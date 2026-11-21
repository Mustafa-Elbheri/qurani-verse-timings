"""Resumable, batched Small alignment and evidence-bound automatic acceptance.

No missing unit receives an invented timestamp. Failed units remain explicit;
the exporter independently recomputes every accepted unit from saved evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import re
import unicodedata

from common import download, ffprobe_duration_ms, load_audio_ffmpeg, log, write_json
from quran_ref import QuranReference, normalize, DATA_DIR
from quality import assess, reference_digest, POLICY_VERSION
from runtime import whisper, digest


def references(kind, number):
    ayahs = QuranReference().track_ayahs(kind, number)
    raw = json.loads((Path(DATA_DIR) / 'quran_imlaei.json').read_text(encoding='utf-8'))
    # Matching normalization must not become a lossy alignment transcript:
    # e.g. على -> علي and حياة -> حياه lower the acoustic token likelihood.
    def alignment_text(text):
        return ' '.join(''.join(c for c in text.replace('ٱ','ا').replace('\ufeff','').replace('ـ','')
                               if unicodedata.category(c) not in ('Mn','Me')).split())
    spoken = {tuple(map(int, v['verse_key'].split(':'))): alignment_text(v['text_imlaei']) for v in raw['verses']}
    for ayah in ayahs:
        ayah['asr_text'] = spoken[(ayah['surah'], ayah['ayah'])]
    return ayahs


def align_window(audio, ayahs, numbers, begin_ms, end_ms, model_name, device, compute_type,
                 source_hash, ref_hash, path):
    model, model_hash = whisper(model_name, device, compute_type)
    identity = dict(source_sha256=source_hash, reference_sha256=ref_hash,
                    model_sha256=model_hash, model_name=model_name,
                    window_start_ms=begin_ms, window_end_ms=end_ms,
                    policy_version=POLICY_VERSION)
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached.get('identity') == identity and cached.get('numbers') == numbers:
            cached['_path'] = str(path)
            return cached
        # Never overwrite evidence from another source, model, reference or window.
        path = path.with_name(path.stem + '-' + hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12] + '.json')
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            if cached.get('identity') != identity or cached.get('numbers') != numbers:
                raise RuntimeError('Evidence identity collision')
            cached['_path']=str(path)
            return cached
    clip = audio[round(begin_ms * 16):round(end_ms * 16)].copy()
    text = '\n'.join(ayahs[n - 1]['asr_text'] for n in numbers)
    result = model.align(clip, text, language='ar', original_split=True, regroup=False,
                         verbose=None, vad=False, fast_mode=False)
    segments = result.to_dict()['segments'] if result else []
    units = {}
    if len(segments) == len(numbers):
        for n, segment in zip(numbers, segments):
            words = [dict(word=w['word'], start_ms=round(w['start']*1000)+begin_ms,
                          end_ms=round(w['end']*1000)+begin_ms, probability=w.get('probability'))
                     for w in segment.get('words', [])]
            units[str(n)] = dict(words=words)
    # Independent free recognition is only used for the units escalated to Large.
    if model_name == 'large-v3':
        from rapidfuzz.fuzz import ratio
        try:
            segments, _ = model.transcribe_original(clip, language='ar', initial_prompt=None,
                         condition_on_previous_text=False, beam_size=1, temperature=0,
                         vad_filter=False, word_timestamps=True)
            free = [dict(word=w.word, start_ms=round(w.start*1000)+begin_ms,
                         end_ms=round(w.end*1000)+begin_ms, probability=w.probability)
                    for segment in segments for w in (segment.words or [])]
        except (IndexError, RuntimeError):
            free = []
        for n, unit in units.items():
            words = unit['words']
            if not words:
                continue
            lo, hi = words[0]['start_ms']-150, words[-1]['end_ms']+150
            local = [w for w in free if w['end_ms'] > lo and w['start_ms'] < hi]
            expected = ''.join(ayahs[int(n)-1]['asr_text'].split())
            observed = ''.join(normalize(w['word']).replace(' ','') for w in local)
            unit['free_words'] = local
            unit['free_identity_verified'] = ratio(expected, observed) >= 92
    document = dict(identity=identity, numbers=numbers, units=units)
    write_json(str(path), document)
    document['_path'] = str(path)
    return document


def groups_from_primary(primary, total, duration_ms):
    """Use observed neighbouring anchors; never interpolate missing timestamps."""
    units = primary.get('units', {})
    locations = {}
    for n in range(1, total + 1):
        words = units.get(str(n), {}).get('words', [])
        if words and words[-1]['end_ms']-words[0]['start_ms'] >= 120:
            if all(w['end_ms'] > w['start_ms'] for w in words):
                locations[n] = (words[0]['start_ms'], words[-1]['end_ms'])
    groups = []
    block = min(8, max(1, total // 2))
    for first in range(1, total+1, block):
        last = min(total, first+block-1)
        context = list(range(max(1, first-1), min(total,last+1)+1))
        if any(n not in locations for n in context):
            groups.append(dict(targets=list(range(first,last+1)), context=context,
                               error='missing_source_location'))
            continue
        begin = max(0, locations[context[0]][0] - 1000)
        end = min(duration_ms, locations[context[-1]][1] + 1000)
        if not 0 < end-begin <= 119000:
            groups.append(dict(targets=list(range(first,last+1)), context=context,
                               error='source_window_exceeds_120_seconds'))
            continue
        groups.append(dict(targets=list(range(first,last+1)), context=context,begin=begin,end=end))
    return groups


def process(args):
    import numpy as np
    if os.name == 'nt' and not getattr(subprocess.Popen, '_qurani_hidden', False):
        parent = subprocess.Popen
        class HiddenProcess(parent):
            _qurani_hidden = True
            def __init__(self, *a, **kw):
                kw['creationflags'] = kw.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW
                super().__init__(*a, **kw)
        subprocess.Popen = HiddenProcess
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    audio_path = directory / 'audio.mp3'
    if args.audio:
        original = Path(args.audio).resolve()
        if not audio_path.exists():
            os.link(original, audio_path)  # Same-drive reference, without copying source audio.
        if digest(audio_path) != digest(original):
            raise RuntimeError('Existing source link differs from the requested native source')
    else:
        download(args.track_url, str(audio_path))
    source_hash = digest(audio_path)
    audio = load_audio_ffmpeg(str(audio_path))
    duration = round(len(audio) * 1000 / 16000)
    ayahs = references(args.unit_kind, args.unit_number)
    ref_hash = reference_digest(ayahs)
    audio.flags.writeable = False
    evidence_dir = directory/'evidence'
    evidence_dir.mkdir(exist_ok=True)
    primary = align_window(audio, ayahs, list(range(1,len(ayahs)+1)), 0, duration,
                           'small', args.device,args.compute_type,source_hash,ref_hash,evidence_dir/'primary.json')
    groups = groups_from_primary(primary,len(ayahs),duration)
    observations = {n:[] for n in range(1,len(ayahs)+1)}
    descriptors = {str(n):[] for n in observations}
    statuses = {n:['missing_source_location'] for n in observations}
    latest = {n:[] for n in observations}
    for model_name in ('small','large-v3'):
        for group in groups:
            targets = [n for n in group['targets'] if statuses[n]]
            if not targets or group.get('error'):
                continue
            documents = []
            for shift in (0,310):
                begin = max(0,group['begin']-shift)
                end = min(duration,group['end']+shift)
                path = evidence_dir/f'{model_name}-{group["targets"][0]:04d}-{shift}.json'
                doc = align_window(audio,ayahs,group['context'],begin,end,model_name,
                                   args.device,args.compute_type,source_hash,ref_hash,path)
                actual = Path(doc.get('_path',str(path)))
                documents.append((doc,actual))
            for n in targets:
                candidates = [{**doc['identity'],**doc.get('units',{}).get(str(n),{})} for doc,_ in documents]
                reasons = assess(ayahs[n-1],candidates,source_hash,ref_hash)
                latest[n] = candidates
                statuses[n] = reasons
                if not reasons:
                    observations[n] = candidates
                    descriptors[str(n)] = [dict(file=p.relative_to(directory).as_posix(),sha256=digest(p)) for _,p in documents]
            save_result(directory,args,ayahs,source_hash,ref_hash,duration,observations,descriptors,statuses)
        log(f'{model_name}: verified {sum(not r for r in statuses.values())}/{len(ayahs)}')
    run_ctc_pending(audio,ayahs,groups,args,directory,duration,source_hash,ref_hash,
                    observations,descriptors,statuses,latest)
    save_result(directory,args,ayahs,source_hash,ref_hash,duration,observations,descriptors,statuses)
    return 0 if not any(statuses.values()) else 2


def run_ctc_pending(audio,ayahs,groups,args,directory,duration,source_hash,ref_hash,observations,descriptors,statuses,latest):
    from arabic_ctc import observe
    for group in groups:
        if group.get('error'):continue
        for number in group['targets']:
            if not statuses[number]:continue
            support=[o for o in latest[number] if o.get('words') and all(w['end_ms']>w['start_ms'] for w in o['words'])]
            if not support:continue
            context=list(range(max(1,number-1),min(len(ayahs),number+1)+1))
            first=primary_word_position(directory,context[0],True)
            last=primary_word_position(directory,context[-1],False)
            if first is None or last is None:continue
            lo=max(0,first-1000);hi=min(duration,last+1000)
            if not 0<hi-lo<=119000:continue
            documents=[];paths=[]
            for shift in (0,310):
                begin=max(0,lo-shift);end=min(duration,hi+shift)
                path=directory/'evidence'/f'ctc-{number:04d}-{shift}-{ref_hash[:10]}.json'
                if path.exists():
                    document=json.loads(path.read_text(encoding='utf-8'))
                    obs={**document['identity'],**document['units'][str(number)]}
                    if obs.get('source_sha256')!=source_hash or obs.get('reference_sha256')!=ref_hash:
                        raise RuntimeError('CTC cache identity mismatch')
                else:
                    try:
                        obs=observe(audio,ayahs,context,begin,end,number,args.device,source_hash,ref_hash)
                    except (ValueError,RuntimeError) as exc:
                        statuses[number]=['ctc_failed:'+type(exc).__name__]
                        break
                    native=obs['words'];obs['ctc_words']=native
                    chosen=min(support,key=lambda o:abs(o['words'][0]['start_ms']-native[0]['start_ms'])+abs(o['words'][-1]['end_ms']-native[-1]['end_ms']))
                    obs['timing_words']=chosen['words']
                    obs['timing_model_sha256']=chosen['model_sha256']
                    if len(native)==len(chosen['words']):
                        obs['words']=[{**w,'probability':n['probability']} for w,n in zip(chosen['words'],native)]
                    document=dict(identity={k:obs[k] for k in ('source_sha256','reference_sha256','model_sha256','model_name','window_start_ms','window_end_ms')},
                                  units={str(number):{k:v for k,v in obs.items() if k not in ('source_sha256','reference_sha256','model_sha256','model_name','window_start_ms','window_end_ms')}})
                    write_json(str(path),document)
                documents.append(obs);paths.append(path)
            if len(documents)==2:
                statuses[number]=assess(ayahs[number-1],documents,source_hash,ref_hash)
                if not statuses[number]:
                    observations[number]=documents
                    descriptors[str(number)]=[dict(file=p.relative_to(directory).as_posix(),sha256=digest(p)) for p in paths]
            save_result(directory,args,ayahs,source_hash,ref_hash,duration,observations,descriptors,statuses)
    log(f'Arabic CTC: verified {sum(not r for r in statuses.values())}/{len(ayahs)}')


def primary_word_position(directory,number,start):
    primary=json.loads((directory/'evidence'/'primary.json').read_text(encoding='utf-8'))
    words=primary.get('units',{}).get(str(number),{}).get('words',[])
    if not words:return None
    return words[0]['start_ms'] if start else words[-1]['end_ms']


def save_result(directory,args,ayahs,source_hash,ref_hash,duration,observations,descriptors,statuses):
    verses=[];pending=[];previous=0
    for n,ayah in enumerate(ayahs,1):
        reasons=list(statuses[n])
        if not reasons:
            obs=observations[n]
            start=round(statistics.median(o['words'][0]['start_ms'] for o in obs))
            end=round(statistics.median(o['words'][-1]['end_ms'] for o in obs))
            start=max(previous,start-30)
            next_n=n+1
            next_starts=[o['words'][0]['start_ms'] for o in observations.get(next_n,[]) if o.get('words')]
            end=min(end+100,duration,min(next_starts)-30 if next_starts else duration)
            if end-start<120:reasons.append('non_positive_or_implausible_after_neighbour_check')
            else:
                verses.append(dict(num=n,surah=ayah['surah'],ayah=ayah['ayah'],text=ayah['text'],start_ms=start,end_ms=end))
                previous=end
        if reasons:pending.append(dict(num=n,surah=ayah['surah'],ayah=ayah['ayah'],reasons=reasons))
    proposal=dict(track_url=args.track_url,source_sha256=source_hash,reference_sha256=ref_hash,
                  duration_ms=duration,unit_kind=args.unit_kind,unit_number=args.unit_number,
                  model='small+targeted-large-v3',verses_count=len(ayahs),approved=not pending,verses=verses)
    review=dict(track_url=args.track_url,source_sha256=source_hash,reference_sha256=ref_hash,
                policy_version=POLICY_VERSION,first_and_last_checked=not statuses[1] and not statuses[len(ayahs)],
                pending_cases=pending,suspects=pending,unit_evidence=descriptors,
                verification_scope='automatic_acoustic_evidence_not_a_claim_of_infallibility')
    write_json(str(directory/'timings.proposed.json'),proposal)
    write_json(str(directory/'review.json'),review)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-url',required=True);ap.add_argument('--audio')
    ap.add_argument('--unit-kind',choices=['juz','surah'],required=True)
    ap.add_argument('--unit-number',type=int,required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--compute-type',default='int8')
    ap.add_argument('--model',choices=['small'],default='small',help='Primary alignment is always Small; escalation is automatic')
    args=ap.parse_args()
    if os.name=='nt':
        base=subprocess.Popen
        class HiddenProcess(base):
            def __init__(self,*a,**kw):
                kw['creationflags']=kw.get('creationflags',0)|subprocess.CREATE_NO_WINDOW
                super().__init__(*a,**kw)
        subprocess.Popen=HiddenProcess
    return process(args)


if __name__=='__main__':sys.exit(main())
