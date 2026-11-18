# -*- coding: utf-8 -*-
"""تصعيد الحالات المشتبهة فقط: تفريغ حر بـ Large-v3 ونافذتان، وعند الحاجة النموذج العربي CTC.

- يقص نافذة حول الآية المشتبهة (±Padding) من المصدر الأصلي.
- تفريغ حر دون تلقين المرجع: beam_size=1 وtemperature=0 وبالعربية — دليل هوية الكلام.
- لا يستخدم أزمنة التفريغ الحر كحدود؛ الحدود تبقى من محاذاة Small.
- ما لم يُحسم يبقى معلقًا ويمنع التصدير (لا اختلاق زمن).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, write_json, ffmpeg_slice
from quran_ref import QuranReference, normalize

PADDING_MS = 6000
SIMILARITY_RESOLVE = 0.90  # تطابق جسم الآية في تفريغين حرين يؤيد الهوية


def fuzzy_ratio(a, b):
    if not a or not b:
        return 0.0
    a, b = normalize(a), normalize(b)
    la, lb = len(a), len(b)
    if abs(la - lb) > max(la, lb) // 2:
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / max(la, lb)


def load_model_large(device, compute_type):
    import stable_whisper
    return stable_whisper.load_faster_whisper('large-v3', device=device, compute_type=compute_type)


def free_transcript(model, clip_path):
    """تفريغ حر مستقل: دون مرجع، بالعربية، beam_size=1 وtemperature=0."""
    r = model.transcribe(clip_path, language='ar', word_timestamps=True,
                         beam_size=1, temperature=0, initial_prompt=None)
    texts = []
    for seg in (getattr(r, 'segments', None) or []):
        t = getattr(seg, 'text', '') or ''
        if t.strip():
            texts.append(t.strip())
    return ' '.join(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--proposal', required=True, help='مسار timings.proposed.json')
    ap.add_argument('--review', required=True, help='مسار review.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--verses', default='', help='أرقام الآيات المطلوب حسمها (مفصولة بفواصل)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compute-type', default='int8')
    ap.add_argument('--with-ctc', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    proposal = json.load(open(args.proposal, encoding='utf-8'))
    review = json.load(open(args.review, encoding='utf-8'))
    duration_ms = proposal['duration_ms']

    wanted = {int(x) for x in args.verses.split(',') if x.strip()} if args.verses else None
    pending = [s for s in review['suspects']
               if wanted is None or s['num'] in wanted]
    if not pending:
        log('لا حالات مشتبهة مطلوبة — لا شيء يُصعّد.')
        return 0
    if len(pending) > 40:
        # فشل شامل في المحاذاة وليس حالات فردية — لا تصعيد عشوائي
        log(f'اشتباهات كثيرة جدًا ({len(pending)}) — فشل شامل؛ تُرفض الآلة ولا تُصعّد.')
        review['pending_cases'].extend(
            {'num': c['num'], 'reason': 'wholesale_failure:' + c['reason']} for c in pending)
        write_json(args.review, review)
        return 2

    ref = QuranReference()
    ayah_by_num = {}
    for kind_key in ('juz', 'surah'):
        try:
            ayahs = ref.track_ayahs(kind_key, proposal['unit_number'])
            if len(ayahs) == proposal['verses_count']:
                ayah_by_num = {i + 1: a for i, a in enumerate(ayahs)}
                break
        except Exception:
            continue
    if not ayah_by_num:
        ayah_by_num = {v['num']: v for v in proposal['verses']}

    model = load_model_large(args.device, args.compute_type)
    evidence = []
    resolved_nums = set()
    for case in pending:
        num = case['num']
        try:
            ayah = ayah_by_num.get(num)
            if ayah is None:
                continue
            # نافذة: من نهاية الآية السابقة المطابقة (أو بداية الصوت) إلى بداية التالية (أو النهاية)
            prev_end = 0
            next_start = duration_ms
            for v in proposal['verses']:
                if v['num'] < num and v['end_ms'] > prev_end:
                    prev_end = v['end_ms']
                if v['num'] > num and v['start_ms'] < next_start:
                    next_start = v['start_ms']
            start = max(0, prev_end - PADDING_MS)
            end = min(duration_ms, next_start + PADDING_MS)
            if end - start > 120000:  # سقف النافذة 120 ثانية
                end = start + 120000
            clip = os.path.join(args.out, f'window-{num:04d}.wav')
            ffmpeg_slice(args.audio, start, end, clip)

            t1 = free_transcript(model, clip)
            t2 = free_transcript(model, clip)
            sim1, sim2 = fuzzy_ratio(t1, ayah['text']), fuzzy_ratio(t2, ayah['text'])
            ev = {'num': num, 'window_start_ms': start, 'window_end_ms': end,
                  'transcript_1': t1, 'transcript_2': t2,
                  'similarity_1': round(sim1, 3), 'similarity_2': round(sim2, 3),
                  'clip_sha256': None, 'ctc': None}
            evidence.append(ev)
            if sim1 >= SIMILARITY_RESOLVE and sim2 >= SIMILARITY_RESOLVE:
                # تأييد الهوية: الآية منطوقة هنا؛ تبقى الحدود من المحاذاة أو تبقى معلقة إن لم توجد حدود
                resolved_nums.add(num)
                review['resolved_cases'].append({
                    'num': num,
                    'reason': f'large-v3 free transcripts match ayah body (sim={sim1:.2f}/{sim2:.2f})',
                    'evidence_files': [os.path.abspath(os.path.join(args.out, f'window-{num:04d}.wav'))],
                })
                if not any(v['num'] == num for v in proposal['verses']):
                    # الآية لم تُطابق أصلًا — لا نختلق زمنًا؛ نطلب حسم الحدود بمحاذاة نطاق متصل لاحقًا
                    review['pending_cases'].append({'num': num,
                                                    'reason': 'identity_confirmed_but_bounds_unresolved',
                                                    'needs_joint_window': True})
                    resolved_nums.discard(num)
        except Exception as ex:
            log(f'تعذّر حسم الآية {num}: {ex}')
            continue

    if args.with_ctc:
        run_ctc(args, pending, proposal, review, resolved_nums)

    review['suspects'] = [s for s in review['suspects'] if s['num'] not in resolved_nums]
    review['pending_cases'] = [c for c in review['pending_cases']
                               if c.get('num') not in resolved_nums]
    write_json(os.path.join(args.out, 'independent-transcript.json'), {'evidence': evidence})
    write_json(args.review, review)
    log(f"حُسمت: {len(resolved_nums)} | معلقة: {len(review['pending_cases'])}")
    return 0


def run_ctc(args, pending, proposal, review, resolved_nums):
    """النموذج العربي Wav2Vec2: تفريغ حر CTC لما بقي مشتبهًا (اختياري)."""
    remaining = [c for c in pending if c['num'] not in resolved_nums]
    if not remaining:
        return
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError:
        log('مكتبات CTC غير مثبتة — تخطي.')
        return
    import subprocess, wave
    processor = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    ctc = Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    ctc.eval()
    ref = QuranReference()
    ayah_by_num = {v['num']: v for v in
                   ref.track_ayahs(proposal['unit_kind'], proposal['unit_number'])}
    for case in remaining:
        num = case['num']
        clip = os.path.join(args.out, f'window-{num:04d}.wav')
        if not os.path.isfile(clip):
            continue
        # تحويل إلى 16kHz mono raw
        raw = clip + '.16k.wav'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', clip, '-ac', '1', '-ar', '16000', raw],
                       capture_output=True, timeout=300)
        with wave.open(raw, 'rb') as w:
            frames = w.readframes(w.getnframes())
        import numpy as np
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        inputs = processor(audio_np, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            logits = ctc(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0]
        text = processor.decode(ids)
        # حفظ الدليل؛ لا يغير الحدود — دليل هوية فقط
        with open(os.path.join(args.out, f'ctc-{num:04d}.json'), 'w', encoding='utf-8') as f:
            json.dump({'num': num, 'free_ctc': text}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
