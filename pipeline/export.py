# -*- coding: utf-8 -*-
"""التصدير النهائي: دمج نتائج التسجيلات، التدقيق الهيكلي الشامل، ورفض الحالات المعلقة.

المخرج: timings/<reciter>.json بصيغة مناسبة للتطبيق + verification.json.
القاعدة (من التعليمات): أي سطر بلا حد محسوم أو أي خطأ هيكلي = لا تسليم.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, write_json
from quran_ref import QuranReference

def structural_audit(tracks, ref, unit_kind):
    """0 أخطاء مطلوب: تصاعد صارم، end>start، ضمن مدة الصوت، عدد الآيات = المرجع."""
    errors = []
    for track_name, track in tracks.items():
        unit = track['unit_number']
        ayahs = ref.track_ayahs(unit_kind, unit)
        if len(track['verses']) != len(ayahs):
            errors.append(f'{track_name}: عدد الآيات {len(track["verses"])} != المرجع {len(ayahs)}')
            continue
        prev_end = -1
        for i, v in enumerate(track['verses']):
            s, e = v['start_ms'], v['end_ms']
            if s < 0 or e <= s:
                errors.append(f'{track_name} آية {v["num"]}: start={s} end={e}')
            if s < prev_end:
                errors.append(f'{track_name} آية {v["num"]}: تراجع زمني {s} < {prev_end}')
            if e > track['duration_ms']:
                errors.append(f'{track_name} آية {v["num"]}: end {e} يتجاوز مدة الصوت {track["duration_ms"]}')
            if e - s < 120:
                errors.append(f'{track_name} آية {v["num"]}: implausible_duration:{e-s}ms')
            if v.get('suspected_boundary'):
                errors.append(f'{track_name} آية {v["num"]}: unresolved_suspected_boundary')
            if v.get('num') != i + 1:
                errors.append(f'{track_name}: non_sequential_unit:{v.get("num")}')
            if v.get('text') != ayahs[i]['text']:
                errors.append(f'{track_name} آية {v["num"]}: reference_text_mismatch')
            if v['surah'] != ayahs[i]['surah'] or v['ayah'] != ayahs[i]['ayah']:
                errors.append(f'{track_name} آية {v["num"]}: مفتاح غير متسلسل {v["surah"]}:{v["ayah"]}')
            prev_end = e
    return errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='مجلد يحوي مجلدات track-N بملفات timings.proposed.json')
    ap.add_argument('--reciter', required=True)
    ap.add_argument('--unit-kind', required=True, choices=['juz', 'surah'])
    ap.add_argument('--unit-name', required=True, help='اسم الوحدة في التطبيق (جزء/سورة)')
    ap.add_argument('--server', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-tracks', default='', help='Explicit expected recording numbers; default is the whole mushaf')
    ap.add_argument('--allow-partial', action='store_true', help='Diagnostic checkpoint only, never ready for app integration')
    args = ap.parse_args()

    ref = QuranReference()
    tracks = {}
    pending_total = []
    acoustic_errors = []
    seen_units = set()
    approved = True
    entries = sorted(os.listdir(args.results))
    for entry in entries:
        prop_path = os.path.join(args.results, entry, 'timings.proposed.json')
        rev_path = os.path.join(args.results, entry, 'review.json')
        if not os.path.isfile(prop_path):
            continue
        prop = json.load(open(prop_path, encoding='utf-8'))
        review = json.load(open(rev_path, encoding='utf-8')) if os.path.isfile(rev_path) else {}
        from quality import audit_review
        from verified_align import references
        try:
            quality_errors = audit_review(prop, review, os.path.dirname(prop_path), references(args.unit_kind, prop['unit_number']))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            quality_errors = ['invalid_or_incomplete_evidence:' + type(exc).__name__]
        acoustic_errors.extend(f'{entry}:{error}' for error in quality_errors)
        if quality_errors:
            approved = False
        if prop['unit_number'] in seen_units:
            acoustic_errors.append(f'{entry}:duplicate_recording_number')
        seen_units.add(prop['unit_number'])
        pending = review.get('pending_cases', [])
        if pending or not prop.get('verses'):
            approved = False
            pending_total.extend([{**c, 'track': entry} for c in pending] or
                                 [{'track': entry, 'reason': 'empty_verses'}])
        tracks[prop['track_url'].rsplit('/', 1)[-1]] = {
            'url': prop['track_url'],
            'sha256': prop['source_sha256'],
            'duration_ms': prop['duration_ms'],
            'unit_number': prop['unit_number'],
            'verses': prop['verses'],
        }

    errors = structural_audit(tracks, ref, args.unit_kind)
    from prepare import parse_range
    maximum = 30 if args.unit_kind == 'juz' else 114
    expected = set(parse_range(args.expected_tracks, maximum)) if args.expected_tracks else set(range(1, maximum+1))
    if not expected or seen_units != expected:
        errors.append(f'recording_coverage_mismatch: expected={sorted(expected)}, actual={sorted(seen_units)}')
    complete = seen_units == set(range(1, maximum+1))
    if not complete and not args.allow_partial:
        errors.append('partial_mushaf_not_approved_for_integration')
    errors.extend(acoustic_errors)
    status = 'PASS' if (approved and not errors) else 'FAIL'
    verification = {
        'structural_checks': status,
        'tracks': len(tracks),
        'errors': errors,
        'pending_cases': pending_total,
        'first_and_last_checked_all_tracks': approved and bool(tracks),
        'automatic_acoustic_checks': 'PASS' if not acoustic_errors and bool(tracks) else 'FAIL',
        'coverage_complete': complete,
        'approved_for_integration': status == 'PASS' and complete,
        'expected_tracks': sorted(expected),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_json(args.out + '.verification.json', verification)

    if status != 'PASS':
        log(f'التدقيق فشل: {len(errors)} خطأ هيكلي و {len(pending_total)} حالة معلقة — لا تسليم.')
        return 1

    final = {
        'reciter': args.reciter + ' (نسخة قرآني)',
        'unit': args.unit_name,
        'source': args.server,
        'tracks': tracks,
        'verification': {'approved_for_integration': status == 'PASS' and complete, 'coverage_complete': complete},
    }
    write_json(args.out, final)
    total_verses = sum(len(t['verses']) for t in tracks.values())
    log(f'تسليم ناجح: {len(tracks)} تسجيلًا و {total_verses} آية — {args.out}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
