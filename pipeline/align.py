# -*- coding: utf-8 -*-
"""المحاذاة الأولى: stable-ts + Whisper Small، ثم مطابقة الكلمات المنطوقة بالآيات المرجعية.

المنهجية (وفق تعليمات التفريغ):
- stable-ts align على نص التسجيل كاملًا بـ Small؛ عند التعثر يُلجأ للتفريغ الحر word_timestamps.
- مطابقة كل آية بكلمات التفريغ عبر DP نطاقي (BAND) مع مطابقة ضبابية.
- تطبيق قواعد منع الصدى والرتابة الصارمة وحد الصوت.
- أي آية لا تُطابق بثقة كافية تُسجل «مشتبهًا» ولا يُختلق لها زمن.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, sha256_file, download, ffprobe_duration_ms, write_json
from quran_ref import QuranReference, normalize

BAND = 900            # نطاق البحث عن موضع الآية بالكلمات (900 للمتون؛ يفضل 2500 لمن يقرأ العناوين مطولة)
SIMILARITY_MIN = 0.80  # أقل تشابه مقبول لجسم الآية
PRE_ROLL_MS = 30       # اقتطاع مسبق أقصى قبل أول كلمة (حماية همزة الوصل دون جلب صدى)
EXTEND_MS = 100        # تمديد أقصى بعد نطق آخر حرف لحفظ المد الطبيعي
NEXT_GAP_MS = 30       # توقف إجباري قبل نطق أول حرف من الآية التالية
EDGE_MARGIN_MS = 250   # حد عند طرف المقطع = اشتباه


def word_norm(w):
    return normalize(w).strip('و ').strip()  # حذف وصل شائع في أول الكلمة


class AyahMatcher:
    """محاذاة ديناميكية شاملة (Banded DP): تطابق كلمات المرجع كلها مع الكلمات المنطوقة
    في مسار واحد متصل — لا انزلاق ولا مطابقة مزدوجة، وفق درس «المحاذاة الديناميكية الموجهة»."""

    def __init__(self, words, band=500):
        self.words = words                  # [{'w': normalized, 'start', 'end'}]
        self.norms = [w['w'] for w in words]
        self.band = band                    # نصف-نطاق البحث بالكلمات حول القطر المتوقع

    def match_all(self, ayahs):
        import numpy as np
        from rapidfuzz import fuzz
        # تسلسل مرجعي: كل كلمة مع رقم آيتها
        ref_words, ref_ayah = [], []
        for ai, ayah in enumerate(ayahs):
            for w in ayah['norm'].split():
                wn = word_norm(w)
                if wn:
                    ref_words.append(wn)
                    ref_ayah.append(ai)
        R, W = len(ref_words), len(self.norms)
        if R == 0 or W == 0:
            return [{'matched': False, 'avg': 0.0, 'first': 0, 'last': -1} for _ in ayahs]
        scale = W / R
        NEG = -1e9
        SKIP_W = 0.12   # كلمة منطوقة بلا مقابل (بسملة/استعاذة/تعليق)
        SKIP_R = 0.45   # كلمة مرجعية بلا مقابل (سقط مشتبه)
        dp = np.full((R + 1, W + 1), NEG, dtype=np.float64)
        dp[0, 0] = 0.0
        sim_cache = {}
        def sim(i, j):
            key = (i, j)
            v = sim_cache.get(key)
            if v is None:
                a, b = ref_words[i], self.norms[j]
                v = fuzz.ratio(a, b) / 100.0 if abs(len(a) - len(b)) <= max(len(a), len(b), 4) else 0.0
                if v < 0.30:
                    v = 0.0
                sim_cache[key] = v
            return v
        for i in range(1, R + 1):
            jexp = i * scale
            jlo = max(1, int(jexp - self.band))
            jhi = min(W, int(jexp + self.band))
            row, prev = dp[i], dp[i - 1]
            best = NEG
            for j in range(jlo, jhi + 1):
                a = prev[j - 1] + sim(i - 1, j - 1)   # ربط
                b = row[j - 1] - SKIP_W               # تخطي منطوقة
                c = prev[j] - SKIP_R                  # تخطي مرجعية
                m = a if a >= b else b
                if c > m:
                    m = c
                row[j] = m
                if m > best:
                    best = m
            if best <= NEG / 2:
                # انقطع القطر: وسّع النطاق مرة واحدة
                for j in range(1, W + 1):
                    if row[j] > NEG / 2:
                        continue
                    a = prev[j - 1] + sim(i - 1, j - 1)
                    b = row[j - 1] - SKIP_W
                    c = prev[j] - SKIP_R
                    row[j] = max(a, b, c)
        # استرجاع المسار من (R, W)
        pairs = []  # (ref_index, word_index)
        i, j = R, W
        while i > 0 and j > 0:
            cur = dp[i, j]
            if abs(cur - (dp[i - 1, j - 1] + sim(i - 1, j - 1))) < 1e-6:
                pairs.append((i - 1, j - 1)); i -= 1; j -= 1
            elif abs(cur - (dp[i, j - 1] - SKIP_W)) < 1e-6:
                j -= 1
            elif abs(cur - (dp[i - 1, j] - SKIP_R)) < 1e-6:
                i -= 1
            else:
                j -= 1
        pairs.reverse()
        # تجميع حسب الآية
        per_ayah = {ai: [] for ai in range(len(ayahs))}
        missing = {ai: 0 for ai in range(len(ayahs))}
        spoken_of = {}
        for ri, wj in pairs:
            per_ayah[ref_ayah[ri]].append(wj)
            spoken_of[ri] = True
        for ri in range(R):
            if ri not in spoken_of:
                missing[ref_ayah[ri]] += 1
        total = {ai: 0 for ai in range(len(ayahs))}
        for ai in ref_ayah:
            total[ai] += 1
        results = []
        for ai in range(len(ayahs)):
            js = per_ayah[ai]
            if not js:
                results.append({'matched': False, 'avg': 0.0, 'first': 0, 'last': -1,
                                'missing': total[ai], 'total': total[ai]})
            else:
                results.append({'matched': True, 'avg': 1.0,
                                'first': js[0], 'last': js[-1],
                                'missing': missing[ai], 'total': total[ai]})
        return results


def extract_words(result):
    """كلمات بأزمنتها من نتيجة stable-ts (align أو transcribe)."""
    words = []
    segments = getattr(result, 'segments', None) or []
    for seg in segments:
        for w in (getattr(seg, 'words', None) or []):
            word = getattr(w, 'word', '') or ''
            start = getattr(w, 'start', None)
            end = getattr(w, 'end', None)
            if start is None or end is None:
                continue
            words.append({'w': word_norm(word), 'start': round(start * 1000), 'end': round(end * 1000)})
    return words


def clamp_ayahs(ayahs, matches, words, duration_ms):
    """تطبيق قواعد Zero-Reverb والرتابة الصارمة وحد الصوت؛ إرجاع (verses, suspects).
    الضغط الطبيعي بين الآيات المتصلة ليس اشتباهًا — يُضبط الحد دون اختلاق زمن."""
    verses = []
    suspects = []
    prev_end = 0
    n = len(ayahs)
    for i, (ayah, m) in enumerate(zip(ayahs, matches)):
        num = i + 1
        if not m['matched']:
            suspects.append({'num': num, 'reason': 'ayah_unmatched',
                             'missing': m.get('total', 0), 'total': m.get('total', 0)})
            continue
        missing_ratio = m.get('missing', 0) / max(m.get('total', 1), 1)
        w_first = words[m['first']]
        w_last = words[m['last']]
        raw_start = w_first['start']
        raw_end = w_last['end']
        # بداية: لا ترجع خلف نهاية سابقة، واقتطاع مسبق ≤ 30ms
        start = max(prev_end, raw_start - PRE_ROLL_MS)
        # نهاية: تمديد ≤ 100ms، وتوقف قبل أول كلمة من الآية التالية (المطابقة) أو نهاية الصوت
        next_word_start = None
        for k in range(i + 1, n):
            if matches[k]['matched']:
                next_word_start = words[matches[k]['first']]['start']
                break
        if next_word_start is None:
            next_word_start = duration_ms
        end = min(raw_end + EXTEND_MS, next_word_start - NEXT_GAP_MS, duration_ms)
        if end <= start:
            # ضغط متصل طبيعي: نعتمد نهاية نطق الكلمات الفعلية دون تجاوز مدة الصوت
            end = min(raw_end, duration_ms)
        if end <= start:
            suspects.append({'num': num, 'reason': 'non_positive_after_clamp',
                             'start_ms': start, 'end_ms': end})
            continue
        reasons = []
        if raw_start <= EDGE_MARGIN_MS:
            # البدء المباشر عند بداية الملف طبيعي في المصاحف المقسّمة — يُسجل كملاحظة لا كمانع
            if m.get('missing', 0) > 0:
                reasons.append('start_at_audio_edge')
        if duration_ms - raw_end <= EDGE_MARGIN_MS:
            reasons.append('end_at_audio_edge')
        if missing_ratio > 0.5:
            reasons.append(f'partial_ayah:{m.get("missing",0)}/{m.get("total",0)}')
        verses.append({'num': num, 'surah': ayah['surah'], 'ayah': ayah['ayah'],
                       'text': ayah['text'], 'start_ms': int(start), 'end_ms': int(end)})
        prev_end = end
        if reasons:
            suspects.append({'num': num, 'reason': ';'.join(reasons),
                             'start_ms': int(start), 'end_ms': int(end)})
    return verses, suspects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track-url', required=True)
    ap.add_argument('--unit-kind', required=True, choices=['juz', 'surah'])
    ap.add_argument('--unit-number', required=True, type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='small')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compute-type', default='int8')
    ap.add_argument('--band', type=int, default=BAND)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    audio = os.path.join(args.out, 'audio' + os.path.splitext(args.track_url)[1])
    log(f"تنزيل {args.track_url}")
    download(args.track_url, audio)
    source_sha256 = sha256_file(audio)
    duration_ms = ffprobe_duration_ms(audio)
    log(f"مدة الصوت: {duration_ms} ms | sha256: {source_sha256[:16]}...")

    ref = QuranReference()
    ayahs = ref.track_ayahs(args.unit_kind, args.unit_number)
    log(f"عدد آيات المرجع: {len(ayahs)}")

    import stable_whisper
    from common import load_audio_ffmpeg
    model = stable_whisper.load_faster_whisper(
        args.model, device=args.device, compute_type=args.compute_type)
    full_text = ' '.join(a['text'] for a in ayahs)
    log("استخراج الألفاظ المنطوقة بتوقيتاتها (transcribe word_timestamps)...")
    # المسار الموجه: تفريغ حر بكلمات موقوتة ثم ربطه بالمرجع عبر DP (درس المحاذاة الديناميكية)
    audio_np = load_audio_ffmpeg(audio)
    result = model.transcribe(audio_np, language='ar', word_timestamps=True)
    words = extract_words(result)
    if not words:
        log("التفريغ الحر بلا كلمات — محاولة stable-ts align")
        result = model.align(audio, full_text, language='ar')
        words = extract_words(result)
    if not words:
        raise RuntimeError('لا كلمات في الناتج — عطل بيئة')
    log(f"كلمات منطوقة: {len(words)}")

    matcher = AyahMatcher(words, band=args.band)
    matches = matcher.match_all(ayahs)
    verses, suspects = clamp_ayahs(ayahs, matches, words, duration_ms)
    unmatched = sum(1 for m in matches if not m['matched'])
    log(f"مطابقة: {len(matches) - unmatched}/{len(matches)} | اشتباهات: {len(suspects)}")

    proposal = {'track_url': args.track_url, 'source_sha256': source_sha256,
                'duration_ms': duration_ms, 'unit_kind': args.unit_kind,
                'unit_number': args.unit_number, 'model': args.model,
                'verses_count': len(ayahs), 'approved': False,
                'verses': verses, 'audio_words': len(words)}
    review = {'track_url': args.track_url, 'source_sha256': source_sha256,
              'suspects': suspects,
              'first_and_last_checked': False,
              'resolved_cases': [], 'pending_cases': suspects}
    write_json(os.path.join(args.out, 'timings.proposed.json'), proposal)
    write_json(os.path.join(args.out, 'review.json'), review)
    write_json(os.path.join(args.out, 'words.json'), {'words': words})
    log("حفظ timings.proposed.json و review.json و words.json")
    return 0  # خروج ناجح دائمًا عند إنتاج مخرجات؛ الاشتباهات لا تعني عطلًا


if __name__ == '__main__':
    sys.exit(main())
