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
    """مطابقة تسلسلية: لكل آية تُبحث عن أفضل موضع مطابق ضمن نطاق BAND بعد موضع الآية السابقة."""

    def __init__(self, words, band=BAND):
        self.words = words          # [{'w': normalized, 'start','end'}]
        self.norms = [w['w'] for w in words]
        self.band = band
        self.ref_words = []         # كلمات المرجع المعيارية لكل آية (للمقارنة)

    def _fuzzy(self, a, b):
        # تشابه سريع على مستوى الحروف مع منح الألف المقصورة/الألف نفس المد
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        if abs(la - lb) > max(la, lb) // 2:
            return 0.0
        same = sum(1 for x, y in zip(a, b) if x == y)
        return same / max(la, lb)

    def _match_ayah(self, ref_norm_words, from_idx, to_idx):
        """DP مصغر: مطابقة كلمات المرجع مع نافذة الكلمات المنطوقة [from_idx, to_idx)."""
        R = len(ref_norm_words)
        best = None  # (score, first_word_idx, last_word_idx)
        W = self.norms[from_idx:to_idx]
        n = len(W)
        if n == 0 or R == 0:
            return None
        # تكلفة DP: لكل زوج (كلمة مرجعية i، كلمة منطوقة j) تشابه؛ مسارات مع فجوات محدودة
        import math
        NEG = -1e9
        dp = [[NEG] * (n + 1) for _ in range(R + 1)]
        par = [[None] * (n + 1) for _ in range(R + 1)]
        dp[0][0] = 0
        for i in range(R + 1):
            for j in range(n + 1):
                cur = dp[i][j]
                if cur <= NEG / 2:
                    continue
                if i < R and j < n:
                    s = self._fuzzy(ref_norm_words[i], W[j])
                    if s > 0.3:  # ربط كلمة بكلمة
                        val = cur + s
                        if val > dp[i + 1][j + 1]:
                            dp[i + 1][j + 1] = val
                            par[i + 1][j + 1] = (i, j, 'm')
                if j < n:  # تخطي كلمة منطوقة (تعليق/تكرار)
                    if cur > dp[i][j + 1]:
                        dp[i][j + 1] = cur - 0.05
                        par[i][j + 1] = (i, j, 's')
                if i < R and j < n:  # فجوة: كلمة مرجعية بلا مقابل (اشتباه سقط)
                    val = cur - 0.6
                    if val > dp[i + 1][j]:
                        dp[i + 1][j] = val
                        par[i + 1][j] = (i, j, 'g')
        # أفضل نهاية عند اكتمال المرجع
        end_j = max(range(n + 1), key=lambda j: dp[R][j])
        score = dp[R][end_j]
        # استرجاع المسار
        i, j = R, end_j
        first_j = None
        while i > 0 or j > 0:
            p = par[i][j]
            if p is None:
                break
            pi, pj, op = p
            if op == 'm' and first_j is None:
                first_j = pj
            i, j = pi, pj
        if first_j is None:
            return None
        avg = score / R
        return {'avg': avg, 'first': from_idx + first_j, 'last': from_idx + end_j - 1}

    def match_all(self, ayahs):
        results = []
        cursor = 0
        for idx, ayah in enumerate(ayahs):
            ref_words = [word_norm(w) for w in ayah['norm'].split() if word_norm(w)]
            window_end = min(len(self.words), cursor + max(self.band, len(ref_words) * 4 + 40))
            m = self._match_ayah(ref_words, cursor, window_end)
            if m and m['avg'] >= SIMILARITY_MIN and m['last'] >= m['first']:
                results.append({'matched': True, **m})
                cursor = m['last'] + 1
            else:
                results.append({'matched': False, 'avg': m['avg'] if m else 0.0,
                                'first': cursor, 'last': cursor - 1})
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
    """تطبيق قواعد Zero-Reverb والرتابة الصارمة وحد الصوت؛ إرجاع (verses, suspects)."""
    verses = []
    suspects = []
    prev_end = 0
    n = len(ayahs)
    for i, (ayah, m) in enumerate(zip(ayahs, matches)):
        num = i + 1
        if not m['matched']:
            suspects.append({'num': num, 'reason': 'ayah_unmatched',
                             'similarity': round(m.get('avg', 0), 3)})
            continue
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
                nf = matches[k]['first']
                next_word_start = words[nf]['start']
                break
        if next_word_start is None:
            next_word_start = duration_ms
        end = min(raw_end + EXTEND_MS, next_word_start - NEXT_GAP_MS, duration_ms)
        reasons = []
        if raw_start <= EDGE_MARGIN_MS:
            reasons.append('start_at_audio_edge')
        if duration_ms - raw_end <= EDGE_MARGIN_MS:
            reasons.append('end_at_audio_edge')
        if end <= start:
            reasons.append('non_positive_after_clamp')
        if end <= start:
            # لا نختلق زمنًا: نسجل اشتباهًا ونتجاوز القيمة
            suspects.append({'num': num, 'reason': ';'.join(reasons) or 'collapsed',
                             'start_ms': start, 'end_ms': end})
            continue
        verses.append({'num': num, 'surah': ayah['surah'], 'ayah': ayah['ayah'],
                       'text': ayah['text'], 'start_ms': int(start), 'end_ms': int(end)})
        prev_end = end
        if reasons:
            suspects.append({'num': num, 'reason': ';'.join(reasons),
                             'start_ms': int(start), 'end_ms': int(end)})
    # رتابة: آية متباعدة بعد آية غير مطابقة قد تكون سقطًا — يبقى للتصعيد
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
    model = stable_whisper.load_faster_whisper(
        args.model, device=args.device, compute_type=args.compute_type)
    full_text = ' '.join(a['text'] for a in ayahs)
    log("المحاذاة بـ stable-ts...")
    result = model.align(audio, full_text, language='ar')
    words = extract_words(result)
    if not words:
        log("لم تنتج المحاذاة كلمات — تحويل للتفريغ الحر word_timestamps")
        result = model.transcribe(audio, language='ar', word_timestamps=True,
                                  beam_size=1, temperature=0)
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
    log("حفظ timings.proposed.json و review.json")
    return 0 if not suspects else 10  # 10 = توجد حالات تحتاج تصعيد (لا يفشل الووركفلو)


if __name__ == '__main__':
    sys.exit(main())
