# -*- coding: utf-8 -*-
"""اختبار منطقي سريع: quran_ref + AyahMatcher + clamp_ayahs على بيانات اصطناعية."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))
sys.stdout.reconfigure(encoding='utf-8')

from quran_ref import QuranReference, normalize
from align import AyahMatcher, clamp_ayahs, word_norm

ref = QuranReference(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
ayahs = ref.track_ayahs('juz', 30)
assert len(ayahs) == 564, len(ayahs)
assert ayahs[0]['surah'] == 78 and ayahs[-1]['surah'] == 114
print('جزء 30:', len(ayahs), 'آية | الأولى:', ayahs[0]['text'][:40], '| الأخيرة:', ayahs[-1]['text'][:40])

# تفريغ اصطناعي: كلمات الجزء 30 بالترتيب مع أزمنة متصاعدة
words, t = [], 0
for a in ayahs[:20]:
    for w in a['norm'].split():
        t += 400
        words.append({'w': word_norm(w), 'start': t, 'end': t + 350})
        t += 350
m = AyahMatcher(words, band=200)
matches = m.match_all(ayahs[:20])
ok = sum(1 for x in matches if x['matched'])
print(f'مطابقة اصطناعية: {ok}/20')
assert ok >= 19, matches[:5]

verses, suspects = clamp_ayahs(ayahs[:20], matches, words, 3600000)
print('آيات مُوقتة:', len(verses), '| اشتباهات:', len(suspects))
assert verses[0]['start_ms'] >= 0 and all(v['end_ms'] > v['start_ms'] for v in verses)
assert all(a['end_ms'] <= b['start_ms'] for a, b in zip(verses, verses[1:]))

# آية ساقطة (حذف كلمات الآية 10 من التفريغ) → يجب أن تظهر غير مطابقة
words2 = [w for w in words if not (verses[9]['start_ms'] - 50 <= w['start'] <= verses[9]['end_ms'] + 50)]
m2 = AyahMatcher(words2, band=200)
matches2 = m2.match_all(ayahs[:20])
unmatched = [i + 1 for i, x in enumerate(matches2) if not x['matched']]
print('غير المطابقة بعد السقط الاصطناعي:', unmatched)
assert unmatched, 'يجب أن تُكتشف الآية الساقطة'
print('ALL LOCAL TESTS PASSED')
