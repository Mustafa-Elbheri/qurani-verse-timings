# -*- coding: utf-8 -*-
"""اختبار منطقي سريع للمطابقة الشاملة (Banded DP): سلامة أساسية + اكتشاف سقط."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))
sys.stdout.reconfigure(encoding='utf-8')

from quran_ref import QuranReference
from align import AyahMatcher, clamp_ayahs, word_norm

ref = QuranReference(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
ayahs = ref.track_ayahs('juz', 30)
assert len(ayahs) == 564 and ayahs[0]['text'][0] != ' ', 'strip text'

# تفريغ اصطناعي متصل لأول 30 آية
words, t = [], 0
for a in ayahs[:30]:
    for w in a['norm'].split():
        t += 400
        words.append({'w': word_norm(w), 'start': t, 'end': t + 350})
        t += 350
m = AyahMatcher(words, band=500)
matches = m.match_all(ayahs[:30])
ok = sum(1 for x in matches if x['matched'])
print(f'مطابقة شاملة: {ok}/30')
assert ok == 30, [(i + 1, x) for i, x in enumerate(matches) if not x['matched']][:5]
assert all(x['missing'] == 0 for x in matches), 'لا كلمات ناقصة في الاصطناعي'

verses, suspects = clamp_ayahs(ayahs[:30], matches, words, 3600000)
assert len(verses) == 30 and not suspects, f'اشتباهات غير متوقعة: {suspects[:3]}'
assert all(a['end_ms'] <= b['start_ms'] + 1 for a, b in zip(verses, verses[1:])), 'رتابة'

# سقط اصطناعي: حذف كلمات الآية 10 بالكامل من التفريغ
v10 = [v for v in verses if v['num'] == 10][0]
words2 = [w for w in words if not (v10['start_ms'] - 200 <= w['start'] <= v10['end_ms'] + 200)]
m2 = AyahMatcher(words2, band=500)
matches2 = m2.match_all(ayahs[:30])
bad = [i + 1 for i, x in enumerate(matches2) if not x['matched'] or x.get('missing', 0) > 0]
print('الحالات المكتشفة بعد السقط الاصطناعي:', bad)
assert 10 in bad, 'يجب اكتشاف الآية الساقطة'
# الآيات السليمة تبقى في أماكنها الصحيحة
verses2, suspects2 = clamp_ayahs(ayahs[:30], matches2, words2, 3600000)
pos = {v['num']: v['start_ms'] for v in verses2}
assert abs(pos.get(11, 0) - [v for v in verses if v['num'] == 11][0]['start_ms']) < 1500, 'انزلاق بعد السقط!'
print('ALL LOCAL TESTS PASSED')
