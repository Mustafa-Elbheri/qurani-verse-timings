# -*- coding: utf-8 -*-
"""المرجع القرآني: تحميل النص العثماني ووحدات التسجيل (جزء/سورة) وتطبيع النص للمطابقة."""
import json, os, re, unicodedata

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

SURAH_NAMES = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف", "الأنفال", "التوبة", "يونس",
    "هود", "يوسف", "الرعد", "إبراهيم", "الحجر", "النحل", "الإسراء", "الكهف", "مريم", "طه",
    "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان", "الشعراء", "النمل", "القصص", "العنكبوت", "الروم",
    "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر", "يس", "الصافات", "ص", "الزمر", "غافر",
    "فصلت", "الشورى", "الزخرف", "الدخان", "الجاثية", "الأحقاف", "محمد", "الفتح", "الحجرات", "ق",
    "الذاريات", "الطور", "النجم", "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة",
    "الصف", "الجمعة", "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة", "المعارج",
    "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان", "المرسلات", "النبأ", "النازعات", "عبس",
    "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج", "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد",
    "الشمس", "الليل", "الضحى", "الشرح", "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات",
    "القارعة", "التكاثر", "العصر", "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر", "الكافرون", "النصر",
    "المسد", "الإخلاص", "الفلق", "الناس",
]

_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640\u06D6-\u06ED]')

def normalize(text):
    """تطبيع نص للمطابقة فقط: حذف التشكيل والمد والتطويل، وتوحيد الألف والياء والهاء."""
    text = unicodedata.normalize('NFC', text)
    text = _DIACRITICS.sub('', text)
    text = (text.replace('ٱ', 'ا').replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
                .replace('ى', 'ي').replace('ة', 'ه'))
    return text.strip()

class QuranReference:
    def __init__(self, data_dir=DATA_DIR):
        with open(os.path.join(data_dir, 'quran_uthmani.json'), encoding='utf-8') as f:
            raw = json.load(f)
        self.verses = raw['verses']
        self.by_key = {(v['surah'], v['ayah']): v['text'] for v in self.verses}
        with open(os.path.join(data_dir, 'juz_ranges.json'), encoding='utf-8') as f:
            self.juz_ranges = json.load(f)

    def track_ayahs(self, unit_kind, unit_number):
        """قائمة {'surah','ayah','text','norm'} للتسجيل: جزء أو سورة كاملة."""
        if unit_kind == 'juz':
            keys = self.juz_ranges[str(unit_number)]
        elif unit_kind == 'surah':
            count = sum(1 for v in self.verses if v['surah'] == unit_number)
            keys = [{'surah': unit_number, 'ayah': a} for a in range(1, count + 1)]
        else:
            raise ValueError('unit_kind must be juz or surah')
        out = []
        for k in keys:
            text = self.by_key[(k['surah'], k['ayah'])].strip()
            out.append({'surah': k['surah'], 'ayah': k['ayah'], 'text': text,
                        'norm': normalize(text)})
        return out
