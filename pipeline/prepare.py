# -*- coding: utf-8 -*-
"""يحضّر مصفوفة الوظائف: يقرأ sources/review_moshafs.json ومدخلات التشغيل ويخرج JSON للمصفوفة."""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_range(spec, maximum):
    nums = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            nums.update(range(int(a), int(b) + 1))
        else:
            nums.add(int(part))
    return sorted(n for n in nums if 1 <= n <= maximum)

def main():
    reciter_index = int(os.environ.get('INPUT_RECITER_INDEX', '6'))
    tracks_spec = os.environ.get('INPUT_TRACKS', '1-30')
    repo = os.environ.get('GITHUB_WORKSPACE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(repo, 'sources', 'review_moshafs.json'), encoding='utf-8') as f:
        catalog = json.load(f)
    moshaf = catalog['moshafs'][reciter_index]
    kind = 'juz' if moshaf['unit'] == 'جزء' else 'surah'
    maximum = len(moshaf['tracks'])
    selected = parse_range(tracks_spec, maximum)
    matrix = []
    for n in selected:
        track = moshaf['tracks'][n - 1]
        matrix.append({'number': n, 'name': track['name'], 'url': track['url']})
    summary = {'reciter': moshaf['reciter'], 'unit': moshaf['unit'], 'kind': kind,
               'server': moshaf['server'], 'selected': selected}
    with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a', encoding='utf-8') as f:
        f.write(f"matrix={json.dumps(matrix, ensure_ascii=False)}\n")
        f.write(f"summary={json.dumps(summary, ensure_ascii=False)}\n")
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
