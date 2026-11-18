# -*- coding: utf-8 -*-
"""تنزيل تسجيل واحد من الأرشيف (لوالإصلاح)."""
import os, sys, urllib.request

url = os.environ['TRACK_URL']
dest_dir = os.environ.get('TRACK_DIR', 'work/track')
os.makedirs(dest_dir, exist_ok=True)
dest = os.path.join(dest_dir, 'audio.mp3')
req = urllib.request.Request(url, headers={'User-Agent': 'qurani-verse-timings/1.0'})
with urllib.request.urlopen(req, timeout=1800) as r, open(dest, 'wb') as f:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
print('downloaded', os.path.getsize(dest))
