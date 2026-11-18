# -*- coding: utf-8 -*-
"""أدوات مشتركة: تنزيل الصوت، البصمات، مدة الملف، كتابة JSON."""
import hashlib, json, os, subprocess, sys, urllib.request

def log(msg):
    print(f"[{os.path.basename(sys.argv[0])}] {msg}", flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def download(url, dest, timeout=1800):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + '.part'
    req = urllib.request.Request(url, headers={'User-Agent': 'qurani-verse-timings/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest

def ffprobe_duration_ms(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'json', path], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError('ffprobe failed: ' + r.stderr[-300:])
    seconds = float(json.loads(r.stdout)['format']['duration'])
    return round(seconds * 1000)

def ffmpeg_slice(path, start_ms, end_ms, dest):
    r = subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-ss', f'{start_ms / 1000:.3f}',
         '-to', f'{end_ms / 1000:.3f}', '-i', path, dest],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg failed: ' + r.stderr[-300:])
    return dest

def load_audio_ffmpeg(path, sr=16000):
    """تحميل الصوت كمصفوفة float32 بـ16kHz عبر ffmpeg — تجاوز كامل لمشكلة توافق PyAV."""
    import numpy as np
    r = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1',
         '-ar', str(sr), 'pipe:1'], capture_output=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg decode failed: ' + (r.stderr or b'')[-300:].decode('utf-8', 'replace'))
    return np.frombuffer(r.stdout, dtype=np.float32).copy()

def write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
