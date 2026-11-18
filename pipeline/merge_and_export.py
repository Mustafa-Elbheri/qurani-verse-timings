# -*- coding: utf-8 -*-
"""دمج نتائج التسجيلات وتشغيل التصدير مع التدقيق الهيكلي (للآلة النهائية في CI)."""
import json, os, subprocess, sys

def main():
    summary = json.loads(os.environ['SUMMARY_JSON'])
    results = os.environ.get('RESULTS_DIR', 'results')
    safe = ''.join(c if c.isalnum() else '-' for c in summary['reciter']).strip('-')
    out = f"timings/{safe}.json"
    cmd = [sys.executable, 'pipeline/export.py', '--results', results,
           '--reciter', summary['reciter'], '--unit-kind', summary['kind'],
           '--unit-name', summary['unit'], '--server', summary['server'], '--out', out]
    print('RUN:', ' '.join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)

if __name__ == '__main__':
    main()
