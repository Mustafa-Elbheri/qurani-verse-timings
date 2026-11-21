"""Bounded CPU-only integration probe; never publish diagnostic audio."""
import argparse,json,os,platform,time
from pathlib import Path
from types import SimpleNamespace
from verified_align import process

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reciter-index',type=int,default=5);ap.add_argument('--track',type=int,default=1)
    args=ap.parse_args();root=Path(__file__).resolve().parents[1]
    m=json.loads((root/'sources/review_moshafs.json').read_text(encoding='utf-8'))['moshafs'][args.reciter_index]
    if m['unit']!='سورة' or args.track!=1:raise ValueError('Short probe supports Al-Fatiha only')
    track=m['tracks'][0];output=root/'results/cpu-probe';output.mkdir(parents=True,exist_ok=True)
    import torch
    report=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),device='cpu',reciter=m['reciter'],track_url=track['url'])
    if report['cuda_available']:raise RuntimeError('CPU-only runner required')
    start=time.monotonic()
    try:
        report['exit_code']=process(SimpleNamespace(out=str(output),audio=None,track_url=track['url'],unit_kind='surah',unit_number=1,device='cpu',compute_type='int8'))
        proposal=json.loads((output/'timings.proposed.json').read_text(encoding='utf-8'))
        review=json.loads((output/'review.json').read_text(encoding='utf-8'))
        report.update(verses=len(proposal['verses']),pending=len(review['pending_cases']),approved=proposal['approved'])
    except Exception as exc:
        import traceback
        report.update(exit_code=1,error=type(exc).__name__,traceback=traceback.format_exc())
    report['elapsed_seconds']=round(time.monotonic()-start,3)
    (output/'cpu-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report['exit_code']

if __name__=='__main__':raise SystemExit(main())
