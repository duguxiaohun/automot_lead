"""本地训练审计包：指标、错误题与源码合同；不含权重、原始大图或本机聊天。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from pathlib import Path
import zipfile


def create(run,output,max_bytes=30_000_000):
    run,output=Path(run).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    files=sorted([p for p in run.glob('*.json') if p.is_file()]+[p for p in run.glob('*cases.jsonl') if p.is_file()])
    files += [p for name in ('metrics.json','cases.jsonl') if (p:=run/'test'/name).is_file()]
    if not files:
        raise ValueError('no audit records')
    pending=output.with_suffix('.pending')
    try:
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:
                if p.resolve() != p or not p.is_relative_to(run):
                    raise ValueError('external audit symlink rejected')
                z.write(p,str(p.relative_to(run)))
        if pending.stat().st_size>max_bytes:
            raise ValueError('audit archive exceeds size budget; metrics were not silently dropped')
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    create(a.run,a.output)
    print(f"[Phase4 audit] archive={a.output.resolve()} bytes={a.output.stat().st_size}", flush=True)

if __name__=='__main__':
    main()
