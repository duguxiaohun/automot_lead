import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description='Independent v2 candidate; never trains or promotes automatically')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('derive')
    for name in ('parent-index','reviews','data-root','output'):
        d.add_argument('--'+name, required=True)
    for cmd in ('prepare','replay'):
        q = sub.add_parser(cmd)
        for name in ('inputs','data-root','reference','output','candidate'):
            q.add_argument('--'+name, required=True)
        if cmd == 'replay':
            for name in ('model-dir','adapter-dir'):
                q.add_argument('--'+name, required=True)
            q.add_argument('--device', default='cuda:0')
    a = vars(p.parse_args()); cmd = a.pop('command')
    if cmd == 'derive':
        from .calibration import derive
        report = derive(**a)
    elif cmd == 'prepare':
        output = Path(a.pop('output'))
        if output.exists():
            raise ValueError('Output already exists')
        from .experiment import prepare
        *_, report = prepare(**a)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
    else:
        if a['device'] != 'cpu':
            if a['device'] != 'cuda:0':
                raise ValueError('Use GPU_IDS for physical GPU, device=cuda:0')
            import os
            from qwen3vl_local.audit_joint.phase1_visual_refine.remote import choose_gpu
            gpu = choose_gpu(); os.environ['GPU_IDS']=gpu; os.environ['CUDA_VISIBLE_DEVICES']=gpu
        from .experiment import replay
        report = replay(**a)
        from .remote import package
        print('[BUNDLE]', package(a['output'], a['inputs']))
    print(json.dumps({k:report[k] for k in ('status','candidate','total','joint_correct') if k in report}))
    if report['status'] == 'review_required':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
