import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Opt-in Phase1 visual calibration and paired prompt audit; never trains automatically')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('derive')
    for name in ('parent-index','reviews','data-root','output'): d.add_argument('--'+name, required=True)
    for command in ('prepare','replay'):
        q = sub.add_parser(command)
        for name in ('inputs','data-root','output'): q.add_argument('--'+name, required=True)
        if command == 'replay':
            q.add_argument('--model-dir', required=True); q.add_argument('--adapter-dir', required=True)
            q.add_argument('--device', default='cuda:0')
    q = sub.add_parser('compare')
    for name in ('inputs','baseline','candidate','output'):q.add_argument('--'+name,required=True)
    args = p.parse_args(); kwargs = vars(args).copy(); command = kwargs.pop('command')
    if command == 'derive':
        from .calibration import derive
        report = derive(**kwargs)
    elif command == 'replay':
        if args.device != 'cpu':
            import os
            from .remote import choose_gpu
            if args.device != 'cuda:0': raise ValueError('Use GPU_IDS to select one physical GPU; device must be cuda:0 or cpu')
            gpu = choose_gpu(); os.environ['GPU_IDS'] = gpu; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        from .experiment import replay
        report = replay(**kwargs)
    else:
        output=Path(kwargs.pop('output'))
        if output.exists(): raise ValueError('Output already exists')
        if command == 'prepare':
            from .experiment import prepare
            _, _, report = prepare(**kwargs)
        else:
            from .regression import compare, read_cases
            report = compare(json.loads(Path(args.inputs).read_text()), read_cases(args.baseline), read_cases(args.candidate))
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','changed_rows','total','joint_correct') if k in report},ensure_ascii=False))
    if report['status'] == 'review_required': raise SystemExit(2)


if __name__ == '__main__': main()
