"""Independent six-action build, budgeted planning, training and diagnosis."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('operation',choices=('build','plan','train','evaluate','trajectory-metrics'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',type=Path)
    p.add_argument('--maps',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--variant',choices=('six',))
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--per-event',type=int,default=1280)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--world-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260904)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--max-epoch-hours',type=float,default=10.)
    p.add_argument('--max-startup-hours',type=float,default=1.)
    args=p.parse_args()
    from . import dataset, planning
    if args.operation=='build':
        if not args.geometry or not args.maps: p.error('build requires --geometry --maps')
        _,report=dataset.build(args.geometry,args.maps,args.output)
        print(json.dumps({k:v for k,v in report.items() if k!='support'})); return
    if not args.dataset: p.error('--dataset required')
    if args.operation=='trajectory-metrics':
        from .expert import matched_metrics
        report=matched_metrics(json.loads(args.dataset.read_text()))
        if args.output.exists(): p.error('output exists')
        args.output.parent.mkdir(parents=True,exist_ok=True); dataset.write(args.output,report); return
    if args.operation=='plan':
        artifact=dataset.load(args.dataset)
        from . import admission
        admission.require(admission.inspect(artifact))
        for protected in (Path(artifact['data_root']),args.dataset.resolve().parent):
            if args.output.resolve().is_relative_to(protected) or protected.is_relative_to(args.output.resolve()):
                p.error('plan output overlaps inputs')
        if args.output.exists(): p.error('output exists')
        report=planning.project(artifact['records'],per_event=args.per_event,epochs=args.epochs,
                                world_size=args.world_size,seed=args.seed)
        args.output.parent.mkdir(parents=True,exist_ok=True); dataset.write(args.output,report)
        print(json.dumps(dict(shared_plan_sha256=report['sha256'],epochs=args.epochs,
            optimizer_steps=[x['optimizer_steps'] for x in report['plans']],
            unseen_frames=[x['unseen_physical_frames'] for x in report['plans']]))); return
    if not args.model_dir or not args.variant:
        p.error('train/evaluate require --variant --model-dir')
    if args.operation == 'evaluate' and not args.checkpoint: p.error('evaluation requires --checkpoint')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0: p.error('invalid_learning_rate')
    # Shell wrapper selects one shared GPU set before torchrun. Direct evaluation
    # also supports automatic GPU selection without user-entered absolute paths.
    if os.environ.get('STUDY_GPU_SELECTED')!='1':
        import subprocess
        if int(os.environ.get('WORLD_SIZE','1'))>1: p.error('distributed training requires run.sh for shared GPU selection')
        explicit=os.environ.get('GPU_IDS')
        if explicit: selected=explicit.split(',')
        else:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            available=sorted((tuple(map(int,line.split(','))) for line in lines.splitlines()),key=lambda x:-x[1])
            selected=[str(available[0][0])] if available else []
        if not selected or len(set(selected))!=len(selected) or not all(x.isdigit() for x in selected): p.error('invalid_GPU_IDS')
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(selected)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from . import train
    if args.operation=='train': train.run(args)
    else: train.evaluate(args)


if __name__=='__main__': main()
