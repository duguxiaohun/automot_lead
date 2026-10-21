"""Explicit build/plan/train/evaluate for geometry_v1. No implicit training."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('operation',choices=('build','plan','train','evaluate'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--collection',type=Path)
    p.add_argument('--maps',type=Path)
    p.add_argument('--scenarios',help='Comma-separated scenario names for a scoped build')
    p.add_argument('--development-manifest',type=Path,action='append',default=[],
                   help='Additional existing manifest with additional_development; repeat to union exposure')
    p.add_argument('--scoped-smoke-only',action='store_true',
                   help='Explicit partial build; artifact cannot enter formal training')
    p.add_argument('--coverage-search-states',type=int,default=100000,
                   help='Exact coverage fallback state budget; exhaustion is inconclusive, not insufficient data')
    p.add_argument('--dataset',type=Path)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--world-size',type=int,default=4,help='CPU plan only; train uses torchrun WORLD_SIZE')
    p.add_argument('--per-event',type=int,default=1024)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--max-epoch-hours',type=float,default=10.)
    p.add_argument('--max-startup-hours',type=float,default=1.,
                   help='Separate cooperative source/model/probe startup budget')
    args=p.parse_args()
    from . import dataset, sources, planning
    if args.operation=='build':
        if not all((args.collection,args.data_root,args.maps)):
            p.error('build requires --collection --data-root --maps')
        selected=set(args.scenarios.split(',')) if args.scenarios else None
        paths=[x for x in args.collection.glob('*_result.json') if not selected or x.stem.removesuffix('_result') in selected]
        if not paths:p.error('no source annotation files')
        from . import splits
        development=splits.development_contract(args.development_manifest)
        routes=dataset.annotation_routes(args.collection,args.data_root,scenarios=selected,development=development['groups'])
        _,report=dataset.build(routes,args.data_root,sources.LaneMap(args.maps),args.output,input_paths=paths,
            development_manifests=args.development_manifest,require_coverage=not args.scoped_smoke_only,
            coverage_search_states=args.coverage_search_states)
        print(json.dumps(report,ensure_ascii=False))
    else:
        if not args.dataset:p.error('--dataset required')
        if args.operation=='plan':
            artifact=dataset.load(args.dataset)
            if args.output.exists():p.error('output already exists')
            output=args.output.resolve()
            for protected in (Path(artifact['data_root']),args.dataset.resolve().parent):
                if output.is_relative_to(protected) or protected.is_relative_to(output):p.error('output overlaps input')
            plans=[]; projected={}
            for epoch in range(args.epochs):
                _,report=planning.plan(artifact['records'],per_event=args.per_event,world_size=args.world_size,
                    epochs=args.epochs,seed=args.seed+epoch,exposure=projected)
                plans.append(report)
                for key in report['ordered_ids']:projected[key]=projected.get(key,0)+1
            result=dict(plans=plans,projection_only=True,actual_exposure_advanced=False,
                warning='Later epochs reselect from successful actual exposure; projections are not consumption.')
            args.output.parent.mkdir(parents=True,exist_ok=True);dataset.write(args.output,result)
            print(json.dumps({k:v for k,v in plans[0].items() if k!='ordered_ids'}))
        else:
            if not args.model_dir:p.error('--model-dir required; local assets only')
            import math
            if not math.isfinite(args.learning_rate) or args.learning_rate<=0:p.error('invalid learning rate')
            # Configure visibility before importing torch, including direct Python callers.
            import os,subprocess
            if os.environ.get('GEOMETRY_GPU_SELECTED')!='1':
                explicit=os.environ.get('GPU_IDS')
                if explicit: selected=explicit.split(',')
                else:
                    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                        '--format=csv,noheader,nounits'],text=True)
                    available=sorted((tuple(map(int,line.split(','))) for line in text.splitlines()),key=lambda x:-x[1])
                    count=int(os.environ.get('WORLD_SIZE','1'))
                    if len(available)<count:p.error('insufficient visible GPUs')
                    selected=[str(i) for i,_ in available[:count]]
                if not selected or len(selected)!=len(set(selected)) or not all(s.isdigit() for s in selected):p.error('invalid GPU_IDS')
                os.environ['CUDA_VISIBLE_DEVICES']=','.join(selected)
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            from . import train
            if args.operation=='train':train.run(args)
            else:
                if not args.checkpoint:p.error('--checkpoint required')
                train.evaluate(args)


if __name__=='__main__':main()
