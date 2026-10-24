"""Worker CLI. The outer pipeline supplies the verified dataset and GPU selection."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('operation',choices=('train','evaluate','probe','validate'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--max-startup-hours',type=float,default=4.)
    p.add_argument('--max-epoch-hours',type=float,default=10.)
    args=p.parse_args()
    args.variant='six';args.per_event=1280;args.epochs=3;args.seed=20260904;args.learning_rate=1e-5
    args.probe=args.operation=='probe'
    from . import train
    if args.operation=='validate':
        from .validation import run
        run(args)
    elif args.operation=='evaluate':
        if not args.checkpoint:p.error('checkpoint required')
        train.evaluate(args)
    else:train.run(args)

if __name__=='__main__':main()
