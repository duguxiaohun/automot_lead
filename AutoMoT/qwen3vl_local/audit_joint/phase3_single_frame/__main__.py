"""CPU dataset build/audit entry. GPU worker integration is separately tested.

No implicit training, source synchronization, dataset rebuild or model download.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
from . import dataset, planning, history
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.sft_new_loop_phase3 import train


def audit(rows,artifact,output,*,count=1024):
    cfg=SimpleNamespace(batch_size=1,grad_accum=1,action_output_mode='choice',
                        seed=20260904,focus_balance_count=count)
    report=dict(epochs={},evaluation={},training_executed=False,
                profile=dict(epochs=3,world_size=4,batch_size=1,grad_accum=1,
                             per_event=count,optimizer_updates=3*10*count//4))
    history=None
    for epoch in range(3):
        work,plan=planning.plan(rows,artifact['records'],cfg,epoch=epoch,history=history,strict=False)
        report['epochs'][str(epoch)]=plan;history=plan['next_history']
    for split in ('val','test'):
        report['evaluation'][split]=planning.evaluation_coverage(
            r for r in artifact['records'].values() if r['split']==split)
    report['status']='cpu_sampling_passed' if all(not p['errors'] for p in report['epochs'].values()) else 'cpu_sampling_blocked'
    report['scope']='Dataset members only; production fixed evaluation/resource admission remains separate'
    trial.write(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('build','audit'))
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--history-policy',choices=('offline_native',),default='offline_native')
    parser.add_argument('--history-fill',choices=history.FILL_MODES,default=history.DEFAULT_FILL)
    parser.add_argument('--lateral-weight',type=float,default=1.5)
    args=parser.parse_args()
    output=args.output.resolve()
    for p in (args.data_root,args.index.parent):
        if output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output):
            raise ValueError('Output overlaps protected input')
    if output.exists():raise ValueError('Output exists; choose an explicit new directory/file')
    sha=trial.file_sha(args.index)
    rows=[r for split in ('train','val','test') for r in train._read_rows(args.index,split,data_root=args.data_root)]
    if sha!=trial.file_sha(args.index):raise ValueError('Index changed while reading')
    if args.operation=='build':
        artifact,report=dataset.build(rows,args.data_root,output,history_policy=args.history_policy,
                                      lateral_weight=args.lateral_weight,index=args.index,history_fill=args.history_fill)
        if artifact['index_sha256']!=sha:raise ValueError('Index changed during generation')
        print(report['status'])
    else:
        if not args.dataset:raise ValueError('--dataset is required for audit; no implicit generation')
        artifact=dataset.load(args.dataset,index=args.index)
        if Path(artifact['data_root'])!=args.data_root.resolve():raise ValueError('Data root mismatch')
        report=audit(rows,artifact,output)
        print(report['status'])
        if report['status']!='cpu_sampling_passed':raise SystemExit(2)


if __name__=='__main__':main()
