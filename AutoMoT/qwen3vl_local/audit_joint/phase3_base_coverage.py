"""Explicit 9600-update independent base adapter around frozen native sources.

No sequence selection, geometric labels, old-checkpoint continuation or source
hash override. Every worker and saved native checkpoint binds this adapter.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import sys

from . import phase3_base_trial as trial, phase3_base_trial_worker as base
from . import phase3_base_coverage_sampling as sampling
from .phase3_base_dense import baseline

POLICY = 'phase3_independent_base_coverage_9600_v2'
FILES = ('phase3_base_coverage.py', 'phase3_base_coverage_sampling.py',
         'phase3_launch.py', 'phase3_launch_worker.py', 'phase3_common_runtime.py',
         'phase3_common_data.py', 'phase3_native_runtime.py', 'phase3_base_pool.py')


def contract():
    return dict(policy=POLICY, sources={n:trial.file_sha(Path(__file__).with_name(n)) for n in FILES},
                per_event=1280, epochs=3, updates=9600, warmup=2500,
                frame_cap_per_epoch=1, frame_cap_per_run=3)


def adapter_files(folder):
    return {str(p.relative_to(folder)):trial.file_sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name!='coverage_receipt.json'}


def preflight(job):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index, check_model
    args=base.make_args(job)
    report=dict(dataset=check_index(args.index,action_output_mode='choice',prompt_variant=args.prompt_variant),
                model=check_model(args.model_dir), coverage_contract=contract(), profile=trial.profile())
    rows=train._read_rows(Path(args.index),'train',data_root=Path(args.data_root))
    history=None; plans=[]
    for epoch in range(3):
        _,plan=sampling.independent_work(rows,args,epoch=epoch,world_size=4,history=history)
        history=plan['next_history']; plans.append({k:v for k,v in plan.items() if k!='next_history'})
        print(f'[base capacity] epoch={epoch+1} updates={plan["optimizer_steps"]} '
              f'unseen={plan["unseen_physical_frames"]} max_repeat={plan["max_frame_repeat"]}',flush=True)
    report.update(epoch_plans=plans, projection_only=True, actual_exposure_advanced=False)
    trial.write(Path(job['output'])/'preflight.json',report)
    return report


def verify_training(job):
    root=Path(job['output'])/'train'
    manifest=trial.read(root/'train_run_manifest.json'); schedule=manifest['learning_rate_schedule']
    if (manifest['total_steps_global']!=9600 or manifest['batch_size']!=1 or manifest['world_size']!=4
        or manifest['train_work_global']!=12800 or schedule['total_optimizer_updates']!=9600
        or schedule['warmup_updates']!=2500):
        raise ValueError('Training did not use the 9600-update coverage plan')
    if job.get('coverage_contract')!=contract():raise ValueError('Coverage source contract changed')
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    expected=dict.fromkeys(CONTEXT_IDS,1280);expected['INVALID']=0
    projection=trial.read(Path(job['output'])/'preflight.json')
    if projection['coverage_contract']!=contract():raise ValueError('Projection contract changed')
    exposure=Counter()
    for epoch in range(1,4):
        consumed=trial.read(root/f'balance/consumed_epoch_{epoch:04d}.json')
        planned=trial.read(root/f'balance/plan_epoch_{epoch:04d}.json')
        if consumed['status']!='verified' or consumed['actual']!=expected:
            raise ValueError('Actual event consumption mismatch')
        if planned!=projection['epoch_plans'][epoch-1]:raise ValueError('Actual sampling differs from CPU projection')
        ids=planned['ordered_ids']
        for rank in range(4):
            path=root/f'epoch_{epoch:04d}_rank{rank}_batches.jsonl'
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            if len(rows)!=3200:raise ValueError('Incomplete rank consumption')
            for step,row in enumerate(rows):
                position=4*step+rank
                if (row['epoch']!=epoch-1 or row['step']!=step or row['batch_size']!=1
                    or row['positions']!=[position] or row['ids']!=[ids[position]]
                    or not math.isfinite(row['loss'])):
                    raise ValueError('Rank consumption differs from admitted plan')
        exposure.update(ids)
        saved=trial.read(root/f'balance/exposure_epoch_{epoch:04d}.json')
        if saved!=dict(policy=sampling.POLICY,next_epoch=epoch,exposure=dict(exposure)):
            raise ValueError('Completed-epoch exposure mismatch')
    cfg=trial.read(root/'final/sft_new_loop_phase3_adapter_config.json')
    if cfg['global_step']!=9600 or cfg['sampling_policy']!=sampling.POLICY:
        raise ValueError('Final adapter budget or sampling mismatch')
    receipt=trial.read(root/'final/coverage_receipt.json')
    if (receipt['coverage_contract']!=contract() or receipt['files']!=adapter_files(root/'final')
        or receipt['index_sha256']!=job['index_sha256']):
        raise ValueError('Final adapter files or coverage binding changed')
    return dict(completed_epochs=3,optimizer_updates=9600,planned_optimizer_updates=9600,
                verified_rank_presentations=38400,coverage_contract=contract(),quality_pending=True)


@contextmanager
def activate():
    """Only process-local hooks; original native files and release stay intact."""
    from qwen3vl_local.sft_new_loop_phase3 import train
    saved=[]
    def patch(obj,name,value):
        saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    old_profile=trial.profile; old_job=trial.build_job; old_validate=base.validate_job
    old_args=base.make_args; old_install=base.install_baseline; old_save=train._save_adapter
    def profile():
        return dict(old_profile(), focus_balance_count=1280,presentations_per_epoch=12800,
                    steps_per_epoch=3200,optimizer_updates=9600,total_presentations=38400,
                    warmup_steps=2500,sampling_policy=sampling.POLICY,coverage_policy=POLICY)
    def build_job(args,output):
        if args.resume or args.stage!='complete' or getattr(args,'experiment','base')!='base':
            raise ValueError('Coverage base requires a new complete base run; no resume/dense')
        job=old_job(args,output)
        job.update(native_policy=job['policy'],policy=POLICY,coverage_contract=contract())
        return job
    def validate(job):
        old_validate(dict(job,policy=job.get('native_policy')))
        if (job.get('policy')!=POLICY or job.get('coverage_contract')!=contract() or job['resume'] or job['stage']!='complete'
            or job.get('mode')!='base'):
            raise ValueError('Coverage job/source/mode mismatch')
    def make_args(job):
        if job.get('coverage_contract')!=contract():raise ValueError('Coverage job contract changed')
        args=old_args(job)
        args.focus_balance_count=1280;args.warmup_steps=2500
        args.eval_steps=args.generation_eval_steps=3200
        args.coverage_contract=contract()
        args.base_trial.update(native_policy=args.base_trial['policy'],policy=POLICY)
        return args
    def install():
        old_install();train._contiguous_work=sampling.independent_work;train.SEQUENCE_POLICY=sampling.POLICY
    def save(bundle,output_dir,args,*,step,name='final'):
        if args.coverage_contract!=contract():raise ValueError('Coverage sources changed before saving')
        folder=old_save(bundle,output_dir,args,step=step,name=name)
        trial.write(folder/'coverage_receipt.json',dict(coverage_contract=contract(),
            index_sha256=args.training_index_sha256,files=adapter_files(folder),global_step=step))
        return folder
    try:
        patch(trial,'profile',profile);patch(trial,'build_job',build_job);patch(trial,'verify_training',verify_training)
        patch(base,'validate_job',validate);patch(base,'make_args',make_args);patch(base,'preflight',preflight)
        patch(base,'independent_work',sampling.independent_work);patch(base,'install_baseline',install)
        patch(baseline,'independent_work',sampling.independent_work)
        patch(train,'_save_adapter',save)
        # Also restore native globals after worker execution or an exception.
        patch(train,'_contiguous_work',train._contiguous_work)
        patch(train,'_loss_batch',train._loss_batch);patch(train,'SEQUENCE_POLICY',train.SEQUENCE_POLICY)
        yield
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


ALLOWED=('phase3_launch_worker','phase3_base_trial_worker','phase3_common_runtime')

def wrap(cmd):
    cmd=list(cmd)
    module=next((v for v in cmd if v in ['qwen3vl_local.audit_joint.'+n for n in ALLOWED]),None)
    if module is None:raise ValueError('Unsupported coverage worker')
    i=cmd.index(module)
    return cmd[:i]+['qwen3vl_local.audit_joint.phase3_base_coverage','--module',
                   module.removeprefix('qwen3vl_local.audit_joint.'),'--']+cmd[i+1:]


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--module',choices=ALLOWED,required=True);p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    with activate():
        sys.argv=[a.module,*(a.args[1:] if a.args[:1]==['--'] else a.args)]
        importlib.import_module('qwen3vl_local.audit_joint.'+a.module).main()


if __name__=='__main__':main()
