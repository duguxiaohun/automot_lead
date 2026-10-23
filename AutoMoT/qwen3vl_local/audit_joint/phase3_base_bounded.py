"""Full-pool base with an explicit scarce-event repetition contract, version 3."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import phase3_base_coverage as coverage, phase3_base_pool as pool
from . import phase3_base_bounded_sampling as sampling
from . import phase3_base_trial as trial, phase3_common_data as common, versioned_data as cache
from . import phase3_base_hash_retry as hash_retry

APP=Path(__file__).resolve().parents[2]
POLICY='phase3_independent_base_scarce_repeat_9600_v3'
FILES=(*coverage.FILES,'phase3_base_bounded.py','phase3_base_bounded_sampling.py','phase3_base_bounded.sh',
       'phase3_base_hash_retry.py')


@contextmanager
def progress(stage):
    start=time.monotonic();done=threading.Event()
    def heartbeat():
        while not done.wait(30):
            print(f'[bounded base CPU] {stage} elapsed={time.monotonic()-start:.0f}s',flush=True)
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:yield
    finally:done.set();worker.join()


def contract():
    return dict(policy=POLICY,sources={n:trial.file_sha(Path(__file__).with_name(n)) for n in FILES},
        native_release=trial.verified_sources(),per_event=1280,epochs=3,updates=9600,warmup=2500,
        case_cap_per_epoch=dict(sufficient=1,scarce=2),frame_cap_per_epoch=2,
        case_cap_per_run='ceil(3*1280/distinct_event_frames) for scarce; 3 otherwise',
        frame_cap_per_run='case cap for single-context frames; 6 for shared-context frames',
        repeat_eligibility='scarce event or shared-context capacity deficit; exhaust each scarce event pool',
        priority='minimize physical repeats, then cumulative exposure; route/segment/action ties within equivalent capacity groups',
        holdout_selection_changed=False,labels_changed=False,input_binding_hash_policy=hash_retry.POLICY)


@contextmanager
def activate():
    previous=coverage.sampling,coverage.POLICY,coverage.contract
    coverage.sampling,coverage.POLICY,coverage.contract=sampling,POLICY,contract
    try:
        with hash_retry.activate(common,trial), coverage.activate():yield
    finally:coverage.sampling,coverage.POLICY,coverage.contract=previous


def verify_training(job):
    with activate():result=coverage.verify_training(job)
    report=trial.read(Path(job['output'])/'preflight.json')
    validate_plans(report['epoch_plans'])
    return dict(result,scarce_repetition_policy=contract())


def validate_plans(plans):
    if len(plans)!=3:raise ValueError('bounded_base_requires_three_complete_projections')
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    for p in plans:
        if (p['policy']!=sampling.POLICY or p['selected']!=12800 or p['optimizer_steps']!=3200
                or p['event_counts']!=dict.fromkeys(CONTEXT_IDS,1280)
                or p['max_frame_repeat']>2 or p['max_run_frame_repeat']>6):
            raise ValueError('bounded_base_budget_or_repetition_mismatch')
        if set(p['event_repeats'])!=set(CONTEXT_IDS):raise ValueError('missing_event_repetition_audit')
        for e,r in p['event_repeats'].items():
            n=r['available_distinct_frames'];scarce=n<1280
            epoch_cap=2 if scarce else 1;run_cap=(3840+n-1)//n if scarce else 3
            if (r['selected_distinct_frames']!=min(1280,n) or r['repeated_presentations']!=max(0,1280-n)
                    or r['case_cap_per_epoch']!=epoch_cap or r['case_cap_per_run']!=run_cap
                    or r['max_case_repeat']>epoch_cap or r['max_run_case_repeat']>run_cap
                    or r['max_frame_repeat']>2 or r['max_run_frame_repeat']>r['frame_cap_per_run']
                    or r['frame_cap_per_run']>6):
                raise ValueError('bounded_base_event_repetition_mismatch: '+e)
        if p['repeated_presentations']!=sum(r['repeated_presentations'] for r in p['event_repeats'].values())+p['shared_context_presentations']:
            raise ValueError('bounded_base_shared_frame_accounting_mismatch')


def project(index,data,reference):
    # Reuse the real native choice eligibility scan. Its immutable generation
    # contract stays unchanged; only this new admission and sampler differ.
    previous=pool.sampling
    pool.sampling=sampling
    try:report=pool.project(index,data,reference)
    finally:pool.sampling=previous
    report.pop('sha256')
    report.update(policy=POLICY,contract=contract(),generation_contract=pool.contract())
    if report['status']=='admitted':validate_plans(report['plans'])
    return dict(report,sha256=cache.digest(report))


@hash_retry.activate(common,trial)
def ensure(root,data,collection,run,*,workers=2):
    root=Path(root).resolve();data=Path(data).resolve(strict=True);collection=Path(collection).resolve()
    for protected in (data,collection):
        if root.is_relative_to(protected) or protected.is_relative_to(root):raise ValueError('bounded_base_output_overlaps_input')
    with cache.lock(root/'.cpu.lock'):
        bound=contract();generation=pool.contract()
        destination=root.parent/'phase3_base_pool_v1/versions'/cache.digest(generation)
        state=dict(schema='phase3_base_bounded_ready_v3',status='preparing',contract=bound,
                   generation_contract=generation)
        pointer=root/'ready_1280.json';cache.write(pointer,state)
        try:
            def native_run(name,cmd,log):
                cmd=list(cmd);module='qwen3vl_local.sft_new_loop_phase3.build_dataset'
                if name!='build_common' or module not in cmd:raise ValueError('unexpected_base_pool_build_stage')
                i=cmd.index(module);cmd[i:i+1]=['qwen3vl_local.audit_joint.phase3_base_pool','build']
                run(name,cmd,log)
            print('[bounded base CPU] validating/reusing full native pool and raw inputs',flush=True)
            with progress('native pool/source verification'):
                index,reference=common.ensure(destination,data,collection,native_run,workers=workers)
            if trial.read(index.with_name('manifest.json')).get('base_pool_contract')!=generation:
                raise ValueError('bounded_base_parent_generation_mismatch')
            print('[bounded base CPU] projecting three epochs with actual eligible frames',flush=True)
            with progress('three-epoch projection'):
                report=project(index,data,reference)
            path=root/'admissions'/(report['sha256']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and trial.read(path)!=report:raise ValueError('bounded_base_admission_changed')
            cache.write(path,report)
            state.update(index=str(index),reference=str(reference),index_sha256=report['index_sha256'],
                reference_sha256=report['reference_sha256'],admission=str(path),admission_sha256=report['sha256'])
            if report['status']!='admitted':raise ValueError('bounded_base_capacity_rejected: '+report['error'])
            if contract()!=bound or pool.contract()!=generation:raise ValueError('bounded_base_sources_changed_during_prepare')
            state.update(status='ready',optimizer_steps=[p['optimizer_steps'] for p in report['plans']])
            cache.write(pointer,state)
            return index,reference
        except BaseException as exc:
            state.update(status='rejected',error=f'{type(exc).__name__}: {exc}');cache.write(pointer,state);raise


@hash_retry.activate(common,trial)
def load_ready(root):
    pointer=Path(root)/'ready_1280.json'
    if not pointer.is_file():raise ValueError('bounded base CPU admission missing; run phase3_compare.sh cpu base first')
    ready=trial.read(pointer)
    if (ready.get('schema')!='phase3_base_bounded_ready_v3' or ready.get('status')!='ready'
            or ready.get('contract')!=contract() or ready.get('generation_contract')!=pool.contract()
            or ready.get('optimizer_steps')!=[3200]*3):raise ValueError('bounded_base_ready_contract_mismatch')
    index=Path(ready['index']);reference=Path(ready['reference'])
    if trial.file_sha(index)!=ready['index_sha256'] or trial.file_sha(reference)!=ready['reference_sha256']:
        raise ValueError('bounded_base_admitted_data_changed')
    manifest=trial.read(index.with_name('manifest.json'))
    if manifest.get('base_pool_contract')!=pool.contract():raise ValueError('bounded_base_parent_changed')
    common.validate(reference.parent,Path(manifest['data_root']).resolve(strict=True))
    report=trial.read(ready['admission'])
    if (report.get('status')!='admitted' or report.get('contract')!=contract()
            or report.get('generation_contract')!=pool.contract()
            or report['sha256']!=ready['admission_sha256']
            or cache.digest({k:v for k,v in report.items() if k!='sha256'})!=report['sha256']
            or report['index_sha256']!=ready['index_sha256'] or report['reference_sha256']!=ready['reference_sha256']):
        raise ValueError('bounded_base_admission_binding_changed')
    validate_plans(report['plans']);return ready


def wrap(cmd):
    cmd=list(cmd)
    module=next((v for v in cmd if v in ['qwen3vl_local.audit_joint.'+n for n in coverage.ALLOWED]),None)
    if module is None:raise ValueError('Unsupported bounded base worker')
    i=cmd.index(module)
    return cmd[:i]+['qwen3vl_local.audit_joint.phase3_base_bounded','worker','--module',
                   module.removeprefix('qwen3vl_local.audit_joint.'),'--']+cmd[i+1:]


def main():
    if sys.argv[1:2]==['worker']:
        p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--module',choices=coverage.ALLOWED,required=True)
        p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args(sys.argv[2:])
        with activate():
            sys.argv=[a.module,*(a.args[1:] if a.args[:1]==['--'] else a.args)]
            importlib.import_module('qwen3vl_local.audit_joint.'+a.module).main()
        return 0
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('operation',choices=('cpu','check'))
    p.add_argument('--root',type=Path,default=APP/'checkpoints/phase3_base_bounded_v3')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--collection-dir',type=Path,default=APP/'keyframe_filter/collection_output')
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    from . import phase3_native_runtime
    if phase3_native_runtime.needed():return phase3_native_runtime.execute('phase3_base_bounded',sys.argv[1:])
    if a.operation=='check':print(json.dumps(load_ready(a.root)),flush=True);return 0
    def run(name,cmd,log):
        result=trial.run_stage(cmd,log,dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=172800,reserve_bytes=20*1024**3)
        if result['exit_code']:raise RuntimeError(f'{name} failed: {result}')
    index,reference=ensure(a.root,a.data_root,a.collection_dir,run,workers=a.workers)
    print(json.dumps(dict(status='ready',index=str(index),reference=str(reference))),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
