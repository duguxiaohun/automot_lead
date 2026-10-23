"""Full native training candidates for coverage base; fixed holdout rules retained.

The historical common index balances events before the epoch sampler. That
truncates otherwise usable training frames. This independent version only
changes train index selection; native labels, holdouts and INVALID stay native.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from . import phase3_common_data as common, phase3_base_trial as trial
from . import phase3_base_coverage_sampling as sampling, versioned_data as cache

APP = Path(__file__).resolve().parents[2]
POLICY = 'native_full_train_candidates_base_v1'


def contract():
    return dict(policy=POLICY, sources={name:trial.file_sha(Path(__file__).with_name(name))
        for name in ('phase3_base_pool.py', 'phase3_base_coverage_sampling.py',
                     'phase3_common_data.py', 'phase3_common_runtime.py')},
        native_release=trial.verified_sources(), per_event=1280, epochs=3,
        seed=20260904, world_size=4, holdout_selection_changed=False)


@contextmanager
def full_train_selection():
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as native
    original = native._sample_context_bucket
    observed = {}
    def select(bucket, *, context_id, target, rng, route_diverse):
        splits = {r['split'] for r in bucket}
        if len(splits) > 1: raise ValueError('mixed_split_sampling_bucket')
        requested = target
        if splits == {'train'}:
            target = len(bucket)
            observed[context_id] = dict(legacy_target=requested, full_candidates=target)
        rows, report = original(bucket, context_id=context_id, target=target,
                                rng=rng, route_diverse=route_diverse)
        if splits == {'train'}:
            report.update(index_selection=POLICY, legacy_target=requested,
                          full_candidates=target, epoch_quota_applied=False)
        return rows, report
    native._sample_context_bucket = select
    try: yield observed
    finally: native._sample_context_bucket = original


def build():
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as native
    args = native.parse_args()
    if args.candidate_cache or args.use_review_cache or args.max_routes or args.max_routes_per_scenario:
        raise ValueError('formal_base_pool_requires_full_native_scan')
    bound = contract()
    with full_train_selection() as observed:
        manifest = native.build_dataset(args)
    if contract() != bound: raise ValueError('base_pool_sources_changed_during_build')
    manifest['base_pool_contract'] = bound
    manifest['balance_contract'] = 'full native train candidates; native val/test and INVALID selection; event quotas at epoch sampling'
    manifest['sampling']['balance']['train']['target_per_context'] = None
    manifest['sampling']['balance']['train']['index_selection'] = POLICY
    manifest['base_pool_candidates'] = observed
    trial.write(Path(args.output_dir)/'manifest.json', manifest)
    print('[base pool] full training candidates: '+json.dumps(observed), flush=True)


def project(index, data, reference):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import unique_cases
    ref = trial.read(reference)
    checked = check_index(index, action_output_mode='choice', prompt_variant=ref['prompt_variant'])
    rows = train._read_rows(index, 'train', data_root=data)
    items = unique_cases([train._make_item(r, seed=20260904, action_output_mode='choice') for r in rows])
    eligible = [x for x in items if train.choice_rejection_reason(x.spec) is None]
    supply = {e:dict(eligible_anchors=sum(x.row.context_id==e for x in eligible),
        physical_frames=len({(x.row.scenario,x.row.route_id,x.row.frame_id) for x in eligible if x.row.context_id==e}))
        for e in train.CONTEXT_IDS}
    args = SimpleNamespace(batch_size=1, grad_accum=1, action_output_mode='choice',
                           focus_balance_count=1280, seed=20260904)
    result = dict(policy=POLICY, status='rejected', index_sha256=trial.file_sha(index),
        reference_sha256=trial.file_sha(reference), contract=contract(), supply=supply,
        dataset=checked, projection_only=True, actual_exposure_advanced=False,
        model_inputs_checked=False, GPU_executed=False, plans=[])
    try:
        history = None
        for epoch in range(3):
            _, plan = sampling.independent_work(rows,args,epoch=epoch,world_size=4,history=history)
            history = plan['next_history']
            result['plans'].append({k:v for k,v in plan.items() if k!='next_history'})
            print(f'[base capacity] epoch={epoch+1} updates={plan["optimizer_steps"]} unseen={plan["unseen_physical_frames"]}', flush=True)
        result['status'] = 'admitted'
    except ValueError as exc:
        result['error'] = str(exc)
    return dict(result, sha256=cache.digest(result))


def ensure(root, data, collection, run, *, workers=2):
    root = Path(root).resolve(); data=Path(data).resolve(strict=True)
    collection=Path(collection).resolve()
    for protected in (data,collection):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('base_pool_output_overlaps_input')
    with cache.lock(root/'.cpu.lock'):
        bound = contract(); destination = root/'versions'/cache.digest(bound)
        pointer = root/'ready_1280.json'
        state = dict(schema='phase3_base_pool_ready_v1', status='preparing', contract=bound)
        cache.write(pointer,state)
        try:
            def native_run(name,cmd,log):
                cmd=list(cmd)
                module='qwen3vl_local.sft_new_loop_phase3.build_dataset'
                if name!='build_common' or module not in cmd:
                    raise ValueError('unexpected_base_pool_build_stage')
                i=cmd.index(module)
                cmd[i:i+1]=['qwen3vl_local.audit_joint.phase3_base_pool','build']
                run(name,cmd,log)
            index,reference=common.ensure(destination,data,collection,native_run,workers=workers)
            if trial.read(index.with_name('manifest.json')).get('base_pool_contract')!=bound:
                raise ValueError('base_pool_generation_contract_mismatch')
            report=project(index,data,reference)
            path=root/'admissions'/(report['sha256']+'.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and trial.read(path)!=report:raise ValueError('base_pool_admission_changed')
            cache.write(path,report)
            state.update(index=str(index), reference=str(reference), index_sha256=report['index_sha256'],
                reference_sha256=report['reference_sha256'], admission=str(path), admission_sha256=report['sha256'])
            if report['status']!='admitted':raise ValueError('base_pool_capacity_rejected: '+report['error'])
            if contract()!=bound:raise ValueError('base_pool_sources_changed_during_prepare')
            state.update(status='ready', optimizer_steps=[p['optimizer_steps'] for p in report['plans']])
            cache.write(pointer,state)
            return index,reference
        except BaseException as exc:
            state.update(status='rejected',error=f'{type(exc).__name__}: {exc}')
            cache.write(pointer,state)
            raise


def main():
    operation=sys.argv[1] if len(sys.argv)>1 else ''
    if operation=='build':
        sys.argv.pop(1);build();return 0
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('operation',choices=('cpu','check'))
    p.add_argument('--root',type=Path,default=APP/'checkpoints/phase3_base_pool_v1')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--collection-dir',type=Path,default=APP/'keyframe_filter/collection_output')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    from . import phase3_native_runtime
    if phase3_native_runtime.needed():
        return phase3_native_runtime.execute('phase3_base_pool',sys.argv[1:])
    if a.operation=='check':
        print(json.dumps(load_ready(a.root)),flush=True)
        return 0
    def run(name,cmd,log):
        result=trial.run_stage(cmd,log,dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            timeout=172800,reserve_bytes=20*1024**3)
        if result['exit_code']:raise RuntimeError(f'{name} failed: {result}')
    index,reference=ensure(a.root,a.data_root,a.collection_dir,run,workers=a.workers)
    print(json.dumps(dict(status='ready',index=str(index),reference=str(reference))),flush=True)
    return 0


def load_ready(root):
    pointer=Path(root)/'ready_1280.json'
    if not pointer.is_file():
        raise ValueError('base CPU admission missing; run phase3_compare.sh cpu base first')
    ready=trial.read(pointer)
    if (ready.get('schema')!='phase3_base_pool_ready_v1' or ready.get('status')!='ready'
            or ready.get('contract')!=contract() or ready.get('optimizer_steps')!=[3200]*3):
        raise ValueError('base CPU admission is not ready for current sources/budget')
    index=Path(ready['index']);reference=Path(ready['reference'])
    if trial.file_sha(index)!=ready['index_sha256'] or trial.file_sha(reference)!=ready['reference_sha256']:
        raise ValueError('base CPU admitted data changed')
    manifest=trial.read(index.with_name('manifest.json'))
    common.validate(reference.parent,Path(manifest['data_root']).resolve(strict=True))
    report=trial.read(ready['admission'])
    if (manifest.get('base_pool_contract')!=contract() or report.get('status')!='admitted'
            or report.get('contract')!=contract() or report['sha256']!=ready['admission_sha256']
            or cache.digest({k:v for k,v in report.items() if k!='sha256'})!=report['sha256']
            or report['index_sha256']!=ready['index_sha256']
            or report['reference_sha256']!=ready['reference_sha256']):
        raise ValueError('base CPU admission binding changed')
    if (len(report['plans'])!=3 or any(p['selected']!=12800 or p['optimizer_steps']!=3200
            or p['max_frame_repeat']!=1 or p['max_run_frame_repeat']>3
            for p in report['plans'])):
        raise ValueError('base CPU admission budget changed')
    return ready


if __name__=='__main__':raise SystemExit(main())
