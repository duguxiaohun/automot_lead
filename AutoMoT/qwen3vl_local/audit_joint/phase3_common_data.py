"""Build once into a fixed local directory; bind a shared, label-blind reference."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys

from . import phase3_base_trial as trial

DEFAULT = trial.APP/'checkpoints/phase3_common_v1'
SOURCE = trial.APP/'qwen3vl_local/sft_new_loop_phase3'
SEED = 20260904


def source_inputs(collection, data):
    """Reject partial local mirrors rather than silently building a smaller pool."""
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _iter_routes_stream
    files = sorted(p for p in collection.glob('*_result.json') if p.stem != 'noScenarios_result')
    if not files: raise ValueError('No formal *_result.json files in '+str(collection))
    routes = 0
    for path in files:
        scenario = path.stem.removesuffix('_result')
        if not (data/scenario).is_dir():
            raise ValueError('Raw scenario directory missing: '+str(data/scenario))
        for route in _iter_routes_stream(path):
            ident = route.get('route_id')
            if not ident or route.get('status') == 'data_missing_skip': continue
            raw = data/scenario/ident
            if not raw.is_dir():
                raise ValueError('Raw route missing; refusing partial general dataset: '+str(raw))
            routes += 1
    if not routes: raise ValueError('Formal collection contains no available routes')
    return dict(collection={str(p): trial.file_sha(p) for p in files}, routes=routes)


def create_reference(index, data):
    """Preserve historical members on identical data, otherwise freeze NEW members."""
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.sft_new_loop_phase3.evaluation_balance import select_work
    from qwen3vl_local.sft_new_loop_phase3.sampling import primary_action_distribution
    old = trial.reference()
    if trial.file_sha(index) == old['index_sha256']:
        return old
    rows = {s: train._read_rows(index, s, data_root=data) for s in ('train','val','test')}
    validation = {}
    for kind, quota, seed in [('loss',16,SEED+1009),('generation',32,SEED)]:
        candidates = evaluate._balanced_cases(rows['val'], cases_per_bin=0, seed=seed, action_output_mode='choice')
        _, plan = select_work(candidates, seed=seed, limit=quota)
        if plan['per_event'] != quota: raise ValueError(f'New val pool needs {quota} distinct cases per event')
        validation[kind] = plan
    candidates = evaluate._balanced_cases(rows['test'], cases_per_bin=0, seed=SEED, action_output_mode='choice')
    _, test = select_work(candidates, seed=SEED, limit=38)
    if test['per_event'] != 38: raise ValueError('New test pool needs 38 distinct cases per event')
    projected = train._balanced_work(rows['train'],target_per_bin=1024,seed=SEED,
                                    route_diverse=True,action_output_mode='choice')
    counts = primary_action_distribution(dict(Counter(x.row.action_signature for x in projected)))['counts']
    return dict(schema='phase3_sequence_trial_reference_v1', reference_kind='new_common_dataset_v1',
        index_sha256=trial.file_sha(index), prompt_variant=old['prompt_variant'], seed=SEED,
        validation=validation, test_plan=test, baseline_training_presentations_per_epoch=counts,
        historical_scores_comparable=False, purpose='Shared fixed members for base and single-frame; no historical model scores')


def validate(root, data):
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    receipt = trial.read(root/'common_receipt.json')
    if receipt.get('schema') != 'phase3_common_dataset_v1' or receipt.get('data_root') != str(data):
        raise ValueError('Common dataset receipt/root mismatch; no implicit relocation or rebuild')
    if receipt['release_sha256'] != trial.verified_sources(): raise ValueError('Native builder sources changed')
    for name, sha in receipt['files'].items():
        if name not in ('native/frame_index.jsonl','native/manifest.json','reference.json'):
            raise ValueError('Unexpected common dataset receipt member')
        if trial.file_sha(root/name) != sha: raise ValueError('Common dataset changed: '+name)
    if set(receipt['files']) != {'native/frame_index.jsonl','native/manifest.json','reference.json'}:
        raise ValueError('Incomplete common dataset receipt')
    ref = trial.read(root/'reference.json')
    if ref['index_sha256'] != receipt['files']['native/frame_index.jsonl']:
        raise ValueError('Reference/index mismatch')
    check_index(root/'native/frame_index.jsonl',action_output_mode='choice',prompt_variant=ref['prompt_variant'])
    return receipt


def ensure(root, data, collection, run, *, workers=2, single_frame=False):
    root, data, collection = (Path(p).resolve() for p in (root,data,collection))
    for protected in (data,collection):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('Common dataset output overlaps raw input')
    if workers < 1: raise ValueError('workers must be positive')
    root.parent.mkdir(parents=True,exist_ok=True)
    # Parent directory lock covers creation too; no stale lock-file deletion.
    fd = os.open(root.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():
            if not (root/'common_receipt.json').is_file():
                raise ValueError('Partial common dataset preserved; inspect build.log/build_failure.json: '+str(root))
            receipt = validate(root,data)
            print('[common data] reuse '+str(root),flush=True)
        else:
            root.mkdir()
            try:
                release = trial.verified_sources()
                inputs = source_inputs(collection,data)
                trial.write(root/'input_bindings.json',inputs)
                cmd = [sys.executable,'-m','qwen3vl_local.sft_new_loop_phase3.build_dataset',
                    '--collection-dir',str(collection),'--data-root',str(data),'--output-dir',str(root/'native'),
                    '--review-root',str(root/'no_local_review'),'--development-groups-extra',
                    str(SOURCE/'development_route_groups_rgb_stage_20261006.json'),
                    '--workers',str(workers),'--split-seed','20260920','--test-ratio','0.10','--val-ratio','0.05',
                    '--require-invalid-true-rs-coverage']
                run('build_common',cmd,root/'build.log')
                from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
                index = root/'native/frame_index.jsonl'
                check_index(index,action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
                trial.write(root/'reference.json',create_reference(index,data))
                if release != trial.verified_sources(): raise ValueError('Sources changed during build')
                paths = sorted(p for p in collection.glob('*_result.json') if p.stem != 'noScenarios_result')
                if inputs['collection'] != {str(p):trial.file_sha(p) for p in paths}:
                    raise ValueError('Collection changed during build')
                receipt = dict(schema='phase3_common_dataset_v1',data_root=str(data),release_sha256=release,
                    files={name:trial.file_sha(root/name) for name in (
                        'native/frame_index.jsonl','native/manifest.json','reference.json')},
                    input_bindings_sha256=trial.file_sha(root/'input_bindings.json'),
                    raw_labels='unchanged native Phase3 rules',training_started=False)
                trial.write(root/'common_receipt.json',receipt)
                validate(root,data)
            except BaseException as exc:
                trial.write(root/'build_failure.json',dict(error=f'{type(exc).__name__}: {exc}'))
                raise
        if single_frame:
            from .phase3_single_frame import dataset
            path = root/'single_frame/dataset.json'
            if path.parent.exists():
                artifact = dataset.load(path,index=root/'native/frame_index.jsonl')
                if artifact['data_root'] != str(data): raise ValueError('Single-frame raw root mismatch')
            else:
                run('derive_common', [sys.executable,'-m','qwen3vl_local.audit_joint.phase3_single_frame','build',
                    '--index',str(root/'native/frame_index.jsonl'),'--data-root',str(data),
                    '--output',str(path.parent)],root/'derive_single_frame.log')
                dataset.load(path,index=root/'native/frame_index.jsonl')
        return root/'native/frame_index.jsonl', root/'reference.json'
    finally:
        os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--data-root',type=Path,default=trial.APP/'lead_data')
    p.add_argument('--collection-dir',type=Path,default=trial.APP/'keyframe_filter/collection_output')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--with-single-frame',action='store_true')
    p.add_argument('--timeout-seconds',type=int,default=172800)
    p.add_argument('--reserve-gib',type=int,default=20)
    a=p.parse_args()
    if a.timeout_seconds<=0 or a.reserve_gib<20: p.error('positive timeout and >=20 GiB reserve required')
    def run(name,cmd,log):
        print('[common data] '+name+'; log='+str(log),flush=True)
        result=trial.run_stage(cmd,log,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
            timeout=a.timeout_seconds,reserve_bytes=a.reserve_gib*1024**3)
        if result['exit_code']: raise RuntimeError(f'{name} failed: {result}')
    index,ref=ensure(a.output,a.data_root,a.collection_dir,run,workers=a.workers,single_frame=a.with_single_frame)
    print('[common data] ready '+str(index),flush=True)


if __name__=='__main__': main()
