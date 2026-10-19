"""Generate one shared native supplemental index; preserve frozen base bytes/splits."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace

from . import phase3_base_trial as trial

POLICY = 'phase3_base_a_b_shared_native_data_v1'


def row_key(r):
    return tuple(r[k] for k in ('scenario','route_id','split','context_id','prompt_road_structure','frame_id'))


def requests(rows, offsets):
    """Targets only; auxiliary labels never enlarge the frozen anchor population."""
    wanted=set(); routes={}
    for r in rows:
        route=(r['scenario'],r['route_id'])
        if any(Path(v).name!=v or v in ('.','..') for v in route):
            raise ValueError('Invalid route path')
        if route in routes and routes[route]!=r['split']:raise ValueError('Route crosses splits')
        routes[route]=r['split']
        if r['invalid_action_context']:continue
        key=row_key(r)
        for delta in set(range(-4,1))|set(offsets):
            if r['frame_id']+delta>=0:wanted.add((*key[:-1],r['frame_id']+delta))
    return wanted,routes


@contextmanager
def frozen_routes(builder,routes):
    """Limit native trajectory work to frozen routes, keeping original annotations."""
    original=builder._source_routes;split=builder._split
    def filtered(collection,scenario,args):
        for route in original(collection,scenario,args):
            if (scenario,str(route.get('route_id',''))) in routes:yield route
    builder._source_routes=filtered
    builder._split=lambda scenario,route,*a:routes[scenario,route]
    try:yield
    finally:builder._source_routes=original;builder._split=split


def added_rows(rows,candidates,offsets):
    """Accept actual native mapped contexts, never copy an anchor to another frame."""
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
    wanted,_=requests(rows,offsets)
    seen={row_key(r) for r in rows if not r['invalid_action_context']}
    for base in candidates:
        k=(base['scenario'],base['route_id'],base['split'],base['context_id'],base['rs'],base['frame_id'])
        if k not in wanted or k in seen:continue
        row=builder._make_row(base=base,context_id=base['context_id'],invalid=False)
        row.update(choice_annotation(row['answers'],row['context_id'],row['action_evidence']))
        seen.add(k)
        yield row


def verify(directory, *, data_root, offsets, release):
    directory=Path(directory).resolve(strict=True)
    receipt=trial.read(directory/'shared_data_receipt.json')
    if (receipt['policy']!=POLICY or receipt['release_sha256']!=release
            or receipt['offset_frames']!=list(offsets)
            or receipt['data_root']!=str(Path(data_root).resolve(strict=True))):
        raise ValueError('Prepared data protocol/source/root mismatch')
    for relative,sha in receipt['files'].items():
        path=directory/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or trial.file_sha(path)!=sha:
            raise ValueError('Prepared data changed: '+relative)
    if receipt['files']['base/frame_index.jsonl']!=trial.reference()['index_sha256']:
        raise ValueError('Prepared base is not the frozen historical index')
    return receipt


def build(job):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    root=Path(job['output']);index=Path(job['index']);data=Path(job['data_root']).resolve(strict=True)
    collection=Path(job['collection_dir']).resolve(strict=True)
    if trial.file_sha(index)!=trial.reference()['index_sha256']:raise ValueError('Frozen index SHA mismatch')
    check_index(index,action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
    rows=[json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    wanted,routes=requests(rows,job['offset_frames'])
    source_paths=[collection/f'{s}_result.json' for s,_ in routes]
    sources={str(p):trial.file_sha(p) for p in sorted(set(source_paths))}
    root.mkdir(parents=True,exist_ok=False)
    (root/'base').mkdir();(root/'auxiliary').mkdir()
    # Base row bytes and IDs are immutable. Only its original manifest is copied.
    shutil.copyfile(index,root/'base/frame_index.jsonl')
    shutil.copyfile(index.with_name('manifest.json'),root/'base/manifest.json')
    args=SimpleNamespace(collection_dir=str(collection),data_root=str(data),workers=0,
        scenarios=','.join(sorted({s for s,_ in routes})),use_review_cache=False,candidate_cache='',
        split_seed=20260920,test_ratio=.1,val_ratio=.05,progress_every_routes=100,
        max_routes=0,max_routes_per_scenario=0,include_visual_risk=False)
    risk=Counter();added=Counter();found={row_key(r) for r in rows if not r['invalid_action_context']}
    auxiliary=root/'auxiliary/frame_index.jsonl'
    with auxiliary.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        with frozen_routes(builder,routes):
            for r in added_rows(rows,builder.iter_base_frames(args,risk),job['offset_frames']):
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
                found.add(row_key(r));added[r['split']+'/'+r['context_id']]+=1
    manifest=trial.read(index.with_name('manifest.json'))
    # Do not inherit the frozen dataset's counts as claims about supplemental rows.
    manifest={k:manifest[k] for k in ('prompt_contract','additional_development') if k in manifest}
    manifest.update(format=builder.FRAME_INDEX_FORMAT,source_scope=POLICY,
        action_rule_version=builder.ACTION_RULE_VERSION,mapping_contract_hash=builder.mapping_contract_hash(),
        frozen_base_index_sha256=trial.file_sha(index),added_rows_by_split_event=dict(added),
        original_rows=len(rows),frame_index=str(auxiliary),automatic_native_labels=True)
    trial.write(auxiliary.with_name('manifest.json'),manifest)
    check_index(auxiliary,action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
    for p,sha in sources.items():
        if trial.file_sha(p)!=sha:raise ValueError('Collection changed during build: '+p)
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=trial.APP,capture_output=True,text=True,timeout=10)
        git=result.stdout.strip() if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):git=None
    files={n:trial.file_sha(root/n) for n in ('base/frame_index.jsonl','base/manifest.json',
           'auxiliary/frame_index.jsonl','auxiliary/manifest.json')}
    receipt=dict(policy=POLICY,release_sha256=job['release_sha256'],git_commit=git,
        git_scope='optional local checkout metadata; published source SHA is authoritative',
        hostname=socket.gethostname(),server_label=os.environ.get('AUTOMOT_SERVER','unspecified'),
        data_root=str(data),collection_dir=str(collection),offset_frames=job['offset_frames'],files=files,
        source_collection_sha256=sources,original_rows=len(rows),added_rows=sum(added.values()),
        added_rows_by_split_event=dict(added),native_risk_counts=dict(risk),
        requested_targets=len(wanted),missing_targets=len(wanted-found),
        missing_by_split_event=dict(Counter(k[2]+'/'+k[3] for k in wanted-found)),
        raw_meta_recomputation='required in each dense CPU admission before training',
        frozen_anchors_unchanged=True,RGB_copied=False,labels_generated_once=True,
        scope='existing routes and exact native mapped contexts; missing labels not imputed')
    trial.write(root/'shared_data_receipt.json',receipt)
    verify(root,data_root=data,offsets=job['offset_frames'],release=job['release_sha256'])
    print(json.dumps(dict(status='shared_data_built',added=receipt['added_rows'],missing=receipt['missing_targets'])),flush=True)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();job=trial.read(a.job)
    if trial.verified_sources()!=job['release_sha256']:raise ValueError('Data builder sources changed')
    build(job)


if __name__=='__main__':main()
