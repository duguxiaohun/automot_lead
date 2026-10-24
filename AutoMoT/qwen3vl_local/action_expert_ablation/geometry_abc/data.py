"""Raw annotation → complete continuous targets, independent of semantic actions.

Route chunks are reusable only after their complete file-content bindings pass.
No model or CARLA server is needed. Map lookup uses the installed CARLA library.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from qwen3vl_local.audit_joint.abc_delivery import digest, sha, write, lock
from qwen3vl_local.audit_joint.geometry_v1 import sources
from qwen3vl_local.audit_joint.geometry_abc_v2.dataset import combine
from .contracts import EVENTS, BACKGROUND, QUOTAS, source_identity, contract

_LANES = {}


def compact_record(r):
    """Keep all model/GT identities, omit duplicated derivation diagnostics."""
    def token(t):
        return {k:t[k] for k in ('valid','token_id','duration_s')}
    r.pop('targets',None)
    for view in r['representations'].values():
        view['targets']=[token(t) for t in view['targets']]
        view['history']=[token(t) for t in view['history']]
    r['observation']['history']=r['representations']['a']['history']
    return r


def candidates(collection,root):
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as old
    from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
    from qwen3vl_local.action_prior.build_event_balance_index import _regular_confirmation_block_reasons
    for path in sorted(Path(collection).glob('*_result.json')):
        scenario=path.stem.removesuffix('_result')
        if scenario=='noScenarios':continue
        for route in old._iter_routes_stream(path):
            rid=route.get('route_id')
            if not rid or route.get('status')=='data_missing_skip':continue
            if any(Path(x).name!=x or x in ('.','..') for x in (scenario,rid)):raise ValueError('unsafe_route')
            run=Path(root)/scenario/rid
            if not run.is_dir() or old.is_abnormal_lead_route(run,scenario)[0]:continue
            anchors=[];seen=set()
            for ann in route.get('annotations',[]):
                frame=int(ann['frame_id'])
                if frame in seen:raise ValueError('duplicate_annotation_frame')
                seen.add(frame)
                rs,primary,codes,repair=repair_annotation(scenario,rid,frame,ann,
                    old._rs_label(ann),str(ann.get('primary_event') or 'UNKNOWN'),old._event_codes(ann))
                contexts,evidence=old.mapped_contexts(scenario,rid,frame,rs,primary,codes)
                # No label_actions, old six-action mask or candidate membership test.
                if not contexts and codes and all(str(c).startswith('R-E') for c in codes):
                    if not _regular_confirmation_block_reasons(contexts,evidence,repair,codes):
                        contexts=[BACKGROUND]
                for event in contexts:
                    if event not in QUOTAS:raise ValueError('unknown_context')
                    anchors.append(dict(frame=frame,event=event,road_structure=rs))
            if anchors:yield dict(scenario=scenario,route_id=rid,town=rid.split('_')[0],anchors=anchors)


def validate_continuous(meta):
    future=np.asarray(meta['future_positions'],dtype=np.float64)
    route=np.asarray(meta['route'],dtype=np.float64)
    if future.ndim!=2 or future.shape[1]!=3 or len(future)<41 or not np.isfinite(future[:41]).all():
        raise ValueError('incomplete_2s_continuous_target')
    if route.ndim!=2 or route.shape[1]!=2 or not len(route) or not np.isfinite(route).all():
        raise ValueError('invalid_continuous_route')
    if np.linalg.norm(future[0])>.1 or (np.linalg.norm(np.diff(future[:41,:2],axis=0),axis=1)>5.).any():
        raise ValueError('invalid_future_origin_or_jump')
    return future[[5,10,15,20,25,30,35,40],:2]


def _route_job(job):
    from PIL import Image
    from qwen3vl_local.audit_joint.geometry_v1 import dataset as native
    from qwen3vl_local.action_prior.build_dataset import route_group, split_for
    from qwen3vl_local.audit_joint import label_quarantine
    route,root,maps,out,identity,development,decode=job
    root=Path(root);out=Path(out);run=root/route['scenario']/route['route_id']
    tag=digest([route['scenario'],route['route_id']]);folder=out/'routes'/tag
    receipt_path=folder/'receipt.json'
    members={kind:sorted(p.name for p in (run/kind).glob('*.'+ext))
             for kind,ext in (('metas','pkl'),('rgb','jpg'),('lidar','laz'))}
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['identity']!=identity or receipt['route_sha256']!=digest(route) or receipt['decoded']!=decode:
            raise ValueError('route_cache_contract_changed_use_new_output')
        if sha(folder/'records.jsonl')!=receipt['records_sha256']:raise ValueError('route_cache_changed')
        if receipt['members']!=members:raise ValueError('route_members_changed')
        for name,value in receipt['bindings'].items():
            if sha(root/name)!=value:raise ValueError('route_source_changed: '+name)
        return str(folder),receipt['count'],True
    folder.mkdir(parents=True,exist_ok=True)
    bindings={};metas={};rejected=Counter();reject_cases=[];result=[]
    def bind(path):
        name=str(path.relative_to(root));value=sha(path)
        if name in bindings and bindings[name]!=value:raise ValueError('source_changed_during_route')
        bindings[name]=value
    needed=sorted({i for a in route['anchors'] for i in range(max(0,a['frame']-8),a['frame']+1)})
    for frame in needed:
        p=run/'metas'/f'{frame:04d}.pkl'
        if p.exists():
            bind(p)
            try:metas[frame]=sources.read_meta(p)
            except (ValueError,EOFError,OSError) as exc:rejected['metadata_decode/'+type(exc).__name__]+=1
    group=route_group(route['scenario'],route['route_id'])
    origin=split_for(group,2026,.1,.1);split='train' if group in development else origin
    route=dict(route,split=split)
    if maps not in _LANES:_LANES[maps]=sources.LaneMap(maps)
    lanes=_LANES[maps]
    risks=label_quarantine.load(root,route=(route['scenario'],route['route_id']))
    for risk in risks:
        for source in risk['sources']:bind(root/source['path'])
    discontinuities=[r['first_affected_frame'] for r in risks]
    decoded={}
    for anchor in route['anchors']:
        frame=anchor['frame']
        try:
            if anchor['road_structure'] not in ('R1','R2','R3','R4','R5'):raise ValueError('unknown_road')
            wp=validate_continuous(metas[frame])
            r=native.record(route,anchor,metas,lanes,discontinuities=discontinuities)
            if not all(t['valid'] for t in r['targets']):raise ValueError('incomplete_four_regions')
            r.update(physical_group=group,parent_split=origin,split=split,
                     waypoint_xy=wp.tolist(),condition_source='oracle')
            for kind,suffix in (('rgb','jpg'),('lidar','laz')):
                p=run/kind/f'{frame:04d}.{suffix}'
                if (frame,kind) not in decoded:
                    bind(p)
                    if decode:
                        if kind=='rgb':
                            with Image.open(p) as image:image.load()
                        else:
                            import laspy
                            cloud=laspy.read(p);xyz=np.stack((cloud.x,cloud.y,cloud.z),axis=1)
                            if not len(xyz) or not np.isfinite(xyz).all():raise ValueError('invalid_lidar_points')
                    decoded[frame,kind]=True
            r['image_sha256']=bindings[r['image']]
            result.append(compact_record(combine(r)))
        except (KeyError,ValueError,TypeError,OSError) as exc:
            reason=f'{type(exc).__name__}:{exc}';rejected[reason]+=1
            reject_cases.append(dict(frame=frame,event=anchor['event'],reason=reason))
    # Stability check includes history, rejected metadata and every decoded sensor.
    for name,value in bindings.items():
        if sha(root/name)!=value:raise ValueError('source_changed_during_route: '+name)
    tmp=folder/'records.jsonl.pending'
    with tmp.open('w') as f:
        for r in result:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(folder/'records.jsonl')
    if members!={kind:sorted(p.name for p in (run/kind).glob('*.'+ext))
                 for kind,ext in (('metas','pkl'),('rgb','jpg'),('lidar','laz'))}:raise ValueError('route_members_changed_during_build')
    receipt=dict(identity=identity,route_sha256=digest(job[0]),count=len(result),decoded=decode,members=members,
                 bindings=bindings,rejected=dict(rejected),reject_cases=reject_cases,
                 records_sha256=sha(folder/'records.jsonl'),maps=lanes.hashes)
    write(receipt_path,receipt)
    return str(folder),len(result),False


def prepare(root,collection,maps,output,*,workers=4,limit_routes=0,decode=True):
    from qwen3vl_local.audit_joint.geometry_v1.splits import development_contract
    from qwen3vl_local.action_prior.build_dataset import route_group
    from qwen3vl_local.sft_new_loop_phase3.split_coverage import complete_context_splits
    from .planning import project
    root=Path(root).resolve(strict=True);collection=Path(collection).resolve(strict=True)
    maps=Path(maps).resolve(strict=True);out=Path(output).resolve()
    for p in (root,collection,maps):
        if out.is_relative_to(p) or p.is_relative_to(out):raise ValueError('output_overlaps_input')
    with lock(out):
        started=time.monotonic();code=source_identity(data_only=True);development=development_contract()
        groups={route_group(*g.split('/',1)) for g in development['groups']}
        inputs={str(p):sha(p) for p in sorted(collection.glob('*_result.json'))}
        from qwen3vl_local.audit_joint import label_quarantine
        ledger_sha=sha(label_quarantine.LEDGER)
        map_hashes={p.name:sha(p) for p in sorted(maps.glob('*.xodr'))}
        identity=digest(dict(code=code,annotations=inputs,maps=map_hashes,development=development,decode=decode,ledger_sha=ledger_sha))
        state=dict(schema='abc_trajectory_preparation_v1',identity=identity,status='enumerating',
                   data_ready=False,plan_ready=False,runtime_ready=False,package_ready=False,
                   scoped_smoke_only=bool(limit_routes or not decode))
        write(out/'status.json',state)
        try:
            routes=list(candidates(collection,root))
            if limit_routes:routes=routes[:limit_routes]
            jobs=[(r,str(root),str(maps),str(out),identity,groups,decode) for r in routes]
            folders=[];state['total_routes']=len(jobs)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i,(folder,count,reused) in enumerate(pool.map(_route_job,jobs),1):
                    folders.append(folder);state.update(status='building',completed_routes=i,
                        last_route_records=count,last_route_reused=reused,seconds=time.monotonic()-started)
                    write(out/'status.json',state)
                    if i%10==0:print(f'[ABC data] routes={i}/{len(jobs)} records_last={count}',flush=True)
            records=[]
            for folder in folders:
                with (Path(folder)/'records.jsonl').open() as f:records.extend(json.loads(line) for line in f)
            support_rows=[dict(group=r['physical_group'],split=r['split'],context_id=r['event']) for r in records]
            report=complete_context_splits(support_rows,contexts=tuple(QUOTAS),splits=('train','val','test'),
                development=groups,group_of=lambda r:r['group'],seed=20260920,min_holdout_frames=32,min_holdout_groups=5)
            write(out/'split_report.json',dict(report,old_run_receipts_provided=False,
                exposure_boundary='code_declared_development_only_unknown_external_exposure_not_certified'))
            for r in records:
                if r['physical_group'] in report['moves']:r['split']=report['moves'][r['physical_group']]['to_split']
            counts=Counter(f'{r["split"]}/{r["event"]}' for r in records)
            paths={}
            for split in ('train','val','test'):
                path=out/(split+'.jsonl')
                with path.open('w') as f:
                    for r in records:
                        if r['split']==split:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
                paths[split]=sha(path)
            state.update(status='data_built',counts=dict(counts),data_ready=not report['unresolved'] and not state['scoped_smoke_only'])
            write(out/'manifest.json',dict(identity=identity,code=code,data_root=str(root),maps=map_hashes,
                annotations=inputs,development=development,splits=paths,
                routes={Path(folder).name:sha(Path(folder)/'receipt.json') for folder in folders},counts=dict(counts),
                contracts={v:contract(v) for v in 'abc'},decoded=decode,ledger_sha256=ledger_sha,scoped_smoke_only=state['scoped_smoke_only']))
            write(out/'status.json',state)
            if report['unresolved']:raise ValueError('effective_holdout_support_incomplete: '+str(report['unresolved']))
            print('[ABC data] projecting all seven epochs',flush=True)
            projection=project(records)
            write(out/'plan.json',projection)
            if code!=source_identity(data_only=True):raise ValueError('source_code_changed_during_prepare')
            if inputs!={str(p):sha(p) for p in sorted(collection.glob('*_result.json'))}:raise ValueError('annotations_changed')
            if map_hashes!={p.name:sha(p) for p in sorted(maps.glob('*.xodr'))}:raise ValueError('maps_changed')
            if ledger_sha!=sha(label_quarantine.LEDGER):raise ValueError('quarantine_ledger_changed')
            state.update(status='data_and_plan_ready' if state['data_ready'] else 'smoke_only',
                         plan_ready=state['data_ready'],plan_sha256=sha(out/'plan.json'),seconds=time.monotonic()-started)
            write(out/'status.json',state)
            return state
        except BaseException as exc:
            state.update(status='rejected',error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-started)
            write(out/'status.json',state)
            raise


def load(directory,*,check_raw=False,data_root=None):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['code']!=source_identity(data_only=True):raise ValueError('ABC_data_code_changed')
    from qwen3vl_local.audit_joint import label_quarantine
    if manifest['ledger_sha256']!=sha(label_quarantine.LEDGER):raise ValueError('ABC_quarantine_changed')
    if manifest['scoped_smoke_only'] or not manifest['decoded']:raise ValueError('smoke_data_not_formally_ready')
    rows={}
    for split,value in manifest['splits'].items():
        p=directory/(split+'.jsonl')
        if sha(p)!=value:raise ValueError('ABC_split_changed')
        with p.open() as f:rows[split]=[json.loads(line) for line in f]
    for folder,value in manifest['routes'].items():
        if sha(directory/'routes'/folder/'receipt.json')!=value:raise ValueError('ABC_route_receipt_changed')
    if check_raw:
        root=Path(data_root or manifest['data_root'])
        for folder in manifest['routes']:
            receipt=json.loads((directory/'routes'/folder/'receipt.json').read_text())
            first=next(iter(receipt['bindings']),None)
            if first:
                parts=Path(first).parts;run=root/parts[0]/parts[1]
                if receipt['members']!={kind:sorted(p.name for p in (run/kind).glob('*.'+ext))
                    for kind,ext in (('metas','pkl'),('rgb','jpg'),('lidar','laz'))}:raise ValueError('ABC_route_members_changed')
            for name,value in receipt['bindings'].items():
                if sha(root/name)!=value:raise ValueError('ABC_raw_source_changed: '+name)
    return rows,manifest
