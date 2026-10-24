"""Explicit data/plan/resource/processor gates before six long-running jobs."""
import json
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from qwen3vl_local.audit_joint.abc_delivery import sha,digest,write
from . import data,contracts


def verify_raw_sources(directory,manifest,root,collection,*,workers=4):
    """Same full byte hashes, bounded IO concurrency, and pre-filter route proof."""
    root=Path(root);directory=Path(directory);routes={}
    for route in data.candidates(collection,root):
        key=digest([route['scenario'],route['route_id']])
        if key in routes:raise ValueError('duplicate_candidate_route')
        routes[key]=(root/route['scenario']/route['route_id'],digest(route))
    if routes.keys()!=manifest['routes'].keys():raise ValueError('candidate_route_members_changed')
    def verify(item):
        key,expected=item;path=directory/'routes'/key/'receipt.json'
        if sha(path)!=expected:raise ValueError('ABC_route_receipt_changed')
        receipt=json.loads(path.read_text());run,route_sha=routes[key]
        if receipt['route_sha256']!=route_sha:raise ValueError('candidate_route_anchors_changed')
        actual={kind:sorted(p.name for p in (run/kind).glob('*.'+ext))
                for kind,ext in (('metas','pkl'),('rgb','jpg'),('lidar','laz'))}
        if actual!=receipt['members']:raise ValueError('ABC_route_members_changed')
        for name,value in receipt['bindings'].items():
            if sha(root/name)!=value:raise ValueError('ABC_raw_source_changed: '+name)
        if sha(path)!=expected:raise ValueError('ABC_route_receipt_changed_during_check')
        return len(receipt['bindings'])
    files=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,n in enumerate(pool.map(verify,sorted(manifest['routes'].items())),1):
            files+=n
            if i%500==0:print(f'[ABC source check] routes={i}/{len(routes)} bound_files={files}',flush=True)
    return dict(routes=len(routes),bound_file_appearances=files,workers=workers,full_content_hashes=True,
                candidate_routes_and_all_members_verified=True)


def history_check(rows,data_root,output):
    """Replay only missing histories; count every slot and retain exact reasons."""
    from qwen3vl_local.audit_joint.geometry_v1 import sources
    from qwen3vl_local.audit_joint import label_quarantine
    from types import SimpleNamespace
    root=Path(data_root);counts=Counter();pending=defaultdict(list);reasons=Counter();physical=0
    for split,records in rows.items():
        for r in records:
            history=r['representations']['a']['history']
            for h,t in enumerate(history):counts[f'{split}/{r["event"]}/{h}/{"valid" if t["valid"] else "missing"}']+=1
            if not all(t['valid'] for t in history):pending[r['scenario'],r['route_id']].append(r)
    for (scenario,rid),records in sorted(pending.items()):
        route=root/scenario/rid;cache={};replayed={}
        breaks=[r['first_affected_frame'] for r in label_quarantine.load(root,route=(scenario,rid))]
        for r in records:
            frame=r['frame']
            if frame not in replayed:
                for f in range(max(0,frame-8),frame+1):
                    if f in cache:continue
                    p=route/'metas'/f'{f:04d}.pkl'
                    if p.exists():
                        try:cache[f]=sources.read_meta(p)
                        except (ValueError,EOFError,OSError):pass
                lanes=SimpleNamespace(at=lambda *args:r['observation']['reference'])
                replayed[frame]=sources.history(cache,frame,rid.split('_')[0],lanes,breaks);physical+=1
            native=replayed[frame]
            for h,(saved,actual) in enumerate(zip(r['representations']['a']['history'],native)):
                if saved['valid']!=actual['valid'] or saved['token_id']!=actual['token_id']:
                    raise ValueError('missing_history_replay_changed')
                if not saved['valid']:reasons[f'{r["split"]}/{r["event"]}/{h}/{actual["reason"]}']+=1
    report=dict(all_history_slot_counts=dict(counts),missing_reasons=dict(reasons),replayed_physical_frames=physical,
        no_future_reads=True,reference='verified stored current reference; native history replay',full_pool_counts=True)
    write(output,report);return report


def resources(model_dir,bev_checkpoint):
    from qwen3vl_local.qwen35.preflight import check
    result=check(model_dir,action=True);errors=list(result['errors'])
    bev=Path(bev_checkpoint)
    if not bev.is_file():errors.append('missing_BEV_weights: '+str(bev))
    try:
        from qwen3vl_local.action_expert_ablation.common import leadmot_train
        leadmot_train()
    except (ImportError,OSError,RuntimeError) as exc:errors.append(f'offline_runner_import: {exc}')
    files={}
    if not errors:
        root=Path(model_dir)
        files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
               if p.is_file() and '.cache' not in p.parts}
    return dict(ready=not errors,errors=errors,qwen=files,bev=sha(bev) if bev.is_file() else None)


def verify_plan(directory,rows):
    from .planning import project
    stored=json.loads((Path(directory)/'plan.json').read_text())
    actual=project([r for values in rows.values() for r in values])
    # JSON object keys are strings; preserve the frozen planner's integer
    # histogram digest while comparing its complete serialized representation.
    if stored!=json.loads(json.dumps(actual,allow_nan=False)):
        raise ValueError('complete_seven_epoch_plan_changed')
    return stored


def plan_digest(projection):
    import copy
    payload=copy.deepcopy({k:v for k,v in projection.items() if k!='sha256'})
    def integer_keys(values):
        result={}
        for key,value in values.items():
            parsed=int(key)
            if str(parsed)!=str(key) or parsed in result:raise ValueError('noncanonical_repeat_histogram')
            result[parsed]=value
        return result
    payload['cumulative_physical_histogram']=integer_keys(payload['cumulative_physical_histogram'])
    for epoch in payload['plans']:epoch['physical_repeat_histogram']=integer_keys(epoch['physical_repeat_histogram'])
    return digest(payload)


def processor_check(directory,model_dir,data_root,output):
    import time
    from .probe_inputs import update,shape
    started=time.monotonic();extrema={}
    from PIL import Image
    from qwen3vl_local.qwen35.backend import AutoProcessor
    from qwen3vl_local.engine import LocalQwen3VLInstructEngine
    from qwen3vl_local.action_prior.image_condition import image_prompt
    from .runtime import observation_text
    rows,manifest=data.load(directory)
    engine=LocalQwen3VLInstructEngine(Path(model_dir),device='cpu')
    engine.processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    maximum=0;longest=None;count=0;per_variant={v:0 for v in 'abc'}
    for split,records in rows.items():
        for r in records:
            image_path=Path(data_root)/r['image']
            if sha(image_path)!=r['image_sha256']:raise ValueError('processor_image_changed')
            with Image.open(image_path) as im:image=im.convert('RGB')
            for variant in 'abc':
                system,text=observation_text(r,variant)
                messages=engine.build_messages(image_prompt(system,1),text,[image])
                packed=engine.prepare_inputs(engine.apply_chat_template(messages),[image])
                tokens=int(packed['input_ids'].shape[-1])
                update(extrema,r,variant,shape(packed,engine.processor))
                if len(packed['image_grid_thw'])!=1 or tokens>8192:raise ValueError('single_image_context_budget_exceeded')
                if tokens>maximum:maximum=tokens;longest=dict(id=r['id'],variant=variant,split=split)
                count+=1;per_variant[variant]+=1
            if count%3000==0:print(f'[ABC processor] texts={count} longest={maximum}',flush=True)
    result=dict(schema='abc_full_pool_processor_v1',dataset=manifest['identity'],splits=manifest['splits'],
                code=contracts.source_identity(),checked=count,per_variant=per_variant,max_tokens=maximum,
                longest=longest,extrema=extrema,seconds=time.monotonic()-started,full_pool=True,no_truncation=True)
    write(output,result);return result


def verify_development(manifest):
    from qwen3vl_local.audit_joint.geometry_v1.splits import development_contract
    current=development_contract();saved=manifest['development']
    def sources(value):
        result={Path(p).name:h for p,h in value['sources'].items()}
        if len(result)!=len(value['sources']):raise ValueError('ambiguous_development_source_names')
        return result
    if current['groups']!=saved['groups'] or sources(current)!=sources(saved) or current['manifests'] or saved['manifests']:
        raise ValueError('known_development_exposure_changed')
    return dict(groups=len(current['groups']),source_sha256=sources(current))


def verify_derivation_sources(manifest,collection,maps):
    from qwen3vl_local.audit_joint.phase3_maps import discover
    expected={Path(p).name:value for p,value in manifest['annotations'].items()}
    if len(expected)!=len(manifest['annotations']):raise ValueError('ambiguous_annotation_basenames')
    observed={p.name:sha(p) for p in sorted(Path(collection).glob('*_result.json'))}
    if observed!=expected:raise ValueError('annotation_sources_changed')
    proof=discover(Path(__file__).resolve().parents[3],maps)
    observed={town+'.xodr':item['sha256'] for town,item in proof['towns'].items()}
    if observed!=manifest['maps']:raise ValueError('geometry_map_sources_changed')
    verify_development(manifest)
    return dict(annotations=expected,maps=observed)


def check(directory,model_dir,bev_checkpoint,data_root,output,*,raw=True,collection=None,maps=None):
    state=dict(data_ready=False,plan_ready=False,runtime_ready=False,package_ready=False,errors=[])
    # Resource inventory is cheap and reports all missing prerequisites together.
    resource=resources(model_dir,bev_checkpoint);state['resources']=resource
    rows,manifest=data.load(directory,check_raw=False,data_root=data_root)
    from .consumption import checked_bytes
    manifest_sha256=sha(Path(directory)/'manifest.json')
    if json.loads(checked_bytes(Path(directory)/'manifest.json',manifest_sha256))!=manifest:
        raise ValueError('manifest_changed_during_admission')
    if raw:
        if collection is None:collection=Path(next(iter(manifest['annotations']))).parent
        verify_derivation_sources(manifest,collection,maps)
        source_report=verify_raw_sources(directory,manifest,data_root,collection)
        write(Path(output).with_name('raw_source_verification.json'),source_report)
        history_check(rows,data_root,Path(output).with_name('history_support.json'))
    status=json.loads((Path(directory)/'status.json').read_text())
    if not status['data_ready'] or not status['plan_ready']:raise ValueError('ABC_full_pool_not_ready')
    projection=verify_plan(directory,rows)
    state.update(data_ready=True,plan_ready=True,dataset=manifest['identity'],splits=manifest['splits'],
                 manifest_sha256=manifest_sha256,
                 plan_sha256=projection['sha256'],code=contracts.source_identity(),raw_sources_verified=raw,history_ready=raw,
                 development_ready=raw)
    checked_bytes(Path(directory)/'manifest.json',manifest_sha256)
    state['errors'].extend(resource['errors'])
    write(output,state);return state
