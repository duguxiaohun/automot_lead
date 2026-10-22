"""Full-pool CPU evidence around the frozen study implementation; no GPU or split repair.

This coordinator lives outside geometry_study so reporting changes do not invalidate
its verified datasets/checkpoints. Reports bind this file and the executed sources.
"""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import time
from .geometry_study import admission, dataset, planning, regions, six
from .geometry_v1 import dataset as native, sources
from . import versioned_data as cache

APP = Path(__file__).resolve().parents[2]


def validation_quality(artifact, check):
    selected=set(check['fixed_validation_ids'])
    rows=[r for r in artifact['records'] if r['id'] in selected]
    result={}
    for v in regions.VARIANTS:
        classes=defaultdict(Counter);events=defaultdict(Counter)
        for r in rows:
            for t in r['representations'][v]['targets']:
                if not t['valid']:continue
                q=dataset.mass(t,len(regions.names(v)))
                if 1. in q:
                    name=regions.names(v)[q.index(1.)]
                    classes[name][r['physical_group']]+=1
                    events[r['event']]['pure']+=1
                else:events[r['event']]['mixed']+=1
        detail={name:dict(pure_queries=sum(c.values()),physical_groups=len(c),
            largest_group_queries=max(c.values(),default=0),
            largest_group_fraction=max(c.values(),default=0)/sum(c.values()) if c else None,
            group_queries=dict(c)) for name in regions.names(v) for c in [classes[name]]}
        n=sum(sum(c.values()) for c in events.values());mixed=sum(c['mixed'] for c in events.values())
        result[v]=dict(classes=detail,mixed_fraction=mixed/n if n else None,
            unsupported_classes=[k for k,c in detail.items() if not c['pure_queries']],
            single_group_classes=[k for k,c in detail.items() if c['physical_groups']==1],
            mixed_only_events=[e for e,c in events.items() if c['mixed'] and not c['pure']],
            events={e:dict(c) for e,c in events.items()},
            reliability_certified=False,selection_changed=False,
            interpretation='Support diagnostics only; computable macro recall is not a reliability certificate.')
    return result


def pool_report(artifact, *, per_event=1280, world_size=4):
    check=admission.inspect(artifact)
    rows=[r for r in artifact['records'] if r['split']=='train' and any(t['valid'] for t in r['targets'])]
    supply={}
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    for e in CONTEXT_IDS:
        group=[r for r in rows if r['event']==e]
        supply[e]=dict(anchors=len(group),physical_frames=len({(r['scenario'],r['route_id'],r['frame']) for r in group}),
                       physical_groups=len({r['physical_group'] for r in group}))
    capacity=dict(per_event=per_event,world_size=world_size,epochs=3,training_only=True,
        actual_exposure_advanced=False,supply=supply)
    missing=[e for e,c in supply.items() if not c['anchors']]
    if missing:
        capacity.update(status='rejected',error='missing_training_events: '+str(missing))
    else:
        try:
            plan=planning.project(artifact['records'],per_event=per_event,world_size=world_size,epochs=3,seed=20261022)
            capacity.update(status='passed',plan=plan)
        except ValueError as exc:capacity.update(status='rejected',error=str(exc))
    return dict(schema='study_full_pool_cpu_evidence_v1',admission=check,capacity=capacity,
        validation_quality=validation_quality(artifact,check),
        formal_conditions_passed=check['status']=='admitted' and capacity['status']=='passed',
        creates_ready=False,repairs_splits=False,thresholds_changed=False,
        equal_information_ablation=False,GPU_executed=False,
        interpretation='Training capacity is measured even after holdout rejection; it never overrides admission.')


def direction_kind(point):
    if point.get('native_action') not in regions.ACTIONS[3:5]:return None
    if point.get('reason')=='ambiguous_current_direction_projection':return 'near_perpendicular'
    if point.get('action') in regions.ACTIONS[3:5] and point['action']!=point['native_action']:return 'flipped'
    if point.get('action') is None:return 'other_invalid'
    return 'unchanged'


def evidence_origin(point):
    """The donor remains authoritative even if map lookup failed before tagging."""
    donor=(point.get('imputation') or {}).get('selected_endpoint_frame') if point.get('imputed') else None
    tagged=point.get('direction_origin_frame')
    if donor is not None and tagged is not None and donor!=tagged:
        raise ValueError('direction_evidence_donor_mismatch')
    origin=donor if donor is not None else tagged
    if origin is None:
        if point.get('imputed'):raise ValueError('missing_direction_evidence_donor')
        origin=point['frame']
    if type(origin) is not int:raise ValueError('invalid_direction_evidence_origin')
    return origin


class CaseBank:
    """Bounded, deterministic diversity candidates, without reading RGB on admission."""
    def __init__(self, limit):
        self.limit=limit;self.buckets={}

    def add(self, case):
        # At most K groups and K distinct event/origins per group for each
        # kind/Town/event. Hash priority spreads candidates beyond early anchors.
        bucket=self.buckets.setdefault((case['kind'],case['town'],case['event']),{})
        group=case['physical_group'];origin=(case['scenario'],case['route_id'],case['native_evidence_origin'])
        if group not in bucket:
            if len(bucket)>=self.limit:
                worst=max(bucket,key=dataset.digest)
                if dataset.digest(group)>=dataset.digest(worst):return
                del bucket[worst]
            bucket[group]={}
        members=bucket[group]
        if origin not in members and len(members)>=self.limit:
            worst=max(members,key=dataset.digest)
            if dataset.digest(origin)>=dataset.digest(worst):return
            del members[worst]
        rank=lambda c:(abs(c['anchor']-c['native_evidence_origin']),c['anchor'],c['id'],c['role'])
        if origin not in members or rank(case)<rank(members[origin]):members[origin]=case

    def select(self):
        all_cases=[c for b in self.buckets.values() for group in b.values() for c in group.values()]
        selected=[]
        for kind in sorted({c['kind'] for c in all_cases}):
            candidates=[c for c in all_cases if c['kind']==kind]
            groups=set();towns=set();events=set();origins=set()
            for _ in range(min(self.limit,len(candidates))):
                score=lambda c:(c['physical_group'] not in groups,c['town'] not in towns,c['event'] not in events,
                    (c['scenario'],c['route_id'],c['native_evidence_origin']) not in origins,
                    dataset.digest(c['case_key']+[c['scenario'],c['route_id']]))
                chosen=max(candidates,key=score);candidates.remove(chosen);selected.append(chosen)
                groups.add(chosen['physical_group']);towns.add(chosen['town']);events.add(chosen['event'])
                origins.add((chosen['scenario'],chosen['route_id'],chosen['native_evidence_origin']))
        return selected


def direction_scan(parent, maps, output, *, max_cases=12):
    """All TRAIN parent candidates, including common-pool exclusions. Never val/test."""
    if type(max_cases) is not int or max_cases<1:raise ValueError('positive_case_limit_required')
    from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
    from . import label_quarantine
    output=Path(output);root=Path(parent['data_root']);lanes=sources.LaneMap(maps)
    grouped=defaultdict(list)
    for r in parent['records']:
        if r['split']=='train':grouped[r['scenario'],r['route_id']].append(r)
    counts=Counter();by_event=defaultdict(Counter);unique=Counter();bank=CaseBank(max_cases)
    examined=0
    for route_index,((scenario,rid),records) in enumerate(sorted(grouped.items()),1):
        trajectory=rule.load_route_trajectory(root/scenario/rid)
        if trajectory is None:raise ValueError('direction_audit_unreadable_route: '+rid)
        quarantine=label_quarantine.load(root,route=(scenario,rid));native_cache={};refs={};seen=set()
        for r in sorted(records,key=lambda r:r['id']):
            result=six.build(r,trajectory,quarantine,lanes,native_cache=native_cache,reference_cache=refs)
            examined+=1
            for role in ('history_points','future_points'):
                for p in result['audit'][role]:
                    kind=direction_kind(p)
                    if kind is None:continue
                    counts[kind]+=1;by_event[r['event']][kind]+=1
                    origin=evidence_origin(p)
                    key=(r['event'],origin,kind)
                    if key not in seen:unique[kind]+=1;seen.add(key)
                    # Limit saved evidence, not scan coverage. Preserve exclusions too.
                    if kind=='unchanged':continue
                    case_key=(r['event'],r['frame'],origin,role,kind)
                    state=refs.get(origin)
                    native_point=native_cache.get((origin,r['event']))
                    if native_point is None:
                        if p.get('imputed'):raise ValueError('missing_donor_native_evidence')
                        native_point=p
                    start=min(r['frame'],origin)-1;end=max(r['frame'],native_point.get('required_end_frame',origin))+1
                    bank.add(dict(case_key=list(case_key),id=r['id'],scenario=scenario,route_id=rid,town=rid.split('_')[0],
                        physical_group=r['physical_group'],split='train',anchor=r['frame'],event=r['event'],
                        kind=kind,role=role,point=p,current_reference=r['observation']['reference'],
                        local_reference=state['reference'] if state else None,
                        native_evidence_origin=origin,native_evidence=native_point.get('evidence'),
                        anchor_action=result['anchor_action'],rgb_range=[start,end],visual_reviewed=False))
        if route_index%25==0:print(f'[direction audit] routes={route_index}/{len(grouped)} anchors={examined} counts={dict(counts)}',flush=True)
    cases=bank.select()
    for case in cases:
        case['RGB']=[]
        start,end=case.pop('rgb_range')
        for f in range(start,end+1):
            path=root/case['scenario']/case['route_id']/'rgb'/f'{f:04d}.jpg'
            case['RGB'].append(dict(frame=f,path=str(path),exists=path.is_file(),
                sha256=sources.file_sha(path) if path.is_file() else None))
    # Verify sources again after replay; no stale evidence passed off as current.
    native.verify(parent)
    for case in cases:
        for frame in case['RGB']:
            path=Path(frame['path'])
            if path.is_file()!=frame['exists'] or frame['exists'] and sources.file_sha(path)!=frame['sha256']:
                raise ValueError('direction_audit_RGB_changed')
    for town,sha in lanes.hashes.items():
        if sources.file_sha(Path(maps)/(town+'.xodr'))!=sha:raise ValueError('direction_audit_map_changed')
    result=dict(schema='study_train_direction_audit_v2',train_only=True,routes=len(grouped),anchors=examined,
        occurrence_counts=dict(counts),event_occurrences={k:dict(c) for k,c in by_event.items()},
        distinct_event_origin_kind_counts=dict(unique),saved_case_limit_per_kind=max_cases,
        case_selection='bounded hash candidates per kind/Town/event; prioritize new physical groups, Towns, events and origins; not blind',
        candidate_bound_per_kind_town_event=max_cases**2,cases=cases,
        selected_support={k:dict(cases=sum(c['kind']==k for c in cases),
            physical_groups=len({c['physical_group'] for c in cases if c['kind']==k}),
            towns=sorted({c['town'] for c in cases if c['kind']==k}),events=sorted({c['event'] for c in cases if c['kind']==k}),
            route_origins=len({(c['scenario'],c['route_id'],c['native_evidence_origin']) for c in cases if c['kind']==k}))
            for k in sorted({c['kind'] for c in cases})},
        independent_maneuver_coverage_certified=False,
        scan_complete=True,visual_review_complete=False,includes_common_exclusions=True,
        map_sha256=dict(lanes.hashes),interpretation='Counts repeat origins across anchors; distinct counts still depend on current anchor reference.')
    dataset.write(output/'directions.json',result)
    panels=['<meta charset="utf-8"><title>Train direction evidence</title><h1>训练池方向边界候选：尚未视觉验收</h1>']
    for c in cases:
        panels.append('<h2>'+html.escape(f"{c['scenario']}/{c['route_id']} anchor {c['anchor']} {c['event']} {c['kind']}")+'</h2>')
        panels.append('<pre>'+html.escape(json.dumps({k:v for k,v in c.items() if k!='RGB'},ensure_ascii=False,indent=2))+'</pre>')
        for frame in c['RGB']:
            if frame['exists']:
                panels.append(f'<figure style="display:inline-block"><img width="400" src="{html.escape(Path(frame["path"]).as_uri(),quote=True)}"><figcaption>frame {frame["frame"]}</figcaption></figure>')
    (output/'directions.html').write_text('\n'.join(panels))
    return result


def project_maps(app, view):
    """Known project CARLA paths only, no filesystem-wide or remote probing."""
    base=app/'CARLA_0915/CarlaUE4/Content/Carla/Maps'
    towns=[f'Town{i:02d}' for i in range(1,8)]+['Town10HD','Town11','Town12','Town13','Town15']
    selected={}
    for town in towns:
        choices=[p for p in (base/'OpenDrive'/f'{town}.xodr',base/town/'OpenDrive'/f'{town}.xodr') if p.is_file()]
        if not choices:raise ValueError('missing_project_map: '+town)
        if len({sources.file_sha(p) for p in choices})!=1:raise ValueError('conflicting_project_maps: '+town)
        selected[town]=choices[0].resolve(strict=True)
    view.mkdir(parents=True,exist_ok=True)
    for town,path in selected.items():
        link=view/(town+'.xodr')
        if link.exists() or link.is_symlink():
            if not link.is_file() or sources.file_sha(link)!=sources.file_sha(path):raise ValueError('existing_map_view_changed')
        else:link.symlink_to(path)
    return view.resolve(strict=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--geometry',type=Path,help='Explicit verified native parent; otherwise use native versioned preparation')
    p.add_argument('--maps',type=Path,help='Default: known project CARLA maps, no cross-machine path guessing')
    p.add_argument('--study-root',type=Path,default=APP/'checkpoints/geometry_study_v1')
    p.add_argument('--output',type=Path)
    p.add_argument('--per-event',type=int,choices=(1024,1280),default=1280)
    p.add_argument('--directions',action='store_true',help='Replay all train parent candidates, including common exclusions')
    args=p.parse_args();output=(args.output or APP/f'checkpoints/geometry_study_cpu_audit_v2_{args.per_event}').resolve()
    study_root=args.study_root.resolve()
    if output.exists():raise FileExistsError('Existing audit preserved; use another explicit semantic --output')
    maps=args.maps.resolve(strict=True) if args.maps else project_maps(APP,APP/'checkpoints/geometry_v1/maps')
    if args.geometry:parent_path=args.geometry.resolve(strict=True)
    else:
        from .data_prepare import geometry
        selected=geometry((APP/'lead_data').resolve(strict=True),(APP/'keyframe_filter/collection_output').resolve(strict=True),
                          maps,APP/'checkpoints')
        parent_path=Path(selected['dataset']).resolve(strict=True)
    parent=native.load(parent_path);protected=[parent_path.parent,Path(parent['data_root']).resolve(),maps,study_root]
    if any(output.is_relative_to(x) or x.is_relative_to(output) for x in protected):raise ValueError('audit_output_overlaps_inputs')
    output.mkdir(parents=True,exist_ok=False)
    del parent  # Do not hold a second full parent during common-pool preparation.
    source_identity=dataset.code_identity();tool_sha=sources.file_sha(__file__)
    status=dict(status='running',parent_geometry=str(parent_path),parent_sha256=sources.file_sha(parent_path),
        sources=source_identity,tool_sha256=tool_sha,start_unix=time.time(),training_executed=False)
    cache.write(output/'status.json',status)
    try:
        from .geometry_study.prepare import prepare
        try:ready=prepare(parent_path,maps,study_root,per_event=args.per_event,world_size=4)
        except ValueError as exc:
            ready_path=study_root/f'ready_{args.per_event}.json'
            if not ready_path.is_file():raise
            ready=json.loads(ready_path.read_text())
            if ready.get('status')!='rejected' or 'dataset' not in ready:raise
            status['prepare_error']=str(exc)
        artifact=dataset.load(ready['dataset'],expected_sha256=ready['dataset_sha256'])
        if artifact['parent_sha256']!=status['parent_sha256']:raise ValueError('audit_parent_binding_mismatch')
        report=pool_report(artifact,per_event=args.per_event,world_size=4)
        report.update(dataset=str(ready['dataset']),dataset_sha256=ready['dataset_sha256'],ready_status=ready['status'])
        dataset.write(output/'pool_report.json',report)
        # Free the common pool before replaying the native parent.
        del artifact
        if args.directions:direction_scan(native.load(parent_path),maps,output)
        if source_identity!=dataset.code_identity() or tool_sha!=sources.file_sha(__file__):raise ValueError('audit_sources_changed')
        status.update(status='completed',formal_conditions_passed=report['formal_conditions_passed'],
            ready_status=ready['status'],end_unix=time.time(),directions_scanned=args.directions)
        cache.write(output/'status.json',status)
        print(json.dumps(dict(status=status['status'],formal_conditions_passed=report['formal_conditions_passed'],
            admission=report['admission']['status'],capacity=report['capacity']['status'],output=str(output))))
        return 0 if report['formal_conditions_passed'] and ready['status']=='ready' else 2
    except Exception as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}',end_unix=time.time())
        cache.write(output/'status.json',status);raise


if __name__=='__main__':raise SystemExit(main())
