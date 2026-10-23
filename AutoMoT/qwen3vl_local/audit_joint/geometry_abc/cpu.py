"""Local geometry-only preparation and CPU evidence. Never starts a GPU job."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from . import admission, dataset, planning, regions, prepare
from ..geometry_v1 import sources
from .. import versioned_data as cache

APP = Path(__file__).resolve().parents[3]

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
    return dict(schema='geometry_abc_cpu_evidence_v1',admission=check,capacity=capacity,
        validation_quality=validation_quality(artifact,check),
        formal_conditions_passed=check['status']=='admitted' and capacity['status']=='passed',
        creates_ready=False,repairs_splits=False,thresholds_changed=False,
        equal_raw_observations=True,GPU_executed=False,
        interpretation='Training capacity is measured even after holdout rejection; it never overrides admission.')


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
    p.add_argument('--geometry',type=Path)
    p.add_argument('--maps',type=Path)
    p.add_argument('--study-root',type=Path,default=APP/'checkpoints/geometry_abc_v1')
    p.add_argument('--per-event',type=int,choices=(1024,1280),default=1280)
    args=p.parse_args(); root=args.study_root.resolve()
    # Hold a single lock for parent preparation as well as derived preparation.
    with cache.lock(root/'.cpu.lock'):
        status=dict(status='running',stage='parent_preparation',start_unix=time.time(),GPU_executed=False)
        cache.write(root/'cpu_status.json',status)
        try:
            maps=args.maps.resolve(strict=True) if args.maps else project_maps(APP,APP/'checkpoints/geometry_v1/maps')
            if args.geometry: parent=args.geometry.resolve(strict=True)
            else:
                from ..data_prepare import geometry
                selected=geometry((APP/'lead_data').resolve(strict=True),
                    (APP/'keyframe_filter/collection_output').resolve(strict=True), maps, APP/'checkpoints')
                parent=Path(selected['dataset']).resolve(strict=True)
            status.update(stage='geometry_regions_and_admission',parent_geometry=str(parent))
            cache.write(root/'cpu_status.json',status)
            try: ready=prepare.prepare(parent,maps,root,per_event=args.per_event,world_size=4)
            except ValueError as exc:
                pointer=root/f'ready_{args.per_event}.json'
                if not pointer.is_file(): raise
                ready=json.loads(pointer.read_text())
                if (ready.get('status')!='rejected' or 'dataset' not in ready
                        or ready.get('error')!=f'ValueError: {exc}'): raise
            artifact=dataset.load(ready['dataset'],check_sources=False,expected_sha256=ready['dataset_sha256'])
            if (artifact['parent_geometry']!=str(parent)
                    or artifact['parent_sha256']!=sources.file_sha(parent)):
                raise ValueError('cpu_parent_binding_mismatch')
            report=pool_report(artifact,per_event=args.per_event,world_size=4)
            report.update(dataset=ready['dataset'],dataset_sha256=ready['dataset_sha256'],
                ready_status=ready['status'],legacy_action_filter_used=False)
            destination=root/'reports'/report['dataset_sha256']/str(args.per_event)
            destination.mkdir(parents=True,exist_ok=True)
            prepare.immutable_report(destination/'pool_report.json',report)
            status.update(status='completed',stage='done',end_unix=time.time(),
                ready_status=ready['status'],formal_conditions_passed=report['formal_conditions_passed'],
                report=str(destination/'pool_report.json'))
            cache.write(root/'cpu_status.json',status)
            print(json.dumps(status,ensure_ascii=False),flush=True)
            return 0 if ready['status']=='ready' and report['formal_conditions_passed'] else 2
        except BaseException as exc:
            status.update(status='failed',error=f'{type(exc).__name__}: {exc}',end_unix=time.time())
            cache.write(root/'cpu_status.json',status)
            raise


if __name__=='__main__': raise SystemExit(main())
