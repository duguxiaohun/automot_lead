"""Freeze all discovered RGB routes before causal annotation; inventory is not supervision."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .identity import ROOT,contract,digest,write_json

POLICY='phase4_filtered_rgb_inventory_v3'


def scan(data_root,output,seed=20260929):
    from .dataset import groups,holdout_reservations,source_route,split_for
    output=Path(output)
    if output.exists():raise FileExistsError('reuse the frozen candidate pool or choose a new output')
    root=Path(data_root).resolve();exposed=groups();reserved=holdout_reservations();routes=[];excluded=[]
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
    for scenario in sorted(root.iterdir()):
        if not scenario.is_dir():continue
        for route in sorted(scenario.iterdir()):
            if not route.is_dir() or not (route/'rgb').is_dir():continue
            if not route.resolve().is_relative_to(root):raise ValueError('candidate route escapes data root')
            rgb=sorted(int(p.stem) for p in (route/'rgb').glob('*.jpg') if p.stem.isdigit())
            metas=sorted(int(p.stem) for p in (route/'metas').glob('*.pkl') if p.stem.isdigit())
            if not rgb:continue
            r=dict(scenario=scenario.name,route_id=route.name)
            group=source_route(r)
            r.update(physical_group=group,split=split_for(group,exposed,seed,reservations=reserved),
                     rgb_frames=rgb,meta_frames=metas,status='pending_event_identity_and_causal_condition_review')
            abnormal,info=is_abnormal_lead_route(route,scenario.name)
            if abnormal:
                excluded.append(dict(r,exclusion_reason='abnormal_duration',duration_evidence=info));continue
            if not metas:
                excluded.append(dict(r,exclusion_reason='missing_all_metas'));continue
            from .route_quality import stationarity_check
            r['stationarity_check']=stationarity_check(route,rgb,metas)
            if r['stationarity_check']['status']=='whole_route_stationary':
                excluded.append(dict(r,exclusion_reason='whole_route_stationary'));continue
            routes.append(r)
    if not routes:raise ValueError('no RGB candidate routes found')
    body=dict(policy=POLICY,contract=contract(),seed=seed,exposure_sha256=digest(sorted(exposed)),
              routes=routes,excluded_routes=excluded,scope='all duration-eligible RGB routes with metadata; exclusions explicitly recorded; labels not inferred',
              raw_rgb_condition_extraction=False,
              summary=dict(routes=len(routes),rgb_frames=sum(len(r['rgb_frames']) for r in routes),
                           split_routes=dict(Counter(r['split'] for r in routes)),discovered_routes=len(routes)+len(excluded),
                           excluded_routes=len(excluded),exclusion_reasons=dict(Counter(r['exclusion_reason'] for r in excluded)),
                           excluded_split_routes=dict(Counter(r['split'] for r in excluded))))
    body['sha256']=digest(body);write_json(output,body)
    return body


def validate(pool):
    from .dataset import groups,holdout_reservations,source_route,split_for
    body={k:v for k,v in pool.items() if k!='sha256'}
    exposed=groups();reserved=holdout_reservations()
    if (pool.get('sha256')!=digest(body) or pool.get('policy')!=POLICY or pool.get('contract')!=contract()
            or pool.get('exposure_sha256')!=digest(sorted(exposed))):
        raise ValueError('candidate inventory identity/exposure/contract mismatch')
    from lead_video_tools.abnormal_duration_filter import classify_abnormal_duration,is_abnormal_duration_allowlisted,DEFAULT_FPS
    from .route_quality import validate_stationarity
    seen=set()
    for r in pool['routes']:
        key=r['scenario'],r['route_id'];group=source_route(r)
        if key in seen or r['physical_group']!=group or r['split']!=split_for(group,exposed,pool['seed'],reservations=reserved):
            raise ValueError('candidate route/split mismatch')
        for field in ('rgb_frames','meta_frames'):
            v=r[field]
            if not isinstance(v,list) or any(type(f) is not int or f<0 for f in v) or v!=sorted(set(v)):
                raise ValueError('invalid candidate frame inventory')
        if classify_abnormal_duration(len(r['rgb_frames']),fps=DEFAULT_FPS) and not is_abnormal_duration_allowlisted(r['scenario']):raise ValueError('abnormal route in eligible pool')
        if not r['meta_frames']:raise ValueError('route lacks metadata')
        validate_stationarity(r.get('stationarity_check'),r['rgb_frames'])
        if r['stationarity_check']['status']=='whole_route_stationary':raise ValueError('stationary capture in eligible pool')
        seen.add(key)
    for r in pool.get('excluded_routes',[]):
        key=r['scenario'],r['route_id']
        if r['physical_group']!=source_route(r) or r['split']!=split_for(r['physical_group'],exposed,pool['seed'],reservations=reserved):raise ValueError('excluded route/split mismatch')
        for field in ('rgb_frames','meta_frames'):
            v=r[field]
            if not isinstance(v,list) or any(type(f) is not int or f<0 for f in v) or v!=sorted(set(v)):raise ValueError('invalid excluded frame inventory')
        if key in seen or r.get('exclusion_reason') not in ('abnormal_duration','missing_all_metas','whole_route_stationary'):raise ValueError('invalid/duplicate exclusion')
        if r['exclusion_reason']=='whole_route_stationary':
            validate_stationarity(r.get('stationarity_check'),r['rgb_frames'])
            if r['stationarity_check']['status']!='whole_route_stationary' or not set(r['rgb_frames'])<=set(r['meta_frames']):raise ValueError('unsubstantiated stationary exclusion')
        abnormal=bool(classify_abnormal_duration(len(r['rgb_frames']),fps=DEFAULT_FPS) and not is_abnormal_duration_allowlisted(r['scenario']))
        if (r['exclusion_reason']=='abnormal_duration' and not abnormal) or (r['exclusion_reason']=='missing_all_metas' and r['meta_frames']):raise ValueError('unsubstantiated exclusion')
        seen.add(key)
    return pool


def coverage(pool,rows,*,production_index=None,approved_classes=()):
    validate(pool)
    routes={(r['scenario'],r['route_id']):r for r in pool['routes']}
    frames={k:set(r['rgb_frames']) for k,r in routes.items()}
    labelled=set();reviewed=set()
    for row in rows:
        key=row['scenario'],row['route_id'];r=routes.get(key)
        if (r is None or row['split']!=r['split']
                or not set(row['observation']['history_frames'])<=frames[key]
                or row['observation']['frame_id'] not in r['meta_frames']):
            raise ValueError('question outside frozen candidate route/split/frame inventory')
        point=(*key,row['observation']['frame_id']);reviewed.add(point)
        if row['target'] in ('YES','NO'):labelled.add(point)
    total=sum(map(len,frames.values()))
    accounting=dict(processed_routes=0,processed_frames=0,dispositions={},rule_targets={},
                    complete_causal_production=not routes,index_sha256=None)
    if production_index is not None:
        from .teacher_replay import accounting as production_accounting
        accounting=production_accounting(pool,production_index)
    return dict(candidate_routes=len(routes),candidate_rgb_frames=total,
                frames_with_questions=len(reviewed),frames_with_supervision=len(labelled),
                frames_without_questions=total-len(reviewed),
                complete_causal_production=bool(total) and accounting['processed_frames']==total and accounting['complete_causal_production'],
                accounting=accounting,approved_rule_classes=list(approved_classes),
                scope='inventory coverage; unreviewed frames may be irrelevant, invalid or awaiting conditions; never implicit NO')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260929)
    a=p.parse_args();print(json.dumps(scan(a.data_root,a.output,a.seed)['summary']))

if __name__=='__main__':main()
