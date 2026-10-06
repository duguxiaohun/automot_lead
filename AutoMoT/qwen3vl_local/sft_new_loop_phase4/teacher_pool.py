"""Prospective answer-blind route discovery for independent teacher review.

Only instance existence is examined. No facts(), answers, expert actions, or
future trajectory are consulted; images are read programmatically, not audited.
The output is a reservation proposal, never an approval or a human reference.
"""
import argparse
from collections import deque,Counter
from concurrent.futures import ProcessPoolExecutor
import json
import lzma
from pathlib import Path
from . import teacher_rules as rules,privileged_geometry as g
from .identity import ROOT,digest,write_json

POLICY='teacher_instance_only_reservations_v1'


def discover(job):
    root,route=job;history=deque(maxlen=7);found={};errors=0
    for n in route['rgb_frames']:
        try:f=g.load_frame(root,route['scenario'],route['route_id'],n)
        except (OSError,ValueError,EOFError,lzma.LZMAError):history.clear();errors+=1;continue
        if n<4 or f.get('source_geometry_issues'):history.clear();continue
        if history and n!=history[-1]['frame_id']+1:history.clear()
        history.append(f)
        if len(history)<7:continue
        frames=list(history)
        for seed in rules.seeds(frames):
            if seed['event'] not in found and not rules.scoped_anomalies(frames,seed['actor_id']):
                found[seed['event']]=dict(frame_id=n,seed=seed,causal_sources=[s for p in frames for s in p['sources']])
        if len(found)==2:break
    return dict(**{k:route[k] for k in ('scenario','route_id','physical_group','split')},instances=found,source_errors=errors)


def eligible_routes(pool,excluded):
    # Physical identity deduplicates alternate recordings before selection.
    seen=set();out=[]
    for r in sorted(pool['routes'],key=lambda r:digest([POLICY,r['scenario'],r['route_id']])):
        group=r['physical_group']
        if r['split'] not in ('val','test') or group in excluded or group in seen:continue
        seen.add(group);out.append(r)
    return out


def select(pool,root,*,per_event=40,workers=4,existing=(),excluded=()):
    from .candidate_pool import validate
    validate(pool)
    if type(per_event) is not int or per_event<1 or type(workers) is not int or not 1<=workers<=64:raise ValueError('invalid reservation budget')
    events=('U-E1','U-E4');counts=Counter();chosen=[];screened=0;errors=0
    routes=eligible_routes(pool,set(excluded)|{r['physical_group'] for r in existing})
    # Stable scenario/Town round robin avoids an early single-scenario prefix.
    buckets={}
    for r in routes:buckets.setdefault((r['scenario'],r['route_id'].split('_')[0]),[]).append(r)
    routes=[]
    while buckets:
        for key in sorted(list(buckets)):
            routes.append(buckets[key].pop(0))
            if not buckets[key]:del buckets[key]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(routes),32):
            for result in executor.map(discover,[(str(root),r) for r in routes[start:start+32]]):
                screened+=1;errors+=result['source_errors']
                needed=[e for e in events if e in result['instances'] and counts[e]<per_event]
                if needed:
                    result['reserved_events']=needed;chosen.append(result)
                    counts.update(needed)
            if all(counts[e]>=per_event for e in events):break
    value=dict(policy=POLICY,teacher=rules.identity(),candidate_pool_sha256=pool['sha256'],
        selection='instance existence only; deterministic scenario/Town rotation; no answer screening',
        per_event=per_event,screened_routes=screened,source_errors=errors,reservations=chosen,
        event_routes=dict(counts),deficits={e:max(0,per_event-counts[e]) for e in events},
        rules_scope='All readiness edges of each reserved event, separately approved per edge and RGB mode; existence does not guarantee YES or a completed episode',
        independently_reviewed=False,supervision_approved=False)
    return dict(value,sha256=digest(value))


def main():
    from .dataset import HOLDOUT_PLAN,PRODUCER_CHECK_PLAN
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-pool',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data');p.add_argument('--per-event',type=int,default=40)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.output.exists():raise FileExistsError('reservation proposals are immutable')
    existing=json.loads((ROOT/PRODUCER_CHECK_PLAN).read_text())['reservations']
    excluded=[r['physical_group'] for r in json.loads((ROOT/HOLDOUT_PLAN).read_text())['reservations']]
    result=select(json.loads(a.candidate_pool.read_text()),a.data_root,per_event=a.per_event,workers=a.workers,existing=existing,excluded=excluded)
    write_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('screened_routes','event_routes','deficits','source_errors')}))

if __name__=='__main__':main()
