"""Versioned post-native-eligibility coverage; existing holdouts are immutable."""
from collections import Counter,defaultdict
import copy
import hashlib
from ..geometry_v1 import splits as inherited
from ..geometry_v1.dataset import digest,validate_splits
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group

POLICY='single_native_coverage_preserve_holdouts_v1'

def coverage(rows,assignment):
    counts={s:Counter() for s in ('train','val','test')};groups={s:defaultdict(set) for s in counts}
    for r in rows:
        if not any(t['valid'] for t in r['targets']):continue
        g=r['physical_group'];s=assignment[g];counts[s][r['event']]+=1;groups[s][r['event']].add(g)
    return {s:{e:dict(anchors=counts[s][e],physical_groups=len(groups[s][e])) for e in CONTEXT_IDS} for s in counts}

def repair(records,*,development,per_event=1280,world_size=4,max_states=100000):
    """Only event validity, identity and training capacity enter selection, not class values."""
    if any(type(x)is not int or x<1 for x in (per_event,world_size)):raise ValueError('positive_training_budget_required')
    if type(max_states)is not int or max_states<1:raise ValueError('positive_split_search_budget_required')
    validate_splits(records)
    rows=[dict(r) for r in records];origins={};by_group=defaultdict(list);blocked=set(development)
    for r in rows:
        g=physical_route_group(r['scenario'],r['route_id'])
        if r['physical_group']!=g:raise ValueError('physical_group_mismatch')
        if origins.setdefault(g,r['split'])!=r['split']:raise ValueError('physical_route_split_leakage')
        if g in blocked and r['split']!='train':raise ValueError('known_exposure_in_existing_holdout')
        if r['event'] not in CONTEXT_IDS:raise ValueError('unknown_single_event')
        by_group[g].append(r)
    before=coverage(rows,origins)
    assignment=dict(origins);moves={};visited=0;capacity_failures=0;inconclusive=False
    events={g:{r['event'] for r in rr if any(t['valid'] for t in r['targets'])} for g,rr in by_group.items()}
    movable={g for g in origins if origins[g]=='train' and g not in blocked}
    # Cheap per-event unique physical-frame floor before the actual shared-frame planner.
    frames={g:{e:{(r['scenario'],r['route_id'],r['frame']) for r in rr
        if r['event']==e and any(t['valid'] for t in r['targets'])} for e in events[g]} for g,rr in by_group.items()}
    train_counts=Counter()
    for g in origins:
        if origins[g]=='train':train_counts.update({e:len(f) for e,f in frames[g].items()})
    supported={s:set().union(*(events[g] for g in origins if origins[g]==s)) for s in ('val','test')}
    candidates={e:sorted((g for g in movable if e in events[g]),key=lambda g:(
        hashlib.sha256(f'20260904:single_coverage:{g}'.encode()).hexdigest(),g)) for e in CONTEXT_IDS}
    def search():
        nonlocal visited,capacity_failures,inconclusive
        if visited>=max_states:inconclusive=True;return False
        visited+=1
        missing=[(s,e) for s in ('val','test') for e in CONTEXT_IDS if e not in supported[s]]
        if not missing:
            from . import planning
            trial=[dict(r,split=assignment[r['physical_group']]) for r in rows]
            try:planning.plan(trial,per_event=per_event,world_size=world_size,epochs=3,seed=20260904)
            except ValueError as exc:
                if not any(tag in str(exc) for tag in ('insufficient_capacity','no_eligible_training','shared_frame_repeat_capacity_exceeded')):raise
                capacity_failures+=1;return False
            return True
        choices={cell:[g for g in candidates[cell[1]] if g not in moves and
            all(train_counts[e]-len(f)>=per_event for e,f in frames[g].items())] for cell in missing}
        cell=min(missing,key=lambda c:(len(choices[c]),c))
        if not choices[cell]:return False
        s,_=cell
        for g in choices[cell]:
            old=set(supported[s]);supported[s].update(events[g]);assignment[g]=s;moves[g]=s
            train_counts.subtract({e:len(f) for e,f in frames[g].items()})
            if search():return True
            train_counts.update({e:len(f) for e,f in frames[g].items()})
            supported[s]=old;assignment[g]='train';del moves[g]
            if inconclusive:return False
        return False
    # Missing train events cannot be supplied by stealing existing holdout groups.
    possible=all(train_counts[e]>=per_event for e in CONTEXT_IDS) and search()
    final=assignment if possible else origins
    if possible:
        for r in rows:r['parent_split']=r['split'];r['split']=final[r['physical_group']]
    else:rows=[dict(r) for r in records]
    report=dict(policy=POLICY,status='feasible' if possible else 'unknown' if inconclusive else 'infeasible',
        before=before,after=coverage(rows,final),
        moves={g:dict(from_split='train',to_split=s,events=sorted(events[g])) for g,s in sorted(moves.items())} if possible else {},
        assignment=final,parent_assignment=origins,per_event=per_event,world_size=world_size,
        visited_states=visited,max_states=max_states,shared_capacity_failures=capacity_failures,
        blocked_groups_sha256=digest(sorted(blocked)),existing_holdouts_preserved=True,
        selection_fields='native eligibility, event coverage, physical group hash, training frame capacity; no action value, RGB or model score',
        exposure_scope='absent from bound recorded development/exposure manifests; unknown external exposure is not certified',
        independent_holdout_claim=False,partial_assignment_published=False)
    return rows,dict(report,sha256=digest(report))

def verify(records,report,development):
    payload={k:v for k,v in report.items() if k!='sha256'}
    if report.get('policy')!=POLICY or report.get('sha256')!=digest(payload):raise ValueError('single_split_report_binding_mismatch')
    if report['blocked_groups_sha256']!=digest(sorted(set(development))):raise ValueError('single_exposure_binding_mismatch')
    for r in records:
        g=r['physical_group'];origin=report['parent_assignment'][g]
        if r['split']!=report['assignment'][g]:raise ValueError('single_split_assignment_mismatch')
        if origin!='train' and r['split']!=origin:raise ValueError('existing_holdout_moved')
        if g in development and r['split']!='train':raise ValueError('exposed_group_moved')
        if r.get('parent_split',origin)!=origin:raise ValueError('parent_split_mismatch')
    if coverage(records,report['assignment'])!=report['after']:raise ValueError('single_split_support_changed')
