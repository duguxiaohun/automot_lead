"""Exact shared-frame capacity allocation with bounded scarce-event repeats.

Scarce events use every eligible physical frame. A compressed min-cost flow
assigns the remaining quotas, minimizing repeated physical presentations first,
then cumulative physical exposure. Per-event and global frame limits both apply.
"""
from collections import Counter, defaultdict
import heapq
import math
import random

from .phase3_base_dense.baseline import digest
from .phase3_base_coverage_sampling import validate_splits

POLICY='independent_full_pool_scarce_repeat_v3'


def _allocate(demands, groups, total, rng):
    """Integer min-cost flow; group members have identical residual eligibility.

    Each event may use each group frame at most once in this residual pass.
    Separate first/second slot edges count cross-event repeats as well as repeats
    of the same event. Integer capacities permit exact group compression.
    """
    graph=[]
    def node():graph.append([]);return len(graph)-1
    def edge(a,b,capacity,cost):
        f=[b,len(graph[b]),capacity,cost,capacity]
        graph[a].append(f);graph[b].append([a,len(graph[a])-1,0,-cost,0]);return f
    source=node();sink=node();events={e:node() for e in sorted(demands)}
    for e,n in demands.items():edge(source,events[e],n,0)
    refs={};keys=sorted(groups);rng.shuffle(keys)
    # Lexicographic objective: no exposure saving can buy one extra repeat.
    repeat_cost=total*7+1
    for key in keys:
        eligible,used,previous,slots=key;count=len(groups[key]);g=node()
        for e in eligible:refs[e,key]=edge(events[e],g,count,0)
        for j in range(slots):edge(g,sink,count,(repeat_cost if used+j else 0)+previous+used+j)
    flow=0;potential=[0]*len(graph);augmentations=0
    while flow<total:
        distance=[math.inf]*len(graph);parent=[None]*len(graph);distance[source]=0;heap=[(0,source)]
        while heap:
            cost,u=heapq.heappop(heap)
            if cost!=distance[u]:continue
            for i,e in enumerate(graph[u]):
                if not e[2]:continue
                candidate=cost+e[3]+potential[u]-potential[e[0]]
                if candidate<distance[e[0]]:
                    distance[e[0]]=candidate;parent[e[0]]=(u,i);heapq.heappush(heap,(candidate,e[0]))
        if parent[sink] is None:raise ValueError('shared_frame_capacity_exceeded_even_with_bounded_repeats')
        for i,d in enumerate(distance):
            if d<math.inf:potential[i]+=d
        amount=total-flow;v=sink
        while v!=source:
            u,i=parent[v];amount=min(amount,graph[u][i][2]);v=u
        v=sink
        while v!=source:
            u,i=parent[v];e=graph[u][i];e[2]-=amount;graph[v][e[1]][2]+=amount;v=u
        flow+=amount;augmentations+=1
    return {(e,key):f[4]-f[2] for (e,key),f in refs.items()},augmentations


def plan(records,*,per_event=1280,world_size=4,epochs=3,seed=20261021,
         exposure=None,max_repeat=1,max_epoch_steps=4096,max_total_steps=12288):
    if any(type(v) is not int or v<=0 for v in (per_event,world_size,epochs,max_repeat,max_epoch_steps,max_total_steps)):
        raise ValueError('positive_integer_budget_required')
    if max_repeat!=1 or epochs>3:raise ValueError('bounded_base_requires_default_cap_one_and_at_most_three_epochs')
    validate_splits(records)
    known={r['id']:r for r in records};events=defaultdict(list);frames=defaultdict(lambda:defaultdict(list))
    frame_of=lambda r:(r['scenario'],r['route_id'],r['frame'])
    for r in records:
        if any(t['valid'] for t in r['targets']):events[r['event']].append(r);frames[frame_of(r)][r['event']].append(r)
    if not events:raise ValueError('no_eligible_training_anchors')
    total=per_event*len(events)
    if total%world_size:raise ValueError('quota_not_divisible_by_world_size')
    steps=total//world_size
    if steps>max_epoch_steps or steps*epochs>max_total_steps:raise ValueError('training_step_budget_exceeded')
    supply={e:len({frame_of(r) for r in rows}) for e,rows in events.items()}
    scarce={e for e,n in supply.items() if n<per_event}
    if any(2*n<per_event for n in supply.values()):raise ValueError('insufficient_capacity_even_with_two_presentations_per_frame')
    case_caps={e:math.ceil(epochs*per_event/n) if e in scarce else epochs for e,n in supply.items()}
    exposure=Counter(exposure or {})
    if any(type(v) is not int or v<0 for v in exposure.values()):raise ValueError('invalid_actual_exposure')
    if set(exposure)-set(known):raise ValueError('exposure_contains_unknown_dataset_members')
    completed,remainder=divmod(sum(exposure.values()),total)
    if remainder or completed>=epochs:raise ValueError('exposure_requires_complete_prior_epochs')
    remaining_epochs=epochs-completed-1
    old_frames=Counter();old_cases=Counter();routes=Counter();segments=Counter()
    for key,count in exposure.items():
        r=known[key];f=frame_of(r);old_frames[f]+=count;old_cases[r['event'],f]+=count
        routes[r['physical_group']]+=count;segments[r['scenario'],r['route_id'],r['frame']//16]+=count
    rng=random.Random(seed);natural=Counter();weights={};buckets=defaultdict(dict)
    for e,rows in events.items():
        for r in rows:
            for cell,q in r['sampling_mass'].items():natural[e+'/'+cell]+=q
    for cell,n in natural.items():buckets[cell.rsplit('/',1)[0]][cell]=n
    for cells in buckets.values():
        mass=sum(cells.values());z=sum(math.sqrt(n) for n in cells.values())
        for cell,n in cells.items():weights[cell]=min(4.,.5+.5*mass/(math.sqrt(n)*z))
    tie={}
    for r in sorted((r for rows in events.values() for r in rows),key=lambda r:r['id']):
        denominator=sum(r['sampling_mass'].values())
        if denominator<=0:raise ValueError('missing_shared_sampling_mass')
        weight=sum(q*weights[r['event']+'/'+cell] for cell,q in r['sampling_mass'].items())/denominator
        tie[r['id']]=-math.log(max(rng.random(),1e-15))/weight
    representatives={(e,f):min(rows,key=lambda r:(exposure[r['id']],tie[r['id']],r['id']))
                     for f,available in frames.items() for e,rows in available.items()}
    selected=[];physical=Counter();cases=Counter();planned=Counter();limits={}
    def priority(r):
        f=frame_of(r)
        return (physical[f],old_cases[r['event'],f]+cases[r['event'],f],routes[r['physical_group']],
            segments[r['scenario'],r['route_id'],r['frame']//16],exposure[r['id']]+planned[r['id']],tie[r['id']],r['id'])
    def assign(r):
        selected.append(r);f=frame_of(r);physical[f]+=1;cases[r['event'],f]+=1;planned[r['id']]+=1
        routes[r['physical_group']]+=1;segments[r['scenario'],r['route_id'],r['frame']//16]+=1
    groups=defaultdict(list)
    for f,available in sorted(frames.items()):
        mandatory=sorted(set(available)&scarce)
        # Single-event frames retain the tighter case cap; shared frames have
        # global cap 2/epoch, 6/run while each event keeps its own run cap.
        limits[f]=case_caps[next(iter(available))] if len(available)==1 else 2*epochs
        allowed=min(2,limits[f]-old_frames[f]-remaining_epochs*len(mandatory))
        if len(mandatory)>allowed:raise ValueError('shared_scarce_frames_cannot_reserve_remaining_epochs')
        for e in mandatory:
            if old_cases[e,f]+1>case_caps[e]-remaining_epochs:raise ValueError('scarce_case_run_capacity_exceeded')
            assign(representatives[e,f])
        eligible=tuple(e for e in sorted(available) if old_cases[e,f]+cases[e,f] <
            case_caps[e]-(remaining_epochs if e in scarce else 0))
        slots=allowed-len(mandatory)
        if eligible and slots:groups[eligible,len(mandatory),old_frames[f],slots].append(f)
    demands={e:per_event-(supply[e] if e in scarce else 0) for e in events}
    allocations,augmentations=_allocate(demands,groups,sum(demands.values()),rng)
    for key in sorted(groups):
        for e in sorted(key[0],key=lambda e:(-allocations[e,key],e)):
            count=allocations[e,key]
            if not count:continue
            choices={f:representatives[e,f] for f in groups[key]}
            heap=[priority(r) for r in choices.values()];heapq.heapify(heap)
            by_id={r['id']:r for r in choices.values()}
            for _ in range(count):
                while heap:
                    previous=heapq.heappop(heap);r=by_id[previous[-1]];f=frame_of(r)
                    if physical[f]>=key[1]+key[3]:continue
                    current=priority(r)
                    if current!=previous:heapq.heappush(heap,current);continue
                    break
                else:raise ValueError('compressed_capacity_realization_failed')
                assign(r)  # removed from this event heap: at most one residual use
    event_frames={e:Counter(frame_of(r) for r in selected if r['event']==e) for e in events}
    event_repeats={e:dict(available_distinct_frames=supply[e],selected_distinct_frames=len(c),
        repeated_presentations=sum(c.values())-len(c),max_case_repeat=max(c.values()),
        max_frame_repeat=max(physical[f] for f in c),
        max_run_case_repeat=max(old_cases[e,f]+n for f,n in c.items()),
        max_run_frame_repeat=max(old_frames[f]+physical[f] for f in c),
        case_cap_per_epoch=2 if e in scarce else 1,case_cap_per_run=case_caps[e],
        frame_cap_per_epoch=2,frame_cap_per_run=max(limits[f] for f in c)) for e,c in event_frames.items()}
    rng.shuffle(selected)
    report=dict(ordered_ids=[r['id'] for r in selected],per_event=per_event,
        event_counts=dict(Counter(r['event'] for r in selected)),selected=len(selected),unique=len(planned),
        optimizer_steps=steps,planned_total_steps=steps*epochs,world_size=world_size,batch_size=1,grad_accum=1,
        max_frame_repeat=max(physical.values()),repeated_fraction=1-len(physical)/len(selected),
        repeated_presentations=len(selected)-len(physical),
        within_event_repeated_presentations=sum(r['repeated_presentations'] for r in event_repeats.values()),
        shared_context_presentations=sum(len(c) for c in event_frames.values())-len(physical),
        allocation_augmentations=augmentations,capacity_algorithm='compressed_integer_min_cost_flow',
        selected_action_mass=dict(sum((Counter({r['event']+'/'+cell:q for cell,q in r['sampling_mass'].items()}) for r in selected),Counter())),
        natural_action_mass=dict(natural),unseen_physical_frames=sum(old_frames[f]==0 for f in physical),
        route_groups=len({r['physical_group'] for r in selected}),
        time_segments=len({(r['scenario'],r['route_id'],r['frame']//16) for r in selected}),
        max_run_frame_repeat=max(old_frames[f]+n for f,n in physical.items()),
        actual_exposure_advanced=False,sampling=POLICY,temporal_segment_seconds=4,
        frame_cap_per_epoch=2,frame_cap_per_run=max(limits.values()),scarce_events=sorted(scarce),event_repeats=event_repeats)
    report['plan_sha256']=digest(report)
    return selected,report


def independent_work(rows,args,*,epoch,world_size,history=None):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity,physical_group_for_item,unique_cases
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    if args.batch_size!=1 or args.grad_accum!=1 or args.action_output_mode!='choice':raise ValueError('Independent coverage requires choice B1 accumulation1')
    if epoch not in range(3):raise ValueError('Invalid coverage epoch')
    if history is None:
        if epoch:raise ValueError('Missing completed-epoch exposure')
        exposure={}
    else:
        if history.get('policy')!=POLICY or history.get('next_epoch')!=epoch:raise ValueError('Foreign or incomplete sampling history')
        exposure=history['exposure']
        if sum(exposure.values())!=epoch*args.focus_balance_count*len(CONTEXT_IDS):raise ValueError('Incomplete successful-epoch exposure')
    seed=int(args.seed)+epoch*1_000_003
    items=unique_cases([train._make_item(r,seed=seed,action_output_mode='choice') for r in rows])
    by_id={digest(case_identity(x)):x for x in items if train.choice_rejection_reason(x.spec) is None}
    records=[dict(id=k,scenario=x.row.scenario,route_id=x.row.route_id,frame=x.row.frame_id,
        split=x.row.split,event=x.row.context_id,physical_group=physical_group_for_item(x),targets=[dict(valid=True)],
        sampling_mass={'current/0/'+train.choice_action_for_answers(x.spec):1.}) for k,x in by_id.items()]
    if {r['event'] for r in records}!=set(CONTEXT_IDS):raise ValueError('All ten training events required')
    _,report=plan(records,per_event=args.focus_balance_count,world_size=world_size,epochs=3,seed=seed,exposure=exposure)
    work=[by_id[k] for k in report['ordered_ids']];next_exposure=Counter(exposure);next_exposure.update(report['ordered_ids'])
    report.update(policy=POLICY,epoch=epoch,seed=seed,
        action_counts=dict(Counter(train.choice_action_for_answers(x.spec) for x in work)),contiguous_priority=False,
        hard_action_quotas=False,next_history=dict(policy=POLICY,next_epoch=epoch+1,exposure=dict(next_exposure)))
    return work,report
