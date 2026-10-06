"""Equal events -> capacity-aware weighted edges -> balanced binary answers.

Defaults fit the smallest event pool. Large teacher pools cannot turn a tiny
manual event into an unlimited source when event equality/repeat caps are hard.
"""
from collections import Counter,defaultdict
import math
import random
from .identity import digest
from .sampling import Flow,CapacityError,frame_key

POLICY='event_weighted_edge_answer_motion_v2'
PRIMARY={'proceed','depart','enter','return','complete'}
WEIGHTS={k:8 for k in PRIMARY}|{'recover_follow':4}


def motion_stratum(row):
    q=row.get('teacher_provenance',{}).get('question',{})
    if q.get('ego_motion') in ('stationary','moving'):return q['ego_motion']
    speed=row.get('observation',{}).get('speed_mps')
    if isinstance(speed,(int,float)) and not isinstance(speed,bool) and math.isfinite(speed):
        return 'stationary' if abs(speed)<.5 else 'moving'
    return 'unknown'


def plan(rows,*,epoch,seed=20260929,cap=8,budget=None,world_size=1,expected_events=None,history=None,max_question_repeat=4):
    if (not rows or any(type(v) is not int or v<1 for v in (cap,world_size,max_question_repeat))
            or type(epoch) is not int or epoch<0):raise ValueError('invalid weighted sampling inputs')
    events=sorted({r['episode']['event'] for r in rows});ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate question IDs')
    if expected_events is not None and set(events)!=set(expected_events):raise ValueError('event pool mismatch')
    cells=defaultdict(list)
    for i,r in enumerate(rows):
        if r['target'] not in ('YES','NO'):raise ValueError('weighted sampling requires binary targets')
        cells[(r['episode']['event'],r['edge'],r['target'],motion_stratum(r),r.get('slice','readiness'))].append(i)
    edges={e:sorted({k[1] for k in cells if k[0]==e}) for e in events}
    capacities={k:sum(min(cap,len(v)*max_question_repeat) for v in _by_frame(rows,indices).values()) for k,indices in cells.items()}
    answer_caps=Counter()
    for k,n in capacities.items():answer_caps[k[:3]]+=n
    edge_caps={};units={};weights={}
    for event in events:
        for edge in edges[event]:
            present=[a for a in ('YES','NO') if (event,edge,a) in answer_caps]
            units[event,edge]=len(present)
            edge_caps[event,edge]=min(answer_caps[event,edge,a] for a in present)*len(present)
            weights[event,edge]=WEIGHTS.get(edge,1) if len(present)==2 else 1
    upper={e:sum(edge_caps[e,k] for k in edges[e]) for e in events}
    unit=math.lcm(2*len(events),world_size);step=unit//len(events)
    ceiling=min(upper.values(),default=0)
    requested=budget
    if requested is not None and (type(requested) is not int or requested<unit or requested%unit):raise ValueError(f'weighted budget must be divisible by {unit}')
    base_binding=dict(policy=POLICY,pool_sha256=digest(sorted(rows,key=lambda r:r['id'])),seed=seed,cap=cap,
                      world_size=world_size,max_question_repeat=max_question_repeat,edge_weights=WEIGHTS,budget_request=budget)
    if history is None:
        history=dict(**base_binding,next_epoch=0,counts={k:0 for k in ids},last_epoch={k:-1 for k in ids},total_presentations=0)
        for n in range(epoch):
            _,audit=plan(rows,epoch=n,seed=seed,cap=cap,budget=budget,world_size=world_size,expected_events=expected_events,history=history,max_question_repeat=max_question_repeat)
            history=audit['next_history']
    if (any(history.get(k)!=v for k,v in base_binding.items()) or history.get('next_epoch')!=epoch
            or set(history.get('counts',{}))!=set(ids) or set(history.get('last_epoch',{}))!=set(ids)):
        raise ValueError('sampling history pool/config/epoch mismatch')
    counts,last=history['counts'],history['last_epoch']
    if (any(type(counts[k]) is not int or counts[k]<0 or type(last[k]) is not int or not -1<=last[k]<epoch or (counts[k]==0)!=(last[k]==-1) for k in ids)
            or sum(counts.values())!=history.get('total_presentations')
            or len({sum(counts[r['id']] for r in rows if r['episode']['event']==e) for e in events})!=1):raise ValueError('invalid weighted exposure history')
    # A reproducible small rotation changes tie priority, not edge weights.
    ranks={k:digest([seed,epoch,k]) for k in edge_caps}
    def demands_for(quota):
        demand={};allocation={k:0 for k in edge_caps}
        for event in events:
            remaining=quota
            while remaining:
                possible=[(event,e) for e in edges[event] if units[event,e]<=remaining and allocation[event,e]+units[event,e]<=edge_caps[event,e]]
                if not possible:return None
                key=min(possible,key=lambda k:((allocation[k]+units[k])/weights[k],ranks[k]))
                allocation[key]+=units[key];remaining-=units[key]
            for edge in edges[event]:
                for answer in ('YES','NO'):
                    keys=[k for k in cells if k[:3]==(event,edge,answer)]
                    if not keys:continue
                    remaining_answer=allocation[event,edge]//units[event,edge]
                    # Capacity-aware equal motion strata inside each answer.
                    # Unknown metadata stays explicit, never guessed stationary.
                    for k in keys:demand[k]=0
                    while remaining_answer:
                        available=[k for k in keys if demand[k]<capacities[k]]
                        if not available:return None
                        phases={k[4] for k in available}
                        phase=min(phases,key=lambda p:((sum(demand[k] for k in keys if k[4]==p)+1)/(1 if p=='catchup' else 4),digest([seed,epoch,p])))
                        k=min((k for k in available if k[4]==phase),key=lambda k:(demand[k],digest([seed,epoch,k])))
                        demand[k]+=1;remaining_answer-=1
        return demand
    def allocate(demand):
        keys=sorted(cells);frames=sorted({frame_key(r) for r in rows});nodes={f:1+len(keys)+j for j,f in enumerate(frames)}
        sink=1+len(keys)+len(frames);flow=Flow(sink+1);links={};source={}
        for node,k in enumerate(keys,1):
            source[k]=flow.add(0,node,demand[k])
            indices=sorted(cells[k],key=lambda i:(counts[ids[i]],last[ids[i]],digest([seed,ids[i]])))
            grouped=_by_frame(rows,indices)
            links[k]={f:(flow.add(node,nodes[f],min(cap,len(v))),v) for f,v in grouped.items()}
        for node in nodes.values():flow.add(node,sink,cap)
        got=flow.run(0,sink)
        # Fill distinct questions before opening repeated exposures. In rich
        # cells the source quota is already satisfied in the first pass.
        # Residual flow may reassign shared frames to satisfy scarce cells.
        for repeat in range(2,max_question_repeat+1):
            if got==sum(demand.values()):break
            for per_cell in links.values():
                for edge,indices in per_cell.values():
                    edge[1]+=min(cap,len(indices)*repeat)-min(cap,len(indices)*(repeat-1))
            got+=flow.run(0,sink)
        return got,links,{'/'.join(k):e[1] for k,e in source.items() if e[1]}
    quota=(requested//len(events)) if requested is not None else (min(ceiling,math.ceil(len(rows)/len(events)))//step)*step
    attempts=0;deficits={}
    while quota>=step:
        demand=demands_for(quota);attempts+=1
        if demand is not None:
            got,links,deficits=allocate(demand)
            if got==quota*len(events):break
        if requested is not None:raise CapacityError(dict(policy=POLICY,requested=requested,event_capacity=upper,cell_deficits=deficits,max_question_repeat=max_question_repeat,shared_frame_cap=cap))
        quota-=step
    else:raise CapacityError(dict(policy=POLICY,event_capacity=upper,cell_deficits=deficits,reason='no feasible equal-event budget'))
    result=[];new_counts=dict(counts);new_last=dict(last);actual=Counter()
    for per_cell in links.values():
        for edge,indices in per_cell.values():
            for _ in range(edge[2][1]):
                available=[i for i in indices if actual[i]<max_question_repeat]
                i=min(available,key=lambda i:(new_counts[ids[i]],last[ids[i]],indices.index(i)))
                result.append(i);actual[i]+=1;new_counts[ids[i]]+=1;new_last[ids[i]]=epoch
    random.Random(seed+epoch).shuffle(result)
    edge_counts=Counter((rows[i]['episode']['event'],rows[i]['edge']) for i in result)
    cell_counts=Counter((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target']) for i in result)
    return result,dict(policy=POLICY,next_history=dict(**base_binding,next_epoch=epoch+1,counts=new_counts,last_epoch=new_last,total_presentations=history['total_presentations']+len(result)),
        epoch=epoch,presentations=len(result),requested_budget=budget,automatic_budget_attempts=attempts,per_event_quota=quota,event_capacity=upper,
        readiness_to_catchup_weight=4,
        edge_answer_phase_counts={'/'.join(k):n for k,n in Counter((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target'],rows[i].get('slice','readiness')) for i in result).items()},
        edge_weights=WEIGHTS,effective_edge_weights={'/'.join(k):v for k,v in weights.items()},
        edge_answer_motion_counts={'/'.join(k):n for k,n in Counter((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target'],motion_stratum(rows[i])) for i in result).items()},
        edge_counts={'/'.join(k):n for k,n in edge_counts.items()},edge_answer_counts={'/'.join(k):n for k,n in cell_counts.items()},
        event_counts=dict(Counter(rows[i]['episode']['event'] for i in result)),max_question_repeat=max(actual.values()),max_question_repeat_limit=max_question_repeat,
        max_frame_repeat=max(Counter(frame_key(rows[i]) for i in result).values()),unique_questions=len(actual),
        primary_edge_presentations=sum(n for (event,edge),n in edge_counts.items() if edge in PRIMARY),
        missing_cell_answers=[e+'/'+edge+'/'+a for e in events for edge in edges[e] for a in ('YES','NO') if (e,edge,a) not in answer_caps],
        cumulative_unique_questions=sum(v>0 for v in new_counts.values()),cumulative_missing_questions=sum(v==0 for v in new_counts.values()),
        event_coverage={e:dict(pool=sum(r['episode']['event']==e for r in rows),seen=sum(new_counts[r['id']]>0 for r in rows if r['episode']['event']==e)) for e in events})


def _by_frame(rows,indices):
    out=defaultdict(list)
    for i in indices:out[frame_key(rows[i])].append(i)
    return out
