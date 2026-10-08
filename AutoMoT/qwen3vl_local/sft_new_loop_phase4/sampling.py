"""事件等额配额，事件内转移/答案/来源轮转，全epoch共享物理帧cap。"""
from collections import Counter, defaultdict, deque
import random


def frame_key(row):
    return (row['scenario'],row['route_id'],row['observation']['frame_id'])


def legacy_plan(rows, *, epoch, seed=20260929, cap=8, budget=None, world_size=1):
    if type(epoch) is not int or epoch < 0 or cap < 1 or world_size < 1 or not rows:
        raise ValueError('invalid sampling inputs')
    rng = random.Random(seed)
    pools = defaultdict(list)
    for i,r in enumerate(rows):
        pools[(r['episode']['event'],r['edge'],r['target'],r['slice'])].append(i)
    queues = {}
    for k,ids in sorted(pools.items()):
        # Group diversity before returning to the same physical route.
        by_group = defaultdict(list)
        for i in ids:
            by_group[rows[i]['physical_group']].append(i)
        grouped = []
        for g in sorted(by_group):
            v = sorted(by_group[g],key=lambda i:rows[i]['id'])
            rng.shuffle(v)
            grouped.append(deque(v))
        order = []
        while any(grouped):
            for q in grouped:
                if q:
                    order.append(q.popleft())
        queues[k] = deque(order)
    # One fixed interleaving, one epoch cursor. Rotating independently at multiple
    # levels can lock budget truncation onto a strict subset (e.g. even indices).
    ring = []
    while any(queues.values()):
        for k in sorted(queues):
            if queues[k]:
                ring.append(queues[k].popleft())
    shift = epoch % len(ring)
    result,counts = [],Counter()
    for i in ring[shift:]+ring[:shift]:
        f = frame_key(rows[i])
        if counts[f] < cap:
            result.append(i)
            counts[f] += 1
    # Every question leads once per len(rows) epochs, so any feasible positive
    # budget includes it then, even with shared-frame caps. No persistent state.
    requested = (len(result)//world_size)*world_size if budget is None else budget
    if requested < world_size or requested % world_size or requested > len(result):
        raise ValueError(f'sampling capacity={len(result)}, requested={requested}, world_size={world_size}')
    result = result[:requested]
    actual = Counter(frame_key(rows[i]) for i in result)
    audit = dict(policy='transition_single_ring_v2',cursor=shift,coverage_bound_epochs=len(ring),epoch=epoch,seed=seed,cap=cap,
                 presentations=len(result),unique_frames=len(actual),max_frame_repeat=max(actual.values()),
                 strata=dict(Counter('/'.join((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target'],rows[i]['slice'])) for i in result)))
    return result,audit


class CapacityError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic=diagnostic
        super().__init__(f'event-equal sampling capacity insufficient: {diagnostic}')


class Flow:
    """Integer residual flow; event/frame nodes avoid expanding repeated samples."""
    def __init__(self,n):self.g=[[] for _ in range(n)]
    def add(self,u,v,cap):
        a=[v,cap,None];b=[u,0,a];a[2]=b
        self.g[u].append(a);self.g[v].append(b)
        return a
    def run(self,s,t):
        total=0
        while True:
            level={s:0};q=deque([s])
            while q:
                u=q.popleft()
                for v,c,_ in self.g[u]:
                    if c and v not in level:level[v]=level[u]+1;q.append(v)
            if t not in level:return total
            pos=[0]*len(self.g)
            def send(u,n):
                if u==t:return n
                while pos[u]<len(self.g[u]):
                    a=self.g[u][pos[u]];v,c,back=a
                    if c and level.get(v)==level[u]+1:
                        sent=send(v,min(n,c))
                        if sent:a[1]-=sent;back[1]+=sent;return sent
                    pos[u]+=1
                return 0
            while (sent:=send(s,10**18)):total+=sent


def plan(rows, *, epoch, seed=20260929, cap=8, budget=None, world_size=1,
         policy='event_equal', expected_events=None, history=None, max_question_repeat=4):
    if policy=='phase3_balanced':
        from .phase3_sampling import plan as phase3
        return phase3(rows,epoch=epoch,seed=seed,budget=budget,world_size=world_size,expected_events=expected_events,history=history)
    if policy=='full_event_equal':
        from .full_sampling import plan as full
        return full(rows,epoch=epoch,seed=seed,budget=budget,world_size=world_size,expected_events=expected_events,history=history)
    if policy=='event_weighted':
        from .weighted_sampling import plan as weighted
        return weighted(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size,expected_events=expected_events,history=history,max_question_repeat=max_question_repeat)
    if policy=='event_edge_answer':
        return balanced_plan(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size,expected_events=expected_events,history=history)
    if policy=='legacy_ring':
        return legacy_plan(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size)
    if policy!='event_equal':raise ValueError('unknown sampling policy')
    import math
    from .taxonomy import EVENTS
    if (type(epoch) is not int or epoch<0 or type(cap) is not int or cap<1
            or type(world_size) is not int or world_size<1 or not rows):
        raise ValueError('invalid sampling inputs')
    events=sorted({r['episode']['event'] for r in rows})
    if expected_events is not None and set(events)!=set(expected_events):
        raise ValueError(f'event pool mismatch: expected={sorted(expected_events)}, present={events}')
    unit=math.lcm(len(events),world_size)
    requested=math.ceil(len(rows)/unit)*unit if budget is None else budget
    if type(requested) is not int or requested<unit or requested%unit:
        raise ValueError(f'event 1:1 budget must be a positive multiple of {unit}; requested={requested}')
    quota=requested//len(events)
    from .identity import digest
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate question IDs in sampling pool')
    binding=dict(policy='event_equal_history_v2',
        pool_sha256=digest(sorted(rows,key=lambda r:r['id'])),seed=seed,cap=cap,
        budget=requested,world_size=world_size)
    if history is None:
        history=dict(**binding,next_epoch=0,counts={k:0 for k in ids},last_epoch={k:-1 for k in ids})
        # Random-access callers get the same deterministic history as training.
        # Sequential callers should pass next_history to avoid replaying prefixes.
        for previous in range(epoch):
            _,audit=plan(rows,epoch=previous,seed=seed,cap=cap,budget=budget,
                         world_size=world_size,expected_events=expected_events,history=history)
            history=audit['next_history']
    if (not isinstance(history,dict) or any(history.get(k)!=v for k,v in binding.items())
            or type(history.get('next_epoch')) is not int or history.get('next_epoch')!=epoch or set(history.get('counts',{}))!=set(ids)
            or set(history.get('last_epoch',{}))!=set(ids)):
        raise ValueError('sampling history pool/config/epoch mismatch')
    counts,last=history['counts'],history['last_epoch']
    if (any(type(counts[k]) is not int or counts[k]<0 or type(last[k]) is not int
            or not -1<=last[k]<epoch or (counts[k]==0)!=(last[k]==-1) for k in ids)
            or any(sum(counts[r['id']] for r in rows if r['episode']['event']==e)!=epoch*quota for e in events)):
        raise ValueError('invalid sampling exposure history')
    # Only actual selections advance exposure. Prefer the least-seen question,
    # then the longest-unselected one, retaining the fixed stratified ring ties.
    rings={};frames=set()
    for event in events:
        ids=[i for i,r in enumerate(rows) if r['episode']['event']==event]
        local=[rows[i] for i in ids]
        order,_=legacy_plan(local,epoch=0,seed=seed,cap=len(local),budget=len(local))
        rings[event]=sorted((ids[i] for i in order),key=lambda i:(counts[rows[i]['id']],last[rows[i]['id']]))
        frames.update(frame_key(rows[i]) for i in ids)
    frame_nodes={f:i+1+len(events) for i,f in enumerate(sorted(frames))}
    sink=1+len(events)+len(frames);flow=Flow(sink+1);links={};demands={}
    for e,event in enumerate(events,1):
        demands[event]=flow.add(0,e,quota)
        grouped=defaultdict(list)
        for i in rings[event]:grouped[frame_key(rows[i])].append(i)
        links[event]={f:(flow.add(e,frame_nodes[f],min(cap,len(ids))),ids)
                      for f,ids in grouped.items()}
    for node in frame_nodes.values():flow.add(node,sink,cap)
    obtained=flow.run(0,sink)
    # First use distinct questions where feasible, then allow repetition within
    # the SAME global physical-frame cap. Never reassign an event's quota.
    for per_event in links.values():
        for edge,ids in per_event.values():edge[1]+=cap-min(cap,len(ids))
    obtained+=flow.run(0,sink)
    if obtained!=requested:
        raise CapacityError(dict(requested=requested,per_event_quota=quota,achieved=obtained,
            event_deficits={e:link[1] for e,link in demands.items() if link[1]},
            standalone_frame_capacity={e:len(links[e])*cap for e in events},
            global_frame_capacity=len(frames)*cap,shared_frame_cap=cap))
    queues={}
    for event in events:
        allocated=Counter()
        for _,(edge,ids) in links[event].items():
            for j in range(edge[2][1]):
                i=min(ids,key=lambda i:(counts[rows[i]['id']]+allocated[i],last[rows[i]['id']],ids.index(i)))
                allocated[i]+=1
        selected=[]
        while allocated:
            for i in rings[event]:
                if allocated[i]:
                    selected.append(i);allocated[i]-=1
                    if not allocated[i]:del allocated[i]
        queues[event]=deque(selected)
    result=[]
    while any(queues.values()):
        for event in events:result.append(queues[event].popleft())
    # Deterministic rank distribution; a rank must not always receive one event.
    random.Random(seed+epoch).shuffle(result)
    actual=Counter(frame_key(rows[i]) for i in result)
    new_counts=dict(counts);new_last=dict(last)
    for i in result:
        key=rows[i]['id'];new_counts[key]+=1;new_last[key]=epoch
    next_history=dict(**binding,next_epoch=epoch+1,counts=new_counts,last_epoch=new_last)
    by_event={e:dict(pool=sum(r['episode']['event']==e for r in rows),
        seen=sum(new_counts[r['id']]>0 for r in rows if r['episode']['event']==e)) for e in events}
    return result,dict(policy='event_equal_history_v2',next_history=next_history,
        cumulative_unique_questions=sum(v>0 for v in new_counts.values()),
        cumulative_missing_questions=sum(v==0 for v in new_counts.values()),event_coverage=by_event,epoch=epoch,seed=seed,cap=cap,
        presentations=len(result),per_event_quota=quota,event_counts=dict(Counter(rows[i]['episode']['event'] for i in result)),
        missing_events=sorted(set(expected_events or EVENTS)-set(events)),
        unique_questions=len(set(result)),unique_frames=len(actual),max_frame_repeat=max(actual.values()),
        coverage_scope='actual exposure history; least-seen then longest-unselected; shared-frame competition may constrain coverage',
        strata=dict(Counter('/'.join((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target'],rows[i]['slice'])) for i in result)))


def coverage_schedule(rows, *, epochs=7, seed=20260929, cap=8, budget=None, world_size=1, policy='event_equal', max_question_repeat=4):
    """Finite-run coverage, separate from the existence of a feasible epoch."""
    if type(epochs) is not int or epochs<1:raise ValueError('invalid coverage epoch count')
    if policy=='full_event_equal':
        # Exhaustive cycles guarantee coverage in every epoch. Do not materialize
        # millions of repetitions 7 times merely to prove the same invariant.
        try:
            _,audit=plan(rows,epoch=0,seed=seed,cap=cap,budget=budget,world_size=world_size,policy=policy,max_question_repeat=max_question_repeat)
        except CapacityError as ex:
            return dict(feasible=False,failed_epoch=0,diagnostic=ex.diagnostic,timeline=[],complete=False)
        return dict(feasible=True,epochs=epochs,world_size=world_size,pool=len(rows),
                    timeline=[dict(epoch=e,seen=len(rows),missing=0,event_coverage=audit['event_coverage']) for e in range(epochs)],
                    complete=True,missing_ids=[],coverage_basis='exhaustive-cycle invariant; actual per-epoch ordering is recorded during training')
    history=None;timeline=[]
    for epoch in range(epochs):
        try:
            _,audit=plan(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size,history=history,policy=policy,max_question_repeat=max_question_repeat)
        except CapacityError as ex:
            return dict(feasible=False,failed_epoch=epoch,diagnostic=ex.diagnostic,timeline=timeline,complete=False)
        history=audit.pop('next_history');timeline.append(dict(epoch=epoch,
            seen=audit['cumulative_unique_questions'],missing=audit['cumulative_missing_questions'],
            event_coverage=audit['event_coverage']))
    return dict(feasible=True,epochs=epochs,world_size=world_size,pool=len(rows),timeline=timeline,
                complete=timeline[-1]['missing']==0,missing_ids=sorted(k for k,n in history['counts'].items() if not n))


def balanced_plan(rows,*,epoch,seed=20260929,cap=8,budget=None,world_size=1,expected_events=None,history=None):
    """Equal events, equal present edge/answer cells; shared frame cap is hard.

    Rare cells may make a requested quota infeasible. Fail with cell deficits,
    never silently replace scarce YES with waiting NO. Exposure stays causal
    and checkpoint-bound. This policy does not promise seven-epoch coverage.
    """
    import math
    from .identity import digest
    if (not rows or type(epoch) is not int or epoch<0 or type(cap) is not int or cap<1
            or type(world_size) is not int or world_size<1):raise ValueError('invalid balanced sampling inputs')
    events=sorted({r['episode']['event'] for r in rows})
    if expected_events is not None and set(expected_events)!=set(events):raise ValueError('event pool mismatch')
    unit=math.lcm(len(events),world_size);requested=math.ceil(len(rows)/unit)*unit if budget is None else budget
    if type(requested) is not int or requested<unit or requested%unit:raise ValueError('balanced budget must be divisible by event/world count')
    quota=requested//len(events);ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate question IDs')
    binding=dict(policy='event_edge_answer_history_v1',pool_sha256=digest(sorted(rows,key=lambda r:r['id'])),seed=seed,cap=cap,budget=requested,world_size=world_size)
    if history is None:
        history=dict(**binding,next_epoch=0,counts={k:0 for k in ids},last_epoch={k:-1 for k in ids})
        for n in range(epoch):
            _,a=balanced_plan(rows,epoch=n,seed=seed,cap=cap,budget=budget,world_size=world_size,expected_events=expected_events,history=history);history=a['next_history']
    if (any(history.get(k)!=v for k,v in binding.items()) or history.get('next_epoch')!=epoch
            or set(history.get('counts',{}))!=set(ids) or set(history.get('last_epoch',{}))!=set(ids)):raise ValueError('sampling history pool/config/epoch mismatch')
    counts,last=history['counts'],history['last_epoch']
    if (any(type(counts[k]) is not int or counts[k]<0 or type(last[k]) is not int or not -1<=last[k]<epoch
            or (counts[k]==0)!=(last[k]==-1) for k in ids)
            or any(sum(counts[r['id']] for r in rows if r['episode']['event']==e)!=epoch*quota for e in events)):
        raise ValueError('invalid sampling exposure history')
    cells=defaultdict(list)
    for i,r in enumerate(rows):cells[(r['episode']['event'],r['edge'],r['target'])].append(i)
    demands={}
    for event in events:
        keys=sorted(k for k in cells if k[0]==event);shift=(epoch*(quota%len(keys)))%len(keys);keys=keys[shift:]+keys[:shift]
        demands.update({k:quota//len(keys)+(j<quota%len(keys)) for j,k in enumerate(keys)})
    keys=sorted(cells);frames=sorted({frame_key(r) for r in rows});nodes={f:1+len(keys)+j for j,f in enumerate(frames)}
    sink=1+len(keys)+len(frames);flow=Flow(sink+1);links={};source={}
    for node,k in enumerate(keys,1):
        source[k]=flow.add(0,node,demands[k]);grouped=defaultdict(list)
        for i in sorted(cells[k],key=lambda i:(counts[ids[i]],last[ids[i]],digest([seed,ids[i]]))):grouped[frame_key(rows[i])].append(i)
        links[k]={f:(flow.add(node,nodes[f],min(cap,len(v))),v) for f,v in grouped.items()}
    for node in nodes.values():flow.add(node,sink,cap)
    obtained=flow.run(0,sink)
    for per_cell in links.values():
        for edge,v in per_cell.values():edge[1]+=cap-min(cap,len(v))
    obtained+=flow.run(0,sink)
    if obtained!=requested:
        raise CapacityError(dict(policy='event_edge_answer',requested=requested,per_event_quota=quota,achieved=obtained,shared_frame_cap=cap,
            cell_deficits={'/'.join(k):e[1] for k,e in source.items() if e[1]},guidance='reduce epoch budget or expand scarce cell routes; NO cannot replace YES'))
    result=[];new_counts=dict(counts);new_last=dict(last)
    for k in keys:
        for edge,v in links[k].values():
            for _ in range(edge[2][1]):
                i=min(v,key=lambda i:(new_counts[ids[i]],last[ids[i]],v.index(i)))
                result.append(i);new_counts[ids[i]]+=1;new_last[ids[i]]=epoch
    random.Random(seed+epoch).shuffle(result)
    return result,dict(policy=binding['policy'],next_history=dict(**binding,next_epoch=epoch+1,counts=new_counts,last_epoch=new_last),
        cumulative_unique_questions=sum(n>0 for n in new_counts.values()),cumulative_missing_questions=sum(n==0 for n in new_counts.values()),
        epoch=epoch,per_event_quota=quota,presentations=len(result),event_counts=dict(Counter(rows[i]['episode']['event'] for i in result)),
        edge_answer_counts={'/'.join(k):sum(rows[i]['episode']['event']==k[0] and rows[i]['edge']==k[1] and rows[i]['target']==k[2] for i in result) for k in keys},
        unique_questions=len(set(result)),max_frame_repeat=max(Counter(frame_key(rows[i]) for i in result).values()),
        missing_cell_answers=[e+'/'+edge+'/'+a for e,edge in sorted({k[:2] for k in keys}) for a in ('YES','NO') if (e,edge,a) not in cells],
        event_coverage={e:dict(pool=sum(r['episode']['event']==e for r in rows),seen=sum(new_counts[r['id']]>0 for r in rows if r['episode']['event']==e)) for e in events})
