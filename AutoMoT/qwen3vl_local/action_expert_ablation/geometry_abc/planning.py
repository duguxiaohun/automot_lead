"""Integer shared-physical-frame allocation before any model is loaded."""
from collections import Counter, defaultdict
import heapq
import math
import random
from qwen3vl_local.audit_joint.abc_delivery import digest
from .contracts import QUOTAS


def physical(r): return (r['scenario'],r['route_id'],r['frame'])


def allocate(demands,groups,cap):
    graph=[]
    def node(): graph.append([]);return len(graph)-1
    def edge(a,b,n,c):
        f=[b,len(graph[b]),n,c,n];graph[a].append(f);graph[b].append([a,len(graph[a])-1,0,-c,0]);return f
    source=node();sink=node();events={e:node() for e in sorted(demands)}
    for e,n in demands.items():edge(source,events[e],n,0)
    refs={};total=sum(demands.values());repeat_cost=total*(max((k[1] for k in groups),default=0)+cap+1)+1
    for key,frames in sorted(groups.items()):
        eligible,previous=key;n=len(frames);g=node()
        for e in eligible:refs[e,key]=edge(events[e],g,n*cap,0)
        for slot in range(cap):edge(g,sink,n,slot*repeat_cost+previous)
    flow=0;potential=[0]*len(graph)
    while flow<total:
        distance=[math.inf]*len(graph);parents=[None]*len(graph);distance[source]=0;heap=[(0,source)]
        while heap:
            cost,u=heapq.heappop(heap)
            if cost!=distance[u]:continue
            for i,e in enumerate(graph[u]):
                candidate=cost+e[3]+potential[u]-potential[e[0]]
                if e[2] and candidate<distance[e[0]]:
                    distance[e[0]]=candidate;parents[e[0]]=(u,i);heapq.heappush(heap,(candidate,e[0]))
        if parents[sink] is None:raise ValueError('joint_physical_capacity_exceeded_cap11')
        for i,d in enumerate(distance):
            if d<math.inf:potential[i]+=d
        amount=total-flow;v=sink
        while v!=source:
            u,i=parents[v];amount=min(amount,graph[u][i][2]);v=u
        v=sink
        while v!=source:
            u,i=parents[v];e=graph[u][i];e[2]-=amount;graph[v][e[1]][2]+=amount;v=u
        flow+=amount
    return {(e,k):f[4]-f[2] for (e,k),f in refs.items() if f[4]>f[2]}


def project(records, *, quotas=None,epochs=7,cap=11,world=4,accumulation=16,seed=2026):
    quotas=QUOTAS if quotas is None else quotas
    if not quotas or any(type(n)!=int or n<=0 for n in (*quotas.values(),epochs,cap,world,accumulation)):
        raise ValueError('positive_integer_budget_required')
    total=sum(quotas.values())
    if total%world:raise ValueError('unequal_DDP_shards')
    seen=set();splits={};frames=defaultdict(dict);natural=Counter()
    for r in records:
        if r['id'] in seen:raise ValueError('duplicate_member')
        seen.add(r['id'])
        group=r['physical_group']
        if splits.setdefault(group,r['split'])!=r['split']:raise ValueError('physical_split_leakage')
        if r['split']!='train':continue
        e=r['event']
        if e not in quotas:raise ValueError('unknown_event')
        f=physical(r)
        if e in frames[f]:raise ValueError('duplicate_event_physical_frame')
        frames[f][e]=r
        for cell,q in r['sampling_mass'].items():natural[e+'/'+cell]+=q
    cells=defaultdict(dict)
    for key,n in natural.items():cells[key.rsplit('/',1)[0]][key]=n
    weights={}
    for counts in cells.values():
        mass=sum(counts.values());z=sum(math.sqrt(n) for n in counts.values())
        weights.update({c:min(4.,.5+.5*mass/(math.sqrt(n)*z)) for c,n in counts.items()})
    support={e:sum(e in rows for rows in frames.values()) for e in quotas}
    deficits={e:quotas[e]-cap*n for e,n in support.items() if cap*n<quotas[e]}
    if deficits:raise ValueError('event_capacity_deficits: '+str(deficits))
    previous=Counter();plans=[];all_exposure=Counter();route_use=Counter();segment_use=Counter()
    for epoch in range(epochs):
        rng=random.Random(seed+epoch);groups=defaultdict(list)
        for f,rows in sorted(frames.items()):groups[tuple(sorted(rows)),previous[f]].append(f)
        assigned=allocate(quotas,groups,cap);used=Counter();ordered=[];counts=Counter();ties={}
        for f,rows in sorted(frames.items()):
            for e,r in rows.items():
                mass=r['sampling_mass'];den=sum(mass.values())
                if not den:raise ValueError('missing_region_sampling_mass')
                w=sum(q*weights[e+'/'+c] for c,q in mass.items())/den
                ties[e,f]=-math.log(max(rng.random(),1e-15))/w
        # Shared group members have identical eligibility; balanced assignment
        # realizes the integer flow while giving scarce events all their frames.
        for key,fs in sorted(groups.items()):
            for e in sorted(key[0],key=lambda e:(support[e],e)):
                n=assigned.get((e,key),0)
                # Static diversity rounds avoid O(N²) heap invalidation when many
                # anchors share a route. Region preference breaks ties within a
                # route/time-slice; round number prioritizes distinct groups.
                routes=defaultdict(lambda:defaultdict(list))
                for f in fs:routes[frames[f][e]['physical_group']][(*f[:2],f[2]//16)].append(f)
                diversity={}
                for g,segments in routes.items():
                    interleaved=[]
                    for segment,values in segments.items():
                        for i,f in enumerate(sorted(values,key=lambda f:(ties[e,f],f))):
                            interleaved.append((i,segment_use[segment],ties[e,f],f))
                    for i,entry in enumerate(sorted(interleaved)):
                        diversity[entry[-1]]=(i,route_use[g],entry[0])
                def priority(f):
                    return (used[f],previous[f],diversity[f],ties[e,f],f)
                heap=[priority(f) for f in fs if used[f]<cap];heapq.heapify(heap)
                for _ in range(n):
                    if not heap:raise AssertionError('flow_realization_failed')
                    item=heapq.heappop(heap);f=item[-1]
                    fresh=priority(f)
                    while item!=fresh:
                        heapq.heappush(heap,fresh);item=heapq.heappop(heap);f=item[-1];fresh=priority(f)
                    r=frames[f][e];ordered.append(r['id']);used[f]+=1;counts[e]+=1
                    route_use[r['physical_group']]+=1;segment_use[(*f[:2],f[2]//16)]+=1
                    if used[f]<cap:heapq.heappush(heap,priority(f))
        if dict(counts)!=quotas or len(ordered)!=total or max(used.values())>cap:
            raise AssertionError('budget_or_cap_mismatch')
        rng.shuffle(ordered);all_exposure.update(ordered)
        report=dict(epoch=epoch+1,ordered_ids=ordered,quotas=dict(counts),samples=total,
            optimizer_steps=math.ceil(total/world/accumulation),tail_global_samples=total%(world*accumulation) or world*accumulation,
            unique_physical=len(used),new_physical=sum(previous[f]==0 for f in used),
            physical_repeat_histogram=dict(Counter(used.values())),max_physical_repeat=max(used.values()),
            rank_samples=[len(ordered[r::world]) for r in range(world)])
        previous.update(used);report['cumulative_max_physical_repeat']=max(previous.values());plans.append(report)
    result=dict(policy='abc_shared_event_natural_sqrt_cap4_v1',plans=plans,support=support,
                total_presentations=sum(all_exposure.values()),total_updates=sum(p['optimizer_steps'] for p in plans),
                cumulative_physical_histogram=dict(Counter(previous.values())),independent_sampling=False)
    result['sha256']=digest(result)
    return result
