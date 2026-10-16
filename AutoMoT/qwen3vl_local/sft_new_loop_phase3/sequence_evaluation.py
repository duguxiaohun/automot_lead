"""Fixed event-equal evaluation: contiguous anchor groups, label-blind membership."""
from collections import defaultdict
from qwen3vl_local.sequence_batch import digest

POLICY='fixed_equal_contiguous_evaluation_v2'


def select(records, expected, *, seed=20261011, limit=0, stride=1):
    expected=tuple(expected)
    if not expected or len(set(expected))!=len(expected) or limit<0 or stride not in (1,2):
        raise ValueError('invalid fixed evaluation configuration')
    pools=defaultdict(lambda:defaultdict(list));seen=set()
    for r in records:
        if r['id'] in seen:raise ValueError('duplicate evaluation ID')
        if r['event'] not in expected:raise ValueError('unexpected evaluation event')
        seen.add(r['id']);pools[r['event']][r['sequence']].append(r)
    available={e:sum(map(len,pools[e].values())) for e in expected}
    if any(n==0 for n in available.values()):raise ValueError('event-equal evaluation missing events')
    quota=min(available.values())
    if limit:quota=min(quota,limit)
    chosen=[]
    for event in expected:
        # Same-frame questions are alternatives, not artificial temporal gaps.
        # Rebuild after each consumed run so remaining alternatives remain eligible.
        remaining={sequence:defaultdict(list) for sequence in pools[event]}
        for sequence,rows in pools[event].items():
            for row in sorted(rows,key=lambda r:digest([seed,r['id']])):
                remaining[sequence][row['frame']].append(row)
        selected=[]
        while len(selected)<quota:
            pieces=[]
            for sequence,frames in remaining.items():
                for residue in range(stride):
                    run=[]
                    for frame in sorted(f for f,values in frames.items() if values and f%stride==residue):
                        if run and frame!=run[-1]+stride:
                            pieces.append((sequence,run));run=[]
                        run.append(frame)
                    if run:pieces.append((sequence,run))
            sequence,frames=min(pieces,key=lambda p:(-len(p[1]),digest([seed,event,p[0],p[1]])))
            for frame in frames[:quota-len(selected)]:
                selected.append(remaining[sequence][frame].pop(0))
        chosen.extend(r['id'] for r in selected)
    return chosen,dict(policy=POLICY,seed=seed,limit=limit,per_event=quota,available=available,
        anchor_stride=stride,selected_ids=chosen,ordered_ids_sha256=digest(chosen),
        membership_sha256=digest(sorted(chosen)),pool_sha256=digest(sorted(records,key=lambda r:r['id'])),
        scope='fixed label-blind longest anchor runs across routes, no replacement; stride may change membership')


def select_work(work, *, seed=20261011, limit=0, stride=1):
    from .context_taxonomy import CONTEXT_IDS
    from .invalid_balance import case_identity,unique_cases
    work=unique_cases(work);valid={};invalid=[]
    for item in work:
        if item.row.answers.get('INVALID_ACTION_CONTEXT',False):invalid.append(item)
        else:valid[digest(case_identity(item))]=item
    records=[dict(id=ident,event=item.row.context_id,
        sequence=digest([item.row.scenario,item.row.route_id,item.row.context_id]),frame=item.row.frame_id)
        for ident,item in valid.items()]
    ids,report=select(records,CONTEXT_IDS,seed=seed,limit=limit,stride=stride)
    invalid.sort(key=lambda item:digest([seed,case_identity(item)]))
    report.update(invalid_unique_cases=len(invalid),invalid_policy='separate eligible unique INVALID')
    return [valid[i] for i in ids]+invalid,report
