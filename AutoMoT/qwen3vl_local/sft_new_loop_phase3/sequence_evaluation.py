"""Fixed event-equal evaluation: contiguous anchor groups, label-blind membership."""
from collections import defaultdict, deque
from qwen3vl_local.sequence_batch import digest

POLICY='fixed_equal_contiguous_evaluation_v1'


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
        queues=[]
        for sequence,rows in sorted(pools[event].items(),key=lambda kv:digest([seed,event,kv[0]])):
            # Stable blocks of up to eight consecutive anchors; both modes use
            # identical membership. Gaps and same-frame questions start new pieces.
            ordered=sorted(rows,key=lambda r:(r['frame'],r['id']))
            pieces=[];piece=[]
            for row in ordered:
                if piece and (row['frame']!=piece[-1]['frame']+1 or len(piece)==8):
                    pieces.append(piece);piece=[]
                piece.append(row)
            if piece:pieces.append(piece)
            pieces.sort(key=lambda p:(-len(p),digest([seed,p[0]['id']])))
            queues.append(deque(pieces))
        selected=[]
        while len(selected)<quota:
            for queue in queues:
                if queue and len(selected)<quota:selected.extend(queue.popleft()[:quota-len(selected)])
        # Reorder chosen anchors only: no mode-specific label/answer selection.
        selected.sort(key=lambda r:(digest([seed,event,r['sequence']]),r['frame']%stride,r['frame'],r['id']))
        chosen.extend(r['id'] for r in selected)
    return chosen,dict(policy=POLICY,seed=seed,limit=limit,per_event=quota,available=available,
        anchor_stride=stride,selected_ids=chosen,ordered_ids_sha256=digest(chosen),
        membership_sha256=digest(sorted(chosen)),pool_sha256=digest(sorted(records,key=lambda r:r['id'])),
        scope='fixed label-blind membership, no replacement; modes differ only in traversal order')


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
