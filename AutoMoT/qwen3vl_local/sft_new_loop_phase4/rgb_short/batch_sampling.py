"""Current student trainer's sequence sampler; frozen native producer is unchanged."""
from qwen3vl_local.sequence_batch import plan as contiguous, phase4_records

# Source-bound training policy; admission agreement is covered by tests.
CRITICAL_EDGES=(('U-E2','return'), ('U-E3','proceed'), ('U-E5','proceed'),
                ('U-E6','proceed'), ('U-E7','complete'), ('R-E5','proceed'))


def plan(rows, *, epoch, seed=20260929, cap=8, budget=None, world_size=1,
         policy='contiguous_event_equal', expected_events=None, history=None,
         max_question_repeat=None, batch_size=1, anchor_stride=2):
    if policy != 'contiguous_event_equal':
        from ..sampling import plan as legacy
        return legacy(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size,
            policy=policy,expected_events=expected_events,history=history,max_question_repeat=4 if max_question_repeat is None else max_question_repeat)
    events={r['episode']['event'] for r in rows}
    if expected_events is not None and events != set(expected_events):
        raise ValueError('event pool mismatch')
    if any(r['target'] not in ('YES','NO') for r in rows):
        raise ValueError('binary supervision required')
    total=1024*len(events) if budget is None else budget
    if not events or type(total) is not int or total%len(events):
        raise ValueError('invalid event-equal budget')
    records=phase4_records(rows)
    supported=set()
    for row,record in zip(rows,records):
        basis=row.get('student_reference',{}).get('original_label_basis',row.get('label_basis'))
        record['reserve_eligible']=(row['split']=='train' and row.get('slice')=='readiness'
                                   and basis in ('weak_rule_teacher','approved_rule_teacher'))
        if record['reserve_eligible']:supported.add((record['event'],record['action']))
    minima={};missing=[]
    for event,edge in CRITICAL_EDGES:
        if event not in events:continue
        for answer in ('YES','NO'):
            action=edge+'/'+answer
            if (event,action) in supported:minima.setdefault(event,{})[action]=1
            else:missing.append(event+'/'+action)
    indices,audit=contiguous(records,per_event=total//len(events),epoch=epoch,
        seed=seed,world_size=world_size,batch_size=batch_size,stride=anchor_stride,history=history,
        max_question_repeat=max_question_repeat,action_minimums=minima)
    audit['unsupported_critical_cells']=missing
    audit['critical_scope']='one native automatic train/readiness question per supported critical edge/answer per epoch; missing support is never synthesized; not independent coverage'
    return indices,audit
