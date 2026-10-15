"""Current student trainer's sequence sampler; frozen native producer is unchanged."""
from qwen3vl_local.sequence_batch import plan as contiguous, phase4_records


def plan(rows, *, epoch, seed=20260929, cap=8, budget=None, world_size=1,
         policy='contiguous_event_equal', expected_events=None, history=None,
         max_question_repeat=4, batch_size=1, anchor_stride=2):
    if policy != 'contiguous_event_equal':
        from ..sampling import plan as legacy
        return legacy(rows,epoch=epoch,seed=seed,cap=cap,budget=budget,world_size=world_size,
            policy=policy,expected_events=expected_events,history=history,max_question_repeat=max_question_repeat)
    events={r['episode']['event'] for r in rows}
    if expected_events is not None and events != set(expected_events):
        raise ValueError('event pool mismatch')
    if any(r['target'] not in ('YES','NO') for r in rows):
        raise ValueError('binary supervision required')
    total=1024*len(events) if budget is None else budget
    if not events or type(total) is not int or total%len(events):
        raise ValueError('invalid event-equal budget')
    return contiguous(phase4_records(rows),per_event=total//len(events),epoch=epoch,
        seed=seed,world_size=world_size,batch_size=batch_size,stride=anchor_stride,history=history)
