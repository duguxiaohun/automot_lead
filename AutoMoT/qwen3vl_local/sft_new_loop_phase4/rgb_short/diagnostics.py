"""Reference diagnostics and compact runtime evidence; no changes to labels or selection."""
from collections import Counter
import json
from ..identity import digest


def binary_diagnostics(cases):
    targets = Counter(r['target'] for r in cases)
    predictions = Counter(r['prediction'] for r in cases)
    recalls = {answer: (sum(r['target'] == answer and r['prediction'] == answer for r in cases)
                        / targets[answer] if targets[answer] else None) for answer in ('YES', 'NO')}
    n = len(cases)
    agreement = sum(r['target'] == r['prediction'] for r in cases) / n if n else None
    majority = max(targets.values()) / n if n else None
    collapsed = next(iter(predictions)) if n and len(predictions) == 1 else None
    warnings = []
    if collapsed:
        warnings.append('constant_prediction:' + collapsed)
    if n and agreement <= majority:
        warnings.append('not_above_majority_reference_baseline')
    if recalls['YES'] == 0:
        warnings.append('zero_yes_recall')
    return dict(count=n, targets=dict(targets), predictions=dict(predictions), recall=recalls,
                balanced_reference_agreement=(sum(recalls.values()) / 2
                    if all(v is not None for v in recalls.values()) else None),
                majority_reference_baseline=majority, constant_prediction=collapsed, warnings=warnings)


def diagnostic_report(cases):
    return dict(overall=binary_diagnostics(cases), by_event={
        event: binary_diagnostics([r for r in cases if r['event'] == event])
        for event in sorted({r['event'] for r in cases})},
        scope='diagnostics only; does not change checkpoint selection or certify reference accuracy')


def sample_records(rows, indices, world):
    return [dict(position=position, rank=position % world, local_index=position // world,
                 id=rows[index]['id'], target=rows[index]['target'],
                 original_label_basis=rows[index].get('student_reference', {}).get(
                     'original_label_basis', rows[index]['label_basis']))
            for position, index in enumerate(indices)]


def sample_binding(records, world):
    return dict(global_order_sha256=digest(records), per_rank_order_sha256={
        str(rank): digest([r for r in records if r['rank'] == rank]) for rank in range(world)})


def append_step(path, record):
    # Each completed microbatch is durable at the Python-stream level, even if a later step fails.
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
