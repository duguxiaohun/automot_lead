"""Fixed label-blind event-equal evaluation, without replacement."""
from collections import defaultdict, deque
import hashlib
import json

POLICY = 'fixed_ten_event_evaluation_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select(records, expected, *, seed=20261011, limit=0):
    """Records: (event, stable question ID, physical group). Labels never enter."""
    expected = tuple(expected)
    if not expected or len(set(expected)) != len(expected) or type(limit) is not int or limit < 0:
        raise ValueError('invalid evaluation event set or limit')
    owners = {}
    pools = defaultdict(lambda: defaultdict(list))
    for event, ident, group in records:
        if event not in expected or not ident or not group:
            raise ValueError('invalid evaluation event/question/group identity')
        if ident in owners:
            if owners[ident] != (event, group):
                raise ValueError('conflicting evaluation question identity')
            continue
        owners[ident] = (event, group)
        pools[event][group].append(ident)
    counts = {e: sum(map(len, pools[e].values())) for e in expected}
    missing = [e for e in expected if not counts[e]]
    if missing:
        raise ValueError(f'event-equal evaluation missing events: {missing}')
    quota = min(counts.values())
    if limit:
        quota = min(quota, limit)
    selected = []
    for event in expected:
        queues = [deque(sorted(ids, key=lambda ident: digest([seed, event, ident])))
                  for group, ids in sorted(pools[event].items(), key=lambda kv: digest([seed, event, kv[0]]))]
        n = 0
        while n < quota:
            for queue in queues:
                if queue and n < quota:
                    selected.append(queue.popleft())
                    n += 1
    return selected, dict(policy=POLICY, seed=seed, limit=limit, per_event=quota,
        available=counts, selected_ids=selected, ordered_ids_sha256=digest(selected),
        pool_sha256=digest(sorted((i, *v) for i, v in owners.items())),
        scope='question equality; not answer balance or independent label certification')


def select_work(work, *, seed=20261011, limit=0):
    from .context_taxonomy import CONTEXT_IDS
    from .invalid_balance import case_identity, unique_cases
    from .build_dataset import physical_route_group
    work = unique_cases(work)
    valid, invalid = {}, []
    for item in work:
        if item.row.answers.get('INVALID_ACTION_CONTEXT', False):
            invalid.append(item)
        else:
            valid[digest(case_identity(item))] = item
    ids, report = select([(item.row.context_id, ident,
                          physical_route_group(item.row.scenario, item.row.route_id))
                         for ident, item in valid.items()], CONTEXT_IDS, seed=seed, limit=limit)
    invalid.sort(key=lambda item: digest([seed, case_identity(item)]))
    report['invalid_unique_cases'] = len(invalid)
    report['invalid_policy'] = 'all eligible unique INVALID, separate from ten valid events'
    return [valid[ident] for ident in ids] + invalid, report
