"""Phase3-style fixed event budget, capacity-aware cells and route diversity.

Production scans all eligible routes. An epoch is a sample of that pool, not an
exhaustive pass inflated to the largest event. Like Phase3 support_aware_quota,
only whole-event cycles may repeat; rare answer cells cannot inflate the epoch.
"""
from collections import Counter, defaultdict, deque
import math
import random
from .identity import digest
from .sampling import frame_key

POLICY = 'phase3_capacity_route_sampling_v1'
DEFAULT_PER_EVENT = 1024


def capacity_quota(capacities, target):
    """Even partial allocation with overflow returned to supported cells."""
    out = dict.fromkeys(sorted(capacities), 0)
    while target:
        active = [k for k in out if out[k] < capacities[k]]
        if not active:
            raise ValueError('partial quota exceeds capacity')
        share, extra = divmod(target, len(active))
        for j, k in enumerate(active):
            take = min(capacities[k] - out[k], share + (j < extra))
            out[k] += take
            target -= take
    return out


def route_order(indices, rows, counts, seed, epoch):
    groups = defaultdict(list)
    for i in indices:
        groups[rows[i]['physical_group']].append(i)
    queues = []
    for group in sorted(groups, key=lambda g: digest([seed, epoch, g])):
        queues.append(deque(sorted(groups[group], key=lambda i: (
            counts[rows[i]['id']], digest([seed, epoch, rows[i]['id']])))))
    out = []
    while queues:
        # Least-seen questions first; physical routes take turns within a round.
        queues.sort(key=lambda q: counts[rows[q[0]]['id']])
        for q in queues:
            out.append(q.popleft())
        queues = [q for q in queues if q]
    return out


def plan(rows, *, epoch, seed=20260929, budget=None, world_size=1,
         expected_events=None, history=None):
    if (not rows or type(epoch) is not int or epoch < 0
            or type(world_size) is not int or world_size < 1):
        raise ValueError('invalid Phase3-style sampling inputs')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate question IDs')
    pools = defaultdict(list)
    for i, row in enumerate(rows):
        if row['target'] not in ('YES', 'NO'):
            raise ValueError('binary supervision required')
        pools[row['episode']['event']].append(i)
    events = sorted(pools)
    if expected_events is not None and set(expected_events) != set(events):
        raise ValueError('event pool mismatch')
    unit = math.lcm(len(events), world_size)
    requested = math.ceil(DEFAULT_PER_EVENT * len(events) / unit) * unit if budget is None else budget
    if type(requested) is not int or requested < unit or requested % unit:
        raise ValueError(f'event/DDP budget must be a positive multiple of {unit}')
    quota = requested // len(events)
    binding = dict(policy=POLICY, seed=seed, world_size=world_size, budget=requested,
                   pool_sha256=digest(sorted([r['id'], r['episode']['event'], r['edge'],
                       r['target'], r['slice'], r['physical_group'], r.get('model_input_sha256'),
                       frame_key(r)] for r in rows)))
    if history is None:
        history = dict(**binding, next_epoch=0, counts=dict.fromkeys(ids, 0))
        for n in range(epoch):
            _, prior = plan(rows, epoch=n, seed=seed, budget=budget, world_size=world_size,
                            expected_events=expected_events, history=history)
            history = prior['next_history']
    if (not isinstance(history, dict) or any(history.get(k) != v for k,v in binding.items())
            or history.get('next_epoch') != epoch or set(history.get('counts', {})) != set(ids)):
        raise ValueError('sampling history pool/config/epoch mismatch')
    counts = history['counts']
    if (any(type(n) is not int or n < 0 for n in counts.values())
            or any(sum(counts[ids[i]] for i in pools[e]) != epoch * quota for e in events)):
        raise ValueError('invalid sampling exposure history')
    result = []
    for event in events:
        indices = pools[event]
        cycles, remainder = divmod(quota, len(indices))
        result.extend(sorted(indices, key=lambda i: ids[i]) * cycles)
        # Match Phase3's capacity-return behavior with transitions as action bins.
        edges = defaultdict(list)
        for i in indices:
            edges[rows[i]['edge']].append(i)
        edge_quota = capacity_quota({e: len(v) for e,v in edges.items()}, remainder)
        for edge in sorted(edges):
            cells = defaultdict(list)
            for i in edges[edge]:
                cells[(rows[i]['target'], rows[i]['slice'])].append(i)
            quotas = capacity_quota({k: len(v) for k,v in cells.items()}, edge_quota[edge])
            for cell in sorted(cells):
                result.extend(route_order(cells[cell], rows, counts, seed, epoch)[:quotas[cell]])
    random.Random(seed + epoch).shuffle(result)
    repeats = Counter(result)
    frames = Counter(frame_key(rows[i]) for i in result)
    updated = {ident: counts[ident] + repeats[i] for i, ident in enumerate(ids)}
    return result, dict(policy=POLICY, epoch=epoch, seed=seed, world_size=world_size,
        presentations=len(result), per_event_quota=quota, event_counts={e:quota for e in events},
        unique_questions=len(repeats), unique_frames=len(frames), max_frame_repeat=max(frames.values()),
        max_question_repeat=max(repeats.values()),
        event_repetition={e:dict(pool=len(pools[e]), presentations=quota,
            unique_questions=sum(i in repeats for i in pools[e]),
            max_question_repeat=max(repeats[i] for i in pools[e]),
            repetition_bound=math.ceil(quota / len(pools[e]))) for e in events},
        strata=dict(Counter('/'.join((rows[i]['episode']['event'], rows[i]['edge'],
                            rows[i]['target'], rows[i]['slice'])) for i in result)),
        cumulative_unique_questions=sum(n > 0 for n in updated.values()),
        cumulative_missing_questions=sum(n == 0 for n in updated.values()),
        event_coverage={e:dict(pool=len(pools[e]), seen=sum(updated[ids[i]] > 0 for i in pools[e])) for e in events},
        coverage_scope='all eligible routes in source pool; bounded sampled epochs, not exhaustive per epoch',
        repetition_policy='whole-event cycles only; per-question repeat <= ceil(event quota/event pool)',
        next_history=dict(**binding, next_epoch=epoch+1, counts=updated))
