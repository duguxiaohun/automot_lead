"""按事件/转移/答案/样本来源轮转，物理帧全局cap；缺类别不伪造。"""
from collections import Counter, defaultdict, deque
import random


def frame_key(row):
    return (row['scenario'],row['route_id'],row['observation']['frame_id'])


def plan(rows, *, epoch, seed=20260929, cap=8, budget=None, world_size=1):
    if cap < 1 or world_size < 1 or not rows:
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
            shift = epoch % len(v)
            grouped.append(deque(v[shift:]+v[:shift]))
        order = []
        while any(grouped):
            for q in grouped:
                if q:
                    order.append(q.popleft())
        shift = epoch % len(order)
        queues[k] = deque(order[shift:]+order[:shift])
    keys = sorted(queues)
    shift = epoch % len(keys)
    keys = keys[shift:]+keys[:shift]
    result,counts = [],Counter()
    # Each unique question at most once/epoch; cap applies across every question/instance sharing an image.
    while any(queues.values()):
        for k in keys:
            q = queues[k]
            while q:
                i = q.popleft()
                f = frame_key(rows[i])
                if counts[f] < cap:
                    result.append(i)
                    counts[f] += 1
                    break
    requested = (len(result)//world_size)*world_size if budget is None else budget
    if requested < world_size or requested % world_size or requested > len(result):
        raise ValueError(f'sampling capacity={len(result)}, requested={requested}, world_size={world_size}')
    result = result[:requested]
    actual = Counter(frame_key(rows[i]) for i in result)
    audit = dict(policy='transition_strata_route_cycle_v1',epoch=epoch,seed=seed,cap=cap,
                 presentations=len(result),unique_frames=len(actual),max_frame_repeat=max(actual.values()),
                 strata=dict(Counter('/'.join((rows[i]['episode']['event'],rows[i]['edge'],rows[i]['target'],rows[i]['slice'])) for i in result)))
    return result,audit
