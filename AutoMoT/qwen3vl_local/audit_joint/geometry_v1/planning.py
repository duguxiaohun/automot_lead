"""Bounded event quotas, actual exposure first; never balance impossible cells."""
from collections import Counter, defaultdict, deque
import heapq
import random

from .dataset import digest, validate_splits


def plan(records, *, per_event=1024, world_size=4, epochs=3, seed=20261021,
         exposure=None, max_repeat=10, max_epoch_steps=4096, max_total_steps=12288):
    if any(type(v) is not int or v <= 0 for v in
           (per_event, world_size, epochs, max_repeat, max_epoch_steps, max_total_steps)):
        raise ValueError('positive_integer_budget_required')
    validate_splits(records)
    events = defaultdict(list)
    for r in records:
        if r['split'] == 'train' and any(t['valid'] for t in r['targets']):
            events[r['event']].append(r)
    if not events: raise ValueError('no_eligible_training_anchors')
    total = per_event*len(events)
    if total % world_size: raise ValueError('quota_not_divisible_by_world_size')
    steps = total//world_size
    if steps > max_epoch_steps or steps*epochs > max_total_steps:
        raise ValueError('training_step_budget_exceeded')
    exposure = Counter(exposure or {})
    if any(type(v) is not int or v < 0 for v in exposure.values()):
        raise ValueError('invalid_actual_exposure')
    known={r['id']:r for r in records}
    if set(exposure)-set(known):raise ValueError('exposure_contains_unknown_dataset_members')
    frame_exposure=Counter()
    for key,count in exposure.items():
        r=known[key];frame_exposure[(r['scenario'],r['route_id'],r['frame'])]+=count
    selected = []; planned = Counter(); physical = Counter(); routes = Counter()
    rng = random.Random(seed)
    frame_of = lambda r: (r['scenario'], r['route_id'], r['frame'])
    tie = {r['id']: rng.random() for event in sorted(events) for r in events[event]}
    by_frame = {}
    for event, rows in events.items():
        grouped = defaultdict(list)
        for r in rows: grouped[frame_of(r)].append(r)
        by_frame[event] = grouped
    allocations = defaultdict(list)
    owners = defaultdict(set)
    repairs = 0

    def priority(r):
        frame = frame_of(r)
        return (frame_exposure[frame]+physical[frame], exposure[r['id']]+planned[r['id']],
                routes[r['physical_group']], tie[r['id']], r['id'])

    def assign(r, index=None):
        if index is None:
            index = len(selected); selected.append(r)
        else:
            old = selected[index]; old_frame = frame_of(old)
            planned[old['id']] -= 1; physical[old_frame] -= 1; routes[old['physical_group']] -= 1
            slots = allocations[old['event'], old_frame]; slots.remove(index)
            if not slots: owners[old_frame].remove(old['event'])
            selected[index] = r
        frame = frame_of(r)
        allocations[r['event'], frame].append(index); owners[frame].add(r['event'])
        planned[r['id']] += 1; physical[frame] += 1; routes[r['physical_group']] += 1

    def augment(start):
        # Residual event -> frame -> occupying event paths. Reassign earlier
        # choices if needed; no path means a true capacitated matching deficit.
        queue = deque([start]); parent_event = {start: None}; parent_frame = {}
        end = None
        while queue and end is None:
            event = queue.popleft()
            choices = sorted((min(rows, key=priority) for rows in by_frame[event].values()), key=priority)
            for r in choices:
                frame = frame_of(r)
                if frame in parent_frame: continue
                parent_frame[frame] = (event, r)
                if physical[frame] < max_repeat:
                    end = frame; break
                for owner in sorted(owners[frame]):
                    if owner not in parent_event:
                        parent_event[owner] = frame; queue.append(owner)
        if end is None: raise ValueError('shared_frame_repeat_capacity_exceeded')
        while True:
            event, row = parent_frame[end]
            previous = parent_event[event]
            if previous is None:
                assign(row); return
            index = allocations[event, previous][-1]
            assign(row, index)
            end = previous

    for event, rows in sorted(events.items()):
        if len(rows)*max_repeat < per_event:
            raise ValueError(f'insufficient_capacity: {event}; no unlimited repetition')
        # Cycle through least-exposed candidates. Random tie keys avoid frame order bias.
        by_id={r['id']:r for r in rows}
        heap=[priority(r) for r in rows];heapq.heapify(heap)
        for _ in range(per_event):
            while heap:
                key=heapq.heappop(heap);r=by_id[key[-1]]
                if planned[r['id']]>=max_repeat or physical[(r['scenario'],r['route_id'],r['frame'])]>=max_repeat:
                    continue
                current=priority(r)
                # All priorities increase monotonically. Refresh stale route/frame
                # counts lazily, retaining exact min selection without O(N*quota).
                if current!=key:heapq.heappush(heap,current);continue
                break
            else:
                augment(event); repairs += 1
                # Rerouting decreases some priorities, invalidating lazy keys.
                heap = [priority(r) for r in rows]; heapq.heapify(heap)
                continue
            assign(r)
            heapq.heappush(heap,priority(r))
    rng.shuffle(selected)
    report = dict(ordered_ids=[r['id'] for r in selected], per_event=per_event,
        event_counts=dict(Counter(r['event'] for r in selected)),
        selected=len(selected), unique=sum(v>0 for v in planned.values()), optimizer_steps=steps,
        planned_total_steps=steps*epochs, world_size=world_size, batch_size=1, grad_accum=1,
        max_frame_repeat=max(physical.values()), repeated_fraction=1-sum(v>0 for v in planned.values())/len(selected),
        capacity_reassignments=repairs,
        query_counts=dict(Counter(f"{i}/{t['name']}" for r in selected for i,t in enumerate(r['targets']) if t['valid'])),
        actual_exposure_advanced=False, sampling='event_only_actual_exposure_then_route_diversity')
    report['plan_sha256'] = digest(report)
    return selected, report


def evaluation(records, split):
    if split not in ('val', 'test'): raise ValueError('invalid_evaluation_split')
    # Freeze all eligible supplied members, with no correctness-dependent re-selection.
    rows = sorted((r for r in records if r['split']==split), key=lambda r:r['id'])
    if not rows: raise ValueError('empty_evaluation_split')
    return rows
