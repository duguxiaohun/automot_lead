"""Independent base selection; bounded event quotas and completed-epoch exposure.

Capacity allocation is derived from the reviewed study planner; no geometric
labels or sequence sampling are used. Action weights break novelty ties only.
"""
from collections import Counter, defaultdict, deque
import heapq
import random
import math

from .phase3_base_dense.baseline import digest


def validate_splits(records):
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate_candidate_identity')
    if any(r['split'] != 'train' for r in records):
        raise ValueError('base_training_candidates_must_be_train')


def plan(records, *, per_event=1280, world_size=4, epochs=3, seed=20261021,
         exposure=None, max_repeat=1, max_epoch_steps=4096, max_total_steps=12288):
    if any(type(v) is not int or v <= 0 for v in
           (per_event, world_size, epochs, max_repeat, max_epoch_steps, max_total_steps)):
        raise ValueError('positive_integer_budget_required')
    if max_repeat != 1 or epochs > 3:
        raise ValueError("study_requires_one_per_frame_per_epoch_and_at_most_three_epochs")
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
    route_exposure=Counter(); segment_exposure=Counter()
    for key,count in exposure.items():
        r=known[key]
        if r['split'] != 'train': raise ValueError('evaluation_member_in_training_exposure')
        route_exposure[r['physical_group']] += count
        segment_exposure[(r['scenario'],r['route_id'],r['frame']//16)] += count
    selected = []; planned = Counter(); physical = Counter(); routes = Counter(); segments = Counter()
    rng = random.Random(seed)
    frame_of = lambda r: (r['scenario'], r['route_id'], r['frame'])
    # Half natural, half square-root future-target mass, within event/horizon/arm.
    # Static importance only breaks novelty ties; no impossible Cartesian quota.
    natural = Counter()
    for event,rows in events.items():
        for r in rows:
            for cell,q in r['sampling_mass'].items(): natural[event+'/'+cell] += q
    buckets=defaultdict(dict)
    for cell,n in natural.items(): buckets[cell.rsplit('/',1)[0]][cell]=n
    weights={}
    for cells in buckets.values():
        total=sum(cells.values()); z=sum(math.sqrt(n) for n in cells.values())
        for cell,n in cells.items(): weights[cell]=min(4., .5+.5*total/(math.sqrt(n)*z))
    tie={}
    for event in sorted(events):
        for r in sorted(events[event],key=lambda r:r['id']):
            mass=r['sampling_mass']; denominator=sum(mass.values())
            if denominator <= 0: raise ValueError('missing_shared_sampling_mass')
            weight=sum(q*weights[event+'/'+cell] for cell,q in mass.items())/denominator
            tie[r['id']]=-math.log(max(rng.random(),1e-15))/weight
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
        segment=(r['scenario'],r['route_id'],r['frame']//16)
        return (int(frame_exposure[frame]>0), route_exposure[r['physical_group']]+routes[r['physical_group']],
                segment_exposure[segment]+segments[segment], frame_exposure[frame]+physical[frame],
                exposure[r['id']]+planned[r['id']], tie[r['id']], r['id'])

    def assign(r, index=None):
        if index is None:
            index = len(selected); selected.append(r)
        else:
            old = selected[index]; old_frame = frame_of(old)
            planned[old['id']] -= 1; physical[old_frame] -= 1; routes[old['physical_group']] -= 1
            segments[(old['scenario'],old['route_id'],old['frame']//16)] -= 1
            slots = allocations[old['event'], old_frame]; slots.remove(index)
            if not slots: owners[old_frame].remove(old['event'])
            selected[index] = r
        frame = frame_of(r)
        allocations[r['event'], frame].append(index); owners[frame].add(r['event'])
        planned[r['id']] += 1; physical[frame] += 1; routes[r['physical_group']] += 1
        segments[(r['scenario'],r['route_id'],r['frame']//16)] += 1

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
                if physical[frame] < max_repeat and frame_exposure[frame]+physical[frame] < 3:
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
                if planned[r['id']]>=max_repeat or physical[frame_of(r)]>=max_repeat or frame_exposure[frame_of(r)]+physical[frame_of(r)]>=3:
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
        selected_action_mass=dict(sum((Counter({r['event']+'/'+cell:q for cell,q in r['sampling_mass'].items()}) for r in selected),Counter())),
        natural_action_mass=dict(natural),
        unseen_physical_frames=sum(frame_exposure[frame_of(r)]==0 for r in selected),
        route_groups=sum(v>0 for v in routes.values()), time_segments=sum(v>0 for v in segments.values()),
        max_run_frame_repeat=max(frame_exposure[f]+n for f,n in physical.items()),
        actual_exposure_advanced=False, sampling='independent_novelty_half_natural_half_sqrt_action_v2',
        temporal_segment_seconds=4, frame_cap_per_epoch=1, frame_cap_per_run=3)
    report['plan_sha256'] = digest(report)
    return selected, report



POLICY = 'independent_novelty_half_natural_half_sqrt_action_v2'


def independent_work(rows, args, *, epoch, world_size, history=None):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (
        case_identity, physical_group_for_item, unique_cases)
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    if args.batch_size != 1 or args.grad_accum != 1 or args.action_output_mode != 'choice':
        raise ValueError('Independent coverage requires choice B1 accumulation1')
    if epoch not in range(3): raise ValueError('Invalid coverage epoch')
    if history is None:
        if epoch: raise ValueError('Missing completed-epoch exposure')
        exposure = {}
    else:
        if history.get('policy') != POLICY or history.get('next_epoch') != epoch:
            raise ValueError('Foreign or incomplete sampling history')
        exposure = history['exposure']
        if sum(exposure.values()) != epoch * args.focus_balance_count * len(CONTEXT_IDS):
            raise ValueError('Incomplete successful-epoch exposure')
    # Stable candidate IDs; prompt option augmentation still follows each epoch.
    seed = int(args.seed) + epoch * 1_000_003
    items = unique_cases([train._make_item(r, seed=seed, action_output_mode='choice') for r in rows])
    items = [x for x in items if train.choice_rejection_reason(x.spec) is None]
    by_id = {digest(case_identity(x)): x for x in items}
    records = [dict(id=k, scenario=x.row.scenario, route_id=x.row.route_id,
        frame=x.row.frame_id, split=x.row.split, event=x.row.context_id,
        physical_group=physical_group_for_item(x), targets=[dict(valid=True)],
        sampling_mass={'current/0/'+train.choice_action_for_answers(x.spec):1.})
        for k,x in by_id.items()]
    if set(r['event'] for r in records) != set(CONTEXT_IDS):
        raise ValueError('All ten training events required')
    _, report = plan(records, per_event=args.focus_balance_count, world_size=world_size,
                     epochs=3, seed=seed, exposure=exposure)
    work = [by_id[k] for k in report['ordered_ids']]
    next_exposure = Counter(exposure); next_exposure.update(report['ordered_ids'])
    report.update(policy=POLICY, epoch=epoch, seed=seed,
        action_counts=dict(Counter(train.choice_action_for_answers(x.spec) for x in work)),
        contiguous_priority=False, hard_action_quotas=False,
        next_history=dict(policy=POLICY, next_epoch=epoch+1, exposure=dict(next_exposure)))
    # The native full-epoch loop advances this proposal only after successful
    # optimization and globally verified consumption. Interrupted runs cannot resume.
    return work, report
