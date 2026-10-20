"""Train-only coverage selection; plans never advance actual exposure history.

The native samplers still set task/answer/subtype/variant quotas. Within those
quotas, choose least-exposed frames first, then soft action deficits and route
coverage. Evaluation continues to use the native, fixed samplers directly.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import heapq
import json
import math
from pathlib import Path
import random

from .training_policy import UNKNOWN, identity, physical_group, signature

POLICY = "coverage_task_action_v2"


def semantic_signature(row):
    # The last old action signature component was a fixed route. Permit new
    # routes here while preserving every answer/event/INVALID/highway subtype.
    return signature(row)[:-1]


def task_identity(item):
    return (*identity(item.row), item.balance_key,
            getattr(item, "augment_balance_key", item.spec.variant))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _encode(counter):
    return [[*key, value] for key, value in sorted(counter.items()) if value]


def _decode(records):
    return Counter({tuple(row[:-1]): int(row[-1]) for row in records})


class ExposureLedger:
    """Counts successful backwards, excluding skipped/zero-loss dummy inputs.

    History is shared across ranks only at epoch boundaries. Per-rank receipts
    are periodically persisted, including pending gradient-accumulation counts;
    they are audit records, not optimizer/RNG resume checkpoints.
    """
    def __init__(self, hints=None):
        self.hints = hints
        self.frames = Counter()
        self.tasks = Counter()
        self.actions = Counter()
        self.events = Counter()
        self.local_frames = Counter()
        self.local_tasks = Counter()
        self.local_actions = Counter()
        self.local_events = Counter()
        self.skipped = 0
        self.padded = 0
        self.pending_backward = 0
        self.applied_backward = 0

    def record(self, item, *, valid, padded=False):
        self.padded += int(padded)
        if not valid:
            self.skipped += 1
            return
        self.local_frames[identity(item.row)] += 1
        self.local_tasks[task_identity(item)] += 1
        self.local_actions[self.hints.label(item.row) if self.hints else UNKNOWN] += 1
        self.local_events[item.row.event] += 1
        self.pending_backward += 1

    def optimizer_applied(self):
        self.applied_backward += self.pending_backward
        self.pending_backward = 0

    def payload(self, step):
        return dict(policy=POLICY, step=int(step), scope="this_run_this_rank",
                    unit="successful_backward_presentations", resume_supported=False,
                    frames=_encode(self.local_frames), tasks=_encode(self.local_tasks),
                    actions=dict(self.local_actions), events=dict(self.local_events),
                    skipped=self.skipped, padded_presentations=self.padded,
                    pending_backward=self.pending_backward, applied_backward=self.applied_backward)

    def save(self, output, rank, step):
        _atomic_json(Path(output) / "exposure" / f"rank_{rank}.json", self.payload(step))

    def finish_run(self, output, *, rank, step, distributed=None):
        self.save(output, rank, step)
        local = dict(successful_backward=sum(self.local_frames.values()),
                     applied_backward=self.applied_backward, pending_backward=self.pending_backward,
                     skipped=self.skipped, padded_presentations=self.padded)
        records = [local]
        if distributed is not None and distributed.is_initialized():
            records = [None] * distributed.get_world_size()
            distributed.all_gather_object(records, local)
        if rank == 0:
            totals = {key: sum(r[key] for r in records) for key in local}
            # The existing audit packer copies top-level JSON only. Include a
            # compact summary there; large rank histories stay on the machine.
            receipt = dict(policy=POLICY, scope="all_ranks_actual", step=step, **totals,
                              unique_frames=len(self.frames), unique_task_pairs=len(self.tasks),
                              action_counts=dict(self.actions), source_event_counts=dict(self.events),
                              resume_supported=False)
            _atomic_json(Path(output) / "exposure" / "summary.json", receipt)
            _atomic_json(Path(output) / "exposure_summary.json", receipt)

    def finish_epoch(self, output, *, rank, step, epoch, distributed=None, planned=0):
        self.save(output, rank, step)
        payload = self.payload(step)
        records = [payload]
        if distributed is not None and distributed.is_initialized():
            records = [None] * distributed.get_world_size()
            distributed.all_gather_object(records, payload)
        frames, tasks, actions, events = Counter(), Counter(), Counter(), Counter()
        for record in records:
            frames.update(_decode(record["frames"]))
            tasks.update(_decode(record["tasks"]))
            actions.update(record["actions"])
            events.update(record["events"])
        delta = frames - self.frames
        new = set(frames) - set(self.frames)
        new_tasks = set(tasks) - set(self.tasks)
        new_routes = {physical_group(k) for k in frames} - {physical_group(k) for k in self.frames}
        task_counts = Counter()
        for key, count in (tasks - self.tasks).items():
            task_counts[key[-2]] += count
        report = dict(policy=POLICY, epoch=epoch, step=step, scope="all_ranks_actual",
                      unit="successful_backward_presentations", planned_presentations=planned,
                      actual_presentations=sum(delta.values()), unique_frames=len(delta),
                      new_frames=len(new), previously_seen_frames=len(delta)-len(new),
                      new_task_pairs=len(new_tasks), new_routes=len(new_routes),
                      cumulative_unique_frames=len(frames), cumulative_unique_task_pairs=len(tasks),
                      max_frame_repeat_this_epoch=max(delta.values(), default=0),
                      max_frame_exposure_cumulative=max(frames.values(), default=0),
                      task_counts=dict(task_counts), action_counts=dict(actions-self.actions),
                      source_event_counts=dict(events-self.events),
                      skipped_cumulative=sum(r["skipped"] for r in records),
                      padded_cumulative=sum(r["padded_presentations"] for r in records),
                      pending_backward=sum(r["pending_backward"] for r in records),
                      applied_backward=sum(r["applied_backward"] for r in records),
                      resume_supported=False)
        self.frames, self.tasks, self.actions, self.events = frames, tasks, actions, events
        if rank == 0:
            _atomic_json(Path(output) / "exposure" / f"epoch_{epoch:04d}.json", report)
            path = Path(output) / "exposure_epochs.json"
            reports = json.loads(path.read_text()) if path.exists() else []
            reports.append(report)
            _atomic_json(path, reports)
            print("[exposure-actual] " + json.dumps(report), flush=True)
        return report


class CoverageSampler:
    def __init__(self, rows, hints):
        self.hints = hints
        pools = defaultdict(dict)
        for row in rows:
            if row.split != "train":
                raise ValueError("Coverage selection only accepts train rows")
            label = hints.label(row)
            key = semantic_signature(row), label != UNKNOWN
            pools[key][identity(row)] = row
        self.pools = {key: list(pool.values()) for key, pool in pools.items()}

    def select(self, work, ledger, *, seed, strength=.5, max_repeat=10):
        if not 0 <= strength <= 1 or max_repeat < 1:
            raise ValueError("Invalid action strength or frame repetition limit")
        rng = random.Random(seed)
        used, route_used = Counter(), Counter()
        route_history = Counter()
        for key, count in ledger.frames.items():
            route_history[physical_group(key)] += count
        slots = defaultdict(list)
        for i, item in enumerate(work):
            slots[semantic_signature(item.row), self.hints.label(item.row) != UNKNOWN].append(i)
        out = list(work)
        # Randomize stratum/slot precedence reproducibly, avoiding a fixed focus
        # always getting first access to shared unseen physical frames.
        keys = list(slots)
        rng.shuffle(keys)
        strata = []
        for key in keys:
            candidates = self.pools[key]
            labels = [self.hints.label(row) for row in candidates]
            ids = [identity(row) for row in candidates]
            routes = [physical_group(k) for k in ids]
            ties = [rng.random() for _ in candidates]
            def priority(i):
                return (ledger.frames[ids[i]] + used[ids[i]],
                        route_history[routes[i]] + route_used[routes[i]], ties[i], i)
            heaps = defaultdict(list)
            for i, label in enumerate(labels):
                heaps[label].append(priority(i))
            for heap in heaps.values():
                heapq.heapify(heap)
            sizes = {label: len(heap) for label, heap in heaps.items()}
            natural = sum(sizes.values())
            roots = sum(math.sqrt(n) for n in sizes.values())
            weights = {label: (1-strength)*n/natural + strength*math.sqrt(n)/roots
                       for label, n in sizes.items()}
            chosen = Counter()
            indices = list(slots[key])
            rng.shuffle(indices)
            for slot in indices:
                tops = {}
                for label, heap in heaps.items():
                    while heap:
                        previous = heap[0]
                        i = previous[-1]
                        if used[ids[i]] >= max_repeat:
                            heapq.heappop(heap)
                        elif previous != priority(i):
                            # Priorities only increase. Lazy refresh also sees
                            # frames/routes consumed through other task strata.
                            heapq.heapreplace(heap, priority(i))
                        else:
                            tops[label] = previous
                            break
                if not tops:
                    raise ValueError("Coverage quota exceeds compatible frame capacity under "
                                     f"repeat cap {max_repeat}: task={work[slot].balance_key}, "
                                     f"slots={len(indices)}, unique_candidates={len(candidates)}. "
                                     "Reduce the quota; no silent duplication or relabeling.")
                # Exposure has priority over action. Thus action balancing can
                # never choose a repeatedly seen frame over an unseen compatible
                # frame merely to hit an action target.
                label = min(tops, key=lambda label: (
                    tops[label][0], chosen[label] - weights[label]*(sum(chosen.values())+1),
                    tops[label][1], tops[label][2]))
                i = heapq.heappop(heaps[label])[-1]
                out[slot] = replace(work[slot], row=candidates[i])
                used[ids[i]] += 1
                route_used[routes[i]] += 1
                chosen[label] += 1
                heapq.heappush(heaps[label], priority(i))
            picked = {identity(out[i].row) for i in indices}
            strata.append(dict(signature=key[0], annotations_available=key[1],
                               task_counts=dict(Counter(work[i].balance_key for i in indices)),
                               action_support=sizes, presentations=len(indices), available_unique_frames=len(candidates),
                               selected_unique_frames=len(picked),
                               new_frames=sum(ledger.frames[k] == 0 for k in picked),
                               remaining_unseen_frames=sum(ledger.frames[k] == 0 and k not in picked for k in ids)))
        before = Counter(self.hints.label(item.row) for item in work)
        after = Counter(self.hints.label(item.row) for item in out)
        task_counts = Counter(item.balance_key for item in out)
        task_frames = defaultdict(set)
        for item in out:
            task_frames[item.balance_key].add(identity(item.row))
        report = dict(policy=POLICY, scope="planned_not_consumed", source=self.hints.receipt,
                      strength=strength, before=dict(before), after=dict(after),
                      total_before=len(work), total_after=len(out),
                      replaced=sum(identity(a.row) != identity(b.row) for a,b in zip(work,out)),
                      unique_frames=len(used), new_frames=sum(ledger.frames[k] == 0 for k in used),
                      previously_seen_frames=sum(ledger.frames[k] > 0 for k in used),
                      new_task_pairs=len({task_identity(x) for x in out} - set(ledger.tasks)),
                      unique_routes=len(route_used), max_frame_repeat=max(used.values(), default=0),
                      repeat_presentations=len(work)-len(used),
                      historical_unique_frames=len(ledger.frames), strata=strata,
                      task_counts=dict(task_counts), task_unique_frames={k:len(v) for k,v in task_frames.items()},
                      priority="task/subtype/annotation-availability; least exposure; soft action; route coverage",
                      warning="Known/unknown annotation coverage is preserved. Counts are frames, not independent scenes; old runs are not imported.")
        if len(used) * 2 < len(work):
            report["repetition_warning"] = "More than half the planned presentations reuse frames within this epoch; inspect scarce strata before increasing quota."
        return out, report


def prepare_epoch_templates(first, *, seed, epoch_count, build, sampler, strength, max_repeat):
    """Exercise every scheduled native sampler before allocating the model.

    Cache native quota templates, not final selected frames: real exposure still
    determines coverage at each boundary. The dry history below is explicitly a
    projection assuming every selected item gets a successful backward.
    """
    templates, reports = {}, []
    projected = ExposureLedger()
    for epoch in range(epoch_count):
        epoch_seed = seed + epoch * 1_000_003
        try:
            template = first if epoch == 0 else build(epoch_seed)
            if len(template) != len(first):
                raise ValueError("Epoch changes the fixed training budget")
            selected, report = sampler.select(template, projected, seed=epoch_seed,
                                               strength=strength, max_repeat=max_repeat)
        except ValueError as exc:
            raise ValueError(f"Epoch {epoch+1} capacity preflight BEFORE model loading: {exc}") from exc
        templates[epoch_seed] = template
        reports.append(dict(epoch=epoch+1, seed=epoch_seed, presentations=len(selected),
                            unique_frames=report['unique_frames'], new_frames=report['new_frames'],
                            max_frame_repeat=report['max_frame_repeat']))
        projected.frames.update(identity(item.row) for item in selected)
        projected.tasks.update(task_identity(item) for item in selected)
    return templates, dict(scope="projection_not_actual_consumption",
                           assumption="all planned presentations succeed; real skips may change later selection",
                           epochs=reports)
