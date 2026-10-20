"""Bounded Phase1/2 training and optional, read-only legacy action strata.

This module has no model/teacher imports. Action annotations affect selection only;
the existing Phase1/2 target, prompt, split and evaluation samplers remain authoritative.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

POLICY = "coverage_task_action_v2"
ACTION_KEYS = ("STOP", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "DECELERATE", "RESUME")
UNKNOWN = "UNAVAILABLE"


def identity(row):
    if isinstance(row, dict):
        return str(row["scenario"]), str(row["route_id"]), int(row["frame_id"])
    return row.scenario, row.route_id, int(row.frame_id)


def signature(row):
    # Preserve EVERY answer and existing subgroup, including highway and invalid.
    return (tuple(sorted(row.answers.items())), row.event,
            getattr(row, "rs", ""), getattr(row, "true_rs", ""),
            getattr(row, "question_domain", ""), getattr(row, "invalid_source", ""),
            getattr(row, "ue3_subtype", ""),
            # Phase2 deliberately preserves route diversity.
            (row.scenario, row.route_id) if hasattr(row, "true_rs") else ())


def physical_group(key):
    scenario, route, _ = key
    route = re.sub(r"_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", "", route)
    route = re.sub(r"_Rep\d+", "", route)
    route = re.sub(r"_route\d+$", "", route)
    return scenario, route


def _action(obj):
    if obj.get("invalid_action_context") or (obj.get("answers") or {}).get("INVALID_ACTION_CONTEXT"):
        return UNKNOWN
    explicit = obj.get("primary_action")
    if explicit is not None:
        if explicit in (*ACTION_KEYS, "KEEP", "NONE"):
            return "KEEP" if explicit == "NONE" else explicit
        return UNKNOWN
    answers = obj.get("answers") or {}
    if any(type(answers.get(k)) is not bool for k in ACTION_KEYS):
        return UNKNOWN
    selected = {key for key in ACTION_KEYS if answers[key]}
    if len(selected & {"STOP", "DECELERATE", "RESUME"}) > 1 or len(selected & {"LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"}) > 1:
        return UNKNOWN
    return next((key for key in ACTION_KEYS if key in selected), "KEEP")


class ActionHints:
    def __init__(self, values=None, receipt=None):
        self.values = values or {}
        self.receipt = receipt or {"status": "disabled", "labels_generated": False}

    def label(self, row):
        return self.values.get(identity(row), UNKNOWN)

    @classmethod
    def load(cls, requested, rows, automot_root):
        if requested == "off":
            return cls()
        path = Path(automot_root) / "checkpoints/sft_new_loop_phase3_data_v23/frame_index.jsonl" if requested == "auto" else Path(requested).expanduser()
        if not path.is_file():
            if requested != "auto":
                raise ValueError(f"explicit ACTION_INDEX is missing: {path}; no generation or fallback")
            return cls(receipt={"status": "unavailable", "path": str(path), "labels_generated": False,
                               "warning": "No legacy action index: task balance only. Set ACTION_INDEX to an existing local index."})
        wanted = {identity(row) for row in rows if row.split == "train"}
        observed = defaultdict(set)
        excluded = set()
        digest = hashlib.sha256()
        stats = Counter()
        with path.open("rb") as handle:
            for number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                    key = identity(obj)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid legacy action record {path}:{number}: {exc}") from exc
                if obj.get("split") != "train":
                    excluded.add(physical_group(key))
                    stats["excluded_nontrain_records"] += 1
                    continue
                if key not in wanted:
                    continue
                label = _action(obj)
                if label != UNKNOWN:
                    observed[key].add(label)
                else:
                    stats["unusable_records"] += 1
        values = {key: next(iter(labels)) for key, labels in observed.items() if len(labels) == 1 and physical_group(key) not in excluded}
        stats["ambiguous_frames"] = sum(len(labels) > 1 for labels in observed.values())
        stats["matched_unique_train_frames"] = len(values)
        stats["requested_unique_train_frames"] = len(wanted)
        return cls(values, {"status": "loaded" if values else "no_compatible_train_frames",
                            "path": str(path.resolve()), "sha256": digest.hexdigest(),
                            "labels_generated": False, "selection_only": True, "counts": dict(stats),
                            "warning": "Partial/ambiguous/unavailable annotations remain UNAVAILABLE; no strict action balance claim."})


def rebalance_actions(work, rows, hints, *, seed, strength=0.5, max_repeat=10):
    """Replace within identical target/subgroup signatures, never append work.

    Preserve the selected known/unknown annotation coverage in each signature.
    Known action proportions blend natural frequency and sqrt frequency. Each
    action cycles its frames; a global repeat cap limits rare support.
    """
    if not 0 <= strength <= 1 or max_repeat < 1:
        raise ValueError("action strength must be in [0,1], repeat cap must be positive")
    before = Counter(hints.label(item.row) for item in work)
    if not hints.values or strength == 0:
        return list(work), {"policy": POLICY, "source": hints.receipt, "before": dict(before), "after": dict(before), "replaced": 0}
    candidates = defaultdict(dict)
    for row in rows:
        if row.split != "train":
            raise ValueError("action balancing is train-only")
        candidates[signature(row)][identity(row)] = row
    slots = defaultdict(list)
    for i, item in enumerate(work):
        slots[signature(item.row)].append(i)
    out = list(work)
    used = Counter(identity(item.row) for item in work)
    rng = random.Random(seed)
    changed = 0
    for sig, indices in slots.items():
        # Unknown rows retain their selected membership; partial coverage never
        # becomes a reason to oversample the small annotated subset wholesale.
        known_slots = [i for i in indices if hints.label(work[i].row) != UNKNOWN]
        groups = defaultdict(list)
        for row in candidates[sig].values():
            if hints.label(row) != UNKNOWN:
                groups[hints.label(row)].append(row)
        if len(groups) < 2 or not known_slots:
            continue
        for bucket in groups.values():
            rng.shuffle(bucket)
        natural_total = sum(map(len, groups.values()))
        sqrt_total = sum(math.sqrt(len(v)) for v in groups.values())
        weights = {k: (1-strength)*len(v)/natural_total + strength*math.sqrt(len(v))/sqrt_total for k,v in groups.items()}
        chosen = Counter()
        cursors = Counter()
        for i in known_slots:
            original = work[i].row
            used[identity(original)] -= 1
            available = {}
            for label, bucket in groups.items():
                # Circular unique passes with a hard global reuse ceiling.
                for offset in range(len(bucket)):
                    cursor = (cursors[label] + offset) % len(bucket)
                    row = bucket[cursor]
                    if used[identity(row)] < max_repeat:
                        available[label] = (row, cursor)
                        break
            if not available:
                row = original
            else:
                label = max(sorted(available), key=lambda k: weights[k]*(sum(chosen.values())+1)-chosen[k])
                row, cursor = available[label]
                cursors[label] = cursor+1
            chosen[hints.label(row)] += 1
            used[identity(row)] += 1
            changed += identity(row) != identity(original)
            out[i] = replace(work[i], row=row)
    after = Counter(hints.label(item.row) for item in out)
    return out, {"policy": POLICY, "source": hints.receipt, "strength": strength,
                 "before": dict(before), "after": dict(after), "replaced": changed,
                 "total_before": len(work), "total_after": len(out), "max_frame_repeat": max(used.values(), default=0)}


def add_arguments(parser):
    parser.add_argument("--action-index", default="auto", help="existing legacy action JSONL; auto checks only the fixed v23 path; off disables")
    parser.add_argument("--action-balance-strength", type=float, default=0.5)
    parser.add_argument("--epoch-step-limit", type=int, default=4096)
    parser.add_argument("--total-step-limit", type=int, default=12288)
    parser.add_argument("--epoch-hours-limit", type=float, default=10.0)
    parser.add_argument("--budget-probe-steps", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true", help="CPU sampling/budget receipt; never load model")


def backend_report():
    module = sys.modules.get("qwen3vl_local.qwen35.vendor.modeling_qwen3_5")
    available = getattr(module, "is_fast_path_available", None)
    return {"qwen35_fast_path_available": available,
            "warning": "Qwen3.5 uses a Torch fallback for missing FLA/causal-conv1d kernels; inspect target-machine acceleration if slow. No dependencies changed automatically." if available is False else None}


def budget_plan(args, global_samples, world_size):
    if world_size < 1 or global_samples < 1:
        raise ValueError("world_size and sampled work must be positive")
    per_epoch = math.ceil(global_samples / world_size)
    total = int(args.max_steps) if args.max_steps > 0 else per_epoch * args.num_epochs
    if args.num_epochs < 1 or args.grad_accum < 1 or args.max_steps < 0:
        raise ValueError("epochs/grad_accum must be positive; max_steps must be nonnegative")
    if args.epoch_step_limit < 1 or args.total_step_limit < 1 or args.epoch_hours_limit <= 0 or args.budget_probe_steps < 1:
        raise ValueError("budget limits and probe steps must be positive")
    if not math.isfinite(args.epoch_hours_limit) or not 0 <= args.action_balance_strength <= 1:
        raise ValueError("finite positive time limit and action strength in [0,1] required")
    if min(per_epoch, total) > args.epoch_step_limit or total > args.total_step_limit:
        raise ValueError(f"Training budget exceeds limit BEFORE model loading: epoch={per_epoch}, total={total}; limits={args.epoch_step_limit}/{args.total_step_limit}. Reduce sampling quota; changing grad_accum does not reduce these forward/backward steps.")
    updates = math.ceil(total / args.grad_accum)
    # Preserve the historical warmup fraction, not an oversized absolute 2000.
    reference = getattr(args, "warmup_reference_steps", 110592)
    warmup = min(2000, max(1, round(updates * 2000 / reference))) if args.warmup_steps < 0 else args.warmup_steps
    if warmup >= updates and total > args.budget_probe_steps:
        raise ValueError(f"warmup={warmup} consumes all {updates} optimizer updates")
    return {"policy": POLICY, "world_size": world_size, "samples_per_epoch_global": global_samples,
            "steps_per_epoch": per_epoch, "total_steps": total, "optimizer_updates": updates,
            "warmup_updates": warmup, "grad_accum": args.grad_accum,
            "epoch_hours_limit": args.epoch_hours_limit, "timing_status": "unmeasured",
            "warning": "Step budget is not a wall-time guarantee; live ETA follows after the probe."}


def main():
    """Cheap launch check, before dataset scans/builds or model loading."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--phase", type=int, choices=(1,2), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--focus-balance-count", type=int, required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=-1)
    parser.add_argument("--regular-focus-multiplier", type=float, default=2.)
    parser.add_argument("--invalid-focus-multiplier", type=float, default=1.)
    add_arguments(parser)
    args = parser.parse_args()
    if args.focus_balance_count <= 0 or (args.phase == 1 and args.focus_balance_count % 6):
        raise ValueError("Positive fixed quota required; Phase1 needs a multiple of 6 for historical exact 4:1:1 variants")
    if args.regular_focus_multiplier <= 0 or args.invalid_focus_multiplier <= 0:
        raise ValueError("RE and INVALID multipliers must be positive")
    samples = (args.focus_balance_count*16 if args.phase == 1 else
               args.focus_balance_count*4 + max(1,round(args.focus_balance_count*args.regular_focus_multiplier)) + max(1,round(args.focus_balance_count*args.invalid_focus_multiplier)))
    args.warmup_reference_steps = 110592 if args.phase == 1 else 10752
    print("[launch-budget] " + json.dumps(budget_plan(args, samples, args.world_size)), flush=True)


class RuntimeBudget:
    def __init__(self, plan, probe_steps):
        self.plan, self.probe_steps = plan, probe_steps
        self.started = time.monotonic()
        self.last_report_step = 0

    def report(self, step):
        elapsed = time.monotonic()-self.started
        seconds_per_step = elapsed/max(1, step)
        epoch_hours = seconds_per_step*self.plan["steps_per_epoch"]/3600
        return {"step": step, "seconds_per_step_including_evaluation": seconds_per_step,
                "projected_epoch_hours": epoch_hours,
                "projected_remaining_hours": seconds_per_step*max(0,self.plan["total_steps"]-step)/3600,
                "limit_hours": self.plan["epoch_hours_limit"],
                "exceeded": step >= self.probe_steps and epoch_hours > self.plan["epoch_hours_limit"]}


def check_runtime(guard, step, *, rank, distributed, device, output_dir, save_adapter):
    """All ranks agree before exiting; only call at an applied optimizer step."""
    if step - guard.last_report_step < guard.probe_steps and step != guard.plan["total_steps"]:
        return
    guard.last_report_step = step
    report = guard.report(step)
    stop = bool(report["exceeded"])
    if distributed is not None and distributed.is_initialized():
        import torch
        flag = torch.tensor(int(stop), device=device)
        distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
        stop = bool(flag.item())
    report["any_rank_exceeded"] = stop
    if rank == 0:
        print("[training-eta] " + json.dumps(report), flush=True)
        Path(output_dir, "training_eta.json").write_text(json.dumps(report, indent=2)+"\n")
        if stop:
            save_adapter()
            Path(output_dir, "budget_stop.json").write_text(json.dumps({**report, "status": "stopped_over_time_budget", "adapter_only": True, "resumable_optimizer": False}, indent=2)+"\n")
    if stop:
        if distributed is not None and distributed.is_initialized():
            # Keep peers alive while rank0 finishes writing the adapter.
            distributed.barrier()
        raise RuntimeError("Projected epoch exceeds time budget; stopped early. budget_stop adapter is preserved for evaluation, NOT an optimizer resume checkpoint. Inspect speed/acceleration before restarting.")


if __name__ == "__main__":
    main()
