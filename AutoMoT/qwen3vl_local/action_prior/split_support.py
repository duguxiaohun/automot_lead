"""Repair Action's own holdout support after development-route isolation.

Assignments depend on event membership and physical route identity, never on
action labels, model errors or Phase3's split assignment. All three entries use
this same plan. The original dataset files remain immutable.
"""
from pathlib import Path
import json

from qwen3vl_local.sft_new_loop_phase3.history_rgb import history_exclusion_reason
from qwen3vl_local.sft_new_loop_phase3.split_coverage import complete_context_splits
from qwen3vl_local.action_prior.build_dataset import route_group

VERSION = "action_unexposed_route_support_v1"
SPLITS = ("train", "val", "test")


def make_split_plan(data_dir, records, development, contexts):
    assignments, origins, support = {}, {}, []
    for split in SPLITS:
        with (Path(data_dir) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("schema") != "action_prior_data_v1" or row.get("split") != split:
                    raise ValueError("wrong Action dataset schema/split in support plan")
                group = route_group(row["scenario"], row["run_id"])
                if row.get("route_group") != group:
                    raise ValueError("inconsistent Action physical route identity")
                if origins.setdefault(group, split) != split:
                    raise ValueError(f"physical route leakage before Action support planning: {group}")
                target = "train" if group in development else split
                assignments[group] = target
                if history_exclusion_reason(row["anchor"]):
                    continue
                key = (row["scenario"], row["run_id"], int(row["anchor"]))
                record = records.get(key, {})
                for context in record.get("eligible_buckets", ()):
                    support.append(dict(group=group, split=target, context_id=context))
    report = complete_context_splits(
        support, contexts=contexts, splits=SPLITS, development=development,
        group_of=lambda r: r["group"], seed=20260920,
        min_holdout_frames=32, min_holdout_groups=5,
    )
    for group, move in report["moves"].items():
        assignments[group] = move["to_split"]
    report.update(version=VERSION, development_groups=len(development),
                  scope="Action effective event support; whole physical groups; independent of Phase3 splits")
    return assignments, report


def split_plan_for_args(args):
    from qwen3vl_local.action_prior.event_balance import (
        SPECIAL_BUCKETS, development_route_groups, source_for_args,
    )
    source = source_for_args(args)
    source.validate_action_dataset(args.data_dir)
    cache = getattr(source, "_split_support_cache", {})
    source._split_support_cache = cache
    key = str(Path(args.data_dir).resolve())
    if key not in cache:
        cache[key] = make_split_plan(key, source.records, development_route_groups(), SPECIAL_BUCKETS)
    assignments, report = cache[key]
    args.event_balance_split_support = report
    return assignments
