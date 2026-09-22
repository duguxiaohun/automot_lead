#!/usr/bin/env python3
"""只读审计三条 Action 入口共享的数据容量；不加载模型或重建数据。

使用正式训练参数，另传 --report 和 --world-sizes；均衡采样需已有 full map。
"""
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.config import parser, read_rows, training_plan
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.action_prior.event_balance import (
    available_counts, build_balanced_epoch, SPECIAL_BUCKETS, REGULAR_BACKGROUND,
)
from qwen3vl_local.action_prior.action_token import token_support, require_conditioned_training


def audit(args):
    if args.num_epochs < 1 or not args.world_sizes or min(args.world_sizes) < 1:
        raise ValueError("positive num_epochs and world-sizes required")
    if not args.event_balance_index:
        raise ValueError("capacity audit requires an existing --event-balance-index for balanced sampling")
    rows = {split: read_rows(args, split) for split in ("train", "val", "test")}
    result = dict(
        scope="shared_action_data_and_planned_full_epochs_only",
        models_loaded=False, production_training_verified=False,
        dataset_hashes={s: file_hash(Path(args.data_dir) / f"{s}.jsonl") for s in rows},
        samples={s: len(rr) for s, rr in rows.items()},
        original_split_counts={s: dict(Counter(r["event_balance_original_split"] for r in rr))
                               for s, rr in rows.items()},
        worlds={},
    )
    if args.event_balance_index:
        result["event_support"] = {}
        for split, rr in rows.items():
            counts = available_counts(rr, for_evaluation=split != "train")
            result["event_support"][split] = dict(
                counts=counts, missing=[k for k in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND) if not counts.get(k)],
            )
    for world in args.world_sizes:
        plan = training_plan(args, rows, world)
        epochs = []
        for epoch in range(args.num_epochs):
            selected, report = build_balanced_epoch(
                rows["train"], mode=args.sampling_mode, total=plan["samples_per_epoch"], seed=args.seed + epoch,
                route_diverse=args.event_balance_route_diverse,
                repeat_cap=args.event_balance_max_frame_repeats,
            )
            if args.high_level_action_token:
                support = token_support({"train": selected})
                require_conditioned_training(support, stage=f"capacity audit world={world} epoch={epoch + 1}")
                report["action_token_support"] = support
            report["scope"] = "planned_full_epoch_before_max_train_steps"
            epochs.append(report)
        result["worlds"][str(world)] = dict(plan=plan, epochs=epochs)
    return result


def main():
    p = parser()
    p.description = __doc__
    p.add_argument("--world-sizes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    try:
        result = dict(ok=True, **audit(args))
    except (ValueError, FileNotFoundError, KeyError, AssertionError) as exc:
        result = dict(ok=False, error=str(exc), models_loaded=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "worlds"}, ensure_ascii=False))
    print(f"capacity report: {args.report}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
