#!/usr/bin/env python3
"""对同帧 base/prior probe 配对比较；事件组由 prior 条件固定，不声称 GT 事件准确率。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.metrics import grouped_counts
from qwen3vl_local.action_prior.train import metrics_from_counts, write_json


def read_cases(path):
    """拒绝重复帧或未包含指标的旧版 probe。"""
    cases = {}
    for file in sorted((Path(path) / "cases").glob("*.json")):
        case = json.loads(file.read_text())
        row = case["sample"]
        key = (row["scenario"], row["run_id"], row["anchor"])
        if key in cases or "metrics" not in case:
            raise ValueError(f"duplicate or old probe case: {file}")
        cases[key] = case
    if not cases:
        raise ValueError(f"no cases in {path}")
    return cases


def compare(base, prior):
    """组归属完全取同一个 prior case，防止两实验各自分类造成分母偏差。"""
    if base.keys() != prior.keys():
        raise ValueError("paired sample identities differ")
    totals = {key: Counter() for key in ("base", "prior")}
    for key, label in prior.items():
        for side, cases in (("base", base), ("prior", prior)):
            if (
                cases[key]["gt_route"] != label["gt_route"]
                or cases[key]["gt_waypoints"] != label["gt_waypoints"]
            ):
                raise ValueError("paired trajectory supervision differs")
            values = cases[key]["metrics"]
            totals[side].update(dict(samples=1, **values))
            totals[side].update(grouped_counts(label, label["sample"], values))
    result = {side: metrics_from_counts(counts) for side, counts in totals.items()}
    result["prior_minus_base"] = {
        k: result["prior"][k] - result["base"][k]
        for k in result["base"]
        if k != "samples" and not k.endswith("/samples")
    }
    return result


def main():
    """核对同 run 超参数与语言条件之外的合同，然后写可配对的小样本比较。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--output", required=True)
    cli = p.parse_args()
    configs = []
    contracts = []
    metadata, plans = [], []
    for folder in (cli.base, cli.prior):
        meta = json.loads((Path(folder) / "metrics.json").read_text())
        metadata.append(meta)
        run = Path(meta["checkpoint"]).parent
        plans.append(json.loads((run / "training_plan.json").read_text()))
        configs.append(json.loads((run / "config.json").read_text()))
        contracts.append(
            json.loads((run / "selected_priors.json").read_text())["identity_payload"]
        )
    if [c["condition_mode"] for c in configs] != ["base", "prior"]:
        raise ValueError("expected separately trained base and prior runs")
    for key in (
        "seed",
        "num_epochs",
        "max_train_steps",
        "grad_accum_steps",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "decoder_dropout",
        "decoder_dtype",
        "loss_type",
        "route_loss_weight",
        "waypoint_loss_weight",
        "ema_decay",
    ):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"ablation training config differs: {key}")
    if (
        metadata[0]["dataset_hashes"] != metadata[1]["dataset_hashes"]
        or metadata[0]["ema"] != metadata[1]["ema"]
    ):
        raise ValueError("ablation dataset/EMA selection differs")
    for key in ("world_size", "effective_batch", "actual_step_limit", "samples"):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"ablation budget differs: {key}")
    for contract in contracts:
        contract.pop("condition_mode", None)
    if contracts[0] != contracts[1]:
        raise ValueError("ablation input/model/execution contracts differ")
    result = compare(read_cases(cli.base), read_cases(cli.prior))
    write_json(
        cli.output,
        dict(
            metrics=result,
            group_source="paired prior predicted conditions; not GT events",
            interpretation="Negative prior_minus_base loss/ADE/FDE indicates lower error on these paired frames.",
        ),
    )


if __name__ == "__main__":
    main()
