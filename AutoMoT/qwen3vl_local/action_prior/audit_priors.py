#!/usr/bin/env python3
"""无训练的真实模型复核对照：同帧独立重问/带答案续问，保存原始问答，不冒称准确率。"""
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.config import (
    parser,
    validate_args,
    build_contract,
    read_rows,
)
from qwen3vl_local.action_prior.launch import ensure_gpu, prepare_run_directory


def main():
    """每个样本只生成先验与摘要；不更新任何参数，不加载 action checkpoint。"""
    p = parser()
    p.set_defaults(recheck_mode="compare", output_dir="checkpoints/action_prior_audit")
    p.add_argument("--max-samples", type=int, default=24)
    p.add_argument("--split", choices=["val", "test"], default="val")
    args = p.parse_args()
    validate_args(args)
    if args.max_samples <= 0 or args.condition_mode != "prior":
        raise ValueError("audit requires prior mode and positive --max-samples")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    ensure_gpu()
    import torch
    from collections import Counter
    from qwen3vl_local.leadmot import train as old, LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.runtime import make_runtime
    from qwen3vl_local.action_prior.train import (
        write_json,
        audit_counts,
        metrics_from_counts,
    )

    args.output_dir = str(prepare_run_directory(args.output_dir))
    out = Path(args.output_dir)
    contract = build_contract(args)
    rows = read_rows(args, args.split)
    rows = random.Random(args.seed + 71).sample(rows, min(len(rows), args.max_samples))
    write_json(out / "config.json", vars(args))
    write_json(out / "selected_priors.json", contract)
    runtime = make_runtime(args, torch.device("cuda", 0), contract)
    loader, _ = old._make_loader(
        rows, args, rank=0, world_size=1, shuffle=False, epoch_seed=args.seed
    )
    decoder_config = LeadMoTPlanningDecoderConfig(use_bev=True, use_final_goal=True)
    counts = Counter()

    def discard_decoder(**kwargs):
        """只消费预处理与语言路径，不产生可被误认作预测轨迹的随机 head 输出。"""
        return {}

    for i, prepared in enumerate(loader):
        if prepared.get("_error"):
            raise RuntimeError(prepared["_error"])
        with torch.no_grad():
            runtime.forward_sample(
                prepared["sample"],
                discard_decoder,
                decoder_config,
                old._dtype(args.decoder_dtype),
                clip=prepared["clip"],
            )
        audit = runtime.prior.last_audit
        counts.update(audit_counts(audit))
        write_json(
            out / "cases" / f"case_{i:06d}.json", dict(audit, sample=prepared["sample"])
        )
        print(f"[audit] {i+1}/{len(rows)}", flush=True)
    write_json(
        out / "summary.json",
        dict(
            metrics=metrics_from_counts(counts),
            ground_truth_accuracy_available=False,
            consistency_is_accuracy=False,
            interpretation="Compare independent/history retention; raw answers require audited RGB/GT to measure error filtering.",
        ),
    )
    print(json.dumps(metrics_from_counts(counts), indent=2))


if __name__ == "__main__":
    main()
