#!/usr/bin/env python3
"""恢复 checkpoint 的完整先验合同，评测 loss/ADE/FDE 和 invalid。"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.progress import current, observed, report


@observed
def main():
    """测试集只用于最终报告，不参与最优 checkpoint 选择。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--model-dir", default="")
    p.add_argument("--phase1-adapter", default="")
    p.add_argument("--phase2-adapter", default="")
    p.add_argument("--lead-bev-ckpt", default="")
    p.add_argument("--phase1-training-index", default="")
    p.add_argument("--phase2-training-index", default="")
    p.add_argument("--output-dir", default="")
    # 默认跟随 checkpoint 自己的先验来源；只有显式关闭才回到 LoRA 推理。
    p.add_argument("--dataset-priors", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prior-labels", default="")
    p.add_argument(
        "--event-balance-index", default="",
        help="relocated current full_event_mapping.jsonl; path is audit-only when bytes match checkpoint contract",
    )
    # 默认仍用训练时的注入噪声；--prior-noise 0 才是干净先验对照。
    p.add_argument("--prior-noise", type=float, default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--dump-cases", action="store_true")
    p.add_argument("--raw", action="store_true", help="默认使用 EMA")
    cli = p.parse_args()
    progress_out = Path(cli.output_dir or str(Path(cli.checkpoint).parent / f"eval_{cli.split}"))
    current().configure(progress_out, "eval")
    report("evaluation/checkpoint_and_contract", announce=True, split=cli.split)
    from qwen3vl_local.action_prior.launch import ensure_gpu

    ensure_gpu()
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"
    import torch
    import torch.distributed as dist
    from qwen3vl_local.action_prior.config import (
        build_contract,
        read_rows,
        validate_args,
    )
    from qwen3vl_local.action_prior.contracts import file_hash, require_contract
    from qwen3vl_local.action_prior.runtime import make_runtime
    from qwen3vl_local.action_prior.train import evaluate, write_json
    from qwen3vl_local.leadmot import train as old
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.flow_matching import (
        ConditionalFlowMatchingDecoder,
        FlowMatchingConfig,
    )

    rank, local, world = old._init_distributed()
    state = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != "action_prior_checkpoint_v4" or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2":
        raise ValueError(
            "requires action_prior v4 joint-trajectory Flow-Matching checkpoint; old independent/regression checkpoints are incompatible"
        )
    args = argparse.Namespace(**state["args"])
    # 权重已由 checkpoint 固定；不依赖训练前选择清单的旧物理路径。
    args.selection_manifest = ""
    args.selection_output = ""
    args.lora_bundle = ""
    # 映射文件路径可迁移；build_contract 比较的是 manifest/index 内容身份而不是绝对路径。
    # 该 override 也让离线 val/test 获得和训练相同的固定 scene contexts。
    if cli.event_balance_index:
        args.event_balance_index = cli.event_balance_index
    trained_with_dataset_priors = bool(state["args"].get("dataset_priors", False))
    args.dataset_priors = (
        trained_with_dataset_priors if cli.dataset_priors is None else cli.dataset_priors
    )
    if cli.prior_labels:
        args.prior_labels = cli.prior_labels
    trained_prior_noise = float(state["args"].get("prior_noise", 0.0))
    args.prior_noise = trained_prior_noise if cli.prior_noise is None else cli.prior_noise
    prior_source_override = (
        args.dataset_priors != trained_with_dataset_priors
        or args.prior_noise != trained_prior_noise
    )
    if args.dataset_priors:
        args.phase1_adapter = args.phase2_adapter = ""
    else:
        args.prior_labels = ""
        args.prior_noise = 0.0
        from qwen3vl_local.action_prior.lora_bundle import restore_paths

        local_paths = (
            restore_paths(state["qwen_backbone"], cli.checkpoint)
            if state["qwen_backbone"].get("phase1")
            else {"phase1": "", "phase2": ""}
        )
        args.phase1_adapter = cli.phase1_adapter or local_paths["phase1"]
        args.phase2_adapter = cli.phase2_adapter or local_paths["phase2"]
    for k in (
        "data_root",
        "data_dir",
        "model_dir",
        "lead_bev_ckpt",
        "phase1_training_index",
        "phase2_training_index",
    ):
        if getattr(cli, k):
            setattr(args, k, getattr(cli, k))
    validate_args(args)
    from qwen3vl_local.action_prior.event_balance import source_audit

    box = [None]
    if rank == 0:
        try:
            box[0] = {"contract": build_contract(args)}
        except Exception as exc:
            box[0] = {"error": str(exc)}
    if world > 1:
        dist.broadcast_object_list(box, src=0)
    if "error" in box[0]:
        raise ValueError(box[0]["error"])
    contract = box[0]["contract"]
    contract_match = require_contract(
        state["qwen_backbone"], contract, allow_prior_source_change=prior_source_override
    )
    if prior_source_override:
        print(
            "[prior source override] trained_with_dataset_priors="
            f"{trained_with_dataset_priors} noise={trained_prior_noise} -> "
            f"dataset_priors={args.dataset_priors} noise={args.prior_noise}; "
            "this is a deliberate condition shift, not a same-condition reproduction",
            flush=True,
        )
    from qwen3vl_local.action_prior.provenance import audit_source_changes

    audit_changes = audit_source_changes(
        state["qwen_backbone"].get("upstream_sources", {}), contract["upstream_sources"]
    )
    if (
        file_hash(Path(args.data_dir) / f"{cli.split}.jsonl")
        != state["dataset_hashes"][cli.split]
    ):
        raise ValueError("evaluation dataset differs from checkpoint split")
    config = LeadMoTPlanningDecoderConfig(**state["decoder_config"])
    flow_config = FlowMatchingConfig(**state["flow_config"])
    device = torch.device("cuda", local)
    dtype = old._dtype(args.decoder_dtype)
    model = ConditionalFlowMatchingDecoder(config, flow_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["decoder"], strict=True)
    if not cli.raw:
        model.load_state_dict(state["ema_state_dict"]["shadow"], strict=True)
    dataset_hashes = state["dataset_hashes"]
    checkpoint_step = state["step"]
    del state
    args.output_dir = str(Path(cli.checkpoint).resolve().parent)
    report("evaluation/load_frozen_models", announce=True)
    runtime = make_runtime(args, device, contract)
    rows = read_rows(args, cli.split)
    from qwen3vl_local.action_prior.provenance import annotate_upstream

    exposure = annotate_upstream(rows, contract["upstream_sources"])
    out = Path(cli.output_dir or str(Path(cli.checkpoint).parent / f"eval_{cli.split}"))
    metrics = evaluate(
        runtime,
        model,
        config,
        rows,
        args,
        dtype,
        rank,
        world,
        cli.max_samples,
        out / "cases" if cli.dump_cases else None,
    )
    if rank == 0:
        write_json(
            out / "metrics.json",
            dict(
                metrics=metrics,
                checkpoint=str(Path(cli.checkpoint).resolve()),
                split=cli.split,
                ema=not cli.raw,
                contract_identity=contract["identity"],
                contract_match=contract_match,
                prior_source=contract.get("prior_source", "phase_loras"),
                trained_with_dataset_priors=trained_with_dataset_priors,
                trained_prior_noise=trained_prior_noise,
                evaluated_prior_noise=args.prior_noise,
                prior_source_override=prior_source_override,
                sample_limit=cli.max_samples,
                event_balance_source_audit=source_audit(args),
                upstream_training_pool_audit=exposure,
                upstream_sources=contract["upstream_sources"],
                upstream_source_changes=audit_changes,
                audit_identity=contract["audit_identity"],
                actual_upstream_seen_routes_verified=False,
                dataset_hashes=dataset_hashes,
                checkpoint_step=checkpoint_step,
            ),
        )
        print(json.dumps(metrics, indent=2))
        from qwen3vl_local.action_prior.audit_bundle import pack
        pack(out)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
