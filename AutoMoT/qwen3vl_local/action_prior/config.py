"""训练默认值、CLI 和纯 CPU preflight。"""

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
from qwen3vl_local.action_prior.contracts import (
    SCHEMA,
    digest,
    file_hash,
    read_json,
    select_adapter,
)
from qwen3vl_local.action_prior.priors import PROTOCOL_VERSION
from qwen3vl_local.action_prior.prompts import ANALYSIS_VERSION, SYSTEM_PROMPT

DEFAULTS = dict(
    model_dir="checkpoints/Qwen3-VL-4B-Instruct",
    data_root="lead_data",
    data_dir="checkpoints/action_prior_data",
    output_dir="checkpoints/action_prior",
    checkpoint_root="checkpoints",
    checkpoint_roots=[],
    selection_policy="available",
    selection_manifest="",
    selection_output="",
    lora_bundle="",
    phase1_adapter="",
    phase2_adapter="",
    lead_bev_ckpt="checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth",
    num_epochs=61,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    grad_accum_steps=16,
    max_grad_norm=1.0,
    val_steps=250,
    val_max_samples=256,
    save_steps=1000,
    logging_steps=10,
    seed=2026,
    num_workers=8,
    prefetch_factor=2,
    analysis_tokens=384,
    recheck_mode="history",
    condition_mode="prior",
    phase1_training_index="",
    phase2_training_index="",
    cache_priors=True,
    max_train_steps=0,
    resume="",
    qwen_dtype="bfloat16",
    decoder_dtype="bfloat16",
    route_points=10,
    waypoint_points=8,
    rgb_frame_count=4,
    rgb_frame_step=1,
    bev_frame_count=1,
    bev_frame_step=1,
    frame_interval_s=0.25,
    target_point_lookahead_s=1.0,
    next_target_point_lookahead_s=2.0,
    tp_mode="route_lookahead",
    tp_min_lookahead_m=5.0,
    use_final_goal=True,
    use_subgoal=False,
    use_bev=True,
    smooth_route=True,
    verbose_samples=False,
    qwen_adapter_dir="",
    leadmot_rope_type="mrope",
    decoder_dropout=0.1,
    qwen_load_stagger_s=0.0,
    persistent_workers=True,
    worker_multiprocessing_context="spawn",
    route_loss_weight=0.5,
    waypoint_loss_weight=1.0,
    loss_type="l1",
    ema_decay=0.999,
)


INPUT_FIELDS = (
    "rgb_frame_count",
    "rgb_frame_step",
    "bev_frame_count",
    "bev_frame_step",
    "frame_interval_s",
    "target_point_lookahead_s",
    "next_target_point_lookahead_s",
    "tp_mode",
    "tp_min_lookahead_m",
    "use_final_goal",
)


def parser():
    """所有正式超参数均可 CLI 覆盖。"""
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, list):
            p.add_argument("--" + k.replace("_", "-"), nargs="+", default=[])
            continue
        p.add_argument(
            "--" + k.replace("_", "-"),
            default=v,
            **(
                {"action": argparse.BooleanOptionalAction}
                if isinstance(v, bool)
                else {"type": type(v)}
            ),
        )
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--models-only", action="store_true")
    return p


def read_rows(args, split):
    """索引必须来自新 builder，并在实际使用前再检查异常 route。"""
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route

    rows, seen, blocked = [], set(), {}
    root = Path(args.data_root).resolve()
    with (Path(args.data_dir) / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("schema") != "action_prior_data_v1" or row.get("split") != split:
                raise ValueError(
                    "wrong dataset schema/split; rebuild action_prior index"
                )
            if (
                row.get("tp_mode") != "route_lookahead"
                or row.get("rgb_frame_count") != 4
                or row.get("rgb_frame_step") != 1
            ):
                raise ValueError(
                    "future-truth navigation or incompatible RGB input is forbidden"
                )
            route = root / row["scenario"] / row["run_id"]
            key = str(route)
            if key not in blocked:
                if not route.is_dir():
                    raise FileNotFoundError(route)
                blocked[key] = is_abnormal_lead_route(route, row["scenario"])[0]
            if blocked[key]:
                continue
            # 索引固定字段仅是构建记录；运行时导航由 CLI 统一决定，避免配置与输入不符。
            for name in INPUT_FIELDS:
                row[name] = getattr(args, name)
            row["route_dir"] = key
            ident = (key, int(row["anchor"]))
            if ident in seen:
                raise ValueError(f"duplicate dataset frame: {ident}")
            seen.add(ident)
            rows.append(row)
    if not rows:
        raise ValueError(f"no valid {split} rows")
    return rows


def validate_args(args):
    """防止兼容参数改变该路线的核心条件。"""
    if getattr(args, "selection_policy", "strict") not in ("available", "strict"):
        raise ValueError("selection_policy must be available or strict")
    if (
        args.use_subgoal
        or args.qwen_adapter_dir
        or not args.use_final_goal
        or not args.use_bev
        or args.tp_mode != "route_lookahead"
        or args.rgb_frame_count != 4
        or args.rgb_frame_step != 1
    ):
        raise ValueError(
            "requires base-only final KV, no subgoal, final_goal, 4 consecutive RGB, route lookahead"
        )
    if args.bev_frame_count != 1 or args.bev_frame_step != 1:
        raise ValueError(
            "action_prior currently supports single-frame BEV only; multi-frame BEV is not implemented"
        )
    if (
        args.frame_interval_s != 0.25
        or args.route_points != 10
        or args.waypoint_points != 8
    ):
        raise ValueError(
            "fixed LEAD 4Hz / route10 / waypoint8 contract; rebuild the pipeline before changing these"
        )
    if args.recheck_mode not in ("history", "independent", "compare"):
        raise ValueError("invalid recheck mode")
    if args.condition_mode not in ("prior", "base"):
        raise ValueError("condition mode must be prior/base")
    for name in (
        "target_point_lookahead_s",
        "next_target_point_lookahead_s",
        "tp_min_lookahead_m",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "num_epochs",
        "grad_accum_steps",
        "val_steps",
        "save_steps",
        "logging_steps",
        "analysis_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.decoder_dtype not in ("bfloat16", "float32") or args.qwen_dtype not in (
        "bfloat16",
        "float32",
    ):
        raise ValueError(
            "supported dtypes are bfloat16/float32; fp16 would require loss scaling"
        )
    if args.max_train_steps < 0 or args.val_max_samples < 0 or args.num_workers < 0:
        raise ValueError("step/sample/worker limits must be nonnegative")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("invalid LR/warmup")


def build_contract(args):
    """冻结上游身份和运行协议；权重路径可迁移，权重字节不能变。"""
    policy = getattr(args, "selection_policy", "strict")
    manifest_path = getattr(args, "selection_manifest", "")
    pinned = read_json(manifest_path) if manifest_path else None
    if pinned and (pinned.get("schema") != "action_prior_selection_v1" or pinned["selection_policy"] != policy):
        raise ValueError("selection manifest policy/schema mismatch")
    roots = getattr(args, "checkpoint_roots", []) or [args.checkpoint_root]
    bundle = getattr(args, "lora_bundle", "")
    bundle_info, packaged_paths = None, {}
    if bundle:
        from qwen3vl_local.action_prior.lora_bundle import verify_bundle, bundle_paths
        bundle_info = verify_bundle(bundle)
        packaged_paths = bundle_paths(bundle)
        if set(packaged_paths) != {"phase1", "phase2"}:
            raise ValueError("training requires a bundle containing both Phase1 and Phase2")
    selected = []
    for phase in (1, 2):
        explicit = getattr(args, f"phase{phase}_adapter")
        if bundle:
            if explicit and Path(explicit).resolve() != Path(packaged_paths[f"phase{phase}"]):
                raise ValueError("explicit adapter conflicts with --lora-bundle")
            explicit = packaged_paths[f"phase{phase}"]
        if pinned:
            # 显式重映射允许搬迁目录，但必须核验实际权重指纹；不得重新择优。
            explicit = explicit or pinned[f"phase{phase}"]["path"]
        if policy == "available":
            from qwen3vl_local.action_prior.available_adapters import select_available
            item = select_available(roots, phase, args.model_dir, explicit)
        else:
            if len(roots) != 1:
                raise ValueError("strict policy uses one checkpoint root; put shared roots under one directory")
            item = select_adapter(roots[0], phase, args.model_dir, explicit)
        if pinned and item["fingerprint"] != pinned[f"phase{phase}"]["fingerprint"]:
            raise ValueError(f"Phase{phase}: selected weight files changed after preflight")
        if bundle_info and item["fingerprint"] != bundle_info["phases"][f"phase{phase}"]["fingerprint"]:
            raise ValueError("bundle identity differs from loaded adapter")
        selected.append(item)
    p1, p2 = selected
    base = Path(args.model_dir).resolve()
    weights = sorted(base.glob("*.safetensors")) or sorted(
        base.glob("pytorch_model*.bin")
    )
    if not weights:
        raise FileNotFoundError(f"{base}: no local Qwen weights")
    base_hashes = {
        p.name: file_hash(p)
        for p in sorted(
            set(
                [
                    *base.glob("*.json"),
                    *base.glob("*.txt"),
                    *base.glob("*.jinja"),
                    *base.glob("*.model"),
                    *weights,
                ]
            )
        )
    }
    if args.use_bev:
        if not args.lead_bev_ckpt or not Path(args.lead_bev_ckpt).is_file():
            raise FileNotFoundError(
                "set LEAD_BEV_CKPT / --lead-bev-ckpt to frozen LEAD BEV weights"
            )
        bev_hash = file_hash(args.lead_bev_ckpt)
    else:
        bev_hash = None
    from qwen3vl_local.action_prior.provenance import (
        execution_fingerprint,
        collect_upstream_sources,
    )
    from qwen3vl_local.action_prior.precision import PRECISION_POLICY

    execution = execution_fingerprint()
    upstream = collect_upstream_sources({"phase1": p1, "phase2": p2}, args)
    identity_payload = dict(
        execution=execution,
        precision=PRECISION_POLICY,
        qwen_dtype=args.qwen_dtype,
        decoder_compute_dtype=args.decoder_dtype,
        recheck_mode=args.recheck_mode,
        condition_mode=args.condition_mode,
        base=base_hashes,
        phase1=p1["fingerprint"],
        phase2=p2["fingerprint"],
        bev=bev_hash,
        protocol=PROTOCOL_VERSION,
        analysis=ANALYSIS_VERSION,
        system=SYSTEM_PROMPT,
        analysis_tokens=args.analysis_tokens,
        navigation={
            k: getattr(args, k)
            for k in (
                "frame_interval_s",
                "target_point_lookahead_s",
                "next_target_point_lookahead_s",
                "tp_min_lookahead_m",
                "bev_frame_count",
                "bev_frame_step",
            )
        },
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    contract = dict(
        git_commit=git,
        upstream_sources=upstream,
        audit_identity=digest(upstream),
        schema=SCHEMA,
        identity=digest(identity_payload),
        identity_payload=identity_payload,
        phase1=p1,
        phase2=p2,
        adapter_enabled=False,
        final_cache_model="base_without_any_adapter",
        selection_policy=policy,
    )
    if pinned and pinned.get("contract_identity") != contract["identity"]:
        raise ValueError("selection manifest execution/base/BEV identity changed after preflight")
    return contract


def training_plan(args, rows, world):
    """全量样本尾部按 rank 不重复分片；报告步数，避免把 micro-step 当 optimizer step。"""
    usable = len(rows["train"]) // world * world
    if usable < world:
        raise ValueError("too few train rows for world size")
    groups = {s: {r["route_group"] for r in rr} for s, rr in rows.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"physical route leakage: {a}/{b}")
    updates = math.ceil((usable // world) / args.grad_accum_steps)
    return dict(
        samples={s: len(v) for s, v in rows.items()},
        effective_batch=world * args.grad_accum_steps,
        world_size=world,
        micro_batch_per_gpu=1,
        condition_mode=args.condition_mode,
        cold_generations_per_unique_frame=(
            0
            if args.condition_mode == "base"
            else 17 if args.recheck_mode == "compare" else 11
        ),
        final_base_prefills_per_presentation=1,
        shared_text_cache=args.cache_priors,
        budget_note="Epoch/LR are initial settings; measure cold and cached throughput with smoke before full training.",
        epochs=args.num_epochs,
        samples_per_epoch=usable,
        ddp_tail_per_epoch=len(rows["train"]) - usable,
        optimizer_steps_per_epoch=updates,
        planned_optimizer_steps=updates * args.num_epochs,
        actual_step_limit=(
            min(args.max_train_steps, updates * args.num_epochs)
            if args.max_train_steps
            else updates * args.num_epochs
        ),
        total_planned_presentations=usable * args.num_epochs,
        partial_final_accumulation=(usable // world) % args.grad_accum_steps,
        learning_rate=args.learning_rate,
        validation_every_optimizer_steps=args.val_steps,
        periodic_validation_samples=args.val_max_samples,
        epoch_validation="all validation frames",
    )
