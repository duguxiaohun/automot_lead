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
    analysis_tokens=160,
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


def parser():
    """所有正式超参数均可 CLI 覆盖。"""
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
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
    p1 = select_adapter(args.checkpoint_root, 1, args.model_dir, args.phase1_adapter)
    p2 = select_adapter(args.checkpoint_root, 2, args.model_dir, args.phase2_adapter)
    base = Path(args.model_dir).resolve()
    weights = sorted(base.glob("*.safetensors")) or sorted(
        base.glob("pytorch_model*.bin")
    )
    if not weights:
        raise FileNotFoundError(f"{base}: no local Qwen weights")
    base_hashes = {
        p.name: file_hash(p)
        for p in sorted(set([*base.glob("*.json"), *base.glob("*.txt"), *weights]))
    }
    if args.use_bev:
        if not args.lead_bev_ckpt or not Path(args.lead_bev_ckpt).is_file():
            raise FileNotFoundError(
                "set LEAD_BEV_CKPT / --lead-bev-ckpt to frozen LEAD BEV weights"
            )
        bev_hash = file_hash(args.lead_bev_ckpt)
    else:
        bev_hash = None
    code_hashes = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    identity_payload = dict(
        code=code_hashes,
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
    return dict(
        git_commit=git,
        schema=SCHEMA,
        identity=digest(identity_payload),
        identity_payload=identity_payload,
        phase1=p1,
        phase2=p2,
        adapter_enabled=False,
        final_cache_model="base_without_any_adapter",
    )


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
