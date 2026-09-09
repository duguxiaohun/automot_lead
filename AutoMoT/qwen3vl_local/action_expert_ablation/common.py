#!/usr/bin/env python3
"""Shared training/evaluation implementation for action expert ablations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from qwen3vl_local.action_prior import config as prior_config
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.flow_matching import (
    ConditionalFlowMatchingDecoder,
    FlowMatchingConfig,
)
from qwen3vl_local.action_prior.precision import PRECISION_POLICY, decoder_forward
from qwen3vl_local.action_prior.training_core import (
    MetricHooks, budget_complete, trim_tensorboard_for_resume,
    with_validation_pending, clear_validation_pending, merge_counts,
    evaluate as evaluate_shared, run_training_loop,
    make_model_and_config, make_optimization, save_training_checkpoint, restore_training_state,
    accumulation_state,
    flow_config_of,
    sampled_trajectory_score,
    write_json,
)
from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig


CHECKPOINT_SCHEMA = "action_expert_ablation_checkpoint_v1"
TRAJECTORY_DECODER = "conditional_joint_trajectory_flow_matching_v2"

VARIANTS = {
    "qwen_simple": {
        "description": "four RGB images + simple LeadMoT/Qwen navigation prompt + frozen BEV",
        "uses_qwen": True,
        "uses_qwen_images": True,
        "output_dir": "checkpoints/action_expert_ablation/qwen_simple",
    },
    "bev_only": {
        "description": (
            "frozen LEAD BEV RGB+LiDAR/navigation/status/query tokens, "
            "with zero-length Qwen prefix KV"
        ),
        "uses_qwen": False,
        "uses_qwen_images": False,
        "output_dir": "checkpoints/action_expert_ablation/bev_only",
    },
}


_OLD_TRAIN = None


def leadmot_train():
    """Lazy import the heavy LeadMoT training helpers."""

    global _OLD_TRAIN
    if _OLD_TRAIN is None:
        from qwen3vl_local.leadmot import train as old_train

        _OLD_TRAIN = old_train
    return _OLD_TRAIN


def parser(variant: str) -> argparse.ArgumentParser:
    """Build a parser that keeps action_prior schedule/flow defaults."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")
    defaults = dict(prior_config.DEFAULTS)
    defaults.update(
        output_dir=VARIANTS[variant]["output_dir"],
        data_dir="checkpoints/action_prior_data",
        condition_mode=variant,
        analysis_review=False,
        cache_priors=False,
        selection_policy="available",
    )
    ignored = {
        "selection_manifest",
        "selection_output",
        "checkpoint_root",
        "selection_policy",
        "lora_bundle",
        "phase1_adapter",
        "phase2_adapter",
        "checkpoint_roots",
        "analysis_review",
        "analysis_tokens",
        "recheck_mode",
        "condition_mode",
        "dataset_priors",
        "prior_labels",
        "prior_noise",
        "prior_noise_invalid_share",
        "phase1_training_index",
        "phase2_training_index",
        "qwen_adapter_dir",
        "use_subgoal",
        "cache_priors",
    }
    if variant == "bev_only":
        ignored.update({"model_dir", "qwen_dtype", "qwen_load_stagger_s"})
    p = argparse.ArgumentParser(description=f"Train/eval {variant} action expert ablation")
    for key, value in defaults.items():
        if key in ignored:
            continue
        kwargs = (
            {"action": argparse.BooleanOptionalAction}
            if isinstance(value, bool)
            else {"type": type(value)}
        )
        p.add_argument("--" + key.replace("_", "-"), default=value, **kwargs)
    p.set_defaults(**{key: defaults[key] for key in ignored if key in defaults})
    p.add_argument("--preflight", action="store_true")
    return p


def _explicit_cli_dests(p: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    """Return argparse destinations that were explicitly set by this invocation."""

    items = list(sys.argv[1:] if argv is None else argv)
    dests: set[str] = set()
    i = 0
    while i < len(items):
        item = items[i]
        if item == "--":
            break
        if not item.startswith("--"):
            i += 1
            continue
        option = item.split("=", 1)[0]
        action = p._option_string_actions.get(option)
        if action is None:
            i += 1
            continue
        dests.add(action.dest)
        if "=" not in item and getattr(action, "nargs", None) in (None, 1):
            i += 2
        else:
            i += 1
    return dests


def parse_train_args(variant: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args, restoring the saved run configuration before resume validation."""

    p = parser(variant)
    cli_args = p.parse_args(argv)
    if not cli_args.resume:
        return cli_args

    resume_path = Path(cli_args.resume).expanduser().resolve()
    config_path = resume_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path}: cannot restore resume defaults; pass the full original train args"
        )
    with config_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    merged = vars(p.parse_args([]))
    merged.update(saved)
    explicit = _explicit_cli_dests(p, argv)
    for dest in explicit:
        merged[dest] = getattr(cli_args, dest)
    merged["resume"] = str(resume_path)
    if "output_dir" not in explicit:
        merged["output_dir"] = str(resume_path.parent)
    return argparse.Namespace(**merged)


def read_rows(args: argparse.Namespace, split: str) -> list[dict]:
    """Reuse the action_prior index so every optimizer step sees the same sample budget."""

    return prior_config.read_rows(args, split)


def validate_args(args: argparse.Namespace, variant: str) -> None:
    """Keep only the knobs that preserve a fair FM ablation contract."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")
    if (
        not args.use_final_goal
        or not args.use_bev
        or args.tp_mode != "route_lookahead"
        or args.rgb_frame_count != 4
        or args.rgb_frame_step != 1
    ):
        raise ValueError(
            "ablations require final_goal, frozen BEV, 4 consecutive RGB rows, and route lookahead"
        )
    if args.bev_frame_count != 1 or args.bev_frame_step != 1:
        raise ValueError("ablations currently follow action_prior single-frame BEV")
    if args.frame_interval_s != 0.25 or args.route_points != 10 or args.waypoint_points != 8:
        raise ValueError("fixed LEAD 4Hz / route10 / waypoint8 contract")
    for name in (
        "target_point_lookahead_s",
        "next_target_point_lookahead_s",
        "tp_min_lookahead_m",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("num_epochs", "grad_accum_steps", "val_steps", "save_steps", "logging_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.decoder_dtype not in ("bfloat16", "float32") or args.qwen_dtype not in (
        "bfloat16",
        "float32",
    ):
        raise ValueError("supported dtypes are bfloat16/float32")
    if args.max_train_steps < 0 or args.val_max_samples < 0 or args.num_workers < 0:
        raise ValueError("step/sample/worker limits must be nonnegative")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("invalid LR/warmup")
    if args.loss_type != "mse":
        raise ValueError("Flow Matching ablations require --loss-type mse")
    FlowMatchingConfig(
        route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
        waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,
        time_embed_dim=args.flow_time_embed_dim,
        sample_steps=args.flow_sample_steps,
        trajectory_layers=args.flow_trajectory_layers,
        trajectory_heads=args.flow_trajectory_heads,
    ).validate()


def training_plan(args: argparse.Namespace, rows: dict[str, list[dict]], world: int, variant: str) -> dict:
    """Mirror action_prior step accounting without prior-specific metrics."""

    plan = prior_config.training_plan(args, rows, world)
    plan.update(
        ablation_variant=variant,
        ablation_description=VARIANTS[variant]["description"],
        prior_source="none",
        condition_mode=variant,
        independent_analysis_review=False,
        cold_generations_per_unique_frame=0,
        final_base_prefills_per_presentation=1 if VARIANTS[variant]["uses_qwen"] else 0,
        shared_text_cache=False,
        tensorboard_note=(
            "Only core FM/planning scalars are logged; no RS/EVENT/prior grouped losses."
        ),
    )
    return plan


def _hash_existing(root: Path, relative_paths: list[str]) -> dict[str, str]:
    """Hash only files that are part of the real ablation execution path."""

    result = {}
    for rel in relative_paths:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[rel] = file_hash(path)
    return result


def _runtime_package_versions() -> dict[str, str | None]:
    """Record key runtime library versions that can change numerical behavior."""

    result = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "numpy",
        "Pillow",
        "timm",
        "laspy",
        "opencv-python",
        "safetensors",
        "tensorboard",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _execution_fingerprint(root: Path, seed_paths: list[str]) -> dict:
    """Hash the explicit ablation execution contract without unused Phase prompts."""

    return dict(
        code=_hash_existing(root, seed_paths),
        packages=_runtime_package_versions(),
    )


def contract_source_paths(variant: str) -> list[str]:
    """List source files that define the ablation execution contract."""

    sources = [
        "qwen3vl_local/action_expert_ablation/__init__.py",
        "qwen3vl_local/action_expert_ablation/common.py",
        "qwen3vl_local/action_expert_ablation/launch.py",
        f"qwen3vl_local/action_expert_ablation/{variant}/__init__.py",
        f"qwen3vl_local/action_expert_ablation/{variant}/train.py",
        f"qwen3vl_local/action_expert_ablation/{variant}/eval.py",
        f"qwen3vl_local/action_expert_ablation/{variant}/train.sh",
        f"qwen3vl_local/action_expert_ablation/{variant}/eval.sh",
        f"qwen3vl_local/action_expert_ablation/{variant}/run_full_pipeline.sh",
        "qwen3vl_local/action_expert_ablation/pipeline_common.sh",
        "qwen3vl_local/action_prior/flow_matching.py",
        "qwen3vl_local/action_prior/precision.py",
        "qwen3vl_local/action_prior/config.py",
        "qwen3vl_local/action_prior/build_dataset.py",
        "qwen3vl_local/action_prior/training_core.py",
        "qwen3vl_local/action_prior/progress.py",
        "qwen3vl_local/action_prior/contracts.py",
        "qwen3vl_local/leadmot/config.py",
        "qwen3vl_local/leadmot/train.py",
        "qwen3vl_local/leadmot/decoder.py",
        "qwen3vl_local/leadmot/heads.py",
        "qwen3vl_local/leadmot/mot_block.py",
        "qwen3vl_local/leadmot/projectors.py",
        "qwen3vl_local/leadmot/query_bank.py",
        "leaderboard/team_code/mot_lead_offline_runner.py",
        "Automot/mot/modeling/bev_encoder/bev_encoder_utils.py",
        "lead_video_tools/abnormal_duration_filter.py",
    ]
    if VARIANTS[variant]["uses_qwen"]:
        sources.extend(
            [
                "qwen3vl_local/engine.py",
                "qwen3vl_local/mrope_utils.py",
            ]
        )
    return sources


def build_contract(args: argparse.Namespace, variant: str) -> dict:
    """Freeze the ablation condition identity without selecting any RS/EVENT adapters."""

    root = Path(__file__).resolve().parents[2]
    sources = contract_source_paths(variant)

    base_hashes = None
    if VARIANTS[variant]["uses_qwen"]:
        base = Path(args.model_dir).resolve()
        weights = sorted(base.glob("*.safetensors")) or sorted(base.glob("pytorch_model*.bin"))
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
    if not args.lead_bev_ckpt or not Path(args.lead_bev_ckpt).is_file():
        raise FileNotFoundError("set --lead-bev-ckpt to frozen LEAD BEV weights")
    bev_hash = file_hash(args.lead_bev_ckpt)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        git_commit = ""
    identity_payload = dict(
        variant=variant,
        condition=VARIANTS[variant],
        execution=_execution_fingerprint(root, sources),
        precision=PRECISION_POLICY,
        qwen_dtype=args.qwen_dtype if VARIANTS[variant]["uses_qwen"] else None,
        decoder_compute_dtype=args.decoder_dtype,
        base=base_hashes,
        bev=bev_hash,
        trajectory_decoder=TRAJECTORY_DECODER,
        flow_matching=dict(
            route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
            waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,
            time_embed_dim=args.flow_time_embed_dim,
            sample_steps=args.flow_sample_steps,
            trajectory_layers=args.flow_trajectory_layers,
            trajectory_heads=args.flow_trajectory_heads,
            train_sampled_metrics=args.train_sampled_metrics,
        ),
        navigation={
            key: getattr(args, key)
            for key in (
                "frame_interval_s",
                "target_point_lookahead_s",
                "next_target_point_lookahead_s",
                "tp_min_lookahead_m",
                "bev_frame_count",
                "bev_frame_step",
            )
        },
    )
    return dict(
        schema="action_expert_ablation_condition_v1",
        git_commit=git_commit,
        identity=digest(identity_payload),
        identity_payload=identity_payload,
        variant=variant,
        prior_source="none",
        adapter_enabled=False,
        final_cache_model=(
            "base_qwen_simple_prompt" if VARIANTS[variant]["uses_qwen"] else "none_zero_length_prefix_kv"
        ),
    )


def require_contract(expected: dict, actual: dict) -> None:
    """Require exact same condition identity for resume/eval."""

    if expected.get("schema") != actual.get("schema") or expected.get("identity") != actual.get("identity"):
        raise ValueError("ablation condition contract mismatch")


def _load_lead_bev_weights(runner, args: argparse.Namespace) -> None:
    """Load the explicit frozen LEAD BEV checkpoint after runner construction."""

    import torch

    weights = torch.load(args.lead_bev_ckpt, map_location="cpu", weights_only=False)
    weights = weights.get("model", weights)
    backbone = {
        key[len("backbone.") :]: value
        for key, value in weights.items()
        if str(key).startswith("backbone.")
    }
    runner.bev_encoder.backbone.load_state_dict(backbone or weights, strict=True)
    runner.bev_encoder.eval()
    for param in runner.bev_encoder.parameters():
        param.requires_grad_(False)


def _patch_timm_pretrained(old):
    """Avoid network-backed ImageNet initialization before explicit BEV weights load."""

    original = old.mot_runner.timm

    def offline_create(*args, **kwargs):
        kwargs["pretrained"] = False
        return original.create_model(*args, **kwargs)

    old.mot_runner.timm = SimpleNamespace(create_model=offline_create)
    return original


class QwenSimpleRuntime:
    """Runtime for the Qwen+simple-prompt ablation."""

    def __init__(self, args: argparse.Namespace, device):
        old = leadmot_train()
        original_timm = _patch_timm_pretrained(old)
        try:
            self.inner = old.LeadMoTTrainRuntime(args, device)
        finally:
            old.mot_runner.timm = original_timm
        self.args = args
        self.device = device
        self.runner = self.inner.runner
        _load_lead_bev_weights(self.inner.runner, args)
        self.last_audit = None

    def forward_sample(
        self,
        sample,
        decoder,
        decoder_config,
        decoder_dtype,
        clip=None,
        flow_state=None,
        flow_time=None,
        flow_sample_noise=None,
        sample_trajectory=True,
    ):
        """Run the standard LeadMoT Qwen prefill and inject FM state into decoder."""

        def wrapped_decoder(**kwargs):
            kwargs["pooled_kv"] = [
                (k.detach().clone(), v.detach().clone())
                for k, v in kwargs["pooled_kv"]
            ]
            if flow_state is not None:
                kwargs["flow_state"] = flow_state
                kwargs["flow_time"] = flow_time
            if flow_sample_noise is not None:
                kwargs["flow_sample_noise"] = flow_sample_noise
            kwargs["sample_trajectory"] = sample_trajectory
            return decoder_forward(decoder, kwargs, decoder_dtype, self.device)

        outputs = self.inner.forward_sample(
            sample,
            wrapped_decoder,
            decoder_config,
            __import__("torch").float32,
            clip,
        )
        self.last_audit = {
            "samples": 1,
            "condition/qwen_simple": 1,
            "condition/qwen_prefills": 1,
            "condition/zero_prefix_kv": 0,
        }
        return outputs


class BevOnlyRuntime:
    """Runtime for the BEV-only ablation; it never initializes the Qwen engine."""

    def __init__(self, args: argparse.Namespace, device):
        import torch

        old = leadmot_train()
        self.args = args
        self.device = device
        self.old = old
        old.mot_runner.LEAD_BEV_CKPT_PATH = Path(args.lead_bev_ckpt).expanduser().resolve()
        original_timm = _patch_timm_pretrained(old)
        try:
            self.runner = old.mot_runner.LeadOfflineMoTRunner(
                device=str(device),
                leadmot_ckpt_path=None,
                leadmot_rope_type=args.leadmot_rope_type,
            )
        finally:
            old.mot_runner.timm = original_timm
        self.runner.leadmot_qwen_engine = None
        _load_lead_bev_weights(self.runner, args)
        self.last_audit = None
        self._torch = torch

    def _build_clip(self, sample):
        """Reuse LeadMoTTrainRuntime's CPU clip construction."""

        return self.old.LeadMoTTrainRuntime._build_clip(self, sample)

    def _empty_pooled_kv(self, config, batch: int, dtype):
        """Create zero-length prefix KV so PrefixKVAttention becomes self-attention only."""

        torch = self._torch
        return [
            (
                torch.empty(
                    batch,
                    config.num_kv_heads,
                    0,
                    config.head_dim,
                    device=self.device,
                    dtype=dtype,
                ),
                torch.empty(
                    batch,
                    config.num_kv_heads,
                    0,
                    config.head_dim,
                    device=self.device,
                    dtype=dtype,
                ),
            )
            for _ in range(config.num_layers)
        ]

    def forward_sample(
        self,
        sample,
        decoder,
        decoder_config,
        decoder_dtype,
        clip=None,
        flow_state=None,
        flow_time=None,
        flow_sample_noise=None,
        sample_trajectory=True,
    ):
        """Prepare BEV/navigation and call the FM decoder with empty Qwen KV."""

        import numpy as np
        import torch

        if clip is None:
            clip = self._build_clip(sample)
        clip_len = int(np.asarray(clip["rgb"]).shape[0])
        group = self.runner._build_group_indices(
            clip_len=clip_len,
            anchor_t=clip_len - 1,
            rgb_frame_step=int(sample.get("rgb_frame_step", self.args.rgb_frame_step)),
            rgb_frame_count=int(sample.get("rgb_frame_count", self.args.rgb_frame_count)),
        )
        with torch.no_grad():
            (
                _rgb_pil_list,
                _lidar_pil_list,
                target_point_speed,
                bev_rgb_tensor,
                bev_lidar_tensor,
                *_rest,
            ) = self.runner._prepare_inference_inputs(clip, group)
            if decoder_config.use_bev:
                bev_features = self.runner.bev_encoder(
                    rgb=bev_rgb_tensor,
                    lidar_bev=bev_lidar_tensor,
                )["bev_feature"]
            else:
                bev_features = None

        status = target_point_speed.to(device=self.device, dtype=decoder_dtype)
        final_goal = status[:, 5:7] if decoder_config.use_final_goal else None
        kwargs = dict(
            pooled_kv=self._empty_pooled_kv(decoder_config, status.shape[0], decoder_dtype),
            bev=(
                bev_features.to(device=self.device, dtype=decoder_dtype)
                if bev_features is not None
                else None
            ),
            speed=status[:, 0],
            target_point=status[:, 1:3],
            target_point_next=status[:, 3:5],
            final_goal=final_goal,
            rope_position_offset=0,
            sample_trajectory=sample_trajectory,
        )
        if flow_state is not None:
            kwargs["flow_state"] = flow_state
            kwargs["flow_time"] = flow_time
        if flow_sample_noise is not None:
            kwargs["flow_sample_noise"] = flow_sample_noise
        outputs = decoder_forward(decoder, kwargs, decoder_dtype, self.device)
        outputs["input_status"] = status.detach()
        self.last_audit = {
            "samples": 1,
            "condition/qwen_simple": 0,
            "condition/qwen_prefills": 0,
            "condition/zero_prefix_kv": 1,
        }
        return outputs


def make_runtime(args: argparse.Namespace, device, variant: str):
    """Construct the runtime for a selected ablation."""

    if variant == "qwen_simple":
        return QwenSimpleRuntime(args, device)
    if variant == "bev_only":
        return BevOnlyRuntime(args, device)
    raise ValueError(f"unknown ablation variant: {variant}")


def scalar_metrics_from_counts(counts: Counter) -> dict[str, float]:
    """Average scalar counters by sample count, preserving raw count/* diagnostics."""

    n = int(counts["samples"])
    if n <= 0:
        raise ValueError("no samples in metrics window")
    result = {key: value / n for key, value in counts.items() if key != "samples"}
    result["samples"] = n
    for key, value in counts.items():
        if key.startswith("condition/"):
            result[f"count/{key}"] = value
    return result


def metric_hooks():
    """消融只保留核心指标，不构造先验标签或 RS/EVENT 审计。"""
    return MetricHooks(
        sample_counts=lambda runtime, sample, planning: Counter(runtime.last_audit or {"samples": 1}),
        summarize=scalar_metrics_from_counts,
        case_record=lambda runtime, sample: dict(sample=sample, condition=runtime.last_audit),
        format_metrics=lambda values: (
            f"route_fm_mse={values['route_fm_mse']:.4f} "
            f"waypoint_fm_mse={values['waypoint_fm_mse']:.4f}"
        ),
    )


def evaluate(runtime, decoder, config, rows, args, dtype, rank, world, max_samples=0, dump_dir=None):
    """使用主线同一验证流程，固定 FM 噪声并采样轨迹。"""
    return evaluate_shared(runtime, decoder, config, rows, args, dtype, rank, world,
                           max_samples, dump_dir, old=leadmot_train(), hooks=metric_hooks())


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    ema,
    args,
    contract,
    dataset_hashes,
    cursor,
    step,
    best,
    rank,
    world,
    variant,
):
    """保留本入口 checkpoint 身份，统一保存优化器、EMA 和各 rank RNG。"""
    save_training_checkpoint(
        path, model, optimizer, scheduler, ema, args, contract, dataset_hashes,
        cursor, step, best, rank, world,
        schema=CHECKPOINT_SCHEMA, contract_key="condition_contract",
        extra={"ablation_variant": variant},
    )


def training_device(local_rank: int):
    """Use CUDA for real training, matching action_prior."""

    import torch

    return torch.device("cuda", local_rank)


def train_main(variant: str) -> None:
    """Train one action expert ablation."""

    args = parse_train_args(variant)
    validate_args(args, variant)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    from qwen3vl_local.action_prior.launch import ensure_gpu, prepare_run_directory

    if not args.preflight:
        if (
            int(os.environ.get("WORLD_SIZE", "1")) > 1
            and os.environ.get("ACTION_ABLATION_RUN_READY") != "1"
        ):
            raise ValueError("use this ablation's train.sh/launch.py for multi-GPU")
        ensure_gpu()
    if not args.lead_bev_ckpt:
        args.lead_bev_ckpt = os.environ.get(
            "LEAD_BEV_CKPT", "checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth"
        )
    rows = {split: read_rows(args, split) for split in ("train", "val", "test")}
    requested_world = (
        len(os.environ["GPU_IDS"].split(","))
        if os.environ.get("GPU_IDS")
        else int(os.environ.get("DDP_GPU_COUNT", "4"))
    )
    plan = training_plan(args, rows, requested_world if args.preflight else int(os.environ.get("WORLD_SIZE", "1")), variant)
    contract = build_contract(args, variant)
    if args.preflight:
        print(json.dumps(dict(plan=plan, contract=contract), indent=2, ensure_ascii=False))
        return

    import torch
    import torch.distributed as dist
    old = leadmot_train()

    rank, local_rank, world = old._init_distributed()
    device = training_device(local_rank)
    if os.environ.get("ACTION_ABLATION_RUN_READY") != "1":
        args.output_dir = str(prepare_run_directory(args.output_dir, args.resume))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_hashes = {split: file_hash(Path(args.data_dir) / f"{split}.jsonl") for split in rows}
    torch.manual_seed(args.seed)
    model, config, flow_config = make_model_and_config(args, device)
    dtype = old._dtype(args.decoder_dtype)
    plan = training_plan(args, rows, world, variant)
    optimizer, scheduler, ema = make_optimization(args, model, plan, old)
    cursor, step, best, resume_rng = {"epoch": 0, "micro": 0}, 0, math.inf, None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schema") != CHECKPOINT_SCHEMA or state.get("trajectory_decoder") != TRAJECTORY_DECODER:
            raise ValueError("requires an action_expert_ablation FM checkpoint")
        if state.get("ablation_variant") != variant:
            raise ValueError("resume variant mismatch")
        require_contract(state["condition_contract"], contract)
        cursor, step, best, resume_rng = restore_training_state(
            state, args=args, dataset_hashes=dataset_hashes, world=world, rank=rank,
            config=config, flow_config=flow_config, model=model, optimizer=optimizer,
            scheduler=scheduler, ema=ema, device=device,
        )
        del state

    if rank == 0:
        write_json(out / "training_plan.json", plan)
        write_json(out / "condition_contract.json", contract)
        write_json(out / "config.json", vars(args))
        print(json.dumps(plan, indent=2), flush=True)

    if budget_complete(step, plan, cursor):
        if rank == 0:
            print(f"[{variant}] training budget already complete at step={step}", flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    runtime = make_runtime(args, device, variant)
    if resume_rng:
        torch.set_rng_state(resume_rng["torch"])
        if resume_rng["cuda"] is not None:
            torch.cuda.set_rng_state(resume_rng["cuda"])
    else:
        torch.manual_seed(args.seed + rank)
    decoder = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        if world > 1
        else model
    )
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        if args.resume:
            trim_tensorboard_for_resume(out / "tb", step)
        writer = SummaryWriter(out / "tb")
    run_training_loop(
        args=args, rows=rows, plan=plan, runtime=runtime, model=model, decoder=decoder,
        config=config, flow_config=flow_config, optimizer=optimizer, scheduler=scheduler,
        ema=ema, cursor=cursor, step=step, best=best, device=device, dtype=dtype,
        rank=rank, world=world, out=out, writer=writer, old=old, hooks=metric_hooks(),
        evaluate_fn=evaluate,
        checkpoint=lambda path, cursor, step, best: save_checkpoint(
            path, model, optimizer, scheduler, ema, args, contract, dataset_hashes,
            cursor, step, best, rank, world, variant),
    )


def eval_main(variant: str) -> None:
    """Evaluate one trained ablation checkpoint."""

    p = argparse.ArgumentParser(description=f"Evaluate {variant} action expert ablation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--data-dir", default="")
    p.add_argument("--model-dir", default="")
    p.add_argument("--lead-bev-ckpt", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--dump-cases", action="store_true")
    p.add_argument("--raw", action="store_true", help="use raw decoder instead of EMA")
    cli = p.parse_args()
    from qwen3vl_local.action_prior.launch import ensure_gpu

    ensure_gpu()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import torch
    import torch.distributed as dist
    old = leadmot_train()

    rank, local_rank, world = old._init_distributed()
    state = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("trajectory_decoder") != TRAJECTORY_DECODER:
        raise ValueError("requires an action_expert_ablation FM checkpoint")
    if state.get("ablation_variant") != variant:
        raise ValueError(f"checkpoint variant {state.get('ablation_variant')} != requested {variant}")
    args = argparse.Namespace(**state["args"])
    for key in ("data_root", "data_dir", "model_dir", "lead_bev_ckpt"):
        if getattr(cli, key):
            setattr(args, key, getattr(cli, key))
    validate_args(args, variant)
    contract = build_contract(args, variant)
    require_contract(state["condition_contract"], contract)
    if file_hash(Path(args.data_dir) / f"{cli.split}.jsonl") != state["dataset_hashes"][cli.split]:
        raise ValueError("evaluation dataset differs from checkpoint split")
    config = LeadMoTPlanningDecoderConfig(**state["decoder_config"])
    flow_config = FlowMatchingConfig(**state["flow_config"])
    device = torch.device("cuda", local_rank)
    dtype = old._dtype(args.decoder_dtype)
    model = ConditionalFlowMatchingDecoder(config, flow_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["decoder"], strict=True)
    if not cli.raw:
        model.load_state_dict(state["ema_state_dict"]["shadow"], strict=True)
    checkpoint_step = state["step"]
    dataset_hashes = state["dataset_hashes"]
    del state
    runtime = make_runtime(args, device, variant)
    rows = read_rows(args, cli.split)
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
                ablation_variant=variant,
                contract_identity=contract["identity"],
                dataset_hashes=dataset_hashes,
                checkpoint_step=checkpoint_step,
            ),
        )
        print(json.dumps(metrics, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
