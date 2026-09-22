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
from qwen3vl_local.action_prior.action_token import separation_contract
from qwen3vl_local.action_prior.priors import PROTOCOL_VERSION
from qwen3vl_local.action_prior.scene_policy import resolve_scene_priors
from qwen3vl_local.action_prior.optimization_config import (
    OPTIMIZATION_DEFAULTS, validate_optimization, optimization_plan,
)
from qwen3vl_local.action_prior.prompts import (
    ANALYSIS_VERSION, PREFILL_VERSION, system_prompt,
)

DEFAULTS = dict(
    **OPTIMIZATION_DEFAULTS,
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
    num_epochs=7,
    learning_rate=2e-4,
    weight_decay=0.01,
    # 第一个 epoch 的 optimizer updates 比例；warmup 占用首周期，不额外加轮数。
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
    # 短摘要最多 80 词；128 token 防截断又不会让失控生成拉长 final KV。
    analysis_tokens=128,
    analysis_review=True,
    # 默认图像+先验 prompt 一次 prefill，显式开启才生成摘要并追加到最终 KV。
    generate_analysis=False,
    # 保留自然 RS/EVENT，显式开启时仅追加所选动作的 Phase3 场景因果句。
    high_level_action_prior=False,
    high_level_action_token=False,
    action_token_separation_weight=0.01,
    action_token_separation_margin=0.5,
    high_level_action_index="",
    recheck_mode="history",
    condition_mode="prior",
    dataset_priors=False,
    prior_labels="",
    prior_noise=0.0,
    prior_noise_invalid_share=0.25,
    # 默认 event_balanced，以全帧语义映射为课程来源：
    # UE1-7、RE2、RE3、RE5 各一份，确认的常规背景池两份。
    sampling_mode="event_balanced",
    event_balance_index="",
    event_balance_route_diverse=True,
    event_balanced_epoch_samples=0,
    # 8 同时保留旧配置缺字段时的恢复兜底；新 CLI 按最终采样模式解析默认值。
    event_balance_max_frame_repeats=8,
    best_selection_metric="natural_ade",
    # 内部保存字段，不再暴露 CLI；新训练按 dataset/action/noise 自动推导。
    event_balanced_scene_priors=False,
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
    # action_prior v5 使用标准 MSE vector-field regression，不再做坐标 L1 回归。
    loss_type="mse",
    flow_route_coordinate_scale_m=30.0,
    flow_waypoint_coordinate_scale_m=20.0,
    flow_time_embed_dim=64,
    flow_sample_steps=10,
    flow_trajectory_layers=2,
    flow_trajectory_heads=8,
    # 训练更新只回归向量场；采样 ADE/FDE 由确定性 validation 报告。显式开启才在
    # 训练日志额外运行 ODE，不让诊断计算改变默认训练随机流或吞吐。
    train_sampled_metrics=False,
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


class SamplingArgumentParser(argparse.ArgumentParser):
    """模式相关默认值在所有别名解析完后确定，不覆盖显式参数或保存值。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args=None, namespace=None):
        parsed, rest = super().parse_known_args(args, namespace)
        from qwen3vl_local.action_prior.event_balance import SAMPLING_MODES
        if parsed.sampling_mode not in SAMPLING_MODES:
            self.error("sampling-mode must be event_balanced or action_balanced; uniform runs require their original source")
        if getattr(parsed, "event_balance_max_frame_repeats", None) is None:
            from qwen3vl_local.action_prior.event_balance import DEFAULT_ACTION_REPEAT_CAP
            parsed.event_balance_max_frame_repeats = (
                DEFAULT_ACTION_REPEAT_CAP if parsed.sampling_mode == "action_balanced" else 8
            )
        return parsed, rest


def parser():
    """所有正式超参数均可 CLI 覆盖。"""
    p = SamplingArgumentParser()
    for k, v in DEFAULTS.items():
        if k == "event_balanced_scene_priors":
            continue
        if isinstance(v, list):
            p.add_argument("--" + k.replace("_", "-"), nargs="+", default=[])
            continue
        p.add_argument(
            "--" + k.replace("_", "-"),
            default=v,
            help=("Warmup fraction of the first epoch optimizer updates (default: 0.05)."
                  if k == "warmup_ratio" else None),
            **(
                {"action": argparse.BooleanOptionalAction}
                if isinstance(v, bool)
                else {"type": type(v)}
            ),
        )
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--models-only", action="store_true")
    p.set_defaults(event_balanced_scene_priors=None, event_balance_max_frame_repeats=None)
    from qwen3vl_local.action_prior.event_balance import add_sampling_aliases
    add_sampling_aliases(p)
    return p


def read_rows(args, split):
    """索引必须来自新 builder，并在实际使用前再检查异常 route。"""
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
    from qwen3vl_local.sft_new_loop_phase3.history_rgb import history_exclusion_reason

    rows, seen, blocked = [], set(), {}
    root = Path(args.data_root).resolve()
    event_active = (
        getattr(args, "sampling_mode", "uniform") in ("event_balanced", "action_balanced")
        or getattr(args, "event_balanced_scene_priors", False)
        or getattr(args, "high_level_action_prior", False)
        or getattr(args, "high_level_action_token", False)
        or bool(getattr(args, "event_balance_index", ""))
    )
    # Phase3 RGB/规则开发路线只能 train；为保持 physical route 隔离，原 val/test 文件中
    # 的同组帧在训练读取时移入 train，holdout 读取时排除。
    from qwen3vl_local.action_prior.event_balance import development_route_groups
    development = development_route_groups() if event_active else frozenset()
    assignments = None
    if event_active and getattr(args, "event_balance_index", ""):
        from qwen3vl_local.action_prior.split_support import split_plan_for_args
        assignments = split_plan_for_args(args)
    source_splits = ("train", "val", "test") if event_active and split == "train" else (split,)
    if assignments is not None:
        source_splits = ("train", "val", "test")
    for source_split in source_splits:
        with (Path(args.data_dir) / f"{source_split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("schema") != "action_prior_data_v1" or row.get("split") != source_split:
                    raise ValueError(
                        "wrong dataset schema/split; rebuild action_prior index"
                    )
                original_split = source_split
                target_split = "train" if row.get("route_group") in development else original_split
                if assignments is not None:
                    target_split = assignments[row["route_group"]]
                if target_split != split:
                    continue
                row["split"] = target_split
                row["event_balance_original_split"] = original_split
                if (
                    row.get("tp_mode") != "route_lookahead"
                    or row.get("rgb_frame_count") != 4
                    or row.get("rgb_frame_step") != 1
                ):
                    raise ValueError(
                        "future-truth navigation or incompatible RGB input is forbidden"
                    )
                if history_exclusion_reason(row["anchor"]):
                    continue
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
    from qwen3vl_local.action_prior.event_balance import annotate_rows

    annotate_rows(args, rows)
    from qwen3vl_local.action_prior.action_token import annotate_tokens
    annotate_tokens(args, rows)
    return rows


def validate_args(args):
    """防止兼容参数改变该路线的核心条件。"""
    resolve_scene_priors(args)
    separation_contract(args)
    # 缺字段只可能来自旧保存配置：其历史行为是生成摘要，不能套用新训练默认值。
    # 后续仍严格核验执行指纹，补字段不表示旧 checkpoint 可以跨源码恢复。
    if not hasattr(args, "generate_analysis"):
        args.generate_analysis = True
    if getattr(args, "high_level_planning", False):
        raise ValueError("high-level-planning was removed; restore old runs with their original source")
    if not hasattr(args, "high_level_action_prior"):
        args.high_level_action_prior = False
    if not hasattr(args, "high_level_action_index"):
        args.high_level_action_index = ""
    # 关闭时路径仅为未使用的配置，允许 CLI 关闭开关覆盖环境中保留的索引路径。
    if args.high_level_action_prior and args.condition_mode != "prior":
        raise ValueError("high-level action prior requires condition-mode prior")
    if getattr(args, "selection_policy", "strict") not in ("available", "strict"):
        raise ValueError("selection_policy must be available or strict")
    if (
        args.use_subgoal
        or args.qwen_adapter_dir
        or not args.use_final_goal
        or not args.use_bev
        or args.tp_mode != "route_lookahead"
        or args.rgb_frame_count not in (1, 4)
        or args.rgb_frame_step != 1
    ):
        raise ValueError(
            "requires base-only final KV, no subgoal, final_goal, current RGB or 4 consecutive RGB, route lookahead"
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
    from qwen3vl_local.action_prior.event_balance import validate_sampling_args

    validate_sampling_args(args)
    if args.event_balanced_scene_priors:
        if args.condition_mode != "prior" or not args.dataset_priors:
            raise ValueError(
                "saved special RE scene priors are a dataset-only privileged prior; "
                "use --dataset-priors --condition-mode prior"
            )
        if args.prior_noise != 0.0:
            raise ValueError(
                "event-balanced scene priors reveal audited context; require --prior-noise 0"
            )
    if getattr(args, "dataset_priors", False):
        if args.condition_mode != "prior":
            raise ValueError("--dataset-priors only replaces the prior source; keep --condition-mode prior")
        if not args.prior_labels:
            raise ValueError("--dataset-priors requires --prior-labels built by build_prior_labels.py")
        if not Path(args.prior_labels).expanduser().is_file():
            raise FileNotFoundError(args.prior_labels)
    elif getattr(args, "prior_labels", ""):
        raise ValueError("--prior-labels is only used together with --dataset-priors")
    if not 0.0 <= getattr(args, "prior_noise", 0.0) <= 1.0 or not (
        0.0 <= getattr(args, "prior_noise_invalid_share", 0.25) <= 1.0
    ):
        raise ValueError("prior noise rate/invalid share must be within [0, 1]")
    if getattr(args, "prior_noise", 0.0) > 0 and not getattr(args, "dataset_priors", False):
        raise ValueError(
            "--prior-noise perturbs dataset ground-truth priors; LoRA priors already carry their own errors"
        )
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
    validate_optimization(args)
    if args.loss_type != "mse":
        raise ValueError("action_prior Flow Matching requires --loss-type mse")
    from qwen3vl_local.action_prior.flow_matching import FlowMatchingConfig

    FlowMatchingConfig(
        route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
        waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,
        time_embed_dim=args.flow_time_embed_dim,
        sample_steps=args.flow_sample_steps,
        trajectory_layers=args.flow_trajectory_layers,
        trajectory_heads=args.flow_trajectory_heads,
    ).validate()


def build_contract(args):
    """冻结上游身份和运行协议；权重路径可迁移，权重字节不能变。"""
    resolve_scene_priors(args)
    from qwen3vl_local.action_prior.action_input import action_input_contract
    action_input = action_input_contract(args)
    from qwen3vl_local.action_prior.action_token import token_contract
    action_token = token_contract(args)
    from qwen3vl_local.action_prior.image_condition import image_contract
    policy = getattr(args, "selection_policy", "strict")
    manifest_path = getattr(args, "selection_manifest", "")
    pinned = read_json(manifest_path) if manifest_path else None
    if pinned and (pinned.get("schema") != "action_prior_selection_v1" or pinned["selection_policy"] != policy):
        raise ValueError("selection manifest policy/schema mismatch")
    roots = getattr(args, "checkpoint_roots", []) or [args.checkpoint_root]
    bundle = getattr(args, "lora_bundle", "")
    dataset = bool(getattr(args, "dataset_priors", False))
    prior_labels = None
    if dataset:
        from qwen3vl_local.action_prior.dataset_labels import PriorNoise, label_source

        if bundle or args.phase1_adapter or args.phase2_adapter or manifest_path:
            raise ValueError(
                "dataset priors never load a LoRA; drop --lora-bundle/--phaseN-adapter/--selection-manifest"
            )
        prior_labels = label_source(
            args.prior_labels,
            PriorNoise(args.prior_noise, args.prior_noise_invalid_share, args.seed),
        )
    bundle_info, packaged_paths = None, {}
    if bundle:
        from qwen3vl_local.action_prior.lora_bundle import verify_bundle, bundle_paths
        bundle_info = verify_bundle(bundle)
        packaged_paths = bundle_paths(bundle)
        if set(packaged_paths) != {"phase1", "phase2"}:
            raise ValueError("training requires a bundle containing both Phase1 and Phase2")
    selected = []
    for phase in (1, 2) if not dataset else ():
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
    p1, p2 = selected if selected else (None, None)
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
    if dataset:
        # 标定真值逐帧命中，没有 LoRA 训练候选池；只声明标签来源，不冒称路线未见。
        upstream = {
            "prior_labels": dict(
                status="dataset_label_lookup",
                routes=[],
                source=prior_labels["path"],
                source_sha256=prior_labels["file_sha256"],
                labeled_frames=prior_labels["labeled_frames"],
                injected_noise=prior_labels["noise"],
                privileged_label_conditioning=True,
                actual_sampled_routes_verified=False,
            )
        }
    else:
        upstream = collect_upstream_sources({"phase1": p1, "phase2": p2}, args)
    identity_payload = dict(
        execution=execution,
        precision=PRECISION_POLICY,
        qwen_dtype=args.qwen_dtype,
        decoder_compute_dtype=args.decoder_dtype,
        recheck_mode=args.recheck_mode,
        condition_mode=args.condition_mode,
        base=base_hashes,
        phase1=p1["fingerprint"] if p1 else None,
        phase2=p2["fingerprint"] if p2 else None,
        bev=bev_hash,
        protocol=PROTOCOL_VERSION,
        analysis=ANALYSIS_VERSION if args.generate_analysis else PREFILL_VERSION,
        system=system_prompt(generate_analysis=args.generate_analysis, rgb_frame_count=args.rgb_frame_count),
        generate_analysis=args.generate_analysis,
        scene_prior_policy=args.scene_prior_policy,
        high_level_action_prior=getattr(args, "high_level_action_prior", False),
        high_level_action_input=action_input,
        high_level_action_token=action_token,
        action_token_separation=separation_contract(args),
        image_condition=image_contract(args),
        rgb_frame_count=args.rgb_frame_count,
        rgb_frame_step=args.rgb_frame_step,
        final_cache_content="inputs_and_analysis" if args.generate_analysis else "inputs_only",
        analysis_tokens=args.analysis_tokens,
        analysis_review=args.analysis_review,
        trajectory_decoder="conditional_joint_trajectory_flow_matching_v2",
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
    from qwen3vl_local.action_prior.event_balance import sampling_contract

    event_balance = sampling_contract(args)
    if event_balance:
        identity_payload["event_balanced_sampling"] = event_balance
    # 旧 LoRA run 的身份负载保持逐字节不变；数据集模式才追加先验来源字段。
    if dataset:
        identity_payload["prior_source"] = prior_labels["prior_source"]
        identity_payload["prior_labels"] = prior_labels["fingerprint"]
        if prior_labels["noise"]:
            identity_payload["prior_noise"] = prior_labels["noise"]
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
    if dataset:
        contract["prior_source"] = prior_labels["prior_source"]
        contract["prior_labels"] = prior_labels
        contract["lora_adapters_loaded"] = False
    if pinned and pinned.get("contract_identity") != contract["identity"]:
        raise ValueError("selection manifest execution/base/BEV identity changed after preflight")
    return contract


def training_plan(args, rows, world):
    """均衡采样先生成全局预算，再按 rank 分片并计算 optimizer steps。"""
    if args.sampling_mode not in ("event_balanced", "action_balanced"):
        raise ValueError("uniform runs require their original source; choose event_balanced or action_balanced")
    action_support = None
    if getattr(args, "high_level_action_token", False):
        from qwen3vl_local.action_prior.action_token import token_support, require_conditioned_training
        action_support = token_support(rows)
        require_conditioned_training(action_support, stage="training plan")
    usable = len(rows["train"]) // world * world
    if usable < world:
        raise ValueError("too few train rows for world size")
    groups = {s: {r["route_group"] for r in rr} for s, rr in rows.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"physical route leakage: {a}/{b}")
    from qwen3vl_local.action_prior.event_balance import (
        REGULAR_BACKGROUND,
        SPECIAL_BUCKETS,
        available_counts,
        event_balanced_total,
        source_contract,
        source_audit,
        weighted_quotas,
    )

    event_available = available_counts(rows["train"])
    missing = [
        key for key in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)
        if int(event_available.get(key, 0)) <= 0
    ]
    if missing:
        raise ValueError(
            "event-balanced sampling needs every UE1-7/RE2/RE3/RE5 bucket and "
            f"a regular background pool; missing={missing} available={event_available}"
        )
    total_function = event_balanced_total
    if args.sampling_mode == "action_balanced":
        from qwen3vl_local.action_prior.action_balance import action_balanced_total
        total_function = action_balanced_total
    usable = total_function(
        rows["train"], requested=args.event_balanced_epoch_samples,
        repeat_cap=args.event_balance_max_frame_repeats, world=world,
    )
    updates = math.ceil((usable // world) / args.grad_accum_steps)
    sampling = dict(mode=args.sampling_mode)
    sampling.update(
        event_balance_source=source_contract(args),
        event_balance_source_audit=source_audit(args),
        train_available=event_available,
        epoch_quotas=weighted_quotas(usable),
        route_diverse=bool(args.event_balance_route_diverse),
        special_bucket_missing=[],
        scene_priors=bool(args.event_balanced_scene_priors),
        max_frame_repeats=int(args.event_balance_max_frame_repeats),
        requested_epoch_samples=int(args.event_balanced_epoch_samples),
        effective_epoch_samples=usable,
        best_selection_metric=args.best_selection_metric,
    )
    if args.sampling_mode == "action_balanced":
        from qwen3vl_local.action_prior.action_balance import action_balance_plan
        sampling["action_balance"] = action_balance_plan(rows["train"], usable, repeat_cap=args.event_balance_max_frame_repeats)
    val_available = available_counts(rows["val"], for_evaluation=True)
    val_missing = [
        key for key in (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)
        if int(val_available.get(key, 0)) <= 0
    ]
    sampling["validation_available"] = val_available
    sampling["validation_bucket_coverage_missing"] = val_missing
    sampling["validation_bucket_coverage_complete"] = not val_missing
    if args.best_selection_metric == "event_balanced_ade" and val_missing:
        raise ValueError(
            "event-balanced best selection needs every bucket in validation; "
            f"missing={val_missing}. Use natural_ade or revise the route split/source."
        )
    return dict(
        samples={s: len(v) for s, v in rows.items()},
        effective_batch=world * args.grad_accum_steps,
        world_size=world,
        micro_batch_per_gpu=1,
        condition_mode=args.condition_mode,
        sampling=sampling,
        prior_source=(
            "dataset_labels" if getattr(args, "dataset_priors", False) else "phase_loras"
        ),
        injected_prior_noise_rate=(
            args.prior_noise if getattr(args, "dataset_priors", False) else 0.0
        ),
        generate_analysis=args.generate_analysis,
        high_level_action_prior=getattr(args, "high_level_action_prior", False),
        high_level_action_token=getattr(args, "high_level_action_token", False),
        action_token_support=action_support,
        action_token_separation=separation_contract(args),
        rgb_frame_count=args.rgb_frame_count,
        prior_image_distribution_shift=(args.rgb_frame_count == 1 and not getattr(args, "dataset_priors", False)
                                        and args.condition_mode == "prior"),
        final_cache_content="inputs_and_analysis" if args.generate_analysis else "inputs_only",
        independent_analysis_review=args.generate_analysis and args.analysis_review,
        cold_generations_per_unique_frame=(
            0
            if args.condition_mode == "base"
            else (
                (0 if getattr(args, "dataset_priors", False)
                 else (15 if args.recheck_mode == "compare" else 9))
                + (1 + int(args.analysis_review) if args.generate_analysis else 0)
            )
        ),
        final_base_prefills_per_presentation=1,
        shared_text_cache=args.cache_priors,
        budget_note="Epoch/LR are initial settings; measure cold and cached throughput with smoke before full training.",
        epochs=args.num_epochs,
        samples_per_epoch=usable,
        ddp_tail_per_epoch=0,
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
        optimization=dict(schedule=optimization_plan(
            args, min(args.max_train_steps, updates * args.num_epochs)
            if args.max_train_steps else updates * args.num_epochs, steps_per_epoch=updates,
        ), parameter_groups=None),
        validation_every_optimizer_steps=args.val_steps,
        periodic_validation_samples=args.val_max_samples,
        epoch_validation="all validation frames; shared cycle boundaries also validate",
    )
