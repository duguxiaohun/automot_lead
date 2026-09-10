#!/usr/bin/env python3
"""全量 frozen Qwen/BEV + LeadMoT 训练，先验 invalid 仍参与轨迹监督。"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.config import (
    parser,
    validate_args,
    build_contract,
    read_rows,
    training_plan,
)
from qwen3vl_local.action_prior.contracts import file_hash, require_contract
from qwen3vl_local.action_prior.metrics import grouped_counts
from qwen3vl_local.action_prior.progress import current, observed, report
from qwen3vl_local.action_prior.training_core import (
    GracefulTerminationExit, MetricHooks, write_json, merge_counts, flow_config_of,
    sampled_trajectory_score,
    accumulation_state, budget_complete, trim_tensorboard_for_resume,
    evaluate as evaluate_shared, run_training_loop,
    make_model_and_config, make_optimization, save_training_checkpoint, restore_training_state,
)


def audit_counts(audit):
    """语义缺失、预计域外 invalid、格式/一致性失败分开统计。"""
    from collections import Counter

    c = Counter(
        {
            "samples": 1,
            "prior/invalid_samples": int(bool(audit["invalid"])),
            "prior/analysis_truncated": int(audit["analysis_truncated"]),
            "prior/analysis_fallback": int(audit.get("analysis_fallback", False)),
            "prior/text_cache_hit": int(audit.get("text_cache_hit", False)),
        }
    )
    rejection = audit.get("analysis_rejection", "none")
    if rejection != "none":
        c[f"prior/analysis_rejection/{rejection}"] += 1
    for criterion, passed in (audit.get("analysis_review") or {}).items():
        c[f"prior/analysis_review/{criterion}/passed"] += int(passed)
        c[f"prior/analysis_review/{criterion}/checked"] += 1
    for field, reason in audit["invalid"].items():
        c[f"prior/reason/{reason}"] += 1
        c[f"prior/field/{field}/{reason}"] += 1
    # confusion 噪声不会进 invalid，必须单独计数才能核对实际注入率。
    noise = (audit.get("dataset_label") or {}).get("noise")
    c["prior/noise_samples"] = int(bool(noise))
    if noise:
        c[f"prior/noise/{noise['channel']}/{noise['mode']}"] += 1
    for item in audit.get("recheck_comparisons", []):
        prefix = f"recheck/{item['mode']}/{item['scope']}"
        c[prefix + "/calls"] += 1
        c[prefix + "/same_prompt"] += int(item["same_prompt"])
        c[prefix + "/compared_fields"] += item["compared_fields"]
        c[prefix + "/accepted_fields"] += item["accepted_fields"]
        for reason in item["errors"].values():
            c[prefix + "/" + reason] += 1
    for item in audit.get("recheck_mode_disagreements", []):
        c[f"recheck/cross_mode/{item['scope']}/cases"] += 1
        c[f"recheck/cross_mode/{item['scope']}/unconfirmed_fields"] += len(
            item["errors"]
        )
    c["prior/unconfirmed_samples"] = int(
        any(r != "domain_inapplicable" for r in audit["invalid"].values())
    )
    c["prior/domain_only_samples"] = int(
        bool(audit["invalid"]) and not c["prior/unconfirmed_samples"]
    )
    return c


def format_prior_metrics(values):
    """显示全 rank 窗口的样本比例与字段原因计数，避免把正常域外当失败。"""
    prefix = "count/prior/reason/"
    # 原因按字段计数，一帧可有多个原因；不能把它们相加当样本比例。
    reasons = {
        key[len(prefix):]: int(value)
        for key, value in sorted(values.items())
        if key.startswith(prefix) and value
    }
    return (
        f'invalid_any={values["prior/invalid_samples"]:.3f} '
        f'domain_only={values["prior/domain_only_samples"]:.3f} '
        f'unconfirmed={values["prior/unconfirmed_samples"]:.3f} '
        f'fallback={values["prior/analysis_fallback"]:.3f} '
        f'reason_fields={json.dumps(reasons, ensure_ascii=False, separators=(",", ":"))}'
    )


def metrics_from_counts(counts):
    """并列报告计数与样本均值，字段 invalid 次数可大于样本数。"""
    n = counts["samples"]
    if not n:
        raise ValueError("no successfully evaluated samples")
    result = {
        k: v / n
        for k, v in counts.items()
        if k != "samples" and not k.startswith("group/")
    }
    for key, value in counts.items():
        if key.startswith("group/"):
            prefix, metric = key.rsplit("/", 1)
            result[key] = (
                value if metric == "samples" else value / counts[prefix + "/samples"]
            )
    result["samples"] = n
    # 自然分布分数仍用于可比的总体报告；事件均衡分数则严格按 1:…:1:2
    # 聚合每个真实全帧桶。缺桶显式标为不完整，不能悄悄以总体 ADE 代替。
    from qwen3vl_local.action_prior.event_balance import (
        EVENT_BALANCE_WEIGHTS,
        REGULAR_BACKGROUND,
        SPECIAL_BUCKETS,
    )

    event_keys = (*SPECIAL_BUCKETS, REGULAR_BACKGROUND)
    has_event_map = any(key.startswith("group/event_balance/") for key in counts)
    missing = [
        key for key in event_keys
        if result.get(f"group/event_balance/{key}/samples", 0) <= 0
    ]
    if has_event_map:
        result["event_balance_bucket_coverage_complete"] = int(not missing)
        result["event_balance_bucket_coverage_missing"] = ",".join(missing)
    sampled_metrics = ("route_ade_m", "waypoint_ade_m", "route_fde_m", "waypoint_fde_m")
    has_event_sampling_metrics = has_event_map and all(
        f"group/event_balance/{bucket}/{metric}" in result
        for bucket in event_keys
        for metric in sampled_metrics
    )
    # 默认训练窗口只记录 FM MSE，不运行 Euler ODE；此时依旧报告桶覆盖，
    # 但绝不能试图从不存在的 ADE/FDE 组指标构造事件均衡分数。
    if has_event_sampling_metrics and not missing:
        weight_sum = sum(EVENT_BALANCE_WEIGHTS[key] for key in event_keys)
        for metric in sampled_metrics:
            result[f"event_balanced_{metric}"] = sum(
                EVENT_BALANCE_WEIGHTS[key]
                * result[f"group/event_balance/{key}/{metric}"]
                for key in event_keys
            ) / weight_sum
        result["event_balanced_trajectory_score"] = (
            result["event_balanced_route_ade_m"]
            + result["event_balanced_waypoint_ade_m"]
        ) / 2.0
    result.update(
        {
            f"count/{k}": v
            for k, v in counts.items()
            if k.startswith(("prior/", "recheck/"))
        }
    )
    return result


def add_dataset_coverage(plan, args, rows):
    """标定先验必须先报命中率；缺帧的样本会退化成完全无条件，不能静默发生。"""
    if not getattr(args, "dataset_priors", False):
        return plan
    from qwen3vl_local.action_prior.dataset_labels import PriorLabelIndex

    plan["dataset_prior_coverage"] = PriorLabelIndex(args.prior_labels).coverage(rows)
    return plan


def metric_hooks():
    """主线保留先验分桶；数值训练流程由 training_core 统一实现。"""
    def sample_counts(runtime, sample, planning):
        counts = audit_counts(runtime.prior.last_audit)
        counts.update(grouped_counts(runtime.prior.last_audit, sample, planning))
        return counts

    def training_audit(runtime, sample, out, epoch, rank, micro, dumped):
        audit = runtime.prior.last_audit
        signature = tuple(sorted(set(audit["invalid"].values()))) or ("accepted",)
        if dumped[signature] < (4 if signature == ("accepted",) else 20):
            write_json(out / "audit" / f"epoch{epoch:03d}_rank{rank}_case{micro:07d}.json",
                       dict(audit, sample=sample))
            dumped[signature] += 1

    return MetricHooks(
        sample_counts=sample_counts, summarize=metrics_from_counts,
        case_record=lambda runtime, sample: dict(runtime.prior.last_audit, sample=sample),
        format_metrics=format_prior_metrics, training_audit=training_audit, epoch_audit=True,
    )


def evaluate(runtime, decoder, config, rows, args, dtype, rank, world, max_samples=0, dump_dir=None):
    """通过共享验证循环执行 FM loss 与纯噪声采样轨迹评测。"""
    from qwen3vl_local.leadmot import train as old
    return evaluate_shared(runtime, decoder, config, rows, args, dtype, rank, world,
                           max_samples, dump_dir, old=old, hooks=metric_hooks())


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
):
    """保留本入口 checkpoint 身份，统一保存优化器、EMA 和各 rank RNG。"""
    save_training_checkpoint(
        path, model, optimizer, scheduler, ema, args, contract, dataset_hashes,
        cursor, step, best, rank, world,
        schema="action_prior_checkpoint_v4", contract_key="qwen_backbone",
    )


def training_device(local_rank):
    """正式训练固定使用对应 CUDA rank；CPU 小模型测试可替换此边界。"""
    import torch

    return torch.device("cuda", local_rank)


@observed
def main():
    """先验证配置和数据，再加载模型；只优化轨迹 decoder。"""
    args = parser().parse_args()
    validate_args(args)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    # launcher 已选卡；直接 Python 入口也先完成选卡再 import torch。
    from qwen3vl_local.action_prior.launch import ensure_gpu, prepare_run_directory

    if not args.preflight:
        if (
            int(os.environ.get("WORLD_SIZE", "1")) > 1
            and os.environ.get("ACTION_PRIOR_RUN_READY") != "1"
        ):
            raise ValueError(
                "use train.sh / launch.py for multi-GPU; the launcher selects one shared GPU mask and run directory"
            )
        ensure_gpu()
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(
        os.environ.get("RANK", "0")
    )
    if not args.lead_bev_ckpt:
        # 本地只读 runner 中的 LEAD 权重路径；preflight 不导入其重依赖。
        args.lead_bev_ckpt = os.environ.get(
            "LEAD_BEV_CKPT", "checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth"
        )
    if args.models_only:
        if not args.preflight:
            raise ValueError("--models-only requires --preflight")
        contract = build_contract(args)
        if args.selection_output:
            selection_path = Path(args.selection_output)
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            with selection_path.open("x", encoding="utf-8") as handle:
                json.dump(dict(schema="action_prior_selection_v1", selection_policy=args.selection_policy,
                               contract_identity=contract["identity"],
                               phase1=contract["phase1"], phase2=contract["phase2"]),
                          handle, indent=2, ensure_ascii=False)
            print(f"[selection pinned] {selection_path}", flush=True)
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return
    rows = {s: read_rows(args, s) for s in ("train", "val", "test")}
    plan = training_plan(args, rows, world)
    if args.preflight:
        requested_world = (
            len(os.environ["GPU_IDS"].split(","))
            if os.environ.get("GPU_IDS")
            else int(os.environ.get("DDP_GPU_COUNT", "4"))
        )
        plan = add_dataset_coverage(training_plan(args, rows, requested_world), args, rows)
        contract = build_contract(args)
        print(
            json.dumps(dict(plan=plan, contract=contract), indent=2, ensure_ascii=False)
        )
        return
    import torch
    import torch.distributed as dist
    from qwen3vl_local.leadmot import train as old
    from qwen3vl_local.action_prior.runtime import make_runtime

    rank, local_rank, world = old._init_distributed()
    device = training_device(local_rank)
    if os.environ.get("ACTION_PRIOR_RUN_READY") != "1":
        args.output_dir = str(prepare_run_directory(args.output_dir, args.resume))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if current():
        current().configure(out)
    report("setup/contract_and_lora_copy", announce=True)
    # 大模型哈希只由 rank0 做，错误包装后广播，避免其它 rank 无期限等待。
    box = [None]
    if rank == 0:
        try:
            from qwen3vl_local.action_prior.lora_bundle import preserve_for_training
            selected_contract = build_contract(args)
            if selected_contract.get("phase1"):
                selected_contract = preserve_for_training(selected_contract, out)
            box[0] = {
                "contract": selected_contract,
                "datasets": {
                    s: file_hash(Path(args.data_dir) / f"{s}.jsonl") for s in rows
                },
            }
        except Exception as exc:
            box[0] = {"error": str(exc)}
    if world > 1:
        dist.broadcast_object_list(box, src=0)
    if "error" in box[0]:
        raise ValueError(box[0]["error"])
    contract, dataset_hashes = box[0]["contract"], box[0]["datasets"]
    from qwen3vl_local.action_prior.provenance import annotate_upstream

    torch.manual_seed(args.seed)
    model, config, flow_config = make_model_and_config(args, device)
    dtype = old._dtype(args.decoder_dtype)
    optimizer, scheduler, ema = make_optimization(args, model, plan, old)
    cursor, step, best, resume_rng = {"epoch": 0, "micro": 0}, 0, math.inf, None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schema") != "action_prior_checkpoint_v4" or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2":
            raise ValueError(
                "requires action_prior v4 joint-trajectory Flow-Matching checkpoint; old independent/regression checkpoints are incompatible"
            )
        require_contract(state["qwen_backbone"], contract)
        cursor, step, best, resume_rng = restore_training_state(
            state, args=args, dataset_hashes=dataset_hashes, world=world, rank=rank,
            config=config, flow_config=flow_config, model=model, optimizer=optimizer,
            scheduler=scheduler, ema=ema, device=device,
        )
        from qwen3vl_local.action_prior.provenance import audit_source_changes

        saved_sources = state["qwen_backbone"].get("upstream_sources", {})
        contract["upstream_source_changes"] = audit_source_changes(
            saved_sources, contract.get("upstream_sources", {})
        )
        contract["retrieved_upstream_sources"] = contract.get("upstream_sources", {})
        # 断点续训沿用原审计路线快照，使半个 epoch 的分组累计仍属于同一套定义。
        contract["upstream_sources"] = saved_sources
        from qwen3vl_local.action_prior.contracts import digest

        contract["audit_identity"] = digest(saved_sources)
        del state
    if contract.get("upstream_sources"):
        plan["upstream_training_pool_audit"] = {
            split: annotate_upstream(items, contract["upstream_sources"])
            for split, items in rows.items()
        }
    if rank == 0:
        add_dataset_coverage(plan, args, rows)
        write_json(out / "selected_priors.json", contract)
        write_json(out / "training_plan.json", plan)
        write_json(out / "config.json", vars(args))
        print(json.dumps(plan, indent=2), flush=True)
    if budget_complete(step, plan, cursor):
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    report("setup/frozen_models", announce=True)
    runtime = make_runtime(args, device, contract)
    if resume_rng:
        torch.set_rng_state(resume_rng["torch"])
        if resume_rng["cuda"] is not None:
            torch.cuda.set_rng_state(resume_rng["cuda"])
    else:
        torch.manual_seed(args.seed + rank)
    report("setup/ddp", announce=True)
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
    termination_signal = run_training_loop(
        args=args, rows=rows, plan=plan, runtime=runtime, model=model, decoder=decoder,
        config=config, flow_config=flow_config, optimizer=optimizer, scheduler=scheduler,
        ema=ema, cursor=cursor, step=step, best=best, device=device, dtype=dtype,
        rank=rank, world=world, out=out, writer=writer, old=old, hooks=metric_hooks(),
        evaluate_fn=evaluate,
        checkpoint=lambda path, cursor, step, best: save_checkpoint(
            path, model, optimizer, scheduler, ema, args, contract, dataset_hashes,
            cursor, step, best, rank, world),
    )
    if termination_signal:
        raise GracefulTerminationExit(termination_signal)


if __name__ == "__main__":
    main()
