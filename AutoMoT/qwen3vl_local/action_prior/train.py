#!/usr/bin/env python3
"""全量 frozen Qwen/BEV + LeadMoT 训练，先验 invalid 仍参与轨迹监督。"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import sys
import time

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


def write_json(path, value):
    """原子写入轻量状态文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


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


def merge_counts(local, world):
    """所有 rank 统一汇总，不用 rank0 的样本代表全局。"""
    import torch.distributed as dist
    from collections import Counter

    if world == 1:
        return Counter(local)
    parts = [None] * world
    dist.all_gather_object(parts, dict(local))
    total = Counter()
    for part in parts:
        total.update(part)
    return total


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


def flow_config_of(decoder):
    """DDP 不透传任意 Python 属性；统一读取未包装 decoder 的 FM 合同。"""
    return getattr(decoder, "module", decoder).flow_config


def sampled_trajectory_score(metrics, args):
    """best.pt 只按实际 Euler 采样轨迹选优，FM loss 仅保留为诊断指标。"""
    required = ("route_ade_m", "waypoint_ade_m")
    if not all(math.isfinite(float(metrics[key])) for key in required):
        raise FloatingPointError("nonfinite sampled trajectory metric")
    return (
        args.route_loss_weight * float(metrics["route_ade_m"])
        + args.waypoint_loss_weight * float(metrics["waypoint_ade_m"])
    )


def evaluate(
    runtime,
    decoder,
    config,
    rows,
    args,
    dtype,
    rank,
    world,
    max_samples=0,
    dump_dir=None,
):
    """独立样本验证；不 padding、不在模型失败时缩小分母。"""
    import torch
    from collections import Counter
    from qwen3vl_local.leadmot import train as old
    from qwen3vl_local.action_prior.flow_matching import (
        flow_matching_loss,
        make_evaluation_flow,
    )

    selected = list(rows)
    if max_samples > 0 and len(selected) > max_samples:
        selected = random.Random(args.seed + 71).sample(selected, max_samples)
    eval_started = time.monotonic()
    progress = current()
    report("validation/start", announce=True, evaluation_samples=len(selected),
           rank_evaluated=0)
    loader, _ = old._make_loader(
        selected, args, rank=rank, world_size=world, shuffle=False, epoch_seed=args.seed
    )
    loader.generator = torch.Generator().manual_seed(args.seed + 72)
    totals = Counter()
    was_training = decoder.training
    decoder.eval()
    try:
        for idx, prepared in enumerate(loader):
            if progress:
                progress.detail = idx == 0
            report("validation/forward", rank_evaluated=idx)
            if prepared.get("_error"):
                raise RuntimeError(prepared["_error"])
            with torch.no_grad():
                gt_r = prepared["gt_route"].unsqueeze(0).to(runtime.device)
                gt_w = prepared["gt_waypoints"].unsqueeze(0).to(runtime.device)
                flow = make_evaluation_flow(
                    gt_r, gt_w, flow_config_of(decoder), prepared["sample"], args.seed
                )
                outputs = runtime.forward_sample(
                    prepared["sample"], decoder, config, dtype, clip=prepared["clip"],
                    flow_state=flow["flow_state"], flow_time=flow["flow_time"],
                    flow_sample_noise=flow["flow_sample_noise"],
                    sample_trajectory=True,
                )
                loss, rl, wl = flow_matching_loss(
                    outputs,
                    flow["flow_target_velocity"],
                    config.num_route_queries,
                    args.route_loss_weight,
                    args.waypoint_loss_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            totals.update(audit_counts(runtime.prior.last_audit))
            planning = dict(
                loss=loss.item(),
                route_fm_mse=rl.item(),
                waypoint_fm_mse=wl.item(),
                **old._compute_planning_metrics(outputs, gt_r, gt_w),
            )
            planning["sampled_trajectory_score"] = sampled_trajectory_score(planning, args)
            totals.update(planning)
            report("validation/sample_done", rank_evaluated=idx + 1,
                   last_validation_loss=planning["loss"])
            if progress:
                progress.detail = False
            totals.update(
                grouped_counts(runtime.prior.last_audit, prepared["sample"], planning)
            )
            if dump_dir is not None:
                audit = dict(
                    runtime.prior.last_audit,
                    sample=prepared["sample"],
                    metrics=planning,
                    pred_route=outputs["pred_route"].float().cpu().tolist(),
                    pred_waypoints=outputs["pred_future_waypoints"]
                    .float()
                    .cpu()
                    .tolist(),
                    gt_route=gt_r.cpu().tolist(),
                    gt_waypoints=gt_w.cpu().tolist(),
                )
                write_json(Path(dump_dir) / f"rank{rank}_case{idx:06d}.json", audit)
            report("validation/data_wait")
    finally:
        decoder.train(was_training)
        if progress:
            progress.detail = False
    report("validation/rank_merge", announce=True, rank_evaluated=int(totals["samples"]))
    metrics = metrics_from_counts(merge_counts(totals, world))
    report("validation/done", announce=True, evaluation_elapsed_s=round(time.monotonic() - eval_started, 1),
           evaluation_samples=metrics["samples"], validation_loss=metrics["loss"],
           route_ade_m=metrics.get("route_ade_m"), waypoint_ade_m=metrics.get("waypoint_ade_m"))
    return metrics


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
    """所有 rank 参与 RNG 采集，resume 从下一 micro-step 精确恢复。"""
    import torch
    import torch.distributed as dist
    from dataclasses import asdict

    report("checkpoint/rng_rank_merge", announce=True, checkpoint=str(path), optimizer_step=step)
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank == 0:
        report("checkpoint/write", announce=True)
        payload = dict(
            schema="action_prior_checkpoint_v4",
            trajectory_decoder="conditional_joint_trajectory_flow_matching_v2",
            decoder=model.state_dict(),
            decoder_config=asdict(model.config),
            flow_config=asdict(model.flow_config),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            ema_state_dict=ema.state_dict(),
            args=vars(args),
            qwen_backbone=contract,
            dataset_hashes=dataset_hashes,
            cursor=cursor,
            step=step,
            best_metric="weighted_sampled_route_waypoint_ade_m",
            best_sampled_trajectory_score=best,
            # 保留这个字段名只为 checkpoint 内部恢复代码稳定；v4 中它明确就是上面的
            # 采样轨迹分数，绝不是 FM vector-field MSE。
            best_val=best,
            rng_by_rank=states,
            world_size=world,
        )
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        report("checkpoint/saved", announce=True)


def accumulation_state(micro, samples, accumulate):
    """返回本窗口实际分母和是否更新，避免残余窗口被错误缩小。"""
    if not 0 <= micro < samples or accumulate <= 0:
        raise ValueError("invalid accumulation cursor")
    window_start = micro // accumulate * accumulate
    divisor = min(accumulate, samples - window_start)
    update = (micro + 1) % accumulate == 0 or micro + 1 == samples
    return divisor, update


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
    from collections import Counter
    from qwen3vl_local.leadmot import train as old
    from qwen3vl_local.leadmot import (
        LeadMoTPlanningDecoderConfig,
    )
    from qwen3vl_local.action_prior.flow_matching import (
        ConditionalFlowMatchingDecoder,
        FlowMatchingConfig,
        flow_matching_loss,
        make_training_flow,
    )
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
    config = LeadMoTPlanningDecoderConfig(
        num_route_queries=args.route_points,
        num_waypoint_queries=args.waypoint_points,
        rope_type=args.leadmot_rope_type,
        dropout=args.decoder_dropout,
        use_bev=args.use_bev,
        use_final_goal=True,
        use_subgoal=False,
    )
    dtype = old._dtype(args.decoder_dtype)
    flow_config = FlowMatchingConfig(
        route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
        waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,
        time_embed_dim=args.flow_time_embed_dim,
        sample_steps=args.flow_sample_steps,
        trajectory_layers=args.flow_trajectory_layers,
        trajectory_heads=args.flow_trajectory_heads,
    )
    model = ConditionalFlowMatchingDecoder(config, flow_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        old._optimizer_param_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    scheduler = old._make_scheduler(
        optimizer, plan["actual_step_limit"], args.warmup_ratio
    )
    ema = old._DecoderEMA(model, args.ema_decay)
    cursor, step, best, resume_rng = {"epoch": 0, "micro": 0}, 0, math.inf, None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("schema") != "action_prior_checkpoint_v4" or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2":
            raise ValueError(
                "requires action_prior v4 joint-trajectory Flow-Matching checkpoint; old independent/regression checkpoints are incompatible"
            )
        require_contract(state["qwen_backbone"], contract)
        if state["dataset_hashes"] != dataset_hashes or state["world_size"] != world:
            raise ValueError("resume dataset/world size differs")
        for key in (
            "num_epochs",
            "grad_accum_steps",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "seed",
            "max_train_steps",
            "ema_decay",
            "max_grad_norm",
            "loss_type",
            "flow_route_coordinate_scale_m",
            "flow_waypoint_coordinate_scale_m",
            "flow_time_embed_dim",
            "flow_sample_steps",
            "flow_trajectory_layers",
            "flow_trajectory_heads",
            "train_sampled_metrics",
            "route_loss_weight",
            "waypoint_loss_weight",
            "smooth_route",
            "decoder_dtype",
            "qwen_dtype",
            "val_max_samples",
            "val_steps",
        ):
            if state["args"][key] != getattr(args, key):
                raise ValueError(f"resume schedule mismatch: {key}")
        if state["decoder_config"] != __import__("dataclasses").asdict(config):
            raise ValueError("resume decoder config mismatch")
        if state.get("flow_config") != __import__("dataclasses").asdict(flow_config):
            raise ValueError("resume flow-matching config mismatch")
        model.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema_state_dict"])
        # 旧 helper 从 CPU checkpoint 恢复 shadow 时不迁移设备，后续 GPU update 会错位。
        ema.shadow = {
            name: value.to(device=device) for name, value in ema.shadow.items()
        }
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
        if state.get("best_metric") != "weighted_sampled_route_waypoint_ade_m":
            raise ValueError("resume checkpoint does not identify sampled trajectory ADE as its best metric")
        cursor, step, best = state["cursor"], state["step"], state["best_sampled_trajectory_score"]
        resume_rng = state["rng_by_rank"][rank]
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
    if step >= plan["actual_step_limit"] and not cursor.get("validation_pending"):
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

        writer = SummaryWriter(out / "tb")
    if cursor.get("validation_pending"):
        # 最后一次 optimizer 更新已保存，但可能在 epoch 验证期间中断；先补完验证和 best 选择。
        with ema.apply_to(model):
            metrics = evaluate(
                runtime,
                model,
                config,
                rows["val"],
                args,
                dtype,
                rank,
                world,
                0 if cursor["full_epoch"] else args.val_max_samples,
            )
        validation_epoch = cursor["validation_epoch"]
        cursor = {"epoch": cursor["epoch"], "micro": cursor["micro"]}
        if rank == 0:
            write_json(
                out
                / "validation"
                / f"epoch_{validation_epoch+1:03d}_step{step:08d}.json",
                metrics,
            )
            for k, v in metrics.items():
                writer.add_scalar(f"val_epoch/{k}", v, step)
        if sampled_trajectory_score(metrics, args) < best:
            best = sampled_trajectory_score(metrics, args)
            save_checkpoint(
                out / "best.pt",
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
            )
        save_checkpoint(
            out / "latest.pt",
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
        )
        if step >= plan["actual_step_limit"]:
            if writer:
                writer.close()
            if dist.is_initialized():
                dist.destroy_process_group()
            return
    window = Counter()
    first_update_step = step + 1
    log_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cursor["epoch"], args.num_epochs):
        ordered = list(rows["train"])
        random.Random(args.seed + epoch).shuffle(ordered)
        usable = len(ordered) // world * world
        rank_rows = ordered[rank:usable:world]
        start = cursor["micro"] if epoch == cursor["epoch"] else 0
        # 固定 generator，DataLoader 建迭代器不得改变 dropout 的全局 RNG。
        loader, _ = old._make_loader(
            rank_rows[start:],
            args,
            rank=0,
            world_size=1,
            shuffle=False,
            epoch_seed=args.seed,
        )
        loader.generator = torch.Generator().manual_seed(args.seed + epoch)
        # 保存的是每个 rank 的计数，恢复后仍按原 world size 汇总完整 epoch。
        epoch_counts = Counter(
            cursor.get("epoch_counts_by_rank", [{}] * world)[rank] if start else {}
        )
        dumped = Counter()
        decoder.train()
        report("train/epoch_start", announce=True, epoch=epoch + 1, epochs=args.num_epochs,
               optimizer_step=step, step_limit=plan["actual_step_limit"],
               rank_completed_micro=start, rank_epoch_samples=len(rank_rows),
               grad_accum_steps=args.grad_accum_steps,
               logging_steps=args.logging_steps, val_steps=args.val_steps)
        report("train/data_wait")
        for local_micro, prepared in enumerate(loader):
            micro = start + local_micro
            if current():
                current().detail = local_micro == 0
            sample_started = time.monotonic()
            report("train/forward", rank_completed_micro=micro,
                   accumulation_micro=micro % args.grad_accum_steps + 1)
            if prepared.get("_error"):
                raise RuntimeError(prepared["_error"])
            divisor, update = accumulation_state(
                micro, len(rank_rows), args.grad_accum_steps
            )
            import contextlib

            with (
                decoder.no_sync()
                if world > 1 and not update
                else contextlib.nullcontext()
            ):
                gt_r = prepared["gt_route"].unsqueeze(0).to(device)
                gt_w = prepared["gt_waypoints"].unsqueeze(0).to(device)
                flow = make_training_flow(gt_r, gt_w, flow_config)
                outputs = runtime.forward_sample(
                    prepared["sample"], decoder, config, dtype, clip=prepared["clip"],
                    flow_state=flow["flow_state"], flow_time=flow["flow_time"],
                    flow_sample_noise=flow["flow_sample_noise"],
                    sample_trajectory=args.train_sampled_metrics,
                )
                loss, rl, wl = flow_matching_loss(
                    outputs,
                    flow["flow_target_velocity"],
                    config.num_route_queries,
                    args.route_loss_weight,
                    args.waypoint_loss_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                report("train/backward_sync" if update else "train/backward_accumulate")
                (loss / divisor).backward()
            audit = runtime.prior.last_audit
            batch_counts = audit_counts(audit)
            batch_counts.update(
                loss=loss.item(), route_fm_mse=rl.item(), waypoint_fm_mse=wl.item()
            )
            if args.train_sampled_metrics:
                batch_counts.update(old._compute_planning_metrics(outputs, gt_r, gt_w))
            batch_counts.update(
                grouped_counts(
                    audit,
                    prepared["sample"],
                    {
                        k: batch_counts[k]
                        for k in (
                            "loss",
                            "route_fm_mse",
                            "waypoint_fm_mse",
                            "route_ade_m",
                            "route_fde_m",
                            "waypoint_ade_m",
                            "waypoint_fde_m",
                        )
                        if k in batch_counts
                    },
                )
            )
            window.update(batch_counts)
            epoch_counts.update(batch_counts)
            report("train/micro_done", announce=step < first_update_step,
                   rank_completed_micro=micro + 1, last_sample_loss=batch_counts["loss"],
                   text_cache_hit=bool(audit.get("text_cache_hit", False)),
                   sample_elapsed_s=round(time.monotonic() - sample_started, 2))
            if current():
                current().detail = False
            signature = tuple(sorted(set(audit["invalid"].values()))) or ("accepted",)
            if dumped[signature] < (4 if signature == ("accepted",) else 20):
                write_json(
                    out / "audit" / f"epoch{epoch:03d}_rank{rank}_case{micro:07d}.json",
                    dict(audit, sample=prepared["sample"]),
                )
                dumped[signature] += 1
            if not update:
                report("train/data_wait")
                continue
            report("train/optimizer")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            step += 1
            report("train/update_done", announce=step == first_update_step,
                   optimizer_step=step, lr=optimizer.param_groups[0]["lr"])
            full_epoch = micro + 1 == len(rank_rows)
            next_cursor = (
                {"epoch": epoch + 1, "micro": 0}
                if full_epoch
                else {"epoch": epoch, "micro": micro + 1}
            )
            if full_epoch or step >= plan["actual_step_limit"]:
                next_cursor.update(
                    validation_pending=True,
                    validation_epoch=epoch,
                    full_epoch=full_epoch,
                )
            if step == first_update_step or step % args.logging_steps == 0:
                report("train/metrics_rank_merge")
                values = metrics_from_counts(merge_counts(window, world))
                if rank == 0:
                    values.update(
                        lr=optimizer.param_groups[0]["lr"],
                        grad_norm=float(grad_norm),
                        samples_per_second=values["samples"]
                        / max(time.monotonic() - log_started, 1e-6),
                        rank0_peak_allocated_gb=torch.cuda.max_memory_allocated(device)
                        / 2**30,
                    )
                    print(
                        f'epoch={epoch+1}/{args.num_epochs} step={step}/{plan["actual_step_limit"]} '
                        f'loss(window_global,{values["samples"]}样本均值)={values["loss"]:.4f} '
                        f'{format_prior_metrics(values)}',
                        flush=True,
                    )
                    for k, v in values.items():
                        writer.add_scalar(f"train/{k}", v, step)
                    writer.flush()
                window.clear()
                log_started = time.monotonic()
            if step % args.val_steps == 0:
                with ema.apply_to(model):
                    metrics = evaluate(
                        runtime,
                        model,
                        config,
                        rows["val"],
                        args,
                        dtype,
                        rank,
                        world,
                        args.val_max_samples,
                    )
                if rank == 0:
                    write_json(out / "validation" / f"step_{step:08d}.json", metrics)
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, step)
                # 小验证集只观察趋势；best.pt 统一由 epoch 全量验证选取。
            if (
                step % args.save_steps == 0
                or step >= plan["actual_step_limit"]
                or full_epoch
            ):
                if not full_epoch:
                    parts = [None] * world
                    if world > 1:
                        dist.all_gather_object(parts, dict(epoch_counts))
                    else:
                        parts[0] = dict(epoch_counts)
                    next_cursor["epoch_counts_by_rank"] = parts
                save_checkpoint(
                    out / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    ema,
                    args,
                    contract,
                    dataset_hashes,
                    next_cursor,
                    step,
                    best,
                    rank,
                    world,
                )
            if step >= plan["actual_step_limit"]:
                break
            report("train/data_wait")
        report("train/epoch_rank_merge")
        counts = merge_counts(epoch_counts, world)
        if rank == 0:
            write_json(
                out / "epoch_audit" / f"epoch_{epoch+1:03d}_step{step:08d}.json",
                dict(counts),
            )
        # 完整 epoch 验证全量 val；smoke 提前停止只跑固定小验证集。
        full_epoch = next_cursor["epoch"] == epoch + 1
        with ema.apply_to(model):
            metrics = evaluate(
                runtime,
                model,
                config,
                rows["val"],
                args,
                dtype,
                rank,
                world,
                0 if full_epoch else args.val_max_samples,
            )
        if rank == 0:
            write_json(
                out / "validation" / f"epoch_{epoch+1:03d}_step{step:08d}.json", metrics
            )
            for k, v in metrics.items():
                writer.add_scalar(f"val_epoch/{k}", v, step)
        next_cursor = {"epoch": next_cursor["epoch"], "micro": next_cursor["micro"]}
        score = sampled_trajectory_score(metrics, args)
        if score < best:
            best = score
            save_checkpoint(
                out / "best.pt",
                model,
                optimizer,
                scheduler,
                ema,
                args,
                contract,
                dataset_hashes,
                next_cursor,
                step,
                best,
                rank,
                world,
            )
        save_checkpoint(
            out / "latest.pt",
            model,
            optimizer,
            scheduler,
            ema,
            args,
            contract,
            dataset_hashes,
            next_cursor,
            step,
            best,
            rank,
            world,
        )
        if step >= plan["actual_step_limit"]:
            break
    if writer:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
