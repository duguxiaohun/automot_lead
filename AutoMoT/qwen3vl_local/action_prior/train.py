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
    for field, reason in audit["invalid"].items():
        c[f"prior/reason/{reason}"] += 1
        c[f"prior/field/{field}/{reason}"] += 1
    c["prior/unconfirmed_samples"] = int(
        any(r != "domain_inapplicable" for r in audit["invalid"].values())
    )
    return c


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
    result = {k: v / n for k, v in counts.items() if k != "samples"}
    result["samples"] = n
    result.update(
        {f"count/{k}": v for k, v in counts.items() if k.startswith("prior/")}
    )
    return result


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

    selected = list(rows)
    if max_samples > 0 and len(selected) > max_samples:
        selected = random.Random(args.seed + 71).sample(selected, max_samples)
    loader, _ = old._make_loader(
        selected, args, rank=rank, world_size=world, shuffle=False, epoch_seed=args.seed
    )
    loader.generator = torch.Generator().manual_seed(args.seed + 72)
    totals = Counter()
    was_training = decoder.training
    decoder.eval()
    try:
        for idx, prepared in enumerate(loader):
            if prepared.get("_error"):
                raise RuntimeError(prepared["_error"])
            with torch.no_grad():
                outputs = runtime.forward_sample(
                    prepared["sample"], decoder, config, dtype, clip=prepared["clip"]
                )
                gt_r = prepared["gt_route"].unsqueeze(0).to(runtime.device)
                gt_w = prepared["gt_waypoints"].unsqueeze(0).to(runtime.device)
                loss, rl, wl = old._planning_loss(
                    outputs,
                    gt_r,
                    gt_w,
                    args.route_loss_weight,
                    args.waypoint_loss_weight,
                    args.loss_type,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            totals.update(audit_counts(runtime.prior.last_audit))
            totals.update(
                dict(
                    loss=loss.item(),
                    route_loss=rl.item(),
                    waypoint_loss=wl.item(),
                    **old._compute_planning_metrics(outputs, gt_r, gt_w),
                )
            )
            if dump_dir is not None:
                audit = dict(
                    runtime.prior.last_audit,
                    sample=prepared["sample"],
                    pred_route=outputs["pred_route"].float().cpu().tolist(),
                    pred_waypoints=outputs["pred_future_waypoints"]
                    .float()
                    .cpu()
                    .tolist(),
                    gt_route=gt_r.cpu().tolist(),
                    gt_waypoints=gt_w.cpu().tolist(),
                )
                write_json(Path(dump_dir) / f"rank{rank}_case{idx:06d}.json", audit)
    finally:
        decoder.train(was_training)
    return metrics_from_counts(merge_counts(totals, world))


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
        payload = dict(
            schema="action_prior_checkpoint_v1",
            decoder=model.state_dict(),
            decoder_config=asdict(model.config),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            ema_state_dict=ema.state_dict(),
            args=vars(args),
            qwen_backbone=contract,
            dataset_hashes=dataset_hashes,
            cursor=cursor,
            step=step,
            best_val=best,
            rng_by_rank=states,
            world_size=world,
        )
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)


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
        print(json.dumps(build_contract(args), indent=2, ensure_ascii=False))
        return
    rows = {s: read_rows(args, s) for s in ("train", "val", "test")}
    plan = training_plan(args, rows, world)
    if args.preflight:
        requested_world = (
            len(os.environ["GPU_IDS"].split(","))
            if os.environ.get("GPU_IDS")
            else int(os.environ.get("DDP_GPU_COUNT", "4"))
        )
        plan = training_plan(args, rows, requested_world)
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
        LeadMoTPlanningDecoder,
        LeadMoTPlanningDecoderConfig,
    )
    from qwen3vl_local.action_prior.runtime import make_runtime

    rank, local_rank, world = old._init_distributed()
    device = training_device(local_rank)
    if os.environ.get("ACTION_PRIOR_RUN_READY") != "1":
        args.output_dir = str(prepare_run_directory(args.output_dir, args.resume))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 大模型哈希只由 rank0 做，错误包装后广播，避免其它 rank 无期限等待。
    box = [None]
    if rank == 0:
        try:
            box[0] = {
                "contract": build_contract(args),
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
    model = LeadMoTPlanningDecoder(config).to(device=device, dtype=dtype)
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
        model.load_state_dict(state["decoder"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema_state_dict"])
        # 旧 helper 从 CPU checkpoint 恢复 shadow 时不迁移设备，后续 GPU update 会错位。
        ema.shadow = {
            name: value.to(device=device) for name, value in ema.shadow.items()
        }
        cursor, step, best = state["cursor"], state["step"], state["best_val"]
        resume_rng = state["rng_by_rank"][rank]
        del state
    if rank == 0:
        write_json(out / "selected_priors.json", contract)
        write_json(out / "training_plan.json", plan)
        write_json(out / "config.json", vars(args))
        print(json.dumps(plan, indent=2), flush=True)
    if step >= plan["actual_step_limit"] and not cursor.get("validation_pending"):
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    runtime = make_runtime(args, device, contract)
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
        if metrics["loss"] < best:
            best = metrics["loss"]
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
        for local_micro, prepared in enumerate(loader):
            micro = start + local_micro
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
                outputs = runtime.forward_sample(
                    prepared["sample"], decoder, config, dtype, clip=prepared["clip"]
                )
                gt_r = prepared["gt_route"].unsqueeze(0).to(device)
                gt_w = prepared["gt_waypoints"].unsqueeze(0).to(device)
                loss, rl, wl = old._planning_loss(
                    outputs,
                    gt_r,
                    gt_w,
                    args.route_loss_weight,
                    args.waypoint_loss_weight,
                    args.loss_type,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / divisor).backward()
            audit = runtime.prior.last_audit
            batch_counts = audit_counts(audit)
            batch_counts.update(
                dict(
                    loss=loss.item(),
                    route_loss=rl.item(),
                    waypoint_loss=wl.item(),
                    **old._compute_planning_metrics(outputs, gt_r, gt_w),
                )
            )
            window.update(batch_counts)
            epoch_counts.update(batch_counts)
            signature = tuple(sorted(set(audit["invalid"].values()))) or ("accepted",)
            if dumped[signature] < (4 if signature == ("accepted",) else 20):
                write_json(
                    out / "audit" / f"epoch{epoch:03d}_rank{rank}_case{micro:07d}.json",
                    dict(audit, sample=prepared["sample"]),
                )
                dumped[signature] += 1
            if not update:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            step += 1
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
            if step % args.logging_steps == 0:
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
                        f'epoch={epoch+1}/{args.num_epochs} step={step}/{plan["actual_step_limit"]} loss={values["loss"]:.4f} invalid={values["prior/invalid_samples"]:.3f}',
                        flush=True,
                    )
                    for k, v in values.items():
                        writer.add_scalar(f"train/{k}", v, step)
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
        if metrics["loss"] < best:
            best = metrics["loss"]
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
