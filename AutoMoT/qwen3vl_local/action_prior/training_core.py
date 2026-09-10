"""Action prior 与两个消融共用的 FM 训练、验证及恢复流程。

入口负责冻结条件模型、合同校验与配置；本模块统一数据步进和数值训练语义。
只有审计计数、case 内容与 checkpoint 身份由入口提供，不能在入口复制训练循环。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Callable

from qwen3vl_local.action_prior.progress import current, report


class GracefulTerminationExit(SystemExit):
    """checkpoint 已安全落盘后，用标准 signal exit code 停止外层流水线。"""

    graceful_termination = True

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(128 + self.signum)


class _TerminationDuringValidation(RuntimeError):
    """内部控制流：所有 rank 一起离开验证并回到可保存 cursor 的调用层。"""


class TerminationCoordinator:
    """只在安全点同步终止请求；signal handler 本身不接触 CUDA、NCCL 或文件。"""

    def __init__(self):
        self.signum = 0
        self.received_at = 0.0
        self._previous = {}
        self._iterators = set()

    def _handler(self, signum, _frame):
        # Python signal handler 里只赋值；真正的同步、保存和日志都留给主循环安全点。
        if not self.signum:
            self.signum = int(signum)
            self.received_at = time.time()

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, *_exc):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def register_iterator(self, iterator):
        self._iterators.add(iterator)
        return iterator

    def close_iterator(self, iterator):
        self._iterators.discard(iterator)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as exc:
                print(f"[worker cleanup warning] {exc}", flush=True)

    def close_iterators(self):
        for iterator in list(self._iterators):
            self.close_iterator(iterator)

    def sync_signal(self, world: int, device) -> int:
        """让只收到 signal 的单个 rank 通知其余 rank；必须由所有 rank 同点调用。"""
        if world == 1:
            return self.signum
        import torch
        import torch.distributed as dist

        value = torch.tensor(self.signum, dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        synced = int(value.item())
        if synced and not self.signum:
            self.signum = synced
        return synced


_ACTIVE_TERMINATION: TerminationCoordinator | None = None


def _archive_stale_termination(out: Path, rank: int, world: int) -> None:
    """resume 前归档上一次正常终止标记，避免旧标记冒充本次运行结果。"""
    import torch.distributed as dist

    marker = Path(out) / "termination.json"
    if rank == 0 and marker.is_file():
        archive = Path(out) / "termination_history"
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / f"termination_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
        marker.replace(target)
    if world > 1:
        dist.barrier()


@dataclass(frozen=True)
class MetricHooks:
    """条件分支的审计接口；消融无需构造任何 RS/EVENT 字段。"""

    sample_counts: Callable
    summarize: Callable
    case_record: Callable
    format_metrics: Callable
    training_audit: Callable | None = None
    epoch_audit: bool = False


def write_json(path, value):
    """原子写入轻量状态文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


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


def accumulation_state(micro, samples, accumulate):
    """返回本窗口实际分母和是否更新，避免残余窗口被错误缩小。"""
    if not 0 <= micro < samples or accumulate <= 0:
        raise ValueError("invalid accumulation cursor")
    window_start = micro // accumulate * accumulate
    divisor = min(accumulate, samples - window_start)
    update = (micro + 1) % accumulate == 0 or micro + 1 == samples
    return divisor, update


def with_validation_pending(cursor: dict, validation_epoch: int, full_epoch: bool) -> dict:
    """标记尚待完成的 epoch/最终验证。"""

    result = dict(cursor)
    result["validation_pending"] = True
    result["validation_epoch"] = validation_epoch
    result["validation_full_epoch"] = bool(full_epoch)
    return result


def clear_validation_pending(cursor: dict) -> dict:
    """验证完成后清除待办字段及已结束的 epoch 计数。"""

    result = dict(cursor)
    for key in ("validation_pending", "validation_epoch", "validation_full_epoch", "full_epoch", "epoch_counts_by_rank"):
        result.pop(key, None)
    return result


def budget_complete(step: int, plan: dict, cursor: dict) -> bool:
    """更新预算用尽且补完验证后，训练才算完成。"""

    return step >= int(plan["actual_step_limit"]) and not cursor.get("validation_pending")


def trim_tensorboard_for_resume(tb_dir: Path, resume_step: int) -> None:
    """续训写日志前归档 checkpoint step 之后的旧 TensorBoard 事件。"""

    if resume_step <= 0 or not tb_dir.exists():
        return
    event_files = sorted(path for path in tb_dir.glob("events.out.tfevents.*") if path.is_file())
    if not event_files:
        return
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except Exception as exc:
        print(f"[resume][warn] TensorBoard trim skipped: cannot import tensorboard: {exc}", flush=True)
        return

    rewrites = []
    for event_file in event_files:
        kept_events = []
        dropped = 0
        try:
            for event in EventFileLoader(str(event_file)).Load():
                if int(getattr(event, "step", 0)) <= int(resume_step):
                    kept_events.append(event)
                else:
                    dropped += 1
        except Exception as exc:
            print(f"[resume][warn] TensorBoard trim skipped for {event_file}: {exc}", flush=True)
            continue
        if dropped:
            rewrites.append((event_file, kept_events, dropped))
    if not rewrites:
        print(f"[resume] TensorBoard already has no events after step {resume_step}", flush=True)
        return

    archive = tb_dir.parent / "tb_resume_archive" / f"from_step_{resume_step}_{time.strftime('%Y%m%d_%H%M%S')}"
    archive.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(tb_dir))
    kept_total = 0
    dropped_total = 0
    try:
        for event_file, kept_events, dropped in rewrites:
            for event in kept_events:
                writer.add_event(event)
            kept_total += len(kept_events)
            dropped_total += int(dropped)
            event_file.replace(archive / event_file.name)
    finally:
        writer.close()
    print(
        f"[resume] TensorBoard trimmed at step {resume_step}: "
        f"kept={kept_total} dropped={dropped_total} archived={archive}",
        flush=True,
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
    *,
    old,
    hooks: MetricHooks,
):
    """独立样本验证；不 padding、不在模型失败时缩小分母。"""
    import torch
    from collections import Counter
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
    # 这份 loader 每次验证后都会销毁，不开启 persistent worker，避免反复验证遗留进程。
    import copy
    loader_args = copy.copy(args)
    loader_args.persistent_workers = False
    loader, _ = old._make_loader(
        selected, loader_args, rank=rank, world_size=world, shuffle=False, epoch_seed=args.seed
    )
    loader.generator = torch.Generator().manual_seed(args.seed + 72)
    iterator = iter(loader)
    if _ACTIVE_TERMINATION is not None:
        _ACTIVE_TERMINATION.register_iterator(iterator)
    totals = Counter()
    was_training = decoder.training
    decoder.eval()
    try:
        idx = 0
        while True:
            try:
                prepared = next(iterator)
                has_sample = 1
            except StopIteration:
                prepared = None
                has_sample = 0
            if _ACTIVE_TERMINATION is not None:
                # 各 rank 的 val shard 可相差一条；短 shard 仍参加同步，避免 collective 数量不一致。
                if _ACTIVE_TERMINATION.sync_signal(world, runtime.device):
                    raise _TerminationDuringValidation
                if world > 1:
                    present = torch.tensor(has_sample, dtype=torch.int32, device=runtime.device)
                    torch.distributed.all_reduce(
                        present, op=torch.distributed.ReduceOp.SUM
                    )
                    if int(present.item()) == 0:
                        break
                elif not has_sample:
                    break
            elif not has_sample:
                break
            if not has_sample:
                continue
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
            planning = dict(
                loss=loss.item(),
                route_fm_mse=rl.item(),
                waypoint_fm_mse=wl.item(),
                **old._compute_planning_metrics(outputs, gt_r, gt_w),
            )
            planning["sampled_trajectory_score"] = sampled_trajectory_score(planning, args)
            totals.update(hooks.sample_counts(runtime, prepared["sample"], planning))
            totals.update(planning)
            report("validation/sample_done", rank_evaluated=idx + 1,
                   last_validation_loss=planning["loss"])
            if progress:
                progress.detail = False
            if dump_dir is not None:
                audit = dict(
                    hooks.case_record(runtime, prepared["sample"]),
                    metrics=planning,
                    pred_route=outputs["pred_route"].float().cpu().tolist(),
                    pred_waypoints=outputs["pred_future_waypoints"].float().cpu().tolist(),
                    gt_route=gt_r.cpu().tolist(),
                    gt_waypoints=gt_w.cpu().tolist(),
                )
                write_json(Path(dump_dir) / f"rank{rank}_case{idx:06d}.json", audit)
            report("validation/data_wait")
            idx += 1
    finally:
        decoder.train(was_training)
        if _ACTIVE_TERMINATION is not None:
            _ACTIVE_TERMINATION.close_iterator(iterator)
        if progress:
            progress.detail = False
    report("validation/rank_merge", announce=True, rank_evaluated=int(totals["samples"]))
    metrics = hooks.summarize(merge_counts(totals, world))
    report("validation/done", announce=True, evaluation_elapsed_s=round(time.monotonic() - eval_started, 1),
           evaluation_samples=metrics["samples"], validation_loss=metrics["loss"],
           route_ade_m=metrics.get("route_ade_m"), waypoint_ade_m=metrics.get("waypoint_ade_m"))
    return metrics


def run_training_loop(*, args, rows, plan, runtime, model, decoder, config, flow_config,
                      optimizer, scheduler, ema, cursor, step, best, device, dtype,
                      rank, world, out, writer, old, hooks: MetricHooks, evaluate_fn, checkpoint):
    """共用累积、更新、日志、验证与保存调度；checkpoint 接收 path/cursor/step/best。"""
    import torch
    import torch.distributed as dist

    global _ACTIVE_TERMINATION
    coordinator = TerminationCoordinator()
    _archive_stale_termination(out, rank, world)
    try:
        with coordinator:
            _ACTIVE_TERMINATION = coordinator
            return _run_training_loop(
                args=args, rows=rows, plan=plan, runtime=runtime, model=model, decoder=decoder,
                config=config, flow_config=flow_config, optimizer=optimizer, scheduler=scheduler,
                ema=ema, cursor=cursor, step=step, best=best, device=device, dtype=dtype,
                rank=rank, world=world, out=out, writer=writer, old=old, hooks=hooks,
                evaluate_fn=evaluate_fn, checkpoint=checkpoint, termination=coordinator,
            )
    finally:
        coordinator.close_iterators()
        _ACTIVE_TERMINATION = None
        if writer:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_training_loop(*, args, rows, plan, runtime, model, decoder, config, flow_config,
                       optimizer, scheduler, ema, cursor, step, best, device, dtype,
                       rank, world, out, writer, old, hooks, evaluate_fn, checkpoint,
                       termination):
    """执行训练；公共包装负责异常时关闭日志和进程组。"""
    import torch
    import torch.distributed as dist

    from qwen3vl_local.action_prior.flow_matching import flow_matching_loss, make_training_flow

    epoch_counts = Counter()

    def save_termination(save_cursor, signum):
        """所有 rank 保存同一个安全 cursor，确认 rank0 原子落盘后再退出。"""
        if hooks.epoch_audit and epoch_counts:
            parts = [None] * world
            if world > 1:
                dist.all_gather_object(parts, dict(epoch_counts))
            else:
                parts[0] = dict(epoch_counts)
            save_cursor = dict(save_cursor, epoch_counts_by_rank=parts)
            if rank == 0 and save_cursor.get("validation_full_epoch"):
                completed_counts = Counter()
                for part in parts:
                    completed_counts.update(part)
                validation_epoch = int(save_cursor["validation_epoch"])
                write_json(
                    out / "epoch_audit" / f"epoch_{validation_epoch+1:03d}_step{step:08d}.json",
                    dict(completed_counts),
                )
        checkpoint(out / "latest.pt", save_cursor, step, best)
        if world > 1:
            dist.barrier()
        if rank == 0:
            write_json(
                out / "termination.json",
                dict(
                    schema="action_training_graceful_termination_v1",
                    signal=int(signum),
                    signal_name=signal.Signals(signum).name,
                    received_at=termination.received_at or time.time(),
                    checkpoint=str((out / "latest.pt").resolve()),
                    optimizer_step=step,
                    cursor=save_cursor,
                    pid=os.getpid(),
                ),
            )
            print(
                f"[termination] {signal.Signals(signum).name} received; "
                f"latest.pt safely saved at optimizer step {step}",
                flush=True,
            )
        report(
            "termination/checkpoint_saved",
            announce=True,
            optimizer_step=step,
            signal=signal.Signals(signum).name,
        )
        return int(signum)

    def finish_pending_validation(pending_cursor):
        """补完 epoch/最终验证，再原子发布无待办的 best/latest。"""
        nonlocal best
        validation_epoch = int(pending_cursor["validation_epoch"])
        # 兼容主线和消融在重构前的两种 cursor 字段名。
        full_epoch = bool(pending_cursor.get("validation_full_epoch", pending_cursor.get("full_epoch", False)))
        try:
            with ema.apply_to(model):
                metrics = evaluate_fn(
                    runtime, model, config, rows["val"], args, dtype, rank, world,
                    0 if full_epoch else args.val_max_samples,
                )
        except _TerminationDuringValidation:
            return pending_cursor, save_termination(
                pending_cursor, termination.signum or signal.SIGTERM
            )
        if rank == 0:
            write_json(out / "validation" / f"epoch_{validation_epoch+1:03d}_step{step:08d}.json", metrics)
            for key, value in metrics.items():
                writer.add_scalar(f"val_epoch/{key}", value, step)
            writer.flush()
        clean_cursor = clear_validation_pending(pending_cursor)
        score = sampled_trajectory_score(metrics, args)
        if score < best:
            best = score
            checkpoint(out / "best.pt", clean_cursor, step, best)
        checkpoint(out / "latest.pt", clean_cursor, step, best)
        requested_signal = termination.sync_signal(world, device)
        if requested_signal:
            return clean_cursor, save_termination(clean_cursor, requested_signal)
        return clean_cursor, None

    if cursor.get("validation_pending"):
        cursor, stopped = finish_pending_validation(cursor)
        if stopped:
            return stopped
    if budget_complete(step, plan, cursor):
        return
    window = Counter()
    first_update_step = step + 1
    log_started = time.monotonic()
    if torch.cuda.is_available():
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
        iterator = termination.register_iterator(iter(loader))
        # 保存的是每个 rank 的计数，恢复后仍按原 world size 汇总完整 epoch。
        epoch_counts = Counter(
            cursor.get("epoch_counts_by_rank", [{}] * world)[rank] if start and hooks.epoch_audit else {}
        )
        dumped = Counter()
        decoder.train()
        report("train/epoch_start", announce=True, epoch=epoch + 1, epochs=args.num_epochs,
               optimizer_step=step, step_limit=plan["actual_step_limit"],
               rank_completed_micro=start, rank_epoch_samples=len(rank_rows),
               grad_accum_steps=args.grad_accum_steps,
               logging_steps=args.logging_steps, val_steps=args.val_steps)
        report("train/data_wait")
        for local_micro, prepared in enumerate(iterator):
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
            planning = dict(loss=loss.item(), route_fm_mse=rl.item(), waypoint_fm_mse=wl.item())
            if args.train_sampled_metrics:
                planning.update(old._compute_planning_metrics(outputs, gt_r, gt_w))
            batch_counts = hooks.sample_counts(runtime, prepared["sample"], planning)
            batch_counts.update(planning)
            window.update(batch_counts)
            if hooks.epoch_audit:
                epoch_counts.update(batch_counts)
            report("train/micro_done", announce=step < first_update_step,
                   rank_completed_micro=micro + 1, last_sample_loss=batch_counts["loss"],
                   sample_elapsed_s=round(time.monotonic() - sample_started, 2))
            if current():
                current().detail = False
            if hooks.training_audit:
                hooks.training_audit(runtime, prepared["sample"], out, epoch, rank, micro, dumped)
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
                next_cursor = with_validation_pending(next_cursor, epoch, full_epoch)
            requested_signal = termination.sync_signal(world, device)
            if requested_signal:
                return save_termination(next_cursor, requested_signal)
            if step == first_update_step or step % args.logging_steps == 0:
                report("train/metrics_rank_merge")
                values = hooks.summarize(merge_counts(window, world))
                if rank == 0:
                    values.update(
                        lr=optimizer.param_groups[0]["lr"],
                        grad_norm=float(grad_norm),
                        samples_seen=epoch * usable + (micro + 1) * world,
                        step_samples=divisor * world,
                        samples_per_second=values["samples"]
                        / max(time.monotonic() - log_started, 1e-6),
                        rank0_peak_allocated_gb=(torch.cuda.max_memory_allocated(device) / 2**30
                                                 if torch.cuda.is_available() else 0.0),
                    )
                    print(
                        f'epoch={epoch+1}/{args.num_epochs} step={step}/{plan["actual_step_limit"]} '
                        f'loss(window_global,{values["samples"]}样本均值)={values["loss"]:.4f} '
                        f'{hooks.format_metrics(values)}',
                        flush=True,
                    )
                    for k, v in values.items():
                        writer.add_scalar(f"train/{k}", v, step)
                    writer.flush()
                window.clear()
                log_started = time.monotonic()
            if step % args.val_steps == 0:
                try:
                    with ema.apply_to(model):
                        metrics = evaluate_fn(
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
                except _TerminationDuringValidation:
                    return save_termination(
                        next_cursor, termination.signum or signal.SIGTERM
                    )
                if rank == 0:
                    write_json(out / "validation" / f"step_{step:08d}.json", metrics)
                    for k, v in metrics.items():
                        writer.add_scalar(f"val/{k}", v, step)
                # 小验证集只观察趋势；best.pt 统一由 epoch 全量验证选取。
                requested_signal = termination.sync_signal(world, device)
                if requested_signal:
                    return save_termination(next_cursor, requested_signal)
            if (
                step % args.save_steps == 0
                or step >= plan["actual_step_limit"]
                or full_epoch
            ):
                if hooks.epoch_audit and not full_epoch:
                    parts = [None] * world
                    if world > 1:
                        dist.all_gather_object(parts, dict(epoch_counts))
                    else:
                        parts[0] = dict(epoch_counts)
                    next_cursor["epoch_counts_by_rank"] = parts
                checkpoint(out / "latest.pt", next_cursor, step, best)
            if step >= plan["actual_step_limit"]:
                break
            report("train/data_wait")
        termination.close_iterator(iterator)
        if hooks.epoch_audit:
            report("train/epoch_rank_merge")
            counts = merge_counts(epoch_counts, world)
            if rank == 0:
                write_json(out / "epoch_audit" / f"epoch_{epoch+1:03d}_step{step:08d}.json", dict(counts))
        cursor, stopped = finish_pending_validation(next_cursor)
        if stopped:
            return stopped
        if step >= plan["actual_step_limit"]:
            break
    return None


def make_model_and_config(args, device):
    """统一构造 FP32 参数的 LeadMoT 与联合轨迹 FM decoder。"""
    import torch
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig

    config = LeadMoTPlanningDecoderConfig(
        num_route_queries=args.route_points,
        num_waypoint_queries=args.waypoint_points,
        rope_type=args.leadmot_rope_type,
        dropout=args.decoder_dropout,
        use_bev=args.use_bev,
        use_final_goal=True,
        use_subgoal=False,
    )
    flow_config = FlowMatchingConfig(
        route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
        waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,
        time_embed_dim=args.flow_time_embed_dim,
        sample_steps=args.flow_sample_steps,
        trajectory_layers=args.flow_trajectory_layers,
        trajectory_heads=args.flow_trajectory_heads,
    )
    model = ConditionalFlowMatchingDecoder(config, flow_config).to(device=device, dtype=torch.float32)
    return model, config, flow_config


def make_optimization(args, model, plan, old):
    """统一 AdamW、学习率调度和 EMA；仅 decoder 参数进入优化器。"""
    import torch
    optimizer = torch.optim.AdamW(
        old._optimizer_param_groups(model, args.weight_decay),
        lr=args.learning_rate, betas=(0.9, 0.95),
    )
    scheduler = old._make_scheduler(optimizer, plan["actual_step_limit"], args.warmup_ratio)
    return optimizer, scheduler, old._DecoderEMA(model, args.ema_decay)


def save_training_checkpoint(
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
    *,
    schema,
    contract_key,
    extra=None,
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = getattr(model, "module", model)
        report("checkpoint/write", announce=True)
        payload = dict(
            schema=schema,
            trajectory_decoder="conditional_joint_trajectory_flow_matching_v2",
            decoder=model.state_dict(),
            decoder_config=asdict(model.config),
            flow_config=asdict(model.flow_config),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            ema_state_dict=ema.state_dict(),
            args=vars(args),
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
        payload[contract_key] = contract
        payload.update(extra or {})
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        report("checkpoint/saved", announce=True)



def restore_training_state(state, *, args, dataset_hashes, world, rank, config, flow_config,
                           model, optimizer, scheduler, ema, device):
    """条件身份由入口先核验；这里统一预算、模型、优化器与恢复游标的校验。"""
    from dataclasses import asdict

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
    if state["decoder_config"] != asdict(config) or state["flow_config"] != asdict(flow_config):
        raise ValueError("resume decoder/flow config mismatch")
    if state.get("best_metric") != "weighted_sampled_route_waypoint_ade_m":
        raise ValueError("resume checkpoint does not identify sampled trajectory ADE as its best metric")
    model.load_state_dict(state["decoder"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    ema.load_state_dict(state["ema_state_dict"])
    ema.shadow = {name: value.to(device=device) for name, value in ema.shadow.items()}
    return state["cursor"], state["step"], state["best_sampled_trajectory_score"], state["rng_by_rank"][rank]
