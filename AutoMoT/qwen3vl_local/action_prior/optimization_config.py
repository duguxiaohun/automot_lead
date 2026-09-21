"""三条 action 训练入口共用的优化配置与步数规划；预检不依赖 torch。"""
from __future__ import annotations

import math
import os


# 仅保留对照实验需要的两个选择；算法细节统一固定并写入训练合同。
OPTIMIZATION_DEFAULTS = dict(optimizer="muon_adamw", lr_scheduler="cosine_restarts")
OPTIMIZATION_SETTINGS = dict(cosine_restart_first_epochs=1, cosine_restart_mult=2,
    warmup_basis="first_epoch_optimizer_steps", cycle_policy="epoch_doubling_warmup_inside_first_v1",
    muon_momentum=0.95, muon_ns_steps=5, muon_lr_scale=1.0,
    decay_policy="shared", optimizer_monitor_steps=100,
    validation_policy="epoch_and_shared_cycle_v1")
OPTIMIZATION_ENV = {"OPTIMIZER": "optimizer", "LR_SCHEDULER": "lr_scheduler",
                    "WARMUP_RATIO": "warmup_ratio", "WEIGHT_DECAY": "weight_decay"}


def optimization_env_args():
    """仅转发支持的环境参数，算法细节不再暴露 CLI/环境开关。"""
    return [value for env, key in OPTIMIZATION_ENV.items() if env in os.environ
            for value in ("--" + key.replace("_", "-"), os.environ[env])]


def legacy_optimization_defaults(saved):
    """缺算法身份的旧配置仍按 AdamW/单余弦解释，旧运行需原源码。"""
    saved.setdefault("optimizer", "adamw")
    saved.setdefault("lr_scheduler", "cosine")
    return saved


def validate_optimization(args):
    """预检保留的优化器选择和基础数值。"""
    if args.optimizer not in ("adamw", "muon_adamw"):
        raise ValueError("optimizer must be adamw or muon_adamw")
    if args.lr_scheduler not in ("cosine", "cosine_restarts"):
        raise ValueError("lr_scheduler must be cosine or cosine_restarts")
    for key in ("learning_rate", "weight_decay", "warmup_ratio"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("LR must be positive, weight decay nonnegative, warmup in [0,1)")


def optimization_plan(args, total_steps, steps_per_epoch):
    """首轮内 warmup，周期占1/2/4/...轮；截断预算不压缩原余弦曲线。"""
    validate_optimization(args)
    total, epoch_steps = int(total_steps), int(steps_per_epoch)
    if total < 1 or epoch_steps < 1:
        raise ValueError("optimization needs positive total and per-epoch updates")
    # 使用梯度累积后每轮实际更新数，包含最后不足一个累积窗口的更新。
    # 极短 smoke 留一次有效更新；一轮只有一次更新时不再额外占用 warmup。
    warmup = (min(total - 1, epoch_steps - 1, max(1, int(epoch_steps * args.warmup_ratio)))
              if args.warmup_ratio > 0 else 0)
    lengths, nominal_lengths, ends = [], [], []
    start, span = 0, epoch_steps * OPTIMIZATION_SETTINGS["cosine_restart_first_epochs"]
    while start < total:
        end = min(total, start + span)
        skip = warmup if start == 0 else 0
        lengths.append(end - start - skip)
        nominal_lengths.append(span - skip)
        ends.append(end)
        start += span
        span *= OPTIMIZATION_SETTINGS["cosine_restart_mult"]
    restarting = args.lr_scheduler == "cosine_restarts"
    return dict(version="action_optimization_v5", optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler, total_steps=total, optimizer_steps_per_epoch=epoch_steps,
        warmup_steps=warmup, cycle_steps=lengths if restarting else [],
        nominal_cycle_steps=nominal_lengths if restarting else [],
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, **OPTIMIZATION_SETTINGS,
        validation_warmup_steps=warmup, validation_cycle_steps=lengths,
        validation_nominal_cycle_steps=nominal_lengths, shared_validation_updates=ends,
        muon_update="nesterov_quintic_ns_match_rms_adamw_v1",
        parameter_policy="hidden_blocks_and_velocity_hidden_v1")


def lr_factor(step, plan):
    """step 为即将执行的零基 optimizer update；最终预算后保持零、不额外重启。"""
    total, warmup = plan["total_steps"], plan["warmup_steps"]
    if step >= total:
        return 0.0
    if step < warmup:
        return (step + 1) / warmup
    if plan["lr_scheduler"] == "cosine":
        # 单余弦基线同样使用首轮 warmup；其后一次衰减覆盖剩余预算。
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    offset = step - warmup
    for length, nominal in zip(plan["cycle_steps"], plan["nominal_cycle_steps"]):
        if offset < length:
            return 1.0 if nominal == 1 else 0.5 * (1 + math.cos(math.pi * offset / (nominal - 1)))
        offset -= length
    return 0.0


def cycle_status(completed_steps, plan):
    """描述刚完成的更新；周期编号从1开始，warmup/单余弦为0。"""
    offset = completed_steps - 1 - plan["warmup_steps"]
    for index, (length, nominal) in enumerate(zip(plan["cycle_steps"], plan["nominal_cycle_steps"]), 1):
        if 0 <= offset < length:
            return dict(cycle=index, progress=offset / max(1, nominal - 1),
                        cycle_end=offset == length - 1)
        offset -= length
    return dict(cycle=0, progress=0.0, cycle_end=False)


def validation_cycle_status(step, plan):
    """两种 LR 都用相同参考周期验证，候选点不依赖实际 scheduler。"""
    return cycle_status(step, dict(warmup_steps=plan["validation_warmup_steps"],
                                   cycle_steps=plan["validation_cycle_steps"],
                                   nominal_cycle_steps=plan["validation_nominal_cycle_steps"]))


def full_validation_due(step, plan, *, full_epoch, cycle_end=False):
    """默认每个epoch及共同周期末完整验证，最终不足一轮也验证。"""
    return full_epoch or step >= plan["total_steps"] or step in plan["shared_validation_updates"]
