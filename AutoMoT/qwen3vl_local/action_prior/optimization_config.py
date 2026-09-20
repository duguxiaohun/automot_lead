"""三条 action 训练入口共用的优化配置与步数规划；预检不依赖 torch。"""
from __future__ import annotations

import math
import os


# 仅保留对照实验需要的两个选择；算法细节统一固定并写入训练合同。
OPTIMIZATION_DEFAULTS = dict(optimizer="muon_adamw", lr_scheduler="cosine_restarts")
OPTIMIZATION_SETTINGS = dict(cosine_restart_cycles=4, cosine_restart_mult=2.0,
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


def optimization_plan(args, total_steps):
    """剩余预算按几何比例分配；短 smoke 自动减少周期，正常每周期至少两次更新。"""
    validate_optimization(args)
    total = int(total_steps)
    if total < 1:
        raise ValueError("optimization needs at least one update")
    warmup = max(1, int(total * args.warmup_ratio))
    reference_warmup = min(total - 1, warmup) if args.warmup_ratio > 0 else 0
    remaining = total - reference_warmup
    cycles, mult = OPTIMIZATION_SETTINGS["cosine_restart_cycles"], OPTIMIZATION_SETTINGS["cosine_restart_mult"]
    while cycles > 1 and remaining / sum(mult ** i for i in range(cycles)) < 2:
        cycles -= 1
    weights = [mult ** i for i in range(cycles)]
    boundaries = [0] + [int(remaining * sum(weights[:i]) / sum(weights)) for i in range(1, cycles)] + [remaining]
    lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
    restarting = args.lr_scheduler == "cosine_restarts"
    return dict(version="action_optimization_v4", optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler, total_steps=total,
        warmup_steps=reference_warmup if restarting else warmup,
        cycle_steps=lengths if restarting else [], learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, **OPTIMIZATION_SETTINGS,
        validation_warmup_steps=reference_warmup, validation_cycle_steps=lengths,
        shared_validation_updates=[reference_warmup + end for end in boundaries[1:]],
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
        # 与原 LeadMoT LambdaLR 完全相同，保留基线实验口径。
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    offset = step - warmup
    for length in plan["cycle_steps"]:
        if offset < length:
            return 1.0 if length == 1 else 0.5 * (1 + math.cos(math.pi * offset / (length - 1)))
        offset -= length
    return 0.0


def cycle_status(completed_steps, plan):
    """描述刚完成的更新；周期编号从1开始，warmup/单余弦为0。"""
    offset = completed_steps - 1 - plan["warmup_steps"]
    for index, length in enumerate(plan["cycle_steps"], 1):
        if 0 <= offset < length:
            return dict(cycle=index, progress=offset / max(1, length - 1),
                        cycle_end=offset == length - 1)
        offset -= length
    return dict(cycle=0, progress=0.0, cycle_end=False)


def validation_cycle_status(step, plan):
    """两种 LR 都用相同参考周期验证，候选点不依赖实际 scheduler。"""
    return cycle_status(step, dict(warmup_steps=plan["validation_warmup_steps"],
                                   cycle_steps=plan["validation_cycle_steps"]))


def full_validation_due(step, plan, *, full_epoch, cycle_end=False):
    """默认每个epoch及共同周期末完整验证，最终不足一轮也验证。"""
    return full_epoch or step >= plan["total_steps"] or step in plan["shared_validation_updates"]
