"""Action 专用 Muon/AdamW 分组和预算内 cosine 重启；不依赖新版本 torch.optim.Muon。

Muon 使用 Newton–Schulz 多项式 (3.4445, -4.775, 2.0315) 与 Moonshot
的 0.2*sqrt(max(rows, cols)) RMS 尺度。算法参考：
https://github.com/KellerJordan/Muon
https://arxiv.org/abs/2502.16982
DDP 完成梯度平均后，各 rank 本地执行相同更新，不另做 optimizer collective。
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import torch

from qwen3vl_local.action_prior.optimization_config import optimization_plan, lr_factor


def parameter_groups(model, args):
    """按用途分组，embedding/query/小输入投影/最终二维输出保留 AdamW。"""
    buckets = {}
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            raise ValueError(f"duplicate optimization parameter: {name}")
        seen.add(id(param))
        hidden = name.startswith(("conditioner.blocks.", "trajectory_blocks.")) or name == "velocity_head.1.weight"
        muon = args.optimizer == "muon_adamw" and hidden and param.ndim == 2 and name.endswith("weight")
        # 保留旧 AdamW baseline 的 decay/no-decay 规则。
        decay = param.ndim >= 2 and not any(s in name for s in ("pos_embed", "query_bank", ".embed."))
        # 衰减策略独立于优化器；shared 两种算法均排除 embedding。
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        if isinstance(model.get_submodule(module_name), torch.nn.Embedding):
            decay = False
            muon = False
        algorithm = "muon" if muon else "adamw"
        key = (algorithm, decay)
        group = buckets.setdefault(key, dict(
            params=[], param_names=[], algorithm=algorithm,
            group_name=algorithm + ("_decay" if decay else "_no_decay"),
            lr=args.learning_rate,
            weight_decay=args.weight_decay if decay else 0.0,
        ))
        group["params"].append(param)
        group["param_names"].append(name)
    if not seen:
        raise ValueError("no trainable decoder parameters")
    # AdamW 的原分组顺序是 decay 在前；混合模式 Muon 在前。
    return [buckets[key] for key in (("muon", True), ("muon", False), ("adamw", True), ("adamw", False)) if key in buckets]


def orthogonalized_update(gradient, steps):
    """矩形矩阵沿较小维计算；CUDA 用 BF16 矩阵乘，参数和动量仍是 FP32。"""
    if gradient.ndim != 2:
        raise ValueError("Muon only accepts hidden matrices")
    with torch.autocast(device_type=gradient.device.type, enabled=False):
        x = gradient.to(dtype=torch.bfloat16 if gradient.is_cuda else torch.float32)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        # 先在 FP32 计算范数，避免 BF16 的小范数舍入。
        x = x / x.float().norm().clamp_min(1e-7).to(x.dtype)
        for _ in range(steps):
            a = x @ x.T
            b = -4.775 * a + 2.0315 * (a @ a)
            x = 3.4445 * x + b @ x
        return (x.T if transposed else x).float()


class MuonAdamW(torch.optim.Optimizer):
    """统一 param_groups/state_dict；辅助 AdamW 使用 PyTorch 的标准更新实现。"""

    def __init__(self, groups, *, momentum, ns_steps):
        super().__init__(groups, dict(momentum=momentum, ns_steps=ns_steps))
        adam_groups = [g for g in self.param_groups if g["algorithm"] == "adamw"]
        self._adam = torch.optim.AdamW(adam_groups, betas=(0.9, 0.95), foreach=False) if adam_groups else None
        self._bind_adam_state()

    def _bind_adam_state(self):
        """恢复后重连共享字典；scheduler 修改的 LR 必须作用于辅助 AdamW。"""
        if self._adam is not None:
            self._adam.param_groups = [g for g in self.param_groups if g["algorithm"] == "adamw"]
            self._adam.state = self.state

    def load_state_dict(self, state_dict):
        """参数角色或次序变化时拒绝把 momentum/Adam moments 对应到错误参数。"""
        expected = [(g["algorithm"], g["param_names"]) for g in self.param_groups]
        actual = [(g.get("algorithm"), g.get("param_names")) for g in state_dict["param_groups"]]
        if actual != expected:
            raise ValueError("Muon/AdamW parameter routing mismatch")
        super().load_state_dict(state_dict)
        self._bind_adam_state()

    @torch.no_grad()
    def step(self, closure=None):
        """一个完整累积窗口只更新一次；无梯度参数不更新动量或 weight decay。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with _algorithm_timer(self, "muon"):
            self._step_muon()
        if self._adam is not None:
            with _algorithm_timer(self, "adamw"):
                self._adam.step()
        return loss

    def _step_muon(self):
        """更新隐藏矩阵，计时与低频监控由公共入口控制。"""
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            beta = group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dtype != torch.float32 or param.grad.is_sparse:
                    raise ValueError("action Muon requires dense gradients and FP32 parameters")
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum = state["momentum_buffer"]
                momentum.lerp_(param.grad, 1 - beta)
                update = orthogonalized_update(param.grad.lerp(momentum, beta), group["ns_steps"])
                param.mul_(1 - group["lr"] * group["weight_decay"])
                scale = 0.2 * math.sqrt(max(param.shape))
                param.add_(update, alpha=-group["lr"] * scale)


@contextmanager
def _algorithm_timer(optimizer, algorithm):
    """仅监控步计时；CUDA event 测量设备执行时间，普通步不创建 event。"""
    timings = getattr(optimizer, "_action_timings", None)
    if timings is None:
        yield
        return
    group = next((g for g in optimizer.param_groups if g["algorithm"] == algorithm), None)
    if group is None:
        yield
        return
    device = group["params"][0].device
    if device.type == "cuda":
        with torch.cuda.device(device):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            end.synchronize()
            timings[algorithm] = start.elapsed_time(end)
    else:
        start = time.perf_counter()
        yield
        timings[algorithm] = (time.perf_counter() - start) * 1000


@torch.no_grad()
def optimizer_step_with_metrics(optimizer, *, monitor=False):
    """各组首个有梯度参数的真实 FP32 更新；包含 decay，最多复制四个参数。"""
    if not monitor:
        optimizer.step()
        return {}
    snapshots = []
    for group in optimizer.param_groups:
        for name, param in zip(group["param_names"], group["params"]):
            if param.grad is not None:
                snapshots.append((group["group_name"], name, param, param.detach().clone()))
                break
    optimizer._action_timings = {}
    try:
        if isinstance(optimizer, MuonAdamW):
            optimizer.step()
        else:
            with _algorithm_timer(optimizer, "adamw"):
                optimizer.step()
        metrics = {f"optimizer_time_ms/{key}": value for key, value in optimizer._action_timings.items()}
        for group, name, param, before in snapshots:
            delta = param - before
            prefix = f"update/{group}/{name}"
            metrics[prefix + "/rms"] = delta.square().mean().sqrt().item()
            metrics[prefix + "/relative_norm"] = (delta.norm() / (before.norm() + 1e-12)).item()
        return metrics
    finally:
        optimizer._action_timings = None


def build_optimization(args, model, plan):
    """共用构造器，将实际周期与逐参数路由写入训练计划和 checkpoint 合同。"""
    schedule = optimization_plan(args, plan["actual_step_limit"])
    groups = parameter_groups(model, args)
    if args.optimizer == "muon_adamw":
        optimizer = MuonAdamW(groups, momentum=schedule["muon_momentum"], ns_steps=schedule["muon_ns_steps"])
    else:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    routing = [dict(name=g["group_name"], algorithm=g["algorithm"],
                    parameters=list(g["param_names"]),
                    numel=sum(p.numel() for p in g["params"]),
                    peak_lr=g["lr"], weight_decay=g["weight_decay"]) for g in groups]
    optimizer.action_contract = dict(schedule=schedule, parameter_groups=routing)
    plan["optimization"] = optimizer.action_contract
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, schedule))
    return optimizer, scheduler
