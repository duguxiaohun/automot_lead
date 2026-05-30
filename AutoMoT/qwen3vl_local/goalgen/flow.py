"""Flow matching 训练目标与简易 Euler 推理积分。

采用最朴素的 rectified flow / OT-CFM 目标：

  z0 ~ N(0, I)
  z1 = VAE(subgoal_keyframe)              # 真值 latent
  t ~ U[0, 1]
  z_t = (1 - t) * z0 + t * z1
  v_target = z1 - z0
  L = || v_pred(z_t, t, condition) - v_target || ^ 2

推理：t = 0, z = z0, 每一步 z <- z + dt * v_pred(z, t, condition)，直到 t = 1。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch


@dataclass
class FlowBatch:
    """一个 step 中前向所需的张量。"""

    z_t: torch.Tensor   # [B, C, H, W]
    z0: torch.Tensor    # [B, C, H, W]
    z1: torch.Tensor    # [B, C, H, W]
    t: torch.Tensor     # [B]
    v_target: torch.Tensor  # [B, C, H, W]


def sample_flow_batch(
    z1: torch.Tensor,
    z0: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> FlowBatch:
    """根据真值 latent z1 采样一组 flow matching 训练用张量。

    使用方式（runner 单 step）：
        batch = sample_flow_batch(z1=vae.encode(target_keyframe))
        v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv)
        loss   = flow_matching_loss(v_pred, batch.v_target)

    z0 / t 都可以由外部固定（便于复现 / 调试）；不传时分别从标准正态与均匀分布采。
    """

    if z0 is None:
        # z0 形状要与 z1 完全一致，randn_like 自动对齐 device / dtype。
        z0 = torch.randn_like(z1)
    if t is None:
        # 每条样本独立采 t；放在 z1.device 上免得后续 broadcast 触发隐式拷贝。
        t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)

    # 把 [B] 形状的 t 重塑成 [B, 1, 1, 1]，与 [B, C, H, W] 的 latent 广播相乘时
    # 沿空间维度和 channel 维度全部 broadcast 出 (1-t)/t 缩放。
    t_b = t.view(-1, 1, 1, 1)
    z_t = (1.0 - t_b) * z0 + t_b * z1
    # rectified flow 的速度真值：v = dz/dt 在直线插值下就是 z1 - z0，与 t 无关。
    v_target = z1 - z0
    return FlowBatch(z_t=z_t, z0=z0, z1=z1, t=t, v_target=v_target)


def flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """简单 MSE 损失：‖v_pred - v_target‖² 对所有维度取平均。

    保留这个独立函数而不是直接在 runner 写 `.mean()`，是为了：
    - v2 想加 t 相关的加权（如 `(1 - t)` 或 `1/(1-t+eps)`）时只改这里；
    - v2 想引入"前景区域 mask"加权或"latent 通道差异"时也只改这里；
    - 训练脚本不感知 loss 形式变化。
    """

    return (v_pred - v_target).pow(2).mean()


@torch.no_grad()
def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: Tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 32,
    z_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler 积分推理，把 z0 推到 z1_hat。

    velocity_fn 必须是 (z_t, t) -> v_pred 的闭包，t 是 [B] 的 float 张量。条件
    （pooled_kv、z_history）由调用方在闭包内捕获，避免本函数依赖具体 DiT 接口。
    """

    # z_init 为空时随机一个标准正态，作为 t=0 处的起点。
    # 用 randn 而不是 zeros 是 rectified flow 的硬性约定：训练时 z0 ~ N(0, I)，
    # 推理时 z0 也必须从同一分布采，否则模型在初值方向上的 v_pred 就是分布外。
    if z_init is None:
        z = torch.randn(shape, device=device, dtype=dtype)
    else:
        # 强制 to(device, dtype)：调用方有可能传不同 device 或 fp32 的 z_init；
        # 不强转会让闭包里 dit 的 attention 在 K/V dtype 不一致时报 SDPA mismatch。
        z = z_init.to(device=device, dtype=dtype)

    # 等步长 Euler：t 从 0 走到 1，步长 dt = 1/num_steps。
    # rectified flow 的优势就是直线轨迹，Euler 在 32 步内通常已经足够好；要更细可以加到 64，
    # 但收益递减很快，超过 100 步对生成质量几乎无提升。
    dt = 1.0 / num_steps
    for step in range(num_steps):
        # 当前时间点 t_val ∈ {0, dt, 2dt, ..., 1-dt}；最后一步落在 1-dt 而不是 1，
        # 是因为我们用"左端点"近似积分（前向 Euler）；用右端点会跑到 t=1 之外的外推区。
        t_val = step * dt
        # 把标量 t 广播到 [B] 张量，方便闭包内部用同样的 broadcast 规则；
        # full() 而不是 expand(randn[]) 是因为 t 必须是确定常量，randn 会引入不需要的噪声。
        t_b = torch.full((shape[0],), t_val, device=device, dtype=dtype)
        # velocity_fn 是 (z, t) -> v 的闭包；runner 用 lambda 把 pooled_kv / z_history 捕获进来。
        # 这样 euler_sample 本身完全不知道 DiT 接口签名，可以无修改给其它生成主干复用。
        v = velocity_fn(z, t_b)
        # 一阶 Euler 更新：z(t + dt) ≈ z(t) + dt * v(z(t), t)。
        # rectified flow 的真值轨迹就是直线，所以一阶 Euler 在理论上是无误差的近似（仅
        # 受 v_pred 自身误差影响），不需要 RK4 / Heun 这类高阶积分器。
        z = z + dt * v
    return z
