"""Flow matching 训练目标 + Euler 推理积分 + EMA + CFG 引导采样。

v2（与第一档改动一起上线）相比 v1 的差异：

1. **z0 prior（image-to-image style）**：旧版 z0 ~ N(0, I)，相当于"从纯噪声生成
   未来帧"。子目标关键帧通常是当前帧的小幅演化（车继续前进、左转完成），完全
   无关的画面极少。新版允许传入 ``z_prior``（一般取 ``z_history[:, -1]`` 即当前
   帧 latent），按 ``z0 = alpha * z_prior + sigma * noise`` 构造起点。推理时
   ``z_init`` 也用同一份 z_prior 构造，保持训练 / 推理分布一致。

2. **logit-normal t 采样**（SD3 论文配方）：旧版 ``t ~ Uniform[0,1]``，t≈0/1 区域
   监督信号弱；logit-normal(0, 1) 把概率密度集中到 t=0.5 附近，实测能显著加快
   收敛。``--t-sampler uniform`` 保留旧行为做对照。

3. **EMA**：``DiTEMA`` 实现标准 EMA（默认 decay=0.9999）；提供
   ``apply_to(model)`` 上下文管理器临时把 shadow 权重塞进模型用于 val / image-log /
   eval / probe，退出时自动恢复。

4. **CFG 引导推理**：``euler_sample_cfg`` 在每步做两次 forward（cond + uncond），
   按 ``v = v_uncond + scale * (v_cond - v_uncond)`` 合成最终速度。null KV 由
   DiT 自身的 ``null_lang_k/v`` 提供，调用 ``dit.forward(..., force_uncond=True)``。

训练目标本身仍是最朴素的 rectified flow / OT-CFM：

  z0 = alpha * z_prior + sigma * eps,  eps ~ N(0, I)   (或者 alpha=0, sigma=1 退回 v1)
  z1 = VAE(subgoal_keyframe)
  z_t = (1 - t) * z0 + t * z1
  v_target = z1 - z0
  L = || v_pred(z_t, t, condition) - v_target || ^ 2
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Flow matching：训练 batch 构造
# --------------------------------------------------------------------------- #


@dataclass
class FlowBatch:
    """一个 step 中前向所需的张量。

    z0 / z1 都已保留：z0 在做 CFG 双前向时复用，z1 在 multi-t velocity 诊断里复用。
    """

    z_t: torch.Tensor   # [B, C, H, W]
    z0: torch.Tensor    # [B, C, H, W]
    z1: torch.Tensor    # [B, C, H, W]
    t: torch.Tensor     # [B]
    v_target: torch.Tensor  # [B, C, H, W]


def sample_t_logit_normal(batch_size: int, device: torch.device, dtype: torch.dtype,
                          mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """SD3 / Stable Diffusion 3 论文里的 logit-normal t 采样。

    步骤：先采 u ~ N(mean, std^2)，再 t = sigmoid(u)。结果是 t ∈ (0, 1) 上一个
    钟形分布，密度集中在 t≈sigmoid(mean) 附近（默认 mean=0 即 t≈0.5）。

    比 uniform 的优势：rectified flow 的 v_target = z1 - z0 在 t→0/1 时被噪声 z0 / 真值 z1
    单独主导，监督信号方向性不强；中间 t 区域（z_t 是噪声与真值的真混合）才是真正的"学习区"。
    把训练时间集中分配到这里能显著加快收敛——SD3 实测约 2× 提速。
    """

    # randn 直接落到目标 device + dtype，避免后续 sigmoid 时再做隐式拷贝。
    u = torch.randn(batch_size, device=device, dtype=dtype) * std + mean
    return torch.sigmoid(u)


def sample_flow_batch(
    z1: torch.Tensor,
    z_prior: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    sigma: float = 1.0,
    z0: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    t_sampler: str = "uniform",
    t_logit_mean: float = 0.0,
    t_logit_std: float = 1.0,
) -> FlowBatch:
    """根据真值 latent z1 采样一组 flow matching 训练用张量。

    参数：
    - z_prior：可选起点 latent，形状与 z1 完全一致。一般取 ``z_history[:, -1]``
      （当前帧 latent）。当 alpha > 0 时使用：``z0 = alpha * z_prior + sigma * eps``。
      不传或 alpha=0 时退回纯噪声起点（v1 行为）。
    - alpha / sigma：z_prior 与噪声的混合系数；默认 (0, 1) 退回 v1。
      推荐 (1.0, 1.0)：起点完全锚定在当前帧 + 同等强度噪声，模型学的是"从当前帧到子目标"的纯 delta。
    - z0：直接指定起点张量（覆盖 alpha / sigma / z_prior）。仅用于复现 / 调试。
    - t：直接指定 t（覆盖 t_sampler）。仅用于复现 / 调试。
    - t_sampler：``uniform``（v1 行为）或 ``logit_normal``（SD3 推荐）。

    使用方式（训练 step）：
        batch = sample_flow_batch(
            z1=vae.encode(target_keyframe),
            z_prior=z_history[:, -1],
            alpha=1.0, sigma=1.0,
            t_sampler="logit_normal",
        )
        v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv)
        loss   = flow_matching_loss(v_pred, batch.v_target)
    """

    device = z1.device
    dtype = z1.dtype
    batch_size = z1.shape[0]

    # ---- 构造 z0 ----
    if z0 is not None:
        # 复现路径：调用方自带 z0，直接采纳。
        pass
    elif z_prior is not None and alpha > 0.0:
        # z_prior 接入：alpha=1.0, sigma=1.0 是推荐配方（起点锚定当前帧 + 同等噪声）。
        # 这条路径让"子目标生成"任务退化成"从当前帧出发学习 delta"，对图像质量提升最大。
        if z_prior.shape != z1.shape:
            raise ValueError(
                f"z_prior 形状 {tuple(z_prior.shape)} 必须等于 z1 形状 {tuple(z1.shape)}"
            )
        eps = torch.randn_like(z1)
        z0 = alpha * z_prior + sigma * eps
    else:
        # v1 行为：纯随机起点。
        z0 = torch.randn_like(z1)

    # ---- 采样 t ----
    if t is None:
        if t_sampler == "uniform":
            t = torch.rand(batch_size, device=device, dtype=dtype)
        elif t_sampler == "logit_normal":
            t = sample_t_logit_normal(batch_size, device, dtype, t_logit_mean, t_logit_std)
        else:
            raise ValueError(f"未知 t_sampler: {t_sampler}（仅支持 uniform / logit_normal）")

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
    - 想加 t 相关的加权（如 `(1 - t)` 或 min-SNR）时只改这里；
    - 想引入"前景区域 mask"加权或"latent 通道差异"时也只改这里；
    - 训练脚本不感知损失形式变化。

    注：当前 v2 改动采用了 logit-normal t 采样（密度集中在中间区域），
    与 min-SNR 加权目标重合，故损失这里仍是均值 MSE。两者通常二选一即可。
    """

    return (v_pred - v_target).pow(2).mean()


# --------------------------------------------------------------------------- #
# Euler 推理：基础版（cond-only）+ CFG 引导版
# --------------------------------------------------------------------------- #


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
    （pooled_kv、z_history、CFG 开关）由调用方在闭包内捕获，避免本函数依赖具体 DiT 接口。

    z_init：可选起点。不传时按 N(0, I) 随机采（v1 行为）；传入时直接当作 t=0 处的 z0。
    若使用 z_prior 路径（z_init = alpha * z_current + sigma * noise），训练 / 推理保持
    同样构造方式，分布才一致。
    """

    if z_init is None:
        z = torch.randn(shape, device=device, dtype=dtype)
    else:
        # 强制 to(device, dtype)：调用方有可能传不同 device 或 fp32 的 z_init；
        # 不强转会让闭包里 dit 的 attention 在 K/V dtype 不一致时报 SDPA 不匹配。
        z = z_init.to(device=device, dtype=dtype)

    # 等步长 Euler：t 从 0 走到 1，步长 dt = 1/num_steps。
    # rectified flow 的优势就是直线轨迹，Euler 在 32 步内通常已经足够好；要更细可以加到 64，
    # 但收益递减很快，超过 100 步对生成质量几乎无提升。
    dt = 1.0 / num_steps
    for step in range(num_steps):
        # 当前时间点 t_val ∈ {0, dt, 2dt, ..., 1-dt}；最后一步落在 1-dt 而不是 1，
        # 是因为我们用"左端点"近似积分（前向 Euler）；用右端点会跑到 t=1 之外的外推区。
        t_val = step * dt
        t_b = torch.full((shape[0],), t_val, device=device, dtype=dtype)
        v = velocity_fn(z, t_b)
        z = z + dt * v
    return z


@torch.no_grad()
def euler_sample_cfg(
    dit: nn.Module,
    z_history: torch.Tensor,
    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    shape: Tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 32,
    cfg_scale: float = 2.0,
    z_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """带 classifier-free guidance 的 Euler 采样。

    每步做两次 DiT 前向：
        v_cond   = dit(z, z_history, t, pooled_kv)                  # 有语言条件
        v_uncond = dit(z, z_history, t, pooled_kv, force_uncond=True)  # null KV
        v        = v_uncond + cfg_scale * (v_cond - v_uncond)

    cfg_scale=1.0 退化成纯 cond（等价于 ``euler_sample(lambda z,t: dit(...))``）；
    cfg_scale=2.0 是 SD3 / Vista / Flux 推荐起点；推 3.0+ 会强化条件对生成的影响
    但容易出现锐化伪影。

    speed：每步 2× DiT 前向；32 步 = 64 次 DiT forward，相比单步 ~30ms 仍可接受。
    """

    if z_init is None:
        z = torch.randn(shape, device=device, dtype=dtype)
    else:
        z = z_init.to(device=device, dtype=dtype)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t_b = torch.full((shape[0],), t_val, device=device, dtype=dtype)
        # 两次 forward；force_uncond 切换走 DiT 自身的 null_lang_k/v 路径，
        # pooled_kv 在该路径下不被使用（但仍按接口契约传入，避免 None 分支）。
        v_cond = dit(z, z_history, t_b, pooled_kv, force_uncond=False)
        v_uncond = dit(z, z_history, t_b, pooled_kv, force_uncond=True)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        z = z + dt * v
    return z


# --------------------------------------------------------------------------- #
# EMA：标准指数滑动平均，配合 apply_to 上下文管理器做无副作用 evaluation
# --------------------------------------------------------------------------- #


class DiTEMA:
    """指数滑动平均（EMA）：训练每步更新 shadow，evaluation 时临时把 shadow 套上模型。

    生命周期：
        ema = DiTEMA(dit_module, decay=0.9999)         # 训练开始时
        ...
        # 每个 optimizer.step() 之后立即调
        ema.update(dit_module)
        ...
        # evaluation / val / image-log 用 EMA 权重
        with ema.apply_to(dit_module):
            v_pred = dit_module(...)                   # 跑的就是 EMA 权重
        # with 退出后，dit_module 的原始权重已恢复，继续训练不受影响

    保存 ckpt：``ema.state_dict()`` 返回 {name -> tensor}，与 ``model.state_dict()``
    格式兼容；加载时用 ``ema.load_state_dict(sd)``。

    SD/SDXL/Vista 默认开 EMA decay=0.9999，需要至少 ~2k step warmup 才看出收益；
    decay=0.999 见效快但晚期略抖；0.9995 是中间值。
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        # 只追踪 requires_grad 的参数：冻结模块（Qwen / VAE）不在 dit_module 里，但
        # DiT 自身可能未来加入 frozen 模块，统一以 requires_grad 为准更安全。
        # detach().clone() 切断对原参数的引用，shadow 自己持有独立存储。
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """EMA 更新公式：shadow = decay * shadow + (1 - decay) * param。

        放在 optimizer.step() 之后调用；放之前会让 shadow 包含未更新的旧参数，
        EMA 会"慢一步"。
        """

        for name, p in model.named_parameters():
            if name in self.shadow:
                # mul_ + add_ 比 *= + += 快：避免 Python 层 op 开销，纯 CUDA kernel；
                # alpha= 让 add_ 直接做 fma 风格融合。
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # 返回 shadow 的浅引用而不是深拷贝：保存时 torch.save 会做序列化拷贝，
        # 这里再 clone 一遍是浪费。
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True) -> None:
        # strict=True：训练时 ckpt 应该覆盖 shadow 全部 key；缺/多 key 都说明
        # 模型结构与训练时不一致，应该立刻报错而不是静默继续。
        if strict:
            missing = set(self.shadow.keys()) - set(sd.keys())
            unexpected = set(sd.keys()) - set(self.shadow.keys())
            if missing or unexpected:
                raise RuntimeError(
                    f"EMA load_state_dict 不匹配：missing={sorted(missing)[:5]} "
                    f"unexpected={sorted(unexpected)[:5]}"
                )
        for name, v in sd.items():
            if name in self.shadow:
                self.shadow[name].copy_(v.float())

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """临时把 shadow 套上 model，with 块退出时自动还原。

        实现：先把当前参数 detach.clone() 存进 backup，再把 shadow copy_ 进 model；
        yield 之后逐一 copy_ 还原。clone 是必要的：直接保留引用会在 yield 期间被
        copy_ 改写，退出时还原会写回到 shadow 自己。
        """

        backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                # 在 no_grad 里 copy_：不切到 no_grad 会让此处对参数的写入被 autograd
                # 记录，未来反传可能误用 EMA 的写入路径。
                with torch.no_grad():
                    p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    with torch.no_grad():
                        p.data.copy_(backup[name])
