"""DiT-MoT：12 层 joint-attention 的 latent 扩散主干（v2 架构对齐 Qwen K/V）。

v2 相对 v1 的核心改动（详见 PROJECT_CONTEXT.md §15）：

- **维度直接对齐 Qwen K/V 子空间**：hidden_dim=1024、n_heads=8、head_dim=128，
  全部跟 Qwen3-VL-4B-Instruct 的 (num_key_value_heads, head_dim) 完全相同。
  作用是**彻底删掉 v1 里的 `lang_k_proj` / `lang_v_proj` 两条 1024→768 跨维线性投影**，
  Qwen 的 K/V 直接当 DiT 的语言 K/V 用，零信息损失。
- **patch_size=4**：视觉 token 数缩到 v1 的 1/4（24×72→6×18），attention FLOPs 大约
  缩 16×。代价是输出空间分辨率粗一倍；GoalGen 的 subgoal latent 不需要像素级细节，
  这个 trade-off 划算。
- **q_norm / k_norm（RMSNorm）**：仿 Qwen3 / AutoMoT PackedAttentionMoT，对 vision Q/K
  在 attention 内做 head-wise RMS 归一化。**对 language K 不再次 norm**——Qwen 那边
  已经 k_norm 过了，再做会重复归一化，破坏量级。这条解决"vision K 和 language K
  量级不一致导致 softmax 偏向某一边"的隐患。
- **LayerNorm → RMSNorm**：norm1 / norm2 / final_norm 全部换 RMSNorm（无 affine，
  scale/shift 由 AdaLN 提供）。SD3 / Flux / Qwen3 一致做法，对训练稳定有帮助。
- **MLP → SwiGLU**：旧版 `Linear→GELU(tanh)→Linear`，新版 `gate * silu(up) → down`。
  参数量上涨 1.5×，但单位参数 quality 更高。

输入是冻结 VAE 编出来的潜变量：含噪目标 z_t 与历史帧 z_history。z_t 与每一帧
历史潜变量共享同一个 Patchify，拼成视觉 token 序列；类型由 type_embed 区分、
时序由 frame_embed 区分。timestep 用 AdaLN-Zero 注入；vision token 不修改 language
部分。输出只读出"z_t 那一段" token，反图块化回 [B, 4, H/8, W/8]，作为速度预测。

关键形状（默认配置，针对 LEAD 1152x384）：

- VAE 潜变量: [B, 4, 48, 144]
- patch_size = 4  -> 图块网格 (12, 36) -> 每个潜变量 432 个 token
- 视觉 token = 432 (z_t) + F * 432 (z_history)，数据构建器默认 F=4 -> 2160 个 token
- hidden_dim = 1024, n_heads = 8, head_dim = 128
- language token = Qwen prefill seq_len, 例如 ~2300
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_2d_sincos_pos_embed(hidden_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """简单 2D sin-cos 位置编码，沿 H 和 W 各占 hidden_dim 的一半。

    返回形状 [grid_h * grid_w, hidden_dim]，写成 float32 让加载时易复用。

    用法：DiTMoT 在 __init__ 里调用一次，注册成 buffer；前向时按当前网格
    切片即可，避免每次都重新计算。

    为什么不用可学习位置编码：vision token 数（grid 尺寸）会随 LEAD vs Vista
    或 patch_size 调整变化，固定 sin-cos 让权重对分辨率变化更鲁棒。
    """

    if hidden_dim % 4 != 0:
        # 实现里把 hidden_dim 拆 H/W 各半，每一半内部再 sin/cos 各半，所以 4 整除。
        raise ValueError("hidden_dim 必须能被 4 整除（H 与 W 各拿一半 sin/cos）")
    half = hidden_dim // 2

    def _axis(length: int, dim: int) -> torch.Tensor:
        """计算单一坐标轴（H 或 W）上长度为 length 的 sin-cos 编码。

        - omega 是 [dim/2] 的频率衰减序列；i 越大频率越低（周期越长）。
        - out[p, i] = pos[p] * omega[i]，再拼 sin/cos 各 dim/2 维。
        - 与经典 ViT / DiT 实现一致：omega = 1 / 10000^(2i/dim)。
        """

        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000.0 ** omega)
        pos = torch.arange(length, dtype=torch.float32)
        # einsum 等价于 pos[:, None] * omega[None, :]，输出 [length, dim/2]。
        out = torch.einsum("p,d->pd", pos, omega)
        # 沿最后一维拼 sin/cos -> [length, dim]。
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    # 两条轴各算一次，再 broadcast 成 (H, W, hidden) 网格。
    emb_h = _axis(grid_h, half)  # [grid_h, hidden_dim/2]
    emb_w = _axis(grid_w, half)  # [grid_w, hidden_dim/2]
    emb = torch.zeros(grid_h, grid_w, hidden_dim)
    # 前一半维度只跟 H 有关 -> 在 W 维上 broadcast。
    emb[..., :half] = emb_h[:, None, :]
    # 后一半维度只跟 W 有关 -> 在 H 维上 broadcast。
    emb[..., half:] = emb_w[None, :, :]
    # 摊平成 token 序列，方便 forward 直接加到 [B, N, hidden] 上。
    return emb.reshape(grid_h * grid_w, hidden_dim)


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """标准 sinusoidal 时间步 embedding。t 形状 [B]，返回 [B, dim]。

    与 DDPM / DiT 通用实现一致：用一组对数等间距的频率给标量 t 编码成 dim 维向量，
    后面接 MLP 投到 cond_dim，再喂给 AdaLN modulation。

    flow matching 里 t ∈ [0, 1] 而不是 1000 步整数，但同样的 sin/cos 编码仍然成立；
    内部统一用 float32 计算可以避免半精度下小 t 值精度损失。
    """

    half = dim // 2
    # freqs[i] = exp(-log(max_period) * i / half) = 1 / max_period^(i/half)。
    # i 越大 -> freq 越小 -> 表达越粗。
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    # args[b, i] = t[b] * freqs[i]，再分别走 cos / sin，拼成 [B, dim]。
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # dim 是奇数时尾巴补一列 0；保留这条分支是为了让 cond_dim=255/257 等也能用。
        emb = F.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    """简单 RMSNorm 实现：x / sqrt(mean(x^2) + eps) * weight。

    本模块多处复用：
    - block 的 norm1 / norm2 / final_norm（不带 affine，scale/shift 由 AdaLN 提供）
    - attention 内的 q_norm / k_norm（带 affine，对 head_dim 这一维做归一化，
      与 Qwen3 / AutoMoT 的 q_norm/k_norm 语义一致）

    elementwise_affine=True 时学习 weight，形状 [dim]；与 nn.LayerNorm 接口对齐。
    内部强制 float32 计算后再转回原 dtype，避免 bf16 下 mean 漂移。
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            # weight=1 是恒等起点；训练初期等价于"只做 RMS 归一化、不缩放"。
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            # 不学 affine 时也注册一个 None 占位，避免外部代码做 hasattr 检查时混乱。
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 升 fp32 计算 RMS：bf16 下 x^2 平均很容易丢精度，归一化后量级会漂。
        # 计算完再转回 x 的 dtype，避免下游 dtype 不一致。
        orig_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        x_fp = x_fp * rms
        x_out = x_fp.to(orig_dtype)
        if self.elementwise_affine:
            x_out = x_out * self.weight
        return x_out


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class Patchify(nn.Module):
    """latent [B,C,H,W] -> token [B, N, hidden] + grid shape。

    使用 stride=patch_size 的 Conv2d 等价于线性 patch embed；同时输出 grid 形状
    方便 unpatchify。
    """

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"latent H/W 不被 patch_size={self.patch_size} 整除：H={h} W={w}")
        feat = self.proj(x)  # [B, hidden, gh, gw]
        gh, gw = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, gh*gw, hidden]
        return tokens, (gh, gw)


class Unpatchify(nn.Module):
    """token [B, gh*gw, hidden] -> latent [B, out_channels, gh*ps, gw*ps]。"""

    def __init__(self, hidden_dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        gh, gw = grid
        b, n, h = tokens.shape
        if n != gh * gw:
            raise ValueError(f"token 数 {n} 不等于 gh*gw={gh*gw}")
        x = self.proj(tokens)  # [B, gh*gw, C*p*p]
        x = x.view(b, gh, gw, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(b, self.out_channels, gh * self.patch_size, gw * self.patch_size)


class AdaLNModulation(nn.Module):
    """AdaLN-Zero 风格的时间 conditioning：从 cond 投影出 6 个 scale/shift。

    顺序为：(shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)。
    输出 gate 初始化为 0，使训练初期等价于跳过 attention/MLP，对稳定 flow matching 有帮助。
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_dim * 6)
        # zero-init：scale=0 + gate=0 让 block 在初始阶段不改 vision token。
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.linear(F.silu(cond))  # [B, 6H]
        return tuple(out.chunk(6, dim=-1))


class SwiGLU(nn.Module):
    """SwiGLU 前馈：gate * silu(up) → down。

    替换旧版 `Linear → GELU(tanh) → Linear`。参数量从 2*hidden*inner 涨到 3*hidden*inner，
    但 LLaMA / Qwen3 / SD3 / Flux 都用 SwiGLU，单位参数 quality 比 GELU 高，
    收敛也更稳。inner = int(hidden * mlp_ratio)；保持跟旧版同一 mlp_ratio=4。
    """

    def __init__(self, hidden_dim: int, inner_dim: int):
        super().__init__()
        # gate / up 两条并列的投影；为了和 Qwen3 的命名风格一致，
        # gate ≡ Qwen 的 gate_proj，up ≡ Qwen 的 up_proj，down ≡ Qwen 的 down_proj。
        self.gate_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(gate) * up 是 SwiGLU 的标准形式（也叫 SiLU-GLU）；
        # Qwen3 / LLaMA-2 等都是 silu，而不是 GELU-GLU。
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class JointAttention(nn.Module):
    """vision Q 与 (vision K/V + language K/V) 做一次 attention。

    v2 关键改动：DiT 的 (n_heads, head_dim) 直接 = Qwen 的 (num_key_value_heads, head_dim)
    = (8, 128)，所以语言 K/V **无需任何线性投影**就能直接 concat。彻底删掉了 v1 的
    `lang_k_proj` / `lang_v_proj` 两条 1024→hidden 跨维投影。

    vision Q/K 在投影后额外走一次 RMSNorm（q_norm / k_norm），与 Qwen3 / AutoMoT 的
    PackedAttentionMoT 的做法一致。这一步对量级对齐至关重要：Qwen K 已经是 k_norm
    过的（单位 RMS），vision K 不 norm 就会量级失配，softmax 偏向某一侧。
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # vision 三投影。bias=False 与 Qwen3 默认对齐（attention bias 通常关掉）。
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # head-wise RMSNorm：作用在最后一维 head_dim 上。Qwen3 q_norm/k_norm 同款。
        # 不 norm V（与 Qwen3 一致），V 只是"被加权求和的内容"，不参与点积匹配。
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        lang_kv_is_projected: bool = False,
    ) -> torch.Tensor:
        """Joint attention：vision Q 同时看 vision K/V 与 language K/V。

        - lang_kv：``(K, V)``，形状 ``[B 或 1, n_heads=8, S, head_dim=128]``；
          与 DiT (n_heads, head_dim) 已经天然同形，无需投影。
        - lang_kv_is_projected：v2 起这个参数**总是 True 语义**——language KV 永远
          在 DiT 自己的 (n_heads, head_dim) 子空间。保留这个参数名只是为了让外层
          DiTMoTBlock 接口最小变动；实际不再有 "is_projected vs not" 的分支。

        与普通 cross-attention 的核心区别：cross-attn 是两次 attention（一次
        self、一次 cross），joint-attn 是**一次** attention，K/V 沿 token 维拼起来。
        这样 vision token 可以在同一个 softmax 内同时挑选"参考自己邻域"还是
        "对齐语言上下文"，更接近 AutoMoT 里"快慢 MoT"原始设计。
        """

        # lang_kv_is_projected 形参保留向后兼容；v2 起 lang_kv 一定已经在 DiT 子空间。
        del lang_kv_is_projected

        b, n_v, _ = vision_tokens.shape
        # vision 三个线性投影 + 切头：用法跟普通 transformer 一致。
        q = self.q_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)

        # head-wise RMSNorm：作用在最后一维 head_dim 上，把 vision Q/K 拉到单位 RMS，
        # 跟 Qwen 已 k_norm 过的 language K 在尺度上对齐。V 不 norm。
        q = self.q_norm(q)
        k_v = self.k_norm(k_v)

        # 语言侧：直接取出（无投影、无 norm）。Qwen K 已经 k_norm 过；DiT 这边再 norm
        # 会重复归一化，反而破坏 Qwen 学到的尺度。
        k_l, v_l = lang_kv
        if k_l.shape[0] != b:
            if k_l.shape[0] == 1:
                # qwen_kv.teacher_forced_prefill 通常以 batch=1 跑 Qwen 预填充；DiT 训练
                # 时可能 batch>1。expand 不拷贝内存，attention 内部按 batch 维只读取，
                # 不会写回，共享语言 KV 是安全的。
                k_l = k_l.expand(b, -1, -1, -1)
                v_l = v_l.expand(b, -1, -1, -1)
            else:
                # batch 既不等也不是 1：上游 dataloader 拼 batch 时没保持"一条样本一份 prefill"
                # 约定，立刻报错而不是错位 broadcast。
                raise ValueError(
                    f"language KV batch {k_l.shape[0]} 不等于 vision batch {b}"
                )

        # 形状一致性最后一道校验：language KV 必须已经在 DiT (n_heads, head_dim) 子空间。
        # 走到这里如果不一致，多半是 qwen_kv.py 或 dataloader 出了维度问题，立刻报错可读性更好。
        if k_l.shape[1] != self.n_heads or k_l.shape[3] != self.head_dim:
            raise ValueError(
                f"language K 形状 {tuple(k_l.shape)} 与 DiT (n_heads={self.n_heads}, "
                f"head_dim={self.head_dim}) 不对齐；v2 起需要严格相同。检查 qwen_kv 输出。"
            )

        # 沿 token 维拼接：[B, H, N_v, D] + [B, H, N_l, D] -> [B, H, N_v + N_l, D]。
        # Q 仍只有 vision token，所以 attention 输出 token 数 = N_v，意味着这一层
        # **只更新 vision 流**；language 部分作为冻结 memory，不在本 block 内被修改。
        k_cat = torch.cat([k_v, k_l], dim=2)
        v_cat = torch.cat([v_v, v_l], dim=2)

        # 用 PyTorch 内置 SDPA，性能更稳；在新版本会自动选 flash-attn / mem-efficient。
        # 显式偏好 flash 后端在 train.py 全局 sdpa_kernel 上下文里设置，本层只调标准接口。
        attn = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=0.0)
        # 把 head 维合回去：[B, H, N_v, D] -> [B, N_v, H*D] -> Linear hidden。
        out = attn.transpose(1, 2).contiguous().view(b, n_v, self.n_heads * self.head_dim)
        return self.out_proj(out)


class DiTMoTBlock(nn.Module):
    """单个 DiT-MoT block：RMSNorm + AdaLN -> JointAttention -> RMSNorm + AdaLN -> SwiGLU，全部带 gate。"""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
        cond_dim: int,
    ):
        super().__init__()
        # norm1 / norm2 不带 affine：scale/shift 由 AdaLN modulation 提供，
        # 否则 norm 的 weight 会和 AdaLN 的 scale 互相打架。
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = JointAttention(hidden_dim, n_heads)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        # SwiGLU 替换旧版 GELU MLP；参数量上涨 1.5×，但与 Qwen3 / SD3 / Flux 一致。
        self.mlp = SwiGLU(hidden_dim, mlp_hidden)
        self.modulation = AdaLNModulation(hidden_dim, cond_dim)

    @staticmethod
    def _apply_mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        cond: torch.Tensor,
        lang_kv_is_projected: bool = False,
    ) -> torch.Tensor:
        """单 block forward：RMSNorm+AdaLN-Zero -> JointAttn -> RMSNorm+AdaLN-Zero -> SwiGLU。

        cond 由外部一次性算好（来自 timestep MLP），所有 block 共用同一个 cond，但每个
        block 内部的 AdaLN modulation 矩阵不共享 -> 每层有自己的 shift/scale/gate。

        ``lang_kv_is_projected`` v2 起总是隐含 True，保留参数仅为兼容旧接口；底层
        JointAttention 也不再依赖它分支。
        """

        # 一次 Linear 出 6 个调制向量（attn 3 + mlp 3）。
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(cond)

        # ---- attention 子层 ----
        # norm1 是无 affine 的 RMSNorm；shift/scale 由 cond 提供，相当于每层独立可学的 affine。
        h = self._apply_mod(self.norm1(vision_tokens), shift_a, scale_a)
        h = self.attn(h, lang_kv, lang_kv_is_projected=lang_kv_is_projected)
        # gate_a 初始为 0（AdaLNModulation.zero_init），让 attention 在训练开始时不
        # 改变 vision_tokens；这是 DiT/MMDiT 标配，对 flow matching 非常友好。
        vision_tokens = vision_tokens + gate_a.unsqueeze(1) * h

        # ---- MLP 子层 ----
        h = self._apply_mod(self.norm2(vision_tokens), shift_m, scale_m)
        h = self.mlp(h)
        vision_tokens = vision_tokens + gate_m.unsqueeze(1) * h
        return vision_tokens


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #


@dataclass
class DiTMoTConfig:
    """DiT-MoT 默认配置；变更默认值时同时更新 PROJECT_CONTEXT.md §15。

    v2 默认值（2026-06 切换）：
    - hidden_dim=1024, n_heads=8, head_dim=128 -> 直接对齐 Qwen K/V (8, 128)
    - patch_size=4：视觉 token 数缩到 v1 的 1/4
    - MLP 走 SwiGLU、所有 norm 走 RMSNorm（不再有 language_kv_input_dim 字段）
    """

    latent_channels: int = 4
    patch_size: int = 4
    hidden_dim: int = 1024
    n_heads: int = 8
    mlp_ratio: float = 4.0
    num_layers: int = 12
    cond_dim: int = 256
    max_grid_h: int = 32
    max_grid_w: int = 96
    max_history_frames: int = 8


class DiTMoT(nn.Module):
    """完整 DiT-MoT 主干（v2 架构）。

    forward 输入：
      - z_t   : [B, C, H, W]，含噪声的目标 latent
      - z_history : [B, F, C, H, W]，历史 VAE latent（旧 -> 新）
      - t     : [B]，flow matching 时间步 ∈ [0,1]
      - pooled_kv : 长度 = num_layers 的列表，元素为 (K_seg, V_seg)；每个张量形状
                    [B, n_heads, S, head_dim] = [B, 8, S, 128]（v2 起必须严格匹配）
    输出：v_pred : [B, C, H, W]，velocity 预测，对应 z_t 的 patch grid。
    """

    def __init__(self, cfg: DiTMoTConfig):
        super().__init__()
        self.cfg = cfg

        # 共享 patchify：z_t（含噪目标）和 z_history（干净历史）走同一组 Conv2d 投影。
        # 旧版本使用两个独立 Patchify 试图"硬区分噪声/干净"，但 SD3 / Flux / Sora /
        # PixArt 都走共享路径，让两类 latent token 落在同一线性子空间，再用
        # type_embed + frame_embed + timestep cond 去区分。t=1 时 z_t = z1（与 z_history
        # 同分布），共享投影后注意力的 prior 更自然；训练初期收敛也更稳，且省一半参数。
        self.patch = Patchify(cfg.latent_channels, cfg.hidden_dim, cfg.patch_size)

        # type_embed[0]=noisy target, [1]=history anchor。
        # 改为 normal 初始化（与 frame_embed 同步）：合并 patchify 后，type_embed 是
        # 区分 z_t / z_history 的主要载体；零初始化会让训练开始时"两类 token 完全相同"，
        # 只能靠 frame_embed（仅 history 有）+ timestep cond 间接区分，反而拖慢收敛。
        # std=0.02 与 BERT / DiT 位置 embed 的常用值一致。
        self.type_embed = nn.Parameter(torch.zeros(2, cfg.hidden_dim))
        nn.init.normal_(self.type_embed, mean=0.0, std=0.02)
        # frame_embed 必须随机初始化：history 多帧走同一个 patchify 卷出来，仅靠卷积权重
        # 无法区分"第 0 帧 vs 第 3 帧"。std=0.02 与 BERT / DiT 位置 embed 的常用值一致。
        self.frame_embed = nn.Parameter(torch.zeros(cfg.max_history_frames, cfg.hidden_dim))
        nn.init.normal_(self.frame_embed, mean=0.0, std=0.02)

        # CFG 训练用的 null language KV：每层独立，直接存于 DiT (n_heads, head_dim) 子空间，
        # 跟 v1 一样、形状跟着新 (8, 128) 走。s_null=1：null 只贡献一个 token，
        # attention softmax 里几乎免费；这是 SD3 / Flux 的标准实践。
        # 用"长度 S~2300 的 zero KV"代替会让 softmax 大量取平均，反而压低有效信号。
        # 零初始化：训练开始时 uncond 路径等价于"忽略语言"，配合 AdaLN-Zero gate=0，
        # 模型在前若干步几乎不被 CFG 干扰；几千步后会逐渐学到非平凡的 null embedding。
        head_dim_dit = cfg.hidden_dim // cfg.n_heads
        self.null_lang_k = nn.ParameterList([
            nn.Parameter(torch.zeros(1, cfg.n_heads, 1, head_dim_dit))
            for _ in range(cfg.num_layers)
        ])
        self.null_lang_v = nn.ParameterList([
            nn.Parameter(torch.zeros(1, cfg.n_heads, 1, head_dim_dit))
            for _ in range(cfg.num_layers)
        ])

        # 2D 位置编码：用预生成最大尺寸表，按实际 grid 切片即可，避免重复计算。
        # persistent=False：不写进 state_dict，节省 ckpt 体积——pos_embed 完全由配置决定，
        # 重新加载时 __init__ 会重算，存进 ckpt 反而冗余且容易在分辨率变更时不匹配。
        pe = _make_2d_sincos_pos_embed(cfg.hidden_dim, cfg.max_grid_h, cfg.max_grid_w)
        self.register_buffer("pos_embed_table", pe, persistent=False)

        # 时间步 conditioning：sin/cos embed(t) → MLP → cond_dim 向量。
        # cond_dim*4 是 DiT / DDPM 常用的 4× 扩展宽度，SiLU 是 diffusion 社区默认激活；
        # 用 ReLU 会让小 t 区域梯度死亡，cond signal 在 warmup 阶段拉不起来。
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim * 4, cfg.cond_dim),
        )

        self.blocks = nn.ModuleList([
            DiTMoTBlock(
                hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads,
                mlp_ratio=cfg.mlp_ratio,
                cond_dim=cfg.cond_dim,
            )
            for _ in range(cfg.num_layers)
        ])

        self.final_norm = RMSNorm(cfg.hidden_dim, eps=1e-6, elementwise_affine=False)
        self.final_mod = AdaLNModulation(cfg.hidden_dim, cfg.cond_dim)
        self.unpatch = Unpatchify(cfg.hidden_dim, cfg.latent_channels, cfg.patch_size)

        # gradient checkpointing 开关：train.py 通过 enable_gradient_checkpointing()
        # 切换。默认 False 保持 forward 路径稳定，开了之后每个 block 用 checkpoint 包裹。
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """启用 / 关闭 per-block gradient checkpointing。

        train.py 在 build_dit 之后按 CLI flag 调一次。打开后每个 DiTMoTBlock 的
        forward 用 torch.utils.checkpoint 包裹，反向传播时再算一次前向；显存省 ~40%，
        wall-clock per step 多 ~30%。

        注意：use_reentrant=False 是新版 PyTorch 推荐，兼容 DDP 和 torch.compile。
        老接口 reentrant=True 在 DDP 下会和 find_unused_parameters 互掐。
        """

        self.gradient_checkpointing = bool(enabled)

    def _pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        if gh > self.cfg.max_grid_h or gw > self.cfg.max_grid_w:
            raise ValueError(
                f"grid ({gh},{gw}) 超过预设上限 ({self.cfg.max_grid_h},{self.cfg.max_grid_w})"
            )
        pe = self.pos_embed_table.view(self.cfg.max_grid_h, self.cfg.max_grid_w, -1)
        return pe[:gh, :gw, :].reshape(gh * gw, -1)

    def _build_cond(self, t: torch.Tensor) -> torch.Tensor:
        # _timestep_embedding 内部强制走 float32 算 sin/cos 保精度；但 t_mlp 的权重
        # 在 bf16/fp16 训练时已经是低精度，直接喂 fp32 输入会 mat1/mat2 dtype mismatch。
        # 这里在 Linear 之前对齐 dtype，与 DiT 权重保持一致。
        t_emb = _timestep_embedding(t, self.cfg.cond_dim)
        t_emb = t_emb.to(dtype=self.t_mlp[0].weight.dtype)
        return self.t_mlp(t_emb)

    def load_patch_unpatch(self, path: str, freeze: bool = True) -> dict:
        """从 ``AutoMoT/vae_standalone/train_patch_unpatch.py`` 训出的 safetensors 加载
        patch + unpatch 权重。

        - ``path`` 指向 ``patch_unpatch_*.safetensors``；它的 state_dict key 是
          ``patch.proj.weight / patch.proj.bias / unpatch.proj.weight / unpatch.proj.bias``，
          与本模块内 ``self.patch`` / ``self.unpatch`` 完全一致，直接 load_state_dict
          即可，无需 rename。
        - **v2 注意**：v1 的 safetensors（hidden=768 / patch=2）与 v2（hidden=1024 / patch=4）
          形状不兼容，无法直接加载；必须用 ``train_patch_unpatch.py`` 的新默认值重训。
        - ``freeze=True``（默认）：加载后把 patch / unpatch 切到 eval、关掉 grad；
          train 的 optimizer 只收 ``requires_grad=True`` 的参数，所以这条路径下
          它们不会被更新。
        - ``freeze=False``：仅加载初值，仍按可训练参数对待——主要给"先暖启再联合
          微调"的实验留口子，常规用法不需要。

        返回一个 dict，列出加载的 key 与 missing/unexpected，便于训练日志打印。
        """

        from safetensors.torch import load_file  # noqa: E402

        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"patch/unpatch 权重不存在: {p}")
        sd = load_file(str(p))

        expected = {
            "patch.proj.weight",
            "patch.proj.bias",
            "unpatch.proj.weight",
            "unpatch.proj.bias",
        }
        pick = {k: v for k, v in sd.items() if k in expected}
        missing = sorted(expected - set(pick.keys()))
        unexpected = sorted(set(sd.keys()) - expected)

        if missing:
            # 缺 key 一定是格式错配——例如把 DiT 全量 ckpt 当 patch_unpatch 加载，
            # 直接抛错而不是 strict=False 静默吞掉，避免 DiT 拿到错误初始化继续训。
            raise ValueError(
                f"patch/unpatch 权重缺少必要 key: {missing}；实际文件键: {sorted(sd.keys())}"
            )

        # 形状校验：用本模块当前的 state_dict 对比每个 key 的 shape；不一致说明
        # 训练 patch/unpatch 时的 hidden_dim / patch_size / latent_channels 与
        # DiTMoTConfig 不匹配，必须立刻报错。v1→v2 切换时这里会精准命中。
        own_sd = self.state_dict()
        for k, v in pick.items():
            if own_sd[k].shape != v.shape:
                raise ValueError(
                    f"patch/unpatch 权重 {k} shape 不匹配："
                    f"DiT 期望 {tuple(own_sd[k].shape)}，文件 {tuple(v.shape)}。"
                    "v1 ckpt（hidden=768/patch=2）与 v2（hidden=1024/patch=4）不兼容，需重训。"
                )

        # strict=False：本模块还有其它 key（blocks/null_lang_*/pos_embed_table 等）
        # 不能要求文件里有；这里 pick 只含 patch/unpatch 的 4 个 key，等价于只覆盖这部分。
        self.load_state_dict(pick, strict=False)

        if freeze:
            for pp in self.patch.parameters():
                pp.requires_grad_(False)
            for pp in self.unpatch.parameters():
                pp.requires_grad_(False)
            self.patch.eval()
            self.unpatch.eval()

        return {
            "loaded_keys": sorted(pick.keys()),
            "missing": missing,
            "unexpected": unexpected,
            "frozen": freeze,
        }

    def forward(
        self,
        z_t: torch.Tensor,
        z_history: torch.Tensor,
        t: torch.Tensor,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """单步前向。

        输入约定：
        - z_t：[B, C, H, W]，noisy 目标 latent。
        - z_history：[B, F, C, H, W] 或旧接口 [B, C, H, W]。F 从旧到新排列，
          最后一帧就是当前 anchor；DiT 会直接看历史视觉 latent。
        - t：[B]，flow matching 时间步 ∈ [0,1]。
        - pooled_kv：长度必须 == num_layers；每段是 (K, V)，形状
          ``[B 或 1, n_heads=8, S, head_dim=128]``，dtype 通常和 DiT 自身一致。
          **v2 要求严格 (n_heads, head_dim) 匹配**，否则在 JointAttention 内会抛错。
        - force_uncond：True 表示用 DiT 自身的 null_lang_k/v 代替 pooled_kv
          走 uncond 路径（CFG 训练 / 引导推理用）。pooled_kv 此时仍需传入，
          只是不被使用——保留位置以保持接口稳定。

        输出：v_pred 与 z_t 同形状，对应 velocity 预测。
        """
        # z_t  torch.Size([1, 4, 48, 144])
        # z_history  torch.Size([1, 4, 4, 48, 144])
        # t  torch.Size([1])

        # 这一步检查在第一次 forward 时定位"runner / 训练脚本里把 num_layers 改了
        # 一边没改另一边"的低级错误，比让 attention 中段越界报错可读得多。
        if len(pooled_kv) != self.cfg.num_layers:
            raise ValueError(
                f"pooled_kv 段数 {len(pooled_kv)} 与 DiT 层数 {self.cfg.num_layers} 不一致"
            )

        # v2 严格性校验：仅用 pooled_kv[0] 抽样检测，避免每层都查浪费 CPU。
        # 不在 JointAttention 内做这步是为了在最早期就抓错，错误信息更直接。
        k0, _ = pooled_kv[0]
        if not force_uncond and (k0.shape[1] != self.cfg.n_heads or k0.shape[3] != (self.cfg.hidden_dim // self.cfg.n_heads)):
            raise ValueError(
                f"pooled_kv[0] K 形状 {tuple(k0.shape)} 与 DiT (n_heads={self.cfg.n_heads}, "
                f"head_dim={self.cfg.hidden_dim // self.cfg.n_heads}) 不匹配。"
                "v2 要求严格相同；检查 qwen_kv 输出或换 Qwen 模型时的 num_key_value_heads。"
            )

        if z_history.ndim == 4:
            z_history = z_history.unsqueeze(1)
        if z_history.ndim != 5:
            raise ValueError(
                f"z_history 应为 [B,F,C,H,W] 或 [B,C,H,W]，实际得到 {tuple(z_history.shape)}"
            )
        if z_history.shape[1] > self.cfg.max_history_frames:
            raise ValueError(
                f"历史帧数 {z_history.shape[1]} > max_history_frames={self.cfg.max_history_frames}"
            )

        # 共享 patchify：z_t 与 z_history 走同一组 Conv2d 投影。
        # type_embed + frame_embed + timestep cond 负责区分 noisy / clean / 时序。

        tok_t, grid_t = self.patch(z_t)
        # tok_t  torch.Size([1, 432, 1024])   <- v2 默认 patch=4 时网格 (12, 36)
        # grid_t (12, 36)
        gh, gw = grid_t

        pe = self._pos_embed(gh, gw).to(dtype=tok_t.dtype, device=tok_t.device)

        tok_t = tok_t + pe + self.type_embed[0]

        history_tokens: List[torch.Tensor] = []
        for frame_idx in range(z_history.shape[1]):
            tok_h, grid_h = self.patch(z_history[:, frame_idx])
            if grid_h != grid_t:
                raise ValueError(
                    f"z_t / z_history patch grid 不一致：{grid_t} vs {grid_h}（要求 H/W 相同）"
                )
            frame_pos = self.frame_embed[frame_idx].to(dtype=tok_h.dtype, device=tok_h.device)
            history_tokens.append(tok_h + pe + self.type_embed[1] + frame_pos)

        # 顺序：先放 z_t（输出要切回来），再放历史 latent tokens（旧 -> 新）。
        vision_tokens = torch.cat([tok_t, *history_tokens], dim=1)
        # patch=4 / F=4 时 torch.Size([1, 2160, 1024])

        n_t = tok_t.shape[1]

        # cond 只算一次，所有 block 共享；AdaLN modulation 矩阵是 per-block 的。
        cond = self._build_cond(t)
        # torch.Size([1, 256])

        # 逐层走 block。每层用 pooled_kv[i] 作为冻结语言 memory；
        # force_uncond=True 时改用 DiT 自带的 null_lang_k/v（CFG 路径）。
        for i, (block, lang_kv) in enumerate(zip(self.blocks, pooled_kv)):
            if force_uncond:
                null_kv = (
                    self.null_lang_k[i].to(dtype=vision_tokens.dtype),
                    self.null_lang_v[i].to(dtype=vision_tokens.dtype),
                )
                vision_tokens = self._maybe_checkpoint(block, vision_tokens, null_kv, cond, True)
            else:
                # lang_kv[0]/[1] torch.Size([1, 8, ~2300, 128])
                vision_tokens = self._maybe_checkpoint(block, vision_tokens, lang_kv, cond, False)

        # final AdaLN：DiT 标准结构，AdaLN 输出 6 个调制向量但 final 只用前 2 个
        # （shift/scale）。后 4 个忽略；保留同一个 AdaLNModulation 类是为了减少
        # 实现分歧。
        shift, scale, _g_a, _s_m, _sc_m, _g_m = self.final_mod(cond)
        vision_out = self.final_norm(vision_tokens)
        vision_out = vision_out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 只取 z_t 那段（前 n_t 个 token），unpatchify 回 latent 形状作为 velocity。
        # history tokens 仅用于让 attention 看到过去到现在的视觉上下文，不预测输出。
        out_t = vision_out[:, :n_t, :]
        return self.unpatch(out_t, (gh, gw))

    def _maybe_checkpoint(
        self,
        block: DiTMoTBlock,
        vision_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        cond: torch.Tensor,
        is_uncond: bool,
    ) -> torch.Tensor:
        """统一 block forward 调用点：根据 self.gradient_checkpointing 决定是否包裹 checkpoint。

        放在独立方法里有两个好处：
        1. forward 主干保持线性可读，不被 if 分叉打断。
        2. 未来想接 selective activation checkpointing（部分层走 checkpoint）只需要
           改这里的 if 条件。

        use_reentrant=False：新版 PyTorch 默认行为，对 DDP / torch.compile 友好；
        老接口的 reentrant=True 在 DDP 下需要 find_unused_parameters=True，开销大。
        """

        # gradient_checkpointing 只在训练且 requires_grad 的张量上启用；eval 模式下走纯前向。
        if self.gradient_checkpointing and self.training and vision_tokens.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                block,
                vision_tokens,
                lang_kv,
                cond,
                is_uncond,  # 对应 block.forward 的 lang_kv_is_projected 形参；语义已在 v2 中作废
                use_reentrant=False,
            )
        return block(vision_tokens, lang_kv, cond, lang_kv_is_projected=is_uncond)
