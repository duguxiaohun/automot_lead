"""Prefix-KV gen 路 decoder block。

frozen Qwen K/V 直接当 prefix concat 进 attention，不过任何 Linear；
gen 段自己走独立 Q/K/V 投影 + 可选 RoPE。要求 gen (num_heads, head_dim)
= Qwen (num_kv_heads, head_dim)（默认 8, 128 -> hidden=1024）。

RoPE 支持两种 freq allocation（参考 JJJYmmm/Multimodal-RoPEs）：
- mrope  : Qwen3-VL 标准 M-RoPE，head_dim//2 切 3 段分别用 t/h/w 旋转
- mhrope : head 维分配 3 轴；不同 head 用不同 axis，剩余 head pad 零不旋转

跟 AutoMoT 严格 MoT 的 forward_inference 数学等价（差别只在层数 12 vs 36
和封装位置）。详见 ARCHITECTURE.md §2。

四个类:
  RMSNorm           - Qwen3 风格 RMSNorm
  SwiGLU            - Qwen/LLaMA 风格 FFN
  PrefixKVAttention - prefix K/V + gen Q/K/V，含 q/k_norm + RoPE
  MoTDecoderBlock   - pre-norm 组装
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RoPE 工具
# ============================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """LLaMA/Qwen RoPE 的半维旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _inv_freq(head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """RoPE 频率：inv_freq[i] = 1 / theta ** (2i / head_dim)，长度 head_dim/2。"""
    return 1.0 / (
        float(theta)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )


def _build_freq_3d(
    position_ids_3d: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """
    position_ids_3d: (3, B, L)
    inv_freq:        (head_dim//2,)
    return:          (3, B, L, head_dim//2)
    """
    pos_f = position_ids_3d.to(device=inv_freq.device, dtype=torch.float32)
    return torch.einsum("abl,d->abld", pos_f, inv_freq)


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids_3d: torch.Tensor,
    rope_theta: float,
    mrope_section: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL 标准 M-RoPE。

    head_dim//2 沿最后一维切 (s0, s1, s2)，3 段分别用 (t, h, w) 三轴 position 旋转。
    所有 head 共享同一份 (cos, sin)（broadcast）。

    Args:
        q, k: (B, H, L, head_dim)
        position_ids_3d: (3, B, L) 三轴 position（t, h, w）
        rope_theta: RoPE 基频
        mrope_section: (s0, s1, s2)，sum 必须 == head_dim//2

    Returns:
        (q_rot, k_rot)，与输入同 shape/dtype
    """
    if q.shape != k.shape:
        raise ValueError(f"q/k 必须同 shape，got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.ndim != 4:
        raise ValueError(f"q 应为 (B,H,L,D)，got {tuple(q.shape)}")
    head_dim = q.shape[-1]
    half = head_dim // 2
    if sum(mrope_section) != half:
        raise ValueError(
            f"M-RoPE: sum(mrope_section)={sum(mrope_section)} != head_dim//2={half}"
        )

    inv_freq = _inv_freq(head_dim, rope_theta, q.device)
    freqs = _build_freq_3d(position_ids_3d, inv_freq)   # (3, B, L, half)

    # 按 JJJYmmm MRoPE：split mrope_section，第 i 段取轴 (i % 3) 的 freq
    freq_parts = freqs.split(list(mrope_section), dim=-1)
    freq_combined = torch.cat(
        [m[i % 3] for i, m in enumerate(freq_parts)],
        dim=-1,
    )  # (B, L, half)

    # 拼回 head_dim：LLaMA RoPE 风格 [freq | freq]
    emb = torch.cat([freq_combined, freq_combined], dim=-1)   # (B, L, head_dim)
    # broadcast 到所有 head
    cos = emb.cos().unsqueeze(1).to(q.dtype)   # (B, 1, L, head_dim)
    sin = emb.sin().unsqueeze(1).to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_mhrope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids_3d: torch.Tensor,
    rope_theta: float,
    mrope_section: Tuple[int, int, int],
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-Head RoPE：head 维分配 3 轴。

    不同 head 用不同 axis：mrope_section[i] 个 head 用第 i 个 axis 的 position。
    剩余 (num_heads - sum) 个 head 不旋转（emb=0 ⇒ cos=1 sin=0，q 不变）。

    Args:
        q, k: (B, H, L, head_dim)
        position_ids_3d: (3, B, L)
        rope_theta: RoPE 基频
        mrope_section: (n0, n1, n2) 每段 head 数，sum 必须 <= num_heads
        num_heads: 总 head 数

    Returns:
        (q_rot, k_rot)
    """
    if q.shape != k.shape:
        raise ValueError(f"q/k 必须同 shape，got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.ndim != 4:
        raise ValueError(f"q 应为 (B,H,L,D)，got {tuple(q.shape)}")
    head_dim = q.shape[-1]
    half = head_dim // 2
    if sum(mrope_section) > num_heads:
        raise ValueError(
            f"MH-RoPE: sum(mrope_section)={sum(mrope_section)} > num_heads={num_heads}"
        )

    b, _, n, _ = q.shape
    inv_freq = _inv_freq(head_dim, rope_theta, q.device)
    freqs = _build_freq_3d(position_ids_3d, inv_freq)   # (3, B, L, half)

    # head-wise allocation：每段 num 个 head 共享同一 axis 的 freq
    parts = []
    for axis_idx, num in enumerate(mrope_section):
        # freqs[axis_idx] (B, L, half) -> (B, num, L, half)
        parts.append(freqs[axis_idx].unsqueeze(1).expand(b, num, n, half))
    freq_combined = torch.cat(parts, dim=1)   # (B, sum(mrope_section), L, half)

    # 不旋转的 head pad 零（emb=0 退化为恒等）
    unrotated = num_heads - sum(mrope_section)
    if unrotated > 0:
        pad = torch.zeros(
            b, unrotated, n, half,
            device=freq_combined.device, dtype=freq_combined.dtype,
        )
        freq_combined = torch.cat([freq_combined, pad], dim=1)

    emb = torch.cat([freq_combined, freq_combined], dim=-1)   # (B, H, L, head_dim)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _build_gen_position_ids_3d(
    rope_position_offset,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """从 scalar/tensor offset 展开成 (3, B, L) 三轴 position。

    LeadMoT gen token 都属于"接在 Qwen prefill 末尾继续生成的新文本 token"，
    在 Qwen3-VL M-RoPE 体系下三轴 position 全等（temporal=height=width=t）。
    所以只需 1 个 scalar offset 就能展开成 3D position。
    """
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    if isinstance(rope_position_offset, torch.Tensor):
        offset = rope_position_offset.to(device=device, dtype=torch.long).reshape(-1)
        if offset.numel() == 1:
            pos = base.unsqueeze(0).expand(batch_size, -1) + offset[0]
        elif offset.numel() == batch_size:
            pos = base.unsqueeze(0) + offset.view(batch_size, 1)
        else:
            raise ValueError(
                f"rope_position_offset tensor 必须 numel=1 或 B，实际 {offset.numel()}"
            )
    else:
        start = int(rope_position_offset)
        pos = (base + start).unsqueeze(0).expand(batch_size, -1)
    # (B, L) -> (3, B, L)，三轴全等
    return pos.unsqueeze(0).expand(3, -1, -1)


# ============================================================
# 基础组件
# ============================================================
class RMSNorm(nn.Module):
    """RMSNorm：y = x / sqrt(mean(x^2) + eps) * gamma，与 Qwen3 / LLaMA 一致。"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bf16 下 x^2 易溢出，先升 fp32 算 RMS 再降回原 dtype
        orig_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        x_out = (x_fp * rms).to(orig_dtype)
        if self.elementwise_affine:
            x_out = x_out * self.weight
        return x_out


class SwiGLU(nn.Module):
    """Qwen/LLaMA 风格 SwiGLU：down(silu(gate(x)) * up(x))，三个 Linear 都不带 bias。"""

    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Attention
# ============================================================
class PrefixKVAttention(nn.Module):
    """gen Q 同时看 gen K/V 与 frozen Qwen prefix K/V 的 attention。

    流程:
        Q_gen = q_proj(gen) -> q_norm -> RoPE(rope_type)
        K_gen = k_proj(gen) -> k_norm -> RoPE(rope_type)
        V_gen = v_proj(gen)
        K = concat(K_gen, K_lang_prefix, dim=token)   # K_lang 不投影、不再 RoPE
        V = concat(V_gen, V_lang_prefix, dim=token)
        out = SDPA(Q_gen, K, V) -> out_proj

    硬约束：gen (num_heads, head_dim) 必须 = lang_kv 的 (heads, head_dim)。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 5000000.0,
        use_rope: bool = True,
        rope_type: str = "mrope",
        mrope_section: Tuple[int, int, int] = (16, 24, 24),
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.use_rope = bool(use_rope)
        self.rope_type = rope_type
        self.mrope_section = tuple(mrope_section)

        # gen 路投影，bias=False (Qwen3 风格)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # head_dim 上 RMSNorm（Qwen3 q_norm/k_norm 同款）
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _check_and_cast_lang_kv(
        self,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """形状校验 + 必要时 batch 广播 + device/dtype 对齐到 ref。"""
        k_l, v_l = lang_kv
        if k_l.ndim != 4 or v_l.ndim != 4:
            raise ValueError(
                f"language KV 应为 [B,H,S,D]，实际 K={tuple(k_l.shape)} V={tuple(v_l.shape)}"
            )
        if k_l.shape != v_l.shape:
            raise ValueError(f"language K/V shape 不一致：K={tuple(k_l.shape)} V={tuple(v_l.shape)}")
        if k_l.shape[1] != self.num_heads or k_l.shape[3] != self.head_dim:
            raise ValueError(
                f"language KV {tuple(k_l.shape)} 与 gen (heads={self.num_heads}, "
                f"head_dim={self.head_dim}) 不匹配"
            )
        if k_l.shape[0] != batch_size:
            if k_l.shape[0] == 1:
                k_l = k_l.expand(batch_size, -1, -1, -1)
                v_l = v_l.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"language KV batch {k_l.shape[0]} != gen batch {batch_size}")
        return (
            k_l.to(device=ref.device, dtype=ref.dtype),
            v_l.to(device=ref.device, dtype=ref.dtype),
        )

    def _apply_rope(
        self,
        q: torch.Tensor,
        k_g: torch.Tensor,
        rope_position_offset,
        prefix_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 rope_type 选 mrope / mhrope。offset=None 时回退到 prefix cache 长度。"""
        b, _, n, _ = q.shape
        if rope_position_offset is None:
            rope_position_offset = prefix_seq_len
        position_ids_3d = _build_gen_position_ids_3d(
            rope_position_offset, seq_len=n, batch_size=b, device=q.device,
        )
        if self.rope_type == "mrope":
            return apply_mrope(q, k_g, position_ids_3d, self.rope_theta, self.mrope_section)
        if self.rope_type == "mhrope":
            return apply_mhrope(
                q, k_g, position_ids_3d, self.rope_theta, self.mrope_section, self.num_heads,
            )
        raise ValueError(f"未知 rope_type={self.rope_type!r}")

    def forward(
        self,
        gen_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        rope_position_offset: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = gen_tokens.shape
        q = self.q_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.v_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Qwen3 风格 head_dim RMSNorm (只 norm gen 自己；lang K 已在 prefill 时 norm)
        q = self.q_norm(q)
        k_g = self.k_norm(k_g)

        k_l, v_l = self._check_and_cast_lang_kv(lang_kv, batch_size=b, ref=q)

        if self.use_rope:
            q, k_g = self._apply_rope(q, k_g, rope_position_offset, prefix_seq_len=int(k_l.shape[2]))

        # 沿 token 维 (dim=2) 拼接，gen 在前 lang 在后（与 goalgen DiT 一致）
        k = torch.cat([k_g, k_l], dim=2)
        v = torch.cat([v_g, v_l], dim=2)

        # PyTorch 2.0+ SDPA 自动选 FlashAttention backend
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # (B, n_heads, L_gen, head_dim) -> (B, L_gen, hidden)
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size)
        return self.out_proj(out)


class MoTDecoderBlock(nn.Module):
    """RMSNorm -> PrefixKVAttention -> +residual -> RMSNorm -> SwiGLU -> +residual。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
        rope_theta: float = 5000000.0,
        use_rope: bool = True,
        rope_type: str = "mrope",
        mrope_section: Tuple[int, int, int] = (16, 24, 24),
    ):
        super().__init__()
        self.norm_attn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.attn = PrefixKVAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            use_rope=use_rope,
            rope_type=rope_type,
            mrope_section=mrope_section,
        )
        self.norm_ffn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gen_seq: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        rope_position_offset: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_attn(gen_seq)
        gen_seq = gen_seq + self.dropout(
            self.attn(h, lang_kv, rope_position_offset=rope_position_offset)
        )
        h = self.norm_ffn(gen_seq)
        gen_seq = gen_seq + self.dropout(self.ffn(h))
        return gen_seq
