"""Prefix-KV gen 路 decoder block。

frozen Qwen K/V 直接当 prefix concat 进 attention，不过任何 Linear；
gen 段自己走独立 Q/K/V 投影。要求 gen (num_heads, head_dim) = Qwen (num_kv_heads, head_dim)
（默认 8, 128 -> hidden=1024）。

跟 AutoMoT 严格 MoT 的 forward_inference 数学等价，差别只在层数（12 vs 36）
和封装位置（独立 transformer vs Qwen LM 内）。详见 ARCHITECTURE.md §2。

四个类:
  RMSNorm           - Qwen3 风格 RMSNorm
  SwiGLU            - Qwen/LLaMA 风格 FFN
  PrefixKVAttention - prefix K/V + gen Q/K/V，含 q/k_norm
  MoTDecoderBlock   - 上面三个的 pre-norm 组装
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PrefixKVAttention(nn.Module):
    """gen Q 同时看 gen K/V 与 frozen Qwen prefix K/V 的 attention。

    流程:
        Q_gen = q_proj(gen) -> q_norm   shape (B, n_heads, L_gen, head_dim)
        K_gen = k_proj(gen) -> k_norm
        V_gen = v_proj(gen)
        K = concat(K_gen, K_lang_prefix, dim=token)   # K_lang 不投影
        V = concat(V_gen, V_lang_prefix, dim=token)
        out = SDPA(Q_gen, K, V) -> out_proj

    硬约束：gen (num_heads, head_dim) 必须 = lang_kv 的 (heads, head_dim)，
    否则 reshape 会失败（hidden_size 必须能被 num_heads 整除）。
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)

        # gen 路投影，bias=False (Qwen3 风格)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # head_dim 上 RMSNorm，与 Qwen3 q_norm/k_norm 同款
        # lang K 已经在 frozen prefill 时 norm 过，这里只 norm gen 自己的 Q/K
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
        # batch=1 时广播到 batch_size（expand 不拷贝）
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

    def forward(
        self,
        gen_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, n, _ = gen_tokens.shape
        # Q/K/V 投影 + reshape 成 (B, n_heads, L_gen, head_dim)
        q = self.q_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.v_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # Qwen3 风格 head_dim RMSNorm (只 norm gen 自己)
        q = self.q_norm(q)
        k_g = self.k_norm(k_g)

        k_l, v_l = self._check_and_cast_lang_kv(lang_kv, batch_size=b, ref=q)

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
    """RMSNorm -> PrefixKVAttention -> +residual -> RMSNorm -> SwiGLU -> +residual。

    标准 pre-norm Transformer block，结构与 Qwen3 LM 层一致。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.attn = PrefixKVAttention(hidden_size, num_heads, dropout=dropout)
        self.norm_ffn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gen_seq: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h = self.norm_attn(gen_seq)
        gen_seq = gen_seq + self.dropout(self.attn(h, lang_kv))
        h = self.norm_ffn(gen_seq)
        gen_seq = gen_seq + self.dropout(self.ffn(h))
        return gen_seq
