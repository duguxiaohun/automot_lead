"""Prefix-KV 版 LeadMoT gen 路 decoder block。

本实现跟 goalgen v2 的关键约定一致：
- gen hidden = Qwen num_kv_heads * head_dim = 8 * 128 = 1024
- frozen Qwen K/V 直接作为 prefix memory 使用，不再经过任何 Linear
- 每个 block 用一个 Qwen K/V segment，默认 36 层 Qwen -> 12 个 select_last segment

gen token 自己仍需要 q/k/v 投影；语言侧 K/V 完全冻结，只在 token 维拼接进 attention。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """简单 RMSNorm，和 goalgen/DiT 侧保持一致。"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        x_out = (x_fp * rms).to(orig_dtype)
        if self.elementwise_affine:
            x_out = x_out * self.weight
        return x_out


class SwiGLU(nn.Module):
    """Qwen/LLaMA 风格 SwiGLU FFN。"""

    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PrefixKVAttention(nn.Module):
    """gen Q 同时看 gen K/V 与 frozen Qwen prefix K/V。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _check_and_cast_lang_kv(
        self,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(
        self,
        gen_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, n, _ = gen_tokens.shape
        q = self.q_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.v_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k_g = self.k_norm(k_g)
        k_l, v_l = self._check_and_cast_lang_kv(lang_kv, batch_size=b, ref=q)

        k = torch.cat([k_g, k_l], dim=2)
        v = torch.cat([v_g, v_l], dim=2)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size)
        return self.out_proj(out)


class MoTDecoderBlock(nn.Module):
    """一层 prefix-KV gen decoder：RMSNorm -> PrefixKVAttention -> RMSNorm -> SwiGLU。"""

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
