"""带 frozen Qwen prefix K/V attention 的 decoder block。

LeadMoT generated token 会自己生成 Q/K/V projection。frozen Qwen prefix K/V
会直接拼进 attention，不重新投影、不重新 RoPE，因为 Qwen prefill 已经
应用过自己的 RoPE/M-RoPE。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """执行 LLaMA/Qwen RoPE 使用的半维旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _inv_freq(head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """构造长度为 ``head_dim // 2`` 的 RoPE inverse frequency。"""
    return 1.0 / (
        float(theta)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )


def _build_freq_3d(position_ids_3d: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """把三轴 position id 展开成 ``(3, B, L, head_dim//2)`` frequency。"""
    pos_f = position_ids_3d.to(device=inv_freq.device, dtype=torch.float32)
    return torch.einsum("abl,d->abld", pos_f, inv_freq)


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids_3d: torch.Tensor,
    rope_theta: float,
    mrope_section: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """给 generated Q/K 应用 Qwen 风格 M-RoPE。

    ``mrope_section`` 把 ``head_dim//2`` 切到 temporal/height/width 三轴。
    所有 attention head 共用同一个组合后的 frequency tensor。
    """
    if q.shape != k.shape:
        raise ValueError(f"q/k must have the same shape, got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.ndim != 4:
        raise ValueError(f"q must be (B,H,L,D), got {tuple(q.shape)}")
    head_dim = q.shape[-1]
    half = head_dim // 2
    if sum(mrope_section) != half:
        raise ValueError(f"M-RoPE section sum {sum(mrope_section)} != head_dim//2={half}")

    inv_freq = _inv_freq(head_dim, rope_theta, q.device)
    freqs = _build_freq_3d(position_ids_3d, inv_freq)

    # 切开最后一个 frequency 维度，并按 t/h/w 轴循环取值；
    # 这和 Qwen3-VL M-RoPE 的 section packing 一致。
    freq_parts = freqs.split(list(mrope_section), dim=-1)
    freq_combined = torch.cat([m[i % 3] for i, m in enumerate(freq_parts)], dim=-1)

    emb = torch.cat([freq_combined, freq_combined], dim=-1)
    cos = emb.cos().unsqueeze(1).to(q.dtype)
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
    """给 generated Q/K 应用 head-wise multi-axis RoPE。

    ``mrope_section`` 表示分给 temporal/height/width 三轴的 head 数。
    如有剩余 head，则用 0 frequency padding 保持不旋转。
    """
    if q.shape != k.shape:
        raise ValueError(f"q/k must have the same shape, got {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.ndim != 4:
        raise ValueError(f"q must be (B,H,L,D), got {tuple(q.shape)}")
    head_dim = q.shape[-1]
    half = head_dim // 2
    if sum(mrope_section) > num_heads:
        raise ValueError(f"MH-RoPE section sum {sum(mrope_section)} > num_heads={num_heads}")

    b, _, n, _ = q.shape
    inv_freq = _inv_freq(head_dim, rope_theta, q.device)
    freqs = _build_freq_3d(position_ids_3d, inv_freq)

    parts = []
    for axis_idx, num in enumerate(mrope_section):
        parts.append(freqs[axis_idx].unsqueeze(1).expand(b, num, n, half))
    freq_combined = torch.cat(parts, dim=1)

    unrotated = num_heads - sum(mrope_section)
    if unrotated > 0:
        pad = torch.zeros(
            b,
            unrotated,
            n,
            half,
            device=freq_combined.device,
            dtype=freq_combined.dtype,
        )
        freq_combined = torch.cat([freq_combined, pad], dim=1)

    emb = torch.cat([freq_combined, freq_combined], dim=-1)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _build_gen_position_ids_3d(
    rope_position_offset,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """为 generated LeadMoT token 构造 ``(3, B, L)`` position id。

    runner 传入 Qwen prefill offset。planning generated token 被视作接在
    prompt 后面的新文本 token，因此 t/h/w 三轴位置相同。
    """
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    if isinstance(rope_position_offset, torch.Tensor):
        offset = rope_position_offset.to(device=device, dtype=torch.long).reshape(-1)
        if offset.numel() == 1:
            pos = base.unsqueeze(0).expand(batch_size, -1) + offset[0]
        elif offset.numel() == batch_size:
            pos = base.unsqueeze(0) + offset.view(batch_size, 1)
        else:
            raise ValueError(f"rope_position_offset tensor must have numel=1 or B, got {offset.numel()}")
    else:
        start = int(rope_position_offset)
        pos = (base + start).unsqueeze(0).expand(batch_size, -1)
    return pos.unsqueeze(0).expand(3, -1, -1)


class RMSNorm(nn.Module):
    """Qwen/LLaMA 风格 RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """用 root-mean-square 统计量归一化最后一维。"""
        # 用 fp32 计算 RMS 保证数值稳定，然后再 cast 回原 dtype。
        orig_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        x_out = (x_fp * rms).to(orig_dtype)
        if self.elementwise_affine:
            x_out = x_out * self.weight
        return x_out


class SwiGLU(nn.Module):
    """Qwen/LLaMA 风格 SwiGLU feed-forward 层。"""

    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 SwiGLU gate/up/down projection。"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PrefixKVAttention(nn.Module):
    """generated token 自注意力，同时 attention 到 frozen Qwen prefix K/V。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 5000000.0,
        rope_type: str = "mrope",
        mrope_section: Tuple[int, int, int] = (16, 24, 24),
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if rope_type not in {"mrope", "mhrope", "none"}:
            raise ValueError(f"rope_type must be mrope/mhrope/none, got {rope_type!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)
        self.rope_type = rope_type
        self.mrope_section = tuple(mrope_section)

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
        """检查 frozen prefix K/V，并对齐到 generated Q 的 dtype/device。"""
        k_l, v_l = lang_kv
        if k_l.ndim != 4 or v_l.ndim != 4:
            raise ValueError(f"language KV must be [B,H,S,D], got K={tuple(k_l.shape)} V={tuple(v_l.shape)}")
        if k_l.shape != v_l.shape:
            raise ValueError(f"language K/V shape mismatch: K={tuple(k_l.shape)} V={tuple(v_l.shape)}")
        if k_l.shape[1] != self.num_heads or k_l.shape[3] != self.head_dim:
            raise ValueError(
                f"language KV {tuple(k_l.shape)} does not match "
                f"(heads={self.num_heads}, head_dim={self.head_dim})"
            )
        if k_l.shape[0] != batch_size:
            if k_l.shape[0] == 1:
                k_l = k_l.expand(batch_size, -1, -1, -1)
                v_l = v_l.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(f"language KV batch {k_l.shape[0]} != gen batch {batch_size}")
        return k_l.to(device=ref.device, dtype=ref.dtype), v_l.to(device=ref.device, dtype=ref.dtype)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k_g: torch.Tensor,
        rope_position_offset,
        prefix_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """只给 generated Q/K 应用当前配置的 RoPE。"""
        b, _, n, _ = q.shape
        if rope_position_offset is None:
            rope_position_offset = prefix_seq_len
        position_ids_3d = _build_gen_position_ids_3d(
            rope_position_offset,
            seq_len=n,
            batch_size=b,
            device=q.device,
        )
        if self.rope_type == "mrope":
            return apply_mrope(q, k_g, position_ids_3d, self.rope_theta, self.mrope_section)
        return apply_mhrope(q, k_g, position_ids_3d, self.rope_theta, self.mrope_section, self.num_heads)

    def forward(
        self,
        gen_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        rope_position_offset: int | torch.Tensor | None = None,
        prefix_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在 generated token 与 frozen prefix K/V 拼接后的序列上跑 attention。

        prefix_key_padding_mask：可选 [B, S_lang] bool，True=有效，False=padding。
        仅在 batched 训练且各 sample 的 Qwen prefill seq_len 不同时需要传入。
        None 时旧 per-sample 路径行为完全不变。
        """
        b, n, _ = gen_tokens.shape
        q = self.q_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.v_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # 对齐 Qwen3 行为：generated Q/K 按 head_dim 做 norm。
        # prefix K 已来自 Qwen，这里不能再额外 norm。
        q = self.q_norm(q)
        k_g = self.k_norm(k_g)

        k_l, v_l = self._check_and_cast_lang_kv(lang_kv, batch_size=b, ref=q)

        if self.rope_type != "none":
            q, k_g = self._apply_rope(q, k_g, rope_position_offset, prefix_seq_len=int(k_l.shape[2]))

        # 沿 token 维拼接：generated self token 在前，frozen language/vision prefix 在后。
        # value 与 key 使用相同顺序。
        k = torch.cat([k_g, k_l], dim=2)
        v = torch.cat([v_g, v_l], dim=2)

        # SDPA 的 attn_mask（仅 batched + padding 真实存在时构造）。bool 语义：
        # True=允许 attend；False=屏蔽。形状 [B, 1, 1, n_gen + S_lang]，SDPA 内部
        # 广播到 [B, H, n_gen, n_gen + S_lang]。gen 段全 True（generated query 永远
        # 可以看自己拼接进来的 gen K/V），lang 段按 padding mask。
        attn_mask = None
        if prefix_key_padding_mask is not None:
            if prefix_key_padding_mask.shape[0] != b:
                if prefix_key_padding_mask.shape[0] == 1:
                    prefix_key_padding_mask = prefix_key_padding_mask.expand(b, -1)
                else:
                    raise ValueError(
                        f"prefix_key_padding_mask batch {prefix_key_padding_mask.shape[0]} != gen batch {b}"
                    )
            if prefix_key_padding_mask.shape[1] != k_l.shape[2]:
                raise ValueError(
                    f"prefix_key_padding_mask seq_len {prefix_key_padding_mask.shape[1]} != language K seq_len {k_l.shape[2]}"
                )
            # 调用方会在批内无 padding 时传 None；这里不做 bool(mask.all())，避免
            # 未来 decoder 编译/图捕获时出现 CUDA tensor -> Python bool 的数据依赖分支。
            gen_mask = torch.ones((b, n), dtype=torch.bool, device=k.device)
            full_key_mask = torch.cat(
                [gen_mask, prefix_key_padding_mask.to(k.device)], dim=1
            )
            attn_mask = full_key_mask[:, None, None, :]

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size)
        return self.out_proj(out)


class MoTDecoderBlock(nn.Module):
    """带 Prefix-KV attention 和 SwiGLU FFN 的 pre-norm transformer block。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
        rope_theta: float = 5000000.0,
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
        prefix_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 attention 与 feed-forward 两段 residual 更新。

        prefix_key_padding_mask 详见 PrefixKVAttention.forward；None 时 per-sample
        行为不变。
        """
        h = self.norm_attn(gen_seq)
        gen_seq = gen_seq + self.dropout(
            self.attn(
                h,
                lang_kv,
                rope_position_offset=rope_position_offset,
                prefix_key_padding_mask=prefix_key_padding_mask,
            )
        )
        h = self.norm_ffn(gen_seq)
        gen_seq = gen_seq + self.dropout(self.ffn(h))
        return gen_seq
