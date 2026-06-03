"""Prefix-KV 版 LeadMoT gen 路 decoder block。

什么是 prefix-KV attention
==========================
传统 cross-attention 是这样的：query (来自 decoder) 和 key/value (来自 encoder hidden)
通过 W_k / W_v 各做一次线性投影，再算 attention。这种做法对 encoder hidden 又做
了一次 K/V 投影，会"重新解释"encoder 的语义。

prefix-KV 是另一种思路（goalgen v2 / AutoMoT 严格 MoT 都用这套）：
- frozen Qwen prefill 阶段已经算好了每一层的 K/V，这些 K/V 已经过 q/k_norm + RoPE
- 我们的 gen 路 attention 把 frozen K/V 直接当 prefix 用，**不再过任何 Linear**
- gen 自己的 token 还是要走自己的 Q/K/V 投影（独立可学参数）
- attention 时 K = [gen_K, frozen_K]，V = [gen_V, frozen_V] 沿 token 维拼接

这样做的好处
============
1. **不破坏 frozen Qwen 的语义空间**：K/V 还是 Qwen 原版的，没被二次投影
2. **完全对齐 AutoMoT 严格 MoT 的 attention 数学**：见
   `AutoMoT/Automot/mot/modeling/automot/qwen3vl_navit.py:678-770` 的
   `forward_inference`，它在每个 Qwen LM 层内做的就是这件事
3. **能直接对接 goalgen 的 qwen_kv pipeline**：DiT 也是这么消费 K/V 的

硬约束
======
要让 frozen K/V 不过 Linear 直接拼接进 attention，gen 路的 (num_heads, head_dim)
必须严格等于 Qwen K/V 的 (num_kv_heads=8, head_dim=128)。所以本子包默认
gen_hidden = 8 * 128 = 1024。这条等式由 LeadMoTPlanningDecoderConfig.validate_qwen_kv_shape
在初始化时校验。

四个类
======
1. **RMSNorm**：跟 goalgen DiT / Qwen3 同名同构的归一化层
2. **SwiGLU**：Qwen/LLaMA 风格 FFN（gate_proj + up_proj + down_proj）
3. **PrefixKVAttention**：单层 prefix-KV attention，本子包核心
4. **MoTDecoderBlock**：把上面三个组成一个完整 decoder block
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """简单 RMSNorm，跟 goalgen DiT / Qwen3 一致。

    为什么用 RMSNorm 而不是 LayerNorm
    =================================
    Qwen3 系（包括 Qwen3-VL-4B-Instruct）的 LM 层用的全部是 RMSNorm。我们 gen
    路也用 RMSNorm，让 gen attention 输入归一化方式与 Qwen K/V 的归一化方式一致，
    数值分布更接近 frozen Qwen，attention 训练更稳定。

    RMSNorm 数学
    ============
    LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:    y = x / sqrt(mean(x^2) + eps) * gamma

    RMSNorm 省掉了 mean 计算和 beta，参数少一半，计算也少；在大模型上效果几乎一样。

    为什么要 .float() 再回 .to(orig_dtype)
    ======================================
    bf16 下 x^2 容易溢出/下溢，必须升精度算 rms 再降回去。这是 LLaMA/Qwen 标准做法。
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        参数:
            dim: 归一化维度（最后一维），通常等于 hidden_size 或 head_dim
            eps: 防 sqrt(0) 的小常数，跟 Qwen3 一致用 1e-6
            elementwise_affine: 是否带可学 gamma（weight 参数）；False 时退化为纯归一化
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 可选的可学缩放因子 gamma，初始化为全 1（不缩放）
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            # 不要 weight 时显式注册成 None，保证 state_dict 不带这个 key
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记下原始 dtype（通常 bf16），最后还原回去
        orig_dtype = x.dtype

        # 升到 fp32 算 RMS，避免 bf16 下 x^2 数值不稳
        x_fp = x.float()

        # rms = 1 / sqrt(mean(x^2) + eps)，写成 inplace 链式调用
        # pow(2): x^2  → mean(-1, keepdim=True): 沿最后一维求均值，保留维度
        # add_(eps): 加 eps  → rsqrt_(): 倒数平方根
        # 用 inplace 是为了少一次中间张量分配（这里 add_ / rsqrt_ 后缀的下划线）
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()

        # 归一化：x / rms_denom 等价于 x * (1 / rms_denom) = x * rms
        # 算完转回原 dtype
        x_out = (x_fp * rms).to(orig_dtype)

        # 可选的逐元素缩放（gamma 参数）
        if self.elementwise_affine:
            x_out = x_out * self.weight
        return x_out


class SwiGLU(nn.Module):
    """Qwen/LLaMA 风格 SwiGLU FFN。

    SwiGLU 数学
    ===========
    传统 FFN（GELU MLP）: x -> Linear(h, 4h) -> GELU -> Linear(4h, h)
    SwiGLU:              x -> [Linear(h, m), Linear(h, m)] -> silu(gate) * up -> Linear(m, h)

    其中：
    - gate_proj 和 up_proj 是两个并行的 Linear（输入相同，权重不同）
    - silu = SiLU 激活，等价于 swish = x * sigmoid(x)
    - down_proj 把中间维 m 投回 hidden_size

    为什么用 SwiGLU 而不是 GELU MLP
    ===============================
    Qwen3-VL-4B-Instruct 自己的 FFN 就是 SwiGLU；gen 路 FFN 也用 SwiGLU，
    数值分布与 frozen Qwen 主干一致，attention 输出语义更兼容。

    为什么没 bias
    =============
    Qwen3 LLaMA 系的 SwiGLU 全部 `bias=False`。bias 在 SwiGLU 里几乎没用（gate 已经
    起了 selective 的作用），去掉可省一点显存。
    """

    def __init__(self, hidden_size: int, inner_size: int):
        """
        参数:
            hidden_size: gen 路 hidden（默认 1024）
            inner_size: SwiGLU 中间维 m（默认 ≈ 2730 = 1024 * 8/3）
        """
        super().__init__()
        # 三个 Linear 都不带 bias，跟 Qwen3 一致
        # gate_proj 和 up_proj 同输入（hidden_size），同输出（inner_size），但权重不同
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        # down_proj 把 inner_size 投回 hidden_size
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 公式: down_proj( silu(gate_proj(x)) * up_proj(x) )
        # F.silu 是 PyTorch 1.7+ 自带的 silu/swish 实现
        # gate 和 up 输出形状都是 (..., inner_size)，逐元素相乘后维持 (..., inner_size)
        # down_proj 把 inner_size -> hidden_size
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PrefixKVAttention(nn.Module):
    """gen 路 Q 同时看 gen K/V 与 frozen Qwen prefix K/V 的 attention。

    本子包的核心模块，attention 数学如下：

        Q_gen = q_proj(gen_tokens) -> (B, n_heads, L_gen, head_dim)
        K_gen = k_proj(gen_tokens) -> (B, n_heads, L_gen, head_dim)
        V_gen = v_proj(gen_tokens) -> (B, n_heads, L_gen, head_dim)

        # Qwen3 风格的 head_dim 上 RMSNorm（gen Q/K 各自归一化）
        Q_gen = q_norm(Q_gen)
        K_gen = k_norm(K_gen)

        # frozen Qwen K/V 不过任何 Linear，只做 device/dtype 对齐
        K_lang, V_lang = lang_kv          # 来自 segment_kv_for_dit 池化的某一段

        # 沿 token 维拼接（gen 在前，lang 在后）
        K = concat([K_gen, K_lang], dim=token)   # (B, n_heads, L_gen + S, head_dim)
        V = concat([V_gen, V_lang], dim=token)

        # 标准 scaled dot-product attention，让 PyTorch 选最优 backend
        # （CUDA 上自动用 FlashAttention 2/3，CPU 用普通实现）
        attn_out = SDPA(Q_gen, K, V)
        out = out_proj(attn_out)

    跟 AutoMoT 严格 MoT forward_inference 的对应关系
    =================================================
    见 `AutoMoT/Automot/mot/modeling/automot/qwen3vl_navit.py:678-770`：
    - 它的 q_proj_mot_gen / k_proj_mot_gen / v_proj_mot_gen
      ↔ 本类的 q_proj / k_proj / v_proj
    - 它的 q_norm_mot_gen / k_norm_mot_gen
      ↔ 本类的 q_norm / k_norm
    - 它把 cache K/V 与新算的 K/V merge 后做 attention
      ↔ 本类把 lang_kv 与 gen K/V concat 后做 attention
    - 它的 o_proj_mot_gen
      ↔ 本类的 out_proj
    数学结构完全一致，唯一差异：AutoMoT 在 Qwen LM 36 层内做这件事，
    我们在外部 12 层独立 transformer 里做（每层用一段池化后的 K/V）。
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        """
        参数:
            hidden_size: gen 路 hidden（默认 1024）
            num_heads:   attention 头数，必须等于 Qwen num_kv_heads（默认 8）
            dropout:     attention 上的 dropout 概率（仅训练时生效）
        """
        super().__init__()
        # 强约束：hidden_size 必须能被 num_heads 整除，否则 reshape 出问题
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # 每头维度，必须等于 Qwen head_dim（默认 128）才能让 frozen K/V 直接接进来
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)

        # gen 路 Q/K/V 投影，全部不带 bias（Qwen3 风格）
        # 注意：frozen Qwen K/V 不通过这里，它直接进 attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # head_dim 上 RMSNorm：作用在最后一维 head_dim 上，把 gen Q/K 各自拉到单位 RMS
        # 这是 Qwen3 q_norm/k_norm 同款做法，让 attention 数值更稳
        # 注意：lang K（来自 frozen Qwen prefill）已经在 prefill 时过了 q_norm/k_norm，
        # 这里只 norm gen 自己的 Q/K，不动 lang K
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=True)

        # attention 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _check_and_cast_lang_kv(
        self,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """校验 frozen Qwen K/V 形状并搬到 gen 同 device/dtype。

        参数:
            lang_kv:    (K, V) 一对张量，来自 segment_kv_for_dit 池化的某一段
                        每个形状期望 [B, n_heads, S, head_dim]
            batch_size: gen 段的 batch（用于检查 K/V 的 batch 是否一致或需要广播）
            ref:        参考张量（通常是 gen 自己投影出的 Q），用来对齐 device/dtype

        返回:
            (k_l, v_l) 形状不变，device/dtype 已对齐 ref

        会在形状错时直接 raise（带详细信息），不让错误延迟到 attention 内部。
        """
        k_l, v_l = lang_kv

        # ---- 形状校验 ----
        # 必须是 4D [B, H, S, D]
        if k_l.ndim != 4 or v_l.ndim != 4:
            raise ValueError(
                f"language KV 应为 [B,H,S,D]，实际 K={tuple(k_l.shape)} V={tuple(v_l.shape)}"
            )
        # K 和 V 必须同形（除非池化模式特殊，否则一定同形）
        if k_l.shape != v_l.shape:
            raise ValueError(f"language K/V shape 不一致：K={tuple(k_l.shape)} V={tuple(v_l.shape)}")
        # heads 数与 head_dim 必须严格等于 gen 路配置；这就是 prefix-KV 的核心约束
        if k_l.shape[1] != self.num_heads or k_l.shape[3] != self.head_dim:
            raise ValueError(
                f"language KV {tuple(k_l.shape)} 与 gen (heads={self.num_heads}, "
                f"head_dim={self.head_dim}) 不匹配"
            )

        # ---- batch 维处理 ----
        # 如果 lang KV 的 batch 与 gen 不同，允许 batch=1 的情况广播到 batch_size
        # （比如离线推理时 gen 是 B=1，cache 也是 B=1，但 batch 维度上的 batch 可能不一致）
        if k_l.shape[0] != batch_size:
            if k_l.shape[0] == 1:
                # expand 不拷贝内存，让所有 gen sample 共享同一份 cache
                k_l = k_l.expand(batch_size, -1, -1, -1)
                v_l = v_l.expand(batch_size, -1, -1, -1)
            else:
                # 既不等也不能广播（lang batch > 1 且 != gen batch），直接报错
                raise ValueError(f"language KV batch {k_l.shape[0]} != gen batch {batch_size}")

        # ---- device/dtype 对齐 ----
        # frozen Qwen cache 通常是 bf16，gen 也可能是 bf16，但保险起见统一搬到 ref 的 device/dtype
        return (
            k_l.to(device=ref.device, dtype=ref.dtype),
            v_l.to(device=ref.device, dtype=ref.dtype),
        )

    def forward(
        self,
        gen_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """单层 prefix-KV attention forward。

        参数:
            gen_tokens: (B, L_gen, hidden) gen 段 token，本层已经过 norm_attn
            lang_kv:    (K, V) frozen Qwen 池化后的一段 K/V，形状 (B 或 1, n_heads, S, head_dim)
        返回:
            (B, L_gen, hidden) attention 输出，与 gen_tokens 同形
        """
        # 解包 batch / 序列长度，hidden 维不用变量名（用 _ 占位）
        b, n, _ = gen_tokens.shape

        # ---- gen Q/K/V 投影 ----
        # 每个 Linear 都把 hidden 投到 hidden，然后 view 拆成 (B, L_gen, n_heads, head_dim)
        # transpose(1, 2) 把 head 维提到第 1 维，得到 (B, n_heads, L_gen, head_dim)
        # 这是 PyTorch SDPA 期望的 layout（head 在 batch 之后、token 之前）
        q = self.q_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.k_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.v_proj(gen_tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Q/K 上的 head_dim RMSNorm（Qwen3 风格）----
        # 只 norm gen 自己的 Q/K，lang K 已经在 frozen Qwen prefill 时 norm 过了
        q = self.q_norm(q)
        k_g = self.k_norm(k_g)

        # ---- 校验+对齐 frozen K/V ----
        # ref=q 是为了让 lang K/V 跟 gen Q 同 device/dtype，避免 attention 时 dtype mismatch
        k_l, v_l = self._check_and_cast_lang_kv(lang_kv, batch_size=b, ref=q)

        # ---- 拼接 gen + lang ----
        # dim=2 是 token 维（前面 transpose 把 head 放第 1 维后，token 在第 2 维）
        # 顺序约定：gen 在前，lang 在后（跟 goalgen DiT 一致）
        # 数学上 attention 对 K/V token 顺序不敏感，但拼接顺序固定方便未来扩展位置编码
        k = torch.cat([k_g, k_l], dim=2)   # (B, n_heads, L_gen + S, head_dim)
        v = torch.cat([v_g, v_l], dim=2)

        # ---- scaled dot-product attention ----
        # 训练时按 self.dropout 概率丢 attention 权重，eval 时关闭
        dropout_p = self.dropout if self.training else 0.0

        # F.scaled_dot_product_attention 是 PyTorch 2.0+ 的统一 attention 接口
        # 自动选最优 backend：CUDA 上选 FlashAttention 2/3 / memory-efficient attention，
        # CPU 上用普通 math impl。比手写 softmax(Q@K^T/sqrt(d))@V 快得多、省显存得多
        # 注意：不传 attn_mask / is_causal 参数，等价于全连接 attention（gen 段内部没有 causal 约束）
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # ---- reshape 回 hidden 维 ----
        # (B, n_heads, L_gen, head_dim) -> (B, L_gen, n_heads, head_dim)
        # contiguous() 是因为 transpose 后内存非连续，view 需要连续内存
        # view 再合并最后两维：(B, L_gen, n_heads * head_dim) = (B, L_gen, hidden)
        out = out.transpose(1, 2).contiguous().view(b, n, self.hidden_size)

        # 输出投影
        return self.out_proj(out)


class MoTDecoderBlock(nn.Module):
    """一层 prefix-KV gen decoder block。

    结构：RMSNorm -> PrefixKVAttention -> +residual -> RMSNorm -> SwiGLU -> +residual

    这是标准的 pre-norm Transformer block，跟 Qwen3 LM 层结构一致，方便未来从 Qwen
    复制权重做 warm start。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
    ):
        """
        参数:
            hidden_size:     gen 路 hidden（默认 1024）
            num_heads:       attention 头数（默认 8）
            ffn_hidden_size: SwiGLU 中间维（默认 ≈ 2730）
            dropout:         attention 和 FFN 输出上的 dropout 概率
        """
        super().__init__()

        # attention 前的 norm（pre-norm 风格，Qwen3 一致）
        self.norm_attn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        # prefix-KV attention 主体
        self.attn = PrefixKVAttention(hidden_size, num_heads, dropout=dropout)

        # FFN 前的 norm
        self.norm_ffn = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        # SwiGLU FFN
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size)

        # residual 路上的 dropout（attention output 和 FFN output 各一次）
        # eval 时自动关闭，所以推理路径上 dropout 没开销
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gen_seq: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        参数:
            gen_seq: (B, L_gen, hidden) 进入本层时的 gen 段 hidden
            lang_kv: 本层使用的 frozen Qwen prefix K/V，形状 [B 或 1, n_heads, S, head_dim]
                     decoder 会按 num_layers 把 Qwen 36 层池化成 12 段，每层用一段
        返回:
            (B, L_gen, hidden) 本层输出，进入下一层
        """
        # ---- attention 子层 ----
        # pre-norm：先 norm 再喂 attention
        h = self.norm_attn(gen_seq)
        # attention 输出加 residual，跟 Transformer 标准做法一致
        # dropout 在 residual 加之前对 attention 输出做（标准实现）
        gen_seq = gen_seq + self.dropout(self.attn(h, lang_kv))

        # ---- FFN 子层 ----
        # 同样 pre-norm 再过 SwiGLU
        h = self.norm_ffn(gen_seq)
        # FFN 输出加 residual + dropout
        gen_seq = gen_seq + self.dropout(self.ffn(h))

        return gen_seq
