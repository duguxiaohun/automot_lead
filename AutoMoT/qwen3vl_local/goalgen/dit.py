"""DiT-MoT：12 层 joint-attention 的 latent 扩散主干。

设计要点（详见 PROJECT_CONTEXT.md §15）：

- 输入是冻结 VAE 编出来的潜变量：含噪目标 z_t 与历史帧 z_history。
- z_t 与每一帧历史潜变量**共享同一个 Patchify**（v2 改动；v1 是两个独立卷积），
  拼成视觉 token 序列；类型由 type_embed 区分、时序由 frame_embed 区分。
- 每层 block 做 MoT 风格的 joint attention：
    Q = vision_token 投出来的 Q
    K = concat[ vision_K, language_K_seg[i] ]
    V = concat[ vision_V, language_V_seg[i] ]
  其中 language K/V 是冻结的 Qwen pooled KV 经过 per-layer 线性投影到 DiT hidden 后，
  按 (n_heads, head_dim) 重排得到。
- timestep 用 AdaLN-Zero 注入；vision token 不修改 language 部分。
- 输出只读出"z_t 那一段" token，反图块化回 [B, 4, H/8, W/8]，作为速度预测。

关键形状（默认配置，针对 LEAD 1152x384）：

- VAE 潜变量: [B, 4, 48, 144]
- patch_size = 2  -> 图块网格 (24, 72) -> 每个潜变量 1728 个 token
- 视觉 token = 1728 (z_t) + F * 1728 (z_history)，数据构建器默认 F=4 -> 8640 个 token
- hidden_dim = 768, n_heads = 12, head_dim = 64
- language token = Qwen prefill seq_len, 例如 ~2332
"""

from __future__ import annotations

import math
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


class JointAttention(nn.Module):
    """vision Q 与 (vision K/V + language K/V) 做一次 attention。

    vision_K/V 来自 vision token 自身的投影；language_K/V 来自外部 pooled Qwen KV，
    经过 per-layer 线性投影到 (n_heads, head_dim)。
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        language_kv_input_dim: int,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # vision 三投影。
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # language K/V 投影：从 Qwen pooled KV flatten 后的维度映射到 DiT hidden。
        self.lang_k_proj = nn.Linear(language_kv_input_dim, hidden_dim)
        self.lang_v_proj = nn.Linear(language_kv_input_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _project_language_kv(
        self,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, n_kv_heads, S, head_dim] -> [B, n_heads, S, dit_head_dim]。

        这是 MoT joint attention 里"语言通道接进 DiT 注意力"的关键一步。Qwen 的
        K/V 头数（默认 8）和 head_dim（默认 128）与 DiT 的 n_heads（12）/ head_dim
        （64）通常对不上，所以走如下三步：

        1. 把 (n_kv_heads, head_dim) **flatten 到最后一维**：从 [B, n_kv, S, Hk]
           permute 成 [B, S, n_kv, Hk] 再 view 成 [B, S, n_kv * Hk]。这一维就是
           "每个 token 在 Qwen 那边的所有 KV 头拼起来"，对应 PLAN §5 中的
           language_kv_input_dim（Qwen3-VL-4B-Instruct: 8 × 128 = 1024）。
        2. Linear(1024 -> 768) 把它投到 DiT hidden 维。**注意**：这条投影是 DiT
           的可训练参数（每个 block 一个独立投影），但梯度**不会回到 Qwen**，
           因为 lang_kv 是 detach 出来的，所以 Qwen 保持冻结。
        3. 再 view 成 (n_heads=12, head_dim=64) 并 transpose 成 attention 期望的
           [B, n_heads, S, head_dim] 形状，方便和 vision 流的 K/V concat。
        """

        k, v = lang_kv
        # qwen_kv.teacher_forced_prefill 通常以 batch=1 跑 Qwen 预填充；DiT 训练
        # 时可能 batch>1。这里允许 1->B 的 expand：expand 不拷贝内存，attention
        # 内部按 batch 维只读取，不会写回，所以共享语言 KV 是安全的。
        if k.shape[0] != batch_size:
            if k.shape[0] == 1:
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            else:
                # batch 既不等也不是 1，说明上游 dataloader 拼 batch 时没保持
                # "一条样本一份 prefill"约定，需要立刻报错而不是错位 broadcast。
                raise ValueError(
                    f"language KV batch {k.shape[0]} 不等于 vision batch {batch_size}"
                )
        b, n_kv, s, hk = k.shape

        # 步骤 1：[B, n_kv, S, Hk] -> [B, S, n_kv*Hk]
        # permute 后内存非连续，所以紧跟 reshape；contiguous 由 reshape 内部处理。
        k_flat = k.permute(0, 2, 1, 3).reshape(b, s, n_kv * hk)
        v_flat = v.permute(0, 2, 1, 3).reshape(b, s, n_kv * hk)

        # 步骤 2 + 3：Linear 投到 hidden_dim，再切成 DiT 的 (n_heads, head_dim)。
        # transpose(1, 2) 把 head 维放到 dim=1，得到 [B, H, S, D]，符合 SDPA 期望。
        k_dit = self.lang_k_proj(k_flat).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v_dit = self.lang_v_proj(v_flat).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        return k_dit, v_dit

    def forward(
        self,
        vision_tokens: torch.Tensor,
        lang_kv: Tuple[torch.Tensor, torch.Tensor],
        lang_kv_is_projected: bool = False,
    ) -> torch.Tensor:
        """Joint attention：vision Q 同时看 vision K/V 与 language K/V。

        与普通 cross-attention 的核心区别：cross-attn 是两次 attention（一次
        self、一次 cross），joint-attn 是**一次** attention，K/V 沿 token 维拼起来。
        这样 vision token 可以在同一个 softmax 内同时挑选"参考自己邻域"还是
        "对齐语言上下文"，更接近 AutoMoT 里"快慢 MoT"原始设计。

        ``lang_kv_is_projected``：True 表示 ``lang_kv`` 已经处于 DiT 的
        ``(n_heads, head_dim)`` 子空间（形状 ``[B 或 1, n_heads, S, head_dim]``），
        跳过 ``lang_k_proj`` / ``lang_v_proj``。这条路径专门服务于 CFG 训练中
        的 ``null_lang_k / null_lang_v``：null KV 由 DiT 自己持有、参数量极小，
        不需要再经过 Qwen → DiT 的语言投影。
        """

        b, n_v, _ = vision_tokens.shape
        # vision 三个线性投影 + 切头：用法跟普通 transformer 一致。
        q = self.q_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_proj(vision_tokens).view(b, n_v, self.n_heads, self.head_dim).transpose(1, 2)

        if lang_kv_is_projected:
            # CFG null KV 路径：lang_kv 已经在 DiT (n_heads, head_dim) 子空间，
            # 不需要再走 lang_k_proj / lang_v_proj。仅做必要的 batch 广播。
            k_l, v_l = lang_kv
            if k_l.shape[0] != b:
                if k_l.shape[0] == 1:
                    k_l = k_l.expand(b, -1, -1, -1)
                    v_l = v_l.expand(b, -1, -1, -1)
                else:
                    raise ValueError(
                        f"projected null KV batch {k_l.shape[0]} 不等于 vision batch {b}"
                    )
        else:
            # 语言侧把 Qwen pooled KV 投到 DiT 自己的 (n_heads, head_dim)；详见上面方法注释。
            k_l, v_l = self._project_language_kv(lang_kv, batch_size=b)

        # 沿 token 维拼接：[B, H, N_v, D] + [B, H, N_l, D] -> [B, H, N_v + N_l, D]。
        # Q 仍只有 vision token，所以 attention 输出 token 数 = N_v，意味着这一层
        # **只更新 vision 流**；language 部分作为冻结 memory，不在本 block 内被修改。
        k_cat = torch.cat([k_v, k_l], dim=2)
        v_cat = torch.cat([v_v, v_l], dim=2)

        # 用 PyTorch 内置 SDPA，性能更稳；在新版本会自动选 flash-attn / mem-efficient。
        attn = F.scaled_dot_product_attention(q, k_cat, v_cat, dropout_p=0.0)
        # 把 head 维合回去：[B, H, N_v, D] -> [B, N_v, H*D] -> Linear hidden。
        out = attn.transpose(1, 2).contiguous().view(b, n_v, self.n_heads * self.head_dim)
        return self.out_proj(out)


class DiTMoTBlock(nn.Module):
    """单个 DiT-MoT block：AdaLN -> JointAttention -> AdaLN -> MLP，全部带 gate。"""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
        language_kv_input_dim: int,
        cond_dim: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = JointAttention(hidden_dim, n_heads, language_kv_input_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
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
        """单 block forward：AdaLN-Zero -> JointAttn -> AdaLN-Zero -> MLP。

        cond 由外部一次性算好（来自 timestep MLP），所有 block 共用同一个 cond，但每个
        block 内部的 AdaLN modulation 矩阵不共享 -> 每层有自己的 shift/scale/gate。

        ``lang_kv_is_projected`` 透传给 JointAttention：CFG uncond 路径下使用 DiT
        自己的 null_lang_k/v，已经在 DiT (n_heads, head_dim) 子空间。
        """

        # 一次 Linear 出 6 个调制向量（attn 3 + mlp 3）。
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(cond)

        # ---- attention 子层 ----
        # norm1 是无仿射的 LayerNorm；shift/scale 由 cond 提供，相当于每层独立可学的 affine。
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
    """DiT-MoT 默认配置；变更默认值时同时更新 PROJECT_CONTEXT.md §15。"""

    latent_channels: int = 4
    patch_size: int = 2
    hidden_dim: int = 768
    n_heads: int = 12
    mlp_ratio: float = 4.0
    num_layers: int = 12
    cond_dim: int = 256
    # Qwen pooled KV 的 (n_kv_heads, head_dim) flatten 后维度。
    # 默认对应 Qwen3-VL-4B-Instruct：n_kv_heads=8, head_dim=128 -> 1024。
    language_kv_input_dim: int = 1024
    max_grid_h: int = 64
    max_grid_w: int = 192
    max_history_frames: int = 8


class DiTMoT(nn.Module):
    """完整 DiT-MoT 主干。

    forward 输入：
      - z_t   : [B, C, H, W]，含噪声的目标 latent
      - z_history : [B, F, C, H, W]，历史 VAE latent（旧 -> 新）
      - t     : [B]，flow matching 时间步 ∈ [0,1]
      - pooled_kv : 长度 = num_layers 的列表，元素为 (K_seg, V_seg)；每个张量形状
                    [B, n_kv_heads, S, head_dim]
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
        # bypass JointAttention 内部的 lang_proj。s_null=1：null 只贡献一个 token，
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
                language_kv_input_dim=cfg.language_kv_input_dim,
                cond_dim=cfg.cond_dim,
            )
            for _ in range(cfg.num_layers)
        ])

        self.final_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = AdaLNModulation(cfg.hidden_dim, cfg.cond_dim)
        self.unpatch = Unpatchify(cfg.hidden_dim, cfg.latent_channels, cfg.patch_size)

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
          [B 或 1, n_kv_heads, S, head_dim]，dtype 通常和 DiT 自身一致。
        - force_uncond：True 表示用 DiT 自身的 null_lang_k/v 代替 pooled_kv
          走 uncond 路径（CFG 训练 / 引导推理用）。pooled_kv 此时仍需传入，
          只是不被使用——保留位置以保持接口稳定。

        输出：v_pred 与 z_t 同形状，对应 velocity 预测。
        """

        # 这一步检查在第一次 forward 时定位"runner / 训练脚本里把 num_layers 改了
        # 一边没改另一边"的低级错误，比让 attention 中段越界报错可读得多。
        if len(pooled_kv) != self.cfg.num_layers:
            raise ValueError(
                f"pooled_kv 段数 {len(pooled_kv)} 与 DiT 层数 {self.cfg.num_layers} 不一致"
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
        n_t = tok_t.shape[1]

        # cond 只算一次，所有 block 共享；AdaLN modulation 矩阵是 per-block 的。
        cond = self._build_cond(t)

        # 逐层走 block。每层用 pooled_kv[i] 作为冻结语言 memory；
        # force_uncond=True 时改用 DiT 自带的 null_lang_k/v（CFG 路径）。
        for i, (block, lang_kv) in enumerate(zip(self.blocks, pooled_kv)):
            if force_uncond:
                null_kv = (
                    self.null_lang_k[i].to(dtype=vision_tokens.dtype),
                    self.null_lang_v[i].to(dtype=vision_tokens.dtype),
                )
                vision_tokens = block(vision_tokens, null_kv, cond, lang_kv_is_projected=True)
            else:
                vision_tokens = block(vision_tokens, lang_kv, cond)

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


def language_kv_input_dim_from_pooled(pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
    """从 pooled_kv 推断 language_kv_input_dim = n_kv_heads * head_dim。

    runner 在构造 DiTMoTConfig 前调用，省得手动算 Qwen KV 维度。
    """

    if not pooled_kv:
        raise ValueError("pooled_kv 为空")
    k0, _ = pooled_kv[0]
    if k0.ndim != 4:
        raise ValueError(f"期望 pooled K 形状为 [B, n_kv_heads, S, head_dim]，实际得到 {k0.shape}")
    return int(k0.shape[1] * k0.shape[3])
