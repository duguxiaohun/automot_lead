"""LeadMoTPlanningDecoder 配置类。

默认数值按 LEAD CARLA leaderboard 模式：route=10, waypoint=8, dt=0.25s。
gen 路 hidden=1024=8*128 直接对齐 Qwen3-VL-4B-Instruct (num_kv_heads, head_dim)，
让 frozen K/V 不过 Linear 直接 concat 进 attention。

RoPE 三种模式（参考 https://github.com/JJJYmmm/Multimodal-RoPEs）：
- mrope    : Qwen3-VL 标准 M-RoPE，head_dim/2 切 3 段分别用 t/h/w 旋转
             —— 与 prefix K 自带 M-RoPE **完全数学等价**，推荐默认。
- mhrope   : Multi-Head RoPE，head 维分配 3 轴（不同 head 用不同 axis）
             —— 要充分生效需同时 patch standalone Qwen prefill 改用 mhrope，
                否则跟默认 M-RoPE 的 prefix K 不匹配。
- none     : 不加 RoPE，gen Q/K 不旋转。attention 仍可计算，只是位置感缺失。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LeadMoTPlanningDecoderConfig:
    # gen 路维度，必须 = num_kv_heads * head_dim
    hidden_size: int = 1024
    qwen_hidden_size: int = 2560      # Qwen 主干，仅元信息
    point_dim: int = 2

    # frozen Qwen prefix K/V 维度
    num_kv_heads: int = 8
    head_dim: int = 128
    num_qwen_layers: int = 36
    kv_segment_mode: str = "select_last"   # 'select_last' / 'mean' / 'concat_layers'

    # ---- RoPE 配置 ----
    # rope_type: "mrope" (Qwen3-VL 标准) | "mhrope" (head-wise) | "none" (不加 RoPE)
    rope_type: str = "mrope"
    rope_theta: float = 5000000.0
    # M-RoPE: head_dim//2 切 3 段，sum 必须 == head_dim//2 (=64)
    # 默认 Qwen3-VL-4B-Instruct 标准 [16, 24, 24]
    mrope_section_dim: Tuple[int, int, int] = (16, 24, 24)
    # MH-RoPE: num_heads 切 3 段（每段共享同一 axis），sum 必须 <= num_heads
    # 默认 (3, 3, 2)：3 个 head 用 temporal，3 个用 height，2 个用 width
    mrope_section_head: Tuple[int, int, int] = (3, 3, 2)

    # BEV (LEAD LeadTransfuserBackbone 输出)
    bev_channels: int = 512
    bev_grid: Tuple[int, int] = (10, 12)   # H*W = 120 token

    # LEAD 头部数量
    num_route_queries: int = 10            # = lead num_route_points_prediction
    num_waypoint_queries: int = 8          # = lead num_way_points_prediction (CARLA)
    waypoint_dt: float = 0.25

    # gen 路 decoder
    num_layers: int = 12                   # 36 Qwen 层 / 3 = 12 段
    num_heads: int = 8                     # = num_kv_heads (MHA)
    # SwiGLU 8/3 让总参数量 ≈ 4*hidden^2，与 GELU MLP 同规模
    mlp_ratio: float = 8.0 / 3.0
    dropout: float = 0.0

    speed_dim: int = 1
    target_point_dim: int = 2

    def total_gen_tokens(self) -> int:
        """BEV + (speed/tp/ntp) + route_q + waypoint_q = 120+3+10+8 = 141"""
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        return bev_tokens + 3 + self.num_route_queries + self.num_waypoint_queries

    def slice_layout(self):
        """packed gen 序列每段 token 的 [start, end) 索引。

        顺序: BEV | speed | tp | ntp | route_q | waypoint_q
        改这里的 cat 顺序必须同步改 decoder._build_gen_sequence。
        """
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        idx = 0
        layout = {}
        layout["bev"] = (idx, idx + bev_tokens); idx += bev_tokens
        layout["speed"] = (idx, idx + 1); idx += 1
        layout["tp"] = (idx, idx + 1); idx += 1
        layout["ntp"] = (idx, idx + 1); idx += 1
        layout["route"] = (idx, idx + self.num_route_queries); idx += self.num_route_queries
        layout["waypoint"] = (idx, idx + self.num_waypoint_queries)
        return layout

    @property
    def ffn_hidden_size(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    def active_mrope_section(self) -> Tuple[int, int, int]:
        """根据 rope_type 返回当前用的 mrope_section（dim 版或 head 版）。

        rope_type=='none' 时返回 mrope_section_dim 仅作占位，运行时不会被 attention 读取。
        """
        if self.rope_type in {"mrope", "none"}:
            return self.mrope_section_dim
        if self.rope_type == "mhrope":
            return self.mrope_section_head
        raise ValueError(f"Unknown rope_type: {self.rope_type!r}")

    def validate_qwen_kv_shape(self) -> None:
        """prefix-KV / RoPE 工作的硬约束。"""
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                f"hidden_size 必须 = num_kv_heads * head_dim，"
                f"当前 {self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads 必须 = num_kv_heads，当前 {self.num_heads} != {self.num_kv_heads}"
            )
        if self.rope_type not in {"mrope", "mhrope", "none"}:
            raise ValueError(
                f"rope_type 必须是 'mrope' / 'mhrope' / 'none'，当前 {self.rope_type!r}"
            )
        if self.rope_type == "none":
            return
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE 要求 head_dim 偶数，当前 {self.head_dim}")
        if self.rope_type == "mrope":
            if sum(self.mrope_section_dim) != self.head_dim // 2:
                raise ValueError(
                    f"M-RoPE: sum(mrope_section_dim)={sum(self.mrope_section_dim)} "
                    f"必须 = head_dim//2={self.head_dim // 2}"
                )
        else:  # mhrope
            if sum(self.mrope_section_head) > self.num_heads:
                raise ValueError(
                    f"MH-RoPE: sum(mrope_section_head)={sum(self.mrope_section_head)} "
                    f"不能超过 num_heads={self.num_heads}"
                )
