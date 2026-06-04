"""LeadMoTPlanningDecoder 配置类。

默认数值按 LEAD CARLA leaderboard 模式：route=10, waypoint=8, dt=0.25s。
gen 路 hidden=1024=8*128 直接对齐 Qwen3-VL-4B-Instruct (num_kv_heads, head_dim)，
让 frozen K/V 不过 Linear 直接 concat 进 attention。
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

    def validate_qwen_kv_shape(self) -> None:
        """prefix-KV 工作的两条硬约束。"""
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                f"hidden_size 必须 = num_kv_heads * head_dim，"
                f"当前 {self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads 必须 = num_kv_heads，当前 {self.num_heads} != {self.num_kv_heads}"
            )
