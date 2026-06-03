"""LeadMoTPlanningDecoder 的配置类。

数值默认值按 LEAD CARLA leaderboard 模式：
- num_route_queries     = 10        对应 lead config.num_route_points_prediction
- num_waypoint_queries  = 8         对应 lead config.num_way_points_prediction
- waypoint_dt           = 0.25 s    waypoints_spacing=5 @ 20Hz = 4Hz

gen 路 hidden 默认 1024 = Qwen3-VL-4B-Instruct 的 num_kv_heads(8) * head_dim(128)。
这样每层可以直接读 frozen Qwen 的 prefix K/V，不需要对语言 K/V 再做线性投影。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LeadMoTPlanningDecoderConfig:
    # ---- 公共维度 ----
    hidden_size: int = 1024              # gen 路维度 = 8 * 128，对齐 Qwen K/V 子空间
    qwen_hidden_size: int = 2560         # Qwen3-VL-4B-Instruct hidden_size，仅作元信息
    point_dim: int = 2

    # ---- frozen Qwen prefix K/V 维度 ----
    num_kv_heads: int = 8
    head_dim: int = 128
    num_qwen_layers: int = 36
    kv_segment_mode: str = "select_last"

    # ---- BEV 输入维度（LEAD LeadTransfuserBackbone 输出）----
    bev_channels: int = 512
    bev_grid: Tuple[int, int] = (10, 12)   # H=10, W=12，flatten 后 120 token

    # ---- LEAD 头部数量（CARLA leaderboard 模式）----
    num_route_queries: int = 10
    num_waypoint_queries: int = 8
    waypoint_dt: float = 0.25

    # ---- gen 路 decoder 结构 ----
    num_layers: int = 12                  # 36 层 Qwen 按 3 层一段 select_last -> 12 段
    num_heads: int = 8                    # 必须等于 Qwen num_kv_heads
    # SwiGLU 有 gate + up 两个投影，按 LLaMA/Qwen 公式 mlp_ratio = 8/3 让总参数量
    # ≈ 4 * hidden^2，与 GELU MLP 同规模；得到 ffn_hidden_size ≈ 2730。
    mlp_ratio: float = 8.0 / 3.0
    dropout: float = 0.0

    # ---- status token 编码 ----
    speed_dim: int = 1
    target_point_dim: int = 2

    def total_gen_tokens(self) -> int:
        """gen 段总 token 数。"""
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        status_tokens = 3  # speed + tp + ntp
        return bev_tokens + status_tokens + self.num_route_queries + self.num_waypoint_queries

    def slice_layout(self):
        """返回 gen 段每一类 token 在 packed_gen_sequence 中的 [start, end) 索引。

        顺序：BEV | speed | tp | ntp | route_q | waypoint_q
        """
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        idx = 0
        layout = {}
        layout["bev"] = (idx, idx + bev_tokens)
        idx += bev_tokens
        layout["speed"] = (idx, idx + 1)
        idx += 1
        layout["tp"] = (idx, idx + 1)
        idx += 1
        layout["ntp"] = (idx, idx + 1)
        idx += 1
        layout["route"] = (idx, idx + self.num_route_queries)
        idx += self.num_route_queries
        layout["waypoint"] = (idx, idx + self.num_waypoint_queries)
        return layout

    @property
    def ffn_hidden_size(self) -> int:
        """FFN 中间维。保留属性名兼容旧代码/文档。"""
        return int(self.hidden_size * self.mlp_ratio)

    def validate_qwen_kv_shape(self) -> None:
        """早期校验配置是否能无投影接 Qwen K/V。"""
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                "hidden_size 必须等于 num_kv_heads * head_dim，"
                f"当前 {self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads 必须等于 num_kv_heads，当前 {self.num_heads} != {self.num_kv_heads}"
            )
