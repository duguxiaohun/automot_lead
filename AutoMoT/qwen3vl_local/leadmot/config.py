"""LeadMoT planning decoder 的配置。

本子包按从 ``AutoMoT`` 目录运行来写路径，例如：
``python qwen3vl_local/leadmot/train_v1.py``。

默认值对齐 LEAD CARLA 设置：
- route head 预测 10 个 ego-frame route 点；
- waypoint head 预测 8 个未来 ego-frame waypoint；
- hidden_size=1024，对齐 Qwen3-VL-4B K/V 的 8 heads * 128 dim。

RoPE 模式：
- ``mrope``：给 LeadMoT 生成 token 使用 Qwen3-VL 风格 M-RoPE；
- ``mhrope``：head-wise multi-axis RoPE，用于消融；
- ``none``：生成 token 不加 RoPE。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LeadMoTPlanningDecoderConfig:
    """LeadMoT decoder 用到的形状和结构开关。

    frozen Qwen prefix K/V 在 attention 前不经过 Linear 投影，因此
    ``hidden_size`` 必须等于 ``num_kv_heads * head_dim``。
    """

    # 生成 token 的 hidden 宽度，必须和 frozen Qwen K/V 宽度一致。
    hidden_size: int = 1024
    qwen_hidden_size: int = 2560
    point_dim: int = 2

    # Frozen Qwen prefix K/V 布局：(B, num_kv_heads, seq, head_dim)。
    num_kv_heads: int = 8
    head_dim: int = 128
    num_qwen_layers: int = 36
    kv_segment_mode: str = "select_last"

    # 只给生成 Q/K 用的 RoPE 配置。Qwen prefix K 已在 prefill 内带位置编码，
    # 这里绝不能重复旋转。
    rope_type: str = "mrope"
    rope_theta: float = 5000000.0
    mrope_section_dim: Tuple[int, int, int] = (16, 24, 24)
    mrope_section_head: Tuple[int, int, int] = (3, 3, 2)

    # LEAD BEV encoder 输出形状：(B, 512, 10, 12)。
    bev_channels: int = 512
    bev_grid: Tuple[int, int] = (10, 12)

    # Query 数量对齐 LEAD planning 标签。
    num_route_queries: int = 10
    num_waypoint_queries: int = 8
    waypoint_dt: float = 0.25

    # Decoder 深度：把 36 层 Qwen 压到 12 个 pooled-prefix block。
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 8.0 / 3.0
    dropout: float = 0.0

    speed_dim: int = 1
    target_point_dim: int = 2

    def total_gen_tokens(self) -> int:
        """返回 packed generated-token 序列长度。

        默认布局是 BEV(120) + speed(1) + target point(1) + next target point(1)
        + route queries(10) + waypoint queries(8) = 141。
        """
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        return bev_tokens + 3 + self.num_route_queries + self.num_waypoint_queries

    def slice_layout(self):
        """返回 packed generated sequence 的 [start, end) 切片。

        这里必须和 ``LeadMoTPlanningDecoder._build_gen_sequence`` 的拼接顺序同步。
        两个 head 只读取 route 和 waypoint 对应切片。
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
        """SwiGLU feed-forward block 的内部宽度。"""
        return int(self.hidden_size * self.mlp_ratio)

    def active_mrope_section(self) -> Tuple[int, int, int]:
        """返回当前 RoPE 模式需要的 section 配置。"""
        if self.rope_type in {"mrope", "none"}:
            return self.mrope_section_dim
        if self.rope_type == "mhrope":
            return self.mrope_section_head
        raise ValueError(f"Unknown rope_type: {self.rope_type!r}")

    def validate_qwen_kv_shape(self) -> None:
        """检查直接 attention 到 Qwen prefix K/V 所需的不变量。"""
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                f"hidden_size must equal num_kv_heads * head_dim: "
                f"{self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads must equal num_kv_heads: {self.num_heads} != {self.num_kv_heads}"
            )
        if self.rope_type not in {"mrope", "mhrope", "none"}:
            raise ValueError(f"rope_type must be 'mrope', 'mhrope', or 'none': {self.rope_type!r}")
        if self.rope_type == "none":
            return
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}")
        if self.rope_type == "mrope":
            if sum(self.mrope_section_dim) != self.head_dim // 2:
                raise ValueError(
                    f"M-RoPE section sum {sum(self.mrope_section_dim)} must equal "
                    f"head_dim//2={self.head_dim // 2}"
                )
        else:
            if sum(self.mrope_section_head) > self.num_heads:
                raise ValueError(
                    f"MH-RoPE head section sum {sum(self.mrope_section_head)} "
                    f"cannot exceed num_heads={self.num_heads}"
                )
