"""LEAD 风格输出头：Linear(hidden -> 2) + cumsum。

跟 LEAD `PlanningDecoder.wp_decoder / route_decoder` 同构（单层 Linear，不上 MLP）。
模型预测相邻点 delta，cumsum 还原成 ego-local 累计坐标。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DeltaCumsumHead(nn.Module):
    """通用 (B, N, hidden) -> Linear(hidden, 2) -> cumsum -> (B, N, 2)。"""

    def __init__(self, hidden_size: int, point_dim: int = 2):
        super().__init__()
        self.point_dim = point_dim
        self.proj = nn.Linear(hidden_size, point_dim)
        # bias=0 保证训练初期预测从原点起步
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        # delta 在 bf16 下计算后，cumsum 必须升到 fp32 累加，否则远端点（~30m）累计误差可达
        # 0.1~0.25m 量级，直接顶死 FDE 精度天花板。这里输出保持 fp32 不再回退 bf16，
        # 下游 loss / metric 本来就用 fp32，避免末步重新量化。
        delta = self.proj(query_hidden)
        return torch.cumsum(delta.float(), dim=1)


class RouteHead(_DeltaCumsumHead):
    """(B, 10, hidden) -> (B, 10, 2)，对齐 LEAD data["route"]。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)


class WaypointHead(_DeltaCumsumHead):
    """(B, 8, hidden) -> (B, 8, 2)，对齐 LEAD data["future_waypoints"]。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)
