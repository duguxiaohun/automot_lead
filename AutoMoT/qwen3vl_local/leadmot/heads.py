"""LEAD 风格输出头：Linear(hidden -> 2) + cumsum。

完全对齐 LEAD `PlanningDecoder` 的 wp_decoder / route_decoder 语义：
模型预测相邻点 delta，再用 cumsum 累计成 ego-local 绝对位置。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DeltaCumsumHead(nn.Module):
    """通用 (B, N, hidden) -> (B, N, 2) 累计输出头。"""

    def __init__(self, hidden_size: int, point_dim: int = 2):
        super().__init__()
        self.point_dim = point_dim
        self.proj = nn.Linear(hidden_size, point_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        delta = self.proj(query_hidden)
        return torch.cumsum(delta, dim=1)


class RouteHead(_DeltaCumsumHead):
    """输出 (B, 10, 2)。对齐 LEAD data["route"]。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)


class WaypointHead(_DeltaCumsumHead):
    """输出 (B, 8, 2)。对齐 LEAD data["future_waypoints"]。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)
