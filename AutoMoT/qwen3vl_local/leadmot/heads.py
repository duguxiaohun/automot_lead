"""Planning 输出头。

两个 LEAD head 都刻意保持简单：``Linear(hidden -> 2)`` 先预测每一步 delta，
再用 ``cumsum`` 转成 ego-frame 累计点。因此训练标签是绝对/累计 ego-frame 点，
不是相邻点之间的 delta。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DeltaCumsumHead(nn.Module):
    """共享的 ``(B, N, hidden) -> (B, N, 2)`` delta-to-point head。"""

    def __init__(self, hidden_size: int, point_dim: int = 2):
        super().__init__()
        self.point_dim = point_dim
        self.proj = nn.Linear(hidden_size, point_dim)
        # bias 初始化为 0，让初始路径尽量贴近 ego 原点附近。
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        """预测点间 delta，并用 fp32 累计，降低远端 FDE 数值误差。"""
        delta = self.proj(query_hidden)
        # cumsum 保持 fp32：bf16 连续累计到轨迹末端时可能积累可见的米级舍入误差。
        return torch.cumsum(delta.float(), dim=1)


class RouteHead(_DeltaCumsumHead):
    """Route head：默认 ``(B, 10, hidden) -> (B, 10, 2)``。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)


class WaypointHead(_DeltaCumsumHead):
    """Waypoint head：默认 ``(B, 8, hidden) -> (B, 8, 2)``。"""

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)
