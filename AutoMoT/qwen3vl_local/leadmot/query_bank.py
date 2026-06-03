"""可学 query embedding 表。

对齐 LEAD `PlanningDecoder` 的 query 设计：一个可学 query 序列 expand 到 batch。
route 和 waypoint 分成两个独立 bank，方便后续分别扩展或冻结。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _QueryBank(nn.Module):
    """通用 query bank。"""

    def __init__(self, num_queries: int, hidden_size: int):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_queries, hidden_size)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """返回 (B, num_queries, hidden) 的 query embedding。"""
        idx = torch.arange(self.num_queries, device=device or self.embed.weight.device)
        q = self.embed(idx).unsqueeze(0).expand(batch_size, -1, -1)
        if dtype is not None:
            q = q.to(dtype)
        return q


class RouteQueryBank(_QueryBank):
    """LEAD route query：默认 10 个，对齐 num_route_points_prediction。"""

    def __init__(self, num_queries: int = 10, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)


class WaypointQueryBank(_QueryBank):
    """LEAD waypoint query：默认 8 个，对齐 4Hz x 2s。"""

    def __init__(self, num_queries: int = 8, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)
