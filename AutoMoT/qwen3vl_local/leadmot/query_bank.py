"""用于 route / waypoint 预测的可学习 query bank。

decoder 会把这些 learned token 拼在 BEV 和 status token 后面。
route head 读取 route query，waypoint head 读取 waypoint query。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _QueryBank(nn.Module):
    """基础 embedding table，forward 时按当前 batch 展开。"""

    def __init__(self, num_queries: int, hidden_size: int):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_queries, hidden_size)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """返回 ``(B, num_queries, hidden)``，不为每个样本额外分配参数。"""
        idx = torch.arange(self.num_queries, device=device or self.embed.weight.device)
        q = self.embed(idx).unsqueeze(0).expand(batch_size, -1, -1)
        if dtype is not None:
            q = q.to(dtype)
        return q


class RouteQueryBank(_QueryBank):
    """默认 10 个可学习 route query，对齐 LEAD route 标签。"""

    def __init__(self, num_queries: int = 10, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)


class WaypointQueryBank(_QueryBank):
    """默认 8 个可学习 waypoint query，对齐 CARLA 2 秒预测窗口。"""

    def __init__(self, num_queries: int = 8, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)
