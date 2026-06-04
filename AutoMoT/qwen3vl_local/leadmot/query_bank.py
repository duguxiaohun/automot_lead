"""可学 query embedding 表。

route 和 waypoint 各一个独立 bank，便于分别冻结/换数量。
forward 用 expand 把单份 embedding 广播到 batch，不开新显存。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _QueryBank(nn.Module):
    def __init__(self, num_queries: int, hidden_size: int):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_queries, hidden_size)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """返回 (B, num_queries, hidden)，所有 batch 共享同一份 embedding (expand 不拷贝)。"""
        idx = torch.arange(self.num_queries, device=device or self.embed.weight.device)
        q = self.embed(idx).unsqueeze(0).expand(batch_size, -1, -1)
        if dtype is not None:
            q = q.to(dtype)
        return q


class RouteQueryBank(_QueryBank):
    """LEAD route：默认 10 个 query。"""

    def __init__(self, num_queries: int = 10, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)


class WaypointQueryBank(_QueryBank):
    """LEAD waypoint：默认 8 个 query（4Hz × 2s）。"""

    def __init__(self, num_queries: int = 8, hidden_size: int = 1024):
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)
