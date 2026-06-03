"""可学 query embedding 表。

什么是"query bank"
==================
在 transformer decoder 出动作/轨迹的范式里（DETR / LEAD PlanningDecoder /
AutoMoT 快推理 head），输出头不是从 sequence 末尾的 token 自回归生成，而是
预先准备好一组"可学习的查询 token"，让 decoder 通过 cross-attention 让这些
query 去"问"图像/语言/BEV 上下文。每个 query 最后输出一个目标点。

LEAD `PlanningDecoder` 就是这套：
    self.query = nn.Embedding(num_queries, hidden_size)
    decoder(query.expand(B,...), context)
    -> 切出 route_queries / wp_queries 各自喂 head

我们这边做了什么
================
- **拆成两个独立 bank**：route 和 waypoint 各自一个 `nn.Embedding`，方便：
  * 单独冻结其中一个（比如只训 waypoint 不动 route）
  * 单独换数量（navsim 模式 waypoint=8 但 route 还想保 10 时）
  * 单独打印/可视化每组 query 的 attention map
- **expand 而非 repeat**：forward 时把 (num_queries, hidden) -> (1, num_queries, hidden)
  -> expand 到 (B, num_queries, hidden)。expand 不开新显存，所有 batch 共享同一份
  embedding 内存，是标准做法。
- **可选 dtype 强转**：forward 接 dtype 参数，如果 decoder 跑 bf16，
  调用方就传 dtype=torch.bfloat16，query 也跟着转 bf16，避免 attention 时
  dtype mismatch。

为什么默认 hidden_size=1024
===========================
跟 LeadMoTPlanningDecoderConfig.hidden_size 默认值对齐。这是 Qwen3-VL-4B
num_kv_heads * head_dim = 8 * 128。允许覆盖以支持未来调整。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _QueryBank(nn.Module):
    """通用 query bank base 类。

    本身不暴露给外部使用（前缀 `_`），让 RouteQueryBank / WaypointQueryBank
    继承它，只改默认 num_queries。
    """

    def __init__(self, num_queries: int, hidden_size: int):
        """
        参数:
            num_queries: query 数量。对应 LEAD CARLA: route=10, waypoint=8
            hidden_size: 每个 query 的 embedding 维度，需与 decoder hidden 一致
        """
        super().__init__()
        # 保存配置，方便 debug 时查看（self.num_queries / self.hidden_size）
        self.num_queries = num_queries
        self.hidden_size = hidden_size

        # nn.Embedding(N, D) 等价于一个 (N, D) 的可学参数表
        # 比 nn.Parameter(torch.zeros(N, D)) 多了 sparse gradient 支持，
        # 但本场景 N 很小（10 或 8），主要好处是表达"按索引取 embedding"语义清晰
        self.embed = nn.Embedding(num_queries, hidden_size)

        # 截断正态初始化 std=0.02 是 Transformer 系（BERT/GPT/LLaMA/Qwen）的标准
        # 不要用默认的 Embedding 初始化（uniform），那个范围对深层网络太大
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """生成 (B, num_queries, hidden) 的 query embedding。

        参数:
            batch_size: 当前 batch 的 B（通常 1，离线推理是单样本）
            device:     可选，强制把 query 放到这个 device。默认跟 embed.weight 同设备
            dtype:      可选，强制把 query 转成这个 dtype。默认跟 embed.weight 同 dtype
                        decoder 跑 bf16 时调用方传 dtype=torch.bfloat16 进来

        返回:
            (B, num_queries, hidden_size) 张量。各 batch 共享同一份 embedding（expand 没拷贝），
            梯度回传时会自动累加到 embed 上。
        """
        # 准备索引 [0, 1, ..., num_queries-1]，让 Embedding 查表
        # 如果调用方没指定 device，就跟 embed 的权重同设备
        idx = torch.arange(self.num_queries, device=device or self.embed.weight.device)

        # 查表得到 (num_queries, hidden)，unsqueeze(0) 加上 batch 维 -> (1, num_queries, hidden)
        # expand 到 (B, num_queries, hidden)：不开新内存，把第 0 维"虚拟广播"成 B
        # 这是 DETR / LEAD 标准做法
        q = self.embed(idx).unsqueeze(0).expand(batch_size, -1, -1)

        # 如果调用方指定了 dtype（比如 bf16），强转一次
        # 注意：这会从 expand 后的张量再创建一份连续内存的新张量，
        # 但因为 N 很小（10 或 8），代价可忽略
        if dtype is not None:
            q = q.to(dtype)
        return q


class RouteQueryBank(_QueryBank):
    """LEAD route query bank：默认 10 个。

    对应 LEAD `data["route"]` 10 个空间路径点。
    输出张量经 decoder forward 后被切片喂 RouteHead，最终出 (B, 10, 2)。
    """

    def __init__(self, num_queries: int = 10, hidden_size: int = 1024):
        # 默认 num_queries=10 对齐 LEAD CARLA num_route_points_prediction
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)


class WaypointQueryBank(_QueryBank):
    """LEAD waypoint query bank：默认 8 个。

    对应 LEAD `data["future_waypoints"]` 8 个时间轨迹点（4Hz × 2s）。
    输出张量经 decoder forward 后被切片喂 WaypointHead，最终出 (B, 8, 2)。
    """

    def __init__(self, num_queries: int = 8, hidden_size: int = 1024):
        # 默认 num_queries=8 对齐 LEAD CARLA num_way_points_prediction
        super().__init__(num_queries=num_queries, hidden_size=hidden_size)
