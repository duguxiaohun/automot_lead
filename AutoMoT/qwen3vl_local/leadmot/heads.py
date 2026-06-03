"""LEAD 风格输出头：Linear(hidden -> 2) + cumsum。

为什么是单层 Linear，不是 MLP
=============================
LEAD `PlanningDecoder.wp_decoder` 和 `route_decoder`（位于
`lead/lead/tfv7/planning_decoder.py`）的实现就是单层 `nn.Linear(hidden, 2)`。
我们这一版要的就是"输出语义对齐 LEAD"，所以不上 MLP，把头部的灵活性集中到
前面 12 层 transformer 的 hidden 表达里。

为什么要 cumsum
================
LEAD 模型实际预测的不是绝对坐标，而是 **相邻点之间的位移 delta**：
    delta[i] = pred_pos[i] - pred_pos[i-1]
然后 cumsum 累计：
    pred_pos = cumsum(delta, dim=1)
得到 ego-local 坐标系下的绝对位置序列。

这样设计的好处：
1. delta 的数值范围比绝对坐标更稳定，训练初期更容易收敛
2. 每个 query 只学"相对上一步该走多远"，物理意义更清晰
3. 跟 LEAD 完全一致，方便直接套 LEAD 评测 / 控制器

什么时候不用 cumsum
====================
如果将来想直接预测绝对位置（比如 navsim 输出风格），把 forward 里的
torch.cumsum 拿掉即可，其它都不用改。

输出单位约定
============
返回张量是 ego-local 米制坐标，与 LEAD `data["route"]` 和
`data["future_waypoints"]` 同坐标系，方便下游：
- 直接算 L1 loss 对齐 LEAD 真值
- 直接算 ADE/FDE（Average/Final Displacement Error）
- 直接喂 LEAD 的 PID 控制器（如果做闭环）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DeltaCumsumHead(nn.Module):
    """通用 (B, N, hidden) -> (B, N, 2) 累计输出头。

    RouteHead 和 WaypointHead 都继承自这个 base，唯一区别只是 N 不同
    （route=10, waypoint=8）。base 实现里不固定 N，由调用方传入的 hidden 形状决定。
    """

    def __init__(self, hidden_size: int, point_dim: int = 2):
        """
        参数:
            hidden_size: query hidden 维度，必须与 decoder 的 gen hidden 一致（默认 1024）
            point_dim:   每个 waypoint 输出几个坐标分量，默认 2 即 (x, y)
        """
        super().__init__()
        # 记录 point_dim 是为了让上层断言 / 评测代码可以从 head 实例拿到这个值，
        # 不必去查 config
        self.point_dim = point_dim

        # 单层 Linear：hidden -> point_dim，bias=True（LEAD 原版也带 bias）
        self.proj = nn.Linear(hidden_size, point_dim)

        # 初始化策略：weight 用截断正态分布 std=0.02（Transformer 系标准），
        # bias 显式置零保证训练初期"delta = 0 -> 累计位置 = 0"，预测从原点起步，
        # 避免初始 epoch 模型输出大幅偏离 GT 拖慢收敛
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query_hidden: (B, N, hidden) 来自 decoder.gen_seq 切片，
                          已经过 num_layers 层 prefix-KV attention + 最终 RMSNorm
        返回:
            (B, N, point_dim) ego-local 米制坐标，累计形式（cumsum）
        """
        # 第一步：线性映射 hidden -> 2，得到每个 query 对应的"相邻点位移 delta"
        # 形状 (B, N, hidden) -> (B, N, 2)
        delta = self.proj(query_hidden)

        # 第二步：沿 query 维 (dim=1) 做累加，把 delta 序列变成绝对坐标序列
        # 数学上：pred[i] = sum_{j<=i} delta[j]
        # 这一步与 LEAD 完全同构（见 lead/lead/tfv7/planning_decoder.py 的
        # `torch.cumsum(self.wp_decoder(...), 1)`）
        return torch.cumsum(delta, dim=1)


class RouteHead(_DeltaCumsumHead):
    """LEAD route 输出头：吃 10 个 query hidden，出 (B, 10, 2)。

    对齐 LEAD `data["route"]`：
    - 10 个空间路径点（不是时间轨迹）
    - ego-local 米制
    - 通常用作横向参考线给下游 PID 跟随

    与 AutoMoT route head 的关键差异：AutoMoT 的 route_head 出 20 个点，
    我们这版输出 10 个点（对齐 LEAD CARLA 默认 num_route_points_prediction=10）。
    """

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        # hidden_size 默认 1024 对应 LeadMoTPlanningDecoderConfig.hidden_size，
        # 但允许覆盖（比如未来调成 2560 接 Qwen 主干 hidden 时）
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)


class WaypointHead(_DeltaCumsumHead):
    """LEAD waypoint 输出头：吃 8 个 query hidden，出 (B, 8, 2)。

    对齐 LEAD `data["future_waypoints"]`：
    - 8 个时间轨迹点
    - 间隔 0.25s（来自 LEAD waypoints_spacing=5 @ 20Hz CARLA = 4Hz）
    - 总时长 2.0s
    - ego-local 米制
    - 通常用作速度参考给下游纵向 PID

    与 AutoMoT waypoint head 的关键差异：AutoMoT 出 6 个点 @ 0.5s 间隔（3s 总时长），
    我们这版按 LEAD CARLA 模式出 8 个点 @ 0.25s 间隔（2s 总时长），
    天然对齐 LEAD 真值，省去时间插值。
    """

    def __init__(self, hidden_size: int = 1024, point_dim: int = 2):
        super().__init__(hidden_size=hidden_size, point_dim=point_dim)
