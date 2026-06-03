"""LEAD-MoT planning decoder 子包。

设计目标：
- 慢推理由冻结 Qwen3-VL 提供 prefix K/V（在本子包外完成）。
- 快推理（本子包）按 MoT 风格在 gen 路上挂可学 query，并直接读取 frozen Qwen K/V，
  输出张量天然对齐
  LEAD `PlanningDecoder`：
    * pred_route             (B, 10, 2)  ego-local 米，对齐 data["route"]
    * pred_future_waypoints  (B,  8, 2)  ego-local 米，对齐 data["future_waypoints"]
- 不含训练代码（数据、loss、optimizer 等）。

详见同目录 ARCHITECTURE.md。
"""

from .config import LeadMoTPlanningDecoderConfig
from .projectors import LeadBEVProjector, StatusTokenEncoder
from .query_bank import RouteQueryBank, WaypointQueryBank
from .heads import RouteHead, WaypointHead
from .mot_block import MoTDecoderBlock, PrefixKVAttention, RMSNorm
from .decoder import LeadMoTPlanningDecoder

__all__ = [
    "LeadMoTPlanningDecoderConfig",
    "LeadBEVProjector",
    "StatusTokenEncoder",
    "RouteQueryBank",
    "WaypointQueryBank",
    "RouteHead",
    "WaypointHead",
    "MoTDecoderBlock",
    "PrefixKVAttention",
    "RMSNorm",
    "LeadMoTPlanningDecoder",
]
