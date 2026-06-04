"""LEAD-MoT 快推理 decoder 子包。

慢推理 (Qwen3-VL prefill) 由 `mot_lead_offline_runner.py` 调
`qwen3vl_local.engine.LocalQwen3VLInstructEngine` 完成，本子包消费 past_key_values。
训练通路（数据/loss/optimizer）不在本子包。

详见同目录 `ARCHITECTURE.md`。
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
