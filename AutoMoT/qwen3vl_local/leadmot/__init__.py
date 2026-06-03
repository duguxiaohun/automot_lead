"""LEAD-MoT 快推理 decoder 子包入口。

本子包做的事
============
- 给离线 runner 提供一个**纯快推理动作头**：吃 frozen Qwen3-VL 的 prefix K/V
  + LEAD BEV 特征 + ego 控制条件，输出对齐 LEAD `PlanningDecoder` 契约的
  `pred_route (B,10,2)` 和 `pred_future_waypoints (B,8,2)`。
- **慢推理（Qwen3-VL prefill）不在本子包**——由 `mot_lead_offline_runner.py`
  通过 AutoMoT `InterleaveInferencer.kv_cache_fixed_inference` 或
  `qwen3vl_local/engine.py` 完成；本子包只消费 `past_key_values`。
- **训练通路不在本子包**——损失函数、数据集、optimizer、训练 loop 由调用方
  另行实现；本子包只暴露 `nn.Module` 架构。

整体在仓库里的位置
================
LEAD pkl
  -> RGB/LiDAR/控制条件 (mot_lead_offline_runner.py)
  -> frozen Qwen3-VL prefill (AutoMoT InterleaveInferencer)
  -> past_key_values: 36 层 (K,V)
  -> _segment_qwen_cache_for_leadmot (runner) 池化成 12 段 prefix K/V
  -> ★ 本子包 LeadMoTPlanningDecoder 快推理 ★
  -> pred_route (B,10,2) + pred_future_waypoints (B,8,2)

为什么单独成子包
================
1. 跟 AutoMoT 自家 `bev_encoder_proj` / `route_head` / `waypoint_head` 解耦：
   AutoMoT 原快推理 head 用的是 (1,1512,8,8) BEV、20 点 route、6 点 waypoint，
   与 LEAD 训练分布不兼容。本子包是"LEAD 语义 + MoT 风格"的全新一套。
2. 跟 frozen Qwen3-VL 解耦：本子包不修改 Qwen 源码，所有可学参数都在本子包内。
3. 跟 `goalgen` 子包并列：goalgen 是 DiT + flow matching 的子目标 latent 生成
   路线；本子包是直接出 waypoint 的 planning 路线。两者公用 `qwen_kv.py` 的
   KV 池化思路。

设计要点（一句话版）
===================
- gen 路 hidden = 1024 = Qwen num_kv_heads (8) × head_dim (128)，让 frozen K/V
  能直接 concat 进 attention，**不需要任何线性投影**——这是用户偏好的核心。
- gen Q/K/V 自己投影（独立可学参数），frozen K/V 不投影。
- 12 层 gen decoder，每层用 Qwen 36 层中按 `select_last` 池化得到的一段 K/V。
- 输出 `Linear + cumsum`，跟 LEAD `PlanningDecoder.wp_decoder / route_decoder` 同构。

更多技术细节
============
详见同目录 `ARCHITECTURE.md`：
- §1 设计目标与输入/输出契约
- §2 与 AutoMoT 严格 MoT 的差异对照（含 attention 计算公式、12 维差异表、
  RoPE 缺失影响、参数初始化补救）
- §3 模块拓扑图
- §4 packed gen 序列 layout（141 token 的索引划分）
- §5 前向张量流（含 shape 全程标注）
- §6 不在本子包做的事（含 padding mask、cache 同源两条警告）
- §7 完整调用模板（含 prefill → segment_kv → decoder 三段式）
"""

# ============================================================
# 子包内的所有可学/纯 nn.Module 类按依赖顺序逐一导出，方便上层 runner /
# 训练脚本一次性 import。导出顺序与 ARCHITECTURE.md §3 模块拓扑一致。
# ============================================================

# 配置类：所有维度、层数、归一化参数都在这里集中管理
from .config import LeadMoTPlanningDecoderConfig

# 输入投影：BEV 特征升维 + 速度/目标点编码
from .projectors import LeadBEVProjector, StatusTokenEncoder

# 可学 query：route 10 个 + waypoint 8 个，分别对齐 LEAD 两条输出契约
from .query_bank import RouteQueryBank, WaypointQueryBank

# 输出头：单层 Linear + cumsum，照 LEAD planning_decoder 风格
from .heads import RouteHead, WaypointHead

# gen 路 decoder block：含 RMSNorm + Qwen3 风格 q/k_norm + prefix-KV attention + SwiGLU
from .mot_block import MoTDecoderBlock, PrefixKVAttention, RMSNorm

# 顶层组装：把上面所有模块串成完整 nn.Module
from .decoder import LeadMoTPlanningDecoder

# __all__ 严格限定外部可见符号，避免 `from leadmot import *` 把内部 helper 拖出去
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
