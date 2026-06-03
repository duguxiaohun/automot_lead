"""LeadMoTPlanningDecoder：完整的 LEAD-MoT 快推理 decoder。

这是子包的顶层组装类，把 config / projectors / query_bank / mot_block / heads
串成一个完整 nn.Module。下游 runner / 训练脚本只需要 import 这一个类。

接口约定
========

输入 (forward kwargs):
    pooled_kv         : list[(K, V)]，长度必须 = num_layers (默认 12)
                        每个 K/V 形状 (B 或 1, 8, S, 128)
                        来源: frozen Qwen3-VL prefill -> past_key_values (36 层)
                        -> segment_kv_for_dit(num_segments=12, mode='select_last')
                        -> 取每段最后一层的 K/V
                        调用方应该在 runner / 训练脚本里完成这一步，
                        本子包不做 prefill 也不做池化（耦合最小化）。
    bev               : (B, 512, 10, 12)  LEAD LeadBEVEncoder 输出
    speed             : (B,) 或 (B, 1)    m/s
    target_point      : (B, 2)            ego frame 米制坐标
    target_point_next : (B, 2)            ego frame 米制坐标

输出 (dict):
    pred_route             : (B, 10, 2)  ego-local 米制累计坐标，对齐 LEAD data["route"]
    pred_future_waypoints  : (B,  8, 2)  ego-local 米制累计坐标，对齐 LEAD data["future_waypoints"]
    gen_hidden             : (B, 141, 1024)  整段 gen 路的最终 hidden（调试/扩展用）

【当前版本是 MoT 形态的工程简化实现】
======================================
- gen 路独立 12 层 transformer，每层用一段 frozen Qwen prefix K/V
- **不是**严格 MoT 投影分流（严格 MoT 在 Qwen LM 36 层内做 q/k/v_proj_mot_gen 分流）
- 两者 attention 数学等价，但严格 MoT 信息融合更细。详见 ARCHITECTURE.md §2

为什么调用方负责池化
====================
- 池化 (segment_kv_for_dit) 需要知道 Qwen 层数和池化策略（select_last/mean/concat），
  这些信息要 frozen Qwen 实例自己提供，本子包不持有 Qwen
- 让调用方做池化，方便统一管理 Qwen 实例和 cache 生命周期
- goalgen 子包也是这个分工（DiT 不持有 Qwen，由 train_v1.py 等调用方做池化）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import LeadMoTPlanningDecoderConfig
from .heads import RouteHead, WaypointHead
from .mot_block import MoTDecoderBlock, RMSNorm
from .projectors import LeadBEVProjector, StatusTokenEncoder
from .query_bank import RouteQueryBank, WaypointQueryBank


class LeadMoTPlanningDecoder(nn.Module):
    """LEAD-MoT 快推理 decoder。

    输出契约严格按 LEAD：route(10) + waypoint(8) + Linear+cumsum。
    架构读 frozen Qwen prefix K/V，语言 K/V 不做线性投影。

    典型用法（更详细的三段式调用见 ARCHITECTURE.md §7）:

        cfg = LeadMoTPlanningDecoderConfig()
        decoder = LeadMoTPlanningDecoder(cfg).cuda().to(torch.bfloat16).eval()

        # 调用方先跑 Qwen prefill + 池化（见 runner._segment_qwen_cache_for_leadmot）
        pooled_kv = ...

        out = decoder(
            pooled_kv=pooled_kv,
            bev=bev,                   # (B, 512, 10, 12)
            speed=speed,               # (B,)
            target_point=tp,           # (B, 2)
            target_point_next=ntp,     # (B, 2)
        )
        # out["pred_route"]            : (B, 10, 2)
        # out["pred_future_waypoints"] : (B,  8, 2)
    """

    def __init__(self, config: Optional[LeadMoTPlanningDecoderConfig] = None):
        """
        参数:
            config: 可选的配置实例。不传则用默认（LEAD CARLA 模式）。
                    所有维度、层数、归一化参数都从 config 读取，
                    本类自己不硬编码任何数值。
        """
        super().__init__()

        # 允许调用方不传 config（使用默认配置），方便快速试验
        self.config = config or LeadMoTPlanningDecoderConfig()

        # 立刻校验配置自洽性：hidden_size == num_kv_heads * head_dim 等
        # 配置错的话，这里 raise 比延迟到 attention 内部 reshape 报错更友好
        self.config.validate_qwen_kv_shape()

        # 局部变量便于阅读后面的初始化代码
        cfg = self.config

        # ============================================================
        # 输入投影模块
        # ============================================================
        # BEV 卷积特征 (B,512,10,12) -> (B,120,hidden)
        self.bev_projector = LeadBEVProjector(cfg)
        # ego status: speed + target_point + target_point_next -> 3 个 (B,1,hidden) token
        self.status_encoder = StatusTokenEncoder(cfg)

        # ============================================================
        # 可学 query embedding bank
        # ============================================================
        # route 10 个 query，对应 LEAD data["route"] 的 10 个空间路径点
        self.route_query_bank = RouteQueryBank(
            num_queries=cfg.num_route_queries,
            hidden_size=cfg.hidden_size,
        )
        # waypoint 8 个 query，对应 LEAD data["future_waypoints"] 的 8 个时间轨迹点
        self.waypoint_query_bank = WaypointQueryBank(
            num_queries=cfg.num_waypoint_queries,
            hidden_size=cfg.hidden_size,
        )

        # ============================================================
        # gen 路 decoder block 堆叠（12 层）
        # ============================================================
        # 每层都是独立的 MoTDecoderBlock（独立参数），按 ModuleList 管理方便循环
        # forward 时 zip(self.blocks, pooled_kv) 一一配对：第 i 层用 pooled_kv[i]
        self.blocks = nn.ModuleList(
            [
                MoTDecoderBlock(
                    hidden_size=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    ffn_hidden_size=cfg.ffn_hidden_size,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # 最后一层 norm：进入 head 之前再做一次归一化（标准 Transformer 做法）
        self.gen_final_norm = RMSNorm(cfg.hidden_size, eps=1e-6, elementwise_affine=True)

        # ============================================================
        # 输出头：单层 Linear + cumsum
        # ============================================================
        # route head: (B,10,hidden) -> (B,10,2)
        self.route_head = RouteHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)
        # waypoint head: (B,8,hidden) -> (B,8,2)
        self.waypoint_head = WaypointHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)

    # ============================================================
    def _build_gen_sequence(
        self,
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> torch.Tensor:
        """把所有 gen 路输入拼成 packed gen 序列 (B, L_gen, hidden)。

        拼接顺序必须严格匹配 LeadMoTPlanningDecoderConfig.slice_layout()：
            BEV | speed | tp | ntp | route_q | waypoint_q

        否则下游切片会取错位置的 hidden 喂 head。

        参数:
            bev:                (B, 512, 10, 12)
            speed:              (B,) 或 (B, 1)
            target_point:       (B, 2)
            target_point_next:  (B, 2)
        返回:
            (B, total_gen_tokens=141, hidden=1024) 拼好的 gen 段序列
        """
        # 从 bev 取 batch size，假设其它输入也是这个 batch（projectors 会校验形状）
        batch_size = bev.shape[0]

        # ---- BEV 段：120 个 token ----
        # 走 LeadBEVProjector：flatten + Linear + 2D pos embedding
        bev_tokens = self.bev_projector(bev)

        # ---- status 段：3 个 token（speed + tp + ntp）----
        # status_encoder 分两次调用：speed 单独，tp/ntp 共享 encoder 一次出两个
        speed_tok = self.status_encoder.encode_speed(speed)
        tp_tok, ntp_tok = self.status_encoder.encode_target_points(
            target_point,
            target_point_next,
        )

        # ---- query 段：route 10 + waypoint 8 个 token ----
        # 跟 bev_tokens 同 device/dtype，避免 attention 时 dtype 不一致
        # device/dtype 从 bev_tokens 取（它已经经过 projector，肯定是 decoder 的 device/dtype）
        route_q = self.route_query_bank(
            batch_size,
            device=bev_tokens.device,
            dtype=bev_tokens.dtype,
        )
        wp_q = self.waypoint_query_bank(
            batch_size,
            device=bev_tokens.device,
            dtype=bev_tokens.dtype,
        )

        # ---- 拼接 ----
        # dim=1 是 token 维（dim=0 是 batch，dim=2 是 hidden）
        # 顺序: BEV(120) | speed(1) | tp(1) | ntp(1) | route_q(10) | waypoint_q(8) = 141
        gen_seq = torch.cat([bev_tokens, speed_tok, tp_tok, ntp_tok, route_q, wp_q], dim=1)

        # ---- 长度校验 ----
        # 与 config.total_gen_tokens() 对比，如果不匹配说明上面 cat 顺序或 query 数错了
        if gen_seq.shape[1] != self.config.total_gen_tokens():
            raise RuntimeError(
                f"gen seq length mismatch: got {gen_seq.shape[1]}, "
                f"expect {self.config.total_gen_tokens()}"
            )
        return gen_seq

    # ============================================================
    def _check_pooled_kv(
        self,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """校验 pooled_kv 的层数和每层形状是否正确。

        早校验比延迟到 attention 内部 reshape 报错更友好——后者堆栈会指向
        SDPA 而不是这里，难以 debug。

        参数:
            pooled_kv: 调用方提供的池化后 K/V 列表
        """
        # ---- 段数校验 ----
        # 必须正好 num_layers 段（默认 12），每段一对 (K, V)
        # 不对的话立即 raise，避免 zip(self.blocks, pooled_kv) 静默截断
        if len(pooled_kv) != self.config.num_layers:
            raise ValueError(
                f"pooled_kv 段数 {len(pooled_kv)} 与 decoder 层数 "
                f"{self.config.num_layers} 不一致"
            )

        # ---- 全部 num_layers 段都逐层校验 ----
        # 不只查 pooled_kv[0]，因为不同段可能 shape 不同（少见但可能，比如调用方
        # 池化时有 bug 只对前几层做了池化）。逐层 enumerate 拿到 index 方便定位
        for i, (k, v) in enumerate(pooled_kv):
            # K 和 V 必须严格同形
            if k.shape != v.shape:
                raise ValueError(
                    f"pooled_kv[{i}] K/V shape 不一致：{tuple(k.shape)} vs {tuple(v.shape)}"
                )
            # 必须是 4D：[B, H, S, D]
            if k.ndim != 4:
                raise ValueError(
                    f"pooled_kv[{i}] K 应为 [B,H,S,D]，实际 {tuple(k.shape)}"
                )
            # heads 维（第 1 维）和 head_dim 维（第 3 维）必须匹配 config
            # 这是 prefix-KV 能直接拼接的核心约束（见 PrefixKVAttention.__init__）
            if k.shape[1] != self.config.num_kv_heads or k.shape[3] != self.config.head_dim:
                raise ValueError(
                    f"pooled_kv[{i}] K 形状 {tuple(k.shape)} 与 "
                    f"(heads={self.config.num_kv_heads}, head_dim={self.config.head_dim}) 不匹配"
                )

    # ============================================================
    def forward(
        self,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """完整前向，从 BEV/status 输入和 frozen K/V 出到 LEAD 风格 route + waypoint。

        参数:
            pooled_kv:          list[(K, V)] 长度 num_layers，每个 [B 或 1, 8, S, 128]
            bev:                (B, 512, 10, 12)
            speed:              (B,) 或 (B, 1)
            target_point:       (B, 2)
            target_point_next:  (B, 2)
        返回:
            {
                "pred_route":            (B, 10, 2),
                "pred_future_waypoints": (B,  8, 2),
                "gen_hidden":            (B, 141, 1024),
            }

        说明:
            - pred_route / pred_future_waypoints 是 ego-local 米制累计坐标
              （已经过 cumsum），可以直接喂 L1 / ADE / 控制器
            - gen_hidden 是最后一层的完整 gen 段输出，主要给调试/扩展用：
                * 切别的段加新 head
                * 做 attention map 可视化
                * 训练时正则化项
              下游不消费可以忽略
        """
        # ---- 1. 提前校验 pooled_kv 形状 ----
        # 早校验，让形状错误的报错堆栈直接指向这里，不要延迟到 attention 内部
        self._check_pooled_kv(pooled_kv)

        # ---- 2. 构建 gen 段输入序列 ----
        # (B, 141, 1024) 的张量，按 layout 排好序
        gen_seq = self._build_gen_sequence(
            bev=bev,
            speed=speed,
            target_point=target_point,
            target_point_next=target_point_next,
        )

        # ---- 3. 12 层 decoder block 主循环 ----
        # 第 i 层用 pooled_kv[i] 作 prefix K/V
        # zip 隐含 len(self.blocks) == len(pooled_kv) == config.num_layers
        # （前面 _check_pooled_kv 已保证）
        for block, lang_kv in zip(self.blocks, pooled_kv):
            # 每个 block 内部: RMSNorm -> PrefixKVAttention -> +residual -> RMSNorm -> SwiGLU -> +residual
            gen_seq = block(gen_seq=gen_seq, lang_kv=lang_kv)

        # ---- 4. 最后一层 RMSNorm ----
        # 进入输出头前做一次归一化（标准 Transformer 做法，让 head 输入分布稳定）
        gen_seq = self.gen_final_norm(gen_seq)

        # ---- 5. 按 layout 切片出 query 段的 hidden ----
        # config.slice_layout() 返回每一类 token 的 [start, end) 索引
        # 这是 packed gen 序列布局的唯一真值，避免在 forward 里硬编码 123/133/141
        layout = self.config.slice_layout()
        r_start, r_end = layout["route"]
        w_start, w_end = layout["waypoint"]

        # 切出 route 段对应的 hidden: (B, 10, 1024)
        route_hidden = gen_seq[:, r_start:r_end, :]
        # 切出 waypoint 段对应的 hidden: (B, 8, 1024)
        wp_hidden = gen_seq[:, w_start:w_end, :]

        # ---- 6. 输出头（Linear + cumsum）----
        # 内部都是 Linear(hidden, 2) -> cumsum(dim=1)，输出 ego-local 累计米制坐标
        pred_route = self.route_head(route_hidden)
        pred_future_waypoints = self.waypoint_head(wp_hidden)

        # ---- 7. 返回 dict ----
        # 用 dict 是为了便于扩展：未来加 heading head / 速度 head / 中间 hidden 输出
        # 时不需要改 forward 签名
        return {
            "pred_route": pred_route,
            "pred_future_waypoints": pred_future_waypoints,
            "gen_hidden": gen_seq,
        }
