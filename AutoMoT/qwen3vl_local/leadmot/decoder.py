"""LeadMoTPlanningDecoder：顶层组装，串起 projectors / query_bank / mot_block / heads。

forward 接口:
    pooled_kv         : list[(K, V)] 长度 = num_layers (=12)
                        每个 K/V (B 或 1, 8, S, 128)
                        来源 frozen Qwen prefill -> segment_kv_for_dit(num_segments=12, mode='select_last')
                        调用方负责池化（详见 ARCHITECTURE.md §7）
    bev               : (B, 512, 10, 12)
    speed             : (B,) 或 (B, 1)
    target_point      : (B, 2)
    target_point_next : (B, 2)

return:
    pred_route             : (B, 10, 2) ego-local 米，对齐 LEAD data["route"]
    pred_future_waypoints  : (B,  8, 2) ego-local 米，对齐 LEAD data["future_waypoints"]
    gen_hidden             : (B, 141, 1024) 调试/扩展用

当前是 MoT 形态的工程简化实现（独立 12 层 transformer 读 frozen K/V），不是严格 MoT
（严格 MoT 在 Qwen LM 36 层内做 q/k/v_proj_mot_gen 分流）。详见 ARCHITECTURE.md §2。
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
    """LEAD-MoT 快推理 decoder（详见模块 docstring）。"""

    def __init__(self, config: Optional[LeadMoTPlanningDecoderConfig] = None):
        super().__init__()
        self.config = config or LeadMoTPlanningDecoderConfig()
        # 提前校验配置自洽，避免延迟到 attention 内部报错
        self.config.validate_qwen_kv_shape()
        cfg = self.config

        # 输入投影
        self.bev_projector = LeadBEVProjector(cfg)
        self.status_encoder = StatusTokenEncoder(cfg)

        # 可学 query
        self.route_query_bank = RouteQueryBank(
            num_queries=cfg.num_route_queries,
            hidden_size=cfg.hidden_size,
        )
        self.waypoint_query_bank = WaypointQueryBank(
            num_queries=cfg.num_waypoint_queries,
            hidden_size=cfg.hidden_size,
        )

        # gen 路 12 层 decoder，每层用一段 pooled_kv
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
        self.gen_final_norm = RMSNorm(cfg.hidden_size, eps=1e-6, elementwise_affine=True)

        # 输出头
        self.route_head = RouteHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)
        self.waypoint_head = WaypointHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)

    def _build_gen_sequence(
        self,
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> torch.Tensor:
        """拼成 packed gen 序列 (B, 141, hidden)，顺序必须匹配 config.slice_layout。"""
        batch_size = bev.shape[0]
        bev_tokens = self.bev_projector(bev)
        speed_tok = self.status_encoder.encode_speed(speed)
        tp_tok, ntp_tok = self.status_encoder.encode_target_points(
            target_point, target_point_next,
        )
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
        # 顺序: BEV(120) | speed(1) | tp(1) | ntp(1) | route_q(10) | wp_q(8) = 141
        gen_seq = torch.cat([bev_tokens, speed_tok, tp_tok, ntp_tok, route_q, wp_q], dim=1)
        if gen_seq.shape[1] != self.config.total_gen_tokens():
            raise RuntimeError(
                f"gen seq length mismatch: got {gen_seq.shape[1]}, "
                f"expect {self.config.total_gen_tokens()}"
            )
        return gen_seq

    def _check_pooled_kv(
        self,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """提前校验所有 num_layers 段的 K/V 形状，避免延迟到 attention 才报错。"""
        if len(pooled_kv) != self.config.num_layers:
            raise ValueError(
                f"pooled_kv 段数 {len(pooled_kv)} 与 decoder 层数 "
                f"{self.config.num_layers} 不一致"
            )
        for i, (k, v) in enumerate(pooled_kv):
            if k.shape != v.shape:
                raise ValueError(
                    f"pooled_kv[{i}] K/V shape 不一致：{tuple(k.shape)} vs {tuple(v.shape)}"
                )
            if k.ndim != 4:
                raise ValueError(
                    f"pooled_kv[{i}] K 应为 [B,H,S,D]，实际 {tuple(k.shape)}"
                )
            if k.shape[1] != self.config.num_kv_heads or k.shape[3] != self.config.head_dim:
                raise ValueError(
                    f"pooled_kv[{i}] K 形状 {tuple(k.shape)} 与 "
                    f"(heads={self.config.num_kv_heads}, head_dim={self.config.head_dim}) 不匹配"
                )

    def forward(
        self,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._check_pooled_kv(pooled_kv)
        gen_seq = self._build_gen_sequence(
            bev=bev, speed=speed,
            target_point=target_point, target_point_next=target_point_next,
        )

        # 第 i 层 block 用 pooled_kv[i] 作 prefix K/V
        for block, lang_kv in zip(self.blocks, pooled_kv):
            gen_seq = block(gen_seq=gen_seq, lang_kv=lang_kv)
        gen_seq = self.gen_final_norm(gen_seq)

        # 按 slice_layout 切出 route / waypoint 段 hidden
        layout = self.config.slice_layout()
        r_start, r_end = layout["route"]
        w_start, w_end = layout["waypoint"]
        route_hidden = gen_seq[:, r_start:r_end, :]
        wp_hidden = gen_seq[:, w_start:w_end, :]

        return {
            "pred_route": self.route_head(route_hidden),
            "pred_future_waypoints": self.waypoint_head(wp_hidden),
            "gen_hidden": gen_seq,
        }
