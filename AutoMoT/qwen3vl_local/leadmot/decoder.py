"""LeadMoT planning decoder 顶层模块。

decoder 从 runner 接收 frozen Qwen prefix K/V 和 frozen LEAD BEV feature。
它先构建 generated-token 序列，再跑 12 层 Prefix-KV decoder block，
最后用 LEAD 风格 head 读取 route / waypoint query 切片。
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
    """接在 frozen Qwen 和 frozen BEV 上训练的快推理 planning decoder。

    Forward 输入：
    - ``pooled_kv``：frozen Qwen K/V 的 12 个 segment，每层 block 用一个；
    - ``bev``：LEAD BEV feature map，形状 ``(B, 512, 10, 12)``；
    - ``speed``、``target_point``、``target_point_next``：ego 状态输入；
    - ``rope_position_offset``：来自 Qwen prefill 的 next-token 位置 offset。
    """

    def __init__(self, config: Optional[LeadMoTPlanningDecoderConfig] = None):
        super().__init__()
        self.config = config or LeadMoTPlanningDecoderConfig()
        # 提前检查：prefix K/V 会直接进 attention，不经过 Linear 投影兜底。
        self.config.validate_qwen_kv_shape()
        cfg = self.config

        self.bev_projector = LeadBEVProjector(cfg)
        self.status_encoder = StatusTokenEncoder(cfg)

        self.route_query_bank = RouteQueryBank(
            num_queries=cfg.num_route_queries,
            hidden_size=cfg.hidden_size,
        )
        self.waypoint_query_bank = WaypointQueryBank(
            num_queries=cfg.num_waypoint_queries,
            hidden_size=cfg.hidden_size,
        )

        # 每个 block attention 到一个 pooled Qwen K/V segment。
        # 所有 block 共用同一套 RoPE 策略，保证 train/eval/runner 对齐。
        active_section = cfg.active_mrope_section()
        self.blocks = nn.ModuleList(
            [
                MoTDecoderBlock(
                    hidden_size=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    ffn_hidden_size=cfg.ffn_hidden_size,
                    dropout=cfg.dropout,
                    rope_theta=cfg.rope_theta,
                    rope_type=cfg.rope_type,
                    mrope_section=active_section,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.gen_final_norm = RMSNorm(cfg.hidden_size, eps=1e-6, elementwise_affine=True)

        self.route_head = RouteHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)
        self.waypoint_head = WaypointHead(hidden_size=cfg.hidden_size, point_dim=cfg.point_dim)

    def _build_gen_sequence(
        self,
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> torch.Tensor:
        """按 ``slice_layout`` 约定顺序打包 BEV/status/query token。"""
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
        # 布局：BEV | speed | target point | next target point | route Q | waypoint Q。
        gen_seq = torch.cat([bev_tokens, speed_tok, tp_tok, ntp_tok, route_q, wp_q], dim=1)
        if gen_seq.shape[1] != self.config.total_gen_tokens():
            raise RuntimeError(
                f"gen seq length mismatch: got {gen_seq.shape[1]}, "
                f"expect {self.config.total_gen_tokens()}"
            )
        return gen_seq

    def _check_pooled_kv(self, pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """attention 开始前检查 K/V segment 数量和形状。"""
        if len(pooled_kv) != self.config.num_layers:
            raise ValueError(
                f"pooled_kv has {len(pooled_kv)} segments, expected {self.config.num_layers}"
            )
        for i, (k, v) in enumerate(pooled_kv):
            if k.shape != v.shape:
                raise ValueError(f"pooled_kv[{i}] K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
            if k.ndim != 4:
                raise ValueError(f"pooled_kv[{i}] K must be [B,H,S,D], got {tuple(k.shape)}")
            if k.shape[1] != self.config.num_kv_heads or k.shape[3] != self.config.head_dim:
                raise ValueError(
                    f"pooled_kv[{i}] shape {tuple(k.shape)} does not match "
                    f"(heads={self.config.num_kv_heads}, head_dim={self.config.head_dim})"
                )

    def forward(
        self,
        pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        bev: torch.Tensor,
        speed: torch.Tensor,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
        rope_position_offset: int | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """返回 route 和 future-waypoint 预测。"""
        self._check_pooled_kv(pooled_kv)
        gen_seq = self._build_gen_sequence(
            bev=bev,
            speed=speed,
            target_point=target_point,
            target_point_next=target_point_next,
        )

        # 第 i 个 block 使用 pooled_kv[i] 作为 language prefix K/V。
        for block, lang_kv in zip(self.blocks, pooled_kv):
            gen_seq = block(
                gen_seq=gen_seq,
                lang_kv=lang_kv,
                rope_position_offset=rope_position_offset,
            )
        gen_seq = self.gen_final_norm(gen_seq)

        # 只有 query 切片进入 planning head；BEV/status token 只作为上下文。
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
