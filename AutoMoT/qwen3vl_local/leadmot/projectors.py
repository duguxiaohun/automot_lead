"""LEAD 输入到 gen 路 token 的投影模块。

- LeadBEVProjector：BEV (B,512,10,12) -> flatten + Linear + 2D pos_embed -> (B,120,1024)
- WaypointInputAdaptor：AutoMoT 同名同构 MLP 2->256->512->hidden
- StatusTokenEncoder：AutoMoT 严格结构，velocity_encoder + 共享 target_point_encoder

dtype/device 通过 next(self.parameters()) 自适应，调用方可以放心 .to(bfloat16)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LeadMoTPlanningDecoderConfig


class LeadBEVProjector(nn.Module):
    """BEV 卷积特征 -> gen 路 BEV token 序列。

    输入: (B, 512, 10, 12) 来源 LEAD LeadBEVEncoder
    输出: (B, 120, hidden) 含可学 2D 位置 embedding
    """

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        super().__init__()
        self.config = config
        h, w = config.bev_grid
        self.proj = nn.Linear(config.bev_channels, config.hidden_size)
        # 2D learnable pos embedding，让 decoder 区分 BEV 不同栅格位置
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        if bev_feature.ndim != 4:
            raise ValueError(f"bev_feature 应为 (B,C,H,W)，实际 {tuple(bev_feature.shape)}")
        _b, c, h, w = bev_feature.shape
        if c != self.config.bev_channels:
            raise ValueError(
                f"BEV channel mismatch: got {c}, expects {self.config.bev_channels}"
            )
        if (h, w) != tuple(self.config.bev_grid):
            raise ValueError(
                f"BEV grid mismatch: got ({h},{w}), expects {self.config.bev_grid}"
            )

        param = next(self.parameters())
        bev_feature = bev_feature.to(device=param.device, dtype=param.dtype)
        # (B,C,H,W) -> (B,H*W,C) -> Linear -> + pos_embed
        x = bev_feature.flatten(2).transpose(1, 2)
        return self.proj(x) + self.pos_embed


class WaypointInputAdaptor(nn.Module):
    """AutoMoT WaypointInputAdaptor 同构：2 -> 256 -> 512 -> token_size。"""

    def __init__(
        self,
        token_size: int,
        hidden_size: int = 256,
        hidden_size2: int = 512,
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.norm_layer = norm_layer
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, token_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return self.mlp(x)


class StatusTokenEncoder(nn.Module):
    """speed + target_point + target_point_next → 3 个 status token。

    严格按 AutoMoT 快路结构：velocity_encoder = 3 层 MLP 1->256->512->hidden；
    target_point_encoder = WaypointInputAdaptor 共享，tp 和 ntp 拼成 (B,2,2) 一次过。
    无归一化（AutoMoT 原版也没有）。
    """

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        super().__init__()
        hidden = config.hidden_size
        self.config = config
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, hidden),
        )
        self.target_point_encoder = WaypointInputAdaptor(token_size=hidden)

    def encode_speed(self, speed: torch.Tensor) -> torch.Tensor:
        """speed: (B,) 或 (B,1) -> (B,1,hidden)。"""
        if speed.ndim == 1:
            speed = speed.unsqueeze(-1)
        if speed.ndim != 2 or speed.shape[-1] != 1:
            raise ValueError(f"speed 应为 (B,) 或 (B,1)，实际 {tuple(speed.shape)}")
        param = next(self.velocity_encoder.parameters())
        speed = speed.to(device=param.device, dtype=param.dtype)
        return self.velocity_encoder(speed).unsqueeze(1)

    def encode_target_points(
        self,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tp/ntp: (B,2) -> 两个 (B,1,hidden) token，共享 encoder。"""
        if target_point.ndim != 2 or target_point.shape[-1] != 2:
            raise ValueError(f"target_point 应为 (B,2)，实际 {tuple(target_point.shape)}")
        if target_point_next.ndim != 2 or target_point_next.shape[-1] != 2:
            raise ValueError(
                f"target_point_next 应为 (B,2)，实际 {tuple(target_point_next.shape)}"
            )
        param = next(self.target_point_encoder.parameters())
        tp = target_point.to(device=param.device, dtype=param.dtype)
        ntp = target_point_next.to(device=param.device, dtype=param.dtype)
        # 拼成 (B, 2, 2) 一次过共享 encoder，与 AutoMoT 等价
        target_points = torch.stack([tp, ntp], dim=1)
        encoded = self.target_point_encoder(target_points)
        return encoded[:, 0:1, :], encoded[:, 1:2, :]
