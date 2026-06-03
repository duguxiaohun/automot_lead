"""LEAD 输入到 gen 路 token 的投影模块。

包含两个子模块：
- LeadBEVProjector：把 LEAD BEV 特征 (B, 512, 10, 12) 投影到 gen hidden=1024。
- StatusTokenEncoder：严格复刻 AutoMoT 快路 status 编码结构：
  speed 走 1 -> 256 -> 512 -> hidden，target_point 与 target_point_next
  共享 WaypointInputAdaptor，输入 (B, 2, 2)，输出 (B, 2, hidden)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LeadMoTPlanningDecoderConfig


class LeadBEVProjector(nn.Module):
    """LEAD BEV 特征 -> gen 路 BEV token 序列。

    输入: bev_feature (B, C=512, H=10, W=12)
    输出: bev_tokens  (B, H*W=120, hidden=1024)
    """

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        super().__init__()
        self.config = config
        h, w = config.bev_grid
        self.proj = nn.Linear(config.bev_channels, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        if bev_feature.ndim != 4:
            raise ValueError(f"bev_feature 应为 (B,C,H,W)，实际 {tuple(bev_feature.shape)}")
        _b, c, h, w = bev_feature.shape
        if c != self.config.bev_channels:
            raise ValueError(
                f"BEV channel mismatch: got {c}, config expects {self.config.bev_channels}"
            )
        if (h, w) != tuple(self.config.bev_grid):
            raise ValueError(
                f"BEV grid mismatch: got ({h},{w}), config expects {self.config.bev_grid}"
            )
        param = next(self.parameters())
        bev_feature = bev_feature.to(device=param.device, dtype=param.dtype)
        x = bev_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return self.proj(x) + self.pos_embed


class WaypointInputAdaptor(nn.Module):
    """AutoMoT 同构 target point adaptor：2 -> 256 -> 512 -> token_size。"""

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
    """AutoMoT 严格结构的 speed / target point 编码器。

    输入仍保留拆开的 speed / tp / ntp，内部会拼成 AutoMoT 的
    target_points=(B,2,2) 一次性过共享 WaypointInputAdaptor。
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
        """tp/ntp: (B,2) -> 两个 (B,1,hidden) token。"""
        if target_point.ndim != 2 or target_point.shape[-1] != 2:
            raise ValueError(f"target_point 应为 (B,2)，实际 {tuple(target_point.shape)}")
        if target_point_next.ndim != 2 or target_point_next.shape[-1] != 2:
            raise ValueError(
                f"target_point_next 应为 (B,2)，实际 {tuple(target_point_next.shape)}"
            )
        param = next(self.target_point_encoder.parameters())
        tp = target_point.to(device=param.device, dtype=param.dtype)
        ntp = target_point_next.to(device=param.device, dtype=param.dtype)
        target_points = torch.stack([tp, ntp], dim=1)  # (B,2,2)
        encoded = self.target_point_encoder(target_points)  # (B,2,hidden)
        return encoded[:, 0:1, :], encoded[:, 1:2, :]
