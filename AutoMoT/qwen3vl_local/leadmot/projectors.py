"""把 LEAD runtime 张量投影成 LeadMoT token 的输入模块。

这些模块刻意保持很小：
- BEV feature 变成 120 个 generated token；
- speed、target_point、next_target_point 变成 3 个 status token；
- 所有输出 token 宽度都对齐 ``config.hidden_size``。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LeadMoTPlanningDecoderConfig


class LeadBEVProjector(nn.Module):
    """把 LEAD BEV feature map 投影成 generated-token embedding。"""

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        super().__init__()
        self.config = config
        h, w = config.bev_grid
        self.proj = nn.Linear(config.bev_channels, config.hidden_size)
        # 可学习 2D 位置 embedding：BEV 的 (H, W) 网格摊平成 token 后，
        # attention 仍能区分不同 grid cell。
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """把 ``(B, C, H, W)`` BEV feature 转成 ``(B, H*W, hidden)``。"""
        if bev_feature.ndim != 4:
            raise ValueError(f"bev_feature must be (B,C,H,W), got {tuple(bev_feature.shape)}")
        _b, c, h, w = bev_feature.shape
        if c != self.config.bev_channels:
            raise ValueError(f"BEV channel mismatch: got {c}, expects {self.config.bev_channels}")
        if (h, w) != tuple(self.config.bev_grid):
            raise ValueError(f"BEV grid mismatch: got ({h},{w}), expects {self.config.bev_grid}")

        param = next(self.parameters())
        bev_feature = bev_feature.to(device=param.device, dtype=param.dtype)
        # 只摊平空间维度；channel 放到最后，交给 Linear 做投影。
        x = bev_feature.flatten(2).transpose(1, 2)
        return self.proj(x) + self.pos_embed


class WaypointInputAdaptor(nn.Module):
    """处理 target point 这类 2D 输入的小 MLP。"""

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
        """把 ``(..., 2)`` 坐标投影成 ``(..., token_size)`` token。"""
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return self.mlp(x)


class StatusTokenEncoder(nn.Module):
    """把 speed、target_point、next_target_point 编码成三个 token。

    speed 保持 runner 中的原始 m/s 量纲；target point 和 next target point
    共享同一个 MLP，让两者 embedding 落在同一坐标语义空间里。
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
        """把 speed ``(B,)`` 或 ``(B,1)`` 编码成一个 ``(B,1,hidden)`` token。"""
        if speed.ndim == 1:
            speed = speed.unsqueeze(-1)
        if speed.ndim != 2 or speed.shape[-1] != 1:
            raise ValueError(f"speed must be (B,) or (B,1), got {tuple(speed.shape)}")
        param = next(self.velocity_encoder.parameters())
        speed = speed.to(device=param.device, dtype=param.dtype)
        return self.velocity_encoder(speed).unsqueeze(1)

    def encode_target_points(
        self,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把 TP/NTP 两个 ``(B,2)`` 张量编码成两个 ``(B,1,hidden)`` token。"""
        if target_point.ndim != 2 or target_point.shape[-1] != 2:
            raise ValueError(f"target_point must be (B,2), got {tuple(target_point.shape)}")
        if target_point_next.ndim != 2 or target_point_next.shape[-1] != 2:
            raise ValueError(f"target_point_next must be (B,2), got {tuple(target_point_next.shape)}")
        param = next(self.target_point_encoder.parameters())
        tp = target_point.to(device=param.device, dtype=param.dtype)
        ntp = target_point_next.to(device=param.device, dtype=param.dtype)
        # 先堆成 (B, 2, 2)，让两个点一次性通过共享 adaptor。
        target_points = torch.stack([tp, ntp], dim=1)
        encoded = self.target_point_encoder(target_points)
        return encoded[:, 0:1, :], encoded[:, 1:2, :]
