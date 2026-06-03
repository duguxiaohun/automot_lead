"""LEAD 输入到 gen 路 token 的投影模块。

为什么有这一层
==============
LeadMoT decoder 内部所有 token 都是 (gen_hidden=1024) 维向量；但 LEAD 数据进来时：
- BEV 特征是 (B, 512, 10, 12) 卷积输出
- speed 是 (B,) 标量
- target_point / target_point_next 是 (B, 2) 米制坐标

projectors 这一层负责把它们都搬到统一的 (B, ?, 1024) gen token 表示，
喂给后面的 12 层 prefix-KV attention 共同处理。

包含三个子模块
==============
1. **LeadBEVProjector**：BEV 卷积特征 → 120 个 BEV token + 2D 位置 embedding
2. **WaypointInputAdaptor**：与 AutoMoT `WaypointInputAdaptor` 同名同构的目标点 MLP
3. **StatusTokenEncoder**：用 AutoMoT 风格的 velocity_encoder + 共享
   target_point_encoder 一次性出 speed/tp/ntp 三个 token

dtype 自适应
============
所有 forward 入口都用 `next(self.<mod>.parameters()).dtype` 自动适配模块参数
的 dtype（bf16/fp16/fp32）。这样调用方可以放心地把 decoder 整体 `.to(bfloat16)`，
而输入张量保持 fp32，projector 会自动转。

为什么用 next(parameters()).dtype 而不是直接 `self.proj.weight.dtype`
====================================================================
- 直接读 `.weight` 假设 Sequential[0] 是 Linear，未来在开头插了 LayerNorm
  或别的 wrapper 模块就会拿到错的 dtype
- `next(self.parameters())` 不假设结构，对任何 Module 都安全
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LeadMoTPlanningDecoderConfig


class LeadBEVProjector(nn.Module):
    """LEAD BEV 卷积特征 → gen 路 BEV token 序列。

    输入: bev_feature (B, C=512, H=10, W=12)
           来源: LEAD `LeadBEVEncoder` 的输出（runner 里调用，权重是 LEAD 训好的
           tfv6_resnet34 backbone-only ckpt，已经 LEAD 训练分布对齐）
    输出: bev_tokens (B, H*W=120, hidden=1024)

    设计要点
    ========
    - **flatten + Linear 升维**：把 (C=512) 维通道空间投影到 (hidden=1024) 维 gen
      空间。Linear 是最简单的升维方式，没有引入空间归纳偏置。
    - **2D 可学位置 embedding**：BEV 是 2D 栅格 (10×12)，flatten 后每个 token 物理
      位置不同，需要 pos embedding 让 decoder 区分哪个 token 对应哪个栅格位置。
      `nn.Parameter(zeros)` + `trunc_normal_(std=0.02)` 是 ViT 标准做法。
    - **没用 sincos pos embedding**：Sincos 更省参数且能外推到不同尺寸，但栅格大小
      是固定的 10×12，learnable 就够，且更易学。

    为什么 BEV channels=512、grid=(10,12)
    =====================================
    这是 LEAD `LeadTransfuserBackbone` 的输出尺寸，由 LEAD 论文/代码定义。
    我们的 BEV projector 必须严格按这个尺寸接，因为 LEAD backbone 是 frozen
    + 训好的，不能改输出 shape。
    """

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        """
        参数:
            config: LeadMoTPlanningDecoderConfig，本类从中读取 bev_channels /
                    bev_grid / hidden_size 三个字段
        """
        super().__init__()
        # 保存 config 引用，forward 时校验输入 shape 用
        self.config = config
        # 解包 bev grid 尺寸，本地变量便于阅读
        h, w = config.bev_grid

        # 升维 Linear：把 (C=512) 投到 (hidden=1024)
        # bias 默认开（nn.Linear 默认行为），跟 LEAD/AutoMoT 风格一致
        self.proj = nn.Linear(config.bev_channels, config.hidden_size)

        # 2D 位置 embedding：(1, H*W, hidden) 的可学参数
        # 第 0 维是 broadcast batch，让一份 pos_embed 给所有 batch 共享
        # zeros 初始化 + 之后 trunc_normal 是为了：开始时是常数 0（不引入随机扰动），
        # 训练时逐渐学出每个栅格位置的 embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """
        参数:
            bev_feature: (B, C, H, W) LEAD BEV encoder 输出
                          dtype 任意（自动转到 self.proj.weight.dtype）
                          device 任意（自动搬到 self.proj.weight.device）
        返回:
            (B, H*W, hidden) BEV token 序列
        """
        # ---- 形状校验 ----
        # 必须是 4D NCHW，否则下面 flatten 的语义就不对
        if bev_feature.ndim != 4:
            raise ValueError(f"bev_feature 应为 (B,C,H,W)，实际 {tuple(bev_feature.shape)}")
        _b, c, h, w = bev_feature.shape

        # channel 数必须与 config 一致：BEV backbone 的输出 channel 是固定的，
        # 出错说明 backbone 或 config 有一个错了，应立即 abort
        if c != self.config.bev_channels:
            raise ValueError(
                f"BEV channel mismatch: got {c}, config expects {self.config.bev_channels}"
            )
        # 栅格大小也必须严格匹配，否则 pos_embed 长度对不上
        if (h, w) != tuple(self.config.bev_grid):
            raise ValueError(
                f"BEV grid mismatch: got ({h},{w}), config expects {self.config.bev_grid}"
            )

        # ---- dtype/device 自适应 ----
        # 从模块参数取一个作为 reference，避免硬编码 self.proj.weight
        # （如果未来在 proj 前面插了别的模块，这种写法仍能拿到正确参数）
        param = next(self.parameters())
        bev_feature = bev_feature.to(device=param.device, dtype=param.dtype)

        # ---- 主投影 ----
        # flatten(2): (B, C, H, W) -> (B, C, H*W)
        # transpose(1,2): (B, C, H*W) -> (B, H*W, C)
        # 最终把每个空间位置变成一个"token"，channel 维当 feature 维
        x = bev_feature.flatten(2).transpose(1, 2)

        # Linear: C=512 -> hidden=1024，再加 pos embedding
        # pos_embed 的 dtype 跟 self.proj.weight 一致（都来自 self.parameters()），
        # 不需要再次显式 .to(dtype=...)
        return self.proj(x) + self.pos_embed


class WaypointInputAdaptor(nn.Module):
    """AutoMoT 同名同构 target point adaptor：2 → 256 → 512 → token_size。

    跟 `AutoMoT/Automot/mot/modeling/automot/automot.py:124` 的 `WaypointInputAdaptor`
    完全同构：3 层 MLP 渐进升维，2 个 ReLU 激活。

    为什么严格抄 AutoMoT 结构
    =========================
    用户明确要求：status 编码必须按 AutoMoT 处理。这样训练时如果想用 AutoMoT
    checkpoint 的 target_point_encoder 权重做 warm start（init_mot 风格），
    state_dict 的 key 可以一一对应，不需要做权重 reshape。

    为什么 tp/ntp 共享一个 encoder
    ===============================
    AutoMoT 原版就是把 tp 和 ntp 拼成 (B, 2, 2) 一次 forward，得到 (B, 2, hidden)
    再切两个 token。共享权重让"目标点编码"这个语义在两个 token 上一致，节约参数。
    """

    def __init__(
        self,
        token_size: int,
        hidden_size: int = 256,
        hidden_size2: int = 512,
        norm_layer: nn.Module | None = None,
    ):
        """
        参数:
            token_size:   最终输出维度，应该等于 decoder hidden（默认 1024）
            hidden_size:  第一隐藏层维度，默认 256，匹配 AutoMoT 原版
            hidden_size2: 第二隐藏层维度，默认 512，匹配 AutoMoT 原版
            norm_layer:   可选的输入归一化层（默认 None；AutoMoT 也没用）
        """
        super().__init__()
        # 可选的输入归一化，AutoMoT 没传，本子包也不用，但保留接口便于未来扩展
        self.norm_layer = norm_layer

        # 3 层 MLP，跟 AutoMoT.modeling.automot.automot.py:146 完全同构：
        # Linear(2, 256) -> ReLU(inplace=True) -> Linear(256, 512) -> ReLU(inplace=True) -> Linear(512, token_size)
        # inplace=True 是为了省一点点显存（直接覆盖输入张量），AutoMoT 也是这么写的
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(True),
            nn.Linear(hidden_size2, token_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, 2) 一组 N 个目标点的米制坐标
        返回:
            (B, N, token_size) 编码后的 token
        """
        # 如果配了 norm_layer 就先归一化（本子包默认不配，所以这行通常 no-op）
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        # MLP forward，输出形状跟输入对齐除了最后一维
        return self.mlp(x)


class StatusTokenEncoder(nn.Module):
    """AutoMoT 严格风格的 speed / target point 编码器。

    把 ego 状态（速度、当前目标点、下一目标点）编码成 3 个 (B, 1, hidden) token，
    供 decoder 的 gen 段消费。

    接口设计的取舍
    ==============
    AutoMoT 原版接口是 `v_target_point=(B, 5)` 一个张量拼起来传，因为它来自
    dataset 阶段的拼接。本子包对外暴露 `encode_speed(speed)` /
    `encode_target_points(tp, ntp)` 两个独立方法：
    - 调用方写起来更显式（speed 是 speed，tp 是 tp，不会把维度搞混）
    - 跟 LEAD 原版 `data["speed"] / data["target_point"] / data["target_point_next"]`
      分开取值的语义一致
    - 内部 `encode_target_points` 仍把 tp 和 ntp 拼成 (B, 2, 2) 一次过共享
      WaypointInputAdaptor，跟 AutoMoT 原版完全等价

    没有归一化
    ==========
    AutoMoT 原版直接喂原始米/秒、米坐标，没做 max_speed 归一化。本子包跟它一致，
    不做归一化。如果未来发现训练不稳，可以在 encode_speed/encode_target_points
    入口加 `/ max_speed` 等做法（LEAD 原版 PlanningContextEncoder 是带归一化的）。
    """

    def __init__(self, config: LeadMoTPlanningDecoderConfig):
        """
        参数:
            config: 本类从中读取 hidden_size，并据此构造内部子模块
        """
        super().__init__()
        # 局部变量便于阅读，避免每次写 self.config.hidden_size
        hidden = config.hidden_size
        # 保存 config 引用方便 debug
        self.config = config

        # velocity_encoder：跟 AutoMoT.modeling.automot.automot.py:288 完全同构
        # 3 层 MLP: Linear(1, 256) -> ReLU -> Linear(256, 512) -> ReLU -> Linear(512, hidden)
        # 这是 AutoMoT 一直用的速度编码结构，方便未来从 AutoMoT ckpt 复制权重做 warm start
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, hidden),
        )

        # target_point_encoder：共享 WaypointInputAdaptor，跟 AutoMoT 同名同构
        # tp 和 ntp 都会过这一个模块，靠 batch 维区分（不再用 type embedding）
        self.target_point_encoder = WaypointInputAdaptor(token_size=hidden)

    def encode_speed(self, speed: torch.Tensor) -> torch.Tensor:
        """编码速度 → 单个 status token。

        参数:
            speed: (B,) 或 (B, 1)，单位 m/s
        返回:
            (B, 1, hidden) 一个 speed token
        """
        # 兼容输入：如果是 (B,) 1D，扩成 (B, 1) 2D
        if speed.ndim == 1:
            speed = speed.unsqueeze(-1)
        # 形状校验：必须是 (B, 1)
        if speed.ndim != 2 or speed.shape[-1] != 1:
            raise ValueError(f"speed 应为 (B,) 或 (B,1)，实际 {tuple(speed.shape)}")

        # dtype/device 自适应：用 velocity_encoder 第一个 Linear 的参数作 reference
        # 这里 next(...) 拿到的是 self.velocity_encoder[0].weight 等价物，
        # 但通过 parameters() 接口拿，避免硬编码 [0].weight
        param = next(self.velocity_encoder.parameters())
        speed = speed.to(device=param.device, dtype=param.dtype)

        # 过 MLP 得到 (B, hidden)，unsqueeze(1) 加上 token 维 -> (B, 1, hidden)
        return self.velocity_encoder(speed).unsqueeze(1)

    def encode_target_points(
        self,
        target_point: torch.Tensor,
        target_point_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码当前目标点 + 下一目标点 → 两个 status token。

        参数:
            target_point:      (B, 2) 当前目标点，ego frame 米制
            target_point_next: (B, 2) 下一目标点，ego frame 米制
        返回:
            (tp_token, ntp_token)，每个 (B, 1, hidden)
        """
        # ---- 形状校验 ----
        # 两个目标点都必须是 (B, 2)，否则下面 stack 出错
        if target_point.ndim != 2 or target_point.shape[-1] != 2:
            raise ValueError(f"target_point 应为 (B,2)，实际 {tuple(target_point.shape)}")
        if target_point_next.ndim != 2 or target_point_next.shape[-1] != 2:
            raise ValueError(
                f"target_point_next 应为 (B,2)，实际 {tuple(target_point_next.shape)}"
            )

        # ---- dtype/device 自适应 ----
        # 用 target_point_encoder 内任意参数作 reference
        param = next(self.target_point_encoder.parameters())
        tp = target_point.to(device=param.device, dtype=param.dtype)
        ntp = target_point_next.to(device=param.device, dtype=param.dtype)

        # ---- 拼成 AutoMoT 风格的 (B, 2, 2) ----
        # stack 沿新维 dim=1：第 0 个是 tp，第 1 个是 ntp
        # 这等价于 AutoMoT 里 `v_target_point[1:5].reshape(1, 2, 2)`
        # 一次过共享 encoder，比分两次 forward 省一次内核启动
        target_points = torch.stack([tp, ntp], dim=1)  # (B, 2, 2)

        # 共享 WaypointInputAdaptor forward → (B, 2, hidden)
        encoded = self.target_point_encoder(target_points)

        # 切回两个独立 token，保持 (B, 1, hidden) 形状方便后续 cat
        # 注意用 0:1 而不是 0，前者保留维度，后者会 squeeze 掉 token 维
        return encoded[:, 0:1, :], encoded[:, 1:2, :]
