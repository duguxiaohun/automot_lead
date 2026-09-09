"""条件轨迹 Flow Matching：以完整图文/分析 KV 为条件生成连续 route 和 waypoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F


FLOW_MATCHING_VERSION = "conditional_joint_trajectory_rectified_flow_v2"


@dataclass(frozen=True)
class FlowMatchingConfig:
    """FM 头、坐标标准化和 ODE 采样配置，独立保存到 action checkpoint。"""

    version: str = FLOW_MATCHING_VERSION
    route_coordinate_scale_m: float = 30.0
    waypoint_coordinate_scale_m: float = 20.0
    time_embed_dim: int = 64
    sample_steps: int = 10
    trajectory_layers: int = 2
    trajectory_heads: int = 8

    def validate(self):
        if self.version != FLOW_MATCHING_VERSION:
            raise ValueError(f"unsupported flow-matching version: {self.version}")
        if self.time_embed_dim < 4 or self.time_embed_dim % 2:
            raise ValueError("flow time_embed_dim must be even and at least 4")
        if self.sample_steps <= 0:
            raise ValueError("flow sample_steps must be positive")
        if self.trajectory_layers <= 0 or self.trajectory_heads <= 0:
            raise ValueError("flow trajectory layers/heads must be positive")
        for key in ("route_coordinate_scale_m", "waypoint_coordinate_scale_m"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")

    def identity(self):
        self.validate()
        return asdict(self)


def _targets(gt_route, gt_waypoints, config):
    """把两类绝对 ego 点分别缩放，仍保留 route/waypoint 的语义边界。"""
    return torch.cat(
        [
            gt_route.float() / config.route_coordinate_scale_m,
            gt_waypoints.float() / config.waypoint_coordinate_scale_m,
        ],
        dim=1,
    )


def _split_targets(points, route_points, config):
    """把归一化连续变量恢复为现有的 route / waypoint 公共接口。"""
    return (
        points[:, :route_points].float() * config.route_coordinate_scale_m,
        points[:, route_points:].float() * config.waypoint_coordinate_scale_m,
    )


def make_training_flow(gt_route, gt_waypoints, config, generator=None):
    """直线条件 FM：x_t=(1-t)eps+t*x_gt，目标向量场为 x_gt-eps。"""
    target = _targets(gt_route, gt_waypoints, config)
    eps = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype, generator=generator)
    state = (1.0 - t[:, None, None]) * eps + t[:, None, None] * target
    # 训练日志里的采样 ADE/FDE 复用同一次已经抽到的高斯噪声。这样不会为了仅作
    # 诊断的 ODE 采样再消耗一份全局 RNG；评测则由 make_evaluation_flow 覆盖成按样本
    # 身份固定的独立噪声。
    return dict(
        flow_state=state,
        flow_time=t,
        flow_target_velocity=target - eps,
        flow_sample_noise=eps,
    )


def _sample_seed(sample, seed, purpose):
    """跨进程/卡数稳定的每帧随机种子，不读取或污染训练 RNG。"""
    identity = (sample["scenario"], sample["run_id"], int(sample["anchor"]), int(seed), purpose)
    return int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], "big") % (2**63 - 1)


def make_evaluation_flow(gt_route, gt_waypoints, config, sample, seed):
    """按样本身份固定 FM x_t/t 和 Euler 初始噪声，支持可复现多卡比较。"""
    generator = torch.Generator(device=gt_route.device)
    generator.manual_seed(_sample_seed(sample, seed, "evaluation_flow"))
    flow = make_training_flow(gt_route, gt_waypoints, config, generator)
    flow["flow_sample_noise"] = torch.randn(
        flow["flow_state"].shape,
        device=gt_route.device,
        dtype=gt_route.dtype,
        generator=generator,
    )
    return flow


def route_policy_generator(device, policy_seed, route_key):
    """为一条闭环 route 建立独立、可复现且不触碰全局 RNG 的 FM 噪声流。"""
    policy_seed = int(policy_seed)
    if policy_seed < 0:
        raise ValueError("policy seed must be nonnegative")
    digest = hashlib.sha256(
        repr(("action_prior_fm_policy_noise_v1", str(route_key), policy_seed)).encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def flow_matching_loss(outputs, target_velocity, route_points, route_weight, waypoint_weight):
    """标准向量场 MSE；route/waypoint 可保留原训练权重但不再用 L1 坐标回归。"""
    velocity = outputs["flow_velocity"]
    if velocity.shape != target_velocity.shape:
        raise ValueError(f"flow velocity target shape mismatch: {velocity.shape} vs {target_velocity.shape}")
    route = F.mse_loss(velocity[:, :route_points].float(), target_velocity[:, :route_points].float())
    waypoint = F.mse_loss(velocity[:, route_points:].float(), target_velocity[:, route_points:].float())
    return route_weight * route + waypoint_weight * waypoint, route, waypoint


class ConditionalFlowMatchingDecoder(nn.Module):
    """冻结 Qwen KV/BEV 条件上的 rectified-flow 向量场。

    ``conditioner`` 只在一次 forward 中运行一次 Prefix-KV decoder；Euler 采样的每一步
    仅调用轻量速度头。因此同一份输入图像、提示词和 base 分析的 KV 会被全部 ODE 步共享。
    """

    def __init__(self, leadmot_config, flow_config: FlowMatchingConfig | None = None):
        super().__init__()
        from qwen3vl_local.leadmot import LeadMoTPlanningDecoder

        self.config = leadmot_config
        self.flow_config = flow_config or FlowMatchingConfig()
        self.flow_config.validate()
        self.conditioner = LeadMoTPlanningDecoder(leadmot_config)
        # encode_conditioning 不会走旧 Linear+cumsum head；移除无梯度参数避免 DDP unused 参数。
        del self.conditioner.route_head
        del self.conditioner.waypoint_head
        hidden = leadmot_config.hidden_size
        points = leadmot_config.num_route_queries + leadmot_config.num_waypoint_queries
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.point_type_embedding = nn.Embedding(2, hidden)
        self.point_index_embedding = nn.Embedding(max(leadmot_config.num_route_queries, leadmot_config.num_waypoint_queries), hidden)
        self.time_encoder = nn.Sequential(
            nn.Linear(self.flow_config.time_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2)
        )
        if hidden % self.flow_config.trajectory_heads:
            raise ValueError("flow trajectory_heads must divide the decoder hidden size")
        # 每个 ODE 步让所有 route/waypoint 的当前 x_t 互相 self-attend。它复用静态
        # 图文 KV/BEV conditioning，但速度场不再逐点独立，能表达整条轨迹的一致分支。
        self.trajectory_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=self.flow_config.trajectory_heads,
                dim_feedforward=hidden * 4,
                dropout=leadmot_config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.flow_config.trajectory_layers,
        )
        self.register_buffer(
            "_time_frequencies",
            torch.exp(torch.linspace(0.0, -math.log(10000.0), self.flow_config.time_embed_dim // 2)),
            persistent=False,
        )
        self._points = points

    def _time_features(self, time, dtype):
        time = time.reshape(-1, 1).float()
        angles = 2.0 * math.pi * time * self._time_frequencies.to(device=time.device)
        return torch.cat([angles.sin(), angles.cos()], dim=-1).to(dtype=dtype)

    def _conditioning(self, **kwargs):
        encoded = self.conditioner.encode_conditioning(**kwargs)
        return torch.cat([encoded["route_hidden"], encoded["waypoint_hidden"]], dim=1)

    def _velocity(self, conditioning, flow_state, flow_time):
        if flow_state.ndim != 3 or flow_state.shape[1:] != (self._points, 2):
            raise ValueError(f"flow_state must be [B,{self._points},2], got {tuple(flow_state.shape)}")
        if flow_state.shape[0] != conditioning.shape[0]:
            raise ValueError("flow state and conditioning batch differ")
        time = flow_time.reshape(-1)
        if time.shape[0] != conditioning.shape[0] or not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
            raise ValueError("flow_time must contain one finite value in [0,1] per batch item")
        route_n = self.config.num_route_queries
        point_type = torch.cat([
            torch.zeros(route_n, device=flow_state.device, dtype=torch.long),
            torch.ones(self._points - route_n, device=flow_state.device, dtype=torch.long),
        ])
        point_index = torch.cat([
            torch.arange(route_n, device=flow_state.device),
            torch.arange(self._points - route_n, device=flow_state.device),
        ])
        token = (
            conditioning
            + self.coordinate_encoder(flow_state.to(dtype=conditioning.dtype))
            + self.point_type_embedding(point_type)[None].to(dtype=conditioning.dtype)
            + self.point_index_embedding(point_index)[None].to(dtype=conditioning.dtype)
            + self.time_encoder(self._time_features(time, conditioning.dtype))[:, None]
        )
        # PyTorch 2.3 的 CPU BF16 eval/no_grad MHA fastpath 会把 BF16 input 直接交给
        # FP32 projection weight。只在这个受影响组合中以 FP32 跑轻量轨迹块，再交还
        # 外层 autocast 运行 velocity head；CUDA BF16 路径不受影响。
        cpu_bf16_autocast = bool(
            getattr(torch, "is_autocast_cpu_enabled", lambda: False)()
        )
        if token.device.type == "cpu" and (token.dtype == torch.bfloat16 or cpu_bf16_autocast):
            with torch.autocast(device_type="cpu", enabled=False):
                trajectory = self.trajectory_blocks(token.float())
            trajectory = trajectory.to(dtype=token.dtype)
        else:
            trajectory = self.trajectory_blocks(token)
        return self.velocity_head(trajectory)

    def _sample(self, conditioning, initial_noise, steps):
        if steps <= 0:
            raise ValueError("flow sample steps must be positive")
        state = initial_noise
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((state.shape[0],), step * dt, device=state.device, dtype=state.dtype)
            state = state + dt * self._velocity(conditioning, state, time)
        return state

    def forward(
        self,
        *,
        flow_state=None,
        flow_time=None,
        flow_sample_noise=None,
        flow_sample_steps=None,
        sample_trajectory=True,
        **condition_kwargs,
    ):
        """训练只需 vector field；采样由 eval/闭环或显式诊断请求启用。"""
        if (flow_state is None) != (flow_time is None):
            raise ValueError("flow_state and flow_time must be supplied together")
        conditioning = self._conditioning(**condition_kwargs)
        batch = conditioning.shape[0]
        velocity = None
        if flow_state is not None:
            velocity = self._velocity(conditioning, flow_state, flow_time)
        result = dict(gen_hidden=conditioning)
        if sample_trajectory:
            if flow_sample_noise is None:
                flow_sample_noise = torch.randn(
                    batch, self._points, 2,
                    device=conditioning.device, dtype=conditioning.dtype,
                )
            else:
                flow_sample_noise = flow_sample_noise.to(
                    device=conditioning.device, dtype=conditioning.dtype
                )
            # 采样结果不参与 vector-field loss。训练模式下明确关掉 trajectory
            # Transformer 的 dropout，固定噪声不会再因 ODE 步数而消耗全局 RNG。
            was_training = self.trajectory_blocks.training
            self.trajectory_blocks.eval()
            try:
                with torch.no_grad():
                    sampled = self._sample(
                        conditioning.detach(), flow_sample_noise,
                        int(flow_sample_steps or self.flow_config.sample_steps),
                    )
            finally:
                self.trajectory_blocks.train(was_training)
            route, waypoint = _split_targets(
                sampled, self.config.num_route_queries, self.flow_config
            )
            result.update(
                pred_route=route,
                pred_future_waypoints=waypoint,
                flow_sample_normalized=sampled,
            )
        if velocity is not None:
            result["flow_velocity"] = velocity
        return result
