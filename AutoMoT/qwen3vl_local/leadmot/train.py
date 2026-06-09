#!/usr/bin/env python3
"""在 frozen Qwen3-VL 和 frozen LeadBEVEncoder 上训练 LeadMoT decoder。"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import lzma
import math
import os
import pickle
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler


AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
TEAM_CODE_DIR = AUTOMOT_ROOT / "leaderboard" / "team_code"
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))
if str(TEAM_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_CODE_DIR))

from mot_lead_offline_runner import (  # noqa: E402
    LEAD_BEV_CKPT_PATH,
    LeadOfflineMoTRunner,
    _segment_qwen_cache_for_leadmot,
    build_cleaned_prompt_and_modes,
    build_clip_from_real_lead_route,
)
import mot_lead_offline_runner as mot_runner  # noqa: E402
from qwen3vl_local.leadmot import LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件，返回字典列表。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_invocation(output_dir: Path, rank: int = 0) -> None:
    """保存 argv/env/git 元信息，方便复现实验。"""
    if rank != 0:
        return
    try:
        # _dt 已在文件顶层 import；platform/shlex/socket/subprocess 只用于写
        # invocations/ 记录，延迟 import 可以减少训练启动路径的额外依赖。
        import platform as _platform
        import shlex as _shlex
        import socket as _socket
        import subprocess as _subprocess

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        host = _socket.gethostname()
        inv_dir = output_dir / "invocations"
        inv_dir.mkdir(parents=True, exist_ok=True)
        out_path = inv_dir / f"{ts}_{host}_pid{os.getpid()}.txt"

        env_keys = (
            "CUDA_VISIBLE_DEVICES",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NCCL_DEBUG",
            "PYTORCH_CUDA_ALLOC_CONF",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "LEADMOT_CUDNN_BENCHMARK",
            "LEADMOT_NCCL_TIMEOUT_MIN",
        )
        try:
            git = _subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(AUTOMOT_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
            )
            git_commit = git.stdout.strip() if git.returncode == 0 else "<unavailable>"
        except Exception:
            git_commit = "<unavailable>"

        lines = [
            f"# saved at {ts}",
            f"# hostname = {host}",
            f"# pid = {os.getpid()}",
            f"# python = {sys.version.split()[0]}",
            f"# torch = {getattr(torch, '__version__', '<unknown>')}",
            f"# platform = {_platform.platform()}",
            f"# git_commit = {git_commit}",
            "",
            "# ---- 选中的环境变量 ----",
            *[f"{key}={os.environ.get(key, '<unset>')}" for key in env_keys],
            "",
            "# ---- sys.argv（每行一个） ----",
            *sys.argv,
            "",
            "# ---- shell 复现命令 ----",
            " ".join(_shlex.quote(arg) for arg in sys.argv),
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[invocation] saved -> {out_path}")
    except Exception as exc:
        print(f"[invocation] save failed (ignored): {exc}")


def _setup_offline_env() -> None:
    """强制当前训练入口使用本地/离线 HuggingFace 行为。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _init_distributed() -> tuple[int, int, int]:
    """初始化 torchrun 分布式训练，并把每个 rank 绑定到对应 GPU。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = _dt.timedelta(minutes=int(os.environ.get("LEADMOT_NCCL_TIMEOUT_MIN", "10")))
        # 让 Qwen/BEV 加载卡死或坏 I/O 导致的 rank 停滞更早暴露。
        init_kwargs: dict[str, Any] = {"backend": backend, "timeout": timeout}
        if backend == "nccl":
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            # 兼容旧 PyTorch：老版本 init_process_group 没有 device_id 参数。
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
    return rank, local_rank, world_size


def _barrier() -> None:
    """只在 torch.distributed 已启用时同步各 rank。"""
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            try:
                dist.barrier(device_ids=[local_rank])
                return
            except TypeError:
                pass
        dist.barrier()


def _dtype(name: str) -> torch.dtype:
    """把 CLI dtype 字符串映射成 torch dtype。"""
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
    """裁剪或重复最后一个点，让轨迹刚好有 rows 个点。"""
    array = np.asarray(array, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] >= rows:
        return array[:rows]
    if array.shape[0] == 0:
        return np.zeros((rows, 2), dtype=np.float32)
    pad = np.repeat(array[-1:], rows - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


def _circle_line_segment_intersection(
    circle_center: np.ndarray,
    circle_radius: float,
    pt1: np.ndarray,
    pt2: np.ndarray,
    tangent_tol: float = 1e-9,
) -> list[tuple[float, float]]:
    """LEAD carla_dataset_utils.circle_line_segment_intersection 的 full_line=True 版本。"""
    if np.linalg.norm(pt1 - pt2) < 1e-9:
        return []
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx**2 + dy**2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius**2 * dr**2 - big_d**2
    if discriminant < 0:
        return []
    intersections = [
        (
            cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**0.5) / dr**2,
            cy + (-big_d * dx + sign * abs(dy) * discriminant**0.5) / dr**2,
        )
        for sign in ((1, -1) if dy < 0 else (-1, 1))
    ]
    if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
        return [intersections[0]]
    return intersections


def _lead_iterative_line_interpolation(route: np.ndarray, rows: int, target_first_distance: float = 2.5) -> np.ndarray:
    """用于 route 标签的最小 LEAD iterative_line_interpolation 等价实现。"""
    interpolated_route_points: list[np.ndarray] = []
    min_distance = 1.0
    last_interpolated_point = np.array([0.0, 0.0], dtype=np.float32)
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = np.array([0.0, 0.0], dtype=np.float32)
    first_iteration = True

    while len(interpolated_route_points) < rows:
        if not first_iteration:
            current_route_index += 1
            last_point = current_point

        if current_route_index < route.shape[0]:
            current_point = route[current_route_index]
            intersection = _circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance if not first_iteration else target_first_distance,
                pt1=last_interpolated_point,
                pt2=current_point,
            )
        else:
            current_point = route[-1]
            last_point = route[-2]
            intersection = _circle_line_segment_intersection(
                circle_center=last_interpolated_point,
                circle_radius=min_distance,
                pt1=last_point,
                pt2=current_point,
            )

        if len(intersection) > 1:
            point_1 = np.array(intersection[0], dtype=np.float32)
            point_2 = np.array(intersection[1], dtype=np.float32)
            direction = current_point - last_point
            if np.dot(point_1, direction) > np.dot(point_2, direction):
                intersection_point = point_1
            else:
                intersection_point = point_2
        elif len(intersection) == 1:
            intersection_point = np.array(intersection[0], dtype=np.float32)
        else:
            return _pad_rows(np.asarray(interpolated_route_points, dtype=np.float32), rows)

        last_interpolated_point = intersection_point
        interpolated_route_points.append(intersection_point)
        min_distance = 1.0
        first_iteration = False

    return np.asarray(interpolated_route_points, dtype=np.float32)


def _smooth_route(route: np.ndarray, rows: int = 10) -> np.ndarray:
    """LEAD smooth_path 等价逻辑：先平滑前 20 个 route 点，再截取预测需要的行数。"""
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    smoothing_rows = max(rows, 20)
    route = route[:smoothing_rows]
    if route.shape[0] == 0:
        return np.zeros((rows, 2), dtype=np.float32)
    _, indices = np.unique(route, return_index=True, axis=0)
    route = route[np.sort(indices).astype(int)]
    if route.shape[0] < 2:
        return _pad_rows(route, rows)
    return _lead_iterative_line_interpolation(route, smoothing_rows)[:rows]


# 这个 cache 保持适中；每个 DDP rank 都会各自持有一份。
# 主要用于避免重复 anchor 反复做 lzma 解压。
@lru_cache(maxsize=4096)
def _load_meta_cached(route_dir: str, anchor: int) -> dict[str, Any]:
    """加载并缓存一个压缩 anchor meta pickle。"""
    with lzma.open(Path(route_dir) / "metas" / f"{anchor:04d}.pkl", "rb") as f:
        return pickle.load(f)


def _load_meta(route_dir: Path, anchor: int) -> dict[str, Any]:
    """给缓存 meta loader 包一层 Path 友好的入口。"""
    return _load_meta_cached(str(route_dir), int(anchor))


def _extract_targets(
    sample: dict[str, Any],
    route_rows: int,
    waypoint_rows: int,
    smooth_route: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回绝对 ego-frame GT 点，对齐 LEAD loss 语义。

    LEAD head 内部先预测点间 delta，再 cumsum 后计算 loss。
    因此这里的标签保持累计 ego-local 坐标，而不是相邻 delta。
    """
    route_dir = Path(sample["route_dir"])
    meta = _load_meta(route_dir, int(sample["anchor"]))

    future_positions = np.asarray(meta["future_positions"], dtype=np.float32).reshape(-1, 3)
    indices = sample.get("future_waypoint_indices") or [5, 10, 15, 20, 25, 30, 35, 40]
    waypoints = []
    for idx in indices[:waypoint_rows]:
        waypoints.append(future_positions[min(int(idx), len(future_positions) - 1), :2])
    gt_waypoints = _pad_rows(np.asarray(waypoints, dtype=np.float32), waypoint_rows)

    route = np.asarray(meta["route"], dtype=np.float32).reshape(-1, 2)
    gt_route = _smooth_route(route, route_rows) if smooth_route else _pad_rows(route, route_rows)
    return torch.from_numpy(gt_route), torch.from_numpy(gt_waypoints)


def _point_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """根据配置分派到对应的逐点回归 loss。"""
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"unsupported loss_type: {loss_type}")


def _planning_loss(
    outputs: dict[str, torch.Tensor],
    gt_route: torch.Tensor,
    gt_waypoints: torch.Tensor,
    route_weight: float,
    waypoint_weight: float,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 waypoint loss 和 route ADE+FDE loss 合成总规划 loss。"""
    pred_wp = outputs["pred_future_waypoints"].float()
    pred_route = outputs["pred_route"].float()
    gt_waypoints = gt_waypoints.float()
    gt_route = gt_route.float()
    wp_loss = _point_loss(pred_wp, gt_waypoints, loss_type)
    route_ade = _point_loss(pred_route, gt_route, loss_type)
    route_fde = _point_loss(pred_route[:, -1, :], gt_route[:, -1, :], loss_type)
    route_loss = route_ade + route_fde
    return route_weight * route_loss + waypoint_weight * wp_loss, route_loss, wp_loss


@torch.no_grad()
def _compute_planning_metrics(
    outputs: dict[str, torch.Tensor],
    gt_route: torch.Tensor,
    gt_waypoints: torch.Tensor,
) -> dict[str, float]:
    """计算 train/eval/probe 共用的米制 ADE/FDE 指标。"""
    pred_route = outputs["pred_route"].float()
    pred_wp = outputs["pred_future_waypoints"].float()
    route_err = torch.linalg.norm(pred_route - gt_route.float(), dim=-1)
    wp_err = torch.linalg.norm(pred_wp - gt_waypoints.float(), dim=-1)
    return {
        "route_ade_m": float(route_err.mean().item()),
        "route_fde_m": float(route_err[:, -1].mean().item()),
        "waypoint_ade_m": float(wp_err.mean().item()),
        "waypoint_fde_m": float(wp_err[:, -1].mean().item()),
    }


def _optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """把 AdamW 参数拆成 decay / no-decay 两组。

    矩阵权重使用 weight decay；bias、norm 权重、embedding 和 query bank 不用，
    因为 decay 会把这些 learned lookup vector 往 0 拉。
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    embed_keywords = ("pos_embed", "query_bank", ".embed.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embed = any(key in name for key in embed_keywords)
        if param.ndim >= 2 and not is_embed:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _param_breakdown(decoder: torch.nn.Module, runtime: "LeadMoTTrainRuntime | None" = None) -> dict[str, Any]:
    """返回启动时 trainable/frozen 模块的参数量统计。"""
    module = decoder.module if hasattr(decoder, "module") else decoder

    def _count(mod: torch.nn.Module | None) -> tuple[int, int]:
        """返回单个模块的总参数量和可训练参数量。"""
        if mod is None:
            return (0, 0)
        total = sum(p.numel() for p in mod.parameters())
        trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return total, trainable

    decoder_breakdown: dict[str, int] = {}
    for name, sub in module.named_children():
        decoder_breakdown[name] = sum(p.numel() for p in sub.parameters() if p.requires_grad)

    out: dict[str, Any] = {
        "decoder_total": int(sum(p.numel() for p in module.parameters())),
        "decoder_trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
        "decoder_breakdown": decoder_breakdown,
    }
    if runtime is not None:
        qwen_total, qwen_train = _count(getattr(runtime.runner.leadmot_qwen_engine, "model", None))
        bev_total, bev_train = _count(runtime.runner.bev_encoder)
        out["qwen_total"] = int(qwen_total)
        out["qwen_trainable"] = int(qwen_train)
        out["bev_encoder_total"] = int(bev_total)
        out["bev_encoder_trainable"] = int(bev_train)
    return out


# ============================================================
# Decoder EMA 辅助类。
# ============================================================
class _DecoderEMA:
    """维护 decoder 参数的 fp32 指数滑动平均。

    shadow key 在 DDP wrap 前创建，因此 update/apply 必须传 unwrapped module。
    eval/probe 默认优先使用这些 EMA 权重。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # 即使 decoder 用 bf16 训练，EMA shadow 也保持 fp32。
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """执行 shadow = decay * shadow + (1 - decay) * param。"""
        one_minus = 1.0 - self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=one_minus)

    def state_dict(self) -> dict[str, Any]:
        """返回可写入 checkpoint 的 EMA 状态。"""
        return {
            "decay": self.decay,
            "shadow": {name: tensor.cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        """从 checkpoint 恢复 EMA shadow。"""
        shadow = state.get("shadow", state)
        if strict and set(shadow) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(shadow))
            extra = sorted(set(shadow) - set(self.shadow))
            raise RuntimeError(f"EMA state mismatch: missing={missing[:5]} extra={extra[:5]}")
        if "decay" in state:
            self.decay = float(state["decay"])
        for name, tensor in shadow.items():
            if name in self.shadow:
                self.shadow[name] = tensor.detach().float().clone()

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """临时把 decoder 参数替换成 EMA shadow，退出时恢复原权重。"""
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                with torch.no_grad():
                    p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    with torch.no_grad():
                        p.data.copy_(backup[name])


class LeadMoTTrainRuntime:
    """复用 runner 预处理路径的 runtime wrapper，Qwen/BEV 全部 frozen。"""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device

        # 训练入口按从 AutoMoT 目录启动设计，因此 model_dir 和 BEV ckpt
        # 默认都相对 AutoMoT/checkpoints。
        qwen_dir = Path(args.model_dir).expanduser().resolve()
        if not qwen_dir.exists():
            raise FileNotFoundError(f"Qwen checkpoint not found: {qwen_dir}")

        # 构造 runner 前先 patch 模块级路径。Qwen/BEV setup 由 runner 统一负责，
        # 这样 train/eval/probe 与 offline inference 的行为保持一致。
        mot_runner.LEAD_BEV_CKPT_PATH = Path(args.lead_bev_ckpt).expanduser().resolve()
        mot_runner._QWEN_INSTRUCT_CHECKPOINT_DIR = qwen_dir
        self.runner = LeadOfflineMoTRunner(
            device=str(device),
            leadmot_ckpt_path=None,
            leadmot_rope_type=args.leadmot_rope_type,
        )
        self.runner.leadmot_qwen_dtype = args.qwen_dtype
        self.runner.bev_encoder.eval()
        for param in self.runner.bev_encoder.parameters():
            param.requires_grad_(False)

        # 多 rank 加载 Qwen 时错峰，避免所有 rank 同时猛读共享文件系统上的
        # safetensors 权重文件。
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank > 0 and args.qwen_load_stagger_s > 0:
            time.sleep(local_rank * args.qwen_load_stagger_s)
        self.runner._ensure_leadmot_qwen_engine()
        if self.runner.leadmot_qwen_engine is None:
            raise RuntimeError("LeadMoT Qwen engine was not initialized")
        self.runner.leadmot_qwen_engine.model.eval()
        for param in self.runner.leadmot_qwen_engine.model.parameters():
            param.requires_grad_(False)

    def _build_clip(self, sample: dict[str, Any]):
        """为一个 JSONL 样本构建 runner 实际使用的 clip dict。"""
        kwargs = dict(
            route_dir=Path(sample["route_dir"]),
            anchor=int(sample["anchor"]),
            rgb_frame_step=int(sample.get("rgb_frame_step", self.args.rgb_frame_step)),
            rgb_frame_count=int(sample.get("rgb_frame_count", self.args.rgb_frame_count)),
            bev_frame_step=int(sample.get("bev_frame_step", self.args.bev_frame_step)),
            bev_frame_count=int(sample.get("bev_frame_count", self.args.bev_frame_count)),
            tp_lookahead_s=float(sample.get("target_point_lookahead_s", self.args.target_point_lookahead_s)),
            ntp_lookahead_s=float(
                sample.get("next_target_point_lookahead_s", self.args.next_target_point_lookahead_s)
            ),
            frame_interval_s=float(sample.get("frame_interval_s", self.args.frame_interval_s)),
            tp_mode=str(sample.get("tp_mode", getattr(self.args, "tp_mode", "route_lookahead"))),
            tp_min_lookahead_m=float(
                sample.get("tp_min_lookahead_m", getattr(self.args, "tp_min_lookahead_m", 5.0))
            ),
            use_final_goal=bool(
                sample.get("use_final_goal", getattr(self.args, "use_final_goal", True))
            ),
        )
        if self.args.verbose_samples:
            return build_clip_from_real_lead_route(**kwargs)
        # builder 会打印逐样本 debug 行；正常训练时静默掉，保持日志可读。
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            return build_clip_from_real_lead_route(**kwargs)

    def _run_subgoal_qwen_prefill(
        self,
        rgb_pil_list: list,
        sample: dict[str, Any],
        navigation_prompt: str,
    ) -> tuple[Any, int | None]:
        """LeadMoT subgoal 模式 Qwen prefill：薄包装，复用 runner 的同名实现。

        训练 / 离线 runner 两边必须走完全同款 prefix prompt + 同款图像顺序，否则
        prefix KV 分布不一致会导致 ckpt 在 runner 端 attention 错配。这里直接转发到
        runner._run_leadmot_qwen_prefill_subgoal，避免出现两份各自维护的实现。
        sample 字段由 build_dataset.py --with-subgoal-fields 写入；上游过滤 +
        forward_sample 内 hard guard 共同保证 subgoal_lookup_ok=True，因此这里只读
        scenario/status/subgoal/subgoal_rgb_path 四个字段。
        """

        return self.runner._run_leadmot_qwen_prefill_subgoal(
            rgb_pil_list=rgb_pil_list,
            navigation_prompt=navigation_prompt,
            subgoal_rgb_path=str(sample["subgoal_rgb_path"]),
            subgoal_scenario=str(sample["scenario"]),
            subgoal_status=str(sample["status"]),
            subgoal_event=str(sample["subgoal"]),
        )

    def forward_sample(
        self,
        sample: dict[str, Any],
        decoder: torch.nn.Module,
        decoder_config: LeadMoTPlanningDecoderConfig,
        decoder_dtype: torch.dtype,
        clip: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """先跑 frozen 预处理/prefill，再调用可训练 decoder。

        ``clip=None``（默认）：和老路径完全兼容，本方法自己调 ``_build_clip``。
        ``clip != None``：复用 DataLoader worker 预取的 clip dict，主进程跳过
        build_clip（最耗 IO 的 lzma/JPG/LAZ 读取）。两条路径数值完全等价。
        """
        if clip is None:
            clip = self._build_clip(sample)
        clip_len = int(np.asarray(clip["rgb"]).shape[0])
        # build_clip_from_real_lead_route 返回的 clip 里 anchor 永远是最后一帧，
        # 因此 anchor_t = clip_len - 1。
        group = self.runner._build_group_indices(
            clip_len=clip_len,
            anchor_t=clip_len - 1,
            rgb_frame_step=int(sample.get("rgb_frame_step", self.args.rgb_frame_step)),
            rgb_frame_count=int(sample.get("rgb_frame_count", self.args.rgb_frame_count)),
        )
        (
            rgb_pil_list,
            _lidar_pil_list,
            target_point_speed,
            bev_rgb_tensor,
            bev_lidar_tensor,
            *_rest,
        ) = self.runner._prepare_inference_inputs(clip, group)

        with torch.no_grad():
            # Qwen 和 BEV 都 frozen；只有这个 block 后面的 decoder 调用参与 autograd。
            prompt_cleaned, _enable_thinking, _enable_mot_reasoning = build_cleaned_prompt_and_modes(target_point_speed)
            if decoder_config.use_subgoal:
                # subgoal 分支：追加 subgoal keyframe RGB + 新 system + STATUS/SUBGOAL 文本块。
                # decoder 走的 prefix KV 与 use_subgoal=False 分布完全不同，跨开关的 ckpt 不兼容。
                # 显式校验 subgoal_lookup_ok：use_subgoal=True 时只能消费
                # build_dataset --with-subgoal-fields 成功反查到 SUBGOAL 的样本。
                # 训练入口会预先过滤；这里保留硬校验，避免旧 jsonl 或手工 sample
                # 静默走错 prefix 分布。
                if sample.get("subgoal_lookup_ok") is not True:
                    raise RuntimeError(
                        "decoder_config.use_subgoal=True requires "
                        "sample.subgoal_lookup_ok is True "
                        f"(got {sample.get('subgoal_lookup_ok')!r}, "
                        f"reason={sample.get('subgoal_skip_reason')})"
                    )
                past_key_values, rope_position_offset = self._run_subgoal_qwen_prefill(
                    rgb_pil_list=rgb_pil_list,
                    sample=sample,
                    navigation_prompt=prompt_cleaned,
                )
            else:
                past_key_values, rope_position_offset = self.runner._run_leadmot_qwen_prefill(
                    rgb_pil_list=rgb_pil_list,
                    user_prompt=prompt_cleaned,
                )
            pooled_kv = _segment_qwen_cache_for_leadmot(past_key_values, decoder_config)
            # use_bev=False：跳过 BEV encoder 调用，省去整套 LEAD TransfuserBackbone
            # 的 forward 时间和显存；decoder 内部自然走 use_bev=False 分支不拼 BEV token。
            if decoder_config.use_bev:
                bev_features = self.runner.bev_encoder(rgb=bev_rgb_tensor, lidar_bev=bev_lidar_tensor)["bev_feature"]
            else:
                bev_features = None

        status = target_point_speed.to(device=self.device, dtype=decoder_dtype)
        # runner 给出的 target_point_speed 布局：
        # [speed, target_x, target_y, next_target_x, next_target_y, final_goal_x, final_goal_y]。
        final_goal = None
        if decoder_config.use_final_goal:
            if status.shape[-1] < 7:
                raise ValueError(
                    "decoder_config.use_final_goal=True 但 target_point_speed 缺少 final_goal；"
                    f"当前 shape={tuple(status.shape)}。请确认 dataset row/use_final_goal 与 build_clip 同步。"
                )
            final_goal = status[:, 5:7]
        outputs = decoder(
            pooled_kv=pooled_kv,
            bev=(bev_features.to(device=self.device, dtype=decoder_dtype)
                 if bev_features is not None else None),
            speed=status[:, 0],
            target_point=status[:, 1:3],
            target_point_next=status[:, 3:5],
            final_goal=final_goal,
            rope_position_offset=rope_position_offset,
        )
        # 供 eval/probe/debug dump 核对输入导航状态；loss 只读取 pred_* 键。
        outputs["input_status"] = status.detach()
        return outputs


# ============================================================
# DataLoader 多 worker 预取支持（B=1，主要为了把 IO+解压搬出 GPU 关键路径）
# ============================================================
class LeadMoTSampleDataset(Dataset):
    """worker 端只做 CPU 部分：build_clip + GT 提取。

    设计意图：训练单 sample 流耗时大头是 4 张 JPG 解码 + 4 个 lzma pickle 解压 +
    4 个 LAZ 点云读取，全是 CPU/磁盘 IO。把这些放进 DataLoader worker 子进程后，
    主进程 GPU 算 Qwen prefill / decoder 时 worker 已经在后台把下一个 sample 备好，
    主进程从 dataloader 取 batch 几乎不等待。

    返回 dict 的字段：
      - sample: 原 jsonl row（passthrough，给 forward_sample 看 use_final_goal 等）
      - clip:   build_clip_from_real_lead_route 的产物（含 rgb numpy / lidar_points
                list / target_point / final_goal 等）
      - gt_route / gt_waypoints: torch.Tensor，CPU
      - 失败时改返回 {"sample": row, "_error": "<msg>"}，主进程检测后走占位 loss，
        保持 DDP collective 对齐
    """

    def __init__(self, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
        self.rows = rows
        self.args = args

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.rows[idx]
        try:
            kwargs = dict(
                route_dir=Path(sample["route_dir"]),
                anchor=int(sample["anchor"]),
                rgb_frame_step=int(sample.get("rgb_frame_step", self.args.rgb_frame_step)),
                rgb_frame_count=int(sample.get("rgb_frame_count", self.args.rgb_frame_count)),
                bev_frame_step=int(sample.get("bev_frame_step", self.args.bev_frame_step)),
                bev_frame_count=int(sample.get("bev_frame_count", self.args.bev_frame_count)),
                tp_lookahead_s=float(sample.get("target_point_lookahead_s", self.args.target_point_lookahead_s)),
                ntp_lookahead_s=float(
                    sample.get("next_target_point_lookahead_s", self.args.next_target_point_lookahead_s)
                ),
                frame_interval_s=float(sample.get("frame_interval_s", self.args.frame_interval_s)),
                tp_mode=str(sample.get("tp_mode", getattr(self.args, "tp_mode", "route_lookahead"))),
                tp_min_lookahead_m=float(
                    sample.get("tp_min_lookahead_m", getattr(self.args, "tp_min_lookahead_m", 5.0))
                ),
                use_final_goal=bool(
                    sample.get("use_final_goal", getattr(self.args, "use_final_goal", True))
                ),
            )
            # build_clip_from_real_lead_route 默认会逐 sample 打 debug 行；正常训练时静默
            # 掉，避免大量 worker 的 stdout 互相打架且压住主进程日志。
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                clip = build_clip_from_real_lead_route(**kwargs)
            gt_route, gt_waypoints = _extract_targets(
                sample,
                int(self.args.route_points),
                int(self.args.waypoint_points),
                bool(self.args.smooth_route),
            )
            return {
                "sample": sample,
                "clip": clip,
                "gt_route": gt_route,
                "gt_waypoints": gt_waypoints,
            }
        except Exception as exc:
            # worker 不抛异常，避免 DataLoader 直接挂；把错误信息回主进程，
            # 主进程走与原 try/except 同样的占位 loss 路径，DDP collective 对齐不破。
            return {"sample": sample, "_error": f"{type(exc).__name__}: {exc}"}


def _identity_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """B=1 专用 collate：直接返回 worker 输出的 dict，不做 stack。

    本路线不做 B>1 的批量化，只用多 worker 把 IO 移出 GPU 关键路径。要做 B>1 还
    需要 PrefixKVAttention 加 lang_kv attention mask；那是后续工作。
    """
    assert len(batch) == 1, f"identity collate requires batch_size=1, got {len(batch)}"
    return batch[0]


class _NoPadDistributedSampler(Sampler[int]):
    """DDP 分片 sampler：只做 rank::world_size，不 padding/复制样本。"""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=generator).tolist()
        else:
            indices = list(range(n))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _make_loader(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    shuffle: bool,
    epoch_seed: int,
) -> tuple[DataLoader, Sampler[int] | None]:
    """统一构造 train/val DataLoader 与 (可选) 无 padding DDP sampler。

    设计点：
    - B=1：见 _identity_collate 注释。
    - DDP：调用方可先传入当前 rank 的无重复 shard；若传 world_size>1，本函数
      也只做 rank::world_size 分片，不 padding/复制样本。
    - persistent_workers：epoch 切换不重启 worker，省 lzma/laspy 等模块 import 开销。
    - prefetch_factor：仅 num_workers>0 时生效；默认 2 已经够预热。
    - multiprocessing_context：默认 spawn，避免在 Qwen/CUDA 初始化后 fork worker。
    """

    dataset = LeadMoTSampleDataset(rows, args)
    sampler: Sampler[int] | None = None
    if world_size > 1:
        sampler = _NoPadDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=epoch_seed,
        )
    num_workers = max(0, int(getattr(args, "num_workers", 0)))
    loader_kwargs: dict[str, Any] = dict(
        batch_size=1,
        sampler=sampler,
        # sampler 已经决定顺序时 shuffle 参数必须留空 / False。
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=_identity_collate,
        pin_memory=False,  # 我们返回的多半是 numpy + dict + Python obj，pin 不到位反而慢
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", True))
        loader_kwargs["prefetch_factor"] = max(1, int(getattr(args, "prefetch_factor", 2)))
        context_name = str(getattr(args, "worker_multiprocessing_context", "spawn"))
        if context_name and context_name != "default":
            loader_kwargs["multiprocessing_context"] = torch.multiprocessing.get_context(context_name)
    return DataLoader(dataset, **loader_kwargs), sampler


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    """按 optimizer update step 创建 warmup + cosine LR schedule。"""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        """返回某个 optimizer step 对应的学习率倍率。"""
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(
    path: Path,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    decoder_config: LeadMoTPlanningDecoderConfig,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    best_val: float | None,
    ema: "_DecoderEMA | None" = None,
) -> None:
    """原子保存 decoder、optimizer、scheduler、config 和可选 EMA。"""
    module = decoder.module if hasattr(decoder, "module") else decoder
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = {
        "schema_version": 1,
        "decoder": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "decoder_config": asdict(decoder_config),
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema_decay"] = ema.decay
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _require_final_goal_checkpoint(state: Any, path: Path) -> None:
    """新路线舍弃旧 ckpt：checkpoint 必须显式记录 use_final_goal=True。"""
    if not isinstance(state, dict):
        raise ValueError(f"{path} 不是带 decoder_config 的 LeadMoT v2 checkpoint，拒绝加载旧 schema")
    cfg = state.get("decoder_config")
    if not isinstance(cfg, dict) or "use_final_goal" not in cfg:
        raise ValueError(f"{path} 缺少 decoder_config.use_final_goal，拒绝加载旧 LeadMoT ckpt")
    if not bool(cfg["use_final_goal"]):
        raise ValueError(f"{path} 的 decoder_config.use_final_goal=False；当前路线要求 final_goal=True")


def _require_subgoal_match(state: Any, path: Path, requested_use_subgoal: bool) -> None:
    """resume/init-from-ckpt 时 ckpt 的 use_subgoal 必须与本次训练设置完全一致。

    use_subgoal 不改 state_dict 形状，但改 prefix KV 分布——cross-load 会让 attention
    完全错配。旧 ckpt 缺字段时按 False 兜底（v1 / v2 prefix）。
    """
    if not isinstance(state, dict):
        return
    cfg = state.get("decoder_config")
    if not isinstance(cfg, dict):
        return
    ckpt_use_subgoal = bool(cfg.get("use_subgoal", False))
    if ckpt_use_subgoal != bool(requested_use_subgoal):
        raise ValueError(
            f"{path} 的 decoder_config.use_subgoal={ckpt_use_subgoal}, "
            f"但当前训练设置 use_subgoal={bool(requested_use_subgoal)}; "
            "subgoal prefix 与 non-subgoal prefix 不兼容，拒绝继续。"
        )


def _write_best_meta(output_dir: Path, checkpoint: Path, val_loss: float, epoch: int, step: int) -> None:
    """原子写出 best.pt 对应的人类可读元信息。"""
    meta = {
        "checkpoint": str(checkpoint),
        "val_loss": float(val_loss),
        "epoch": int(epoch),
        "step": int(step),
    }
    tmp_path = output_dir / f".best.json.tmp.{os.getpid()}"
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, output_dir / "best.json")


def _prune_old_checkpoints(output_dir: Path, keep_recent: int, glob_pat: str) -> None:
    """对某个滚动 checkpoint 池只保留最新的若干文件。"""
    if keep_recent <= 0:
        return
    ckpts = sorted(output_dir.glob(glob_pat))
    for stale in ckpts[:-keep_recent]:
        try:
            stale.unlink()
        except OSError:
            pass


def _load_checkpoint(
    path: Path,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: "_DecoderEMA | None" = None,
    requested_use_subgoal: bool = False,
) -> tuple[int, int, float | None]:
    """恢复 decoder、optimizer、scheduler 和可选 EMA 状态。"""
    state = torch.load(path, map_location="cpu")
    _require_final_goal_checkpoint(state, path)
    _require_subgoal_match(state, path, requested_use_subgoal)
    schema = int(state.get("schema_version", 0))
    if schema != 1:
        print(f"[resume] warning: checkpoint schema_version={schema} != 1, fields may have drifted: {path}")
    module = decoder.module if hasattr(decoder, "module") else decoder
    module.load_state_dict(state["decoder"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema_sd = state.get("ema_state_dict")
        if ema_sd is not None:
            ema.load_state_dict(ema_sd, strict=True)
            print(f"[resume] EMA restored (decay={state.get('ema_decay', ema.decay)})")
        else:
            print("[resume] WARN: --ema-decay set but checkpoint has no ema_state_dict; "
                  "EMA shadow re-initialized from current decoder weights.")
    return int(state.get("epoch", 0)), int(state.get("step", 0)), state.get("best_val")


def _load_decoder_only(path: Path, decoder: torch.nn.Module, requested_use_subgoal: bool = False) -> None:
    """只加载 decoder 权重，不加载 optimizer/scheduler，用于 init-from-ckpt。"""
    state = torch.load(path, map_location="cpu")
    _require_final_goal_checkpoint(state, path)
    _require_subgoal_match(state, path, requested_use_subgoal)
    state_dict = state["decoder"] if isinstance(state, dict) and "decoder" in state else state
    module = decoder.module if hasattr(decoder, "module") else decoder
    module.load_state_dict(state_dict, strict=True)


# ============================================================
# TensorBoard planning overlay 辅助函数。
# ============================================================
def _render_planning_overlay_np(
    pred_route: torch.Tensor,
    gt_route: torch.Tensor,
    pred_wp: torch.Tensor,
    gt_wp: torch.Tensor,
    title: str = "",
) -> np.ndarray:
    """把 route/waypoint 的预测与 GT 渲染成 TensorBoard RGB 图。

    这里用 PIL 而不是 matplotlib，避免远程训练任务依赖图形后端。
    """
    from PIL import Image, ImageDraw  # 延迟 import，减轻 train 启动路径。

    width, height = 540, 360
    margin = 32
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    all_pts = torch.cat([pred_route, gt_route, pred_wp, gt_wp], dim=0).cpu().float()
    max_x = max(float(all_pts[:, 0].max().item()), 10.0)
    min_x = min(float(all_pts[:, 0].min().item()), 0.0)
    max_abs_y = max(float(all_pts[:, 1].abs().max().item()), 5.0)

    def project(pt):
        """把 ego-frame 米制坐标映射到图像像素。"""
        x = float(pt[0])
        y = float(pt[1])
        px = margin + (x - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * margin)
        py = height / 2 - y / max(max_abs_y, 1e-6) * (height / 2 - margin)
        return (px, py)

    # 参考轴能让横向/前向尺度错误在 TensorBoard 上更容易被看出来。
    draw.line([(margin, height / 2), (width - margin, height / 2)], fill=(220, 220, 220), width=1)
    draw.line([project((0.0, -max_abs_y)), project((0.0, max_abs_y))], fill=(220, 220, 220), width=1)

    def poly(points, color, legend_row, label):
        """画一条轨迹折线和对应图例。"""
        pts = [project(p) for p in points]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        for p in pts:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color)
        draw.text((margin, 8 + legend_row * 14), label, fill=color)

    poly(gt_route, (40, 120, 220), 0, "gt route")
    poly(pred_route, (20, 70, 150), 1, "pred route")
    poly(gt_wp, (30, 160, 70), 2, "gt waypoint")
    poly(pred_wp, (210, 90, 40), 3, "pred waypoint")
    if title:
        draw.text((margin, height - 18), title, fill=(80, 80, 80))
    return np.asarray(img, dtype=np.uint8)


@torch.no_grad()
def _log_image_samples(
    writer: Any,
    runtime: "LeadMoTTrainRuntime",
    decoder: torch.nn.Module,
    decoder_config: LeadMoTPlanningDecoderConfig,
    val_samples: list[dict[str, Any]],
    args: argparse.Namespace,
    decoder_dtype: torch.dtype,
    step: int,
    ema: "_DecoderEMA | None",
) -> None:
    """周期性记录 raw/EMA 权重的 prediction-vs-GT overlay。

    decoder 参数必须传未包 DDP 的原始 module，因为 EMA shadow key 不带 DDP
    的 "module." 前缀。
    """
    if writer is None or not val_samples or args.image_log_samples <= 0:
        return

    def _to_tb(panels: list[np.ndarray]) -> torch.Tensor | None:
        """把 HWC uint8 panel 转成 TensorBoard 需要的 NCHW float tensor。"""
        if not panels:
            return None
        arr = np.stack(panels, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    decoder.eval()
    try:
        take = min(args.image_log_samples, len(val_samples))
        rng = random.Random(args.image_log_seed + step)
        picked = rng.sample(val_samples, take) if len(val_samples) > take else list(val_samples)

        panels_raw: list[np.ndarray] = []
        panels_ema: list[np.ndarray] = []
        for sample in picked:
            try:
                gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
                gt_route_b = gt_route.unsqueeze(0).to(runtime.device)
                gt_wp_b = gt_wp.unsqueeze(0).to(runtime.device)

                outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
                metrics = _compute_planning_metrics(outputs, gt_route_b, gt_wp_b)
                panels_raw.append(
                    _render_planning_overlay_np(
                        outputs["pred_route"][0].cpu().float(),
                        gt_route.cpu().float(),
                        outputs["pred_future_waypoints"][0].cpu().float(),
                        gt_wp.cpu().float(),
                        title=f"raw step={step} route_fde={metrics['route_fde_m']:.2f}m wp_fde={metrics['waypoint_fde_m']:.2f}m",
                    )
                )

                if ema is not None:
                    with ema.apply_to(decoder):
                        outputs_ema = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
                    metrics_ema = _compute_planning_metrics(outputs_ema, gt_route_b, gt_wp_b)
                    panels_ema.append(
                        _render_planning_overlay_np(
                            outputs_ema["pred_route"][0].cpu().float(),
                            gt_route.cpu().float(),
                            outputs_ema["pred_future_waypoints"][0].cpu().float(),
                            gt_wp.cpu().float(),
                            title=f"ema step={step} route_fde={metrics_ema['route_fde_m']:.2f}m wp_fde={metrics_ema['waypoint_fde_m']:.2f}m",
                        )
                    )
            except Exception as exc:
                print(f"[image-log] skip {sample.get('route_dir')}@{sample.get('anchor')}: {exc}", flush=True)
                continue

        raw_grid = _to_tb(panels_raw)
        if raw_grid is not None:
            writer.add_images("samples/planning_overlay_raw", raw_grid, step, dataformats="NCHW")
        ema_grid = _to_tb(panels_ema)
        if ema_grid is not None:
            writer.add_images("samples/planning_overlay_ema", ema_grid, step, dataformats="NCHW")
    finally:
        decoder.train()

@torch.no_grad()
def _validate(
    runtime: LeadMoTTrainRuntime,
    decoder: torch.nn.Module,
    decoder_config: LeadMoTPlanningDecoderConfig,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    decoder_dtype: torch.dtype,
    rank: int = 0,
    world_size: int = 1,
    ema: "_DecoderEMA | None" = None,
) -> dict[str, float]:
    """运行分片 validation，并在各 rank 间 all-reduce 汇总。

    decoder 必须是 unwrapped module。DDP 只用于训练梯度；
    validation 走 no_grad，并只 reduce scalar 累加量。
    """
    nan_summary = {"loss": float("nan"), "route_ade_m": float("nan"), "route_fde_m": float("nan"), "waypoint_ade_m": float("nan"), "waypoint_fde_m": float("nan")}
    if not samples:
        return nan_summary
    decoder.eval()
    if args.val_max_samples > 0 and len(samples) > args.val_max_samples:
        rng = random.Random(args.val_sample_seed)
        limited_samples = rng.sample(samples, args.val_max_samples)
    else:
        limited_samples = list(samples)

    # val 也走 DataLoader：先手动 rank::world_size 分片，再交给本 rank 的 worker。
    # 不用 DistributedSampler(drop_last=False)，避免它为了等分而 padding/复制样本，
    # 导致验证指标重复计入少量 case。
    val_rank_samples = limited_samples[rank::world_size]
    val_loader, val_sampler = _make_loader(
        val_rank_samples, args,
        rank=0, world_size=1,
        shuffle=False, epoch_seed=args.val_sample_seed,
    )
    if val_sampler is not None:
        # eval 路径只跑一次，set_epoch 给个固定值即可，避免 sampler 内部用默认 0 时
        # 在某些 PyTorch 版本里 emit warning。
        val_sampler.set_epoch(0)

    def _run_pass() -> tuple[float, dict[str, float], int]:
        """运行当前 rank 的 validation shard，并返回本地累加量。"""
        total_loss = 0.0
        agg = {"route_ade_m": 0.0, "route_fde_m": 0.0, "waypoint_ade_m": 0.0, "waypoint_fde_m": 0.0}
        local_count = 0
        for prepared in val_loader:
            sample = prepared["sample"]
            worker_error = prepared.get("_error")
            try:
                if worker_error:
                    raise RuntimeError(f"worker preprocess failed: {worker_error}")
                outputs = runtime.forward_sample(
                    sample, decoder, decoder_config, decoder_dtype,
                    clip=prepared["clip"],
                )
                gt_route = prepared["gt_route"].unsqueeze(0).to(runtime.device)
                gt_wp = prepared["gt_waypoints"].unsqueeze(0).to(runtime.device)
                loss, _route_loss, _wp_loss = _planning_loss(
                    outputs, gt_route, gt_wp,
                    args.route_loss_weight, args.waypoint_loss_weight, args.loss_type,
                )
                metrics = _compute_planning_metrics(outputs, gt_route, gt_wp)
            except Exception as exc:
                if rank == 0:
                    print(
                        f"[skip-val] sample={sample.get('route_dir')}@{sample.get('anchor')}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue
            total_loss += float(loss.item())
            for key in agg:
                agg[key] += metrics[key]
            local_count += 1
        return total_loss, agg, local_count

    try:
        if ema is not None:
            with ema.apply_to(decoder):
                total, agg, count = _run_pass()
        else:
            total, agg, count = _run_pass()
    finally:
        # 与 _log_image_samples 同理：无论 _run_pass 是否抛异常都恢复 train。
        # apply_to 自己的 try/finally 会把 shadow 写回 backup，不污染原权重。
        decoder.train()

    if dist.is_available() and dist.is_initialized():
        packed = torch.tensor(
            [total, agg["route_ade_m"], agg["route_fde_m"], agg["waypoint_ade_m"], agg["waypoint_fde_m"], float(count)],
            device=runtime.device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        total = float(packed[0].item())
        agg = {
            "route_ade_m": float(packed[1].item()),
            "route_fde_m": float(packed[2].item()),
            "waypoint_ade_m": float(packed[3].item()),
            "waypoint_fde_m": float(packed[4].item()),
        }
        count = int(packed[5].item())

    if count <= 0:
        return nan_summary
    return {
        "loss": total / count,
        "route_ade_m": agg["route_ade_m"] / count,
        "route_fde_m": agg["route_fde_m"] / count,
        "waypoint_ade_m": agg["waypoint_ade_m"] / count,
        "waypoint_fde_m": agg["waypoint_fde_m"] / count,
    }


def parse_args() -> argparse.Namespace:
    """解析 train CLI 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", default="checkpoints/leadmot_v1_data/train.jsonl")
    parser.add_argument("--val-jsonl", default="checkpoints/leadmot_v1_data/val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/leadmot_v1_decoder")
    parser.add_argument("--model-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lead-bev-ckpt", default=str(LEAD_BEV_CKPT_PATH))
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-from-ckpt", default="", help="Load decoder weights only and reset optimizer/scheduler.")
    parser.add_argument("--seed", type=int, default=2026)
    # decoder 从零初始化（约 150M），backbone 全冻结：多跑几个 epoch + 略高 LR 更易收敛。
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=3, help="Roll over epoch checkpoints; 0 keeps all. best.pt/latest.pt are never pruned.")
    parser.add_argument("--step-save-every", type=int, default=10000, help="Write an extra step-checkpoint-NNNNNN.pt every N optimizer steps; 0 disables step snapshots.")
    parser.add_argument("--keep-recent-step-checkpoints", type=int, default=3, help="Roll over step checkpoints; default 3 (last 30k steps at step-save-every=10000). Independent from epoch pool.")
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--val-max-samples", type=int, default=64)
    parser.add_argument("--val-sample-seed", type=int, default=202607)
    parser.add_argument("--route-loss-weight", type=float, default=0.5)
    parser.add_argument("--waypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-type", default="l1", choices=["l1", "smooth_l1"])
    parser.add_argument("--leadmot-rope-type", default="mrope", choices=["mrope", "mhrope", "none"])
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-dtype", default="bfloat16")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-load-stagger-s", type=float, default=2.0)
    parser.add_argument("--route-points", type=int, default=10)
    parser.add_argument("--waypoint-points", type=int, default=8)
    parser.add_argument("--smooth-route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rgb-frame-count", type=int, default=4)
    parser.add_argument("--rgb-frame-step", type=int, default=1)
    parser.add_argument("--bev-frame-count", type=int, default=1)
    parser.add_argument("--bev-frame-step", type=int, default=1)
    # use_bev：decoder 是否在 gen 序列里拼 BEV(120) token。默认 True（v1 全套行为）。
    # --no-use-bev 时 gen 序列只剩 22 个 status/query token（4 status + 18 query），decoder 完全靠 frozen
    # Qwen prefix K/V + 自车状态做 planning。注意切换 use_bev 时 state_dict 不兼容，
    # 切档必须从头训或单独 warm start，不能直接 --init-from-ckpt 跨 use_bev 加载。
    parser.add_argument("--use-bev", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-interval-s", type=float, default=0.25)
    # 默认 1.0s / 2.0s 对齐 LeadMoT v2 (wp 视野 2s, ntp 落 wp 末端)。
    # sample dict 里 target_point_lookahead_s 优先于 CLI；通过 build_dataset 写入。
    parser.add_argument("--target-point-lookahead-s", type=float, default=1.0)
    parser.add_argument("--next-target-point-lookahead-s", type=float, default=2.0)
    parser.add_argument(
        "--tp-mode", type=str, default="route_lookahead",
        choices=["route_lookahead", "future_truth"],
        help="route_lookahead 与在线 agent 完全同款; future_truth 是 v1 兼容选项."
    )
    parser.add_argument("--tp-min-lookahead-m", type=float, default=5.0)
    parser.add_argument(
        "--use-final-goal", action=argparse.BooleanOptionalAction, default=True,
        help="是否传 final_goal token 给 decoder; 必须与 decoder use_final_goal 一致."
    )
    # use_subgoal：仅离线 train/eval/probe 支持。开启后 prefix 多 1 张 subgoal
    # keyframe RGB + STATUS/SUBGOAL 文本块；ckpt 的 decoder_config.use_subgoal 必须与之同步。
    # 与 use_bev 正交，4 种组合都允许；cross-load 不兼容，由 _require_final_goal_checkpoint
    # / _require_subgoal_match 校验拒绝。
    parser.add_argument(
        "--use-subgoal", action=argparse.BooleanOptionalAction, default=False,
        help="离线 subgoal 模式: prefix 追加 subgoal keyframe RGB + STATUS/SUBGOAL 文本块. "
             "需配套 build_dataset --with-subgoal-fields 生成的 jsonl 字段一起使用."
    )
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--verbose-samples", action="store_true")
    parser.add_argument("--no-tb", action="store_true")
    # ---- DataLoader 多 worker 预取 ----
    # 默认 8 / rank：4 张 JPG + 4 个 lzma pickle + 4 个 LAZ 解压全在 worker 里并发，
    # 主进程不再等 IO。H20 节点（~96 物理核）单卡 / 4 卡 / 8 卡 DDP 跑下来 8 足够把
    # GPU util 推到 90%+。CPU 内存代价 ~800MB / rank，可忽略。
    # 想退回同步 IO（debug 用）显式 --num-workers 0。
    # 注意：worker 数只影响 GPU 计算利用率，**不影响 GPU 显存**；显存利用率高需要 B>1。
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker 数；0 表示主进程同步 IO，>0 启用后台预取。默认 8。")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="每个 worker 预取的 batch 数（仅 num_workers>0 生效）。默认 2。")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
                        help="num_workers>0 时 epoch 切换不重启 worker，省 lzma/laspy 等模块 import 开销。")
    parser.add_argument(
        "--worker-multiprocessing-context",
        default="spawn",
        choices=["spawn", "forkserver", "fork", "default"],
        help="DataLoader worker 启动方式；默认 spawn，避免 CUDA/Qwen 初始化后 fork。"
    )
    # ---- EMA ----
    # 默认开 EMA，decay=0.999 适配 LeadMoT 默认 3 epoch 短 schedule（warmup ~500 step）；
    # 长 schedule（>= 10 epoch）可考虑 0.9999 但需相应延长 warmup。--no-ema 关闭。
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    # ---- TB 图像样例（预测 vs 真值 overlay） ----
    parser.add_argument("--image-log-every", type=int, default=1000,
                        help="Write planning overlay images to TB every N optimizer steps; 0 disables.")
    parser.add_argument("--image-log-samples", type=int, default=4,
                        help="How many val samples to render per image-log step.")
    parser.add_argument("--image-log-seed", type=int, default=20260101,
                        help="Base seed for picking image-log samples; full seed = base + step.")
    return parser.parse_args()


def main() -> None:
    """在 Qwen 和 BEV encoder 保持 frozen 时训练 decoder。"""
    args = parse_args()
    _setup_offline_env()
    os.environ.setdefault("QWEN3VL_LOCAL_DTYPE", args.qwen_dtype)
    if args.resume and args.init_from_ckpt:
        raise ValueError("--resume and --init-from-ckpt are mutually exclusive")

    rank, local_rank, world_size = _init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir)
    _dump_invocation(output_dir, rank)

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 在 Ampere/Hopper GPU 上允许 TF32 matmul，换取更高吞吐。
    torch.set_float32_matmul_precision("high")
    # cuDNN benchmark 默认关：首见 conv shape 时探测多种 algorithm 会抬高瞬时 workspace 峰值；
    # 显存有余量想要那点速度时设 LEADMOT_CUDNN_BENCHMARK=1 显式开启。
    torch.backends.cudnn.benchmark = os.environ.get("LEADMOT_CUDNN_BENCHMARK", "0") == "1"
    train_rows = _read_jsonl(Path(args.train_jsonl))
    val_rows = _read_jsonl(Path(args.val_jsonl)) if Path(args.val_jsonl).exists() else []
    subgoal_filter_stats: dict[str, int] | None = None
    if bool(args.use_subgoal):
        # use_subgoal 是 prefix 分布开关，不是普通标签增强。开启后只允许
        # build_dataset --with-subgoal-fields 成功反查到 STATUS/SUBGOAL/keyframe RGB
        # 的样本进入训练；旧 jsonl 或 subgoal_lookup_ok=False 的行直接过滤掉。
        train_before = len(train_rows)
        val_before = len(val_rows)
        train_rows = [row for row in train_rows if row.get("subgoal_lookup_ok") is True]
        val_rows = [row for row in val_rows if row.get("subgoal_lookup_ok") is True]
        subgoal_filter_stats = {
            "train_before": train_before,
            "train_after": len(train_rows),
            "train_dropped": train_before - len(train_rows),
            "val_before": val_before,
            "val_after": len(val_rows),
            "val_dropped": val_before - len(val_rows),
        }
        if rank == 0:
            print(json.dumps({"subgoal_filter": subgoal_filter_stats}, ensure_ascii=False), flush=True)
    if args.limit_train_samples > 0:
        train_rows = train_rows[: args.limit_train_samples]
    if args.limit_val_samples > 0:
        val_rows = val_rows[: args.limit_val_samples]
    if not train_rows:
        if bool(args.use_subgoal):
            raise ValueError(
                "no training samples with subgoal_lookup_ok=True found; "
                "rebuild jsonl with leadmot/build_dataset.py --with-subgoal-fields "
                "and a valid --keyframes path."
            )
        raise ValueError("no training samples found")

    usable = (len(train_rows) // world_size) * world_size
    if usable == 0:
        raise ValueError(f"train samples ({len(train_rows)}) fewer than world_size ({world_size})")
    train_rows_total = len(train_rows)
    train_rows = train_rows[:usable]
    rank_rows = train_rows[rank:usable:world_size]
    # 每 rank 每 epoch 实际看到的样本数完全相同；尾部不足 world_size 的样本按旧逻辑截掉，
    # 避免 DistributedSampler padding 复制样本造成重复训练/重复计数。
    rank_samples_per_epoch = len(rank_rows)

    decoder_dtype = _dtype(args.decoder_dtype)
    decoder_config = LeadMoTPlanningDecoderConfig(
        num_route_queries=args.route_points,
        num_waypoint_queries=args.waypoint_points,
        rope_type=args.leadmot_rope_type,
        dropout=args.decoder_dropout,
        use_bev=args.use_bev,
        use_final_goal=bool(args.use_final_goal),
        use_subgoal=bool(args.use_subgoal),
    )
    decoder = LeadMoTPlanningDecoder(decoder_config).to(device=device, dtype=decoder_dtype)
    if args.init_from_ckpt:
        _load_decoder_only(Path(args.init_from_ckpt), decoder, requested_use_subgoal=bool(args.use_subgoal))
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(decoder, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(rank_samples_per_epoch / max(1, args.grad_accum_steps))
    total_steps = max(1, updates_per_epoch * args.num_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = _make_scheduler(optimizer, total_steps, args.warmup_ratio)

    # EMA shadow 必须在 DDP wrap 之前对 unwrapped 创建：apply_to / update 都 walk
    # named_parameters，DDP 包后 key 多 "module." 前缀会全部 miss。Resume 时把
    # EMA 一起 load 进来，让恢复后的 EMA shadow 跟训练状态一致。
    ema = _DecoderEMA(decoder, decay=args.ema_decay) if args.ema else None

    start_epoch = 0
    global_step = 0
    best_val: float | None = None
    if args.resume:
        start_epoch, global_step, best_val = _load_checkpoint(
            Path(args.resume), decoder, optimizer, scheduler, ema=ema,
            requested_use_subgoal=bool(args.use_subgoal),
        )

    runtime = LeadMoTTrainRuntime(args, device)
    _barrier()

    if world_size > 1:
        # find_unused_parameters=False：占位 loss 已显式触及全部可训参数（sum(p.sum()) * 0），
        # 每个 micro-step 上每个 param 都进入计算图；不需要 reducer 多扫一遍 unused。
        # 关掉省 1-3% 吞吐，且避免少数坏 sample 时 reducer 的额外路径。
        decoder = DistributedDataParallel(
            decoder,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    writer = None
    if rank == 0 and not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(Path(args.output_dir) / "tb")
        except Exception as exc:
            print(f"[rank0] TensorBoard disabled: {exc}")

    if rank == 0:
        module = decoder.module if hasattr(decoder, "module") else decoder
        startup_log = {
            "train_rows": len(train_rows),
            "train_rows_total_before_ddp_trim": train_rows_total,
            "train_rows_dropped_for_even_ddp": train_rows_total - len(train_rows),
            "rank_samples_per_epoch": rank_samples_per_epoch,
            "val_rows": len(val_rows),
            "num_workers": int(getattr(args, "num_workers", 0)),
            "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
            "persistent_workers": bool(getattr(args, "persistent_workers", True)),
            "world_size": world_size,
            "device": str(device),
            "lr": args.learning_rate,
            "epochs": args.num_epochs,
            "grad_accum": args.grad_accum_steps,
            "total_steps": total_steps,
            "loss_type": args.loss_type,
            "leadmot_rope_type": args.leadmot_rope_type,
            "ema": bool(args.ema),
            "ema_decay": float(args.ema_decay) if args.ema else None,
            "image_log_every": int(args.image_log_every),
            "use_subgoal": bool(args.use_subgoal),
            "subgoal_filter": subgoal_filter_stats,
            "label_semantics": "absolute ego-frame route/future_waypoints; decoder heads cumsum internal deltas",
            **_param_breakdown(decoder, runtime),
        }
        print(json.dumps(startup_log, ensure_ascii=False))

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    log_start = time.time()
    stop_training = False

    # Train DataLoader 在 epoch 之间复用同一份（persistent_workers=True 时不重启 worker）。
    # rank_rows 已经是当前 rank 的等长无重复 shard；shuffle=True 只在本 rank 内打乱，
    # 不再让 DistributedSampler padding/复制样本。
    train_loader, train_sampler = _make_loader(
        rank_rows, args,
        rank=0, world_size=1,
        shuffle=True, epoch_seed=args.seed,
    )

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        decoder_module = decoder.module if hasattr(decoder, "module") else decoder
        epoch_samples_len = len(train_loader)
        for micro_idx, prepared in enumerate(train_loader):
            is_update = (micro_idx + 1) % args.grad_accum_steps == 0 or (micro_idx + 1) == epoch_samples_len
            sync_ctx = decoder.no_sync() if world_size > 1 and not is_update else contextlib.nullcontext()
            sample = prepared["sample"]
            with sync_ctx:
                # 大规模离线数据里可能出现坏样本。失败样本也必须让每个 rank
                # 走一次 backward，否则 DDP collective 可能错位。
                # Worker 已经在 __getitem__ 里 try 一遍：失败时返回 _error 字段；
                # 这里把它当与原 raise 一样处理，统一走占位 loss 路径。
                worker_error = prepared.get("_error")
                try:
                    if worker_error:
                        raise RuntimeError(f"worker preprocess failed: {worker_error}")
                    outputs = runtime.forward_sample(
                        sample, decoder, decoder_config, decoder_dtype,
                        clip=prepared["clip"],
                    )
                    gt_route = prepared["gt_route"].unsqueeze(0).to(device)
                    gt_wp = prepared["gt_waypoints"].unsqueeze(0).to(device)
                    loss, route_loss, waypoint_loss = _planning_loss(
                        outputs, gt_route, gt_wp, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type
                    )
                    # 米制指标和 loss 一起记，方便看曲线时做 sanity check。
                    train_metrics = _compute_planning_metrics(outputs, gt_route, gt_wp)
                    micro_loss = loss / max(1, args.grad_accum_steps)
                except Exception as exc:
                    if rank == 0 or args.verbose_samples:
                        print(
                            f"[skip] sample={sample.get('route_dir')}@{sample.get('anchor')}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    # 用 0 loss 触碰所有可训练参数。梯度仍是 0，
                    # 但 DDP reducer 能看到一致的计算图。
                    placeholder = sum(
                        p.sum() for p in decoder_module.parameters() if p.requires_grad
                    )
                    micro_loss = placeholder * 0.0
                    loss = torch.zeros((), device=device)
                    route_loss = torch.zeros((), device=device)
                    waypoint_loss = torch.zeros((), device=device)
                    train_metrics = {
                        "route_ade_m": float("nan"),
                        "route_fde_m": float("nan"),
                        "waypoint_ade_m": float("nan"),
                        "waypoint_fde_m": float("nan"),
                    }
                micro_loss.backward()

            if is_update:
                module = decoder.module if hasattr(decoder, "module") else decoder
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    # EMA 必须在 optimizer.step() 后更新，否则会慢一个 step。
                    ema.update(decoder_module)
                global_step += 1

                if rank == 0 and global_step % args.logging_steps == 0:
                    elapsed = max(time.time() - log_start, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"step={global_step} epoch={epoch + 1} loss={loss.item():.4f} "
                        f"route={route_loss.item():.4f} wp={waypoint_loss.item():.4f} "
                        f"route_fde={train_metrics['route_fde_m']:.2f}m "
                        f"wp_fde={train_metrics['waypoint_fde_m']:.2f}m "
                        f"grad={float(grad_norm):.3f} lr={lr:.3e} updates/s={args.logging_steps / elapsed:.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/route_loss", route_loss.item(), global_step)
                        writer.add_scalar("train/waypoint_loss", waypoint_loss.item(), global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                        for mkey, mval in train_metrics.items():
                            if math.isfinite(mval):
                                writer.add_scalar(f"train/{mkey}", mval, global_step)
                    log_start = time.time()

                # 图像日志由 rank0 用 no_grad 额外跑一遍，前后用 barrier 包住。
                # 用未包 DDP 的 module + no_grad，不触发 DDP 梯度同步，对后续训练 collective 无副作用。
                # 每个 rank 每个 micro-step 仍会跑 backward，因此 NCCL 保持对齐。
                do_image_log = (
                    args.image_log_every > 0
                    and global_step % args.image_log_every == 0
                    and bool(val_rows)
                )
                if do_image_log:
                    _barrier()
                    if rank == 0:
                        # rank0 渲染失败（PIL / OOM / TB writer）不应卡死其它 rank：
                        # 其它 rank 已 barrier 等本步 rank0 完成，未捕获异常会让后续 barrier 永远等不到。
                        # try/except 兜底：log 一行警告然后继续训练。decoder.eval()/train() 已在
                        # _log_image_samples 内部 try/finally 保护；这里再包一层防御。
                        try:
                            _log_image_samples(
                                writer=writer,
                                runtime=runtime,
                                decoder=module,
                                decoder_config=decoder_config,
                                val_samples=val_rows,
                                args=args,
                                decoder_dtype=decoder_dtype,
                                step=global_step,
                                ema=ema,
                            )
                        except Exception as exc:
                            print(f"[image-log] step={global_step} failed: {type(exc).__name__}: {exc}", flush=True)
                            # 保险：万一 decoder 残留在 eval 模式或 EMA 没还原，强制恢复。
                            module.train()
                    _barrier()
                    decoder.train()

                do_val = args.val_steps > 0 and bool(val_rows) and global_step % args.val_steps == 0
                if do_val:
                    _barrier()
                    val_summary = _validate(
                        runtime, module, decoder_config, val_rows, args, decoder_dtype,
                        rank, world_size, ema=None,
                    )
                    val_summary_ema: dict[str, float] | None = None
                    if ema is not None:
                        _barrier()
                        val_summary_ema = _validate(
                            runtime, module, decoder_config, val_rows, args, decoder_dtype,
                            rank, world_size, ema=ema,
                        )
                    if rank == 0:
                        msg = (
                            f"val step={global_step} loss={val_summary['loss']:.4f} "
                            f"route_fde={val_summary['route_fde_m']:.2f}m "
                            f"wp_fde={val_summary['waypoint_fde_m']:.2f}m"
                        )
                        if val_summary_ema is not None:
                            msg += (
                                f" | ema loss={val_summary_ema['loss']:.4f} "
                                f"route_fde={val_summary_ema['route_fde_m']:.2f}m "
                                f"wp_fde={val_summary_ema['waypoint_fde_m']:.2f}m"
                            )
                        print(msg)
                        if writer is not None:
                            for vkey, vval in val_summary.items():
                                if math.isfinite(vval):
                                    writer.add_scalar(f"val/{vkey}", vval, global_step)
                            if val_summary_ema is not None:
                                for vkey, vval in val_summary_ema.items():
                                    if math.isfinite(vval):
                                        writer.add_scalar(f"val_ema/{vkey}", vval, global_step)
                        best_source = val_summary_ema if val_summary_ema is not None else val_summary
                        best_candidate = best_source["loss"]
                        if math.isfinite(best_candidate) and (best_val is None or best_candidate < best_val):
                            best_val = best_candidate
                            _save_checkpoint(
                                output_dir / "best.pt",
                                decoder, optimizer, scheduler, decoder_config, args,
                                epoch, global_step, best_val, ema=ema,
                            )
                            _write_best_meta(output_dir, output_dir / "best.pt", best_val, epoch + 1, global_step)
                    _barrier()
                    decoder.train()

                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(
                        output_dir / "latest.pt",
                        decoder, optimizer, scheduler, decoder_config, args,
                        epoch, global_step, best_val, ema=ema,
                    )

                # step 级独立快照池：长 epoch 下也能拿到中间产物，与 epoch 池互不淘汰。
                if rank == 0 and args.step_save_every > 0 and global_step % args.step_save_every == 0:
                    _save_checkpoint(
                        output_dir / f"step-checkpoint-{global_step:06d}.pt",
                        decoder, optimizer, scheduler, decoder_config, args,
                        epoch, global_step, best_val, ema=ema,
                    )
                    _prune_old_checkpoints(output_dir, args.keep_recent_step_checkpoints, "step-checkpoint-*.pt")

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    stop_training = True
                    break

        if rank == 0:
            _save_checkpoint(
                output_dir / f"checkpoint-epoch{epoch + 1:02d}.pt",
                decoder, optimizer, scheduler, decoder_config, args,
                epoch + 1, global_step, best_val, ema=ema,
            )
            _save_checkpoint(
                output_dir / "latest.pt",
                decoder, optimizer, scheduler, decoder_config, args,
                epoch + 1, global_step, best_val, ema=ema,
            )
            _prune_old_checkpoints(output_dir, args.keep_recent_checkpoints, "checkpoint-epoch*.pt")
        if stop_training:
            break

    if writer is not None:
        writer.close()
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        print(f"done: output_dir={output_dir} step={global_step} best_val={best_val}")


if __name__ == "__main__":
    main()
