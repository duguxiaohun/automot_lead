#!/usr/bin/env python3
"""Train LeadMoT decoder with frozen Qwen3-VL and frozen LeadBEVEncoder."""

from __future__ import annotations

import argparse
import contextlib
import json
import lzma
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_invocation(output_dir: Path, rank: int = 0) -> None:
    if rank != 0:
        return
    try:
        import datetime as _dt
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
            "# ---- selected env vars ----",
            *[f"{key}={os.environ.get(key, '<unset>')}" for key in env_keys],
            "",
            "# ---- sys.argv (one per line) ----",
            *sys.argv,
            "",
            "# ---- shell replay ----",
            " ".join(_shlex.quote(arg) for arg in sys.argv),
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[invocation] saved -> {out_path}")
    except Exception as exc:
        print(f"[invocation] save failed (ignored): {exc}")


def _setup_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
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
    """LEAD carla_dataset_utils.circle_line_segment_intersection with full_line=True."""
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
    """Minimal LEAD iterative_line_interpolation equivalent for route labels."""
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
    """LEAD smooth_path equivalent: smooth first 20 route points, then keep prediction rows."""
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


@lru_cache(maxsize=16384)
def _load_meta_cached(route_dir: str, anchor: int) -> dict[str, Any]:
    with lzma.open(Path(route_dir) / "metas" / f"{anchor:04d}.pkl", "rb") as f:
        return pickle.load(f)


def _load_meta(route_dir: Path, anchor: int) -> dict[str, Any]:
    return _load_meta_cached(str(route_dir), int(anchor))


def _extract_targets(
    sample: dict[str, Any],
    route_rows: int,
    waypoint_rows: int,
    smooth_route: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute ego-frame GT points, matching LEAD loss semantics.

    LEAD heads predict point deltas internally, then cumsum them before loss.
    Therefore labels here stay as accumulated ego-local coordinates.
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
    pred_wp = outputs["pred_future_waypoints"].float()
    pred_route = outputs["pred_route"].float()
    gt_waypoints = gt_waypoints.float()
    gt_route = gt_route.float()
    wp_loss = _point_loss(pred_wp, gt_waypoints, loss_type)
    route_ade = _point_loss(pred_route, gt_route, loss_type)
    route_fde = _point_loss(pred_route[:, -1, :], gt_route[:, -1, :], loss_type)
    route_loss = route_ade + route_fde
    return route_weight * route_loss + waypoint_weight * wp_loss, route_loss, wp_loss


def _optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class LeadMoTTrainRuntime:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device

        qwen_dir = Path(args.model_dir).expanduser().resolve()
        if not qwen_dir.exists():
            raise FileNotFoundError(f"Qwen checkpoint not found: {qwen_dir}")
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
        )
        if self.args.verbose_samples:
            return build_clip_from_real_lead_route(**kwargs)
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            return build_clip_from_real_lead_route(**kwargs)

    def forward_sample(
        self,
        sample: dict[str, Any],
        decoder: torch.nn.Module,
        decoder_config: LeadMoTPlanningDecoderConfig,
        decoder_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        clip = self._build_clip(sample)
        clip_len = int(np.asarray(clip["rgb"]).shape[0])
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
            prompt_cleaned, _enable_thinking, _enable_mot_reasoning = build_cleaned_prompt_and_modes(target_point_speed)
            past_key_values, rope_position_offset = self.runner._run_leadmot_qwen_prefill(
                rgb_pil_list=rgb_pil_list,
                user_prompt=prompt_cleaned,
            )
            pooled_kv = _segment_qwen_cache_for_leadmot(past_key_values, decoder_config)
            bev_features = self.runner.bev_encoder(rgb=bev_rgb_tensor, lidar_bev=bev_lidar_tensor)["bev_feature"]

        status = target_point_speed.to(device=self.device, dtype=decoder_dtype)
        return decoder(
            pooled_kv=pooled_kv,
            bev=bev_features.to(device=self.device, dtype=decoder_dtype),
            speed=status[:, 0],
            target_point=status[:, 1:3],
            target_point_next=status[:, 3:5],
            rope_position_offset=rope_position_offset,
        )


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
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
) -> None:
    module = decoder.module if hasattr(decoder, "module") else decoder
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema_version": 1,
            "decoder": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "decoder_config": asdict(decoder_config),
            "args": vars(args),
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _write_best_meta(output_dir: Path, checkpoint: Path, val_loss: float, epoch: int, step: int) -> None:
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
    # 按 glob 滚动淘汰对应池；best.pt / latest.pt 不在任何池里，永远保留。
    # epoch 池 glob "checkpoint-epoch*.pt" 与 step 池 "step-checkpoint-*.pt" 前缀不同，互不污染。
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
) -> tuple[int, int, float | None]:
    state = torch.load(path, map_location="cpu")
    module = decoder.module if hasattr(decoder, "module") else decoder
    module.load_state_dict(state["decoder"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state.get("epoch", 0)), int(state.get("step", 0)), state.get("best_val")


def _load_decoder_only(path: Path, decoder: torch.nn.Module) -> None:
    state = torch.load(path, map_location="cpu")
    state_dict = state["decoder"] if isinstance(state, dict) and "decoder" in state else state
    module = decoder.module if hasattr(decoder, "module") else decoder
    module.load_state_dict(state_dict, strict=True)


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
) -> float:
    if not samples:
        return float("nan")
    decoder.eval()
    total = 0.0
    count = 0
    if args.val_max_samples > 0 and len(samples) > args.val_max_samples:
        rng = random.Random(args.val_sample_seed)
        limited_samples = rng.sample(samples, args.val_max_samples)
    else:
        limited_samples = list(samples)
    for sample in limited_samples[rank::world_size]:
        # 与 eval 同理：val 的 all_reduce 在循环后一次完成，坏样本 skip 不破坏 DDP 同步。
        try:
            outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
            gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
            gt_route = gt_route.unsqueeze(0).to(runtime.device)
            gt_wp = gt_wp.unsqueeze(0).to(runtime.device)
            loss, _route_loss, _wp_loss = _planning_loss(
                outputs, gt_route, gt_wp, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type
            )
        except Exception as exc:
            if rank == 0:
                print(
                    f"[skip-val] sample={sample.get('route_dir')}@{sample.get('anchor')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            continue
        total += float(loss.item())
        count += 1
    decoder.train()
    if dist.is_available() and dist.is_initialized():
        packed = torch.tensor([total, float(count)], device=runtime.device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        total = float(packed[0].item())
        count = int(packed[1].item())
    return total / max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", default="checkpoints/leadmot_v1_data/train.jsonl")
    parser.add_argument("--val-jsonl", default="checkpoints/leadmot_v1_data/val.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/leadmot_v1_decoder")
    parser.add_argument("--model-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lead-bev-ckpt", default=str(LEAD_BEV_CKPT_PATH))
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-from-ckpt", default="", help="Load decoder weights only and reset optimizer/scheduler.")
    parser.add_argument("--seed", type=int, default=2026)
    # decoder 从零初始化（~150M），backbone 全冻结：多跑几个 epoch + 略高 LR 更易收敛。
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
    parser.add_argument("--frame-interval-s", type=float, default=0.25)
    parser.add_argument("--target-point-lookahead-s", type=float, default=1.5)
    parser.add_argument("--next-target-point-lookahead-s", type=float, default=3.0)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--verbose-samples", action="store_true")
    parser.add_argument("--no-tb", action="store_true")
    return parser.parse_args()


def main() -> None:
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

    train_rows = _read_jsonl(Path(args.train_jsonl))
    val_rows = _read_jsonl(Path(args.val_jsonl)) if Path(args.val_jsonl).exists() else []
    if args.limit_train_samples > 0:
        train_rows = train_rows[: args.limit_train_samples]
    if args.limit_val_samples > 0:
        val_rows = val_rows[: args.limit_val_samples]
    if not train_rows:
        raise ValueError("no training samples found")

    usable = (len(train_rows) // world_size) * world_size
    if usable == 0:
        raise ValueError(f"train samples ({len(train_rows)}) fewer than world_size ({world_size})")
    train_rows = train_rows[:usable]
    rank_rows = train_rows[rank:usable:world_size]

    decoder_dtype = _dtype(args.decoder_dtype)
    decoder_config = LeadMoTPlanningDecoderConfig(
        num_route_queries=args.route_points,
        num_waypoint_queries=args.waypoint_points,
        rope_type=args.leadmot_rope_type,
        dropout=args.decoder_dropout,
    )
    decoder = LeadMoTPlanningDecoder(decoder_config).to(device=device, dtype=decoder_dtype)
    if args.init_from_ckpt:
        _load_decoder_only(Path(args.init_from_ckpt), decoder)
    optimizer = torch.optim.AdamW(
        _optimizer_param_groups(decoder, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(rank_rows) / max(1, args.grad_accum_steps))
    total_steps = max(1, updates_per_epoch * args.num_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = _make_scheduler(optimizer, total_steps, args.warmup_ratio)

    start_epoch = 0
    global_step = 0
    best_val: float | None = None
    if args.resume:
        start_epoch, global_step, best_val = _load_checkpoint(Path(args.resume), decoder, optimizer, scheduler)

    runtime = LeadMoTTrainRuntime(args, device)
    _barrier()

    if world_size > 1:
        decoder = DistributedDataParallel(decoder, device_ids=[local_rank] if device.type == "cuda" else None)

    writer = None
    if rank == 0 and not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(Path(args.output_dir) / "tb")
        except Exception as exc:
            print(f"[rank0] TensorBoard disabled: {exc}")

    if rank == 0:
        module = decoder.module if hasattr(decoder, "module") else decoder
        print(
            json.dumps(
                {
                    "train_rows": len(train_rows),
                    "rank_rows": len(rank_rows),
                    "val_rows": len(val_rows),
                    "world_size": world_size,
                    "device": str(device),
                    "lr": args.learning_rate,
                    "epochs": args.num_epochs,
                    "grad_accum": args.grad_accum_steps,
                    "total_steps": total_steps,
                    "loss_type": args.loss_type,
                    "leadmot_rope_type": args.leadmot_rope_type,
                    "trainable_params": sum(p.numel() for p in module.parameters() if p.requires_grad),
                    "label_semantics": "absolute ego-frame route/future_waypoints; decoder heads cumsum internal deltas",
                },
                ensure_ascii=False,
            )
        )

    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    log_start = time.time()
    stop_training = False

    for epoch in range(start_epoch, args.num_epochs):
        epoch_rows = list(rank_rows)
        random.Random(args.seed + epoch).shuffle(epoch_rows)
        decoder_module = decoder.module if hasattr(decoder, "module") else decoder
        for micro_idx, sample in enumerate(epoch_rows):
            is_update = (micro_idx + 1) % args.grad_accum_steps == 0 or (micro_idx + 1) == len(epoch_rows)
            sync_ctx = decoder.no_sync() if world_size > 1 and not is_update else contextlib.nullcontext()
            with sync_ctx:
                # 坏数据（缺帧 / LAZ 损坏 / meta 解压失败）随时可能 raise。这里做 DDP-safe
                # 兜底：失败时构造一个触及全部可训练参数的 0 loss，backward + all-reduce 仍能走完，
                # 各 rank 的 micro 步数和 is_update 边界完全一致，不会让 NCCL collective 错位挂死。
                try:
                    outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
                    gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
                    gt_route = gt_route.unsqueeze(0).to(device)
                    gt_wp = gt_wp.unsqueeze(0).to(device)
                    loss, route_loss, waypoint_loss = _planning_loss(
                        outputs, gt_route, gt_wp, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type
                    )
                    micro_loss = loss / max(1, args.grad_accum_steps)
                except Exception as exc:
                    if rank == 0 or args.verbose_samples:
                        print(
                            f"[skip] sample={sample.get('route_dir')}@{sample.get('anchor')}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    # 触及全部参数的占位 loss：保证 DDP reducer 对所有参数都 fire（与正常 forward 一致），
                    # 且梯度恒为 0，不污染本次 accumulation。
                    placeholder = sum(
                        p.sum() for p in decoder_module.parameters() if p.requires_grad
                    )
                    micro_loss = placeholder * 0.0
                    loss = torch.zeros((), device=device)
                    route_loss = torch.zeros((), device=device)
                    waypoint_loss = torch.zeros((), device=device)
                micro_loss.backward()

            if is_update:
                module = decoder.module if hasattr(decoder, "module") else decoder
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if rank == 0 and global_step % args.logging_steps == 0:
                    elapsed = max(time.time() - log_start, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"step={global_step} epoch={epoch + 1} loss={loss.item():.4f} "
                        f"route={route_loss.item():.4f} wp={waypoint_loss.item():.4f} "
                        f"grad={float(grad_norm):.3f} lr={lr:.3e} updates/s={args.logging_steps / elapsed:.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/route_loss", route_loss.item(), global_step)
                        writer.add_scalar("train/waypoint_loss", waypoint_loss.item(), global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                    log_start = time.time()

                do_val = args.val_steps > 0 and bool(val_rows) and global_step % args.val_steps == 0
                if do_val:
                    _barrier()
                    val_loss = _validate(runtime, module, decoder_config, val_rows, args, decoder_dtype, rank, world_size)
                    if rank == 0:
                        print(f"val step={global_step} loss={val_loss:.4f}")
                        if writer is not None:
                            writer.add_scalar("val/loss", val_loss, global_step)
                        if best_val is None or val_loss < best_val:
                            best_val = val_loss
                            _save_checkpoint(
                                output_dir / "best.pt",
                                decoder,
                                optimizer,
                                scheduler,
                                decoder_config,
                                args,
                                epoch,
                                global_step,
                                best_val,
                            )
                            _write_best_meta(output_dir, output_dir / "best.pt", best_val, epoch + 1, global_step)
                    _barrier()
                    decoder.train()

                if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(output_dir / "latest.pt", decoder, optimizer, scheduler, decoder_config, args, epoch, global_step, best_val)

                # step 级独立快照池：长 epoch 下也能拿到中间产物，与 epoch 池互不淘汰。
                if rank == 0 and args.step_save_every > 0 and global_step % args.step_save_every == 0:
                    _save_checkpoint(output_dir / f"step-checkpoint-{global_step:06d}.pt", decoder, optimizer, scheduler, decoder_config, args, epoch, global_step, best_val)
                    _prune_old_checkpoints(output_dir, args.keep_recent_step_checkpoints, "step-checkpoint-*.pt")

                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    stop_training = True
                    break

        if rank == 0:
            _save_checkpoint(output_dir / f"checkpoint-epoch{epoch + 1:02d}.pt", decoder, optimizer, scheduler, decoder_config, args, epoch + 1, global_step, best_val)
            _save_checkpoint(output_dir / "latest.pt", decoder, optimizer, scheduler, decoder_config, args, epoch + 1, global_step, best_val)
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
