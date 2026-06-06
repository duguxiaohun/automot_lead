#!/usr/bin/env python3
"""离线评估 LeadMoT v1 decoder checkpoint。"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from qwen3vl_local.leadmot import LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig
from qwen3vl_local.leadmot.train import (
    LEAD_BEV_CKPT_PATH,
    LeadMoTTrainRuntime,
    _barrier,
    _compute_planning_metrics,
    _dump_invocation,
    _dtype,
    _extract_targets,
    _planning_loss,
    _read_jsonl,
    _setup_offline_env,
)


DEFAULT_OUTPUT_ROOT = Path("checkpoints/leadmot_v1_decoder")


def _unwrap_ema_state_dict(ema_sd: Any) -> tuple[dict[str, Any] | None, Any]:
    """Return decoder-shaped EMA weights from either old or current checkpoint schema."""

    if not isinstance(ema_sd, dict):
        return None, None
    shadow = ema_sd.get("shadow")
    if isinstance(shadow, dict):
        return shadow, ema_sd.get("decay")
    return ema_sd, None


def _pick_idle_gpus(n: int = 1) -> str:
    """从 nvidia-smi 中选择占用最低的 GPU，失败则返回空字符串。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                rows.append((int(parts[1]), parts[0]))
            except ValueError:
                pass
    rows.sort()
    return ",".join(row[1] for row in rows[:n])


# DDP race 兜底：torchrun 多 worker 且外部未预设 CVD 时，只让 rank0 跑 nvidia-smi 挑卡
# → atomic 写共享文件，其它 rank 阻塞读，避免每 worker 各自挑卡导致 set_device 撞同一张卡。
_GPU_PICK_IMPORT_TIME = time.time()
_GPU_PICK_WAIT_TIMEOUT_S = 60.0
_GPU_PICK_STALE_TOLERANCE_S = 30.0
_GPU_PICK_LOCK_PREFIX = "leadmot_eval_cvd"


def _share_cvd_via_file_for_ddp(want_count: int) -> str:
    """rank0 挑卡 → 共享文件 → 其它 rank 读；锁文件按 MASTER_ADDR+MASTER_PORT 命名隔离
    不同 run，非 rank0 用 mtime >= 本进程 import 时刻 - 容差 拒绝上一轮残留旧文件。"""
    rank = int(os.environ.get("RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    lock_path = Path(tempfile.gettempdir()) / f"{_GPU_PICK_LOCK_PREFIX}_{master_addr}_{master_port}.txt"
    min_mtime = _GPU_PICK_IMPORT_TIME - _GPU_PICK_STALE_TOLERANCE_S
    if rank == 0:
        selected = _pick_idle_gpus(want_count)
        if not selected:
            return ""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        tmp_path = lock_path.with_suffix(f".tmp_{os.getpid()}")
        tmp_path.write_text(selected, encoding="utf-8")
        os.replace(tmp_path, lock_path)
        return selected
    deadline = time.time() + _GPU_PICK_WAIT_TIMEOUT_S
    while True:
        try:
            mtime = lock_path.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0
        if mtime >= min_mtime:
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"rank {rank} timed out waiting {_GPU_PICK_WAIT_TIMEOUT_S:.0f}s for "
                f"rank0 to publish CUDA_VISIBLE_DEVICES at {lock_path}"
            )
        time.sleep(0.05)
    return lock_path.read_text(encoding="utf-8").strip()


def _maybe_set_gpu(device: str) -> None:
    """自动选择空闲 GPU 并覆盖外层残留的 CUDA_VISIBLE_DEVICES。

    仅 --device 显式 cpu/cuda[:N] 时尊重用户锁卡，不覆盖。torchrun 多 worker 时
    由 rank0 挑 N 张经文件 IPC 同步给各 rank（避免每 worker 各自 nvidia-smi 重挑撞卡）。
    """
    if device and device.strip().lower() not in ("", "auto"):
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        selected = _share_cvd_via_file_for_ddp(world_size)
    else:
        selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected CUDA_VISIBLE_DEVICES={selected}; "
            f"world_size={world_size}"
        )


def _init_dist() -> tuple[int, int, int]:
    """当 eval 由 torchrun 启动时初始化 torch.distributed。

    与 train._init_distributed 使用同款 NCCL timeout：默认 10 分钟，
    可用 LEADMOT_NCCL_TIMEOUT_MIN 覆盖。eval 通常是分钟级任务，
    timeout 短一点更便于定位单 rank 卡死（Qwen load 慢 / NFS 抖动）。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = _dt.timedelta(minutes=int(os.environ.get("LEADMOT_NCCL_TIMEOUT_MIN", "10")))
        dist.init_process_group(backend=backend, timeout=timeout)
    return rank, local_rank, world_size


def _load_decoder(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    use_ema: bool = True,
) -> tuple[LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig]:
    """从 checkpoint 加载 decoder 权重。

    use_ema=True 时优先使用 ema_state_dict；旧 checkpoint 没有 EMA 字段时
    自动回退到 raw decoder 权重。
    """
    state = torch.load(path, map_location="cpu")
    cfg_dict = dict(state.get("decoder_config", {}))
    if "route_points" in cfg_dict and "num_route_queries" not in cfg_dict:
        cfg_dict["num_route_queries"] = cfg_dict.pop("route_points")
    if "waypoint_points" in cfg_dict and "num_waypoint_queries" not in cfg_dict:
        cfg_dict["num_waypoint_queries"] = cfg_dict.pop("waypoint_points")

    ema_sd = state.get("ema_state_dict") if isinstance(state, dict) else None
    ema_state_dict, ema_decay = _unwrap_ema_state_dict(ema_sd)
    using_ema = bool(use_ema and ema_state_dict is not None)
    state_dict = ema_state_dict if using_ema else state["decoder"]
    if "use_bev" not in cfg_dict:
        cfg_dict["use_bev"] = any(str(key).startswith("bev_projector.") for key in state_dict)
        print(
            "[leadmot] checkpoint has no decoder_config.use_bev; "
            f"inferred use_bev={cfg_dict['use_bev']} from state_dict keys"
        )

    config = LeadMoTPlanningDecoderConfig(**{k: v for k, v in cfg_dict.items() if k in LeadMoTPlanningDecoderConfig.__dataclass_fields__})
    decoder = LeadMoTPlanningDecoder(config).to(device=device, dtype=dtype)

    if using_ema:
        decoder.load_state_dict(state_dict, strict=True)
        decay = ema_decay if ema_decay is not None else state.get("ema_decay", "unknown")
        print(f"[leadmot] using EMA weights (decay={decay})")
    else:
        if use_ema and ema_sd is None:
            print("[leadmot] WARN: --use-ema set but checkpoint has no ema_state_dict; using raw decoder weights")
        decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()
    return decoder, config


def _checkpoint_root(save_root: str) -> Path:
    """解析默认 checkpoint 搜索根目录。"""
    return Path(save_root) if save_root else DEFAULT_OUTPUT_ROOT


def _resolve_checkpoint(checkpoint: str, save_root: str) -> Path:
    """先解析显式 checkpoint，再尝试 best/latest，最后找最新快照。"""
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    root = _checkpoint_root(save_root)
    best = root / "best.pt"
    latest = root / "latest.pt"
    if best.exists():
        return best
    if latest.exists():
        return latest
    # 早期 smoke test 可能还没有 best/latest，此时使用最新快照。
    for glob_pat in ("step-checkpoint-*.pt", "checkpoint-epoch*.pt"):
        pool = sorted(root.glob(glob_pat))
        if pool:
            return pool[-1]
    raise FileNotFoundError(f"no default LeadMoT checkpoint found under {root}")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """对齐 GoalGen 布局：未显式覆盖时写到 <save-root>/eval。"""
    if args.output_dir:
        return Path(args.output_dir)
    root = Path(args.save_root) if args.save_root else DEFAULT_OUTPUT_ROOT
    return root / "eval"


# 旧版本本地实现的 _compute_metrics 已删除：现在统一从 train import
# _compute_planning_metrics。所有 ADE/FDE 都带 `_m` 后缀（米），与
# train/val/val_ema 同口径，TB 同名 scalar 可以叠在一张图上对比。
# 兼容 alias：让旧的 probe import 路径短期内继续可用，未来可以删。
_compute_metrics = _compute_planning_metrics


def parse_args() -> argparse.Namespace:
    """解析离线评估 CLI 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="checkpoints/leadmot_v1_data/val.jsonl")
    parser.add_argument("--checkpoint", default="", help="Default: <save-root>/best.pt -> latest.pt -> newest step/epoch checkpoint.")
    parser.add_argument("--save-root", default="", help="GoalGen-style root; eval artifacts go to <save-root>/eval when --output-dir is omitted.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lead-bev-ckpt", default=str(LEAD_BEV_CKPT_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--decoder-dtype", default="bfloat16")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-load-stagger-s", type=float, default=0.0)
    parser.add_argument("--route-loss-weight", type=float, default=0.5)
    parser.add_argument("--waypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-type", default="l1", choices=["l1", "smooth_l1"])
    parser.add_argument("--leadmot-rope-type", default="mrope", choices=["mrope", "mhrope", "none"])
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
    parser.add_argument("--verbose-samples", action="store_true")
    # 默认用 EMA 权重做 eval；--no-use-ema 强制用 raw decoder。
    # 旧 ckpt（不带 ema_state_dict）下任一选项都会回落到 raw 权重，并 print 提示。
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """运行分片评估，并由 rank0 合并每个 rank 的 JSONL。"""
    args = parse_args()
    _setup_offline_env()
    _maybe_set_gpu(args.device)
    rank, local_rank, world_size = _init_dist()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    rows = _read_jsonl(Path(args.jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    shard = rows[rank::world_size]
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_root = Path(args.save_root) if args.save_root else output_dir.parent
    _dump_invocation(invocation_root, rank)

    decoder_dtype = _dtype(args.decoder_dtype)
    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.save_root)
    decoder, decoder_config = _load_decoder(checkpoint_path, device, decoder_dtype, use_ema=args.use_ema)
    runtime = LeadMoTTrainRuntime(args, device)

    # sums 的 key 直接对齐 train._compute_planning_metrics 返回的 `_m` 后缀键，
    # 这样 eval/* TB scalar 跟训练 train/* val/* val_ema/* 三组 scalar 完全同名，
    # 可以叠在同一张 TB 板上比对。
    sums = {
        "loss": 0.0,
        "route_loss": 0.0,
        "waypoint_loss": 0.0,
        "route_ade_m": 0.0,
        "route_fde_m": 0.0,
        "waypoint_ade_m": 0.0,
        "waypoint_fde_m": 0.0,
    }
    perline_path = output_dir / f"eval_v1_perline.rank{rank}.jsonl"
    count = 0
    with perline_path.open("w", encoding="utf-8", newline="\n") as f:
        with torch.no_grad():
            for idx, sample in enumerate(shard):
                try:
                    # eval 只在循环结束后 all-reduce 一次。坏样本可以在各 rank
                    # 独立跳过，最后按全局有效样本数平均。
                    outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
                    gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
                    gt_route = gt_route.unsqueeze(0).to(device)
                    gt_wp = gt_wp.unsqueeze(0).to(device)
                    loss, route_loss, wp_loss = _planning_loss(outputs, gt_route, gt_wp, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type)
                    metrics = _compute_planning_metrics(outputs, gt_route, gt_wp)
                except Exception as exc:
                    print(
                        f"[skip] rank={rank} sample={sample.get('route_dir')}@{sample.get('anchor')}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                row = {"index": idx * world_size + rank, "route_dir": sample.get("route_dir"), "anchor": sample.get("anchor"), "loss": float(loss.item()), "route_loss": float(route_loss.item()), "waypoint_loss": float(wp_loss.item()), **metrics}
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                for key in sums:
                    sums[key] += row[key]
                count += 1

    packed = torch.tensor([sums[key] for key in sums] + [float(count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    keys = list(sums)
    total_count = max(1, int(packed[-1].item()))
    summary = {key: float(packed[i].item() / total_count) for i, key in enumerate(keys)}
    summary.update({"count": total_count, "checkpoint": str(checkpoint_path), "jsonl": str(Path(args.jsonl)), "world_size": world_size, "use_ema": bool(args.use_ema)})

    _barrier()
    if rank == 0:
        merged = output_dir / "eval_v1_perline.jsonl"
        with merged.open("w", encoding="utf-8", newline="\n") as out:
            for r in range(world_size):
                part = output_dir / f"eval_v1_perline.rank{r}.jsonl"
                if part.exists():
                    out.write(part.read_text(encoding="utf-8"))
        (output_dir / "eval_v1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        # 可选 TensorBoard summary；未安装 tensorboard 不影响 eval 结果。
        try:
            import time as _time

            from torch.utils.tensorboard import SummaryWriter

            run_tag = f"{checkpoint_path.stem}_{_time.strftime('%Y%m%d_%H%M%S')}"
            tb_dir = invocation_root / "eval_tb" / run_tag
            with SummaryWriter(tb_dir) as tb_writer:
                for key, val in summary.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        tb_writer.add_scalar(f"eval/{key}", float(val), 0)
            print(f"[eval-tb] {tb_dir}")
        except Exception as exc:
            print(f"[eval-tb] disabled: {exc}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
