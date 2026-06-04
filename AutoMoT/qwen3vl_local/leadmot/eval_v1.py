#!/usr/bin/env python3
"""Offline evaluation for LeadMoT v1 decoder checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from qwen3vl_local.leadmot import LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig
from qwen3vl_local.leadmot.train_v1 import (
    LEAD_BEV_CKPT_PATH,
    LeadMoTTrainRuntime,
    _barrier,
    _dump_invocation,
    _dtype,
    _extract_targets,
    _planning_loss,
    _read_jsonl,
    _setup_offline_env,
)


DEFAULT_OUTPUT_ROOT = Path("checkpoints/leadmot_v1_decoder")


def _pick_idle_gpu() -> str:
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
    return rows[0][1] if rows else ""


def _maybe_set_gpu(device: str) -> None:
    if device != "auto" or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    selected = _pick_idle_gpu()
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[gpu] auto selected CUDA_VISIBLE_DEVICES={selected}")


def _init_dist() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size


def _load_decoder(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig]:
    state = torch.load(path, map_location="cpu")
    cfg_dict = state.get("decoder_config", {})
    if "route_points" in cfg_dict and "num_route_queries" not in cfg_dict:
        cfg_dict["num_route_queries"] = cfg_dict.pop("route_points")
    if "waypoint_points" in cfg_dict and "num_waypoint_queries" not in cfg_dict:
        cfg_dict["num_waypoint_queries"] = cfg_dict.pop("waypoint_points")
    config = LeadMoTPlanningDecoderConfig(**{k: v for k, v in cfg_dict.items() if k in LeadMoTPlanningDecoderConfig.__dataclass_fields__})
    decoder = LeadMoTPlanningDecoder(config).to(device=device, dtype=dtype)
    decoder.load_state_dict(state["decoder"], strict=True)
    decoder.eval()
    return decoder, config


def _checkpoint_root(save_root: str) -> Path:
    return Path(save_root) if save_root else DEFAULT_OUTPUT_ROOT


def _resolve_checkpoint(checkpoint: str, save_root: str) -> Path:
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
    # 训练中途可能还没 best/latest，再回退到最新 step / epoch 快照。
    for glob_pat in ("step-checkpoint-*.pt", "checkpoint-epoch*.pt"):
        pool = sorted(root.glob(glob_pat))
        if pool:
            return pool[-1]
    raise FileNotFoundError(f"no default LeadMoT checkpoint found under {root}")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = Path(args.save_root) if args.save_root else DEFAULT_OUTPUT_ROOT
    return root / "eval"


def _compute_metrics(outputs: dict[str, torch.Tensor], gt_route: torch.Tensor, gt_wp: torch.Tensor) -> dict[str, float]:
    pred_route = outputs["pred_route"].float()
    pred_wp = outputs["pred_future_waypoints"].float()
    route_err = torch.linalg.norm(pred_route - gt_route.float(), dim=-1)
    wp_err = torch.linalg.norm(pred_wp - gt_wp.float(), dim=-1)
    return {
        "route_ade": float(route_err.mean().item()),
        "route_fde": float(route_err[:, -1].mean().item()),
        "waypoint_ade": float(wp_err.mean().item()),
        "waypoint_fde": float(wp_err[:, -1].mean().item()),
    }


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
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
    decoder, decoder_config = _load_decoder(checkpoint_path, device, decoder_dtype)
    runtime = LeadMoTTrainRuntime(args, device)

    sums = {"loss": 0.0, "route_loss": 0.0, "waypoint_loss": 0.0, "route_ade": 0.0, "route_fde": 0.0, "waypoint_ade": 0.0, "waypoint_fde": 0.0}
    perline_path = output_dir / f"eval_v1_perline.rank{rank}.jsonl"
    count = 0
    with perline_path.open("w", encoding="utf-8", newline="\n") as f:
        with torch.no_grad():
            for idx, sample in enumerate(shard):
                try:
                    # Eval only all-reduces once after the loop. Bad samples can be
                    # skipped independently per rank and averaged by global valid count.
                    outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
                    gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
                    gt_route = gt_route.unsqueeze(0).to(device)
                    gt_wp = gt_wp.unsqueeze(0).to(device)
                    loss, route_loss, wp_loss = _planning_loss(outputs, gt_route, gt_wp, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type)
                    metrics = _compute_metrics(outputs, gt_route, gt_wp)
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
    summary.update({"count": total_count, "checkpoint": str(checkpoint_path), "jsonl": str(Path(args.jsonl)), "world_size": world_size})

    _barrier()
    if rank == 0:
        merged = output_dir / "eval_v1_perline.jsonl"
        with merged.open("w", encoding="utf-8", newline="\n") as out:
            for r in range(world_size):
                part = output_dir / f"eval_v1_perline.rank{r}.jsonl"
                if part.exists():
                    out.write(part.read_text(encoding="utf-8"))
        (output_dir / "eval_v1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        # eval_tb：每次 eval 一个独立 run_tag，多 ckpt 可在同一 TensorBoard 叠加对比（对齐 goalgen）。
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
