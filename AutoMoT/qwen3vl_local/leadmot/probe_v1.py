#!/usr/bin/env python3
"""Dump LeadMoT v1 case-level prediction probes."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

AUTOMOT_ROOT = Path(__file__).resolve().parents[2]
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from qwen3vl_local.leadmot import LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig
from qwen3vl_local.leadmot.train_v1 import (
    LEAD_BEV_CKPT_PATH,
    LeadMoTTrainRuntime,
    _dump_invocation,
    _dtype,
    _extract_targets,
    _planning_loss,
    _read_jsonl,
    _setup_offline_env,
)
from qwen3vl_local.leadmot.eval_v1 import _compute_metrics


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
    return root / "eval_cases"


def _case_name(sample: dict[str, Any], idx: int) -> str:
    scenario = str(sample.get("scenario", "scenario")).replace("/", "_").replace("\\", "_")
    route_id = str(sample.get("route_id", "route")).replace("/", "_").replace("\\", "_")
    anchor = int(sample.get("anchor", 0))
    return f"{idx:05d}__{scenario}__{route_id}__anchor{anchor:04d}"


def _draw_plot(path: Path, pred_route, gt_route, pred_wp, gt_wp) -> None:
    width, height = 720, 540
    margin = 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    all_pts = torch.cat([pred_route, gt_route, pred_wp, gt_wp], dim=0).cpu().float()
    max_x = max(float(all_pts[:, 0].max().item()), 10.0)
    min_x = min(float(all_pts[:, 0].min().item()), 0.0)
    max_abs_y = max(float(all_pts[:, 1].abs().max().item()), 5.0)

    def project(pt):
        x = float(pt[0])
        y = float(pt[1])
        px = margin + (x - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * margin)
        py = height / 2 - y / max(max_abs_y, 1e-6) * (height / 2 - margin)
        return (px, py)

    draw.line([(margin, height / 2), (width - margin, height / 2)], fill=(220, 220, 220), width=1)
    draw.line([project((0.0, -max_abs_y)), project((0.0, max_abs_y))], fill=(220, 220, 220), width=1)

    def poly(points, color, label):
        pts = [project(p) for p in points]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)
        for p in pts:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)
        draw.text((margin, 12 + label[0] * 18), label[1], fill=color)

    poly(gt_route, (40, 120, 220), (0, "gt route"))
    poly(pred_route, (20, 70, 150), (1, "pred route"))
    poly(gt_wp, (30, 160, 70), (2, "gt waypoint"))
    poly(pred_wp, (210, 90, 40), (3, "pred waypoint"))
    img.save(path)


def _select_samples(rows: list[dict[str, Any]], num_per_scenario: int, max_cases: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("scenario", "scenario"))].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        vals = list(buckets[key])
        rng.shuffle(vals)
        selected.extend(vals[:num_per_scenario])
    rng.shuffle(selected)
    return selected[:max_cases] if max_cases > 0 else selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="checkpoints/leadmot_v1_data/val.jsonl")
    parser.add_argument("--checkpoint", default="", help="Default: <save-root>/best.pt -> latest.pt -> newest step/epoch checkpoint.")
    parser.add_argument("--save-root", default="", help="GoalGen-style root; case dumps go to <save-root>/eval_cases when --output-dir is omitted.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--lead-bev-ckpt", default=str(LEAD_BEV_CKPT_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-per-scenario", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=24)
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
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    rows = _read_jsonl(Path(args.jsonl))
    samples = _select_samples(rows, args.num_per_scenario, args.max_cases, args.seed)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_root = Path(args.save_root) if args.save_root else output_dir.parent
    _dump_invocation(invocation_root, rank=0)

    decoder_dtype = _dtype(args.decoder_dtype)
    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.save_root)
    decoder, decoder_config = _load_decoder(checkpoint_path, device, decoder_dtype)
    runtime = LeadMoTTrainRuntime(args, device)

    index_rows = []
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            case_dir = output_dir / _case_name(sample, idx)
            case_dir.mkdir(parents=True, exist_ok=True)
            outputs = runtime.forward_sample(sample, decoder, decoder_config, decoder_dtype)
            gt_route, gt_wp = _extract_targets(sample, args.route_points, args.waypoint_points, args.smooth_route)
            gt_route_b = gt_route.unsqueeze(0).to(device)
            gt_wp_b = gt_wp.unsqueeze(0).to(device)
            loss, route_loss, wp_loss = _planning_loss(outputs, gt_route_b, gt_wp_b, args.route_loss_weight, args.waypoint_loss_weight, args.loss_type)
            metrics = _compute_metrics(outputs, gt_route_b, gt_wp_b)

            pred_route = outputs["pred_route"][0].detach().cpu().float()
            pred_wp = outputs["pred_future_waypoints"][0].detach().cpu().float()
            gt_route_cpu = gt_route.cpu().float()
            gt_wp_cpu = gt_wp.cpu().float()
            _draw_plot(case_dir / "planning_overlay.png", pred_route, gt_route_cpu, pred_wp, gt_wp_cpu)

            pred = {
                "pred_route": pred_route.tolist(),
                "gt_route": gt_route_cpu.tolist(),
                "pred_future_waypoints": pred_wp.tolist(),
                "gt_future_waypoints": gt_wp_cpu.tolist(),
            }
            (case_dir / "predictions.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
            metric_row = {"loss": float(loss.item()), "route_loss": float(route_loss.item()), "waypoint_loss": float(wp_loss.item()), **metrics}
            (case_dir / "metrics.json").write_text(json.dumps(metric_row, ensure_ascii=False, indent=2), encoding="utf-8")
            (case_dir / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
            (case_dir / "overview.md").write_text(
                f"# LeadMoT Probe\n\n![planning](planning_overlay.png)\n\n```json\n{json.dumps(metric_row, ensure_ascii=False, indent=2)}\n```\n",
                encoding="utf-8",
            )
            index_rows.append({"case_dir": str(case_dir), **metric_row, "route_dir": sample.get("route_dir"), "anchor": sample.get("anchor")})
            print(f"wrote {case_dir}")

    (output_dir / "probe_index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
