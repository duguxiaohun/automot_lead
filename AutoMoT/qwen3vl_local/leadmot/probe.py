#!/usr/bin/env python3
"""导出 LeadMoT v1 case-level 预测 probe。"""

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
from qwen3vl_local.leadmot.train import (
    LEAD_BEV_CKPT_PATH,
    LeadMoTTrainRuntime,
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


def _maybe_set_gpu(device: str) -> None:
    """自动选择空闲 GPU，并覆盖外层残留的 CUDA_VISIBLE_DEVICES。

    --device 显式传 cpu / cuda[:N] 时尊重用户锁卡，不覆盖 CVD；仅 auto / 空值走自动覆盖。
    """
    if device and device.strip().lower() not in ("", "auto"):
        return
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected CUDA_VISIBLE_DEVICES={selected}; "
            f"process uses cuda:0/auto; previous={previous or '<unset>'}"
        )


def _load_decoder(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    use_ema: bool = True,
) -> tuple[LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig]:
    """加载 decoder 权重；可用时优先使用 EMA 权重。"""
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
    if "use_final_goal" not in cfg_dict:
        raise ValueError(f"{path} 缺少 decoder_config.use_final_goal，拒绝加载旧 LeadMoT ckpt")
    if not bool(cfg_dict["use_final_goal"]):
        raise ValueError(f"{path} 的 decoder_config.use_final_goal=False；当前路线要求 final_goal=True")

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
    """对齐 GoalGen 布局：未显式覆盖时写到 <save-root>/eval_cases。"""
    if args.output_dir:
        return Path(args.output_dir)
    root = Path(args.save_root) if args.save_root else DEFAULT_OUTPUT_ROOT
    return root / "eval_cases"


def _case_name(sample: dict[str, Any], idx: int) -> str:
    """构造稳定且文件系统安全的 case 目录名。"""
    scenario = str(sample.get("scenario", "scenario")).replace("/", "_").replace("\\", "_")
    route_id = str(sample.get("route_id", "route")).replace("/", "_").replace("\\", "_")
    anchor = int(sample.get("anchor", 0))
    return f"{idx:05d}__{scenario}__{route_id}__anchor{anchor:04d}"


def _draw_plot(path: Path, pred_route, gt_route, pred_wp, gt_wp) -> None:
    """渲染简单俯视角 prediction-vs-ground-truth planning 图。"""
    width, height = 720, 540
    margin = 40
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

    draw.line([(margin, height / 2), (width - margin, height / 2)], fill=(220, 220, 220), width=1)
    draw.line([project((0.0, -max_abs_y)), project((0.0, max_abs_y))], fill=(220, 220, 220), width=1)

    def poly(points, color, label):
        """画一条轨迹及其图例标签。"""
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
    """按 scenario 抽样，让 probe 覆盖多个 scenario bucket。"""
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
    """解析 case-probe CLI 参数。"""
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
    parser.add_argument("--target-point-lookahead-s", type=float, default=1.0)
    parser.add_argument("--next-target-point-lookahead-s", type=float, default=2.0)
    parser.add_argument("--tp-mode", type=str, default="route_lookahead", choices=["route_lookahead", "future_truth"])
    parser.add_argument("--tp-min-lookahead-m", type=float, default=5.0)
    parser.add_argument("--use-final-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-samples", action="store_true")
    # 默认用 EMA 权重做 probe；--no-use-ema 强制 raw decoder。
    # 旧 ckpt 缺 ema_state_dict 时自动回落 raw 并 print 提示。
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """导出预测图和 JSON 指标，方便人工检查。"""
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
    decoder, decoder_config = _load_decoder(checkpoint_path, device, decoder_dtype, use_ema=args.use_ema)
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
            metrics = _compute_planning_metrics(outputs, gt_route_b, gt_wp_b)

            pred_route = outputs["pred_route"][0].detach().cpu().float()
            pred_wp = outputs["pred_future_waypoints"][0].detach().cpu().float()
            input_status = outputs.get("input_status")
            navigation_input = None
            if input_status is not None:
                status = input_status[0].detach().cpu().float().view(-1)
                navigation_input = {
                    "speed_mps": float(status[0].item()),
                    "target_point": status[1:3].tolist(),
                    "target_point_next": status[3:5].tolist(),
                    "final_goal": status[5:7].tolist() if status.numel() >= 7 else None,
                }
            gt_route_cpu = gt_route.cpu().float()
            gt_wp_cpu = gt_wp.cpu().float()
            _draw_plot(case_dir / "planning_overlay.png", pred_route, gt_route_cpu, pred_wp, gt_wp_cpu)

            pred = {
                "navigation_input": navigation_input,
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
    # 顶层 meta：记录 ckpt、jsonl 与 use_ema，便于后续对比 raw vs ema 两次 probe 的结果。
    (output_dir / "probe_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "jsonl": str(Path(args.jsonl)),
                "use_ema": bool(args.use_ema),
                "num_cases": len(index_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
