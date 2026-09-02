#!/usr/bin/env python3
"""比较 base-Qwen 与 Phase2-LoRA-Qwen 两组 LeadMoT 离线结果。

两组必须来自同一份 validation JSONL。脚本逐 case 对齐后按 route 聚合，再做
route-cluster bootstrap；连续帧不会被当成相互独立证据。所有距离指标均为 B-A，
负值表示 LoRA 组更好。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "loss",
    "route_loss",
    "waypoint_loss",
    "route_ade_m",
    "route_fde_m",
    "waypoint_ade_m",
    "waypoint_fde_m",
)
GATE_METRICS = ("route_ade_m", "route_fde_m", "waypoint_ade_m", "waypoint_fde_m")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 eval per-line JSONL。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            rows.append(row)
    return rows


def _identity(row: dict[str, Any]) -> tuple[int, str, int]:
    """返回 eval.py 保存的稳定 case 身份。"""

    return int(row["index"]), str(row.get("route_dir", "")), int(row.get("anchor", -1))


def _quantile(values: list[float], probability: float) -> float:
    """在线性插值下返回经验分位数。"""

    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * probability
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mean(values: Iterable[float]) -> float:
    """计算非空浮点序列平均值。"""

    vals = list(values)
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return sum(vals) / len(vals)


def compare_rows(
    base_rows: list[dict[str, Any]],
    lora_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 20260902,
    min_routes: int = 10,
) -> dict[str, Any]:
    """严格对齐两组 case 并生成 route-cluster bootstrap 报告。"""

    base = {_identity(row): row for row in base_rows}
    lora = {_identity(row): row for row in lora_rows}
    if len(base) != len(base_rows) or len(lora) != len(lora_rows):
        raise ValueError("duplicate case identity in base or LoRA eval rows")
    missing_lora = sorted(set(base) - set(lora))
    missing_base = sorted(set(lora) - set(base))
    if missing_lora or missing_base:
        raise ValueError(
            f"A/B case identity mismatch: missing_lora={missing_lora[:5]} "
            f"missing_base={missing_base[:5]}"
        )
    if not base:
        raise ValueError("A/B eval rows are empty")

    route_metric_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    frame_deltas: dict[str, list[float]] = defaultdict(list)
    for key in sorted(base):
        base_row = base[key]
        lora_row = lora[key]
        route = key[1]
        for metric in METRICS:
            if metric not in base_row or metric not in lora_row:
                raise ValueError(f"missing metric {metric!r} for case {key}")
            delta = float(lora_row[metric]) - float(base_row[metric])
            frame_deltas[metric].append(delta)
            route_metric_deltas[route][metric].append(delta)

    routes = sorted(route_metric_deltas)
    per_route = {
        route: {
            metric: _mean(route_metric_deltas[route][metric])
            for metric in METRICS
        }
        for route in routes
    }
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = {metric: [] for metric in METRICS}
    if len(routes) >= 2 and bootstrap_samples > 0:
        for _ in range(bootstrap_samples):
            sampled = [routes[rng.randrange(len(routes))] for _ in routes]
            for metric in METRICS:
                bootstrap[metric].append(_mean(per_route[route][metric] for route in sampled))

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        route_macro_delta = _mean(per_route[route][metric] for route in routes)
        samples = bootstrap[metric]
        metrics[metric] = {
            "frame_mean_delta_lora_minus_base": _mean(frame_deltas[metric]),
            "route_macro_delta_lora_minus_base": route_macro_delta,
            "route_bootstrap_ci95": (
                [_quantile(samples, 0.025), _quantile(samples, 0.975)] if samples else None
            ),
        }

    reasons: list[str] = []
    if len(routes) < min_routes:
        reasons.append(f"route_count={len(routes)} < required {min_routes}")
    for metric in GATE_METRICS:
        ci = metrics[metric]["route_bootstrap_ci95"]
        if ci is None or float(ci[1]) >= 0.0:
            reasons.append(f"{metric} route-bootstrap upper CI is not below zero")
    decision = "LORA_OFFLINE_WIN" if not reasons else "INSUFFICIENT_OR_NO_IMPROVEMENT"
    return {
        "format": "leadmot_qwen_adapter_ab_v1",
        "comparison": "LoRA minus base; negative distance delta is better",
        "paired_cases": len(base),
        "unique_routes": len(routes),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "minimum_routes": int(min_routes),
        "metrics": metrics,
        "decision": decision,
        "gate_reasons": reasons,
        "carla_allowed": decision == "LORA_OFFLINE_WIN",
    }


def _markdown(report: dict[str, Any]) -> str:
    """渲染简洁 Markdown 报告。"""

    lines = [
        "# LeadMoT Qwen base / Phase2 LoRA A/B",
        "",
        f"- paired cases: `{report['paired_cases']}`",
        f"- unique routes: `{report['unique_routes']}`",
        f"- decision: `{report['decision']}`",
        f"- CARLA allowed: `{report['carla_allowed']}`",
        "",
        "所有 delta 均为 `LoRA - base`，距离指标为负才表示 LoRA 更好。",
        "",
        "| metric | frame mean delta | route-macro delta | route bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = report["metrics"][metric]
        ci = item["route_bootstrap_ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"
        lines.append(
            f"| {metric} | {item['frame_mean_delta_lora_minus_base']:.6f} | "
            f"{item['route_macro_delta_lora_minus_base']:.6f} | {ci_text} |"
        )
    if report["gate_reasons"]:
        lines.extend(["", "## Gate reasons", ""])
        lines.extend(f"- {reason}" for reason in report["gate_reasons"])
    lines.extend(
        [
            "",
            "只有四个 ADE/FDE 指标的 route-cluster bootstrap 上界都小于 0，才进入 CARLA。",
            "该自动门槛不修改 Phase2 prompt，也不打开 unseen-456。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base arm eval_v1_perline.jsonl")
    parser.add_argument("--lora", required=True, help="LoRA arm eval_v1_perline.jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--min-routes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """运行比较并写 JSON/Markdown。"""

    args = parse_args()
    report = compare_rows(
        _read_jsonl(Path(args.base)),
        _read_jsonl(Path(args.lora)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_routes=args.min_routes,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
