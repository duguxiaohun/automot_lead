#!/usr/bin/env python3
"""根据已逐帧审计的 UE3 决策表生成诊断性子集指标。

该工具不修改 dataset、checkpoint 选优或正式 validation 分数；它只用来区分
VISIBLE_ACTIVE 模型漏判与 PRE/POST/DOMAIN/2RGB/AMBIGUOUS 标签责任。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Tuple


Identity = Tuple[str, str, int, str]
VISUAL_CLASSES = {
    "VISIBLE_ACTIVE",
    "PRE_EVENT",
    "POST_EVENT",
    "DOMAIN_CONFLICT",
    "2RGB_UNOBSERVABLE",
    "AMBIGUOUS",
}


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """读取 JSONL object。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            yield payload


def _identity(row: Mapping[str, Any]) -> Identity:
    """返回与 RGB audit manifest 一致的稳定身份。"""

    return (
        str(row.get("scenario", "")),
        str(row.get("route_id", "")),
        int(row.get("frame_id", -1)),
        str(row.get("question_domain", "")),
    )


def build_report(manifest_path: pathlib.Path, decisions_path: pathlib.Path) -> Dict[str, Any]:
    """硬校验决策覆盖并计算诊断性指标。"""

    manifest = {_identity(row): row for row in _iter_jsonl(manifest_path)}
    decisions: Dict[Identity, Dict[str, Any]] = {}
    for row in _iter_jsonl(decisions_path):
        key = _identity(row)
        visual_class = str(row.get("visual_class", ""))
        if visual_class not in VISUAL_CLASSES:
            raise ValueError(f"invalid visual_class={visual_class!r}: {key}")
        if key in decisions:
            raise ValueError(f"duplicate RGB decision: {key}")
        decisions[key] = row
    missing = sorted(set(manifest) - set(decisions))
    extra = sorted(set(decisions) - set(manifest))
    if missing or extra:
        raise ValueError(f"RGB decision identity mismatch: missing={missing[:10]} extra={extra[:10]}")

    class_counts = Counter(str(row["visual_class"]) for row in decisions.values())
    seed_class_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    seed_route_visible: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for key, case in manifest.items():
        visual_class = str(decisions[key]["visual_class"])
        route = f"{key[0]}/{key[1]}"
        pattern = case.get("prediction_pattern") or {}
        for seed, result in pattern.items():
            seed = str(seed)
            seed_class_counts[seed][f"{visual_class}/total"] += 1
            seed_class_counts[seed][f"{visual_class}/tp"] += int(str(result) == "TP")
            if visual_class == "VISIBLE_ACTIVE":
                seed_route_visible[seed][route]["total"] += 1
                seed_route_visible[seed][route]["tp"] += int(str(result) == "TP")

    visible_report: Dict[str, Any] = {}
    for seed, counters in sorted(seed_class_counts.items()):
        total = int(counters["VISIBLE_ACTIVE/total"])
        tp = int(counters["VISIBLE_ACTIVE/tp"])
        routes = {
            route: {
                "cases": int(values["total"]),
                "true_positives": int(values["tp"]),
                "recall": float(values["tp"]) / max(1.0, float(values["total"])),
            }
            for route, values in sorted(seed_route_visible[seed].items())
        }
        visible_report[seed] = {
            "cases": total,
            "true_positives": tp,
            "frame_recall": float(tp) / max(1.0, float(total)),
            "unique_routes": len(routes),
            "route_macro_recall": (
                sum(float(item["recall"]) for item in routes.values()) / max(1, len(routes))
            ),
            "routes": routes,
        }

    class_seed_report = {
        visual_class: {
            seed: {
                "cases": int(counters[f"{visual_class}/total"]),
                "predicted_yes": int(counters[f"{visual_class}/tp"]),
                "predicted_yes_rate": float(counters[f"{visual_class}/tp"])
                / max(1.0, float(counters[f"{visual_class}/total"])),
            }
            for seed, counters in sorted(seed_class_counts.items())
        }
        for visual_class in sorted(VISUAL_CLASSES)
    }
    return {
        "format": "sft_new_loop_phase2_ue3_rgb_decision_rescore_v1",
        "official_metric": False,
        "contract": (
            "Diagnostic only. Decisions were made after viewing validation RGB and must not replace frozen "
            "checkpoint selection, train labels, or unseen acceptance."
        ),
        "manifest": str(manifest_path),
        "decisions": str(decisions_path),
        "cases": len(manifest),
        "visual_class_counts": dict(sorted(class_counts.items())),
        "visible_active": visible_report,
        "prediction_by_visual_class": class_seed_report,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    """生成简短的人类可读诊断摘要。"""

    lines = [
        "# UE3 RGB decision diagnostic rescore",
        "",
        "- official metric: `False`",
        f"- cases: `{report['cases']}`",
        f"- visual classes: `{report['visual_class_counts']}`",
        "",
        "| seed | visible cases | TP | frame recall | routes | route-macro recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for seed, item in sorted((report.get("visible_active") or {}).items()):
        lines.append(
            f"| {seed} | {item['cases']} | {item['true_positives']} | "
            f"{item['frame_recall']:.4f} | {item['unique_routes']} | {item['route_macro_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "该表只用于定位模型责任，不能替代 frozen validation 或触发 unseen。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", default="checkpoints/ue3_route_diverse_full_rgb_audit")
    parser.add_argument(
        "--decisions",
        default=str(pathlib.Path(__file__).with_name("ue3_route_diverse_rgb_decisions_v1.jsonl")),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    audit_root = pathlib.Path(args.audit_root)
    output_json = pathlib.Path(args.output_json) if args.output_json else audit_root / "decision_rescore.json"
    output_md = pathlib.Path(args.output_md) if args.output_md else audit_root / "decision_rescore.md"
    report = build_report(audit_root / "manifest.jsonl", pathlib.Path(args.decisions))
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
