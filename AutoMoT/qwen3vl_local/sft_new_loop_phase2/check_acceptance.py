#!/usr/bin/env python3
"""按冻结阈值检查 unseen production metrics，生成一次性验收记录。"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict


def evaluate_acceptance(metrics: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """从 eval metrics 计算所有硬门槛。"""

    variant = (metrics.get("variant_reports") or {}).get("all_random_order") or {}
    slices = metrics.get("slice_reports") or {}
    questions = metrics.get("per_question") or {}
    values = {
        "overall_exact": float(metrics.get("exact_match_accuracy", 0.0)),
        "format_valid_rate": float(variant.get("format_valid_rate", 0.0)),
        "ue3_recall": float((questions.get("UE3") or {}).get("recall", 0.0)),
        "ue6_recall": float((questions.get("UE6") or {}).get("recall", 0.0)),
        "invalid_recall": float((questions.get("INVALID_EVENT_CONTEXT") or {}).get("recall", 0.0)),
        "applicable_regular_exact": float(
            (slices.get("applicable_regular") or {}).get("exact_match_accuracy", 0.0)
        ),
    }
    floors = {
        "overall_exact": float(args.min_overall_exact),
        "format_valid_rate": float(args.min_format_valid_rate),
        "ue3_recall": float(args.min_ue3_recall),
        "ue6_recall": float(args.min_ue6_recall),
        "invalid_recall": float(args.min_invalid_recall),
        "applicable_regular_exact": float(args.min_applicable_regular_exact),
    }
    passed = {key: values[key] >= floors[key] for key in values}
    return {
        "accepted": all(passed.values()),
        "contract": "frozen unseen thresholds; do not tune prompt on this result",
        "metrics_path": str(pathlib.Path(args.metrics).resolve()),
        "total_cases": int(metrics.get("total_cases", 0)),
        "prompt_name": metrics.get("prompt_name"),
        "production_prompt_sha256": metrics.get("production_prompt_sha256"),
        "values": values,
        "floors": floors,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-overall-exact", type=float, default=0.80)
    parser.add_argument("--min-format-valid-rate", type=float, default=1.0)
    parser.add_argument("--min-ue3-recall", type=float, default=0.80)
    parser.add_argument("--min-ue6-recall", type=float, default=0.80)
    parser.add_argument("--min-invalid-recall", type=float, default=0.80)
    parser.add_argument("--min-applicable-regular-exact", type=float, default=0.50)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    metrics = json.loads(pathlib.Path(args.metrics).read_text(encoding="utf-8"))
    result = evaluate_acceptance(metrics, args)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["accepted"] and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
