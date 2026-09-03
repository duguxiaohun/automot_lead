#!/usr/bin/env python3
"""比较旧/新 Phase2 index，守住 frozen val/test 并审计 train route 集中度。"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Tuple


Identity = Tuple[str, str, int, str, str, str]
CLASSES = ("UE1", "UE3", "UE5", "UE6", "RE", "INVALID")


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """读取 JSONL object。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            yield row


def _class(row: Mapping[str, Any]) -> str:
    """返回六桶类别。"""

    return "INVALID" if bool(row.get("invalid_event_context")) else str(row.get("target_event_class"))


def _identity(row: Mapping[str, Any]) -> Identity:
    """冻结比较所需的完整 case 身份。"""

    return (
        str(row.get("scenario", "")),
        str(row.get("route_id", "")),
        int(row.get("frame_id", -1)),
        str(row.get("question_domain", "")),
        _class(row),
        str(row.get("invalid_source", "")),
    )


def _route_report(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """按 split/class 汇总 route 集中度。"""

    groups: Dict[str, Dict[str, Counter[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in rows:
        split = str(row.get("split", "UNKNOWN"))
        target_class = _class(row)
        groups[split][target_class][
            (str(row.get("scenario", "")), str(row.get("route_id", "")))
        ] += 1
    return {
        split: {
            target_class: {
                "cases": sum(counts.values()),
                "unique_routes": len(counts),
                "max_cases_per_route": max(counts.values(), default=0),
                "top_routes": {
                    f"{scenario}/{route_id}": count
                    for (scenario, route_id), count in sorted(
                        counts.items(), key=lambda pair: (-pair[1], pair[0])
                    )[:10]
                },
            }
            for target_class, counts in sorted(classes.items())
        }
        for split, classes in sorted(groups.items())
    }


def compare_indices(old_path: pathlib.Path, new_path: pathlib.Path) -> Dict[str, Any]:
    """比较 frozen 身份和 train route 分布。"""

    old_rows = list(_iter_jsonl(old_path))
    new_rows = list(_iter_jsonl(new_path))
    old_report = _route_report(old_rows)
    new_report = _route_report(new_rows)
    frozen: Dict[str, Any] = {}
    for split in ("val", "test"):
        old_ids = Counter(_identity(row) for row in old_rows if str(row.get("split")) == split)
        new_ids = Counter(_identity(row) for row in new_rows if str(row.get("split")) == split)
        removed = list((old_ids - new_ids).elements())
        added = list((new_ids - old_ids).elements())
        frozen[split] = {
            "old_cases": sum(old_ids.values()),
            "new_cases": sum(new_ids.values()),
            "identity_multiset_equal": old_ids == new_ids,
            "removed_examples": removed[:10],
            "added_examples": added[:10],
        }

    train_classes: Dict[str, Any] = {}
    for target_class in CLASSES:
        old = (old_report.get("train") or {}).get(target_class) or {
            "cases": 0,
            "unique_routes": 0,
            "max_cases_per_route": 0,
        }
        new = (new_report.get("train") or {}).get(target_class) or {
            "cases": 0,
            "unique_routes": 0,
            "max_cases_per_route": 0,
        }
        route_guard = True
        if target_class != "INVALID":
            route_guard = (
                int(new["unique_routes"]) >= int(old["unique_routes"])
                and int(new["max_cases_per_route"]) <= int(old["max_cases_per_route"])
            )
        train_classes[target_class] = {
            "old": old,
            "new": new,
            "case_count_equal": int(old["cases"]) == int(new["cases"]),
            "route_diversity_improved_or_equal": route_guard,
        }

    guards = {
        "val_identity_multiset_equal": bool(frozen["val"]["identity_multiset_equal"]),
        "test_identity_multiset_equal": bool(frozen["test"]["identity_multiset_equal"]),
        "train_class_counts_equal": all(item["case_count_equal"] for item in train_classes.values()),
        "train_noninvalid_route_diversity_improved_or_equal": all(
            item["route_diversity_improved_or_equal"]
            for key, item in train_classes.items()
            if key != "INVALID"
        ),
    }
    return {
        "format": "sft_new_loop_phase2_dataset_route_diversity_comparison_v1",
        "old_index": str(old_path),
        "new_index": str(new_path),
        "frozen_splits": frozen,
        "train_classes": train_classes,
        "guards": guards,
        "passed": all(guards.values()),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    """渲染简短比较表。"""

    lines = [
        "# Phase2 route-diverse dataset comparison",
        "",
        f"- passed: `{report['passed']}`",
        f"- guards: `{report['guards']}`",
        "",
        "| class | old routes | new routes | old max/route | new max/route | count equal | route guard |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for target_class, item in report["train_classes"].items():
        old = item["old"]
        new = item["new"]
        lines.append(
            f"| {target_class} | {old['unique_routes']} | {new['unique_routes']} | "
            f"{old['max_cases_per_route']} | {new['max_cases_per_route']} | "
            f"{item['case_count_equal']} | {item['route_diversity_improved_or_equal']} |"
        )
    lines.extend(["", "## Frozen splits", ""])
    for split, item in report["frozen_splits"].items():
        lines.append(
            f"- {split}: old `{item['old_cases']}`, new `{item['new_cases']}`, "
            f"identity multiset equal `{item['identity_multiset_equal']}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-index", required=True)
    parser.add_argument("--new-index", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = compare_indices(pathlib.Path(args.old_index), pathlib.Path(args.new_index))
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("route-diverse dataset comparison failed; do not train")


if __name__ == "__main__":
    main()
