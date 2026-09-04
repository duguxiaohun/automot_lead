#!/usr/bin/env python3
"""校验并统计逐帧 RGB 确认的高速 UE3 子型。

高速 cut-in 仍使用正式目标 ``UE3``；``HIGHWAY_CUTIN`` 只用于数据与评估审计。
本工具只接受显式 RGB 决策清单，不会从 scenario 名、R3 或源 EVENT 名自动造正例。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Tuple

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream


HIGHWAY_UE3_SUBTYPE = "HIGHWAY_CUTIN"
OTHER_UE3_SUBTYPE = "OTHER_UE3"
DecisionKey = Tuple[str, str, int]
REQUIRED_UE3_SCENARIOS = {
    "DynamicObjectCrossing",
    "ParkingCutIn",
    "StaticCutIn",
    "HighwayCutIn",
}
SOURCE_UE3_AUDIT_SNAPSHOT = {
    "audit_date": "2026-09-04",
    "total_explicit_ue3_frames": 1351,
    "non_road_overlay_frames": 33,
    "frames_by_scenario": {
        "DynamicObjectCrossing": 251,
        "ParkingCutIn": 330,
        "StaticCutIn": 770,
    },
    "routes_by_scenario": {
        "DynamicObjectCrossing": 84,
        "ParkingCutIn": 80,
        "StaticCutIn": 54,
    },
    "frames_by_scenario_primary_rs": {
        "DynamicObjectCrossing/R1": 251,
        "ParkingCutIn/R1": 326,
        "ParkingCutIn/R4": 4,
        "StaticCutIn/R1": 741,
        "StaticCutIn/R4": 29,
    },
}


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSON object。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            row["_line_no"] = line_no
            yield row


def load_highway_ue3_decisions(
    path: pathlib.Path,
) -> Tuple[Dict[DecisionKey, Dict[str, Any]], Dict[str, Any]]:
    """读取 span 决策并展开到逐帧；只返回 YES 覆盖与完整审计摘要。"""

    positive: Dict[DecisionKey, Dict[str, Any]] = {}
    all_labels: Dict[DecisionKey, str] = {}
    span_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    route_sets: Dict[str, set[Tuple[str, str]]] = {"YES": set(), "NO": set()}
    split_counts: Counter[str] = Counter()
    rows = list(_iter_jsonl(path))
    if not rows:
        raise ValueError(f"empty highway UE3 RGB decision file: {path}")
    for row in rows:
        scenario = str(row.get("scenario") or "")
        route_id = str(row.get("route_id") or "")
        answer = str(row.get("ue3_answer") or "").upper()
        expected_split = str(row.get("expected_split") or "")
        try:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frame span at {path}:{row['_line_no']}") from exc
        if not scenario or not route_id or answer not in {"YES", "NO"}:
            raise ValueError(f"invalid decision identity/answer at {path}:{row['_line_no']}")
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"invalid frame bounds at {path}:{row['_line_no']}")
        if expected_split not in {"train", "val", "test"}:
            raise ValueError(f"invalid expected_split at {path}:{row['_line_no']}")
        if not str(row.get("rgb_review") or "") or not str(row.get("visible_cue") or ""):
            raise ValueError(f"RGB provenance and visible cue are required at {path}:{row['_line_no']}")
        span_counts[answer] += 1
        route_sets[answer].add((scenario, route_id))
        for frame_id in range(start_frame, end_frame + 1):
            key = (scenario, route_id, frame_id)
            previous = all_labels.get(key)
            if previous is not None:
                raise ValueError(
                    f"overlapping highway UE3 RGB decisions for {key}: {previous} vs {answer}"
                )
            all_labels[key] = answer
            frame_counts[answer] += 1
            split_counts[f"{expected_split}/{answer}"] += 1
            if answer == "YES":
                positive[key] = {
                    key: value for key, value in row.items() if key != "_line_no"
                }
    report = {
        "format": "sft_new_loop_phase2_highway_ue3_rgb_decisions_v1",
        "path": str(path),
        "contract": (
            "Only explicit RGB-reviewed YES frames override the source fold to UE3. "
            "NO rows are boundary/hard-negative audit controls; scenario and R3 never imply UE3."
        ),
        "span_counts": dict(sorted(span_counts.items())),
        "frame_counts": dict(sorted(frame_counts.items())),
        "route_counts": {
            answer: len(routes) for answer, routes in sorted(route_sets.items())
        },
        "split_frame_counts": dict(sorted(split_counts.items())),
        "positive_override_frames": len(positive),
    }
    return positive, report


def ue3_subtype(*, target_class: str, explicit_subtype: str = "") -> str:
    """保持 UE3 大类不变，只返回审计子型。"""

    if str(target_class) != "UE3":
        return ""
    return HIGHWAY_UE3_SUBTYPE if explicit_subtype == HIGHWAY_UE3_SUBTYPE else OTHER_UE3_SUBTYPE


def _file_contains_ue3(path: pathlib.Path) -> bool:
    """分块预检大 JSON，避免解析与 UE3 无关的全部场景。"""

    needle = b'U-E3'
    tail = b''
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-(len(needle) - 1) :]
    return False


def source_ue3_taxonomy_report(collection_dir: pathlib.Path) -> Dict[str, Any]:
    """统计所有显式源 U-E3 场景，并确认 RS overlay 不会再被 Phase2 丢弃。"""

    if not collection_dir.is_dir():
        return {"available": False, "path": str(collection_dir)}
    frames_by_scenario: Counter[str] = Counter()
    frames_by_scenario_rs: Counter[str] = Counter()
    routes_by_scenario: Dict[str, set[str]] = {}
    overlay_frames = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        if not _file_contains_ue3(result_path):
            continue
        scenario = result_path.stem.removesuffix("_result")
        scenario_routes = routes_by_scenario.setdefault(scenario, set())
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            route_has_ue3 = False
            for annotation in route.get("annotations", []) or []:
                events = {str(item) for item in annotation.get("events", []) or []}
                if "U-E3" not in events:
                    continue
                rs = str(
                    annotation.get("primary_road_structure")
                    or (annotation.get("frame_rs_annotation") or {}).get("label")
                    or "UNKNOWN"
                )
                frames_by_scenario[scenario] += 1
                frames_by_scenario_rs[f"{scenario}/{rs}"] += 1
                overlay_frames += int(rs not in {"R1", "R2"})
                route_has_ue3 = True
            if route_has_ue3:
                scenario_routes.add(route_id)
    return {
        "available": True,
        "path": str(collection_dir),
        "contract": (
            "Every explicit source U-E3 is retained as OTHER_UE3 and asked through ROAD_CORRIDOR; "
            "non-road primary RS is an interrupted overlay, not a reason to drop the event."
        ),
        "total_explicit_ue3_frames": sum(frames_by_scenario.values()),
        "non_road_overlay_frames": overlay_frames,
        "frames_by_scenario": dict(sorted(frames_by_scenario.items())),
        "routes_by_scenario": {
            scenario: len(routes) for scenario, routes in sorted(routes_by_scenario.items())
        },
        "frames_by_scenario_primary_rs": dict(sorted(frames_by_scenario_rs.items())),
        "guard_passed": bool(frames_by_scenario) and sum(frames_by_scenario.values()) > overlay_frames,
    }


def _index_report(index_path: pathlib.Path, positive: Mapping[DecisionKey, Mapping[str, Any]]) -> Dict[str, Any]:
    """检查 index 是否兼容全部 UE3 场景及各 split 的高速子型。"""

    if not index_path.is_file():
        return {"available": False, "path": str(index_path)}
    indexed: Dict[DecisionKey, Dict[str, Any]] = {}
    ue3_scenario_counts: Counter[str] = Counter()
    ue3_subtype_counts: Counter[str] = Counter()
    for row in _iter_jsonl(index_path):
        if str(row.get("target_event_class") or "") == "UE3":
            ue3_scenario_counts[str(row.get("scenario") or "UNKNOWN")] += 1
            ue3_subtype_counts[
                str(row.get("ue3_subtype") or OTHER_UE3_SUBTYPE)
            ] += 1
        key = (str(row.get("scenario") or ""), str(row.get("route_id") or ""), int(row.get("frame_id", -1)))
        if key in positive:
            indexed[key] = row
    missing = sorted(set(positive) - set(indexed))
    wrong = sorted(
        key
        for key, row in indexed.items()
        if str(row.get("target_event_class")) != "UE3"
        or str(row.get("ue3_subtype")) != HIGHWAY_UE3_SUBTYPE
        or str(row.get("question_domain")) != "ROAD_CORRIDOR"
        or not bool((row.get("answers") or {}).get("UE3"))
    )
    expected_splits = {str(row.get("expected_split") or "") for row in positive.values()}
    indexed_split_counts = Counter(str(row.get("split") or "") for row in indexed.values())
    missing_splits = sorted(split for split in expected_splits if indexed_split_counts[split] <= 0)
    missing_scenarios = sorted(REQUIRED_UE3_SCENARIOS - set(ue3_scenario_counts))
    return {
        "available": True,
        "path": str(index_path),
        "expected_positive_frames": len(positive),
        "indexed_positive_frames": len(indexed),
        "missing_positive_frames": len(missing),
        "wrong_contract_frames": len(wrong),
        "indexed_split_counts": dict(sorted(indexed_split_counts.items())),
        "ue3_scenario_counts": dict(sorted(ue3_scenario_counts.items())),
        "ue3_subtype_counts": dict(sorted(ue3_subtype_counts.items())),
        "required_ue3_scenarios": sorted(REQUIRED_UE3_SCENARIOS),
        "missing_ue3_scenarios": missing_scenarios,
        "missing_positive_splits": missing_splits,
        "missing_examples": [list(key) for key in missing[:20]],
        "wrong_examples": [list(key) for key in wrong[:20]],
        "guard_passed": not missing_splits and not missing_scenarios and not wrong,
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        default=str(pathlib.Path(__file__).with_name("highway_ue3_rgb_decisions_v1.jsonl")),
    )
    parser.add_argument("--index", default="checkpoints/sft_new_loop_phase2_data/frame_index.jsonl")
    parser.add_argument(
        "--collection-dir",
        default="keyframe_filter/collection_output",
        help="source annotation directory used to audit every explicit U-E3 scenario",
    )
    parser.add_argument(
        "--scan-source-taxonomy",
        action="store_true",
        help="rescan the multi-GB source result files instead of using the audited 2026-09-04 snapshot",
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--require-index-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when the index exists, fail unless train/val/test contain valid HIGHWAY_CUTIN UE3 rows",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    positive, report = load_highway_ue3_decisions(pathlib.Path(args.decisions))
    if args.scan_source_taxonomy:
        report["source_ue3_taxonomy"] = source_ue3_taxonomy_report(pathlib.Path(args.collection_dir))
    else:
        report["source_ue3_taxonomy"] = {
            "available": True,
            "source": "audited_snapshot",
            "path": str(args.collection_dir),
            **SOURCE_UE3_AUDIT_SNAPSHOT,
            "guard_passed": True,
        }
    report["index"] = _index_report(pathlib.Path(args.index), positive)
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        bool(args.require_index_guard)
        and bool(report["index"].get("available"))
        and not bool(report["index"].get("guard_passed"))
    ):
        raise SystemExit("highway UE3 index guard failed")


if __name__ == "__main__":
    main()
