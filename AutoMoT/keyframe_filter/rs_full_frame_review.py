#!/usr/bin/env python3
"""Full-frame ROAD_STRUCTURE review runner.

For each scenario and each town, sample up to N readable LEAD runs, annotate
all frames with the runtime ROAD_STRUCTURE rules, and generate RGB contact
sheets with the predicted label/confidence overlaid on every frame.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

KEYFRAME_DIR = Path(__file__).resolve().parent
AUTOMOT_ROOT = KEYFRAME_DIR.parent
if str(KEYFRAME_DIR) not in sys.path:
    sys.path.insert(0, str(KEYFRAME_DIR))
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from collector import (  # noqa: E402
    SCENARIO_TO_ROAD_STRUCTURE,
    ScenarioCollector,
    _DEFAULT_CARLA_ROOT,
    _DEFAULT_LEAD_DATA_ROOT,
    _DEFAULT_XML_ROOT,
)
from rs_research import (  # noqa: E402
    _scenario_runs_by_town,
    _select_research_runs,
)


RS_COLORS = {
    "R1": (95, 116, 142),
    "R2": (214, 89, 76),
    "R3": (85, 139, 214),
    "R4": (50, 151, 91),
    "R5": (156, 105, 205),
    "R6": (211, 142, 62),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _rgb_path_for_frame(run_dir: Path, frame_id: int) -> Optional[Path]:
    rgb_dir = run_dir / "rgb"
    for name in (f"{frame_id:04d}.jpg", f"{frame_id:05d}.jpg", f"{frame_id}.jpg"):
        path = rgb_dir / name
        if path.exists():
            return path
    matches = sorted(rgb_dir.glob(f"{frame_id:04d}.*"))
    return matches[0] if matches else None


def _annotation_review_reasons(ann: Dict[str, Any]) -> List[str]:
    frame_rs = ann.get("frame_rs_annotation", {}) or {}
    evidence = ann.get("evidence", {}) or {}
    reasons = list(frame_rs.get("review_reasons") or evidence.get("review_reasons") or [])
    diag = evidence.get("diagnostic_attribution", {}) or {}
    reasons.extend(diag.get("weak_or_missing_inputs") or [])
    return sorted({str(reason) for reason in reasons if reason})


def _hard_review_reasons(ann: Dict[str, Any]) -> List[str]:
    frame_rs = ann.get("frame_rs_annotation", {}) or {}
    evidence = ann.get("evidence", {}) or {}
    reasons = list(frame_rs.get("review_reasons") or evidence.get("review_reasons") or [])
    return sorted({str(reason) for reason in reasons if reason})


def _issue_bucket(reasons: List[str], label: str) -> str:
    reason_text = " ".join(reasons)
    if "route_projection_error" in reason_text or "xml_route_projection_error" in reason_text:
        return "xml_projection_or_boundary_parameter"
    if "candidate_score_gap" in reason_text:
        return "arbitration_or_threshold_margin"
    if "temporal_smoothing" in reason_text:
        return "temporal_boundary_smoothing"
    if "lacks_xodr" in reason_text or "xodr_topology_untrusted" in reason_text:
        if label in {"R2", "R3", "R6"}:
            return "topology_confirmation_missing"
        return "xodr_evidence_weak"
    if "signalized_policy_without_meta_tl" in reason_text:
        return "traffic_light_meta_or_signalized_rule"
    if "low_confidence" in reason_text:
        return "threshold_or_evidence_strength"
    return "needs_rgb_visual_review"


def _is_candidate_anomaly(ann: Dict[str, Any], prev_label: Optional[str]) -> Tuple[bool, List[str]]:
    label = str(ann.get("primary_road_structure") or "")
    confidence = float(ann.get("confidence") or 0.0)
    reasons = _hard_review_reasons(ann)
    if reasons:
        return True, _annotation_review_reasons(ann)
    if confidence < 0.75:
        return True, ["confidence_lt_0.75"]
    if prev_label is not None and label and label != prev_label:
        return True, ["primary_rs_transition"]
    return False, []


def _safe_open_rgb(path: Optional[Path], size: Tuple[int, int]) -> Image.Image:
    if Image is None:
        raise RuntimeError("PIL is required to render review sheets")
    if path is None:
        img = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "RGB missing", fill=(20, 20, 20))
        return img
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"RGB open failed\n{path.name}", fill=(20, 20, 20))
        return img
    img.thumbnail(size)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _draw_frame_tile(
    ann: Dict[str, Any],
    run_dir: Path,
    tile_size: Tuple[int, int],
    selected_reason: Optional[str] = None,
) -> Image.Image:
    frame_id = int(ann.get("frame_id", 0))
    label = str(ann.get("primary_road_structure") or "-")
    confidence = ann.get("confidence")
    reasons = _annotation_review_reasons(ann)
    frame_rs = ann.get("frame_rs_annotation", {}) or {}
    img = _safe_open_rgb(_rgb_path_for_frame(run_dir, frame_id), tile_size)
    draw = ImageDraw.Draw(img)
    color = RS_COLORS.get(label, (70, 70, 70))
    draw.rectangle((0, 0, tile_size[0], 26), fill=color)
    text = f"f={frame_id} {label} conf={float(confidence or 0.0):.2f}"
    if frame_rs.get("review_required") or reasons:
        text += " REVIEW"
    draw.text((5, 6), text, fill=(255, 255, 255))
    bottom_text = selected_reason or ", ".join(reasons[:2]) or str(frame_rs.get("decision_source") or "")
    if bottom_text:
        draw.rectangle((0, tile_size[1] - 22, tile_size[0], tile_size[1]), fill=(0, 0, 0))
        draw.text((5, tile_size[1] - 18), bottom_text[:90], fill=(255, 255, 255))
    return img


def _write_sheets(
    annotations: List[Dict[str, Any]],
    run_dir: Path,
    out_dir: Path,
    frames_per_sheet: int,
    cols: int,
    prefix: str,
    selected_reasons: Optional[Dict[int, str]] = None,
) -> List[str]:
    if Image is None or not annotations:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_size = (288, 116)
    rows = max(1, math.ceil(frames_per_sheet / cols))
    paths: List[str] = []
    for chunk_index, start in enumerate(range(0, len(annotations), frames_per_sheet)):
        chunk = annotations[start:start + frames_per_sheet]
        sheet = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), (235, 235, 235))
        for local_idx, ann in enumerate(chunk):
            frame_id = int(ann.get("frame_id", 0))
            reason = selected_reasons.get(frame_id) if selected_reasons else None
            tile = _draw_frame_tile(ann, run_dir, tile_size, reason)
            x = (local_idx % cols) * tile_size[0]
            y = (local_idx // cols) * tile_size[1]
            sheet.paste(tile, (x, y))
        first_frame = chunk[0].get("frame_id")
        last_frame = chunk[-1].get("frame_id")
        path = out_dir / f"{prefix}_{chunk_index:04d}_f{first_frame}_to_f{last_frame}.jpg"
        sheet.save(path, quality=88)
        paths.append(str(path))
    return paths


def _route_anomaly_rows(
    scenario: str,
    town: str,
    route_id: str,
    annotations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    rows: List[Dict[str, Any]] = []
    selected_reasons: Dict[int, str] = {}
    prev_label: Optional[str] = None
    for ann in annotations:
        label = str(ann.get("primary_road_structure") or "")
        is_anom, reasons = _is_candidate_anomaly(ann, prev_label)
        if is_anom:
            frame_id = int(ann.get("frame_id", 0))
            bucket = _issue_bucket(reasons, label)
            selected_reasons[frame_id] = bucket
            frame_rs = ann.get("frame_rs_annotation", {}) or {}
            rows.append(
                {
                    "scenario": scenario,
                    "town": town,
                    "route_id": route_id,
                    "frame_id": frame_id,
                    "label": label,
                    "confidence": ann.get("confidence"),
                    "issue_bucket": bucket,
                    "reasons": reasons,
                    "decision_source": frame_rs.get("decision_source"),
                    "comment": frame_rs.get("comment") or ann.get("annotation_comment", ""),
                    "rgb_review_status": "pending_visual_check",
                    "suspected_cause": bucket,
                }
            )
        if label:
            prev_label = label
    return rows, selected_reasons


def _route_summary(route_result: Dict[str, Any], anomaly_count: int, sheet_paths: List[str], anomaly_sheets: List[str]) -> Dict[str, Any]:
    return {
        "route_id": route_result.get("route_id"),
        "status": route_result.get("status"),
        "xml_path": route_result.get("xml_path"),
        "xml_town": route_result.get("xml_town"),
        "xml_available": route_result.get("xml_available"),
        "num_frames": route_result.get("num_frames", 0),
        "primary_rs_distribution": route_result.get("primary_rs_distribution", {}),
        "review_required_frames": route_result.get("review_required_frames", 0),
        "review_required_ratio": route_result.get("review_required_ratio", 0.0),
        "review_reason_distribution": route_result.get("review_reason_distribution", {}),
        "xodr_source_distribution": route_result.get("xodr_source_distribution", {}),
        "confidence_stats": route_result.get("confidence_stats", {}),
        "primary_rs_transitions": route_result.get("primary_rs_transitions", []),
        "candidate_anomaly_frames": anomaly_count,
        "full_frame_sheets": sheet_paths,
        "anomaly_sheets": anomaly_sheets,
    }


def _run_review(args: argparse.Namespace) -> Dict[str, Any]:
    lead_data_root = Path(args.lead_data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    anomaly_jsonl = out_root / "candidate_anomalies.jsonl"
    if anomaly_jsonl.exists():
        anomaly_jsonl.unlink()

    collector = ScenarioCollector(
        lead_data_root=str(lead_data_root),
        output_dir=str(out_root / "_collector_tmp"),
        xml_root=args.xml_root,
        carla_root=args.carla_root,
        rule_config_json=args.rule_config_json,
    )

    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE) if args.scenario.lower() == "all" else [
        item.strip() for item in args.scenario.split(",") if item.strip()
    ]
    invalid = [scenario for scenario in scenarios if scenario not in SCENARIO_TO_ROAD_STRUCTURE]
    if invalid:
        raise ValueError(f"未知场景: {invalid}")
    if args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]

    global_summary: Dict[str, Any] = {
        "output_dir": str(out_root),
        "lead_data_root": str(lead_data_root),
        "xml_root": args.xml_root,
        "carla_root": args.carla_root,
        "samples_per_town": args.samples_per_town,
        "scenarios": {},
    }
    total_frames = 0
    total_routes = 0
    total_anomalies = 0

    for scenario_index, scenario in enumerate(scenarios, 1):
        print(f"[{scenario_index}/{len(scenarios)}] scenario={scenario}", flush=True)
        scenario_dir = out_root / scenario
        scenario_summary: Dict[str, Any] = {"towns": {}, "scenario": scenario}
        runs_by_town = _scenario_runs_by_town(lead_data_root, scenario)
        for town, town_runs in runs_by_town.items():
            sampled_runs = _select_research_runs(town_runs, args.samples_per_town)
            if args.max_routes_per_town > 0:
                sampled_runs = sampled_runs[:args.max_routes_per_town]
            print(f"  town={town} routes={len(sampled_runs)}", flush=True)
            town_summary: Dict[str, Any] = {
                "sampled_run_ids": [run.name for run in sampled_runs],
                "routes": [],
                "primary_rs_distribution": {},
                "review_reason_distribution": {},
                "candidate_anomaly_frames": 0,
                "num_frames": 0,
            }
            town_primary = Counter()
            town_review = Counter()
            for route_idx, run_dir in enumerate(sampled_runs, 1):
                print(f"    [{route_idx}/{len(sampled_runs)}] {run_dir.name}", flush=True)
                route_result = collector._process_route(  # pylint: disable=protected-access
                    scenario,
                    run_dir,
                    max_frames_per_route=args.max_frames_per_route if args.max_frames_per_route > 0 else None,
                )
                annotations = route_result.get("annotations", [])
                route_out = scenario_dir / town / run_dir.name
                _write_json(route_out / "route_annotations.json", route_result)
                anomaly_rows, selected_reasons = _route_anomaly_rows(scenario, town, run_dir.name, annotations)
                if anomaly_rows:
                    _append_jsonl(anomaly_jsonl, anomaly_rows)
                full_sheets = _write_sheets(
                    annotations,
                    run_dir,
                    route_out / "sheets",
                    args.frames_per_sheet,
                    args.sheet_cols,
                    "all_frames",
                )
                anomaly_annotations = [
                    ann for ann in annotations
                    if int(ann.get("frame_id", 0)) in selected_reasons
                ]
                anomaly_sheets = _write_sheets(
                    anomaly_annotations,
                    run_dir,
                    route_out / "sheets",
                    args.frames_per_sheet,
                    args.sheet_cols,
                    "candidate_anomalies",
                    selected_reasons=selected_reasons,
                )
                route_summary = _route_summary(route_result, len(anomaly_rows), full_sheets, anomaly_sheets)
                _write_json(route_out / "route_review_summary.json", route_summary)
                town_summary["routes"].append(route_summary)
                total_routes += 1
                total_frames += int(route_result.get("num_frames", 0) or 0)
                total_anomalies += len(anomaly_rows)
                town_summary["candidate_anomaly_frames"] += len(anomaly_rows)
                town_summary["num_frames"] += int(route_result.get("num_frames", 0) or 0)
                town_primary.update(route_result.get("primary_rs_distribution", {}))
                town_review.update(route_result.get("review_reason_distribution", {}))
            town_summary["primary_rs_distribution"] = dict(sorted(town_primary.items()))
            town_summary["review_reason_distribution"] = dict(sorted(town_review.items()))
            scenario_summary["towns"][town] = town_summary
            _write_json(scenario_dir / town / "town_review_summary.json", town_summary)
        scenario_counter = Counter()
        scenario_review = Counter()
        scenario_frames = 0
        scenario_anomalies = 0
        for town_summary in scenario_summary["towns"].values():
            scenario_counter.update(town_summary.get("primary_rs_distribution", {}))
            scenario_review.update(town_summary.get("review_reason_distribution", {}))
            scenario_frames += int(town_summary.get("num_frames", 0) or 0)
            scenario_anomalies += int(town_summary.get("candidate_anomaly_frames", 0) or 0)
        scenario_summary["num_frames"] = scenario_frames
        scenario_summary["candidate_anomaly_frames"] = scenario_anomalies
        scenario_summary["primary_rs_distribution"] = dict(sorted(scenario_counter.items()))
        scenario_summary["review_reason_distribution"] = dict(sorted(scenario_review.items()))
        global_summary["scenarios"][scenario] = scenario_summary
        _write_json(scenario_dir / "scenario_review_summary.json", scenario_summary)

    global_summary["total_routes"] = total_routes
    global_summary["total_frames"] = total_frames
    global_summary["total_candidate_anomaly_frames"] = total_anomalies
    global_summary["candidate_anomalies_jsonl"] = str(anomaly_jsonl)
    _write_json(out_root / "global_review_summary.json", global_summary)
    return global_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-frame ROAD_STRUCTURE RGB review")
    parser.add_argument("--scenario", default="all", help="Scenario name, comma list, or all")
    parser.add_argument("--samples-per-town", type=int, default=5)
    parser.add_argument("--max-routes-per-town", type=int, default=0, help="Debug cap; 0 means no extra cap")
    parser.add_argument("--max-scenarios", type=int, default=0, help="Debug cap; 0 means all selected scenarios")
    parser.add_argument("--max-frames-per-route", type=int, default=0, help="Debug cap; 0 means full route")
    parser.add_argument("--lead-data-root", default=str(_DEFAULT_LEAD_DATA_ROOT))
    parser.add_argument("--xml-root", default=str(_DEFAULT_XML_ROOT))
    parser.add_argument("--carla-root", default=str(_DEFAULT_CARLA_ROOT))
    parser.add_argument("--output-dir", default=str(KEYFRAME_DIR / "collection_output" / "rs_full_frame_review"))
    parser.add_argument("--rule-config-json", default="")
    parser.add_argument("--frames-per-sheet", type=int, default=40)
    parser.add_argument("--sheet-cols", type=int, default=4)
    args = parser.parse_args()

    summary = _run_review(args)
    print(
        "done: "
        f"routes={summary['total_routes']} frames={summary['total_frames']} "
        f"candidate_anomalies={summary['total_candidate_anomaly_frames']} "
        f"output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
