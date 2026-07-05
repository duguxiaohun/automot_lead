#!/usr/bin/env python3
"""RGB-first blind audit for ROAD_STRUCTURE / EVENT labels.

The audit is intentionally two-stage:

1. Read every RGB frame and build blind visual evidence without looking at the
   generated RS/EVENT labels.
2. Load the current rule labels only after the blind pass, then compare spans
   and write mismatch reports plus optional contact sheets.

The automatic blind guess is conservative and is mainly a triage signal. For
high-stakes fixes, use the generated blind sheets to write manual span answers
first, then compare those spans with the current labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTOMOT_ROOT = REPO_ROOT / "AutoMoT"
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))
if str(AUTOMOT_ROOT / "keyframe_filter") not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT / "keyframe_filter"))

from collector import ScenarioCollector, _extract_town  # noqa: E402
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from rgb_r4_r5_audit import _front_view, _junction_visual_evidence, _load_pickle_file, _traffic_light_evidence  # noqa: E402


_WORKER_COLLECTOR: ScenarioCollector | None = None
_WORKER_OUTPUT_DIR: pathlib.Path | None = None
_WORKER_WRITE_SHEETS = False
_WORKER_SHEET_LIMIT = 80


OBSTACLE_RECOVERY_SCENARIOS = {
    "Accident",
    "AccidentTwoWays",
    "ConstructionObstacle",
    "ConstructionObstacleTwoWays",
    "ParkedObstacle",
    "ParkedObstacleTwoWays",
    "HazardAtSideLane",
    "HazardAtSideLaneTwoWays",
    "VehicleOpensDoorTwoWays",
}

EVENT_CORE_SCENARIOS = OBSTACLE_RECOVERY_SCENARIOS | {
    "ParkingCutIn",
    "StaticCutIn",
    "InvadingTurn",
    "OppositeVehicleRunningRedLight",
    "OppositeVehicleTakingPriority",
    "BlockedIntersection",
    "CrossJunctionDefectTrafficLight",
    "DynamicObjectCrossing",
    "PedestrianCrossing",
    "CrossingBicycleFlow",
    "VehicleTurningRoutePedestrian",
}


@dataclass
class BlindFrame:
    frame_id: int
    rgb_path: str
    blind_rs: str
    blind_event: str
    signal_score: float
    junction_score: float
    motion_score: float
    vehicle_tail_score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class RouteAudit:
    scenario: str
    route_id: str
    town: str
    status: str
    frame_count: int = 0
    label_frame_count: int = 0
    blind_rs_distribution: dict[str, int] = field(default_factory=dict)
    blind_event_distribution: dict[str, int] = field(default_factory=dict)
    label_rs_distribution: dict[str, int] = field(default_factory=dict)
    label_event_distribution: dict[str, int] = field(default_factory=dict)
    mismatch_counts: dict[str, int] = field(default_factory=dict)
    issue_reasons: list[str] = field(default_factory=list)
    first_blind_sheet: str = ""
    first_compare_sheet: str = ""
    message: str = ""


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {part for chunk in value.split(",") for part in chunk.split() if part}
    return out or None


def _frame_id(path: pathlib.Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _iter_routes(data_root: pathlib.Path, scenarios: set[str] | None) -> list[tuple[str, str, pathlib.Path]]:
    routes: list[tuple[str, str, pathlib.Path]] = []
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if (route_dir / "rgb").is_dir():
                routes.append((scenario, route_dir.name, route_dir))
    return routes


def _select_evenly(items: list[tuple[str, str, pathlib.Path]], limit: int | None) -> list[tuple[str, str, pathlib.Path]]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indexes = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in indexes]


def _apply_route_sampling(
    routes: list[tuple[str, str, pathlib.Path]],
    *,
    max_routes_per_scenario: int,
    samples_per_town: int,
) -> list[tuple[str, str, pathlib.Path]]:
    if max_routes_per_scenario <= 0 and samples_per_town <= 0:
        return routes
    by_scenario: dict[str, list[tuple[str, str, pathlib.Path]]] = defaultdict(list)
    for item in routes:
        by_scenario[item[0]].append(item)
    selected: list[tuple[str, str, pathlib.Path]] = []
    for scenario, items in sorted(by_scenario.items()):
        if samples_per_town > 0:
            by_town: dict[str, list[tuple[str, str, pathlib.Path]]] = defaultdict(list)
            for item in items:
                by_town[_extract_town(item[1]) or "unknown"].append(item)
            scenario_items: list[tuple[str, str, pathlib.Path]] = []
            for town_items in by_town.values():
                scenario_items.extend(_select_evenly(sorted(town_items, key=lambda x: x[1]), samples_per_town))
        else:
            scenario_items = items
        selected.extend(_select_evenly(sorted(scenario_items, key=lambda x: x[1]), max_routes_per_scenario or None))
    return sorted(selected, key=lambda x: (x[0], x[1]))


def _vehicle_tail_score(img_rgb: np.ndarray) -> float:
    front = _front_view(img_rgb)
    h, w = front.shape[:2]
    roi = front[int(h * 0.42) :, int(w * 0.18) : int(w * 0.82)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 90, 80]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 90, 80]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 3 <= area <= 600:
            score += min(1.0, area / 80.0)
    return float(min(score, 8.0))


def _motion_score(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    h, w = diff.shape[:2]
    roi = diff[int(h * 0.35) :, int(w * 0.15) : int(w * 0.85)]
    return float(np.mean(roi) / 255.0)


def _traffic_light_confirmed(route_dir: pathlib.Path, frame_id: int) -> tuple[bool, list[str]]:
    sources: list[str] = []
    meta_path = route_dir / "metas" / f"{frame_id:04d}.pkl"
    if meta_path.exists():
        try:
            meta = _load_pickle_file(meta_path)
            tl_state = str(meta.get("traffic_light_state", "None")) if isinstance(meta, dict) else "None"
            if tl_state not in {"None", "NONE", "Unknown", "unknown", ""}:
                sources.append(f"meta_tl={tl_state}")
        except Exception:
            pass
    bbox_path = route_dir / "bboxes" / f"{frame_id:04d}.pkl"
    if bbox_path.exists():
        try:
            boxes = _load_pickle_file(bbox_path)
            for box in boxes if isinstance(boxes, list) else []:
                if not isinstance(box, dict):
                    continue
                cls = str(box.get("class", "") or box.get("type", "")).lower()
                if "traffic_light" in cls or "traffic light" in cls:
                    sources.append("bbox_traffic_light")
                    break
        except Exception:
            pass
    return bool(sources), sources


def _junction_confirmed(route_dir: pathlib.Path, frame_id: int) -> tuple[bool, list[str]]:
    sources: list[str] = []
    meta_path = route_dir / "metas" / f"{frame_id:04d}.pkl"
    if meta_path.exists():
        try:
            meta = _load_pickle_file(meta_path)
            if isinstance(meta, dict):
                if bool(meta.get("is_junction", False)):
                    sources.append("meta_junction")
                if bool(meta.get("stop_sign", False) or meta.get("stop_hazard", False)):
                    sources.append("meta_stop")
        except Exception:
            pass
    bbox_path = route_dir / "bboxes" / f"{frame_id:04d}.pkl"
    if bbox_path.exists():
        try:
            boxes = _load_pickle_file(bbox_path)
            for box in boxes if isinstance(boxes, list) else []:
                if not isinstance(box, dict):
                    continue
                cls = str(box.get("class", "") or box.get("type", "")).lower()
                if "stop" in cls or "yield" in cls:
                    sources.append("bbox_stop_yield")
                    break
        except Exception:
            pass
    return bool(sources), sources


def _smooth_streaks(frames: list[BlindFrame]) -> None:
    """Promote short but stable RGB evidence streaks to blind RS guesses."""
    def _has_confirmed_signal(frame: BlindFrame) -> bool:
        return any(note.startswith("meta_tl=") or note == "bbox_traffic_light" for note in frame.notes)

    def _has_stopline_junction(frame: BlindFrame) -> bool:
        return "rgb_stopline_candidate" in frame.notes and any(
            note in {"meta_junction", "meta_stop", "bbox_stop_yield"} for note in frame.notes
        )

    idx = 0
    while idx < len(frames):
        active = _has_stopline_junction(frames[idx]) and frames[idx].junction_score >= 0.85
        if not active:
            idx += 1
            continue
        start = idx
        while idx < len(frames) and _has_stopline_junction(frames[idx]) and frames[idx].junction_score >= 0.85:
            idx += 1
        end = idx
        if end - start < 4:
            continue
        for frame in frames[start:end]:
            frame.blind_rs = "R5"
            frame.blind_event = "R-E5"
            frame.notes.append("blind_r5_stable_stopline_streak")

    for attr, rs_label, event_label, min_len in (
        ("signal_score", "R4", "R-E4", 2),
    ):
        idx = 0
        while idx < len(frames):
            score = getattr(frames[idx], attr)
            confirmed_signal = (
                attr != "signal_score"
                or _has_confirmed_signal(frames[idx])
            )
            threshold = 0.75 if attr == "signal_score" else 1.80
            active = confirmed_signal and score >= threshold
            if not active:
                idx += 1
                continue
            start = idx
            while idx < len(frames):
                confirmed_signal = (
                    attr != "signal_score"
                    or _has_confirmed_signal(frames[idx])
                )
                threshold = 0.75 if attr == "signal_score" else 1.80
                if not (confirmed_signal and getattr(frames[idx], attr) >= threshold):
                    break
                idx += 1
            end = idx
            if end - start < min_len:
                continue
            for frame in frames[start:end]:
                frame.blind_rs = rs_label
                frame.blind_event = event_label
                frame.notes.append(f"blind_{rs_label.lower()}_stable_rgb_streak")


def build_blind_frames(route_dir: pathlib.Path) -> list[BlindFrame]:
    """Read every RGB frame and create blind guesses without current labels."""
    frame_paths = sorted((route_dir / "rgb").glob("*.jpg"), key=_frame_id)
    frames: list[BlindFrame] = []
    prev_gray: np.ndarray | None = None
    for path in frame_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        front_rgb = _front_view(img_rgb)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        possible_signal, red, yellow, green, signal_score = _traffic_light_evidence(front_rgb)
        possible_junction, stopline_crosswalk, turn_marking, junction_score = _junction_visual_evidence(img_rgb)
        motion = _motion_score(prev_gray, gray)
        tail_score = _vehicle_tail_score(img_rgb)
        prev_gray = gray

        blind_rs = "R1"
        blind_event = "R-E1"
        notes: list[str] = []
        tl_confirmed, tl_sources = _traffic_light_confirmed(route_dir, _frame_id(path))
        if possible_signal and signal_score >= 0.75:
            colors = [name for flag, name in ((red, "red"), (yellow, "yellow"), (green, "green")) if flag]
            notes.append("rgb_signal_candidate:" + ",".join(colors or ["unknown"]))
            if tl_confirmed:
                notes.extend(tl_sources)
        if possible_junction and junction_score >= 0.85:
            if stopline_crosswalk:
                notes.append("rgb_stopline_candidate")
            if turn_marking:
                notes.append("rgb_turn_marking_candidate")
            junction_confirmed, junction_sources = _junction_confirmed(route_dir, _frame_id(path))
            if junction_confirmed:
                notes.extend(junction_sources)
        if tail_score >= 2.0:
            notes.append("rgb_near_vehicle_tail")
        if motion >= 0.10:
            notes.append("rgb_high_temporal_change")
        frames.append(
            BlindFrame(
                frame_id=_frame_id(path),
                rgb_path=str(path),
                blind_rs=blind_rs,
                blind_event=blind_event,
                signal_score=round(signal_score, 4),
                junction_score=round(junction_score, 4),
                motion_score=round(motion, 4),
                vehicle_tail_score=round(tail_score, 4),
                notes=notes,
            )
        )
    _smooth_streaks(frames)
    return frames


def _spans(values: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
    ordered = list(values)
    spans: list[dict[str, Any]] = []
    idx = 0
    while idx < len(ordered):
        frame_id, label = ordered[idx]
        start = frame_id
        count = 1
        idx += 1
        while idx < len(ordered) and ordered[idx][1] == label:
            frame_id = ordered[idx][0]
            count += 1
            idx += 1
        spans.append({"label": label, "start_frame": start, "end_frame": frame_id, "frames": count})
    return spans


def _load_current_labels(
    collector: ScenarioCollector,
    scenario: str,
    route_dir: pathlib.Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = collector._process_route(scenario, route_dir)
    if result.get("status") != "success":
        return [], result
    return result.get("annotations", []), result


def _compare_frames(
    scenario: str,
    blind_frames: list[BlindFrame],
    label_frames: list[dict[str, Any]],
) -> tuple[Counter, list[dict[str, Any]], list[str]]:
    label_by_frame = {int(ann.get("frame_id")): ann for ann in label_frames}
    label_events_by_frame = {
        int(ann.get("frame_id")): str(ann.get("primary_event") or "")
        for ann in label_frames
    }
    mismatch = Counter()
    examples: list[dict[str, Any]] = []
    reasons: set[str] = set()

    def _label_has_signal_control(ann: dict[str, Any]) -> bool:
        evidence = ann.get("evidence") or {}
        tl = str(evidence.get("traffic_light_state", "")).strip().lower()
        if tl not in {"", "none", "null", "nan", "unknown"}:
            return True
        bbox = evidence.get("bbox_semantics") or {}
        if bool(bbox.get("traffic_light")):
            return True
        rules = set(evidence.get("rules_fired", []) or [])
        return any(rule.startswith("r4_") and "demoted" not in rule for rule in rules)

    def _label_has_nonsignal_control(ann: dict[str, Any]) -> bool:
        evidence = ann.get("evidence") or {}
        bbox = evidence.get("bbox_semantics") or {}
        if bool(bbox.get("stop_sign") or bbox.get("yield_sign")):
            return True
        if bool(evidence.get("meta_stop_hazard") or evidence.get("meta_is_junction")):
            return True
        flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
        if bool(flags.get("strong_control_context") and (flags.get("near_junction") or flags.get("close_trigger_for_junction"))):
            return True
        rules = set(evidence.get("rules_fired", []) or [])
        return any(rule.startswith("r5_") or rule.endswith("_r5") for rule in rules)

    def _label_has_defect_r5_control(ann: dict[str, Any]) -> bool:
        if scenario != "CrossJunctionDefectTrafficLight":
            return False
        return _label_has_signal_control(ann) or _label_has_nonsignal_control(ann)

    def _label_has_r3_merge_or_highway_control(ann: dict[str, Any]) -> bool:
        evidence = ann.get("evidence") or {}
        if str(evidence.get("rule_kind", "")).lower() in {"highway_merge", "highway_exit"}:
            return True
        if str(evidence.get("route_semantic_bucket", "")).lower() == "highway_rgb_route":
            return True
        rules = set(evidence.get("rules_fired", []) or [])
        if any(rule.startswith("r3_") or "r3_" in rule for rule in rules):
            return True
        diag = evidence.get("diagnostic_attribution") or {}
        if str(diag.get("decision_source", "")).lower() in {"merge_actor_flow_or_topology_window", "highway_or_exit_space"}:
            return True
        return False

    def _label_has_twoways_obstacle_core(ann: dict[str, Any]) -> bool:
        if scenario not in {
            "AccidentTwoWays",
            "ConstructionObstacleTwoWays",
            "HazardAtSideLaneTwoWays",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
        }:
            return False
        evidence = ann.get("evidence") or {}
        if str(evidence.get("rule_kind", "")).lower() not in {"twoways_obstacle", "vehicle_opens_door_twoways"}:
            return False
        rules = set(evidence.get("rules_fired", []) or [])
        if any(rule.startswith("r2_") for rule in rules):
            return True
        return ann.get("primary_event") == "U-E2"

    def _twoways_regular_after_completed_return(frame_id: int) -> bool:
        if scenario not in {
            "AccidentTwoWays",
            "ConstructionObstacleTwoWays",
            "HazardAtSideLaneTwoWays",
            "ParkedObstacleTwoWays",
            "VehicleOpensDoorTwoWays",
        }:
            return False
        previous = [
            event
            for fid, event in label_events_by_frame.items()
            if fid < frame_id
        ]
        if "U-E2" not in previous or "R-E2" not in previous:
            return False
        recent = [
            event
            for fid, event in label_events_by_frame.items()
            if frame_id - 8 <= fid <= frame_id
        ]
        return "U-E2" not in recent and "R-E2" not in recent

    for blind in blind_frames:
        ann = label_by_frame.get(blind.frame_id)
        if ann is None:
            mismatch["missing_label_frame"] += 1
            continue
        label_rs = ann.get("primary_road_structure")
        label_event = ann.get("primary_event")
        frame_reasons: list[str] = []
        if blind.blind_rs in {"R4", "R5"} and label_rs != blind.blind_rs:
            explained = (
                label_rs == "R4"
                and _label_has_signal_control(ann)
                and blind.blind_rs == "R5"
            ) or (
                label_rs == "R5"
                and (_label_has_nonsignal_control(ann) or _label_has_defect_r5_control(ann))
                and blind.blind_rs == "R4"
            ) or (
                label_rs == "R3"
                and blind.blind_rs in {"R4", "R5"}
                and _label_has_r3_merge_or_highway_control(ann)
            ) or (
                label_rs == "R2"
                and blind.blind_rs in {"R4", "R5"}
                and _label_has_twoways_obstacle_core(ann)
            )
            if not explained:
                key = f"blind_{blind.blind_rs}_label_{label_rs}"
                mismatch[key] += 1
                frame_reasons.append(key)
        if label_rs in {"R4", "R5"} and blind.blind_rs == "R1":
            explained = (label_rs == "R4" and _label_has_signal_control(ann)) or (
                label_rs == "R5" and (_label_has_nonsignal_control(ann) or _label_has_defect_r5_control(ann))
            )
            if not explained:
                key = f"label_{label_rs}_without_rgb_junction_signal"
                mismatch[key] += 1
                frame_reasons.append(key)
        if scenario in EVENT_CORE_SCENARIOS and label_event == "R-E1":
            if (
                (blind.vehicle_tail_score >= 2.0 or blind.motion_score >= 0.10)
                and not _twoways_regular_after_completed_return(blind.frame_id)
            ):
                key = "event_regular_during_rgb_object_or_motion_activity"
                mismatch[key] += 1
                frame_reasons.append(key)
        if label_event in {"U-E2", "U-E3"} and label_rs in {"R4", "R5"}:
            key = "junction_label_keeps_u2_u3"
            mismatch[key] += 1
            frame_reasons.append(key)
        if frame_reasons and len(examples) < 80:
            examples.append(
                {
                    "frame_id": blind.frame_id,
                    "rgb_path": blind.rgb_path,
                    "blind_rs": blind.blind_rs,
                    "blind_event": blind.blind_event,
                    "label_rs": label_rs,
                    "label_event": label_event,
                    "reasons": frame_reasons,
                    "blind_notes": blind.notes,
                    "signal_score": blind.signal_score,
                    "junction_score": blind.junction_score,
                    "motion_score": blind.motion_score,
                    "vehicle_tail_score": blind.vehicle_tail_score,
                }
            )
            reasons.update(frame_reasons)
    return mismatch, examples, sorted(reasons)


def _draw_sheet(
    out_path: pathlib.Path,
    frames: list[BlindFrame],
    *,
    title: str,
    labels_by_frame: dict[int, dict[str, Any]] | None,
    max_tiles: int,
) -> str:
    chosen = frames[:max_tiles]
    if not chosen:
        return ""
    cols = 5
    thumb_w, thumb_h, label_h = 288, 96, 58
    rows = math.ceil(len(chosen) / cols)
    header_h = 70
    sheet = np.full((header_h + rows * (thumb_h + label_h), cols * thumb_w, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, title[:170], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(sheet, "blind rows are generated before current labels are loaded", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA)
    for idx, frame in enumerate(chosen):
        bgr = cv2.imread(frame.rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.full((thumb_h, thumb_w, 3), 32, dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x = (idx % cols) * thumb_w
        y = header_h + (idx // cols) * (thumb_h + label_h)
        sheet[y : y + thumb_h, x : x + thumb_w] = bgr
        cv2.rectangle(sheet, (x, y + thumb_h), (x + thumb_w - 1, y + thumb_h + label_h - 1), (255, 255, 255), -1)
        line1 = f"f={frame.frame_id} blind={frame.blind_rs}/{frame.blind_event}"
        cv2.putText(sheet, line1[:48], (x + 4, y + thumb_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        if labels_by_frame is not None:
            ann = labels_by_frame.get(frame.frame_id, {})
            line2 = f"label={ann.get('primary_road_structure')}/{ann.get('primary_event')}"
        else:
            line2 = "label=hidden"
        cv2.putText(sheet, line2[:48], (x + 4, y + thumb_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 180), 1, cv2.LINE_AA)
        note = ",".join(frame.notes) or f"sig={frame.signal_score} junc={frame.junction_score}"
        cv2.putText(sheet, note[:54], (x + 4, y + thumb_h + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 80, 80), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    return str(out_path)


def audit_route(
    collector: ScenarioCollector,
    scenario: str,
    route_id: str,
    route_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    write_sheets: bool,
    sheet_limit: int,
) -> tuple[RouteAudit, list[dict[str, Any]], dict[str, Any]]:
    bad, info = is_abnormal_lead_route(route_dir, scenario)
    if bad:
        return (
            RouteAudit(
                scenario=scenario,
                route_id=route_id,
                town=_extract_town(route_id) or "unknown",
                status="skipped_abnormal_duration",
                message=str(info.get("reason", "abnormal_duration")),
            ),
            [],
            {},
        )
    blind_frames = build_blind_frames(route_dir)
    label_frames, label_result = _load_current_labels(collector, scenario, route_dir)
    if label_result.get("status") != "success":
        return (
            RouteAudit(
                scenario=scenario,
                route_id=route_id,
                town=_extract_town(route_id) or "unknown",
                status=label_result.get("status", "label_error"),
                frame_count=len(blind_frames),
                message=str(label_result.get("skip_reason") or label_result.get("error") or ""),
            ),
            [],
            label_result,
        )
    blind_rs = Counter(frame.blind_rs for frame in blind_frames)
    blind_event = Counter(frame.blind_event for frame in blind_frames)
    label_rs = Counter(str(ann.get("primary_road_structure")) for ann in label_frames)
    label_event = Counter(str(ann.get("primary_event")) for ann in label_frames)
    mismatch, examples, reasons = _compare_frames(scenario, blind_frames, label_frames)
    labels_by_frame = {int(ann.get("frame_id")): ann for ann in label_frames}

    first_blind_sheet = ""
    first_compare_sheet = ""
    if write_sheets:
        route_out = output_dir / "blind_sheets" / scenario / route_id
        first_blind_sheet = _draw_sheet(
            route_out / "blind_page_000.jpg",
            blind_frames,
            title=f"{scenario}/{route_id}",
            labels_by_frame=None,
            max_tiles=sheet_limit,
        )
        interesting_ids = {int(ex["frame_id"]) for ex in examples[:sheet_limit]}
        interesting = [frame for frame in blind_frames if frame.frame_id in interesting_ids]
        if not interesting:
            interesting = blind_frames[: min(sheet_limit, len(blind_frames))]
        first_compare_sheet = _draw_sheet(
            route_out / "compare_page_000.jpg",
            interesting,
            title=f"{scenario}/{route_id} | blind-vs-label examples",
            labels_by_frame=labels_by_frame,
            max_tiles=sheet_limit,
        )

    audit = RouteAudit(
        scenario=scenario,
        route_id=route_id,
        town=_extract_town(route_id) or "unknown",
        status="success",
        frame_count=len(blind_frames),
        label_frame_count=len(label_frames),
        blind_rs_distribution=dict(sorted(blind_rs.items())),
        blind_event_distribution=dict(sorted(blind_event.items())),
        label_rs_distribution=dict(sorted(label_rs.items())),
        label_event_distribution=dict(sorted(label_event.items())),
        mismatch_counts=dict(sorted(mismatch.items())),
        issue_reasons=reasons,
        first_blind_sheet=first_blind_sheet,
        first_compare_sheet=first_compare_sheet,
    )
    route_detail = {
        "scenario": scenario,
        "route_id": route_id,
        "blind_spans_rs": _spans((f.frame_id, f.blind_rs) for f in blind_frames),
        "blind_spans_event": _spans((f.frame_id, f.blind_event) for f in blind_frames),
        "label_spans_rs": _spans((int(a.get("frame_id")), str(a.get("primary_road_structure"))) for a in label_frames),
        "label_spans_event": _spans((int(a.get("frame_id")), str(a.get("primary_event"))) for a in label_frames),
        "examples": examples,
    }
    return audit, examples, route_detail


def _init_worker(data_root: str, output_dir: str, write_sheets: bool, sheet_limit: int) -> None:
    global _WORKER_COLLECTOR, _WORKER_OUTPUT_DIR, _WORKER_WRITE_SHEETS, _WORKER_SHEET_LIMIT
    _WORKER_OUTPUT_DIR = pathlib.Path(output_dir)
    _WORKER_WRITE_SHEETS = bool(write_sheets)
    _WORKER_SHEET_LIMIT = int(sheet_limit)
    _WORKER_COLLECTOR = ScenarioCollector(
        lead_data_root=data_root,
        output_dir=str(_WORKER_OUTPUT_DIR / "_label_cache"),
    )


def _audit_route_worker(task: tuple[str, str, str]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if _WORKER_COLLECTOR is None or _WORKER_OUTPUT_DIR is None:
        raise RuntimeError("worker not initialized")
    scenario, route_id, route_dir = task
    audit, examples, detail = audit_route(
        _WORKER_COLLECTOR,
        scenario,
        route_id,
        pathlib.Path(route_dir),
        _WORKER_OUTPUT_DIR,
        write_sheets=_WORKER_WRITE_SHEETS,
        sheet_limit=_WORKER_SHEET_LIMIT,
    )
    return asdict(audit), examples, detail


def _write_outputs(output_dir: pathlib.Path, audits: list[RouteAudit], details: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dicts = [asdict(audit) for audit in audits]
    (output_dir / "route_blind_rs_event_audit.json").write_text(json.dumps(audit_dicts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "route_blind_rs_event_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "scenario",
        "town",
        "route_id",
        "status",
        "frame_count",
        "label_frame_count",
        "blind_rs_distribution",
        "label_rs_distribution",
        "blind_event_distribution",
        "label_event_distribution",
        "mismatch_counts",
        "issue_reasons",
        "first_blind_sheet",
        "first_compare_sheet",
        "message",
    ]
    with (output_dir / "route_blind_rs_event_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in audit_dicts:
            flat = dict(row)
            for key in ("blind_rs_distribution", "label_rs_distribution", "blind_event_distribution", "label_event_distribution", "mismatch_counts", "issue_reasons"):
                flat[key] = json.dumps(flat.get(key), ensure_ascii=False, sort_keys=True)
            writer.writerow({field: flat.get(field) for field in fields})

    manual_fields = [
        "scenario",
        "town",
        "route_id",
        "frame_count",
        "first_blind_sheet",
        "manual_rs_spans",
        "manual_event_spans",
        "manual_notes",
    ]
    # This template intentionally excludes current labels. Fill it while viewing
    # blind sheets, then use the audit/detail JSON only after the blind answer is
    # written down.
    with (output_dir / "manual_blind_answer_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manual_fields)
        writer.writeheader()
        for audit in audits:
            if audit.status != "success":
                continue
            writer.writerow(
                {
                    "scenario": audit.scenario,
                    "town": audit.town,
                    "route_id": audit.route_id,
                    "frame_count": audit.frame_count,
                    "first_blind_sheet": audit.first_blind_sheet,
                    "manual_rs_spans": "",
                    "manual_event_spans": "",
                    "manual_notes": "",
                }
            )

    by_scenario: dict[str, list[RouteAudit]] = defaultdict(list)
    for audit in audits:
        by_scenario[audit.scenario].append(audit)
    summary: list[dict[str, Any]] = []
    for scenario, rows in sorted(by_scenario.items()):
        mismatch = Counter()
        status = Counter(row.status for row in rows)
        for row in rows:
            mismatch.update(row.mismatch_counts)
        summary.append(
            {
                "scenario": scenario,
                "routes": len(rows),
                "frames": sum(row.frame_count for row in rows),
                "status_counts": dict(sorted(status.items())),
                "mismatch_counts": dict(sorted(mismatch.items())),
                "top_issue_reasons": [key for key, _ in mismatch.most_common(8)],
            }
        )
    (output_dir / "scenario_blind_rs_event_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "scenario_blind_rs_event_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields2 = ["scenario", "routes", "frames", "status_counts", "mismatch_counts", "top_issue_reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields2)
        writer.writeheader()
        for row in summary:
            flat = dict(row)
            for key in ("status_counts", "mismatch_counts", "top_issue_reasons"):
                flat[key] = json.dumps(flat.get(key), ensure_ascii=False, sort_keys=True)
            writer.writerow(flat)


def run(args: argparse.Namespace) -> None:
    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    scenarios = _parse_csv(args.scenarios)
    routes = _iter_routes(data_root, scenarios)
    routes = _apply_route_sampling(
        routes,
        max_routes_per_scenario=args.max_routes_per_scenario,
        samples_per_town=args.samples_per_town,
    )
    audits: list[RouteAudit] = []
    details: list[dict[str, Any]] = []
    start = time.time()
    print(f"[blind-audit] routes={len(routes)} workers={args.workers} output={output_dir}", flush=True)
    tasks = [(scenario, route_id, str(route_dir)) for scenario, route_id, route_dir in routes]

    def _record(done: int, total: int, last: str) -> None:
        elapsed = time.time() - start
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(f"[blind-audit] {done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s last={last}", flush=True)

    if args.workers <= 1:
        collector = ScenarioCollector(lead_data_root=str(data_root), output_dir=str(output_dir / "_label_cache"))
        for idx, (scenario, route_id, route_dir) in enumerate(routes, 1):
            audit, _examples, detail = audit_route(
                collector,
                scenario,
                route_id,
                route_dir,
                output_dir,
                write_sheets=args.write_sheets,
                sheet_limit=args.sheet_limit,
            )
            audits.append(audit)
            if detail:
                details.append(detail)
            if idx == len(routes) or idx % max(1, args.progress_interval) == 0:
                _record(idx, len(routes), f"{scenario}/{route_id}")
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(data_root), str(output_dir), args.write_sheets, args.sheet_limit),
        ) as pool:
            future_to_task = {pool.submit(_audit_route_worker, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), 1):
                scenario, route_id, _route_dir = future_to_task[future]
                audit_dict, _examples, detail = future.result()
                audits.append(RouteAudit(**audit_dict))
                if detail:
                    details.append(detail)
                if idx == len(tasks) or idx % max(1, args.progress_interval) == 0:
                    _record(idx, len(tasks), f"{scenario}/{route_id}")
    audits.sort(key=lambda row: (row.scenario, row.route_id))
    details.sort(key=lambda row: (row.get("scenario", ""), row.get("route_id", "")))
    _write_outputs(output_dir, audits, details)
    print(f"[blind-audit] wrote {output_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(AUTOMOT_ROOT / "lead_data"))
    parser.add_argument("--output-dir", default="/tmp/automot_rgb_blind_rs_event_audit")
    parser.add_argument("--scenarios", default=None, help="Comma/space separated scenario subset.")
    parser.add_argument("--max-routes-per-scenario", type=int, default=0)
    parser.add_argument("--samples-per-town", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--write-sheets", action="store_true")
    parser.add_argument("--sheet-limit", type=int, default=80)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
