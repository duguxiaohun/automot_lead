#!/usr/bin/env python3
"""Full RGB audit for R4/R5 scene-level candidate pruning.

The script reads every RGB frame under LEAD routes, computes conservative visual
evidence for signalized junctions and non-signalized junctions, and writes
scenario/route summaries plus compact contact sheets for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import os
import pathlib
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "AutoMoT") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "AutoMoT"))

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402

VIEW_W = 384
VIEW_H = 384
RGB_FRAME_GLOB = "*.jpg"


@dataclass
class FrameVisualEvidence:
    frame_id: int
    has_signal: bool = False
    has_possible_signal_color: bool = False
    has_confirming_signal_source: bool = False
    has_red_signal: bool = False
    has_yellow_signal: bool = False
    has_green_signal: bool = False
    has_junction: bool = False
    has_confirming_junction_source: bool = False
    has_stopline_or_crosswalk: bool = False
    has_turn_marking: bool = False
    signal_score: float = 0.0
    junction_score: float = 0.0
    rgb_path: str = ""
    confirm_source: str = ""


@dataclass
class RouteRgbAudit:
    scenario: str
    route_id: str
    rgb_dir: str
    total_frames: int
    analyzed_frames: int
    skipped_abnormal: bool = False
    abnormal_reason: str = ""
    signal_frames: int = 0
    possible_signal_color_frames: int = 0
    red_signal_frames: int = 0
    yellow_signal_frames: int = 0
    green_signal_frames: int = 0
    junction_frames: int = 0
    possible_junction_frames: int = 0
    nonsignal_junction_frames: int = 0
    stopline_crosswalk_frames: int = 0
    turn_marking_frames: int = 0
    max_signal_score: float = 0.0
    max_junction_score: float = 0.0
    route_class: str = "none"
    signal_examples: list[dict[str, Any]] = field(default_factory=list)
    nonsignal_examples: list[dict[str, Any]] = field(default_factory=list)
    junction_examples: list[dict[str, Any]] = field(default_factory=list)


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


def _load_pickle_file(file_path: pathlib.Path) -> Any:
    try:
        with file_path.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError):
        with lzma.open(file_path, "rb") as f:
            return pickle.load(f)


def _iter_routes(data_root: pathlib.Path, scenarios: set[str] | None) -> list[tuple[str, str, pathlib.Path]]:
    routes: list[tuple[str, str, pathlib.Path]] = []
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            rgb_dir = route_dir / "rgb"
            if rgb_dir.is_dir():
                routes.append((scenario, route_dir.name, route_dir))
    return routes


def _count_light_components(
    mask: np.ndarray,
    hsv_roi: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    max_w: int,
    max_h: int,
    require_dark_context: bool = True,
) -> tuple[int, float, list[tuple[float, float]]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    count = 0
    score = 0.0
    centers: list[tuple[float, float]] = []
    roi_h, roi_w = mask.shape[:2]
    value = hsv_roi[:, :, 2]
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        if w > max_w or h > max_h:
            continue
        if h < 2 or w < 2:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 2.8:
            continue
        if y + h / 2.0 > roi_h * 0.82:
            continue
        if require_dark_context:
            pad = 9
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(roi_w, x + w + pad), min(roi_h, y + h + pad)
            context = value[y0:y1, x0:x1]
            component = labels[y0:y1, x0:x1] == idx
            ring = context[~component]
            if ring.size == 0:
                continue
            dark_fraction = float((ring < 85).sum()) / float(ring.size)
            if dark_fraction < 0.10:
                continue
        count += 1
        score += min(1.0, area / max(float(min_area), 1.0))
        centers.append((x + w / 2.0, y + h / 2.0))
    return count, score, centers


def _traffic_light_evidence(img_rgb: np.ndarray) -> tuple[bool, bool, bool, bool, float]:
    """Detect compact bright red/yellow/green blobs in the upper road scene."""

    h, w = img_rgb.shape[:2]
    roi = img_rgb[: int(h * 0.58), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Keep compact, saturated lights; reject large signs, lamps, lane paint, and brake lights.
    red1 = cv2.inRange(hsv, np.array([0, 105, 105]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 105, 105]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    yellow = cv2.inRange(hsv, np.array([18, 95, 130]), np.array([36, 255, 255]))
    green = cv2.inRange(hsv, np.array([45, 90, 105]), np.array([88, 255, 255]))

    kernel = np.ones((2, 2), dtype=np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)

    red_count, red_score, red_centers = _count_light_components(red, hsv, min_area=4, max_area=75, max_w=18, max_h=22)
    green_count, green_score, green_centers = _count_light_components(green, hsv, min_area=4, max_area=85, max_w=20, max_h=24)
    yellow_count, yellow_score, yellow_centers = _count_light_components(
        yellow,
        hsv,
        min_area=4,
        max_area=80,
        max_w=18,
        max_h=22,
        require_dark_context=True,
    )
    primary_centers = red_centers + green_centers
    if yellow_centers and primary_centers:
        yellow_count = sum(
            1
            for x, y in yellow_centers
            if any((x - px) ** 2 + (y - py) ** 2 <= 32.0**2 for px, py in primary_centers)
        )
        yellow_score = min(float(yellow_count), yellow_score)
    else:
        yellow_count = 0
        yellow_score = 0.0

    has_red = red_count >= 1
    has_green = green_count >= 1
    has_yellow = yellow_count >= 1
    score = red_score + yellow_score + green_score
    return has_red or has_green, has_red, has_yellow, has_green, float(score)


def _front_view(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if w >= VIEW_W * 3 - 4:
        return img_rgb[:, VIEW_W : VIEW_W * 2]
    return img_rgb


def _lane_marking_evidence(front_rgb: np.ndarray) -> tuple[bool, bool, float]:
    """Look for visible stopline/crosswalk bands and turn arrows in the road area."""

    h, w = front_rgb.shape[:2]
    road = front_rgb[int(h * 0.42) :, :]
    gray = cv2.cvtColor(road, cv2.COLOR_RGB2GRAY)
    bright = cv2.inRange(gray, 185, 255)
    sat = cv2.cvtColor(road, cv2.COLOR_RGB2HSV)[:, :, 1]
    bright = cv2.bitwise_and(bright, cv2.inRange(sat, 0, 95))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    horizontal = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (34, 3)))
    rows = (horizontal > 0).sum(axis=1)
    long_rows = int((rows > w * 0.18).sum())
    zebra_rows = int((rows > w * 0.08).sum())
    stopline_or_crosswalk = long_rows >= 2 or zebra_rows >= 9

    # Turn arrows are usually compact white markings near the lower center.
    lower = bright[int(bright.shape[0] * 0.30) :, int(w * 0.22) : int(w * 0.78)]
    contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turn_like = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 45 or area > 1800:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if 8 <= bw <= 95 and 8 <= bh <= 105 and 0.25 <= bw / max(bh, 1) <= 4.0:
            turn_like += 1
    has_turn_marking = turn_like >= 1
    score = min(1.0, long_rows / 12.0) + min(1.0, zebra_rows / 24.0) + min(1.0, turn_like / 2.0)
    return stopline_or_crosswalk, has_turn_marking, float(score)


def _junction_visual_evidence(img_rgb: np.ndarray) -> tuple[bool, bool, bool, float]:
    front = _front_view(img_rgb)
    stopline_or_crosswalk, turn_marking, marking_score = _lane_marking_evidence(front)
    # Turn arrows are useful visual hints, but ordinary curved roads and lane
    # arrows create too many false positives. They are reported but do not
    # auto-confirm a junction by themselves.
    has_junction = stopline_or_crosswalk
    return has_junction, stopline_or_crosswalk, turn_marking, marking_score


def _auxiliary_frame_context(route_dir: pathlib.Path, frame_id: int, need_bbox: bool) -> dict[str, Any]:
    context: dict[str, Any] = {
        "valid_tl_meta": False,
        "meta_junction": False,
        "meta_stop": False,
        "bbox_traffic_light": False,
        "bbox_stop_sign": False,
        "sources": [],
    }
    meta_path = route_dir / "metas" / f"{frame_id:04d}.pkl"
    if meta_path.exists():
        try:
            meta = _load_pickle_file(meta_path)
            tl_state = str(meta.get("traffic_light_state", "None"))
            context["valid_tl_meta"] = tl_state not in {"None", "NONE", "Unknown", "unknown", ""}
            context["meta_junction"] = bool(meta.get("is_junction")) or float(meta.get("dist_to_junction", 9999.0) or 9999.0) < 18.0
            context["meta_stop"] = bool(meta.get("stop_sign_hazard")) or bool(meta.get("stop_sign_close"))
            if context["valid_tl_meta"]:
                context["sources"].append(f"meta_tl={tl_state}")
            if context["meta_junction"]:
                context["sources"].append("meta_junction")
            if context["meta_stop"]:
                context["sources"].append("meta_stop")
        except Exception:
            pass
    if need_bbox:
        bbox_path = route_dir / "bboxes" / f"{frame_id:04d}.pkl"
        if bbox_path.exists():
            try:
                boxes = _load_pickle_file(bbox_path)
                for box in boxes if isinstance(boxes, list) else []:
                    cls = str(box.get("class", "")).lower()
                    if "traffic_light" in cls or "traffic light" in cls:
                        context["bbox_traffic_light"] = True
                    if "stop_sign" in cls or "stop sign" in cls:
                        context["bbox_stop_sign"] = True
                if context["bbox_traffic_light"]:
                    context["sources"].append("bbox_traffic_light")
                if context["bbox_stop_sign"]:
                    context["sources"].append("bbox_stop_sign")
            except Exception:
                pass
    return context


def _analyze_frame(path: pathlib.Path, route_dir: pathlib.Path) -> FrameVisualEvidence | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    possible_signal, red, yellow, green, signal_score = _traffic_light_evidence(img)
    possible_junction, stopline_crosswalk, turn_marking, junction_score = _junction_visual_evidence(img)
    frame_id = _frame_id(path)
    context = _auxiliary_frame_context(route_dir, frame_id, need_bbox=possible_signal or possible_junction)
    signal = bool(possible_signal and (context["valid_tl_meta"] or context["bbox_traffic_light"]))
    junction = bool(
        possible_junction
        and (
            context["meta_junction"]
            or context["meta_stop"]
            or context["bbox_stop_sign"]
            or context["bbox_traffic_light"]
            or signal
        )
    )
    if signal:
        junction = True
        junction_score = max(junction_score, 0.75)
    return FrameVisualEvidence(
        frame_id=frame_id,
        has_signal=signal,
        has_possible_signal_color=possible_signal,
        has_confirming_signal_source=bool(context["valid_tl_meta"] or context["bbox_traffic_light"]),
        has_red_signal=red,
        has_yellow_signal=yellow,
        has_green_signal=green,
        has_junction=junction,
        has_confirming_junction_source=bool(
            context["meta_junction"] or context["meta_stop"] or context["bbox_stop_sign"] or context["bbox_traffic_light"]
        ),
        has_stopline_or_crosswalk=stopline_crosswalk,
        has_turn_marking=turn_marking,
        signal_score=signal_score,
        junction_score=junction_score,
        rgb_path=str(path),
        confirm_source=",".join(context["sources"]),
    )


def _push_example(examples: list[dict[str, Any]], frame: FrameVisualEvidence, score_name: str, limit: int) -> None:
    item = asdict(frame)
    examples.append(item)
    examples.sort(key=lambda x: (float(x.get(score_name, 0.0)), x.get("frame_id", -1)), reverse=True)
    del examples[limit:]


def _classify_route(audit: RouteRgbAudit) -> str:
    if audit.analyzed_frames <= 0:
        return "empty"
    signal_ratio = audit.signal_frames / audit.analyzed_frames
    junction_ratio = audit.junction_frames / audit.analyzed_frames
    nonsignal_ratio = audit.nonsignal_junction_frames / audit.analyzed_frames
    has_r4 = audit.signal_frames >= 2 or signal_ratio >= 0.006
    has_r5 = audit.nonsignal_junction_frames >= 4 or nonsignal_ratio >= 0.012
    if has_r4 and has_r5:
        return "R4_R5_shared"
    if has_r4:
        return "R4_only"
    if has_r5:
        return "R5_only"
    if junction_ratio >= 0.006 or audit.junction_frames >= 2:
        return "junction_uncertain_control"
    return "no_visible_R4_R5"


def analyze_route(task: tuple[str, str, str, bool, int]) -> dict[str, Any]:
    scenario, route_id, route_dir_str, skip_abnormal, example_limit = task
    route_dir = pathlib.Path(route_dir_str)
    rgb_dir = route_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob(RGB_FRAME_GLOB), key=_frame_id)
    audit = RouteRgbAudit(
        scenario=scenario,
        route_id=route_id,
        rgb_dir=str(rgb_dir),
        total_frames=len(frame_paths),
        analyzed_frames=0,
    )
    if skip_abnormal:
        is_bad, info = is_abnormal_lead_route(route_dir, scenario)
        if is_bad:
            audit.skipped_abnormal = True
            audit.abnormal_reason = str(info.get("reason", "abnormal_duration"))
            audit.route_class = "skipped_abnormal_duration"
            return asdict(audit)

    for path in frame_paths:
        evidence = _analyze_frame(path, route_dir)
        if evidence is None:
            continue
        audit.analyzed_frames += 1
        if evidence.has_possible_signal_color:
            audit.possible_signal_color_frames += 1
        if evidence.has_signal:
            audit.signal_frames += 1
            _push_example(audit.signal_examples, evidence, "signal_score", example_limit)
        if evidence.has_red_signal:
            audit.red_signal_frames += 1
        if evidence.has_yellow_signal:
            audit.yellow_signal_frames += 1
        if evidence.has_green_signal:
            audit.green_signal_frames += 1
        if evidence.has_junction:
            audit.junction_frames += 1
            _push_example(audit.junction_examples, evidence, "junction_score", example_limit)
        elif evidence.has_stopline_or_crosswalk or evidence.has_turn_marking:
            audit.possible_junction_frames += 1
        if evidence.has_junction and not evidence.has_signal:
            audit.nonsignal_junction_frames += 1
            _push_example(audit.nonsignal_examples, evidence, "junction_score", example_limit)
        if evidence.has_stopline_or_crosswalk:
            audit.stopline_crosswalk_frames += 1
        if evidence.has_turn_marking:
            audit.turn_marking_frames += 1
        audit.max_signal_score = max(audit.max_signal_score, evidence.signal_score)
        audit.max_junction_score = max(audit.max_junction_score, evidence.junction_score)

    audit.route_class = _classify_route(audit)
    return asdict(audit)


def _scenario_class(route_counts: dict[str, int], route_total: int) -> str:
    if route_total <= 0:
        return "no_data"
    r4_routes = route_counts.get("R4_only", 0) + route_counts.get("R4_R5_shared", 0)
    r5_routes = route_counts.get("R5_only", 0) + route_counts.get("R4_R5_shared", 0)
    uncertain = route_counts.get("junction_uncertain_control", 0)
    r4_ratio = r4_routes / route_total
    r5_ratio = r5_routes / route_total
    if r4_ratio >= 0.08 and r5_ratio >= 0.08:
        return "R4_R5_shared"
    if r4_ratio >= 0.08 and r5_ratio < 0.03:
        return "R4_only"
    if r5_ratio >= 0.08 and r4_ratio < 0.03:
        return "R5_only"
    if r4_ratio < 0.03 and r5_ratio < 0.03 and uncertain / route_total < 0.05:
        return "no_visible_R4_R5"
    return "mixed_or_uncertain_review"


def _aggregate(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        by_scenario.setdefault(route["scenario"], []).append(route)

    rows: list[dict[str, Any]] = []
    for scenario, items in sorted(by_scenario.items()):
        valid = [x for x in items if not x.get("skipped_abnormal")]
        analyzed_frames = sum(int(x["analyzed_frames"]) for x in valid)
        route_counts: dict[str, int] = {}
        for item in valid:
            route_counts[item["route_class"]] = route_counts.get(item["route_class"], 0) + 1
        signal_frames = sum(int(x["signal_frames"]) for x in valid)
        junction_frames = sum(int(x["junction_frames"]) for x in valid)
        nonsignal_frames = sum(int(x["nonsignal_junction_frames"]) for x in valid)
        rows.append(
            {
                "scenario": scenario,
                "routes_total": len(items),
                "routes_valid": len(valid),
                "routes_skipped_abnormal": len(items) - len(valid),
                "frames_analyzed": analyzed_frames,
                "route_class_counts": route_counts,
                "scenario_rgb_class": _scenario_class(route_counts, len(valid)),
                "signal_frame_ratio": signal_frames / analyzed_frames if analyzed_frames else 0.0,
                "junction_frame_ratio": junction_frames / analyzed_frames if analyzed_frames else 0.0,
                "nonsignal_junction_frame_ratio": nonsignal_frames / analyzed_frames if analyzed_frames else 0.0,
                "signal_routes_ratio": (
                    (route_counts.get("R4_only", 0) + route_counts.get("R4_R5_shared", 0)) / len(valid)
                    if valid
                    else 0.0
                ),
                "nonsignal_routes_ratio": (
                    (route_counts.get("R5_only", 0) + route_counts.get("R4_R5_shared", 0)) / len(valid)
                    if valid
                    else 0.0
                ),
            }
        )
    return rows


def _write_outputs(out_dir: pathlib.Path, routes: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "route_rgb_r4_r5_audit.json").write_text(json.dumps(routes, indent=2), encoding="utf-8")
    (out_dir / "scenario_rgb_r4_r5_summary.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")

    with (out_dir / "scenario_rgb_r4_r5_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario",
            "scenario_rgb_class",
            "routes_total",
            "routes_valid",
            "routes_skipped_abnormal",
            "frames_analyzed",
            "signal_routes_ratio",
            "nonsignal_routes_ratio",
            "signal_frame_ratio",
            "junction_frame_ratio",
            "nonsignal_junction_frame_ratio",
            "route_class_counts",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scenarios:
            out = dict(row)
            out["route_class_counts"] = json.dumps(out["route_class_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)

    with (out_dir / "route_rgb_r4_r5_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario",
            "route_id",
            "route_class",
            "total_frames",
            "analyzed_frames",
            "skipped_abnormal",
            "signal_frames",
            "possible_signal_color_frames",
            "junction_frames",
            "possible_junction_frames",
            "nonsignal_junction_frames",
            "red_signal_frames",
            "yellow_signal_frames",
            "green_signal_frames",
            "stopline_crosswalk_frames",
            "turn_marking_frames",
            "max_signal_score",
            "max_junction_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for route in routes:
            writer.writerow({field: route.get(field) for field in fields})
    _write_contact_sheets(out_dir, routes, scenarios)


def _load_thumb(path: str, label: str, size: tuple[int, int] = (384, 128)) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        thumb = np.full((size[1], size[0], 3), 35, dtype=np.uint8)
    else:
        thumb = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (size[0] - 1, 22), (0, 0, 0), -1)
    cv2.putText(thumb, label[:88], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def _best_examples(routes: list[dict[str, Any]], scenario: str, key: str, limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    score_key = "signal_score" if key == "signal_examples" else "junction_score"
    for route in routes:
        if route.get("scenario") != scenario or route.get("skipped_abnormal"):
            continue
        for example in route.get(key, []) or []:
            pairs.append((route, example))
    pairs.sort(key=lambda pair: (float(pair[1].get(score_key, 0.0)), int(pair[1].get("frame_id", -1))), reverse=True)
    seen: set[str] = set()
    diverse: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for route, example in pairs:
        rid = str(route.get("route_id"))
        if rid in seen and len(diverse) < max(2, limit // 2):
            continue
        diverse.append((route, example))
        seen.add(rid)
        if len(diverse) >= limit:
            break
    return diverse


def _write_contact_sheets(out_dir: pathlib.Path, routes: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> None:
    sheet_dir = out_dir / "evidence_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cols = 3
    cell_w, cell_h = 384, 128
    text_h = 42
    for scenario_row in scenarios:
        scenario = scenario_row["scenario"]
        signal = _best_examples(routes, scenario, "signal_examples", 9)
        nonsignal = _best_examples(routes, scenario, "nonsignal_examples", 9)
        junction = _best_examples(routes, scenario, "junction_examples", 6)
        sections = [("R4 signal evidence", signal), ("R5 no-signal junction evidence", nonsignal), ("junction visual evidence", junction)]
        rows: list[np.ndarray] = []
        header = np.full((56, cols * cell_w, 3), 255, dtype=np.uint8)
        title = (
            f"{scenario} | class={scenario_row['scenario_rgb_class']} | "
            f"routes={scenario_row['routes_valid']}/{scenario_row['routes_total']} | "
            f"signal_routes={scenario_row['signal_routes_ratio']:.3f} | "
            f"nonsignal_routes={scenario_row['nonsignal_routes_ratio']:.3f}"
        )
        cv2.putText(header, title[:150], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            header,
            json.dumps(scenario_row["route_class_counts"], sort_keys=True)[:150],
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        rows.append(header)
        for section_title, pairs in sections:
            section_header = np.full((28, cols * cell_w, 3), 238, dtype=np.uint8)
            cv2.putText(section_header, section_title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            rows.append(section_header)
            if not pairs:
                blank = np.full((cell_h + text_h, cols * cell_w, 3), 245, dtype=np.uint8)
                cv2.putText(blank, "no RGB evidence selected", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
                rows.append(blank)
                continue
            for offset in range(0, len(pairs), cols):
                chunk = pairs[offset : offset + cols]
                row = np.full((cell_h + text_h, cols * cell_w, 3), 245, dtype=np.uint8)
                for col, (route, example) in enumerate(chunk):
                    label = (
                        f"{route['route_id']} f={example.get('frame_id')} "
                        f"sig={example.get('signal_score', 0):.2f} junc={example.get('junction_score', 0):.2f}"
                    )
                    thumb = _load_thumb(str(example.get("rgb_path", "")), label, (cell_w, cell_h))
                    x = col * cell_w
                    row[:cell_h, x : x + cell_w] = thumb
                    flags = (
                        f"class={route.get('route_class')} "
                        f"R={int(bool(example.get('has_red_signal')))} "
                        f"Y={int(bool(example.get('has_yellow_signal')))} "
                        f"G={int(bool(example.get('has_green_signal')))} "
                        f"src={str(example.get('confirm_source', ''))[:24]} "
                        f"stop/cw={int(bool(example.get('has_stopline_or_crosswalk')))} "
                        f"turn={int(bool(example.get('has_turn_marking')))}"
                    )
                    cv2.putText(row, flags[:90], (x + 5, cell_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                rows.append(row)
        sheet = np.vstack(rows)
        cv2.imwrite(str(sheet_dir / f"{scenario}.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _progress(done: int, total: int, start: float, last: str = "") -> None:
    elapsed = max(1e-6, time.time() - start)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else math.inf
    print(
        f"[rgb-audit] {done}/{total} ({100.0 * done / max(total, 1):5.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s {last}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    data_root = pathlib.Path(args.data_root)
    out_dir = pathlib.Path(args.output_dir)
    scenarios = _parse_csv(args.scenarios)
    routes = _iter_routes(data_root, scenarios)
    if args.max_routes_per_scenario > 0:
        kept: list[tuple[str, str, pathlib.Path]] = []
        counts: dict[str, int] = {}
        for scenario, route_id, route_dir in routes:
            count = counts.get(scenario, 0)
            if count < args.max_routes_per_scenario:
                kept.append((scenario, route_id, route_dir))
                counts[scenario] = count + 1
        routes = kept

    print(f"[rgb-audit] discovered routes={len(routes)} data_root={data_root}", flush=True)
    tasks = [
        (scenario, route_id, str(route_dir), not args.keep_abnormal_duration, args.example_limit)
        for scenario, route_id, route_dir in routes
    ]
    results: list[dict[str, Any]] = []
    start = time.time()
    _progress(0, len(tasks), start, "starting")
    if args.workers <= 1:
        for idx, task in enumerate(tasks, 1):
            results.append(analyze_route(task))
            if idx == len(tasks) or idx % max(1, args.progress_interval) == 0:
                _progress(idx, len(tasks), start, f"last={task[0]}/{task[1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_task = {pool.submit(analyze_route, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                results.append(future.result())
                if idx == len(tasks) or idx % max(1, args.progress_interval) == 0:
                    _progress(idx, len(tasks), start, f"last={task[0]}/{task[1]}")
    results.sort(key=lambda x: (x["scenario"], x["route_id"]))
    scenario_rows = _aggregate(results)
    _write_outputs(out_dir, results, scenario_rows)
    print(f"[rgb-audit] wrote {out_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "AutoMoT" / "lead_data"))
    parser.add_argument("--output-dir", default="/tmp/automot_rgb_r4_r5_full_audit")
    parser.add_argument("--scenarios", default=None, help="Comma/space separated scenario subset.")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--example-limit", type=int, default=6)
    parser.add_argument("--max-routes-per-scenario", type=int, default=0)
    parser.add_argument("--keep-abnormal-duration", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
