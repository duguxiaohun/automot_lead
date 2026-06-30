#!/usr/bin/env python3
"""Rule-based key-frame filtering for all CARLA scenarios.

Outputs a JSON index with initial/middle/final key frames per run.
Primary signals are metas/*.pkl fields (xz-compressed). Falls back to
bboxes/rgb when a meta field is absent.

Middle frames are named per scenario to reflect meaningful key steps:
  Accident            → hazard_detect, max_brake_or_min_gap, recover_or_pass
  BlockedIntersection → obstacle_detect, wait_stop, bypass_resume
  CrossingBicycleFlow → wait_start, wait_peak, proceed_resume
  HighwayCutIn        → cutin_onset, caution_peak, stabilize_follow
  PedestrianCrossing  → pedestrian_detect, wait_to_cross, proceed_resume
  ConstructionObstacle→ construction_detect, slow_and_navigate, obstacle_clear
  … (see SCENARIO_CONFIG for all 43 scenarios)
"""

from __future__ import annotations

import argparse
import glob
import json
import lzma
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Scenario labels  (used as initial-frame label_text)
# ---------------------------------------------------------------------------
SCENARIO_LABELS: Dict[str, str] = {
    "Accident":                              "Brake and avoid accident hazard",
    "AccidentTwoWays":                       "Brake and avoid head-on accident hazard",
    "BlockedIntersection":                   "Bypass blocked intersection",
    "ConstructionObstacle":                  "Navigate around construction obstacle",
    "ConstructionObstacleTwoWays":           "Navigate construction obstacle on two-way road",
    "ControlLoss":                           "Regain control after vehicle instability",
    "CrossJunctionDefectTrafficLight":       "Cross junction with defective traffic light",
    "CrossingBicycleFlow":                   "Yield to crossing bicycle",
    "DynamicObjectCrossing":                 "Yield to dynamic object crossing",
    "EnterActorFlow":                        "Merge into actor flow",
    "EnterActorFlowV2":                      "Merge into actor flow (variant)",
    "HardBreakRoute":                        "React to lead vehicle hard braking",
    "HazardAtSideLane":                      "Pass hazard at side lane",
    "HazardAtSideLaneTwoWays":               "Pass side-lane hazard on two-way road",
    "HighwayCutIn":                          "Decelerate and prepare for cut-in vehicle",
    "HighwayExit":                           "Navigate highway exit",
    "InterurbanActorFlow":                   "Merge into interurban actor flow",
    "InterurbanAdvancedActorFlow":           "Merge into advanced interurban actor flow",
    "InvadingTurn":                          "React to vehicle invading lane on turn",
    "MergerIntoSlowTraffic":                 "Merge into slow-moving traffic",
    "MergerIntoSlowTrafficV2":               "Merge into slow-moving traffic (variant)",
    "NonSignalizedJunctionLeftTurn":         "Left turn at non-signalized junction",
    "NonSignalizedJunctionLeftTurnEnterFlow":"Left turn and enter flow at non-signalized junction",
    "NonSignalizedJunctionRightTurn":        "Right turn at non-signalized junction",
    "OppositeVehicleRunningRedLight":        "React to opposite vehicle running red light",
    "OppositeVehicleTakingPriority":         "React to opposite vehicle taking priority",
    "ParkedObstacle":                        "Navigate around parked obstacle",
    "ParkedObstacleTwoWays":                 "Navigate around parked obstacle on two-way road",
    "ParkingCrossingPedestrian":             "Yield to pedestrian near parking area",
    "ParkingCutIn":                          "React to vehicle cutting in from parking",
    "ParkingExit":                           "Yield to exiting parking vehicle",
    "PedestrianCrossing":                    "Yield to crossing pedestrian",
    "PriorityAtJunction":                    "Assert and navigate junction priority",
    "RedLightWithoutLeadVehicle":            "Stop at red light without lead vehicle",
    "SignalizedJunctionLeftTurn":            "Left turn at signalized junction",
    "SignalizedJunctionLeftTurnEnterFlow":   "Left turn and enter flow at signalized junction",
    "SignalizedJunctionRightTurn":           "Right turn at signalized junction",
    "StaticCutIn":                           "React to static vehicle cutting in",
    "T_Junction":                            "Navigate T-junction",
    "VehicleOpensDoorTwoWays":               "React to vehicle door opening on two-way road",
    "VehicleTurningRoute":                   "React to vehicle turning across route",
    "VehicleTurningRoutePedestrian":         "React to turning vehicle with pedestrian present",
}

# ---------------------------------------------------------------------------
# Per-scenario rule configuration
# ---------------------------------------------------------------------------
# Each entry: (dist_meta_field, approach_threshold_m, (event_A, event_B, event_C))
#   dist_meta_field    – primary metas field holding scenario distance signal
#   approach_threshold – distance (m) below which the interaction has started
#   event_A            – first detection / approach key step
#   event_B            – peak moment (closest gap / hardest brake / full stop)
#   event_C            – resolution / resume
# ---------------------------------------------------------------------------
SCENARIO_CONFIG: Dict[str, Tuple[str, float, Tuple[str, str, str]]] = {
    # ── Accident / head-on hazard ─────────────────────────────────────────
    "Accident": (
        "dist_to_accident_site", 30.0,
        ("hazard_detect", "max_brake_or_min_gap", "recover_or_pass"),
    ),
    "AccidentTwoWays": (
        "dist_to_accident_site", 30.0,
        ("hazard_detect", "max_brake_or_min_gap", "recover_or_pass"),
    ),
    # ── Construction obstacle ─────────────────────────────────────────────
    "ConstructionObstacle": (
        "dist_to_construction_site", 35.0,
        ("construction_detect", "slow_and_navigate", "obstacle_clear"),
    ),
    "ConstructionObstacleTwoWays": (
        "dist_to_construction_site", 35.0,
        ("construction_detect", "slow_and_navigate", "obstacle_clear"),
    ),
    # ── Parked obstacle / side hazard ─────────────────────────────────────
    "ParkedObstacle": (
        "dist_to_parked_obstacle", 25.0,
        ("obstacle_detect", "decelerate_around", "clear_obstacle"),
    ),
    "ParkedObstacleTwoWays": (
        "dist_to_parked_obstacle", 25.0,
        ("obstacle_detect", "decelerate_around", "clear_obstacle"),
    ),
    "HazardAtSideLane": (
        "dist_to_parked_obstacle", 30.0,
        ("side_hazard_detect", "passing_hazard", "clear_hazard"),
    ),
    "HazardAtSideLaneTwoWays": (
        "dist_to_parked_obstacle", 30.0,
        ("side_hazard_detect", "passing_hazard", "clear_hazard"),
    ),
    "ParkingExit": (
        "dist_to_parked_obstacle", 20.0,
        ("parking_exit_detect", "brake_for_exit", "exit_clear"),
    ),
    # ── Cut-in / merge ────────────────────────────────────────────────────
    "HighwayCutIn": (
        "dist_to_cutin_vehicle", 35.0,
        ("cutin_onset", "caution_peak", "stabilize_follow"),
    ),
    "StaticCutIn": (
        "dist_to_cutin_vehicle", 30.0,
        ("cutin_onset", "caution_peak", "stabilize_follow"),
    ),
    "ParkingCutIn": (
        "dist_to_cutin_vehicle", 25.0,
        ("cutin_onset", "caution_peak", "stabilize_follow"),
    ),
    "MergerIntoSlowTraffic": (
        "dist_to_cutin_vehicle", 30.0,
        ("slow_traffic_detect", "match_speed", "merge_complete"),
    ),
    "MergerIntoSlowTrafficV2": (
        "dist_to_cutin_vehicle", 30.0,
        ("slow_traffic_detect", "match_speed", "merge_complete"),
    ),
    "InvadingTurn": (
        "dist_to_cutin_vehicle", 30.0,
        ("invading_vehicle_detect", "evasive_decelerate", "lane_clear"),
    ),
    # ── Pedestrian / bicycle ──────────────────────────────────────────────
    "CrossingBicycleFlow": (
        "dist_to_biker", 20.0,
        ("wait_start", "wait_peak", "proceed_resume"),
    ),
    "PedestrianCrossing": (
        "dist_to_pedestrian", 20.0,
        ("pedestrian_detect", "wait_to_cross", "proceed_resume"),
    ),
    "ParkingCrossingPedestrian": (
        "dist_to_pedestrian", 20.0,
        ("pedestrian_detect", "wait_to_cross", "proceed_resume"),
    ),
    "DynamicObjectCrossing": (
        "dist_to_pedestrian", 20.0,
        ("object_detect", "wait_or_slow", "proceed_resume"),
    ),
    "VehicleTurningRoutePedestrian": (
        "dist_to_pedestrian", 20.0,
        ("pedestrian_detect", "yield_and_slow", "proceed_resume"),
    ),
    # ── Vehicle door ──────────────────────────────────────────────────────
    "VehicleOpensDoorTwoWays": (
        "dist_to_vehicle_opens_door", 25.0,
        ("door_open_detect", "avoid_door", "clear_hazard"),
    ),
    # ── Blocked intersection ──────────────────────────────────────────────
    "BlockedIntersection": (
        "dist_to_junction", 25.0,
        ("obstacle_detect", "wait_stop", "bypass_resume"),
    ),
    # ── Traffic-light / signalized junctions ─────────────────────────────
    "CrossJunctionDefectTrafficLight": (
        "dist_to_junction", 25.0,
        ("junction_approach", "check_and_proceed", "junction_clear"),
    ),
    "RedLightWithoutLeadVehicle": (
        "dist_to_junction", 25.0,
        ("junction_approach", "brake_at_light", "proceed_on_green"),
    ),
    "SignalizedJunctionLeftTurn": (
        "dist_to_junction", 30.0,
        ("junction_approach", "wait_or_turn_on_green", "turn_complete"),
    ),
    "SignalizedJunctionLeftTurnEnterFlow": (
        "dist_to_junction", 30.0,
        ("junction_approach", "wait_and_enter_flow", "flow_established"),
    ),
    "SignalizedJunctionRightTurn": (
        "dist_to_junction", 30.0,
        ("junction_approach", "turn_on_green", "turn_complete"),
    ),
    # ── Non-signalized junctions ──────────────────────────────────────────
    "NonSignalizedJunctionLeftTurn": (
        "dist_to_junction", 30.0,
        ("junction_approach", "yield_and_turn", "turn_complete"),
    ),
    "NonSignalizedJunctionLeftTurnEnterFlow": (
        "dist_to_junction", 30.0,
        ("junction_approach", "yield_and_enter_flow", "flow_established"),
    ),
    "NonSignalizedJunctionRightTurn": (
        "dist_to_junction", 30.0,
        ("junction_approach", "yield_and_turn", "turn_complete"),
    ),
    "PriorityAtJunction": (
        "dist_to_junction", 30.0,
        ("junction_approach", "assert_priority", "junction_clear"),
    ),
    "T_Junction": (
        "dist_to_junction", 30.0,
        ("junction_approach", "yield_and_turn", "turn_complete"),
    ),
    "HighwayExit": (
        "dist_to_junction", 40.0,
        ("exit_approach", "decelerate_and_diverge", "exit_complete"),
    ),
    # ── Actor flows ───────────────────────────────────────────────────────
    "EnterActorFlow": (
        "dist_to_junction", 35.0,
        ("flow_approach", "gap_accept_merge", "flow_established"),
    ),
    "EnterActorFlowV2": (
        "dist_to_junction", 35.0,
        ("flow_approach", "gap_accept_merge", "flow_established"),
    ),
    "InterurbanActorFlow": (
        "dist_to_junction", 40.0,
        ("flow_approach", "gap_accept_merge", "flow_established"),
    ),
    "InterurbanAdvancedActorFlow": (
        "dist_to_junction", 40.0,
        ("flow_approach", "gap_accept_merge", "flow_established"),
    ),
    # ── Opposite vehicle ─────────────────────────────────────────────────
    "OppositeVehicleRunningRedLight": (
        "dist_to_junction", 30.0,
        ("threat_detect", "evasive_action", "threat_clear"),
    ),
    "OppositeVehicleTakingPriority": (
        "dist_to_junction", 30.0,
        ("threat_detect", "evasive_action", "threat_clear"),
    ),
    # ── Speed / brake response (brake/accel is the primary mid-event signal)
    "HardBreakRoute": (
        "dist_to_junction", 50.0,   # junction used only as rough route marker
        ("follow_normal", "hard_brake_response", "recover_speed"),
    ),
    "ControlLoss": (
        "dist_to_junction", 50.0,
        ("normal_driving", "loss_of_control", "regain_control"),
    ),
    # ── Turning vehicle across route ──────────────────────────────────────
    "VehicleTurningRoute": (
        "dist_to_junction", 30.0,
        ("turning_vehicle_detect", "yield_or_slow", "route_clear"),
    ),
}

# Scenarios where brake/accel is the primary mid-event signal.
BRAKE_ACCEL_PRIMARY: set = {"HardBreakRoute", "ControlLoss"}

# Directories inside dataset root that are NOT scenario folders.
SKIPPED_DIRS: set = {"noScenarios", "verification_tool"}

# All distance meta fields to extract from metas/*.pkl.
_ALL_DIST_FIELDS = [
    "dist_to_accident_site",
    "dist_to_construction_site",
    "dist_to_parked_obstacle",
    "dist_to_vehicle_opens_door",
    "dist_to_cutin_vehicle",
    "dist_to_pedestrian",
    "dist_to_biker",
    "dist_to_junction",
]


@dataclass
class RunContext:
    scenario: str
    run_id: str
    run_path: str
    total_frames: int
    duration_game: float
    seconds_per_frame: float
    status: str
    num_infractions: int
    route_id: str
    final_success: bool
    infractions: Dict[str, Any]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        return float(v)
    except Exception:
        return default


def load_pickle(path: str) -> Any:
    # Dataset pickles are xz-compressed.
    try:
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def list_sorted(glob_pattern: str) -> List[str]:
    return sorted(glob.glob(glob_pattern))


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_frame_count(run_path: str) -> int:
    rgb = list_sorted(os.path.join(run_path, "rgb", "*.jpg"))
    if rgb:
        return len(rgb)
    metas = list_sorted(os.path.join(run_path, "metas", "*.pkl"))
    return len(metas)


def build_run_context(scenario: str, run_path: str) -> Optional[RunContext]:
    results_path = os.path.join(run_path, "results.json")
    if not os.path.exists(results_path):
        return None
    results = read_json(results_path)
    total_frames = get_frame_count(run_path)
    if total_frames <= 0:
        return None

    meta = results.get("meta", {})
    duration_game = safe_float(meta.get("duration_game"), default=0.0)
    if duration_game <= 0:
        duration_game = max(total_frames - 1, 1) * 0.25

    seconds_per_frame = duration_game / max(total_frames - 1, 1)
    infractions = results.get("infractions", {})
    status = str(results.get("status", "Unknown"))
    final_success = status in {"Perfect", "Completed"} and not (
        infractions.get("route_timeout")
        or infractions.get("vehicle_blocked")
        or infractions.get("scenario_timeouts")
    )

    return RunContext(
        scenario=scenario,
        run_id=os.path.basename(run_path.rstrip("/")),
        run_path=run_path,
        total_frames=total_frames,
        duration_game=duration_game,
        seconds_per_frame=seconds_per_frame,
        status=status,
        num_infractions=int(results.get("num_infractions", 0)),
        route_id=str(results.get("route_id", "")),
        final_success=bool(final_success),
        infractions=infractions,
    )


# ---------------------------------------------------------------------------
# Meta series extraction
# ---------------------------------------------------------------------------

def extract_meta_series(run_path: str, total_frames: int) -> Dict[str, List[float]]:
    meta_files = list_sorted(os.path.join(run_path, "metas", "*.pkl"))
    n = min(len(meta_files), total_frames)

    series: Dict[str, List[float]] = {
        "speed":   [math.nan] * total_frames,
        "accel_x": [math.nan] * total_frames,
        "brake":   [math.nan] * total_frames,
        "throttle":[math.nan] * total_frames,
    }
    for f in _ALL_DIST_FIELDS:
        series[f] = [math.nan] * total_frames

    for i in range(n):
        m = load_pickle(meta_files[i])
        series["speed"][i]    = safe_float(m.get("speed"))
        series["accel_x"][i]  = safe_float(m.get("accel_x"))
        series["brake"][i]    = safe_float(m.get("brake"))
        series["throttle"][i] = safe_float(m.get("throttle"))
        for f in _ALL_DIST_FIELDS:
            series[f][i] = safe_float(m.get(f))

    return series


def series_has_signal(values: Sequence[float]) -> bool:
    return any(not math.isnan(v) and abs(v) < 1e8 for v in values)


# ---------------------------------------------------------------------------
# Bbox fallback signals
# ---------------------------------------------------------------------------

def nearest_vehicle_distance_from_bboxes(run_path: str, total_frames: int) -> List[float]:
    bbox_files = list_sorted(os.path.join(run_path, "bboxes", "*.pkl"))
    out = [math.nan] * total_frames
    n = min(total_frames, len(bbox_files))
    for i in range(n):
        arr = load_pickle(bbox_files[i])
        best = math.nan
        for obj in arr if isinstance(arr, list) else []:
            if obj.get("class") in {"car", "static_prop_car"}:
                d = safe_float(obj.get("distance"))
                if math.isnan(d):
                    pos = obj.get("position")
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        d = math.hypot(safe_float(pos[0], 0.0), safe_float(pos[1], 0.0))
                if not math.isnan(d) and d > 0:
                    if math.isnan(best) or d < best:
                        best = d
        out[i] = best
    return out


def nearest_pedestrian_from_bboxes(run_path: str, total_frames: int) -> List[float]:
    bbox_files = list_sorted(os.path.join(run_path, "bboxes", "*.pkl"))
    out = [math.nan] * total_frames
    n = min(total_frames, len(bbox_files))
    for i in range(n):
        arr = load_pickle(bbox_files[i])
        best = math.nan
        for obj in arr if isinstance(arr, list) else []:
            cls = obj.get("class", "")
            if "pedestrian" not in cls and "walker" not in cls:
                continue
            d = safe_float(obj.get("distance"))
            if math.isnan(d):
                pos = obj.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    d = math.hypot(safe_float(pos[0], 0.0), safe_float(pos[1], 0.0))
            if not math.isnan(d):
                if math.isnan(best) or d < best:
                    best = d
        out[i] = best
    return out


def bicycle_like_distance_from_bboxes(run_path: str, total_frames: int) -> List[float]:
    bbox_files = list_sorted(os.path.join(run_path, "bboxes", "*.pkl"))
    out = [math.nan] * total_frames
    n = min(total_frames, len(bbox_files))
    for i in range(n):
        arr = load_pickle(bbox_files[i])
        best = math.nan
        for obj in arr if isinstance(arr, list) else []:
            if obj.get("class") != "car":
                continue
            ext = obj.get("extent")
            if not isinstance(ext, (list, tuple)) or len(ext) < 2:
                continue
            if safe_float(ext[0], 999.0) > 1.3 or safe_float(ext[1], 999.0) > 0.6:
                continue
            d = safe_float(obj.get("distance"))
            if math.isnan(d):
                pos = obj.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    d = math.hypot(safe_float(pos[0], 0.0), safe_float(pos[1], 0.0))
            if not math.isnan(d):
                if math.isnan(best) or d < best:
                    best = d
        out[i] = best
    return out


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------

def smooth(values: Sequence[float], window: int = 3) -> List[float]:
    n = len(values)
    out = [math.nan] * n
    half = max(1, window // 2)
    for i in range(n):
        xs = [values[j] for j in range(max(0, i - half), min(n, i + half + 1))
              if not math.isnan(values[j])]
        out[i] = sum(xs) / len(xs) if xs else math.nan
    return out


def first_where(start: int, end: int, predicate, min_consecutive: int = 1) -> Optional[int]:
    count = 0
    begin = None
    for i in range(max(0, start), max(0, end)):
        if predicate(i):
            if begin is None:
                begin = i
            count += 1
            if count >= min_consecutive:
                return begin
        else:
            count = 0
            begin = None
    return None


def argmin_valid(values: Sequence[float], start: int = 0, end: Optional[int] = None) -> Optional[int]:
    if end is None:
        end = len(values)
    best_i, best_v = None, None
    for i in range(max(0, start), min(len(values), end)):
        v = values[i]
        if not math.isnan(v) and (best_v is None or v < best_v):
            best_v = v
            best_i = i
    return best_i


def argmax_valid(values: Sequence[float], start: int = 0, end: Optional[int] = None) -> Optional[int]:
    if end is None:
        end = len(values)
    best_i, best_v = None, None
    for i in range(max(0, start), min(len(values), end)):
        v = values[i]
        if not math.isnan(v) and (best_v is None or v > best_v):
            best_v = v
            best_i = i
    return best_i


def enforce_event_order(
    initial: int,
    mids: List[Tuple[str, int, float]],
    final: int,
    min_gap: int = 2,
) -> List[Tuple[str, int, float]]:
    available = max(1, final - initial)
    adaptive_gap = max(1, min(min_gap, available // max(1, len(mids) + 1)))
    ordered: List[Tuple[str, int, float]] = []
    last = initial
    total = len(mids)
    for pos, (name, idx, conf) in enumerate(mids):
        remaining = total - pos
        max_allowed = final - adaptive_gap * remaining
        idx = max(idx, last + adaptive_gap)
        idx = min(idx, max(initial + adaptive_gap, max_allowed))
        if ordered and idx <= ordered[-1][1]:
            idx = min(max_allowed, ordered[-1][1] + adaptive_gap)
        ordered.append((name, idx, conf))
        last = idx
    return ordered


def frame_to_time(frame: int, spf: float) -> float:
    return round(frame * spf, 4)


def rgb_motion_peak_indices(run_path: str, total_frames: int) -> Tuple[int, int, int]:
    rgb_files = list_sorted(os.path.join(run_path, "rgb", "*.jpg"))[:total_frames]
    if len(rgb_files) < 6:
        q1 = max(1, total_frames // 4)
        q2 = max(q1 + 1, total_frames // 2)
        q3 = max(q2 + 1, (3 * total_frames) // 4)
        return q1, q2, q3
    sizes = [os.path.getsize(p) for p in rgb_files]
    diffs = [0.0] + [abs(float(sizes[i] - sizes[i - 1])) for i in range(1, len(sizes))]
    half = len(diffs) // 2
    a = max(range(1, max(2, half)), key=lambda i: diffs[i])
    b = max(range(max(1, half), len(diffs)), key=lambda i: diffs[i])
    c = min(len(diffs) - 2, max(a + 2, b + 2))
    return a, b, c


# ---------------------------------------------------------------------------
# Generic 3-event extractor  (distance-driven)
# ---------------------------------------------------------------------------

def _pick_distance_events(
    dist: List[float],
    speed: List[float],
    accel: List[float],
    brake: List[float],
    n: int,
    approach_thresh: float,
    event_names: Tuple[str, str, str],
    conf_base: float,
) -> List[Tuple[str, int, float]]:
    name_a, name_b, name_c = event_names
    i_min = argmin_valid(dist) or max(1, n // 2)

    # A: first frame below threshold where deceleration begins
    detect = first_where(
        0, i_min + 1,
        lambda i: (
            not math.isnan(dist[i]) and dist[i] < approach_thresh
            and (
                (not math.isnan(accel[i]) and accel[i] < -0.4)
                or (not math.isnan(brake[i]) and brake[i] > 0.05)
            )
        ),
        min_consecutive=1,
    )
    if detect is None:
        detect = max(1, i_min - max(4, n // 8))

    # B: minimum speed (or closest approach) near the event peak
    peak = argmin_valid(speed, start=detect, end=min(n, i_min + n // 6))
    if peak is None:
        peak = i_min

    # C: first sustained acceleration / speed recovery after peak
    recover = first_where(
        peak + 1, n,
        lambda i: (
            not math.isnan(speed[i]) and speed[i] > 2.0
            and not math.isnan(accel[i]) and accel[i] > 0.1
        ),
        min_consecutive=2,
    )
    if recover is None:
        recover = min(n - 2, peak + max(4, n // 8))

    return [
        (name_a, detect,  conf_base),
        (name_b, peak,    conf_base),
        (name_c, recover, conf_base - 0.04),
    ]


# ---------------------------------------------------------------------------
# Brake/accel-primary event extractor  (HardBreakRoute, ControlLoss)
# ---------------------------------------------------------------------------

def _pick_brake_accel_events(
    speed: List[float],
    accel: List[float],
    brake: List[float],
    n: int,
    event_names: Tuple[str, str, str],
    conf_base: float,
) -> List[Tuple[str, int, float]]:
    name_a, name_b, name_c = event_names

    # A: normal driving — first fifth of the sequence
    normal = max(1, n // 5)

    # B: peak braking (most negative acceleration or highest brake pedal)
    peak = argmin_valid(accel, start=normal, end=n)
    if peak is None:
        peak = argmax_valid(brake, start=normal, end=n) or max(normal + 1, n // 2)

    # C: first sustained speed recovery
    recover = first_where(
        peak + 1, n,
        lambda i: not math.isnan(speed[i]) and speed[i] > 3.0,
        min_consecutive=2,
    )
    if recover is None:
        recover = min(n - 2, peak + max(4, n // 7))

    return [
        (name_a, normal,  conf_base - 0.05),
        (name_b, peak,    conf_base),
        (name_c, recover, conf_base - 0.05),
    ]


# ---------------------------------------------------------------------------
# CrossingBicycleFlow specialised rule  (finer bicycle-wait detection)
# ---------------------------------------------------------------------------

def _pick_bicycle_flow_events(
    dist: List[float],
    speed: List[float],
    accel: List[float],
    brake: List[float],
    n: int,
    conf_base: float,
) -> List[Tuple[str, int, float]]:
    i_min = argmin_valid(dist) or max(1, n // 2)
    wait_start = first_where(
        0, i_min + 1,
        lambda i: (
            not math.isnan(dist[i]) and dist[i] < 20.0
            and (
                (not math.isnan(accel[i]) and accel[i] < -0.6)
                or (not math.isnan(brake[i]) and brake[i] > 0.1)
            )
        ),
        min_consecutive=2,
    )
    if wait_start is None:
        wait_start = max(1, i_min - 8)
    wait_peak = argmin_valid(speed, start=wait_start, end=min(n, i_min + 16)) or i_min
    resume = first_where(
        max(wait_peak + 1, i_min + 1), n,
        lambda i: (
            not math.isnan(speed[i]) and speed[i] > 2.0
            and not math.isnan(accel[i]) and accel[i] > 0.2
            and i >= 2
            and not math.isnan(dist[i]) and not math.isnan(dist[i - 2])
            and dist[i] > dist[i - 2]
        ),
        min_consecutive=2,
    )
    if resume is None:
        resume = min(n - 2, wait_peak + 8)
    return [
        ("wait_start",     wait_start, conf_base),
        ("wait_peak",      wait_peak,  conf_base),
        ("proceed_resume", resume,     conf_base - 0.02),
    ]


# ---------------------------------------------------------------------------
# Cut-in specialised rule  (onset → closest approach → stabilise)
# ---------------------------------------------------------------------------

def _pick_cutin_events(
    dist: List[float],
    speed: List[float],
    n: int,
    thresh: float,
    ev_names: Tuple[str, str, str],
    conf_base: float,
) -> List[Tuple[str, int, float]]:
    ev_a, ev_b, ev_c = ev_names
    caution_peak = argmin_valid(dist) or max(1, n // 2)
    onset = first_where(
        0, caution_peak + 1,
        lambda i: (
            not math.isnan(dist[i]) and dist[i] < thresh
            and i >= 3
            and not math.isnan(dist[i - 3])
            and dist[i] < dist[i - 3]
        ),
        min_consecutive=2,
    )
    if onset is None:
        onset = max(1, caution_peak - 12)
    stabilize = first_where(
        caution_peak + 1, n,
        lambda i: (
            not math.isnan(dist[i]) and dist[i] > (dist[caution_peak] + 5.0)
            and not math.isnan(speed[i]) and speed[i] > 4.0
        ),
        min_consecutive=2,
    )
    if stabilize is None:
        stabilize = min(n - 2, caution_peak + 12)
    return [
        (ev_a, onset,        conf_base),
        (ev_b, caution_peak, conf_base + 0.02),
        (ev_c, stabilize,    conf_base - 0.02),
    ]


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def pick_middle_events(
    ctx: RunContext,
    meta: Dict[str, List[float]],
) -> Tuple[List[Tuple[str, int, float]], str]:
    n = ctx.total_frames
    speed = smooth(meta["speed"],   3)
    accel = smooth(meta["accel_x"], 3)
    brake = smooth(meta["brake"],   3)
    signal_source = "metas"

    # ── CrossingBicycleFlow — bicycle-wait specialised rule ────────────────
    if ctx.scenario == "CrossingBicycleFlow":
        dist = smooth(meta["dist_to_biker"], 5)
        if not series_has_signal(dist):
            dist = smooth(bicycle_like_distance_from_bboxes(ctx.run_path, n), 5)
            signal_source = "bboxes"
        if series_has_signal(dist):
            conf = 0.9 if signal_source == "metas" else 0.72
            mids = _pick_bicycle_flow_events(dist, speed, accel, brake, n, conf)
            return mids, signal_source

    # ── Cut-in / merge specialised rule ───────────────────────────────────
    if ctx.scenario in {
        "HighwayCutIn", "StaticCutIn", "ParkingCutIn",
        "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2", "InvadingTurn",
    }:
        cfg = SCENARIO_CONFIG[ctx.scenario]
        dist = smooth(meta[cfg[0]], 5)
        if not series_has_signal(dist):
            dist = smooth(nearest_vehicle_distance_from_bboxes(ctx.run_path, n), 5)
            signal_source = "bboxes"
        if series_has_signal(dist):
            conf = 0.9 if signal_source == "metas" else 0.7
            mids = _pick_cutin_events(dist, speed, n, cfg[1], cfg[2], conf)
            return mids, signal_source

    # ── Brake/accel primary scenarios ─────────────────────────────────────
    if ctx.scenario in BRAKE_ACCEL_PRIMARY:
        cfg = SCENARIO_CONFIG[ctx.scenario]
        mids = _pick_brake_accel_events(speed, accel, brake, n, cfg[2], conf_base=0.85)
        return mids, signal_source

    # ── Generic config-driven distance rule ───────────────────────────────
    if ctx.scenario in SCENARIO_CONFIG:
        dist_field, thresh, ev_names = SCENARIO_CONFIG[ctx.scenario]
        dist = smooth(meta[dist_field], 5)

        # Choose bbox fallback based on signal class
        if dist_field == "dist_to_pedestrian":
            bbox_fn = nearest_pedestrian_from_bboxes
        elif dist_field == "dist_to_biker":
            bbox_fn = bicycle_like_distance_from_bboxes
        else:
            bbox_fn = nearest_vehicle_distance_from_bboxes

        if not series_has_signal(dist):
            dist = smooth(bbox_fn(ctx.run_path, n), 5)
            signal_source = "bboxes"

        if series_has_signal(dist):
            conf = 0.88 if signal_source == "metas" else 0.68
            mids = _pick_distance_events(
                dist, speed, accel, brake, n, thresh, ev_names, conf_base=conf
            )
            return mids, signal_source

    # ── RGB fallback for any unknown / unrecognised scenario ──────────────
    signal_source = "rgb_fallback"
    a, b, c = rgb_motion_peak_indices(ctx.run_path, n)
    cfg_names = SCENARIO_CONFIG.get(ctx.scenario, (None, None, ("event_1", "event_2", "event_3")))
    ev_names = cfg_names[2]
    return [
        (ev_names[0], a, 0.5),
        (ev_names[1], b, 0.5),
        (ev_names[2], c, 0.5),
    ], signal_source


# ---------------------------------------------------------------------------
# Run output assembly
# ---------------------------------------------------------------------------

def build_run_output(ctx: RunContext) -> Dict[str, Any]:
    meta = extract_meta_series(ctx.run_path, ctx.total_frames)
    mids, source = pick_middle_events(ctx, meta)

    initial_idx = 0
    final_idx = max(0, ctx.total_frames - 1)
    mids = enforce_event_order(initial_idx, mids, final_idx, min_gap=2)

    initial = {
        "event": "initial",
        "frame": initial_idx,
        "t": frame_to_time(initial_idx, ctx.seconds_per_frame),
        "label_text": SCENARIO_LABELS.get(ctx.scenario, f"Handle {ctx.scenario} scenario"),
        "confidence": 1.0,
    }
    middle = [
        {
            "event": name,
            "frame": idx,
            "t": frame_to_time(idx, ctx.seconds_per_frame),
            "confidence": round(conf, 3),
        }
        for name, idx, conf in mids
    ]
    final = {
        "event": "final",
        "frame": final_idx,
        "t": frame_to_time(final_idx, ctx.seconds_per_frame),
        "final_success": ctx.final_success,
        "confidence": 1.0 if ctx.final_success else 0.8,
    }
    avg_conf = sum(m["confidence"] for m in middle) / max(1, len(middle))

    return {
        "scenario": ctx.scenario,
        "run_id": ctx.run_id,
        "route_id": ctx.route_id,
        "status": ctx.status,
        "num_infractions": ctx.num_infractions,
        "signal_source": source,
        "rule_confidence": round(avg_conf, 3),
        "initial": initial,
        "middle": middle,
        "final": final,
        "diagnostics": {
            "total_frames": ctx.total_frames,
            "duration_game": round(ctx.duration_game, 4),
            "seconds_per_frame": round(ctx.seconds_per_frame, 6),
        },
    }


# ---------------------------------------------------------------------------
# Scenario discovery and run collection
# ---------------------------------------------------------------------------

def discover_scenarios(dataset_root: str) -> List[str]:
    """Return sorted list of valid scenario directory names, skipping meta-dirs."""
    return sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
        and d not in SKIPPED_DIRS
        and not d.startswith(".")
    )


def collect_runs(dataset_root: str, scenarios: Sequence[str]) -> List[RunContext]:
    runs: List[RunContext] = []
    for scenario in scenarios:
        scenario_path = os.path.join(dataset_root, scenario)
        if not os.path.isdir(scenario_path):
            continue
        for child in sorted(os.listdir(scenario_path)):
            run_path = os.path.join(scenario_path, child)
            if not os.path.isdir(run_path):
                continue
            ctx = build_run_context(scenario, run_path)
            if ctx is not None:
                runs.append(ctx)
    return runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rule-based key-frame filtering for all CARLA scenarios"
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/cruser1/lda/lead/cache_ln/data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario folders to process (default: all discovered in dataset-root)",
    )
    parser.add_argument(
        "--output",
        default="/home/cruser1/lda/lead/cache_ln/data/keyframes_all_scenarios.json",
        help="Output JSON path (default: keyframes_all_scenarios.json)",
    )
    args = parser.parse_args()

    scenarios = args.scenarios or discover_scenarios(args.dataset_root)
    print(f"Scenarios ({len(scenarios)}): {scenarios}")

    runs = collect_runs(args.dataset_root, scenarios)
    print(f"Total runs found: {len(runs)}")

    output: Dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "scenarios": list(scenarios),
        "num_runs": len(runs),
        "runs": [],
    }

    failed = []
    for i, ctx in enumerate(runs):
        try:
            output["runs"].append(build_run_output(ctx))
        except Exception as e:
            failed.append({"scenario": ctx.scenario, "run_id": ctx.run_id, "error": str(e)})
        if (i + 1) % 100 == 0:
            print(f"  … processed {i + 1}/{len(runs)}")

    output["failed_runs"] = failed
    output["num_failed_runs"] = len(failed)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Processed runs : {len(runs)}")
    print(f"Failed runs    : {len(failed)}")
    print(f"Saved          : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())