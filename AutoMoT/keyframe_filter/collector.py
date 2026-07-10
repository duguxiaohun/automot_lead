"""
场景事件采集系统 - 精简版
核心采集器 + 完整策略 + 结构分析（一个文件）

特性：
- 47个场景的采集策略
- 多标签和重叠控制
- 结构分析验证
- 单场景采集API
"""

import json
import pickle
import logging
import lzma
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict
import numpy as np


_KEYFRAME_DIR = Path(__file__).resolve().parent
_AUTOMOT_ROOT = _KEYFRAME_DIR.parent
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))
_DEFAULT_LEAD_DATA_ROOT = _AUTOMOT_ROOT / "lead_data"
_DEFAULT_OUTPUT_DIR = _KEYFRAME_DIR / "collection_output"
_DEFAULT_XML_ROOT = _AUTOMOT_ROOT / "data" / "lead"
_DEFAULT_CARLA_ROOT = _AUTOMOT_ROOT / "CARLA_0915"

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402

# ============================================================================
# 辅助函数
# ============================================================================

def load_pickle_file(file_path: Path):
    """
    加载 pickle 文件，支持 XZ 压缩格式

    某些 .pkl 文件实际上是 XZ 压缩的。此函数自动检测并处理。
    """
    try:
        # 首先尝试标准 pickle 加载
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        # 如果标准加载失败，尝试 XZ 解压
        try:
            with lzma.open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as xz_error:
            # 两种方式都失败，抛出原始错误
            raise e


def _bbox_object_iter(raw: Any):
    """尽量兼容 LEAD bbox pkl 的 list/dict/ndarray 结构，逐个产出对象 dict。"""
    if raw is None:
        return
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, dict):
        for key in ("objects", "bboxes", "boxes", "actors", "data"):
            value = raw.get(key)
            if value is not None:
                yield from _bbox_object_iter(value)
                return
        yield raw
        return
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, dict):
                yield item


def _bbox_class_text(obj: Dict[str, Any]) -> str:
    for key in ("class", "type", "label", "name", "category", "semantic_class"):
        value = obj.get(key)
        if value is not None:
            return str(value).strip().lower()
    return ""


def summarize_bbox_semantics(bbox_path: Path) -> Dict[str, Any]:
    """读取单帧 bbox，提取 R4/R5 需要的轻量语义证据。"""
    if not bbox_path.exists():
        return {
            "bbox_available": False,
            "bbox_has_traffic_light": False,
            "bbox_has_stop_sign": False,
            "bbox_has_yield_sign": False,
            "bbox_has_junction_hint": False,
            "bbox_semantic_classes": {},
            "bbox_semantic_metrics": {},
        }
    try:
        raw = load_pickle_file(bbox_path)
    except Exception:
        return {
            "bbox_available": False,
            "bbox_load_error": True,
            "bbox_has_traffic_light": False,
            "bbox_has_stop_sign": False,
            "bbox_has_yield_sign": False,
            "bbox_has_junction_hint": False,
            "bbox_semantic_classes": {},
            "bbox_semantic_metrics": {},
        }

    counts: Dict[str, int] = defaultdict(int)
    has_tl = False
    has_stop = False
    has_yield = False
    has_junction = False
    metrics: Dict[str, Any] = {
        "traffic_light_count": 0,
        "traffic_light_min_distance_m": None,
        "traffic_light_min_forward_x_m": None,
        "traffic_light_min_physical_distance_m": None,
        "traffic_light_affects_ego": False,
        "traffic_light_same_lane": False,
        "traffic_light_overhead": False,
        "stop_sign_count": 0,
        "stop_sign_min_distance_m": None,
        "yield_sign_count": 0,
        "yield_sign_min_distance_m": None,
        "red_light_conflict_vehicle_count": 0,
        "red_light_conflict_vehicle_min_distance_m": None,
        "red_light_conflict_vehicle_min_forward_x_m": None,
        "red_light_conflict_vehicle_min_abs_lateral_y_m": None,
        "red_light_conflict_vehicle_max_speed_mps": None,
    }

    def _update_min(key: str, value: Any) -> None:
        value_f = _safe_float(value, default=math.inf)
        if not math.isfinite(value_f):
            return
        current = _safe_float(metrics.get(key), default=math.inf)
        if value_f < current:
            metrics[key] = round(value_f, 3)

    for obj in _bbox_object_iter(raw) or []:
        cls = _bbox_class_text(obj)
        if not cls:
            continue
        counts[cls] += 1
        compact = cls.replace("_", "").replace("-", "").replace(" ", "")
        if "trafficlight" in compact or compact in {"tl", "light"}:
            has_tl = True
            metrics["traffic_light_count"] = int(metrics["traffic_light_count"]) + 1
            _update_min("traffic_light_min_distance_m", obj.get("distance"))
            position = obj.get("position")
            if isinstance(position, (list, tuple)) and position:
                _update_min("traffic_light_min_forward_x_m", position[0])
            _update_min("traffic_light_min_physical_distance_m", obj.get("distance_to_physical_traffic_light"))
            metrics["traffic_light_affects_ego"] = bool(metrics["traffic_light_affects_ego"]) or _safe_bool(obj.get("affects_ego"))
            metrics["traffic_light_same_lane"] = bool(metrics["traffic_light_same_lane"]) or _safe_bool(obj.get("same_lane_as_ego"))
            metrics["traffic_light_overhead"] = bool(metrics["traffic_light_overhead"]) or _safe_bool(obj.get("is_over_head_traffic_light"))
        if "stopsign" in compact or compact == "stop":
            has_stop = True
            metrics["stop_sign_count"] = int(metrics["stop_sign_count"]) + 1
            _update_min("stop_sign_min_distance_m", obj.get("distance"))
        if "yield" in compact or "giveway" in compact:
            has_yield = True
            metrics["yield_sign_count"] = int(metrics["yield_sign_count"]) + 1
            _update_min("yield_sign_min_distance_m", obj.get("distance"))
        if "junction" in compact or "intersection" in compact or "crosswalk" in compact:
            has_junction = True
        if (
            ("car" in compact or "vehicle" in compact)
            and "ego" not in compact
            and "static" not in compact
        ):
            position = obj.get("position")
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                x = _safe_float(position[0], default=math.inf)
                y = _safe_float(position[1], default=math.inf)
                dist = _safe_float(obj.get("distance"), default=math.inf)
                speed = _safe_float(obj.get("speed"), default=0.0)
                yaw = _safe_float(obj.get("yaw"), default=math.inf)
                abs_y = abs(y) if math.isfinite(y) else math.inf
                yaw_abs = abs(((yaw + math.pi) % (2.0 * math.pi)) - math.pi) if math.isfinite(yaw) else math.inf
                crossing_or_opposite_heading = (
                    yaw_abs >= 2.2
                    or abs(yaw_abs - (math.pi / 2.0)) <= 0.7
                    or speed >= 5.0
                )
                if (
                    0.0 <= x <= 30.0
                    and 2.0 <= abs_y <= 7.5
                    and dist <= 35.0
                    and speed >= 1.0
                    and crossing_or_opposite_heading
                ):
                    metrics["red_light_conflict_vehicle_count"] = int(metrics["red_light_conflict_vehicle_count"]) + 1
                    _update_min("red_light_conflict_vehicle_min_distance_m", dist)
                    _update_min("red_light_conflict_vehicle_min_forward_x_m", x)
                    _update_min("red_light_conflict_vehicle_min_abs_lateral_y_m", abs_y)
                    current_speed = _safe_float(metrics.get("red_light_conflict_vehicle_max_speed_mps"), default=-math.inf)
                    if speed > current_speed:
                        metrics["red_light_conflict_vehicle_max_speed_mps"] = round(speed, 3)

    return {
        "bbox_available": True,
        "bbox_has_traffic_light": has_tl,
        "bbox_has_stop_sign": has_stop,
        "bbox_has_yield_sign": has_yield,
        "bbox_has_junction_hint": has_junction,
        "bbox_semantic_classes": dict(sorted(counts.items())),
        "bbox_semantic_metrics": metrics,
    }

# ============================================================================
# 数据定义
# ============================================================================

class RoadStructure(Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"


class EventType(Enum):
    R_E1 = "R-E1"
    R_E2 = "R-E2"
    R_E3 = "R-E3"
    R_E4 = "R-E4"
    R_E5 = "R-E5"
    U_E1 = "U-E1"
    U_E2 = "U-E2"
    U_E3 = "U-E3"
    U_E4 = "U-E4"
    U_E5 = "U-E5"
    U_E6 = "U-E6"
    U_E7 = "U-E7"
    U_E8 = "U-E8"


ROAD_STRUCTURE_LABELS = {
    "R1": "常规道路 / 跟车 / 车道保持",
    "R2": "双向单车道 / 借对向绕行规则空间",
    "R3": "匝道合流 / 并线 / 高速驶出",
    "R4": "信号灯控制路口",
    "R5": "无信号灯 / 路权路口",
}


EVENT_LABELS = {
    "R-E1": "常规跟车 / 车道保持",
    "R-E2": "目标导向型变道 / 回目标车道",
    "R-E3": "常规匝道合流 / 并线 / 驶出",
    "R-E4": "信号灯路口常规通行",
    "R-E5": "无信号灯 / 路权路口常规通行",
    "U-E1": "前车急刹 / 突然减速",
    "U-E2": "静态障碍物占道 / 被迫绕行",
    "U-E3": "动态车辆切入 / 动态占道",
    "U-E4": "行人 / 自行车横穿或侧向进入路径",
    "U-E5": "对向车辆异常侵占自车道",
    "U-E6": "违规车辆冲突",
    "U-E7": "信号灯故障 / 路口规则失效",
    "U-E8": "前方道路暂时阻塞 / 阻塞解除",
}


@dataclass
class FrameAnnotation:
    frame_id: int
    road_structures: Set[RoadStructure]
    events: Set[EventType]
    confidence: float = 1.0
    reason: str = ""
    primary_road_structure: Optional[RoadStructure] = None
    secondary_road_structures: Set[RoadStructure] = field(default_factory=set)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    annotation_comment: str = ""
    primary_event: Optional[EventType] = None
    event_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            'frame_id': self.frame_id,
            'road_structures': [rs.value for rs in self.road_structures],
            'events': [ev.value for ev in self.events],
            'primary_event': (
                self.primary_event.value
                if isinstance(self.primary_event, EventType)
                else self.primary_event
            ),
            'confidence': self.confidence,
            'reason': self.reason,
            'primary_road_structure': (
                self.primary_road_structure.value
                if isinstance(self.primary_road_structure, RoadStructure)
                else self.primary_road_structure
            ),
            'secondary_road_structures': [
                rs.value if isinstance(rs, RoadStructure) else rs
                for rs in sorted(self.secondary_road_structures, key=lambda x: x.value if isinstance(x, RoadStructure) else str(x))
            ],
            'road_structure_candidates': self.candidate_scores,
            'road_structure_overlay': None,
            'evidence': self.evidence,
            'event_evidence': self.event_evidence,
            'annotation_comment': self.annotation_comment,
        }


# ============================================================================
# 全局映射表
# ============================================================================

SCENARIO_TO_ROAD_STRUCTURE = {
    "Accident": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "AccidentTwoWays": [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "BlockedIntersection": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ConstructionObstacle": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ConstructionObstacleTwoWays": [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "ControlLoss": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "CrossingBicycleFlow": [RoadStructure.R1, RoadStructure.R4],
    "CrossJunctionDefectTrafficLight": [RoadStructure.R1, RoadStructure.R4],
    "DynamicObjectCrossing": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "EnterActorFlow": [RoadStructure.R1, RoadStructure.R3],
    "EnterActorFlowV2": [RoadStructure.R1, RoadStructure.R3],
    "HardBreakRoute": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R5],
    "HazardAtSideLane": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "HazardAtSideLaneTwoWays": [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "HighwayCutIn": [RoadStructure.R3, RoadStructure.R4],
    "HighwayExit": [RoadStructure.R3],
    "InterurbanActorFlow": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R5],
    "InterurbanAdvancedActorFlow": [RoadStructure.R1, RoadStructure.R5],
    "InvadingTurn": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "MergerIntoSlowTraffic": [RoadStructure.R3, RoadStructure.R4],
    "MergerIntoSlowTrafficV2": [RoadStructure.R3],
    "NonSignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "noScenarios": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R5],
    "OppositeVehicleRunningRedLight": [RoadStructure.R1, RoadStructure.R4],
    "OppositeVehicleTakingPriority": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ParkedObstacle": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ParkedObstacleTwoWays": [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "ParkingCrossingPedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ParkingCutIn": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ParkingExit": [RoadStructure.R1, RoadStructure.R4],
    "PedestrianCrossing": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "PriorityAtJunction": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "RedLightWithoutLeadVehicle": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R4],
    "StaticCutIn": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R5],
    "T_Junction": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "VehicleOpensDoorTwoWays": [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5],
    "VehicleTurningRoute": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "VehicleTurningRoutePedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
}

LAYOUT_R2_ROUTE_IDS: Dict[str, Set[str]] = {
    # 非 TwoWays 场景必须先逐 route / 逐帧 RGB 复核，确认确实是对向单车道，
    # 再写入这里动态开放 R2。当前先保持空白，避免普通 R1/R-E1 场景被误升 R2。
}

MIXED_SCENARIO_HIGHWAY_ROUTE_IDS = {
    # RGB reviewed HardBreakRoute fast-road/highway bucket.
    # These routes show divided multi-lane fast roads, guardrails, ramps/bridges, or highway-style lane geometry.
    "HardBreakRoute": {
        "Town12_Rep0_1428_0_route0_01_10_00_18_08",
        "Town12_Rep0_1439_0_route0_01_10_04_51_04",
        "Town12_Rep0_2452_0_route0_01_11_03_14_24",
        "Town12_Rep0_2510_0_route0_01_09_22_05_04",
        "Town12_Rep0_2585_0_route0_01_09_13_23_09",
        "Town12_Rep0_4115_0_route0_01_09_08_38_49",
        "Town12_Rep0_4118_0_route0_01_09_11_51_26",
        "Town12_Rep0_4139_0_route0_01_11_01_24_29",
        "Town12_Rep0_947_0_route0_01_10_04_45_34",
        "Town12_Rep0_954_0_route0_01_10_01_42_01",
        "Town13_Rep0_1258_0_route0_01_09_16_21_37",
        "Town13_Rep0_1269_0_route0_01_07_23_43_15",
        "Town13_Rep0_1275_0_route0_01_09_14_40_40",
        "Town13_Rep0_1387_0_route0_01_08_06_02_20",
        "Town13_Rep0_1663_0_route0_01_09_13_45_51",
        "Town13_Rep0_1666_0_route0_01_10_04_27_25",
    },
    # RGB reviewed mixed scenarios. Empty sets mean every route stayed in the non-highway bucket.
    "InterurbanActorFlow": set(),
    "InterurbanAdvancedActorFlow": set(),
    "StaticCutIn": {
        "Town12_Rep0_1419_0_route0_01_09_22_23_57",
        "Town12_Rep0_1429_0_route0_01_08_20_38_15",
        "Town12_Rep0_1429_1_route0_01_10_03_05_37",
        "Town12_Rep0_2450_0_route0_01_08_05_41_18",
        "Town12_Rep0_2462_0_route0_01_08_23_27_47",
        "Town12_Rep0_2579_0_route0_01_10_06_04_27",
        "Town12_Rep0_2974_0_route0_01_10_17_47_12",
        "Town12_Rep0_4001_1_route0_01_09_20_16_43",
        "Town12_Rep0_4003_0_route0_01_10_13_44_50",
        "Town12_Rep0_4015_0_route0_01_08_08_47_29",
        "Town12_Rep0_4015_1_route0_01_10_18_20_11",
        "Town12_Rep0_4029_0_route0_01_10_07_58_19",
        "Town12_Rep0_4041_0_route0_01_10_19_55_39",
        "Town12_Rep0_4133_0_route0_01_10_02_10_03",
        "Town12_Rep0_4135_0_route0_01_09_00_50_04",
        "Town12_Rep0_4135_1_route0_01_09_00_38_11",
        "Town12_Rep0_4476_0_route0_01_09_04_31_11",
        "Town12_Rep0_4476_1_route0_01_10_03_41_06",
        "Town12_Rep0_4488_0_route0_01_09_10_22_28",
        "Town12_Rep0_4490_0_route0_01_11_15_05_07",
        "Town12_Rep0_4490_1_route0_01_10_19_02_40",
        "Town12_Rep0_4495_0_route0_01_09_20_22_50",
        "Town12_Rep0_4495_1_route0_01_08_08_10_03",
        "Town12_Rep0_911_0_route0_01_10_14_53_36",
        "Town12_Rep0_913_0_route0_01_11_01_32_11",
        "Town12_Rep0_934_0_route0_01_09_20_20_25",
        "Town12_Rep0_949_0_route0_01_07_22_14_09",
        "Town13_Rep0_1264_0_route0_01_09_19_23_58",
        "Town13_Rep0_1264_1_route0_01_08_01_53_22",
        "Town13_Rep0_1264_2_route0_01_10_06_31_25",
        "Town13_Rep0_1266_0_route0_01_10_11_13_19",
        "Town13_Rep0_1266_1_route0_01_09_05_38_22",
        "Town13_Rep0_1266_2_route0_01_07_22_56_05",
        "Town13_Rep0_1267_0_route0_01_09_18_13_31",
        "Town13_Rep0_1267_1_route0_01_08_12_46_57",
        "Town13_Rep0_1267_2_route0_01_09_21_18_03",
        "Town13_Rep0_1386_0_route0_01_10_22_42_39",
        "Town13_Rep0_1386_1_route0_01_08_16_01_37",
        "Town13_Rep0_1624_0_route0_01_09_02_55_07",
        "Town13_Rep0_1624_1_route0_01_10_11_10_25",
        "Town13_Rep0_1626_0_route0_01_08_13_53_39",
        "Town13_Rep0_1626_1_route0_01_09_12_51_05",
        "Town13_Rep0_1668_0_route0_01_11_09_17_39",
        "Town13_Rep0_1668_1_route0_01_10_21_25_32",
    },
    "ParkingCutIn": set(),
}

MIXED_SCENARIO_RGB_REVIEW_COUNTS = {
    "HardBreakRoute": 97,
    "InterurbanActorFlow": 91,
    "InterurbanAdvancedActorFlow": 78,
    "StaticCutIn": 100,
    "ParkingCutIn": 99,
}

SCENARIOS_WITH_RGB_NO_R4 = {
    # 2026-07-04 full lead_video RGB overview: no stable signalized intersections.
    "EnterActorFlow",
    "EnterActorFlowV2",
    "HighwayExit",
    "InterurbanActorFlow",
    "InterurbanAdvancedActorFlow",
    "MergerIntoSlowTrafficV2",
    # Explicit no-signal left-turn families: RGB review shows STOP/no-light control.
    # Bbox occasionally reports traffic_light together with stop_sign here; do not
    # dynamically reopen R4 from that weak hint.
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionLeftTurnEnterFlow",
}


def _mixed_route_semantic_bucket(scenario_name: str, route_id: Optional[str]) -> str:
    if scenario_name not in MIXED_SCENARIO_HIGHWAY_ROUTE_IDS:
        return "not_mixed"
    if not route_id:
        return "mixed_unknown_no_route_id"
    if route_id in MIXED_SCENARIO_HIGHWAY_ROUTE_IDS.get(scenario_name, set()):
        return "highway_rgb_route"
    return "mixed_reviewed_non_highway"


def _mixed_route_allowed_structures(
    scenario_name: str,
    route_id: Optional[str],
    base_allowed: Set[RoadStructure],
) -> Set[RoadStructure]:
    bucket = _mixed_route_semantic_bucket(scenario_name, route_id)
    if bucket == "highway_rgb_route":
        allowed = {rs for rs in base_allowed if rs in {RoadStructure.R3, RoadStructure.R4}}
        allowed.add(RoadStructure.R3)
        return allowed
    allowed = set(base_allowed)
    if route_id and route_id in LAYOUT_R2_ROUTE_IDS.get(scenario_name, set()):
        allowed.add(RoadStructure.R2)
    return allowed

SCENARIO_TO_FINE_EVENTS = {
    "Accident": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "AccidentTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "BlockedIntersection": [EventType.R_E1, EventType.R_E4, EventType.U_E1, EventType.U_E8],
    "ConstructionObstacle": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "ConstructionObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "ControlLoss": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "CrossingBicycleFlow": [EventType.R_E1, EventType.R_E4, EventType.U_E4],
    "CrossJunctionDefectTrafficLight": [EventType.R_E1, EventType.R_E4, EventType.U_E6, EventType.U_E7],
    "DynamicObjectCrossing": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E3, EventType.U_E4],
    "EnterActorFlow": [EventType.R_E1, EventType.R_E2, EventType.R_E3],
    "EnterActorFlowV2": [EventType.R_E1, EventType.R_E2, EventType.R_E3],
    "HardBreakRoute": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E1],
    "HazardAtSideLane": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E4],
    "HazardAtSideLaneTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E4],
    "HighwayCutIn": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4],
    "HighwayExit": [EventType.R_E1, EventType.R_E2, EventType.R_E3],
    "InterurbanActorFlow": [EventType.R_E1, EventType.R_E2, EventType.R_E5],
    "InterurbanAdvancedActorFlow": [EventType.R_E1, EventType.R_E2, EventType.R_E5],
    "InvadingTurn": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E5],
    "MergerIntoSlowTraffic": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4],
    "MergerIntoSlowTrafficV2": [EventType.R_E1, EventType.R_E2, EventType.R_E3],
    "NonSignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "noScenarios": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4, EventType.R_E5],
    "OppositeVehicleRunningRedLight": [EventType.R_E1, EventType.R_E4, EventType.U_E6],
    "OppositeVehicleTakingPriority": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E7],
    "ParkedObstacle": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "ParkedObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "ParkingCrossingPedestrian": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "ParkingCutIn": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E3],
    "ParkingExit": [EventType.R_E1, EventType.R_E2, EventType.R_E4],
    "PedestrianCrossing": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "PriorityAtJunction": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "RedLightWithoutLeadVehicle": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E4],
    "StaticCutIn": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4, EventType.R_E5, EventType.U_E3],
    "T_Junction": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "VehicleOpensDoorTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E5, EventType.U_E2],
    "VehicleTurningRoute": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "VehicleTurningRoutePedestrian": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
}

ROAD_STRUCTURE_TO_FINE_EVENTS = {
    RoadStructure.R1: {EventType.R_E1, EventType.R_E2, EventType.U_E1, EventType.U_E2, EventType.U_E3, EventType.U_E4},
    RoadStructure.R2: {EventType.R_E1, EventType.R_E2, EventType.U_E2, EventType.U_E5},
    RoadStructure.R3: {EventType.R_E1, EventType.R_E2, EventType.R_E3},
    RoadStructure.R4: {EventType.R_E4, EventType.U_E4, EventType.U_E6, EventType.U_E8},
    RoadStructure.R5: {EventType.R_E5, EventType.U_E4, EventType.U_E5, EventType.U_E6, EventType.U_E7, EventType.U_E8},
}

OBSTACLE_EVENT_DISTANCE_FIELDS = {
    "Accident": ("dist_to_accident_site", 29.0),
    "AccidentTwoWays": ("dist_to_accident_site", 29.0),
    "ConstructionObstacle": ("dist_to_construction_site", 32.0),
    "ConstructionObstacleTwoWays": ("dist_to_construction_site", 32.0),
    "ParkedObstacle": ("dist_to_parked_obstacle", 25.0),
    "ParkedObstacleTwoWays": ("dist_to_parked_obstacle", 25.0),
    "VehicleOpensDoorTwoWays": ("dist_to_vehicle_opens_door", 26.0),
}

PEDESTRIAN_BICYCLE_EVENT_FIELDS = {
    "CrossingBicycleFlow": ("dist_to_biker", 22.0),
    "DynamicObjectCrossing": ("dist_to_pedestrian", 22.0),
    "HazardAtSideLane": ("dist_to_biker", 30.0),
    "HazardAtSideLaneTwoWays": ("dist_to_biker", 30.0),
    "PedestrianCrossing": ("dist_to_pedestrian", 22.0),
    "ParkingCrossingPedestrian": ("dist_to_pedestrian", 24.0),
    "VehicleTurningRoute": ("dist_to_biker", 16.0),
    "VehicleTurningRoutePedestrian": ("dist_to_pedestrian", 22.0),
}

CROSSING_U4_SINGLE_SPAN_SCENARIOS = set(PEDESTRIAN_BICYCLE_EVENT_FIELDS)
CROSSING_U4_SUPPORT_PAD_M = 6.0
CROSSING_U4_MAX_INTERNAL_GAP_FRAMES = 10


def _crossing_u4_support_pad_m(scenario_name: str) -> float:
    if scenario_name == "VehicleTurningRoute":
        return 2.0
    if scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}:
        return 2.0
    return CROSSING_U4_SUPPORT_PAD_M


def _crossing_u4_max_internal_gap_frames(scenario_name: str) -> int:
    if scenario_name == "CrossingBicycleFlow":
        return 14
    return CROSSING_U4_MAX_INTERNAL_GAP_FRAMES


INTERRUPTED_UNUSUAL_OVERLAY_EVENTS = {EventType.U_E1, EventType.U_E2, EventType.U_E3, EventType.U_E4}
INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES = 12
INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES = 24
STATIC_U2_RE2_CLEAR_DELTA_M = 4.5
CUTIN_U3_ACTIVE_DISTANCE_M = 28.0
RECOVERY_AFTER_LATERAL_PEAK_FRAMES = 3
RECOVERY_LATERAL_DROP_START_M = 0.18
RECOVERY_LATERAL_DROP_STRONG_M = 0.35

R3_MERGE_SCENARIOS = {
    "EnterActorFlow",
    "EnterActorFlowV2",
    "MergerIntoSlowTraffic",
    "MergerIntoSlowTrafficV2",
    "HighwayExit",
}

R2_RETURN_SCENARIOS = {
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

@dataclass
class RouteXmlInfo:
    """Route XML 解析结果，供 ROAD_STRUCTURE 规则使用。"""

    path: Path
    scenario: str
    route_id: str
    town: str
    waypoints: List[Tuple[float, float]]
    scenario_tags: List[Dict[str, Any]]
    weathers: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def trigger_points(self) -> List[Tuple[float, float]]:
        points = []
        for tag in self.scenario_tags:
            trigger = tag.get("trigger_point")
            if trigger is not None:
                points.append(trigger)
        return points


def _safe_float(value: Any, default: float = math.inf) -> float:
    """把 meta/XML 字段稳妥转成 float，失败时返回 default。"""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _safe_bool(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        value = value.reshape(-1)[0]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _valid_traffic_light(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"red", "yellow", "green"}


def _finite_min(*values: Any) -> float:
    finite = [_safe_float(v) for v in values]
    finite = [v for v in finite if math.isfinite(v)]
    return min(finite) if finite else math.inf


def _point_at_geometry_s(geom: Dict[str, float], local_s: float) -> Tuple[float, float]:
    """按 OpenDRIVE planView geometry 粗略计算 s 位置坐标。"""
    local_s = max(0.0, min(float(local_s), float(geom.get("length", 0.0))))
    x = float(geom.get("x", 0.0))
    y = float(geom.get("y", 0.0))
    hdg = float(geom.get("hdg", 0.0))
    curvature = geom.get("curvature")
    if curvature is None or abs(float(curvature)) < 1e-9:
        return (x + local_s * math.cos(hdg), y + local_s * math.sin(hdg))

    curvature = float(curvature)
    radius = 1.0 / curvature
    cx = x - radius * math.sin(hdg)
    cy = y + radius * math.cos(hdg)
    theta = hdg + local_s * curvature
    return (cx + radius * math.sin(theta), cy - radius * math.cos(theta))


def _geometry_bbox(geom: Dict[str, float]) -> Tuple[float, float, float, float]:
    """给 geometry 生成采样 bbox，用于静态 XODR 近邻粗筛。"""
    length = float(geom.get("length", 0.0))
    sample_count = 2 if geom.get("curvature") is None else 8
    points = [
        _point_at_geometry_s(geom, length * idx / max(1, sample_count - 1))
        for idx in range(sample_count)
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = 8.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _bbox_min_distance(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> float:
    """点到 bbox 的最小可能距离，用于跳过明显无关 road geometry。"""
    px, py = point
    min_x, min_y, max_x, max_y = bbox
    dx = max(min_x - px, 0.0, px - max_x)
    dy = max(min_y - py, 0.0, py - max_y)
    return math.hypot(dx, dy)


def _distance_point_to_segment(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """计算点到线段距离。"""
    px, py = point
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / denom))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(px - qx, py - qy)


def _distance_to_geometry(point: Tuple[float, float], geom: Dict[str, float]) -> float:
    """用线段采样近似点到 OpenDRIVE geometry 的距离。"""
    length = float(geom.get("length", 0.0))
    if length <= 1e-6:
        gx, gy = _point_at_geometry_s(geom, 0.0)
        return math.hypot(point[0] - gx, point[1] - gy)
    sample_count = max(2, min(24, int(length / 8.0) + 2))
    best = math.inf
    prev = _point_at_geometry_s(geom, 0.0)
    for idx in range(1, sample_count):
        local_s = length * idx / (sample_count - 1)
        cur = _point_at_geometry_s(geom, local_s)
        best = min(best, _distance_point_to_segment(point, prev, cur))
        prev = cur
    return best


def _extract_route_num(text: str) -> Optional[str]:
    match = re.search(r"route[_-]?(\d{3,6})", text, re.IGNORECASE)
    if match:
        return match.group(1).zfill(6)
    match = re.search(r"(\d{3,6})", text)
    if match:
        return match.group(1).zfill(6)
    return None


def _extract_town(text: str) -> Optional[str]:
    match = re.search(r"(Town\d+(?:HD)?)", str(text), re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _route_key_aliases(text: str) -> Set[str]:
    name = Path(str(text)).stem
    out = {name.lower()}
    run_match = re.match(
        r"^(?P<town>Town\d+(?:HD)?)_Rep\d+_(?P<key>.+)_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$",
        name,
        re.IGNORECASE,
    )
    if run_match:
        town = run_match.group("town")
        raw = run_match.group("key")
        if raw.endswith("_route0"):
            raw = raw[: -len("_route0")]
        out.add(raw.lower())
        out.add(f"{town}_Rep0_{raw}".lower())
        if raw.lower().startswith("route_"):
            out.add(f"{town}_{raw}".lower())
            out.add(raw[len("route_"):].lower())
        else:
            out.add(f"{town}_route_{raw}".lower())
    xml_match = re.match(r"^(?P<town>Town\d+(?:HD)?)_route_(?P<key>.+)$", name, re.IGNORECASE)
    if xml_match:
        town = xml_match.group("town")
        raw = xml_match.group("key")
        out.add(raw.lower())
        out.add(f"{town}_Rep0_{raw}".lower())
        if raw.lower().startswith("route_"):
            out.add(f"{town}_{raw}".lower())
            out.add(raw[len("route_"):].lower())
        else:
            out.add(f"{town}_Rep0_route_{raw}".lower())
    return out


def _strict_xml_stem_from_run_name(text: str) -> Optional[str]:
    """按 LEAD run_id 命名规范反推 XML stem，优先于模糊 alias 匹配。"""
    name = Path(str(text)).stem
    run_match = re.match(
        r"^(?P<town>Town\d+(?:HD)?)_Rep\d+_(?P<key>.+)_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$",
        name,
        re.IGNORECASE,
    )
    if not run_match:
        return None
    town = run_match.group("town")
    raw = run_match.group("key")
    if raw.endswith("_route0"):
        raw = raw[: -len("_route0")]
    if raw.startswith("route_"):
        return f"{town}_{raw}"
    return f"{town}_route_{raw}"


def _parse_position_node(node: ET.Element) -> Optional[Tuple[float, float]]:
    x = _safe_float(node.get("x"))
    y = _safe_float(node.get("y"))
    if math.isfinite(x) and math.isfinite(y):
        return (x, y)
    return None


def _parse_weather_node(node: ET.Element) -> Dict[str, Any]:
    weather: Dict[str, Any] = {}
    for key, value in node.attrib.items():
        parsed = _safe_float(value, default=math.nan)
        weather[key] = parsed if math.isfinite(parsed) else value
    return weather


def _parse_route_xml(path: Path, fallback_scenario: str) -> Optional[RouteXmlInfo]:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    route = root.find(".//route")
    if route is None:
        return None
    waypoints = []
    for node in route.findall(".//waypoints/position"):
        point = _parse_position_node(node)
        if point is not None:
            waypoints.append(point)

    weathers = [_parse_weather_node(node) for node in route.findall(".//weathers/weather")]

    tags = []
    for scenario_node in route.findall(".//scenarios/scenario"):
        item: Dict[str, Any] = {
            "name": scenario_node.get("name", ""),
            "type": scenario_node.get("type", fallback_scenario),
        }
        for child in list(scenario_node):
            if child.tag == "trigger_point":
                item["trigger_point"] = _parse_position_node(child)
                item["trigger_yaw"] = _safe_float(child.get("yaw"), default=math.nan)
                continue
            attrs = {}
            for key, value in child.attrib.items():
                attrs[key] = _safe_float(value, default=value)
            if "value" in attrs and len(attrs) == 1:
                item[child.tag] = attrs["value"]
            else:
                item[child.tag] = attrs

        tags.append(item)

    return RouteXmlInfo(
        path=path,
        scenario=fallback_scenario,
        route_id=route.get("id", path.stem),
        town=route.get("town", ""),
        waypoints=waypoints,
        scenario_tags=tags,
        weathers=weathers,
    )


class RouteXmlIndex:
    """按 scenario/route id 建 XML 索引，支持本地 data/lead 的多种命名。"""

    def __init__(self, xml_root: Path = _DEFAULT_XML_ROOT):
        self.xml_root = Path(xml_root)
        self.by_scenario: Dict[str, List[RouteXmlInfo]] = defaultdict(list)
        self.by_stem: Dict[Tuple[str, str], RouteXmlInfo] = {}
        self.by_route_num: Dict[Tuple[str, str], List[RouteXmlInfo]] = defaultdict(list)
        self.by_town_route_num: Dict[Tuple[str, str, str], List[RouteXmlInfo]] = defaultdict(list)
        self._build()

    @staticmethod
    def _append_unique(bucket: List[RouteXmlInfo], info: RouteXmlInfo) -> None:
        if not any(existing.path == info.path for existing in bucket):
            bucket.append(info)

    def _build(self) -> None:
        if not self.xml_root.exists():
            return
        for scenario_dir in sorted(p for p in self.xml_root.iterdir() if p.is_dir()):
            scenario = scenario_dir.name
            for xml_path in sorted(scenario_dir.glob("*.xml")):
                info = _parse_route_xml(xml_path, scenario)
                if info is None:
                    continue
                self.by_scenario[scenario].append(info)
                self.by_stem[(scenario, xml_path.stem.lower())] = info
                for key in {xml_path.stem, info.route_id}:
                    route_num = _extract_route_num(key)
                    if route_num:
                        self._append_unique(self.by_route_num[(scenario, route_num)], info)
                        if info.town:
                            self._append_unique(self.by_town_route_num[(scenario, info.town.lower(), route_num)], info)

    def match(self, scenario: str, route_name: str) -> Optional[RouteXmlInfo]:
        strict_stem = _strict_xml_stem_from_run_name(route_name)
        if strict_stem:
            hit = self.by_stem.get((scenario, strict_stem.lower()))
            if hit is not None:
                return hit

        candidates = self.by_scenario.get(scenario, [])
        route_name_lower = route_name.lower()
        route_aliases = _route_key_aliases(route_name)
        route_town = _extract_town(route_name)
        route_town_key = route_town.lower() if route_town else None
        for info in candidates:
            if info.path.stem and info.path.stem.lower() in route_name_lower:
                return info
            if _route_key_aliases(info.path.stem) & route_aliases:
                return info
        route_num = _extract_route_num(route_name)
        if route_num:
            if route_town_key:
                town_hits = self.by_town_route_num.get((scenario, route_town_key, route_num), [])
                if len(town_hits) == 1:
                    return town_hits[0]
            hits = self.by_route_num.get((scenario, route_num), [])
            if len(hits) == 1:
                return hits[0]
        if not candidates:
            return None
        for info in candidates:
            if route_town_key and info.town and info.town.lower() != route_town_key:
                continue
            if info.route_id and len(str(info.route_id)) >= 3 and str(info.route_id) in route_name:
                return info
        return None


class XodrTopologyProbe:
    """可选 XODR 查询器：有 carla Python API 时读取地图拓扑，无则自动降级。"""

    def __init__(self, carla_root: Path = _DEFAULT_CARLA_ROOT):
        self.carla_root = Path(carla_root)
        self._carla = None
        self._maps: Dict[str, Any] = {}
        self._static_indexes: Dict[str, Dict[str, Any]] = {}
        self._import_failed = False

    def _find_xodr(self, town: str) -> Optional[Path]:
        if not town:
            return None
        candidates = [
            self.carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}.xodr",
            self.carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}.xodr",
            self.carla_root / "AdditionalMaps_0.9.15" / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / f"{town}.xodr",
            self.carla_root / "AdditionalMaps_0.9.15" / "CarlaUE4" / "Content" / "Carla" / "Maps" / town / "OpenDrive" / f"{town}.xodr",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_map(self, town: str):
        if not town or self._import_failed:
            return None
        if town in self._maps:
            return self._maps[town]
        try:
            if self._carla is None:
                import carla  # type: ignore
                self._carla = carla
            xodr = self._find_xodr(town)
            if xodr is None:
                return None
            map_obj = self._carla.Map(town, xodr.read_text(encoding="utf-8", errors="ignore"))
            self._maps[town] = map_obj
            return map_obj
        except Exception:
            self._import_failed = True
            return None

    def _load_static_index(self, town: str) -> Dict[str, Any]:
        """无 CARLA API 时解析 XODR 几何/信号/lane 粗索引，保证规则可执行。"""
        if town in self._static_indexes:
            return self._static_indexes[town]
        xodr = self._find_xodr(town)
        index: Dict[str, Any] = {"available": False, "path": str(xodr) if xodr else None, "roads": {}, "signals": [], "roundabout_junctions": []}
        if xodr is None:
            self._static_indexes[town] = index
            return index
        try:
            root = ET.parse(xodr).getroot()
            junction_roads = set()
            junction_connections: Dict[str, List[str]] = defaultdict(list)
            for junction in root.findall(".//junction"):
                junction_id = str(junction.get("id", ""))
                for connection in junction.findall("connection"):
                    incoming = connection.get("incomingRoad")
                    connecting = connection.get("connectingRoad")
                    if incoming:
                        junction_roads.add(str(incoming))
                        if junction_id:
                            junction_connections[junction_id].append(str(incoming))
                    if connecting:
                        junction_roads.add(str(connecting))
                        if junction_id:
                            junction_connections[junction_id].append(str(connecting))

            for road in root.findall(".//road"):
                road_id = str(road.get("id", ""))
                if not road_id:
                    continue
                geometries = []
                for geom in road.findall(".//planView/geometry"):
                    try:
                        item: Dict[str, float] = {
                            "s": float(geom.get("s", 0.0)),
                            "x": float(geom.get("x", 0.0)),
                            "y": float(geom.get("y", 0.0)),
                            "hdg": float(geom.get("hdg", 0.0)),
                            "length": float(geom.get("length", 0.0)),
                        }
                    except Exception:
                        continue
                    arc = geom.find("arc")
                    if arc is not None and arc.get("curvature") is not None:
                        item["curvature"] = float(arc.get("curvature", "0"))
                    item["bbox"] = _geometry_bbox(item)  # type: ignore[assignment]
                    geometries.append(item)

                lanes_by_side = {"left": [], "right": [], "center": []}
                for side in ("left", "right", "center"):
                    for lane in road.findall(f".//lanes/laneSection/{side}/lane"):
                        lanes_by_side[side].append(
                            {
                                "id": int(_safe_float(lane.get("id"), 0)),
                                "type": str(lane.get("type", "")),
                            }
                        )

                max_abs_curvature = max(abs(float(g.get("curvature", 0.0))) for g in geometries) if geometries else 0.0
                total_geometry_length = sum(float(g.get("length", 0.0)) for g in geometries)
                total_abs_heading_change = sum(
                    abs(float(g.get("curvature", 0.0)) * float(g.get("length", 0.0)))
                    for g in geometries
                )
                loop_closure_distance = math.inf
                if geometries:
                    first_point = _point_at_geometry_s(geometries[0], 0.0)
                    last_geom = geometries[-1]
                    last_point = _point_at_geometry_s(last_geom, float(last_geom.get("length", 0.0) or 0.0))
                    loop_closure_distance = math.hypot(first_point[0] - last_point[0], first_point[1] - last_point[1])
                road_entry = {
                    "id": road_id,
                    "junction": road.get("junction", "-1"),
                    "is_junction_road": road.get("junction", "-1") not in {"", "-1"} or road_id in junction_roads,
                    "junction_connection_count": len(set(junction_connections.get(str(road.get("junction", "-1")), []))),
                    "max_abs_curvature": max_abs_curvature,
                    "total_geometry_length": total_geometry_length,
                    "total_abs_heading_change": total_abs_heading_change,
                    "loop_closure_distance": loop_closure_distance,
                    "geometries": geometries,
                    "lanes_by_side": lanes_by_side,
                }
                index["roads"][road_id] = road_entry
                for signal in road.findall(".//signals/signal"):
                    sig_s = _safe_float(signal.get("s"), 0.0)
                    geom = None
                    for candidate in reversed(geometries):
                        if sig_s >= float(candidate.get("s", 0.0)):
                            geom = candidate
                            break
                    if geom is None and geometries:
                        geom = geometries[0]
                    point = None
                    if geom is not None:
                        point = _point_at_geometry_s(geom, sig_s - float(geom.get("s", 0.0)))
                    index["signals"].append(
                        {
                            "road_id": road_id,
                            "id": signal.get("id"),
                            "name": signal.get("name"),
                            "type": signal.get("type"),
                            "subtype": signal.get("subtype"),
                            "point": point,
                        }
                    )
            roundabout_junctions = set()
            roads_by_junction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for road in index["roads"].values():
                junction_id = str(road.get("junction", "-1"))
                if junction_id not in {"", "-1"}:
                    roads_by_junction[junction_id].append(road)
            for junction_id, roads in roads_by_junction.items():
                curved_roads = [r for r in roads if float(r.get("max_abs_curvature", 0.0) or 0.0) >= 0.035]
                if len(roads) < 6 or len(curved_roads) < 4:
                    continue
                points = []
                for road in roads:
                    for geom in road.get("geometries", []):
                        length = float(geom.get("length", 0.0) or 0.0)
                        if length <= 0.0:
                            continue
                        for local_s in (0.0, length * 0.5, length):
                            points.append(_point_at_geometry_s(geom, local_s))
                if len(points) < 12:
                    continue
                cx = sum(p[0] for p in points) / len(points)
                cy = sum(p[1] for p in points) / len(points)
                radii = [math.hypot(p[0] - cx, p[1] - cy) for p in points]
                mean_radius = sum(radii) / len(radii)
                if mean_radius < 8.0 or mean_radius > 55.0 or min(radii) < 3.5:
                    continue
                radius_std = math.sqrt(sum((r - mean_radius) ** 2 for r in radii) / len(radii))
                if radius_std / mean_radius > 0.42:
                    continue
                angles = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in points)
                gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
                gaps.append((angles[0] + 2.0 * math.pi) - angles[-1])
                angular_coverage_deg = math.degrees(2.0 * math.pi - max(gaps))
                if angular_coverage_deg < 300.0:
                    continue
                nearest_signal_to_center = math.inf
                for signal in index.get("signals", []):
                    point = signal.get("point")
                    if point is None:
                        continue
                    nearest_signal_to_center = min(nearest_signal_to_center, math.hypot(cx - point[0], cy - point[1]))
                if math.isfinite(nearest_signal_to_center) and nearest_signal_to_center <= 45.0:
                    continue
                roundabout_junctions.add(junction_id)
            index["roundabout_junctions"] = sorted(roundabout_junctions)
            signal_points = [signal.get("point") for signal in index.get("signals", []) if signal.get("point") is not None]
            for road in index["roads"].values():
                closed_loop_candidate = (
                    float(road.get("total_abs_heading_change", 0.0) or 0.0) >= 4.5
                    and float(road.get("total_geometry_length", 0.0) or 0.0) <= 320.0
                    and float(road.get("loop_closure_distance", math.inf) or math.inf) <= 45.0
                )
                nearest_signal_to_road = math.inf
                if closed_loop_candidate and signal_points:
                    road_points = []
                    for geom in road.get("geometries", []):
                        length = float(geom.get("length", 0.0) or 0.0)
                        if length <= 0.0:
                            continue
                        for local_s in (0.0, length * 0.25, length * 0.5, length * 0.75, length):
                            road_points.append(_point_at_geometry_s(geom, local_s))
                    for road_point in road_points:
                        nearest_signal_to_road = min(
                            nearest_signal_to_road,
                            min(math.hypot(road_point[0] - point[0], road_point[1] - point[1]) for point in signal_points),
                        )
                closed_loop_roundabout = closed_loop_candidate and not (
                    math.isfinite(nearest_signal_to_road) and nearest_signal_to_road <= 45.0
                )
                road["is_roundabout_road"] = (
                    str(road.get("junction", "-1")) in roundabout_junctions
                    or closed_loop_roundabout
                )
            index["available"] = True
        except Exception as exc:
            index["parse_error"] = str(exc)
        self._static_indexes[town] = index
        return index

    def _probe_static(self, town: str, ego_xy: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """普通 Python 可用的 XODR 静态几何探针。"""
        out: Dict[str, Any] = {"xodr_available": False, "xodr_source": "static_xodr", "xodr_topology_trusted": False}
        if ego_xy is None:
            return out
        index = self._load_static_index(town)
        if not index.get("available"):
            out["xodr_path"] = index.get("path")
            if index.get("parse_error"):
                out["parse_error"] = index.get("parse_error")
            return out

        best = None
        for road in index.get("roads", {}).values():
            for geom in road.get("geometries", []):
                bbox = geom.get("bbox")
                if best is not None and bbox is not None and _bbox_min_distance(ego_xy, bbox) > best["distance_m"]:
                    continue
                dist = _distance_to_geometry(ego_xy, geom)
                if best is None or dist < best["distance_m"]:
                    best = {"road": road, "distance_m": dist}
        if best is None:
            out["xodr_path"] = index.get("path")
            return out

        road = best["road"]
        lanes_by_side = road.get("lanes_by_side", {})
        left_driving = [lane for lane in lanes_by_side.get("left", []) if "driving" in lane.get("type", "").lower()]
        right_driving = [lane for lane in lanes_by_side.get("right", []) if "driving" in lane.get("type", "").lower()]
        parking_or_shoulder = any(
            ("parking" in lane.get("type", "").lower() or "shoulder" in lane.get("type", "").lower())
            for side_lanes in lanes_by_side.values()
            for lane in side_lanes
        )

        signal_distances = []
        for signal in index.get("signals", []):
            point = signal.get("point")
            if point is None:
                continue
            signal_distances.append(math.hypot(ego_xy[0] - point[0], ego_xy[1] - point[1]))
        nearest_signal = min(signal_distances) if signal_distances else math.inf

        same_dir_lanes = max(len(left_driving), len(right_driving), 1)
        has_opposite = bool(left_driving and right_driving)
        topology_trusted = best["distance_m"] <= 20.0
        junction_id = str(road.get("junction", "-1"))
        junction_connection_count = int(road.get("junction_connection_count", 0) or 0)
        max_abs_curvature = float(road.get("max_abs_curvature", 0.0) or 0.0)
        total_geometry_length = float(road.get("total_geometry_length", 0.0) or 0.0)
        roundabout_hint = topology_trusted and bool(road.get("is_roundabout_road", False))
        out.update(
            {
                "xodr_available": True,
                "xodr_path": index.get("path"),
                "xodr_topology_trusted": topology_trusted,
                "map_road_id": road.get("id"),
                "map_lane_id": None,
                "map_lane_type": "Driving" if left_driving or right_driving else "unknown",
                "map_is_junction": bool(road.get("is_junction_road")) and best["distance_m"] <= 12.0,
                "map_junction_id": road.get("junction"),
                "map_is_roundabout": roundabout_hint,
                "junction_connection_count": junction_connection_count,
                "road_max_abs_curvature": round(max_abs_curvature, 6),
                "road_geometry_length_m": round(total_geometry_length, 3),
                "road_total_abs_heading_change": round(float(road.get("total_abs_heading_change", 0.0) or 0.0), 6),
                "road_loop_closure_distance_m": (
                    round(float(road.get("loop_closure_distance")), 3)
                    if math.isfinite(float(road.get("loop_closure_distance", math.inf)))
                    else None
                ),
                "map_projection_error_m": round(best["distance_m"], 3),
                "nearest_signal_m": round(nearest_signal, 3) if math.isfinite(nearest_signal) else None,
                "lane_count_same_dir": same_dir_lanes,
                "has_opposite_driving_lane": has_opposite and topology_trusted,
                "has_parking_or_shoulder_nearby": parking_or_shoulder and topology_trusted,
                "ramp_merge_split_hint": topology_trusted and bool(road.get("is_junction_road")) and not (math.isfinite(nearest_signal) and nearest_signal <= 60.0),
            }
        )
        return out

    def probe(self, town: str, ego_xy: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"xodr_available": False, "xodr_topology_trusted": False}
        if ego_xy is None:
            return out
        map_obj = self._load_map(town)
        if map_obj is None or self._carla is None:
            return self._probe_static(town, ego_xy)
        try:
            loc = self._carla.Location(x=float(ego_xy[0]), y=float(ego_xy[1]), z=0.0)
            wp = map_obj.get_waypoint(loc, project_to_road=True)
            if wp is None:
                return out
            out.update(
                {
                    "xodr_available": True,
                    "xodr_source": "carla_api",
                    "xodr_topology_trusted": True,
                    "map_road_id": int(wp.road_id),
                    "map_lane_id": int(wp.lane_id),
                    "map_lane_type": str(wp.lane_type),
                    "map_is_junction": bool(wp.is_junction),
                    "map_junction_id": int(wp.junction_id),
                    "map_lane_width": float(wp.lane_width),
                }
            )

            lane_count_same_dir = 1
            has_opposite = False
            has_parking = "Parking" in str(wp.lane_type) or "Shoulder" in str(wp.lane_type)
            origin_positive = int(wp.lane_id) > 0
            for getter_name in ("get_left_lane", "get_right_lane"):
                cur = getattr(wp, getter_name)()
                depth = 0
                while cur is not None and depth < 5:
                    lane_type = str(cur.lane_type)
                    if "Parking" in lane_type or "Shoulder" in lane_type:
                        has_parking = True
                    if "Driving" not in lane_type:
                        break
                    if (int(cur.lane_id) > 0) == origin_positive:
                        lane_count_same_dir += 1
                    else:
                        has_opposite = True
                    cur = getattr(cur, getter_name)()
                    depth += 1

            out["lane_count_same_dir"] = lane_count_same_dir
            out["has_opposite_driving_lane"] = has_opposite
            out["has_parking_or_shoulder_nearby"] = has_parking
            out["ramp_merge_split_hint"] = bool(wp.is_junction and not _valid_traffic_light(out.get("traffic_light_state")))
            static_hint = self._probe_static(town, ego_xy)
            for key in (
                "map_is_roundabout",
                "junction_connection_count",
                "road_max_abs_curvature",
                "road_geometry_length_m",
                "nearest_signal_m",
            ):
                if key in static_hint:
                    out[key] = static_hint[key]
            return out
        except Exception:
            return out


def _project_route_s(point: Optional[Tuple[float, float]], waypoints: List[Tuple[float, float]]) -> Tuple[float, float]:
    """把点投影到 XML route polyline，返回 (s, distance)。"""
    if point is None or len(waypoints) < 2:
        return (math.nan, math.inf)
    px, py = point
    best_s = math.nan
    best_d = math.inf
    acc = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-6:
            continue
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / seg_len2))
        qx = ax + t * vx
        qy = ay + t * vy
        d = math.hypot(px - qx, py - qy)
        seg_len = math.sqrt(seg_len2)
        if d < best_d:
            best_d = d
            best_s = acc + t * seg_len
        acc += seg_len
    return (best_s, best_d)


def _route_length_m(waypoints: List[Tuple[float, float]]) -> float:
    if len(waypoints) < 2:
        return math.nan
    total = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _current_xml_weather(
    xml_info: Optional[RouteXmlInfo],
    route_s: float,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """按 route 百分比选择最接近当前帧的 XML weather。"""
    if xml_info is None or not xml_info.weathers:
        return {}, None
    route_len = _route_length_m(xml_info.waypoints)
    route_percent: Optional[float] = None
    if math.isfinite(route_s) and math.isfinite(route_len) and route_len > 1e-6:
        route_percent = max(0.0, min(100.0, route_s / route_len * 100.0))
        best = min(
            xml_info.weathers,
            key=lambda item: abs(_safe_float(item.get("route_percentage"), default=0.0) - route_percent),
        )
        return best, route_percent
    return xml_info.weathers[0], route_percent


def _low_visibility_junction_factor(weather: Dict[str, Any]) -> Tuple[float, List[str]]:
    """低能见度下压缩路口判定距离；1.0 表示不压缩。"""
    if not weather:
        return 1.0, []
    factor = 1.0
    reasons: List[str] = []
    precipitation = _safe_float(weather.get("precipitation"), default=0.0)
    fog = _safe_float(weather.get("fog_density"), default=0.0)
    sun_altitude = _safe_float(weather.get("sun_altitude_angle"), default=45.0)
    cloudiness = _safe_float(weather.get("cloudiness"), default=0.0)

    if sun_altitude <= 0.0:
        factor *= 0.78
        reasons.append("night")
    elif sun_altitude < 15.0:
        factor *= 0.90
        reasons.append("low_sun")

    if fog >= 50.0:
        factor *= 0.85
        reasons.append("heavy_fog")
    elif fog >= 20.0:
        factor *= 0.92
        reasons.append("fog")

    # 雨天 RGB 中红绿灯通常仍可见；单纯 precipitation 不压缩 R4/R5 范围。
    # 只有雨与夜间/低太阳/雾叠加时，才轻微增强低能见度收缩。
    if precipitation >= 60.0:
        if reasons:
            factor *= 0.96
            reasons.append("rain_compounds_low_visibility")
    elif precipitation >= 20.0 and reasons:
        factor *= 0.98
        reasons.append("rain_compounds_low_visibility")

    if cloudiness >= 80.0 and (fog > 0.0 or sun_altitude < 20.0):
        factor *= 0.95
        reasons.append("heavy_cloud")

    if not reasons:
        return 1.0, []
    return max(0.65, min(1.0, factor)), reasons


def _extract_ego_xy(meta: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = meta.get("pos_global")
    if isinstance(pos, np.ndarray):
        pos = pos.reshape(-1).tolist()
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        x = _safe_float(pos[0])
        y = _safe_float(pos[1])
        if math.isfinite(x) and math.isfinite(y):
            return (x, y)
    matrix = meta.get("ego_matrix")
    if isinstance(matrix, np.ndarray) and matrix.shape[0] >= 3 and matrix.shape[1] >= 4:
        return (_safe_float(matrix[0, 3]), _safe_float(matrix[1, 3]))
    return None


def _xml_numeric(info: Optional[RouteXmlInfo], key: str, default: float = math.nan) -> float:
    if info is None:
        return default
    vals = []
    for tag in info.scenario_tags:
        value = tag.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        f = _safe_float(value, default=math.nan)
        if math.isfinite(f):
            vals.append(f)
    return max(vals) if vals else default


def _nearest_trigger_distance(ego_xy: Optional[Tuple[float, float]], info: Optional[RouteXmlInfo]) -> float:
    if ego_xy is None or info is None:
        return math.inf
    distances = [math.hypot(ego_xy[0] - x, ego_xy[1] - y) for x, y in info.trigger_points]
    return min(distances) if distances else math.inf


def _actor_flow_distance(ego_xy: Optional[Tuple[float, float]], info: Optional[RouteXmlInfo]) -> float:
    """返回 ego 到 XML actor-flow 线段的最近距离，用于 merge 结构兜底。"""
    if ego_xy is None or info is None:
        return math.inf
    best = math.inf
    for tag in info.scenario_tags:
        start = tag.get("start_actor_flow")
        end = tag.get("end_actor_flow")
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        sx = _safe_float(start.get("x"))
        sy = _safe_float(start.get("y"))
        ex = _safe_float(end.get("x"))
        ey = _safe_float(end.get("y"))
        if not all(math.isfinite(v) for v in (sx, sy, ex, ey)):
            continue
        best = min(best, _distance_point_to_segment(ego_xy, (sx, sy), (ex, ey)))
    return best


def _route_trigger_window(route_s: float, trigger_s: float, pre_m: float, post_m: float) -> bool:
    if not math.isfinite(route_s) or not math.isfinite(trigger_s):
        return False
    return (trigger_s - pre_m) <= route_s <= (trigger_s + post_m)


def apply_rule_config_overrides(overrides: Dict[str, Dict[str, Any]]) -> None:
    """把人工调参后的场景阈值覆盖到运行时规则表。

    覆盖文件只允许修改已存在场景的 `SCENARIO_RULE_CONFIG` 字段，避免拼错场景名时静默新增规则。
    """
    for scenario, patch in (overrides or {}).items():
        if scenario not in SCENARIO_RULE_CONFIG:
            raise ValueError(f"未知场景规则覆盖: {scenario}")
        if not isinstance(patch, dict):
            raise ValueError(f"{scenario} 的规则覆盖必须是 object")
        SCENARIO_RULE_CONFIG[scenario].update(patch)


def load_rule_config_overrides(path: str) -> None:
    """从 JSON 文件加载每场景阈值覆盖，便于 smoke test 后快速调参。"""
    if not path:
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "scenarios" in payload:
        payload = payload["scenarios"]
    apply_rule_config_overrides(payload)


SCENARIO_RULE_KIND = {
    "Accident": "same_direction_obstacle",
    "AccidentTwoWays": "twoways_obstacle",
    "BlockedIntersection": "blocked_intersection",
    "ConstructionObstacle": "same_direction_obstacle",
    "ConstructionObstacleTwoWays": "twoways_obstacle",
    "ControlLoss": "default_meta_map",
    "CrossingBicycleFlow": "default_meta_map",
    "CrossJunctionDefectTrafficLight": "defect_junction",
    "DynamicObjectCrossing": "default_meta_map",
    "EnterActorFlow": "highway_merge",
    "EnterActorFlowV2": "highway_merge",
    "HardBreakRoute": "hardbreak_route",
    "HazardAtSideLane": "default_meta_map",
    "HazardAtSideLaneTwoWays": "twoways_obstacle",
    "HighwayCutIn": "highway_merge",
    "HighwayExit": "highway_merge",
    "InterurbanActorFlow": "interurban",
    "InterurbanAdvancedActorFlow": "interurban_advanced",
    "InvadingTurn": "invading_turn",
    "MergerIntoSlowTraffic": "highway_merge",
    "MergerIntoSlowTrafficV2": "highway_merge",
    "NonSignalizedJunctionLeftTurn": "nonsignalized_junction",
    "NonSignalizedJunctionLeftTurnEnterFlow": "nonsignalized_junction",
    "NonSignalizedJunctionRightTurn": "nonsignalized_junction",
    "noScenarios": "noscenario",
    "OppositeVehicleRunningRedLight": "signalized_junction",
    "OppositeVehicleTakingPriority": "nonsignalized_junction",
    "ParkedObstacle": "same_direction_obstacle",
    "ParkedObstacleTwoWays": "twoways_obstacle",
    "ParkingCrossingPedestrian": "parking",
    "ParkingCutIn": "parking",
    "ParkingExit": "parking_exit",
    "PedestrianCrossing": "pedestrian_crossing",
    "PriorityAtJunction": "nonsignalized_junction",
    "RedLightWithoutLeadVehicle": "signalized_junction",
    "SignalizedJunctionLeftTurn": "signalized_junction",
    "SignalizedJunctionLeftTurnEnterFlow": "signalized_junction",
    "SignalizedJunctionRightTurn": "signalized_junction",
    "StaticCutIn": "static_cutin",
    "T_Junction": "signalized_junction",
    "VehicleOpensDoorTwoWays": "vehicle_opens_door_twoways",
    "VehicleTurningRoute": "vehicle_turning",
    "VehicleTurningRoutePedestrian": "vehicle_turning",
}


SCENARIO_RULE_CONFIG: Dict[str, Dict[str, Any]] = {
    # 同向静态障碍：只让灯态/受控路口提升到 R4，障碍距离不改变 RS。
    "Accident": {"kind": "same_direction_obstacle", "junction_pre_m": 54, "junction_post_m": 22, "junction_tighten_factor": 0.85, "veto": ["no_r2"]},
    "ConstructionObstacle": {"kind": "same_direction_obstacle", "junction_pre_m": 42, "junction_post_m": 25, "junction_tighten_factor": 0.85, "veto": ["no_r2"]},
    "ParkedObstacle": {"kind": "same_direction_obstacle", "junction_pre_m": 72, "junction_post_m": 25, "junction_tighten_factor": 0.85, "veto": ["parked_not_parking_rs"]},
    # TwoWays：道路空间按“可行驶通道”等效对向单车道处理；障碍前后普通行驶回 R2，路口回 R4/R5。
    "AccidentTwoWays": {"kind": "twoways_obstacle", "junction_pre_m": 50, "junction_post_m": 20, "junction_tighten_factor": 0.85, "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "ConstructionObstacleTwoWays": {"kind": "twoways_obstacle", "junction_pre_m": 42, "junction_tighten_factor": 0.70, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "HazardAtSideLaneTwoWays": {"kind": "twoways_obstacle", "junction_tighten_factor": 0.75, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "two_way_min_pre_m": 75, "two_way_post_pad_m": 20, "trigger_close_m": 75, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 20, "two_way_approach_obstacle_m": 30, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "ParkedObstacleTwoWays": {"kind": "twoways_obstacle", "junction_tighten_factor": 0.75, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "two_way_min_pre_m": 55, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "InvadingTurn": {"kind": "invading_turn", "two_way_min_pre_m": 80, "two_way_post_pad_m": 20, "trigger_close_m": 75, "rule_note": "passive_oncoming_invasion"},
    # 阻塞路口：阻塞是 EVENT；RS 由路口控制源决定，STOP/无灯路口不能默认 R4。
    "BlockedIntersection": {"kind": "blocked_intersection", "junction_pre_m": 32, "junction_post_m": 18, "rule_note": "blocked_is_event_not_rs"},
    "OppositeVehicleRunningRedLight": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 20, "rule_note": "violation_not_r5"},
    "RedLightWithoutLeadVehicle": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 14, "scenario_active_signal_max_m": 52},
    "SignalizedJunctionLeftTurn": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 25},
    "SignalizedJunctionLeftTurnEnterFlow": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 25, "initial_weak_r4_towns": ["town01", "town02"], "initial_weak_r4_trigger_m": 22, "veto": ["enter_flow_not_r3"]},
    "SignalizedJunctionRightTurn": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    "T_Junction": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 32, "junction_tighten_factor": 0.80, "junction_min_post_m": 8.0, "review_if_no_tl": True},
    # 无灯/路权/故障路口。
    "CrossJunctionDefectTrafficLight": {"kind": "defect_junction", "junction_pre_m": 60, "junction_post_m": 20, "junction_tighten_factor": 0.65, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "rule_note": "signalized_rs_with_defect_event"},
    "NonSignalizedJunctionLeftTurn": {"kind": "nonsignalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    "NonSignalizedJunctionLeftTurnEnterFlow": {"kind": "nonsignalized_junction", "junction_pre_m": 84, "junction_post_m": 20, "veto": ["enter_flow_not_r3"]},
    "NonSignalizedJunctionRightTurn": {"kind": "nonsignalized_junction", "junction_pre_m": 63, "junction_post_m": 20, "junction_tighten_factor": 0.75, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "rightturn_core_pre_m": 45.0, "rightturn_core_post_m": 5.0, "rightturn_core_dist_to_junction_m": 18.0, "rightturn_core_trigger_m": 12.0},
    "OppositeVehicleTakingPriority": {"kind": "nonsignalized_junction", "junction_pre_m": 75, "junction_post_m": 20, "junction_tighten_factor": 0.60},
    "PriorityAtJunction": {"kind": "nonsignalized_junction", "junction_pre_m": 50, "junction_post_m": 20, "junction_tighten_factor": 0.65, "junction_lock_signal_protect_m": 70, "priority_signal_pre_frames": 4, "junction_regular_gap_merge_max_frames": 28, "junction_regular_gap_min_neighbor_frames": 4},
    # R3 高速/匝道/合流。
    "EnterActorFlow": {"kind": "highway_merge", "merge_pre_m": 24, "merge_post_m": 36, "trigger_close_m": 75, "actor_flow_near_m": 40, "highway_default_r3": False},
    "EnterActorFlowV2": {"kind": "highway_merge", "merge_pre_m": 24, "merge_post_m": 36, "trigger_close_m": 75, "actor_flow_near_m": 40, "highway_default_r3": False},
    "HighwayCutIn": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 40, "trigger_close_m": 90, "highway_default_r3": True},
    "HighwayExit": {"kind": "highway_merge", "merge_pre_m": 50, "merge_post_m": 50, "trigger_close_m": 90, "highway_default_r3": True},
    "MergerIntoSlowTraffic": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True, "actor_flow_near_m": 20, "highway_default_r3": True},
    "MergerIntoSlowTrafficV2": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True, "actor_flow_near_m": 20, "highway_default_r3": True},
    "InterurbanActorFlow": {"kind": "interurban", "merge_pre_m": 50, "merge_post_m": 45, "junction_pre_m": 55, "junction_post_m": 25},
    "InterurbanAdvancedActorFlow": {"kind": "interurban_advanced", "junction_pre_m": 72, "junction_post_m": 33, "r3_requires_topology": True},
    # 停车/路边占道。
    "ParkingCrossingPedestrian": {"kind": "parking", "parking_pre_m": 35, "parking_post_m": 60, "veto": ["pedestrian_not_rs"]},
    "ParkingCutIn": {"kind": "parking", "parking_pre_m": 30, "parking_post_m": 50},
    "ParkingExit": {"kind": "parking_exit", "parking_pre_m": 20, "parking_post_m": 60, "rule_note": "parking_to_driving_transition"},
    "VehicleOpensDoorTwoWays": {"kind": "vehicle_opens_door_twoways", "junction_tighten_factor": 0.85, "two_way_min_pre_m": 55, "two_way_post_pad_m": 20, "parking_pre_m": 35, "parking_post_m": 55},
    "StaticCutIn": {"kind": "static_cutin", "junction_tighten_factor": 0.70, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "parking_pre_m": 35, "parking_post_m": 55, "merge_pre_m": 35, "merge_post_m": 55},
    # 按道路空间拆分的横穿/转弯/普通场景。
    "PedestrianCrossing": {"kind": "pedestrian_crossing", "junction_pre_m": 36, "junction_post_m": 60, "junction_tighten_factor": 0.70, "junction_min_pre_m": 12.0, "junction_min_post_m": 5.0, "junction_regular_gap_merge_max_frames": 20, "junction_regular_gap_min_neighbor_frames": 4, "pedestrian_exit_tail_frames": 6, "veto": ["pedestrian_not_rs"]},
    "VehicleTurningRoute": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 20, "junction_tighten_factor": 0.65, "junction_min_pre_m": 8.0, "junction_min_post_m": 3.0, "turning_tl_requires_local_junction": True, "turning_trigger_core_m": 8.0, "turning_nosignal_trigger_core_m": 5.0, "multi_trigger": True},
    "VehicleTurningRoutePedestrian": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 40, "junction_tighten_factor": 0.60, "junction_min_pre_m": 8.0, "junction_min_post_m": 3.0, "turning_tl_requires_local_junction": True, "turning_trigger_core_m": 8.0, "turning_nosignal_trigger_core_m": 5.0, "turning_final_regular_gap_max_frames": 6, "turning_final_regular_gap_min_neighbor_frames": 4, "veto": ["pedestrian_not_rs"]},
    "CrossingBicycleFlow": {"kind": "default_meta_map", "junction_pre_m": 35, "junction_post_m": 25, "junction_tighten_factor": 0.90, "veto": ["actor_flow_not_r3"]},
    "DynamicObjectCrossing": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "junction_tighten_factor": 0.65, "junction_min_pre_m": 8.0, "junction_min_post_m": 2.5, "veto": ["crossing_event_not_rs"]},
    "ControlLoss": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "junction_tighten_factor": 0.60, "junction_min_pre_m": 8.0, "junction_min_post_m": 3.0, "veto": ["control_loss_not_rs"]},
    "HardBreakRoute": {"kind": "hardbreak_route", "junction_pre_m": 50, "junction_post_m": 25, "junction_tighten_factor": 0.80, "junction_min_pre_m": 10.0, "junction_min_post_m": 3.0, "veto": ["brake_not_rs"]},
    "HazardAtSideLane": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "junction_tighten_factor": 0.90, "veto": ["side_lane_not_twoways"]},
    "noScenarios": {"kind": "noscenario", "junction_pre_m": 50, "junction_post_m": 25, "conservative": True},
}


JUNCTION_PRE_WINDOW_SCALE = 0.36
JUNCTION_POST_WINDOW_SCALE = 0.28
JUNCTION_PRE_WINDOW_MIN_M = 16.0
JUNCTION_POST_WINDOW_MIN_M = 5.0
JUNCTION_META_NEAR_M = 35.0
JUNCTION_STRONG_MAX_M = 22.0
STATIC_SIGNAL_NEAR_M = 35.0
JUNCTION_CLOSE_TRIGGER_MAX_M = 25.0
RE2_EXIT_CENTER_TOLERANCE_SCALE = 1.10
RE2_EXIT_STABLE_FUTURE_FRAMES = 2


def _shrink_junction_window(pre_m: float, post_m: float) -> Tuple[float, float]:
    """收紧路口影响区，避免十字路口标签过早/过晚覆盖普通路段。"""
    pre = max(JUNCTION_PRE_WINDOW_MIN_M, float(pre_m) * JUNCTION_PRE_WINDOW_SCALE)
    post = max(JUNCTION_POST_WINDOW_MIN_M, float(post_m) * JUNCTION_POST_WINDOW_SCALE)
    return pre, post


def _diagnose_rs_decision(
    scenario_name: str,
    kind: str,
    primary: RoadStructure,
    scores: Dict[RoadStructure, float],
    rules: List[str],
    xml_info: Optional[RouteXmlInfo],
    xodr: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    """给单帧 RS 决策生成可追责诊断，方便定位 XML/XODR/meta 哪侧没用好。"""
    used_inputs = {
        "scenario_prior": scenario_name in SCENARIO_TO_ROAD_STRUCTURE,
        "xml_matched": xml_info is not None,
        "xodr_runtime_available": bool(xodr.get("xodr_available", False)),
        "xodr_topology_trusted": bool(xodr.get("xodr_topology_trusted", xodr.get("xodr_available", False))),
        "meta_traffic_light_valid": bool(flags.get("has_tl")),
        "meta_light_hazard": bool(flags.get("light_hazard")),
        "bbox_traffic_light": bool(flags.get("bbox_traffic_light")),
        "meta_junction_hint": bool(flags.get("is_junction")) or bool(flags.get("dist_to_junction_near")),
        "bbox_junction_hint": bool(flags.get("bbox_junction_hint")),
        "xodr_roundabout_hint": bool(flags.get("map_is_roundabout")),
        "meta_active_scenario": bool(flags.get("scenario_active")),
        "meta_stop_hint": bool(flags.get("stop_hazard")),
        "bbox_stop_or_yield_hint": bool(flags.get("bbox_stop_or_yield")),
    }

    weak_or_missing = []
    if not used_inputs["xml_matched"]:
        weak_or_missing.append("xml_not_matched_to_route")
    if not used_inputs["xodr_runtime_available"]:
        weak_or_missing.append("xodr_runtime_probe_unavailable")
    elif not used_inputs["xodr_topology_trusted"]:
        weak_or_missing.append("xodr_topology_untrusted")
    if flags.get("route_projection_error_high"):
        weak_or_missing.append("xml_route_projection_error_high")
    if kind in {"signalized_junction", "defect_junction"} and not (
        used_inputs["meta_traffic_light_valid"] or used_inputs["meta_light_hazard"] or used_inputs["bbox_traffic_light"] or used_inputs["xodr_runtime_available"]
    ):
        weak_or_missing.append("signalized_policy_lacks_meta_light_and_xodr")
    if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"} and not bool(xodr.get("has_opposite_driving_lane", False)):
        weak_or_missing.append("r2_lacks_xodr_opposite_lane_confirmation")
    if kind in {"parking", "parking_exit", "static_cutin", "vehicle_opens_door_twoways"} and not bool(xodr.get("has_parking_or_shoulder_nearby", False)):
        weak_or_missing.append("parking_or_curbside_context_unconfirmed")
    if kind in {"highway_merge", "interurban", "interurban_advanced", "static_cutin"} and not bool(xodr.get("ramp_merge_split_hint", False)):
        weak_or_missing.append("r3_lacks_xodr_merge_split_confirmation")

    if primary == RoadStructure.R4:
        if "r4_tl_confirmed" in rules:
            decision_source = "meta_traffic_light"
        elif "r4_bbox_traffic_light_confirmed" in rules:
            decision_source = "bbox_traffic_light"
        elif "r4_meta_tl_without_strong_context_review" in rules:
            decision_source = "meta_traffic_light_without_strong_context"
        elif "r4_light_hazard" in rules:
            decision_source = "meta_light_hazard"
        else:
            decision_source = "scenario_xml_or_junction_window"
    elif primary == RoadStructure.R5:
        decision_source = "defect_or_nonsignalized_junction_window"
    elif primary == RoadStructure.R3:
        decision_source = "merge_actor_flow_or_topology_window"
    elif primary == RoadStructure.R2:
        if "r2_layout_xodr_effective_twoway_confirmed" in rules:
            decision_source = "xodr_effective_twoway_layout"
        else:
            decision_source = "twoways_trigger_window"
    else:
        decision_source = "conservative_default_or_outside_special_window"

    sorted_scores = sorted(((rs.value, score) for rs, score in scores.items()), key=lambda item: item[1], reverse=True)
    score_gap = None
    if len(sorted_scores) >= 2:
        score_gap = round(sorted_scores[0][1] - sorted_scores[1][1], 3)

    checks = []
    if primary == RoadStructure.R1:
        checks.extend(["检查 trigger_distance/route_progress 是否落在特殊窗口外", "检查 meta 灯态或 active_scenario 是否缺失"])
        if flags.get("map_is_roundabout"):
            checks.append("XODR 判定为 roundabout，按规则保留 R1 而不是 R4/R5")
    if primary in {RoadStructure.R4, RoadStructure.R5}:
        checks.extend(["检查 meta traffic_light_state/light_hazard 是否可信", "检查 XODR junction/signal/controller 与 XML trigger 是否同源"])
        if not flags.get("strong_control_context"):
            checks.append("当前缺少 strong_control_context，应优先按 RGB 核查是否真有路口控制区")
    if primary == RoadStructure.R2:
        checks.extend(["检查 XODR 是否找到相邻对向 driving lane", "检查 same_dir_lanes 是否被地图误判为足够绕行"])
        if flags.get("twoway_core_obstruction"):
            checks.append("R2 由近距离障碍/车辆交互 meta 支撑，需核查 RGB 是否确有占道或借对向需求")
        if flags.get("twoway_xml_core_confirmed"):
            checks.append("R2 由 XML trigger 极近或 XML 场景障碍近距离召回，需核查 RGB 是否正处于核心绕障区")
    if primary == RoadStructure.R3:
        checks.extend(["检查 XML actor-flow/other_actor_location 是否投影到正确窗口", "检查 XODR ramp_merge_split_hint 是否过弱"])
    return {
        "decision_source": decision_source,
        "used_inputs": used_inputs,
        "weak_or_missing_inputs": weak_or_missing,
        "window_flags": {
            "close_trigger": bool(flags.get("close_trigger")),
            "close_trigger_for_junction": bool(flags.get("close_trigger_for_junction")),
            "defect_local_control_context": bool(flags.get("defect_local_control_context")),
            "turning_local_junction_context": bool(flags.get("turning_local_junction_context")),
            "turning_local_junction_evidence": bool(flags.get("turning_local_junction_evidence")),
            "near_junction": bool(flags.get("near_junction")),
            "strong_control_context": bool(flags.get("strong_control_context")),
            "static_signal_near": bool(flags.get("static_signal_near")),
            "junction_window": bool(flags.get("junction_window")),
            "roundabout_context": bool(flags.get("map_is_roundabout")),
            "two_way_window": bool(flags.get("two_way_window")),
            "two_way_layout_prior": bool(flags.get("two_way_layout_prior")),
            "layout_r2_enabled": bool(flags.get("layout_r2_enabled")),
            "effective_twoway_drivable_layout": bool(flags.get("effective_twoway_drivable_layout")),
            "layout_effective_twoway_drivable": bool(flags.get("layout_effective_twoway_drivable")),
            "twoway_core_obstruction": bool(flags.get("twoway_core_obstruction")),
            "twoway_strict_core_obstruction": bool(flags.get("twoway_strict_core_obstruction")),
            "twoway_xml_core_close": bool(flags.get("twoway_xml_core_close")),
            "twoway_xml_obstacle_close": bool(flags.get("twoway_xml_obstacle_close")),
            "twoway_xml_core_confirmed": bool(flags.get("twoway_xml_core_confirmed")),
            "scenario_active": bool(flags.get("scenario_active")),
        },
        "score_ranking": sorted_scores,
        "top_score_gap": score_gap,
        "if_this_frame_is_wrong_check": checks,
    }


class RoadEventRuleEngine:
    """把 scenario 先验、RS、meta/XML 证据合成逐帧 EVENT 标签。"""

    @staticmethod
    def _scenario_active(frame_data: Dict[str, Any], scenario_name: str) -> bool:
        active = str(frame_data.get("current_active_scenario_type") or "")
        previous = str(frame_data.get("previous_active_scenario_type") or "")
        return scenario_name in {active, previous}

    @staticmethod
    def _near_trigger(evidence: Dict[str, Any], max_m: float = 45.0) -> bool:
        return _safe_float(evidence.get("trigger_distance_m"), default=math.inf) <= max_m

    @staticmethod
    def _speed(frame_data: Dict[str, Any]) -> float:
        return _safe_float(frame_data.get("speed"), default=0.0)

    @staticmethod
    def _target_speed(frame_data: Dict[str, Any]) -> float:
        return _safe_float(frame_data.get("target_speed"), default=math.inf)

    @staticmethod
    def _route_lateral_offset(frame_data: Dict[str, Any]) -> float:
        """用 ego-frame route 的近前方局部切线估计自车相对目标中心线的位置。"""
        route = frame_data.get("route")
        if route is None:
            return math.inf
        try:
            arr = np.asarray(route, dtype=float)
        except (TypeError, ValueError):
            return math.inf
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            return math.inf
        xs = arr[:, 0]
        ys = arr[:, 1]
        finite = np.isfinite(xs) & np.isfinite(ys)
        ahead = finite & (xs >= 1.0) & (xs <= 8.0)
        if not np.any(ahead):
            ahead = finite & (xs >= 1.0)
        if not np.any(ahead):
            return math.inf
        candidates = arr[ahead, :2]
        order = np.argsort(candidates[:, 0])
        candidates = candidates[order]
        anchor = candidates[0]
        if len(candidates) >= 2:
            far_idx = min(len(candidates) - 1, 3)
            tangent = candidates[far_idx] - anchor
        else:
            tangent = np.array([1.0, 0.0], dtype=float)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-6:
            return float(anchor[1])
        tangent = tangent / norm
        origin_delta = -anchor
        # 2D cross(tangent, origin-anchor): signed perpendicular offset to the local centerline.
        return float(tangent[0] * origin_delta[1] - tangent[1] * origin_delta[0])

    @staticmethod
    def _lane_center_tolerance(frame_data: Dict[str, Any]) -> float:
        lane_width = _safe_float(frame_data.get("target_lane_width"), default=math.inf)
        if not math.isfinite(lane_width):
            lane_width = _safe_float(frame_data.get("ego_lane_width"), default=3.5)
        if not math.isfinite(lane_width) or lane_width <= 0.0:
            lane_width = 3.5
        return max(0.45, min(0.75, lane_width * 0.18))

    @staticmethod
    def _hard_decel(frame_data: Dict[str, Any]) -> bool:
        accel_x = _safe_float(frame_data.get("accel_x"), default=0.0)
        brake = _safe_bool(frame_data.get("brake", False))
        speed = RoadEventRuleEngine._speed(frame_data)
        target_speed = RoadEventRuleEngine._target_speed(frame_data)
        target_drop = math.isfinite(target_speed) and target_speed <= max(0.5, speed * 0.45)
        return brake or accel_x <= -3.5 or target_drop

    @staticmethod
    def _changed_route(frame_data: Dict[str, Any]) -> bool:
        if _safe_bool(frame_data.get("changed_route", False)):
            return True
        signed = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
        return math.isfinite(signed) and signed <= 3.5

    @staticmethod
    def _route_centered(frame_data: Dict[str, Any]) -> bool:
        route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
        if not math.isfinite(route_lateral_offset):
            return False
        return abs(route_lateral_offset) <= RoadEventRuleEngine._lane_center_tolerance(frame_data)

    @staticmethod
    def _target_lane_change_active(frame_data: Dict[str, Any]) -> bool:
        route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
        route_abs = abs(route_lateral_offset) if math.isfinite(route_lateral_offset) else math.inf
        tol = RoadEventRuleEngine._lane_center_tolerance(frame_data)
        signed = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
        changed_route = _safe_bool(frame_data.get("changed_route", False))
        route_offset_active = math.isfinite(route_abs) and route_abs > tol
        signed_active = math.isfinite(signed) and signed <= 3.5
        if math.isfinite(route_abs):
            return route_offset_active and (changed_route or signed_active or route_abs >= tol + 0.35)
        return changed_route or signed_active

    @staticmethod
    def _noscenario_target_lane_change_active(frame_data: Dict[str, Any]) -> bool:
        """noScenarios 没有场景先验，RE2 必须依赖显式换道轨迹证据。"""
        route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
        route_abs = abs(route_lateral_offset) if math.isfinite(route_lateral_offset) else math.inf
        tol = RoadEventRuleEngine._lane_center_tolerance(frame_data)
        signed = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
        changed_route = _safe_bool(frame_data.get("changed_route", False))
        lane_change_str = str(frame_data.get("lane_change_str", "") or "").upper()
        explicit_cmd = "CHANGELANE" in lane_change_str or lane_change_str in {"LEFT", "RIGHT"}
        route_offset_active = math.isfinite(route_abs) and route_abs > max(tol, 0.18)
        signed_active = math.isfinite(signed) and signed <= 3.5
        if math.isfinite(route_abs):
            return route_offset_active and (changed_route or signed_active or explicit_cmd)
        return changed_route or signed_active or explicit_cmd

    @staticmethod
    def _signed_lane_change_intent(frame_data: Dict[str, Any]) -> bool:
        signed = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
        return math.isfinite(signed) and signed <= 2.0

    @staticmethod
    def _highway_exit_ramp_transition_active(frame_data: Dict[str, Any]) -> bool:
        commands = frame_data.get("next_commands")
        if not isinstance(commands, (list, tuple)) or not commands:
            return False
        try:
            first_command = int(commands[0])
        except (TypeError, ValueError):
            return False
        return first_command == 3

    @staticmethod
    def _highway_r3_core_event_active(
        scenario_name: str,
        frame_data: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """R3 是道路空间；匝道进入/驶出/合流过渡段输出 R-E3。"""
        if scenario_name == "HighwayCutIn":
            return False, ["event_highway_cutin_regular_follow_default"], {"highway_r3_core_active": False}
        if scenario_name == "HighwayExit":
            command3_active = RoadEventRuleEngine._highway_exit_ramp_transition_active(frame_data)
            trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
            actor_flow_distance = _safe_float(evidence.get("actor_flow_distance_m"), default=math.inf)
            scenario_active = _safe_bool(frame_data.get("scenario_active", False))
            active = command3_active or (
                scenario_active
                and (
                    trigger_distance <= 90.0
                    or actor_flow_distance <= 18.0
                )
            )
            rules = []
            if command3_active:
                rules.append("event_highway_exit_ramp_transition_r3")
            if active and not command3_active:
                rules.append("event_highway_exit_approach_or_ramp_r3")
            if not rules:
                rules.append("event_highway_exit_regular_follow_default")
            return active, (
                rules
            ), {
                "highway_r3_core_active": active,
                "highway_exit_command3_transition": command3_active,
                "highway_exit_trigger_distance_m": trigger_distance if math.isfinite(trigger_distance) else None,
                "highway_exit_actor_flow_distance_m": actor_flow_distance if math.isfinite(actor_flow_distance) else None,
            }

        trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
        actor_flow_distance = _safe_float(evidence.get("actor_flow_distance_m"), default=math.inf)
        xodr = evidence.get("xodr") or {}
        ramp_hint = bool(xodr.get("ramp_merge_split_hint", False))
        scenario_active = _safe_bool(frame_data.get("scenario_active", False))

        trigger_core_m = {
            "EnterActorFlow": 16.0,
            "EnterActorFlowV2": 16.0,
            "MergerIntoSlowTraffic": 0.0,
            "MergerIntoSlowTrafficV2": 0.0,
        }.get(scenario_name, 20.0)
        actor_core_m = {
            "EnterActorFlow": 40.0,
            "EnterActorFlowV2": 40.0,
            "MergerIntoSlowTraffic": 32.0,
            "MergerIntoSlowTrafficV2": 32.0,
        }.get(scenario_name, math.inf)
        actor_trigger_guard_m = {
            "EnterActorFlow": 140.0,
            "EnterActorFlowV2": 140.0,
            "MergerIntoSlowTraffic": 90.0,
            "MergerIntoSlowTrafficV2": 90.0,
        }.get(scenario_name, math.inf)
        active_scenario_guard_m = {
            "EnterActorFlow": 220.0,
            "EnterActorFlowV2": 220.0,
            "MergerIntoSlowTraffic": 0.0,
            "MergerIntoSlowTrafficV2": 0.0,
        }.get(scenario_name, math.inf)

        trigger_core = trigger_distance <= trigger_core_m
        actor_core = (
            math.isfinite(actor_core_m)
            and actor_flow_distance <= actor_core_m
            and trigger_distance <= actor_trigger_guard_m
        )
        if scenario_name == "noScenarios":
            ramp_core = ramp_hint
        elif scenario_name in {"MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
            ramp_core = ramp_hint and actor_flow_distance <= max(actor_core_m, 35.0)
        else:
            ramp_core = ramp_hint and trigger_distance <= max(trigger_core_m, 25.0)
        if scenario_name in {"MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
            scenario_merge_approach = (
                scenario_active
                and math.isfinite(actor_core_m)
                and actor_flow_distance <= actor_core_m
            )
        else:
            scenario_merge_approach = (
                scenario_active
                and math.isfinite(active_scenario_guard_m)
                and (
                    trigger_distance <= active_scenario_guard_m
                    or actor_flow_distance <= actor_core_m
                )
            )
        active = trigger_core or actor_core or ramp_core or scenario_merge_approach

        rules = []
        if trigger_core:
            rules.append("event_highway_trigger_core_r3")
        if actor_core:
            rules.append("event_highway_actor_flow_core_r3")
        if ramp_core:
            rules.append("event_highway_xodr_ramp_core_r3")
        if scenario_merge_approach:
            rules.append("event_highway_merge_approach_r3")
        if not rules:
            rules.append("event_highway_r3_space_regular_follow")
        return active, rules, {
            "highway_r3_core_active": active,
            "highway_trigger_core_m": trigger_core_m,
            "highway_actor_flow_core_m": None if math.isinf(actor_core_m) else actor_core_m,
            "highway_actor_trigger_guard_m": None if math.isinf(actor_trigger_guard_m) else actor_trigger_guard_m,
            "highway_active_scenario_guard_m": None if math.isinf(active_scenario_guard_m) else active_scenario_guard_m,
            "highway_trigger_distance_m": trigger_distance if math.isfinite(trigger_distance) else None,
            "highway_actor_flow_distance_m": actor_flow_distance if math.isfinite(actor_flow_distance) else None,
            "highway_ramp_merge_split_hint": ramp_hint,
            "highway_scenario_merge_approach": scenario_merge_approach,
        }

    @staticmethod
    def _regular_event_details(
        scenario_name: str,
        primary_rs: RoadStructure,
        frame_data: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[EventType, List[str], Dict[str, Any]]:
        if primary_rs == RoadStructure.R4:
            return EventType.R_E4, ["event_regular_by_r4_signalized_junction"], {}
        if primary_rs == RoadStructure.R5:
            return EventType.R_E5, ["event_regular_by_r5_nonsignalized_or_defect_junction"], {}
        if primary_rs == RoadStructure.R3:
            core_active, core_rules, core_metrics = RoadEventRuleEngine._highway_r3_core_event_active(
                scenario_name,
                frame_data,
                evidence,
            )
            if scenario_name == "noScenarios":
                lane_change_active = RoadEventRuleEngine._noscenario_target_lane_change_active(frame_data)
            else:
                lane_change_active = RoadEventRuleEngine._target_lane_change_active(frame_data)
            if scenario_name in {"EnterActorFlow", "EnterActorFlowV2"} and not lane_change_active:
                actor_flow_distance = _safe_float(evidence.get("actor_flow_distance_m"), default=math.inf)
                signed = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
                route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
                route_abs = abs(route_lateral_offset) if math.isfinite(route_lateral_offset) else math.inf
                lane_change_active = (
                    RoadEventRuleEngine._changed_route(frame_data)
                    and math.isfinite(actor_flow_distance)
                    and actor_flow_distance <= 8.0
                    and math.isfinite(signed)
                    and signed <= 2.0
                    and math.isfinite(route_abs)
                    and route_abs >= 0.08
                )
            if lane_change_active:
                metrics = dict(core_metrics)
                metrics["highway_lane_change_regular"] = True
                return EventType.R_E2, ["event_highway_target_lane_change_r2", *core_rules], metrics
            if core_active:
                return EventType.R_E3, core_rules, core_metrics
            return EventType.R_E1, core_rules, core_metrics
        if scenario_name in {"HighwayCutIn", "HighwayExit", "InterurbanActorFlow", "ParkingExit", "StaticCutIn"}:
            if RoadEventRuleEngine._target_lane_change_active(frame_data):
                return EventType.R_E2, ["event_regular_target_lane_change_r2"], {}
        if primary_rs in {RoadStructure.R1, RoadStructure.R2} and (
            RoadEventRuleEngine._noscenario_target_lane_change_active(frame_data)
            if scenario_name == "noScenarios"
            else RoadEventRuleEngine._target_lane_change_active(frame_data)
        ):
            cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
            xodr = evidence.get("xodr") or {}
            trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
            if cfg.get("kind") == "same_direction_obstacle":
                field_cfg = OBSTACLE_EVENT_DISTANCE_FIELDS.get(scenario_name)
                field, threshold = field_cfg if field_cfg else ("", 0.0)
                specific_obstacle_close = bool(field) and _safe_float(frame_data.get(field), default=math.inf) <= threshold + 5.0
                xodr_multileg_curved_junction = (
                    bool(xodr.get("map_is_junction"))
                    and int(xodr.get("junction_connection_count", 0) or 0) >= 5
                    and _safe_float(xodr.get("road_total_abs_heading_change"), default=0.0) >= 1.0
                    and _safe_float(xodr.get("nearest_signal_m"), default=math.inf) > 35.0
                )
                roundabout_or_curved_junction = bool(xodr.get("map_is_roundabout")) or (
                    bool(xodr.get("map_is_junction"))
                    and bool(xodr.get("ramp_merge_split_hint"))
                    and _safe_float(xodr.get("nearest_signal_m"), default=math.inf) > 60.0
                ) or xodr_multileg_curved_junction
                if roundabout_or_curved_junction and not specific_obstacle_close and trigger_distance > 70.0:
                    return EventType.R_E1, ["event_regular_lane_change_suppressed_in_roundabout_like_junction"], {}
            return EventType.R_E2, ["event_regular_target_lane_change_r2"], {}
        return EventType.R_E1, ["event_regular_by_road_structure"], {}

    @staticmethod
    def _regular_event(
        scenario_name: str,
        primary_rs: RoadStructure,
        frame_data: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> EventType:
        event, _rules, _metrics = RoadEventRuleEngine._regular_event_details(
            scenario_name,
            primary_rs,
            frame_data,
            evidence,
        )
        return event

    @staticmethod
    def _accident_twoways_r2_overlay_active(
        scenario_name: str,
        primary_rs: RoadStructure,
        evidence: Dict[str, Any],
    ) -> bool:
        """AccidentTwoWays 中 R4/R5 可与 R2 核心叠加，事件层优先保留绕障/回正。"""
        if scenario_name != "AccidentTwoWays" or primary_rs not in {RoadStructure.R4, RoadStructure.R5}:
            return False
        rules = set(evidence.get("rules_fired", []) or [])
        if any(
            rule in rules
            for rule in (
                "r2_core_obstruction_confirmed",
                "r2_xml_trigger_core_confirmed",
                "r2_strict_core_obstruction_window",
                "r2_core_obstruction_meta_confirmed_without_trusted_xodr",
                "r2_opposite_lane_confirmed",
            )
        ):
            return True
        diagnostic = evidence.get("diagnostic_attribution", {}) or {}
        window_flags = diagnostic.get("window_flags", {}) or {}
        if bool(window_flags.get("twoway_core_obstruction")) or bool(window_flags.get("twoway_xml_core_confirmed")):
            return True
        twoway = evidence.get("twoway_obstruction_evidence") or {}
        return bool(twoway.get("core_confirmed")) or bool(twoway.get("stuck")) or bool(twoway.get("vehicle_hazard"))

    @staticmethod
    def _obstacle_event(
        scenario_name: str,
        frame_data: Dict[str, Any],
        primary_rs: RoadStructure,
        evidence: Dict[str, Any],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        if scenario_name not in OBSTACLE_EVENT_DISTANCE_FIELDS:
            return False, [], {}
        field, threshold = OBSTACLE_EVENT_DISTANCE_FIELDS[scenario_name]
        dist = _safe_float(frame_data.get(field), default=math.inf)
        speed_obj_dist = _safe_float(frame_data.get("speed_reduced_by_obj_distance"), default=math.inf)
        scenario_active = RoadEventRuleEngine._scenario_active(frame_data, scenario_name)
        near_trigger = RoadEventRuleEngine._near_trigger(evidence, 55.0)
        kind = evidence.get("rule_kind") or SCENARIO_RULE_CONFIG.get(scenario_name, {}).get("kind")
        twoway = evidence.get("twoway_obstruction_evidence") or {}
        speed_obj_threshold = max(0.0, threshold - 2.0)
        close_specific_obstacle = dist <= threshold
        scenario_obstacles = frame_data.get("scenario_obstacles_ids")
        has_scenario_obstacles = bool(scenario_obstacles) and str(scenario_obstacles) not in {"[]", "None", "nan"}
        trigger_only_same_direction = (
            kind == "same_direction_obstacle"
            and not scenario_active
            and not close_specific_obstacle
            and not has_scenario_obstacles
        )
        speed_obj_close_near_xml = (
            speed_obj_dist <= speed_obj_threshold
            and near_trigger
            and not trigger_only_same_direction
        )
        close_obstacle = close_specific_obstacle or speed_obj_close_near_xml
        hard_response = RoadEventRuleEngine._hard_decel(frame_data) or _safe_bool(frame_data.get("vehicle_hazard", False))
        hard_response_near_object = hard_response and speed_obj_dist <= threshold + 8.0 and (
            near_trigger or close_specific_obstacle
        )
        if (
            trigger_only_same_direction
            and hard_response_near_object
            and RoadEventRuleEngine._route_centered(frame_data)
            and not _safe_bool(frame_data.get("vehicle_hazard", False))
        ):
            hard_response_near_object = False
        twoway_core = bool(
            (evidence.get("diagnostic_attribution", {}) or {})
            .get("window_flags", {})
            .get("twoway_core_obstruction", False)
        )
        twoway_r2_lane_change_core = (
            kind in {"twoways_obstacle", "vehicle_opens_door_twoways"}
            and (
                (primary_rs == RoadStructure.R2 and (twoway_core or close_specific_obstacle or hard_response_near_object))
                or RoadEventRuleEngine._accident_twoways_r2_overlay_active(scenario_name, primary_rs, evidence)
            )
        )
        door_open = scenario_name == "VehicleOpensDoorTwoWays" and _safe_bool(frame_data.get("vehicle_opened_door", False))
        active_window = near_trigger or close_specific_obstacle or twoway_core or twoway_r2_lane_change_core or door_open
        should = active_window and (close_obstacle or twoway_core or twoway_r2_lane_change_core or door_open or hard_response_near_object)
        rules = []
        if scenario_active:
            rules.append("event_active_scenario")
        if near_trigger:
            rules.append("event_near_xml_trigger")
        if close_specific_obstacle:
            rules.append(f"event_obstacle_distance:{field}")
        if speed_obj_close_near_xml:
            rules.append("event_speed_reduced_object_near_xml_trigger")
        if hard_response_near_object:
            rules.append("event_obstacle_speed_or_brake_response")
        if twoway_core:
            rules.append("event_twoway_core_obstruction")
        if twoway_r2_lane_change_core:
            rules.append("event_twoway_r2_lane_change_core")
        if door_open:
            rules.append("event_vehicle_door_opened")
        metrics = {
            field: dist if math.isfinite(dist) else None,
            "speed_reduced_by_obj_distance": speed_obj_dist if math.isfinite(speed_obj_dist) else None,
            "event_obstacle_trigger_threshold_m": threshold,
            "event_speed_object_trigger_threshold_m": speed_obj_threshold,
            "twoway_obstruction": twoway,
            "twoway_r2_lane_change_core": twoway_r2_lane_change_core,
            "accident_twoways_r2_overlay_active": RoadEventRuleEngine._accident_twoways_r2_overlay_active(
                scenario_name,
                primary_rs,
                evidence,
            ),
        }
        return should, rules, metrics

    @staticmethod
    def _ped_bike_event(
        scenario_name: str,
        frame_data: Dict[str, Any],
        primary_rs: RoadStructure,
        evidence: Dict[str, Any],
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        if scenario_name not in PEDESTRIAN_BICYCLE_EVENT_FIELDS:
            return False, [], {}
        field, threshold = PEDESTRIAN_BICYCLE_EVENT_FIELDS[scenario_name]
        dist = _safe_float(frame_data.get(field), default=math.inf)
        alt_dist = min(
            _safe_float(frame_data.get("dist_to_pedestrian"), default=math.inf),
            _safe_float(frame_data.get("dist_to_biker"), default=math.inf),
        )
        scenario_active = RoadEventRuleEngine._scenario_active(frame_data, scenario_name)
        near_trigger = RoadEventRuleEngine._near_trigger(evidence, 45.0)
        hazard = (
            _safe_bool(frame_data.get("walker_hazard", False))
            or _safe_bool(frame_data.get("does_emergency_brake_for_pedestrians", False))
        )
        close = dist <= threshold or alt_dist <= threshold
        if scenario_name in {"VehicleTurningRoute", "VehicleTurningRoutePedestrian"}:
            should = (scenario_active or near_trigger or primary_rs in {RoadStructure.R4, RoadStructure.R5}) and (close or hazard)
        elif scenario_name == "CrossingBicycleFlow":
            should = (scenario_active or near_trigger or primary_rs == RoadStructure.R4) and (close or hazard)
        else:
            should = (scenario_active or near_trigger or close or hazard) and (close or hazard)
        rules = []
        if scenario_active:
            rules.append("event_active_scenario")
        if near_trigger:
            rules.append("event_near_xml_trigger")
        if close:
            rules.append(f"event_crossing_distance:{field}")
        if hazard:
            rules.append("event_walker_or_emergency_brake_hazard")
        metrics = {
            field: dist if math.isfinite(dist) else None,
            "nearest_ped_bike_m": alt_dist if math.isfinite(alt_dist) else None,
        }
        return should, rules, metrics

    @staticmethod
    def analyze(
        scenario_name: str,
        frame_data: Dict[str, Any],
        primary_rs: RoadStructure,
        evidence: Dict[str, Any],
    ) -> Tuple[Set[EventType], EventType, Dict[str, Any]]:
        scenario_allowed = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [EventType.R_E1]))
        road_allowed = set(ROAD_STRUCTURE_TO_FINE_EVENTS.get(primary_rs, {EventType.R_E1}))
        if scenario_name == "InvadingTurn" and primary_rs in {
            RoadStructure.R1,
            RoadStructure.R2,
            RoadStructure.R4,
            RoadStructure.R5,
        }:
            road_allowed.add(EventType.U_E5)
        accident_twoways_r2_overlay = RoadEventRuleEngine._accident_twoways_r2_overlay_active(
            scenario_name,
            primary_rs,
            evidence,
        )
        if accident_twoways_r2_overlay:
            road_allowed.update({EventType.R_E2, EventType.U_E2, EventType.U_E3})
        if scenario_name == "HazardAtSideLaneTwoWays" and primary_rs == RoadStructure.R2:
            road_allowed.add(EventType.U_E4)
        if scenario_name == "CrossJunctionDefectTrafficLight" and primary_rs == RoadStructure.R4:
            road_allowed.update({EventType.U_E6, EventType.U_E7})
        regular, regular_rules, regular_metrics = RoadEventRuleEngine._regular_event_details(
            scenario_name,
            primary_rs,
            frame_data,
            evidence,
        )
        allowed = (scenario_allowed & road_allowed) | {regular}

        unusual: Optional[EventType] = None
        extra_events: Set[EventType] = set()
        rules: List[str] = []
        metrics: Dict[str, Any] = {}

        obstacle_hit, obstacle_rules, obstacle_metrics = RoadEventRuleEngine._obstacle_event(
            scenario_name, frame_data, primary_rs, evidence
        )
        if obstacle_hit and EventType.U_E2 in allowed:
            unusual = EventType.U_E2
            rules.extend(obstacle_rules)
            metrics.update(obstacle_metrics)

        ped_hit, ped_rules, ped_metrics = RoadEventRuleEngine._ped_bike_event(
            scenario_name, frame_data, primary_rs, evidence
        )
        if unusual is None and ped_hit and EventType.U_E4 in allowed:
            unusual = EventType.U_E4
            rules.extend(ped_rules)
            metrics.update(ped_metrics)

        scenario_active = RoadEventRuleEngine._scenario_active(frame_data, scenario_name)
        near_trigger = RoadEventRuleEngine._near_trigger(evidence, 45.0)
        speed_obj_dist = _safe_float(frame_data.get("speed_reduced_by_obj_distance"), default=math.inf)
        cutin_dist = _safe_float(frame_data.get("dist_to_cutin_vehicle"), default=math.inf)
        hard_decel = RoadEventRuleEngine._hard_decel(frame_data)
        vehicle_hazard = _safe_bool(frame_data.get("vehicle_hazard", False))
        route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
        route_center_tolerance = RoadEventRuleEngine._lane_center_tolerance(frame_data)
        route_centered = math.isfinite(route_lateral_offset) and abs(route_lateral_offset) <= route_center_tolerance
        target_lane_change_active = RoadEventRuleEngine._target_lane_change_active(frame_data)
        changed_route = RoadEventRuleEngine._changed_route(frame_data)
        signed_dist = _safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf)
        route_lateral_abs = abs(route_lateral_offset) if math.isfinite(route_lateral_offset) else math.inf

        if unusual is None and scenario_name == "HardBreakRoute" and EventType.U_E1 in allowed:
            close_lead_stop = vehicle_hazard and speed_obj_dist <= 14.0 and RoadEventRuleEngine._speed(frame_data) <= 2.5
            if (scenario_active or speed_obj_dist <= 30.0) and (hard_decel or close_lead_stop):
                unusual = EventType.U_E1
                rules.extend(["event_hard_brake_response"])
        if unusual is None and scenario_name == "BlockedIntersection":
            blocked_near_trigger = RoadEventRuleEngine._near_trigger(evidence, 52.0)
            if EventType.U_E8 in allowed and (scenario_active or blocked_near_trigger) and (
                _safe_bool(frame_data.get("slower_occluded_junction", False))
                or (RoadEventRuleEngine._speed(frame_data) <= 0.9 and _safe_bool(frame_data.get("brake", False)))
            ):
                unusual = EventType.U_E8
                rules.extend(["event_blocked_intersection_wait_or_clear"])
            elif EventType.U_E1 in allowed and hard_decel and speed_obj_dist <= 25.0:
                unusual = EventType.U_E1
                rules.extend(["event_blocked_intersection_lead_vehicle_decel"])
        if unusual is None and scenario_name in {"ParkingCutIn", "StaticCutIn", "DynamicObjectCrossing"} and EventType.U_E3 in allowed:
            if scenario_name == "ParkingCutIn":
                cutin_motion = (
                    _safe_bool(frame_data.get("brake_cutin", False))
                    or vehicle_hazard
                    or (changed_route and math.isfinite(route_lateral_abs) and route_lateral_abs >= 0.04)
                    or target_lane_change_active
                )
                parking_cutin_core = (
                    math.isfinite(cutin_dist)
                    and cutin_dist <= 26.0
                    and cutin_motion
                )
                if parking_cutin_core:
                    unusual = EventType.U_E3
                    rules.extend(["event_parking_cutin_dynamic_core"])
            elif (scenario_active or near_trigger or cutin_dist <= 35.0) and (
                cutin_dist <= 30.0 or _safe_bool(frame_data.get("brake_cutin", False)) or vehicle_hazard
            ):
                unusual = EventType.U_E3
                rules.extend(["event_dynamic_cutin_or_occupancy"])
        if unusual is None and scenario_name == "InvadingTurn" and EventType.U_E5 in allowed:
            rs_rules = set(str(rule) for rule in (evidence.get("rules_fired") or []))
            trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
            r2_local_invasion_motion = (
                changed_route
                and math.isfinite(route_lateral_abs)
                and route_lateral_abs >= 0.01
                and math.isfinite(trigger_distance)
                and trigger_distance >= 45.0
            )
            r2_invasion_core = (
                primary_rs == RoadStructure.R2
                and r2_local_invasion_motion
                and bool(
                    rs_rules
                    & {
                        "passive_oncoming_invasion",
                        "r2_passive_invading_turn",
                        "r2_opposite_lane_confirmed",
                        "r2_scenario_trigger_medium",
                    }
                )
            )
            if r2_invasion_core:
                unusual = EventType.U_E5
                rules.extend(["event_oncoming_lane_invasion_r2_core"])
                metrics["invading_turn_r2_core_u5"] = True
            elif (
                vehicle_hazard
                and math.isfinite(speed_obj_dist)
                and speed_obj_dist <= 35.0
            ):
                unusual = EventType.U_E5
                rules.extend(["event_oncoming_lane_invasion_vehicle_hazard_confirmed"])
        if unusual is None and scenario_name == "OppositeVehicleRunningRedLight" and EventType.U_E6 in allowed:
            bbox_metrics = frame_data.get("bbox_semantic_metrics", {}) or {}
            bbox_conflict_count = int(
                _safe_float(bbox_metrics.get("red_light_conflict_vehicle_count"), default=0.0) or 0
            )
            bbox_conflict_dist = _safe_float(
                bbox_metrics.get("red_light_conflict_vehicle_min_distance_m"),
                default=math.inf,
            )
            bbox_conflict_forward_x = _safe_float(
                bbox_metrics.get("red_light_conflict_vehicle_min_forward_x_m"),
                default=math.inf,
            )
            bbox_conflict_lateral_y = _safe_float(
                bbox_metrics.get("red_light_conflict_vehicle_min_abs_lateral_y_m"),
                default=math.inf,
            )
            bbox_conflict_speed = _safe_float(
                bbox_metrics.get("red_light_conflict_vehicle_max_speed_mps"),
                default=math.inf,
            )
            vehicle_hazard_conflict = (
                primary_rs == RoadStructure.R4
                and vehicle_hazard
                and math.isfinite(speed_obj_dist)
                and speed_obj_dist <= 30.0
            )
            bbox_crossing_conflict = (
                primary_rs == RoadStructure.R4
                and bbox_conflict_count > 0
                and bbox_conflict_dist <= 32.0
                and bbox_conflict_forward_x <= 26.0
            )
            if vehicle_hazard_conflict or bbox_crossing_conflict:
                unusual = EventType.U_E6
                if vehicle_hazard_conflict:
                    rules.extend(["event_opposite_vehicle_running_red_light_hazard_confirmed"])
                if bbox_crossing_conflict:
                    rules.extend(["event_opposite_red_light_bbox_crossing_vehicle_confirmed"])
                    metrics.update(
                        {
                            "red_light_conflict_vehicle_count": bbox_conflict_count,
                            "red_light_conflict_vehicle_min_distance_m": bbox_conflict_dist,
                            "red_light_conflict_vehicle_min_forward_x_m": bbox_conflict_forward_x,
                            "red_light_conflict_vehicle_min_abs_lateral_y_m": bbox_conflict_lateral_y,
                            "red_light_conflict_vehicle_max_speed_mps": bbox_conflict_speed,
                        }
                    )
        defect_traffic_light_state = str(frame_data.get("traffic_light_state") or "")
        defect_has_valid_light_state = defect_traffic_light_state in {"Red", "Yellow", "Green"}
        defect_is_junction_frame = _safe_bool(frame_data.get("is_junction", False))
        defect_speed_obj_type = str(frame_data.get("speed_reduced_by_obj_type") or "")
        defect_conflict_vehicle = (
            vehicle_hazard
            or (
                math.isfinite(speed_obj_dist)
                and speed_obj_dist <= 30.0
                and defect_speed_obj_type
                and not defect_speed_obj_type.startswith("traffic.traffic_light")
            )
        )
        defect_light_unavailable_in_context = (
            not defect_has_valid_light_state
            and (defect_is_junction_frame or scenario_active or near_trigger)
        )
        if unusual is None and scenario_name == "CrossJunctionDefectTrafficLight":
            if EventType.U_E7 in allowed and primary_rs == RoadStructure.R4 and (
                defect_conflict_vehicle
                or defect_light_unavailable_in_context
            ):
                unusual = EventType.U_E7
                rules.extend(["event_defect_junction_rule_failure"])
                if EventType.U_E6 in allowed and defect_conflict_vehicle:
                    extra_events.add(EventType.U_E6)
                    rules.extend(["event_defect_junction_vehicle_conflict"])
        if unusual is None and scenario_name == "OppositeVehicleTakingPriority" and EventType.U_E7 in allowed:
            if primary_rs == RoadStructure.R5 and (scenario_active or near_trigger or vehicle_hazard or hard_decel):
                unusual = EventType.U_E7
                rules.extend(["event_priority_rule_failure_or_extra_yield"])

        if unusual is not None and unusual not in allowed:
            unusual = None

        if unusual is None:
            events = {regular}
            primary_event = regular
        elif (
            primary_rs in {RoadStructure.R4, RoadStructure.R5}
            and regular in {EventType.R_E4, EventType.R_E5}
            and unusual in {EventType.U_E1, EventType.U_E4, EventType.U_E6, EventType.U_E7, EventType.U_E8}
        ):
            events = {regular, unusual, *extra_events}
            primary_event = unusual
        else:
            events = {unusual, *extra_events}
            primary_event = unusual

        metrics.update(
            {
                "speed": RoadEventRuleEngine._speed(frame_data),
                "target_speed": RoadEventRuleEngine._target_speed(frame_data),
                "speed_reduced_by_obj_distance": speed_obj_dist if math.isfinite(speed_obj_dist) else None,
                "dist_to_cutin_vehicle": cutin_dist if math.isfinite(cutin_dist) else None,
                "brake_cutin": _safe_bool(frame_data.get("brake_cutin", False)),
                "vehicle_hazard": vehicle_hazard,
                "hard_decel": hard_decel,
                "changed_route": changed_route,
                "signed_dist_to_lane_change": (
                    signed_dist
                    if math.isfinite(signed_dist)
                    else None
                ),
                "route_lateral_offset_m": route_lateral_offset if math.isfinite(route_lateral_offset) else None,
                "route_lateral_abs_m": route_lateral_abs if math.isfinite(route_lateral_abs) else None,
                "route_center_tolerance_m": route_center_tolerance,
                "route_centered": route_centered,
                "target_lane_change_active": target_lane_change_active,
                "route_projection_error_m": evidence.get("route_projection_error_m"),
                "xodr_road_id": (evidence.get("xodr") or {}).get("map_road_id"),
                "xodr_lane_id": (evidence.get("xodr") or {}).get("map_lane_id"),
                "xodr_junction_id": (evidence.get("xodr") or {}).get("map_junction_id"),
                "xodr_topology_trusted": (evidence.get("xodr") or {}).get("xodr_topology_trusted"),
                "scenario_active": scenario_active,
                "near_trigger": near_trigger,
                "primary_road_structure": primary_rs.value,
                "accident_twoways_r2_overlay_active": accident_twoways_r2_overlay,
                "traffic_light_state": frame_data.get("traffic_light_state"),
                "light_hazard": _safe_bool(frame_data.get("light_hazard", False)),
                "defect_has_valid_light_state": defect_has_valid_light_state,
                "defect_light_unavailable_in_context": defect_light_unavailable_in_context,
                "defect_conflict_vehicle": defect_conflict_vehicle,
            }
        )
        metrics.update(regular_metrics)
        event_evidence = {
            "primary_event": primary_event.value,
            "events": [ev.value for ev in sorted(events, key=lambda ev: ev.value)],
            "regular_event": regular.value,
            "unusual_event": unusual.value if unusual else None,
            "secondary_unusual_events": [ev.value for ev in sorted(extra_events, key=lambda ev: ev.value)],
            "allowed_events": [ev.value for ev in allowed],
            "rules_fired": rules or regular_rules or ["event_regular_by_road_structure"],
            "metrics": metrics,
            "review_required": False,
            "review_reasons": [],
        }
        return events, primary_event, event_evidence


@dataclass(frozen=True)
class TwoWayObstructionEvidence:
    """TwoWays R2 的核心对象/动作证据，独立于 XODR 拓扑可信度。"""

    nearest_obstacle_m: Optional[float]
    stuck: bool
    vehicle_hazard: bool
    has_scenario_obstacles: bool
    signed_lane_change_abs_m: Optional[float]
    core_confirmed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nearest_obstacle_m": self.nearest_obstacle_m,
            "stuck": self.stuck,
            "vehicle_hazard": self.vehicle_hazard,
            "has_scenario_obstacles": self.has_scenario_obstacles,
            "signed_lane_change_abs_m": self.signed_lane_change_abs_m,
            "core_confirmed": self.core_confirmed,
        }


def _twoway_obstruction_evidence(frame_data: Dict[str, Any]) -> TwoWayObstructionEvidence:
    """提取图像优先复核后允许支撑 R2 high 的近距离障碍证据。"""
    nearest_obstacle = _finite_min(
        frame_data.get("dist_to_accident_site"),
        frame_data.get("dist_to_construction_site"),
        frame_data.get("dist_to_parked_obstacle"),
    )
    nearest_obstacle_out = round(nearest_obstacle, 3) if math.isfinite(nearest_obstacle) else None
    stuck = (
        _safe_bool(frame_data.get("accident_two_ways_stuck", False))
        or _safe_bool(frame_data.get("construction_obstacle_two_ways_stuck", False))
        or _safe_bool(frame_data.get("parked_obstacle_two_ways_stuck", False))
    )
    vehicle_hazard = _safe_bool(frame_data.get("vehicle_hazard", False))
    scenario_obstacles = frame_data.get("scenario_obstacles_ids")
    has_scenario_obstacles = bool(scenario_obstacles) and str(scenario_obstacles) not in {"[]", "None", "nan"}
    signed_lane_change = abs(_safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
    signed_lane_change_out = round(signed_lane_change, 3) if math.isfinite(signed_lane_change) else None
    core_confirmed = (
        stuck
        or vehicle_hazard
        or nearest_obstacle <= 18.0
        or (has_scenario_obstacles and nearest_obstacle <= 30.0)
        or signed_lane_change <= 1.5
    )
    return TwoWayObstructionEvidence(
        nearest_obstacle_m=nearest_obstacle_out,
        stuck=stuck,
        vehicle_hazard=vehicle_hazard,
        has_scenario_obstacles=has_scenario_obstacles,
        signed_lane_change_abs_m=signed_lane_change_out,
        core_confirmed=core_confirmed,
    )


def _twoway_strict_core_confirmed(
    evidence: TwoWayObstructionEvidence,
    cfg: Dict[str, Any],
) -> bool:
    """TwoWays R2 high 的严格核心门控：只覆盖正在借/等对向的局部障碍段。"""
    nearest = evidence.nearest_obstacle_m
    lane_change_abs = evidence.signed_lane_change_abs_m
    core_m = float(cfg.get("two_way_obstacle_core_m", 18.0))
    approach_m = float(cfg.get("two_way_approach_obstacle_m", max(core_m, 28.0)))
    if evidence.stuck or evidence.vehicle_hazard:
        return True
    if nearest is None:
        return False
    if nearest <= core_m:
        return True
    return (
        evidence.has_scenario_obstacles
        and nearest <= approach_m
        and lane_change_abs is not None
        and lane_change_abs <= 1.5
    )


def _strong_control_context(
    *,
    is_junction: bool,
    xodr_near_junction: bool,
    stop_hazard: bool,
    static_signal_near: bool,
    dist_to_junction: float,
    junction_pre: float,
) -> bool:
    """R4/R5 high 需要能在 RGB 或可信拓扑中解释的路口控制上下文。"""
    static_signal_strong_m = min(junction_pre, 25.0)
    return (
        is_junction
        or xodr_near_junction
        or stop_hazard
        or (static_signal_near and dist_to_junction < static_signal_strong_m)
    )


def _frame_annotation_comment(
    primary: RoadStructure,
    secondary: Set[RoadStructure],
    confidence: float,
    evidence: Dict[str, Any],
) -> str:
    """把单帧规则命中与关键证据转成可人工浏览的中文注释。"""
    rule_kind = evidence.get("rule_kind", "unknown")
    rules = evidence.get("rules_fired", [])
    diag = evidence.get("diagnostic_attribution", {})
    source = diag.get("decision_source", "unknown")
    weak = diag.get("weak_or_missing_inputs", [])
    secondary_text = ",".join(sorted(rs.value if isinstance(rs, RoadStructure) else str(rs) for rs in secondary)) or "无"
    review_text = "需复核" if evidence.get("review_required") else "可自动使用"
    metric_parts = []
    route_s = evidence.get("route_progress_m")
    route_error = evidence.get("route_projection_error_m")
    trigger_distance = evidence.get("trigger_distance_m")
    if route_s is not None:
        metric_parts.append(f"route_s={route_s:.1f}m")
    if route_error is not None:
        metric_parts.append(f"proj_err={route_error:.1f}m")
    if trigger_distance is not None:
        metric_parts.append(f"trigger_dist={trigger_distance:.1f}m")
    route_bucket = evidence.get("route_semantic_bucket")
    if route_bucket:
        metric_parts.append(f"route_bucket={route_bucket}")
    metric_text = ", ".join(metric_parts) if metric_parts else "无可用route/trigger度量"
    weak_text = f"；弱证据={','.join(weak[:4])}" if weak else ""
    return (
        f"{primary.value}：规则族={rule_kind}，来源={source}，置信={confidence:.2f}，"
        f"secondary={secondary_text}，{metric_text}，{review_text}；"
        f"命中={','.join(rules[:5])}{weak_text}"
    )


class RoadStructureRuleEngine:
    """按 ROAD_EVENT_CLASSIFICATION_PLAN.md 生成帧级主 RS。"""

    PRIORITY = [RoadStructure.R4, RoadStructure.R5, RoadStructure.R3, RoadStructure.R2, RoadStructure.R1]

    def __init__(self, xodr_probe: Optional[XodrTopologyProbe] = None):
        self.xodr_probe = xodr_probe

    def _add(self, scores: Dict[RoadStructure, float], rs: RoadStructure, score: float) -> None:
        scores[rs] = max(scores.get(rs, 0.0), score)

    def analyze(
        self,
        scenario_name: str,
        frame_id: int,
        frame_data: Dict[str, Any],
        xml_info: Optional[RouteXmlInfo] = None,
        route_id: Optional[str] = None,
    ) -> Tuple[RoadStructure, Set[RoadStructure], Dict[str, float], Dict[str, Any], float, str]:
        base_allowed = set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [RoadStructure.R1]))
        route_semantic_bucket = _mixed_route_semantic_bucket(scenario_name, route_id)
        route_highway_bucket = route_semantic_bucket == "highway_rgb_route"
        allowed = _mixed_route_allowed_structures(scenario_name, route_id, base_allowed)
        scores: Dict[RoadStructure, float] = {RoadStructure.R1: 0.35}
        rules: List[str] = ["r1_default_candidate"]
        if route_highway_bucket:
            rules.append("mixed_route_rgb_highway_bucket_r3_r4_only")
        elif route_semantic_bucket == "mixed_reviewed_non_highway":
            rules.append("mixed_route_rgb_non_highway_keeps_base_candidates")

        ego_xy = _extract_ego_xy(frame_data)
        town = xml_info.town if xml_info is not None else str(frame_data.get("town", ""))
        route_s, route_error = _project_route_s(ego_xy, xml_info.waypoints if xml_info else [])
        xml_weather, xml_weather_route_percentage = _current_xml_weather(xml_info, route_s)
        low_visibility_factor, low_visibility_reasons = _low_visibility_junction_factor(xml_weather)
        trigger_s = math.nan
        if xml_info and xml_info.trigger_points and xml_info.waypoints:
            trigger_s, _ = _project_route_s(xml_info.trigger_points[0], xml_info.waypoints)
        trigger_distance = _nearest_trigger_distance(ego_xy, xml_info)
        xodr = self.xodr_probe.probe(town, ego_xy) if self.xodr_probe else {"xodr_available": False}

        tl = frame_data.get("traffic_light_state")
        has_tl = _valid_traffic_light(tl)
        light_hazard = _safe_bool(frame_data.get("light_hazard", False))
        bbox_traffic_light = _safe_bool(frame_data.get("bbox_has_traffic_light", False))
        bbox_stop_sign = _safe_bool(frame_data.get("bbox_has_stop_sign", False))
        bbox_yield_sign = _safe_bool(frame_data.get("bbox_has_yield_sign", False))
        bbox_junction_hint = _safe_bool(frame_data.get("bbox_has_junction_hint", False))
        bbox_metrics = frame_data.get("bbox_semantic_metrics", {}) or {}
        bbox_tl_count = int(_safe_float(bbox_metrics.get("traffic_light_count"), default=0.0) or 0)
        bbox_tl_min_distance = _safe_float(bbox_metrics.get("traffic_light_min_distance_m"), default=math.inf)
        bbox_tl_forward_x = _safe_float(bbox_metrics.get("traffic_light_min_forward_x_m"), default=math.inf)
        bbox_tl_physical_distance = _safe_float(
            bbox_metrics.get("traffic_light_min_physical_distance_m"),
            default=math.inf,
        )
        bbox_tl_affects_ego = _safe_bool(bbox_metrics.get("traffic_light_affects_ego", False))
        bbox_tl_same_lane = _safe_bool(bbox_metrics.get("traffic_light_same_lane", False))
        bbox_tl_overhead = _safe_bool(bbox_metrics.get("traffic_light_overhead", False))
        bbox_stop_min_distance = _safe_float(bbox_metrics.get("stop_sign_min_distance_m"), default=math.inf)
        bbox_yield_min_distance = _safe_float(bbox_metrics.get("yield_sign_min_distance_m"), default=math.inf)
        bbox_stop_yield_min_distance = min(bbox_stop_min_distance, bbox_yield_min_distance)
        rgb_no_r4 = scenario_name in SCENARIOS_WITH_RGB_NO_R4
        if (
            (has_tl or light_hazard or bbox_traffic_light)
            and not rgb_no_r4
        ):
            allowed.add(RoadStructure.R4)
        meta_is_junction = _safe_bool(frame_data.get("is_junction", False)) or _safe_bool(frame_data.get("is_intersection", False))
        is_junction = meta_is_junction or bbox_junction_hint
        map_is_junction = bool(xodr.get("map_is_junction", False))
        map_is_roundabout = bool(xodr.get("map_is_roundabout", False))
        dist_to_junction = _finite_min(frame_data.get("dist_to_junction"), frame_data.get("distance_to_next_junction"))
        meta_stop_hazard = _safe_bool(frame_data.get("stop_sign_hazard", False)) or _safe_bool(frame_data.get("stop_sign_close", False))
        stop_hazard = meta_stop_hazard or bbox_stop_sign or bbox_yield_sign
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {"kind": SCENARIO_RULE_KIND.get(scenario_name, "default_meta_map")})
        kind = str(cfg.get("kind", SCENARIO_RULE_KIND.get(scenario_name, "default_meta_map")))
        active = str(frame_data.get("current_active_scenario_type", "") or "")
        scenario_active = scenario_name in active or active in {scenario_name, scenario_name.replace("V2", "")}
        trigger_close_m = float(cfg.get("trigger_close_m", 70.0))
        close_trigger = trigger_distance < trigger_close_m
        actor_flow_distance = _actor_flow_distance(ego_xy, xml_info)
        actor_flow_near = actor_flow_distance < float(cfg.get("actor_flow_near_m", 0.0))
        xodr_trusted = bool(xodr.get("xodr_topology_trusted", xodr.get("xodr_available", False)))
        xodr_source = str(xodr.get("xodr_source", ""))
        static_topology_only = xodr_source == "static_xodr"
        xodr_junction_id = str(xodr.get("map_junction_id", ""))
        xodr_junction_connection_count = int(xodr.get("junction_connection_count", 0) or 0)
        xodr_nearest_signal_m = _safe_float(xodr.get("nearest_signal_m"), default=math.inf)
        xodr_heading_change = _safe_float(xodr.get("road_total_abs_heading_change"), default=0.0)
        xodr_roundabout_like_junction = (
            xodr_trusted
            and map_is_junction
            and xodr_junction_connection_count >= 5
            and (
                (bool(xodr.get("ramp_merge_split_hint", False)) and xodr_nearest_signal_m > 60.0)
                or (xodr_heading_change >= 1.0 and xodr_nearest_signal_m > 35.0)
            )
            and not has_tl
            and not bbox_traffic_light
        )
        if kind == "same_direction_obstacle" and xodr_roundabout_like_junction:
            xodr = dict(xodr)
            xodr["map_is_roundabout"] = True
            xodr["roundabout_inferred_by_static_junction_loop"] = True
            map_is_roundabout = True
            rules.append("roundabout_like_static_junction_loop_forces_r1")
        xodr_structured_junction = (
            map_is_junction
            and (
                not static_topology_only
                or (
                    xodr_junction_id not in {"", "-1", "None", "none"}
                    and xodr_junction_connection_count > 0
                )
            )
        )
        if static_topology_only and map_is_junction and not xodr_structured_junction:
            rules.append("static_xodr_junction_hint_demoted_unstructured")
        dist_to_junction_near = dist_to_junction < JUNCTION_META_NEAR_M
        xml_distance = _xml_numeric(xml_info, "distance", default=50.0)
        two_way_pre = max(xml_distance, float(cfg.get("two_way_min_pre_m", 45.0)))
        two_way_post = two_way_pre + float(cfg.get("two_way_post_pad_m", 20.0))
        junction_pre = float(cfg.get("junction_pre_m", 60.0))
        junction_post = float(cfg.get("junction_post_m", 25.0))
        junction_pre_window, junction_post_window = _shrink_junction_window(junction_pre, junction_post)
        if low_visibility_factor < 1.0:
            junction_pre_window = max(10.0, junction_pre_window * low_visibility_factor)
            junction_post_window = max(3.0, junction_post_window * low_visibility_factor)
            rules.append(
                f"low_visibility_junction_window_scaled_{low_visibility_factor:.2f}"
            )
        scenario_junction_factor = _safe_float(cfg.get("junction_tighten_factor"), default=1.0)
        if not math.isfinite(scenario_junction_factor) or scenario_junction_factor <= 0.0:
            scenario_junction_factor = 1.0
        scenario_junction_factor = min(1.0, scenario_junction_factor)
        scenario_min_pre = _safe_float(cfg.get("junction_min_pre_m"), default=JUNCTION_PRE_WINDOW_MIN_M)
        scenario_min_post = _safe_float(cfg.get("junction_min_post_m"), default=JUNCTION_POST_WINDOW_MIN_M)
        if scenario_junction_factor < 1.0:
            junction_pre_window = max(scenario_min_pre, junction_pre_window * scenario_junction_factor)
            junction_post_window = max(scenario_min_post, junction_post_window * scenario_junction_factor)
            rules.append(f"scenario_junction_window_tightened_{scenario_junction_factor:.2f}")
        effective_meta_near_m = max(10.0, JUNCTION_META_NEAR_M * low_visibility_factor * scenario_junction_factor)
        effective_strong_max_m = max(8.0, JUNCTION_STRONG_MAX_M * low_visibility_factor * scenario_junction_factor)
        effective_static_signal_near_m = max(10.0, STATIC_SIGNAL_NEAR_M * low_visibility_factor * scenario_junction_factor)
        effective_close_trigger_max_m = max(8.0, JUNCTION_CLOSE_TRIGGER_MAX_M * low_visibility_factor * scenario_junction_factor)
        dist_to_junction_near = dist_to_junction < effective_meta_near_m
        dist_to_junction_strong = dist_to_junction < min(junction_pre_window, effective_strong_max_m)
        route_s_for_window = route_s
        route_projection_error_high = route_error > 5.0 if math.isfinite(route_error) else False
        if route_projection_error_high:
            route_s_for_window = math.nan
            rules.append("route_s_window_disabled_projection_error_gt_5m")
        static_topology_strong = not (static_topology_only and route_projection_error_high)
        if static_topology_only and route_projection_error_high and xodr.get("xodr_available"):
            rules.append("static_xodr_topology_demoted_projection_error")
        static_signal_near = (
            xodr_trusted
            and static_topology_strong
            and xodr_nearest_signal_m <= effective_static_signal_near_m
        )
        bbox_traffic_light_for_r4 = bbox_traffic_light
        static_signal_near_for_r4 = static_signal_near
        has_tl_for_r4 = has_tl
        noscenario_local_signal_control = True
        noscenario_stop_yield_overrides_weak_light = False
        noscenario_far_single_light_demoted = False
        if kind == "noscenario":
            noscenario_local_signal_control = bool(
                bbox_tl_overhead
                or (bbox_tl_affects_ego and bbox_tl_forward_x <= 18.0)
                or (bbox_tl_affects_ego and bbox_tl_min_distance <= 12.0)
                or (bbox_tl_count >= 2 and bbox_tl_affects_ego and bbox_tl_forward_x <= 32.0)
                or (bbox_tl_physical_distance <= 24.0 and bbox_tl_forward_x <= 25.0)
                or (meta_is_junction and bbox_tl_forward_x <= 12.0)
            )
            local_stop_or_yield = bool(meta_stop_hazard or bbox_stop_yield_min_distance <= 35.0)
            noscenario_stop_yield_overrides_weak_light = (
                local_stop_or_yield
                and not noscenario_local_signal_control
                and not light_hazard
            )
            noscenario_far_single_light_demoted = (
                (has_tl or bbox_traffic_light)
                and not noscenario_local_signal_control
                and not local_stop_or_yield
                and not light_hazard
            )
            if noscenario_stop_yield_overrides_weak_light:
                has_tl_for_r4 = False
                bbox_traffic_light_for_r4 = False
                static_signal_near_for_r4 = False
                rules.append("noscenario_stop_yield_overrides_weak_far_light")
            elif noscenario_far_single_light_demoted:
                has_tl_for_r4 = False
                bbox_traffic_light_for_r4 = False
                static_signal_near_for_r4 = False
                rules.append("noscenario_far_single_light_demoted_to_r1")
        stop_yield_overrides_weak_signal_hint = (
            stop_hazard
            and not has_tl
            and (
                kind == "nonsignalized_junction"
                or scenario_name == "T_Junction"
            )
        )
        if stop_yield_overrides_weak_signal_hint:
            bbox_traffic_light_for_r4 = False
            static_signal_near_for_r4 = False
            rules.append("stop_yield_suppresses_bbox_or_static_signal_r4")
        meta_near_junction = is_junction or dist_to_junction_strong
        xodr_near_junction = xodr_trusted and static_topology_strong and xodr_structured_junction
        near_junction = (meta_near_junction or xodr_near_junction) and not map_is_roundabout
        stop_hazard_for_r5 = stop_hazard
        junction_context_for_r5 = is_junction
        conservative_outside_xml_guard = kind in {"same_direction_obstacle", "invading_turn"} or (
            kind == "default_meta_map" and scenario_name == "ControlLoss"
        )
        if conservative_outside_xml_guard:
            conservative_r5_context = bbox_junction_hint or (
                meta_stop_hazard
                and (meta_is_junction or (dist_to_junction_strong and not static_topology_only))
            )
            stop_hazard_for_r5 = stop_hazard and (
                conservative_r5_context
            )
            junction_context_for_r5 = conservative_r5_context
            if (stop_hazard or is_junction) and not conservative_r5_context:
                rules.append(f"{kind}_stop_yield_without_local_junction_demoted")
        strong_control_context = _strong_control_context(
            is_junction=is_junction,
            xodr_near_junction=xodr_near_junction,
            stop_hazard=stop_hazard_for_r5,
            static_signal_near=static_signal_near,
            dist_to_junction=dist_to_junction,
            junction_pre=junction_pre_window,
        )
        close_trigger_for_structure = close_trigger and (
            not route_projection_error_high
            or trigger_distance < min(trigger_close_m, 25.0)
        )
        close_trigger_for_junction = trigger_distance < min(
            trigger_close_m,
            junction_pre_window,
            effective_close_trigger_max_m,
        ) and not map_is_roundabout and (
            not route_projection_error_high
            or trigger_distance < min(effective_close_trigger_max_m, 25.0)
        )
        if kind == "twoways_obstacle":
            twoways_local_junction_context = (
                meta_is_junction
                or bbox_junction_hint
                or (meta_stop_hazard and (bbox_stop_sign or bbox_yield_sign))
                or (has_tl and bbox_traffic_light)
                or xodr_near_junction
            )
            if close_trigger_for_junction and not twoways_local_junction_context:
                close_trigger_for_junction = False
                rules.append("twoways_xml_trigger_not_junction_context")
        rightturn_local_core = True
        rightturn_intersection_distance = _safe_float(
            frame_data.get("distance_to_intersection_index_ego"),
            default=math.inf,
        )
        if scenario_name == "NonSignalizedJunctionRightTurn":
            rightturn_core_pre = _safe_float(cfg.get("rightturn_core_pre_m"), default=32.0)
            rightturn_core_post = _safe_float(cfg.get("rightturn_core_post_m"), default=5.0)
            rightturn_dist_core = _safe_float(cfg.get("rightturn_core_dist_to_junction_m"), default=18.0)
            rightturn_trigger_core = _safe_float(cfg.get("rightturn_core_trigger_m"), default=12.0)
            rightturn_local_core = (
                is_junction
                or bbox_junction_hint
                or (
                    math.isfinite(rightturn_intersection_distance)
                    and -rightturn_core_post <= rightturn_intersection_distance <= rightturn_core_pre
                )
                or (
                    math.isfinite(dist_to_junction)
                    and dist_to_junction <= rightturn_dist_core
                )
                or (
                    trigger_distance <= rightturn_trigger_core
                    and (bbox_stop_sign or bbox_yield_sign or bbox_traffic_light or has_tl)
                )
            ) and not map_is_roundabout
            if not rightturn_local_core:
                stop_hazard_for_r5 = False
                junction_context_for_r5 = False
                close_trigger_for_junction = False
                strong_control_context = _strong_control_context(
                    is_junction=False,
                    xodr_near_junction=False,
                    stop_hazard=False,
                    static_signal_near=False,
                    dist_to_junction=math.inf,
                    junction_pre=junction_pre_window,
                )
                rules.append("nonsig_rightturn_far_from_local_core_demoted")
        scenario_active_for_structure = scenario_active and not route_projection_error_high

        two_way_window = (
            _route_trigger_window(route_s_for_window, trigger_s, two_way_pre, two_way_post)
            or close_trigger_for_structure
            or scenario_active_for_structure
        )
        junction_window = (
            near_junction
            or (_route_trigger_window(route_s_for_window, trigger_s, junction_pre_window, junction_post_window) and not map_is_roundabout)
            or close_trigger_for_junction
        )
        if scenario_name == "NonSignalizedJunctionRightTurn" and not rightturn_local_core:
            junction_window = False
        turning_local_junction_evidence = True
        if kind == "vehicle_turning" and "turning_trigger_core_m" in cfg:
            turning_local_junction_evidence = (
                is_junction
                or dist_to_junction_near
                or bbox_junction_hint
                or xodr_near_junction
                or static_signal_near
            ) and not map_is_roundabout
            turning_trigger_core_m = float(cfg.get("turning_trigger_core_m", 8.0))
            if not (has_tl or bbox_traffic_light_for_r4 or light_hazard):
                turning_trigger_core_m = float(
                    cfg.get("turning_nosignal_trigger_core_m", turning_trigger_core_m)
                )
            turning_trigger_core = trigger_distance <= turning_trigger_core_m
            if junction_window and not (turning_local_junction_evidence or turning_trigger_core):
                junction_window = False
                rules.append("vehicle_turning_far_trigger_without_local_junction_demoted")
        turning_tl_requires_local_junction = bool(
            kind == "vehicle_turning" and cfg.get("turning_tl_requires_local_junction")
        )
        turning_local_junction_context = (
            not turning_tl_requires_local_junction
            or junction_window
            or near_junction
            or bbox_junction_hint
            or turning_local_junction_evidence
        )
        defect_local_control_context = True
        if kind == "defect_junction":
            defect_local_control_context = (
                is_junction
                or dist_to_junction_near
                or bbox_junction_hint
                or xodr_near_junction
                or static_signal_near
            ) and not map_is_roundabout
            if junction_window and not defect_local_control_context:
                junction_window = False
                rules.append("defect_far_trigger_or_meta_light_without_local_junction_demoted")
        dynamic_crossing_strict_junction_context = (
            scenario_name != "DynamicObjectCrossing"
            or close_trigger_for_junction
            or bbox_junction_hint
            or (has_tl and dist_to_junction_strong)
        )

        if map_is_roundabout:
            self._add(scores, RoadStructure.R1, 0.92)
            rules.append("roundabout_xodr_forces_r1")

        twoway_obstruction = _twoway_obstruction_evidence(frame_data)
        two_way_layout_prior_enabled = bool(cfg.get("two_way_layout_prior"))
        light_hazard_control_context = strong_control_context and (
            meta_near_junction
            or xodr_near_junction
            or stop_hazard
        )
        conservative_light_hazard_kind = kind in {"same_direction_obstacle", "default_meta_map", "noscenario"}
        noscenario_signalized_approach_context = (
            kind == "noscenario"
            and junction_window
            and has_tl_for_r4
            and bbox_traffic_light_for_r4
            and noscenario_local_signal_control
            and not stop_hazard
            and not map_is_roundabout
        )
        if conservative_light_hazard_kind and not (meta_near_junction or stop_hazard):
            light_hazard_control_context = False
        elif (not conservative_light_hazard_kind) and static_signal_near and strong_control_context:
            light_hazard_control_context = True

        if rgb_no_r4 and (has_tl or light_hazard or bbox_traffic_light):
            rules.append("rgb_review_no_signalized_intersection_ignores_meta_tl")
        elif noscenario_signalized_approach_context:
            self._add(scores, RoadStructure.R4, 0.90)
            rules.append("r4_noscenario_stable_tl_bbox_approach")
        elif (not map_is_roundabout) and has_tl_for_r4 and strong_control_context and dynamic_crossing_strict_junction_context:
            self._add(scores, RoadStructure.R4, 0.95)
            rules.append("r4_tl_confirmed")
        elif (
            not map_is_roundabout
            and scenario_name == "DynamicObjectCrossing"
            and has_tl_for_r4
            and bbox_traffic_light_for_r4
            and (near_junction or close_trigger_for_junction or bbox_junction_hint)
        ):
            self._add(scores, RoadStructure.R4, 0.88)
            self._add(scores, RoadStructure.R1, 0.70)
            rules.append("r4_dynamic_crossing_meta_bbox_light_near_control")
        elif (not map_is_roundabout) and has_tl_for_r4:
            if kind == "highway_merge":
                self._add(scores, RoadStructure.R4, 0.70)
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("r4_highway_meta_tl_without_control_context_demoted")
            elif scenario_name == "NonSignalizedJunctionRightTurn" and not rightturn_local_core:
                self._add(scores, RoadStructure.R1, 0.86)
                rules.append("nonsig_rightturn_far_meta_tl_demoted_to_r1")
            elif (
                kind == "same_direction_obstacle"
                and bbox_traffic_light
                and xodr_nearest_signal_m <= effective_static_signal_near_m
            ):
                self._add(scores, RoadStructure.R4, 0.88)
                rules.append("r4_same_direction_stable_meta_bbox_light_near_signal")
            elif conservative_light_hazard_kind:
                self._add(scores, RoadStructure.R1, 0.80)
                rules.append("r4_meta_tl_without_control_context_demoted_to_r1")
            elif kind == "defect_junction" and not defect_local_control_context:
                self._add(scores, RoadStructure.R1, 0.82)
                rules.append("defect_meta_tl_far_from_local_junction_demoted_to_r1")
            elif turning_tl_requires_local_junction and not turning_local_junction_context:
                self._add(scores, RoadStructure.R1, 0.82)
                rules.append("vehicle_turning_far_meta_tl_without_local_junction_demoted_to_r1")
            else:
                self._add(scores, RoadStructure.R4, 0.86)
                self._add(scores, RoadStructure.R1, 0.62)
                rules.append("r4_meta_tl_without_strong_context_review")
        elif kind == "noscenario" and has_tl and not has_tl_for_r4:
            if noscenario_stop_yield_overrides_weak_light:
                self._add(scores, RoadStructure.R5, 0.86)
                self._add(scores, RoadStructure.R1, 0.70)
                rules.append("noscenario_weak_light_keeps_stop_yield_r5")
            else:
                self._add(scores, RoadStructure.R1, 0.86)
                rules.append("noscenario_weak_far_light_keeps_r1")
        elif (
            not map_is_roundabout
            and scenario_name == "DynamicObjectCrossing"
            and bbox_traffic_light_for_r4
            and not has_tl_for_r4
            and (bbox_stop_sign or bbox_yield_sign or not close_trigger_for_junction)
        ):
            self._add(scores, RoadStructure.R1, 0.82)
            rules.append("dynamic_crossing_bbox_light_without_meta_or_clean_context_demoted_to_r1")
        elif (not map_is_roundabout) and bbox_traffic_light_for_r4 and strong_control_context and dynamic_crossing_strict_junction_context:
            self._add(scores, RoadStructure.R4, 0.90)
            rules.append("r4_bbox_traffic_light_confirmed")
        elif (not map_is_roundabout) and bbox_traffic_light_for_r4:
            if kind == "highway_merge":
                self._add(scores, RoadStructure.R4, 0.68)
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("r4_highway_bbox_tl_without_control_context_demoted")
            elif scenario_name == "NonSignalizedJunctionRightTurn" and not rightturn_local_core:
                self._add(scores, RoadStructure.R1, 0.84)
                rules.append("nonsig_rightturn_far_bbox_tl_demoted_to_r1")
            elif conservative_light_hazard_kind:
                self._add(scores, RoadStructure.R1, 0.80)
                rules.append("r4_bbox_tl_without_control_context_demoted_to_r1")
            elif turning_tl_requires_local_junction and not turning_local_junction_context:
                self._add(scores, RoadStructure.R1, 0.82)
                rules.append("vehicle_turning_far_bbox_tl_without_local_junction_demoted_to_r1")
            else:
                self._add(scores, RoadStructure.R4, 0.78)
                self._add(scores, RoadStructure.R1, 0.70)
                rules.append("r4_bbox_tl_without_strong_context_review")
        elif (not map_is_roundabout) and light_hazard and light_hazard_control_context:
            self._add(scores, RoadStructure.R4, 0.90)
            rules.append("r4_light_hazard")
        elif light_hazard:
            self._add(scores, RoadStructure.R1, 0.78)
            rules.append("light_hazard_ignored_without_junction_context")
        elif (not map_is_roundabout) and static_signal_near_for_r4 and strong_control_context:
            self._add(scores, RoadStructure.R4, 0.74)
            rules.append("r4_static_xodr_signal_near")
        elif (not map_is_roundabout) and static_signal_near_for_r4:
            self._add(scores, RoadStructure.R1, 0.76)
            rules.append("r4_static_signal_without_visual_junction_demoted")

        if (
            RoadStructure.R5 in allowed
            and not map_is_roundabout
            and not has_tl_for_r4
            and not light_hazard
            and junction_window
        ):
            r5_control = stop_hazard_for_r5 or junction_context_for_r5 or (
                xodr_near_junction and not route_projection_error_high and not static_topology_only
            )
            dynamic_crossing_initial_weak_r5 = (
                scenario_name == "DynamicObjectCrossing"
                and route_s < 8.0
                and not near_junction
                and not bbox_junction_hint
            )
            dynamic_crossing_billboard_stop_hint = (
                scenario_name == "DynamicObjectCrossing"
                and route_projection_error_high
                and bbox_traffic_light
                and (bbox_stop_sign or bbox_yield_sign)
                and not meta_stop_hazard
                and not is_junction
                and not bbox_junction_hint
            )
            if r5_control:
                if dynamic_crossing_initial_weak_r5:
                    self._add(scores, RoadStructure.R1, 0.80)
                    rules.append("dynamic_crossing_initial_weak_r5_demoted")
                elif dynamic_crossing_billboard_stop_hint:
                    self._add(scores, RoadStructure.R1, 0.82)
                    rules.append("dynamic_crossing_bbox_stop_light_billboard_hint_demoted_to_r1")
                elif route_projection_error_high and not (stop_hazard_for_r5 or junction_context_for_r5):
                    self._add(scores, RoadStructure.R5, 0.58)
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("r5_generic_demoted_projection_error_rgb_required")
                else:
                    self._add(scores, RoadStructure.R5, 0.84 if (stop_hazard_for_r5 or junction_context_for_r5) else 0.72)
                    rules.append("r5_generic_stop_or_junction_control")

        if (
            scenario_name == "HazardAtSideLaneTwoWays"
            and frame_id < 30
            and not meta_is_junction
            and not bbox_junction_hint
            and not meta_stop_hazard
            and not has_tl
            and not light_hazard
            and bbox_stop_sign
        ):
            self._add(scores, RoadStructure.R2, 0.90)
            rules.append("hazard_side_twoways_initial_bbox_only_stop_demoted_to_r2")

        if (
            (
                kind == "default_meta_map"
                and scenario_name == "ControlLoss"
            )
            or kind == "same_direction_obstacle"
        ) and not map_is_roundabout:
            outside_xml_r4_context = (
                bbox_junction_hint
                or meta_near_junction
                or xodr_near_junction
                or close_trigger_for_junction
            )
            if (
                RoadStructure.R5 in allowed
                and not has_tl
                and not light_hazard
                and stop_hazard_for_r5
            ):
                self._add(scores, RoadStructure.R5, 0.86)
                rules.append(f"{kind}_visible_stop_yield_r5_outside_xml_window")
            if (
                RoadStructure.R4 in allowed
                and has_tl
                and bbox_traffic_light
                and not stop_hazard
                and outside_xml_r4_context
            ):
                self._add(scores, RoadStructure.R4, 0.88)
                rules.append(f"{kind}_meta_tl_bbox_light_r4_outside_xml_window")
            elif (
                RoadStructure.R4 in allowed
                and has_tl
                and bbox_traffic_light
                and not stop_hazard
                and not outside_xml_r4_context
            ):
                rules.append(f"{kind}_traffic_light_without_local_junction_demoted")

        for note in cfg.get("veto", []):
            rules.append(str(note))
        if cfg.get("rule_note"):
            rules.append(str(cfg["rule_note"]))
        raw_has_opposite = xodr_trusted and bool(xodr.get("has_opposite_driving_lane", False))
        raw_has_parking = xodr_trusted and bool(xodr.get("has_parking_or_shoulder_nearby", False))
        has_opposite = xodr_trusted and static_topology_strong and raw_has_opposite
        same_dir_lanes = int(xodr.get("lane_count_same_dir", 1) or 1)
        has_parking = xodr_trusted and static_topology_strong and raw_has_parking
        effective_twoway_drivable_layout = has_opposite and (same_dir_lanes <= 1 or has_parking)
        layout_r2_enabled = bool(route_id and route_id in LAYOUT_R2_ROUTE_IDS.get(scenario_name, set()))
        static_layout_r2_fallback = (
            layout_r2_enabled
            and static_topology_only
            and route_projection_error_high
            and raw_has_opposite
            and (same_dir_lanes <= 1 or raw_has_parking)
        )
        layout_effective_twoway_drivable = effective_twoway_drivable_layout or static_layout_r2_fallback
        ramp_hint = xodr_trusted and static_topology_strong and bool(xodr.get("ramp_merge_split_hint", False))
        twoway_xml_core_close = trigger_distance <= float(cfg.get("two_way_xml_core_close_m", 8.0))
        twoway_strict_core_confirmed = _twoway_strict_core_confirmed(twoway_obstruction, cfg)
        twoway_xml_obstacle_close = (
            twoway_obstruction.has_scenario_obstacles
            and (
                twoway_obstruction.nearest_obstacle_m is None
                or twoway_obstruction.nearest_obstacle_m <= float(cfg.get("two_way_obstacle_core_m", 18.0))
            )
        )
        twoway_xml_core_confirmed = (
            kind == "twoways_obstacle"
            and RoadStructure.R2 in allowed
            and two_way_window
            and (twoway_xml_core_close or (close_trigger and twoway_xml_obstacle_close))
            and (has_opposite or not xodr_trusted or route_projection_error_high)
        )
        if not static_topology_strong:
            if bool(xodr.get("has_opposite_driving_lane", False)):
                rules.append("opposite_lane_hint_demoted_projection_error")
            if bool(xodr.get("has_parking_or_shoulder_nearby", False)):
                rules.append("parking_hint_demoted_projection_error")
            if bool(xodr.get("ramp_merge_split_hint", False)):
                rules.append("merge_split_hint_demoted_projection_error")
        if (
            layout_r2_enabled
            and RoadStructure.R2 in allowed
            and layout_effective_twoway_drivable
            and not map_is_roundabout
            and not route_highway_bucket
            and not (strong_control_context and (junction_window or stop_hazard or has_tl or light_hazard))
        ):
            r2_layout_score = 0.88
            if (has_parking or raw_has_parking) and same_dir_lanes > 1:
                r2_layout_score = 0.86
                rules.append("r2_effective_lane_count_reduced_by_parking_or_shoulder")
            if static_layout_r2_fallback:
                rules.append("r2_layout_static_xodr_projection_error_review")
            self._add(scores, RoadStructure.R2, r2_layout_score)
            rules.append("r2_layout_xodr_effective_twoway_confirmed")

        if kind == "defect_junction":
            if junction_window and defect_local_control_context:
                defect_score = 0.98 if near_junction or has_tl or static_signal_near else 0.74
                self._add(scores, RoadStructure.R4, defect_score)
                rules.append("defect_signal_keeps_r4_with_u7_event")
                if defect_score < 0.90:
                    rules.append("defect_junction_window_without_strong_junction_evidence")
        elif kind == "signalized_junction":
            active_signal_max_m = _safe_float(cfg.get("scenario_active_signal_max_m"), default=trigger_close_m)
            active_signal_window = scenario_active and trigger_distance <= active_signal_max_m
            if junction_window or active_signal_window:
                if has_tl:
                    r4_score = 0.96
                elif near_junction or static_signal_near_for_r4:
                    r4_score = 0.82
                    rules.append("r4_signalized_without_meta_tl_requires_rgb_review")
                else:
                    r4_score = 0.66
                    self._add(scores, RoadStructure.R1, 0.76)
                    rules.append("r4_signalized_window_lacks_visualizable_junction_evidence")
                self._add(scores, RoadStructure.R4, r4_score)
                rules.append("r4_signalized_scenario_window")
                if (
                    scenario_name == "T_Junction"
                    and RoadStructure.R5 in allowed
                    and not (has_tl or light_hazard or static_signal_near_for_r4)
                    and stop_hazard
                ):
                    self._add(scores, RoadStructure.R5, 0.84)
                    rules.append("t_junction_stop_or_yield_no_light_r5")
        elif kind == "blocked_intersection":
            if junction_window or scenario_active:
                signal_control = has_tl or (
                    (bbox_traffic_light or light_hazard or static_signal_near)
                    and strong_control_context
                    and not stop_hazard
                )
                no_light_control = stop_hazard or meta_near_junction or close_trigger_for_junction
                if signal_control:
                    r4_score = 0.96 if has_tl else 0.84
                    self._add(scores, RoadStructure.R4, r4_score)
                    rules.append("blocked_intersection_signalized_r4")
                    if not has_tl:
                        rules.append("blocked_intersection_r4_without_meta_tl_review")
                elif no_light_control:
                    r5_score = 0.97 if stop_hazard else 0.82
                    self._add(scores, RoadStructure.R5, r5_score)
                    rules.append("blocked_intersection_stop_or_nolight_r5")
                else:
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("blocked_intersection_window_lacks_control_source_review")
        elif kind == "nonsignalized_junction":
            nonsig_scenario_active = scenario_active
            if scenario_name == "NonSignalizedJunctionRightTurn":
                nonsig_scenario_active = scenario_active and rightturn_local_core
            if junction_window or stop_hazard_for_r5 or nonsig_scenario_active:
                if meta_near_junction or stop_hazard_for_r5:
                    r5_score = 0.86
                elif xodr_near_junction:
                    r5_score = 0.74
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r5_static_xodr_only_junction_review")
                elif nonsig_scenario_active and close_trigger_for_junction:
                    r5_score = 0.78
                    self._add(scores, RoadStructure.R1, 0.70)
                    rules.append("r5_active_close_trigger_without_stop_review")
                else:
                    r5_score = 0.56
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("r5_demoted_without_stop_yield_or_junction_evidence")
                self._add(scores, RoadStructure.R5, r5_score)
                rules.append("r5_nonsignalized_junction_window")
            local_route_window = _route_trigger_window(
                route_s_for_window,
                trigger_s,
                junction_pre_window,
                junction_post_window,
            )
            if (
                not scenario_active
                and not is_junction
                and not close_trigger_for_junction
                and not local_route_window
            ):
                self._add(scores, RoadStructure.R1, 0.90)
                rules.append("nonsignalized_outside_local_junction_core_demoted_to_r1")
            route_lateral_offset = RoadEventRuleEngine._route_lateral_offset(frame_data)
            route_centered_straight = (
                math.isfinite(route_lateral_offset)
                and abs(route_lateral_offset) < 0.04
                and RoadEventRuleEngine._speed(frame_data) > 5.0
            )
            if route_centered_straight and not is_junction:
                self._add(scores, RoadStructure.R1, 0.90)
                rules.append("nonsignalized_moving_centered_straight_segment_demoted_to_r1")
            if has_tl or static_signal_near:
                rules.append("nonsig_with_signal_conflict_review")
        elif kind in {"twoways_obstacle", "invading_turn"}:
            r2_layout_prior_allowed = (
                kind == "twoways_obstacle"
                and two_way_layout_prior_enabled
                and RoadStructure.R2 in allowed
                and not (
                    xodr_trusted
                    and not static_topology_only
                    and static_topology_strong
                    and not has_opposite
                    and same_dir_lanes > 1
                    and not has_parking
                )
            )
            r2_topology_confirmed = effective_twoway_drivable_layout
            r2_core_meta_confirmed = (
                kind == "twoways_obstacle"
                and twoway_strict_core_confirmed
                and (scenario_active_for_structure or twoway_obstruction.has_scenario_obstacles)
            )
            r2_core_allowed_by_meta = r2_core_meta_confirmed and (
                two_way_window
                or close_trigger_for_structure
                or route_projection_error_high
                or scenario_active_for_structure
            )
            if RoadStructure.R2 in allowed and (two_way_window or r2_core_allowed_by_meta):
                if (
                    (r2_topology_confirmed and (twoway_obstruction.core_confirmed or kind == "invading_turn"))
                    or r2_core_meta_confirmed
                    or twoway_xml_core_confirmed
                ):
                    r2_score = 0.93 if r2_core_allowed_by_meta else (0.90 if not twoway_xml_core_confirmed else 0.88)
                    self._add(scores, RoadStructure.R2, r2_score)
                    if r2_topology_confirmed:
                        rules.append("r2_opposite_lane_confirmed")
                    elif twoway_xml_core_confirmed:
                        rules.append("r2_xml_trigger_core_confirmed")
                    else:
                        rules.append("r2_core_obstruction_meta_confirmed_without_trusted_xodr")
                    if kind == "twoways_obstacle":
                        rules.append("r2_core_obstruction_confirmed")
                        if twoway_strict_core_confirmed:
                            rules.append("r2_strict_core_obstruction_window")
                else:
                    if r2_layout_prior_allowed and r2_topology_confirmed:
                        self._add(scores, RoadStructure.R2, 0.86)
                        rules.append("r2_twoways_effective_single_lane_topology_confirmed")
                        if has_parking and same_dir_lanes > 1:
                            rules.append("r2_effective_lane_count_reduced_by_parking")
                    elif r2_layout_prior_allowed:
                        self._add(scores, RoadStructure.R2, 0.84)
                        rules.append("r2_twoways_layout_prior_effective_drivable_default")
                        if not has_opposite:
                            rules.append("r2_layout_prior_lacks_xodr_opposite_confirmation")
                    else:
                        self._add(scores, RoadStructure.R2, 0.76)
                        rules.append("r2_scenario_trigger_medium")
                    if r2_topology_confirmed:
                        rules.append("r2_waits_for_close_obstruction_or_vehicle_interaction")
                    else:
                        rules.append("r2_requires_visible_or_topology_occupancy_confirmation")
            elif r2_layout_prior_allowed and r2_topology_confirmed:
                self._add(scores, RoadStructure.R2, 0.86)
                rules.append("r2_twoways_effective_single_lane_topology_confirmed")
                if has_parking and same_dir_lanes > 1:
                    rules.append("r2_effective_lane_count_reduced_by_parking")
            elif r2_layout_prior_allowed:
                self._add(scores, RoadStructure.R2, 0.84)
                rules.append("r2_twoways_layout_prior_effective_drivable_default")
                if not has_opposite:
                    rules.append("r2_layout_prior_lacks_xodr_opposite_confirmation")
            if kind == "invading_turn":
                rules.append("r2_passive_invading_turn")
                invading_bbox_stop_context = (bbox_stop_sign or bbox_yield_sign) and (
                    bbox_junction_hint
                    or meta_near_junction
                    or (xodr_near_junction and not route_projection_error_high and not static_topology_only)
                )
                invading_stop_or_junction_control = meta_stop_hazard or is_junction or invading_bbox_stop_context
                if RoadStructure.R5 in allowed and junction_window and not map_is_roundabout:
                    if invading_stop_or_junction_control:
                        self._add(scores, RoadStructure.R5, 0.86)
                        rules.append("invading_turn_nonsignalized_stop_or_junction_r5")
                    elif scenario_active and close_trigger_for_junction and (
                        bbox_junction_hint
                        or meta_near_junction
                        or (xodr_near_junction and not route_projection_error_high and not static_topology_only)
                    ):
                        self._add(scores, RoadStructure.R5, 0.80)
                        rules.append("invading_turn_active_close_trigger_r5")
                    elif scenario_active and close_trigger_for_junction:
                        self._add(scores, RoadStructure.R1, 0.78)
                        rules.append("invading_turn_close_trigger_without_junction_control_demoted")
                    elif scenario_active and near_junction:
                        self._add(scores, RoadStructure.R5, 0.82)
                        self._add(scores, RoadStructure.R1, 0.72)
                        rules.append("invading_turn_active_meta_junction_r5")
                    elif near_junction:
                        self._add(scores, RoadStructure.R5, 0.78)
                        self._add(scores, RoadStructure.R1, 0.72)
                        rules.append("invading_turn_meta_junction_r5_review")
                    else:
                        self._add(scores, RoadStructure.R1, 0.72)
                        rules.append("invading_turn_junction_window_lacks_nonsig_control_review")
                elif RoadStructure.R5 in allowed and not map_is_roundabout:
                    if invading_stop_or_junction_control and close_trigger:
                        self._add(scores, RoadStructure.R5, 0.82)
                        rules.append("invading_turn_stop_close_trigger_r5")
                    elif scenario_active and trigger_distance < min(trigger_close_m, 45.0):
                        self._add(scores, RoadStructure.R1, 0.78)
                        rules.append("invading_turn_near_trigger_without_junction_control_demoted")
            if kind == "twoways_obstacle" and RoadStructure.R4 in allowed and not twoway_xml_core_confirmed:
                post_core_signal_near = (
                    xodr_trusted
                    and static_topology_strong
                    and not map_is_roundabout
                    and _safe_float(xodr.get("nearest_signal_m"), default=math.inf)
                    <= float(cfg.get("two_way_post_core_signal_m", 45.0)) * low_visibility_factor
                )
                post_core_junction_context = (
                    meta_near_junction
                    or stop_hazard
                    or light_hazard
                    or xodr_near_junction
                )
                if has_tl:
                    self._add(scores, RoadStructure.R4, 0.88)
                    rules.append("twoways_post_core_meta_tl_r4")
                elif post_core_signal_near and post_core_junction_context and strong_control_context:
                    self._add(scores, RoadStructure.R4, 0.84)
                    rules.append("twoways_post_core_xodr_signal_r4")
        elif kind == "highway_merge":
            keep_r3_when_slow = bool(cfg.get("keep_r3_when_slow"))
            enter_actor_flow_tight = scenario_name in {"EnterActorFlow", "EnterActorFlowV2"}
            highway_default_r3 = bool(cfg.get("highway_default_r3"))
            merge_xml_fallback = (
                bool(cfg.get("keep_r3_when_slow"))
                or enter_actor_flow_tight
            ) and (
                actor_flow_near
                or (close_trigger and not enter_actor_flow_tight)
            )
            scenario_active_merge_window = scenario_active_for_structure and not keep_r3_when_slow
            merge_window = (
                _route_trigger_window(
                    route_s_for_window,
                    trigger_s,
                    float(cfg.get("merge_pre_m", 50.0)),
                    float(cfg.get("merge_post_m", 50.0)),
                )
                or close_trigger_for_structure
                or (keep_r3_when_slow and close_trigger)
                or scenario_active_merge_window
                or actor_flow_near
            )
            if merge_window and RoadStructure.R3 in allowed:
                if ramp_hint:
                    r3_score = 0.88
                elif merge_xml_fallback:
                    r3_score = 0.84 if actor_flow_near or (close_trigger and not enter_actor_flow_tight) else 0.80
                    rules.append("r3_merger_actor_flow_or_trigger_fallback")
                elif xodr.get("xodr_available"):
                    r3_score = 0.78 if highway_default_r3 else 0.50
                    if not highway_default_r3:
                        self._add(scores, RoadStructure.R1, 0.80)
                    rules.append(
                        "r3_highway_scene_default_without_merge_split"
                        if highway_default_r3
                        else "r3_xodr_available_without_merge_split_review"
                    )
                else:
                    r3_score = 0.76 if highway_default_r3 else 0.45
                    if not highway_default_r3:
                        self._add(scores, RoadStructure.R1, 0.80)
                    rules.append(
                        "r3_highway_scene_default_without_xodr"
                        if highway_default_r3
                        else "r3_without_xodr_topology_low"
                    )
                self._add(scores, RoadStructure.R3, r3_score)
                rules.append("r3_merge_or_exit_window")
            elif highway_default_r3 and RoadStructure.R3 in allowed:
                self._add(scores, RoadStructure.R3, 0.76)
                rules.append("r3_highway_scene_default_outside_trigger_window")
        elif kind == "interurban":
            merge_window = (
                _route_trigger_window(
                    route_s_for_window,
                    trigger_s,
                    float(cfg.get("merge_pre_m", 50.0)),
                    float(cfg.get("merge_post_m", 45.0)),
                )
                or scenario_active_for_structure
            )
            if merge_window and RoadStructure.R3 in allowed:
                if ramp_hint:
                    self._add(scores, RoadStructure.R3, 0.82)
                else:
                    self._add(scores, RoadStructure.R3, 0.50)
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("interurban_r3_lacks_merge_topology_review")
                rules.append("interurban_r3_actor_flow_window")
            if junction_window:
                if has_tl:
                    self._add(scores, RoadStructure.R4, 0.95)
                    rules.append("interurban_junction_r4")
                elif stop_hazard or is_junction or (xodr_near_junction and not route_projection_error_high and not static_topology_only):
                    if route_projection_error_high and (stop_hazard or is_junction):
                        self._add(scores, RoadStructure.R5, 0.86 if stop_hazard else 0.82)
                        rules.append("interurban_rgb_reviewed_stop_or_junction_r5")
                    elif route_projection_error_high:
                        self._add(scores, RoadStructure.R5, 0.62)
                        self._add(scores, RoadStructure.R1, 0.78)
                        rules.append("interurban_r5_demoted_projection_error_rgb_required")
                    else:
                        self._add(scores, RoadStructure.R5, 0.82 if (stop_hazard or is_junction) else 0.72)
                        rules.append("interurban_junction_r5_medium")
                elif scenario_active and close_trigger_for_junction:
                    self._add(scores, RoadStructure.R1, 0.84)
                    rules.append("interurban_active_close_trigger_without_junction_control_demoted_to_r1")
                else:
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("interurban_junction_window_lacks_visible_control_review")
        elif kind == "interurban_advanced":
            if junction_window:
                if has_tl:
                    self._add(scores, RoadStructure.R4, 0.85)
                    rules.append("advanced_actor_flow_junction_r4")
                elif stop_hazard or is_junction or (xodr_near_junction and not route_projection_error_high and not static_topology_only):
                    if route_projection_error_high:
                        self._add(scores, RoadStructure.R5, 0.62)
                        self._add(scores, RoadStructure.R1, 0.78)
                        rules.append("advanced_actor_flow_r5_demoted_projection_error_rgb_required")
                    else:
                        self._add(scores, RoadStructure.R5, 0.82 if (stop_hazard or is_junction) else 0.72)
                        rules.append("advanced_actor_flow_junction_r5")
                else:
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("advanced_actor_flow_junction_window_lacks_visible_control_review")
            if ramp_hint:
                self._add(scores, RoadStructure.R3, 0.58)
                rules.append("advanced_actor_flow_r3_only_with_topology")
        elif kind in {"parking", "parking_exit"}:
            parking_window = (
                _route_trigger_window(
                    route_s_for_window,
                    trigger_s,
                    float(cfg.get("parking_pre_m", 35.0)),
                    float(cfg.get("parking_post_m", 60.0)),
                )
                or close_trigger_for_structure
                or scenario_active_for_structure
            )
            if parking_window and RoadStructure.R1 in allowed:
                self._add(scores, RoadStructure.R1, 0.84 if has_parking else 0.80)
                rules.append("parking_context_kept_as_r1")
                if kind == "parking_exit":
                    rules.append("parking_exit_merge_expressed_by_re2_event")
        elif kind == "vehicle_opens_door_twoways":
            door_window = two_way_window
            if RoadStructure.R2 in allowed:
                if effective_twoway_drivable_layout:
                    self._add(scores, RoadStructure.R2, 0.88)
                    rules.append("vehicle_open_door_effective_single_lane_r2")
                    if has_parking and same_dir_lanes > 1:
                        rules.append("r2_effective_lane_count_reduced_by_parking")
                else:
                    self._add(scores, RoadStructure.R2, 0.84 if two_way_layout_prior_enabled else 0.76)
                    rules.append("vehicle_open_door_r2_lacks_opposite_confirmation")
                if door_window:
                    rules.append("vehicle_open_door_r2_possible")
        elif kind == "static_cutin":
            cutin_window = (
                _route_trigger_window(
                    route_s_for_window,
                    trigger_s,
                    float(cfg.get("parking_pre_m", 35.0)),
                    float(cfg.get("parking_post_m", 55.0)),
                )
                or close_trigger_for_structure
                or scenario_active_for_structure
            )
            if cutin_window and ramp_hint and RoadStructure.R3 in allowed:
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("static_cutin_r3_merge_side")
            elif cutin_window:
                self._add(scores, RoadStructure.R1, 0.80)
                rules.append("static_cutin_same_direction_r1")
        elif kind == "pedestrian_crossing":
            if junction_window:
                rs = RoadStructure.R4 if has_tl else RoadStructure.R5
                self._add(scores, rs, 0.86 if has_tl else 0.70)
                rules.append("pedestrian_crossing_junction_space")
        elif kind == "vehicle_turning":
            if junction_window:
                if has_tl:
                    self._add(scores, RoadStructure.R4, 0.86)
                    rules.append("vehicle_turning_junction_space")
                elif route_projection_error_high and not (
                    stop_hazard
                    or is_junction
                    or (xodr_near_junction and not static_topology_only)
                ):
                    self._add(scores, RoadStructure.R1, 0.78)
                    self._add(scores, RoadStructure.R5, 0.58)
                    rules.append("vehicle_turning_r5_demoted_projection_error_rgb_required")
                else:
                    self._add(scores, RoadStructure.R5, 0.70)
                    rules.append("vehicle_turning_junction_space")
        elif kind == "hardbreak_route":
            if route_highway_bucket:
                self._add(scores, RoadStructure.R3, 0.84 if ramp_hint else 0.80)
                rules.append("hardbreak_rgb_route_bucket_r3_no_r1")
            else:
                self._add(scores, RoadStructure.R1, 0.78)
                rules.append("hardbreak_event_keeps_r1_unless_highway_like_or_tl")
        elif kind == "noscenario":
            noscenario_r3_allowed = (
                RoadStructure.R3 in allowed
                and not map_is_roundabout
                and not (
                    strong_control_context
                    and (junction_window or stop_hazard or has_tl or light_hazard)
                )
            )
            if noscenario_r3_allowed and ramp_hint:
                self._add(scores, RoadStructure.R3, 0.82)
                rules.append("r3_noscenario_xodr_ramp_merge_split")
            elif noscenario_r3_allowed and route_highway_bucket:
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("r3_noscenario_highway_route_bucket")
            if (
                not has_tl
                and not light_hazard
                and RoadStructure.R5 not in scores
                and RoadStructure.R2 not in scores
                and RoadStructure.R3 not in scores
            ):
                scores = {RoadStructure.R1: max(scores.get(RoadStructure.R1, 0.0), 0.86)}
                rules.append("noscenario_conservative_r1_without_meta_light")
        else:
            if kind == "default_meta_map" and RoadStructure.R5 in allowed and junction_window:
                if stop_hazard or is_junction or (
                    xodr_near_junction and not route_projection_error_high and not static_topology_only
                ):
                    if (
                        scenario_name == "DynamicObjectCrossing"
                        and route_s < 8.0
                        and not near_junction
                        and not bbox_junction_hint
                    ):
                        self._add(scores, RoadStructure.R1, 0.80)
                        rules.append("dynamic_crossing_initial_default_r5_demoted")
                    elif (
                        scenario_name == "DynamicObjectCrossing"
                        and route_projection_error_high
                        and bbox_traffic_light
                        and (bbox_stop_sign or bbox_yield_sign)
                        and not meta_stop_hazard
                        and not is_junction
                        and not bbox_junction_hint
                    ):
                        self._add(scores, RoadStructure.R1, 0.84)
                        rules.append("dynamic_crossing_default_bbox_stop_light_billboard_hint_demoted_to_r1")
                    elif route_projection_error_high and not (stop_hazard or is_junction):
                        self._add(scores, RoadStructure.R1, 0.78)
                        self._add(scores, RoadStructure.R5, 0.58)
                        rules.append("default_meta_map_r5_demoted_projection_error_rgb_required")
                    else:
                        self._add(scores, RoadStructure.R5, 0.82 if (stop_hazard or is_junction) else 0.70)
                        rules.append("default_meta_map_stop_or_junction_r5")
                else:
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append(f"{kind}_junction_window_lacks_stop_or_junction_control")
            # SameDirectionObstacle 与 DefaultMetaMapPolicy 默认只允许灯态把主标签提升到 R4；
            # 少数明确开放 R5 的场景需额外满足上面的 STOP/无灯路口同源证据。
            self._add(scores, RoadStructure.R1, 0.78)
            rules.append(f"{kind}_keeps_default_r1_unless_tl")

        if map_is_roundabout:
            if RoadStructure.R4 in scores or RoadStructure.R5 in scores:
                rules.append("roundabout_removed_junction_rs_scores")
            scores.pop(RoadStructure.R4, None)
            scores.pop(RoadStructure.R5, None)
            self._add(scores, RoadStructure.R1, 0.92)

        if (
            scores.get(RoadStructure.R1, 0.0) <= 0.35
            and not any(rs != RoadStructure.R1 and score >= 0.60 for rs, score in scores.items())
        ):
            self._add(scores, RoadStructure.R1, 0.78)
            rules.append("r1_stable_no_special_structure_confirmed")

        # 只保留原始候选表允许的 RS，保留强行填充候选全集但不让规则输出越界。
        scores = {rs: score for rs, score in scores.items() if rs in allowed}
        if not scores:
            if (kind == "highway_merge" or route_highway_bucket) and RoadStructure.R3 in allowed:
                scores = {RoadStructure.R3: 0.76}
                rules.append("r3_highway_candidate_fallback_no_r1")
            elif RoadStructure.R1 in allowed:
                scores = {RoadStructure.R1: 0.35}
            elif allowed:
                fallback_rs = next(iter(allowed))
                scores = {fallback_rs: 0.35}
                rules.append(f"fallback_to_allowed_{fallback_rs.value}_without_r1_candidate")
            else:
                scores = {RoadStructure.R1: 0.35}

        if (
            kind == "noscenario"
            and RoadStructure.R4 not in scores
            and RoadStructure.R5 not in scores
            and RoadStructure.R2 not in scores
            and RoadStructure.R3 not in scores
        ):
            primary = RoadStructure.R1
        else:
            max_score = max(scores.values())
            priority = self.PRIORITY
            if (
                scenario_name == "AccidentTwoWays"
                and RoadStructure.R2 in scores
                and any(
                    rule in rules
                    for rule in (
                        "r2_core_obstruction_confirmed",
                        "r2_strict_core_obstruction_window",
                        "r2_core_obstruction_meta_confirmed_without_trusted_xodr",
                        "r2_xml_trigger_core_confirmed",
                    )
                )
            ):
                priority = [RoadStructure.R2, RoadStructure.R4, RoadStructure.R5, RoadStructure.R3, RoadStructure.R1]
                rules.append("accidenttwoways_r2_core_priority_over_junction")
            primary = max(scores, key=lambda rs: (scores[rs], -priority.index(rs) if rs in priority else -99))
            # 分数接近时按全局优先级仲裁，但视觉复核显示弱特殊 RS 不能低分压过 R1。
            for rs in priority:
                if rs not in scores or scores[rs] < max_score - 0.08:
                    continue
                if primary == RoadStructure.R1 and rs != RoadStructure.R1 and scores[rs] < scores[RoadStructure.R1]:
                    rules.append("priority_tiebreak_kept_r1_over_weaker_special_rs")
                    continue
                if rs != RoadStructure.R1 and RoadStructure.R1 in scores and scores[rs] < scores[RoadStructure.R1]:
                    rules.append("priority_tiebreak_skipped_weaker_special_rs")
                    continue
                else:
                    primary = rs
                    break

        secondary = {
            rs for rs, score in scores.items()
            if rs != primary and score >= 0.60 and (primary in {RoadStructure.R4, RoadStructure.R5} or rs in {RoadStructure.R2, RoadStructure.R3})
        }
        confidence = scores.get(primary, 0.35)
        reason = f"{scenario_name}: primary={primary.value}, rules={','.join(rules[:4])}"
        diagnostic_attribution = _diagnose_rs_decision(
            scenario_name=scenario_name,
            kind=kind,
            primary=primary,
            scores=scores,
            rules=rules,
            xml_info=xml_info,
            xodr=xodr,
            flags={
                "has_tl": has_tl,
                "light_hazard": light_hazard,
                "bbox_traffic_light": bbox_traffic_light,
                "bbox_stop_or_yield": bbox_stop_sign or bbox_yield_sign,
                "bbox_junction_hint": bbox_junction_hint,
                "is_junction": is_junction,
                "map_is_roundabout": map_is_roundabout,
                "dist_to_junction_near": dist_to_junction_near,
                "dist_to_junction_strong": dist_to_junction_strong,
                "stop_hazard": stop_hazard,
                "scenario_active": scenario_active,
                "close_trigger": close_trigger,
                "close_trigger_for_structure": close_trigger_for_structure,
                "close_trigger_for_junction": close_trigger_for_junction,
                "defect_local_control_context": defect_local_control_context,
                "turning_local_junction_context": turning_local_junction_context,
                "turning_local_junction_evidence": turning_local_junction_evidence,
                "scenario_active_for_structure": scenario_active_for_structure,
                "near_junction": near_junction,
                "strong_control_context": strong_control_context,
                "static_signal_near": static_signal_near,
                "junction_window": junction_window,
                "two_way_window": two_way_window,
                "two_way_layout_prior": bool(
                    kind == "twoways_obstacle"
                    and two_way_layout_prior_enabled
                    and RoadStructure.R2 in allowed
                    and not (
                        xodr_trusted
                        and not static_topology_only
                        and static_topology_strong
                        and not has_opposite
                        and same_dir_lanes > 1
                        and not has_parking
                    )
                ),
                "layout_r2_enabled": layout_r2_enabled,
                "effective_twoway_drivable_layout": effective_twoway_drivable_layout,
                "layout_effective_twoway_drivable": layout_effective_twoway_drivable,
                "twoway_core_obstruction": twoway_obstruction.core_confirmed
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else False,
                "twoway_strict_core_obstruction": twoway_strict_core_confirmed
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else False,
                "twoway_xml_core_close": twoway_xml_core_close,
                "twoway_xml_obstacle_close": twoway_xml_obstacle_close,
                "twoway_xml_core_confirmed": twoway_xml_core_confirmed,
                "rightturn_local_core": rightturn_local_core,
                "route_projection_error_high": route_error > 5.0 if math.isfinite(route_error) else False,
            },
        )
        review_reasons = []
        if confidence < 0.70:
            review_reasons.append("low_confidence")
        if route_projection_error_high:
            review_reasons.append("route_projection_error_high")
        score_gap = diagnostic_attribution.get("top_score_gap")
        if score_gap is not None and score_gap < 0.15:
            review_reasons.append("candidate_score_gap_lt_0.15")
        weak_inputs = diagnostic_attribution.get("weak_or_missing_inputs", [])
        if primary in {RoadStructure.R2, RoadStructure.R3} and weak_inputs:
            review_reasons.append("special_rs_lacks_full_topology_confirmation")
        if route_projection_error_high and (close_trigger or scenario_active) and not (close_trigger_for_structure or scenario_active_for_structure):
            review_reasons.append("structure_window_demoted_by_projection_error")
        if static_topology_only and route_projection_error_high and xodr.get("xodr_available"):
            review_reasons.append("static_xodr_topology_demoted_by_projection_error")
        if "priority_tiebreak_kept_r1_over_weaker_special_rs" in rules:
            review_reasons.append("weaker_special_rs_kept_as_candidate_not_primary")
        if kind == "signalized_junction" and cfg.get("review_if_no_tl") and primary == RoadStructure.R4 and not has_tl:
            review_reasons.append("signalized_policy_without_meta_tl")
        if kind == "signalized_junction" and primary == RoadStructure.R4 and not has_tl:
            review_reasons.append("signalized_r4_without_meta_tl_requires_rgb_confirmation")
        if kind == "blocked_intersection" and primary == RoadStructure.R4 and not has_tl:
            review_reasons.append("blocked_r4_without_meta_tl_requires_rgb_confirmation")
        if primary == RoadStructure.R4 and "r4_meta_tl_without_strong_context_review" in rules:
            review_reasons.append("r4_meta_tl_without_strong_context_requires_rgb_confirmation")
        if primary == RoadStructure.R4 and "r4_bbox_tl_without_strong_context_review" in rules:
            review_reasons.append("r4_bbox_tl_without_strong_context_requires_rgb_confirmation")
        if any(rule.startswith("r4_highway_") and rule.endswith("_demoted") for rule in rules):
            review_reasons.append("highway_light_hint_demoted_without_control_context")
        if "r5_demoted_without_stop_yield_or_junction_evidence" in rules:
            review_reasons.append("r5_requires_stop_yield_priority_or_junction_evidence")
        if kind == "nonsignalized_junction" and "nonsig_with_signal_conflict_review" in rules:
            review_reasons.append("nonsignalized_with_signal_topology_conflict")
        evidence = {
            "rules_fired": rules,
            "rule_kind": kind,
            "rule_config": cfg,
            "xml_path": str(xml_info.path) if xml_info else None,
            "xml_town": town,
            "route_progress_m": route_s if math.isfinite(route_s) else None,
            "route_projection_error_m": route_error if math.isfinite(route_error) else None,
            "trigger_distance_m": trigger_distance if math.isfinite(trigger_distance) else None,
            "distance_to_intersection_index_ego_m": (
                rightturn_intersection_distance
                if math.isfinite(rightturn_intersection_distance)
                else None
            ),
            "rightturn_local_core": rightturn_local_core,
            "actor_flow_distance_m": actor_flow_distance if math.isfinite(actor_flow_distance) else None,
            "junction_window_config": {
                "junction_pre_m": junction_pre,
                "junction_post_m": junction_post,
                "effective_pre_m": round(junction_pre_window, 3),
                "effective_post_m": round(junction_post_window, 3),
                "pre_scale": JUNCTION_PRE_WINDOW_SCALE,
                "post_scale": JUNCTION_POST_WINDOW_SCALE,
                "scenario_tighten_factor": round(scenario_junction_factor, 3),
                "scenario_min_pre_m": round(scenario_min_pre, 3),
                "scenario_min_post_m": round(scenario_min_post, 3),
                "meta_near_m": JUNCTION_META_NEAR_M,
                "effective_meta_near_m": round(effective_meta_near_m, 3),
                "strong_max_m": JUNCTION_STRONG_MAX_M,
                "effective_strong_max_m": round(effective_strong_max_m, 3),
                "static_signal_near_m": STATIC_SIGNAL_NEAR_M,
                "effective_static_signal_near_m": round(effective_static_signal_near_m, 3),
                "close_trigger_max_m": JUNCTION_CLOSE_TRIGGER_MAX_M,
                "effective_close_trigger_max_m": round(effective_close_trigger_max_m, 3),
                "low_visibility_factor": round(low_visibility_factor, 3),
                "low_visibility_reasons": low_visibility_reasons,
            },
            "xml_weather": xml_weather,
            "xml_weather_route_percentage": (
                round(xml_weather_route_percentage, 3)
                if xml_weather_route_percentage is not None
                else None
            ),
            "traffic_light_state": str(tl) if tl is not None else None,
            "bbox_semantics": {
                "available": _safe_bool(frame_data.get("bbox_available", False)),
                "traffic_light": bbox_traffic_light,
                "stop_sign": bbox_stop_sign,
                "yield_sign": bbox_yield_sign,
                "junction_hint": bbox_junction_hint,
                "classes": frame_data.get("bbox_semantic_classes", {}),
                "metrics": bbox_metrics,
            },
            "meta_is_junction": meta_is_junction,
            "meta_stop_hazard": meta_stop_hazard,
            "combined_stop_hazard": stop_hazard,
            "current_active_scenario_type": active or None,
            "strong_control_context": strong_control_context,
            "twoway_obstruction_evidence": (
                twoway_obstruction.to_dict()
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else None
            ),
            "xodr": xodr,
            "diagnostic_attribution": diagnostic_attribution,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "route_id": route_id,
            "route_semantic_bucket": route_semantic_bucket,
            "mixed_scenario_rgb_review_total_routes": MIXED_SCENARIO_RGB_REVIEW_COUNTS.get(scenario_name),
            "mixed_scenario_highway_route_count": (
                len(MIXED_SCENARIO_HIGHWAY_ROUTE_IDS[scenario_name])
                if scenario_name in MIXED_SCENARIO_HIGHWAY_ROUTE_IDS
                else None
            ),
        }
        return primary, secondary, {rs.value: round(score, 3) for rs, score in scores.items()}, evidence, confidence, reason


# 简化策略工厂 - 保留原始候选全集，同时追加规则生成的主 ROAD_STRUCTURE
class SimpleFrameAnalyzer:
    """帧分析器：保留候选全集，并根据 XML/XODR/meta 生成 primary RS。"""

    _engine = RoadStructureRuleEngine()

    @staticmethod
    def configure_xodr_probe(xodr_probe: Optional[XodrTopologyProbe]) -> None:
        SimpleFrameAnalyzer._engine = RoadStructureRuleEngine(xodr_probe)

    @staticmethod
    def analyze(
        scenario_name: str,
        frame_id: int,
        frame_data: dict,
        xml_info: Optional[RouteXmlInfo] = None,
        route_id: Optional[str] = None,
    ) -> FrameAnnotation:
        """通用帧分析"""
        # 保留旧逻辑：road_structures 仍是该 scenario 的候选全集。
        road_structures = _mixed_route_allowed_structures(
            scenario_name,
            route_id,
            set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [RoadStructure.R1])),
        )
        if (
            (
                _valid_traffic_light(frame_data.get("traffic_light_state"))
                or _safe_bool(frame_data.get("light_hazard", False))
                or _safe_bool(frame_data.get("bbox_has_traffic_light", False))
            )
            and scenario_name not in SCENARIOS_WITH_RGB_NO_R4
        ):
            road_structures.add(RoadStructure.R4)
        primary, secondary, scores, evidence, confidence, reason = SimpleFrameAnalyzer._engine.analyze(
            scenario_name=scenario_name,
            frame_id=frame_id,
            frame_data=frame_data,
            xml_info=xml_info,
            route_id=route_id,
        )
        events, primary_event, event_evidence = RoadEventRuleEngine.analyze(
            scenario_name=scenario_name,
            frame_data=frame_data,
            primary_rs=primary,
            evidence=evidence,
        )

        comment = _frame_annotation_comment(primary, secondary, confidence, evidence)
        return FrameAnnotation(
            frame_id=frame_id,
            road_structures=road_structures,
            events=events,
            primary_event=primary_event,
            confidence=confidence,
            reason=reason,
            primary_road_structure=primary,
            secondary_road_structures=secondary,
            candidate_scores=scores,
            evidence=evidence,
            event_evidence=event_evidence,
            annotation_comment=comment,
        )


class ScenarioCollector:
    """灵活的采集器 - 支持4种采集模式"""

    def __init__(self, lead_data_root: str = "",
                 output_dir: str = "",
                 xml_root: str = "",
                 carla_root: str = "",
                 rule_config_json: str = ""):
        if rule_config_json:
            load_rule_config_overrides(rule_config_json)
        self.lead_data_root = Path(
            lead_data_root or os.environ.get("LEAD_DATA_ROOT", "") or _DEFAULT_LEAD_DATA_ROOT
        )
        self.output_dir = Path(
            output_dir or os.environ.get("KEYFRAME_COLLECTION_OUTPUT", "") or _DEFAULT_OUTPUT_DIR
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.xml_index = RouteXmlIndex(Path(xml_root) if xml_root else _DEFAULT_XML_ROOT)
        self.xodr_probe = XodrTopologyProbe(Path(carla_root) if carla_root else _DEFAULT_CARLA_ROOT)
        SimpleFrameAnalyzer.configure_xodr_probe(self.xodr_probe)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # ========================================================================
    # 4种采集模式
    # ========================================================================

    def collect_one_scenario_all(
        self,
        scenario_name: str,
        max_frames_per_route: Optional[int] = None,
        samples_per_town: Optional[int] = None,
    ) -> Dict:
        """模式1: 单场景全部采集 - 采集该场景的所有routes"""
        return self._collect_scenario(
            scenario_name,
            max_routes=None,
            max_frames_per_route=max_frames_per_route,
            samples_per_town=samples_per_town,
        )

    def collect_one_scenario(
        self,
        scenario_name: str,
        max_routes: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
        samples_per_town: Optional[int] = None,
    ) -> Dict:
        """模式2: 单场景采集；max_routes=None 时采集全部合法 routes。"""
        return self._collect_scenario(
            scenario_name,
            max_routes=max_routes,
            max_frames_per_route=max_frames_per_route,
            samples_per_town=samples_per_town,
        )

    def collect_multiple_scenarios(
        self,
        scenario_names: List[str],
        max_routes_per_scenario: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
        samples_per_town: Optional[int] = None,
    ) -> Dict:
        """模式3: 多场景全部采集 - 采集多个指定场景"""
        self.logger.info(f"采集多场景: {scenario_names}")

        all_results = {}
        total_frames = 0

        for scenario in scenario_names:
            try:
                result = self._collect_scenario(
                    scenario,
                    max_routes=max_routes_per_scenario,
                    max_frames_per_route=max_frames_per_route,
                    samples_per_town=samples_per_town,
                )
                all_results[scenario] = result
                total_frames += result.get('total_frames', 0)
                self.logger.info(f"  ✓ {scenario}: {result.get('total_frames', 0)} 帧")
            except Exception as e:
                self.logger.error(f"  ✗ {scenario}: {e}")
                all_results[scenario] = {"status": "error", "error": str(e)}

        # 保存综合结果
        summary = {
            "status": "success",
            "scenarios_collected": len([r for r in all_results.values() if r.get('status') == 'success']),
            "total_scenarios": len(scenario_names),
            "total_frames": total_frames,
            "results": all_results
        }

        output_file = self.output_dir / "multi_scenario_collection.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        return summary

    def collect_all_scenarios(
        self,
        max_routes_per_scenario: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
        samples_per_town: Optional[int] = None,
    ) -> Dict:
        """模式4: 全部采集 - 默认采集所有场景的全部合法 routes。"""
        all_scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
        self.logger.info(f"采集所有场景 ({len(all_scenarios)}个)")

        return self.collect_multiple_scenarios(
            all_scenarios,
            max_routes_per_scenario,
            max_frames_per_route=max_frames_per_route,
            samples_per_town=samples_per_town,
        )

    # ========================================================================
    # 内部实现
    # ========================================================================

    def _select_route_dirs(self, route_dirs: List[Path], max_routes: Optional[int]) -> List[Path]:
        """分散抽样 route，避免只看排序最前面的同 town/id。"""
        if max_routes is None or len(route_dirs) <= max_routes:
            return route_dirs
        max_routes = max(1, max_routes)
        if max_routes == 1:
            return [route_dirs[0]]
        indexes = sorted({round(i * (len(route_dirs) - 1) / (max_routes - 1)) for i in range(max_routes)})
        selected = [route_dirs[i] for i in indexes]
        if len(selected) < max_routes:
            selected_names = {p.name for p in selected}
            for route_dir in route_dirs:
                if route_dir.name in selected_names:
                    continue
                selected.append(route_dir)
                selected_names.add(route_dir.name)
                if len(selected) >= max_routes:
                    break
        return selected

    def _select_route_dirs_per_town(self, route_dirs: List[Path], samples_per_town: Optional[int]) -> List[Path]:
        """每个 town 分散抽样 N 条 route；None 表示不启用 per-town 抽样。"""
        if samples_per_town is None:
            return route_dirs
        samples_per_town = max(1, int(samples_per_town))
        by_town: Dict[str, List[Path]] = defaultdict(list)
        for route_dir in route_dirs:
            town = _extract_town(route_dir.name) or "unknown"
            by_town[town].append(route_dir)
        selected: List[Path] = []
        for town in sorted(by_town):
            selected.extend(self._select_route_dirs(sorted(by_town[town]), samples_per_town))
        return selected

    def _route_data_quality_skip(self, scenario_name: str, route_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[RouteXmlInfo]]:
        """缺 meta/XML 的 route 不进入规则标定，只作为数据质量 skip 记录。"""
        reasons: List[str] = []
        metas_dir = route_path / "metas"
        meta_files: List[Path] = []
        if not metas_dir.exists():
            reasons.append("missing_metas_dir")
        else:
            meta_files = sorted(metas_dir.glob("*.pkl"))
            if not meta_files:
                reasons.append("missing_meta_pkl")

        xml_info = self.xml_index.match(scenario_name, route_path.name)
        if xml_info is None:
            reasons.append("missing_route_xml")

        if not reasons:
            return None, xml_info

        skip = {
            "route_id": route_path.name,
            "status": "data_missing_skip",
            "skip_reason": ";".join(reasons),
            "skip_reasons": reasons,
            "num_frames": 0,
            "data_quality": {
                "usable_for_annotation": False,
                "missing_meta": "missing_metas_dir" in reasons or "missing_meta_pkl" in reasons,
                "missing_xml": "missing_route_xml" in reasons,
                "metas_dir": str(metas_dir),
                "meta_file_count": len(meta_files),
                "xml_available": xml_info is not None,
                "xml_path": str(xml_info.path) if xml_info else None,
            },
            "xml_path": str(xml_info.path) if xml_info else None,
            "xml_town": xml_info.town if xml_info else None,
            "xml_available": xml_info is not None,
            "annotations": [],
        }
        return skip, xml_info

    def _collect_scenario(
        self,
        scenario_name: str,
        max_routes: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
        samples_per_town: Optional[int] = None,
    ) -> Dict:
        """采集单个场景的内部实现"""
        self.logger.info(f"采集场景: {scenario_name}")

        scenario_dir = self.lead_data_root / scenario_name
        if not scenario_dir.exists():
            return {
                "scenario": scenario_name,
                "status": "error",
                "error": f"场景目录不存在: {scenario_dir}",
                "lead_data_root": str(self.lead_data_root),
                "routes": [],
                "total_frames": 0,
            }

        # 获取该场景的所有 routes，并先剔除异常时长采集。
        discovered_route_dirs = sorted([d for d in scenario_dir.iterdir() if d.is_dir()])
        all_route_dirs = []
        abnormal_skipped = []
        data_missing_skipped = []
        for route_dir in discovered_route_dirs:
            should_exclude, info = is_abnormal_lead_route(route_dir, scenario_name)
            if should_exclude:
                abnormal_skipped.append(info)
                continue
            data_skip, _ = self._route_data_quality_skip(scenario_name, route_dir)
            if data_skip is not None:
                data_missing_skipped.append(data_skip)
                continue
            all_route_dirs.append(route_dir)

        # 根据 per-town / max_routes 参数决定采集多少
        if samples_per_town is not None:
            route_dirs = self._select_route_dirs_per_town(all_route_dirs, samples_per_town)
        elif max_routes is None:
            # None 表示采集所有
            route_dirs = all_route_dirs
        else:
            # 采集前 max_routes 个
            route_dirs = self._select_route_dirs(all_route_dirs, max_routes)

        self.logger.info(
            f"  发现 {len(discovered_route_dirs)} 个routes, "
            f"异常时长剔除 {len(abnormal_skipped)} 个, "
            f"数据缺失剔除 {len(data_missing_skipped)} 个, 将采集 {len(route_dirs)} 个"
            + (f" (每 town {samples_per_town} 条)" if samples_per_town is not None else "")
        )

        routes = []
        for i, route_dir in enumerate(route_dirs, 1):
            self.logger.info(f"    [{i}/{len(route_dirs)}] 处理 {route_dir.name}")
            route_result = self._process_route(scenario_name, route_dir, max_frames_per_route=max_frames_per_route)
            routes.append(route_result)

        result = {
            "scenario": scenario_name,
            "status": "success",
            "road_candidates": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [])],
            "event_candidates": [ev.value for ev in SCENARIO_TO_FINE_EVENTS.get(scenario_name, [])],
            "abnormal_duration_rule": "exclude duration_s > 90 unless scenario is BlockedIntersection or ControlLoss",
            "abnormal_duration_skipped": abnormal_skipped,
            "data_missing_skip_rule": "skip routes whose metas/*.pkl or matched route XML is missing; these are data-quality failures, not rule failures",
            "data_missing_skipped": data_missing_skipped,
            "data_missing_skip_count": len(data_missing_skipped),
            "data_quality_summary": {
                "discovered_routes": len(discovered_route_dirs),
                "abnormal_duration_skipped": len(abnormal_skipped),
                "data_missing_skipped": len(data_missing_skipped),
                "eligible_routes": len(all_route_dirs),
                "selected_routes": len(route_dirs),
            },
            "samples_per_town": samples_per_town,
            "routes": routes,
            "total_frames": sum(r.get('num_frames', 0) for r in routes)
        }

        # 保存结果
        output_file = self.output_dir / f"{scenario_name}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"  ✓ {scenario_name} 采集完成: {result['total_frames']} 帧")

        return result

    def _frame_rs_annotation_payload(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        """生成显式逐帧 RS 标注块，和候选全集字段分开。"""
        evidence = ann.get("evidence", {})
        diagnostic = evidence.get("diagnostic_attribution", {})
        return {
            "label": ann.get("primary_road_structure"),
            "secondary": ann.get("secondary_road_structures", []),
            "overlay": ann.get("road_structure_overlay"),
            "confidence": ann.get("confidence"),
            "comment": ann.get("annotation_comment", ""),
            "rule_kind": evidence.get("rule_kind"),
            "rules_fired": evidence.get("rules_fired", []),
            "decision_source": diagnostic.get("decision_source"),
            "review_required": bool(evidence.get("review_required")),
            "review_reasons": evidence.get("review_reasons", []),
            "metrics": {
                "route_progress_m": evidence.get("route_progress_m"),
                "route_projection_error_m": evidence.get("route_projection_error_m"),
                "trigger_distance_m": evidence.get("trigger_distance_m"),
                "actor_flow_distance_m": evidence.get("actor_flow_distance_m"),
                "traffic_light_state": evidence.get("traffic_light_state"),
                "active_scenario": evidence.get("current_active_scenario_type"),
            },
            "xodr_summary": {
                "available": evidence.get("xodr", {}).get("xodr_available"),
                "source": evidence.get("xodr", {}).get("xodr_source"),
                "trusted": evidence.get("xodr", {}).get("xodr_topology_trusted"),
                "road_id": evidence.get("xodr", {}).get("map_road_id"),
                "lane_id": evidence.get("xodr", {}).get("map_lane_id"),
                "is_junction": evidence.get("xodr", {}).get("map_is_junction"),
                "is_roundabout": evidence.get("xodr", {}).get("map_is_roundabout"),
                "opposite_lane": evidence.get("xodr", {}).get("has_opposite_driving_lane"),
                "parking_or_shoulder": evidence.get("xodr", {}).get("has_parking_or_shoulder_nearby"),
                "merge_split_hint": evidence.get("xodr", {}).get("ramp_merge_split_hint"),
            },
        }

    def _attach_overlay_base_rs(
        self,
        ann: Dict[str, Any],
        base_rs: Any,
        reason: str = "interrupted_event_overlay_base_rs",
    ) -> None:
        """在路口 overlay 帧保留被截断突发事件所属的基础 RS。"""
        base_label = str(base_rs or "")
        if base_label not in RoadStructure._value2member_map_:
            return
        primary_label = str(ann.get("primary_road_structure") or "")
        if not primary_label or base_label == primary_label:
            return
        secondary = set(str(x) for x in (ann.get("secondary_road_structures") or []) if x)
        if base_label not in secondary:
            secondary.add(base_label)
            ann["secondary_road_structures"] = sorted(secondary)
        overlay = ann.get("road_structure_overlay")
        if not isinstance(overlay, dict):
            overlay = {}
            ann["road_structure_overlay"] = overlay
        overlay["active"] = True
        overlay["source"] = "interrupted_event_overlay"
        overlay["base_road_structure"] = base_label
        overlay["intersection_road_structure"] = primary_label
        overlay["secondary_road_structure"] = base_label
        reasons = overlay.setdefault("reasons", [])
        if reason not in reasons:
            reasons.append(reason)
        evidence = ann.setdefault("evidence", {})
        overlay_secondary = evidence.setdefault("overlay_secondary_road_structures", [])
        marker = {
            "base_road_structure": base_label,
            "primary_road_structure": primary_label,
            "reason": reason,
        }
        if marker not in overlay_secondary:
            overlay_secondary.append(marker)
        rules = evidence.setdefault("rules_fired", [])
        rule = f"rs_{reason}"
        if rule not in rules:
            rules.append(rule)
        ann["annotation_comment"] = _frame_annotation_comment(
            RoadStructure(primary_label),
            {
                RoadStructure(x)
                for x in ann.get("secondary_road_structures", [])
                if x in RoadStructure._value2member_map_
            },
            float(ann.get("confidence", 0.0) or 0.0),
            evidence,
        )
        ann["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann)

    def _frame_event_annotation_payload(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        """生成显式逐帧 EVENT 标注块，和候选全集字段分开。"""
        evidence = ann.get("event_evidence", {}) or {}
        return {
            "label": ann.get("primary_event"),
            "events": ann.get("events", []),
            "regular_event": evidence.get("regular_event"),
            "unusual_event": evidence.get("unusual_event"),
            "allowed_events": evidence.get("allowed_events", []),
            "rules_fired": evidence.get("rules_fired", []),
            "metrics": evidence.get("metrics", {}),
            "interrupted_event_overlay": evidence.get("interrupted_event_overlay"),
            "review_required": bool(evidence.get("review_required")),
            "review_reasons": evidence.get("review_reasons", []),
            "comment": self._event_comment(ann),
        }

    @staticmethod
    def _event_comment(ann: Dict[str, Any]) -> str:
        label = ann.get("primary_event") or "unknown"
        events = ann.get("events", [])
        evidence = ann.get("event_evidence", {}) or {}
        rules = evidence.get("rules_fired", [])
        overlay = evidence.get("interrupted_event_overlay") or {}
        if overlay.get("active"):
            return (
                f"EVENT {label}: interrupted-overlay "
                f"{overlay.get('base_road_structure')}+{overlay.get('intersection_road_structure')} "
                f"events={events}; phase={overlay.get('phase')}; rules={','.join(rules[:3])}"
            )
        if events and len(events) > 1:
            return f"EVENT {label}: 路口双触发 {events}; rules={','.join(rules[:3])}"
        return f"EVENT {label}: rules={','.join(rules[:3])}"

    @staticmethod
    def _rewrite_event_label(
        ann: Dict[str, Any],
        events: Set[EventType],
        primary_event: EventType,
        reason: str,
    ) -> None:
        ann["events"] = [ev.value for ev in sorted(events, key=lambda ev: ev.value)]
        ann["primary_event"] = primary_event.value
        evidence = ann.setdefault("event_evidence", {})
        evidence["events"] = ann["events"]
        evidence["primary_event"] = primary_event.value
        if primary_event.value.startswith("R-E"):
            evidence["regular_event"] = primary_event.value
            evidence["unusual_event"] = None
        else:
            evidence["unusual_event"] = primary_event.value
        evidence.setdefault("rules_fired", []).append(reason)

    def _apply_event_route_postprocess(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Route 级 EVENT 修正：障碍核心后短恢复段按 R-E2，短孤立 U-E 去抖。"""
        if not annotations:
            return {"enabled": True, "changes": []}
        changes: List[Dict[str, Any]] = []

        obstacle_field_cfg = OBSTACLE_EVENT_DISTANCE_FIELDS.get(scenario_name)

        def _event_metrics(ann: Dict[str, Any]) -> Dict[str, Any]:
            return ((ann.get("event_evidence") or {}).get("metrics") or {})

        def _trigger_distance(ann: Dict[str, Any]) -> float:
            return _safe_float((ann.get("evidence") or {}).get("trigger_distance_m"), default=math.inf)

        def _signed_lane_change(ann: Dict[str, Any]) -> float:
            return _safe_float(_event_metrics(ann).get("signed_dist_to_lane_change"), default=math.inf)

        def _signed_lane_change_active(ann: Dict[str, Any], *, limit_m: float = 3.5) -> bool:
            signed = _signed_lane_change(ann)
            return math.isfinite(signed) and abs(signed) <= limit_m

        def _route_lateral_abs(ann: Dict[str, Any]) -> float:
            return _safe_float(_event_metrics(ann).get("route_lateral_abs_m"), default=math.inf)

        def _route_center_tolerance(ann: Dict[str, Any]) -> float:
            tol = _safe_float(_event_metrics(ann).get("route_center_tolerance_m"), default=math.inf)
            return tol if math.isfinite(tol) and tol > 0.0 else 0.55

        def _route_centered(ann: Dict[str, Any]) -> bool:
            metrics = _event_metrics(ann)
            if "route_centered" in metrics:
                return bool(metrics.get("route_centered"))
            route_abs = _route_lateral_abs(ann)
            return math.isfinite(route_abs) and route_abs <= _route_center_tolerance(ann)

        def _route_centered_for_re2_exit(ann: Dict[str, Any]) -> bool:
            """R-E2 退出稍早于严格中心线完成，避免恢复段尾部粘滞。"""
            route_abs = _route_lateral_abs(ann)
            if math.isfinite(route_abs):
                return route_abs <= _route_center_tolerance(ann) * RE2_EXIT_CENTER_TOLERANCE_SCALE
            return _route_centered(ann)

        def _route_change_hint(ann: Dict[str, Any], *, allow_abs: bool = True) -> bool:
            metrics = _event_metrics(ann)
            signed = _signed_lane_change(ann)
            if bool(metrics.get("target_lane_change_active")):
                return True
            # In curved Town06-style obstacle routes the ego-frame route centerline can
            # remain close to the vehicle while the lane-change signal is continuous.
            # Treat signed lane-change + changed_route as a recovery hint, but only in
            # route-level postprocess where the surrounding U-E2 context is known.
            if bool(metrics.get("changed_route")) and math.isfinite(signed) and abs(signed) <= 3.5:
                return True
            if bool(metrics.get("changed_route")) and not _route_centered(ann):
                return True
            return allow_abs and math.isfinite(signed) and abs(signed) <= 4.5 and not _route_centered(ann)

        def _lane_change_re2_supported(index: int, *, allow_centering: bool = True) -> bool:
            """R-E2 必须来自真实横向换道/回正证据，不能只靠标签顺序。"""
            ann = annotations[index]
            rs = ann.get("primary_road_structure")
            if rs in {RoadStructure.R4.value, RoadStructure.R5.value}:
                metrics = _event_metrics(ann)
                if not (scenario_name == "AccidentTwoWays" and bool(metrics.get("accident_twoways_r2_overlay_active"))):
                    return False
            if _return_lane_change_hint(ann):
                return True
            if _route_change_hint(ann) and (not _route_centered(ann) or _signed_lane_change_active(ann)):
                return True
            if allow_centering and _route_centering_trend(index):
                prev_window = range(max(0, index - 5), index + 1)
                return any(_route_change_hint(annotations[j]) or _signed_lane_change_active(annotations[j]) for j in prev_window)
            return False

        def _avoidance_lane_u2_supported(index: int) -> bool:
            """U-E2 避障阶段：仍偏离 route 中心且尚未开始明显回正。"""
            ann = annotations[index]
            rs = ann.get("primary_road_structure")
            if rs in {RoadStructure.R4.value, RoadStructure.R5.value}:
                return False
            if not (_route_change_hint(ann) or _signed_lane_change_active(ann)):
                return False
            if _route_centered(ann) or _return_lane_change_hint(ann) or _route_centering_trend(index):
                return False
            return True

        def _return_lane_change_hint(ann: Dict[str, Any]) -> bool:
            signed = _signed_lane_change(ann)
            return math.isfinite(signed) and signed <= -0.45

        def _route_centering_trend(index: int) -> bool:
            if index <= 0 or index >= len(annotations):
                return False
            cur = _route_lateral_abs(annotations[index])
            if not math.isfinite(cur):
                return False
            tol = _route_center_tolerance(annotations[index])
            if cur <= tol:
                return False
            prev_values = [
                _route_lateral_abs(annotations[j])
                for j in range(max(0, index - 2), index)
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            next_values = [
                _route_lateral_abs(annotations[j])
                for j in range(index + 1, min(len(annotations), index + 3))
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            prev_high = max(prev_values) if prev_values else math.inf
            next_low = min(next_values) if next_values else math.inf
            return (
                (math.isfinite(prev_high) and prev_high - cur >= 0.15)
                or (math.isfinite(next_low) and cur - next_low >= 0.15)
            )

        def _lateral_recovery_started(
            index: int,
            span_start: int,
            span_end: int,
            *,
            require_after_peak: bool = True,
        ) -> bool:
            """自车已从绕障侧向峰值开始回目标/原车道。"""
            if index < span_start or index >= span_end:
                return False
            cur = _route_lateral_abs(annotations[index])
            if not math.isfinite(cur):
                return False
            prior_values = [
                (j, _route_lateral_abs(annotations[j]))
                for j in range(span_start, index + 1)
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            if len(prior_values) < 3:
                return False
            peak_idx, peak_val = max(prior_values, key=lambda item: item[1])
            if require_after_peak and peak_idx >= index:
                return False
            future_values = [
                _route_lateral_abs(annotations[j])
                for j in range(index + 1, min(span_end, index + 4))
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            recent_values = [
                _route_lateral_abs(annotations[j])
                for j in range(max(span_start, index - 3), index)
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            future_drop = bool(future_values) and cur - min(future_values) >= RECOVERY_LATERAL_DROP_START_M
            past_drop = bool(recent_values) and max(recent_values) - cur >= RECOVERY_LATERAL_DROP_START_M
            peak_drop = peak_val - cur >= RECOVERY_LATERAL_DROP_STRONG_M
            signed_return = _return_lane_change_hint(annotations[index])
            signed_near_zero_after_peak = (
                peak_drop
                and _signed_lane_change_active(annotations[index], limit_m=4.5)
                and _signed_lane_change(annotations[index]) <= 0.25
            )
            return signed_return or future_drop or past_drop or signed_near_zero_after_peak

        def _route_recovery_start(
            start: int,
            end: int,
            closest_idx: Optional[int],
        ) -> Optional[int]:
            """在 U-E2 span 内找回原/目标车道的开始，而不是等回正后补 R-E2。"""
            finite_points = [
                (j, _route_lateral_abs(annotations[j]))
                for j in range(start, end)
                if math.isfinite(_route_lateral_abs(annotations[j]))
            ]
            if len(finite_points) < 4:
                return None

            def _signed_recovery_start() -> Optional[int]:
                signed_points = [
                    (j, abs(_signed_lane_change(annotations[j])))
                    for j in range(start, end)
                    if _signed_lane_change_active(annotations[j], limit_m=4.5)
                    and bool(_event_metrics(annotations[j]).get("changed_route"))
                ]
                if len(signed_points) < 4:
                    return None
                first_high = max((val for j, val in signed_points if j <= start + max(6, (end - start) // 3)), default=0.0)
                if first_high < 0.45:
                    return None
                for j, signed_abs in signed_points:
                    if j < start + 6:
                        continue
                    if closest_idx is not None and j < max(start + 6, closest_idx - 10):
                        continue
                    if signed_abs > 0.15 and _signed_lane_change(annotations[j]) > -0.25:
                        continue
                    future = range(j, min(end, j + 5))
                    stable_low = sum(
                        1
                        for k in future
                        if _signed_lane_change_active(annotations[k], limit_m=4.5)
                        and abs(_signed_lane_change(annotations[k])) <= 0.25
                    )
                    if stable_low < 3 and _signed_lane_change(annotations[j]) > -0.25:
                        continue
                    prior_peak = max(
                        (
                            abs(_signed_lane_change(annotations[k]))
                            for k in range(start, j)
                            if _signed_lane_change_active(annotations[k], limit_m=4.5)
                        ),
                        default=0.0,
                    )
                    if prior_peak - signed_abs < 0.30 and _signed_lane_change(annotations[j]) > -0.25:
                        continue
                    cur = _route_lateral_abs(annotations[j])
                    tol = _route_center_tolerance(annotations[j])
                    lateral_reasonable = (
                        not math.isfinite(cur)
                        or cur <= max(tol + 0.25, 0.80)
                        or _route_centering_trend(j)
                    )
                    if not lateral_reasonable:
                        continue
                    return max(start, j - 2)
                return None

            signed_start = _signed_recovery_start()
            if closest_idx is None:
                if signed_start is not None:
                    return signed_start
                closest_idx = max(finite_points, key=lambda item: item[1])[0]
            # Positive signed-distance decay before the closest obstacle point is still
            # the avoidance lane-change, not the return/recovery lane-change.
            elif signed_start is not None and signed_start >= max(start + 3, closest_idx - 2):
                return signed_start
            lateral_peak_idx, _lateral_peak = max(finite_points, key=lambda item: item[1])
            closest_dist = _specific_obstacle_distance(annotations[closest_idx])
            if math.isfinite(closest_dist):
                for j in range(max(start + 3, closest_idx + 1), min(end, closest_idx + 5)):
                    dist = _specific_obstacle_distance(annotations[j])
                    signed = _signed_lane_change(annotations[j])
                    if (
                        math.isfinite(dist)
                        and dist > closest_dist + 0.05
                        and math.isfinite(signed)
                        and abs(signed) <= 1.20
                        and _route_centered_for_re2_exit(annotations[j])
                    ):
                        return j
            search_start = max(start + 2, min(closest_idx - 1, lateral_peak_idx + 1))
            peak_so_far = -math.inf
            peak_idx = search_start
            for j in range(start, end):
                cur = _route_lateral_abs(annotations[j])
                if not math.isfinite(cur):
                    continue
                if cur > peak_so_far:
                    peak_so_far = cur
                    peak_idx = j
                if j < search_start:
                    continue
                tol = _route_center_tolerance(annotations[j])
                signed_return = _return_lane_change_hint(annotations[j])
                if cur <= tol and not signed_return:
                    continue
                prev_vals = [
                    _route_lateral_abs(annotations[k])
                    for k in range(max(start, j - 3), j)
                    if math.isfinite(_route_lateral_abs(annotations[k]))
                ]
                next_vals = [
                    _route_lateral_abs(annotations[k])
                    for k in range(j + 1, min(end, j + 4))
                    if math.isfinite(_route_lateral_abs(annotations[k]))
                ]
                prev_peak = max(prev_vals) if prev_vals else peak_so_far
                next_min = min(next_vals) if next_vals else math.inf
                has_left_peak = (
                    math.isfinite(prev_peak)
                    and prev_peak >= max(tol + 0.35, cur + 0.12)
                    and peak_idx <= j
                )
                has_forward_drop = math.isfinite(next_min) and cur - next_min >= 0.12
                lateral_recovery = _lateral_recovery_started(j, start, end)
                trend = _route_centering_trend(j) or has_left_peak or has_forward_drop or signed_return or lateral_recovery
                if not trend:
                    continue
                passed_core = j >= max(start + 3, closest_idx - 1)
                no_longer_specific_core = not _specific_obstacle_close(annotations[j], pad_m=-3.0)
                passed_lateral_peak = j >= max(start + 3, lateral_peak_idx + 1)
                if not (
                    passed_core
                    or no_longer_specific_core
                    or (passed_lateral_peak and lateral_recovery and (has_forward_drop or signed_return or _route_centering_trend(j)))
                ):
                    continue
                # R-E2 起点应略早于可观测下降，表达“准备回原/目标车道”。
                early = 1 if lateral_recovery else (2 if has_forward_drop or signed_return else 0)
                return max(start, min(j, max(search_start - 1, j - early)))
            return None

        def _obstacle_still_close(ann: Dict[str, Any], *, pad_m: float = 0.0) -> bool:
            if obstacle_field_cfg is None:
                return False
            field, threshold = obstacle_field_cfg
            metrics = _event_metrics(ann)
            dist = _safe_float(metrics.get(field), default=math.inf)
            if dist <= threshold + pad_m:
                return True
            speed_obj = _safe_float(metrics.get("speed_reduced_by_obj_distance"), default=math.inf)
            return speed_obj <= threshold + pad_m and _trigger_distance(ann) <= 55.0

        def _specific_obstacle_close(ann: Dict[str, Any], *, pad_m: float = 0.0) -> bool:
            if obstacle_field_cfg is None:
                return False
            field, threshold = obstacle_field_cfg
            dist = _safe_float(_event_metrics(ann).get(field), default=math.inf)
            return dist <= threshold + pad_m

        def _specific_obstacle_distance(ann: Dict[str, Any]) -> float:
            if obstacle_field_cfg is None:
                return math.inf
            field, _threshold = obstacle_field_cfg
            return _safe_float(_event_metrics(ann).get(field), default=math.inf)

        def _specific_obstacle_core_or_approaching(index: int) -> bool:
            """Keep U-E2 while ego is still beside or approaching the static obstacle."""
            cur = _specific_obstacle_distance(annotations[index])
            if not math.isfinite(cur):
                return False
            if cur <= 6.0:
                return True
            if cur > 12.0:
                return False
            prev_vals = [
                _specific_obstacle_distance(annotations[j])
                for j in range(max(0, index - 3), index)
                if math.isfinite(_specific_obstacle_distance(annotations[j]))
            ]
            if not prev_vals:
                return cur <= 8.0
            return cur <= min(prev_vals) + 0.8

        def _cutin_distance(ann: Dict[str, Any]) -> float:
            return _safe_float(_event_metrics(ann).get("dist_to_cutin_vehicle"), default=math.inf)

        def _cutin_response_active(ann: Dict[str, Any]) -> bool:
            metrics = _event_metrics(ann)
            return bool(metrics.get("brake_cutin")) or bool(metrics.get("vehicle_hazard"))

        def _regular_event_for_annotation(ann: Dict[str, Any]) -> EventType:
            rs = ann.get("primary_road_structure")
            if rs == RoadStructure.R4.value:
                return EventType.R_E4
            if rs == RoadStructure.R5.value:
                return EventType.R_E5
            if rs == RoadStructure.R3.value:
                event_evidence = ann.get("event_evidence") or {}
                regular = event_evidence.get("regular_event")
                if regular in {EventType.R_E1.value, EventType.R_E2.value, EventType.R_E3.value}:
                    return EventType(regular)
                metrics = event_evidence.get("metrics") or {}
                if bool(metrics.get("highway_r3_core_active")):
                    return EventType.R_E3
                if bool(metrics.get("target_lane_change_active")) or bool(metrics.get("highway_lane_change_regular")):
                    return EventType.R_E2
                return EventType.R_E1
            return EventType.R_E1

        def _intersection_or_signal_control(ann: Dict[str, Any]) -> bool:
            rs = ann.get("primary_road_structure")
            if rs in {RoadStructure.R4.value, RoadStructure.R5.value}:
                return True
            evidence = ann.get("evidence") or {}
            tl_state = evidence.get("traffic_light_state")
            return _valid_traffic_light(tl_state) or bool(evidence.get("light_hazard"))

        def _release_to_regular(ann: Dict[str, Any], reason: str) -> Optional[EventType]:
            old = ann.get("primary_event")
            replacement = _regular_event_for_annotation(ann)
            evidence = ann.get("evidence") or {}
            if replacement == EventType.R_E1 and _valid_traffic_light(evidence.get("traffic_light_state")):
                replacement = EventType.R_E4
            if old == replacement.value:
                return None
            self._rewrite_event_label(ann, {replacement}, replacement, reason)
            return replacement

        def _force_highway_r3(index: int, reason: str) -> bool:
            ann = annotations[index]
            if ann.get("primary_road_structure") != RoadStructure.R3.value:
                return False
            if ann.get("primary_event") != EventType.R_E1.value:
                return False
            self._rewrite_event_label(ann, {EventType.R_E3}, EventType.R_E3, reason)
            metrics = _event_metrics(ann)
            metrics["highway_r3_core_active"] = True
            metrics["highway_route_postprocess_r3"] = True
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
            return True

        if scenario_name == "noScenarios":
            def _noscenario_re2_support(index: int) -> bool:
                ann = annotations[index]
                if ann.get("primary_road_structure") not in {RoadStructure.R1.value, RoadStructure.R3.value}:
                    return False
                if ann.get("primary_event") not in {EventType.R_E1.value, EventType.R_E2.value, EventType.R_E3.value}:
                    return False
                metrics = _event_metrics(ann)
                signed = _signed_lane_change(ann)
                route_abs = _route_lateral_abs(ann)
                changed_route = bool(metrics.get("changed_route"))
                return (
                    bool(metrics.get("target_lane_change_active"))
                    or (changed_route and math.isfinite(signed) and abs(signed) <= 4.0)
                    or (
                        changed_route
                        and math.isfinite(route_abs)
                        and route_abs > max(_route_center_tolerance(ann), 0.18)
                    )
                    or (
                        math.isfinite(signed)
                        and abs(signed) <= 2.5
                        and math.isfinite(route_abs)
                        and route_abs > max(_route_center_tolerance(ann), 0.18)
                    )
                )

            def _force_noscenario_re2(index: int, reason: str, *, require_support: bool = True) -> bool:
                ann = annotations[index]
                if require_support and not _noscenario_re2_support(index):
                    return False
                if ann.get("primary_road_structure") not in {RoadStructure.R1.value, RoadStructure.R3.value}:
                    return False
                if ann.get("primary_event") == EventType.R_E2.value:
                    return False
                old = ann.get("primary_event")
                self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, reason)
                metrics = _event_metrics(ann)
                metrics["noscenario_re2_span_smoothed"] = True
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E2.value,
                        "reason": reason,
                    }
                )
                return True

            re2_spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                re2_spans.append((start, idx))

            for start, end in re2_spans:
                new_start = start
                while new_start > 0 and start - new_start <= 2 and _noscenario_re2_support(new_start - 1):
                    new_start -= 1
                new_end = end
                while new_end < len(annotations) and new_end - end <= 4 and _noscenario_re2_support(new_end):
                    new_end += 1
                for j in range(new_start, new_end):
                    _force_noscenario_re2(j, "event_noscenario_lane_change_span_expanded_by_trajectory")

            # Fill tiny regular gaps between two supported RE2 chunks; this keeps a single
            # lane-change maneuver from being split by one noisy centered frame.
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                    continue
                gap_start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                gap_end = idx
                if gap_start == 0 or gap_end >= len(annotations) or gap_end - gap_start > 3:
                    continue
                if (
                    annotations[gap_start - 1].get("primary_event") == EventType.R_E2.value
                    and annotations[gap_end].get("primary_event") == EventType.R_E2.value
                    and (
                        gap_end - gap_start == 1
                        or all(_noscenario_re2_support(j) for j in range(gap_start, gap_end))
                    )
                    and all(
                        annotations[j].get("primary_road_structure") in {RoadStructure.R1.value, RoadStructure.R3.value}
                        for j in range(gap_start, gap_end)
                    )
                ):
                    for j in range(gap_start, gap_end):
                        _force_noscenario_re2(
                            j,
                            "event_noscenario_short_re2_gap_merged",
                            require_support=gap_end - gap_start != 1,
                        )

        highway_enter_merge_scenarios = {
            "EnterActorFlow",
            "EnterActorFlowV2",
            "MergerIntoSlowTraffic",
            "MergerIntoSlowTrafficV2",
        }
        if scenario_name in highway_enter_merge_scenarios:
            max_backfill = 36 if scenario_name.startswith("EnterActorFlow") else 32
            min_enter_r3_frames = 16 if scenario_name.startswith("EnterActorFlow") else 0
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                back = start - 1
                while (
                    back >= 0
                    and start - back <= max_backfill
                    and annotations[back].get("primary_road_structure") == RoadStructure.R3.value
                    and annotations[back].get("primary_event") in {EventType.R_E1.value, EventType.R_E3.value}
                ):
                    back -= 1
                if min_enter_r3_frames > 0:
                    desired_start = max(0, start - min_enter_r3_frames)
                    back = min(back, desired_start - 1)
                for j in range(back + 1, start):
                    old = annotations[j].get("primary_event")
                    if _force_highway_r3(j, "event_highway_merge_approach_backfilled_to_r3"):
                        changes.append(
                            {
                                "frame_id": annotations[j].get("frame_id"),
                                "from": old,
                                "to": EventType.R_E3.value,
                                "reason": "highway_merge_approach_backfilled_to_r3",
                            }
                        )

        if scenario_name in {"EnterActorFlow", "EnterActorFlowV2"}:
            enter_actor_re2_actor_guard_m = 35.0

            def _enter_actor_re2_start_support(index: int) -> bool:
                ann = annotations[index]
                metrics = _event_metrics(ann)
                actor_flow = _safe_float(metrics.get("highway_actor_flow_distance_m"), default=math.inf)
                signed = _signed_lane_change(ann)
                return (
                    ann.get("primary_road_structure") in {RoadStructure.R1.value, RoadStructure.R3.value}
                    and ann.get("primary_event") in {EventType.R_E1.value, EventType.R_E3.value, EventType.R_E2.value}
                    and bool(metrics.get("changed_route"))
                    and math.isfinite(actor_flow)
                    and actor_flow <= enter_actor_re2_actor_guard_m
                    and math.isfinite(signed)
                    and abs(signed) <= 2.6
                )

            def _enter_actor_re2_tail_support(index: int) -> bool:
                ann = annotations[index]
                metrics = _event_metrics(ann)
                actor_flow = _safe_float(metrics.get("highway_actor_flow_distance_m"), default=math.inf)
                signed = _signed_lane_change(ann)
                if (
                    ann.get("primary_road_structure") not in {RoadStructure.R1.value, RoadStructure.R3.value}
                    or ann.get("primary_event") not in {EventType.R_E1.value, EventType.R_E3.value, EventType.R_E2.value}
                    or not bool(metrics.get("changed_route"))
                    or not math.isfinite(actor_flow)
                    or actor_flow > enter_actor_re2_actor_guard_m
                    or not math.isfinite(signed)
                ):
                    return False
                route_abs = _route_lateral_abs(ann)
                return (
                    abs(signed) <= 2.2
                    or (math.isfinite(route_abs) and route_abs >= 0.06)
                    or not _route_centered_for_re2_exit(ann)
                )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                new_start = start
                while new_start > 0 and start - new_start <= 8 and _enter_actor_re2_start_support(new_start - 1):
                    new_start -= 1
                new_end = end
                while new_end < len(annotations) and new_end - end <= 8 and _enter_actor_re2_tail_support(new_end):
                    new_end += 1
                if new_start == start and new_end == end:
                    continue
                for j in range(new_start, new_end):
                    if annotations[j].get("primary_event") == EventType.R_E2.value:
                        continue
                    old = annotations[j].get("primary_event")
                    self._rewrite_event_label(
                        annotations[j],
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_enter_actor_flow_lane_change_span_expanded_by_trajectory",
                    )
                    metrics = _event_metrics(annotations[j])
                    metrics["enter_actor_flow_re2_span_expanded"] = True
                    annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                    changes.append(
                        {
                            "frame_id": annotations[j].get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "enter_actor_flow_lane_change_span_expanded_by_trajectory",
                        }
                    )

        if scenario_name in {"HighwayCutIn", "HighwayExit", "MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
            if scenario_name == "HighwayExit":
                backward_pad = 2
                forward_pad = 6
            elif scenario_name == "HighwayCutIn":
                backward_pad = 3
                forward_pad = 4
            elif scenario_name in {"MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
                backward_pad = 5
                forward_pad = 5
            else:
                backward_pad = 3
                forward_pad = 3

            def _highway_re2_boundary_support(index: int) -> bool:
                ann = annotations[index]
                if (
                    ann.get("primary_road_structure") != RoadStructure.R3.value
                    or ann.get("primary_event")
                    not in {EventType.R_E1.value, EventType.R_E2.value, EventType.R_E3.value}
                ):
                    return False
                metrics = _event_metrics(ann)
                signed = _signed_lane_change(ann)
                route_abs = _route_lateral_abs(ann)
                return bool(metrics.get("changed_route")) and (
                    (math.isfinite(signed) and abs(signed) <= 3.5)
                    or (math.isfinite(route_abs) and route_abs >= 0.05)
                    or not _route_centered_for_re2_exit(ann)
                )

            original_re2_spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                original_re2_spans.append((start, idx))

            for start, end in original_re2_spans:
                new_start = start
                while (
                    new_start > 0
                    and start - new_start < backward_pad
                    and _highway_re2_boundary_support(new_start - 1)
                ):
                    new_start -= 1
                new_end = end
                while (
                    new_end < len(annotations)
                    and new_end - end < forward_pad
                    and _highway_re2_boundary_support(new_end)
                ):
                    new_end += 1
                for j in range(new_start, new_end):
                    if annotations[j].get("primary_event") == EventType.R_E2.value:
                        continue
                    old = annotations[j].get("primary_event")
                    self._rewrite_event_label(
                        annotations[j],
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_highway_lane_change_span_expanded_by_trajectory",
                    )
                    metrics = _event_metrics(annotations[j])
                    metrics["highway_re2_span_expanded"] = True
                    annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                    changes.append(
                        {
                            "frame_id": annotations[j].get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "highway_lane_change_span_expanded_by_trajectory",
                        }
                    )

            if scenario_name in {"MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
                max_tail_frames = 64
                tail_bridge_gap = 8

                def _merger_tail_r3_support(index: int) -> bool:
                    ann = annotations[index]
                    if (
                        ann.get("primary_road_structure") != RoadStructure.R3.value
                        or ann.get("primary_event") not in {EventType.R_E1.value, EventType.R_E3.value}
                    ):
                        return False
                    metrics = _event_metrics(ann)
                    actor_flow = _safe_float(metrics.get("highway_actor_flow_distance_m"), default=math.inf)
                    trigger = _safe_float(metrics.get("highway_trigger_distance_m"), default=math.inf)
                    route_abs = _route_lateral_abs(ann)
                    changed_route = bool(metrics.get("changed_route"))
                    close_actor_flow = math.isfinite(actor_flow) and actor_flow <= 35.0
                    close_trigger_merge = math.isfinite(trigger) and trigger <= 45.0 and (
                        changed_route or bool(metrics.get("highway_ramp_merge_split_hint"))
                    )
                    near_route_transition = (
                        changed_route
                        and math.isfinite(actor_flow)
                        and actor_flow <= 45.0
                        and (
                            (math.isfinite(route_abs) and route_abs >= 0.04)
                            or not _route_centered_for_re2_exit(ann)
                        )
                    )
                    return bool(close_actor_flow or close_trigger_merge or near_route_transition)

                current_re2_spans: List[Tuple[int, int]] = []
                idx = 0
                while idx < len(annotations):
                    if annotations[idx].get("primary_event") != EventType.R_E2.value:
                        idx += 1
                        continue
                    start = idx
                    while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                        idx += 1
                    current_re2_spans.append((start, idx))

                for _start, end in current_re2_spans:
                    segment_start = None
                    last_support = None
                    misses = 0
                    for j in range(end, min(len(annotations), end + max_tail_frames)):
                        if annotations[j].get("primary_event") == EventType.R_E2.value:
                            misses = 0
                            continue
                        if _merger_tail_r3_support(j):
                            if segment_start is None:
                                segment_start = j
                            last_support = j
                            misses = 0
                            continue
                        if segment_start is None:
                            continue
                        misses += 1
                        if misses > tail_bridge_gap:
                            break
                    if segment_start is None or last_support is None:
                        continue
                    for j in range(segment_start, last_support + 1):
                        old = annotations[j].get("primary_event")
                        if _force_highway_r3(j, "event_merger_post_lane_change_merge_tail_preserved_as_r3"):
                            metrics = _event_metrics(annotations[j])
                            metrics["merger_post_re2_tail_r3_preserved"] = True
                            annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                            changes.append(
                                {
                                    "frame_id": annotations[j].get("frame_id"),
                                    "from": old,
                                    "to": EventType.R_E3.value,
                                    "reason": "merger_post_lane_change_merge_tail_preserved_as_r3",
                                }
                            )

        if scenario_name in {"InterurbanActorFlow", "InterurbanAdvancedActorFlow"}:
            backward_pad = 3
            forward_pad = 4
            is_advanced_interurban = scenario_name == "InterurbanAdvancedActorFlow"

            def _interurban_re2_boundary_support(index: int) -> bool:
                ann = annotations[index]
                allowed_rs = (
                    {RoadStructure.R1.value, RoadStructure.R5.value}
                    if is_advanced_interurban
                    else {RoadStructure.R1.value, RoadStructure.R3.value}
                )
                allowed_events = (
                    {EventType.R_E1.value, EventType.R_E2.value, EventType.R_E5.value}
                    if is_advanced_interurban
                    else {EventType.R_E1.value, EventType.R_E2.value}
                )
                if (
                    ann.get("primary_road_structure") not in allowed_rs
                    or ann.get("primary_event") not in allowed_events
                ):
                    return False
                metrics = _event_metrics(ann)
                signed = _signed_lane_change(ann)
                route_abs = _route_lateral_abs(ann)
                route_change = bool(metrics.get("changed_route")) and (
                    (math.isfinite(signed) and abs(signed) <= 3.5)
                    or (math.isfinite(route_abs) and route_abs >= 0.05)
                    or not _route_centered_for_re2_exit(ann)
                )
                if not route_change:
                    return False
                if not is_advanced_interurban:
                    return True
                # Advanced interurban routes often perform the target lane-change
                # while the primary RS is already the no-signal junction.  Only
                # allow that R-E2 when the frame is in or immediately adjacent to
                # the R5 junction span, so ordinary R1 cruising stays R-E1.
                if ann.get("primary_road_structure") == RoadStructure.R5.value:
                    return True
                near_r5 = any(
                    annotations[j].get("primary_road_structure") == RoadStructure.R5.value
                    for j in range(max(0, index - 6), min(len(annotations), index + 7))
                )
                return near_r5

            original_re2_spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                original_re2_spans.append((start, idx))

            for start, end in original_re2_spans:
                new_start = start
                while (
                    new_start > 0
                    and start - new_start < backward_pad
                    and _interurban_re2_boundary_support(new_start - 1)
                ):
                    new_start -= 1
                new_end = end
                while (
                    new_end < len(annotations)
                    and new_end - end < forward_pad
                    and _interurban_re2_boundary_support(new_end)
                ):
                    new_end += 1
                for j in range(new_start, new_end):
                    if annotations[j].get("primary_event") == EventType.R_E2.value:
                        continue
                    old = annotations[j].get("primary_event")
                    self._rewrite_event_label(
                        annotations[j],
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_interurban_lane_change_span_expanded_by_trajectory",
                    )
                    metrics = _event_metrics(annotations[j])
                    metrics["interurban_re2_span_expanded"] = True
                    annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                    changes.append(
                        {
                            "frame_id": annotations[j].get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "interurban_lane_change_span_expanded_by_trajectory",
                        }
                    )
            if is_advanced_interurban:
                supported_re2_spans: List[Tuple[int, int]] = []
                idx = 0
                while idx < len(annotations):
                    if not _interurban_re2_boundary_support(idx):
                        idx += 1
                        continue
                    start = idx
                    while idx < len(annotations) and _interurban_re2_boundary_support(idx):
                        idx += 1
                    end = idx
                    has_junction = any(
                        annotations[j].get("primary_road_structure") == RoadStructure.R5.value
                        for j in range(start, end)
                    )
                    if has_junction and end - start >= 2:
                        supported_re2_spans.append((start, end))

                for start, end in supported_re2_spans:
                    new_start = start
                    while (
                        new_start > 0
                        and start - new_start < backward_pad
                        and _interurban_re2_boundary_support(new_start - 1)
                    ):
                        new_start -= 1
                    new_end = end
                    while (
                        new_end < len(annotations)
                        and new_end - end < forward_pad
                        and _interurban_re2_boundary_support(new_end)
                    ):
                        new_end += 1
                    for j in range(new_start, new_end):
                        if annotations[j].get("primary_event") == EventType.R_E2.value:
                            continue
                        old = annotations[j].get("primary_event")
                        self._rewrite_event_label(
                            annotations[j],
                            {EventType.R_E2},
                            EventType.R_E2,
                            "event_interurban_advanced_junction_lane_change_re2_by_trajectory",
                        )
                        metrics = _event_metrics(annotations[j])
                        metrics["interurban_advanced_re2_span_expanded"] = True
                        metrics["interurban_re2_span_expanded"] = True
                        annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                        changes.append(
                            {
                                "frame_id": annotations[j].get("frame_id"),
                                "from": old,
                                "to": EventType.R_E2.value,
                                "reason": "interurban_advanced_junction_lane_change_re2_by_trajectory",
                            }
                        )

        if scenario_name == "HighwayExit":
            transition_start = None
            for j, ann in enumerate(annotations):
                if ann.get("primary_event") in {EventType.R_E2.value, EventType.R_E3.value}:
                    transition_start = j
                    break
            if transition_start is not None:
                for j in range(transition_start, len(annotations)):
                    if annotations[j].get("primary_event") == EventType.R_E2.value:
                        continue
                    old = annotations[j].get("primary_event")
                    if _force_highway_r3(j, "event_highway_exit_ramp_span_backfilled_to_r3"):
                        changes.append(
                            {
                                "frame_id": annotations[j].get("frame_id"),
                                "from": old,
                                "to": EventType.R_E3.value,
                                "reason": "highway_exit_ramp_span_backfilled_to_r3",
                            }
                        )

        if scenario_name in (highway_enter_merge_scenarios | {"HighwayExit"}):
            idx = 0
            while idx < len(annotations):
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.R_E3.value or end - start > 3:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != EventType.R_E2.value or next_label != EventType.R_E2.value:
                    continue
                if not any(
                    _route_change_hint(annotations[j])
                    for j in range(max(0, start - 2), min(len(annotations), end + 2))
                ):
                    continue
                for j in range(start, end):
                    old = annotations[j].get("primary_event")
                    self._rewrite_event_label(
                        annotations[j],
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_highway_short_r3_gap_inside_lane_change_merged",
                    )
                    annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                    changes.append(
                        {
                            "frame_id": annotations[j].get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "highway_short_r3_gap_inside_lane_change_merged",
                        }
                    )

        if scenario_name in {"MergerIntoSlowTraffic", "MergerIntoSlowTrafficV2"}:
            idx = 0
            while idx < len(annotations):
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.R_E3.value or end - start > 3:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != EventType.R_E1.value or next_label != EventType.R_E1.value:
                    continue
                if any(
                    annotations[j].get("primary_event") == EventType.R_E2.value
                    for j in range(max(0, start - 8), min(len(annotations), end + 8))
                ):
                    continue
                for j in range(start, end):
                    old = annotations[j].get("primary_event")
                    self._rewrite_event_label(
                        annotations[j],
                        {EventType.R_E1},
                        EventType.R_E1,
                        "event_merger_short_isolated_r3_gap_smoothed_to_r1",
                    )
                    metrics = _event_metrics(annotations[j])
                    metrics["merger_short_isolated_r3_gap_smoothed"] = True
                    annotations[j]["frame_event_annotation"] = self._frame_event_annotation_payload(annotations[j])
                    changes.append(
                        {
                            "frame_id": annotations[j].get("frame_id"),
                            "from": old,
                            "to": EventType.R_E1.value,
                            "reason": "merger_short_isolated_r3_gap_smoothed_to_r1",
                        }
                    )

        def _twoways_current_core_active(ann: Dict[str, Any]) -> bool:
            """TwoWays U-E2/U-E3 starts only when the current frame is in the actual opposite-lane core."""
            if "TwoWays" not in scenario_name:
                return False
            rules = set((ann.get("evidence") or {}).get("rules_fired") or [])
            strong_rs_rules = {
                "r2_core_obstruction_confirmed",
                "r2_strict_core_obstruction_window",
                "r2_core_obstruction_meta_confirmed_without_trusted_xodr",
                "r2_xml_trigger_core_confirmed",
                "accidenttwoways_r2_core_priority_over_junction",
            }
            r2_removed_as_disturbance = "temporal_smoothing_twoways_non_longest_r2_disturbance" in rules
            if rules & strong_rs_rules and not r2_removed_as_disturbance:
                return True
            event_metrics = ((ann.get("event_evidence") or {}).get("metrics") or {})
            twoway_metrics = event_metrics.get("twoway_obstruction") or {}
            if (
                bool(twoway_metrics.get("core_confirmed"))
                or bool(twoway_metrics.get("stuck"))
                or bool(twoway_metrics.get("vehicle_hazard"))
            ):
                return True
            dist = _specific_obstacle_distance(ann)
            return math.isfinite(dist) and dist <= 18.0

        if scenario_name in R2_RETURN_SCENARIOS:
            max_u2_gap_frames = 6
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                left_end = idx + 1
                while left_end < len(annotations) and annotations[left_end].get("primary_event") == EventType.U_E2.value:
                    left_end += 1
                gap_end = left_end
                while gap_end < len(annotations) and annotations[gap_end].get("primary_event") != EventType.U_E2.value:
                    gap_end += 1
                if (
                    gap_end < len(annotations)
                    and 0 < gap_end - left_end <= max_u2_gap_frames
                    and all(str(annotations[j].get("primary_event", "")).startswith("R-E") for j in range(left_end, gap_end))
                ):
                    for ann in annotations[left_end:gap_end]:
                        old = ann.get("primary_event")
                        self._rewrite_event_label(ann, {EventType.U_E2}, EventType.U_E2, "event_short_u2_gap_merged")
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": EventType.U_E2.value,
                                "reason": "short_u2_gap_merged",
                            }
                        )
                    idx = left_end
                    continue
                idx = left_end

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                end = idx
                if start > 2 or end > 18:
                    continue
                lookahead_end = min(len(annotations), end + 8)
                has_specific_obstacle = any(_specific_obstacle_close(annotations[j], pad_m=4.0) for j in range(start, lookahead_end))
                has_near_route_change = any(_route_change_hint(annotations[j]) for j in range(start, lookahead_end))
                has_twoway_core = any(
                    "event_twoway_r2_lane_change_core" in ((ann.get("event_evidence") or {}).get("rules_fired") or [])
                    or "event_twoway_core_obstruction" in ((ann.get("event_evidence") or {}).get("rules_fired") or [])
                    for ann in annotations[start:lookahead_end]
                )
                if has_specific_obstacle or has_near_route_change or has_twoway_core:
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    replacement = _regular_event_for_annotation(ann)
                    self._rewrite_event_label(ann, {replacement}, replacement, "event_initial_trigger_only_u2_suppressed")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": "initial_trigger_only_u2_suppressed",
                        }
                    )

            if scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}:
                def _hazard_recovery_supported(index: int) -> bool:
                    ann = annotations[index]
                    metrics = _event_metrics(ann)
                    rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
                    signed = _signed_lane_change(ann)
                    route_abs = _route_lateral_abs(ann)
                    tol = _route_center_tolerance(ann)
                    return (
                        "event_regular_target_lane_change_r2" in rules
                        or bool(metrics.get("target_lane_change_active"))
                        or (
                            bool(metrics.get("changed_route"))
                            and math.isfinite(signed)
                            and abs(signed) <= 3.5
                        )
                        or (
                            math.isfinite(route_abs)
                            and route_abs > max(0.35, tol * 0.75)
                        )
                    )

                def _hazard_recovery_complete(index: int, first_support: int) -> bool:
                    if index < first_support + 2:
                        return False
                    ann = annotations[index]
                    metrics = _event_metrics(ann)
                    route_abs = _route_lateral_abs(ann)
                    tol = _route_center_tolerance(ann)
                    if bool(metrics.get("target_lane_change_active")):
                        return False
                    return math.isfinite(route_abs) and route_abs <= max(0.16, tol * 0.30)

                def _hazard_regular_for_rs(ann: Dict[str, Any]) -> EventType:
                    rs_label = ann.get("primary_road_structure")
                    if rs_label == RoadStructure.R4.value:
                        return EventType.R_E4
                    if rs_label == RoadStructure.R5.value:
                        return EventType.R_E5
                    return EventType.R_E1

                def _rewrite_hazard_recovery(
                    ann: Dict[str, Any],
                    source_rs: str,
                    source_frame: Any,
                    recovery_start_frame: Any,
                    reason: str,
                ) -> None:
                    rs_label = ann.get("primary_road_structure")
                    regular = _hazard_regular_for_rs(ann)
                    if rs_label in {RoadStructure.R4.value, RoadStructure.R5.value}:
                        ann["events"] = [ev.value for ev in sorted({regular, EventType.R_E2}, key=lambda ev: ev.value)]
                        ann["primary_event"] = EventType.R_E2.value
                        event_evidence = ann.setdefault("event_evidence", {})
                        event_evidence["events"] = ann["events"]
                        event_evidence["primary_event"] = EventType.R_E2.value
                        event_evidence["regular_event"] = regular.value
                        event_evidence["unusual_event"] = None
                        event_evidence["overlay_recovery_event"] = EventType.R_E2.value
                        age = max(1, int(ann.get("frame_id", 0) or 0) - int(source_frame or 0))
                        recovery_age = max(1, int(ann.get("frame_id", 0) or 0) - int(recovery_start_frame or 0) + 1)
                        event_evidence["interrupted_event_overlay"] = {
                            "active": True,
                            "base_road_structure": source_rs,
                            "intersection_road_structure": rs_label,
                            "regular_event": regular.value,
                            "overlay_event": EventType.R_E2.value,
                            "source_unusual_event": EventType.U_E4.value,
                            "age_frames": age,
                            "age_seconds": round(age * 0.25, 3),
                            "max_frames": INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES,
                            "recovery_age_frames": recovery_age,
                            "recovery_max_frames": INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES,
                            "phase": "hazard_side_post_u4_recovery_to_target_lane",
                            "reason": "hazard_side_u4_recovery_requires_re2",
                        }
                        self._attach_overlay_base_rs(
                            ann,
                            source_rs,
                            "hazard_side_u4_recovery_overlay_base_rs",
                        )
                        event_evidence.setdefault("rules_fired", []).append(reason)
                    else:
                        self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, reason)

                idx_h = 0
                while idx_h < len(annotations):
                    if annotations[idx_h].get("primary_event") != EventType.U_E4.value:
                        idx_h += 1
                        continue
                    u4_start = idx_h
                    while idx_h < len(annotations) and annotations[idx_h].get("primary_event") == EventType.U_E4.value:
                        idx_h += 1
                    u4_end = idx_h
                    if u4_end >= len(annotations):
                        continue
                    search_end = min(len(annotations), u4_end + 9)
                    support_idx = next((j for j in range(u4_end, search_end) if _hazard_recovery_supported(j)), None)
                    if support_idx is None:
                        continue
                    recovery_start = u4_end
                    recovery_end = min(len(annotations), support_idx + 33)
                    for j in range(support_idx, recovery_end):
                        if _hazard_recovery_complete(j, support_idx):
                            recovery_end = j + 1
                            break
                    source_rs = str(annotations[u4_end - 1].get("primary_road_structure") or RoadStructure.R1.value)
                    source_frame = annotations[u4_end - 1].get("frame_id")
                    recovery_start_frame = annotations[recovery_start].get("frame_id")
                    for j in range(recovery_start, recovery_end):
                        if annotations[j].get("primary_event") == EventType.U_E4.value:
                            continue
                        old = annotations[j].get("primary_event")
                        _rewrite_hazard_recovery(
                            annotations[j],
                            source_rs,
                            source_frame,
                            recovery_start_frame,
                            "event_hazard_side_post_u4_recovery_to_r2",
                        )
                        if old != EventType.R_E2.value:
                            changes.append(
                                {
                                    "frame_id": annotations[j].get("frame_id"),
                                    "from": old,
                                    "to": EventType.R_E2.value,
                                    "reason": "hazard_side_post_u4_recovery_to_r2",
                                    "u4_start_frame": annotations[u4_start].get("frame_id"),
                                    "u4_end_frame": annotations[u4_end - 1].get("frame_id"),
                                    "support_frame": annotations[support_idx].get("frame_id"),
                                }
                            )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                end = idx
                if end - start < 3:
                    continue
                extended = 0
                cursor = end
                while cursor < len(annotations) and extended < 3:
                    ann = annotations[cursor]
                    label = ann.get("primary_event")
                    if label == EventType.R_E2.value:
                        break
                    if label not in {EventType.R_E1.value, EventType.R_E3.value}:
                        break
                    if not _avoidance_lane_u2_supported(cursor):
                        break
                    if obstacle_field_cfg is not None:
                        nearby_obstacle_context = any(
                            _specific_obstacle_close(annotations[j], pad_m=5.0)
                            or _obstacle_still_close(annotations[j], pad_m=3.0)
                            for j in range(max(start, cursor - 4), min(len(annotations), cursor + 2))
                        )
                        if not nearby_obstacle_context:
                            break
                    old = label
                    self._rewrite_event_label(ann, {EventType.U_E2}, EventType.U_E2, "event_u2_extended_until_avoidance_lane_complete")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "u2_extended_until_avoidance_lane_complete",
                        }
                    )
                    extended += 1
                    cursor += 1

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                pre_start = start
                while pre_start > 0 and annotations[pre_start - 1].get("primary_event") == EventType.R_E2.value:
                    pre_start -= 1
                if pre_start == start or start - pre_start > 10:
                    continue
                previous_u2_nearby = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(max(0, pre_start - 16), pre_start)
                )
                if previous_u2_nearby:
                    continue
                if not any(_route_change_hint(annotations[j]) for j in range(pre_start, start)):
                    continue
                for ann in annotations[pre_start:start]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.U_E2}, EventType.U_E2, "event_pre_u2_avoidance_lane_change_absorbed")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "pre_u2_avoidance_lane_change_absorbed",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                end = idx
                if end - start < 8:
                    continue
                closest_idx = None
                closest_dist = math.inf
                for j in range(start, end):
                    dist = _specific_obstacle_distance(annotations[j])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = j
                if closest_idx is None or not math.isfinite(closest_dist):
                    closest_idx = None
                return_start = _route_recovery_start(start, end, closest_idx)
                if return_start is None:
                    continue
                return_end = return_start
                hold_without_hint = 0
                center_completed = False
                while return_end < end:
                    ann = annotations[return_end]
                    if return_end < return_start + 2:
                        return_end += 1
                        continue
                    if return_end > return_start + 1 and _route_centered(ann):
                        future = range(return_end + 1, min(end, return_end + 4))
                        future_centered = all(
                            _route_centered(annotations[j])
                            and not _signed_lane_change_active(annotations[j])
                            for j in future
                        )
                        if future_centered and not _signed_lane_change_active(ann):
                            center_completed = True
                            break
                    if _route_change_hint(ann) or _return_lane_change_hint(ann):
                        hold_without_hint = 0
                        return_end += 1
                        continue
                    if _route_centering_trend(return_end):
                        hold_without_hint = 0
                        return_end += 1
                        continue
                    if hold_without_hint < 1 and _trigger_distance(ann) > 45.0:
                        hold_without_hint += 1
                        return_end += 1
                        continue
                    break
                for ann in annotations[return_start:return_end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_u2_return_lane_change_to_r2")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "u2_return_lane_change_to_r2",
                        }
                    )
                if center_completed:
                    for ann in annotations[return_end:end]:
                        old = ann.get("primary_event")
                        replacement = _regular_event_for_annotation(ann)
                        self._rewrite_event_label(ann, {replacement}, replacement, "event_u2_after_recovery_center_released")
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": replacement.value,
                                "reason": "u2_after_recovery_center_released",
                            }
                        )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                end = idx
                if end >= len(annotations) or annotations[end].get("primary_event") != EventType.R_E2.value:
                    continue
                closest_idx = None
                closest_dist = math.inf
                for j in range(start, end):
                    dist = _specific_obstacle_distance(annotations[j])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_idx = j
                if closest_idx is None or not math.isfinite(closest_dist):
                    closest_idx = end - 1
                pull_start = end
                for j in range(end - 1, max(start - 1, end - 4), -1):
                    passed_core = j >= max(start + 3, closest_idx - 1)
                    next_active = _route_change_hint(annotations[end]) or _route_centering_trend(end)
                    preparing_return = _return_lane_change_hint(annotations[j]) or _route_centering_trend(j) or next_active
                    if not (passed_core and preparing_return):
                        break
                    pull_start = j
                    if end - pull_start >= 2:
                        break
                for ann in annotations[pull_start:end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_u2_r2_boundary_pulled_left_by_centerline")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "u2_r2_boundary_pulled_left_by_centerline",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                ann = annotations[idx]
                if ann.get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                far_from_trigger = _trigger_distance(ann) > 80.0
                if far_from_trigger and not _obstacle_still_close(ann, pad_m=6.0) and not _route_change_hint(ann):
                    replacement = _regular_event_for_annotation(ann)
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {replacement}, replacement, "event_u2_far_from_trigger_without_obstacle_released")
                    event_evidence = ann.setdefault("event_evidence", {})
                    event_evidence["review_required"] = True
                    event_evidence.setdefault("review_reasons", []).append("u2_released_far_from_xml_trigger")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": "u2_far_from_trigger_without_obstacle_released",
                        }
                    )
                    idx += 1
                    continue
                idx += 1

            if annotations[-1].get("primary_event") == EventType.U_E2.value:
                tail_start = len(annotations) - 1
                while tail_start > 0 and annotations[tail_start - 1].get("primary_event") == EventType.U_E2.value:
                    tail_start -= 1
                tail_reason = "u2_reaches_route_end_requires_review"
                for ann in annotations[tail_start:]:
                    event_evidence = ann.setdefault("event_evidence", {})
                    event_evidence["review_required"] = True
                    reasons = event_evidence.setdefault("review_reasons", [])
                    if tail_reason not in reasons:
                        reasons.append(tail_reason)
                changes.append(
                    {
                        "start_frame": annotations[tail_start].get("frame_id"),
                        "end_frame": annotations[-1].get("frame_id"),
                        "from": EventType.U_E2.value,
                        "to": EventType.U_E2.value,
                        "reason": tail_reason,
                    }
                )

            u2_run_len_ending_at: Dict[int, int] = {}
            scan_idx = 0
            while scan_idx < len(annotations):
                if annotations[scan_idx].get("primary_event") != EventType.U_E2.value:
                    scan_idx += 1
                    continue
                run_start = scan_idx
                while scan_idx < len(annotations) and annotations[scan_idx].get("primary_event") == EventType.U_E2.value:
                    scan_idx += 1
                u2_run_len_ending_at[scan_idx - 1] = scan_idx - run_start

            last_u2_end = None
            for idx, ann in enumerate(annotations):
                primary = ann.get("primary_event")
                if primary == EventType.U_E2.value:
                    last_u2_end = idx
                    continue
                if last_u2_end is None:
                    continue
                if u2_run_len_ending_at.get(last_u2_end, 0) < 4:
                    last_u2_end = None
                    continue
                if idx - last_u2_end > 16:
                    last_u2_end = None
                    continue
                rs = ann.get("primary_road_structure")
                if (
                    primary == EventType.R_E1.value
                    and rs not in {RoadStructure.R4.value, RoadStructure.R5.value}
                    and _route_change_hint(ann)
                    and (not _route_centered(ann) or _signed_lane_change_active(ann))
                ):
                    self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_post_u2_recovery_to_target_lane")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": primary,
                            "to": EventType.R_E2.value,
                            "reason": "post_u2_recovery_window",
                        }
                    )
                elif primary not in {EventType.R_E1.value, EventType.R_E2.value}:
                    last_u2_end = None

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E1.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                    idx += 1
                end = idx
                if end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                recent_u2 = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(max(0, start - 16), start)
                )
                if prev_label != EventType.R_E2.value or next_label != EventType.R_E2.value or not recent_u2:
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_short_r1_gap_inside_recovery_r2_merged")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "short_r1_gap_inside_recovery_r2_merged",
                            }
                        )

            if "TwoWays" not in scenario_name:
                idx = 0
                while idx < len(annotations):
                    if annotations[idx].get("primary_event") != EventType.U_E2.value:
                        idx += 1
                        continue
                    start = idx
                    while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                        idx += 1
                    end = idx
                    if end - start < 8:
                        continue
                    finite_points = [
                        (j, _specific_obstacle_distance(annotations[j]))
                        for j in range(start, end)
                        if math.isfinite(_specific_obstacle_distance(annotations[j]))
                    ]
                    if len(finite_points) < 4:
                        continue
                    closest_idx, closest_dist = min(finite_points, key=lambda item: item[1])
                    release_start = None
                    for j in range(max(start + 4, closest_idx + 2), end):
                        dist = _specific_obstacle_distance(annotations[j])
                        if not math.isfinite(dist) or dist < closest_dist + 2.0:
                            continue
                        future = range(j, min(end, j + 3))
                        future_centered = all(
                            _route_centered(annotations[k])
                            and not _signed_lane_change_active(annotations[k])
                            for k in future
                        )
                        still_recovering = _route_change_hint(annotations[j]) or _return_lane_change_hint(annotations[j]) or _route_centering_trend(j)
                        if not (future_centered or still_recovering):
                            continue
                        release_start = max(start, j - 1 if still_recovering else j)
                        break
                    if release_start is None:
                        continue
                    for j in range(release_start, end):
                        ann = annotations[j]
                        old = ann.get("primary_event")
                        still_recovering = (
                            _route_change_hint(ann)
                            or _return_lane_change_hint(ann)
                            or _route_centering_trend(j)
                        ) and (not _route_centered(ann) or _signed_lane_change_active(ann))
                        if still_recovering:
                            self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_u2_passed_obstacle_recovery_to_r2")
                            replacement = EventType.R_E2
                            reason = "u2_passed_obstacle_recovery_to_r2"
                        else:
                            replacement = _regular_event_for_annotation(ann)
                            self._rewrite_event_label(ann, {replacement}, replacement, "event_u2_passed_obstacle_centered_released")
                            reason = "u2_passed_obstacle_centered_released"
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": replacement.value,
                                "reason": reason,
                            }
                        )

            idx = 0
            while idx < len(annotations):
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.U_E2.value or end - start > 4:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != EventType.R_E2.value:
                    continue
                recovery_continues = next_label in {
                    EventType.R_E2.value,
                    EventType.R_E1.value,
                    EventType.R_E4.value,
                    EventType.R_E5.value,
                }
                center_or_return = any(
                    _route_centered(annotations[j])
                    or _return_lane_change_hint(annotations[j])
                    or _route_centering_trend(j)
                    for j in range(start, end)
                )
                if not (recovery_continues or center_or_return):
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_short_u2_after_recovery_r2_merged")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "short_u2_after_recovery_r2_merged",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.R_E2.value or end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                recent_recovery = any(
                    annotations[j].get("primary_event") == EventType.R_E2.value
                    for j in range(max(0, start - 8), start)
                )
                if prev_label == EventType.U_E2.value and next_label == EventType.U_E2.value and not recent_recovery:
                    for ann in annotations[start:end]:
                        old = ann.get("primary_event")
                        self._rewrite_event_label(ann, {EventType.U_E2}, EventType.U_E2, "event_short_r2_between_u2_smoothed")
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": EventType.U_E2.value,
                                "reason": "short_r2_between_u2_smoothed",
                            }
                        )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                has_recent_u2 = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(max(0, start - 24), start)
                )
                if not has_recent_u2:
                    continue
                complete_idx = None
                for j in range(start + 2, end):
                    if not _route_centered_for_re2_exit(annotations[j]) or _signed_lane_change_active(annotations[j]):
                        continue
                    future = range(j + 1, min(end, j + 1 + RE2_EXIT_STABLE_FUTURE_FRAMES))
                    future_centered = all(
                        _route_centered_for_re2_exit(annotations[k])
                        and not _signed_lane_change_active(annotations[k])
                        for k in future
                    )
                    if future_centered:
                        complete_idx = j
                        break
                if complete_idx is None:
                    continue
                for ann in annotations[complete_idx:end]:
                    old = ann.get("primary_event")
                    replacement = _regular_event_for_annotation(ann)
                    self._rewrite_event_label(ann, {replacement}, replacement, "event_r2_recovery_completed_at_route_center")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": "r2_recovery_completed_at_route_center",
                        }
                    )

            if scenario_name == "AccidentTwoWays":
                idx = 0
                while idx < len(annotations):
                    if annotations[idx].get("primary_event") != EventType.R_E1.value:
                        idx += 1
                        continue
                    start = idx
                    while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                        idx += 1
                    end = idx
                    if end - start > 12:
                        continue
                    prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                    next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                    if prev_label != EventType.R_E2.value:
                        continue
                    if next_label not in {EventType.R_E4.value, EventType.R_E5.value, None}:
                        continue
                    prev_r2_start = start - 1
                    while prev_r2_start > 0 and annotations[prev_r2_start - 1].get("primary_event") == EventType.R_E2.value:
                        prev_r2_start -= 1
                    if start - prev_r2_start < 4:
                        continue
                    recent_u2 = any(
                        annotations[j].get("primary_event") == EventType.U_E2.value
                        for j in range(max(0, prev_r2_start - 24), prev_r2_start)
                    )
                    if not recent_u2:
                        continue
                    if any(
                        annotations[j].get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}
                        for j in range(start, end)
                    ):
                        continue
                    for ann in annotations[start:end]:
                        old = ann.get("primary_event")
                        self._rewrite_event_label(
                            ann,
                            {EventType.R_E2},
                            EventType.R_E2,
                            "event_accidenttwoways_recovery_tail_r1_to_r2_final",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": EventType.R_E2.value,
                                "reason": "accidenttwoways_recovery_tail_r1_to_r2_final",
                                "previous_r2_start_frame": annotations[prev_r2_start].get("frame_id"),
                                "next_event": next_label,
                            }
                        )

            if scenario_name == "AccidentTwoWays":
                for index, ann in enumerate(annotations):
                    primary = ann.get("primary_event")
                    if primary not in {EventType.R_E1.value, EventType.R_E2.value}:
                        continue
                    if primary == EventType.R_E2.value and _lane_change_re2_supported(index):
                        event_evidence = ann.setdefault("event_evidence", {})
                        rules = event_evidence.setdefault("rules_fired", [])
                        if "event_accidenttwoways_core_keeps_existing_recovery_r2" not in rules:
                            rules.append("event_accidenttwoways_core_keeps_existing_recovery_r2")
                        continue
                    event_rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
                    event_metrics = ((ann.get("event_evidence") or {}).get("metrics") or {})
                    twoway_metrics = event_metrics.get("twoway_obstruction") or {}
                    dist = _specific_obstacle_distance(ann)
                    still_core = (
                        "event_twoway_core_obstruction" in event_rules
                        or "event_twoway_r2_lane_change_core" in event_rules
                        or bool(twoway_metrics.get("core_confirmed"))
                        or bool(twoway_metrics.get("stuck"))
                        or bool(twoway_metrics.get("vehicle_hazard"))
                        or (math.isfinite(dist) and dist <= 18.0)
                    )
                    if not still_core:
                        continue
                    old = primary
                    self._rewrite_event_label(
                        ann,
                        {EventType.U_E2},
                        EventType.U_E2,
                        "event_accidenttwoways_core_obstruction_keeps_u2",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "accidenttwoways_core_obstruction_keeps_u2",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.R_E2.value or end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"} and any(
                    "event_hazard_side_post_u4_recovery_to_r2" in ((annotations[j].get("event_evidence") or {}).get("rules_fired") or [])
                    for j in range(start, end)
                ):
                    continue
                if prev_label == EventType.R_E1.value and next_label == EventType.R_E1.value:
                    for ann in annotations[start:end]:
                        self._rewrite_event_label(ann, {EventType.R_E1}, EventType.R_E1, "event_short_r2_recovery_smoothed")
                    changes.append(
                        {
                            "start_frame": annotations[start].get("frame_id"),
                            "end_frame": annotations[end - 1].get("frame_id"),
                            "from": EventType.R_E2.value,
                            "to": EventType.R_E1.value,
                            "reason": "short_r2_recovery_smoothed",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                left_end = idx + 1
                while left_end < len(annotations) and annotations[left_end].get("primary_event") == EventType.U_E2.value:
                    left_end += 1
                gap_end = left_end
                while gap_end < len(annotations) and annotations[gap_end].get("primary_event") != EventType.U_E2.value:
                    gap_end += 1
                if gap_end >= len(annotations) or gap_end - left_end > 6:
                    idx = left_end
                    continue
                gap_labels = {annotations[j].get("primary_event") for j in range(left_end, gap_end)}
                if not gap_labels <= {EventType.R_E1.value, EventType.R_E2.value, EventType.R_E4.value, EventType.R_E5.value}:
                    idx = left_end
                    continue
                same_obstacle_core = any(
                    _specific_obstacle_close(annotations[j], pad_m=4.0)
                    or "event_twoway_r2_lane_change_core" in ((annotations[j].get("event_evidence") or {}).get("rules_fired") or [])
                    or "event_twoway_core_obstruction" in ((annotations[j].get("event_evidence") or {}).get("rules_fired") or [])
                    for j in range(max(0, left_end - 2), min(len(annotations), gap_end + 2))
                )
                if not same_obstacle_core:
                    idx = left_end
                    continue
                for ann in annotations[left_end:gap_end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(ann, {EventType.U_E2}, EventType.U_E2, "event_short_u2_gap_after_recovery_rules_merged")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "short_u2_gap_after_recovery_rules_merged",
                        }
                    )
                idx = left_end

        idx = 0
        while idx < len(annotations):
            label = annotations[idx].get("primary_event")
            start = idx
            while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                idx += 1
            end = idx
            if label != EventType.U_E3.value or end - start > 4:
                continue
            prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
            next_label = annotations[end].get("primary_event") if end < len(annotations) else None
            merge_start = start
            if prev_label != EventType.R_E2.value:
                gap_start = start
                while (
                    gap_start > 0
                    and start - gap_start <= 2
                    and annotations[gap_start - 1].get("primary_event")
                    in {
                        EventType.R_E1.value,
                        EventType.R_E3.value,
                        EventType.R_E4.value,
                        EventType.R_E5.value,
                    }
                ):
                    gap_start -= 1
                prior_label = annotations[gap_start - 1].get("primary_event") if gap_start > 0 else None
                if prior_label != EventType.R_E2.value:
                    continue
                merge_start = gap_start
            if prev_label != EventType.R_E2.value and merge_start == start:
                continue
            recovery_continues = next_label in {
                EventType.R_E2.value,
                EventType.R_E1.value,
                EventType.R_E3.value,
                EventType.R_E4.value,
                EventType.R_E5.value,
            }
            center_or_return = any(
                _route_centered(annotations[j])
                or _return_lane_change_hint(annotations[j])
                or _route_centering_trend(j)
                for j in range(start, end)
            )
            if not (recovery_continues or center_or_return):
                continue
            for ann in annotations[merge_start:end]:
                old = ann.get("primary_event")
                self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_short_u3_after_recovery_r2_merged")
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E2.value,
                        "reason": "short_u3_after_recovery_r2_merged",
                    }
                )

        if scenario_name in {"ParkingCutIn", "StaticCutIn"}:
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E3.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E3.value:
                    idx += 1
                end = idx
                if end - start < 4:
                    continue
                finite_points = [
                    (j, _cutin_distance(annotations[j]))
                    for j in range(start, end)
                    if math.isfinite(_cutin_distance(annotations[j]))
                ]
                if len(finite_points) < 3:
                    continue
                closest_idx, closest_dist = min(finite_points, key=lambda item: item[1])
                release_start = None
                for j in range(max(start + 2, closest_idx + 2), end):
                    dist = _cutin_distance(annotations[j])
                    if not math.isfinite(dist) or dist < closest_dist + 2.0:
                        continue
                    future = range(j, min(end, j + 3))
                    future_response = any(_cutin_response_active(annotations[k]) for k in future)
                    if future_response:
                        continue
                    release_start = j
                    break
                if release_start is None:
                    continue
                for j in range(release_start, end):
                    ann = annotations[j]
                    old = ann.get("primary_event")
                    if _route_change_hint(ann) and not _route_centered(ann):
                        replacement = EventType.R_E2
                        self._rewrite_event_label(ann, {replacement}, replacement, "event_u3_passed_cutin_recovery_to_r2")
                        reason = "u3_passed_cutin_recovery_to_r2"
                    else:
                        replacement = _regular_event_for_annotation(ann)
                        self._rewrite_event_label(ann, {replacement}, replacement, "event_u3_passed_cutin_released")
                        reason = "u3_passed_cutin_released"
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": reason,
                        }
                    )

        idx = 0
        while idx < len(annotations):
            label = annotations[idx].get("primary_event")
            start = idx
            while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                idx += 1
            end = idx
            if label != EventType.R_E2.value or end - start > 2:
                continue
            prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
            next_label = annotations[end].get("primary_event") if end < len(annotations) else None
            if prev_label != next_label or prev_label not in {
                EventType.R_E1.value,
                EventType.R_E3.value,
                EventType.R_E4.value,
                EventType.R_E5.value,
            }:
                continue
            if any(
                annotations[j].get("primary_event") in {EventType.U_E2.value, EventType.U_E3.value}
                for j in range(max(0, start - 16), start)
            ):
                continue
            if any(_lane_change_re2_supported(j) for j in range(start, end)):
                continue
            replacement = EventType(prev_label)
            for ann in annotations[start:end]:
                old = ann.get("primary_event")
                self._rewrite_event_label(ann, {replacement}, replacement, "event_short_isolated_r2_smoothed_by_centerline")
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": replacement.value,
                        "reason": "short_isolated_r2_smoothed_by_centerline",
                    }
                )

        if scenario_name == "InvadingTurn":
            r2_core_rules = {
                "passive_oncoming_invasion",
                "r2_passive_invading_turn",
                "r2_opposite_lane_confirmed",
                "r2_scenario_trigger_medium",
            }
            r2_tail_rules = {
                "passive_oncoming_invasion",
                "r2_passive_invading_turn",
                "r2_opposite_lane_confirmed",
                "r2_scenario_trigger_medium",
            }

            def _invading_rules(ann: Dict[str, Any]) -> Set[str]:
                return set(str(rule) for rule in ((ann.get("evidence") or {}).get("rules_fired") or []))

            def _invading_trigger_distance(ann: Dict[str, Any]) -> float:
                return _safe_float((ann.get("evidence") or {}).get("trigger_distance_m"), default=math.inf)

            def _invading_tail_supported(index: int) -> bool:
                ann = annotations[index]
                rs = ann.get("primary_road_structure")
                if rs not in {RoadStructure.R1.value, RoadStructure.R2.value}:
                    return False
                rules = _invading_rules(ann)
                if "passive_oncoming_invasion" not in rules:
                    return False
                if not (rules & {"r2_passive_invading_turn", "r2_opposite_lane_confirmed", "r2_scenario_trigger_medium"}):
                    return False
                trigger_distance = _invading_trigger_distance(ann)
                if not math.isfinite(trigger_distance) or trigger_distance < 35.0:
                    return False
                metrics = _event_metrics(ann)
                route_abs = _route_lateral_abs(ann)
                still_responding = (
                    bool(metrics.get("changed_route"))
                    or bool(metrics.get("hard_decel"))
                    or bool(metrics.get("vehicle_hazard"))
                    or (math.isfinite(route_abs) and route_abs >= 0.05)
                )
                # RGB review shows long cone rows remain after the ego lateral response
                # recenters; keep the tail while the R2 invasion rules persist.
                return still_responding or rs == RoadStructure.R2.value

            def _rewrite_invading_u5(index: int, reason: str) -> None:
                ann = annotations[index]
                old = ann.get("primary_event")
                regular = _regular_event_for_annotation(ann)
                events = {EventType.U_E5}
                if ann.get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}:
                    events.add(regular)
                self._rewrite_event_label(ann, events, EventType.U_E5, reason)
                metrics = _event_metrics(ann)
                metrics["invading_turn_cone_tail_u5"] = True
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.U_E5.value,
                        "reason": reason,
                    }
                )

            for ann in annotations:
                if (
                    ann.get("primary_road_structure") != RoadStructure.R2.value
                    or ann.get("primary_event") != EventType.R_E1.value
                ):
                    continue
                evidence = ann.get("evidence") or {}
                rs_rules = set(str(rule) for rule in (evidence.get("rules_fired") or []))
                if not (rs_rules & r2_core_rules):
                    continue
                metrics = _event_metrics(ann)
                trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
                route_lateral_abs = _safe_float(metrics.get("route_lateral_abs_m"), default=math.inf)
                if not (
                    bool(metrics.get("changed_route"))
                    and math.isfinite(trigger_distance)
                    and trigger_distance >= 45.0
                    and math.isfinite(route_lateral_abs)
                    and route_lateral_abs >= 0.01
                ):
                    continue
                old = ann.get("primary_event")
                self._rewrite_event_label(
                    ann,
                    {EventType.U_E5},
                    EventType.U_E5,
                    "event_invading_turn_final_r2_local_invasion_u5",
                )
                metrics["invading_turn_final_r2_local_invasion_u5"] = True
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.U_E5.value,
                        "reason": "invading_turn_final_r2_local_invasion_u5",
                    }
                )

            # Existing U-E5 seeds mark the point where the ego reaches the invaded lane.
            # Continue the event while the cone/oncoming-occupation rules persist; this
            # avoids ending at the lateral-response peak while the cone row is still visible.
            max_tail_frames = 56
            max_r1_tail_frames = 32
            max_gap_frames = 4
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E5.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E5.value:
                    idx += 1
                end = idx
                tail_idx = end
                gap = 0
                last_r2_idx: Optional[int] = next(
                    (
                        j
                        for j in range(end - 1, start - 1, -1)
                        if annotations[j].get("primary_road_structure") == RoadStructure.R2.value
                    ),
                    None,
                )
                while tail_idx < len(annotations) and tail_idx - start < max_tail_frames:
                    if annotations[tail_idx].get("primary_event") == EventType.U_E5.value:
                        gap = 0
                        if annotations[tail_idx].get("primary_road_structure") == RoadStructure.R2.value:
                            last_r2_idx = tail_idx
                        tail_idx += 1
                        continue
                    if _invading_tail_supported(tail_idx):
                        if annotations[tail_idx].get("primary_road_structure") == RoadStructure.R1.value:
                            if last_r2_idx is None or tail_idx - last_r2_idx > max_r1_tail_frames:
                                break
                        else:
                            last_r2_idx = tail_idx
                        _rewrite_invading_u5(tail_idx, "event_invading_turn_cone_occupation_tail_u5")
                        gap = 0
                        tail_idx += 1
                        continue
                    gap += 1
                    if gap > max_gap_frames:
                        break
                    tail_idx += 1
                idx = max(idx, tail_idx)

            # Some routes have the same long cone row but the initial lateral-response
            # peak is too brief to create a seed. Seed from the supported R2/R1 cluster,
            # then let the same tail rule carry it to the visual clear point.
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") == EventType.U_E5.value or not _invading_tail_supported(idx):
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") != EventType.U_E5.value and _invading_tail_supported(idx):
                    idx += 1
                end = idx
                if end - start < 8:
                    continue
                if any(
                    annotations[j].get("primary_event") == EventType.U_E5.value
                    for j in range(start, min(len(annotations), end + max_tail_frames))
                ):
                    continue
                has_r2 = any(annotations[j].get("primary_road_structure") == RoadStructure.R2.value for j in range(start, end))
                has_motion_or_response = any(
                    bool(_event_metrics(annotations[j]).get("changed_route"))
                    or bool(_event_metrics(annotations[j]).get("hard_decel"))
                    or _route_lateral_abs(annotations[j]) >= 0.12
                    for j in range(start, end)
                )
                # Some InvadingTurn routes show a long cone row / occupied oncoming lane
                # in RGB, but the ego response is small because the lane is already narrow.
                # In that case the stable R2 invasion rules are enough to seed U-E5.
                if not has_r2:
                    continue
                seed_start = next(
                    (
                        j
                        for j in range(start, end)
                        if _invading_trigger_distance(annotations[j]) >= 25.0
                        and (
                            annotations[j].get("primary_road_structure") == RoadStructure.R2.value
                            or has_motion_or_response
                            or bool(_event_metrics(annotations[j]).get("changed_route"))
                            or _route_lateral_abs(annotations[j]) >= 0.12
                        )
                    ),
                    start,
                )
                for j in range(seed_start, min(end, seed_start + max_tail_frames)):
                    if annotations[j].get("primary_event") == EventType.U_E5.value:
                        continue
                    _rewrite_invading_u5(j, "event_invading_turn_cone_occupation_seeded_u5")

        sticky_gap_by_scenario = {
            "InvadingTurn": (EventType.U_E5, 5),
            "OppositeVehicleRunningRedLight": (EventType.U_E6, 5),
        }
        sticky_config = sticky_gap_by_scenario.get(scenario_name)
        if sticky_config is not None:
            sticky_event, max_gap = sticky_config
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != sticky_event.value:
                    idx += 1
                    continue
                left_end = idx + 1
                while left_end < len(annotations) and annotations[left_end].get("primary_event") == sticky_event.value:
                    left_end += 1
                gap_end = left_end
                while gap_end < len(annotations) and annotations[gap_end].get("primary_event") != sticky_event.value:
                    gap_end += 1
                if (
                    gap_end < len(annotations)
                    and 0 < gap_end - left_end <= max_gap
                    and all(str(annotations[j].get("primary_event", "")).startswith("R-E") for j in range(left_end, gap_end))
                ):
                    for ann in annotations[left_end:gap_end]:
                        old = ann.get("primary_event")
                        regular = EventType.R_E5 if sticky_event == EventType.U_E5 else EventType.R_E4
                        events = {sticky_event}
                        if ann.get("primary_road_structure") == RoadStructure.R4.value and regular == EventType.R_E4:
                            events.add(EventType.R_E4)
                        if ann.get("primary_road_structure") == RoadStructure.R5.value and regular == EventType.R_E5:
                            events.add(EventType.R_E5)
                        self._rewrite_event_label(ann, events, sticky_event, f"event_short_{sticky_event.value.lower()}_gap_merged")
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": sticky_event.value,
                                "reason": f"short_{sticky_event.value}_gap_merged",
                            }
                        )
                    idx = left_end
                    continue
                idx = left_end

        if scenario_name == "OppositeVehicleRunningRedLight":
            u6_spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E6.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E6.value:
                    idx += 1
                u6_spans.append((start, idx))
            if len(u6_spans) > 1:
                def _u6_span_score(span: Tuple[int, int]) -> Tuple[int, int, int, float]:
                    start, end = span
                    closest = min(
                        (
                            _safe_float(
                                _event_metrics(annotations[j]).get("speed_reduced_by_obj_distance"),
                                default=math.inf,
                            )
                            for j in range(start, end)
                        ),
                        default=math.inf,
                    )
                    stopped_wait_frames = 0
                    slow_wait_frames = 0
                    bbox_conflict_frames = 0
                    for j in range(max(0, start - 6), min(len(annotations), end + 6)):
                        metrics = _event_metrics(annotations[j])
                        speed = _safe_float(metrics.get("speed"), default=math.inf)
                        target_speed = _safe_float(metrics.get("target_speed"), default=math.inf)
                        if speed <= 1.5 and target_speed <= 1.5:
                            stopped_wait_frames += 1
                        elif speed <= 3.0 and target_speed <= 2.5:
                            slow_wait_frames += 1
                    for j in range(start, end):
                        metrics = _event_metrics(annotations[j])
                        if int(_safe_float(metrics.get("red_light_conflict_vehicle_count"), default=0.0) or 0) > 0:
                            bbox_conflict_frames += 1
                    return stopped_wait_frames, slow_wait_frames, bbox_conflict_frames, end - start, -closest

                keep_span = max(u6_spans, key=_u6_span_score)
                for start, end in u6_spans:
                    if (start, end) == keep_span:
                        continue
                    for j in range(start, end):
                        ann = annotations[j]
                        old = ann.get("primary_event")
                        replacement = _regular_event_for_annotation(ann)
                        self._rewrite_event_label(
                            ann,
                            {replacement},
                            replacement,
                            "event_opposite_red_light_nonprimary_u6_span_released",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": replacement.value,
                                "reason": "opposite_red_light_nonprimary_u6_span_released",
                            }
                        )

            # Keep the selected red-light violation visible while ego is still stopped
            # by the conflict. A fixed 2-frame tail misses cases where the violating
            # vehicle has not cleared the intersection and ego remains near-zero speed.
            u6_context_pre = 6
            u6_context_post = 32

            def _u6_wait_or_conflict_context(ann: Dict[str, Any], *, before_seed: bool) -> bool:
                metrics = _event_metrics(ann)
                speed = _safe_float(metrics.get("speed"), default=math.inf)
                target_speed = _safe_float(metrics.get("target_speed"), default=math.inf)
                obj_dist = _safe_float(metrics.get("speed_reduced_by_obj_distance"), default=math.inf)
                vehicle_hazard = bool(metrics.get("vehicle_hazard")) or bool((ann.get("event_evidence") or {}).get("vehicle_hazard"))
                hard_decel = bool(metrics.get("hard_decel")) or bool(metrics.get("brake_cutin"))
                if obj_dist <= 14.0 or vehicle_hazard:
                    return True
                if before_seed:
                    return bool(speed <= 3.0 and (target_speed <= 2.5 or obj_dist <= 35.0 or hard_decel))
                return bool(speed <= 1.5 and (target_speed <= 1.5 or obj_dist <= 25.0 or hard_decel))

            u6_spans = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E6.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E6.value:
                    idx += 1
                u6_spans.append((start, idx))
            for start, end in u6_spans:
                ext_start = start
                while (
                    ext_start > 0
                    and start - ext_start <= u6_context_pre
                    and annotations[ext_start - 1].get("primary_road_structure") == RoadStructure.R4.value
                    and annotations[ext_start - 1].get("primary_event") == EventType.R_E4.value
                    and _u6_wait_or_conflict_context(annotations[ext_start - 1], before_seed=True)
                ):
                    ext_start -= 1
                ext_end = end
                while (
                    ext_end < len(annotations)
                    and ext_end - end < u6_context_post
                    and annotations[ext_end].get("primary_road_structure") == RoadStructure.R4.value
                    and annotations[ext_end].get("primary_event") == EventType.R_E4.value
                    and _u6_wait_or_conflict_context(annotations[ext_end], before_seed=False)
                ):
                    ext_end += 1
                for ann in annotations[ext_start:start] + annotations[end:ext_end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(
                        ann,
                        {EventType.R_E4, EventType.U_E6},
                        EventType.U_E6,
                        "event_opposite_red_light_conflict_context_u6",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E6.value,
                            "reason": "opposite_red_light_conflict_context_u6",
                        }
                    )

        min_unusual_frames = 2
        idx = 0
        while idx < len(annotations):
            label = annotations[idx].get("primary_event")
            start = idx
            while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                idx += 1
            end = idx
            if not str(label).startswith("U-E") or end - start >= min_unusual_frames:
                continue
            prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
            next_label = annotations[end].get("primary_event") if end < len(annotations) else None
            replacement = next_label or prev_label
            if replacement and str(replacement).startswith("R-E"):
                replacement_ev = EventType(replacement)
                for ann in annotations[start:end]:
                    self._rewrite_event_label(ann, {replacement_ev}, replacement_ev, "event_short_unusual_span_smoothed")
                changes.append(
                    {
                        "start_frame": annotations[start].get("frame_id"),
                        "end_frame": annotations[end - 1].get("frame_id"),
                        "from": label,
                        "to": replacement,
                        "reason": "short_unusual_span_smoothed",
                    }
                )

        for idx, ann in enumerate(annotations):
            label = ann.get("primary_event")
            if label not in {EventType.U_E2.value, EventType.U_E3.value}:
                continue
            if not _intersection_or_signal_control(ann):
                continue
            metrics = ((ann.get("event_evidence") or {}).get("metrics") or {})
            if scenario_name == "AccidentTwoWays" and bool(metrics.get("accident_twoways_r2_overlay_active")):
                event_evidence = ann.setdefault("event_evidence", {})
                rules = event_evidence.setdefault("rules_fired", [])
                if "event_intersection_release_skipped_for_r2_overlay" not in rules:
                    rules.append("event_intersection_release_skipped_for_r2_overlay")
                continue
            replacement = _release_to_regular(ann, f"event_{label.lower()}_intersection_wait_released")
            if replacement is None:
                continue
            changes.append(
                {
                    "frame_id": ann.get("frame_id"),
                    "from": label,
                    "to": replacement.value,
                    "reason": f"{label.lower()}_intersection_wait_released",
                }
            )

        for unusual_event, reason in (
            (EventType.U_E2, "repeated_u2_suppressed_single_core"),
            (EventType.U_E3, "repeated_u3_suppressed_single_core"),
        ):
            def _single_core_span_score(span: Tuple[int, int]) -> Tuple[float, int, int]:
                start, end = span
                duration = end - start
                score = 0.0
                if unusual_event == EventType.U_E2:
                    specific_distances = [
                        _specific_obstacle_distance(annotations[j])
                        for j in range(start, end)
                    ]
                    finite_specific = [d for d in specific_distances if math.isfinite(d)]
                    has_specific_obstacle = any(_specific_obstacle_close(annotations[j], pad_m=4.0) for j in range(start, end))
                    has_route_change = any(_route_change_hint(annotations[j]) for j in range(start, end))
                    has_twoway_event_core = any(
                        {
                            "event_twoway_core_obstruction",
                            "event_twoway_r2_lane_change_core",
                        }
                        & set((annotations[j].get("event_evidence") or {}).get("rules_fired") or [])
                        for j in range(start, end)
                    )
                    twoway_r2_overlap = (
                        sum(1 for j in range(start, end) if annotations[j].get("primary_road_structure") == RoadStructure.R2.value)
                        if "TwoWays" in scenario_name and scenario_name in R2_RETURN_SCENARIOS
                        else 0
                    )
                    if has_specific_obstacle:
                        score += 10.0
                    if has_twoway_event_core:
                        if "TwoWays" in scenario_name and scenario_name in R2_RETURN_SCENARIOS:
                            score += 20.0 if twoway_r2_overlap else 2.0
                        else:
                            score += 20.0
                    if has_twoway_event_core and twoway_r2_overlap:
                        score += min(18.0, twoway_r2_overlap * 0.75)
                    if finite_specific:
                        min_specific = min(finite_specific)
                        score += max(0.0, 5.0 - min_specific / 10.0)
                    if has_route_change:
                        score += 4.0
                    if any(_obstacle_still_close(annotations[j], pad_m=0.0) for j in range(start, end)):
                        score += 1.0
                    # Moving-lead distance alone can create an early false U-E2; do not let
                    # duration beat the span with concrete static-obstacle evidence.
                    if not has_specific_obstacle and not has_route_change:
                        score -= 3.0
                elif unusual_event == EventType.U_E3:
                    cutin_distances = [
                        _safe_float(_event_metrics(annotations[j]).get("dist_to_cutin_vehicle"), default=math.inf)
                        for j in range(start, end)
                    ]
                    finite_cutin = [d for d in cutin_distances if math.isfinite(d)]
                    has_cutin_response = any(
                        bool(_event_metrics(annotations[j]).get("brake_cutin"))
                        or bool(_event_metrics(annotations[j]).get("vehicle_hazard"))
                        for j in range(start, end)
                    )
                    if finite_cutin:
                        score += max(0.0, 6.0 - min(finite_cutin) / 8.0)
                    if has_cutin_response:
                        score += 4.0
                    if any(
                        "event_dynamic_cutin_or_occupancy" in ((annotations[j].get("event_evidence") or {}).get("rules_fired") or [])
                        for j in range(start, end)
                    ):
                        score += 2.0
                    if not finite_cutin and not has_cutin_response:
                        score -= 3.0
                return (score, min(duration, 20), -start)

            spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != unusual_event.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == unusual_event.value:
                    idx += 1
                spans.append((start, idx))
            if len(spans) <= 1:
                continue
            keep_start, keep_end = max(spans, key=_single_core_span_score)
            for start, end in spans:
                if start == keep_start and end == keep_end:
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    replacement = _release_to_regular(ann, f"event_{reason}")
                    if replacement is None:
                        continue
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": reason,
                            "kept_start_frame": annotations[keep_start].get("frame_id"),
                            "kept_end_frame": annotations[keep_end - 1].get("frame_id"),
                        }
                    )

        idx = 0
        while idx < len(annotations):
            if annotations[idx].get("primary_event") != EventType.R_E1.value:
                idx += 1
                continue
            start = idx
            while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                idx += 1
            end = idx
            if end - start > 4:
                continue
            prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
            next_label = annotations[end].get("primary_event") if end < len(annotations) else None
            if prev_label not in {EventType.U_E2.value, EventType.U_E3.value} or next_label != EventType.R_E2.value:
                continue
            max_bridge = 8 if scenario_name in R2_RETURN_SCENARIOS and prev_label == EventType.U_E2.value else 4
            if end - start > max_bridge:
                continue
            for ann in annotations[start:end]:
                old = ann.get("primary_event")
                self._rewrite_event_label(ann, {EventType.R_E2}, EventType.R_E2, "event_pre_recovery_r1_gap_merged_to_r2")
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E2.value,
                        "reason": "pre_recovery_r1_gap_merged_to_r2",
                        "previous_unusual": prev_label,
                    }
                )

        if scenario_name in R2_RETURN_SCENARIOS:
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                has_recent_u2 = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(max(0, start - 24), start)
                )
                if has_recent_u2:
                    continue
                has_lane_change_evidence = any(_lane_change_re2_supported(j) for j in range(start, end))
                if has_lane_change_evidence:
                    for ann in annotations[start:end]:
                        event_evidence = ann.setdefault("event_evidence", {})
                        reasons = event_evidence.setdefault("rules_fired", [])
                        if "event_independent_route_lane_change_r2_kept" not in reasons:
                            reasons.append("event_independent_route_lane_change_r2_kept")
                    changes.append(
                        {
                            "start_frame": annotations[start].get("frame_id"),
                            "end_frame": annotations[end - 1].get("frame_id"),
                            "from": EventType.R_E2.value,
                            "to": EventType.R_E2.value,
                            "reason": "independent_route_lane_change_r2_kept",
                        }
                    )
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    replacement = _regular_event_for_annotation(ann)
                    self._rewrite_event_label(ann, {replacement}, replacement, "event_obstacle_r2_without_recent_u2_suppressed")
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": "obstacle_r2_without_recent_u2_suppressed",
                        }
                    )

        if "TwoWays" in scenario_name and scenario_name in R2_RETURN_SCENARIOS:
            for idx, ann in enumerate(annotations):
                label = ann.get("primary_event")
                if label not in {EventType.U_E2.value, EventType.U_E3.value}:
                    continue
                if _twoways_current_core_active(ann):
                    continue
                recent_core = any(
                    annotations[j].get("primary_event") in {EventType.U_E2.value, EventType.U_E3.value}
                    and _twoways_current_core_active(annotations[j])
                    for j in range(max(0, idx - 24), idx)
                )
                if recent_core and _lane_change_re2_supported(idx):
                    replacement = EventType.R_E2
                    reason = "event_twoways_post_core_recovery_to_r2"
                else:
                    replacement = _regular_event_for_annotation(ann)
                    reason = "event_twoways_pre_core_unusual_suppressed"
                old = label
                self._rewrite_event_label(
                    ann,
                    {replacement},
                    replacement,
                    reason,
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": replacement.value,
                        "reason": reason.replace("event_", ""),
                    }
                )

            for idx, ann in enumerate(annotations):
                if ann.get("primary_event") != EventType.U_E4.value:
                    continue
                recent_core = any(
                    annotations[j].get("primary_event") in {EventType.U_E2.value, EventType.U_E3.value, EventType.R_E2.value}
                    for j in range(max(0, idx - 16), idx)
                )
                if not recent_core or not _lane_change_re2_supported(idx):
                    continue
                metrics = _event_metrics(ann)
                ped_dist = _safe_float(metrics.get("nearest_ped_bike_m"), default=math.inf)
                has_ped_hazard = "event_walker_or_emergency_brake_hazard" in set(
                    ((ann.get("event_evidence") or {}).get("rules_fired") or [])
                )
                if (math.isfinite(ped_dist) and ped_dist <= 12.0) or has_ped_hazard:
                    continue
                old = ann.get("primary_event")
                self._rewrite_event_label(
                    ann,
                    {EventType.R_E2},
                    EventType.R_E2,
                    "event_twoways_recovery_overrides_weak_u4",
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E2.value,
                        "reason": "twoways_recovery_overrides_weak_u4",
                    }
                )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E1.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                    idx += 1
                end = idx
                if end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label not in {EventType.U_E2.value, EventType.U_E3.value} or next_label != EventType.R_E2.value:
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(
                        ann,
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_twoways_post_core_short_r1_gap_to_r2",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "twoways_post_core_short_r1_gap_to_r2",
                        }
                    )

        for ann in annotations:
            rs_label = ann.get("primary_road_structure")
            event_label = ann.get("primary_event")
            regular_for_rs = _regular_event_for_annotation(ann)
            event_rules = set(str(rule) for rule in ((ann.get("event_evidence") or {}).get("rules_fired") or []))
            invading_r1_u5_tail = (
                scenario_name == "InvadingTurn"
                and event_label == EventType.U_E5.value
                and rs_label == RoadStructure.R1.value
                and (
                    "event_invading_turn_cone_occupation_tail_u5" in event_rules
                    or "event_invading_turn_cone_occupation_seeded_u5" in event_rules
                )
            )
            if (
                event_label in {EventType.R_E4.value, EventType.R_E5.value}
                and event_label != regular_for_rs.value
            ) or (
                event_label == EventType.R_E1.value
                and rs_label in {RoadStructure.R4.value, RoadStructure.R5.value}
            ) or (
                event_label == EventType.U_E5.value
                and rs_label not in {RoadStructure.R2.value, RoadStructure.R4.value, RoadStructure.R5.value}
                and not invading_r1_u5_tail
            ):
                old = event_label
                self._rewrite_event_label(
                    ann,
                    {regular_for_rs},
                    regular_for_rs,
                    "event_regular_resynced_after_rs_smoothing",
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": regular_for_rs.value,
                        "reason": "regular_resynced_after_rs_smoothing",
                    }
                )
                event_label = regular_for_rs.value
            if rs_label == RoadStructure.R4.value and event_label in {EventType.U_E2.value, EventType.U_E3.value}:
                metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
                overlay = ((ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {})
                if (
                    overlay.get("active")
                    or (scenario_name == "AccidentTwoWays" and bool(metrics.get("accident_twoways_r2_overlay_active")))
                ):
                    event_evidence = ann.setdefault("event_evidence", {})
                    rules = event_evidence.setdefault("rules_fired", [])
                    if "event_r4_r2_overlay_keeps_obstacle_priority" not in rules:
                        rules.append("event_r4_r2_overlay_keeps_obstacle_priority")
                    continue
                old = event_label
                self._rewrite_event_label(ann, {EventType.R_E4}, EventType.R_E4, "event_forced_regular_by_r4_candidate_pool")
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E4.value,
                        "reason": "r4_candidate_pool_excludes_u2_u3",
                    }
                )
            elif rs_label == RoadStructure.R5.value and event_label in {EventType.U_E2.value, EventType.U_E3.value}:
                metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
                overlay = ((ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {})
                if (
                    overlay.get("active")
                    or (scenario_name == "AccidentTwoWays" and bool(metrics.get("accident_twoways_r2_overlay_active")))
                ):
                    event_evidence = ann.setdefault("event_evidence", {})
                    rules = event_evidence.setdefault("rules_fired", [])
                    if "event_r5_r2_overlay_keeps_obstacle_priority" not in rules:
                        rules.append("event_r5_r2_overlay_keeps_obstacle_priority")
                    continue
                old = event_label
                self._rewrite_event_label(ann, {EventType.R_E5}, EventType.R_E5, "event_forced_regular_by_r5_candidate_pool")
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E5.value,
                        "reason": "r5_candidate_pool_excludes_u2_u3",
                    }
                )
        if scenario_name == "InvadingTurn":
            final_u5_rules = {
                "passive_oncoming_invasion",
                "r2_passive_invading_turn",
                "r2_opposite_lane_confirmed",
                "r2_scenario_trigger_medium",
            }

            def _final_invading_support(index: int) -> bool:
                ann = annotations[index]
                rs = ann.get("primary_road_structure")
                if rs not in {RoadStructure.R1.value, RoadStructure.R2.value}:
                    return False
                rules = set(str(rule) for rule in ((ann.get("evidence") or {}).get("rules_fired") or []))
                if "passive_oncoming_invasion" not in rules:
                    return False
                if not (rules & (final_u5_rules - {"passive_oncoming_invasion"})):
                    return False
                trigger_distance = _trigger_distance(ann)
                if not math.isfinite(trigger_distance) or trigger_distance < 25.0:
                    return False
                metrics = _event_metrics(ann)
                route_abs = _route_lateral_abs(ann)
                still_responding = (
                    bool(metrics.get("changed_route"))
                    or bool(metrics.get("hard_decel"))
                    or bool(metrics.get("vehicle_hazard"))
                    or (math.isfinite(route_abs) and route_abs >= 0.03)
                )
                return rs == RoadStructure.R2.value or still_responding

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") == EventType.U_E5.value or not _final_invading_support(idx):
                    idx += 1
                    continue
                start = idx
                while (
                    idx < len(annotations)
                    and annotations[idx].get("primary_event") != EventType.U_E5.value
                    and _final_invading_support(idx)
                ):
                    idx += 1
                end = idx
                if end - start < 8:
                    continue
                has_r2 = any(annotations[j].get("primary_road_structure") == RoadStructure.R2.value for j in range(start, end))
                if not has_r2 and end - start < 12:
                    continue
                max_final_cluster_frames = 48 if has_r2 else 40
                limited_end = min(end, start + max_final_cluster_frames)
                for ann in annotations[start:limited_end]:
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.U_E5}
                    if ann.get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.U_E5,
                        "event_invading_turn_final_cone_occupation_u5",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E5.value,
                            "reason": "invading_turn_final_cone_occupation_u5",
                        }
                    )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E5.value:
                    idx += 1
                    continue
                left_end = idx + 1
                while left_end < len(annotations) and annotations[left_end].get("primary_event") == EventType.U_E5.value:
                    left_end += 1
                gap_end = left_end
                while gap_end < len(annotations) and annotations[gap_end].get("primary_event") != EventType.U_E5.value:
                    gap_end += 1
                if (
                    gap_end < len(annotations)
                    and 0 < gap_end - left_end <= 5
                    and all(str(annotations[j].get("primary_event", "")).startswith("R-E") for j in range(left_end, gap_end))
                ):
                    for ann in annotations[left_end:gap_end]:
                        old = ann.get("primary_event")
                        self._rewrite_event_label(
                            ann,
                            {EventType.U_E5},
                            EventType.U_E5,
                            "event_invading_turn_final_short_u5_gap_merged",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": EventType.U_E5.value,
                                "reason": "invading_turn_final_short_u5_gap_merged",
                            }
                        )
                    idx = left_end
                    continue
                idx = left_end

        if scenario_name == "ParkedObstacleTwoWays":
            max_short_junction_gap = 8
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") not in {EventType.R_E4.value, EventType.R_E5.value}:
                    idx += 1
                    continue
                start = idx
                while (
                    idx < len(annotations)
                    and annotations[idx].get("primary_event") in {EventType.R_E4.value, EventType.R_E5.value}
                ):
                    idx += 1
                end = idx
                if end - start > max_short_junction_gap:
                    continue
                prev_event = annotations[start - 1].get("primary_event") if start > 0 else None
                next_event = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_event != EventType.R_E2.value or next_event != EventType.R_E2.value:
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    old_rs = ann.get("primary_road_structure")
                    if old_rs != RoadStructure.R2.value:
                        self._rewrite_rs_label(
                            ann,
                            RoadStructure.R2.value,
                            "rs_short_junction_gap_between_twoways_re2_merged",
                            "route_level_event_twoways_gap_merge",
                        )
                    self._rewrite_event_label(
                        ann,
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_short_junction_gap_between_twoways_re2_merged",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "rs_from": old_rs,
                            "rs_to": RoadStructure.R2.value,
                            "reason": "short_junction_gap_between_twoways_re2_merged",
                        }
                    )
        if {EventType.R_E4, EventType.R_E5} & set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [])):
            max_junction_gap = int(
                SCENARIO_RULE_CONFIG.get(scenario_name, {}).get(
                    "junction_regular_gap_merge_max_frames",
                    12,
                )
            )
            min_junction_neighbor = int(
                SCENARIO_RULE_CONFIG.get(scenario_name, {}).get(
                    "junction_regular_gap_min_neighbor_frames",
                    1,
                )
            )

            def _same_junction_cluster_len(anchor_start: int, anchor_end: int, label: str, direction: int) -> int:
                total = anchor_end - anchor_start
                if direction < 0:
                    pos = anchor_start - 1
                    while pos >= 0:
                        gap_end = pos + 1
                        while pos >= 0 and annotations[pos].get("primary_event") in {
                            EventType.R_E1.value,
                            EventType.R_E2.value,
                        }:
                            pos -= 1
                        gap_len = gap_end - (pos + 1)
                        if gap_len <= 0 or gap_len > max_junction_gap:
                            break
                        seg_end = pos + 1
                        while pos >= 0 and annotations[pos].get("primary_event") == label:
                            pos -= 1
                        seg_len = seg_end - (pos + 1)
                        if seg_len <= 0:
                            break
                        total += seg_len
                    return total

                pos = anchor_end
                while pos < len(annotations):
                    gap_start = pos
                    while (
                        pos < len(annotations)
                        and annotations[pos].get("primary_event") in {EventType.R_E1.value, EventType.R_E2.value}
                    ):
                        pos += 1
                    gap_len = pos - gap_start
                    if gap_len <= 0 or gap_len > max_junction_gap:
                        break
                    seg_start = pos
                    while pos < len(annotations) and annotations[pos].get("primary_event") == label:
                        pos += 1
                    seg_len = pos - seg_start
                    if seg_len <= 0:
                        break
                    total += seg_len
                return total

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") not in {EventType.R_E1.value, EventType.R_E2.value}:
                    idx += 1
                    continue
                start = idx
                while (
                    idx < len(annotations)
                    and annotations[idx].get("primary_event") in {EventType.R_E1.value, EventType.R_E2.value}
                ):
                    idx += 1
                end = idx
                if end - start > max_junction_gap:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != next_label or prev_label not in {EventType.R_E4.value, EventType.R_E5.value}:
                    continue
                prev_start = start - 1
                while prev_start > 0 and annotations[prev_start - 1].get("primary_event") == prev_label:
                    prev_start -= 1
                next_end = end
                while next_end < len(annotations) and annotations[next_end].get("primary_event") == next_label:
                    next_end += 1
                prev_cluster_len = _same_junction_cluster_len(prev_start, start, prev_label, -1)
                next_cluster_len = _same_junction_cluster_len(end, next_end, next_label, 1)
                if prev_cluster_len < min_junction_neighbor or next_cluster_len < min_junction_neighbor:
                    continue
                fill_event = EventType(prev_label)
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    old_rs = ann.get("primary_road_structure")
                    fill_rs = RoadStructure.R4 if fill_event == EventType.R_E4 else RoadStructure.R5
                    if old_rs != fill_rs.value:
                        self._rewrite_rs_label(
                            ann,
                            fill_rs.value,
                            "rs_short_regular_gap_between_same_junction_event_merged",
                            "route_level_event_junction_gap_merge",
                        )
                    self._rewrite_event_label(
                        ann,
                        {fill_event},
                        fill_event,
                        "event_short_regular_gap_between_same_junction_event_merged",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": fill_event.value,
                            "rs_from": old_rs,
                            "rs_to": fill_rs.value,
                            "reason": "short_regular_gap_between_same_junction_event_merged",
                        }
                    )
        if scenario_name == "AccidentTwoWays":
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E1.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                    idx += 1
                end = idx
                if end - start > 16:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != EventType.U_E2.value or next_label != EventType.R_E2.value:
                    continue
                if any(
                    annotations[j].get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}
                    for j in range(start, end)
                ):
                    continue
                split = None
                for j in range(start, end):
                    if (
                        _lane_change_re2_supported(j)
                        or _return_lane_change_hint(annotations[j])
                        or _route_centering_trend(j)
                    ):
                        split = j
                        break
                if split is None:
                    split = start + max(1, (end - start) // 2)
                split = min(max(split, start + 1), end)
                for j in range(start, end):
                    ann = annotations[j]
                    old = ann.get("primary_event")
                    if j < split and (
                        _twoways_current_core_active(ann)
                        or _specific_obstacle_close(ann, pad_m=5.0)
                        or not _lane_change_re2_supported(j)
                    ):
                        replacement = EventType.U_E2
                        reason = "event_accidenttwoways_final_r1_gap_to_u2"
                    else:
                        replacement = EventType.R_E2
                        reason = "event_accidenttwoways_final_r1_gap_to_r2"
                    self._rewrite_event_label(ann, {replacement}, replacement, reason)
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": reason.replace("event_", ""),
                            "gap_start_frame": annotations[start].get("frame_id"),
                            "gap_end_frame": annotations[end - 1].get("frame_id"),
                            "split_frame": annotations[split].get("frame_id") if split < len(annotations) else None,
                        }
                    )

        if "TwoWays" in scenario_name and scenario_name in R2_RETURN_SCENARIOS:
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E1.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E1.value:
                    idx += 1
                end = idx
                if end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label not in {EventType.U_E2.value, EventType.U_E3.value} or next_label != EventType.R_E2.value:
                    continue
                if any(
                    annotations[j].get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}
                    for j in range(start, end)
                ):
                    continue
                for ann in annotations[start:end]:
                    old = ann.get("primary_event")
                    self._rewrite_event_label(
                        ann,
                        {EventType.R_E2},
                        EventType.R_E2,
                        "event_final_twoways_short_r1_gap_to_r2",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "final_twoways_short_r1_gap_to_r2",
                        }
                    )

        if scenario_name in CROSSING_U4_SINGLE_SPAN_SCENARIOS:
            field, threshold = PEDESTRIAN_BICYCLE_EVENT_FIELDS[scenario_name]
            support_pad_m = _crossing_u4_support_pad_m(scenario_name)
            max_internal_gap_frames = _crossing_u4_max_internal_gap_frames(scenario_name)

            def _u4_support(index: int) -> Tuple[bool, float]:
                ann = annotations[index]
                metrics = _event_metrics(ann)
                rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
                primary = ann.get("primary_event")
                dist = _finite_min(
                    metrics.get(field),
                    metrics.get("nearest_ped_bike_m"),
                    metrics.get("dist_to_pedestrian"),
                    metrics.get("dist_to_biker"),
                )
                has_crossing_rule = any(str(rule).startswith("event_crossing_distance") for rule in rules)
                has_hazard = "event_walker_or_emergency_brake_hazard" in rules
                if primary == EventType.U_E4.value:
                    return True, dist if math.isfinite(dist) else threshold
                if has_crossing_rule or has_hazard:
                    return True, dist if math.isfinite(dist) else threshold
                if math.isfinite(dist) and dist <= threshold + support_pad_m:
                    return True, dist
                return False, dist

            support = []
            for idx_support in range(len(annotations)):
                ok, dist = _u4_support(idx_support)
                support.append((ok, dist))

            raw_spans = []
            idx = 0
            while idx < len(annotations):
                while idx < len(annotations) and not support[idx][0]:
                    idx += 1
                if idx >= len(annotations):
                    break
                start = idx
                last_support = idx
                gap = 0
                idx += 1
                while idx < len(annotations):
                    if support[idx][0]:
                        last_support = idx
                        gap = 0
                    else:
                        gap += 1
                        if gap > max_internal_gap_frames:
                            break
                    idx += 1
                raw_spans.append((start, last_support + 1))

            if raw_spans:
                def _span_score(span: Tuple[int, int]) -> Tuple[int, float, int]:
                    start, end = span
                    support_count = sum(1 for j in range(start, end) if support[j][0])
                    closest = min(
                        (support[j][1] for j in range(start, end) if math.isfinite(support[j][1])),
                        default=threshold + support_pad_m,
                    )
                    return support_count, -closest, end - start

                main_start, main_end = max(raw_spans, key=_span_score)
                for idx_ann, ann in enumerate(annotations):
                    inside = main_start <= idx_ann < main_end
                    label = ann.get("primary_event")
                    if inside:
                        if label == EventType.U_E4.value:
                            continue
                        regular = _regular_event_for_annotation(ann)
                        events = {EventType.U_E4}
                        if regular in {EventType.R_E4, EventType.R_E5}:
                            events.add(regular)
                        old = label
                        self._rewrite_event_label(
                            ann,
                            events,
                            EventType.U_E4,
                            "event_crossing_u4_single_span_gap_merged",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": EventType.U_E4.value,
                                "reason": "crossing_u4_single_span_gap_merged",
                            }
                        )
                    elif label == EventType.U_E4.value:
                        replacement = _regular_event_for_annotation(ann)
                        old = label
                        self._rewrite_event_label(
                            ann,
                            {replacement},
                            replacement,
                            "event_crossing_u4_outside_main_span_released",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": replacement.value,
                                "reason": "crossing_u4_outside_main_span_released",
                            }
                        )

        if scenario_name in {"Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane"}:
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                has_recent_u2 = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(max(0, start - 12), start)
                )
                if not has_recent_u2:
                    continue
                recent_u2_start = start
                left_budget = 0
                while (
                    recent_u2_start > 0
                    and left_budget < 64
                    and annotations[recent_u2_start - 1].get("primary_event") in {EventType.U_E2.value, EventType.R_E2.value}
                ):
                    recent_u2_start -= 1
                    left_budget += 1
                recent_closest_idx = None
                recent_closest_dist = math.inf
                for recent_idx in range(recent_u2_start, start):
                    recent_dist = _specific_obstacle_distance(annotations[recent_idx])
                    if math.isfinite(recent_dist) and recent_dist < recent_closest_dist:
                        recent_closest_dist = recent_dist
                        recent_closest_idx = recent_idx
                recent_lateral_peak_idx = None
                recent_lateral_points = [
                    (j, _route_lateral_abs(annotations[j]))
                    for j in range(recent_u2_start, end)
                    if math.isfinite(_route_lateral_abs(annotations[j]))
                ]
                if recent_lateral_points:
                    recent_lateral_peak_idx = max(recent_lateral_points, key=lambda item: item[1])[0]
                for j in range(start, end):
                    ann = annotations[j]
                    if ann.get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}:
                        continue
                    dist = _specific_obstacle_distance(ann)
                    prev_dist = _specific_obstacle_distance(annotations[j - 1]) if j > 0 else math.inf
                    obstacle_has_cleared_core = (
                        math.isfinite(recent_closest_dist)
                        and dist >= recent_closest_dist + STATIC_U2_RE2_CLEAR_DELTA_M
                    )
                    rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
                    recovery_after_lateral_peak = (
                        recent_lateral_peak_idx is not None
                        and j >= recent_lateral_peak_idx + RECOVERY_AFTER_LATERAL_PEAK_FRAMES
                        and (recent_closest_idx is None or j >= recent_closest_idx)
                        and _lateral_recovery_started(j, recent_u2_start, end)
                        and (
                            _route_centered_for_re2_exit(ann)
                            or _route_centering_trend(j)
                            or _return_lane_change_hint(ann)
                        )
                    )
                    post_closest_prepare_re2 = (
                        recent_closest_idx is not None
                        and j > recent_closest_idx
                        and math.isfinite(recent_closest_dist)
                        and math.isfinite(dist)
                        and dist > recent_closest_dist + 0.05
                        and _signed_lane_change_active(ann, limit_m=1.20)
                        and _route_centered_for_re2_exit(ann)
                    ) or (
                        "event_u2_return_lane_change_to_r2" in rules
                        and math.isfinite(dist)
                        and math.isfinite(prev_dist)
                        and dist > prev_dist + 0.05
                        and _signed_lane_change_active(ann, limit_m=1.20)
                    )
                    recovery_supported_after_core = (
                        obstacle_has_cleared_core
                        or recovery_after_lateral_peak
                        or post_closest_prepare_re2
                    ) and (
                        _return_lane_change_hint(ann)
                        or _lane_change_re2_supported(j)
                        or _route_centered_for_re2_exit(ann)
                        or _lateral_recovery_started(j, recent_u2_start, end)
                        or post_closest_prepare_re2
                    )
                    if (
                        not math.isfinite(dist)
                        or dist > 21.5
                        or recovery_supported_after_core
                    ):
                        continue
                    old = ann.get("primary_event")
                    self._rewrite_event_label(
                        ann,
                        {EventType.U_E2},
                        EventType.U_E2,
                        "event_recovery_delayed_until_static_obstacle_clear",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "recovery_delayed_until_static_obstacle_clear",
                        }
                    )

        idx = 0
        while idx < len(annotations):
            if annotations[idx].get("primary_event") != EventType.R_E2.value:
                idx += 1
                continue
            start = idx
            while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                idx += 1
            end = idx
            if end - start > 3 or start <= 0 or end >= len(annotations):
                continue
            prev_label = annotations[start - 1].get("primary_event")
            next_label = annotations[end].get("primary_event")
            if prev_label != next_label or prev_label not in {EventType.U_E2.value, EventType.U_E3.value}:
                continue
            support_window = range(max(0, start - 2), min(len(annotations), end + 2))
            if prev_label == EventType.U_E2.value:
                supported = any(
                    _specific_obstacle_core_or_approaching(j)
                    or _specific_obstacle_close(annotations[j], pad_m=4.0)
                    or _obstacle_still_close(annotations[j], pad_m=4.0)
                    for j in support_window
                )
            else:
                supported = any(
                    _cutin_distance(annotations[j]) <= CUTIN_U3_ACTIVE_DISTANCE_M or _cutin_response_active(annotations[j])
                    for j in support_window
                )
            if not supported:
                continue
            for ann in annotations[start:end]:
                old = ann.get("primary_event")
                self._rewrite_event_label(
                    ann,
                    {EventType(prev_label)},
                    EventType(prev_label),
                    "event_short_r2_between_same_unusual_merged",
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": prev_label,
                        "reason": "short_r2_between_same_unusual_merged",
                    }
                )

        def _overlay_recovery_supported(index: int) -> bool:
            ann = annotations[index]
            if _return_lane_change_hint(ann):
                return True
            if _route_change_hint(ann) and (not _route_centered(ann) or _signed_lane_change_active(ann)):
                return True
            if _route_centering_trend(index):
                prev_window = range(max(0, index - 5), index + 1)
                return any(_route_change_hint(annotations[j]) or _signed_lane_change_active(annotations[j]) for j in prev_window)
            return False

        def _overlay_u2_still_active(index: int) -> bool:
            ann = annotations[index]
            if _specific_obstacle_core_or_approaching(index):
                return True
            if _obstacle_still_close(ann, pad_m=4.0) or _specific_obstacle_close(ann, pad_m=6.0):
                return True
            metrics = _event_metrics(ann)
            hard_response = bool(metrics.get("hard_decel")) or bool(metrics.get("vehicle_hazard"))
            if hard_response and _safe_float(metrics.get("speed_reduced_by_obj_distance"), default=math.inf) <= 40.0:
                return True
            if (_route_change_hint(ann) or _signed_lane_change_active(ann)) and not _route_centered(ann) and not _return_lane_change_hint(ann):
                return True
            return False

        def _overlay_u1_still_active(index: int) -> bool:
            ann = annotations[index]
            metrics = _event_metrics(ann)
            speed_obj = _safe_float(metrics.get("speed_reduced_by_obj_distance"), default=math.inf)
            return bool(metrics.get("hard_decel")) or (
                bool(metrics.get("vehicle_hazard"))
                and math.isfinite(speed_obj)
                and speed_obj <= 25.0
            )

        def _overlay_u3_still_active(index: int) -> bool:
            ann = annotations[index]
            if scenario_name == "ParkingCutIn":
                if _cutin_response_active(ann):
                    return True
                return _route_change_hint(ann) and not _route_centered(ann)
            return _cutin_distance(ann) <= CUTIN_U3_ACTIVE_DISTANCE_M or _cutin_response_active(ann)

        def _overlay_u4_still_active(index: int, age_frames: int) -> bool:
            ann = annotations[index]
            metrics = _event_metrics(ann)
            ped_dist = _safe_float(metrics.get("nearest_ped_bike_m"), default=math.inf)
            has_ped_hazard = "event_walker_or_emergency_brake_hazard" in set(
                ((ann.get("event_evidence") or {}).get("rules_fired") or [])
            )
            if (math.isfinite(ped_dist) and ped_dist <= 12.0) or has_ped_hazard:
                return True

            if age_frames > 10:
                return False

            if math.isfinite(ped_dist) and ped_dist <= 28.0:
                return True

            # Some U-E4 routes expose the crossing/turning conflict actor as a
            # generic vehicle/object distance after the junction RS takes over.
            dynamic_obj_dist = _safe_float(metrics.get("speed_reduced_by_obj_distance"), default=math.inf)
            return (
                scenario_name
                in {
                    "DynamicObjectCrossing",
                    "CrossingBicycleFlow",
                    "ParkingCrossingPedestrian",
                    "PedestrianCrossing",
                    "VehicleTurningRoute",
                    "VehicleTurningRoutePedestrian",
                }
                and math.isfinite(dynamic_obj_dist)
                and dynamic_obj_dist <= 35.0
            )

        def _choose_interrupted_overlay_event(
            index: int,
            source_event: EventType,
            recovery_age: int,
            age_frames: int,
        ) -> Optional[Tuple[EventType, str]]:
            scenario_allowed = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, []))
            if source_event == EventType.U_E1 and _overlay_u1_still_active(index):
                return EventType.U_E1, "unusual_still_active"
            if source_event == EventType.U_E2:
                if _specific_obstacle_core_or_approaching(index):
                    return EventType.U_E2, "unusual_still_active"
                if (
                    EventType.R_E2 in scenario_allowed
                    and recovery_age < INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES
                    and _overlay_recovery_supported(index)
                ):
                    return EventType.R_E2, "recovery_to_target_lane"
                if _overlay_u2_still_active(index):
                    return EventType.U_E2, "unusual_still_active"
                return None
            if source_event == EventType.U_E3:
                if _overlay_u3_still_active(index):
                    return EventType.U_E3, "unusual_still_active"
                if (
                    EventType.R_E2 in scenario_allowed
                    and recovery_age < INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES
                    and _overlay_recovery_supported(index)
                ):
                    return EventType.R_E2, "recovery_to_target_lane"
                return None
            if source_event == EventType.U_E4:
                if _overlay_u4_still_active(index, age_frames):
                    return EventType.U_E4, "unusual_still_active"
                if (
                    EventType.R_E2 in scenario_allowed
                    and recovery_age < INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES
                    and _overlay_recovery_supported(index)
                ):
                    return EventType.R_E2, "recovery_to_target_lane"
                return None
            return None

        def _set_interrupted_overlay(
            ann: Dict[str, Any],
            overlay_event: EventType,
            source_event: EventType,
            source_rs: str,
            age_frames: int,
            recovery_age_frames: int,
            phase: str,
        ) -> None:
            regular = _regular_event_for_annotation(ann)
            events = {regular, overlay_event}
            ann["events"] = [ev.value for ev in sorted(events, key=lambda ev: ev.value)]
            ann["primary_event"] = overlay_event.value
            event_evidence = ann.setdefault("event_evidence", {})
            event_evidence["events"] = ann["events"]
            event_evidence["primary_event"] = overlay_event.value
            event_evidence["regular_event"] = regular.value
            if overlay_event.value.startswith("U-E"):
                event_evidence["unusual_event"] = overlay_event.value
                event_evidence.pop("overlay_recovery_event", None)
            else:
                event_evidence["unusual_event"] = None
                event_evidence["overlay_recovery_event"] = overlay_event.value
            event_evidence["interrupted_event_overlay"] = {
                "active": True,
                "base_road_structure": source_rs,
                "intersection_road_structure": ann.get("primary_road_structure"),
                "regular_event": regular.value,
                "overlay_event": overlay_event.value,
                "source_unusual_event": source_event.value,
                "age_frames": age_frames,
                "age_seconds": round(age_frames * 0.25, 3),
                "max_frames": INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES,
                "recovery_age_frames": recovery_age_frames,
                "recovery_max_frames": INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES,
                "phase": phase,
                "reason": "unusual_event_interrupted_by_intersection_rs",
            }
            self._attach_overlay_base_rs(ann, source_rs)
            rules = event_evidence.setdefault("rules_fired", [])
            if "event_interrupted_unusual_overlay" not in rules:
                rules.append("event_interrupted_unusual_overlay")

        for idx, ann in enumerate(annotations):
            rs_label = ann.get("primary_road_structure")
            if rs_label not in {RoadStructure.R4.value, RoadStructure.R5.value} or idx <= 0:
                continue
            prev = annotations[idx - 1]
            prev_overlay = (prev.get("event_evidence") or {}).get("interrupted_event_overlay") or {}
            if prev_overlay.get("active"):
                source_label = prev_overlay.get("source_unusual_event")
                source_event = EventType(source_label) if source_label in EventType._value2member_map_ else None
                source_rs = str(prev_overlay.get("base_road_structure") or RoadStructure.R1.value)
                age_frames = int(prev_overlay.get("age_frames") or 0) + 1
                recovery_age = int(prev_overlay.get("recovery_age_frames") or 0)
            else:
                prev_event_label = prev.get("primary_event")
                prev_rs = prev.get("primary_road_structure")
                source_event = None
                source_rs = str(prev_rs or RoadStructure.R1.value)
                seed_age_frames = 1
                seed_recovery_age = 0
                if (
                    prev_event_label in EventType._value2member_map_
                    and EventType(prev_event_label) in INTERRUPTED_UNUSUAL_OVERLAY_EVENTS
                    and (
                        prev_rs not in {RoadStructure.R4.value, RoadStructure.R5.value}
                        or (
                            scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}
                            and prev_event_label == EventType.U_E4.value
                        )
                    )
                ):
                    source_event = EventType(prev_event_label)
                elif (
                    EventType.R_E2 in set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, []))
                    and prev_event_label == EventType.R_E2.value
                    and prev_rs not in {RoadStructure.R4.value, RoadStructure.R5.value}
                ):
                    recent_sources = [
                        EventType(annotations[j].get("primary_event"))
                        for j in range(max(0, idx - 16), idx)
                        if annotations[j].get("primary_event")
                        in {EventType.U_E2.value, EventType.U_E3.value, EventType.U_E4.value}
                    ]
                    if recent_sources:
                        candidate_source = recent_sources[-1]
                        if (
                            (
                                candidate_source == EventType.U_E2
                                and (_overlay_u2_still_active(idx) or _overlay_recovery_supported(idx))
                            )
                            or (
                                candidate_source == EventType.U_E3
                                and (_overlay_u3_still_active(idx) or _overlay_recovery_supported(idx))
                            )
                            or (
                                candidate_source == EventType.U_E4
                                and (
                                    _overlay_u4_still_active(idx, 1)
                                    or _overlay_recovery_supported(idx)
                                )
                            )
                            ):
                                source_event = candidate_source
                elif EventType.R_E2 in set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [])):
                    # R4/R5 may arrive one or two frames before the recovery
                    # evidence becomes visible. Keep a short grace window so an
                    # interrupted U-E2/U-E3 can still surface as overlay R-E2
                    # once the return maneuver starts.
                    recent_unusual = [
                        j
                        for j in range(max(0, idx - 8), idx)
                        if annotations[j].get("primary_event")
                        in {EventType.U_E2.value, EventType.U_E3.value, EventType.U_E4.value}
                        and (
                            annotations[j].get("primary_road_structure")
                            not in {RoadStructure.R4.value, RoadStructure.R5.value}
                            or (
                                scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}
                                and annotations[j].get("primary_event") == EventType.U_E4.value
                            )
                        )
                    ]
                    if recent_unusual:
                        source_idx = recent_unusual[-1]
                        candidate_source = EventType(annotations[source_idx].get("primary_event"))
                        source_rs = str(annotations[source_idx].get("primary_road_structure") or source_rs)
                        if (
                            (
                                candidate_source == EventType.U_E2
                                and (_overlay_u2_still_active(idx) or _overlay_recovery_supported(idx))
                            )
                            or (
                                candidate_source == EventType.U_E3
                                and (_overlay_u3_still_active(idx) or _overlay_recovery_supported(idx))
                            )
                            or (
                                candidate_source == EventType.U_E4
                                and (
                                    _overlay_u4_still_active(idx, max(1, idx - source_idx))
                                    or _overlay_recovery_supported(idx)
                                )
                            )
                        ):
                            source_event = candidate_source
                            seed_age_frames = max(1, idx - source_idx)
                            seed_recovery_age = 0
                if source_event is None:
                    continue
                age_frames = seed_age_frames
                recovery_age = seed_recovery_age
            if source_event is None or age_frames > INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES:
                continue
            choice = _choose_interrupted_overlay_event(idx, source_event, recovery_age, age_frames)
            if choice is None:
                continue
            overlay_event, phase = choice
            recovery_age = recovery_age + 1 if overlay_event == EventType.R_E2 else 0
            old = ann.get("primary_event")
            _set_interrupted_overlay(
                ann,
                overlay_event,
                source_event,
                source_rs,
                age_frames,
                recovery_age,
                phase,
            )
            changes.append(
                {
                    "frame_id": ann.get("frame_id"),
                    "from": old,
                    "to": overlay_event.value,
                    "reason": "interrupted_unusual_overlay",
                    "primary_rs": rs_label,
                    "base_rs": source_rs,
                    "regular_event": _regular_event_for_annotation(ann).value,
                    "source_unusual_event": source_event.value,
                    "phase": phase,
                    "age_frames": age_frames,
                }
            )

        if EventType.R_E2 in set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [])):
            for idx, ann in enumerate(annotations):
                rs_label = ann.get("primary_road_structure")
                if rs_label not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                    continue
                overlay = ((ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {})
                if overlay.get("active"):
                    continue
                if not _overlay_recovery_supported(idx):
                    continue
                recent_unusual = [
                    j
                    for j in range(max(0, idx - 8), idx)
                    if annotations[j].get("primary_event")
                    in {EventType.U_E2.value, EventType.U_E3.value, EventType.U_E4.value}
                    and (
                        annotations[j].get("primary_road_structure")
                        not in {RoadStructure.R4.value, RoadStructure.R5.value}
                        or (
                            scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}
                            and annotations[j].get("primary_event") == EventType.U_E4.value
                        )
                    )
                ]
                if not recent_unusual:
                    continue
                source_idx = recent_unusual[-1]
                source_event = EventType(annotations[source_idx].get("primary_event"))
                source_rs = str(annotations[source_idx].get("primary_road_structure") or RoadStructure.R1.value)
                if source_event == EventType.U_E2:
                    supported = _overlay_u2_still_active(idx) or _overlay_recovery_supported(idx)
                elif source_event == EventType.U_E3:
                    supported = _overlay_u3_still_active(idx) or _overlay_recovery_supported(idx)
                else:
                    supported = (
                        _overlay_u4_still_active(idx, max(1, idx - source_idx))
                        or _overlay_recovery_supported(idx)
                    )
                if not supported:
                    continue
                age_frames = max(1, idx - source_idx)
                if age_frames > INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES:
                    continue
                old = ann.get("primary_event")
                _set_interrupted_overlay(
                    ann,
                    EventType.R_E2,
                    source_event,
                    source_rs,
                    age_frames,
                    1,
                    "recovery_to_target_lane_grace",
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.R_E2.value,
                        "reason": "interrupted_unusual_overlay_recovery_grace",
                        "primary_rs": rs_label,
                        "base_rs": source_rs,
                        "regular_event": _regular_event_for_annotation(ann).value,
                        "source_unusual_event": source_event.value,
                        "phase": "recovery_to_target_lane_grace",
                        "age_frames": age_frames,
                    }
                )

        if scenario_name in {"Accident", "ConstructionObstacle", "ParkedObstacle", "HazardAtSideLane"}:
            def _static_u2_r2_cluster_bounds(r2_start: int, r2_end: int) -> Tuple[int, int]:
                cluster_start = r2_start
                left_budget = 0
                while cluster_start > 0 and left_budget < 56:
                    prev_label = annotations[cluster_start - 1].get("primary_event")
                    if prev_label not in {EventType.U_E2.value, EventType.R_E2.value}:
                        break
                    cluster_start -= 1
                    left_budget += 1
                cluster_end = r2_end
                right_budget = 0
                while cluster_end < len(annotations) and right_budget < 56:
                    next_label = annotations[cluster_end].get("primary_event")
                    if next_label not in {EventType.U_E2.value, EventType.R_E2.value}:
                        break
                    cluster_end += 1
                    right_budget += 1
                return cluster_start, cluster_end

            def _cluster_closest_index(cluster_start: int, cluster_end: int) -> Tuple[Optional[int], float]:
                closest_index = None
                closest_distance = math.inf
                for cluster_idx in range(cluster_start, cluster_end):
                    dist = _specific_obstacle_distance(annotations[cluster_idx])
                    if math.isfinite(dist) and dist < closest_distance:
                        closest_distance = dist
                        closest_index = cluster_idx
                return closest_index, closest_distance

            def _cluster_lateral_peak_index(cluster_start: int, cluster_end: int) -> Optional[int]:
                finite_points = [
                    (cluster_idx, _route_lateral_abs(annotations[cluster_idx]))
                    for cluster_idx in range(cluster_start, cluster_end)
                    if math.isfinite(_route_lateral_abs(annotations[cluster_idx]))
                ]
                if not finite_points:
                    return None
                return max(finite_points, key=lambda item: item[1])[0]

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                cluster_start, cluster_end = _static_u2_r2_cluster_bounds(start, end)
                if not any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    for j in range(cluster_start, cluster_end)
                ):
                    continue
                closest_idx, closest_dist = _cluster_closest_index(cluster_start, cluster_end)
                if closest_idx is None or not math.isfinite(closest_dist):
                    continue
                lateral_peak_idx = _cluster_lateral_peak_index(cluster_start, cluster_end)
                for j in range(start, end):
                    ann = annotations[j]
                    dist = _specific_obstacle_distance(ann)
                    if not math.isfinite(dist):
                        continue
                    prev_dist = _specific_obstacle_distance(annotations[j - 1]) if j > 0 else math.inf
                    rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
                    recovery_after_lateral_peak = (
                        lateral_peak_idx is not None
                        and j >= lateral_peak_idx + RECOVERY_AFTER_LATERAL_PEAK_FRAMES
                        and j > closest_idx
                        and _lateral_recovery_started(j, cluster_start, cluster_end)
                        and (
                            _route_centered_for_re2_exit(ann)
                            or _route_centering_trend(j)
                            or _return_lane_change_hint(ann)
                        )
                    )
                    post_closest_prepare_re2 = (
                        j > closest_idx
                        and math.isfinite(closest_dist)
                        and math.isfinite(dist)
                        and dist > closest_dist + 0.05
                        and _signed_lane_change_active(ann, limit_m=1.20)
                        and _route_centered_for_re2_exit(ann)
                    ) or (
                        "event_u2_return_lane_change_to_r2" in rules
                        and math.isfinite(prev_dist)
                        and dist > prev_dist + 0.05
                        and _signed_lane_change_active(ann, limit_m=1.20)
                    )
                    before_or_near_core_exit = (
                        (j <= closest_idx and not recovery_after_lateral_peak and not post_closest_prepare_re2)
                        or (
                            dist < closest_dist + STATIC_U2_RE2_CLEAR_DELTA_M
                            and not recovery_after_lateral_peak
                            and not post_closest_prepare_re2
                        )
                    )
                    if not before_or_near_core_exit:
                        continue
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.U_E2}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.U_E2,
                        "event_re2_inside_unfinished_static_u2_core_merged",
                    )
                    overlay = (ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {}
                    if overlay.get("active"):
                        overlay["overlay_event"] = EventType.U_E2.value
                        overlay["phase"] = "unfinished_static_obstacle_core"
                        event_evidence = ann.setdefault("event_evidence", {})
                        event_evidence["interrupted_event_overlay"] = overlay
                        event_evidence["unusual_event"] = EventType.U_E2.value
                        event_evidence.pop("overlay_recovery_event", None)
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E2.value,
                            "reason": "re2_inside_unfinished_static_u2_core_merged",
                            "cluster_start_frame": annotations[cluster_start].get("frame_id"),
                            "cluster_end_frame": annotations[cluster_end - 1].get("frame_id"),
                            "closest_frame": annotations[closest_idx].get("frame_id"),
                            "clear_delta_m": round(dist - closest_dist, 3),
                        }
                    )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                u2_start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                if idx >= len(annotations) or annotations[idx].get("primary_event") != EventType.R_E2.value:
                    continue
                r2_start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                if idx >= len(annotations) or annotations[idx].get("primary_event") != EventType.U_E2.value:
                    continue
                tail_start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                tail_end = idx
                cluster_start, cluster_end = u2_start, tail_end
                closest_idx, closest_dist = _cluster_closest_index(cluster_start, cluster_end)
                if closest_idx is None or not math.isfinite(closest_dist):
                    continue
                tail_distances = [
                    _specific_obstacle_distance(annotations[j])
                    for j in range(tail_start, tail_end)
                    if math.isfinite(_specific_obstacle_distance(annotations[j]))
                ]
                tail_min_dist = min(tail_distances) if tail_distances else math.inf
                tail_len = tail_end - tail_start
                if (
                    tail_len <= 2
                    and math.isfinite(tail_min_dist)
                    and tail_min_dist >= closest_dist + STATIC_U2_RE2_CLEAR_DELTA_M
                ):
                    rewrite_range = range(tail_start, tail_end)
                    replacement = EventType.R_E2
                    reason = "event_short_clear_u2_tail_absorbed_into_re2"
                    change_reason = "short_clear_u2_tail_absorbed_into_re2"
                else:
                    rewrite_range = range(r2_start, tail_start)
                    replacement = EventType.U_E2
                    reason = "event_re2_before_u2_tail_merged_to_unfinished_core"
                    change_reason = "re2_before_u2_tail_merged_to_unfinished_core"
                for j in rewrite_range:
                    ann = annotations[j]
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {replacement}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(ann, events, replacement, reason)
                    overlay = (ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {}
                    if overlay.get("active"):
                        overlay["overlay_event"] = replacement.value
                        overlay["phase"] = (
                            "recovery_to_target_lane"
                            if replacement == EventType.R_E2
                            else "unfinished_static_obstacle_core"
                        )
                        event_evidence = ann.setdefault("event_evidence", {})
                        event_evidence["interrupted_event_overlay"] = overlay
                        if replacement == EventType.U_E2:
                            event_evidence["unusual_event"] = EventType.U_E2.value
                            event_evidence.pop("overlay_recovery_event", None)
                        else:
                            event_evidence["unusual_event"] = None
                            event_evidence["overlay_recovery_event"] = EventType.R_E2.value
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": replacement.value,
                            "reason": change_reason,
                            "u2_start_frame": annotations[u2_start].get("frame_id"),
                            "r2_start_frame": annotations[r2_start].get("frame_id"),
                            "tail_start_frame": annotations[tail_start].get("frame_id"),
                            "closest_frame": annotations[closest_idx].get("frame_id"),
                            "tail_clear_delta_m": round(tail_min_dist - closest_dist, 3)
                            if math.isfinite(tail_min_dist)
                            else None,
                        }
                    )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E2.value:
                    idx += 1
                end = idx
                if end - start > 2 or start <= 0 or end >= len(annotations):
                    continue
                prev_label = annotations[start - 1].get("primary_event")
                next_label = annotations[end].get("primary_event")
                if prev_label != EventType.R_E2.value or next_label != EventType.R_E2.value:
                    continue
                cluster_start, cluster_end = _static_u2_r2_cluster_bounds(start - 1, end + 1)
                closest_idx, closest_dist = _cluster_closest_index(cluster_start, cluster_end)
                if closest_idx is None or start <= closest_idx:
                    continue
                lateral_peak_idx = _cluster_lateral_peak_index(cluster_start, cluster_end)
                if lateral_peak_idx is None or start <= lateral_peak_idx:
                    continue
                supported = False
                support_range = range(max(cluster_start, start - 3), min(cluster_end, end + 4))
                for j in support_range:
                    ann = annotations[j]
                    if (
                        _lateral_recovery_started(j, cluster_start, cluster_end)
                        or _lane_change_re2_supported(j)
                        or _return_lane_change_hint(ann)
                        or _route_centering_trend(j)
                    ):
                        supported = True
                        break
                if not supported:
                    continue
                for j in range(start, end):
                    ann = annotations[j]
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.R_E2}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.R_E2,
                        "event_short_u2_inside_static_recovery_absorbed",
                    )
                    overlay = (ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {}
                    if overlay.get("active"):
                        overlay["overlay_event"] = EventType.R_E2.value
                        overlay["phase"] = "recovery_to_target_lane"
                        event_evidence = ann.setdefault("event_evidence", {})
                        event_evidence["interrupted_event_overlay"] = overlay
                        event_evidence["unusual_event"] = None
                        event_evidence["overlay_recovery_event"] = EventType.R_E2.value
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "short_u2_inside_static_recovery_absorbed",
                            "cluster_start_frame": annotations[cluster_start].get("frame_id"),
                            "cluster_end_frame": annotations[cluster_end - 1].get("frame_id"),
                            "lateral_peak_frame": annotations[lateral_peak_idx].get("frame_id"),
                        }
                    )
        if scenario_name == "ParkingExit":
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                new_end = end
                while new_end < len(annotations) and new_end - end < 4:
                    ann = annotations[new_end]
                    if ann.get("primary_event") not in {
                        EventType.R_E1.value,
                        EventType.R_E4.value,
                        EventType.R_E5.value,
                    }:
                        break
                    if not (
                        _signed_lane_change_active(ann, limit_m=1.35)
                        or _route_change_hint(ann)
                        or not _route_centered(ann)
                    ):
                        break
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.R_E2}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.R_E2,
                        "event_parking_exit_re2_extended_until_lane_change_complete",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "parking_exit_re2_extended_until_lane_change_complete",
                        }
                    )
                    new_end += 1

            initial_trim_frames = 5
            if annotations and annotations[0].get("primary_event") == EventType.R_E2.value:
                end = 0
                while end < len(annotations) and annotations[end].get("primary_event") == EventType.R_E2.value:
                    end += 1
                if end > initial_trim_frames + 4:
                    trim_start = end - initial_trim_frames
                    for ann in annotations[trim_start:end]:
                        old = ann.get("primary_event")
                        regular = _regular_event_for_annotation(ann)
                        self._rewrite_event_label(
                            ann,
                            {regular},
                            regular,
                            "event_parking_exit_initial_re2_end_pulled_early_by_rgb",
                        )
                        changes.append(
                            {
                                "frame_id": ann.get("frame_id"),
                                "from": old,
                                "to": regular.value,
                                "reason": "parking_exit_initial_re2_end_pulled_early_by_rgb",
                            }
                        )

        if scenario_name == "PedestrianCrossing":
            max_tail = int(SCENARIO_RULE_CONFIG.get(scenario_name, {}).get("pedestrian_exit_tail_frames", 6))

            def _pedestrian_tail_supported(ann: Dict[str, Any], age: int) -> bool:
                evidence = ann.get("evidence") or {}
                event_evidence = ann.get("event_evidence") or {}
                metrics = event_evidence.get("metrics") or {}
                trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
                tl_state = str(evidence.get("traffic_light_state") or "")
                has_light = tl_state in {"Red", "Yellow", "Green"} or bool(evidence.get("light_hazard"))
                bbox = evidence.get("bbox_semantics") or {}
                has_visual_control = bool(bbox.get("traffic_light")) or bool(bbox.get("junction_hint"))
                has_crossing_event = ann.get("primary_event") == EventType.U_E4.value
                ped_dist = _safe_float(metrics.get("nearest_ped_bike_m"), default=math.inf)
                return (
                    age <= max_tail
                    and (
                        has_crossing_event
                        or (math.isfinite(ped_dist) and ped_dist <= 24.0)
                        or (has_light and trigger_distance <= 36.0)
                        or (has_visual_control and trigger_distance <= 30.0)
                        or bool(evidence.get("meta_is_junction"))
                    )
                )

            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") not in {EventType.R_E4.value, EventType.R_E5.value}:
                    idx += 1
                    continue
                fill_event = EventType(annotations[idx].get("primary_event"))
                fill_rs = RoadStructure.R4 if fill_event == EventType.R_E4 else RoadStructure.R5
                while idx < len(annotations) and annotations[idx].get("primary_event") == fill_event.value:
                    idx += 1
                tail = idx
                age = 1
                while tail < len(annotations) and age <= max_tail:
                    ann = annotations[tail]
                    if ann.get("primary_event") not in {EventType.R_E1.value, EventType.R_E2.value, EventType.U_E4.value}:
                        break
                    if not _pedestrian_tail_supported(ann, age):
                        break
                    old_event = ann.get("primary_event")
                    old_rs = ann.get("primary_road_structure")
                    if old_rs != fill_rs.value:
                        self._rewrite_rs_label(
                            ann,
                            fill_rs.value,
                            "rs_pedestrian_crossing_exit_tail_extended_by_rgb",
                            "route_level_pedestrian_crossing_exit_tail",
                        )
                    if old_event == EventType.U_E4.value:
                        self._rewrite_event_label(
                            ann,
                            {fill_event, EventType.U_E4},
                            EventType.U_E4,
                            "event_pedestrian_crossing_exit_tail_keeps_junction_regular",
                        )
                    else:
                        self._rewrite_event_label(
                            ann,
                            {fill_event},
                            fill_event,
                            "event_pedestrian_crossing_exit_tail_extended_by_rgb",
                        )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old_event,
                            "to": ann.get("primary_event"),
                            "rs_from": old_rs,
                            "rs_to": fill_rs.value,
                            "reason": "pedestrian_crossing_exit_tail_extended_by_rgb",
                        }
                    )
                    tail += 1
                    age += 1

            short_gap_max = 8
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") not in {EventType.R_E1.value, EventType.R_E2.value}:
                    idx += 1
                    continue
                start = idx
                while (
                    idx < len(annotations)
                    and annotations[idx].get("primary_event") in {EventType.R_E1.value, EventType.R_E2.value}
                ):
                    idx += 1
                end = idx
                if end - start > short_gap_max:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
                if prev_label != next_label or prev_label not in {EventType.R_E4.value, EventType.R_E5.value}:
                    continue
                prev_start = start - 1
                while prev_start > 0 and annotations[prev_start - 1].get("primary_event") == prev_label:
                    prev_start -= 1
                next_end = end
                while next_end < len(annotations) and annotations[next_end].get("primary_event") == next_label:
                    next_end += 1
                if (start - prev_start) + (next_end - end) < 8:
                    continue
                fill_event = EventType(prev_label)
                fill_rs = RoadStructure.R4 if fill_event == EventType.R_E4 else RoadStructure.R5
                for ann in annotations[start:end]:
                    old_event = ann.get("primary_event")
                    old_rs = ann.get("primary_road_structure")
                    if old_rs != fill_rs.value:
                        self._rewrite_rs_label(
                            ann,
                            fill_rs.value,
                            "rs_pedestrian_crossing_short_regular_gap_merged",
                            "route_level_pedestrian_crossing_gap_merge",
                        )
                    self._rewrite_event_label(
                        ann,
                        {fill_event},
                        fill_event,
                        "event_pedestrian_crossing_short_regular_gap_merged",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old_event,
                            "to": fill_event.value,
                            "rs_from": old_rs,
                            "rs_to": fill_rs.value,
                            "reason": "pedestrian_crossing_short_regular_gap_merged",
                        }
                    )

        if scenario_name == "ParkingCutIn":
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.U_E3.value:
                    idx += 1
                    continue
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.U_E3.value:
                    idx += 1
                end = idx
                new_end = end
                while new_end < len(annotations) and new_end - end < 6:
                    ann = annotations[new_end]
                    if ann.get("primary_event") not in {
                        EventType.R_E1.value,
                        EventType.R_E4.value,
                        EventType.R_E5.value,
                    }:
                        break
                    if not (
                        _cutin_response_active(ann)
                        or (_route_change_hint(ann) and not _route_centered(ann))
                    ):
                        break
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.U_E3}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.U_E3,
                        "event_parking_cutin_u3_tail_extended_by_rgb_trajectory",
                    )
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.U_E3.value,
                            "reason": "parking_cutin_u3_tail_extended_by_rgb_trajectory",
                        }
                    )
                    new_end += 1

        if scenario_name in {"VehicleOpensDoorTwoWays", "ParkedObstacle", "ParkedObstacleTwoWays"}:
            recovery_spans: List[Tuple[int, int]] = []
            idx = 0
            while idx < len(annotations):
                if annotations[idx].get("primary_event") != EventType.R_E2.value:
                    idx += 1
                    continue
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == EventType.R_E2.value:
                    idx += 1
                end = idx
                recent_u2 = any(
                    annotations[j].get("primary_event") == EventType.U_E2.value
                    or (
                        ((annotations[j].get("event_evidence") or {}).get("interrupted_event_overlay") or {}).get(
                            "source_unusual_event"
                        )
                        == EventType.U_E2.value
                    )
                    for j in range(max(0, start - 8), start)
                )
                if recent_u2:
                    recovery_spans.append((start, end))

            for start, end in recovery_spans:
                new_start = start
                while (
                    new_start > 0
                    and start - new_start < 3
                    and annotations[new_start - 1].get("primary_event") == EventType.U_E2.value
                ):
                    new_start -= 1
                for j in range(new_start, start):
                    ann = annotations[j]
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    events = {EventType.R_E2}
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        events.add(regular)
                    self._rewrite_event_label(
                        ann,
                        events,
                        EventType.R_E2,
                        "event_static_obstacle_re2_start_advanced_3f",
                    )
                    event_evidence = ann.setdefault("event_evidence", {})
                    if regular in {EventType.R_E4, EventType.R_E5}:
                        event_evidence["overlay_recovery_event"] = EventType.R_E2.value
                        event_evidence["interrupted_event_overlay"] = {
                            "active": True,
                            "base_road_structure": RoadStructure.R2.value,
                            "intersection_road_structure": ann.get("primary_road_structure"),
                            "regular_event": regular.value,
                            "overlay_event": EventType.R_E2.value,
                            "source_unusual_event": EventType.U_E2.value,
                            "age_frames": max(1, start - j),
                            "age_seconds": round(max(1, start - j) * 0.25, 3),
                            "max_frames": INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES,
                            "recovery_age_frames": 1,
                            "recovery_max_frames": INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES,
                            "phase": "vehicle_door_recovery_start_advanced",
                            "reason": "unusual_event_interrupted_by_intersection_rs",
                        }
                        self._attach_overlay_base_rs(
                            ann,
                            RoadStructure.R2.value,
                            "vehicle_door_recovery_overlay_base_rs",
                        )
                    event_evidence["unusual_event"] = None
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": EventType.R_E2.value,
                            "reason": "static_obstacle_re2_start_advanced_3f",
                        }
                    )

                release_count = min(4, max(0, end - new_start - 4))
                for j in range(end - release_count, end):
                    ann = annotations[j]
                    if ann.get("primary_event") != EventType.R_E2.value:
                        continue
                    old = ann.get("primary_event")
                    regular = _regular_event_for_annotation(ann)
                    self._rewrite_event_label(
                        ann,
                        {regular},
                        regular,
                        "event_static_obstacle_re2_end_advanced_4f",
                    )
                    event_evidence = ann.setdefault("event_evidence", {})
                    event_evidence["unusual_event"] = None
                    event_evidence.pop("overlay_recovery_event", None)
                    event_evidence.pop("interrupted_event_overlay", None)
                    changes.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": old,
                            "to": regular.value,
                            "reason": "static_obstacle_re2_end_advanced_4f",
                        }
                    )

        for ann in annotations:
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
        return {"enabled": True, "changes": changes}

    def _apply_crossing_u4_single_span_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """横穿类事件每条 route 保留一段连续 U-E4，避免距离/RS 抖动切成多段。"""
        if scenario_name not in CROSSING_U4_SINGLE_SPAN_SCENARIOS:
            return {"enabled": False, "changes": []}
        if not annotations:
            return {"enabled": True, "changes": []}

        field, threshold = PEDESTRIAN_BICYCLE_EVENT_FIELDS[scenario_name]
        support_pad_m = _crossing_u4_support_pad_m(scenario_name)
        max_internal_gap_frames = _crossing_u4_max_internal_gap_frames(scenario_name)
        changes: List[Dict[str, Any]] = []

        def _current_regular_event(ann: Dict[str, Any]) -> EventType:
            primary_rs = ann.get("primary_road_structure")
            if primary_rs == RoadStructure.R4.value:
                return EventType.R_E4
            if primary_rs == RoadStructure.R5.value:
                return EventType.R_E5
            return EventType.R_E1

        def _support(index: int) -> Tuple[bool, float]:
            ann = annotations[index]
            metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
            rules = set(((ann.get("event_evidence") or {}).get("rules_fired") or []))
            dist = _finite_min(
                metrics.get(field),
                metrics.get("nearest_ped_bike_m"),
                metrics.get("dist_to_pedestrian"),
                metrics.get("dist_to_biker"),
            )
            has_crossing_rule = any(str(rule).startswith("event_crossing_distance") for rule in rules)
            has_hazard = "event_walker_or_emergency_brake_hazard" in rules
            if ann.get("primary_event") == EventType.U_E4.value:
                return True, dist if math.isfinite(dist) else threshold
            if has_crossing_rule or has_hazard:
                return True, dist if math.isfinite(dist) else threshold
            if math.isfinite(dist) and dist <= threshold + support_pad_m:
                return True, dist
            return False, dist

        support = [_support(i) for i in range(len(annotations))]
        spans: List[Tuple[int, int]] = []
        idx = 0
        while idx < len(annotations):
            while idx < len(annotations) and not support[idx][0]:
                idx += 1
            if idx >= len(annotations):
                break
            start = idx
            last_support = idx
            gap = 0
            idx += 1
            while idx < len(annotations):
                if support[idx][0]:
                    last_support = idx
                    gap = 0
                else:
                    gap += 1
                    if gap > max_internal_gap_frames:
                        break
                idx += 1
            spans.append((start, last_support + 1))

        if not spans:
            return {"enabled": True, "changes": []}

        def _score(span: Tuple[int, int]) -> Tuple[int, float, int]:
            start, end = span
            support_count = sum(1 for j in range(start, end) if support[j][0])
            closest = min(
                (support[j][1] for j in range(start, end) if math.isfinite(support[j][1])),
                default=threshold + support_pad_m,
            )
            return support_count, -closest, end - start

        main_start, main_end = max(spans, key=_score)
        for idx_ann, ann in enumerate(annotations):
            label = ann.get("primary_event")
            inside = main_start <= idx_ann < main_end
            if inside:
                if label == EventType.U_E4.value:
                    continue
                regular = _current_regular_event(ann)
                events = {EventType.U_E4}
                if regular in {EventType.R_E4, EventType.R_E5}:
                    events.add(regular)
                old = label
                self._rewrite_event_label(
                    ann,
                    events,
                    EventType.U_E4,
                    "event_crossing_u4_final_single_span_gap_merged",
                )
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": EventType.U_E4.value,
                        "reason": "crossing_u4_final_single_span_gap_merged",
                    }
                )
            elif label == EventType.U_E4.value:
                replacement = _current_regular_event(ann)
                old = label
                self._rewrite_event_label(
                    ann,
                    {replacement},
                    replacement,
                    "event_crossing_u4_outside_final_main_span_released",
                )
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": replacement.value,
                        "reason": "crossing_u4_outside_final_main_span_released",
                    }
                )

        return {
            "enabled": True,
            "changes": changes,
            "main_span": {
                "start_frame": annotations[main_start].get("frame_id"),
                "end_frame": annotations[main_end - 1].get("frame_id"),
                "length": main_end - main_start,
            },
        }

    def _apply_event_candidate_clamp(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """最终兜底：route 级 event 后处理不能产出 scenario/当前 RS 候选池外的事件。"""
        scenario_allowed = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [EventType.R_E1]))
        allowed_values = {ev.value for ev in scenario_allowed}
        changes: List[Dict[str, Any]] = []

        for ann in annotations:
            overlay = ((ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {})
            if overlay.get("active"):
                self._attach_overlay_base_rs(
                    ann,
                    overlay.get("base_road_structure"),
                    "interrupted_event_overlay_base_rs_final_sync",
                )
            metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
            if (
                scenario_name == "AccidentTwoWays"
                and bool(metrics.get("accident_twoways_r2_overlay_active"))
                and ann.get("primary_road_structure") in {RoadStructure.R4.value, RoadStructure.R5.value}
            ):
                self._attach_overlay_base_rs(
                    ann,
                    RoadStructure.R2.value,
                    "accident_twoways_r2_overlay_base_rs_final_sync",
                )

        def _current_regular_event(ann: Dict[str, Any]) -> EventType:
            return {
                RoadStructure.R4.value: EventType.R_E4,
                RoadStructure.R5.value: EventType.R_E5,
            }.get(str(ann.get("primary_road_structure") or ""), EventType.R_E1)

        def _current_rs_allowed(ann: Dict[str, Any]) -> Set[EventType]:
            primary_rs_value = ann.get("primary_road_structure")
            primary_rs = (
                RoadStructure(primary_rs_value)
                if primary_rs_value in RoadStructure._value2member_map_
                else RoadStructure.R1
            )
            road_allowed = set(ROAD_STRUCTURE_TO_FINE_EVENTS.get(primary_rs, {EventType.R_E1}))
            if scenario_name == "InvadingTurn" and primary_rs in {
                RoadStructure.R1,
                RoadStructure.R2,
                RoadStructure.R4,
                RoadStructure.R5,
            }:
                road_allowed.add(EventType.U_E5)
            if scenario_name == "HazardAtSideLaneTwoWays" and primary_rs == RoadStructure.R2:
                road_allowed.add(EventType.U_E4)
            metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
            if scenario_name == "AccidentTwoWays" and bool(metrics.get("accident_twoways_r2_overlay_active")):
                road_allowed.update({EventType.R_E2, EventType.U_E2})
            if scenario_name == "CrossJunctionDefectTrafficLight" and primary_rs == RoadStructure.R4:
                road_allowed.update({EventType.U_E6, EventType.U_E7})
            event_rules = set(str(rule) for rule in ((ann.get("event_evidence") or {}).get("rules_fired") or []))
            current_event = str(ann.get("primary_event") or "")
            if (
                "event_short_regular_gap_between_same_junction_event_merged" in event_rules
                and current_event in {EventType.R_E4.value, EventType.R_E5.value}
            ):
                road_allowed.add(EventType(current_event))
            if (
                scenario_name == "InterurbanAdvancedActorFlow"
                and primary_rs == RoadStructure.R5
                and bool(metrics.get("interurban_advanced_re2_span_expanded"))
            ):
                road_allowed.add(EventType.R_E2)
            overlay = ((ann.get("event_evidence") or {}).get("interrupted_event_overlay") or {})
            overlay_event = overlay.get("overlay_event") if overlay.get("active") else None
            if overlay_event in EventType._value2member_map_:
                road_allowed.add(EventType(overlay_event))
            # Preserve the current RS regular event, not a stale regular_event
            # left by earlier route-level EVENT rewrites.
            regular_allowed: Set[EventType] = {_current_regular_event(ann)}
            return (scenario_allowed & road_allowed) | regular_allowed or {EventType.R_E1}

        def _fallback_regular(ann: Dict[str, Any], current_allowed: Set[EventType]) -> EventType:
            previous_regular = (ann.get("event_evidence") or {}).get("regular_event")
            current_values = {ev.value for ev in current_allowed}
            if previous_regular in current_values and previous_regular in EventType._value2member_map_:
                return EventType(previous_regular)
            primary_rs = ann.get("primary_road_structure")
            by_rs = {
                RoadStructure.R2.value: EventType.R_E1,
                RoadStructure.R3.value: EventType.R_E1,
                RoadStructure.R4.value: EventType.R_E4,
                RoadStructure.R5.value: EventType.R_E5,
            }.get(primary_rs, EventType.R_E1)
            if by_rs in current_allowed:
                return by_rs
            if EventType.R_E1 in current_allowed:
                return EventType.R_E1
            regular_options = sorted(
                (ev for ev in current_allowed if ev.value.startswith("R-E")),
                key=lambda ev: ev.value,
            )
            return regular_options[0] if regular_options else sorted(current_allowed, key=lambda ev: ev.value)[0]

        def _final_recovery_overlay_supported(ann: Dict[str, Any]) -> bool:
            metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
            signed = _safe_float(metrics.get("signed_dist_to_lane_change"), default=math.inf)
            if math.isfinite(signed) and signed <= -0.45:
                return True
            if bool(metrics.get("target_lane_change_active")):
                return True
            lateral = _safe_float(metrics.get("route_lateral_abs_m"), default=math.inf)
            tolerance = _safe_float(metrics.get("route_center_tolerance_m"), default=0.55)
            return math.isfinite(lateral) and math.isfinite(tolerance) and lateral > max(0.35, tolerance * 0.75)

        def _seed_final_interrupted_overlay(index: int, ann: Dict[str, Any]) -> None:
            if EventType.R_E2 not in scenario_allowed:
                return
            rs_label = ann.get("primary_road_structure")
            if rs_label not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                return
            event_evidence = ann.setdefault("event_evidence", {})
            overlay = event_evidence.get("interrupted_event_overlay") or {}
            if overlay.get("active"):
                return
            if not _final_recovery_overlay_supported(ann):
                return
            recent = [
                j
                for j in range(max(0, index - 8), index)
                if annotations[j].get("primary_event")
                in {EventType.U_E2.value, EventType.U_E3.value, EventType.U_E4.value}
                and (
                    annotations[j].get("primary_road_structure")
                    not in {RoadStructure.R4.value, RoadStructure.R5.value}
                    or (
                        scenario_name in {"HazardAtSideLane", "HazardAtSideLaneTwoWays"}
                        and annotations[j].get("primary_event") == EventType.U_E4.value
                    )
                )
            ]
            if not recent:
                return
            source_idx = recent[-1]
            source_event = annotations[source_idx].get("primary_event")
            source_rs = annotations[source_idx].get("primary_road_structure") or RoadStructure.R1.value
            regular = _current_regular_event(ann)
            ann["events"] = [ev.value for ev in sorted({regular, EventType.R_E2}, key=lambda ev: ev.value)]
            ann["primary_event"] = EventType.R_E2.value
            event_evidence["events"] = ann["events"]
            event_evidence["primary_event"] = EventType.R_E2.value
            event_evidence["regular_event"] = regular.value
            event_evidence["unusual_event"] = None
            event_evidence["overlay_recovery_event"] = EventType.R_E2.value
            event_evidence["interrupted_event_overlay"] = {
                "active": True,
                "base_road_structure": source_rs,
                "intersection_road_structure": rs_label,
                "regular_event": regular.value,
                "overlay_event": EventType.R_E2.value,
                "source_unusual_event": source_event,
                "age_frames": max(1, index - source_idx),
                "age_seconds": round(max(1, index - source_idx) * 0.25, 3),
                "max_frames": INTERRUPTED_UNUSUAL_OVERLAY_TOTAL_MAX_FRAMES,
                "recovery_age_frames": 1,
                "recovery_max_frames": INTERRUPTED_UNUSUAL_OVERLAY_RECOVERY_MAX_FRAMES,
                "phase": "recovery_to_target_lane_final_grace",
                "reason": "unusual_event_interrupted_by_intersection_rs",
            }
            self._attach_overlay_base_rs(ann, source_rs, "final_grace_overlay_base_rs")
            event_evidence.setdefault("rules_fired", []).append("event_interrupted_unusual_overlay_final_grace")

        for index, ann in enumerate(annotations):
            _seed_final_interrupted_overlay(index, ann)
            current_allowed = _current_rs_allowed(ann)
            current_allowed_values = {ev.value for ev in current_allowed}
            primary_label = ann.get("primary_event")
            primary_event = EventType(primary_label) if primary_label in EventType._value2member_map_ else None
            event_values = [
                ev
                for ev in (ann.get("events") or [])
                if ev in EventType._value2member_map_ and EventType(ev) in current_allowed
            ]
            primary_ok = primary_event in current_allowed
            events_ok = sorted(event_values) == sorted(ev for ev in (ann.get("events") or []) if ev in EventType._value2member_map_)
            if primary_ok and events_ok:
                event_evidence = ann.setdefault("event_evidence", {})
                event_evidence["allowed_events"] = [ev.value for ev in sorted(current_allowed, key=lambda ev: ev.value)]
                continue
            fallback = primary_event if primary_ok else _fallback_regular(ann, current_allowed)
            if fallback not in current_allowed:
                fallback = _fallback_regular(ann, current_allowed)
            replacement_events = {EventType(ev) for ev in event_values}
            replacement_events.add(fallback)
            change = {
                "frame_id": ann.get("frame_id"),
                "from": primary_label,
                "to": fallback.value,
                "reason": "event_outside_current_rs_candidate_pool" if primary_event in scenario_allowed else "event_outside_scenario_candidate_pool",
                "primary_rs": ann.get("primary_road_structure"),
                "allowed_events": sorted(current_allowed_values),
            }
            self._rewrite_event_label(
                ann,
                replacement_events,
                fallback,
                "event_candidate_pool_final_clamp",
            )
            event_evidence = ann.setdefault("event_evidence", {})
            event_evidence["allowed_events"] = [ev.value for ev in sorted(current_allowed, key=lambda ev: ev.value)]
            event_evidence.setdefault("candidate_clamp", []).append(change)
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
            changes.append(change)

        return {"enabled": True, "changes": changes}

    def _apply_final_junction_regular_gap_merge(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """最终输出前再次缝合同一 R4/R5 路口段内的短 R-E1/R-E2 噪声。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        has_gap_merge_cfg = "junction_regular_gap_merge_max_frames" in cfg
        if (cfg.get("kind") != "vehicle_turning" and not has_gap_merge_cfg) or not annotations:
            return {"enabled": False, "changes": []}
        max_gap = int(cfg.get("junction_regular_gap_merge_max_frames", cfg.get("turning_final_regular_gap_max_frames", 6)))
        min_neighbor = int(cfg.get("junction_regular_gap_min_neighbor_frames", cfg.get("turning_final_regular_gap_min_neighbor_frames", 4)))
        changes: List[Dict[str, Any]] = []

        def _same_event_run_len(pos: int, label: str, step: int) -> int:
            count = 0
            while 0 <= pos < len(annotations) and annotations[pos].get("primary_event") == label:
                count += 1
                pos += step
            return count

        idx = 0
        while idx < len(annotations):
            if annotations[idx].get("primary_event") not in {EventType.R_E1.value, EventType.R_E2.value}:
                idx += 1
                continue
            start = idx
            while (
                idx < len(annotations)
                and annotations[idx].get("primary_event") in {EventType.R_E1.value, EventType.R_E2.value}
            ):
                idx += 1
            end = idx
            gap_len = end - start
            if gap_len <= 0 or gap_len > max_gap or start == 0 or end >= len(annotations):
                continue
            prev_label = annotations[start - 1].get("primary_event")
            next_label = annotations[end].get("primary_event")
            if prev_label != next_label or prev_label not in {EventType.R_E4.value, EventType.R_E5.value}:
                continue
            if (
                _same_event_run_len(start - 1, prev_label, -1) < min_neighbor
                or _same_event_run_len(end, next_label, 1) < min_neighbor
            ):
                continue
            fill_event = EventType(prev_label)
            fill_rs = RoadStructure.R4 if fill_event == EventType.R_E4 else RoadStructure.R5
            for ann in annotations[start:end]:
                old_event = ann.get("primary_event")
                old_rs = ann.get("primary_road_structure")
                if old_rs != fill_rs.value:
                    self._rewrite_rs_label(
                        ann,
                        fill_rs.value,
                        "rs_final_junction_short_regular_gap_merged",
                        "route_level_final_junction_regular_gap_merge",
                    )
                self._rewrite_event_label(
                    ann,
                    {fill_event},
                    fill_event,
                    "event_final_junction_short_regular_gap_merged",
                )
                ann["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann)
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old_event,
                        "to": fill_event.value,
                        "rs_from": old_rs,
                        "rs_to": fill_rs.value,
                        "reason": "final_junction_short_regular_gap_merged",
                    }
                )
        return {
            "enabled": True,
            "max_gap_frames": max_gap,
            "min_neighbor_frames": min_neighbor,
            "changes": changes,
        }

    def _confidence_stats(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        values = [float(ann.get("confidence", 0.0)) for ann in annotations if ann.get("confidence") is not None]
        if not values:
            return {"min": None, "avg": None, "max": None}
        return {
            "min": round(min(values), 4),
            "avg": round(sum(values) / len(values), 4),
            "max": round(max(values), 4),
        }

    def _min_rs_segment_frames(self, rs_label: str) -> int:
        """所有 RS 共用的最短稳定片段长度；4Hz 下 4 帧约 1 秒。"""
        return {
            "R1": 2,
            "R2": 4,
            "R3": 4,
            "R4": 4,
            "R5": 4,
        }.get(str(rs_label), 4)

    def _rs_runs(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not annotations:
            return runs
        start = 0
        current = annotations[0].get("primary_road_structure")
        for idx, ann in enumerate(annotations[1:], 1):
            label = ann.get("primary_road_structure")
            if label == current:
                continue
            runs.append({"label": current, "start": start, "end": idx, "length": idx - start})
            start = idx
            current = label
        runs.append({"label": current, "start": start, "end": len(annotations), "length": len(annotations) - start})
        return runs

    def _rewrite_rs_label(
        self,
        ann: Dict[str, Any],
        new_label: str,
        reason: str,
        inherited_from: Optional[str],
    ) -> None:
        old_label = ann.get("primary_road_structure")
        if old_label == new_label:
            return
        candidates = ann.setdefault("road_structure_candidates", {})
        candidates[new_label] = max(float(candidates.get(new_label, 0.0) or 0.0), 0.74)
        ann["primary_road_structure"] = new_label
        ann["confidence"] = float(candidates.get(new_label, 0.74) or 0.74)
        secondary = set(ann.get("secondary_road_structures", []) or [])
        secondary.discard(new_label)
        if old_label:
            secondary.add(str(old_label))
        ann["secondary_road_structures"] = sorted(secondary)
        evidence = ann.setdefault("evidence", {})
        smoothing = evidence.setdefault("temporal_smoothing", [])
        smoothing.append(
            {
                "from": old_label,
                "to": new_label,
                "reason": reason,
                "inherited_from": inherited_from,
            }
        )
        rules = evidence.setdefault("rules_fired", [])
        rules.append(f"temporal_smoothing_{reason}")
        review_reasons = evidence.setdefault("review_reasons", [])
        if "temporal_smoothing_applied" not in review_reasons:
            review_reasons.append("temporal_smoothing_applied")
        evidence["review_required"] = True
        ann["reason"] = f"{ann.get('reason', '')}; temporal_smoothing {old_label}->{new_label}"
        ann["annotation_comment"] = _frame_annotation_comment(
            RoadStructure(new_label),
            {RoadStructure(x) for x in ann.get("secondary_road_structures", []) if x in RoadStructure._value2member_map_},
            float(ann.get("confidence", 0.0) or 0.0),
            evidence,
        )
        ann["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann)

    def _apply_r4_context_recovery(self, scenario_name: str, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """把连续稳定的真路口灯控段从过度保守的 R1 恢复为 R4。"""
        changes: List[Dict[str, Any]] = []
        if len(annotations) < 4:
            return {"enabled": True, "changes": changes}

        def _is_demoted_r4_candidate(ann: Dict[str, Any]) -> bool:
            if ann.get("primary_road_structure") != RoadStructure.R1.value:
                return False
            rules = (ann.get("evidence") or {}).get("rules_fired", []) or []
            return any(
                rule in rules
                for rule in (
                    "r4_meta_tl_without_control_context_demoted_to_r1",
                    "r4_bbox_tl_without_control_context_demoted_to_r1",
                )
            )

        def _has_light_evidence(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            tl = str(evidence.get("traffic_light_state", "")).strip().lower()
            bbox_tl = bool((evidence.get("bbox_semantics") or {}).get("traffic_light"))
            return tl not in {"", "none", "null", "nan"} or bbox_tl

        def _has_meta_light_evidence(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            tl = str(evidence.get("traffic_light_state", "")).strip().lower()
            return tl not in {"", "none", "null", "nan"}

        def _xodr_topology_untrusted(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            trusted = (evidence.get("xodr") or {}).get("xodr_topology_trusted")
            if trusted is None:
                trusted = (
                    ((evidence.get("diagnostic_attribution") or {}).get("used_inputs") or {}).get(
                        "xodr_topology_trusted"
                    )
                )
            return trusted is False

        def _has_local_intersection_context(ann: Dict[str, Any]) -> bool:
            flags = (((ann.get("evidence") or {}).get("diagnostic_attribution") or {}).get("window_flags") or {})
            return bool(
                flags.get("strong_control_context")
                or flags.get("close_trigger_for_junction")
                or flags.get("bbox_junction_hint")
            )

        def _has_dynamic_crossing_strict_context(ann: Dict[str, Any]) -> bool:
            flags = (((ann.get("evidence") or {}).get("diagnostic_attribution") or {}).get("window_flags") or {})
            return bool(
                flags.get("bbox_junction_hint")
                or (
                    flags.get("close_trigger_for_junction")
                    and _has_meta_light_evidence(ann)
                )
            )

        def _blocked_by_stop_yield_without_light(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            tl = str(evidence.get("traffic_light_state", "")).strip().lower()
            has_meta_tl = tl not in {"", "none", "null", "nan"}
            return bool(flags.get("bbox_stop_or_yield") or flags.get("stop_hazard")) and not has_meta_tl

        idx = 0
        while idx < len(annotations):
            if not _is_demoted_r4_candidate(annotations[idx]):
                idx += 1
                continue
            start = idx
            while idx < len(annotations) and _is_demoted_r4_candidate(annotations[idx]):
                idx += 1
            end = idx
            segment = annotations[start:end]
            length = end - start
            light_count = sum(1 for ann in segment if _has_light_evidence(ann))
            meta_light_count = sum(1 for ann in segment if _has_meta_light_evidence(ann))
            context_count = sum(1 for ann in segment if _has_local_intersection_context(ann))
            xodr_untrusted_count = sum(1 for ann in segment if _xodr_topology_untrusted(ann))
            stop_yield_without_light = sum(1 for ann in segment if _blocked_by_stop_yield_without_light(ann))
            stable_context_recovery = length >= 4 and light_count >= 4 and context_count >= 2
            stable_meta_light_recovery = (
                length >= 6
                and meta_light_count >= 6
                and xodr_untrusted_count >= max(4, length // 2)
            )
            if scenario_name == "DynamicObjectCrossing":
                strict_context_count = sum(1 for ann in segment if _has_dynamic_crossing_strict_context(ann))
                stable_context_recovery = (
                    length >= 6
                    and light_count >= 6
                    and meta_light_count >= max(3, length // 4)
                    and context_count >= max(4, length // 3)
                    and strict_context_count >= max(4, length // 3)
                )
                stable_meta_light_recovery = (
                    length >= 24
                    and light_count >= 24
                    and meta_light_count >= 12
                )
            if (
                not (stable_context_recovery or stable_meta_light_recovery)
                or stop_yield_without_light >= max(2, length // 2)
            ):
                continue
            recovery_reason = (
                "stable_meta_light_with_untrusted_xodr"
                if stable_meta_light_recovery and not stable_context_recovery
                else "stable_light_plus_intersection_context"
            )
            recovery_rule = (
                "r4_context_recovery_stable_meta_light_untrusted_xodr"
                if recovery_reason == "stable_meta_light_with_untrusted_xodr"
                else "r4_context_recovery_stable_light_plus_intersection_context"
            )
            recovery_segment = segment
            if recovery_reason == "stable_meta_light_with_untrusted_xodr" and scenario_name != "DynamicObjectCrossing":
                meta_offsets = [offset for offset, ann in enumerate(segment) if _has_meta_light_evidence(ann)]
                if not meta_offsets:
                    continue
                rel_start = max(0, meta_offsets[0] - 2)
                rel_end = min(len(segment), meta_offsets[-1] + 3)
                recovery_segment = segment[rel_start:rel_end]
            elif recovery_reason == "stable_meta_light_with_untrusted_xodr" and scenario_name == "DynamicObjectCrossing":
                meta_offsets = [offset for offset, ann in enumerate(segment) if _has_meta_light_evidence(ann)]
                if not meta_offsets:
                    continue
                entry_offsets = []
                for offset, ann in enumerate(segment):
                    flags = (((ann.get("evidence") or {}).get("diagnostic_attribution") or {}).get("window_flags") or {})
                    if flags.get("close_trigger_for_junction") or flags.get("bbox_junction_hint"):
                        entry_offsets.append(offset)
                if entry_offsets:
                    rel_start = max(0, entry_offsets[0])
                else:
                    rel_start = min(len(segment) - 1, meta_offsets[0] + 14)
                recovery_segment = segment[rel_start:]
            for ann in recovery_segment:
                recovery_note = {
                    "from": RoadStructure.R1.value,
                    "to": RoadStructure.R4.value,
                    "reason": recovery_reason,
                    "segment_start_frame": recovery_segment[0].get("frame_id"),
                    "segment_end_frame": recovery_segment[-1].get("frame_id"),
                }
                self._rewrite_rs_label(
                    ann,
                    RoadStructure.R4.value,
                    "r4_stable_light_context_recovered",
                    inherited_from=RoadStructure.R1.value,
                )
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("r4_context_recovery", []).append(recovery_note)
                rules = evidence.setdefault("rules_fired", [])
                if recovery_rule not in rules:
                    rules.append(recovery_rule)
                review_reasons = evidence.setdefault("review_reasons", [])
                if "r4_context_recovery_applied" not in review_reasons:
                    review_reasons.append("r4_context_recovery_applied")
                self._rewrite_event_label(
                    ann,
                    {EventType.R_E4},
                    EventType.R_E4,
                    "event_recomputed_after_r4_context_recovery",
                )
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
            changes.append(
                {
                    "start_frame": recovery_segment[0].get("frame_id"),
                    "end_frame": recovery_segment[-1].get("frame_id"),
                    "from": RoadStructure.R1.value,
                    "to": RoadStructure.R4.value,
                    "length": len(recovery_segment),
                    "source_segment_length": length,
                    "light_count": light_count,
                    "meta_light_count": meta_light_count,
                    "context_count": context_count,
                    "xodr_untrusted_count": xodr_untrusted_count,
                    "reason": recovery_reason,
                }
            )
        if scenario_name == "DynamicObjectCrossing":
            runs = self._rs_runs(annotations)
            for run in runs:
                if run.get("label") != RoadStructure.R4.value:
                    continue
                start = int(run["start"])
                end = int(run["end"])
                if end - start < 6 or start <= 0:
                    continue
                r4_segment = annotations[start:end]
                if sum(1 for ann in r4_segment if _has_meta_light_evidence(ann)) < 3:
                    continue
                tail: List[Dict[str, Any]] = []
                idx_back = start - 1
                while idx_back >= 0 and len(tail) < 4:
                    ann = annotations[idx_back]
                    if ann.get("primary_road_structure") != RoadStructure.R1.value:
                        break
                    if not _is_demoted_r4_candidate(ann):
                        break
                    flags = (((ann.get("evidence") or {}).get("diagnostic_attribution") or {}).get("window_flags") or {})
                    bbox_tl = bool(((ann.get("evidence") or {}).get("bbox_semantics") or {}).get("traffic_light"))
                    if not (bbox_tl and flags.get("close_trigger_for_junction")):
                        break
                    tail.append(ann)
                    idx_back -= 1
                if not tail:
                    continue
                recovery_segment = list(reversed(tail))
                for ann in recovery_segment:
                    recovery_note = {
                        "from": RoadStructure.R1.value,
                        "to": RoadStructure.R4.value,
                        "reason": "dynamic_crossing_pre_r4_close_light_tail",
                        "segment_start_frame": recovery_segment[0].get("frame_id"),
                        "segment_end_frame": recovery_segment[-1].get("frame_id"),
                    }
                    self._rewrite_rs_label(
                        ann,
                        RoadStructure.R4.value,
                        "r4_dynamic_crossing_pre_r4_tail_recovered",
                        inherited_from=RoadStructure.R1.value,
                    )
                    evidence = ann.setdefault("evidence", {})
                    evidence.setdefault("r4_context_recovery", []).append(recovery_note)
                    rules = evidence.setdefault("rules_fired", [])
                    recovery_rule = "r4_dynamic_crossing_pre_r4_tail_recovered"
                    if recovery_rule not in rules:
                        rules.append(recovery_rule)
                    review_reasons = evidence.setdefault("review_reasons", [])
                    if "r4_context_recovery_applied" not in review_reasons:
                        review_reasons.append("r4_context_recovery_applied")
                    self._rewrite_event_label(
                        ann,
                        {EventType.R_E4},
                        EventType.R_E4,
                        "event_recomputed_after_r4_context_recovery",
                    )
                    ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "start_frame": recovery_segment[0].get("frame_id"),
                        "end_frame": recovery_segment[-1].get("frame_id"),
                        "from": RoadStructure.R1.value,
                        "to": RoadStructure.R4.value,
                        "length": len(recovery_segment),
                        "source_segment_length": end - start,
                        "light_count": sum(1 for ann in recovery_segment if _has_light_evidence(ann)),
                        "meta_light_count": sum(1 for ann in recovery_segment if _has_meta_light_evidence(ann)),
                        "context_count": len(recovery_segment),
                        "xodr_untrusted_count": sum(1 for ann in recovery_segment if _xodr_topology_untrusted(ann)),
                        "reason": "dynamic_crossing_pre_r4_close_light_tail",
                    }
                )
        return {"enabled": True, "changes": changes}

    def _apply_blocked_signalized_tail_recovery(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """BlockedIntersection 灯控路口内灯态丢失时，不把出口尾段误判为 R5。"""
        changes: List[Dict[str, Any]] = []
        if scenario_name != "BlockedIntersection" or len(annotations) < 4:
            return {"enabled": True, "changes": changes}

        def _has_signal_evidence(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            tl = str(evidence.get("traffic_light_state", "")).strip().lower()
            bbox_tl = bool((evidence.get("bbox_semantics") or {}).get("traffic_light"))
            return tl in {"red", "yellow", "green"} or bbox_tl

        def _has_stop_or_yield_evidence(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            bbox = evidence.get("bbox_semantics") or {}
            return bool(evidence.get("meta_stop_hazard")) or bool(evidence.get("combined_stop_hazard")) or bool(
                bbox.get("stop_sign") or bbox.get("yield_sign")
            )

        runs = self._rs_runs(annotations)
        for run in runs:
            if run.get("label") != RoadStructure.R5.value:
                continue
            start = int(run["start"])
            end = int(run["end"])
            if start <= 0:
                continue
            rules_in_run = [
                rule
                for ann in annotations[start:end]
                for rule in ((ann.get("evidence") or {}).get("rules_fired") or [])
            ]
            if "blocked_intersection_stop_or_nolight_r5" not in rules_in_run:
                continue
            if any(_has_stop_or_yield_evidence(ann) for ann in annotations[start:end]):
                continue
            prior_window = annotations[max(0, start - 24):start]
            prior_signalized = [
                ann for ann in prior_window
                if ann.get("primary_road_structure") == RoadStructure.R4.value and _has_signal_evidence(ann)
            ]
            if len(prior_signalized) < 4:
                continue
            for ann in annotations[start:end]:
                old = ann.get("primary_road_structure")
                self._rewrite_rs_label(
                    ann,
                    RoadStructure.R4.value,
                    "blocked_signalized_tail_kept_r4_without_stop_yield",
                    RoadStructure.R4.value,
                )
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("blocked_signalized_tail_recovery", []).append(
                    {
                        "from": old,
                        "to": RoadStructure.R4.value,
                        "reason": "prior_stable_r4_signal_no_stop_yield",
                        "prior_signalized_frames": len(prior_signalized),
                    }
                )
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old,
                        "to": RoadStructure.R4.value,
                        "reason": "blocked_signalized_tail_kept_r4_without_stop_yield",
                        "run_start_frame": annotations[start].get("frame_id"),
                        "run_end_frame": annotations[end - 1].get("frame_id"),
                    }
                )
        return {"enabled": True, "changes": changes}

    def _apply_temporal_rs_smoothing(self, scenario_name: str, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """通用 RS 去抖：所有短片段都必须持续足够久才作为真实结构切换。"""
        changes: List[Dict[str, Any]] = []
        if len(annotations) < 3:
            return {"enabled": True, "changes": changes}

        def _dynamic_crossing_weak_r4_run(run_annotations: List[Dict[str, Any]]) -> bool:
            if scenario_name != "DynamicObjectCrossing":
                return False
            if not run_annotations:
                return False
            weak_count = 0
            for ann in run_annotations:
                evidence = ann.get("evidence") or {}
                rules = set(evidence.get("rules_fired") or [])
                flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
                has_local_context = bool(
                    flags.get("near_junction")
                    or flags.get("strong_control_context")
                    or flags.get("bbox_junction_hint")
                )
                if (
                    "r4_dynamic_crossing_meta_bbox_light_near_control" in rules
                    and not has_local_context
                ):
                    weak_count += 1
            return weak_count >= max(1, len(run_annotations) // 2)

        for iteration in range(max(8, len(annotations))):
            iteration_changes: List[Dict[str, Any]] = []
            runs = self._rs_runs(annotations)
            for run_index, run in enumerate(runs):
                label = run.get("label")
                if not label:
                    continue
                min_frames = self._min_rs_segment_frames(str(label))
                run_annotations = annotations[int(run["start"]): int(run["end"])]
                if str(label) == RoadStructure.R4.value and _dynamic_crossing_weak_r4_run(run_annotations):
                    min_frames = max(min_frames, 6)
                if int(run["length"]) >= min_frames:
                    continue
                prev_run = runs[run_index - 1] if run_index > 0 else None
                next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
                replacement = None
                reason = f"short_{label}_run_lt_{min_frames}_frames"
                inherited_from = None
                if prev_run and next_run and prev_run.get("label") == next_run.get("label"):
                    replacement = str(prev_run["label"])
                    inherited_from = "both_neighbors"
                elif prev_run or next_run:
                    neighbor_options = [r for r in (prev_run, next_run) if r]
                    chosen = max(
                        neighbor_options,
                        key=lambda r: (
                            int(r.get("length", 0)),
                            -abs(int(r.get("start", 0)) - int(run.get("start", 0))),
                        ),
                    )
                    replacement = str(chosen["label"])
                    inherited_from = "previous_neighbor" if chosen is prev_run else "next_neighbor"
                if replacement is None or replacement == label:
                    continue
                if not self._can_temporal_smoothing_promote(scenario_name, run_annotations, str(label), replacement):
                    continue
                change = {
                    "start_frame": annotations[int(run["start"])].get("frame_id"),
                    "end_frame": annotations[int(run["end"]) - 1].get("frame_id"),
                    "from": label,
                    "to": replacement,
                    "length": int(run["length"]),
                    "min_frames": min_frames,
                    "inherited_from": inherited_from,
                    "iteration": iteration + 1,
                }
                iteration_changes.append(change)
                for ann in run_annotations:
                    self._rewrite_rs_label(ann, replacement, reason, inherited_from)
                break
            changes.extend(iteration_changes)
            if not iteration_changes:
                break

        return {"enabled": True, "min_frames": {"R1": 2, "R2": 4, "R3": 4, "R4": 4, "R5": 4}, "changes": changes}

    def _apply_vehicle_turning_junction_gap_recovery(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """转弯路口内部短 R1 空洞回填，避免 STOP/路口过程中 R5/R4 抖成 R1。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        if cfg.get("kind") != "vehicle_turning" or len(annotations) < 5:
            return {"enabled": False, "changes": []}

        max_gap = int(cfg.get("turning_junction_gap_max_frames", 16))
        min_neighbor_frames = int(cfg.get("turning_junction_gap_min_neighbor_frames", 8))
        changes: List[Dict[str, Any]] = []

        def _gap_has_control_context(segment: List[Dict[str, Any]]) -> bool:
            if not segment:
                return False
            supported = 0
            for ann in segment:
                evidence = ann.get("evidence") or {}
                flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
                rules = set(evidence.get("rules_fired") or [])
                if (
                    flags.get("turning_local_junction_evidence")
                    or flags.get("strong_control_context")
                    or flags.get("bbox_stop_or_yield")
                    or flags.get("stop_hazard")
                    or "r5_generic_stop_or_junction_control" in rules
                    or "vehicle_turning_junction_space" in rules
                ):
                    supported += 1
            return supported >= max(1, len(segment) // 2)

        def _regular_event_for_rs(rs_label: str) -> EventType:
            if rs_label == RoadStructure.R4.value:
                return EventType.R_E4
            if rs_label == RoadStructure.R5.value:
                return EventType.R_E5
            return EventType.R_E1

        runs = self._rs_runs(annotations)
        for run_index, run in enumerate(runs):
            if run.get("label") != RoadStructure.R1.value:
                continue
            if int(run.get("length", 0)) > max_gap:
                continue
            if run_index <= 0:
                continue
            if run_index + 1 >= len(runs):
                continue
            prev_run = runs[run_index - 1]
            next_run = runs[run_index + 1]
            replacement = str(prev_run.get("label"))
            if replacement not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                continue
            if replacement != str(next_run.get("label")):
                continue
            if (
                int(prev_run.get("length", 0)) < min_neighbor_frames
                or int(next_run.get("length", 0)) < min_neighbor_frames
            ):
                continue
            start = int(run["start"])
            end = int(run["end"])
            segment = annotations[start:end]
            if not _gap_has_control_context(segment):
                continue

            regular = _regular_event_for_rs(replacement)
            for ann in segment:
                old_rs = ann.get("primary_road_structure")
                old_event = ann.get("primary_event")
                self._rewrite_rs_label(
                    ann,
                    replacement,
                    "vehicle_turning_short_junction_r1_gap_recovered",
                    inherited_from="both_neighbors",
                )
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("vehicle_turning_junction_gap_recovery", []).append(
                    {
                        "from": old_rs,
                        "to": replacement,
                        "reason": "short_r1_gap_between_same_intersection_rs",
                        "prev_start_frame": annotations[int(prev_run["start"])].get("frame_id"),
                        "next_end_frame": annotations[int(next_run["end"]) - 1].get("frame_id"),
                    }
                )
                events = {regular}
                primary_event = regular
                if old_event == EventType.U_E4.value:
                    events.add(EventType.U_E4)
                    primary_event = EventType.U_E4
                self._rewrite_event_label(
                    ann,
                    events,
                    primary_event,
                    "event_resynced_after_vehicle_turning_junction_gap_recovery",
                )
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                changes.append(
                    {
                        "frame_id": ann.get("frame_id"),
                        "from": old_rs,
                        "to": replacement,
                        "old_event": old_event,
                        "new_event": primary_event.value,
                        "reason": "vehicle_turning_short_junction_r1_gap_recovered",
                    }
                )

        return {
            "enabled": True,
            "max_gap_frames": max_gap,
            "min_neighbor_frames": min_neighbor_frames,
            "changes": changes,
        }

    def _apply_accident_initial_no_junction_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Accident 起始段只压制弱静态路口 hint；真实初始路口保留原 RS。"""
        first_n_frames = 30
        if scenario_name not in {"Accident", "AccidentTwoWays"} or not annotations:
            return {"enabled": False, "changes": []}
        first_ann = annotations[0]
        first_evidence = first_ann.get("evidence") or {}
        route_town = first_evidence.get("xml_town") or _extract_town(str(first_evidence.get("route_id") or ""))
        if str(route_town).lower() == "town13":
            return {
                "enabled": True,
                "first_n_frames": first_n_frames,
                "skipped": True,
                "reason": "town13_keeps_original_rs",
                "changes": [],
            }

        changes: List[Dict[str, Any]] = []

        def _has_initial_control_source(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            tl = str(evidence.get("traffic_light_state", "")).strip().lower()
            bbox = evidence.get("bbox_semantics") or {}
            xodr = evidence.get("xodr") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            xodr_junction = (
                bool(xodr.get("xodr_topology_trusted"))
                and (
                    str(xodr.get("map_junction_id") or "") not in {"", "-1", "None"}
                    or int(xodr.get("junction_connection_count", 0) or 0) > 0
                )
            )
            if (
                tl in {"red", "yellow", "green"}
                and (bool(bbox.get("traffic_light")) or bool(flags.get("near_junction")) or bool(evidence.get("meta_is_junction")))
            ):
                return True
            if bool(evidence.get("meta_is_junction")) or xodr_junction or bool(bbox.get("junction_hint")):
                return True
            if bool(evidence.get("meta_stop_hazard")):
                return True
            return bool(
                (bbox.get("stop_sign") or bbox.get("yield_sign"))
                and (bool(flags.get("near_junction")) or bool(flags.get("bbox_junction_hint")))
            )

        for index, ann in enumerate(annotations[:first_n_frames]):
            old_rs = ann.get("primary_road_structure")
            if old_rs not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                continue
            if _has_initial_control_source(ann):
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("accident_initial_no_junction_filter", []).append(
                    {
                        "frame_index": index,
                        "frame_id": ann.get("frame_id"),
                        "from": old_rs,
                        "to": old_rs,
                        "reason": "accident_initial_kept_true_control_source",
                        "scenario": scenario_name,
                    }
                )
                continue
            old_event = ann.get("primary_event")
            replacement_rs = RoadStructure.R2.value if scenario_name == "AccidentTwoWays" else RoadStructure.R1.value
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": old_rs,
                "to": replacement_rs,
                "old_primary_event": old_event,
                "reason": "accident_initial_30_frames_no_junction",
                "scenario": scenario_name,
            }
            self._rewrite_rs_label(
                ann,
                replacement_rs,
                "accident_initial_30_frames_no_junction",
                "route_level_accident_initial_filter",
            )
            if old_event in {EventType.R_E4.value, EventType.R_E5.value}:
                self._rewrite_event_label(
                    ann,
                    {EventType.R_E1},
                    EventType.R_E1,
                    "event_recomputed_after_accident_initial_no_junction",
                )
                ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
                change["new_primary_event"] = EventType.R_E1.value
            else:
                change["new_primary_event"] = ann.get("primary_event")
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("accident_initial_no_junction_filter", []).append(change)
            changes.append(change)

        return {"enabled": True, "first_n_frames": first_n_frames, "changes": changes}

    def _apply_hazard_side_twoways_initial_bbox_stop_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """清除 HazardAtSideLaneTwoWays 起始直道上的 bbox-only STOP 伪路口。"""
        first_n_frames = 30
        if scenario_name != "HazardAtSideLaneTwoWays" or not annotations:
            return {"enabled": False, "changes": []}
        changes: List[Dict[str, Any]] = []
        for index, ann in enumerate(annotations[:first_n_frames]):
            if ann.get("primary_road_structure") != RoadStructure.R5.value:
                continue
            evidence = ann.get("evidence") or {}
            bbox = evidence.get("bbox_semantics") or {}
            xodr = evidence.get("xodr") or {}
            tl = str(evidence.get("traffic_light_state") or "")
            bbox_only_stop = (
                bool(bbox.get("stop_sign"))
                and not bool(bbox.get("junction_hint"))
                and not bool(evidence.get("meta_is_junction"))
                and not bool(evidence.get("meta_stop_hazard"))
                and tl not in {"Red", "Yellow", "Green"}
                and not bool(bbox.get("traffic_light"))
            )
            unstructured_xodr = (
                str(xodr.get("map_junction_id") or "") in {"", "-1", "None"}
                and int(xodr.get("junction_connection_count", 0) or 0) == 0
            )
            if not (bbox_only_stop and unstructured_xodr):
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                RoadStructure.R2.value,
                "hazard_side_twoways_initial_bbox_only_stop_demoted_to_r2",
                "route_level_hazard_side_initial_filter",
            )
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": RoadStructure.R5.value,
                "to": RoadStructure.R2.value,
                "old_primary_event": old_event,
                "reason": "hazard_side_twoways_initial_bbox_only_stop",
            }
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("hazard_side_twoways_initial_filter", []).append(change)
            changes.append(change)
        return {"enabled": True, "first_n_frames": first_n_frames, "changes": changes}

    def _apply_hazard_side_initial_weak_junction_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """清除 HazardAtSideLane 起始直道被弱 stop/trigger 证据抬成 R4/R5 的伪路口。"""
        first_n_frames = 30
        if scenario_name != "HazardAtSideLane" or not annotations:
            return {"enabled": False, "changes": []}

        changes: List[Dict[str, Any]] = []

        def _has_strong_initial_junction_source(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            bbox = evidence.get("bbox_semantics") or {}
            xodr = evidence.get("xodr") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            tl = str(evidence.get("traffic_light_state") or "").strip().lower()
            xodr_junction = (
                bool(xodr.get("xodr_topology_trusted"))
                and (
                    str(xodr.get("map_junction_id") or "") not in {"", "-1", "None"}
                    or int(xodr.get("junction_connection_count", 0) or 0) > 0
                )
            )
            visible_signalized = bool(bbox.get("traffic_light")) and tl in {"red", "yellow", "green"}
            visible_unsignalized = (
                bool(bbox.get("junction_hint"))
                and (bool(bbox.get("stop_sign")) or bool(bbox.get("yield_sign")) or bool(evidence.get("meta_stop_hazard")))
            )
            local_junction = bool(flags.get("bbox_junction_hint")) or (
                bool(flags.get("near_junction"))
                and (bool(evidence.get("meta_stop_hazard")) or bool(evidence.get("combined_stop_hazard")))
            )
            meta_junction = bool(evidence.get("meta_is_junction"))
            return bool(meta_junction or xodr_junction or visible_signalized or visible_unsignalized or local_junction)

        for index, ann in enumerate(annotations[:first_n_frames]):
            old_rs = ann.get("primary_road_structure")
            if old_rs not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                continue
            if _has_strong_initial_junction_source(ann):
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                RoadStructure.R1.value,
                "hazard_side_initial_weak_junction_demoted_to_r1",
                "route_level_hazard_side_initial_filter",
            )
            self._rewrite_event_label(
                ann,
                {EventType.R_E1},
                EventType.R_E1,
                "event_resynced_after_hazard_side_initial_weak_junction_demotion",
            )
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": old_rs,
                "to": RoadStructure.R1.value,
                "old_primary_event": old_event,
                "new_primary_event": EventType.R_E1.value,
                "reason": "hazard_side_initial_weak_junction",
            }
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("hazard_side_initial_weak_junction_filter", []).append(change)
            changes.append(change)

        return {"enabled": True, "first_n_frames": first_n_frames, "changes": changes}

    def _apply_control_loss_initial_weak_junction_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """清除 ControlLoss Town01-04 起始直道上的弱证据伪路口。"""
        first_n_frames = 30
        if scenario_name != "ControlLoss" or not annotations:
            return {"enabled": False, "changes": []}
        first_evidence = annotations[0].get("evidence") or {}
        route_town = str(first_evidence.get("xml_town") or _extract_town(str(first_evidence.get("route_id") or ""))).lower()
        if route_town not in {"town01", "town02", "town03", "town04"}:
            return {
                "enabled": True,
                "first_n_frames": first_n_frames,
                "skipped": True,
                "reason": "control_loss_initial_filter_only_town01_to_town04",
                "changes": [],
            }

        changes: List[Dict[str, Any]] = []

        def _has_strong_initial_control_source(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            bbox = evidence.get("bbox_semantics") or {}
            xodr = evidence.get("xodr") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            tl = str(evidence.get("traffic_light_state") or "").strip().lower()
            xodr_junction = (
                bool(xodr.get("xodr_topology_trusted"))
                and (
                    str(xodr.get("map_junction_id") or "") not in {"", "-1", "None"}
                    or int(xodr.get("junction_connection_count", 0) or 0) > 0
                )
            )
            signalized = (
                tl in {"red", "yellow", "green"}
                and bool(bbox.get("traffic_light"))
                and (
                    bool(flags.get("near_junction"))
                    or bool(flags.get("bbox_junction_hint"))
                    or bool(bbox.get("junction_hint"))
                )
            )
            unsignalized = (
                (bool(flags.get("bbox_junction_hint")) or bool(bbox.get("junction_hint")))
                and (
                    bool(evidence.get("meta_is_junction"))
                    or bool(evidence.get("meta_stop_hazard"))
                    or bool(bbox.get("stop_sign"))
                    or bool(bbox.get("yield_sign"))
                    or xodr_junction
                )
            )
            return bool(signalized or unsignalized)

        for index, ann in enumerate(annotations[:first_n_frames]):
            old_rs = ann.get("primary_road_structure")
            if old_rs not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                continue
            if _has_strong_initial_control_source(ann):
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                RoadStructure.R1.value,
                "control_loss_town01_04_initial_weak_junction_demoted_to_r1",
                "route_level_control_loss_initial_filter",
            )
            self._rewrite_event_label(
                ann,
                {EventType.R_E1},
                EventType.R_E1,
                "event_resynced_after_control_loss_initial_weak_junction_demotion",
            )
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": old_rs,
                "to": RoadStructure.R1.value,
                "old_primary_event": old_event,
                "new_primary_event": EventType.R_E1.value,
                "reason": "control_loss_town01_04_initial_weak_junction",
                "town": route_town,
            }
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("control_loss_initial_weak_junction_filter", []).append(change)
            changes.append(change)

        return {"enabled": True, "first_n_frames": first_n_frames, "town": route_town, "changes": changes}

    def _apply_signalized_enter_flow_initial_weak_r4_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """清除 SignalizedJunctionLeftTurnEnterFlow Town01/02 起步直道上的远灯 R4。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        towns = {str(t).lower() for t in cfg.get("initial_weak_r4_towns", [])}
        first_n_frames = 30
        if scenario_name != "SignalizedJunctionLeftTurnEnterFlow" or not annotations or not towns:
            return {"enabled": False, "changes": []}
        first_evidence = annotations[0].get("evidence") or {}
        route_town = str(first_evidence.get("xml_town") or _extract_town(str(first_evidence.get("route_id") or ""))).lower()
        if route_town not in towns:
            return {
                "enabled": True,
                "first_n_frames": first_n_frames,
                "skipped": True,
                "reason": "signalized_enter_flow_initial_filter_town_not_selected",
                "town": route_town,
                "changes": [],
            }

        trigger_limit = _safe_float(cfg.get("initial_weak_r4_trigger_m"), default=22.0)
        changes: List[Dict[str, Any]] = []

        def _has_local_junction_core(ann: Dict[str, Any]) -> bool:
            evidence = ann.get("evidence") or {}
            bbox = evidence.get("bbox_semantics") or {}
            xodr = evidence.get("xodr") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            xodr_junction = (
                bool(xodr.get("xodr_topology_trusted"))
                and (
                    str(xodr.get("map_junction_id") or "") not in {"", "-1", "None"}
                    or int(xodr.get("junction_connection_count", 0) or 0) > 0
                )
            )
            return bool(
                evidence.get("meta_is_junction")
                or bool(bbox.get("junction_hint"))
                or bool(flags.get("bbox_junction_hint"))
                or xodr_junction
            )

        for index, ann in enumerate(annotations[:first_n_frames]):
            if ann.get("primary_road_structure") != RoadStructure.R4.value:
                continue
            trigger_distance = _safe_float((ann.get("evidence") or {}).get("trigger_distance_m"), default=math.inf)
            if trigger_distance > trigger_limit or _has_local_junction_core(ann):
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                RoadStructure.R1.value,
                "signalized_enter_flow_town01_02_initial_weak_r4_demoted_to_r1",
                "route_level_signalized_enter_flow_initial_filter",
            )
            self._rewrite_event_label(
                ann,
                {EventType.R_E1},
                EventType.R_E1,
                "event_resynced_after_signalized_enter_flow_initial_weak_r4_demotion",
            )
            ann["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann)
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": RoadStructure.R4.value,
                "to": RoadStructure.R1.value,
                "old_primary_event": old_event,
                "new_primary_event": EventType.R_E1.value,
                "trigger_distance_m": trigger_distance,
                "reason": "signalized_enter_flow_town01_02_initial_weak_r4",
                "town": route_town,
            }
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("signalized_enter_flow_initial_weak_r4_filter", []).append(change)
            changes.append(change)

        return {
            "enabled": True,
            "first_n_frames": first_n_frames,
            "town": route_town,
            "trigger_limit_m": trigger_limit,
            "changes": changes,
        }

    def _apply_red_light_exit_tail_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """RedLightWithoutLeadVehicle 出口段远离路口后释放为 R1/R-E1。"""
        if scenario_name != "RedLightWithoutLeadVehicle" or not annotations:
            return {"enabled": False, "changes": []}
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        trigger_limit = _safe_float(cfg.get("scenario_active_signal_max_m"), default=52.0)
        changes: List[Dict[str, Any]] = []
        for index, ann in enumerate(annotations):
            if ann.get("primary_road_structure") != RoadStructure.R4.value:
                continue
            if ann.get("primary_event") != EventType.R_E4.value:
                continue
            evidence = ann.get("evidence") or {}
            flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
            trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
            metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
            speed = _safe_float(metrics.get("speed"), default=0.0)
            if (
                trigger_distance <= trigger_limit
                or speed < 2.0
                or bool(evidence.get("meta_is_junction"))
                or bool(flags.get("near_junction"))
                or bool(flags.get("strong_control_context"))
                or bool(flags.get("junction_window"))
            ):
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                RoadStructure.R1.value,
                "red_light_exit_tail_far_signal_released_to_r1",
                "route_level_red_light_exit_tail_filter",
            )
            self._rewrite_event_label(
                ann,
                {EventType.R_E1},
                EventType.R_E1,
                "event_resynced_after_red_light_exit_tail_release",
            )
            ann["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann)
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
            change = {
                "frame_index": index,
                "frame_id": ann.get("frame_id"),
                "from": RoadStructure.R4.value,
                "to": RoadStructure.R1.value,
                "old_primary_event": old_event,
                "new_primary_event": EventType.R_E1.value,
                "trigger_distance_m": trigger_distance,
                "speed": speed,
                "reason": "red_light_exit_tail_far_signal",
            }
            evidence = ann.setdefault("evidence", {})
            evidence.setdefault("red_light_exit_tail_filter", []).append(change)
            changes.append(change)
        return {"enabled": True, "trigger_limit_m": trigger_limit, "changes": changes}

    @staticmethod
    def _noscenario_local_signal_control(evidence: Dict[str, Any]) -> bool:
        bbox = evidence.get("bbox_semantics") or {}
        metrics = bbox.get("metrics") or {}
        tl_count = int(_safe_float(metrics.get("traffic_light_count"), default=0.0) or 0)
        tl_distance = _safe_float(metrics.get("traffic_light_min_distance_m"), default=math.inf)
        tl_forward_x = _safe_float(metrics.get("traffic_light_min_forward_x_m"), default=math.inf)
        tl_physical = _safe_float(metrics.get("traffic_light_min_physical_distance_m"), default=math.inf)
        tl_affects_ego = _safe_bool(metrics.get("traffic_light_affects_ego", False))
        tl_overhead = _safe_bool(metrics.get("traffic_light_overhead", False))
        meta_is_junction = _safe_bool(evidence.get("meta_is_junction", False))
        return bool(
            tl_overhead
            or (tl_affects_ego and tl_forward_x <= 18.0)
            or (tl_affects_ego and tl_distance <= 12.0)
            or (tl_count >= 2 and tl_affects_ego and tl_forward_x <= 32.0)
            or (tl_physical <= 24.0 and tl_forward_x <= 25.0)
            or (meta_is_junction and tl_forward_x <= 12.0)
        )

    @staticmethod
    def _noscenario_local_stop_or_yield(evidence: Dict[str, Any]) -> bool:
        bbox = evidence.get("bbox_semantics") or {}
        metrics = bbox.get("metrics") or {}
        stop_distance = _safe_float(metrics.get("stop_sign_min_distance_m"), default=math.inf)
        yield_distance = _safe_float(metrics.get("yield_sign_min_distance_m"), default=math.inf)
        return bool(
            _safe_bool(evidence.get("meta_stop_hazard", False))
            or stop_distance <= 35.0
            or yield_distance <= 35.0
        )

    def _apply_route_junction_control_lock(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """同一 route 的路口控制源只能在 R4/R5 中选一种，避免有灯/无灯互跳。"""
        lock_scenarios = {
            "T_Junction",
            "PedestrianCrossing",
            "PriorityAtJunction",
            "OppositeVehicleTakingPriority",
        }
        parking_guard_scenarios = {"ParkingExit", "ParkingCutIn"}
        if not annotations:
            return {"enabled": False, "changes": []}
        allowed_rs = set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, []))
        allows_r4_r5 = {RoadStructure.R4, RoadStructure.R5}.issubset(allowed_rs)
        has_r4 = any(ann.get("primary_road_structure") == RoadStructure.R4.value for ann in annotations)
        has_r5 = any(ann.get("primary_road_structure") == RoadStructure.R5.value for ann in annotations)
        mixed_r45_route = has_r4 and has_r5
        if (
            scenario_name not in lock_scenarios | parking_guard_scenarios
            and not (allows_r4_r5 and mixed_r45_route)
        ):
            return {"enabled": False, "changes": []}
        has_valid_light = any(
            str((ann.get("evidence") or {}).get("traffic_light_state") or "") in {"Red", "Yellow", "Green"}
            for ann in annotations
        )
        bbox_light_frames = sum(
            1
            for ann in annotations
            if bool(((ann.get("evidence") or {}).get("bbox_semantics") or {}).get("traffic_light"))
        )
        has_visible_light = bbox_light_frames >= 3
        route_is_signalized = bool(has_valid_light or has_visible_light)
        priority_signal_support_indices = []
        if scenario_name == "PriorityAtJunction" and route_is_signalized:
            for idx_ann, ann in enumerate(annotations):
                evidence = ann.get("evidence") or {}
                bbox = evidence.get("bbox_semantics") or {}
                if (
                    str(evidence.get("traffic_light_state") or "") in {"Red", "Yellow", "Green"}
                    or bool(evidence.get("light_hazard"))
                    or bool(bbox.get("traffic_light"))
                    or bool(evidence.get("meta_is_junction"))
                ):
                    priority_signal_support_indices.append(idx_ann)
        priority_signal_start = min(priority_signal_support_indices) if priority_signal_support_indices else None
        priority_signal_end = max(priority_signal_support_indices) if priority_signal_support_indices else None
        priority_existing_control_indices = []
        if scenario_name == "PriorityAtJunction" and route_is_signalized:
            priority_existing_control_indices = [
                idx_ann
                for idx_ann, ann in enumerate(annotations)
                if ann.get("primary_road_structure") == RoadStructure.R4.value
            ]
        priority_existing_control_start = (
            min(priority_existing_control_indices) if priority_existing_control_indices else priority_signal_start
        )
        priority_signal_pre_frames = int(
            SCENARIO_RULE_CONFIG.get(scenario_name, {}).get("priority_signal_pre_frames", 2)
        )
        changes: List[Dict[str, Any]] = []
        centered_far_threshold = {
            "T_Junction": 30.0,
            "PedestrianCrossing": 24.0,
            "PriorityAtJunction": 22.0,
            "OppositeVehicleTakingPriority": 18.0,
        }.get(scenario_name)

        def _regular_event_for_target(target_label: str) -> EventType:
            if target_label == RoadStructure.R4.value:
                return EventType.R_E4
            if target_label == RoadStructure.R5.value:
                return EventType.R_E5
            return EventType.R_E1

        for idx_ann, ann in enumerate(annotations):
            old = ann.get("primary_road_structure")
            local_control_protected = False
            if scenario_name in lock_scenarios:
                priority_signal_pre_window = bool(
                    scenario_name == "PriorityAtJunction"
                    and route_is_signalized
                    and priority_existing_control_start is not None
                    and priority_existing_control_start >= 30
                    and idx_ann >= max(0, priority_existing_control_start - priority_signal_pre_frames)
                    and idx_ann < priority_existing_control_start
                )
                if old not in {RoadStructure.R4.value, RoadStructure.R5.value} and not priority_signal_pre_window:
                    continue
                metrics = (ann.get("event_evidence") or {}).get("metrics") or {}
                evidence = ann.get("evidence") or {}
                trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
                lateral = _safe_float(metrics.get("route_lateral_abs_m"), default=math.inf)
                bbox = evidence.get("bbox_semantics") or {}
                lock_signal_protect_m = _safe_float(
                    SCENARIO_RULE_CONFIG.get(scenario_name, {}).get("junction_lock_signal_protect_m"),
                    default=math.nan,
                )
                local_signal_frame = bool(
                    bool(bbox.get("traffic_light"))
                    or bool(evidence.get("light_hazard"))
                    or str(evidence.get("traffic_light_state") or "") in {"Red", "Yellow", "Green"}
                    or bool(evidence.get("meta_is_junction"))
                    or bool(bbox.get("junction_hint"))
                )
                priority_signal_window = bool(
                    scenario_name == "PriorityAtJunction"
                    and route_is_signalized
                    and priority_existing_control_start is not None
                    and priority_signal_end is not None
                    and idx_ann >= max(
                        0,
                        priority_existing_control_start
                        - (priority_signal_pre_frames if priority_existing_control_start >= 30 else 0),
                    )
                    and idx_ann <= min(len(annotations) - 1, priority_signal_end + 8)
                )
                local_control_protected = bool(
                    evidence.get("meta_is_junction")
                    or bool(bbox.get("junction_hint"))
                    or (
                        math.isfinite(lock_signal_protect_m)
                        and trigger_distance <= lock_signal_protect_m
                        and (
                            bool(bbox.get("traffic_light"))
                            or str(evidence.get("traffic_light_state") or "") in {"Red", "Yellow", "Green"}
                        )
                    )
                    or priority_signal_window
                )
                if scenario_name == "PriorityAtJunction" and route_is_signalized:
                    if priority_signal_window or local_signal_frame:
                        target = RoadStructure.R4.value
                    else:
                        target = RoadStructure.R1.value
                    moving_centered_far = False
                    local_control_protected = bool(priority_signal_window or local_signal_frame)
                else:
                    moving_centered_far = (
                        centered_far_threshold is not None
                        and trigger_distance > centered_far_threshold
                        and _safe_float(metrics.get("speed"), default=0.0) > 5.0
                        and math.isfinite(lateral)
                        and lateral < 0.04
                        and not local_control_protected
                    )
                    if moving_centered_far:
                        target = RoadStructure.R1.value
                    else:
                        target = RoadStructure.R4.value if route_is_signalized else RoadStructure.R5.value
            elif scenario_name in parking_guard_scenarios:
                if has_valid_light:
                    if scenario_name == "ParkingCutIn" and old == RoadStructure.R5.value:
                        target = RoadStructure.R4.value
                    else:
                        continue
                elif old != RoadStructure.R4.value:
                    continue
                else:
                    evidence = ann.get("evidence") or {}
                    bbox = evidence.get("bbox_semantics") or {}
                    strong_nolight = bool(evidence.get("meta_is_junction")) and (
                        bool(evidence.get("meta_stop_hazard"))
                        or bool(bbox.get("stop_sign"))
                        or bool(bbox.get("yield_sign"))
                    )
                    target = (
                        RoadStructure.R5.value
                        if scenario_name == "ParkingCutIn" and strong_nolight
                        else RoadStructure.R1.value
                    )
            elif scenario_name == "noScenarios":
                if old not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                    continue
                evidence = ann.get("evidence") or {}
                local_signal = self._noscenario_local_signal_control(evidence)
                local_stop_yield = self._noscenario_local_stop_or_yield(evidence)
                if local_stop_yield and not local_signal:
                    target = RoadStructure.R5.value
                elif local_signal:
                    target = RoadStructure.R4.value
                else:
                    target = RoadStructure.R1.value
            else:
                if old not in {RoadStructure.R4.value, RoadStructure.R5.value}:
                    continue
                target = RoadStructure.R4.value if route_is_signalized else RoadStructure.R5.value
            if target == old:
                continue
            old_event = ann.get("primary_event")
            self._rewrite_rs_label(
                ann,
                target,
                "route_junction_control_source_locked",
                "route_level_junction_control_lock",
            )
            target_regular = _regular_event_for_target(target)
            if old_event in {EventType.R_E4.value, EventType.R_E5.value, EventType.R_E1.value}:
                self._rewrite_event_label(
                    ann,
                    {target_regular},
                    target_regular,
                    "event_resynced_after_route_junction_control_source_lock",
                )
            elif old_event in {EventType.U_E4.value, EventType.U_E6.value, EventType.U_E7.value, EventType.U_E8.value}:
                unusual = EventType(old_event)
                self._rewrite_event_label(
                    ann,
                    {target_regular, unusual},
                    unusual,
                    "event_regular_resynced_after_route_junction_control_source_lock",
                )
            changes.append(
                {
                    "frame_id": ann.get("frame_id"),
                    "from": old,
                    "to": target,
                    "has_valid_light": has_valid_light,
                    "has_visible_light": has_visible_light,
                    "local_control_protected": local_control_protected,
                    "old_primary_event": old_event,
                    "new_regular_event": target_regular.value,
                }
            )
        return {
            "enabled": True,
            "has_valid_light": has_valid_light,
            "has_visible_light": has_visible_light,
            "route_is_signalized": route_is_signalized,
            "mixed_r45_route": mixed_r45_route,
            "changes": changes,
        }

    def _fallback_after_twoways_core(self, ann: Dict[str, Any]) -> str:
        """TwoWays 核心段结束后回到真实路网标签：路口回 R4/R5，否则保持 R2。"""
        candidates = ann.get("road_structure_candidates", {}) or {}
        evidence = ann.get("evidence", {}) or {}
        rules = set(evidence.get("rules_fired", []) or [])
        r4_score = float(candidates.get("R4", 0.0) or 0.0)
        r5_score = float(candidates.get("R5", 0.0) or 0.0)
        if r4_score >= 0.82 and any(rule.startswith("r4_") or rule.startswith("twoways_post_core") for rule in rules):
            return "R4"
        if r5_score >= 0.82 and any(rule.startswith("r5_") or rule.endswith("_r5") for rule in rules):
            return "R5"
        return "R2"

    @staticmethod
    def _twoways_core_metric(ann: Dict[str, Any]) -> Tuple[Optional[float], bool, bool]:
        evidence = ann.get("evidence", {}) or {}
        obs = evidence.get("twoway_obstruction_evidence") or {}
        nearest = obs.get("nearest_obstacle_m")
        nearest_f = _safe_float(nearest, default=math.inf)
        return (
            nearest_f if math.isfinite(nearest_f) else None,
            _safe_bool(obs.get("stuck", False)),
            _safe_bool(obs.get("vehicle_hazard", False)),
        )

    @staticmethod
    def _twoways_topology_r2_confirmed(ann: Dict[str, Any]) -> bool:
        rules = set(((ann.get("evidence") or {}).get("rules_fired") or []))
        return bool(
            {
                "r2_twoways_single_lane_topology_confirmed",
                "r2_twoways_effective_single_lane_topology_confirmed",
                "r2_twoways_layout_prior_effective_drivable_default",
                "r2_opposite_lane_confirmed",
            }
            & rules
        )

    def _apply_twoways_core_span_clipping(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """只裁掉无双向单车道拓扑支撑的事件型 R2，保留正常 TwoWays 道路 R2。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        if cfg.get("kind") != "twoways_obstacle" or not annotations:
            return {"enabled": False, "changes": []}

        exit_delta = float(cfg.get("two_way_exit_delta_m", 3.0))
        exit_hold = max(1, int(cfg.get("two_way_exit_hold_frames", 8)))
        changes: List[Dict[str, Any]] = []
        runs = self._rs_runs(annotations)
        for run in runs:
            if run.get("label") != "R2":
                continue
            start = int(run["start"])
            end = int(run["end"])
            run_annotations = annotations[start:end]
            if all(self._twoways_topology_r2_confirmed(ann) for ann in run_annotations):
                continue
            finite_metrics = []
            for offset, ann in enumerate(run_annotations):
                nearest, stuck, vehicle_hazard = self._twoways_core_metric(ann)
                if nearest is not None:
                    finite_metrics.append((offset, nearest, stuck, vehicle_hazard))
            if not finite_metrics:
                continue
            min_offset, min_nearest, _, _ = min(finite_metrics, key=lambda item: item[1])
            cut_offset = None
            quiet_count = 0
            for offset, nearest, stuck, vehicle_hazard in finite_metrics:
                if offset <= min_offset:
                    continue
                moving_away = nearest >= min_nearest + exit_delta
                quiet = (not stuck) and (not vehicle_hazard)
                if moving_away and quiet:
                    quiet_count += 1
                    if quiet_count >= exit_hold:
                        cut_offset = offset - exit_hold + 1
                        break
                else:
                    quiet_count = 0
            if cut_offset is None:
                continue

            cut_start = start + cut_offset
            change = {
                "start_frame": annotations[cut_start].get("frame_id"),
                "end_frame": annotations[end - 1].get("frame_id"),
                "from": "R2",
                "to": "R1/R4_by_evidence",
                "nearest_obstacle_min_m": round(min_nearest, 3),
                "exit_delta_m": exit_delta,
                "exit_hold_frames": exit_hold,
                "reason": "twoways_core_passed_obstacle_distance_increasing",
            }
            changes.append(change)
            for ann in annotations[cut_start:end]:
                if self._twoways_topology_r2_confirmed(ann):
                    continue
                replacement = self._fallback_after_twoways_core(ann)
                self._rewrite_rs_label(
                    ann,
                    replacement,
                    "twoways_core_span_clipped_after_obstacle",
                    "route_level_twoways_core_exit",
                )
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("twoways_core_span_clipping", []).append(change)

        return {"enabled": True, "changes": changes}

    def _apply_twoways_longest_r2_filter(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """只清理无双向单车道拓扑支撑的 R2 碎片；正常 TwoWays 直道可以持续 R2。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        if cfg.get("kind") not in {"twoways_obstacle", "vehicle_opens_door_twoways"} or not annotations:
            return {"enabled": False, "changes": []}

        r2_runs = []
        for run in self._rs_runs(annotations):
            if run.get("label") != "R2":
                continue
            start = int(run["start"])
            end = int(run["end"])
            if all(self._twoways_topology_r2_confirmed(ann) for ann in annotations[start:end]):
                continue
            r2_runs.append(run)
        if len(r2_runs) <= 1:
            return {"enabled": True, "kept": None, "changes": []}

        def run_score(run: Dict[str, Any]) -> Tuple[int, float, float]:
            start = int(run["start"])
            end = int(run["end"])
            run_annotations = annotations[start:end]
            confidences = [
                float(ann.get("confidence", 0.0) or 0.0)
                for ann in run_annotations
            ]
            r2_scores = [
                float((ann.get("road_structure_candidates", {}) or {}).get("R2", 0.0) or 0.0)
                for ann in run_annotations
            ]
            return (
                int(run.get("length", 0) or 0),
                sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
                sum(confidences) / len(confidences) if confidences else 0.0,
            )

        keep_run = max(r2_runs, key=run_score)
        kept = {
            "start_frame": annotations[int(keep_run["start"])].get("frame_id"),
            "end_frame": annotations[int(keep_run["end"]) - 1].get("frame_id"),
            "length": int(keep_run["length"]),
            "reason": "longest_continuous_r2_span",
        }
        changes: List[Dict[str, Any]] = []
        for run in r2_runs:
            if run is keep_run:
                continue
            start = int(run["start"])
            end = int(run["end"])
            change = {
                "start_frame": annotations[start].get("frame_id"),
                "end_frame": annotations[end - 1].get("frame_id"),
                "from": "R2",
                "to": "R1/R4_by_evidence",
                "length": int(run["length"]),
                "kept_r2_span": kept,
                "reason": "twoways_non_longest_r2_disturbance",
            }
            changes.append(change)
            for ann in annotations[start:end]:
                if self._twoways_topology_r2_confirmed(ann):
                    continue
                replacement = self._fallback_after_twoways_core(ann)
                self._rewrite_rs_label(
                    ann,
                    replacement,
                    "twoways_non_longest_r2_disturbance",
                    "route_level_longest_r2_span",
                )
                evidence = ann.setdefault("evidence", {})
                evidence.setdefault("twoways_longest_r2_filter", []).append(change)

        return {"enabled": True, "kept": kept, "changes": changes}

    @staticmethod
    def _can_temporal_smoothing_promote(scenario_name: str, run_annotations: List[Dict[str, Any]], old_label: str, replacement: str) -> bool:
        """避免把只有弱证据的普通路段，因邻居继承提升成特殊 ROAD_STRUCTURE。"""
        if old_label in {"R4", "R5"} and replacement != old_label and len(run_annotations) < 4:
            # A real intersection/T-junction traversal should last around a second or more
            # at 4 Hz. 2-3 frame R4/R5 spikes are usually transient TL/bbox/XODR hints.
            return True
        if old_label == "R5" and replacement == "R1":
            for ann in run_annotations:
                rules = set((ann.get("evidence", {}) or {}).get("rules_fired", []) or [])
                if any(
                    rule.startswith("invading_turn_") and rule.endswith("_r5")
                    for rule in rules
                ):
                    return False
        if old_label != "R1" or replacement not in {"R2", "R3", "R4", "R5"}:
            return True
        for ann in run_annotations:
            evidence = ann.get("evidence", {}) or {}
            rules = set(evidence.get("rules_fired", []) or [])
            if replacement == "R2" and "r2_core_obstruction_confirmed" not in rules:
                return False
            if replacement == "R3" and not any(
                rule.startswith("r3_") or rule.startswith("hardbreak_rgb_route_bucket")
                for rule in rules
            ):
                return False
            if replacement == "R4" and not any(rule.startswith("r4_") for rule in rules):
                return False
            if scenario_name == "DynamicObjectCrossing" and replacement == "R4":
                flags = ((evidence.get("diagnostic_attribution") or {}).get("window_flags") or {})
                if not (flags.get("close_trigger_for_junction") or flags.get("bbox_junction_hint")):
                    return False
            if replacement == "R5" and not any(
                rule.startswith("r5_")
                or (rule.startswith("invading_turn_") and rule.endswith("_r5"))
                or (rule.startswith("interurban_") and rule.endswith("_r5"))
                for rule in rules
            ):
                diagnostic = evidence.get("diagnostic_attribution", {}) or {}
                window_flags = diagnostic.get("window_flags", {}) or {}
                used_inputs = diagnostic.get("used_inputs", {}) or {}
                trigger_distance = _safe_float(evidence.get("trigger_distance_m"), default=math.inf)
                invading_gap_bridge = (
                    evidence.get("rule_kind") == "invading_turn"
                    and (
                        bool(window_flags.get("junction_window"))
                        or bool(used_inputs.get("meta_stop_hint"))
                        or trigger_distance < 45.0
                    )
                )
                if not invading_gap_bridge:
                    return False
        return True

    def _process_route(self, scenario_name: str, route_path: Path, max_frames_per_route: Optional[int] = None) -> Dict:
        """处理单个route"""
        data_skip, xml_info = self._route_data_quality_skip(scenario_name, route_path)
        if data_skip is not None:
            return data_skip

        metas_dir = route_path / "metas"
        meta_files = sorted(metas_dir.glob("*.pkl"))
        if max_frames_per_route is not None and max_frames_per_route > 0:
            meta_files = meta_files[:max_frames_per_route]
        bboxes_dir = route_path / "bboxes"
        annotations = []

        for meta_file in meta_files:
            try:
                frame_id = int(meta_file.stem)
                # 使用支持 XZ 压缩的加载函数
                frame_data = load_pickle_file(meta_file)
                if isinstance(frame_data, dict):
                    bbox_summary = summarize_bbox_semantics(bboxes_dir / meta_file.name)
                    frame_data.update(bbox_summary)

                ann = SimpleFrameAnalyzer.analyze(scenario_name, frame_id, frame_data, xml_info=xml_info, route_id=route_path.name)
                ann_dict = ann.to_dict()
                ann_dict["frame_time_s"] = round(frame_id * 0.25, 3)
                ann_dict["meta_path"] = str(meta_file)
                ann_dict["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann_dict)
                annotations.append(ann_dict)
            except Exception as e:
                self.logger.warning(f"处理 {meta_file} 出错: {e}")
                continue

        twoways_core_span_clipping = self._apply_twoways_core_span_clipping(scenario_name, annotations)
        twoways_longest_r2_filter = self._apply_twoways_longest_r2_filter(scenario_name, annotations)
        r4_context_recovery = self._apply_r4_context_recovery(scenario_name, annotations)
        blocked_signalized_tail_recovery = self._apply_blocked_signalized_tail_recovery(scenario_name, annotations)
        temporal_smoothing_summary = self._apply_temporal_rs_smoothing(scenario_name, annotations)
        vehicle_turning_junction_gap_recovery = self._apply_vehicle_turning_junction_gap_recovery(scenario_name, annotations)
        accident_initial_no_junction_filter = self._apply_accident_initial_no_junction_filter(scenario_name, annotations)
        hazard_side_twoways_initial_filter = self._apply_hazard_side_twoways_initial_bbox_stop_filter(
            scenario_name,
            annotations,
        )
        hazard_side_initial_weak_junction_filter = self._apply_hazard_side_initial_weak_junction_filter(
            scenario_name,
            annotations,
        )
        control_loss_initial_weak_junction_filter = self._apply_control_loss_initial_weak_junction_filter(
            scenario_name,
            annotations,
        )
        signalized_enter_flow_initial_weak_r4_filter = self._apply_signalized_enter_flow_initial_weak_r4_filter(
            scenario_name,
            annotations,
        )
        red_light_exit_tail_filter = self._apply_red_light_exit_tail_filter(
            scenario_name,
            annotations,
        )
        route_junction_control_lock = self._apply_route_junction_control_lock(scenario_name, annotations)
        event_postprocess_summary = self._apply_event_route_postprocess(scenario_name, annotations)
        crossing_u4_single_span = self._apply_crossing_u4_single_span_filter(scenario_name, annotations)
        event_candidate_clamp = self._apply_event_candidate_clamp(scenario_name, annotations)
        final_junction_regular_gap_merge = self._apply_final_junction_regular_gap_merge(scenario_name, annotations)

        primary_counter = defaultdict(int)
        event_counter = defaultdict(int)
        review_counter = defaultdict(int)
        event_review_counter = defaultdict(int)
        xodr_source_counter = defaultdict(int)
        route_bucket_counter = defaultdict(int)
        review_frame_count = 0
        event_review_frame_count = 0
        transition_frames = []
        event_transition_frames = []
        prev_primary = None
        prev_event = None
        for ann in annotations:
            primary = ann.get("primary_road_structure")
            if primary:
                primary_counter[primary] += 1
                if prev_primary is not None and primary != prev_primary:
                    transition_frames.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": prev_primary,
                            "to": primary,
                            "comment": ann.get("annotation_comment", ""),
                        }
                    )
                prev_primary = primary
            primary_event = ann.get("primary_event")
            if primary_event:
                event_counter[primary_event] += 1
                if prev_event is not None and primary_event != prev_event:
                    event_transition_frames.append(
                        {
                            "frame_id": ann.get("frame_id"),
                            "from": prev_event,
                            "to": primary_event,
                            "comment": ann.get("frame_event_annotation", {}).get("comment", ""),
                        }
                    )
                prev_event = primary_event
            evidence = ann.get("evidence", {})
            if evidence.get("review_required"):
                review_frame_count += 1
                for reason in evidence.get("review_reasons", ["review_required"]):
                    review_counter[reason] += 1
            event_evidence = ann.get("event_evidence", {})
            if event_evidence.get("review_required"):
                event_review_frame_count += 1
                for reason in event_evidence.get("review_reasons", ["event_review_required"]):
                    event_review_counter[reason] += 1
            xodr_source = evidence.get("xodr", {}).get("xodr_source") or "unavailable"
            xodr_source_counter[xodr_source] += 1
            route_bucket_counter[evidence.get("route_semantic_bucket") or "unknown"] += 1

        return {
            "route_id": route_path.name,
            "status": "success",
            "xml_path": str(xml_info.path) if xml_info else None,
            "xml_town": xml_info.town if xml_info else None,
            "xml_available": xml_info is not None,
            "num_frames": len(annotations),
            "primary_rs_distribution": dict(sorted(primary_counter.items())),
            "primary_event_distribution": dict(sorted(event_counter.items())),
            "review_required_frames": review_frame_count,
            "review_required_ratio": round(review_frame_count / len(annotations), 4) if annotations else 0.0,
            "event_review_required_frames": event_review_frame_count,
            "event_review_required_ratio": round(event_review_frame_count / len(annotations), 4) if annotations else 0.0,
            "review_reason_distribution": dict(sorted(review_counter.items())),
            "event_review_reason_distribution": dict(sorted(event_review_counter.items())),
            "xodr_source_distribution": dict(sorted(xodr_source_counter.items())),
            "route_semantic_bucket_distribution": dict(sorted(route_bucket_counter.items())),
            "twoways_core_span_clipping": twoways_core_span_clipping,
            "twoways_longest_r2_filter": twoways_longest_r2_filter,
            "r4_context_recovery": r4_context_recovery,
            "blocked_signalized_tail_recovery": blocked_signalized_tail_recovery,
            "temporal_smoothing": temporal_smoothing_summary,
            "vehicle_turning_junction_gap_recovery": vehicle_turning_junction_gap_recovery,
            "accident_initial_no_junction_filter": accident_initial_no_junction_filter,
            "hazard_side_twoways_initial_filter": hazard_side_twoways_initial_filter,
            "hazard_side_initial_weak_junction_filter": hazard_side_initial_weak_junction_filter,
            "control_loss_initial_weak_junction_filter": control_loss_initial_weak_junction_filter,
            "signalized_enter_flow_initial_weak_r4_filter": signalized_enter_flow_initial_weak_r4_filter,
            "red_light_exit_tail_filter": red_light_exit_tail_filter,
            "route_junction_control_lock": route_junction_control_lock,
            "event_postprocess": event_postprocess_summary,
            "event_candidate_clamp": event_candidate_clamp,
            "final_junction_regular_gap_merge": final_junction_regular_gap_merge,
            "crossing_u4_single_span": crossing_u4_single_span,
            "confidence_stats": self._confidence_stats(annotations),
            "primary_rs_transitions": transition_frames[:50],
            "primary_event_transitions": event_transition_frames[:80],
            "annotations": annotations
        }
