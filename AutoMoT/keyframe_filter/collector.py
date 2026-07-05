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
        }

    counts: Dict[str, int] = defaultdict(int)
    has_tl = False
    has_stop = False
    has_yield = False
    has_junction = False
    for obj in _bbox_object_iter(raw) or []:
        cls = _bbox_class_text(obj)
        if not cls:
            continue
        counts[cls] += 1
        compact = cls.replace("_", "").replace("-", "").replace(" ", "")
        if "trafficlight" in compact or compact in {"tl", "light"}:
            has_tl = True
        if "stopsign" in compact or compact == "stop":
            has_stop = True
        if "yield" in compact or "giveway" in compact:
            has_yield = True
        if "junction" in compact or "intersection" in compact or "crosswalk" in compact:
            has_junction = True

    return {
        "bbox_available": True,
        "bbox_has_traffic_light": has_tl,
        "bbox_has_stop_sign": has_stop,
        "bbox_has_yield_sign": has_yield,
        "bbox_has_junction_hint": has_junction,
        "bbox_semantic_classes": dict(sorted(counts.items())),
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
    R6 = "R6"


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
    "R5": "无信号灯 / 信号灯失效 / 路权路口",
    "R6": "路边停车 / 停车占道道路结构",
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
    "U-E4": "行人 / 自行车横穿",
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
            'evidence': self.evidence,
            'event_evidence': self.event_evidence,
            'annotation_comment': self.annotation_comment,
        }


# ============================================================================
# 全局映射表
# ============================================================================

SCENARIO_TO_ROAD_STRUCTURE = {
    "Accident": [RoadStructure.R1, RoadStructure.R4],
    "AccidentTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "BlockedIntersection": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ConstructionObstacle": [RoadStructure.R1, RoadStructure.R4],
    "ConstructionObstacleTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "ControlLoss": [RoadStructure.R1, RoadStructure.R4],
    "CrossingBicycleFlow": [RoadStructure.R1, RoadStructure.R4],
    "CrossJunctionDefectTrafficLight": [RoadStructure.R1, RoadStructure.R5],
    "DynamicObjectCrossing": [RoadStructure.R1, RoadStructure.R4],
    "EnterActorFlow": [RoadStructure.R3],
    "EnterActorFlowV2": [RoadStructure.R3],
    "HardBreakRoute": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "HazardAtSideLane": [RoadStructure.R1, RoadStructure.R4],
    "HazardAtSideLaneTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "HighwayCutIn": [RoadStructure.R3, RoadStructure.R4],
    "HighwayExit": [RoadStructure.R3],
    "InterurbanActorFlow": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R5],
    "InterurbanAdvancedActorFlow": [RoadStructure.R1, RoadStructure.R5],
    "InvadingTurn": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R5],
    "MergerIntoSlowTraffic": [RoadStructure.R3, RoadStructure.R4],
    "MergerIntoSlowTrafficV2": [RoadStructure.R3],
    "NonSignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "noScenarios": [RoadStructure.R1, RoadStructure.R4],
    "OppositeVehicleRunningRedLight": [RoadStructure.R1, RoadStructure.R4],
    "OppositeVehicleTakingPriority": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "ParkedObstacle": [RoadStructure.R1, RoadStructure.R4],
    "ParkedObstacleTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "ParkingCrossingPedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "ParkingCutIn": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "ParkingExit": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "PedestrianCrossing": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "PriorityAtJunction": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "RedLightWithoutLeadVehicle": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R4],
    "StaticCutIn": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R6],
    "T_Junction": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "VehicleOpensDoorTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4, RoadStructure.R6],
    "VehicleTurningRoute": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "VehicleTurningRoutePedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
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
    "InvadingTurn",
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
    return set(base_allowed)

SCENARIO_TO_FINE_EVENTS = {
    "Accident": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "AccidentTwoWays": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "BlockedIntersection": [EventType.R_E1, EventType.R_E4, EventType.U_E1, EventType.U_E8],
    "ConstructionObstacle": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "ConstructionObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "ControlLoss": [EventType.R_E1, EventType.R_E4],
    "CrossingBicycleFlow": [EventType.R_E1, EventType.R_E4, EventType.U_E4],
    "CrossJunctionDefectTrafficLight": [EventType.R_E1, EventType.R_E5, EventType.U_E6, EventType.U_E7],
    "DynamicObjectCrossing": [EventType.R_E1, EventType.R_E4],
    "EnterActorFlow": [EventType.R_E1, EventType.R_E3],
    "EnterActorFlowV2": [EventType.R_E1, EventType.R_E3],
    "HardBreakRoute": [EventType.R_E1, EventType.R_E4, EventType.U_E1],
    "HazardAtSideLane": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "HazardAtSideLaneTwoWays": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "HighwayCutIn": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4],
    "HighwayExit": [EventType.R_E1, EventType.R_E2, EventType.R_E3],
    "InterurbanActorFlow": [EventType.R_E1, EventType.R_E2, EventType.R_E5],
    "InterurbanAdvancedActorFlow": [EventType.R_E1, EventType.R_E5],
    "InvadingTurn": [EventType.R_E1, EventType.R_E5, EventType.U_E5],
    "MergerIntoSlowTraffic": [EventType.R_E1, EventType.R_E3, EventType.R_E4],
    "MergerIntoSlowTrafficV2": [EventType.R_E1, EventType.R_E3],
    "NonSignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "noScenarios": [EventType.R_E1, EventType.R_E4],
    "OppositeVehicleRunningRedLight": [EventType.R_E1, EventType.R_E4, EventType.U_E6],
    "OppositeVehicleTakingPriority": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E7],
    "ParkedObstacle": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "ParkedObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "ParkingCrossingPedestrian": [EventType.R_E1, EventType.R_E4, EventType.U_E4],
    "ParkingCutIn": [EventType.R_E1, EventType.R_E4, EventType.U_E3],
    "ParkingExit": [EventType.R_E1, EventType.R_E2, EventType.R_E4],
    "PedestrianCrossing": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "PriorityAtJunction": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "RedLightWithoutLeadVehicle": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E4],
    "StaticCutIn": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4, EventType.U_E3],
    "T_Junction": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "VehicleOpensDoorTwoWays": [EventType.R_E1, EventType.R_E2, EventType.U_E2],
    "VehicleTurningRoute": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "VehicleTurningRoutePedestrian": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
}

OBSTACLE_EVENT_DISTANCE_FIELDS = {
    "Accident": ("dist_to_accident_site", 32.0),
    "AccidentTwoWays": ("dist_to_accident_site", 32.0),
    "ConstructionObstacle": ("dist_to_construction_site", 35.0),
    "ConstructionObstacleTwoWays": ("dist_to_construction_site", 35.0),
    "ParkedObstacle": ("dist_to_parked_obstacle", 28.0),
    "ParkedObstacleTwoWays": ("dist_to_parked_obstacle", 28.0),
    "HazardAtSideLane": ("dist_to_parked_obstacle", 30.0),
    "HazardAtSideLaneTwoWays": ("dist_to_parked_obstacle", 30.0),
    "VehicleOpensDoorTwoWays": ("dist_to_vehicle_opens_door", 28.0),
}

PEDESTRIAN_BICYCLE_EVENT_FIELDS = {
    "CrossingBicycleFlow": ("dist_to_biker", 22.0),
    "PedestrianCrossing": ("dist_to_pedestrian", 22.0),
    "ParkingCrossingPedestrian": ("dist_to_pedestrian", 24.0),
    "VehicleTurningRoute": ("dist_to_biker", 22.0),
    "VehicleTurningRoutePedestrian": ("dist_to_pedestrian", 22.0),
}

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
    "Accident": {"kind": "same_direction_obstacle", "junction_pre_m": 60, "junction_post_m": 25, "veto": ["no_r2", "no_r6"]},
    "ConstructionObstacle": {"kind": "same_direction_obstacle", "junction_pre_m": 60, "junction_post_m": 25, "veto": ["no_r2", "no_r6"]},
    "ParkedObstacle": {"kind": "same_direction_obstacle", "junction_pre_m": 60, "junction_post_m": 25, "veto": ["parked_not_parking_rs"]},
    # TwoWays：R2 只覆盖必须借/等对向的核心障碍片段；障碍前后普通双向道路回 R1/R4。
    "AccidentTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "ConstructionObstacleTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "HazardAtSideLaneTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 75, "two_way_post_pad_m": 20, "trigger_close_m": 75, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 20, "two_way_approach_obstacle_m": 30, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True},
    "ParkedObstacleTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 55, "two_way_post_pad_m": 20, "trigger_close_m": 70, "two_way_xml_core_close_m": 8, "two_way_obstacle_core_m": 18, "two_way_approach_obstacle_m": 28, "two_way_exit_delta_m": 2, "two_way_exit_hold_frames": 3, "two_way_post_core_signal_m": 45, "two_way_layout_prior": True, "veto": ["parked_not_r6"]},
    "InvadingTurn": {"kind": "invading_turn", "two_way_min_pre_m": 80, "two_way_post_pad_m": 20, "trigger_close_m": 75, "rule_note": "passive_oncoming_invasion"},
    # 阻塞路口：阻塞是 EVENT；RS 由路口控制源决定，STOP/无灯路口不能默认 R4。
    "BlockedIntersection": {"kind": "blocked_intersection", "junction_pre_m": 48, "junction_post_m": 20, "rule_note": "blocked_is_event_not_rs"},
    "OppositeVehicleRunningRedLight": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 20, "rule_note": "violation_not_r5"},
    "RedLightWithoutLeadVehicle": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 20},
    "SignalizedJunctionLeftTurn": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 25},
    "SignalizedJunctionLeftTurnEnterFlow": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 25, "veto": ["enter_flow_not_r3"]},
    "SignalizedJunctionRightTurn": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    "T_Junction": {"kind": "signalized_junction", "junction_pre_m": 50, "junction_post_m": 20, "review_if_no_tl": True},
    # 无灯/路权/故障路口。
    "CrossJunctionDefectTrafficLight": {"kind": "defect_junction", "junction_pre_m": 60, "junction_post_m": 20, "override": "r5_over_r4"},
    "NonSignalizedJunctionLeftTurn": {"kind": "nonsignalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    "NonSignalizedJunctionLeftTurnEnterFlow": {"kind": "nonsignalized_junction", "junction_pre_m": 60, "junction_post_m": 20, "veto": ["enter_flow_not_r3"]},
    "NonSignalizedJunctionRightTurn": {"kind": "nonsignalized_junction", "junction_pre_m": 45, "junction_post_m": 20},
    "OppositeVehicleTakingPriority": {"kind": "nonsignalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    "PriorityAtJunction": {"kind": "nonsignalized_junction", "junction_pre_m": 50, "junction_post_m": 20},
    # R3 高速/匝道/合流。
    "EnterActorFlow": {"kind": "highway_merge", "merge_pre_m": 30, "merge_post_m": 40, "trigger_close_m": 90, "highway_default_r3": True},
    "EnterActorFlowV2": {"kind": "highway_merge", "merge_pre_m": 30, "merge_post_m": 40, "trigger_close_m": 90, "highway_default_r3": True},
    "HighwayCutIn": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 40, "trigger_close_m": 90, "highway_default_r3": True},
    "HighwayExit": {"kind": "highway_merge", "merge_pre_m": 50, "merge_post_m": 50, "trigger_close_m": 90, "highway_default_r3": True},
    "MergerIntoSlowTraffic": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True, "actor_flow_near_m": 20, "highway_default_r3": True},
    "MergerIntoSlowTrafficV2": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True, "actor_flow_near_m": 20, "highway_default_r3": True},
    "InterurbanActorFlow": {"kind": "interurban", "merge_pre_m": 50, "merge_post_m": 45, "junction_pre_m": 55, "junction_post_m": 25},
    "InterurbanAdvancedActorFlow": {"kind": "interurban_advanced", "junction_pre_m": 55, "junction_post_m": 25, "r3_requires_topology": True},
    # 停车/路边占道。
    "ParkingCrossingPedestrian": {"kind": "parking", "parking_pre_m": 35, "parking_post_m": 60, "veto": ["pedestrian_not_rs"]},
    "ParkingCutIn": {"kind": "parking", "parking_pre_m": 30, "parking_post_m": 50},
    "ParkingExit": {"kind": "parking_exit", "parking_pre_m": 20, "parking_post_m": 60, "rule_note": "parking_to_driving_transition"},
    "VehicleOpensDoorTwoWays": {"kind": "vehicle_opens_door_twoways", "two_way_min_pre_m": 55, "two_way_post_pad_m": 20, "parking_pre_m": 35, "parking_post_m": 55},
    "StaticCutIn": {"kind": "static_cutin", "parking_pre_m": 35, "parking_post_m": 55, "merge_pre_m": 35, "merge_post_m": 55},
    # 按道路空间拆分的横穿/转弯/普通场景。
    "PedestrianCrossing": {"kind": "pedestrian_crossing", "junction_pre_m": 40, "junction_post_m": 40, "veto": ["pedestrian_not_rs"]},
    "VehicleTurningRoute": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 20, "multi_trigger": True},
    "VehicleTurningRoutePedestrian": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 40, "veto": ["pedestrian_not_rs"]},
    "CrossingBicycleFlow": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["actor_flow_not_r3"]},
    "DynamicObjectCrossing": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["crossing_event_not_rs"]},
    "ControlLoss": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["control_loss_not_rs"]},
    "HardBreakRoute": {"kind": "hardbreak_route", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["brake_not_rs"]},
    "HazardAtSideLane": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["side_lane_not_twoways"]},
    "noScenarios": {"kind": "noscenario", "junction_pre_m": 50, "junction_post_m": 25, "conservative": True},
}


JUNCTION_WINDOW_SCALE = 0.85
JUNCTION_META_NEAR_M = 45.0
JUNCTION_STRONG_MAX_M = 30.0
STATIC_SIGNAL_NEAR_M = 45.0
JUNCTION_CLOSE_TRIGGER_MAX_M = 35.0


def _shrink_junction_window(pre_m: float, post_m: float) -> Tuple[float, float]:
    """轻微收窄路口影响区，避免十字路口标签覆盖过早/过晚。"""
    pre = max(30.0, float(pre_m) * JUNCTION_WINDOW_SCALE)
    post = max(15.0, float(post_m) * JUNCTION_WINDOW_SCALE)
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
        weak_or_missing.append("r6_lacks_xodr_parking_or_shoulder_confirmation")
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
        decision_source = "twoways_trigger_window"
    elif primary == RoadStructure.R6:
        decision_source = "parking_or_curbside_window"
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
    if primary == RoadStructure.R6:
        checks.extend(["检查 XODR parking/shoulder 是否缺失", "检查 bbox/meta 是否有路边静态车证据但当前规则未接入"])

    return {
        "decision_source": decision_source,
        "used_inputs": used_inputs,
        "weak_or_missing_inputs": weak_or_missing,
        "window_flags": {
            "close_trigger": bool(flags.get("close_trigger")),
            "close_trigger_for_junction": bool(flags.get("close_trigger_for_junction")),
            "near_junction": bool(flags.get("near_junction")),
            "strong_control_context": bool(flags.get("strong_control_context")),
            "static_signal_near": bool(flags.get("static_signal_near")),
            "junction_window": bool(flags.get("junction_window")),
            "roundabout_context": bool(flags.get("map_is_roundabout")),
            "two_way_window": bool(flags.get("two_way_window")),
            "two_way_layout_prior": bool(flags.get("two_way_layout_prior")),
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
    def _regular_event(scenario_name: str, primary_rs: RoadStructure, frame_data: Dict[str, Any]) -> EventType:
        if primary_rs == RoadStructure.R4:
            return EventType.R_E4
        if primary_rs == RoadStructure.R5:
            return EventType.R_E5
        if scenario_name in R3_MERGE_SCENARIOS and primary_rs == RoadStructure.R3:
            return EventType.R_E3
        if scenario_name in {"HighwayCutIn", "HighwayExit", "InterurbanActorFlow", "ParkingExit", "StaticCutIn"}:
            if RoadEventRuleEngine._changed_route(frame_data):
                return EventType.R_E2
        if primary_rs == RoadStructure.R3:
            return EventType.R_E3
        if primary_rs in {RoadStructure.R1, RoadStructure.R2, RoadStructure.R6} and RoadEventRuleEngine._changed_route(frame_data):
            return EventType.R_E2
        return EventType.R_E1

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
        twoway = evidence.get("twoway_obstruction_evidence") or {}
        twoway_core = bool(
            (evidence.get("diagnostic_attribution", {}) or {})
            .get("window_flags", {})
            .get("twoway_core_obstruction", False)
        )
        close_specific_obstacle = dist <= threshold
        speed_obj_close_near_xml = speed_obj_dist <= threshold and near_trigger
        close_obstacle = close_specific_obstacle or speed_obj_close_near_xml
        hard_response = RoadEventRuleEngine._hard_decel(frame_data) or _safe_bool(frame_data.get("vehicle_hazard", False))
        door_open = scenario_name == "VehicleOpensDoorTwoWays" and _safe_bool(frame_data.get("vehicle_opened_door", False))
        active_window = near_trigger or primary_rs == RoadStructure.R2 or close_specific_obstacle or twoway_core or door_open
        hard_response_near_object = hard_response and speed_obj_dist <= threshold + 10.0 and (
            near_trigger or close_specific_obstacle or primary_rs == RoadStructure.R2
        )
        should = active_window and (close_obstacle or twoway_core or door_open or hard_response_near_object)
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
        if door_open:
            rules.append("event_vehicle_door_opened")
        metrics = {
            field: dist if math.isfinite(dist) else None,
            "speed_reduced_by_obj_distance": speed_obj_dist if math.isfinite(speed_obj_dist) else None,
            "twoway_obstruction": twoway,
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
        allowed = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [EventType.R_E1]))
        regular = RoadEventRuleEngine._regular_event(scenario_name, primary_rs, frame_data)
        if regular not in allowed:
            if primary_rs == RoadStructure.R4 and EventType.R_E4 in allowed:
                regular = EventType.R_E4
            elif primary_rs == RoadStructure.R5 and EventType.R_E5 in allowed:
                regular = EventType.R_E5
            else:
                regular = EventType.R_E1

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

        if unusual is None and scenario_name == "HardBreakRoute" and EventType.U_E1 in allowed:
            close_lead_stop = vehicle_hazard and speed_obj_dist <= 14.0 and RoadEventRuleEngine._speed(frame_data) <= 2.5
            if (scenario_active or speed_obj_dist <= 30.0) and (hard_decel or close_lead_stop):
                unusual = EventType.U_E1
                rules.extend(["event_hard_brake_response"])
        if unusual is None and scenario_name == "BlockedIntersection":
            if EventType.U_E8 in allowed and (scenario_active or near_trigger) and (
                _safe_bool(frame_data.get("slower_occluded_junction", False))
                or (RoadEventRuleEngine._speed(frame_data) <= 0.7 and _safe_bool(frame_data.get("brake", False)))
            ):
                unusual = EventType.U_E8
                rules.extend(["event_blocked_intersection_wait_or_clear"])
            elif EventType.U_E1 in allowed and hard_decel and speed_obj_dist <= 25.0:
                unusual = EventType.U_E1
                rules.extend(["event_blocked_intersection_lead_vehicle_decel"])
        if unusual is None and scenario_name in {"ParkingCutIn", "StaticCutIn"} and EventType.U_E3 in allowed:
            if (scenario_active or near_trigger or cutin_dist <= 35.0) and (
                cutin_dist <= 30.0 or _safe_bool(frame_data.get("brake_cutin", False)) or vehicle_hazard
            ):
                unusual = EventType.U_E3
                rules.extend(["event_dynamic_cutin_or_occupancy"])
        if unusual is None and scenario_name == "InvadingTurn" and EventType.U_E5 in allowed:
            if (scenario_active or near_trigger or primary_rs in {RoadStructure.R2, RoadStructure.R5}) and (
                vehicle_hazard or speed_obj_dist <= 35.0
            ):
                unusual = EventType.U_E5
                rules.extend(["event_oncoming_lane_invasion"])
        if unusual is None and scenario_name == "OppositeVehicleRunningRedLight" and EventType.U_E6 in allowed:
            if primary_rs == RoadStructure.R4 and (scenario_active or near_trigger) and (vehicle_hazard or speed_obj_dist <= 30.0 or hard_decel):
                unusual = EventType.U_E6
                rules.extend(["event_opposite_vehicle_running_red_light"])
        if unusual is None and scenario_name == "CrossJunctionDefectTrafficLight":
            if EventType.U_E7 in allowed and primary_rs == RoadStructure.R5 and (scenario_active or near_trigger or _safe_bool(frame_data.get("is_junction", False))):
                unusual = EventType.U_E7
                rules.extend(["event_defect_junction_rule_failure"])
                if EventType.U_E6 in allowed and vehicle_hazard:
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
                "changed_route": RoadEventRuleEngine._changed_route(frame_data),
                "signed_dist_to_lane_change": (
                    signed_dist
                    if math.isfinite(signed_dist := _safe_float(frame_data.get("signed_dist_to_lane_change"), default=math.inf))
                    else None
                ),
                "scenario_active": scenario_active,
                "near_trigger": near_trigger,
                "primary_road_structure": primary_rs.value,
            }
        )
        event_evidence = {
            "primary_event": primary_event.value,
            "events": [ev.value for ev in sorted(events, key=lambda ev: ev.value)],
            "regular_event": regular.value,
            "unusual_event": unusual.value if unusual else None,
            "secondary_unusual_events": [ev.value for ev in sorted(extra_events, key=lambda ev: ev.value)],
            "allowed_events": [ev.value for ev in allowed],
            "rules_fired": rules or ["event_regular_by_road_structure"],
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

    PRIORITY = [RoadStructure.R4, RoadStructure.R5, RoadStructure.R3, RoadStructure.R2, RoadStructure.R6, RoadStructure.R1]

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
        rgb_no_r4 = scenario_name in SCENARIOS_WITH_RGB_NO_R4
        if (
            (has_tl or light_hazard or bbox_traffic_light)
            and scenario_name != "CrossJunctionDefectTrafficLight"
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
        dist_to_junction_strong = dist_to_junction < min(junction_pre_window, JUNCTION_STRONG_MAX_M)
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
            and _safe_float(xodr.get("nearest_signal_m"), default=math.inf) <= STATIC_SIGNAL_NEAR_M
        )
        bbox_traffic_light_for_r4 = bbox_traffic_light
        static_signal_near_for_r4 = static_signal_near
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
        strong_control_context = _strong_control_context(
            is_junction=is_junction,
            xodr_near_junction=xodr_near_junction,
            stop_hazard=stop_hazard,
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
            JUNCTION_CLOSE_TRIGGER_MAX_M,
        ) and not map_is_roundabout and (
            not route_projection_error_high
            or trigger_distance < min(JUNCTION_CLOSE_TRIGGER_MAX_M, 25.0)
        )
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
        conservative_light_hazard_kind = kind in {"same_direction_obstacle", "default_meta_map"}
        if conservative_light_hazard_kind and not (meta_near_junction or stop_hazard):
            light_hazard_control_context = False
        elif (not conservative_light_hazard_kind) and static_signal_near and strong_control_context:
            light_hazard_control_context = True

        if rgb_no_r4 and (has_tl or light_hazard or bbox_traffic_light):
            rules.append("rgb_review_no_signalized_intersection_ignores_meta_tl")
        elif (not map_is_roundabout) and has_tl and strong_control_context:
            self._add(scores, RoadStructure.R4, 0.95)
            rules.append("r4_tl_confirmed")
        elif (not map_is_roundabout) and has_tl:
            if kind == "highway_merge":
                self._add(scores, RoadStructure.R4, 0.70)
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("r4_highway_meta_tl_without_control_context_demoted")
            else:
                self._add(scores, RoadStructure.R4, 0.86)
                self._add(scores, RoadStructure.R1, 0.62)
                rules.append("r4_meta_tl_without_strong_context_review")
        elif (not map_is_roundabout) and bbox_traffic_light_for_r4 and strong_control_context:
            self._add(scores, RoadStructure.R4, 0.90)
            rules.append("r4_bbox_traffic_light_confirmed")
        elif (not map_is_roundabout) and bbox_traffic_light_for_r4:
            if kind == "highway_merge":
                self._add(scores, RoadStructure.R4, 0.68)
                self._add(scores, RoadStructure.R3, 0.78)
                rules.append("r4_highway_bbox_tl_without_control_context_demoted")
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

        for note in cfg.get("veto", []):
            rules.append(str(note))
        if cfg.get("rule_note"):
            rules.append(str(cfg["rule_note"]))
        has_opposite = xodr_trusted and static_topology_strong and bool(xodr.get("has_opposite_driving_lane", False))
        same_dir_lanes = int(xodr.get("lane_count_same_dir", 1) or 1)
        has_parking = xodr_trusted and static_topology_strong and bool(xodr.get("has_parking_or_shoulder_nearby", False))
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

        if kind == "defect_junction":
            if junction_window:
                defect_score = 0.98 if near_junction or has_tl or static_signal_near else 0.74
                self._add(scores, RoadStructure.R5, defect_score)
                rules.append("defect_signal_overrides_R4")
                if defect_score < 0.90:
                    rules.append("defect_junction_window_without_strong_junction_evidence")
        elif kind == "signalized_junction":
            if junction_window or scenario_active:
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
            if junction_window or stop_hazard or scenario_active:
                if meta_near_junction or stop_hazard:
                    r5_score = 0.86
                elif xodr_near_junction:
                    r5_score = 0.74
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r5_static_xodr_only_junction_review")
                elif scenario_active and close_trigger_for_junction:
                    r5_score = 0.78
                    self._add(scores, RoadStructure.R1, 0.70)
                    rules.append("r5_active_close_trigger_without_stop_review")
                else:
                    r5_score = 0.56
                    self._add(scores, RoadStructure.R1, 0.78)
                    rules.append("r5_demoted_without_stop_yield_or_junction_evidence")
                self._add(scores, RoadStructure.R5, r5_score)
                rules.append("r5_nonsignalized_junction_window")
            if has_tl or static_signal_near:
                rules.append("nonsig_with_signal_conflict_review")
        elif kind in {"twoways_obstacle", "invading_turn"}:
            r2_layout_prior_allowed = (
                kind == "twoways_obstacle"
                and two_way_layout_prior_enabled
                and RoadStructure.R2 in allowed
                and not (xodr_trusted and not static_topology_only and static_topology_strong and not has_opposite and same_dir_lanes > 1)
            )
            if RoadStructure.R2 in allowed and two_way_window:
                r2_topology_confirmed = has_opposite and same_dir_lanes <= 1
                r2_core_meta_confirmed = (
                    kind == "twoways_obstacle"
                    and twoway_strict_core_confirmed
                    and scenario_active_for_structure
                )
                if (
                    (r2_topology_confirmed and (twoway_obstruction.core_confirmed or kind == "invading_turn"))
                    or r2_core_meta_confirmed
                    or twoway_xml_core_confirmed
                ):
                    self._add(scores, RoadStructure.R2, 0.90 if not twoway_xml_core_confirmed else 0.88)
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
                    if r2_layout_prior_allowed:
                        self._add(scores, RoadStructure.R2, 0.58)
                        self._add(scores, RoadStructure.R1, 0.78)
                        rules.append("r2_twoways_layout_prior_weak_non_core")
                        if not has_opposite:
                            rules.append("r2_layout_prior_lacks_xodr_opposite_confirmation")
                    else:
                        self._add(scores, RoadStructure.R2, 0.58)
                        self._add(scores, RoadStructure.R1, 0.76)
                        rules.append("r2_scenario_trigger_medium")
                    if r2_topology_confirmed:
                        rules.append("r2_waits_for_close_obstruction_or_vehicle_interaction")
                    else:
                        rules.append("r2_requires_visible_or_topology_occupancy_confirmation")
            elif r2_layout_prior_allowed:
                self._add(scores, RoadStructure.R2, 0.58)
                self._add(scores, RoadStructure.R1, 0.78)
                rules.append("r2_twoways_layout_prior_weak_non_core")
                if not has_opposite:
                    rules.append("r2_layout_prior_lacks_xodr_opposite_confirmation")
            if kind == "invading_turn":
                rules.append("r2_passive_invading_turn")
                if RoadStructure.R5 in allowed and junction_window and not map_is_roundabout:
                    if stop_hazard or is_junction:
                        self._add(scores, RoadStructure.R5, 0.86)
                        rules.append("invading_turn_nonsignalized_stop_or_junction_r5")
                    elif scenario_active and close_trigger_for_junction:
                        self._add(scores, RoadStructure.R5, 0.80)
                        rules.append("invading_turn_active_close_trigger_r5")
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
                    if stop_hazard and close_trigger:
                        self._add(scores, RoadStructure.R5, 0.82)
                        rules.append("invading_turn_stop_close_trigger_r5")
                    elif scenario_active and trigger_distance < min(trigger_close_m, 45.0):
                        self._add(scores, RoadStructure.R5, 0.80)
                        rules.append("invading_turn_active_near_trigger_r5")
            if kind == "twoways_obstacle" and RoadStructure.R4 in allowed and not twoway_xml_core_confirmed:
                post_core_signal_near = (
                    xodr_trusted
                    and static_topology_strong
                    and not map_is_roundabout
                    and _safe_float(xodr.get("nearest_signal_m"), default=math.inf) <= float(cfg.get("two_way_post_core_signal_m", 45.0))
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
            highway_default_r3 = bool(cfg.get("highway_default_r3"))
            merge_xml_fallback = bool(cfg.get("keep_r3_when_slow")) and (
                actor_flow_near
                or close_trigger
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
                    r3_score = 0.84 if (actor_flow_near or close_trigger) else 0.80
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
                    self._add(scores, RoadStructure.R5, 0.80)
                    rules.append("interurban_rgb_reviewed_active_close_trigger_r5")
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
            if parking_window and RoadStructure.R6 in allowed:
                if has_parking:
                    self._add(scores, RoadStructure.R6, 0.88)
                else:
                    self._add(scores, RoadStructure.R6, 0.50)
                    self._add(scores, RoadStructure.R1, 0.80)
                    rules.append("r6_requires_parking_or_curbside_confirmation")
                rules.append("r6_parking_context_window")
        elif kind == "vehicle_opens_door_twoways":
            door_window = two_way_window
            if door_window and RoadStructure.R2 in allowed:
                if has_opposite and same_dir_lanes <= 1:
                    self._add(scores, RoadStructure.R2, 0.88)
                else:
                    self._add(scores, RoadStructure.R2, 0.50)
                    self._add(scores, RoadStructure.R1, 0.80)
                    rules.append("vehicle_open_door_r2_lacks_opposite_confirmation")
                rules.append("vehicle_open_door_r2_possible")
            if door_window and RoadStructure.R6 in allowed:
                if has_parking:
                    self._add(scores, RoadStructure.R6, 0.82)
                else:
                    self._add(scores, RoadStructure.R6, 0.48)
                    self._add(scores, RoadStructure.R1, 0.80)
                    rules.append("vehicle_open_door_r6_lacks_parking_confirmation")
                rules.append("vehicle_open_door_r6_parking_context")
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
            if cutin_window and has_parking and RoadStructure.R6 in allowed:
                self._add(scores, RoadStructure.R6, 0.84)
                rules.append("static_cutin_r6_parking_side")
            elif cutin_window and ramp_hint and RoadStructure.R3 in allowed:
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
            if not has_tl and not light_hazard:
                scores = {RoadStructure.R1: max(scores.get(RoadStructure.R1, 0.0), 0.86)}
                rules.append("noscenario_conservative_r1_without_meta_light")
        else:
            # SameDirectionObstacle 与 DefaultMetaMapPolicy 都只允许灯态把主标签提升到 R4。
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

        if scenario_name == "CrossJunctionDefectTrafficLight" and RoadStructure.R5 in scores:
            primary = RoadStructure.R5
        elif kind == "noscenario" and RoadStructure.R4 not in scores:
            primary = RoadStructure.R1
        else:
            max_score = max(scores.values())
            primary = max(scores, key=lambda rs: (scores[rs], -self.PRIORITY.index(rs) if rs in self.PRIORITY else -99))
            # 分数接近时按全局优先级仲裁，但视觉复核显示弱特殊 RS 不能低分压过 R1。
            for rs in self.PRIORITY:
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
            if rs != primary and score >= 0.60 and (primary in {RoadStructure.R4, RoadStructure.R5} or rs in {RoadStructure.R2, RoadStructure.R3, RoadStructure.R6})
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
                    and not (xodr_trusted and not static_topology_only and static_topology_strong and not has_opposite and same_dir_lanes > 1)
                ),
                "twoway_core_obstruction": twoway_obstruction.core_confirmed
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else False,
                "twoway_strict_core_obstruction": twoway_strict_core_confirmed
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else False,
                "twoway_xml_core_close": twoway_xml_core_close,
                "twoway_xml_obstacle_close": twoway_xml_obstacle_close,
                "twoway_xml_core_confirmed": twoway_xml_core_confirmed,
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
        if primary in {RoadStructure.R2, RoadStructure.R3, RoadStructure.R6} and weak_inputs:
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
            "actor_flow_distance_m": actor_flow_distance if math.isfinite(actor_flow_distance) else None,
            "junction_window_config": {
                "junction_pre_m": junction_pre,
                "junction_post_m": junction_post,
                "effective_pre_m": round(junction_pre_window, 3),
                "effective_post_m": round(junction_post_window, 3),
                "scale": JUNCTION_WINDOW_SCALE,
                "meta_near_m": JUNCTION_META_NEAR_M,
                "strong_max_m": JUNCTION_STRONG_MAX_M,
                "static_signal_near_m": STATIC_SIGNAL_NEAR_M,
                "close_trigger_max_m": JUNCTION_CLOSE_TRIGGER_MAX_M,
            },
            "traffic_light_state": str(tl) if tl is not None else None,
            "bbox_semantics": {
                "available": _safe_bool(frame_data.get("bbox_available", False)),
                "traffic_light": bbox_traffic_light,
                "stop_sign": bbox_stop_sign,
                "yield_sign": bbox_yield_sign,
                "junction_hint": bbox_junction_hint,
                "classes": frame_data.get("bbox_semantic_classes", {}),
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
            and scenario_name != "CrossJunctionDefectTrafficLight"
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
        for route_dir in discovered_route_dirs:
            should_exclude, info = is_abnormal_lead_route(route_dir, scenario_name)
            if should_exclude:
                abnormal_skipped.append(info)
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
            f"异常时长剔除 {len(abnormal_skipped)} 个, 将采集 {len(route_dirs)} 个"
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

        def _route_change_hint(ann: Dict[str, Any], *, allow_abs: bool = True) -> bool:
            metrics = _event_metrics(ann)
            signed = _signed_lane_change(ann)
            if bool(metrics.get("changed_route")):
                return True
            return allow_abs and math.isfinite(signed) and abs(signed) <= 4.5

        def _return_lane_change_hint(ann: Dict[str, Any]) -> bool:
            signed = _signed_lane_change(ann)
            return math.isfinite(signed) and signed <= -1.0

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

        def _regular_event_for_annotation(ann: Dict[str, Any]) -> EventType:
            rs = ann.get("primary_road_structure")
            if rs == RoadStructure.R4.value:
                return EventType.R_E4
            if rs == RoadStructure.R5.value:
                return EventType.R_E5
            if rs == RoadStructure.R3.value:
                return EventType.R_E3
            return EventType.R_E1

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
                if has_specific_obstacle or has_near_route_change:
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
                return_start: Optional[int] = None
                for j in range(start + 6, end):
                    if _return_lane_change_hint(annotations[j]) and (
                        _trigger_distance(annotations[j]) > 45.0
                        or not _obstacle_still_close(annotations[j], pad_m=-2.0)
                    ):
                        return_start = j
                        break
                if return_start is None:
                    continue
                return_end = return_start
                hold_without_hint = 0
                while return_end < end:
                    ann = annotations[return_end]
                    if _route_change_hint(ann) or _return_lane_change_hint(ann):
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

            last_u2_end = None
            for idx, ann in enumerate(annotations):
                primary = ann.get("primary_event")
                if primary == EventType.U_E2.value:
                    last_u2_end = idx
                    continue
                if last_u2_end is None:
                    continue
                if idx - last_u2_end > 16:
                    last_u2_end = None
                    continue
                rs = ann.get("primary_road_structure")
                if (
                    primary == EventType.R_E1.value
                    and rs not in {RoadStructure.R4.value, RoadStructure.R5.value}
                    and _route_change_hint(ann)
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
                label = annotations[idx].get("primary_event")
                start = idx
                while idx < len(annotations) and annotations[idx].get("primary_event") == label:
                    idx += 1
                end = idx
                if label != EventType.R_E2.value or end - start > 2:
                    continue
                prev_label = annotations[start - 1].get("primary_event") if start > 0 else None
                next_label = annotations[end].get("primary_event") if end < len(annotations) else None
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
        for ann in annotations:
            ann["frame_event_annotation"] = self._frame_event_annotation_payload(ann)
        return {"enabled": True, "changes": changes}

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
            "R6": 4,
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

    def _apply_temporal_rs_smoothing(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """通用 RS 去抖：所有短片段都必须持续足够久才作为真实结构切换。"""
        changes: List[Dict[str, Any]] = []
        if len(annotations) < 3:
            return {"enabled": True, "changes": changes}

        runs = self._rs_runs(annotations)
        for run_index, run in enumerate(runs):
            label = run.get("label")
            if not label:
                continue
            min_frames = self._min_rs_segment_frames(str(label))
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
                chosen = max(neighbor_options, key=lambda r: (int(r.get("length", 0)), -abs(int(r.get("start", 0)) - int(run.get("start", 0)))))
                replacement = str(chosen["label"])
                inherited_from = "previous_neighbor" if chosen is prev_run else "next_neighbor"
            if replacement is None or replacement == label:
                continue
            run_annotations = annotations[int(run["start"]): int(run["end"])]
            if not self._can_temporal_smoothing_promote(run_annotations, str(label), replacement):
                continue
            change = {
                "start_frame": annotations[int(run["start"])].get("frame_id"),
                "end_frame": annotations[int(run["end"]) - 1].get("frame_id"),
                "from": label,
                "to": replacement,
                "length": int(run["length"]),
                "min_frames": min_frames,
                "inherited_from": inherited_from,
            }
            changes.append(change)
            for ann in run_annotations:
                self._rewrite_rs_label(ann, replacement, reason, inherited_from)

        return {"enabled": True, "min_frames": {"R1": 2, "R2": 4, "R3": 4, "R4": 4, "R5": 4, "R6": 4}, "changes": changes}

    def _fallback_after_twoways_core(self, ann: Dict[str, Any]) -> str:
        """TwoWays 核心段结束后回到真实路网标签：有强 R4 证据才回 R4，否则 R1。"""
        candidates = ann.get("road_structure_candidates", {}) or {}
        evidence = ann.get("evidence", {}) or {}
        rules = set(evidence.get("rules_fired", []) or [])
        r4_score = float(candidates.get("R4", 0.0) or 0.0)
        if r4_score >= 0.82 and any(rule.startswith("r4_") or rule.startswith("twoways_post_core") for rule in rules):
            return "R4"
        return "R1"

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

    def _apply_twoways_core_span_clipping(
        self,
        scenario_name: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """把 TwoWays R2 限定在局部核心段，避免绕过障碍后整段继续 R2。"""
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
        """TwoWays route 只保留最长 R2 核心段，清理偶发短 R2 扰动。"""
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {})
        if cfg.get("kind") not in {"twoways_obstacle", "vehicle_opens_door_twoways"} or not annotations:
            return {"enabled": False, "changes": []}

        r2_runs = [run for run in self._rs_runs(annotations) if run.get("label") == "R2"]
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
    def _can_temporal_smoothing_promote(run_annotations: List[Dict[str, Any]], old_label: str, replacement: str) -> bool:
        """避免把只有弱证据的普通路段，因邻居继承提升成特殊 ROAD_STRUCTURE。"""
        if old_label == "R4" and replacement != "R4":
            for ann in run_annotations:
                rules = set((ann.get("evidence", {}) or {}).get("rules_fired", []) or [])
                if any(rule.startswith("r4_") for rule in rules):
                    return False
        if old_label == "R5" and replacement == "R1":
            for ann in run_annotations:
                rules = set((ann.get("evidence", {}) or {}).get("rules_fired", []) or [])
                if any(
                    rule.startswith("invading_turn_") and rule.endswith("_r5")
                    for rule in rules
                ):
                    return False
        if old_label != "R1" or replacement not in {"R2", "R3", "R4", "R5", "R6"}:
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
            if replacement == "R6" and not any(rule.startswith("r6_") for rule in rules):
                return False
        return True

    def _process_route(self, scenario_name: str, route_path: Path, max_frames_per_route: Optional[int] = None) -> Dict:
        """处理单个route"""
        metas_dir = route_path / "metas"
        if not metas_dir.exists():
            return {"route_id": route_path.name, "status": "skip", "num_frames": 0}

        xml_info = self.xml_index.match(scenario_name, route_path.name)
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
        temporal_smoothing_summary = self._apply_temporal_rs_smoothing(annotations)
        event_postprocess_summary = self._apply_event_route_postprocess(scenario_name, annotations)

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
            "temporal_smoothing": temporal_smoothing_summary,
            "event_postprocess": event_postprocess_summary,
            "confidence_stats": self._confidence_stats(annotations),
            "primary_rs_transitions": transition_frames[:50],
            "primary_event_transitions": event_transition_frames[:80],
            "annotations": annotations
        }
