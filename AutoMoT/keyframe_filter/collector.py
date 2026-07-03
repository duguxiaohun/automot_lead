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

    def to_dict(self):
        return {
            'frame_id': self.frame_id,
            'road_structures': [rs.value for rs in self.road_structures],
            'events': [ev.value for ev in self.events],
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
            'annotation_comment': self.annotation_comment,
        }


# ============================================================================
# 全局映射表
# ============================================================================

SCENARIO_TO_ROAD_STRUCTURE = {
    "Accident": [RoadStructure.R1, RoadStructure.R4],
    "AccidentTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "BlockedIntersection": [RoadStructure.R1, RoadStructure.R4],
    "ConstructionObstacle": [RoadStructure.R1, RoadStructure.R4],
    "ConstructionObstacleTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "ControlLoss": [RoadStructure.R1, RoadStructure.R4],
    "CrossingBicycleFlow": [RoadStructure.R1, RoadStructure.R4],
    "CrossJunctionDefectTrafficLight": [RoadStructure.R1, RoadStructure.R5],
    "DynamicObjectCrossing": [RoadStructure.R1, RoadStructure.R4],
    "EnterActorFlow": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "EnterActorFlowV2": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "HardBreakRoute": [RoadStructure.R1, RoadStructure.R4],
    "HazardAtSideLane": [RoadStructure.R1, RoadStructure.R4],
    "HazardAtSideLaneTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "HighwayCutIn": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "HighwayExit": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "InterurbanActorFlow": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R5],
    "InterurbanAdvancedActorFlow": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "InvadingTurn": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "MergerIntoSlowTraffic": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "MergerIntoSlowTrafficV2": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4],
    "NonSignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R5],
    "NonSignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R5],
    "noScenarios": [RoadStructure.R1, RoadStructure.R4],
    "OppositeVehicleRunningRedLight": [RoadStructure.R1, RoadStructure.R4],
    "OppositeVehicleTakingPriority": [RoadStructure.R1, RoadStructure.R5],
    "ParkedObstacle": [RoadStructure.R1, RoadStructure.R4],
    "ParkedObstacleTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4],
    "ParkingCrossingPedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "ParkingCutIn": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "ParkingExit": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R6],
    "PedestrianCrossing": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "PriorityAtJunction": [RoadStructure.R1, RoadStructure.R5],
    "RedLightWithoutLeadVehicle": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurn": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionLeftTurnEnterFlow": [RoadStructure.R1, RoadStructure.R4],
    "SignalizedJunctionRightTurn": [RoadStructure.R1, RoadStructure.R4],
    "StaticCutIn": [RoadStructure.R1, RoadStructure.R3, RoadStructure.R4, RoadStructure.R6],
    "T_Junction": [RoadStructure.R1, RoadStructure.R4],
    "VehicleOpensDoorTwoWays": [RoadStructure.R1, RoadStructure.R2, RoadStructure.R4, RoadStructure.R6],
    "VehicleTurningRoute": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
    "VehicleTurningRoutePedestrian": [RoadStructure.R1, RoadStructure.R4, RoadStructure.R5],
}

SCENARIO_TO_FINE_EVENTS = {
    "Accident": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "AccidentTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "BlockedIntersection": [EventType.R_E1, EventType.R_E4, EventType.U_E8],
    "ConstructionObstacle": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "ConstructionObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "ControlLoss": [EventType.R_E1, EventType.R_E4],
    "CrossingBicycleFlow": [EventType.R_E1, EventType.R_E4, EventType.U_E4],
    "CrossJunctionDefectTrafficLight": [EventType.R_E1, EventType.R_E5, EventType.U_E7],
    "DynamicObjectCrossing": [EventType.R_E1, EventType.R_E4, EventType.U_E3, EventType.U_E4],
    "EnterActorFlow": [EventType.R_E1, EventType.R_E3, EventType.R_E4],
    "EnterActorFlowV2": [EventType.R_E1, EventType.R_E3, EventType.R_E4],
    "HardBreakRoute": [EventType.R_E1, EventType.R_E4, EventType.U_E1],
    "HazardAtSideLane": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "HazardAtSideLaneTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "HighwayCutIn": [EventType.R_E1, EventType.R_E3, EventType.R_E4, EventType.U_E3],
    "HighwayExit": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4],
    "InterurbanActorFlow": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4, EventType.R_E5],
    "InterurbanAdvancedActorFlow": [EventType.R_E1, EventType.R_E4, EventType.R_E5],
    "InvadingTurn": [EventType.R_E1, EventType.R_E4, EventType.U_E5],
    "MergerIntoSlowTraffic": [EventType.R_E1, EventType.R_E3, EventType.R_E4],
    "MergerIntoSlowTrafficV2": [EventType.R_E1, EventType.R_E3, EventType.R_E4],
    "NonSignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E5],
    "NonSignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E5],
    "noScenarios": [EventType.R_E1, EventType.R_E4],
    "OppositeVehicleRunningRedLight": [EventType.R_E1, EventType.R_E4, EventType.U_E6],
    "OppositeVehicleTakingPriority": [EventType.R_E1, EventType.R_E5],
    "ParkedObstacle": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "ParkedObstacleTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "ParkingCrossingPedestrian": [EventType.R_E1, EventType.R_E4, EventType.U_E4],
    "ParkingCutIn": [EventType.R_E1, EventType.R_E4, EventType.U_E3],
    "ParkingExit": [EventType.R_E1, EventType.R_E2, EventType.R_E4],
    "PedestrianCrossing": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "PriorityAtJunction": [EventType.R_E1, EventType.R_E5],
    "RedLightWithoutLeadVehicle": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurn": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionLeftTurnEnterFlow": [EventType.R_E1, EventType.R_E4],
    "SignalizedJunctionRightTurn": [EventType.R_E1, EventType.R_E4],
    "StaticCutIn": [EventType.R_E1, EventType.R_E2, EventType.R_E3, EventType.R_E4, EventType.U_E3],
    "T_Junction": [EventType.R_E1, EventType.R_E4],
    "VehicleOpensDoorTwoWays": [EventType.R_E1, EventType.R_E2, EventType.R_E4, EventType.U_E2],
    "VehicleTurningRoute": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
    "VehicleTurningRoutePedestrian": [EventType.R_E1, EventType.R_E4, EventType.R_E5, EventType.U_E4],
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


def _route_key_aliases(text: str) -> Set[str]:
    name = Path(str(text)).stem
    out = {name.lower()}
    run_match = re.match(
        r"^(?P<town>Town\d+(?:HD)?)_Rep\d+_(?P<key>.+)_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$",
        name,
        re.IGNORECASE,
    )
    if run_match:
        raw = run_match.group("key")
        if raw.endswith("_route0"):
            raw = raw[: -len("_route0")]
        out.add(raw.lower())
        out.add(f"{run_match.group('town')}_route_{raw}".lower())
    xml_match = re.match(r"^(?P<town>Town\d+(?:HD)?)_route_(?P<key>.+)$", name, re.IGNORECASE)
    if xml_match:
        raw = xml_match.group("key")
        out.add(raw.lower())
        out.add(f"{xml_match.group('town')}_Rep0_{raw}".lower())
    return out


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
        self.by_route_num: Dict[Tuple[str, str], RouteXmlInfo] = {}
        self._build()

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
                for key in {xml_path.stem, info.route_id}:
                    route_num = _extract_route_num(key)
                    if route_num:
                        self.by_route_num[(scenario, route_num)] = info

    def match(self, scenario: str, route_name: str) -> Optional[RouteXmlInfo]:
        candidates = self.by_scenario.get(scenario, [])
        for info in candidates:
            if info.path.stem and info.path.stem in route_name:
                return info
            if _route_key_aliases(info.path.stem) & _route_key_aliases(route_name):
                return info
        route_num = _extract_route_num(route_name)
        if route_num:
            hit = self.by_route_num.get((scenario, route_num))
            if hit is not None:
                return hit
        if not candidates:
            return None
        for info in candidates:
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
    "BlockedIntersection": "signalized_junction",
    "ConstructionObstacle": "same_direction_obstacle",
    "ConstructionObstacleTwoWays": "twoways_obstacle",
    "ControlLoss": "default_meta_map",
    "CrossingBicycleFlow": "default_meta_map",
    "CrossJunctionDefectTrafficLight": "defect_junction",
    "DynamicObjectCrossing": "default_meta_map",
    "EnterActorFlow": "highway_merge",
    "EnterActorFlowV2": "highway_merge",
    "HardBreakRoute": "default_meta_map",
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
    # TwoWays：必须在 trigger/active/opposite-lane 窗口内才 R2。
    "AccidentTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 45, "two_way_post_pad_m": 20, "trigger_close_m": 70},
    "ConstructionObstacleTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 45, "two_way_post_pad_m": 20, "trigger_close_m": 70},
    "HazardAtSideLaneTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 70, "two_way_post_pad_m": 20, "trigger_close_m": 75},
    "ParkedObstacleTwoWays": {"kind": "twoways_obstacle", "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "trigger_close_m": 70, "veto": ["parked_not_r6"]},
    "InvadingTurn": {"kind": "invading_turn", "two_way_min_pre_m": 80, "two_way_post_pad_m": 20, "trigger_close_m": 75, "rule_note": "passive_oncoming_invasion"},
    # 信号灯路口：stopline 前也保持 R4。
    "BlockedIntersection": {"kind": "signalized_junction", "junction_pre_m": 60, "junction_post_m": 25, "rule_note": "blocked_is_event_not_rs"},
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
    "EnterActorFlow": {"kind": "highway_merge", "merge_pre_m": 30, "merge_post_m": 40, "trigger_close_m": 90},
    "EnterActorFlowV2": {"kind": "highway_merge", "merge_pre_m": 30, "merge_post_m": 40, "trigger_close_m": 90},
    "HighwayCutIn": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 40, "trigger_close_m": 90},
    "HighwayExit": {"kind": "highway_merge", "merge_pre_m": 50, "merge_post_m": 50, "trigger_close_m": 90},
    "MergerIntoSlowTraffic": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True},
    "MergerIntoSlowTrafficV2": {"kind": "highway_merge", "merge_pre_m": 40, "merge_post_m": 50, "trigger_close_m": 90, "keep_r3_when_slow": True},
    "InterurbanActorFlow": {"kind": "interurban", "merge_pre_m": 50, "merge_post_m": 45, "junction_pre_m": 55, "junction_post_m": 25},
    "InterurbanAdvancedActorFlow": {"kind": "interurban_advanced", "junction_pre_m": 55, "junction_post_m": 25, "r3_requires_topology": True},
    # 停车/路边占道。
    "ParkingCrossingPedestrian": {"kind": "parking", "parking_pre_m": 35, "parking_post_m": 60, "veto": ["pedestrian_not_rs"]},
    "ParkingCutIn": {"kind": "parking", "parking_pre_m": 30, "parking_post_m": 50},
    "ParkingExit": {"kind": "parking_exit", "parking_pre_m": 20, "parking_post_m": 60, "rule_note": "parking_to_driving_transition"},
    "VehicleOpensDoorTwoWays": {"kind": "vehicle_opens_door_twoways", "two_way_min_pre_m": 50, "two_way_post_pad_m": 20, "parking_pre_m": 35, "parking_post_m": 55},
    "StaticCutIn": {"kind": "static_cutin", "parking_pre_m": 35, "parking_post_m": 55, "merge_pre_m": 35, "merge_post_m": 55},
    # 按道路空间拆分的横穿/转弯/普通场景。
    "PedestrianCrossing": {"kind": "pedestrian_crossing", "junction_pre_m": 40, "junction_post_m": 40, "veto": ["pedestrian_not_rs"]},
    "VehicleTurningRoute": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 20, "multi_trigger": True},
    "VehicleTurningRoutePedestrian": {"kind": "vehicle_turning", "junction_pre_m": 50, "junction_post_m": 40, "veto": ["pedestrian_not_rs"]},
    "CrossingBicycleFlow": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["actor_flow_not_r3"]},
    "DynamicObjectCrossing": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["crossing_event_not_rs"]},
    "ControlLoss": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["control_loss_not_rs"]},
    "HardBreakRoute": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["brake_not_rs"]},
    "HazardAtSideLane": {"kind": "default_meta_map", "junction_pre_m": 50, "junction_post_m": 25, "veto": ["side_lane_not_twoways"]},
    "noScenarios": {"kind": "noscenario", "junction_pre_m": 50, "junction_post_m": 25, "conservative": True},
}


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
        "meta_junction_hint": bool(flags.get("is_junction")) or bool(flags.get("dist_to_junction_near")),
        "xodr_roundabout_hint": bool(flags.get("map_is_roundabout")),
        "meta_active_scenario": bool(flags.get("scenario_active")),
        "meta_stop_hint": bool(flags.get("stop_hazard")),
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
        used_inputs["meta_traffic_light_valid"] or used_inputs["meta_light_hazard"] or used_inputs["xodr_runtime_available"]
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
            "near_junction": bool(flags.get("near_junction")),
            "strong_control_context": bool(flags.get("strong_control_context")),
            "static_signal_near": bool(flags.get("static_signal_near")),
            "junction_window": bool(flags.get("junction_window")),
            "roundabout_context": bool(flags.get("map_is_roundabout")),
            "two_way_window": bool(flags.get("two_way_window")),
            "twoway_core_obstruction": bool(flags.get("twoway_core_obstruction")),
            "scenario_active": bool(flags.get("scenario_active")),
        },
        "score_ranking": sorted_scores,
        "top_score_gap": score_gap,
        "if_this_frame_is_wrong_check": checks,
    }


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
    return (
        is_junction
        or xodr_near_junction
        or stop_hazard
        or (static_signal_near and dist_to_junction < min(junction_pre, 35.0))
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
    ) -> Tuple[RoadStructure, Set[RoadStructure], Dict[str, float], Dict[str, Any], float, str]:
        allowed = set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [RoadStructure.R1]))
        scores: Dict[RoadStructure, float] = {RoadStructure.R1: 0.35}
        rules: List[str] = ["r1_default_candidate"]

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
        is_junction = _safe_bool(frame_data.get("is_junction", False)) or _safe_bool(frame_data.get("is_intersection", False))
        map_is_junction = bool(xodr.get("map_is_junction", False))
        map_is_roundabout = bool(xodr.get("map_is_roundabout", False))
        dist_to_junction = _finite_min(frame_data.get("dist_to_junction"), frame_data.get("distance_to_next_junction"))
        stop_hazard = _safe_bool(frame_data.get("stop_sign_hazard", False)) or _safe_bool(frame_data.get("stop_sign_close", False))
        cfg = SCENARIO_RULE_CONFIG.get(scenario_name, {"kind": SCENARIO_RULE_KIND.get(scenario_name, "default_meta_map")})
        active = str(frame_data.get("current_active_scenario_type", "") or "")
        scenario_active = scenario_name in active or active in {scenario_name, scenario_name.replace("V2", "")}
        trigger_close_m = float(cfg.get("trigger_close_m", 70.0))
        close_trigger = trigger_distance < trigger_close_m
        xodr_trusted = bool(xodr.get("xodr_topology_trusted", xodr.get("xodr_available", False)))
        xodr_source = str(xodr.get("xodr_source", ""))
        static_topology_only = xodr_source == "static_xodr"
        dist_to_junction_near = dist_to_junction < 55.0
        xml_distance = _xml_numeric(xml_info, "distance", default=50.0)
        two_way_pre = max(xml_distance, float(cfg.get("two_way_min_pre_m", 45.0)))
        two_way_post = two_way_pre + float(cfg.get("two_way_post_pad_m", 20.0))
        junction_pre = float(cfg.get("junction_pre_m", 60.0))
        junction_post = float(cfg.get("junction_post_m", 25.0))
        dist_to_junction_strong = dist_to_junction < min(junction_pre, 35.0)
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
            and _safe_float(xodr.get("nearest_signal_m"), default=math.inf) <= 60.0
        )
        meta_near_junction = is_junction or dist_to_junction_strong
        xodr_near_junction = xodr_trusted and static_topology_strong and map_is_junction
        near_junction = (meta_near_junction or xodr_near_junction) and not map_is_roundabout
        strong_control_context = _strong_control_context(
            is_junction=is_junction,
            xodr_near_junction=xodr_near_junction,
            stop_hazard=stop_hazard,
            static_signal_near=static_signal_near,
            dist_to_junction=dist_to_junction,
            junction_pre=junction_pre,
        )
        close_trigger_for_structure = close_trigger and (
            not route_projection_error_high
            or trigger_distance < min(trigger_close_m, 25.0)
        )
        scenario_active_for_structure = scenario_active and not route_projection_error_high

        two_way_window = (
            _route_trigger_window(route_s_for_window, trigger_s, two_way_pre, two_way_post)
            or close_trigger_for_structure
            or scenario_active_for_structure
        )
        junction_window = (
            near_junction
            or (_route_trigger_window(route_s_for_window, trigger_s, junction_pre, junction_post) and not map_is_roundabout)
            or (close_trigger_for_structure and not map_is_roundabout)
        )

        if map_is_roundabout:
            self._add(scores, RoadStructure.R1, 0.92)
            rules.append("roundabout_xodr_forces_r1")

        twoway_obstruction = _twoway_obstruction_evidence(frame_data)

        if (not map_is_roundabout) and has_tl and strong_control_context:
            self._add(scores, RoadStructure.R4, 0.95)
            rules.append("r4_tl_confirmed")
        elif (not map_is_roundabout) and has_tl:
            self._add(scores, RoadStructure.R4, 0.62)
            self._add(scores, RoadStructure.R1, 0.76)
            rules.append("r4_tl_seen_without_strong_junction_context")
        elif (not map_is_roundabout) and light_hazard and (near_junction or static_signal_near):
            self._add(scores, RoadStructure.R4, 0.90)
            rules.append("r4_light_hazard")
        elif light_hazard:
            rules.append("light_hazard_ignored_without_junction_context")
        elif (not map_is_roundabout) and static_signal_near and strong_control_context:
            self._add(scores, RoadStructure.R4, 0.74)
            rules.append("r4_static_xodr_signal_near")
        elif (not map_is_roundabout) and static_signal_near:
            self._add(scores, RoadStructure.R1, 0.76)
            rules.append("r4_static_signal_without_visual_junction_demoted")

        kind = str(cfg.get("kind", SCENARIO_RULE_KIND.get(scenario_name, "default_meta_map")))
        for note in cfg.get("veto", []):
            rules.append(str(note))
        if cfg.get("rule_note"):
            rules.append(str(cfg["rule_note"]))
        has_opposite = xodr_trusted and static_topology_strong and bool(xodr.get("has_opposite_driving_lane", False))
        same_dir_lanes = int(xodr.get("lane_count_same_dir", 1) or 1)
        has_parking = xodr_trusted and static_topology_strong and bool(xodr.get("has_parking_or_shoulder_nearby", False))
        ramp_hint = xodr_trusted and static_topology_strong and bool(xodr.get("ramp_merge_split_hint", False))
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
                elif near_junction or static_signal_near:
                    r4_score = 0.82
                else:
                    r4_score = 0.66
                    self._add(scores, RoadStructure.R1, 0.76)
                    rules.append("r4_signalized_window_lacks_visualizable_junction_evidence")
                self._add(scores, RoadStructure.R4, r4_score)
                rules.append("r4_signalized_scenario_window")
        elif kind == "nonsignalized_junction":
            if junction_window or stop_hazard or scenario_active:
                if meta_near_junction or stop_hazard:
                    r5_score = 0.86
                elif xodr_near_junction:
                    r5_score = 0.74
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r5_static_xodr_only_junction_review")
                else:
                    r5_score = 0.66
                    self._add(scores, RoadStructure.R1, 0.76)
                    rules.append("r5_window_lacks_strong_junction_evidence")
                self._add(scores, RoadStructure.R5, r5_score)
                rules.append("r5_nonsignalized_junction_window")
            if has_tl or static_signal_near:
                rules.append("nonsig_with_signal_conflict_review")
        elif kind in {"twoways_obstacle", "invading_turn"}:
            if RoadStructure.R2 in allowed and two_way_window:
                r2_topology_confirmed = has_opposite and same_dir_lanes <= 1
                r2_core_meta_confirmed = (
                    kind == "twoways_obstacle"
                    and twoway_obstruction.core_confirmed
                    and scenario_active_for_structure
                )
                if (r2_topology_confirmed and (twoway_obstruction.core_confirmed or kind == "invading_turn")) or r2_core_meta_confirmed:
                    self._add(scores, RoadStructure.R2, 0.90)
                    if r2_topology_confirmed:
                        rules.append("r2_opposite_lane_confirmed")
                    else:
                        rules.append("r2_core_obstruction_meta_confirmed_without_trusted_xodr")
                    if kind == "twoways_obstacle":
                        rules.append("r2_core_obstruction_confirmed")
                else:
                    self._add(scores, RoadStructure.R2, 0.58)
                    self._add(scores, RoadStructure.R1, 0.76)
                    rules.append("r2_scenario_trigger_medium")
                    if r2_topology_confirmed:
                        rules.append("r2_waits_for_close_obstruction_or_vehicle_interaction")
                    else:
                        rules.append("r2_requires_visible_or_topology_occupancy_confirmation")
            if kind == "invading_turn":
                rules.append("r2_passive_invading_turn")
        elif kind == "highway_merge":
            merge_window = (
                _route_trigger_window(
                    route_s_for_window,
                    trigger_s,
                    float(cfg.get("merge_pre_m", 50.0)),
                    float(cfg.get("merge_post_m", 50.0)),
                )
                or close_trigger_for_structure
                or scenario_active_for_structure
            )
            if merge_window and RoadStructure.R3 in allowed:
                if ramp_hint:
                    r3_score = 0.88
                elif xodr.get("xodr_available"):
                    r3_score = 0.58
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r3_xodr_available_without_merge_split_review")
                else:
                    r3_score = 0.52
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r3_without_xodr_topology_low")
                self._add(scores, RoadStructure.R3, r3_score)
                rules.append("r3_merge_or_exit_window")
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
                    self._add(scores, RoadStructure.R3, 0.58)
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("interurban_r3_lacks_merge_topology_review")
                rules.append("interurban_r3_actor_flow_window")
            if junction_window:
                if has_tl:
                    self._add(scores, RoadStructure.R4, 0.95)
                    rules.append("interurban_junction_r4")
                elif stop_hazard or is_junction or (xodr_near_junction and not route_projection_error_high and not static_topology_only):
                    self._add(scores, RoadStructure.R5, 0.70)
                    rules.append("interurban_junction_r5_medium")
                else:
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("interurban_junction_window_lacks_visible_control_review")
        elif kind == "interurban_advanced":
            if junction_window:
                self._add(scores, RoadStructure.R4 if has_tl else RoadStructure.R5, 0.85)
                rules.append("advanced_actor_flow_junction_primary")
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
                    self._add(scores, RoadStructure.R6, 0.58)
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("r6_requires_parking_or_curbside_confirmation")
                rules.append("r6_parking_context_window")
        elif kind == "vehicle_opens_door_twoways":
            door_window = two_way_window
            if door_window and RoadStructure.R2 in allowed:
                if has_opposite and same_dir_lanes <= 1:
                    self._add(scores, RoadStructure.R2, 0.88)
                else:
                    self._add(scores, RoadStructure.R2, 0.58)
                    self._add(scores, RoadStructure.R1, 0.72)
                    rules.append("vehicle_open_door_r2_lacks_opposite_confirmation")
                rules.append("vehicle_open_door_r2_possible")
            if door_window and RoadStructure.R6 in allowed:
                if has_parking:
                    self._add(scores, RoadStructure.R6, 0.82)
                else:
                    self._add(scores, RoadStructure.R6, 0.56)
                    self._add(scores, RoadStructure.R1, 0.72)
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
                self._add(scores, RoadStructure.R1, 0.72)
                rules.append("static_cutin_same_direction_r1")
        elif kind == "pedestrian_crossing":
            if junction_window:
                rs = RoadStructure.R4 if has_tl else RoadStructure.R5
                self._add(scores, rs, 0.86 if has_tl else 0.70)
                rules.append("pedestrian_crossing_junction_space")
        elif kind == "vehicle_turning":
            if junction_window:
                rs = RoadStructure.R4 if has_tl else RoadStructure.R5
                self._add(scores, rs, 0.86 if has_tl else 0.70)
                rules.append("vehicle_turning_junction_space")
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

        # 只保留原始候选表允许的 RS，保留强行填充候选全集但不让规则输出越界。
        scores = {rs: score for rs, score in scores.items() if rs in allowed}
        if not scores:
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
                "is_junction": is_junction,
                "map_is_roundabout": map_is_roundabout,
                "dist_to_junction_near": dist_to_junction_near,
                "dist_to_junction_strong": dist_to_junction_strong,
                "stop_hazard": stop_hazard,
                "scenario_active": scenario_active,
                "close_trigger": close_trigger,
                "close_trigger_for_structure": close_trigger_for_structure,
                "scenario_active_for_structure": scenario_active_for_structure,
                "near_junction": near_junction,
                "strong_control_context": strong_control_context,
                "static_signal_near": static_signal_near,
                "junction_window": junction_window,
                "two_way_window": two_way_window,
                "twoway_core_obstruction": twoway_obstruction.core_confirmed
                if kind in {"twoways_obstacle", "invading_turn", "vehicle_opens_door_twoways"}
                else False,
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
            "traffic_light_state": str(tl) if tl is not None else None,
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
    ) -> FrameAnnotation:
        """通用帧分析"""
        # 保留旧逻辑：road_structures 仍是该 scenario 的候选全集。
        road_structures = set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [RoadStructure.R1]))
        events = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [EventType.R_E1]))
        primary, secondary, scores, evidence, confidence, reason = SimpleFrameAnalyzer._engine.analyze(
            scenario_name=scenario_name,
            frame_id=frame_id,
            frame_data=frame_data,
            xml_info=xml_info,
        )

        # 基础逻辑：根据场景类型添加默认事件
        if not events:
            events.add(EventType.R_E1)  # 默认为正常行驶

        comment = _frame_annotation_comment(primary, secondary, confidence, evidence)
        return FrameAnnotation(
            frame_id=frame_id,
            road_structures=road_structures,
            events=events,
            confidence=confidence,
            reason=reason,
            primary_road_structure=primary,
            secondary_road_structures=secondary,
            candidate_scores=scores,
            evidence=evidence,
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

    def collect_one_scenario_all(self, scenario_name: str, max_frames_per_route: Optional[int] = None) -> Dict:
        """模式1: 单场景全部采集 - 采集该场景的所有routes"""
        return self._collect_scenario(scenario_name, max_routes=None, max_frames_per_route=max_frames_per_route)

    def collect_one_scenario(self, scenario_name: str, max_routes: Optional[int] = None, max_frames_per_route: Optional[int] = None) -> Dict:
        """模式2: 单场景采集；max_routes=None 时采集全部合法 routes。"""
        return self._collect_scenario(scenario_name, max_routes=max_routes, max_frames_per_route=max_frames_per_route)

    def collect_multiple_scenarios(
        self,
        scenario_names: List[str],
        max_routes_per_scenario: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
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

    def collect_all_scenarios(self, max_routes_per_scenario: Optional[int] = None, max_frames_per_route: Optional[int] = None) -> Dict:
        """模式4: 全部采集 - 默认采集所有场景的全部合法 routes。"""
        all_scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
        self.logger.info(f"采集所有场景 ({len(all_scenarios)}个)")

        return self.collect_multiple_scenarios(all_scenarios, max_routes_per_scenario, max_frames_per_route=max_frames_per_route)

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

    def _collect_scenario(
        self,
        scenario_name: str,
        max_routes: Optional[int] = None,
        max_frames_per_route: Optional[int] = None,
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

        # 根据 max_routes 参数决定采集多少
        if max_routes is None:
            # None 表示采集所有
            route_dirs = all_route_dirs
        else:
            # 采集前 max_routes 个
            route_dirs = self._select_route_dirs(all_route_dirs, max_routes)

        self.logger.info(
            f"  发现 {len(discovered_route_dirs)} 个routes, "
            f"异常时长剔除 {len(abnormal_skipped)} 个, 将采集 {len(route_dirs)} 个"
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
        ann["confidence"] = max(float(ann.get("confidence", 0.0) or 0.0), float(candidates.get(new_label, 0.74)))
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
            for ann in annotations[int(run["start"]): int(run["end"])]:
                self._rewrite_rs_label(ann, replacement, reason, inherited_from)

        return {"enabled": True, "min_frames": {"R1": 2, "R2": 4, "R3": 4, "R4": 4, "R5": 4, "R6": 4}, "changes": changes}

    def _process_route(self, scenario_name: str, route_path: Path, max_frames_per_route: Optional[int] = None) -> Dict:
        """处理单个route"""
        metas_dir = route_path / "metas"
        if not metas_dir.exists():
            return {"route_id": route_path.name, "status": "skip", "num_frames": 0}

        xml_info = self.xml_index.match(scenario_name, route_path.name)
        meta_files = sorted(metas_dir.glob("*.pkl"))
        if max_frames_per_route is not None and max_frames_per_route > 0:
            meta_files = meta_files[:max_frames_per_route]
        annotations = []

        for meta_file in meta_files:
            try:
                frame_id = int(meta_file.stem)
                # 使用支持 XZ 压缩的加载函数
                frame_data = load_pickle_file(meta_file)

                ann = SimpleFrameAnalyzer.analyze(scenario_name, frame_id, frame_data, xml_info=xml_info)
                ann_dict = ann.to_dict()
                ann_dict["frame_time_s"] = round(frame_id * 0.25, 3)
                ann_dict["meta_path"] = str(meta_file)
                ann_dict["frame_rs_annotation"] = self._frame_rs_annotation_payload(ann_dict)
                annotations.append(ann_dict)
            except Exception as e:
                self.logger.warning(f"处理 {meta_file} 出错: {e}")
                continue

        temporal_smoothing_summary = self._apply_temporal_rs_smoothing(annotations)

        primary_counter = defaultdict(int)
        review_counter = defaultdict(int)
        xodr_source_counter = defaultdict(int)
        review_frame_count = 0
        transition_frames = []
        prev_primary = None
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
            evidence = ann.get("evidence", {})
            if evidence.get("review_required"):
                review_frame_count += 1
                for reason in evidence.get("review_reasons", ["review_required"]):
                    review_counter[reason] += 1
            xodr_source = evidence.get("xodr", {}).get("xodr_source") or "unavailable"
            xodr_source_counter[xodr_source] += 1

        return {
            "route_id": route_path.name,
            "status": "success",
            "xml_path": str(xml_info.path) if xml_info else None,
            "xml_town": xml_info.town if xml_info else None,
            "xml_available": xml_info is not None,
            "num_frames": len(annotations),
            "primary_rs_distribution": dict(sorted(primary_counter.items())),
            "review_required_frames": review_frame_count,
            "review_required_ratio": round(review_frame_count / len(annotations), 4) if annotations else 0.0,
            "review_reason_distribution": dict(sorted(review_counter.items())),
            "xodr_source_distribution": dict(sorted(xodr_source_counter.items())),
            "temporal_smoothing": temporal_smoothing_summary,
            "confidence_stats": self._confidence_stats(annotations),
            "primary_rs_transitions": transition_frames[:50],
            "annotations": annotations
        }
