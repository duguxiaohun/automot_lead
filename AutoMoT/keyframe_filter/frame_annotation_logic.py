"""
逐帧标注逻辑模块 - 根据metas信息生成差异化的帧级标签

注意：本文件保留的是早期 legacy analyzer，EventType 注释和当前
ROAD_EVENT_CLASSIFICATION_PLAN.md 的 R-E/U-E 语义不完全一致。新的帧级候选
与 ROAD_STRUCTURE 标注以 collector.py、ROAD_EVENT_CLASSIFICATION_PLAN.md 和
ROAD_EVENT_CANDIDATE_MAPPING.md 为准；不要把本文件输出直接当作新版 EVENT 真值。

设计原则：
1. RS（Road Structure）优先级：交叉口 > 高速 > 双向 > 直道；停车/遮挡只进 EVENT 或 R2 等效窄路
2. Event 优先级：危险/异常 > 场景特定 > 正常
3. 每帧只输出一个主要RS和一个主要Event（但可扩展多标签）
"""

import pickle
import lzma
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, List


class RoadStructure(Enum):
    """道路结构类型"""
    R1 = "R1"      # 直道/一般道路
    R2 = "R2"      # 双向道路
    R3 = "R3"      # 高速/多车道
    R4 = "R4"      # 交叉口/转弯
    R5 = "R5"      # 非信号化路口


class EventType(Enum):
    """事件类型 - Road相关(R-E) vs Unusual(U-E)"""
    # Road Events (道路可预期事件)
    R_E1 = "R-E1"    # 正常行驶/无异常
    R_E2 = "R-E2"    # 前方有静态障碍
    R_E3 = "R-E3"    # 并道/变道
    R_E4 = "R-E4"    # 行人/自行车/动态物体
    R_E5 = "R-E5"    # 红灯等候/信号限制

    # Unusual Events (异常危险事件)
    U_E1 = "U-E1"    # 紧急制动(Hard Brake)
    U_E2 = "U-E2"    # 碰撞/事故
    U_E3 = "U-E3"    # 停泊/长时间静止
    U_E4 = "U-E4"    # 行人近距通行
    U_E5 = "U-E5"    # 急转/高速转向
    U_E6 = "U-E6"    # 红灯违反(闯红灯)
    U_E7 = "U-E7"    # 交通灯异常/故障
    U_E8 = "U-E8"    # 路口拥堵/通行困难


@dataclass
class FrameAnnotation:
    """单帧标注结果"""
    frame_id: int
    road_structures: List[str]        # [主RS, 可选备选RS]
    events: List[str]                 # [主Event, 可选其他Event]
    confidence: float = 1.0
    reason: str = ""                  # 判断理由
    debug_info: Dict = None           # 调试信息（原始特征值）


class FrameAnnotationAnalyzer:
    """
    逐帧标注分析器 - 使用metas信息生成差异化标签

    流程：
    1. 解析frame metas
    2. 判断RS（道路结构）
    3. 根据RS和当前状态判断Events
    4. 返回FrameAnnotation
    """

    # ========================================================================
    # 阈值配置（可后续调参）
    # ========================================================================

    # RS判断阈值
    JUNCTION_DISTANCE_THRESHOLD = 20.0      # m，靠近交叉口距离
    JUNCTION_ENTER_THRESHOLD = 10.0         # m，已进入交叉口
    SPEED_LIMIT_HIGH_SPEED = 80.0           # km/h，高速判断
    PARKED_OBSTACLE_DISTANCE = 10.0         # m，停泊障碍距离
    PARKING_ZONE_DISTANCE = 15.0            # m，停车侧/遮挡事件参考距离

    # Event判断阈值
    BRAKE_HIGH = 0.7                        # 制动强度
    BRAKE_HARD = 0.8                        # 严重制动
    ACCEL_HARD_BRAKE = -8.0                 # m/s²，紧急制动加速度
    ACCEL_COLLISION = -12.0                 # m/s²，碰撞级加速度

    ACCIDENT_DISTANCE = 10.0                # m，事故点距离
    PEDESTRIAN_DISTANCE = 20.0              # m，行人检测距离
    PEDESTRIAN_DANGER = 10.0                # m，行人危险距离
    BIKER_DISTANCE = 15.0                   # m，自行车检测距离

    STEER_THRESHOLD = 0.3                   # 方向盘转角阈值
    STEER_HARD = 0.6                        # 急转阈值

    SPEED_EPSILON = 0.1                     # km/h，静止判定
    STOPPED_DURATION_FRAMES = 30            # 帧数，30帧=约1秒

    def __init__(self):
        self.stopped_frames_counter = {}  # {frame_id: consecutive_stop_count}

    @staticmethod
    def load_frame_meta(meta_file: Path) -> Optional[Dict]:
        """加载单帧的meta数据（支持xz压缩）"""
        try:
            with lzma.open(meta_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load {meta_file}: {e}")
            return None

    def safe_get(self, data: Dict, key: str, default=None):
        """安全获取字典值，返回标量数值"""
        val = data.get(key, default)
        if isinstance(val, np.ndarray):
            return val.item() if val.size == 1 else val[0]
        return val

    def judge_road_structure(self, meta: Dict, prev_meta: Optional[Dict] = None) -> Tuple[RoadStructure, float]:
        """
        判断当前帧的道路结构 (RS)

        返回: (主要RS, 置信度)
        优先级: 交叉口 > 高速 > 双向 > 停泊 > 直道
        """

        # 1. 检查交叉口/路口优先级最高
        is_intersection = bool(self.safe_get(meta, 'is_intersection', False))
        is_junction = bool(self.safe_get(meta, 'is_junction', False))
        dist_to_junction = float(self.safe_get(meta, 'distance_to_junction', 999))

        if is_intersection or is_junction or dist_to_junction < self.JUNCTION_DISTANCE_THRESHOLD:
            # 进一步区分信号化(R4)还是非信号化(R5)
            traffic_light = self.safe_get(meta, 'traffic_light_state', None)
            if traffic_light is None or traffic_light == 'off' or traffic_light == 'unknown':
                return RoadStructure.R5, 0.9  # 非信号化路口
            else:
                return RoadStructure.R4, 0.95  # 信号化路口

        # 2. 检查双向道路。停车/遮挡不再单独生成 RS；若压缩成有效对向单车道，由 R2 表达。
        lane_change_str = str(self.safe_get(meta, 'lane_change_str', ''))
        if 'opposite' in lane_change_str.lower() or 'bidirectional' in lane_change_str.lower():
            return RoadStructure.R2, 0.80  # 双向道路

        # 3. 检查高速/多车道
        speed_limit = float(self.safe_get(meta, 'speed_limit', 50))
        if speed_limit > self.SPEED_LIMIT_HIGH_SPEED:
            return RoadStructure.R3, 0.85  # 高速

        # 4. 默认直道
        return RoadStructure.R1, 0.70

    def judge_events(self, meta: Dict, prev_meta: Optional[Dict], rs: RoadStructure) -> List[str]:
        """
        根据当前RS判断发生的Events

        返回: [主Event, 可选备选Event, ...]

        优先级原则：
        - 危险事件(U-E*)优先于正常事件(R-E*)
        - 碰撞/闯红灯最高优先级
        - 每帧主要返回1个Event，可选返回多个
        """

        events = []

        # ====== 通用危险事件 (对所有RS) ======

        # 碰撞检测 (U-E2) - 最高优先级
        dist_to_accident = float(self.safe_get(meta, 'dist_to_accident_site', 999))
        accel_x = float(self.safe_get(meta, 'accel_x', 0))
        brake = float(self.safe_get(meta, 'brake', 0))

        # 碰撞判断：距离近 + 强制动，或距离近 + 大减速
        if dist_to_accident < self.ACCIDENT_DISTANCE:
            if brake > self.BRAKE_HARD or accel_x < self.ACCEL_HARD_BRAKE:
                return [EventType.U_E2.value]  # 碰撞 - 最高优先级，立即返回

        # 紧急制动 (U-E1) - 单独的brake或accel信号都可触发
        if brake > self.BRAKE_HARD or accel_x < self.ACCEL_HARD_BRAKE:
            events.append(EventType.U_E1.value)

        # 急转/高速转向 (U-E5)
        steer = abs(float(self.safe_get(meta, 'steer', 0)))
        speed = float(self.safe_get(meta, 'speed', 0))  # km/h
        if steer > self.STEER_HARD and speed > 10:
            events.append(EventType.U_E5.value)

        # ====== RS特定事件 ======

        if rs == RoadStructure.R4:  # 信号化交叉口
            traffic_light = str(self.safe_get(meta, 'traffic_light_state', 'unknown')).lower()

            # 红灯违反 (U-E6)
            if traffic_light == 'red' and speed > 1.0:
                return [EventType.U_E6.value]  # 闯红灯 - 高优先级

            # 交通灯异常 (U-E7)
            if traffic_light not in ['green', 'yellow', 'red', 'off']:
                events.append(EventType.U_E7.value)

            # 红灯等候 (R-E5)
            if traffic_light == 'red' and brake > 0.3:
                events.append(EventType.R_E5.value)

        elif rs == RoadStructure.R5:  # 非信号化路口
            # 红灯概念不适用，检查路口拥堵
            dist_to_junction = float(self.safe_get(meta, 'distance_to_junction', 999))
            if dist_to_junction < self.JUNCTION_ENTER_THRESHOLD and speed < 5:
                events.append(EventType.U_E8.value)  # 路口拥堵

        # ====== 通用动态物体事件 ======

        # 行人近距通行 (U-E4)
        dist_to_pedestrian = float(self.safe_get(meta, 'dist_to_pedestrian', 999))
        if dist_to_pedestrian < self.PEDESTRIAN_DANGER:
            events.append(EventType.U_E4.value)

        # 行人/自行车检测 (R-E4)
        dist_to_biker = float(self.safe_get(meta, 'dist_to_biker', 999))
        if (dist_to_pedestrian < self.PEDESTRIAN_DISTANCE or dist_to_biker < self.BIKER_DISTANCE):
            if EventType.U_E4.value not in events:  # 避免重复
                events.append(EventType.R_E4.value)

        # ====== 道路事件 ======

        # 障碍物 (R-E2)
        dist_to_parked = float(self.safe_get(meta, 'dist_to_parked_obstacle', 999))
        if dist_to_parked < self.PARKED_OBSTACLE_DISTANCE and EventType.R_E4.value not in events:
            events.append(EventType.R_E2.value)

        # 并道 (R-E3)
        if steer > self.STEER_THRESHOLD and speed > 10:
            events.append(EventType.R_E3.value)

        # ====== 默认事件 ======
        if not events:
            events.append(EventType.R_E1.value)  # 正常行驶

        return events

    def analyze(self, frame_id: int, meta: Dict, prev_meta: Optional[Dict] = None) -> FrameAnnotation:
        """
        完整的逐帧分析流程

        入参:
            frame_id: 帧号
            meta: 当前帧的metas字典
            prev_meta: 前一帧的metas字典（可选，用于时序分析）

        返回:
            FrameAnnotation 对象
        """

        # 1. 判断RS
        rs, rs_confidence = self.judge_road_structure(meta, prev_meta)

        # 2. 判断Events
        events = self.judge_events(meta, prev_meta, rs)

        # 构建标注
        reason = f"RS={rs.value} (conf={rs_confidence:.2f})"

        ann = FrameAnnotation(
            frame_id=frame_id,
            road_structures=[rs.value],
            events=events,
            confidence=rs_confidence,
            reason=reason,
            debug_info={
                'speed': self.safe_get(meta, 'speed', 0),
                'brake': self.safe_get(meta, 'brake', 0),
                'accel_x': self.safe_get(meta, 'accel_x', 0),
                'steer': self.safe_get(meta, 'steer', 0),
                'traffic_light': self.safe_get(meta, 'traffic_light_state', None),
                'dist_to_pedestrian': self.safe_get(meta, 'dist_to_pedestrian', 999),
                'dist_to_accident': self.safe_get(meta, 'dist_to_accident_site', 999),
            }
        )

        return ann

    def to_dict(self, ann: FrameAnnotation) -> Dict:
        """转为JSON兼容字典"""
        return {
            'frame_id': ann.frame_id,
            'road_structures': ann.road_structures,
            'events': ann.events,
            'confidence': ann.confidence,
            'reason': ann.reason,
            'debug_info': ann.debug_info,
        }
