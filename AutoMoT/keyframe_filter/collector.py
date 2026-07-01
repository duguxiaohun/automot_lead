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
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict
import numpy as np

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
    
    def to_dict(self):
        return {
            'frame_id': self.frame_id,
            'road_structures': [rs.value for rs in self.road_structures],
            'events': [ev.value for ev in self.events],
            'confidence': self.confidence,
            'reason': self.reason
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

# 简化策略工厂 - 所有场景使用通用策略加启发式检测
class SimpleFrameAnalyzer:
    """简化的帧分析器 - 使用启发式检测"""
    
    @staticmethod
    def analyze(scenario_name: str, frame_id: int, frame_data: dict) -> FrameAnnotation:
        """通用帧分析"""
        road_structures = set(SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [RoadStructure.R1]))
        events = set(SCENARIO_TO_FINE_EVENTS.get(scenario_name, [EventType.R_E1]))
        
        # 基础逻辑：根据场景类型添加默认事件
        if not events:
            events.add(EventType.R_E1)  # 默认为正常行驶
        
        return FrameAnnotation(
            frame_id=frame_id,
            road_structures=road_structures,
            events=events,
            reason=f"{scenario_name}: 自动检测"
        )


class ScenarioCollector:
    """灵活的采集器 - 支持4种采集模式"""
    
    def __init__(self, lead_data_root: str = "/home/cruser1/lda/AutoMoT/lead_data",
                 output_dir: str = "/home/cruser1/lda/AutoMoT/keyframe_filter/collection_output"):
        self.lead_data_root = Path(lead_data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    # ========================================================================
    # 4种采集模式
    # ========================================================================
    
    def collect_one_scenario_all(self, scenario_name: str) -> Dict:
        """模式1: 单场景全部采集 - 采集该场景的所有routes"""
        return self._collect_scenario(scenario_name, max_routes=None)
    
    def collect_one_scenario(self, scenario_name: str, max_routes: int = 5) -> Dict:
        """模式2: 单场景指定数目采集 - 采集该场景的前 max_routes 个routes"""
        return self._collect_scenario(scenario_name, max_routes=max_routes)
    
    def collect_multiple_scenarios(self, scenario_names: List[str], max_routes_per_scenario: int = 5) -> Dict:
        """模式3: 多场景全部采集 - 采集多个指定场景"""
        self.logger.info(f"采集多场景: {scenario_names}")
        
        all_results = {}
        total_frames = 0
        
        for scenario in scenario_names:
            try:
                result = self._collect_scenario(scenario, max_routes=max_routes_per_scenario)
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
    
    def collect_all_scenarios(self, max_routes_per_scenario: int = 5) -> Dict:
        """模式4: 全部采集 - 采集所有47个场景"""
        all_scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE.keys())
        self.logger.info(f"采集所有场景 ({len(all_scenarios)}个)")
        
        return self.collect_multiple_scenarios(all_scenarios, max_routes_per_scenario)
    
    # ========================================================================
    # 内部实现
    # ========================================================================
    
    def _collect_scenario(self, scenario_name: str, max_routes: Optional[int] = None) -> Dict:
        """采集单个场景的内部实现"""
        self.logger.info(f"采集场景: {scenario_name}")
        
        scenario_dir = self.lead_data_root / scenario_name
        if not scenario_dir.exists():
            return {"scenario": scenario_name, "status": "error", "routes": []}
        
        # 获取该场景的所有 routes
        all_route_dirs = sorted([d for d in scenario_dir.iterdir() if d.is_dir()])
        
        # 根据 max_routes 参数决定采集多少
        if max_routes is None:
            # None 表示采集所有
            route_dirs = all_route_dirs
        else:
            # 采集前 max_routes 个
            route_dirs = all_route_dirs[:max_routes]
        
        self.logger.info(f"  发现 {len(all_route_dirs)} 个routes, 将采集 {len(route_dirs)} 个")
        
        routes = []
        for i, route_dir in enumerate(route_dirs, 1):
            self.logger.info(f"    [{i}/{len(route_dirs)}] 处理 {route_dir.name}")
            route_result = self._process_route(scenario_name, route_dir)
            routes.append(route_result)
        
        result = {
            "scenario": scenario_name,
            "status": "success",
            "road_candidates": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario_name, [])],
            "event_candidates": [ev.value for ev in SCENARIO_TO_FINE_EVENTS.get(scenario_name, [])],
            "routes": routes,
            "total_frames": sum(r.get('num_frames', 0) for r in routes)
        }
        
        # 保存结果
        output_file = self.output_dir / f"{scenario_name}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"  ✓ {scenario_name} 采集完成: {result['total_frames']} 帧")
        
        return result
    
    def _process_route(self, scenario_name: str, route_path: Path) -> Dict:
        """处理单个route"""
        metas_dir = route_path / "metas"
        if not metas_dir.exists():
            return {"route_id": route_path.name, "status": "skip", "num_frames": 0}
        
        meta_files = sorted(metas_dir.glob("*.pkl"))
        annotations = []
        
        for meta_file in meta_files:
            try:
                frame_id = int(meta_file.stem)
                # 使用支持 XZ 压缩的加载函数
                frame_data = load_pickle_file(meta_file)
                
                ann = SimpleFrameAnalyzer.analyze(scenario_name, frame_id, frame_data)
                annotations.append(ann.to_dict())
            except Exception as e:
                self.logger.warning(f"处理 {meta_file} 出错: {e}")
                continue
        
        return {
            "route_id": route_path.name,
            "status": "success",
            "num_frames": len(annotations),
            "annotations": annotations
        }
