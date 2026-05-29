"""Qwen3-VL 范式 A 使用的 prompt 与状态机流水线。

这个文件把“驾驶场景状态机”从旧的 vlm_paradigm_a_runner.py 迁移成可复用模块：

- SCENARIO_LABELS：场景名到人类可读任务说明。
- SCENARIO_EVENT_SEQUENCES：每个场景允许的事件推进顺序。
- EVENT_DESCRIPTIONS：每个事件 token 的自然语言解释。
- DrivingMemory：跨帧保留的当前状态、下一目标和已完成事件。
- build_*_prompt：把 memory 和视觉输入说明拼成 Qwen 对话 prompt。
- parse_vlm_output/update_memory：把 VLM 自由文本收回到结构化状态机。

注意：这个模块只生成文字 prompt 和解析文字输出，不读取图片，也不负责真实 image
token 注入。图片由 engine.build_messages() 的 structured image content 处理。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# 第一段：迁移自 vlm_prompt_pipeline.py（原样保留,保证状态机语义一致）
# ============================================================================

# ---------------------------------------------------------------------------
# Scenario labels
# ---------------------------------------------------------------------------
# 这些 label 只给 prompt 提供场景语义提示，不参与合法性校验。
# 真正的可选 STATUS/SUBGOAL 来自 SCENARIO_EVENT_SEQUENCES。

SCENARIO_LABELS: Dict[str, str] = {
    "Accident":                               "Brake and avoid accident hazard",
    "AccidentTwoWays":                        "Brake and avoid head-on accident hazard",
    "BlockedIntersection":                    "Bypass blocked intersection",
    "ConstructionObstacle":                   "Navigate around construction obstacle",
    "ConstructionObstacleTwoWays":            "Navigate construction obstacle on two-way road",
    "ControlLoss":                            "Regain control after vehicle instability",
    "CrossJunctionDefectTrafficLight":        "Cross junction with defective traffic light",
    "CrossingBicycleFlow":                    "Yield to crossing bicycle",
    "DynamicObjectCrossing":                  "Yield to dynamic object crossing",
    "EnterActorFlow":                         "Merge into actor flow",
    "EnterActorFlowV2":                       "Merge into actor flow (variant)",
    "HardBreakRoute":                         "React to lead vehicle hard braking",
    "HazardAtSideLane":                       "Pass hazard at side lane",
    "HazardAtSideLaneTwoWays":                "Pass side-lane hazard on two-way road",
    "HighwayCutIn":                           "Decelerate and prepare for cut-in vehicle",
    "HighwayExit":                            "Navigate highway exit",
    "InterurbanActorFlow":                    "Merge into interurban actor flow",
    "InterurbanAdvancedActorFlow":            "Merge into advanced interurban actor flow",
    "InvadingTurn":                           "React to vehicle invading lane on turn",
    "MergerIntoSlowTraffic":                  "Merge into slow-moving traffic",
    "MergerIntoSlowTrafficV2":                "Merge into slow-moving traffic (variant)",
    "NonSignalizedJunctionLeftTurn":          "Left turn at non-signalized junction",
    "NonSignalizedJunctionLeftTurnEnterFlow": "Left turn and enter flow at non-signalized junction",
    "NonSignalizedJunctionRightTurn":         "Right turn at non-signalized junction",
    "OppositeVehicleRunningRedLight":         "React to opposite vehicle running red light",
    "OppositeVehicleTakingPriority":          "React to opposite vehicle taking priority",
    "ParkedObstacle":                         "Navigate around parked obstacle",
    "ParkedObstacleTwoWays":                  "Navigate around parked obstacle on two-way road",
    "ParkingCrossingPedestrian":              "Yield to pedestrian near parking area",
    "ParkingCutIn":                           "React to vehicle cutting in from parking",
    "ParkingExit":                            "Yield to exiting parking vehicle",
    "PedestrianCrossing":                     "Yield to crossing pedestrian",
    "PriorityAtJunction":                     "Assert and navigate junction priority",
    "RedLightWithoutLeadVehicle":             "Stop at red light without lead vehicle",
    "SignalizedJunctionLeftTurn":             "Left turn at signalized junction",
    "SignalizedJunctionLeftTurnEnterFlow":    "Left turn and enter flow at signalized junction",
    "SignalizedJunctionRightTurn":            "Right turn at signalized junction",
    "StaticCutIn":                            "React to static vehicle cutting in",
    "T_Junction":                             "Navigate T-junction",
    "VehicleOpensDoorTwoWays":                "React to vehicle door opening on two-way road",
    "VehicleTurningRoute":                    "React to vehicle turning across route",
    "VehicleTurningRoutePedestrian":          "React to turning vehicle with pedestrian present",
}

# ---------------------------------------------------------------------------
# Per-scenario ordered event sequences  (initial → middle[0..2] → final)
# ---------------------------------------------------------------------------

# 每个场景是一条严格有序状态机：完整序列会由 get_full_sequence() 补成
# initial -> middle[0] -> middle[1] -> middle[2] -> final。
# 模型只能保持当前 STATUS 或向前推进，不能跳阶段、不能倒退。
SCENARIO_EVENT_SEQUENCES: Dict[str, Tuple[str, str, str]] = {
    "Accident":                               ("hazard_detect",            "max_brake_or_min_gap",       "recover_or_pass"),
    "AccidentTwoWays":                        ("hazard_detect",            "max_brake_or_min_gap",       "recover_or_pass"),
    "BlockedIntersection":                    ("obstacle_detect",          "wait_stop",                  "bypass_resume"),
    "ConstructionObstacle":                   ("construction_detect",      "slow_and_navigate",          "obstacle_clear"),
    "ConstructionObstacleTwoWays":            ("construction_detect",      "slow_and_navigate",          "obstacle_clear"),
    "ControlLoss":                            ("normal_driving",           "loss_of_control",            "regain_control"),
    "CrossJunctionDefectTrafficLight":        ("junction_approach",        "check_and_proceed",          "junction_clear"),
    "CrossingBicycleFlow":                    ("wait_start",               "wait_peak",                  "proceed_resume"),
    "DynamicObjectCrossing":                  ("object_detect",            "wait_or_slow",               "proceed_resume"),
    "EnterActorFlow":                         ("flow_approach",            "gap_accept_merge",           "flow_established"),
    "EnterActorFlowV2":                       ("flow_approach",            "gap_accept_merge",           "flow_established"),
    "HardBreakRoute":                         ("follow_normal",            "hard_brake_response",        "recover_speed"),
    "HazardAtSideLane":                       ("side_hazard_detect",       "passing_hazard",             "clear_hazard"),
    "HazardAtSideLaneTwoWays":                ("side_hazard_detect",       "passing_hazard",             "clear_hazard"),
    "HighwayCutIn":                           ("cutin_onset",              "caution_peak",               "stabilize_follow"),
    "HighwayExit":                            ("exit_approach",            "decelerate_and_diverge",     "exit_complete"),
    "InterurbanActorFlow":                    ("flow_approach",            "gap_accept_merge",           "flow_established"),
    "InterurbanAdvancedActorFlow":            ("flow_approach",            "gap_accept_merge",           "flow_established"),
    "InvadingTurn":                           ("invading_vehicle_detect",  "evasive_decelerate",         "lane_clear"),
    "MergerIntoSlowTraffic":                  ("slow_traffic_detect",      "match_speed",                "merge_complete"),
    "MergerIntoSlowTrafficV2":                ("slow_traffic_detect",      "match_speed",                "merge_complete"),
    "NonSignalizedJunctionLeftTurn":          ("junction_approach",        "yield_and_turn",             "turn_complete"),
    "NonSignalizedJunctionLeftTurnEnterFlow": ("junction_approach",        "yield_and_enter_flow",       "flow_established"),
    "NonSignalizedJunctionRightTurn":         ("junction_approach",        "yield_and_turn",             "turn_complete"),
    "OppositeVehicleRunningRedLight":         ("threat_detect",            "evasive_action",             "threat_clear"),
    "OppositeVehicleTakingPriority":          ("threat_detect",            "evasive_action",             "threat_clear"),
    "ParkedObstacle":                         ("obstacle_detect",          "decelerate_around",          "clear_obstacle"),
    "ParkedObstacleTwoWays":                  ("obstacle_detect",          "decelerate_around",          "clear_obstacle"),
    "ParkingCrossingPedestrian":              ("pedestrian_detect",        "wait_to_cross",              "proceed_resume"),
    "ParkingCutIn":                           ("cutin_onset",              "caution_peak",               "stabilize_follow"),
    "ParkingExit":                            ("parking_exit_detect",      "brake_for_exit",             "exit_clear"),
    "PedestrianCrossing":                     ("pedestrian_detect",        "wait_to_cross",              "proceed_resume"),
    "PriorityAtJunction":                     ("junction_approach",        "assert_priority",            "junction_clear"),
    "RedLightWithoutLeadVehicle":             ("junction_approach",        "brake_at_light",             "proceed_on_green"),
    "SignalizedJunctionLeftTurn":             ("junction_approach",        "wait_or_turn_on_green",      "turn_complete"),
    "SignalizedJunctionLeftTurnEnterFlow":    ("junction_approach",        "wait_and_enter_flow",        "flow_established"),
    "SignalizedJunctionRightTurn":            ("junction_approach",        "turn_on_green",              "turn_complete"),
    "StaticCutIn":                            ("cutin_onset",              "caution_peak",               "stabilize_follow"),
    "T_Junction":                             ("junction_approach",        "yield_and_turn",             "turn_complete"),
    "VehicleOpensDoorTwoWays":                ("door_open_detect",         "avoid_door",                 "clear_hazard"),
    "VehicleTurningRoute":                    ("turning_vehicle_detect",   "yield_or_slow",              "route_clear"),
    "VehicleTurningRoutePedestrian":          ("pedestrian_detect",        "yield_and_slow",             "proceed_resume"),
}

# ---------------------------------------------------------------------------
# Human-readable event descriptions
# ---------------------------------------------------------------------------
# prompt 会把当前场景涉及到的事件说明展开给模型，让它知道每个 event token
# 在驾驶语义上代表什么。

EVENT_DESCRIPTIONS: Dict[str, str] = {
    # --- shared terminal events ---
    "initial":                  "Scenario has started; vehicle is approaching the challenge zone.",
    "final":                    "Scenario is complete; vehicle has cleared the challenge zone.",
    # --- detection / onset ---
    "hazard_detect":            "Accident hazard has been detected ahead.",
    "construction_detect":      "Construction zone has been detected ahead.",
    "obstacle_detect":          "Stationary obstacle is detected blocking the path.",
    "side_hazard_detect":       "Hazard has been detected at the side lane.",
    "cutin_onset":              "A vehicle has begun cutting into the ego lane.",
    "invading_vehicle_detect":  "An oncoming vehicle is detected invading the lane on a turn.",
    "door_open_detect":         "A vehicle door opening into the lane has been detected.",
    "parking_exit_detect":      "A vehicle is detected exiting a parking space.",
    "object_detect":            "A dynamic object (e.g., cyclist, animal) is detected crossing.",
    "pedestrian_detect":        "A pedestrian has been detected near the crossing zone.",
    "turning_vehicle_detect":   "A vehicle is detected turning across the ego route.",
    "threat_detect":            "An oncoming vehicle threat (e.g., red-light runner) is detected.",
    "junction_approach":        "Ego vehicle is approaching a junction or intersection.",
    "exit_approach":            "Ego vehicle is approaching a highway exit.",
    "flow_approach":            "Ego vehicle is approaching an active traffic flow.",
    "slow_traffic_detect":      "Slow-moving traffic ahead has been detected.",
    "normal_driving":           "Ego vehicle is driving normally before any control event.",
    "wait_start":               "Ego vehicle has begun waiting for a crossing agent (bicycle/pedestrian).",
    "follow_normal":            "Ego vehicle is following the lead vehicle at normal speed.",
    # --- mid-scenario actions ---
    "max_brake_or_min_gap":     "Ego vehicle is at maximum braking or minimum safe gap.",
    "slow_and_navigate":        "Ego vehicle is slowing and navigating around the construction zone.",
    "decelerate_around":        "Ego vehicle is decelerating to navigate around an obstacle.",
    "passing_hazard":           "Ego vehicle is actively passing the side-lane hazard.",
    "caution_peak":             "Ego vehicle is at peak caution response to the cut-in vehicle.",
    "evasive_decelerate":       "Ego vehicle is performing evasive deceleration.",
    "avoid_door":               "Ego vehicle is steering to avoid the open door.",
    "brake_for_exit":           "Ego vehicle is braking to yield to the exiting parking vehicle.",
    "wait_or_slow":             "Ego vehicle is waiting or slowing for the crossing dynamic object.",
    "wait_to_cross":            "Ego vehicle is waiting for the pedestrian to finish crossing.",
    "wait_peak":                "Ego vehicle is waiting while bicycle/pedestrian flow is at its peak.",
    "wait_stop":                "Ego vehicle has stopped to yield at the blocked intersection.",
    "yield_and_turn":           "Ego vehicle is yielding and beginning the turn.",
    "yield_and_enter_flow":     "Ego vehicle is yielding and merging into the traffic flow.",
    "yield_or_slow":            "Ego vehicle is yielding or slowing for the turning vehicle.",
    "yield_and_slow":           "Ego vehicle is yielding and slowing for the pedestrian.",
    "assert_priority":          "Ego vehicle is asserting priority and proceeding through the junction.",
    "check_and_proceed":        "Ego vehicle is checking the defective signal and proceeding with caution.",
    "brake_at_light":           "Ego vehicle is braking at the red light.",
    "wait_or_turn_on_green":    "Ego vehicle is waiting for green or turning when signal permits.",
    "wait_and_enter_flow":      "Ego vehicle is waiting for a gap and entering the flow on green.",
    "turn_on_green":            "Ego vehicle is turning on the green signal.",
    "gap_accept_merge":         "Ego vehicle has accepted a gap and is merging into the flow.",
    "decelerate_and_diverge":   "Ego vehicle is decelerating and diverging onto the exit ramp.",
    "match_speed":              "Ego vehicle is matching the speed of slow-moving traffic.",
    "evasive_action":           "Ego vehicle is taking evasive action against the threat vehicle.",
    "loss_of_control":          "Ego vehicle is experiencing loss of control.",
    "hard_brake_response":      "Ego vehicle is responding with hard braking to the lead vehicle.",
    # --- resolution / completion ---
    "recover_or_pass":          "Ego vehicle has recovered from braking or passed the hazard.",
    "obstacle_clear":           "Construction/obstacle zone has been cleared.",
    "clear_obstacle":           "Obstacle has been cleared and ego vehicle resumes normal driving.",
    "clear_hazard":             "Side-lane hazard has been cleared.",
    "stabilize_follow":         "Ego vehicle has stabilized and is following at a safe distance.",
    "lane_clear":               "Invading vehicle has cleared; ego lane is safe again.",
    "exit_clear":               "Parking exit is clear; ego vehicle resumes.",
    "proceed_resume":           "Crossing agent has cleared; ego vehicle resumes driving.",
    "bypass_resume":            "Blocked intersection has been bypassed; ego vehicle resumes.",
    "turn_complete":            "Turn has been completed successfully.",
    "flow_established":         "Ego vehicle has successfully merged and flow is established.",
    "junction_clear":           "Junction has been crossed and is now clear.",
    "exit_complete":            "Highway exit has been completed.",
    "merge_complete":           "Merge into slow traffic has been completed.",
    "proceed_on_green":         "Traffic light is green; ego vehicle is proceeding.",
    "route_clear":              "Turning vehicle has cleared; ego route is unobstructed.",
    "threat_clear":             "Threat vehicle has been neutralized or passed.",
    "regain_control":           "Ego vehicle has regained control after the instability event.",
    "recover_speed":            "Ego vehicle has recovered to normal cruising speed.",
}


def get_full_sequence(scenario: str) -> Tuple[str, ...]:
    """返回某场景的完整事件序列（initial + 3 中段事件 + final）。"""
    # 返回某个场景的完整事件序列。不存在的 scenario 直接报错，
    # 这样可以尽早发现 route 自动识别或命令行参数传错。
    middle = SCENARIO_EVENT_SEQUENCES.get(scenario)
    if middle is None:
        raise ValueError(f"Unknown scenario: '{scenario}'. "
                         f"Available: {sorted(SCENARIO_EVENT_SEQUENCES)}")
    return ("initial",) + middle + ("final",)


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------

@dataclass
class DrivingMemory:
    """每次 VLM 调用后传入并被更新的持久化状态。"""

    # 跨帧/跨步持久化的驾驶任务状态。当前 runner 只跑单步，但该结构为未来
    # 多步循环预留：每次 VLM 输出 STATUS 后，update_memory 会产出下一次 prompt
    # 可继续使用的新 memory。
    scenario: str
    scenario_label: str
    event_sequence: Tuple[str, ...]
    status: str
    subgoal: str
    completed_events: List[str] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, scenario: str) -> "DrivingMemory":
        # 初始状态固定为 initial，下一目标是该场景序列里的第一个真实事件。
        label = SCENARIO_LABELS.get(scenario, f"Handle {scenario} scenario")
        seq = get_full_sequence(scenario)
        return cls(
            scenario=scenario,
            scenario_label=label,
            event_sequence=seq,
            status="initial",
            subgoal=seq[1],
            completed_events=["initial"],
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DrivingMemory":
        # 从 JSON/dict 恢复 memory，方便未来多步推理续跑。
        scenario = d["scenario"]
        seq = get_full_sequence(scenario)
        return cls(
            scenario=scenario,
            scenario_label=d.get("scenario_label", SCENARIO_LABELS.get(scenario, scenario)),
            event_sequence=seq,
            status=d["status"],
            subgoal=d["subgoal"],
            completed_events=list(d.get("completed_events", [])),
        )

    def to_dict(self) -> dict:
        # 转成 JSON 可写结构；tuple 序列显式转 list。
        return {
            "scenario":         self.scenario,
            "scenario_label":   self.scenario_label,
            "event_sequence":   list(self.event_sequence),
            "status":           self.status,
            "subgoal":          self.subgoal,
            "completed_events": self.completed_events,
        }

    def is_complete(self) -> bool:
        # final 是状态机唯一终点。
        return self.status == "final"

    def status_description(self) -> str:
        # 给 prompt 注释当前 STATUS 的自然语言含义。
        return EVENT_DESCRIPTIONS.get(self.status, self.status)

    def subgoal_description(self) -> str:
        # 给 prompt 注释当前 SUBGOAL 的自然语言含义。
        return EVENT_DESCRIPTIONS.get(self.subgoal, self.subgoal)

    def _next_event_after(self, event: str) -> Optional[str]:
        # 根据状态机顺序推导下一事件；不存在或已到末尾时返回 None。
        try:
            idx = self.event_sequence.index(event)
        except ValueError:
            return None
        if idx + 1 < len(self.event_sequence):
            return self.event_sequence[idx + 1]
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an autonomous driving agent controlling an ego vehicle.

Input:
- You receive a short RGB clip ordered oldest to newest.
- Each frame is a stitched three-camera view: left, front, and right.
- These are your recent observations while driving. The last frame is the \
current moment; earlier frames show how the scene has evolved.

Your task:
1. Read MEMORY to understand the driving scenario, the ordered \
EVENT_SEQUENCE, and each event description.
2. Use the scenario and event descriptions to infer the expected overall \
evolution of this driving task.
3. Use the recent observations to understand the current scene, surrounding \
agents, road layout, and task progress.
4. Decide the current STATUS at the newest frame.
5. Identify the next SUBGOAL the ego vehicle should pursue.
6. Briefly explain the visual evidence.

Definitions:
- EVENT_SEQUENCE is the only valid ordered state machine for this scenario.
- EVENT_DESCRIPTIONS explains what each event stage means.
- An event_name is exactly one token copied verbatim from EVENT_SEQUENCE.
- STATUS is the current task stage reached at the newest frame.
- SUBGOAL is the event immediately after STATUS in EVENT_SEQUENCE. If STATUS \
is final, SUBGOAL is final.
- If the visual evidence is ambiguous, keep the previous STATUS from MEMORY.

Output format - respond EXACTLY as shown below, with no extra text before or \
after the block:

ANALYSIS: <2-4 sentence description of scene evolution and task-stage evidence>
STATUS: <event_name>
SUBGOAL: <event_name>

Rules:
- STATUS and SUBGOAL must each be copied verbatim from EVENT_SEQUENCE.
- Do NOT invent event names outside EVENT_SEQUENCE.
- Do NOT skip event stages. STATUS may stay the same as MEMORY STATUS or \
advance by one event only.
- SUBGOAL must be the immediate next event after STATUS, unless STATUS is \
final.
- Be concise and precise."""


def build_system_prompt() -> str:
    """返回固定 system prompt。

    system prompt 负责定义模型角色、输入含义、状态机规则和严格输出格式。
    """

    return _SYSTEM_PROMPT


def build_memory_block(memory: DrivingMemory) -> str:
    """把当前 memory 展开成 prompt 中的 [MEMORY] 块。

    这里故意把 EVENT_SEQUENCE 和对应 EVENT_DESCRIPTIONS 全量写入 prompt：
    模型不需要记住项目里的状态机表，只需从当前 prompt 复制合法 event token。
    """

    seq_str = " -> ".join(memory.event_sequence)
    event_desc_str = "\n".join(
        f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}"
        for event in memory.event_sequence
    )
    completed_str = ", ".join(memory.completed_events) if memory.completed_events else "none"
    status_desc = memory.status_description()
    subgoal_desc = memory.subgoal_description()

    return (
        "[MEMORY]\n"
        f"SCENARIO: {memory.scenario}  # {memory.scenario_label}\n"
        f"EVENT_SEQUENCE: {seq_str}\n"
        f"EVENT_DESCRIPTIONS:\n{event_desc_str}\n"
        f"STATUS: {memory.status}  # {status_desc}\n"
        f"SUBGOAL: {memory.subgoal}  # {subgoal_desc}\n"
        f"COMPLETED: {completed_str}\n"
        "[/MEMORY]"
    )


def build_user_prompt(
    memory: DrivingMemory,
    image_description: str = "Refer to the visual observation(s) above.",
) -> str:
    """构造 user prompt。

    image_description 是普通文本，用来描述调用方已经插入的视觉输入。例如
    “N 张图片按 oldest -> newest 排列”。真实 Qwen image token 来自 structured
    image messages 和 processor，不来自这个字符串。
    """
    memory_block = build_memory_block(memory)
    return (
        f"{image_description}\n\n"
        f"{memory_block}\n\n"
        "Given the observations above and the memory context, output your "
        "ANALYSIS, STATUS, and SUBGOAL."
    )


def build_combined_prompt(
    memory: DrivingMemory,
    image_description: str = "Refer to the visual observation(s) above.",
) -> str:
    """范式 A 实际喂给 inferencer 的合并 prompt = system + user。

    背景:AutoMoT 的 kv_cache_fixed_inference(input_lists) 把 input_lists 里
    唯一的字符串当作 instruction_prompt,套进固定的 USER_PROMPT 模板。它没有
    单独的 system slot,所以把 vlm_prompt_pipeline 的 system 串和 user 串拼成
    一段文本作为 instruction 喂进去(把"系统指令"和"用户问题"放在同一个
    user-turn,Qwen3-VL 依然能听懂)。
    """
    system = build_system_prompt()
    user = build_user_prompt(memory, image_description)
    return (
        "[SYSTEM INSTRUCTION]\n"
        f"{system}\n"
        "[/SYSTEM INSTRUCTION]\n\n"
        "[USER]\n"
        f"{user}\n"
        "[/USER]"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
# 三个正则只负责“从自由文本里找到字段”，不做合法性判断。
# 合法性由 update_memory 根据当前场景的 EVENT_SEQUENCE 再统一处理。

_STATUS_RE   = re.compile(r"^\s*STATUS\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_SUBGOAL_RE  = re.compile(r"^\s*SUBGOAL\s*:\s*(\S+)", re.MULTILINE | re.IGNORECASE)
_ANALYSIS_RE = re.compile(
    r"^\s*ANALYSIS\s*:\s*(.+?)(?=\nSTATUS\s*:|\nSUBGOAL\s*:|$)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)


def parse_vlm_output(text: str) -> Dict[str, Optional[str]]:
    """从 VLM 原始字符串里抽出 STATUS / SUBGOAL / ANALYSIS。

    宽松匹配:大小写不敏感,允许前后空白。任意字段抽不到返回 None。
    """
    # 匹配策略刻意宽松：大小写不敏感，允许前后空白；抽不到的字段返回 None。
    status_m   = _STATUS_RE.search(text)
    subgoal_m  = _SUBGOAL_RE.search(text)
    analysis_m = _ANALYSIS_RE.search(text)

    return {
        "status":   status_m.group(1).strip()   if status_m  else None,
        "subgoal":  subgoal_m.group(1).strip()  if subgoal_m else None,
        "analysis": analysis_m.group(1).strip() if analysis_m else None,
    }


# ---------------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------------

def update_memory(
    memory: DrivingMemory,
    parsed: Dict[str, Optional[str]],
    *,
    strict: bool = False,
) -> DrivingMemory:
    """把 parsed VLM 响应应用到当前 memory,返回新 memory 实例(原值不变)。

    校验规则:
    - 事件名必须落在 memory.event_sequence 内,否则丢弃
    - status 只允许沿序列前进或保持,不允许回退
    - subgoal 由代码从 status 推导(防止 VLM 跳号),只在 VLM 给出的 subgoal
      与推导值一致时采纳
    """
    # parsed 来自自由文本，不能直接信任。下面先做事件名白名单校验，再做顺序校验。
    new_status  = parsed.get("status")
    new_subgoal = parsed.get("subgoal")

    valid_events = set(memory.event_sequence)

    if new_status and new_status not in valid_events:
        # 模型发明了不存在的状态名。非 strict 模式下丢弃该字段，让 memory 保守保持。
        if strict:
            raise ValueError(
                f"VLM returned STATUS '{new_status}' which is not in the "
                f"event sequence for '{memory.scenario}': {memory.event_sequence}"
            )
        new_status = None

    if new_subgoal and new_subgoal not in valid_events:
        # SUBGOAL 同理，必须来自当前场景状态机。
        if strict:
            raise ValueError(
                f"VLM returned SUBGOAL '{new_subgoal}' which is not in the "
                f"event sequence for '{memory.scenario}': {memory.event_sequence}"
            )
        new_subgoal = None

    final_status = memory.status
    if new_status:
        try:
            current_idx = memory.event_sequence.index(memory.status)
            new_idx     = memory.event_sequence.index(new_status)
            if new_idx >= current_idx:
                # 允许保持或向前推进；不允许倒退。
                final_status = new_status
        except ValueError:
            pass

    derived_subgoal = memory._next_event_after(final_status) or "final"

    if new_subgoal and new_subgoal == derived_subgoal:
        # 只有和状态机推导一致的 SUBGOAL 才信任模型输出。
        final_subgoal = new_subgoal
    else:
        final_subgoal = derived_subgoal

    completed = list(memory.completed_events)
    if final_status not in completed:
        completed.append(final_status)

    return DrivingMemory(
        scenario=memory.scenario,
        scenario_label=memory.scenario_label,
        event_sequence=memory.event_sequence,
        status=final_status,
        subgoal=final_subgoal,
        completed_events=completed,
    )
