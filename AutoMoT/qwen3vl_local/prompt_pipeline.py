"""Local copy of the vlm_prompt_pipeline state machine used by Qwen3-VL paradigm A.

This file is migrated from AutoMoT/leaderboard/team_code/vlm_paradigm_a_runner.py
so the standalone Qwen runner uses the same prompts, event descriptions, parser,
and memory update semantics as the original paradigm-A smoke test.
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

    scenario: str
    scenario_label: str
    event_sequence: Tuple[str, ...]
    status: str
    subgoal: str
    completed_events: List[str] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, scenario: str) -> "DrivingMemory":
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
        return {
            "scenario":         self.scenario,
            "scenario_label":   self.scenario_label,
            "event_sequence":   list(self.event_sequence),
            "status":           self.status,
            "subgoal":          self.subgoal,
            "completed_events": self.completed_events,
        }

    def is_complete(self) -> bool:
        return self.status == "final"

    def status_description(self) -> str:
        return EVENT_DESCRIPTIONS.get(self.status, self.status)

    def subgoal_description(self) -> str:
        return EVENT_DESCRIPTIONS.get(self.subgoal, self.subgoal)

    def _next_event_after(self, event: str) -> Optional[str]:
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
    return _SYSTEM_PROMPT


def build_memory_block(memory: DrivingMemory) -> str:
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


def build_user_prompt(memory: DrivingMemory, image_description: str = "<image>") -> str:
    """构造 user 段提示词。image_description 用于在文本里给图像留占位标记,
    真正的图像通过 inferencer 的 image_list 参数另路注入(范式 A 复用 AutoMoT
    现成的 USER_PROMPT 模板,模板中已含 <|vision_start|>...<|vision_end|>
    视觉占位 token,所以这里的字符串占位仅是给人读的注释)。"""
    memory_block = build_memory_block(memory)
    return (
        f"{image_description}\n\n"
        f"{memory_block}\n\n"
        "Given the observations above and the memory context, output your "
        "ANALYSIS, STATUS, and SUBGOAL."
    )


def build_combined_prompt(memory: DrivingMemory,
                          image_description: str = "<image>") -> str:
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
    new_status  = parsed.get("status")
    new_subgoal = parsed.get("subgoal")

    valid_events = set(memory.event_sequence)

    if new_status and new_status not in valid_events:
        if strict:
            raise ValueError(
                f"VLM returned STATUS '{new_status}' which is not in the "
                f"event sequence for '{memory.scenario}': {memory.event_sequence}"
            )
        new_status = None

    if new_subgoal and new_subgoal not in valid_events:
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
                final_status = new_status
        except ValueError:
            pass

    derived_subgoal = memory._next_event_after(final_status) or "final"

    if new_subgoal and new_subgoal == derived_subgoal:
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
