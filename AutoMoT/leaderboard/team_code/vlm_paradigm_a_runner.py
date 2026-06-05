"""范式 A 在线运行脚本：把 vlm_prompt_pipeline.py 迁移到 AutoMoT 工程内,
并接上本地 Qwen / AutoMoT 两条对照路径。

设计原则（参考 PROJECT_CONTEXT.md §14.7 第 2 点）:
    范式 A = LLM 当对话模型用 → 自回归 decode 出文本 → 正则解析。
    本文件保留两条互相隔离的范式 A backend:
        - qwen:     本地 checkpoints/Qwen3-VL-4B + HF past_key_values 显式 prefill/decode
        - automot:  AutoMoT ckpt + kv_cache_fixed_inference + gen_text（实测通常立即 EOS）
    AutoMoT inferencer 还自带范式 B:
        - 范式 B:   kv_cache_fixed_inference + based_kv_cache_context_fast_qwen3vl_dp

qwen backend 调用流程（一次 step）:
    1. build_system_prompt() / build_user_prompt(memory, ...) → 拼装文本
    2. processor.apply_chat_template(...) + processor(...)
    3. model(**inputs, use_cache=True)
           ↑ prefill: 一次 forward 把 prompt + 图像编码进 HF past_key_values
    4. _decode_with_explicit_cache(...)
           ↑ decode: 每步只喂上一个 token + past_key_values
    5. parse_vlm_output(raw) → {"analysis":..., "status":..., "subgoal":...}
    6. update_memory(memory, parsed) → 推进状态机

每次 step 都把所有文本输入和输出落盘到 save_dir/step_xxxxxx/ 下,
便于事后审计模型行为。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


def _pick_idle_gpus(n: int = 1) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


def _maybe_set_idle_gpu_mask() -> None:
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"process uses cuda:0; previous={previous or '<unset>'}"
        )


_maybe_set_idle_gpu_mask()

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


# ============================================================================
# 第二段：AutoMoT 路径/tokenizer 预初始化(参考 mot_lead_offline_runner.py)
# ============================================================================
#
# 这一段必须在 import AutoMoT 内部模块之前跑完,因为 automot.py 模块级别会创建
# 全局 tokenizer / processor,如果路径不对就会从 hub 拉默认 ckpt。
# 详见 mot_lead_offline_runner.py 头部相同处理。

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]                        # .../AutoMoT
_AUTOMOT_PROJECT_ROOT = _THIS_FILE.parents[2] / "Automot"    # .../AutoMoT/Automot
_MOT_ROOT = _AUTOMOT_PROJECT_ROOT / "mot"                    # .../AutoMoT/Automot/mot

for _p in (str(_AUTOMOT_ROOT), str(_AUTOMOT_PROJECT_ROOT), str(_MOT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _lazy_setup_automot_paths() -> None:
    """在第一次创建 ParadigmARunner 时才触发(避免 import 该模块就强制加载
    Qwen3 tokenizer / processor,这样在本机做单元测试也能 import 通过)。"""
    qwen3vl_path = str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B")

    import mot.modeling.automot.automot as _automot_module_preset
    _automot_module_preset.QWEN3VL_TOKENIZER_PATH = qwen3vl_path
    _automot_module_preset.QWEN3VL_PROCESSOR_PATH = qwen3vl_path

    if not hasattr(_automot_module_preset, "_tokenizer_reinitialized"):
        from transformers import AutoTokenizer as Qwen3Tokenizer
        from data.reasoning.data_utils import add_special_tokens as _add_special_tokens
        try:
            _tmp_tokenizer = Qwen3Tokenizer.from_pretrained(
                qwen3vl_path, local_files_only=True, trust_remote_code=True
            )
            _tmp_tokenizer, _, _ = _add_special_tokens(_tmp_tokenizer)
            _automot_module_preset.tokenizer = _tmp_tokenizer
            _automot_module_preset._tokenizer_reinitialized = True
            print(f"[vlm_paradigm_a_runner] pre-initialized tokenizer from {qwen3vl_path}")
        except Exception as e:
            print(f"[vlm_paradigm_a_runner] WARNING: could not pre-init tokenizer: {e}")


# ============================================================================
# 第三段：范式 A runner
# ============================================================================

# 适配 PIL.Image:延迟 import,允许本文件在没有 PIL 的环境下也能加载状态机部分
try:
    from PIL import Image as _PILImage  # noqa: F401
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


@dataclass
class ParadigmAStepRecord:
    """一次 step 的完整审计记录。所有字符串与 parsed 字典都会序列化到 JSON。

    图像有两层语义,落盘也分两份:
      - image_files     : 实际喂给 inferencer/processor 的图(可能已被 caller
                          做过 resize / crop / 色彩调整,对齐训练分布)
      - raw_image_files : 最原始未处理的图(caller 显式传 raw_images 才有,
                          用于审计预处理链路是否引入失真)
    """
    step_idx: int
    timestamp: str
    scenario: str
    num_images: int
    memory_before: dict
    system_prompt: str
    user_prompt: str
    combined_prompt: str
    raw_vlm_text: str
    parsed: Dict[str, Optional[str]]
    memory_after: dict
    save_dir: Optional[str] = None
    image_files: List[str] = field(default_factory=list)
    raw_image_files: List[str] = field(default_factory=list)
    num_raw_images: int = 0

    def to_dict(self) -> dict:
        return {
            "step_idx":         self.step_idx,
            "timestamp":        self.timestamp,
            "scenario":         self.scenario,
            "num_images":       self.num_images,
            "num_raw_images":   self.num_raw_images,
            "memory_before":    self.memory_before,
            "system_prompt":    self.system_prompt,
            "user_prompt":      self.user_prompt,
            "combined_prompt":  self.combined_prompt,
            "raw_vlm_text":     self.raw_vlm_text,
            "parsed":           self.parsed,
            "memory_after":     self.memory_after,
            "save_dir":         self.save_dir,
            "image_files":      self.image_files,
            "raw_image_files":  self.raw_image_files,
        }


class ParadigmARunner:
    """把 vlm_prompt_pipeline 状态机接到真实 Qwen3-VL 上的范式 A 执行器。

    用法:
        runner = ParadigmARunner(save_root="eval_json/paradigm_a")
        memory = DrivingMemory.from_scenario("MergerIntoSlowTraffic")
        for step_idx, (images, _) in enumerate(my_data_loader):
            memory, record = runner.run_paradigm_a_step(
                memory=memory,
                images=images,        # list[PIL.Image]
                step_idx=step_idx,
            )
            if memory.is_complete():
                break

    images: list[PIL.Image.Image]
        正面 RGB 图序列。runner 不做 resize / 颜色变换,完全照 AutoMoT 训练分布
        透传(kv_cache_fixed_inference 内部自带必要的 image preprocessing)。
    """

    def __init__(
        self,
        save_root: Optional[str] = None,
        device: str = "cuda",
        max_gen_tokens: int = 256,
        text_temperature: float = 0.0,
        do_sample: bool = False,
    ):
        self.save_root = pathlib.Path(save_root).resolve() if save_root else None
        if self.save_root is not None:
            self.save_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        self.text_temperature = text_temperature
        self.do_sample = do_sample

        self.inferencer = None   # 延迟初始化
        self.automot = None

    # ------------------------------------------------------------------
    # 模型初始化(等真要跑模型时才执行,失败原因不会污染 import 流程)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self.inferencer is not None:
            return

        _lazy_setup_automot_paths()

        from transformers import HfArgumentParser, AutoTokenizer
        from data.reasoning.data_utils import add_special_tokens
        from mot.evaluation.inference import InterleaveInferencer

        from team_code.automot_utils import (
            InferenceArguments,
            ModelArguments,
            load_model_mot,
        )
        from team_code import automot_utils as _automot_utils_module

        parser = HfArgumentParser((ModelArguments, InferenceArguments))
        model_args, inference_args = parser.parse_args_into_dataclasses(args=[])

        actual_model_path = str(_AUTOMOT_ROOT / "checkpoints" / "AutoMoT")
        if os.path.isfile(os.path.join(actual_model_path, "model.safetensors")) or \
           os.path.isfile(os.path.join(actual_model_path, "model.safetensors.index.json")):
            model_args.model_path = actual_model_path

        model_args.qwen3vl_path = str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B")

        import mot.modeling.automot.automot as _automot_module
        _automot_module.QWEN3VL_TOKENIZER_PATH = model_args.qwen3vl_path
        _automot_module.QWEN3VL_PROCESSOR_PATH = model_args.qwen3vl_path

        _automot_utils_module._AUTOMOT_ROOT = str(_AUTOMOT_ROOT)
        _automot_utils_module.ModelArguments.__dataclass_fields__["model_path"].default = model_args.model_path
        _automot_utils_module.ModelArguments.__dataclass_fields__["qwen3vl_path"].default = model_args.qwen3vl_path

        self.automot = load_model_mot(self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.qwen3vl_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.automot.language_model.tokenizer = tokenizer

        self.inferencer = InterleaveInferencer(
            model=self.automot,
            vae_model=None,
            tokenizer=tokenizer,
            vae_transform=None,
            vit_transform=None,
            new_token_ids=new_token_ids,
            max_num_tokens=inference_args.max_num_tokens,
            visual_gen=True,
            visual_und=True,
        )
        print("[vlm_paradigm_a_runner] InterleaveInferencer ready")

    # ------------------------------------------------------------------
    # 一次完整的 step:prefill -> decode -> parse -> update -> dump
    # ------------------------------------------------------------------

    def run_paradigm_a_step(
        self,
        memory: DrivingMemory,
        images: List[Any],
        step_idx: int,
        image_description: str = "<image>",
        save_dir: Optional[str] = None,
        raw_images: Optional[List[Any]] = None,
    ) -> Tuple[DrivingMemory, ParadigmAStepRecord]:
        """跑一次范式 A 推理 + 落盘。

        参数:
            memory:     当前 DrivingMemory。
            images:     list[PIL.Image.Image],**实际喂给 inferencer 的图**
                        (已经过 caller 的 resize / crop / 色彩调整,对齐训练分布)。
                        按时间顺序排列的前视 RGB。
            step_idx:   该 step 在整段 trajectory 内的序号(只用于命名/索引)。
            image_description: 文本里给图像留的占位字符串(纯人读注释)。
            save_dir:   可选,覆盖 self.save_root 决定的默认 step 目录。
            raw_images: 可选,**最原始未处理的图**(例如直接从相机 sensor 出来,
                        或从数据集 .png 读到内存,还没做任何 resize / crop /
                        色彩调整)。提供后会另存一份 raw_image_xxx.png,便于
                        审计预处理链路是否引入失真。

        返回:
            (updated_memory, ParadigmAStepRecord)
        """
        self._ensure_model_loaded()

        system_prompt   = build_system_prompt()
        user_prompt     = build_user_prompt(memory, image_description=image_description)
        combined_prompt = build_combined_prompt(memory, image_description=image_description)

        # ----- prefill 阶段 -----
        # 仿照 mot_lead_offline_runner.run_step 的 slow_input_lists 拼法:
        #   [PIL, PIL, ..., str]  → kv_cache_fixed_inference 内部按类型分流
        slow_input_lists: List[Any] = list(images) + [combined_prompt]

        gen_context = self.inferencer.kv_cache_fixed_inference(slow_input_lists)

        # ----- decode 阶段 -----
        # gen_text 内部对 gen_context 做 deepcopy,所以 gen_context 还能复用给
        # 范式 B(based_kv_cache_context_fast_qwen3vl_dp)——若以后想 A/B 双跑可以
        # 把 gen_context 一起 return,这里先专注范式 A,只返回文本。
        raw_text = self.inferencer.gen_text(
            gen_context,
            max_length=self.max_gen_tokens,
            do_sample=self.do_sample,
            temperature=self.text_temperature,
        )

        # ----- parse + 推进状态机 -----
        parsed = parse_vlm_output(raw_text)
        new_memory = update_memory(memory, parsed)

        # ----- 落盘 -----
        record = ParadigmAStepRecord(
            step_idx=step_idx,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            scenario=memory.scenario,
            num_images=len(images),
            num_raw_images=len(raw_images) if raw_images is not None else 0,
            memory_before=memory.to_dict(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            combined_prompt=combined_prompt,
            raw_vlm_text=raw_text,
            parsed=parsed,
            memory_after=new_memory.to_dict(),
        )

        if self.save_root is not None or save_dir is not None:
            target_dir = pathlib.Path(save_dir) if save_dir is not None \
                else (self.save_root / f"step_{step_idx:06d}")
            self._dump_record(record, images, target_dir, raw_images=raw_images)

        return new_memory, record

    # ------------------------------------------------------------------
    # 文件落盘 helper
    # ------------------------------------------------------------------

    @staticmethod
    def _dump_record(record: ParadigmAStepRecord,
                     images: List[Any],
                     target_dir: pathlib.Path,
                     raw_images: Optional[List[Any]] = None) -> None:
        """把 record 全量落盘,文本/JSON/图像分离存放,便于人工 review 与 diff。

        目录结构:
            target_dir/
                inputs/
                    system_prompt.txt
                    user_prompt.txt
                    combined_prompt.txt
                    memory_before.json
                    image_000.png          ← 实际喂给 inferencer 的图(model input)
                    image_001.png
                    ...
                    raw_image_000.png      ← (可选) 最原始未处理的图
                    raw_image_001.png
                    ...
                outputs/
                    raw_vlm_text.txt
                    parsed.json
                    memory_after.json
                step.json                  # 汇总索引,所有字段一份冗余 JSON
        """
        target_dir = pathlib.Path(target_dir)
        inputs_dir  = target_dir / "inputs"
        outputs_dir = target_dir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # 文本输入
        (inputs_dir / "system_prompt.txt").write_text(record.system_prompt, encoding="utf-8")
        (inputs_dir / "user_prompt.txt").write_text(record.user_prompt, encoding="utf-8")
        (inputs_dir / "combined_prompt.txt").write_text(record.combined_prompt, encoding="utf-8")
        (inputs_dir / "memory_before.json").write_text(
            json.dumps(record.memory_before, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 实际喂给模型的图(已预处理)
        image_files: List[str] = ParadigmARunner._save_pil_list(
            images, inputs_dir, prefix="image_"
        )
        record.image_files = image_files

        # 最原始未处理的图(可选)
        raw_image_files: List[str] = []
        if raw_images is not None and len(raw_images) > 0:
            raw_image_files = ParadigmARunner._save_pil_list(
                raw_images, inputs_dir, prefix="raw_image_"
            )
        record.raw_image_files = raw_image_files

        record.save_dir = str(target_dir)

        # 文本输出
        (outputs_dir / "raw_vlm_text.txt").write_text(record.raw_vlm_text, encoding="utf-8")
        (outputs_dir / "parsed.json").write_text(
            json.dumps(record.parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (outputs_dir / "memory_after.json").write_text(
            json.dumps(record.memory_after, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 汇总索引
        (target_dir / "step.json").write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _save_pil_list(imgs: List[Any],
                       out_dir: pathlib.Path,
                       prefix: str) -> List[str]:
        """把一组 PIL 图按 ``{prefix}{idx:03d}.png`` 命名落盘,返回相对路径列表。"""
        saved: List[str] = []
        for i, img in enumerate(imgs):
            fname = f"{prefix}{i:03d}.png"
            fpath = out_dir / fname
            try:
                if _HAS_PIL and hasattr(img, "save"):
                    img.save(str(fpath))
                    # 这里给 step.json 用的相对路径:相对 target_dir(即 out_dir.parent)
                    saved.append(str(fpath.relative_to(out_dir.parent)))
                else:
                    print(f"[vlm_paradigm_a_runner] {prefix}{i} is not PIL.Image, skip save")
            except Exception as e:
                print(f"[vlm_paradigm_a_runner] failed to save {prefix}{i}: {e}")
        return saved


# ============================================================================
# Baseline runner:AutoMoT/checkpoints 里的本地 Qwen3-VL-4B
# ============================================================================
#
# 与 ParadigmARunner 的关系(对比实验设计):
#
#   ParadigmARunner       <-- AutoMoT ckpt(驾驶 SFT 微调,自定义 MoT 架构)
#                              通过 InterleaveInferencer.kv_cache_fixed_inference
#                              + gen_text 调用
#
#   BaselineQwen3VLRunner <-- AutoMoT/checkpoints/Qwen3-VL-4B 本地 ckpt
#                              通过 AutoProcessor.apply_chat_template
#                              + HF past_key_values 显式 prefill/decode 调用
#
# 两条路径输入一致(同一张/同一组 PIL RGB + 同一个 system/user prompt + 同一份
# DrivingMemory),输出对比能反映"驾驶 SFT 微调对 STATUS/SUBGOAL 指令跟随能力"
# 的影响。
#
# 为什么不能复用同一个 inferencer:
#   AutoMoT 的 layer_module="Qwen3VLMoTDecoderLayer",且额外定义了
#   reasoning_queries / action_queries / waypoints_head 等模块,state_dict
#   完全不兼容 Qwen3-VL-4B。所以两个 ckpt 必须各走各的加载链路。
# ============================================================================


class BaselineQwen3VLRunner:
    """直接用 AutoMoT/checkpoints/Qwen3-VL-4B 本地权重跑范式 A。

    用法:
        runner = BaselineQwen3VLRunner(save_root="eval_json/paradigm_a_qwen_baseline")
        memory = DrivingMemory.from_scenario("MergerIntoSlowTraffic")
        new_memory, record = runner.run_paradigm_a_step(
            memory=memory, images=[pil1, pil2, ...], step_idx=0,
        )
    """

    def __init__(
        self,
        save_root: Optional[str] = None,
        device: str = "cuda",
        max_gen_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        torch_dtype_str: str = "bfloat16",
    ):
        self.save_root = pathlib.Path(save_root).resolve() if save_root else None
        if self.save_root is not None:
            self.save_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.torch_dtype_str = torch_dtype_str

        self.model = None
        self.processor = None

    # ------------------------------------------------------------------
    # 模型加载:只读本地 AutoMoT/checkpoints/Qwen3-VL-4B,不联网下载
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        import torch
        qwen3vl_path = str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B")
        print(f"[BaselineQwen3VLRunner] loading local Qwen3-VL-4B from {qwen3vl_path} ...")

        # transformers >= 4.45 自带 Qwen3VLForConditionalGeneration;
        # 若版本太旧请升级 transformers。
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            # 退一步走通用 AutoModelForVision2Seq
            from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.bfloat16)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen3vl_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            qwen3vl_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("[BaselineQwen3VLRunner] local Qwen3-VL-4B ready")

    def _decode_with_explicit_cache(self, inputs: Any) -> Any:
        """本地 Qwen 专用:显式 prefill KV cache,再逐 token decode 文本。

        这里故意不复用 AutoMoT 的 InterleaveInferencer.gen_text:那套函数吃的是
        AutoMoT 自定义 MoT 模型的 NaiveCache / new_token_ids / start-token 逻辑。
        本地 Qwen baseline 使用 transformers 标准 past_key_values。
        """
        import torch

        # ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
        # input_ids shape: torch.Size([1, 2316])
        # attention_mask shape: torch.Size([1, 2316])
        # pixel_values shape: torch.Size([6912, 1536])
        # image_grid_thw shape: torch.Size([4, 3])




        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids = set()
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_token_ids = {int(x) for x in eos_token_id}
        else:
            eos_token_ids = {int(eos_token_id)}

        attention_mask = inputs.get("attention_mask", None)
        decoded_input_ids = inputs["input_ids"]
        generated_tokens: List[Any] = []

        # Prefill:完整多模态 prompt 只 forward 一次,拿到 KV cache 和首个 token logits。
        outputs = self.model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

        # keys: ['logits', 'past_key_values', 'rope_deltas']
        # 'logits': torch.Size([1, 2316, 151936])
        # 'past_key_values': <class 'transformers.cache_utils.DynamicCache'>, 
        # 'rope_deltas': torch.Size([1, 1])}

        past_key_values = outputs.past_key_values
        # outputs.logits 形状为 (batch_size, seq_len, vocab_size)
        # 取最后一个时间步 logits (index -1) 的原因:
        # 在自回归/生成场景中，模型会对输入序列中每个位置预测下一个 token 的分布。
        # 当我们把完整的 prompt 一次性前向（prefill）后，序列的最后一个位置对应的是
        # "在已给定 prompt 之后模型接下来要生成的第一个 token" 的 logits。
        # 因此使用 outputs.logits[:, -1, :] 作为下一步采样/贪心决策的依据。
        next_logits = outputs.logits[:, -1, :]
        # torch.Size([1, 151936])

        for _ in range(self.max_gen_tokens):
            if self.do_sample:
                logits = next_logits / max(self.temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # torch.Size([1, 1])
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token)
            decoded_input_ids = torch.cat([decoded_input_ids, next_token], dim=1)

            token_id = int(next_token[0, 0].item())
            if token_id in eos_token_ids:
                break

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                    dim=1,
                )


            # Decode:后续每步只喂刚生成的 token + 上一步 KV cache。
            if hasattr(self.model, "prepare_inputs_for_generation"):
                cache_position = torch.arange(
                    decoded_input_ids.shape[1] - 1,
                    decoded_input_ids.shape[1],
                    device=decoded_input_ids.device,
                )
                model_inputs = self.model.prepare_inputs_for_generation(
                    decoded_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )
                # keys: ['cache_position', 'past_key_values', 'input_ids', 'inputs_embeds', 'position_ids', 'attention_mask', 'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'use_cache']
                # model_inputs input_ids shape: torch.Size([1, 1])
            else:
                model_inputs = {
                    "input_ids": next_token,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask

            outputs = self.model(**model_inputs, return_dict=True)
            #  {'logits': torch.Size([1, 1, 151936]), 
            # 'past_key_values': <class 'transformers.cache_utils.DynamicCache'>, 
            # 'rope_deltas': torch.Size([1, 1])}


            past_key_values = outputs.past_key_values
            # 同样道理：每次 decode 步骤模型返回的 logits 形状仍为
            # (batch_size, seq_len, vocab_size)。当我们只喂入最新生成的 token
            # 并带上 past_key_values 时，最后一个位置的 logits 对应下一步要选的 token。
            next_logits = outputs.logits[:, -1, :]

        if not generated_tokens:
            return inputs["input_ids"].new_empty((1, 0))
        return torch.cat(generated_tokens, dim=1)

    # ------------------------------------------------------------------
    # 一次完整 step:apply_chat_template -> processor -> prefill -> decode
    # ------------------------------------------------------------------

    def run_paradigm_a_step(
        self,
        memory: DrivingMemory,
        images: List[Any],
        step_idx: int,
        save_dir: Optional[str] = None,
        raw_images: Optional[List[Any]] = None,
    ) -> Tuple[DrivingMemory, ParadigmAStepRecord]:
        """与 ParadigmARunner.run_paradigm_a_step 接口完全一致。

        raw_images 可选;提供后会另存一份 raw_image_xxx.png,语义同 ParadigmARunner。
        """
        import torch
        self._ensure_model_loaded()

        system_prompt = build_system_prompt()
        user_prompt   = build_user_prompt(memory, image_description="<image>")

        # 用本地 processor 的 Qwen3-VL chat template 组消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": user_prompt}]
            )},
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


        inputs = self.processor(
            text=[chat_text],
            images=images if len(images) > 0 else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        # processor inputs keys: ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

        with torch.no_grad():
            new_ids = self._decode_with_explicit_cache(inputs)

        raw_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        raw_text = raw_text.lstrip("\n ")
        # ANALYSIS: The ego vehicle is approaching a dark, foggy road where a hazard (a vehicle ahead) is visible, indicating the hazard detection phase has begun. The scene shows reduced visibility and the presence of an obstacle ahead, consistent with hazard detection.
        # STATUS: hazard_detect
        # SUBGOAL: max_brake_or_min_gap

        parsed = parse_vlm_output(raw_text)
        new_memory = update_memory(memory, parsed)

        record = ParadigmAStepRecord(
            step_idx=step_idx,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            scenario=memory.scenario,
            num_images=len(images),
            num_raw_images=len(raw_images) if raw_images is not None else 0,
            memory_before=memory.to_dict(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            # 这里 combined_prompt 字段复用为 "实际喂给 processor 的 chat 模板化文本",
            # 便于和 ParadigmARunner 的 combined_prompt(AutoMoT instruction 拼法)直接对比
            combined_prompt=chat_text,
            raw_vlm_text=raw_text,
            parsed=parsed,
            memory_after=new_memory.to_dict(),
        )

        if self.save_root is not None or save_dir is not None:
            target_dir = pathlib.Path(save_dir) if save_dir is not None \
                else (self.save_root / f"step_{step_idx:06d}")
            ParadigmARunner._dump_record(record, images, target_dir, raw_images=raw_images)

        return new_memory, record


# ---------------------------------------------------------------------------
# Framework-agnostic helper(便于不挂模型也能单跑文本流程做单元测试)
# ---------------------------------------------------------------------------

VLMFn = Callable[[str, str, List[Any]], str]


def run_pipeline_step(
    memory: DrivingMemory,
    images: List[Any],
    vlm_fn: VLMFn,
    image_description: str = "<image>",
) -> Tuple[DrivingMemory, Dict[str, Optional[str]], str]:
    """vlm_prompt_pipeline.py 原版 run_pipeline_step 的扩展版。

    与原版差异:
    - vlm_fn 签名从 ``(system, user) -> str`` 改成
      ``(system, user, images) -> str``,显式带上图像列表,匹配范式 A 多模态
      调用(原版只把图像当占位字符串,没法接 InterleaveInferencer)。
    - 返回值多带回 raw_text,方便上层一起落盘(原版只返回 parsed,
      想审计 raw 还得在 vlm_fn 内部 hack)。

    用途:
        - 单元测试(用 dummy vlm_fn 模拟"模型按格式输出三行")
        - 想自己接非 AutoMoT 模型(如 OpenAI / Anthropic 多模态 API)时,
          只要实现一个 vlm_fn 就可以复用整套状态机
    """
    system = build_system_prompt()
    user   = build_user_prompt(memory, image_description=image_description)
    raw    = vlm_fn(system, user, images)
    parsed = parse_vlm_output(raw)
    updated_memory = update_memory(memory, parsed)
    return updated_memory, parsed, raw


# ---------------------------------------------------------------------------
# __main__:默认走本地 Qwen3-VL-4B + AutoMoT 权重 smoke test;--dry-run 走
# 纯文本桩(不加载模型)。
# ---------------------------------------------------------------------------

def _dry_run_self_test() -> None:
    """不依赖 Qwen 权重,仅验证 prompt / parse / update / dump 全链路语义。
    需要显式 --dry-run 才会走;默认 main 直接拉真模型。"""
    print("=" * 60)
    print("vlm_paradigm_a_runner DRY-RUN self-test (no real VLM)")
    print("=" * 60)

    memory = DrivingMemory.from_scenario("MergerIntoSlowTraffic")
    print(f"initial memory: {memory.to_dict()}\n")

    def dummy_vlm_fn(system: str, user: str, images: List[Any]) -> str:
        return (
            "ANALYSIS: The ego vehicle observes slower vehicles ahead in the "
            "right lane and is decelerating to match their pace.\n"
            "STATUS: match_speed\n"
            "SUBGOAL: merge_complete\n"
        )

    new_memory, parsed, raw = run_pipeline_step(memory, images=[], vlm_fn=dummy_vlm_fn)
    print(f"raw vlm text:\n{raw}\n")
    print(f"parsed:        {parsed}")
    print(f"updated memory:{new_memory.to_dict()}")

    # 演示双份图像落盘:raw 用 (1920,1080),model_input 用 (1152,384)
    raw_imgs, mi_imgs = _build_synthetic_raw_and_model_input(num_frames=2)

    tmp_dir = pathlib.Path(__file__).parent / "_paradigm_a_self_test_out"
    record = ParadigmAStepRecord(
        step_idx=0,
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        scenario=memory.scenario,
        num_images=len(mi_imgs),
        num_raw_images=len(raw_imgs),
        memory_before=memory.to_dict(),
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(memory),
        combined_prompt=build_combined_prompt(memory),
        raw_vlm_text=raw,
        parsed=parsed,
        memory_after=new_memory.to_dict(),
    )
    ParadigmARunner._dump_record(
        record, images=mi_imgs, target_dir=tmp_dir / "step_000000",
        raw_images=raw_imgs,
    )
    print(f"\ndumped DRY-RUN record to: {tmp_dir / 'step_000000'}")
    print(f"  inputs/image_*.png        (model input, {mi_imgs[0].size})")
    print(f"  inputs/raw_image_*.png    (raw,         {raw_imgs[0].size})")


def _build_synthetic_images(num_frames: int = 4,
                            height: int = 384,
                            width: int = 1152) -> List[Any]:
    """生成 N 张合成 PIL RGB 图(三色横条 + 帧号角标),目标尺寸 (height, width)。

    - 默认 shape (H=384, W=1152) 对齐 mot_lead_offline_runner 的 LEAD 风格
      训练分布(input term: <PIL.Image.Image image mode=RGB size=1152x384>)。
    - 设计目标:落盘 PNG 一眼看清是 3 通道 RGB,且每帧明显不同:
        * 上 1/3 整片纯红 / 中 1/3 整片纯绿 / 下 1/3 整片纯蓝
        * 横向叠加 0→100 灰度衰减,让图有结构而非纯色块
        * 左上角 80×80 色块,每帧用不同饱和色编码帧号(黄/青/品红/橙循环)
    """
    if not _HAS_PIL:
        raise RuntimeError("PIL 不可用,无法生成合成图像")
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"smoke test 需要 numpy: {e}")

    images: List[Any] = []
    third = height // 3

    fade = np.linspace(0, 100, width, dtype=np.int16)
    fade_2d = np.tile(fade[None, :], (height, 1))

    marker_colors = [
        (255, 255,   0),   # 黄
        (  0, 255, 255),   # 青
        (255,   0, 255),   # 品红
        (255, 128,   0),   # 橙
    ]

    # 帧号角标大小按图像尺寸自适应(取短边的 1/5,夹在 [40, 200])
    marker_size = max(40, min(200, min(height, width) // 5))

    for t in range(num_frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:third,            :, 0] = 255
        rgb[third:2 * third,   :, 1] = 255
        rgb[2 * third:,        :, 2] = 255

        rgb_int16 = rgb.astype(np.int16) - fade_2d[..., None]
        rgb = np.clip(rgb_int16, 0, 255).astype(np.uint8)

        mc = marker_colors[t % len(marker_colors)]
        rgb[:marker_size, :marker_size, 0] = mc[0]
        rgb[:marker_size, :marker_size, 1] = mc[1]
        rgb[:marker_size, :marker_size, 2] = mc[2]

        images.append(_PILImage.fromarray(rgb))
    return images


def _build_synthetic_raw_and_model_input(
    num_frames: int = 4,
    raw_size: Tuple[int, int] = (1920, 1080),       # (W, H) 模拟原始相机分辨率
    model_input_size: Tuple[int, int] = (1152, 384), # (W, H) LEAD 风格训练分布
) -> Tuple[List[Any], List[Any]]:
    """生成 "原始 raw 图" 与 "已预处理 model-input 图" 两份配对图像。

    返回:
        (raw_imgs, model_input_imgs) —— 两个 list 长度都是 num_frames,
        同一索引位置内容一致,只是分辨率/纵横比不同。
        model_input_imgs 是把 raw_imgs 各自 resize 到 model_input_size 的结果,
        模拟"caller 拿到原始相机帧 → resize 到训练分布 → 喂进 inferencer"。
    """
    raw_w, raw_h = raw_size
    mi_w,  mi_h  = model_input_size

    raw_imgs = _build_synthetic_images(
        num_frames=num_frames, height=raw_h, width=raw_w,
    )
    # 用 PIL.Image.resize 做"预处理",生成 model input
    model_input_imgs = [
        img.resize((mi_w, mi_h), resample=_PILImage.BILINEAR) for img in raw_imgs
    ]
    return raw_imgs, model_input_imgs


# ---------------------------------------------------------------------------
# 真实 LEAD 数据加载(借鉴 mot_lead_offline_runner.build_clip_from_real_lead_route
# 的 RGB-only 部分;不依赖 laspy/AutoMoT 重型 import,启动开销小)
# ---------------------------------------------------------------------------

# LEAD 路径里典型 scenario 段位置:
#   /datashare/IOL4SGH/data/data/<Scenario>/Town03_Rep0_route_..../
#                       ^^^^^^^^^^
# 自动从 route_dir 提取 scenario 名,若名字在 SCENARIO_LABELS 里就直接用。

def _auto_detect_scenario_from_route(route_dir: str) -> Optional[str]:
    """尝试从 LEAD 路径里抠出 scenario 名;失败返回 None。"""
    parts = pathlib.Path(route_dir).resolve().parts
    for p in reversed(parts):
        if p in SCENARIO_LABELS:
            return p
    return None


def _ensure_hwc_uint8(img: Any) -> Any:
    """复刻 LeadOfflineMoTRunner._ensure_hwc_uint8。

    源出处: mot_lead_offline_runner.py:_ensure_hwc_uint8 (约 L754)。
    这里**逐字复刻**而非 import,避免触发 AutoMoT 重型模块初始化(tokenizer / processor 等)。

    功能:把任意输入规范化为 (H, W, 3) uint8 RGB:
        - CHW → HWC(若首维是 1/3/4)
        - float → uint8(假定 [0,1] 范围)
        - 通道 >3 → 截前 3 个
        - 通道 ==1 → 复制成 3 通道
    """
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"_ensure_hwc_uint8 需要 numpy: {e}")

    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"RGB frame ndim invalid: {arr.ndim}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _load_lead_rgb_clip(
    route_dir: str,
    anchor: int = 12,
    rgb_frame_step: int = 1,
    rgb_frame_count: int = 4,
) -> Tuple[List[Any], List[Any]]:
    """从真实 LEAD route 目录读 anchor 时刻的 RGB 历史帧(按时间顺序)。

    数据路径**完全镜像** mot_lead_offline_runner,分两阶段:

      阶段 1 (对应 build_clip_from_real_lead_route L1456-L1460):
          cv2.imread → cv2.cvtColor(BGR2RGB) → 追加到 rgb_list
          最后 np.stack(rgb_list) → (T, H, W, 3) uint8

      阶段 2 (对应 _prepare_inference_inputs L925-L933):
          for idx in asc:
              rgb_i = rgb_clip[idx]                          # (H, W, 3)
              rgb_hwc = _ensure_hwc_uint8(rgb_i)              # 规范化
              rgb_pil_list.append(Image.fromarray(rgb_hwc, mode="RGB"))

    采样规则同 _build_group_indices(L856-L870):
          desc = [max(anchor - i*step, 0) for i in range(count)]
          asc  = list(reversed(desc))
      不足处通过 max(..., 0) 钳到 frame 0(会重复采样,与 runner 同款 warning 行为)。

    LEAD 在线模式 RGB 已是 (1152, 384) 训练分布,runner 不做 resize,
    所以 model_input == raw(共享同一组 PIL 对象,落盘时仍会另存一份保持接口对称)。

    返回:
        (raw_imgs, model_input_imgs) —— 两份 list 长度相同,内容一致。

    注意:本函数**只读 RGB**,不读 meta/lidar/pose,所以不需要 laspy / xz pickle
    依赖,也不会触发 AutoMoT 模块的重型 import,启动开销 ≈ 仅 cv2 + PIL + numpy。
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"需要 opencv-python + numpy: {e}")
    if not _HAS_PIL:
        raise RuntimeError("PIL 不可用,无法构造 PIL.Image")

    route = pathlib.Path(route_dir)
    if not route.exists():
        raise FileNotFoundError(f"route_dir 不存在: {route_dir}")
    rgb_dir = route / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"route_dir 下缺少 rgb 子目录: {rgb_dir}")

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"{rgb_dir} 下没有 .jpg 文件")
    total = len(rgb_files)

    if anchor < 0:
        raise ValueError(f"anchor 必须 >= 0,当前 anchor={anchor}")
    if anchor >= total:
        raise ValueError(
            f"anchor={anchor} 超出 route 范围(总帧数={total},合法 [0, {total - 1}])"
        )

    rgb_frame_step = max(1, rgb_frame_step)
    rgb_frame_count = max(1, rgb_frame_count)

    # 同 mot_lead_offline_runner._build_group_indices
    desc = [max(anchor - i * rgb_frame_step, 0) for i in range(rgb_frame_count)]
    asc = list(reversed(desc))

    # 历史不足时的 warning(与 runner 同款语义)
    ideal_start = anchor - (rgb_frame_count - 1) * rgb_frame_step
    if ideal_start < 0:
        print(
            f"[警告] anchor={anchor} 历史不足:需要 {(rgb_frame_count - 1) * rgb_frame_step} "
            f"帧历史但 route 起点仅到 0,采样里会重复 frame 0 共 {-ideal_start} 次"
        )

    print(
        f"[load] route={route} total_frames={total} anchor={anchor} "
        f"rgb_step={rgb_frame_step} rgb_count={rgb_frame_count}"
    )
    print(f"[load] sampled rgb indices (asc): {asc}")

    # =====================================================================
    # 阶段 1: cv2 读图 → BGR→RGB → 追加到 list (镜像 build_clip_from_real_lead_route)
    # =====================================================================
    rgb_ndarray_list: List[Any] = []
    for idx in asc:
        stem = f"{idx:04d}"
        rgb_path = rgb_dir / f"{stem}.jpg"
        if not rgb_path.exists():
            # 兼容部分 LEAD 命名(其它位宽),退一步用文件列表索引
            rgb_path = rgb_files[idx]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 读图失败: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # (H, W, 3) uint8
        rgb_ndarray_list.append(rgb)

    # np.stack 成 (T, H, W, 3) —— 对齐 runner clip["rgb"] 的存储形态。
    # 即使后续不真的索引,这一步保留是为了让数据路径与 runner 完全一致。
    rgb_clip = np.stack(rgb_ndarray_list, axis=0)
    # 期望 shape (T, 384, 1152, 3),dtype=uint8

    # =====================================================================
    # 阶段 2: 逐帧 _ensure_hwc_uint8 → PIL.fromarray (镜像 _prepare_inference_inputs)
    # =====================================================================
    raw_imgs: List[Any] = []
    for t in range(rgb_clip.shape[0]):
        rgb_i = rgb_clip[t]                              # (H, W, 3)
        rgb_hwc = _ensure_hwc_uint8(rgb_i)               # 规范化(此处对 LEAD jpg 是 no-op,
                                                          # 但保留以兼容其它 dtype/shape 输入)
        raw_imgs.append(_PILImage.fromarray(rgb_hwc, mode="RGB"))

    # LEAD 原图已对齐训练分布(1152, 384),无需再 resize;
    # 双份落盘接口保持对称,所以 model_input == raw(共享同一组 PIL 对象)。
    model_input_imgs = list(raw_imgs)

    if len(raw_imgs) > 0:
        print(
            f"[load] loaded {len(raw_imgs)} frames, "
            f"clip_shape={tuple(rgb_clip.shape)}, "
            f"PIL size={raw_imgs[0].size}, mode={raw_imgs[0].mode}"
        )
    return raw_imgs, model_input_imgs


def _run_one_backend(backend: str,
                     memory: DrivingMemory,
                     images: List[Any],
                     save_root: str,
                     max_gen_tokens: int,
                     raw_images: Optional[List[Any]] = None,
                     ) -> Tuple[DrivingMemory, ParadigmAStepRecord]:
    """跑某一个 backend 的范式 A 推理,统一接口便于对比。

    backend:
        "automot"  -> ParadigmARunner       (AutoMoT 微调 ckpt)
        "qwen"     -> BaselineQwen3VLRunner (AutoMoT/checkpoints/Qwen3-VL-4B)
    raw_images: 可选,原始未预处理图,会额外落盘到 inputs/raw_image_xxx.png。
    """
    print("\n" + "=" * 60)
    print(f"[backend={backend}] loading model & running paradigm A step ...")
    print("=" * 60)

    if backend == "automot":
        runner = ParadigmARunner(
            save_root=save_root,
            device="cuda",
            max_gen_tokens=max_gen_tokens,
            text_temperature=0.0,
            do_sample=False,
        )
    elif backend == "qwen":
        runner = BaselineQwen3VLRunner(
            save_root=save_root,
            device="cuda",
            max_gen_tokens=max_gen_tokens,
            temperature=0.0,
            do_sample=False,
            torch_dtype_str="bfloat16",
        )
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    runner._ensure_model_loaded()
    new_memory, record = runner.run_paradigm_a_step(
        memory=memory, images=images, step_idx=0, raw_images=raw_images,
    )

    print("-" * 60)
    print(f"[backend={backend}] raw VLM text (len={len(record.raw_vlm_text)} chars):")
    print(record.raw_vlm_text if record.raw_vlm_text else "<EMPTY STRING>")
    if len(record.raw_vlm_text) < 5:
        # 极短输出多半是模型立即吐 EOS;给出诊断提示而不是让用户看一片空白
        print(f"[backend={backend}] ⚠ raw_vlm_text 长度 < 5,repr={record.raw_vlm_text!r}")
        if backend == "automot":
            print(f"[backend={backend}]   AutoMoT 的 lm_head 训练时只为 reasoning_query 第 2 位做")
            print(f"[backend={backend}]   stop/keep 二分类(见 PROJECT_CONTEXT §14.3.4),autoregressive")
            print(f"[backend={backend}]   自由文本生成路径未受 SFT,常见现象是立即生成 EOS = 空字符串。")
            print(f"[backend={backend}]   想要文字输出,要么换本地 Qwen3-VL(--backend qwen),")
            print(f"[backend={backend}]   要么改走范式 B 直接拿 reasoning_hidden_states 接 head。")
    print("-" * 60)
    print(f"[backend={backend}] parsed:       {record.parsed}")
    print(f"[backend={backend}] memory after: {new_memory.to_dict()}")
    print(f"[backend={backend}] saved to:     {record.save_dir}")
    print(f"[backend={backend}] image files (model input): {record.image_files}")
    print(f"[backend={backend}] raw image files:           {record.raw_image_files}")
    return new_memory, record


def _real_smoke_test(backend: str = "automot",
                     save_root: Optional[str] = None,
                     scenario: str = "MergerIntoSlowTraffic",
                     num_frames: int = 4,
                     max_gen_tokens: int = 256,
                     route_dir: Optional[str] = None,
                     anchor: int = 12,
                     rgb_frame_step: int = 1) -> None:
    """真实加载模型跑一次 paradigm A 端到端。

    backend:
        "automot" -> 只跑 AutoMoT 微调 ckpt
        "qwen"    -> 只跑 AutoMoT/checkpoints/Qwen3-VL-4B
        "both"    -> 两个都跑(分别落到 <save_root>/automot/ 和 <save_root>/qwen/),
                     便于直接 diff 两个 raw_vlm_text.txt

    route_dir:
        None        -> 用合成图(三色横条 + 角标),仅验证 prefill+decode 通路
        <real path> -> 用真实 LEAD route 目录里的 RGB,语义可信
                       (借鉴 mot_lead_offline_runner.build_clip_from_real_lead_route)
    """
    print("=" * 60)
    print(f"vlm_paradigm_a_runner REAL smoke test (backend={backend})")
    print("=" * 60)
    print(f"AutoMoT root: {_AUTOMOT_ROOT}")
    print(f"qwen3vl ckpt: {_AUTOMOT_ROOT / 'checkpoints' / 'Qwen3-VL-4B'}  (local baseline)")
    print(f"automot ckpt: {_AUTOMOT_ROOT / 'checkpoints' / 'AutoMoT'}      (driving SFT)")

    if save_root is None:
        save_root = str(_AUTOMOT_ROOT / "eval_json" / "paradigm_a_smoke_test")
    print(f"save_root:    {save_root}\n")

    # 优先用真实 LEAD RGB;否则退回合成图
    if route_dir:
        print(f"[input source] real LEAD route: {route_dir}")
        raw_images, model_input_images = _load_lead_rgb_clip(
            route_dir=route_dir,
            anchor=anchor,
            rgb_frame_step=rgb_frame_step,
            rgb_frame_count=num_frames,
        )

        # 尝试从路径自动识别 scenario(覆盖 --scenario 默认值)
        auto_scen = _auto_detect_scenario_from_route(route_dir)
        if auto_scen is not None and auto_scen != scenario:
            print(f"[scenario] auto-detected '{auto_scen}' from route path,"
                  f" overriding --scenario='{scenario}'")
            scenario = auto_scen
        elif auto_scen is None:
            print(f"[scenario] could not auto-detect from path, using --scenario='{scenario}'")
    else:
        print("[input source] synthetic test pattern (no --route-dir given)")
        raw_images, model_input_images = _build_synthetic_raw_and_model_input(
            num_frames=num_frames,
            raw_size=(1920, 1080),               # (W, H) 模拟原始相机分辨率
            model_input_size=(1152, 384),        # (W, H) LEAD 训练分布
        )

    print(f"prepared {len(raw_images)} frames:")
    print(f"  raw         : size={raw_images[0].size}, mode={raw_images[0].mode}")
    print(f"  model input : size={model_input_images[0].size}, mode={model_input_images[0].mode}")

    if backend in ("automot", "qwen"):
        memory = DrivingMemory.from_scenario(scenario)
        print(f"initial memory: {memory.to_dict()}")
        _run_one_backend(
            backend, memory, model_input_images,
            save_root=save_root,
            max_gen_tokens=max_gen_tokens,
            raw_images=raw_images,
        )

    elif backend == "both":
        # 两个 backend 各跑一次,起始 memory 一致,落盘到不同子目录便于 diff
        for sub in ("automot", "qwen"):
            memory = DrivingMemory.from_scenario(scenario)   # 每个 backend 都用全新 memory
            print(f"\ninitial memory: {memory.to_dict()}")
            sub_root = str(pathlib.Path(save_root) / sub)
            _run_one_backend(
                sub, memory, model_input_images,
                save_root=sub_root,
                max_gen_tokens=max_gen_tokens,
                raw_images=raw_images,
            )

        # 对比提示
        print("\n" + "=" * 60)
        print("对比文件路径(diff 这两个看微调对指令跟随的影响):")
        print(f"  automot raw : {save_root}/automot/step_000000/outputs/raw_vlm_text.txt")
        print(f"  qwen    raw : {save_root}/qwen/step_000000/outputs/raw_vlm_text.txt")
        print("=" * 60)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    if route_dir:
        print("\n[OK] 输入是真实 LEAD RGB,语义可信。raw_vlm_text 反映模型真实视觉理解能力。")
    else:
        print("\n[注意] 输入是合成渐变图,语义不可信 —— 仅验证 prefill+decode 通路。")
        print("       想做真实语义对比,加 --route-dir <LEAD route 路径> 即可。")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="vlm_paradigm_a_runner CLI")
    p.add_argument("--backend", choices=["automot", "qwen", "both"], default="both",
                   help="automot = AutoMoT 微调 ckpt;qwen = checkpoints/Qwen3-VL-4B 本地权重;both = 两个都跑做对比")
    p.add_argument("--dry-run", action="store_true",
                   help="不加载任何权重,只跑文本桩测试 prompt/parse/update 链路")
    p.add_argument("--scenario", type=str, default="MergerIntoSlowTraffic",
                   help="DrivingMemory.from_scenario 用的场景名;"
                        "若 --route-dir 路径里能自动识别 scenario,会覆盖此值")
    p.add_argument("--num-frames", type=int, default=4,
                   help="RGB 历史帧数(默认 4,对齐 LEAD 风格采样)。"
                        "合成模式下决定 _build_synthetic 的帧数;真实模式下决定采样窗口")
    p.add_argument("--max-gen-tokens", type=int, default=256,
                   help="decode 最大新生成 token 数")
    p.add_argument("--save-root", type=str, default=None,
                   help="落盘根目录;默认 <AutoMoT>/eval_json/paradigm_a_smoke_test")

    # ---- LEAD 真实数据入口(借鉴 mot_lead_offline_runner 同名参数) ----
    p.add_argument("--route-dir", type=str, default='/datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46',
                   help="真实 LEAD 路由目录,目录下需有 rgb/*.jpg 子目录。"
                        "示例: /datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_...。"
                        "不传 → 用合成图(仅验证通路,语义不可信)。")
    p.add_argument("--anchor", type=int, default=12,
                   help="待处理的 anchor 帧索引(route 内绝对索引,0-based)。"
                        "仅在 --route-dir 提供时生效")
    p.add_argument("--rgb-frame-step", type=int, default=1,
                   help="RGB 历史采样步长(单位:帧)。默认 1(LEAD 每帧 ~0.25s),"
                        "设 5 约 1.25s 间隔。仅 --route-dir 时生效")
    args = p.parse_args()

    if args.dry_run:
        _dry_run_self_test()
    else:
        _real_smoke_test(
            backend=args.backend,
            save_root=args.save_root,
            scenario=args.scenario,
            num_frames=args.num_frames,
            max_gen_tokens=args.max_gen_tokens,
            route_dir=args.route_dir,
            anchor=args.anchor,
            rgb_frame_step=args.rgb_frame_step,
        )
