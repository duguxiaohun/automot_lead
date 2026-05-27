"""范式 A 在线运行脚本：把 vlm_prompt_pipeline.py 迁移到 AutoMoT 工程内,
并接上真实的 Qwen3-VL（通过 AutoMoT 的 InterleaveInferencer）。

设计原则（参考 PROJECT_CONTEXT.md §14.7 第 2 点）:
    范式 A = LLM 当对话模型用 → `.generate()` 出文本 → 正则解析。
    AutoMoT 同一个 inferencer 自带两种范式:
        - 范式 A:   kv_cache_fixed_inference + gen_text         (本文件)
        - 范式 B:   kv_cache_fixed_inference + based_kv_cache_context_fast_qwen3vl_dp

调用流程（一次 step）:
    1. build_system_prompt() / build_user_prompt(memory, ...) → 拼装文本
    2. inferencer.kv_cache_fixed_inference([img1, img2, ..., combined_prompt])
           ↑ prefill: 一次 forward 把 prompt + 图像编码进 KV cache
    3. inferencer.gen_text(gen_context, max_length=200)
           ↑ decode: autoregressive 出文本字符串
    4. parse_vlm_output(raw) → {"analysis":..., "status":..., "subgoal":...}
    5. update_memory(memory, parsed) → 推进状态机

每次 step 都把所有文本输入和输出落盘到 save_dir/step_xxxxxx/ 下,
便于事后审计模型行为。

只新增本文件,不修改 AutoMoT/lead 其它任何文件。
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
You are an autonomous driving assistant evaluating a single front-camera \
image from an ego vehicle.

Your task:
1. Examine the image carefully.
2. Read the current MEMORY (scenario context, status, and subgoal).
3. Determine whether the current driving STATUS has changed based on visual \
evidence in the image.
4. Identify the appropriate next SUBGOAL the ego vehicle should pursue.
5. Provide a brief ANALYSIS (2-4 sentences) explaining your reasoning.

Output format - respond EXACTLY as shown below, with no extra text before or \
after the block:

ANALYSIS: <2-4 sentence description of what you observe and how it maps to \
the scenario events>
STATUS: <event_name>
SUBGOAL: <event_name>

Rules:
- STATUS and SUBGOAL must each be a single event name from the scenario's \
event sequence (e.g., "slow_traffic_detect", "match_speed").
- STATUS may stay the same as the memory STATUS if the situation has not \
changed.
- SUBGOAL must be the event immediately after STATUS in the sequence, unless \
STATUS is already the last middle event, in which case SUBGOAL is "final".
- Do NOT invent event names outside the provided sequence.
- Be concise and precise."""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_memory_block(memory: DrivingMemory) -> str:
    seq_str = " -> ".join(memory.event_sequence)
    completed_str = ", ".join(memory.completed_events) if memory.completed_events else "none"
    status_desc = memory.status_description()
    subgoal_desc = memory.subgoal_description()

    return (
        "[MEMORY]\n"
        f"SCENARIO: {memory.scenario_label}\n"
        f"EVENT_SEQUENCE: {seq_str}\n"
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
        "Given the image above and the memory context, output your ANALYSIS, "
        "STATUS, and SUBGOAL."
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
    """一次 step 的完整审计记录。所有字符串与 parsed 字典都会序列化到 JSON。"""
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

    def to_dict(self) -> dict:
        return {
            "step_idx":         self.step_idx,
            "timestamp":        self.timestamp,
            "scenario":         self.scenario,
            "num_images":       self.num_images,
            "memory_before":    self.memory_before,
            "system_prompt":    self.system_prompt,
            "user_prompt":      self.user_prompt,
            "combined_prompt":  self.combined_prompt,
            "raw_vlm_text":     self.raw_vlm_text,
            "parsed":           self.parsed,
            "memory_after":     self.memory_after,
            "save_dir":         self.save_dir,
            "image_files":      self.image_files,
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

        tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
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
    ) -> Tuple[DrivingMemory, ParadigmAStepRecord]:
        """跑一次范式 A 推理 + 落盘。

        参数:
            memory:    当前 DrivingMemory。
            images:    list[PIL.Image.Image],按时间顺序排列的前视 RGB。
            step_idx:  该 step 在整段 trajectory 内的序号(只用于命名 / 索引)。
            image_description: 文本里给图像留的占位字符串(纯人读注释)。
            save_dir:  可选,覆盖 self.save_root 决定的默认 step 目录。

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
            self._dump_record(record, images, target_dir)

        return new_memory, record

    # ------------------------------------------------------------------
    # 文件落盘 helper
    # ------------------------------------------------------------------

    @staticmethod
    def _dump_record(record: ParadigmAStepRecord,
                     images: List[Any],
                     target_dir: pathlib.Path) -> None:
        """把 record 全量落盘,文本/JSON/图像分离存放,便于人工 review 与 diff。

        目录结构:
            target_dir/
                inputs/
                    system_prompt.txt
                    user_prompt.txt
                    combined_prompt.txt
                    memory_before.json
                    image_000.png
                    image_001.png
                    ...
                outputs/
                    raw_vlm_text.txt
                    parsed.json
                    memory_after.json
                step.json                   # 汇总索引,所有字段一份冗余 JSON
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

        # 图像输入(PIL 直接 save;非 PIL 则跳过并打印 warning)
        image_files: List[str] = []
        for i, img in enumerate(images):
            fname = f"image_{i:03d}.png"
            fpath = inputs_dir / fname
            try:
                if _HAS_PIL and hasattr(img, "save"):
                    img.save(str(fpath))
                    image_files.append(str(fpath.relative_to(target_dir)))
                else:
                    print(f"[vlm_paradigm_a_runner] image[{i}] is not PIL.Image, skip save")
            except Exception as e:
                print(f"[vlm_paradigm_a_runner] failed to save image[{i}]: {e}")
        record.image_files = image_files
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
# __main__:不挂模型的烟囱测试
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """不依赖 Qwen 权重,仅验证 prompt / parse / update / dump 全链路语义。"""
    print("=" * 60)
    print("vlm_paradigm_a_runner self-test (no real VLM)")
    print("=" * 60)

    memory = DrivingMemory.from_scenario("MergerIntoSlowTraffic")
    print(f"initial memory: {memory.to_dict()}\n")

    def dummy_vlm_fn(system: str, user: str, images: List[Any]) -> str:
        # 模拟一个完美遵循格式的 VLM 响应
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

    # 测试落盘(走 ParadigmARunner._dump_record,但不真的调模型)
    tmp_dir = pathlib.Path(__file__).parent / "_paradigm_a_self_test_out"
    record = ParadigmAStepRecord(
        step_idx=0,
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        scenario=memory.scenario,
        num_images=0,
        memory_before=memory.to_dict(),
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(memory),
        combined_prompt=build_combined_prompt(memory),
        raw_vlm_text=raw,
        parsed=parsed,
        memory_after=new_memory.to_dict(),
    )
    ParadigmARunner._dump_record(record, images=[], target_dir=tmp_dir / "step_000000")
    print(f"\ndumped self-test record to: {tmp_dir / 'step_000000'}")


if __name__ == "__main__":
    _self_test()
