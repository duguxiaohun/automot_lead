#!/usr/bin/env python3
"""Phase3 上下文分类：把 Phase1/Phase2 的 YES/NO 答案恢复为可观察的
ROAD_STRUCTURE 与异常 EVENT，再映射到 high-level 动作问题域。

上游完整结构四问 RS1/RS2/RS4/RS5 单 YES 恢复 R1/R2/R4/R5，完整全 NO
恢复 R3；subset 未问不能当 NO，hierarchical RS_HIGHWAY 可提供独立结构答案。
Phase1 HIGHWAY 是独立可见事实，不能代替 RS_HIGHWAY。

STATIC_OBSTACLE / VULNERABLE / TRAFFIC_LIGHT_ABNORMAL 对应 U-E2 / U-E4 / U-E7
的候选语义（U4 包含沿路骑行，U7 必须是真正灯故障，普通无灯路口不是 U7）。
Phase2 在 INVALID_EVENT_CONTEXT=NO 时提供 UE1/UE3/UE5/UE6，可与 Phase1 事件并存。
上游 all-NO 不能确定任何常规事件；R-E2/R-E3/R-E5 需要独立可见/导航 transition gate。
历史 raw taxonomy 的 U7 不等于 Phase1 TRAFFIC_LIGHT_ABNORMAL；生产数据走
source_mapping.py 读取人工答案表和显式高速 UE3 RGB 决定，不能直接照搬事件名。

context 的 RS 范围表示事件可与该道路空间共存，不是从道路结构自动推导异常。
五种动作不变；七个异常 + R-E2/R-E3/R-E5 共十个平衡 context。
POST_BYPASS_RETURN 是兼容 ID，覆盖通用 R-E2，具体已发生的绕障历史另行提供。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Phase1 / Phase2 答案 -> RS / EVENT
# ---------------------------------------------------------------------------

PHASE1_RS_KEYS: Tuple[str, ...] = ("RS1", "RS2", "RS4", "RS5")
PHASE1_EVENT_KEYS: Tuple[str, ...] = ("STATIC_OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL")
PHASE2_EVENT_KEYS: Tuple[str, ...] = ("UE1", "UE3", "UE5", "UE6")

PHASE1_RS_TO_RS: Dict[str, str] = {"RS1": "R1", "RS2": "R2", "RS4": "R4", "RS5": "R5"}
PHASE1_EVENT_TO_EVENT: Dict[str, str] = {
    "STATIC_OBSTACLE": "U-E2",
    "VULNERABLE": "U-E4",
    "TRAFFIC_LIGHT_ABNORMAL": "U-E7",
}
PHASE2_EVENT_TO_EVENT: Dict[str, str] = {
    "UE1": "U-E1",
    "UE3": "U-E3",
    "UE5": "U-E5",
    "UE6": "U-E6",
}
def road_structure_from_answers(answers: Mapping[str, bool]) -> str:
    """只使用实际问过的结构答案；可见事实 HIGHWAY 与结构 R3 独立。"""

    positives = [key for key in PHASE1_RS_KEYS if answers.get(key) is True]
    if answers.get("RS_HIGHWAY") is True:
        return "UNKNOWN" if positives else "R3"
    if len(positives) == 1:
        return PHASE1_RS_TO_RS[positives[0]]
    if not positives and all(answers.get(key) is False for key in PHASE1_RS_KEYS):
        if answers.get("RS_HIGHWAY") is False:
            return "UNKNOWN"
        return "R3"
    return "UNKNOWN"


def abnormal_events_from_answers(answers: Mapping[str, bool]) -> Tuple[str, ...]:
    """恢复所有被 Phase1/2 明确回答 YES 的异常事件标志。

    Phase1 的可见事实和 Phase2 的 direct-event 问题可能同时为 YES（特别是
    interrupted overlay）。因此这里返回集合而不是人为优先级选一个“主事件”。
    ``INVALID_EVENT_CONTEXT=YES`` 时 Phase2 的 UE 输出不可用；Phase1 的独立可见
    事实仍可保留。
    """

    events = []
    for key in PHASE1_EVENT_KEYS:
        if answers.get(key) is True:
            events.append(PHASE1_EVENT_TO_EVENT[key])
    if answers.get("INVALID_EVENT_CONTEXT") is False:
        for key in PHASE2_EVENT_KEYS:
            if answers.get(key) is True:
                events.append(PHASE2_EVENT_TO_EVENT[key])
    return tuple(events)


def event_from_answers(answers: Mapping[str, bool]) -> str:
    """返回唯一可恢复的异常 EVENT，否则返回 ``UNKNOWN``。

    这不是 regular-event 分类器：没有异常 YES 时，Phase1/2 并未给出 R-E1、
    R-E2、R-E3、R-E4 或 R-E5 的身份。多个 YES 也不能由本函数任意挑一个主事件。
    """

    events = abnormal_events_from_answers(answers)
    return events[0] if len(events) == 1 else "UNKNOWN"


# ---------------------------------------------------------------------------
# Phase3 动作问题域与上下文
# ---------------------------------------------------------------------------

DOMAIN_LONGITUDINAL = "LONGITUDINAL_YIELD"
DOMAIN_MANEUVER = "FULL_MANEUVER"
QUESTION_DOMAINS: Tuple[str, ...] = (DOMAIN_LONGITUDINAL, DOMAIN_MANEUVER)

ACTION_DECELERATE = "DECELERATE"
ACTION_STOP = "STOP"
ACTION_RESUME = "RESUME"
ACTION_LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
ACTION_LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
ACTION_KEYS: Tuple[str, ...] = (
    ACTION_DECELERATE,
    ACTION_STOP,
    ACTION_RESUME,
    ACTION_LANE_CHANGE_LEFT,
    ACTION_LANE_CHANGE_RIGHT,
)

# 纵向域故意不给横向选项：U-E1/U-E3/U-E5/U-E6/U-E7 的本任务高层动作只在
# 减速、停车等待、恢复三者之间，给出变道行会制造无法由轨迹验证的监督。
DOMAIN_ACTION_KEYS: Dict[str, Tuple[str, ...]] = {
    DOMAIN_LONGITUDINAL: (ACTION_DECELERATE, ACTION_STOP, ACTION_RESUME),
    DOMAIN_MANEUVER: ACTION_KEYS,
}


@dataclass(frozen=True)
class ActionContext:
    """一个 phase3 平衡类别；异常 UE 来自上游答案，R-E context 来自 transition gate。"""

    context_id: str
    question_domain: str
    source_event: str
    allowed_rs: Tuple[str, ...]
    road_structure_text: Dict[str, str]
    situation_text: str
    scope_text: str

    @property
    def action_keys(self) -> Tuple[str, ...]:
        """返回该上下文需要回答的动作行。"""

        return DOMAIN_ACTION_KEYS[self.question_domain]


ROAD_STRUCTURE_TEXT: Dict[str, str] = {
    "R1": (
        "an ordinary same-direction road where lane keeping, car following and safe-gap "
        "control are the governing rules"
    ),
    "R2": (
        "a road whose usable forward space is close to a single lane, so the opposing lane "
        "can take part in the decision"
    ),
    "R3": (
        "a limited-access highway, ramp, merge or exit corridor where speed matching, "
        "rear-side gaps and the target lane are the governing rules"
    ),
    "R4": "a local junction whose traffic-signal head is the governing rule",
    "R5": (
        "a local junction with no usable signal rule, so right of way, gap acceptance and "
        "crossing or oncoming flow govern the decision"
    ),
}


ACTION_CONTEXTS: Tuple[ActionContext, ...] = (
    ActionContext(
        context_id="LEAD_BRAKE",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="U-E1",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "the vehicle already in ego's forward path has braked hard or suddenly slowed enough "
            "to interrupt normal following"
        ),
        scope_text=(
            "Ego is following that vehicle in its own lane. Only longitudinal speed control is "
            "asked here; no lane change is available as an answer."
        ),
    ),
    ActionContext(
        context_id="STATIC_BLOCKAGE",
        question_domain=DOMAIN_MANEUVER,
        source_event="U-E2",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "a static blockage such as a crashed vehicle, construction barrier, parked or stalled "
            "vehicle or roadside hazard occupies ego's current path"
        ),
        scope_text=(
            "Ego must assess how much of its lane the blockage occupies and whether to slow, "
            "hold still and wait for a usable gap, or leave the blocked lane to the left or to the "
            "right. Leaving to the left may mean borrowing the opposing lane. Vehicles wholly "
            "inside a parking bay with an open ego corridor do not establish a blockage."
        ),
    ),
    ActionContext(
        context_id="DYNAMIC_CUTIN",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="U-E3",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "another vehicle is moving into, or is visibly about to occupy, ego's immediate "
            "forward corridor"
        ),
        scope_text=(
            "Answer longitudinal speed control for that intruding vehicle. Ego's lateral "
            "decision is not asked and is not implied by this question set."
        ),
    ),
    ActionContext(
        context_id="VULNERABLE_CROSSING",
        question_domain=DOMAIN_MANEUVER,
        source_event="U-E4",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "a pedestrian or cyclist is relevant to ego's immediate decision: crossing, entering "
            "the path, or travelling along its edge with insufficient passing clearance"
        ),
        scope_text=(
            "Distinguish a crossing user from a same-direction cyclist. Slow or wait while the "
            "path is unsafe; a deliberate lane change to pass with clearance requires a usable "
            "gap and an actual lane boundary crossing. A distant sidewalk user alone is insufficient."
        ),
    ),
    ActionContext(
        context_id="ONCOMING_INVASION",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="U-E5",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "an oncoming or opposite-direction vehicle is abnormally intruding into ego's usable "
            "corridor"
        ),
        scope_text=(
            "Ego is the passive side and must give the invading vehicle time and space by "
            "longitudinal control only. Check successive oncoming actors: one vehicle passing "
            "does not mean a second intruder has cleared. Cones alone do not establish intrusion."
        ),
    ),
    ActionContext(
        context_id="JUNCTION_RULE_CONFLICT",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="U-E6",
        allowed_rs=("R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "another vehicle is violating the junction rule and entering ego's conflict path while "
            "ego should have priority"
        ),
        scope_text=(
            "Ego is inside or entering that junction. Only longitudinal yielding, waiting for the "
            "conflict to clear and resuming are asked here."
        ),
    ),
    ActionContext(
        context_id="SIGNAL_FAILURE",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="U-E7",
        allowed_rs=("R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "installed traffic signals have an established malfunction, so ego cannot rely on a red or green "
            "phase and must watch every approach"
        ),
        scope_text=(
            "Ego must also respect applicable stop/yield priority. An ordinary unsignalized "
            "junction, a red light, or unreadable lamps in fog do not establish a signal failure. "
            "One visible green lamp also cannot by itself disprove an established system fault."
        ),
    ),
    ActionContext(
        context_id="POST_BYPASS_RETURN",
        question_domain=DOMAIN_MANEUVER,
        source_event="R-E2",
        allowed_rs=("R1", "R2", "R3", "R4", "R5"),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "a route-lane transition is pending: ego may need to recover its lane after a bypass "
            "or move into a navigation-required target lane"
        ),
        scope_text=(
            "The question is whether the next high-level step is to move back toward ego's original "
            "or route-target lane, and on which side that lane now lies. Answer both lane-change "
            "lines NO when ego must still stay in the current lane. Both NO answers only describe "
            "the next three seconds; they do not prove the recovery has completed."
        ),
    ),
    ActionContext(
        context_id="UNSIGNALIZED_PRIORITY",
        question_domain=DOMAIN_LONGITUDINAL,
        source_event="R-E5",
        allowed_rs=("R5",),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "ego is negotiating a normal unsignalized junction using stop/yield rules and "
            "the priority of crossing or oncoming traffic"
        ),
        scope_text=(
            "Respect any visible STOP requirement even without other traffic. A vehicle with "
            "priority is not a rule violator; absence of traffic lights is not a signal failure. "
            "The junction must govern the current local decision; a distant intersection ahead "
            "of an otherwise continuous car-following corridor is insufficient."
        ),
    ),
    ActionContext(
        context_id="RAMP_MERGE_EXIT",
        question_domain=DOMAIN_MANEUVER,
        source_event="R-E3",
        allowed_rs=("R3",),
        road_structure_text=ROAD_STRUCTURE_TEXT,
        situation_text=(
            "ego is in an active ramp, merge, lane-join or exit transition toward its route target"
        ),
        scope_text=(
            "The question is whether the next high-level step is a lane change into the target "
            "lane, and on which side. Answer both lane-change lines NO when ego only has to keep "
            "the current lane and match speed along an actual ramp transition. Stable main-line "
            "following or parallel traffic alone is not an active merge. An ordinary navigation "
            "lane change on a continuous main line belongs to the route-lane-transition question, "
            "not automatically to this ramp/merge/exit situation."
        ),
    ),
)

CONTEXT_BY_ID: Dict[str, ActionContext] = {ctx.context_id: ctx for ctx in ACTION_CONTEXTS}
CONTEXT_IDS: Tuple[str, ...] = tuple(ctx.context_id for ctx in ACTION_CONTEXTS)
EVENT_TO_CONTEXT: Dict[str, str] = {ctx.source_event: ctx.context_id for ctx in ACTION_CONTEXTS}


def context_for_event(event: str) -> Optional[ActionContext]:
    """把 EVENT code 映射到 phase3 上下文；U-E8 等未覆盖事件返回 None。"""

    context_id = EVENT_TO_CONTEXT.get(str(event))
    return CONTEXT_BY_ID.get(context_id) if context_id else None


def context_from_answers(answers: Mapping[str, bool]) -> Optional[ActionContext]:
    """由单一异常 YES 直接推出 phase3 上下文。

    这是七个异常 UE 的便捷入口；regular、multiple-event、R-E2 与 R-E3 都必须由
    外层状态机/transition gate 显式处理，不能从这里臆测。
    """

    return context_for_event(event_from_answers(answers))


# 旧 API 常量只为读取历史产物保留；新映射不使用优先级/固定超时。
EVENT_PRECEDENCE: Tuple[str, ...] = ("U-E2", "U-E4", "U-E7", "U-E1", "U-E3", "U-E5", "U-E6")
POST_BYPASS_MAX_GAP_FRAMES = 24


def resolve_context_id(
    road_structure: str,
    event_codes: Sequence[str],
    frames_since_bypass: Optional[int] = None,
) -> Optional[str]:
    """单上下文兼容接口；多个事件时返回 None，调用者应使用 resolve_context_ids。

    frames_since_bypass 不再截断 R-E2，历史只作为提示词的额外事实。
    """

    contexts = resolve_context_ids(road_structure, event_codes)
    return contexts[0] if len(contexts) == 1 else None


def resolve_context_ids(
    road_structure: str, event_codes: Sequence[str], *,
    signal_failure_confirmed: Optional[bool] = None,
) -> Tuple[str, ...]:
    """离线候选映射，保留并发事件；普通无灯 U7 历史标签不能冒充灯故障。

    R-E2/R-E3/R-E5 必须来自显式标注/transition gate，不从 all-NO 推导。
    allowed_rs 只定义可容纳事件的几何空间，不从 RS 自动产生任何异常正例。
    """
    rs, events = str(road_structure), set(event_codes)
    if "U-E7" in events and signal_failure_confirmed is not True:
        if signal_failure_confirmed is False or rs != "R4":
            events.discard("U-E7")
    abnormal = [ctx.context_id for ctx in ACTION_CONTEXTS
                if ctx.source_event.startswith("U-") and ctx.source_event in events
                and rs in ctx.allowed_rs]
    if abnormal:
        return tuple(abnormal)
    if "U-E8" in events:
        return ()  # 七问没有该异常的可观测输出，不能伪装为普通 R-E5。
    return tuple(ctx.context_id for ctx in ACTION_CONTEXTS
                 if ctx.source_event.startswith("R-") and ctx.source_event in events
                 and rs in ctx.allowed_rs)
