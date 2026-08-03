"""SFT baseline 的高速/非高速 + RE/UE 标签协议。

本文件保留 sft_baseline 的 RS/EVENT 解析工具作为数据来源，但当前路线最终只监督
两个二分类值：

- `ROAD`: `HIGHWAY` / `NON_HIGHWAY`
- `EVENT`: `RE` / `UE`
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DATASET_VERSION = "sft_baseline_highway_reue_joint_v1"
# DATASET_VERSION 描述的是本路线自己的样本协议：两问直接输出语义 token，不含
# OPSD/CoT/teacher，也不再有任何 A/B/C 选项字母。数据 schema 里的 `rs_option` /
# `event_option_map` 两个字母字段已被删除，Q2 候选改为有序 list
# `event_candidates_ordered`；regular EVENT 不再折叠成单个 RE。当前标签以 RS
# 为准做 canonical regular 映射：路口通行类 R-E4/R-E5 由 R4/R5 决定，非路口
# 越界 regular 映射回该 RS 默认 regular。旧 schema / 旧未映射 index 直接作废
# 重建，不保留兼容读取。
#
# EVENT 候选顺序故意继续使用 v5 的 namespace。这样同一 route/frame/seed
# 在 sft_baseline 与 sft_v5 中候选集合和展示顺序一致，便于做逐样本直接对照；
# 同时 adapter config 仍记录 DATASET_VERSION，避免把两条训练路线混淆。
CHOICE_ORDER_DATASET_VERSION = "sft_v5_rs_event_sequence"


# ---------------------------------------------------------------------------
# RS: Q1 中给学生看的固定语义 token。内部仍保留 R1-R5，便于和 keyframe_filter 对齐。
# ---------------------------------------------------------------------------

RS_LABELS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
ROAD_LABELS: Tuple[str, ...] = ("HIGHWAY", "NON_HIGHWAY")
EVENT_FAMILY_LABELS: Tuple[str, ...] = ("RE", "UE")

RS_LABEL_TO_TOKEN: Dict[str, str] = {
    "R1": "ORDINARY_ROAD",
    "R2": "BIDIRECTIONAL_NARROW",
    "R3": "HIGHWAY_MERGE_EXIT",
    "R4": "SIGNAL_INTERSECTION",
    "R5": "PRIORITY_INTERSECTION",
}
RS_TOKEN_TO_LABEL: Dict[str, str] = {v: k for k, v in RS_LABEL_TO_TOKEN.items()}

RS_DESCRIPTIONS: Dict[str, str] = {
    # 直接按内部 R1-R5 索引。旧版本这里按 A-E 索引，需要先 label -> letter 再取描述；
    # token 协议下字母已经没有任何作用，多一层映射只会制造对不齐的机会。
    "R1": (
        "Ordinary same-direction road: continuous lanes in the same travel direction, "
        "with parallel lane markings or road edges extending ahead and no nearby junction "
        "control, branch connector, ramp, or opposing-lane-sharing constraint."
    ),
    "R2": (
        "Bidirectional narrow or opposing-lane-sharing road: a tight corridor where the "
        "oncoming lane is part of the usable space, with little separation between ego "
        "and opposite-direction traffic."
    ),
    "R3": (
        "Highway, ramp, lane-join, split, or exit structure: a high-speed road geometry with "
        "ramps, acceleration/deceleration lanes, gore areas, lane splits, lane joins, "
        "or entry/exit connectors."
    ),
    "R4": (
        "Signalized intersection: an intersection approach or interior where visible, "
        "working traffic lights define the right-of-way structure."
    ),
    "R5": (
        "Unsignalized or priority-controlled intersection: an intersection approach or "
        "interior without a working traffic-light rule, using stop/yield signs, priority "
        "layout, or road geometry for right-of-way."
    ),
}

ROAD_DESCRIPTIONS: Dict[str, str] = {
    "HIGHWAY": (
        "High-speed road context: highway mainline, ramp, acceleration/deceleration lane, "
        "merge, split, exit, connector, or lane-join geometry."
    ),
    "NON_HIGHWAY": (
        "Non-highway context: ordinary urban or suburban road, rural/unstructured local road, "
        "narrow bidirectional road, signalized intersection, unsignalized intersection, "
        "or priority-controlled junction."
    ),
}


def road_label_from_rs(rs_label: str) -> str:
    """把内部 R1-R5 道路结构折叠成高速/非高速二分类。"""

    return "HIGHWAY" if str(rs_label) == "R3" else "NON_HIGHWAY"


def event_family_from_label(event_label: str) -> str:
    """把 R-E*/U-E* 折叠成 RE/UE 二分类。"""

    return "UE" if is_unusual(str(event_label)) else "RE"


# ---------------------------------------------------------------------------
# EVENT: Q2 候选由当前 RS 的静态全集决定；allowed_events 只用于解析/审计 GT。
# ---------------------------------------------------------------------------

EVENT_CANDIDATES_BY_RS: Dict[str, List[str]] = {
    # 这里直接保留原始 R-E*/U-E* 候选表。旧协议会把所有 R-E* 折成一个 RE；
    # 新协议把 regular 也展开成可监督 token，避免 R3 这类“纯 regular RS”退化成
    # 单候选送分题。
    # UE 静态表按 2026-07 全量共现审计的严格口径维护：过滤异常/缺失 route 后，
    # 仅保留 count >= 20 且占该 RS 帧数 rate >= 0.1% 的 RS x UE 组合。低频/零频
    # 组合交给 dataset_candidate_mismatch 剔除，不作为所有该 RS 帧的永久干扰项。
    "R1": ["R-E1", "R-E2", "U-E1", "U-E2", "U-E3", "U-E4", "U-E5"],
    "R2": ["R-E1", "R-E2", "U-E2", "U-E4", "U-E5"],
    "R3": ["R-E1", "R-E2", "R-E3"],
    "R4": ["R-E4", "U-E4", "U-E6", "U-E7", "U-E8"],
    "R5": ["R-E5", "U-E4", "U-E5", "U-E7", "U-E8"],
}

RS_REGULAR_EVENTS: Dict[str, List[str]] = {
    # 每个 RS 下有哪些 regular 行为可被解释成 RE。R3 特意保留 3 个 regular，
    # 因为 highway/ramp/merge/exit 结构下，“无异常”不只是普通跟车，也可能是
    # 正常汇入、分流、驶离或目标车道跟踪。
    "R1": ["R-E1", "R-E2"],
    "R2": ["R-E1", "R-E2"],
    "R3": ["R-E1", "R-E2", "R-E3"],
    "R4": ["R-E4"],
    "R5": ["R-E5"],
}

REGULAR_EVENT_DESCRIPTIONS: Dict[str, str] = {
    "R-E1": (
        "Regular lane following: the ego vehicle stays stably within its current lane or path, "
        "with no visible lane-line crossing, ramp/exit branch selection, or short-horizon conflict."
    ),
    "R-E2": (
        "Regular lane change or recovery: the latest frames show lateral movement across lane "
        "markings or between adjacent lanes, returning to a lane, or completing a normal path "
        "adjustment on clear drivable space."
    ),
    "R-E3": (
        "Regular highway/ramp maneuver: the ego vehicle is actively taking a highway branch "
        "action, such as entering a connector, leaving the mainline through an exit or "
        "deceleration lane, joining the mainline from an acceleration lane, or choosing a split "
        "branch, without an unusual blocking event."
    ),
    "R-E4": (
        "Regular traffic-light compliance: at a working signalized intersection, the ego vehicle "
        "waits, proceeds, or turns according to the visible signal control without an unusual "
        "road-user conflict."
    ),
    "R-E5": (
        "Regular priority negotiation: at an unsignalized or priority-controlled intersection, "
        "the ego vehicle follows stop/yield, priority layout, or safe-gap reasoning without an "
        "unusual road-user conflict."
    ),
}

UE_DESCRIPTIONS: Dict[str, str] = {
    "U-E1": (
        "A lead vehicle suddenly brakes or decelerates, requiring the ego vehicle to react "
        "with reduced speed or increased following distance."
    ),
    "U-E2": (
        "A static obstacle, accident, construction object, parked vehicle, open door, or "
        "blocked lane occupies the ego path and forces avoidance, stopping, or borrowing space."
    ),
    "U-E3": (
        "A moving vehicle cuts in, pulls out, or dynamically occupies the ego path, creating "
        "a short-horizon conflict."
    ),
    "U-E4": (
        "A pedestrian, cyclist, or small vulnerable road user crosses or laterally enters "
        "the ego vehicle's intended path."
    ),
    "U-E5": (
        "An oncoming vehicle abnormally invades the ego lane or priority space, forcing the "
        "ego vehicle to yield or adjust."
    ),
    "U-E6": (
        "Another vehicle violates the expected intersection rule, such as running a red light "
        "or crossing against the ego vehicle's priority, creating conflict."
    ),
    "U-E7": (
        "The intersection rule source is unreliable or failed, such as defective traffic "
        "lights or ambiguous priority, so normal right-of-way reasoning is broken."
    ),
    "U-E8": (
        "The forward road or intersection space is temporarily blocked or reopening, requiring "
        "waiting, queue handling, or cautious release."
    ),
}

EVENT_DESCRIPTIONS: Dict[str, str] = {
    **REGULAR_EVENT_DESCRIPTIONS,
    **UE_DESCRIPTIONS,
}

EVENT_LABEL_TO_TOKEN: Dict[str, str] = {
    "R-E1": "LANE_FOLLOWING",
    "R-E2": "LANE_CHANGE",
    "R-E3": "HIGHWAY_MANEUVER",
    "R-E4": "SIGNAL_COMPLIANCE",
    "R-E5": "PRIORITY_NEGOTIATION",
    "U-E1": "LEAD_BRAKE",
    "U-E2": "STATIC_OBSTACLE",
    "U-E3": "MOVING_CUT_IN",
    "U-E4": "VULNERABLE_CROSSING",
    "U-E5": "ONCOMING_INVASION",
    "U-E6": "RULE_VIOLATION",
    "U-E7": "RULE_UNCERTAIN",
    "U-E8": "BLOCKED_SPACE",
}
EVENT_TOKEN_TO_LABEL: Dict[str, str] = {v: k for k, v in EVENT_LABEL_TO_TOKEN.items()}
EVENT_LABELS: Tuple[str, ...] = tuple(EVENT_LABEL_TO_TOKEN.keys())
REGULAR_EVENT_LABELS: Tuple[str, ...] = ("R-E1", "R-E2", "R-E3", "R-E4", "R-E5")

# 全量映射后数据上的 GT-RS oracle EVENT 参照：假设已知正确 RS，永远回答该
# RS 下最高频 regular 子类；UE 帧自然计 0。eval 另报真正端到端零信息下界：
# 永远回答全局最高频 regular token（当前为 R-E1 / LANE_FOLLOWING）。
REGULAR_MAJORITY_EVENT_BY_RS: Dict[str, str] = {
    "R1": "R-E1",
    "R2": "R-E1",
    "R3": "R-E1",
    "R4": "R-E4",
    "R5": "R-E5",
}
REGULAR_ZERO_INFO_BASELINE_BY_RS: Dict[str, float] = {
    "R1": 0.805,
    "R2": 0.535,
    "R3": 0.589,
    "R4": 0.901,
    "R5": 0.813,
}
REGULAR_ZERO_INFO_BASELINE_END_TO_END = 0.7685

EVENT_ORDER: Tuple[str, ...] = (
    # 多标签没有置信度或 primary 不可用时，用这个全局顺序做确定性兜底。
    # 这样同一份数据在不同机器/不同 Python hash seed 下不会得到不同 teacher target。
    "R-E1", "R-E2", "R-E3", "R-E4", "R-E5",
    "U-E1", "U-E2", "U-E3", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8",
)


@dataclass(frozen=True)
class RSTarget:
    """单帧 RS 训练目标。

    `label` 是内部 R1-R5，学生实际输出的是 `RS_LABEL_TO_TOKEN[label]`。
    `candidates` 保存原始打分，方便后续 probe 回查“为什么双标签最后选了哪个”。
    """

    label: str
    description: str
    confidence: float
    secondary: Tuple[str, ...]
    candidates: Dict[str, float]


@dataclass(frozen=True)
class EventTarget:
    """单帧 EVENT 训练目标。

    `label` 是监督用标签：R-E* 或 U-E*。UE 保留原始异常标签；regular 会按
    当前 RS 映射到 canonical R-E 标签。`event_code` 保留映射前选中的原始 code，
    便于重建数据后继续审计 RS / regular 原标注是否互相矛盾。
    """

    label: str
    event_code: str
    abnormal: bool
    raw_events: Tuple[str, ...]
    regular_event_codes: Tuple[str, ...] = ()


def normalize_event_code(code: Any) -> Optional[str]:
    """把各种来源里的 EVENT 字符串规范化成 `R-E*` / `U-E*`。

    collection_output 中通常已经是规范 code；这里多做一层大小写/空白容错，让测试和
    旧中间产物不至于因为格式小差异直接崩。
    """

    if code is None:
        return None
    text = str(code).strip().upper().replace("_", "-")
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[RU]-E[1-8]", text):
        return text
    if text == "RE":
        return "RE"
    return None


def is_unusual(code: str) -> bool:
    """判断某个 EVENT code 是否是 UE。"""

    return str(code).startswith("U-E")


def is_regular_event(code: str) -> bool:
    """判断某个 EVENT code 是否是 regular R-E。"""

    return str(code).startswith("R-E")


def default_regular_event_for_rs(rs_label: str) -> str:
    """返回某个 RS 下 memory/非法兜底用的默认 regular EVENT。"""

    values = RS_REGULAR_EVENTS.get(str(rs_label), [])
    return values[0] if values else "R-E1"


def canonical_regular_event_for_rs(rs_label: str, event_code: str) -> str:
    """按 RS 把原始 regular EVENT 映射成训练/评估使用的 canonical 标签。

    远端全量归因显示，R-E4/R-E5 在路口类型上不稳定：R5 里大量出现
    SIGNAL_COMPLIANCE，且分散在 NonSignalizedJunctionRightTurn、
    PriorityAtJunction 等明确无信号/优先权 scenario。这里采用“RS 更可信”的口径：
    - R4 下任意 regular 都映射为 R-E4；
    - R5 下任意 regular 都映射为 R-E5；
    - R1/R2/R3 只保留各自静态表内的道路行为 regular，越界项映射为默认 R-E1。

    映射只改变监督标签；原始 code 会继续写入 event_code / regular_event_codes 审计字段。
    """

    code = normalize_event_code(event_code)
    rs = str(rs_label)
    if not code or not is_regular_event(code):
        return default_regular_event_for_rs(rs)
    allowed = RS_REGULAR_EVENTS.get(rs, [])
    if code in allowed:
        return code
    return default_regular_event_for_rs(rs)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """轻量数值转换，用在候选分数/置信度解析。"""

    try:
        return float(value)
    except Exception:
        return default


def resolve_rs_target(frame: Mapping[str, Any]) -> RSTarget:
    """从 collection_output 的单帧 annotation 中解析 RS 单标签目标。

    RS 的双标签不按 student 输出动态接受，直接取最高置信。优先使用
    `road_structure_candidates` 的最高分；缺分数时保持 primary label。
    """

    primary = (
        (frame.get("frame_rs_annotation") or {}).get("label")
        or frame.get("primary_road_structure")
        or frame.get("rs_label")
    )
    candidates_raw = frame.get("road_structure_candidates") or frame.get("rs_candidates") or {}
    candidates: Dict[str, float] = {}
    if isinstance(candidates_raw, Mapping):
        for key, value in candidates_raw.items():
            if str(key) in RS_LABELS:
                candidates[str(key)] = _safe_float(value)

    label = str(primary) if primary in RS_LABELS else "R1"
    if candidates:
        # RS 多标签只训练一个标签：优先最高置信度；置信度相同则按 R1-R5 稳定顺序。
        # 这里不采用 student 动态答案，是为了避免 Q1 的路结构监督变成多目标漂移。
        best_label, _ = max(
            candidates.items(),
            key=lambda item: (item[1], -RS_LABELS.index(item[0]) if item[0] in RS_LABELS else -99),
        )
        label = best_label

    rs_ann = frame.get("frame_rs_annotation") or {}
    secondary = tuple(str(x) for x in (rs_ann.get("secondary") or frame.get("secondary_road_structures") or []) if x)
    return RSTarget(
        label=label,
        description=RS_DESCRIPTIONS[label],
        confidence=_safe_float(rs_ann.get("confidence", frame.get("confidence", 0.0))),
        secondary=secondary,
        candidates=candidates,
    )


def raw_events_from_frame(frame: Mapping[str, Any]) -> Tuple[str, ...]:
    """按权威优先级读取原始 EVENT 列表。"""

    event_ann = frame.get("frame_event_annotation") or {}
    raw = event_ann.get("events") or frame.get("events") or frame.get("event_labels_raw") or None
    if not raw:
        primary = event_ann.get("label") or frame.get("primary_event") or frame.get("event_code")
        raw = [primary] if primary else []
    out: List[str] = []
    for item in raw:
        code = normalize_event_code(item)
        if code and code != "RE" and code not in out:
            out.append(code)
    return tuple(out)


def _stable_first(codes: Iterable[str]) -> str:
    """按全局 EVENT_ORDER 取稳定第一项。"""

    items = set(codes)
    for code in EVENT_ORDER:
        if code in items:
            return code
    return sorted(items)[0] if items else "R-E1"


def resolve_event_target(
    frame: Mapping[str, Any],
    *,
    student_event: Optional[str] = None,
    rs_label: Optional[str] = None,
) -> EventTarget:
    """把可能多标签的原始 EVENT 解析成单标签目标。

    规则对应 plan §4.3：
    - 有 UE 时 UE 优先；
    - 多个 UE 时，如果 student 输出是其中之一，teacher target 采用 student 输出；
      否则用 primary_event / 稳定顺序；
    - 全 regular 时先确定原始 R-E*，再按 RS 映射到 canonical regular 标签。
    """

    raw_events = raw_events_from_frame(frame)
    rs_for_mapping = str(rs_label) if rs_label in RS_LABELS else resolve_rs_target(frame).label
    student = normalize_event_code(student_event)
    ue = [code for code in raw_events if is_unusual(code)]
    regular = [code for code in raw_events if code.startswith("R-E")]
    if ue:
        # EVENT 多标签里只要出现 UE，就按“异常优先”训练，避免异常帧被 RE 稀释。
        # 如果 student 已经选择了 raw UE 之一，teacher 也接受这个选择并围绕它解释；
        # 否则才退回 primary/稳定顺序。这就是用户要求的“单标签但兼容双 UE”的口径。
        if student in ue:
            chosen = str(student)
        else:
            primary = normalize_event_code((frame.get("frame_event_annotation") or {}).get("label") or frame.get("primary_event"))
            chosen = str(primary) if primary in ue else _stable_first(ue)
        return EventTarget(
            label=chosen,
            event_code=chosen,
            abnormal=True,
            raw_events=tuple(raw_events),
            regular_event_codes=tuple(regular),
        )

    if student in regular:
        # 全 regular 的多标签帧允许 student 选 raw regular 之一；否则按
        # primary/稳定顺序确定一个训练标签，保证直接监督仍是单标签。
        chosen_re = str(student)
    else:
        primary = normalize_event_code((frame.get("frame_event_annotation") or {}).get("label") or frame.get("primary_event"))
        chosen_re = str(primary) if primary in regular else _stable_first(regular or ["R-E1"])
    regular_codes = tuple(regular or [chosen_re])
    mapped_re = canonical_regular_event_for_rs(rs_for_mapping, chosen_re)
    return EventTarget(label=mapped_re, event_code=chosen_re, abnormal=False, raw_events=tuple(raw_events), regular_event_codes=regular_codes)


def scenario_event_candidates_from_result(result: Mapping[str, Any]) -> List[str]:
    """读取单场景 result 顶层 event_candidates。

    如果旧文件缺少该字段，就返回全量 EVENT_ORDER 作为审计兜底。sft_baseline 当前
    Q2 出题不再用 scenario 级候选缩窄，只按当前 RS 的静态候选全集生成选项。
    """

    raw = result.get("event_candidates") or []
    out: List[str] = []
    for item in raw:
        code = normalize_event_code(item)
        if code and code != "RE" and code not in out:
            out.append(code)
    return out or list(EVENT_ORDER)


def q2_raw_candidates(scenario_candidates: Sequence[str], rs_label: str) -> List[str]:
    """按当前 RS 返回 Q2 原始候选全集。

    `scenario_candidates` 只保留在签名里兼容旧调用；当前协议下 Q2 选项不再由帧级
    或 scenario 级候选缩窄，避免候选长度直接泄漏“这帧有没有异常”。
    """

    del scenario_candidates
    return list(EVENT_CANDIDATES_BY_RS.get(rs_label, []))


def allowed_events_from_frame(frame: Mapping[str, Any]) -> List[str]:
    """读取 collector 写入的逐帧 EVENT 候选池。

    新版 `collection_output` 会把最终 candidate clamp / interrupted overlay 后的候选
    写入 `frame_event_annotation.allowed_events`，旧版可能只在 `event_evidence`
    里有同名字段。这里按 plan 约定的优先级读取，不做额外 RS/scenario 覆盖。
    """

    sources = [
        (frame.get("frame_event_annotation") or {}).get("allowed_events"),
        (frame.get("event_evidence") or {}).get("allowed_events"),
        frame.get("frame_allowed_events_raw"),
    ]
    out: List[str] = []
    for raw in sources:
        if not raw:
            continue
        for item in raw:
            code = normalize_event_code(item)
            if code and code != "RE" and code not in out:
                out.append(code)
        if out:
            return out
    return []


def q2_raw_candidates_for_frame(
    frame: Mapping[str, Any],
    *,
    scenario_candidates: Sequence[str],
    rs_label: str,
) -> List[str]:
    """得到 Q2 原始候选。

    sft_baseline 当前协议使用 RS 条件全集出题：allowed_events 只保留为 GT 解析和审计
    字段，不参与候选构造。
    """

    del frame
    return q2_raw_candidates(scenario_candidates, rs_label)


def collapse_regular_to_re(candidates: Sequence[str], rs_label: str) -> List[str]:
    """返回 Q2 展示候选。

    函数名保留旧调用兼容；当前协议已经不再 collapse regular。输出会保留
    R-E1..R-E5 具体标签，并只做去重与空候选兜底。
    """

    out: List[str] = []
    for code in candidates:
        norm = normalize_event_code(code)
        if norm and norm != "RE" and norm not in out:
            out.append(norm)
    if not out:
        out.append(default_regular_event_for_rs(rs_label))
    return out


def event_description_for_display(
    label: str,
    rs_label: str,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """返回 Q2 选项展示文本。"""

    del rs_label, regular_event_codes
    if is_regular_event(label):
        return REGULAR_EVENT_DESCRIPTIONS.get(label, label)
    return UE_DESCRIPTIONS.get(label, label)


def stable_event_choice_order(
    *,
    run_id: str,
    frame_id: int,
    rs_label: str,
    scenario_candidates: Sequence[str],
    raw_candidates: Optional[Sequence[str]] = None,
    seed: int = 0,
    dataset_version: str = CHOICE_ORDER_DATASET_VERSION,
) -> List[str]:
    """为某一帧生成可复现随机的 Q2 候选展示顺序。

    同一个 dataset_version + run_id + frame_id + seed 永远得到同样顺序；不同帧会
    打乱，避免模型学到“列在第一行的总是 REGULAR”这种位置捷径。默认 dataset_version
    使用 CHOICE_ORDER_DATASET_VERSION，而不是本路线 DATASET_VERSION，是为了让
    base/v5 在候选顺序扰动上完全同相位。

    返回值是有序 list，不是字母 -> 标签的 dict：token 协议下学生直接输出语义 token，
    候选字母没有任何作用，只有“第几个”这个展示顺序还需要保留。旧版返回的
    `{"A": ..., "B": ...}` 按 index 分配字母，因此 `sorted(map)` 的遍历顺序就等于
    这里的 list 顺序，两者渲染出的 prompt 完全一致。
    """

    raw = list(raw_candidates) if raw_candidates is not None else q2_raw_candidates(scenario_candidates, rs_label)
    display = collapse_regular_to_re(raw, rs_label)
    # 随机只打乱展示顺序，不改变本帧候选集合；seed 源里包含 dataset_version，
    # 以后如果候选协议变化，可以自然得到一套新顺序，避免旧缓存混用。
    seed_src = f"{dataset_version}::{run_id}::{frame_id}::{seed}".encode("utf-8")
    rng_seed = int(hashlib.sha256(seed_src).hexdigest(), 16) % (2**31)
    items = list(display)
    random.Random(rng_seed).shuffle(items)
    return items


def event_in_candidates(label: Optional[str], candidates: Sequence[str]) -> bool:
    """判断某个 EVENT label 是否出现在本帧候选里。

    替代旧的 `option_for_event`：那时需要反查字母才能判断“在不在选项里”，现在
    候选本身就是标签 list，成员判断即可。
    """

    return bool(label) and str(label) in {str(item) for item in candidates}


def weather_to_text(weather: Mapping[str, Any] | None) -> str:
    """把 XML weather 数值压成 teacher 用的短英文描述。

    Student 不直接看到这段文字；它只进入 teacher privileged prompt 和数据审计字段。
    """

    if not weather:
        return "weather is not available from XML"
    cloud = _safe_float(weather.get("cloudiness"), 0.0)
    rain = _safe_float(weather.get("precipitation"), 0.0)
    wet = _safe_float(weather.get("wetness"), 0.0)
    fog = _safe_float(weather.get("fog_density"), 0.0)
    sun = _safe_float(weather.get("sun_altitude_angle"), 45.0)
    parts: List[str] = []
    if sun <= 0:
        parts.append("night or very low-sun lighting")
    elif sun < 15:
        parts.append("low-sun lighting")
    else:
        parts.append("daytime lighting")
    if cloud >= 80:
        parts.append("heavy cloud cover")
    elif cloud >= 30:
        parts.append("moderate cloudiness")
    else:
        parts.append("clear or lightly cloudy sky")
    if rain >= 60:
        parts.append("heavy precipitation")
    elif rain >= 20:
        parts.append("rain")
    else:
        parts.append("no active rain")
    if wet >= 40:
        parts.append("wet road surface")
    else:
        parts.append("mostly dry road surface")
    if fog >= 25:
        parts.append("reduced visibility from fog")
    elif fog > 0:
        parts.append("light fog")
    return ", ".join(parts)


