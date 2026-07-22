"""SFT v5 的 ROAD_STRUCTURE / EVENT 标签协议。

本文件只做纯 Python 标签和候选集逻辑，不读图片、不加载模型。这样 build_dataset、
train、eval、probe 和测试可以共用同一份“什么能出现在 Q2 选项里”的规则，避免
后续出现训练和评估各写一套隐式候选表。

调用约定：构建数据使用 ``resolve_rs_target`` / ``resolve_event_target`` 和
``stable_event_option_map``；训练、eval、probe 只消费这些函数返回的 ``RSTarget`` /
``EventTarget``，不要在入口脚本里重新实现双标签优先级或候选过滤。

标签一共分三层：原始 collection code（R-E*/U-E*）用于审计，canonical 训练标签
（RE/U-E*）用于 memory 与 loss，本帧随机字母（A-Z）只用于 prompt。三层不能混写：
尤其不能把某一帧的字母存进下一帧 memory，也不能丢掉 regular 原始 code 后再猜 RE 文案。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DATASET_VERSION = "sft_v5_rs_event_sequence"


# ---------------------------------------------------------------------------
# RS: Q1 中给学生看的 A-E 选择题。内部仍保留 R1-R5，便于和 keyframe_filter 对齐。
# ---------------------------------------------------------------------------

RS_OPTION_TO_LABEL: Dict[str, str] = {
    "A": "R1",
    "B": "R2",
    "C": "R3",
    "D": "R4",
    "E": "R5",
}
RS_LABEL_TO_OPTION: Dict[str, str] = {v: k for k, v in RS_OPTION_TO_LABEL.items()}

RS_OPTION_DESCRIPTIONS: Dict[str, str] = {
    "A": (
        "Ordinary same-direction drivable road: the ego vehicle is mainly following, "
        "lane-keeping, making same-direction lane adjustments, or recovering on a normal "
        "drivable path; there is no dominant intersection rule, traffic-light control, "
        "highway merge/exit structure, or opposing-lane borrowing requirement."
    ),
    "B": (
        "Bidirectional single-lane or opposing-lane-sharing road: the usable corridor is "
        "narrow enough that the oncoming lane affects the decision, including borrowing "
        "the opposing lane to pass a blockage or yielding because an oncoming vehicle "
        "invades the ego lane."
    ),
    "C": (
        "Highway, ramp, merge, split, or exit structure: the ego vehicle is in a "
        "high-speed or ramp-like decision space where speed matching, gap selection, "
        "target-lane tracking, merging, diverging, or exiting dominates the driving rule."
    ),
    "D": (
        "Signalized intersection: the ego vehicle is inside or approaching an intersection "
        "where working traffic lights are the main right-of-way rule, including red-light "
        "waiting, green-light crossing, and protected or permissive turning under signal control."
    ),
    "E": (
        "Unsignalized or priority-controlled intersection: the ego vehicle is inside or "
        "approaching an intersection without a reliable traffic-light rule, so it must use "
        "stop/yield signs, priority, road geometry, cross traffic, pedestrians, or safe-gap "
        "reasoning to proceed."
    ),
}


# ---------------------------------------------------------------------------
# EVENT: Q2 首选逐帧 allowed_events；旧数据缺字段时才退回 scenario ∩ 当前 RS。
# ---------------------------------------------------------------------------

EVENT_CANDIDATES_BY_RS: Dict[str, List[str]] = {
    # 这里保留原始 R-E*/U-E* 候选表，而不是直接写 prompt 里的 RE。
    # 原因是 build_dataset 还需要保存 event_code / regular_event_codes 供审计；
    # 真正给学生看的选项会在 collapse_regular_to_re 里把所有 R-E* 折成一个 RE。
    "R1": ["R-E1", "R-E2", "U-E1", "U-E2", "U-E3", "U-E4"],
    "R2": ["R-E1", "R-E2", "U-E2", "U-E5"],
    "R3": ["R-E1", "R-E2", "R-E3"],
    "R4": ["R-E4", "U-E4", "U-E6", "U-E7", "U-E8"],
    "R5": ["R-E5", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8"],
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

RE_DESCRIPTIONS_BY_RS: Dict[str, str] = {
    "R1": (
        "No unusual event; continue ordinary same-direction lane keeping, following, "
        "safe-distance keeping, same-direction lane adjustment, or recovery after a completed maneuver."
    ),
    "R2": (
        "No unusual event; continue along the bidirectional narrow-road space while keeping "
        "safe clearance from oncoming traffic, without an active blockage or invading oncoming vehicle."
    ),
    "R3": (
        "No unusual event; continue normal highway, ramp, merge, split, or exit behavior such "
        "as speed matching, gap keeping, target-lane tracking, merging, or exiting."
    ),
    "R4": (
        "No unusual event; obey normal traffic-light intersection rules such as stopping for "
        "red, proceeding on green, or turning under signal control."
    ),
    "R5": (
        "No unusual event; negotiate the unsignalized or priority intersection using stop/yield "
        "rules, right-of-way, and safe-gap reasoning."
    ),
}

REGULAR_EVENT_DESCRIPTIONS: Dict[str, str] = {
    "R-E1": (
        "regular following, lane keeping, safe-distance keeping, or speed matching "
        "without a short-horizon conflict"
    ),
    "R-E2": (
        "regular target-directed lane change, return-to-lane, or completed recovery "
        "maneuver on a drivable path"
    ),
    "R-E3": (
        "regular highway or ramp behavior such as merging, diverging, splitting, "
        "exiting, or tracking the target lane"
    ),
    "R-E4": (
        "regular traffic-light intersection behavior, including red-light waiting, "
        "green-light crossing, or signal-controlled turning"
    ),
    "R-E5": (
        "regular unsignalized or priority-intersection behavior using stop, yield, "
        "right-of-way, cross-traffic, and safe-gap reasoning"
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
    **UE_DESCRIPTIONS,
    "RE": "No unusual event is currently interrupting the driving task; continue the regular behavior implied by the current road structure.",
}

EVENT_ORDER: Tuple[str, ...] = (
    # 多标签没有置信度或 primary 不可用时，用这个全局顺序做确定性兜底。
    # 这样同一份数据在不同机器/不同 Python hash seed 下不会得到不同 teacher target。
    "R-E1", "R-E2", "R-E3", "R-E4", "R-E5",
    "U-E1", "U-E2", "U-E3", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8",
)


@dataclass(frozen=True)
class RSTarget:
    """单帧 RS 训练目标。

    `option` 是学生输出的 A-E；`label` 是内部 R1-R5。`candidates` 保存原始打分，
    方便后续 probe 回查“为什么双标签最后选了哪个”。
    """

    label: str
    option: str
    description: str
    confidence: float
    secondary: Tuple[str, ...]
    candidates: Dict[str, float]


@dataclass(frozen=True)
class EventTarget:
    """单帧 EVENT 训练目标。

    `label` 是 v5 监督用标签：RE 或 U-E*。`event_code` 保留原始 R-E*/U-E*，
    即使 regular 被折叠成 RE，也可以从这里知道标定原来是哪一种 regular。
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
    """判断某个 canonical/raw EVENT code 是否属于 unusual family。

    正常行为 ``RE`` 与原始 ``R-E*`` 都返回 ``False``；只有 ``U-E*`` 返回 ``True``。
    该函数只区分 family，不负责验证编号是否合法。
    """

    return str(code).startswith("U-E")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把候选分数/置信度容错转换为浮点，失败时返回显式默认值。"""

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
            if str(key) in RS_LABEL_TO_OPTION:
                candidates[str(key)] = _safe_float(value)

    label = str(primary) if primary in RS_LABEL_TO_OPTION else "R1"
    if candidates:
        # RS 多标签只训练一个标签：优先最高置信度；置信度相同则按 R1-R5 稳定顺序。
        # 这里不采用 student 动态答案，是为了避免 Q1 的路结构监督变成多目标漂移。
        best_label, _ = max(
            candidates.items(),
            key=lambda item: (item[1], -list(RS_OPTION_TO_LABEL.values()).index(item[0]) if item[0] in RS_OPTION_TO_LABEL.values() else -99),
        )
        label = best_label

    option = RS_LABEL_TO_OPTION[label]
    rs_ann = frame.get("frame_rs_annotation") or {}
    secondary = tuple(str(x) for x in (rs_ann.get("secondary") or frame.get("secondary_road_structures") or []) if x)
    return RSTarget(
        label=label,
        option=option,
        description=RS_OPTION_DESCRIPTIONS[option],
        confidence=_safe_float(rs_ann.get("confidence", frame.get("confidence", 0.0))),
        secondary=secondary,
        candidates=candidates,
    )


def raw_events_from_frame(frame: Mapping[str, Any]) -> Tuple[str, ...]:
    """按权威优先级读取、规范化并去重原始 EVENT 列表。

    优先读取新版 ``frame_event_annotation.events``，再兼容旧顶层字段，最后才退回
    primary 单标签。显式 ``RE`` 不属于原始细分 code，因而不会进入返回值。
    """

    event_ann = frame.get("frame_event_annotation") or {}
    raw = event_ann.get("events") or frame.get("events") or frame.get("event_labels_raw") or None
    if not raw:
        primary = event_ann.get("label") or frame.get("primary_event") or frame.get("event_code")
        raw = [primary] if primary else []
    # 保持源数据顺序去重，便于 JSON 审计；后续需要单目标时再走 _stable_first。
    out: List[str] = []
    for item in raw:
        code = normalize_event_code(item)
        if code and code != "RE" and code not in out:
            out.append(code)
    return tuple(out)


def _stable_first(codes: Iterable[str]) -> str:
    """按全局 ``EVENT_ORDER`` 取稳定第一项。

    输入可能来自 set/dict 等无稳定顺序容器，因此不能直接 ``next(iter(...))``。
    空集合返回 R-E1 只用于旧数据的 regular 审计兜底，最终监督仍会折叠为 RE。
    """

    items = set(codes)
    for code in EVENT_ORDER:
        if code in items:
            return code
    return sorted(items)[0] if items else "R-E1"


def resolve_event_target(
    frame: Mapping[str, Any],
    *,
    student_event: Optional[str] = None,
) -> EventTarget:
    """把可能多标签的原始 EVENT 解析成 v5 单标签目标。

    规则对应 plan §4.3：
    - 有 UE 时 UE 优先；
    - 多个 UE 时，如果 student 输出是其中之一，teacher target 采用 student 输出；
      否则用 primary_event / 稳定顺序；
    - 全 regular 时折叠为 RE，但 `event_code` 保留具体 R-E*。
    """

    raw_events = raw_events_from_frame(frame)
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
        # 全 regular 的双标签会折叠成 RE；event_code 仍记住具体 R-E*。
        # 若 student 选中了 raw regular 之一，teacher target 使用该 regular 作为审计 code，
        # 否则按 primary/稳定顺序选一个，不影响最终监督 label=RE。
        chosen_re = str(student)
    else:
        primary = normalize_event_code((frame.get("frame_event_annotation") or {}).get("label") or frame.get("primary_event"))
        chosen_re = str(primary) if primary in regular else _stable_first(regular or ["R-E1"])
    regular_codes = tuple(regular or [chosen_re])
    return EventTarget(label="RE", event_code=chosen_re, abnormal=False, raw_events=tuple(raw_events), regular_event_codes=regular_codes)


def scenario_event_candidates_from_result(result: Mapping[str, Any]) -> List[str]:
    """读取单场景 result 顶层 event_candidates。

    如果旧文件缺少该字段，就返回全量 EVENT_ORDER 作为保守兜底；真正进入 Q2 前仍会
    与当前 RS 的候选池取交集，所以不会把其它 RS 的事件放进选项。
    """

    raw = result.get("event_candidates") or []
    out: List[str] = []
    for item in raw:
        code = normalize_event_code(item)
        if code and code != "RE" and code not in out:
            out.append(code)
    return out or list(EVENT_ORDER)


def q2_raw_candidates(scenario_candidates: Sequence[str], rs_label: str) -> List[str]:
    """计算旧数据的 Q2 fallback：scenario 候选与当前 RS 候选取交集。

    返回顺序以 ``EVENT_CANDIDATES_BY_RS`` 为准，保留 R-E*/U-E* 原始 code。新版逐帧
    ``allowed_events`` 存在时不调用本规则，避免静态表覆盖人工审核后的 frame 候选。
    """

    scenario_set = {str(x) for x in scenario_candidates}
    return [code for code in EVENT_CANDIDATES_BY_RS.get(rs_label, []) if code in scenario_set]


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
    """按 v5 口径得到 Q2 原始候选。

    首选逐帧 allowed_events；只有缺失时才 fallback 到
    `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[current_rs]`。
    """

    allowed = allowed_events_from_frame(frame)
    if allowed:
        # allowed_events 是逐帧最终候选，已经融合了 keyframe_filter 的 clamp/overlay。
        # 只要存在就直接采用，不能再用静态表强行改写，否则会把人工修正的候选冲掉。
        return allowed
    return q2_raw_candidates(scenario_candidates, rs_label)


def collapse_regular_to_re(candidates: Sequence[str], rs_label: str) -> List[str]:
    """把逐帧 allowed candidates 里的所有 regular 分支折叠成 prompt 里的 RE。

    逐帧 `allowed_events` 已经包含 collector 的最终 clamp / overlay 结果，不能再
    用当前 RS 的静态 regular 表二次过滤；否则会丢掉 final clamp 或 interrupted
    overlay 留下的例外 regular code。合并后的 EVENT_FAST 是一次性的
    REGULAR-vs-UNUSUAL 具体事件选择，所以即使原始 allowed 只列 UE，prompt 也必须
    保留一个 RE 负类选项，让模型能根据当帧 RGB 否定诱导性 UE 候选。
    """

    out: List[str] = ["RE"]
    for code in candidates:
        if is_unusual(code) and code not in out:
            out.append(code)
    return out


def event_description_for_display(
    label: str,
    rs_label: str,
    regular_event_codes: Optional[Sequence[str]] = None,
) -> str:
    """把 canonical EVENT label 展开成 Q2 可读描述。

    UE 直接查固定描述；RE 先按当前 RS 解释“正常驾驶意味着什么”，再附加本帧允许的
    R-E* 细分模式。``regular_event_codes=None`` 表示不追加细分，显式空序列则回退当前
    RS 的标准 regular 列表，这两种语义由调用方有意区分。
    """

    if label == "RE":
        base = RE_DESCRIPTIONS_BY_RS.get(rs_label, EVENT_DESCRIPTIONS["RE"])
        codes: List[str] = []
        raw_codes = list(regular_event_codes or []) if regular_event_codes is not None else []
        if regular_event_codes is not None and not raw_codes:
            raw_codes = RS_REGULAR_EVENTS.get(rs_label, [])
        for item in raw_codes:
            code = normalize_event_code(item)
            if code and code.startswith("R-E") and code not in codes:
                codes.append(code)
        # 细分 code 只扩充自然语言，不会把一个 RE 重新拆回多个答案。
        details = [REGULAR_EVENT_DESCRIPTIONS[code] for code in codes if code in REGULAR_EVENT_DESCRIPTIONS]
        if details:
            return f"{base} Regular modes allowed for this frame include: {'; '.join(details)}."
        return base
    return UE_DESCRIPTIONS.get(label, label)


def stable_event_option_map(
    *,
    run_id: str,
    frame_id: int,
    rs_label: str,
    scenario_candidates: Sequence[str],
    raw_candidates: Optional[Sequence[str]] = None,
    seed: int = 0,
    dataset_version: str = DATASET_VERSION,
) -> Dict[str, str]:
    """为某一帧生成可复现随机的 Q2 选项字母映射。

    同一个 dataset_version + run_id + frame_id + seed 永远得到同样顺序；不同帧会
    打乱，避免模型学到“A 总是 RE”这种捷径。
    """

    raw = list(raw_candidates) if raw_candidates is not None else q2_raw_candidates(scenario_candidates, rs_label)
    display = collapse_regular_to_re(raw, rs_label)
    # 随机只打乱“字母到标签”的映射，不改变本帧候选集合；seed 源里包含 dataset_version，
    # 以后如果候选协议变化，可以自然得到一套新映射，避免旧缓存混用。
    seed_src = f"{dataset_version}::{run_id}::{frame_id}::{seed}".encode("utf-8")
    rng_seed = int(hashlib.sha256(seed_src).hexdigest(), 16) % (2**31)
    items = list(display)
    random.Random(rng_seed).shuffle(items)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(items) > len(letters):
        raise ValueError(f"too many Q2 event choices: {len(items)}")
    return {letters[i]: item for i, item in enumerate(items)}


def option_for_event(label: str, option_map: Mapping[str, str]) -> Optional[str]:
    """从 canonical event label 反查本帧随机 option letter。

    找不到返回 ``None``，上层据此记录 candidate mismatch；不能默认 A，因为每帧映射都
    会稳定随机打乱，默认字母会制造错误监督。
    """

    for letter, value in option_map.items():
        if value == label:
            return str(letter)
    return None


def weather_to_text(weather: Mapping[str, Any] | None) -> str:
    """把 XML weather 数值压成 teacher 用的短英文描述。

    Student 不直接看到这段文字；它只进入 teacher privileged prompt 和数据审计字段。
    阈值只是把连续 XML 数值压缩为基础模型更易理解的短语，不参与 RS/EVENT 真值规则；
    teacher prompt 也被要求在 XML 与 RGB 冲突时以图像证据为准。
    """

    if not weather:
        return "weather is not available from XML"
    cloud = _safe_float(weather.get("cloudiness"), 0.0)
    rain = _safe_float(weather.get("precipitation"), 0.0)
    wet = _safe_float(weather.get("wetness"), 0.0)
    fog = _safe_float(weather.get("fog_density"), 0.0)
    sun = _safe_float(weather.get("sun_altitude_angle"), 45.0)
    # 五类因素分别追加，最终逗号连接成一行；保留多个因素比只输出晴/雨更能描述
    # 夜间、湿地但已停雨、轻雾等组合条件。
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
