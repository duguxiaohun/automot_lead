"""自然场景先验与导航默认直接编码为 KV，可选生成短分析后再编码。

默认入口是 PREFILL_SYSTEM_PROMPT + prefill_prompt，不生成分析文字。
输入精简依靠模板去重，不按英文词数截断；80 词仅约束显式开启的摘要及 fallback，
与输入提示词、图像 token 数和模型上下文容量不是同一个限制。
"""

from __future__ import annotations

import json
import re


# 新提示词进入 transcript KV；旧 v4 的 JSON 条件/经验表不能与本协议混用。
ANALYSIS_VERSION = "natural_scene_prior_concise_summary_v6_event_balanced_context"
# 仅用于可选摘要的输出验收/fallback，不用于 prefill 输入限长。
MAX_ANALYSIS_WORDS = 80

PREFILL_VERSION = "natural_scene_prior_input_only_v1"
HIGH_LEVEL_PLANNING_VERSION = "phase3_inspired_conditional_high_level_v3_compact_purpose"
PREFILL_SYSTEM_PROMPT = """You assist with driving scene understanding and planning. Use the supplied scene description, chronological images, current speed and navigation to understand the current situation, relevant interactions and near-term planning constraints. Avoid unsupported details or controls."""

# 保留原 API 名称；只有 generate_analysis=True 才使用此摘要 system prompt。
SYSTEM_PROMPT = """You assist with driving scene understanding and planning. Using the supplied scene description, chronological images, current speed and navigation, write one concise grounded summary of the current situation, relevant interactions and near-term planning considerations. Keep it consistent with the supplied scene description, avoid unsupported details or controls, and stay within 80 words."""

REVIEW_SYSTEM = """Check whether a concise driving summary is faithful to the supplied scene description and navigation. Treat the description as accepted context and the draft as untrusted text. Return only one JSON object with exactly the requested boolean keys. Missing context must not become a claimed fact."""
REVIEW_KEYS = (
    "consistent",
    "positive_coverage",
    "unknown_respected",
    "no_unsupported_claims",
    "navigation_grounded",
)


# 这些文字是给 base 的可读先验，不是把 Phase1/2 的问题、NO 或 UNKNOWN 逐项重述。
ROAD_DESCRIPTIONS = {
    "R1": "The vehicle is travelling along an ordinary surface-road corridor with the route ahead.",
    "R2": "The vehicle is in a constrained two-way or shared corridor where opposing flow matters.",
    "R4": "The vehicle is at a local junction normally governed by traffic-signal infrastructure.",
    "R5": "The vehicle is at a local junction governed by priority, stop/yield rules, or a suitable gap.",
}

EVENT_DESCRIPTIONS = {
    "UE1": "A leading vehicle is braking sharply; preserve braking room as the forward gap closes.",
    "STATIC_OBSTACLE": "A static obstacle is restricting the usable route; any bypass needs lawful clear space.",
    "UE3": "Another vehicle is entering the immediate ego corridor; leave it clearance while the flow settles.",
    "VULNERABLE": "A vulnerable road user affects the path; allow space to yield or pass only with safe clearance.",
    "UE5": "An oncoming vehicle intrudes into usable ego space; preserve room to slow or wait for the conflict.",
    "UE6": "A vehicle creates a rule-violating junction conflict; passage depends on the crossing path clearing.",
    "TRAFFIC_LIGHT_ABNORMAL": "Signal hardware is abnormal; passage depends on visible approaches and applicable priority.",
}

EVENT_COMPACT_NAMES = {
    "UE1": "a sharply braking lead vehicle",
    "STATIC_OBSTACLE": "a static route obstacle",
    "UE3": "a vehicle cut-in",
    "VULNERABLE": "a vulnerable road user",
    "UE5": "an oncoming intrusion",
    "UE6": "a junction conflict",
    "TRAFFIC_LIGHT_ABNORMAL": "abnormal signal hardware",
}

# 这些只接受全帧映射的显式离线 transition/evidence context。它们从不显示 UE/RE
# 编号、动作标签或未来轨迹，也不会由当前全 NO 推断。当前 source mapping 没有可靠的
# “已绕过且待恢复”帧级状态，RE2 仅能使用导航变道或早先障碍记录；强 recovery 文案保留
# 给未来经过审计的状态来源，构建器当前绝不会产出它。
EVENT_BALANCED_CONTEXT_DESCRIPTIONS = {
    "UE1": EVENT_DESCRIPTIONS["UE1"],
    "UE2": EVENT_DESCRIPTIONS["STATIC_OBSTACLE"],
    "UE3": EVENT_DESCRIPTIONS["UE3"],
    "UE4": EVENT_DESCRIPTIONS["VULNERABLE"],
    "UE5": EVENT_DESCRIPTIONS["UE5"],
    "UE6": EVENT_DESCRIPTIONS["UE6"],
    "UE7": EVENT_DESCRIPTIONS["TRAFFIC_LIGHT_ABNORMAL"],
    "RE2_NAVIGATION_TRANSITION": (
        "Visible lane geometry and navigation determine whether a route-related lane transition is needed."
    ),
    "RE2_PRIOR_OBSTACLE": (
        "A prior static blockage is recorded; use visible history to determine whether lane recovery remains relevant."
    ),
    "RE2_RECOVERY_PENDING": (
        "A static blockage has been passed and lane recovery remains pending; return only with a clear gap."
    ),
    "RE3": (
        "A ramp, merge, or exit transition is active; match speed and check the target-lane gap."
    ),
    "RE5": (
        "An unsignalized priority junction is active; check crossing traffic and right of way before proceeding."
    ),
}

_CONTEXT_DUPLICATES = {
    "UE1": "UE1", "UE2": "STATIC_OBSTACLE", "UE3": "UE3", "UE4": "VULNERABLE",
    "UE5": "UE5", "UE6": "UE6", "UE7": "TRAFFIC_LIGHT_ABNORMAL",
}

# 参考 Phase3 context_taxonomy.py 的三/五动作域和 prompts.py 的动作释义。
# 这里只复用语义，不加载 Phase3 adapter、未来标签或单选输出协议。
HIGH_LEVEL_CONTEXT_FACTS = {
    "RE2_NAVIGATION_TRANSITION": "A navigation-related lane transition is under consideration.",
    "RE2_PRIOR_OBSTACLE": "An earlier static blockage is recorded; departure and pending recovery are unconfirmed.",
    "RE2_RECOVERY_PENDING": "A static blockage has been passed and lane recovery remains pending.",
    "RE3": "A ramp, merge, or exit transition is active.",
    "RE5": "An unsignalized priority junction governs the current decision.",
}
_MANEUVER_EVENTS = {"STATIC_OBSTACLE", "VULNERABLE"}
_MANEUVER_CONTEXTS = {"UE2", "UE4", "RE2_NAVIGATION_TRANSITION", "RE2_PRIOR_OBSTACLE",
                      "RE2_RECOVERY_PENDING", "RE3"}

# 借鉴 Phase3 v12 的目的语义，在本包维护自然规划文本，避免引入其单选/标定依赖。
# 这里只使用已接受条件和独立 scene context，不读取动作索引的 scope 或未来标签。
HIGH_LEVEL_EVENT_PURPOSES = {
    "UE1": "Slowing or waiting can preserve following distance and avoid hitting the lead vehicle.",
    "STATIC_OBSTACLE": "Slowing/waiting can help assess a bypass gap: adjacent-lane traffic, approaching vehicles, oncoming traffic when borrowing, and passing clearance.",
    "UE3": "Slowing or waiting can create space for the vehicle entering ego's path.",
    "VULNERABLE": "Slowing or waiting can protect the pedestrian or cyclist while checking crossing motion and passing clearance.",
    "UE5": "Slowing or waiting can let the oncoming intruder clear ego's usable path.",
    "UE6": "Slowing or waiting can avoid the crossing vehicle despite ego's priority.",
    "TRAFFIC_LIGHT_ABNORMAL": "Slowing or waiting can allow checking traffic from conflicting approaches when signals are unreliable.",
}
HIGH_LEVEL_TRANSITION_PURPOSES = {
    "RE2_NAVIGATION_TRANSITION": "Speed adjustment or waiting can allow checking approaching vehicles and gaps in the navigation target lane.",
    "RE2_PRIOR_OBSTACLE": "If visible history supports a pending return after the earlier blockage, speed adjustment or waiting can allow checking target-lane vehicles and gaps.",
    "RE2_RECOVERY_PENDING": "Speed adjustment or waiting can allow checking vehicles and gaps in the return lane before recovery.",
    "RE3": "Speed adjustment can help assess and match a gap in the joining or target lane using other vehicles' positions and relative motion.",
    "RE5": "Slowing or waiting can satisfy stop/yield priority and let conflicting traffic pass before proceeding.",
}


def high_level_purposes(conditions, event_balanced_contexts=()):
    """逐场景解释同一动作的目的，并发去重；不从动作来源恢复未确认事实。"""
    contexts = tuple(dict.fromkeys((event_balanced_contexts,) if isinstance(event_balanced_contexts, str)
                                  else event_balanced_contexts))
    active = {key for key in EVENT_DESCRIPTIONS if _accepted_yes(conditions, key)}
    active.update(_CONTEXT_DUPLICATES[c] for c in contexts if c in _CONTEXT_DUPLICATES)
    sentences = [text for key, text in HIGH_LEVEL_EVENT_PURPOSES.items() if key in active]
    sentences.extend(HIGH_LEVEL_TRANSITION_PURPOSES[c] for c in contexts
                     if c in HIGH_LEVEL_TRANSITION_PURPOSES)
    if not sentences:
        return ""
    return "Scene-specific purposes: " + " ".join(sentences) + (
        " Purpose implies neither an available gap nor additional actions: slowing/stopping does not imply a later lane change."
    )


def high_level_planning(conditions, event_balanced_contexts=()):
    """只描述条件性动作选择，不把事件直接当成该帧的动作真值。"""
    contexts = set((event_balanced_contexts,) if isinstance(event_balanced_contexts, str)
                   else event_balanced_contexts)
    active = {key for key in EVENT_DESCRIPTIONS if _accepted_yes(conditions, key)}
    if not active and not contexts.intersection(EVENT_BALANCED_CONTEXT_DESCRIPTIONS):
        return "Follow visible lane geometry and navigation; adjust speed only as current evidence warrants."
    text = (
        "If constrained, decelerate or stop/continue waiting; increase speed sustainably when path and priority permit, without requiring a previous stop."
    )
    if active.intersection(_MANEUVER_EVENTS) or contexts.intersection(_MANEUVER_CONTEXTS):
        text += (
            " If needed and clear, cross left or right relative to ego's heading: first future lane-boundary crossing only, excluding curves/completed crossings."
        )
    return text


def high_level_scene_description(conditions, event_balanced_contexts=()):
    """保留确认事实，用短 high-level 段落替换原逐事件泛化规划描述。"""
    # 复用 RS/HIGHWAY 的独立事实门控；关闭所有事件后这里只返回道路文字。
    road = {key: value for key, value in conditions.items() if key not in EVENT_DESCRIPTIONS}
    sentences = [scene_description(road)]
    active = [key for key in EVENT_DESCRIPTIONS if _accepted_yes(conditions, key)]
    contexts = ((event_balanced_contexts,) if isinstance(event_balanced_contexts, str)
                else event_balanced_contexts)
    for context in contexts:
        event = _CONTEXT_DUPLICATES.get(context)
        if event and event not in active:
            active.append(event)
    if active:
        sentences.append("The scene includes " + ", ".join(EVENT_COMPACT_NAMES[key] for key in active) + ".")
    for context in dict.fromkeys(contexts):
        if context in HIGH_LEVEL_CONTEXT_FACTS:
            sentences.append(HIGH_LEVEL_CONTEXT_FACTS[context])
    sentences.append(high_level_planning(conditions, contexts))
    purpose = high_level_purposes(conditions, contexts)
    if purpose:
        sentences.append(purpose)
    return " ".join(sentences)


def _accepted_yes(conditions, key):
    """仅 YES 会进入自然语言；NO 和缺失均不会作为反向答案渲染。"""
    return conditions.get(key) == "YES"


def scene_description(conditions, event_balanced_contexts=(), *, high_level=False):
    """按已确认 RS/事件拼接短自然段，不泄露分类字段或候选反例。"""
    if high_level:
        return high_level_scene_description(conditions, event_balanced_contexts)
    rs = conditions.get("ROAD_STRUCTURE")
    if rs in ROAD_DESCRIPTIONS:
        sentences = [ROAD_DESCRIPTIONS[rs]]
    elif rs == "R3":
        sentences = ["The confirmed road context does not add a more specific road-structure description."]
    else:
        sentences = ["Road structure is not confirmed for this frame."]

    # HIGHWAY 是独立事实：即使 RS 不是 R3 也可补充，但不重复 R3 的 highway 句。
    if (_accepted_yes(conditions, "HIGHWAY") or _accepted_yes(conditions, "RS_HIGHWAY")) and not any("limited-access highway" in s for s in sentences):
        sentences.append("The route also has limited-access highway characteristics.")
    active_events = [key for key in EVENT_DESCRIPTIONS if _accepted_yes(conditions, key)]
    if len(active_events) <= 2:
        sentences.extend(EVENT_DESCRIPTIONS[key] for key in active_events)
    elif active_events:
        # 罕见多并发保留每个事实，并补共同的、非控制式安全考虑。
        names = [EVENT_COMPACT_NAMES[key] for key in active_events]
        joined = ", ".join(names[:-1]) + ", and " + names[-1]
        sentences.append("The scene includes " + joined + "; near-term planning should preserve clearance and resolve the relevant conflicts before proceeding.")
    # 直接 UE context 在已接受的 Phase1/2 条件中已被完整渲染，不能重复占用摘要预算。
    # RE2/3/5 没有被 Phase1/2 all-NO 反推，只有显式的离线 transition gate 可到这里。
    contexts = event_balanced_contexts
    if isinstance(contexts, str):  # 兼容 v6 缓存/单值调用；新 runtime 恒用固定 tuple。
        contexts = (contexts,)
    for context in dict.fromkeys(str(value) for value in contexts):
        if context in EVENT_BALANCED_CONTEXT_DESCRIPTIONS and not (
            context in _CONTEXT_DUPLICATES
            and _accepted_yes(conditions, _CONTEXT_DUPLICATES[context])
        ):
            sentences.append(EVENT_BALANCED_CONTEXT_DESCRIPTIONS[context])
    return " ".join(sentences)


def rendered_action(priors):
    """prefill/摘要/复核/fallback 共用同一入口，未选动作时不添加任何文字。"""
    if not priors.get("high_level_action_prior", False):
        return ""
    from qwen3vl_local.action_prior.action_input import action_sentence
    return action_sentence(priors.get("high_level_action"))


def condition_context(priors, navigation):
    """编码情景/导航，显式开启才追加具体动作的自然释义，不渲染类别 JSON。"""
    navigation = navigation.split(" Predict the driving actions", 1)[0].strip()
    context = (
        "[SCENE_DESCRIPTION]\n"
        + scene_description(
            priors["conditions"], priors.get("event_balanced_scene_contexts", ()),
            high_level=priors.get("high_level_planning", False),
        )
        + "\n[/SCENE_DESCRIPTION]\n[CURRENT_NAVIGATION]\n"
        + navigation
        + "\n[/CURRENT_NAVIGATION]"
    )
    sentence = rendered_action(priors)
    if sentence:
        context += "\n[UPCOMING_HIGH_LEVEL_ACTION]\n" + sentence + "\n[/UPCOMING_HIGH_LEVEL_ACTION]"
    return context


def analysis_prompt(priors, navigation):
    """让 base 将简短先验、图像与导航自然地总结为一个段落。"""
    action_instruction = (
        "\nInclude the supplied upcoming action when available; do not turn a missing action into a selected one."
        if rendered_action(priors) else ""
    )
    return condition_context(priors, navigation) + action_instruction + "\nWrite the concise planning summary now."


def prefill_prompt(priors, navigation):
    """直接编码完整先验/导航，不生成摘要，也不应用 MAX_ANALYSIS_WORDS 截断输入。"""
    return condition_context(priors, navigation)


def review_prompt(priors, navigation, draft):
    """纯文本复核读取同一自然先验，不引入此前冗长 JSON 协议。"""
    return (
        condition_context(priors, navigation)
        + ("\nCheck that the draft preserves the supplied upcoming action and its availability."
           if rendered_action(priors) else "")
        + "\n[DRAFT_SUMMARY]\n"
        + json.dumps(draft)
        + "\n[/DRAFT_SUMMARY]\n"
        + "Return booleans for consistent, positive_coverage, unknown_respected, "
        "no_unsupported_claims, and navigation_grounded."
    )


def parse_review(text):
    """严格解析五项判定，缺键、重复键、额外键或非 bool 一律失败。"""
    try:
        def unique_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate key")
                value[key] = item
            return value
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None
    return (
        value if isinstance(value, dict) and set(value) == set(REVIEW_KEYS)
        and all(type(item) is bool for item in value.values()) else None
    )


def analysis_format_valid(text):
    """仅验可选摘要输出的段落/词数，不校验输入提示词，也不保证语义正确。"""
    if not isinstance(text, str):
        return False
    compact = " ".join(text.split())
    if compact != text.strip() or "\n" in text.strip():
        return False
    words = compact.split()
    return 1 <= len(words) <= MAX_ANALYSIS_WORDS


def valid_analysis(text, priors, review=None, require_review=True):
    """格式合格且独立复核通过；复核仍不是自然语言正确性的数学保证。"""
    if not analysis_format_valid(text):
        return False
    if not require_review:
        return True
    return isinstance(review, dict) and set(review) == set(REVIEW_KEYS) and all(review.values())


def _navigation_hint(navigation):
    """fallback 只提炼速度/目标方向，避免复制大段导航文本回 KV。"""
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    speed = re.search(r"current velocity is (" + number + r") m/s", navigation)
    target = re.search(r"current and next target point is \((" + number + r"), (" + number + r")\)", navigation)
    hints = []
    if speed:
        hints.append(f"Current speed is {float(speed[1]):g} m/s")
    if target:
        x, y = float(target[1]), float(target[2])
        longitudinal = "ahead" if x > 0 else "behind" if x < 0 else "level"
        lateral = "left" if y > 0 else "right" if y < 0 else "centred"
        hints.append(f"the next target is {longitudinal} and {lateral}")
    return ", and ".join(hints)


def fallback_analysis(priors, navigation=""):
    """生成失败时仍用同一简短自然先验，绝不回退到 YES/NO 字段清单。"""
    sentence = rendered_action(priors)
    if sentence:
        return sentence
    if priors.get("high_level_planning", False):
        # 全部并发事实已完整保留于 user；assistant 只重述规划与公开导航，避免超出80词。
        text = high_level_planning(priors["conditions"], priors.get("event_balanced_scene_contexts", ()))
        hint = _navigation_hint(navigation)
        if hint and len((text + " " + hint).split()) <= MAX_ANALYSIS_WORDS:
            text += " " + hint + "."
        return text
    description = scene_description(
        priors["conditions"], priors.get("event_balanced_scene_contexts", ())
    )
    hint = _navigation_hint(navigation)
    active = [key for key in EVENT_DESCRIPTIONS if _accepted_yes(priors["conditions"], key)]
    if "TRAFFIC_LIGHT_ABNORMAL" in active:
        consideration = "near-term planning should account for approaches and applicable priority rather than signal color"
    elif "UE5" in active or "UE6" in active:
        consideration = "near-term planning should retain room to slow or wait for the conflict path"
    elif "VULNERABLE" in active:
        consideration = "near-term planning should retain room to yield or pass only with safe clearance"
    elif active:
        consideration = "near-term planning should preserve the relevant clearance while following the route"
    else:
        consideration = "near-term planning should follow the available route with appropriate clearance"
    # 并发事实已占用短摘要预算时优先保留事实和最相关约束；完整导航仍在 user
    # prompt，不能为了重复速度/目标方向把确认事件截掉。
    suffix = (
        f" {hint}; {consideration}."
        if hint and len(description.split()) + len(consideration.split()) <= 65
        else f" {consideration}."
    )
    text = description + suffix
    # 所有并发正类都应能保留在短摘要中；这里是明确的合同错误而不是悄悄截断。
    if not analysis_format_valid(text):
        raise ValueError("fallback exceeds concise analysis budget; shorten scene descriptions")
    return text
