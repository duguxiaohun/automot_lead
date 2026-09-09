"""把接受的 RS/EVENT 条件压缩为自然场景先验，供 base Qwen 形成短分析。"""

from __future__ import annotations

import json
import re


# 新提示词进入 transcript KV；旧 v4 的 JSON 条件/经验表不能与本协议混用。
ANALYSIS_VERSION = "natural_scene_prior_concise_summary_v5"
MAX_ANALYSIS_WORDS = 80

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


def _accepted_yes(conditions, key):
    """仅 YES 会进入自然语言；NO 和缺失均不会作为反向答案渲染。"""
    return conditions.get(key) == "YES"


def scene_description(conditions):
    """按已确认 RS/事件拼接短自然段，不泄露分类字段或候选反例。"""
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
    return " ".join(sentences)


def condition_context(priors, navigation):
    """最终 transcript 只有一段情景先验和公开导航，不含 JSON 类别表或反例。"""
    navigation = navigation.split(" Predict the driving actions", 1)[0].strip()
    return (
        "[SCENE_DESCRIPTION]\n"
        + scene_description(priors["conditions"])
        + "\n[/SCENE_DESCRIPTION]\n[CURRENT_NAVIGATION]\n"
        + navigation
        + "\n[/CURRENT_NAVIGATION]"
    )


def analysis_prompt(priors, navigation):
    """让 base 将简短先验、图像与导航自然地总结为一个段落。"""
    return condition_context(priors, navigation) + "\nWrite the concise planning summary now."


def review_prompt(priors, navigation, draft):
    """纯文本复核读取同一自然先验，不引入此前冗长 JSON 协议。"""
    return (
        condition_context(priors, navigation)
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
    """仅验一个完整短段和词数；不把格式验收误称为语义验收。"""
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
    description = scene_description(priors["conditions"])
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
