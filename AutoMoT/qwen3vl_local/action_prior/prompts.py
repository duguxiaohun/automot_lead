"""base Qwen 只压缩既有先验，不接收 expert action 或未来标签。"""

import json

ANALYSIS_VERSION = "bounded_prior_summary_v1"
SYSTEM_PROMPT = """You summarize perception priors for a driving trajectory decoder.
Treat each non-null supplied prior as the accepted condition. Do not independently reclassify it.
A null/INVALID field is missing conditioning, not NO and not an instruction to invent a replacement.
Use the four chronological stitched RGB images only to give brief context consistent with accepted priors.
Write exactly three short paragraphs, each one sentence (at most 25 words):
Scene: summarize the accepted road structure and visible facts.
Interaction: summarize accepted event conditions, or state that event conditioning is unavailable.
Planning context: relate those conditions to the supplied current speed and navigation, without choosing a new semantic action class.
Do not invent actor identity, future behavior, traffic-light color, right of way, or missing events.
If all semantic priors are missing, simply acknowledge the limited conditioning and available navigation.
Do not output trajectory coordinates, action labels, extended reasoning, or additional paragraphs."""


def analysis_prompt(priors, navigation):
    """白名单序列化；原始问答/GT/场景名不会进入分析 prompt。"""
    navigation = navigation.split(" Predict the driving actions", 1)[0]
    return (
        "[CONDITION_MEANINGS]\n"
        + json.dumps(CONDITION_MEANINGS, sort_keys=True)
        + "\n[/CONDITION_MEANINGS]\n[ACCEPTED_PERCEPTION_PRIORS]\n"
        + json.dumps(priors["conditions"], sort_keys=True)
        + "\n[/ACCEPTED_PERCEPTION_PRIORS]\n[CURRENT_NAVIGATION]\n"
        + navigation
        + "\n[/CURRENT_NAVIGATION]\nProvide the three short summaries now."
    )


CONDITION_MEANINGS = {
    "R1": "ordinary same-direction surface-road lane-following corridor",
    "R2": "opposing-lane or two-way shared-corridor constraint",
    "R3": "none of the four queried road structures; highway is an independent fact",
    "R4": "local junction with traffic-signal hardware governing the path",
    "R5": "local junction governed by priority, stop/yield or gap acceptance",
    "UE1": "lead-vehicle hard braking or sudden slowdown",
    "UE3": "another vehicle cutting into or dynamically occupying the immediate ego corridor, including highway cut-in",
    "UE5": "abnormal oncoming invasion into the usable ego corridor",
    "UE6": "visibly rule-violating vehicle conflict at a local junction",
    "HIGHWAY": "visible limited-access highway topology",
    "STATIC_OBSTACLE": "visible static obstacle condition",
    "VULNERABLE": "visible vulnerable road-user condition",
    "TRAFFIC_LIGHT_ABNORMAL": "visible abnormal traffic-light hardware condition",
    "RS_HIGHWAY": "independently checked highway/ramp/connector topology",
}


def valid_analysis(text):
    """检测截断、额外段落和过长分析；语义是否臆造仍需抽检原始回答。"""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    headings = ("Scene:", "Interaction:", "Planning context:")
    return len(lines) == 3 and all(
        line.startswith(h) and len(line[len(h) :].split()) in range(1, 26)
        for line, h in zip(lines, headings)
    )


def fallback_analysis(priors):
    """格式失败时用可追溯的三句简述，保留先验而不创造缺失事实。"""
    c = priors["conditions"]
    rs = c.get("ROAD_STRUCTURE")
    scene = CONDITION_MEANINGS.get(rs, "road-structure conditioning unavailable")
    active = [k for k in ("UE1", "UE3", "UE5", "UE6") if c.get(k) == "YES"]
    known = [k for k in ("UE1", "UE3", "UE5", "UE6") if c.get(k) is not None]
    event = (
        ("Accepted active event conditions: " + ", ".join(active))
        if active
        else (
            "Available event conditions are negative; unconfirmed fields remain unknown"
            if known
            else "Event conditioning is unavailable"
        )
    )
    return f"Scene: {scene}.\n\nInteraction: {event}.\n\nPlanning context: Use the accepted conditions with current speed and navigation; retain uncertainty for missing conditions."
