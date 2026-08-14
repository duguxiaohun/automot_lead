"""Phase2 的四项 ROAD_STRUCTURE 提示词与严格 YES/NO 解析。"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional

from qwen3vl_local.sft_loop_phase2.history_rgb import (
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODE_ALL4,
    history_rgb_prompt_description,
    validate_history_rgb_mode,
)


PROMPT_NAME = "sft_loop_phase2_rs_four_binary_visual"
ANSWER_KEYS = ("RS1", "RS2", "RS4", "RS5")
ANSWER_VALUES = ("YES", "NO")


SYSTEM_PROMPT = """You are the second perception step of an autonomous-driving agent.
The input is a stitched three-camera RGB history, ordered from oldest to newest. Classify only the newest moment. Inspect the complete left/front/right scene and use older frames only to confirm road geometry, motion, visibility, or an approaching junction. Do not use scenario names, dataset labels, maps, hidden metadata, or memory. The questions ask about the driving-rule structure visible now, not about the event: an obstacle, a pedestrian, turning vehicle, rain, darkness, or braking does not by itself decide the road structure."""


def build_phase2_prompt(*, audit: bool = False, history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE) -> str:
    """Build the no-memory Phase2 prompt; audit only asks for visible evidence."""

    mode = validate_history_rgb_mode(history_rgb_mode)
    history = history_rgb_prompt_description(mode)
    endpoint_notice = "" if mode == HISTORY_RGB_MODE_ALL4 else " Only the first and fourth frames of the original four-frame history are visible; do not assume intermediate evidence exists."
    text = f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[VISUAL_CHECK_ORDER]
Classify the newest frame using the {history}. First trace the ego vehicle's usable corridor and its lane boundaries. Then inspect whether an opposing direction must participate, whether a junction conflict area is already being approached or entered, and whether traffic-signal hardware at that same junction is the active right-of-way rule. Scan the full left/front/right scene for a side road, cross street, stop/yield sign, signal mast or overhead arm, stop line, crosswalk, median, oncoming lane, and lane split. Answer RS1, RS2, RS4, and RS5 independently from their own definitions and the visible RGB evidence. Do not make one answer a prerequisite, exclusion rule, or correction for another answer. Use older frames only to confirm a cue that is visible in the newest moment; never invent a hidden junction or road topology.{endpoint_notice}
[/VISUAL_CHECK_ORDER]

[DECISION_RULES]
RS1:
Ask: "Is the ego currently on an ordinary same-direction drivable road, outside a controlling junction, opposing-lane-sharing constraint, and highway/ramp decision structure?"
YES when the usable path is an ordinary lane or ordinary same-direction roadway: lane markings or road edges continue ahead, ordinary following/lane keeping/ordinary lane change is the main rule, and there is no active junction control or need to negotiate an opposing lane. This includes local, urban, suburban, rural, parking-side, roundabout, and ordinary surface-road segments.
NO when an unseparated opposing direction governs the usable corridor, local signal/priority intersection control governs the ego path, or the road is a limited-access highway/ramp/merge/exit structure. Do not call a road RS1 merely because it is straight, empty, wide, rainy, foggy, or has a lead vehicle. A roundabout is RS1 unless a separate signal/priority junction visibly governs the ego path.

RS2:
Ask: "Does the opposing direction currently constrain the ego's usable corridor, so the ego may need to share, borrow, yield around, or reason about the oncoming lane?"
YES needs visible road-layout evidence, not a scenario name or a nearby parked car. Strong evidence is a narrow two-way corridor with a centre line and little physical separation, an oncoming vehicle/lane immediately relevant to the ego path, or a fixed obstruction/parked vehicles/door leaving only a passage that requires entering or yielding to the opposite lane. A narrow street with recurring opposing headlights/vehicle fronts across its centre line and parked vehicles at the curb is RS2 even when ego can still move forward in its own lane in this exact frame. Four nominal lanes can still be RS2 when parked vehicles or obstacles remove the side lanes so that the remaining usable space is effectively one lane each way.
NO for a normal same-direction multi-lane road, ordinary traffic in a separate opposing carriageway behind a median/barrier, a curbside parked vehicle that leaves ego's lane open, or a road where the opposite lane is visible but clearly does not constrain ego's current usable space. A double-yellow line alone is not enough when the road is broad and ego can continue normally without any opposing-lane decision.

RS4:
Ask: "Is the ego in the approach, stop line, conflict area, or immediate exit of an intersection where traffic-signal hardware is the governing right-of-way rule?"
YES when the current visible junction has readable signal heads/masts/overhead signal arms controlling the ego approach or the intersecting approaches, together with local junction geometry such as a cross street, T-junction, turning pocket, stop line, crosswalk, median opening, or crossing traffic. A signal head/arm plus a local stop line or crosswalk is sufficient even when rain or fog partly hides the cross street. Actively inspect the upper front view and both side views: the physical signal head/arm can be small, high, partly masked, or on the far side. A failed, dark, flashing, or contradictory signal system is still RS4 when visible traffic-signal hardware is the rule source at this junction; Phase1 handles whether it is abnormal.
NO for a distant tiny signal that does not govern the current junction, a traffic light seen down another road, a normal road with a far signal but no local junction geometry, a pedestrian signal only, a signalized junction already clearly left behind, or a vehicle running a normal red light. Do not infer RS4 from a scenario name, a light reflection, a green/red pixel, or an at-grade road alone.

RS5:
Ask: "Is the ego in the approach, conflict area, or immediate exit of an intersection whose rule is priority, stop/yield, gap acceptance, or geometry rather than a working traffic signal?"
YES when a cross street, T-junction, angled junction, side-road merge, stop/yield control, STOP marking, visible priority conflict, crossing traffic, or clear junction box is local to ego and no working traffic-signal rule governs that conflict. A readable STOP or yield sign beside the ego approach is a strong RS5 cue when the road visibly meets/crosses another road, even in fog. T-shaped intersections count here when unlit or stop/yield/priority controlled. The vehicle may be turning or going straight; the deciding evidence is the local no-signal/priority junction structure.
NO for ordinary road bends, driveways, parking exits, a far side street with no immediate conflict, a roundabout, or a junction visibly governed by working traffic-signal hardware. Do not turn RS5 into a catch-all for any turn, any braking, any crosswalk, or a missing/too-small signal in fog.

HIGHWAY/RAMP ROBUSTNESS:
A limited-access highway mainline, ramp, merge, split, exit, connector, gore area, or controlled high-speed corridor is a visual road type to distinguish from an ordinary surface road: several same-direction high-speed lanes with continuous guardrails/barriers, a merge/split/gore/exit, or a grade-separated connector are useful positive clues. Decide whether this road type is present only from those RGB clues, not from the answers to other questions. Do not call it an ordinary same-direction road merely because it has lane markings. Do not treat another carriageway behind a barrier as an opposing-lane constraint, or a distant lamp or bridge as a local junction control.

INDEPENDENT ANSWER CHECK:
For each question, output only the YES or NO supported by that question's RGB evidence. Do not revise one answer because of another answer, and do not use the final four-line answer pattern to infer hidden road topology.
[/DECISION_RULES]
""".strip()
    if audit:
        output = """
[AUDIT_OUTPUT]
For every answer, give one short externally checkable observation from the RGB history, then the answer. Do not give hidden reasoning, dataset knowledge, or guessed topology. Name the clearest frame position for small/brief evidence. For RS2=YES name the opposing-lane constraint; for RS4=YES name the local signal hardware and junction cue; for RS5=YES name the local no-signal/priority junction cue; for RS1=YES name the continuing ordinary same-direction corridor. For a NO, name the main rejected cue only when useful.
EVIDENCE_RS1: <short visible observation>
RS1: <YES or NO>
EVIDENCE_RS2: <short visible observation>
RS2: <YES or NO>
EVIDENCE_RS4: <short visible observation>
RS4: <YES or NO>
EVIDENCE_RS5: <short visible observation>
RS5: <YES or NO>
[/AUDIT_OUTPUT]
"""
    else:
        output = """
[OUTPUT]
Output exactly four lines and nothing else:
RS1: <YES or NO>
RS2: <YES or NO>
RS4: <YES or NO>
RS5: <YES or NO>
[/OUTPUT]
"""
    return text + "\n" + output.strip()


def build_phase2_target(answers: Dict[str, bool]) -> str:
    """Render the strict four-line SFT target."""

    return "\n".join(f"{key}: {'YES' if bool(answers[key]) else 'NO'}" for key in ANSWER_KEYS)


def parse_phase2_output(text: str) -> Dict[str, Optional[bool]]:
    """Strictly parse standalone YES/NO values without guessing omitted lines."""

    out: Dict[str, Optional[bool]] = {key: None for key in ANSWER_KEYS}
    for key in ANSWER_KEYS:
        matches = re.findall(rf"(?im)^\s*{re.escape(key)}\s*:\s*(YES|NO)\s*$", text)
        if len(matches) == 1:
            out[key] = matches[0] == "YES"
    return out


def phase2_prompt_sha256(*, audit: bool = False, history_rgb_mode: str = DEFAULT_HISTORY_RGB_MODE) -> str:
    """Return a reproducible prompt fingerprint persisted with each adapter/eval."""

    payload = SYSTEM_PROMPT + "\n" + build_phase2_prompt(audit=audit, history_rgb_mode=history_rgb_mode)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
