"""第一轮四项视觉事实问答的提示词与严格答案解析。

生产 prompt 不要求模型输出思维链，只返回四个 YES/NO。``audit=True`` 是提示词
调试专用：它要求可核对的短视觉证据，不要求或保存隐藏推理过程，以便定位误判到底
来自道路拓扑、目标物还是交通灯识别。
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional


PROMPT_NAME = "sft_loop_phase1_static_obstacle_prompt"

ANSWER_KEYS = ("HIGHWAY", "STATIC_OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL")
ANSWER_VALUES = ("YES", "NO")


SYSTEM_PROMPT = """You are the perception first step of an autonomous-driving agent.
The input is a stitched three-camera RGB history, ordered from oldest to newest. Classify only the newest moment. Inspect the complete scene across left/front/right views: road topology, lane markings, shoulders, ramps, exits, merges, crossings, traffic lights, vehicles, pedestrians, cyclists, and objects that can affect the ego vehicle. Use older frames only to confirm motion, visibility, occlusion, or a changing signal. Do not use scenario names, dataset labels, maps, hidden metadata, or memory. Be conservative: a broad, straight, empty, open, foggy, tree-lined, or fast-looking road is not a highway unless limited-access topology is visible; another road user violating a normal signal is not a traffic-light fault."""


def build_phase1_prompt(*, audit: bool = False) -> str:
    """构造不含 memory 的第一轮问题。

    ``audit`` 仅用于人工改 prompt 时查看可见依据；训练和部署都应使用默认严格答案。
    """

    criteria = f"""
[PROMPT_NAME]
{PROMPT_NAME}
[/PROMPT_NAME]

[VISUAL_CHECK_ORDER]
Classify the newest frame, but use the short history to confirm motion and visibility. First scan left/front/right for road layout and lane topology. Then trace the ego vehicle's drivable corridor for the next few seconds. Make one dedicated near-to-horizon lane-closure pass over every history frame: follow the ego lane from nearby pavement to the vanishing point, then inspect the junction box, crosswalks, curb corners, sidewalks, shoulders, and the left/right camera edges. This pass is for small pedestrians, bicycles, vehicle noses, doors, cones, roadwork boards/trailers, and signal heads that are easy to miss in a broad scene summary. Then inspect traffic lights that govern ego, nearby vehicles, and vulnerable road users. Answer each question independently; one YES does not force another YES.
[/VISUAL_CHECK_ORDER]

[DECISION_RULES]
HIGHWAY:
Ask: "Can I SEE that the ego path is on a limited-access fast road or a ramp/connector, where ordinary street access and crossings are restricted right now?"
YES needs a visible topology chain, not one isolated clue. A ramp/merge/exit is sufficient but NOT required: a highway or expressway mainline can be YES when the current view shows a controlled high-speed corridor. Strong positive cues include:
- highway/expressway mainline, on-ramp, off-ramp, connector, merge, split, exit, acceleration lane, deceleration lane, gore area, lane divergence, or lane convergence;
- physically separated carriageways, median/barrier/guardrail separating directions, long controlled corridor, or several same-direction lanes with shoulders/edge barriers;
- absence of ordinary surface-street access along the ego path: no storefront/driveway access, no normal at-grade intersection controlling ego, no crosswalk governing ego, no pedestrian-facing street edge.
If the newest frame shows multiple same-direction lanes, continuous metal/concrete guardrails or barriers, no sidewalks/crosswalks/driveways/intersections, and an open controlled corridor, answer HIGHWAY=YES even if no ramp, merge, exit sign, or speed-limit sign is visible in this exact frame.
Open land, grass, trees, few buildings, and a far view can support YES only when the limited-access/ramp/merge chain is also visible. They are never enough alone.
NO traps from the reviewed RGB:
- R3-like or highway-looking geometry is still NO if it is a country/interurban surface road with trees/grass, a double-yellow opposing-traffic centreline, or ordinary two-way access;
- a wet/foggy straight road, mountain road with guardrails, broad city arterial, divided boulevard, urban overpass, bridge, underpass, tunnel-like/enclosed road, street lamps, or an empty wide multi-lane road is NO when there is no ramp/mainline/access-control evidence;
- buildings, sidewalks, crosswalks, storefronts, street parking, driveways, frequent junctions, or ordinary traffic-light intersections mean surface road unless a ramp/limited-access structure still visibly governs ego;
- after an exit, answer NO once ego has reached a normal surface street.
Never infer HIGHWAY from a dataset road label, event code, town, scenario name, speed, or a single guardrail.

STATIC_OBSTACLE:
Ask only: "Is a non-moving physical object currently occupying or closing the ego vehicle's usable corridor?" This is NOT the general hazard question. Do not answer YES for an object merely because it is nearby, looks dangerous, or might move later.
Trace the ego lane and intended turning/through corridor from the newest frame a short distance ahead. YES only when the object is stationary across the four-frame history, or clearly remains fixed relative to the road, AND its body, door, debris, cones, barrier, or work equipment occupies the lane, turning arc, or the only practical passable gap so ego must stop or steer around it now.
RGB-grounded YES patterns from the reviewed static-obstacle routes:
- a crashed, disabled, or parked car is stopped partly in the travel lane or diagonally across it; the lane line runs underneath/along its body, or there is no remaining ego-width passage without moving around it;
- a construction board, cone/barrier line, roadwork object, or work vehicle closes the forward lane, narrows it to an unusable gap, or forces ego around the work zone. A trailer-mounted temporary lane-arrow board with a yellow base, flashing lamps, an arrow panel, or cones around it is roadwork equipment: answer YES when it occupies the traced ego lane or diverts that lane. A small cone at the curb alone is not enough;
- a small but readable orange/yellow lane-closure board, barricade, cone cluster, or work-zone trailer remains in the same road position across the history and sits inside the traced ego lane. It is still YES when far ahead if its lane overlap or lane diversion is visible. Do not reject it merely because it is distant; reject it only when its path overlap cannot be seen;
- a curbside car has an open door protruding into the travel lane, or its fixed body extends out of an unmarked shoulder into ego's corridor;
- a stationary vehicle remains in the same road position in the history while ego approaches, and it blocks the only usable lane or the intended turn. It can be an ordinary-looking car: the required evidence is fixed occupation of the path, not a dramatic crash shape.
Before calling an orange/yellow object a moving vehicle, identify its visible structure. A mobile roadwork trailer/closure normally has a raised rectangular board or arrow panel above a compact base, often with a cone cluster, narrow support, lamps, or small trailer wheels; it is not an ordinary road-going car merely because its base is orange/yellow. Compare the object against lane markings, curb, road edge, and background. Ego approaching makes every stationary object shift or grow in the image; size growth, a changing pixel box, or apparent closing distance alone is NOT proof that it is moving toward ego. Call it dynamic only when its own road-relative position changes: for example it advances, crosses, turns, pulls out, changes lanes, or changes lateral alignment against lane markings/background independently of ego motion. A true static obstacle stays road-fixed while ego closes distance or drives around it. In fog, rain, glare, or night, one older frame may establish that the same fixed car, barrier, cone line, trailer-arrow board, or open door still blocks the corridor. Do not invent a static object when the history is unreadable.
NO traps from the reviewed RGB:
- a normal lead vehicle, stopped traffic queue, ambulance, or car waiting at a junction; even if ego must brake, it belongs to the later dynamic-obstacle question unless it is visibly disabled/parked and road-fixed. A vehicle that only looks still for this short history, but is advancing, turning, cutting in, pulling out, crossing, or changing road-relative position, is NO;
- a vehicle cutting in, pulling out, crossing, driving the wrong way, or running a red light; those are dynamic conflicts for the next loop, not STATIC_OBSTACLE;
- a car moving in its own lane, oncoming traffic in its proper lane, or traffic separated by a median/barrier;
- a curbside car fully inside a marked parking bay/shoulder, a background parked car that leaves the lane open, or a residual accident vehicle after ego has already passed and the corridor is clear;
- a distant object whose path overlap cannot be seen, an object only at the image edge with no path overlap, brake lights, a short gap, slow traffic, or a dark/foggy view where a fixed obstruction cannot be seen.
When uncertain between static and dynamic, answer STATIC_OBSTACLE=NO. The next loop is responsible for asking about moving vehicles and dynamic intrusions. Never infer this answer from a scenario name, event label, town, road type, or a previous route state.

VULNERABLE:
Look specifically for pedestrians, cyclists, scooter riders, wheelchair users, or other unprotected road users. YES only if one is in the ego path, crossing it, approaching the conflict zone, emerging from occlusion, or close enough to affect ego's current decision.
Before answering, deliberately inspect every crosswalk, curb corner, sidewalk edge, shoulder, bus-stop area, and left/right image edge in all four history frames. A person or bicycle may be small, partly hidden, or visible clearly in only one frame; one clear frame is enough for YES when that person can enter ego's forward or turning conflict area.
Use the history to check lateral movement and whether the person's/bicycle's path intersects ego's path. Crosswalk users, cyclists entering from a side street, and pedestrians close to a turning path can be YES.
Do not require the person or bicycle to already be inside the ego lane. A cyclist or pedestrian near the curb, sidewalk edge, side lane, crosswalk, or turning conflict zone is VULNERABLE=YES when ego may soon pass, turn, or merge near them, especially if they are oriented toward the roadway, standing near a crosswalk, walking beside the road, or partly entering from the side.
For the first loop question, treat active unprotected people/bicycles near the upcoming junction, sidewalk corner, shoulder, or curb as decision-relevant unless they are clearly far away or separated by a barrier. A standing pedestrian beside a crosswalk and a cyclist on the sidewalk edge near ego's forward/turning path are YES, not NO, even before they step into the lane.
NO for a distant sidewalk pedestrian, a person standing safely behind a barrier, a person far outside ego's conflict zone, a parked bicycle, a bicycle rack, a sign/billboard image, reflection, or a human-shaped object that is not an active road user.
Do not mark VULNERABLE just because there is a traffic light or another vehicle conflict; this question is only about unprotected road users.

TRAFFIC_LIGHT_ABNORMAL:
First identify the signal heads around the same junction and compare left/front/right views. Decide whether the signal system at the ego conflict point is self-consistent, not only whether the ego-facing lamp is green. Separate this junction from distant junctions and pedestrian-only signals.
Use a visual-witness test before deciding YES: locate one junction box and compare readable illuminated heads across left/front/right views. At a clear wide cross, T-junction, angled junction, or multi-arm junction, two GREEN heads visibly facing different approach arms are sufficient YES evidence when those arms feed the same broad conflict area. Do not require an exact lane-by-lane reconstruction or visible moving cars before recognizing this pattern. A red head plus a green head is normal by default. Two heads clearly serving the same approach, compatible turn lanes, or different/distant junctions are not a conflicting pair.
YES requires the signal/control system itself to be visibly unreliable, broken, or contradictory at the same conflict point. Positive patterns include:
- conflicting approaches or incompatible movements visibly permitted at the same time while vehicles are released through the shared conflict area;
- several signal heads on visibly different arms of the same cross junction show green at the same time; answer YES even if ego's own signal is green and appears bright/normal;
- if readable green heads on the ego/front approach and a left/right crossing or merging approach are visible in one broad conflict area, this is TRAFFIC_LIGHT_ABNORMAL=YES. This includes four-way crosses, T-junctions, angled junctions, and multi-arm junctions. Do not dismiss these as a "consistent green phase" merely because exact turn-lane geometry is small in the image.
- one clear witness frame is sufficient. If an older history frame clearly shows two crossing approaches green, answer YES even if the newest frame is partially occluded, foggy, or one signal head has moved out of view. Conversely, do not infer YES from a defect scenario when no history frame contains a readable contradictory signal witness.
- the ego-governing signal head is present but dark/off/broken when it should control the junction;
- impossible flashing, stuck, or inconsistent signal behavior across the short history;
- a clear all-green pattern across visibly different approach arms of one junction. Do not use all-red or mixed red/green alone: those can be normal phasing.
NO traps:
- ordinary red/yellow/green lights, normal phase changes, a normal red light with ego waiting, RedLightWithoutLeadVehicle behavior, non-signalized junctions, absent lights, unreadable tiny/distant lamps, fog/rain/glare, or different colors for non-conflicting lanes when that is normal phasing;
- another vehicle running a red light, taking priority, blocking an intersection, or crossing ego's path. That is a later dynamic-conflict question, but TRAFFIC_LIGHT_ABNORMAL=NO if the lights themselves look normal.
Several visible green heads are not sufficient only when they are clearly on one approach/same gantry, serve compatible lanes, or belong to different junctions. When a clear wide junction shows green heads facing distinct left/front/right approach arms, treat it as a visible fault witness; do not reject it merely because the exact lane mapping is small.
Do not infer a signal fault from a scenario name, event label, ego waiting, or the presence of a traffic light.
[/DECISION_RULES]
""".strip()

    if audit:
        output = """
[AUDIT_OUTPUT]
For each item, write one short, externally checkable visual observation (not hidden reasoning), then the answer. Mention only what is visible in the RGB history. For a small/brief object or signal, name the history-frame position where it is clearest. For TRAFFIC_LIGHT_ABNORMAL=YES, name the two visibly distinct approach arms with green heads or the broken head; do not invent exact lane details. For TRAFFIC_LIGHT_ABNORMAL=NO, state the main reason a visible cue is normal, such as red-versus-green normal phasing, same-arm lights, or no readable signal. If the answer is NO, name the main rejected false-positive cue when useful, such as "wide straight city road but no ramp/access control" or "vehicle violates signal but lamps look normal".
EVIDENCE_HIGHWAY: <short visible road-topology evidence>
EVIDENCE_STATIC_OBSTACLE: <short visible fixed object/path evidence>
EVIDENCE_VULNERABLE: <short visible vulnerable-road-user evidence>
EVIDENCE_TRAFFIC_LIGHT_ABNORMAL: <short visible signal evidence>
HIGHWAY: <YES or NO>
STATIC_OBSTACLE: <YES or NO>
VULNERABLE: <YES or NO>
TRAFFIC_LIGHT_ABNORMAL: <YES or NO>
[/AUDIT_OUTPUT]
""".strip()
    else:
        output = """
[OUTPUT]
Output exactly these four lines and nothing else:
HIGHWAY: <YES or NO>
STATIC_OBSTACLE: <YES or NO>
VULNERABLE: <YES or NO>
TRAFFIC_LIGHT_ABNORMAL: <YES or NO>
[/OUTPUT]
""".strip()
    return f"{criteria}\n\n{output}"


def phase1_prompt_sha256(*, audit: bool = False) -> str:
    """返回实际送入模型的 system + user prompt 内容指纹。"""

    payload = f"{SYSTEM_PROMPT}\n\0{build_phase1_prompt(audit=audit)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_ANSWER_RE = re.compile(r"(?im)^\s*([A-Z_]+)\s*:\s*(YES|NO)\b")


def parse_phase1_output(text: str) -> Dict[str, Optional[str]]:
    """解析四个二值答案；漏答、重复或非法值都返回 ``None``。"""

    found: Dict[str, Optional[str]] = {key: None for key in ANSWER_KEYS}
    duplicate = set()
    for key, value in _ANSWER_RE.findall(text or ""):
        key = key.upper()
        if key not in found:
            continue
        if found[key] is not None:
            duplicate.add(key)
        found[key] = value.upper()
    for key in duplicate:
        found[key] = None
    return found


def build_phase1_target(answers: Dict[str, bool]) -> str:
    """把人工复核后的四项布尔标签编码成训练目标。"""

    missing = [key for key in ANSWER_KEYS if key not in answers]
    if missing:
        raise ValueError(f"phase1 target missing answers: {missing}")
    return "\n".join(f"{key}: {'YES' if bool(answers[key]) else 'NO'}" for key in ANSWER_KEYS)
