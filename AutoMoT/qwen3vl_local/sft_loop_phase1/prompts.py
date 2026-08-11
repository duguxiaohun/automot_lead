"""第一轮四项视觉事实问答的提示词与严格答案解析。

生产 prompt 不要求模型输出思维链，只返回四个 YES/NO。``audit=True`` 是提示词
调试专用：它要求可核对的短视觉证据，不要求或保存隐藏推理过程，以便定位误判到底
来自道路拓扑、目标物还是交通灯识别。
"""

from __future__ import annotations

import re
from typing import Dict, Optional


PROMPT_NAME = "sft_loop_phase1_zero_shot_prompt"

ANSWER_KEYS = ("HIGHWAY", "OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL")
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
Classify the newest frame, but use the short history to confirm motion and visibility. First scan left/front/right for road layout and lane topology. Then trace the ego vehicle's drivable corridor for the next few seconds. Make one dedicated close-target pass over every history frame: the ego lane, the junction box, crosswalks, curb corners, sidewalks, shoulders, and the left/right camera edges. This pass is for small pedestrians, bicycles, vehicle noses, doors, cones, and signal heads that are easy to miss in a broad scene summary. Then inspect traffic lights that govern ego, nearby vehicles, and vulnerable road users. Answer each question independently; one YES does not force another YES.
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

OBSTACLE:
Trace the ego lane/path in the newest frame and the short distance ahead. YES only if a physical object occupies, enters, blocks, or sharply compresses that usable corridor so ego may need braking, yielding, stopping, or avoidance now.
YES examples: crashed vehicle, stalled/parked/construction vehicle in lane, construction object, open vehicle door protruding into lane, vehicle pulling out or cutting in, static car intruding from the side, oncoming vehicle invading ego lane, queue or blocked intersection physically preventing ego from clearing the lane/junction, or a vehicle violating right-of-way into ego's conflict zone.
For blocked-intersection/accident/construction scenes, do not require a dramatic crash shape. A stopped or very slow vehicle/queue directly in the ego lane or inside the intersection is OBSTACLE=YES when it makes ego brake, wait, or unable to clear the junction, even if the visible object looks like an ordinary car.
In fog/rain/night, use the short history, but do not promote an ordinary lead vehicle to YES from brake lights, a short-looking gap, or slow traffic alone. It is YES only when the history shows the ego lane/junction is actually closed or sharply constrained: for example a stopped queue prevents clearing the junction, a vehicle has stopped in the usable lane, or an object is visibly moving into that corridor.
Use a path-overlap test, not a dramatic-appearance test: a vehicle, ambulance, construction board, cone line, door, or queue is a YES when it crosses into or occupies the lane, junction box, turning arc, or only practical gap that ego must use. Do not treat a vehicle as an intrusion merely because it is beside the lane, parked in a curb/parking bay, travelling normally ahead in the same lane, or its front is visually near a lane boundary. A side vehicle can be YES before it fills the lane only when its body/nose is visibly crossing the lane boundary or its motion is entering ego's path. Inspect all four frames because a real intrusion may be clearest in only one of them.
Use older frames for partly occluded static obstacles only when they clearly show the same object still constraining ego's path.
NO examples: a normal lead vehicle, including one braking or travelling slowly in ordinary flow; a vehicle in its own/oncoming lane; traffic separated by a median/barrier; a distant object; safely parked roadside/background cars; residual accident vehicles after ego's path is open; or a queue visible far away but not blocking ego's usable corridor.
This question includes vehicle/path conflicts such as a red-light-running or oncoming vehicle; that is an obstacle/conflict, not a traffic-light fault.

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
Use a visual-witness test before deciding YES: locate one junction box, assign each readable illuminated head to an approach or movement, and find two GREEN heads that authorize vehicle paths which would cross in that same box. A red head plus a green head is normal by default. Two heads serving the same approach, different lanes with compatible turns, or different/distant junctions are not a conflicting pair. If the camera view does not let you map the green heads to conflicting approaches, there is no readable fault witness; do not infer a fault from green color alone.
YES requires the signal/control system itself to be visibly unreliable, broken, or contradictory at the same conflict point. Positive patterns include:
- conflicting approaches or incompatible movements visibly permitted at the same time while vehicles are released through the shared conflict area;
- several signal heads on different arms of the same cross junction show green at the same time for directions that would cross or collide; answer YES even if ego's own signal is green and appears bright/normal;
- if readable green heads on the ego approach and a crossing/merging approach both authorize paths through the same conflict area, this is TRAFFIC_LIGHT_ABNORMAL=YES. This includes four-way crosses, T-junctions, angled junctions, and multi-arm junctions. The required fact is conflicting permitted movements, not simply several green pixels.
- one clear witness frame is sufficient. If an older history frame clearly shows two crossing approaches green, answer YES even if the newest frame is partially occluded, foggy, or one signal head has moved out of view. Conversely, do not infer YES from a defect scenario when no history frame contains a readable contradictory signal witness.
- the ego-governing signal head is present but dark/off/broken when it should control the junction;
- impossible flashing, stuck, or inconsistent signal behavior across the short history;
- an all-red/all-green or mixed-color pattern only when the readable approach mapping proves that it authorizes a collision, not merely different phases for different approaches.
NO traps:
- ordinary red/yellow/green lights, normal phase changes, a normal red light with ego waiting, RedLightWithoutLeadVehicle behavior, non-signalized junctions, absent lights, unreadable tiny/distant lamps, fog/rain/glare, or different colors for non-conflicting lanes when that is normal phasing;
- another vehicle running a red light, taking priority, blocking an intersection, or crossing ego's path. That may be OBSTACLE=YES, but TRAFFIC_LIGHT_ABNORMAL=NO if the lights themselves look normal.
Several visible green heads are not sufficient by themselves: first verify that they govern crossing movements in the same junction. When that mapping is unreadable because of angle, distance, fog, glare, or occlusion, do not invent a fault.
Do not infer a signal fault from a scenario name, event label, ego waiting, or the presence of a traffic light.
[/DECISION_RULES]
""".strip()

    if audit:
        output = """
[AUDIT_OUTPUT]
For each item, write one short, externally checkable visual observation (not hidden reasoning), then the answer. Mention only what is visible in the RGB history. For a small/brief object or signal, name the history-frame position where it is clearest. For TRAFFIC_LIGHT_ABNORMAL=YES, name the two visibly conflicting green approaches or the broken head. For TRAFFIC_LIGHT_ABNORMAL=NO, state that no readable conflicting same-junction witness pair was seen, rather than merely calling the phase normal. If the answer is NO, name the main rejected false-positive cue when useful, such as "wide straight city road but no ramp/access control" or "vehicle violates signal but lamps look normal".
EVIDENCE_HIGHWAY: <short visible road-topology evidence>
EVIDENCE_OBSTACLE: <short visible object/path evidence>
EVIDENCE_VULNERABLE: <short visible vulnerable-road-user evidence>
EVIDENCE_TRAFFIC_LIGHT_ABNORMAL: <short visible signal evidence>
HIGHWAY: <YES or NO>
OBSTACLE: <YES or NO>
VULNERABLE: <YES or NO>
TRAFFIC_LIGHT_ABNORMAL: <YES or NO>
[/AUDIT_OUTPUT]
""".strip()
    else:
        output = """
[OUTPUT]
Output exactly these four lines and nothing else:
HIGHWAY: <YES or NO>
OBSTACLE: <YES or NO>
VULNERABLE: <YES or NO>
TRAFFIC_LIGHT_ABNORMAL: <YES or NO>
[/OUTPUT]
""".strip()
    return f"{criteria}\n\n{output}"


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
