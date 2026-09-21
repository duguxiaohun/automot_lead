"""Action 的单当前图条件；SFT adapter 原始训练提示词和元数据保持不变。"""
import re

IMAGE_CONDITION_VERSION = "action_current_or_four_rgb_v1"
CURRENT_IMAGE_NOTICE = (
    "The input is one current stitched left/front/right RGB image. "
    "No earlier images are provided. Use only currently visible evidence; "
    "do not infer observed motion or changes from an unavailable image sequence."
)


def image_prompt(text, count):
    """最终规划/摘要的视觉说明随实际图片数变化。"""
    if count == 4:
        return text
    if count != 1:
        raise ValueError("RGB frame count must be 1 or 4")
    text = text.replace("chronological images", "the current image")
    text = text.replace("use visible history", "use the current image and supplied context")
    text = text.replace("using visible history", "using the current image and supplied context")
    return CURRENT_IMAGE_NOTICE + "\n" + text


_CURRENT_FACT_RULES = """HIGHWAY:
YES only for visible limited-access highway/ramp topology. Width, speed, an empty road, barriers alone, fog, or a straight corridor do not establish highway status.
STATIC_OBSTACLE:
YES only for a visibly fixed obstruction occupying ego's usable lane, turning arc, or necessary passing gap: barriers, cones, debris, work equipment, an open door, or a clearly parked/disabled obstructing vehicle. A normal lead vehicle, queue, junction wait, or an apparently still vehicle is insufficient. Do not infer stationarity from one image when static/dynamic status is ambiguous.
VULNERABLE:
Inspect all three views, crosswalks, curb corners, sidewalks, shoulders and image edges for a visible pedestrian or cyclist affecting the corridor or its immediate approach. A crosswalk or setting alone is insufficient; do not invent an occluded actor.
TRAFFIC_LIGHT_ABNORMAL:
YES requires readable abnormal signal hardware governing the local junction. Normal red lights, unreadable lamps, streetlights, and another actor violating a normal signal do not establish a signal fault. A flashing pattern cannot be established from one image.
Judge every question independently from its own evidence. Missing visible support is not a positive fact."""

_CURRENT_EVENTS = {
    "UE1": "YES only for a same-lane lead vehicle with clear currently visible braking evidence affecting the immediate following path. Brake lights need a relevant lead/following relation; proximity, an ordinary queue, and an apparently stationary vehicle alone do not prove sudden deceleration. Do not invent a speed change.",
    "UE3": "YES only for visible entry or encroachment into ego's immediate corridor by a side, adjacent-lane, or pulling-out vehicle, supported by its body/nose and lane-boundary relation. A parked angle, ordinary parallel traffic, a construction/crash actor or ego's planned lane change alone is insufficient. Do not invent lateral displacement.",
    "UE5": "YES only for an opposite-facing vehicle visibly intruding into ego's lane or usable corridor and causing an immediate yielding conflict. Ordinary oncoming traffic on its own side, narrow-road sharing and ego borrowing the opposite lane do not establish invasion.",
    "UE6": "YES only at a visible local junction with both a vehicle occupying ego's conflict path and visible signal/priority evidence showing a violation while ego has priority. A crossing vehicle alone, normal yielding or unknown right of way is insufficient.",
}


def current_prior_prompt(module, phase, spec):
    """复用问题/输出/道路几何合同，只替换需要多帧证据的指令。"""
    builder = module.build_phase1_prompt if phase == 1 else module.build_event_prompt
    text = builder(spec=spec, history_rgb_mode="4rgb")
    text = text.replace(module.PROMPT_NAME, module.PROMPT_NAME + "_action_current_rgb_v1")
    def block(name, value):
        nonlocal text
        pattern = rf"\[{name}\].*?\[/{name}\]"
        text, count = re.subn(pattern, lambda _: f"[{name}]\n{value}\n[/{name}]", text, flags=re.S)
        if count != 1:
            raise ValueError(f"unsupported Phase{phase} single-image prompt section: {name}")
    block("VISUAL_CHECK_ORDER", CURRENT_IMAGE_NOTICE + " Inspect the full left/front/right scene and answer the listed questions independently.")
    if phase == 1:
        block("PHASE1_VISIBLE_FACT_RULES", _CURRENT_FACT_RULES)
    else:
        block("OBSERVABILITY_AND_TIMING", "Judge whether the actor and conflict are visibly supported now. Insufficient motion or priority evidence is not YES. Poor visibility alone does not make the road layout invalid.")
        for key, definition in module.EVENT_DEFINITIONS.items():
            if key in _CURRENT_EVENTS:
                text = text.replace(definition, key + ":\n" + _CURRENT_EVENTS[key])
        text = text.replace("visible progressive lane-divider crossing", "visible lane-divider encroachment")
    system = (
        "You answer driving scene questions from visible RGB evidence. " + CURRENT_IMAGE_NOTICE
        + " Do not use scenario names, hidden metadata, maps, or future frames. Follow the requested output format."
    )
    return system, text


def current_base_prefill(runner, images, user_prompt, system_prompt):
    """单图 simple/base 路径直接 prefill；不加载新模型，不复制当前图冒充历史。"""
    if len(images) != 1:
        raise ValueError("single-image Qwen prefill requires exactly the current image")
    engine = runner.leadmot_qwen_engine
    engine._last_decode_state = None
    engine._system_prompt_cache = None
    messages = engine.build_messages(image_prompt(system_prompt, 1), user_prompt, images)
    chat = engine.apply_chat_template(messages)
    inputs = engine.prepare_inputs(chat, images)
    output = engine.prefill(inputs)
    if output.rope_deltas is None:
        raise RuntimeError("missing current-image Qwen M-RoPE delta")
    offset = int(inputs["input_ids"].shape[-1]) + int(output.rope_deltas.reshape(-1)[0].item())
    return output.past_key_values, offset


def image_contract(args):
    """同时绑定实际图数与先验问答适配策略，不篡改 adapter 的训练元数据。"""
    count = args.rgb_frame_count
    if count not in (1, 4):
        raise ValueError("RGB frame count must be 1 or 4")
    return dict(version=IMAGE_CONDITION_VERSION, frame_count=count, frame_step=args.rgb_frame_step,
                selection="current_anchor" if count == 1 else "chronological_four_ending_at_anchor",
                stitched_views="left_front_right", bev="current_rgb_lidar_unchanged",
                prior_prompt_policy="current_only_adapted" if count == 1 else "saved_adapter_history")
