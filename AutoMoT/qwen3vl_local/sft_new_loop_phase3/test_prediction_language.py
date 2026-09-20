"""v16：全部上下文、输出和图像模式区分历史与未来，不暴露离线数值判定器。"""
from dataclasses import replace

import pytest

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID, DOMAIN_MANEUVER
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import CONTEXT_ACTION_DESCRIPTIONS
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    make_prompt_spec, build_action_prompt, build_action_target, parse_action_output, spec_answers,
    action_prompt_sha256,
)


@pytest.mark.parametrize("context_id", CONTEXT_BY_ID)
@pytest.mark.parametrize("rgb", ["4rgb", "2rgb_endpoints"])
@pytest.mark.parametrize("output", ["choice", "binary"])
def test_prediction_instruction_across_all_contexts(context_id, rgb, output):
    context = CONTEXT_BY_ID[context_id]
    spec = make_prompt_spec(variant="all_random_order", answers=dict.fromkeys(ACTION_KEYS, False),
        seed_key="time-boundary", context_id=context_id, road_structure=context.allowed_rs[0],
        current_speed_mps=6, action_output_mode=output)
    prompt = build_action_prompt(spec=spec, history_rgb_mode=rgb)
    assert "judge ego's upcoming driving behavior" in prompt
    assert "repeat an action already completed" in prompt
    assert "current waiting still counts even if ego moves off later" in prompt
    assert "first meaningful speed change" in prompt and "sustained speed increase" in prompt
    for forbidden in ("seconds", "second window", "prediction window", "4-Hz", "0.5 m/s", "1.2 m/s", "20%", "two consecutive", "qualifying", "threshold", "max("):
        assert forbidden not in prompt
    assert ("t-0.50" in prompt) == (rgb == "4rgb")
    scene = prompt.split("[SCENE_CONTEXT]",1)[1].split("[/SCENE_CONTEXT]",1)[0]
    for action in context.action_keys:
        causal = CONTEXT_ACTION_DESCRIPTIONS[context_id][action]
        assert prompt.count(causal) == 1 and causal not in scene
    if output == "choice":
        assert build_action_target(spec) == "KEEP"
        if context.question_domain != DOMAIN_MANEUVER:
            assert "makes no claim about lane changes" in prompt
    changed = replace(spec, questions=tuple(replace(q, answer=not q.answer) for q in spec.questions))
    assert build_action_prompt(spec=changed, history_rgb_mode=rgb) == prompt
    assert parse_action_output(build_action_target(spec), spec=spec) == spec_answers(spec)


@pytest.mark.parametrize("rgb", ["4rgb", "2rgb_endpoints"])
@pytest.mark.parametrize("output", ["choice", "binary"])
def test_time_instruction_is_bound_to_prompt_hash(monkeypatch, rgb, output):
    from qwen3vl_local.sft_new_loop_phase3 import prompts
    before = action_prompt_sha256(history_rgb_mode=rgb, action_output_mode=output)
    monkeypatch.setattr(prompts, "OBSERVATION_RULES", prompts.OBSERVATION_RULES + " Changed boundary.")
    assert before != action_prompt_sha256(history_rgb_mode=rgb, action_output_mode=output)
