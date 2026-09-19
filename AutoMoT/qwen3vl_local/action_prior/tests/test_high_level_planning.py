"""高层规划保持事实门控、Phase3 动作域和默认提示词边界。"""

import itertools

import pytest

from qwen3vl_local.action_prior import prompts
from qwen3vl_local.action_prior.config import parser, validate_args


@pytest.mark.parametrize("event,maneuver", [
    ("UE1", False), ("UE3", False), ("UE5", False), ("UE6", False),
    ("TRAFFIC_LIGHT_ABNORMAL", False), ("STATIC_OBSTACLE", True), ("VULNERABLE", True),
])
def test_confirmed_event_replaces_generic_planning_with_conditional_actions(event, maneuver):
    """纵向三动作语义通用，五动作域才补横向；无真值、类别或旧规划重复。"""
    conditions = {"ROAD_STRUCTURE": "R1", event: "YES"}
    prior = dict(conditions=conditions, high_level_planning=True)
    text = prompts.prefill_prompt(prior, "current velocity is 4 m/s")
    assert prompts.EVENT_COMPACT_NAMES[event] in text
    assert prompts.EVENT_DESCRIPTIONS[event] not in text
    assert "If constrained" in text
    assert "stop/continue waiting" in text and "without requiring a previous stop" in text
    assert ("first future lane-boundary crossing" in text) is maneuver
    assert prompts.scene_description(conditions, high_level=True) in prompts.review_prompt(prior, "nav", "draft")
    for token in (event, "YES", "UNKNOWN", "DECELERATE", "LANE_CHANGE_LEFT"):
        assert token not in text


def test_negative_unknown_and_r3_do_not_invent_actions_or_transitions():
    """缺失事实不能变成高速、常规事件或已经绕障的历史。"""
    conditions = dict(ROAD_STRUCTURE="R3", UE1="NO", STATIC_OBSTACLE=None)
    text = prompts.scene_description(conditions, high_level=True)
    for word in ("highway", "blockage", "recovery", "stop/continue waiting", "cross left"):
        assert word not in text
    assert "limited-access highway" in prompts.scene_description(dict(conditions, HIGHWAY="YES"), high_level=True)


@pytest.mark.parametrize("context,maneuver", [
    ("UE1", False), ("UE2", True), ("UE4", True), ("RE3", True), ("RE5", False),
    ("RE2_NAVIGATION_TRANSITION", True), ("RE2_PRIOR_OBSTACLE", True),
])
def test_explicit_contexts_keep_domain_and_history_boundaries(context, maneuver):
    """仅显式离线上下文能补特殊常规域，未知绕障阶段不写成完成恢复。"""
    text = prompts.scene_description({}, [context], high_level=True)
    assert ("cross left or right" in text) is maneuver
    assert context not in text
    assert "has been passed" not in text
    if context == "RE2_NAVIGATION_TRANSITION":
        assert "blockage" not in text
    if context == "RE2_PRIOR_OBSTACLE":
        assert "pending recovery are unconfirmed" in text


def test_all_event_combinations_preserve_facts_and_fallback_budget():
    """并发事件不丢失；fallback 只重述规划，事实仍完整留在 user。"""
    keys = tuple(prompts.EVENT_DESCRIPTIONS)
    for values in itertools.product(("YES", "NO"), repeat=len(keys)):
        conditions = dict(zip(keys, values), ROAD_STRUCTURE="R2", HIGHWAY="YES")
        prior = dict(conditions=conditions, high_level_planning=True)
        text = prompts.prefill_prompt(prior, "navigation")
        for key, value in zip(keys, values):
            assert (prompts.EVENT_COMPACT_NAMES[key] in text) is (value == "YES")
        assert prompts.analysis_format_valid(prompts.fallback_analysis(prior, "current velocity is 4 m/s"))


def test_default_and_legacy_config_keep_original_prompt():
    """新旧配置默认均关闭替换；无先验消融不能宣称消费该开关。"""
    args = parser().parse_args([])
    assert args.high_level_planning is False and args.generate_analysis is False
    del args.high_level_planning
    validate_args(args)
    assert args.high_level_planning is False
    prior = dict(conditions={"ROAD_STRUCTURE": "R1", "UE1": "YES"})
    assert prompts.prefill_prompt(prior, "nav") == prompts.prefill_prompt(dict(prior, high_level_planning=False), "nav")
    args.high_level_planning, args.condition_mode = True, "base"
    with pytest.raises(ValueError, match="condition-mode prior"):
        validate_args(args)


@pytest.mark.parametrize("event", list(prompts.HIGH_LEVEL_EVENT_PURPOSES))
@pytest.mark.parametrize("action", ["DECELERATE", "STOP"])
def test_shared_actions_keep_scene_specific_purposes_across_prompt_paths(event, action):
    """同一减速/停车释义不变，prefill/摘要/复核共用各自场景目的。"""
    from qwen3vl_local.action_prior.action_input import action_sentence
    prior = dict(conditions={event: "YES"}, high_level_planning=True,
                 high_level_action_prior=True,
                 high_level_action=dict(status="selected", actions=[action]))
    purpose = prompts.HIGH_LEVEL_EVENT_PURPOSES[event]
    sentence = action_sentence(prior["high_level_action"])
    for text in (prompts.prefill_prompt(prior, "nav"), prompts.analysis_prompt(prior, "nav"),
                 prompts.review_prompt(prior, "nav", "draft")):
        assert text.count(purpose) == 1
        assert sentence in text
        assert "Purpose implies neither an available gap nor additional actions" in text
        assert "does not imply a later lane change" in text
        assert all(value not in text for key, value in prompts.HIGH_LEVEL_EVENT_PURPOSES.items() if key != event)
    assert prompts.fallback_analysis(prior) == sentence
    assert prompts.analysis_format_valid(prompts.fallback_analysis(prior))


def test_purpose_ignores_raw_action_contexts_and_preserves_unselected_prompt():
    """未确认 UE/无独立 RE gate 时，动作 scope 不能额外提供目的或场景事实。"""
    from qwen3vl_local.action_prior.action_input import gate_action
    prior = dict(conditions={"ROAD_STRUCTURE": "R1", "STATIC_OBSTACLE": "NO"}, high_level_planning=True)
    baseline = prompts.prefill_prompt(prior, "nav")
    for context in ("UE2", "RE2_PRIOR_OBSTACLE", "RE2_NAVIGATION_TRANSITION"):
        effective, audit = gate_action(dict(status="selected", actions=["STOP"]), prior["conditions"], (context,))
        supplied = dict(prior, high_level_action_prior=True, high_level_action=effective,
                        high_level_action_contexts=(context,), high_level_action_gate=audit)
        assert prompts.prefill_prompt(supplied, "nav") == baseline
        assert "Scene-specific purposes" not in baseline
    confirmed = dict(prior, conditions={"ROAD_STRUCTURE": "R1", "STATIC_OBSTACLE": "YES"})
    for status in ("no_action", "unavailable", "not_applicable"):
        supplied = dict(confirmed, high_level_action_prior=True, high_level_action=dict(status=status, actions=[]))
        assert prompts.prefill_prompt(supplied, "nav") == prompts.prefill_prompt(confirmed, "nav")


def test_static_obstacle_purpose_uses_observation_not_a_maneuver_sequence():
    """绕障目的具体到相邻车道/接近车辆/借道对向车，不能从减速直接指定跨线。"""
    conditions = {"STATIC_OBSTACLE": "YES"}
    text = prompts.high_level_purposes(conditions, ("UE2", "UE2"))
    assert text.count(prompts.HIGH_LEVEL_EVENT_PURPOSES["STATIC_OBSTACLE"]) == 1
    for cue in ("adjacent-lane traffic", "approaching vehicles", "oncoming traffic", "passing clearance"):
        assert cue in text
    assert "does not imply a later lane change" in text
    assert not prompts.high_level_purposes({"STATIC_OBSTACLE": None})
    assert "Scene-specific purposes" not in prompts.prefill_prompt(dict(conditions=conditions), "nav")


@pytest.mark.parametrize("context", list(prompts.HIGH_LEVEL_TRANSITION_PURPOSES))
def test_transition_purpose_requires_explicit_scene_context(context):
    """RE 目的只来自独立 scene context，并保留导航/先前障碍/确认恢复的边界。"""
    prior = dict(conditions={"ROAD_STRUCTURE": "R3"}, high_level_planning=True)
    purpose = prompts.HIGH_LEVEL_TRANSITION_PURPOSES[context]
    assert purpose not in prompts.prefill_prompt(prior, "nav")
    text = prompts.prefill_prompt(dict(prior, event_balanced_scene_contexts=(context,)), "nav")
    assert text.count(purpose) == 1
    if context == "RE2_NAVIGATION_TRANSITION":
        assert "blockage" not in text and "recovery" not in text
    elif context == "RE2_PRIOR_OBSTACLE":
        assert "If visible history supports" in text and "has been passed" not in text
