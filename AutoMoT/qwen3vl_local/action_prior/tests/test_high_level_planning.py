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
    assert "If progress is constrained" in text
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
