"""主要动作优先级共用；Phase3 KEEP 与 action 的旧 NONE 协议明确区分。"""

from collections import Counter
import itertools

import pytest

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    make_prompt_spec, build_action_target, build_action_prompt, parse_action_output, spec_answers,
)
from qwen3vl_local.sft_new_loop_phase3.primary_action import primary_action, count_none_prediction
from qwen3vl_local.action_prior.action_input import select_primary, choice_action_input, gate_action


@pytest.mark.parametrize("context_id", CONTEXT_BY_ID)
@pytest.mark.parametrize("speed,lane", itertools.product(
    (None, "DECELERATE", "STOP", "RESUME"), (None, "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT")))
def test_training_target_parser_and_action_condition_share_projection(context_id, speed, lane):
    """每种合法纵横组合按域投影，正动作单选监督与下游一致，保持的两套协议不得静默混用。"""
    context = CONTEXT_BY_ID[context_id]
    answers = {key: key in (speed, lane) for key in ACTION_KEYS}
    spec = make_prompt_spec(variant="all_random_order", answers=answers, seed_key="fixed",
        context_id=context_id, road_structure=context.allowed_rs[0], action_output_mode="choice")
    target = build_action_target(spec)
    raw = [key for key in context.action_keys if answers[key]]
    projected = select_primary(dict(status="selected" if raw else "no_action", actions=raw))
    if target == "KEEP":
        assert projected["status"] == "no_action"
        assert choice_action_input(target)["status"] == "unavailable"  # 旧下游协议不静默混用。
    else:
        assert choice_action_input(target) == projected
    assert parse_action_output(target, spec=spec) == spec_answers(spec)
    assert sum(q.answer for q in spec.questions) <= 1
    assert answers == {key: key in (speed, lane) for key in ACTION_KEYS}
    empty = make_prompt_spec(variant="all_random_order", answers={key: False for key in ACTION_KEYS}, seed_key="fixed",
        context_id=context_id, road_structure=context.allowed_rs[0], action_output_mode="choice")
    assert build_action_prompt(spec=spec) == build_action_prompt(spec=empty)


def test_wait_is_not_none_and_crossing_does_not_override_stop():
    """已确认等待、配合跨线、空证据三者不能混为一类。"""
    assert primary_action({"STOP": True, "LANE_CHANGE_LEFT": True}) == "STOP"
    assert primary_action({"DECELERATE": True, "LANE_CHANGE_LEFT": True}) == "LANE_CHANGE_LEFT"
    assert primary_action({}) == "NONE"
    for output in ("NONE extra", "STOP\nNONE", "INVALID_ACTION_CONTEXT", "", None):
        assert choice_action_input(output)["status"] == "unavailable"


def test_primary_selection_runs_after_upstream_domain_gate():
    """未确认障碍物不能使 LEFT 抢掉已确认前车事件的减速。"""
    evidence = dict(status="selected", actions=["DECELERATE", "LANE_CHANGE_LEFT"])
    conditions = {"ROAD_STRUCTURE": "R1", "UE1": "YES", "STATIC_OBSTACLE": "NO",
                  "ROAD_CORRIDOR/INVALID_EVENT_CONTEXT": "NO"}
    assert gate_action(evidence, conditions, ("UE1", "UE2"))[0]["actions"] == ["DECELERATE"]
    conditions["STATIC_OBSTACLE"] = "YES"
    effective, audit = gate_action(evidence, conditions, ("UE1", "UE2"))
    assert effective["actions"] == ["LANE_CHANGE_LEFT"]
    assert audit["eligible_actions"] == evidence["actions"]


def test_none_metrics_count_false_positives_and_malformed_predictions():
    """格式错不能算 NONE；NONE 的 precision/recall 同时有分母。"""
    counts = Counter()
    for gt, predicted in ((True, True), (True, False), (False, True)):
        count_none_prediction(counts, gt_none=gt, predicted_none=predicted)
    assert counts == {"NONE/gt_yes": 2, "NONE/pred_yes": 2,
                      "NONE/recall_hit": 1, "NONE/precision_hit": 1}
