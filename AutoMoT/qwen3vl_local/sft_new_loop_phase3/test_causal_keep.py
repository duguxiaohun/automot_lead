"""v15 保持阶段、未知隔离、场景与动作因果分离的无模型回归。"""
from collections import Counter
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3.choice_semantics import (
    choice_annotation, primary_choice, count_keep_prediction, CONTEXT_ACTION_DESCRIPTIONS, validate_choice_row,
)
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    make_prompt_spec, build_action_prompt, build_action_target, parse_action_output,
)
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import longitudinal_decision
from qwen3vl_local.sft_new_loop_phase3.quality_guards import choice_generation_guards


def evidence(speeds=None, direction=""):
    return dict(longitudinal_decision=longitudinal_decision(speeds or [8.0]*9),
                lateral_observation_complete=True, lane_change_direction=direction)


def test_keep_is_positive_continuation_with_domain_scope_not_unknown():
    labels = dict.fromkeys(ACTION_KEYS, False)
    assert choice_annotation(labels, "STATIC_BLOCKAGE", evidence())["keep_scope"] == "lane_and_speed"
    # 真实 UE5 #513 的关键边界：题域只问速度，不能从 KEEP 推断没有变道。
    labels["LANE_CHANGE_LEFT"] = True
    assert choice_annotation(labels, "ONCOMING_INVASION", evidence(direction="LEFT"))["keep_scope"] == "speed_only"
    assert choice_annotation(labels, "STATIC_BLOCKAGE", evidence(direction="LEFT"))["primary_action"] == "LANE_CHANGE_LEFT"


@pytest.mark.parametrize("bad", [{}, {"STOP": None}, {"STOP": "NO"}, {"STOP": 0}, {"INVALID_ACTION_CONTEXT": True}])
def test_missing_or_invalid_answers_never_become_keep(bad):
    labels = {} if not bad else {**dict.fromkeys(ACTION_KEYS, False), **bad}
    with pytest.raises(ValueError):
        primary_choice(labels)


def test_incomplete_ambiguous_or_inconsistent_trajectory_cannot_supply_keep():
    labels = dict.fromkeys(ACTION_KEYS, False)
    for speed in ([8]*8, [8, 10, 10, 6, 6, 6, 6, 6, 6]):
        with pytest.raises(ValueError, match="eligible"):
            choice_annotation(labels, "STATIC_BLOCKAGE", evidence(speed))
    missing = evidence(); missing["lateral_observation_complete"] = False
    with pytest.raises(ValueError, match="lateral"):
        choice_annotation(labels, "STATIC_BLOCKAGE", missing)
    assert choice_annotation(labels, "LEAD_BRAKE", missing)["primary_action"] == "KEEP"
    with pytest.raises(ValueError, match="longitudinal"):
        choice_annotation(labels, "STATIC_BLOCKAGE", evidence([0]*9))
    with pytest.raises(ValueError, match="lateral"):
        choice_annotation(labels, "STATIC_BLOCKAGE", evidence(direction="LEFT"))


def test_wait_release_and_small_adjustments_are_distinct():
    labels = dict.fromkeys(ACTION_KEYS, False)
    assert choice_annotation(labels, "STATIC_BLOCKAGE", evidence([8,8.2,8.8,8.1,7.9,8.1,8.3,8,8]))["primary_action"] == "KEEP"
    labels["STOP"] = True
    assert choice_annotation(labels, "STATIC_BLOCKAGE", evidence([0,0,0,1,2,3,4,5,6]))["primary_action"] == "STOP"
    labels["STOP"] = False; labels["RESUME"] = True
    assert choice_annotation(labels, "STATIC_BLOCKAGE", evidence([0,.6,1.7,3,4,4,4,4,4]))["primary_action"] == "RESUME"


@pytest.mark.parametrize("context_id", CONTEXT_BY_ID)
def test_all_contexts_render_causal_actions_and_positive_keep_without_answer_leak(context_id):
    context = CONTEXT_BY_ID[context_id]
    assert set(CONTEXT_ACTION_DESCRIPTIONS[context_id]) == {*context.action_keys, "KEEP"}
    labels = dict.fromkeys(ACTION_KEYS, False)
    spec = make_prompt_spec(variant="all_random_order", answers=labels, seed_key="v15",
        context_id=context_id, road_structure=context.allowed_rs[0], action_output_mode="choice")
    text = build_action_prompt(spec=spec)
    assert build_action_target(spec) == "KEEP"
    assert all(v is None for v in parse_action_output("NONE", spec=spec).values())
    assert len(text.split()) <= 650
    scene = text.split("[SCENE_CONTEXT]",1)[1].split("[/SCENE_CONTEXT]",1)[0]
    for action in context.action_keys:
        labels[action] = True
        other = make_prompt_spec(variant="all_random_order", answers=labels, seed_key="v15",
            context_id=context_id, road_structure=context.allowed_rs[0], action_output_mode="choice")
        assert build_action_prompt(spec=other) == text
        assert CONTEXT_ACTION_DESCRIPTIONS[context_id][action] not in scene
        labels[action] = False


def test_keep_metrics_and_guard_require_real_support_precision_and_recall():
    counts = Counter()
    for truth, pred in ((True,True),(True,False),(False,True)):
        count_keep_prediction(counts, gt_keep=truth, predicted_keep=pred)
    assert counts == {"KEEP/gt_yes":2,"KEEP/pred_yes":2,"KEEP/recall_hit":1,"KEEP/precision_hit":1}
    metrics = dict(format_valid_rate=1, exact_accuracy=1)
    for action in (*ACTION_KEYS, "KEEP"):
        metrics.update({f"action/{action.lower()}_{metric}":1 for metric in ("gt_yes","precision","recall")})
    assert choice_generation_guards(metrics,min_format_valid_rate=.99)["all_ok"]
    for key in ("gt_yes","precision","recall"):
        bad = deepcopy(metrics); bad[f"action/keep_{key}"]=0
        assert not choice_generation_guards(bad,min_format_valid_rate=.99)["all_ok"]


def test_rereview_covers_all_contexts_and_keeps_input_future_boundary():
    notes = [json.loads(line) for line in Path(__file__).with_name("causal_action_rgb_notes_20260920.jsonl").read_text().splitlines()]
    assert len(notes) == 26 and {n["context_id"] for n in notes} == set(CONTEXT_BY_ID)
    for note in notes:
        f = note["frame_id"]
        assert note["input_frames"] == list(range(f-3,f+1))
        assert note["reviewed_frames"] == list(range(f-3,f+14))
        assert note["future_frames_role"] == "offline_audit_only"
        assert note["primary_action"] == primary_choice(note["raw_action_labels"], CONTEXT_BY_ID[note["context_id"]].action_keys)


def test_review_display_separates_keep_unknown_and_legacy_none():
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import PRIMARY_CHOICE_VERSION
    from qwen3vl_local.sft_new_loop_phase3.prepare_error_review import display_action
    row = dict(prompt_spec=dict(action_output_mode="choice", primary_action_version=PRIMARY_CHOICE_VERSION))
    answers = dict.fromkeys((*ACTION_KEYS, "INVALID_ACTION_CONTEXT"), "NO")
    assert display_action(row, answers) == "KEEP"
    assert display_action({}, answers) == "NONE"
    assert display_action(row, {**answers, "STOP": None}) == "UNPARSED"
    assert display_action(row, {**answers, "STOP": "YES"}) == "STOP"


def index_row():
    from qwen3vl_local.sft_new_loop_phase3 import DATASET_NAME
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
    labels = {**dict.fromkeys(ACTION_KEYS, False), "KEEP": True, "INVALID_ACTION_CONTEXT": False}
    raw = {**evidence(), "rule_version": ACTION_RULE_VERSION, "rule_code_sha256": action_rule_sha256()}
    return dict(dataset_name=DATASET_NAME, split="train", scenario="synthetic", route_id="r", frame_id=0,
        context_id="STATIC_BLOCKAGE", invalid_action_context=False, answers=labels, action_evidence=raw,
        mapping_contract_hash=mapping_contract_hash(), current_speed_mps=8.0, goal_ego_xy=[10,0],
        **choice_annotation(labels, "STATIC_BLOCKAGE", raw))


@pytest.mark.parametrize("change", ["missing_version", "unknown", "string", "scope", "invalid"])
def test_index_validation_rejects_stale_ambiguous_and_inconsistent_rows(change):
    row = index_row()
    validate_choice_row(row)
    if change == "missing_version":
        row.pop("primary_action_version")
    elif change == "unknown":
        row["answers"]["STOP"] = None
    elif change == "string":
        row["answers"]["STOP"] = "NO"
    elif change == "scope":
        row["keep_scope"] = "speed_only"
    else:
        row["invalid_action_context"] = True
    with pytest.raises(ValueError):
        validate_choice_row(row)


@pytest.mark.parametrize("module_name", ["train", "eval"])
def test_actual_index_readers_validate_before_boolean_conversion(tmp_path, module_name):
    """执行真实读取函数，抽离无关模型初始化，使 CPU 能查出 bool(None)/bool('NO') 漏洞。"""
    from qwen3vl_local.sft_new_loop_phase3 import DATASET_NAME
    path = Path(__file__).with_name(module_name + ".py")
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_read_rows")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(pathlib=__import__("pathlib"), json=json, math=math, DATASET_NAME=DATASET_NAME,
        _AUTOMOT_ROOT=path.parents[2], FrameRow=SimpleNamespace, _resolve_rgb_path=lambda value, root:value)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), namespace)
    index = tmp_path/"index.jsonl"
    row = index_row(); index.write_text(json.dumps(row)+"\n")
    loaded = namespace["_read_rows"](index, "train")
    assert loaded[0].answers == row["answers"]
    for value in (None, "NO", 0):
        broken = deepcopy(row); broken["answers"]["STOP"] = value
        index.write_text(json.dumps(broken)+"\n")
        with pytest.raises(ValueError, match="boolean"):
            namespace["_read_rows"](index, "train")
    row.pop("primary_action_version"); index.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="version"):
        namespace["_read_rows"](index, "train")


@pytest.mark.parametrize("mode", ["binary", "choice"])
def test_preflight_rejects_legacy_rows_without_manifest(tmp_path, mode):
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    row = index_row(); row.pop("primary_action_version")
    path = tmp_path/"index.jsonl"; path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="version"):
        check_index(path, mode)
