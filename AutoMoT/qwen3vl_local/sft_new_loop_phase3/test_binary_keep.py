"""判断题 KEEP 与选择题、索引、真实计数分支和错误审计的一致性。"""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import binary_answers, count_keep_prediction
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    build_action_prompt, build_action_target, make_prompt_spec, parse_action_output, spec_answers,
)


def spec_for(context_id, labels, mode="binary"):
    return make_prompt_spec(variant="all_random_order", answers=labels, seed_key="binary-keep",
        context_id=context_id, road_structure=CONTEXT_BY_ID[context_id].allowed_rs[0],
        action_output_mode=mode)


@pytest.mark.parametrize("context_id", CONTEXT_BY_ID)
def test_binary_and_choice_share_keep_for_every_valid_action_combination(context_id):
    context = CONTEXT_BY_ID[context_id]
    # 包括同时减速/跨线；binary 保留两者，choice 只投影主要动作。
    for speed in (None, "DECELERATE", "STOP", "RESUME"):
        for side in (None, "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            labels = {key: key in (speed, side) for key in ACTION_KEYS}
            binary = spec_for(context_id, labels)
            choice = spec_for(context_id, labels, "choice")
            parsed = parse_action_output(build_action_target(binary), spec=binary)
            assert parsed == spec_answers(binary)
            assert parsed["KEEP"] == (build_action_target(choice) == "KEEP")
            assert all(parsed[key] == labels[key] for key in context.action_keys)
            assert len(parsed) == len(context.action_keys) + 2
            prompt = build_action_prompt(spec=binary)
            assert "KEEP:" in prompt and "KEEP cannot coexist" in prompt
            assert "all speed answers are NO" not in prompt
            invalid = spec_for(context_id, {**labels, "INVALID_ACTION_CONTEXT": True})
            assert all(value == (key == "INVALID_ACTION_CONTEXT")
                       for key, value in spec_answers(invalid).items())


def test_keep_is_required_in_production_and_audit_answers():
    spec = spec_for("STATIC_BLOCKAGE", dict.fromkeys(ACTION_KEYS, False))
    target = build_action_target(spec)
    missing = "\n".join(line for line in target.splitlines() if not line.startswith("KEEP:"))
    assert all(value is None for value in parse_action_output(missing, spec=spec).values())
    audit = target + "\n" + "\n".join(f"EVIDENCE_{key}: Visible continued driving." for key in spec.output_keys)
    assert parse_action_output(audit, spec=spec, audit=True)["KEEP"] is True
    missing = "\n".join(line for line in audit.splitlines() if not line.startswith("EVIDENCE_KEEP:"))
    assert all(value is None for value in parse_action_output(missing, spec=spec, audit=True).values())


def test_index_generation_requires_evidence_and_marks_invalid_keep_no():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _answers_for
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import validate_choice_row
    from qwen3vl_local.sft_new_loop_phase3.test_causal_keep import index_row
    assert _answers_for("STATIC_BLOCKAGE", dict.fromkeys(ACTION_KEYS, False), invalid=False)["KEEP"]
    assert not _answers_for("STATIC_BLOCKAGE", None, invalid=True)["KEEP"]
    for labels in (None, {}, {**dict.fromkeys(ACTION_KEYS, False), "STOP": "NO"}):
        with pytest.raises(ValueError):
            _answers_for("STATIC_BLOCKAGE", labels, invalid=False)
    for keep in (None, False, "YES"):
        row = index_row()
        row["answers"]["KEEP"] = keep
        with pytest.raises(ValueError):
            validate_choice_row(row)


@pytest.mark.parametrize("module", ["train", "eval"])
def test_actual_metric_branch_requires_explicit_binary_keep(module):
    # 抽取实际生产计数分支，CPU 检验两端一致，不导入无关模型依赖。
    path = Path(__file__).with_name(module + ".py")
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and "spec.action_output_mode == 'choice'" == ast.unparse(n.test)
                and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                        and c.func.id == "count_keep_prediction" for c in ast.walk(n)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")
    yes, no = (True, False) if module == "train" else ("YES", "NO")
    namespace = dict(spec=SimpleNamespace(action_output_mode="binary"), ACTION_KEYS=ACTION_KEYS,
                     count_keep_prediction=count_keep_prediction)
    gt = {**dict.fromkeys(ACTION_KEYS, no), "KEEP": yes}
    for prediction, hits in ((yes, 1), (no, 0), (None, 0)):
        counts = Counter()
        namespace.update(gt=gt, parsed={**gt, "KEEP": prediction}, action_counts=counts)
        exec(code, namespace)
        assert counts["KEEP/gt_yes"] == 1
        assert counts["KEEP/pred_yes"] == counts["KEEP/recall_hit"] == hits


def test_binary_keep_guard_and_error_review():
    from qwen3vl_local.sft_new_loop_phase3.test_repair_contract import _guard, _healthy_metrics
    from qwen3vl_local.sft_new_loop_phase3.prepare_error_review import compare_action_labels
    from qwen3vl_local.sft_new_loop_phase3.audit_eval_cases import TARGETS, _target_matches
    for metric in ("gt_yes", "precision", "recall"):
        assert not _guard({**_healthy_metrics(), f"action/keep_{metric}": 0})["all_ok"]
    raw = dict.fromkeys(ACTION_KEYS, False)
    gt = {key: "YES" if value else "NO" for key, value in binary_answers(raw).items()}
    row = dict(gt=gt, action_answers=gt, prompt_spec=dict(action_output_mode="binary"))
    assert compare_action_labels(row, raw) == ([], [])
    row["gt"] = {**gt, "KEEP": "NO"}
    assert compare_action_labels(row, raw) == ([], ["KEEP"])
    errors = _target_matches(dict(gt=gt, parsed={**gt, "KEEP": "NO"}))
    assert "keep_fn" in errors and set(errors) <= set(TARGETS)
    errors = _target_matches(dict(gt=gt, parsed={**gt, "STOP": "YES"}))
    assert "keep_action_conflict" in errors and set(errors) <= set(TARGETS)
