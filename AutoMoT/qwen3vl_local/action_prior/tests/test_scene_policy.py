"""自动特殊 RE 的条件、实际注入、准备与恢复回归；不加载权重。"""

import json
from types import SimpleNamespace

import pytest

from qwen3vl_local.action_prior.config import parser
from qwen3vl_local.action_prior.scene_policy import (
    SCENE_PRIOR_POLICY, LEGACY_SCENE_PRIOR_POLICY, resolve_scene_priors,
)
from qwen3vl_local.action_prior.runtime import sample_condition_inputs
from qwen3vl_local.action_prior.action_input import gate_action
from qwen3vl_local.action_prior import prompts, prepare_action_priors


@pytest.mark.parametrize("options,enabled", [
    ([], False), (["--dataset-priors"], False), (["--high-level-planning"], False),
    (["--dataset-priors", "--high-level-planning"], True),
    (["--dataset-priors", "--high-level-planning", "--high-level-action-prior"], True),
    (["--dataset-priors", "--high-level-planning", "--prior-noise", "0.1"], False),
    (["--dataset-priors", "--high-level-planning", "--no-high-level-planning"], False),
])
def test_new_run_derives_scene_policy_once(options, enabled):
    """无需额外开关，采样模式不决定输入条件，带噪声实验不补干净标签。"""
    args = parser().parse_args(options)
    assert resolve_scene_priors(args) is enabled
    assert args.scene_prior_policy == SCENE_PRIOR_POLICY
    args.sampling_mode = "event_balanced"
    assert resolve_scene_priors(args) is enabled


@pytest.mark.parametrize("context,rs", [
    ("RE2_NAVIGATION_TRANSITION", "R1"), ("RE2_PRIOR_OBSTACLE", "R2"),
    ("RE2_RECOVERY_PENDING", "R1"), ("RE3", "R3"), ("RE5", "R5"),
])
def test_auto_re_reaches_planning_and_action_but_none_retains_context(context, rs):
    """走真实运行时输入选择与门控；不把并发 UE scope 偷补为上游 YES。"""
    args = parser().parse_args(["--dataset-priors", "--high-level-planning", "--high-level-action-prior"])
    resolve_scene_priors(args)
    sample = dict(scenario="S", run_id="R", anchor=2,
                  event_balance_scene_contexts=[context, "UE2"])
    conditions = {"ROAD_STRUCTURE": rs, "STATIC_OBSTACLE": "NO"}
    index = SimpleNamespace(candidate_evidence=lambda _: dict(status="selected", actions=["STOP"]),
                            planning_contexts=lambda _: (context,))
    inputs = sample_condition_inputs(args, sample, index)
    assert inputs["event_balanced_scene_contexts"] == (context,)
    for status in ("selected", "no_action"):
        effective, _ = gate_action(dict(status=status, actions=["STOP"] if status == "selected" else []),
                                   conditions, inputs["high_level_action_contexts"],
                                   inputs["event_balanced_scene_contexts"])
        priors = dict(conditions=conditions, high_level_planning=True, high_level_action_prior=True,
                      high_level_action=effective, event_balanced_scene_contexts=(context,))
        prompt = prompts.prefill_prompt(priors, "nav")
        assert prompts.HIGH_LEVEL_CONTEXT_FACTS[context] in prompt
        assert prompts.HIGH_LEVEL_TRANSITION_PURPOSES[context] in prompt
        assert ("UPCOMING_HIGH_LEVEL_ACTION" in prompt) is (status == "selected")
        assert prompts.EVENT_COMPACT_NAMES["STATIC_OBSTACLE"] not in prompt
    # 普通背景没有 context，即使有索引动作也不能凭空注入特殊 RE。
    sample["event_balance_scene_contexts"] = []
    assert sample_condition_inputs(args, sample, index)["event_balanced_scene_contexts"] == ()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("policy", [None, SCENE_PRIOR_POLICY])
def test_resume_and_eval_keep_saved_scene_condition(tmp_path, enabled, policy):
    """原 run 的显式关闭不会被新默认开启覆盖；eval 也不重新推导条件。"""
    saved = dict(event_balanced_scene_priors=enabled, dataset_priors=True, high_level_planning=True)
    if policy:
        saved["scene_prior_policy"] = policy
    (tmp_path / "config.json").write_text(json.dumps(saved))
    args = parser().parse_args(["--resume", str(tmp_path / "latest.pt")])
    assert resolve_scene_priors(args) is enabled
    assert args.scene_prior_policy == (policy or LEGACY_SCENE_PRIOR_POLICY)
    assert resolve_scene_priors(SimpleNamespace(**saved)) is enabled


def test_planning_only_prepares_map_and_never_action_labels(tmp_path, monkeypatch):
    """直接 Python planning-only 入口也准备映射，恢复缺失产物时不重标。"""
    args = parser().parse_args(["--dataset-priors", "--high-level-planning"])
    resolve_scene_priors(args)
    calls = []
    monkeypatch.setattr(prepare_action_priors, "ensure_full_mapping", lambda a: calls.append(a))
    monkeypatch.setattr(prepare_action_priors, "prepare_actions", lambda *a: pytest.fail("unexpected action labels"))
    prepare_action_priors.ensure_scene_inputs(args)
    assert calls == [args]
    args.resume = str(tmp_path / "latest.pt")
    with pytest.raises(ValueError, match="do not regenerate"):
        prepare_action_priors.ensure_scene_inputs(args)
    args.event_balance_index = "saved/mapping.jsonl"
    prepare_action_priors.ensure_scene_inputs(args)
    assert calls == [args]


def test_retired_cli_not_exposed_or_accepted():
    """主线公开 CLI 移除旧开关，保存用的内部字段仍保留。"""
    p = parser()
    assert "event-balanced-scene-priors" not in p.format_help()
    for flag in ("--event-balanced-scene-priors", "--no-event-balanced-scene-priors"):
        with pytest.raises(SystemExit):
            p.parse_args([flag])


def test_legacy_checkpoint_without_scene_field_stays_disabled():
    """旧 checkpoint 缺字段不能被误认为是请求自动推导的新训练。"""
    args = SimpleNamespace(dataset_priors=True, high_level_planning=True, prior_noise=0)
    assert resolve_scene_priors(args) is False
    assert args.scene_prior_policy == LEGACY_SCENE_PRIOR_POLICY
