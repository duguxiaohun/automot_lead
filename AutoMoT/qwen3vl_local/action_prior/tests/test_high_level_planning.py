"""移除候选规划菜单，逐场景复用 Phase3 choice 的唯一动作因果句。"""

from pathlib import Path

import pytest

from qwen3vl_local.action_prior import prompts
from qwen3vl_local.action_prior.action_input import action_sentence, gate_action
from qwen3vl_local.action_prior.config import parser, validate_args
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import action_description
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_CONTEXTS
from test_action_gating import conditions
from test_action_input import record, write_index

CASES = [(spec, action) for spec in ACTION_CONTEXTS for action in spec.action_keys]


@pytest.mark.parametrize("spec,action", CASES, ids=[f"{s.source_event}-{a}" for s, a in CASES])
def test_every_event_action_uses_exact_choice_sentence_after_natural_facts(spec, action):
    """十场景全部动作通过真实门控；preﬁll/摘要/复核/fallback 内容一致，无候选菜单。"""
    bucket = spec.source_event.replace("-", "")
    context = "RE2_NAVIGATION_TRANSITION" if bucket == "RE2" else bucket
    event = prompts._CONTEXT_DUPLICATES.get(context)
    upstream = conditions(spec.allowed_rs[0], **({event: "YES"} if event else {}))
    scenes = (context,) if bucket.startswith("RE") else ()
    effective, audit = gate_action(dict(status="selected", actions=[action]), upstream, (context,), scenes)
    assert effective["actions"] == [action]
    assert audit["description_context_id"] == spec.context_id
    priors = dict(conditions=upstream, event_balanced_scene_contexts=scenes,
                  high_level_action_prior=True, high_level_action=effective, high_level_action_gate=audit)
    sentence = "Next action: " + action_description(spec.context_id, action)
    natural = prompts.scene_description(upstream, scenes)
    text = prompts.prefill_prompt(priors, "nav")
    assert text == f"[SCENE_DESCRIPTION]\n{natural}\n{sentence}\n[/SCENE_DESCRIPTION]\n[CURRENT_NAVIGATION]\nnav\n[/CURRENT_NAVIGATION]"
    assert sentence.count(".") == 1 and len(sentence.split()) <= 45
    for rendered in (text, prompts.analysis_prompt(priors, "nav"),
                     prompts.review_prompt(priors, "nav", "draft"), prompts.fallback_analysis(priors)):
        assert rendered.count(sentence) == 1
        assert "If constrained" not in rendered
        assert "Action meanings" not in rendered
        for other in spec.action_keys:
            if other != action:
                assert action_description(spec.context_id, other) not in rendered
    assert prompts.analysis_format_valid(prompts.fallback_analysis(priors))
    disabled = dict(priors, high_level_action_prior=False)
    assert prompts.prefill_prompt(disabled, "nav") == prompts.prefill_prompt(
        dict(conditions=upstream, event_balanced_scene_contexts=scenes), "nav")


def test_concurrent_context_selects_one_supported_cause_independent_of_index_order():
    """并发原因采用 taxonomy 固定顺序，只在支持所选动作的已接受场景内选择。"""
    upstream = conditions("R1", UE1="YES", STATIC_OBSTACLE="YES")
    for contexts in (("UE1", "UE2"), ("UE2", "UE1")):
        action, audit = gate_action(dict(status="selected", actions=["LANE_CHANGE_LEFT"]), upstream, contexts)
        assert audit["description_context_id"] == "STATIC_BLOCKAGE"
        assert action_sentence(action, audit["accepted_contexts"]) == "Next action: " + action_description("STATIC_BLOCKAGE", "LANE_CHANGE_LEFT")
        action, audit = gate_action(dict(status="selected", actions=["STOP"]), upstream, contexts)
        assert audit["description_context_id"] == "LEAD_BRAKE"
        assert action_sentence(action, audit["accepted_contexts"]) == "Next action: " + action_description("LEAD_BRAKE", "STOP")


def test_selected_action_without_accepted_context_cannot_invent_causal_sentence():
    """门控缺失或只有不支持横向的场景时拒绝渲染，不回退通用原因。"""
    for contexts in ((), ("UE1",)):
        with pytest.raises(ValueError, match="confirmed compatible context"):
            action_sentence(dict(status="selected", actions=["LANE_CHANGE_LEFT"]), contexts)
    for status in ("no_action", "unavailable", "not_applicable"):
        assert action_sentence(dict(status=status, actions=[])) == ""


def test_removed_cli_and_old_planning_checkpoint_are_rejected():
    """新训练不再暴露 planning，旧模式不能被静默重解释为新 prompt。"""
    p = parser()
    assert "high-level-planning" not in p.format_help()
    for flag in ("--high-level-planning", "--no-high-level-planning"):
        with pytest.raises(SystemExit):
            p.parse_args([flag])
    args = p.parse_args([])
    args.high_level_planning = True
    with pytest.raises(ValueError, match="original source"):
        validate_args(args)


def test_action_contract_binds_live_phase3_choice_wording(tmp_path, monkeypatch):
    """动作索引内容相同但 choice 文字改动时，也必须拒绝沿用旧 decoder 条件。"""
    from test_dataset_priors import dataset_args
    from qwen3vl_local.action_prior import provenance
    from qwen3vl_local.action_prior.config import build_contract
    from qwen3vl_local.action_prior.contracts import require_contract
    monkeypatch.setattr(provenance, "execution_fingerprint", lambda: {"fixture": "fixed"})
    args = dataset_args(tmp_path)
    args.high_level_action_prior = True
    args.high_level_action_index = str(write_index(tmp_path / "actions.jsonl", [record()]))
    before = build_contract(args)
    source = Path(__file__).parents[2] / "sft_new_loop_phase3/choice_semantics.py"
    original_read = Path.read_bytes

    def changed_wording(path):
        """模拟新版因果措辞，不写入真实 Phase3 源码或动作数据。"""
        raw = original_read(path)
        if path.resolve() == source.resolve():
            return raw.replace(b"prevent a rear-end collision", b"avoid a rear-end collision")
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_wording)
    after = build_contract(args)
    assert before["identity_payload"]["high_level_action_input"]["sha256"] == after["identity_payload"]["high_level_action_input"]["sha256"]
    with pytest.raises(ValueError):
        require_contract(before, after, allow_prior_source_change=True)
