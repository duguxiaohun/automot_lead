"""Phase1/2 到动作条件的门控与真实 prompt 回归，不加载 Qwen 或离线 runner。"""

from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from qwen3vl_local.action_prior import prompts
from qwen3vl_local.action_prior.action_input import HighLevelActionIndex, gate_action
from qwen3vl_local.action_prior.runtime import PriorEngine, sample_condition_inputs
from qwen3vl_local.action_prior.text_cache import TextCache
from qwen3vl_local.action_prior.train import audit_counts
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
from test_action_input import record, write_index

STOP = dict(status="selected", actions=["STOP"])


def conditions(rs="R4", **events):
    """提供已通过域校验的条件，具体事件仍须逐项确认 YES。"""
    return {"ROAD_STRUCTURE": rs, **events,
            **{f"{d}/{p2.INVALID_KEY}": "NO" for d in p2.QUESTION_DOMAINS}}


@pytest.mark.parametrize("scope,event,rs", [
    ("UE1", "UE1", "R1"), ("UE2", "STATIC_OBSTACLE", "R1"), ("UE3", "UE3", "R3"),
    ("UE4", "VULNERABLE", "R2"), ("UE5", "UE5", "R1"),
    ("UE6", "UE6", "R4"), ("UE7", "TRAFFIC_LIGHT_ABNORMAL", "R5"),
])
@pytest.mark.parametrize("answer", ["YES", "NO", None])
def test_all_ue_actions_require_accepted_upstream_event(scope, event, rs, answer):
    """源 scope 即便为真值，也不能把 NO/未知事件重新激活。"""
    upstream = conditions(rs, **{event: answer})
    before = dict(upstream)
    action, audit = gate_action(STOP, upstream, (scope,), (scope,))
    assert (action == STOP) is (answer == "YES")
    assert audit["reason"] == ("accepted" if answer == "YES" else "upstream_unconfirmed")
    assert upstream == before


@pytest.mark.parametrize("upstream", [
    conditions("R1", UE6="YES"), conditions(None, UE6="YES"),
    {**conditions(UE6="YES"), f"{p2.JUNCTION_DOMAIN}/{p2.INVALID_KEY}": "YES"},
    {**conditions(UE6="YES"), f"{p2.JUNCTION_DOMAIN}/{p2.INVALID_KEY}": None},
])
def test_unknown_or_incompatible_road_and_invalid_phase2_domain_block_action(upstream):
    """事件 YES 不能绕过 RS 或 Phase2 问题域失效。"""
    assert gate_action(STOP, upstream, ("UE6",))[0]["status"] == "unavailable"


@pytest.mark.parametrize("context,rs", [
    ("RE2_NAVIGATION_TRANSITION", "R1"), ("RE2_PRIOR_OBSTACLE", "R2"),
    ("RE2_RECOVERY_PENDING", "R1"), ("RE3", "R3"), ("RE5", "R5"),
])
def test_re_needs_independent_matching_transition_gate(context, rs):
    """动作文件里的 RE scope 不等于上游 transition 证据。"""
    upstream = conditions(rs)
    assert gate_action(STOP, upstream, (context,))[0]["status"] == "unavailable"
    assert gate_action(STOP, upstream, (context,), (context,))[0] == STOP
    assert gate_action(STOP, upstream, (context,), ("UE6",))[0]["status"] == "unavailable"


def test_concurrent_domains_keep_only_supported_actions():
    """只有纵向事件确认时，不借未确认机动事件注入横向动作。"""
    combined = dict(status="selected", actions=["STOP", "LANE_CHANGE_LEFT"])
    upstream = conditions("R1", UE1="YES", STATIC_OBSTACLE="NO")
    action, audit = gate_action(combined, upstream, ("UE1", "UE2"))
    assert action == STOP and audit["dropped_actions"] == ["LANE_CHANGE_LEFT"]
    assert audit["reason"] == "domain_filtered"
    upstream["STATIC_OBSTACLE"] = "YES"
    assert gate_action(combined, upstream, ("UE1", "UE2"))[0] == STOP


@pytest.fixture
def engine():
    """记录最终图文 transcript；空生成结果使摘要分支走真实 fallback。"""
    transcripts = []

    def template(messages, **kwargs):
        """保留实际的 system/user/assistant，便于逐字比较。"""
        assert kwargs["add_generation_prompt"] is False
        transcripts.append(messages)
        return "fixture"

    fake = SimpleNamespace(
        build_messages=lambda system, user, images: [dict(role="system", content=system), dict(role="user", content=user)],
        processor=SimpleNamespace(apply_chat_template=template),
        prepare_inputs=lambda *a: {"input_ids": torch.zeros((1, 4), dtype=torch.long)},
        prefill=lambda *a: SimpleNamespace(past_key_values="kv", rope_deltas=torch.tensor([[0]])),
    )
    return fake, transcripts


def runtime(engine, labels, cache, enabled, generate=False):
    """模拟同图/同条件下是否开启动作开关，其余输入固定。"""
    value = PriorEngine(engine, {"identity": "fixture"}, labels=labels, text_cache=cache,
                        high_level_planning=True, high_level_action_prior=enabled,
                        generate_analysis=generate, analysis_review=False)
    value.generate_messages = lambda *a, **kw: ("", SimpleNamespace(decode_steps=[SimpleNamespace(is_eos=True)]))
    return value


@pytest.mark.parametrize("status", ["selected", "no_action", "unavailable", "not_applicable", "missing"])
@pytest.mark.parametrize("generate", [False, True])
def test_no_effective_action_preserves_entire_transcript_with_index_contexts(tmp_path, engine, status, generate):
    """UE6=NO 或无动作时，冷/热缓存的最终 transcript 与只开 planning 完全相同。"""
    fake, transcripts = engine
    source = record(status="unavailable" if status == "missing" else status,
                    actions=["STOP"] if status == "selected" else [],
                    event_buckets=["UE6"], planning_contexts=["UE6"])
    if status == "not_applicable":
        source.update(event_status="confirmed_regular", event_buckets=[], planning_contexts=[])
    index = HighLevelActionIndex(write_index(tmp_path / "action.jsonl", [source]))
    frame = 99 if status == "missing" else 2
    sample = dict(scenario="S", run_id="R", anchor=frame, event_balance_scene_contexts=["UE6"])
    args = SimpleNamespace(high_level_action_prior=True, event_balanced_scene_priors=False)
    inputs = sample_condition_inputs(args, sample, index)
    assert inputs["event_balanced_scene_contexts"] == ()
    upstream = conditions("R1", UE6="NO")
    labels = SimpleNamespace(path="synthetic", rows=1, priors=lambda identity: dict(conditions=upstream, invalid={}, calls=[]))
    images = [Image.new("RGB", (2, 2)) for _ in range(4)]
    cache = TextCache(tmp_path / "cache")
    baseline = runtime(fake, labels, cache, False, generate)
    enabled = runtime(fake, labels, cache, True, generate)
    baseline.condition(images, "nav", "same", ("S", "R", frame))
    original = transcripts[-1]
    for hit in (False, True):
        enabled.condition(images, "nav", "same", ("S", "R", frame), **inputs)
        assert enabled.last_audit["text_cache_hit"] is hit
        assert transcripts[-1] == original
        assert enabled.last_audit["high_level_action_input"] == inputs["high_level_action"]
        assert "UPCOMING_HIGH_LEVEL_ACTION" not in transcripts[-1][1]["content"]
        assert prompts.EVENT_COMPACT_NAMES["UE6"] not in transcripts[-1][1]["content"]
        for render in (lambda p: prompts.prefill_prompt(p, "nav"), lambda p: prompts.analysis_prompt(p, "nav"),
                       lambda p: prompts.review_prompt(p, "nav", "draft"), lambda p: prompts.fallback_analysis(p, "nav")):
            assert render(enabled.last_audit) == render(baseline.last_audit)
    if status == "selected":
        counts = audit_counts(enabled.last_audit)
        assert counts["prior/high_level_action_input/selected"] == 1
        assert counts["prior/high_level_action/unavailable"] == 1
        assert counts["prior/high_level_action_gate/upstream_unconfirmed"] == 1


@pytest.mark.parametrize("mode", ["confusion", "invalid"])
def test_actual_prior_noise_cannot_be_repaired_by_action_context(tmp_path, engine, mode, monkeypatch):
    """真实 dataset-priors 噪声路径先改事件，动作 oracle 不得回填噪声前真值。"""
    from test_dataset_priors import row, write_index as write_labels
    from qwen3vl_local.action_prior.dataset_labels import PriorLabelIndex, PriorNoise
    path = write_labels(tmp_path / "labels", [row(frame=2, rs="R4", target="UE6",
                       domain=p2.JUNCTION_DOMAIN, invalid_domain=p2.ROAD_DOMAIN)])
    noise = PriorNoise(1, invalid_share=1 if mode == "invalid" else 0)
    monkeypatch.setattr(noise, "_unit", lambda key, salt: 0.9 if salt == "channel" else 0.0)
    labels = PriorLabelIndex(path, noise)
    fake, transcripts = engine
    value = runtime(fake, labels, TextCache(tmp_path / "cache"), True)
    value.condition([Image.new("RGB", (2, 2)) for _ in range(4)], "nav", "same", ("S", "R", 2),
                    high_level_action=STOP, high_level_action_contexts=("UE6",))
    assert value.last_audit["conditions"]["UE6"] != "YES"
    assert value.last_audit["high_level_action_gate"]["reason"] == "upstream_unconfirmed"
    assert "UPCOMING_HIGH_LEVEL_ACTION" not in transcripts[-1][1]["content"]
    assert prompts.EVENT_COMPACT_NAMES["UE6"] not in transcripts[-1][1]["content"]


def test_empty_actions_preserve_prompt_even_when_event_and_scene_priors_are_confirmed():
    """已确认的特殊事件全 NO 也不能追加负动作说明；原显式 scene priors 保留。"""
    baseline = dict(conditions=conditions("R5", UE6="YES"), high_level_planning=True,
                    event_balanced_scene_contexts=("RE5",))
    for status in ("no_action", "unavailable", "not_applicable"):
        effective, _ = gate_action(dict(status=status, actions=[]), baseline["conditions"], ("UE6",), ("RE5",))
        with_action = dict(baseline, high_level_action_prior=True, high_level_action=effective)
        assert prompts.prefill_prompt(with_action, "nav") == prompts.prefill_prompt(baseline, "nav")
        assert prompts.analysis_prompt(with_action, "nav") == prompts.analysis_prompt(baseline, "nav")
        assert prompts.review_prompt(with_action, "nav", "draft") == prompts.review_prompt(baseline, "nav", "draft")
        assert prompts.fallback_analysis(with_action, "nav") == prompts.fallback_analysis(baseline, "nav")


def test_re_scope_does_not_enable_scene_prior_switch(tmp_path):
    """RE 索引始终只提供门控 scope，独立开关才使 sample 的 transition 成为事实。"""
    source = record(event_buckets=["RE3"], planning_contexts=["RE3"], actions=["LANE_CHANGE_LEFT"])
    index = HighLevelActionIndex(write_index(tmp_path / "action.jsonl", [source]))
    sample = dict(scenario="S", run_id="R", anchor=2, event_balance_scene_contexts=["RE3"])
    for enabled in (False, True):
        args = SimpleNamespace(high_level_action_prior=True, event_balanced_scene_priors=enabled)
        values = sample_condition_inputs(args, sample, index)
        assert values["event_balanced_scene_contexts"] == (("RE3",) if enabled else ())
        action, _ = gate_action(values["high_level_action"], conditions("R3"),
                               values["high_level_action_contexts"], values["event_balanced_scene_contexts"])
        assert (action["status"] == "selected") is enabled
