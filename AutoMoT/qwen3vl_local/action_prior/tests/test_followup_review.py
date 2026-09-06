"""后续审查回归：自然语言验收、执行边界、来源迁移与正常域外分组。"""

from collections import Counter
import json
from pathlib import Path
import pytest
from qwen3vl_local.action_prior import prompts, config, provenance
from qwen3vl_local.action_prior.metrics import grouped_counts
from qwen3vl_local.action_prior.train import metrics_from_counts


@pytest.mark.parametrize(
    "draft",
    [
        "Scene: The road is a same-direction surface corridor.\nInteraction: Another vehicle is cutting into the ego corridor.\nPlanning context: At 4 m/s the forward target and cut-in jointly constrain the available corridor.",
        "Scene: Lane following is the accepted surface-road context.\nInteraction: A vehicle intrusion is present in the immediate path.\nPlanning context: Current motion is 4 m/s toward the target ahead, with the accepted intrusion relevant to that path.",
    ],
)
def test_paraphrases_can_pass_without_template_match(draft):
    priors = {"conditions": {"ROAD_STRUCTURE": "R1", "UE3": "YES"}}
    review = {k: True for k in prompts.REVIEW_KEYS}
    assert draft != prompts.fallback_analysis(priors)
    assert prompts.valid_analysis(draft, priors, review)
    for criterion in prompts.REVIEW_KEYS:
        assert not prompts.valid_analysis(
            draft, priors, dict(review, **{criterion: False})
        )
    assert "VERIFIED_SUMMARY" not in prompts.analysis_prompt(
        priors, "current velocity is 4 m/s"
    )


@pytest.mark.parametrize(
    "text",
    [
        "true",
        "{}",
        '{"consistent": "true"}',
        '{"consistent":false,"consistent":true,"positive_coverage":true,"unknown_respected":true,"no_unsupported_claims":true,"navigation_grounded":true}',
        "```json\n{}\n```",
    ],
)
def test_review_malformed_or_duplicate_cannot_pass(text):
    assert prompts.parse_review(text) is None


def test_navigation_and_events_change_fallback_only_not_generation_answer():
    priors = {"conditions": {"ROAD_STRUCTURE": "R1", "UE3": "YES"}}
    nav1 = "Your current and next target point is (8.000000, 2.000000), and your current velocity is 4.00 m/s."
    nav2 = "Your current and next target point is (5.000000, -1.000000), and your current velocity is 0.00 m/s."
    a, b = [prompts.fallback_analysis(priors, nav) for nav in (nav1, nav2)]
    assert a != b and "4 m/s" in a and "ahead and left" in a
    assert "0 m/s" in b and "ahead and right" in b
    assert "UE3" in a
    assert a not in prompts.analysis_prompt(priors, nav1)


def test_unrelated_phase3_mutation_does_not_invalidate(tmp_path):
    for name in provenance.EXECUTION_SEEDS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# dependency\n")
    # 实际模块级相对导入必须被展开。
    engine = tmp_path / "qwen3vl_local/engine.py"
    engine.write_text("from . import helper\n")
    helper = engine.with_name("helper.py")
    helper.write_text("VALUE=1\n")
    unused = tmp_path / "qwen3vl_local/sft_new_loop_phase3/train.py"
    unused.parent.mkdir(parents=True)
    unused.write_text("VALUE=1\n")
    a = provenance.execution_fingerprint(tmp_path)
    unused.write_text("VALUE=2\n")
    assert provenance.execution_fingerprint(tmp_path) == a
    helper.write_text("VALUE=2\n")
    assert provenance.execution_fingerprint(tmp_path) != a
    real = provenance.execution_fingerprint()
    assert not any("sft_new_loop_phase3" in path for path in real["code"])
    assert "qwen3vl_local/sft_loop_phase2_augment/prompts.py" in real["code"]


def test_generation_identity_survives_same_index_remap_and_missing_source(
    tmp_path, monkeypatch
):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"fixture weights")
    bev = tmp_path / "bev.pt"
    bev.write_bytes(b"fixture BEV")
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            dict(scenario="X", route_id="Town01_Rep0_route_1_route0", split="train")
        )
        + "\n"
    )
    run = tmp_path / "adapter"
    run.mkdir()
    (run / "train_run_manifest.json").write_text(
        json.dumps(dict(index=str(index), split="train"))
    )
    monkeypatch.setattr(
        config,
        "select_adapter",
        lambda root, phase, *a: dict(
            path=str(run / "best_generation"), fingerprint=f"phase{phase}", metadata={}
        ),
    )
    args = config.parser().parse_args(
        ["--model-dir", str(base), "--lead-bev-ckpt", str(bev), "--selection-policy", "strict"]
    )
    automatic = config.build_contract(args)
    args.phase1_training_index = str(index)
    explicit = config.build_contract(args)
    assert automatic["identity"] == explicit["identity"]
    assert automatic["audit_identity"] != explicit["audit_identity"]
    moved = tmp_path / "moved.jsonl"
    moved.write_bytes(index.read_bytes())
    args.phase1_training_index = str(moved)
    relocated = config.build_contract(args)
    assert automatic["identity"] == relocated["identity"]
    changes = provenance.audit_source_changes(
        automatic["upstream_sources"], relocated["upstream_sources"]
    )
    assert changes["phase1"]["status"] == "same_content"
    args.phase1_training_index = str(tmp_path / "unavailable.jsonl")
    unavailable = config.build_contract(args)
    assert unavailable["identity"] == automatic["identity"]
    assert unavailable["upstream_sources"]["phase1"]["status"] == "unknown"
    moved.write_text(
        json.dumps(
            dict(scenario="X", route_id="Town01_Rep0_route_2_route0", split="train")
        )
        + "\n"
    )
    args.phase1_training_index = str(moved)
    changed = config.build_contract(args)
    assert changed["identity"] == automatic["identity"]
    assert (
        provenance.audit_source_changes(
            automatic["upstream_sources"], changed["upstream_sources"]
        )["phase1"]["status"]
        == "changed_content"
    )


def test_normal_domain_exclusion_and_unconfirmed_have_distinct_metrics():
    totals = Counter(samples=3, loss=10)
    for invalid, loss in [
        ({}, 1),
        ({"UE6": "domain_inapplicable"}, 2),
        ({"UE6": "domain_inapplicable", "UE3": "disagreement"}, 7),
    ]:
        totals.update(
            grouped_counts(dict(invalid=invalid, conditions={}), {}, dict(loss=loss))
        )
    report = metrics_from_counts(totals)
    assert report["group/confirmation/all_confirmed/loss"] == 1
    assert report["group/confirmation/expected_domain_only/loss"] == 2
    assert report["group/confirmation/unconfirmed/loss"] == 7
    assert report["group/invalid/loss"] == 4.5  # 旧总组保留，但不能解释为复核失败能力。


def test_compare_disagreement_is_reported_but_history_policy_unchanged():
    from qwen3vl_local.action_prior.priors import collect_priors
    from test_contracts import ask_fixture

    counts = Counter()

    def ask(phase, spec, history):
        text, prompt = ask_fixture()(phase, spec, history)
        if phase == 2 and spec.question_domain == "ROAD_CORRIDOR":
            counts["road"] += 1
            value = "NO" if counts["road"] == 2 else "YES"
            text = text.replace("UE3: NO", f"UE3: {value}")
        return text, prompt

    result = collect_priors(ask, "case", recheck_mode="compare")
    assert result["conditions"]["UE3"] == "YES"
    assert result["condition_acceptance_policy"] == "history_consistency"
    assert result["compare_requires_consensus"] is False
    road = next(
        x for x in result["recheck_mode_disagreements"] if x["scope"] == "ROAD_CORRIDOR"
    )
    assert road["errors"]["UE3"] == "disagreement"
