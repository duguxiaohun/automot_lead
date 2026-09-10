"""v5 自然场景先验：只传已确认 RS/EVENT，不把 Phase1/2 反例塞进 KV。"""

import pytest

from qwen3vl_local.action_prior import prompts


NAVIGATION = (
    "Your current and next target point is (8.0, 2.0), (12.0, 3.0), "
    "your final destination is (100.0, 20.0), and your current velocity is 4.0 m/s."
)


@pytest.mark.parametrize("rs,expected", [
    ("R1", "ordinary surface-road corridor"),
    ("R2", "two-way or shared corridor"),
    ("R4", "local junction normally governed"),
    ("R5", "local junction governed by priority"),
])
def test_road_description_uses_confirmed_road_only(rs, expected):
    text = prompts.scene_description({"ROAD_STRUCTURE": rs})
    assert expected in text
    assert "YES" not in text and "NO" not in text and rs not in text


@pytest.mark.parametrize("field,description", prompts.EVENT_DESCRIPTIONS.items())
def test_each_confirmed_event_becomes_a_helpful_natural_sentence(field, description):
    prior = {"ROAD_STRUCTURE": "R1", field: "YES"}
    prompt = prompts.analysis_prompt({"conditions": prior}, NAVIGATION)
    assert description in prompt
    assert field not in prompt and "NO" not in prompt and "UNKNOWN" not in prompt
    assert description not in prompts.scene_description(dict(prior, **{field: "NO"}))


def test_unknown_and_negative_conditions_are_not_rendered_as_facts():
    conditions = {
        "ROAD_STRUCTURE": "R1", "UE3": "NO", "VULNERABLE": None,
        "STATIC_OBSTACLE": "NO", "UE1": None,
    }
    prompt = prompts.analysis_prompt({"conditions": conditions}, NAVIGATION)
    assert "cut-in" not in prompt and "vulnerable" not in prompt.lower()
    assert "UE3" not in prompt and "NO" not in prompt and "UNKNOWN" not in prompt


def test_highway_is_independent_and_r3_does_not_silently_claim_it():
    assert "highway" not in prompts.scene_description({"ROAD_STRUCTURE": "R3"}).lower()
    text = prompts.scene_description({"ROAD_STRUCTURE": "R3", "RS_HIGHWAY": "YES"})
    assert "limited-access highway" in text


def test_all_concurrent_events_are_retained_within_the_short_budget():
    conditions = {key: "YES" for key in prompts.EVENT_DESCRIPTIONS}
    conditions["ROAD_STRUCTURE"] = "R2"
    text = prompts.fallback_analysis({"conditions": conditions}, NAVIGATION)
    assert prompts.analysis_format_valid(text)
    for name in prompts.EVENT_COMPACT_NAMES.values():
        assert name in text
    assert "YES" not in text and "NO" not in text


def test_generation_review_and_fallback_share_the_same_scene_prior_without_json_leak():
    priors = {"conditions": {"ROAD_STRUCTURE": "R4", "UE6": "YES", "UE3": "YES"}}
    generation = prompts.analysis_prompt(priors, NAVIGATION)
    review = prompts.review_prompt(priors, NAVIGATION, "short draft")
    fallback = prompts.fallback_analysis(priors, NAVIGATION)
    description = prompts.scene_description(priors["conditions"])
    assert description in generation and description in review and description in fallback
    for value in (generation, review, fallback):
        assert "[ACCEPTED_PERCEPTION_PRIORS]" not in value
        assert "[PLANNING_EXPERIENCE]" not in value
        assert "ROAD_STRUCTURE" not in value and "UE6" not in value


@pytest.mark.parametrize(
    "context, phrase",
    [
        ("RE2_NAVIGATION_TRANSITION", "Visible lane geometry and navigation"),
        ("RE2_PRIOR_OBSTACLE", "prior static blockage is recorded"),
        ("RE2_RECOVERY_PENDING", "static blockage has been passed"),
        ("RE3", "ramp, merge, or exit transition"),
        ("RE5", "unsignalized priority junction"),
    ],
)
def test_explicit_special_regular_context_is_short_natural_and_has_no_category_leak(context, phrase):
    priors = {"conditions": {"ROAD_STRUCTURE": "R1"}, "event_balanced_scene_contexts": [context]}
    text = prompts.fallback_analysis(priors, NAVIGATION)
    assert phrase in text
    assert context not in text and "YES" not in text and "NO" not in text
    assert prompts.analysis_format_valid(text)


def test_navigation_re2_never_claims_a_bypass_or_static_obstacle():
    text = prompts.scene_description(
        {"ROAD_STRUCTURE": "R1"}, ["RE2_NAVIGATION_TRANSITION"]
    )
    assert "passed" not in text and "static blockage" not in text
