"""七类 UE 经验的来源一致性、缺失边界、并发和长度回归。"""

import itertools
import json

import pytest

from qwen3vl_local.action_prior import dataset_labels, prompts
from qwen3vl_local.action_prior.priors import collect_priors
from test_contracts import ask_fixture


EVENT_SOURCES = {
    "UE1": "UE1", "STATIC_OBSTACLE": "UE2", "UE3": "UE3",
    "VULNERABLE": "UE4", "UE5": "UE5", "UE6": "UE6",
    "TRAFFIC_LIGHT_ABNORMAL": "UE7",
}
NAVIGATION = (
    "Your current and next target point is (8.0, 2.0), (12.0, 3.0), "
    "your final destination is (100.0, 20.0), and your current velocity is 4.0 m/s."
)


def experience_block(prompt):
    """读取实际发给模型的经验 JSON。"""
    return json.loads(prompt.split("[PLANNING_EXPERIENCE]\n", 1)[1].split(
        "\n[/PLANNING_EXPERIENCE]", 1
    )[0])


@pytest.mark.parametrize("source,event", EVENT_SOURCES.items())
def test_only_accepted_yes_selects_each_ue(source, event):
    for rs in (None, "R1", "R2", "R3", "R4", "R5"):
        c = {"ROAD_STRUCTURE": rs, source: "YES"}
        guidance = prompts.planning_experience(c)
        assert set(guidance) == {event}
        assert guidance[event] in prompts.fallback_planning(c, NAVIGATION)
        for value in (None, "NO"):
            assert set(prompts.planning_experience(dict(c, **{source: value}))) == {"DEFAULT"}


def test_lora_and_dataset_share_ue2_experience_without_source_metadata_leak():
    base_ask = ask_fixture()

    def ask(phase, spec, history):
        answer, prompt = base_ask(phase, spec, history)
        return answer.replace("STATIC_OBSTACLE: NO", "STATIC_OBSTACLE: YES"), prompt

    lora = collect_priors(ask, "case")
    packed = dataset_labels.pack("R1", "RE", "ROAD_CORRIDOR", "LOCAL_JUNCTION", {
        "HIGHWAY": False, "STATIC_OBSTACLE": True,
        "VULNERABLE": False, "TRAFFIC_LIGHT_ABNORMAL": False,
    })
    conditions, invalid, label = dataset_labels.conditions_from_label(packed)
    dataset = dict(conditions=conditions, invalid=invalid, dataset_label=label)
    assert lora["conditions"] == conditions
    a, b = [prompts.analysis_prompt(p, NAVIGATION) for p in (lora, dataset)]
    assert a == b
    assert set(experience_block(a)) == {"UE2"}  # Phase2 RE 不能覆盖 Phase1 障碍。
    assert "change lanes to bypass" in a and "safe legal gap" in a
    dataset["conditions"]["STATIC_OBSTACLE"] = None
    # 即使来源标签仍是障碍，最终条件被 mask 后也不得重新从真值恢复经验。
    assert set(experience_block(prompts.analysis_prompt(dataset, NAVIGATION))) == {"DEFAULT"}


def test_generation_and_review_receive_same_concurrent_experience():
    priors = {"conditions": {
        "ROAD_STRUCTURE": "R4", "STATIC_OBSTACLE": "YES", "UE3": "YES",
        "VULNERABLE": "YES", "TRAFFIC_LIGHT_ABNORMAL": "YES",
    }}
    generation = prompts.analysis_prompt(priors, NAVIGATION)
    review = prompts.review_prompt(priors, NAVIGATION, "draft")
    assert experience_block(generation) == experience_block(review)
    assert set(experience_block(generation)) == {"UE2", "UE3", "UE4", "UE7"}


def test_regular_and_unknown_do_not_invent_special_contexts():
    for rs in (None, "R3", "R5"):
        for value in (None, "NO"):
            c = dict.fromkeys(EVENT_SOURCES, value)
            c.update(ROAD_STRUCTURE=rs, HIGHWAY="YES", RS_HIGHWAY="YES")
            experience = prompts.planning_experience(c)
            assert set(experience) == {"DEFAULT"}
            assert "normal driving" in experience["DEFAULT"]
            assert "missing event conditions remain unknown" in experience["DEFAULT"]
            assert not any(k in str(experience) for k in ("UE7", "RE2", "RE3", "RE5"))
    # 数据集的正常域外 null 不阻止普通驾驶，也不被改写成 NO。
    packed = dataset_labels.pack("R5", "RE", "LOCAL_JUNCTION", None, {
        "HIGHWAY": False, "STATIC_OBSTACLE": False,
        "VULNERABLE": False, "TRAFFIC_LIGHT_ABNORMAL": False,
    })
    c, _, _ = dataset_labels.conditions_from_label(packed)
    assert c["UE3"] is None and c["UE6"] == "NO"
    assert set(prompts.planning_experience(c)) == {"DEFAULT"}


def test_all_event_combinations_fit_fallback_without_dropping_positives():
    for values in itertools.product(("YES", "NO", None), repeat=7):
        conditions = dict(zip(EVENT_SOURCES, values), ROAD_STRUCTURE="R2")
        expected = {EVENT_SOURCES[k] for k, v in conditions.items() if v == "YES"}
        assert set(prompts.planning_experience(conditions)) == (expected or {"DEFAULT"})
        for nav in ("", NAVIGATION, NAVIGATION.replace("(8.0, 2.0)", "(0.0, 0.0)")):
            fallback = prompts.fallback_analysis({"conditions": conditions}, nav)
            assert prompts.analysis_format_valid(fallback), fallback
            planning = fallback.split("Planning context: ", 1)[1]
            assert all(event in planning for event in expected)
            assert "unknown fields remain unresolved" in planning
            if nav:
                assert "4 m/s" in planning and "target:" in planning
