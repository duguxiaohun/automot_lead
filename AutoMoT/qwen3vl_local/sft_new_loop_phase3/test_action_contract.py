#!/usr/bin/env python3
"""sft_new_loop_phase3 的合同测试：上下文映射、动作标定、prompt 与严格解析。

从 AutoMoT/ 目录运行：

```bash
python -m pytest qwen3vl_local/sft_new_loop_phase3/test_action_contract.py -q
```
"""

from __future__ import annotations

import pathlib
import random
import sys

import pytest

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (  # noqa: E402
    ACTION_KEYS,
    CONTEXT_BY_ID,
    CONTEXT_IDS,
    DOMAIN_ACTION_KEYS,
    DOMAIN_LONGITUDINAL,
    POST_BYPASS_MAX_GAP_FRAMES,
    abnormal_events_from_answers,
    context_for_event,
    event_from_answers,
    resolve_context_id,
    resolve_context_ids,
    road_structure_from_answers,
)
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (  # noqa: E402
    balanced_invalid_items,
    mismatched_contexts,
)
from qwen3vl_local.sft_new_loop_phase3.navigation_goal import goal_sentence, goal_side  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.prompts import (  # noqa: E402
    ANSWER_KEYS,
    INVALID_KEY,
    action_prompt_sha256,
    build_action_prompt,
    build_action_target,
    make_prompt_spec,
    parse_action_output,
    spec_answers,
)
from qwen3vl_local.sft_new_loop_phase3.sampling import even_quota_with_capacity  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (  # noqa: E402
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    LONGITUDINAL_HORIZON_FRAMES,
    action_evidence,
    label_actions,
    lane_change_direction_from_ids,
)
from qwen3vl_local.sft_new_loop_phase3.build_dataset import action_signature  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _balanced_invalid_rows  # noqa: E402


def _no_answers() -> dict:
    """返回全 NO 的答案字典。"""

    return {key: False for key in ANSWER_KEYS}


def _signals(**overrides) -> dict:
    """构造一份可用的动作信号；默认是稳定巡航。"""

    base = {
        "frame_id": 10,
        "speed": 8.0,
        "speed_min": 8.0,
        "speed_max": 8.0,
        "immediate_speed_min": 8.0,
        "immediate_speed_max": 8.0,
        "future_speed_count": LONGITUDINAL_HORIZON_FRAMES + 1,
        "future_speeds": [8.0] * 9,
        "brake": False,
        "throttle": 0.5,
        "speed_limit": 8.33,
        "lane_id": -1,
        "road_id": 4,
        "lane_width": 3.5,
        "is_junction": False,
        "distance_to_next_junction": 90.0,
        "changed_route": False,
        "lateral_shift": 0.0,
        "lane_change_direction": None,
        "goal_x": 60.0,
        "goal_y": -1.0,
        "goal_available": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Phase1/Phase2 答案 -> RS / EVENT / 上下文
# ---------------------------------------------------------------------------


def test_road_structure_is_uniquely_recovered_from_phase1_answers() -> None:
    """四个 RS 问题各自唯一确定 R1/R2/R4/R5，全 NO + HIGHWAY 确定 R3。"""

    assert road_structure_from_answers({"RS1": True}) == "R1"
    assert road_structure_from_answers({"RS2": True}) == "R2"
    assert road_structure_from_answers({"RS4": True}) == "R4"
    assert road_structure_from_answers({"RS5": True}) == "R5"
    assert road_structure_from_answers({"HIGHWAY": True}) == "UNKNOWN"
    assert road_structure_from_answers(dict(RS1=False, RS2=False, RS4=False, RS5=False, HIGHWAY=False)) == "R3"
    assert road_structure_from_answers({"RS_HIGHWAY": True}) == "R3"
    assert road_structure_from_answers({}) == "UNKNOWN"
    assert road_structure_from_answers({"RS1": True, "RS4": True}) == "UNKNOWN"


def test_abnormal_events_are_recovered_but_regular_events_are_not_invented() -> None:
    """Phase1/2 覆盖七个异常 UE；all-NO 绝不能伪造 R-E*。"""

    assert event_from_answers({"RS1": True, "STATIC_OBSTACLE": True}) == "U-E2"
    assert event_from_answers({"RS1": True, "VULNERABLE": True}) == "U-E4"
    assert event_from_answers({"RS4": True, "TRAFFIC_LIGHT_ABNORMAL": True}) == "U-E7"
    assert event_from_answers({"RS1": True, "UE1": True, "INVALID_EVENT_CONTEXT": False}) == "U-E1"
    assert event_from_answers({"RS1": True, "UE3": True, "INVALID_EVENT_CONTEXT": False}) == "U-E3"
    assert event_from_answers({"RS2": True, "UE5": True, "INVALID_EVENT_CONTEXT": False}) == "U-E5"
    assert event_from_answers({"RS4": True, "UE6": True, "INVALID_EVENT_CONTEXT": False}) == "U-E6"
    assert event_from_answers({"RS4": True}) == "UNKNOWN"
    assert event_from_answers({"RS5": True}) == "UNKNOWN"
    assert event_from_answers({"HIGHWAY": True}) == "UNKNOWN"
    assert abnormal_events_from_answers({"RS1": True, "STATIC_OBSTACLE": True, "UE1": True, "INVALID_EVENT_CONTEXT": False}) == (
        "U-E2",
        "U-E1",
    )
    assert event_from_answers({"RS1": True, "STATIC_OBSTACLE": True, "UE1": True, "INVALID_EVENT_CONTEXT": False}) == "UNKNOWN"
    assert abnormal_events_from_answers({"UE3": True, "INVALID_EVENT_CONTEXT": True}) == ()


def test_every_ue_maps_to_exactly_one_action_context() -> None:
    """七个可唯一确定的 U-E 各自映射到一个上下文；U-E8 不进入本阶段。"""

    for code in ("U-E1", "U-E2", "U-E3", "U-E4", "U-E5", "U-E6", "U-E7"):
        context = context_for_event(code)
        assert context is not None, code
    assert context_for_event("U-E8") is None
    assert len(CONTEXT_IDS) == 10
    assert len(set(CONTEXT_IDS)) == 10


def test_longitudinal_contexts_never_offer_lane_change_lines() -> None:
    """纵向让行上下文只问减速/停车/恢复。"""

    for context_id in CONTEXT_IDS:
        context = CONTEXT_BY_ID[context_id]
        keys = set(context.action_keys)
        if context.question_domain == DOMAIN_LONGITUDINAL:
            assert keys == {"DECELERATE", "STOP", "RESUME"}, context_id
        else:
            assert keys == set(ACTION_KEYS), context_id


def test_longitudinal_signature_ignores_unasked_lane_change() -> None:
    """纵向桶不能被未提问的横向轨迹拆成稀有 signature。"""

    labels = {key: False for key in ACTION_KEYS}
    labels["DECELERATE"] = True
    labels["LANE_CHANGE_LEFT"] = True
    assert action_signature(labels, context_id="LEAD_BRAKE") == "DECELERATE"
    assert action_signature(labels, context_id="STATIC_BLOCKAGE") == "DECELERATE+LANE_CHANGE_LEFT"


def test_resolve_context_id_matches_rs_and_event_pairing() -> None:
    """RS+EVENT 折叠成上下文时必须尊重 RS 白名单、绕障窗口与 R3 匝道口径。"""

    assert resolve_context_id("R3", ["R-E2"], None) == "POST_BYPASS_RETURN"
    assert resolve_context_id("R3", ["R-E3"], None) == "RAMP_MERGE_EXIT"
    assert resolve_context_id("R2", ["R-E2"], 35) == "POST_BYPASS_RETURN"
    assert resolve_context_id("R4", ["R-E4", "U-E6"]) == "JUNCTION_RULE_CONFLICT"
    assert resolve_context_id("R1", ["U-E4", "U-E2"]) is None
    assert set(resolve_context_ids("R1", ["U-E4", "U-E2"])) == {"STATIC_BLOCKAGE", "VULNERABLE_CROSSING"}
    assert resolve_context_id("R4", ["U-E2"]) == "STATIC_BLOCKAGE"
    assert resolve_context_id("R2", ["U-E4"]) == "VULNERABLE_CROSSING"
    assert resolve_context_id("R5", ["U-E7", "R-E5"]) == "UNSIGNALIZED_PRIORITY"
    assert resolve_context_id("R1", ["R-E1"]) is None



# ---------------------------------------------------------------------------
# 轨迹 -> 动作标签
# ---------------------------------------------------------------------------


def test_steady_cruising_produces_no_action() -> None:
    """稳定巡航时五个动作全 NO。"""

    labels = label_actions(_signals())
    assert labels == {key: False for key in ACTION_KEYS}


def test_stop_wins_over_decelerate_when_ego_comes_to_rest() -> None:
    """即时窗内会静止且保持静止时是 STOP，不是 DECELERATE。"""

    labels = label_actions(
        _signals(future_speeds=[6, 4, 2, .2, .1, 0, 0, 0, 0], speed=6.0, speed_min=0.0, speed_max=6.0, immediate_speed_min=0.0, immediate_speed_max=1.2)
    )
    assert labels["STOP"] is True
    assert labels["DECELERATE"] is False
    assert labels["RESUME"] is False


def test_already_stopped_but_pulling_away_is_resume_not_stop() -> None:
    """已经停稳但即将起步时是 RESUME，不再继续 STOP。"""

    labels = label_actions(
        _signals(future_speeds=[0, .5, 1, 2, 3, 4, 5, 6, 6], speed=0.0, speed_min=0.0, speed_max=6.0, immediate_speed_min=0.0, immediate_speed_max=5.0)
    )
    assert labels["STOP"] is False
    assert labels["RESUME"] is True
    assert labels["DECELERATE"] is False


def test_slowdown_without_rest_is_decelerate() -> None:
    """明显减速但未静止时是 DECELERATE。"""

    labels = label_actions(
        _signals(future_speeds=[9, 8, 7, 6, 5.5, 5, 5, 5, 5], speed=9.0, speed_min=5.0, speed_max=9.0, immediate_speed_min=5.5, immediate_speed_max=9.0)
    )
    assert labels["DECELERATE"] is True
    assert labels["STOP"] is False
    assert labels["RESUME"] is False


def test_longitudinal_actions_stay_mutually_exclusive() -> None:
    """随机窗口下 DECELERATE/STOP/RESUME 最多只有一个为 YES。"""

    rng = random.Random(20260904)
    for _ in range(400):
        speed = rng.uniform(0.0, 18.0)
        low = max(0.0, speed - rng.uniform(0.0, speed + 2.0))
        high = speed + rng.uniform(0.0, 6.0)
        labels = label_actions(
            _signals(
                future_speeds=[speed] + [rng.uniform(low, high) for _ in range(8)],
                speed=speed,
                speed_min=low,
                speed_max=high,
                immediate_speed_min=rng.uniform(low, speed),
                immediate_speed_max=rng.uniform(speed, high),
            )
        )
        assert sum(int(labels[key]) for key in ("DECELERATE", "STOP", "RESUME")) <= 1


def test_short_future_window_yields_no_label() -> None:
    """未来帧不足时不生成动作标签，避免用截断轨迹伪造监督。"""

    assert label_actions(_signals(future_speed_count=LONGITUDINAL_HORIZON_FRAMES)) is None


def test_lane_change_direction_uses_entry_lane_not_current_lane() -> None:
    """借对向车道是左变道，回原车道是右变道。"""

    assert lane_change_direction_from_ids(-1, 1, -1) == DIRECTION_LEFT
    assert lane_change_direction_from_ids(1, -1, -1) == DIRECTION_RIGHT
    assert lane_change_direction_from_ids(-3, -2, -3) == DIRECTION_LEFT
    assert lane_change_direction_from_ids(-3, -4, -3) == DIRECTION_RIGHT
    assert lane_change_direction_from_ids(-1, -1, -1) is None


def test_lane_change_direction_flips_for_opposite_travel_direction() -> None:
    """自车沿 -s 行驶时，lane id 增大是向右。"""

    assert lane_change_direction_from_ids(2, 3, 2) == DIRECTION_RIGHT
    assert lane_change_direction_from_ids(2, 1, 2) == DIRECTION_LEFT


def test_lane_change_labels_follow_detected_direction() -> None:
    """检测到的方向唯一决定左右变道行。"""

    left = label_actions(_signals(lane_change_direction=DIRECTION_LEFT))
    right = label_actions(_signals(lane_change_direction=DIRECTION_RIGHT))
    assert (left["LANE_CHANGE_LEFT"], left["LANE_CHANGE_RIGHT"]) == (True, False)
    assert (right["LANE_CHANGE_LEFT"], right["LANE_CHANGE_RIGHT"]) == (False, True)


def test_action_evidence_records_auditable_fields() -> None:
    """索引里必须留下可复核的速度、车道与阈值证据。"""

    evidence = action_evidence(_signals(lane_change_direction=DIRECTION_LEFT))
    for key in (
        "speed_mps",
        "future_speed_min_mps",
        "future_speed_max_mps",
        "longitudinal_threshold_mps",
        "lane_change_direction",
        "lane_id",
        "road_id",
        "horizon_frames",
    ):
        assert key in evidence


# ---------------------------------------------------------------------------
# 导航目标坐标口径
# ---------------------------------------------------------------------------


def test_goal_side_uses_carla_left_handed_frame() -> None:
    """y 负为左、y 正为右，接近 0 时视为正前方。"""

    assert goal_side(-8.0) == "left"
    assert goal_side(8.0) == "right"
    assert goal_side(0.4) == "straight ahead"
    assert "left" in goal_sentence(40.0, -12.0)
    assert "right" in goal_sentence(40.0, 12.0)


# ---------------------------------------------------------------------------
# Prompt 与严格解析
# ---------------------------------------------------------------------------


def _spec(context_id: str = "STATIC_BLOCKAGE", **answers):
    """构造一次 prompt spec。"""

    payload = _no_answers()
    payload.update(answers)
    context = CONTEXT_BY_ID[context_id]
    return make_prompt_spec(
        variant="all_random_order",
        answers=payload,
        seed_key=f"test:{context_id}",
        context_id=context_id,
        road_structure=context.allowed_rs[0],
        goal_xy=(42.0, -3.0),
    )


def test_prompt_contains_context_goal_and_only_asked_action_lines() -> None:
    """prompt 必须带上场景前提与目标坐标，并只列出该域的动作行。"""

    spec = _spec("LEAD_BRAKE")
    text = build_action_prompt(spec=spec)
    assert "[SCENE_CONTEXT]" in text
    assert "[NAVIGATION_GOAL]" in text
    assert "ROUTE_TARGET_XY" in text
    assert set(spec.output_keys) == {"DECELERATE", "STOP", "RESUME", INVALID_KEY}
    assert "LANE_CHANGE_LEFT" not in text


def test_prompt_never_leaks_dataset_codes() -> None:
    """prompt 不能出现 RS/EVENT 的数据集 code。"""

    for context_id in CONTEXT_IDS:
        text = build_action_prompt(spec=_spec(context_id))
        for token in ("U-E", "R-E", "RS1", "RS2", "RS4", "RS5", "UE1", "UE3", "UE5", "UE6"):
            assert token not in text, (context_id, token)


def test_prompt_forbids_curved_lane_as_lane_change_evidence() -> None:
    """横向边界必须显式排除弯道/转向角。"""

    text = build_action_prompt(spec=_spec("POST_BYPASS_RETURN"))
    assert "curved lane" in text
    assert "Steering input" in text


def test_target_and_strict_parser_round_trip() -> None:
    """渲染出的 target 必须能被严格解析器完整还原。"""

    spec = _spec("STATIC_BLOCKAGE", LANE_CHANGE_LEFT=True, DECELERATE=True)
    target = build_action_target(spec)
    parsed = parse_action_output(target, spec=spec)
    assert parsed == spec_answers(spec)


def test_strict_parser_rejects_reordered_missing_and_trailing_text() -> None:
    """行乱序、缺行或尾随解释都必须整条失效。"""

    spec = _spec("DYNAMIC_CUTIN", DECELERATE=True)
    target = build_action_target(spec)
    lines = target.splitlines()
    assert all(value is None for value in parse_action_output("\n".join(reversed(lines)), spec=spec).values())
    assert all(value is None for value in parse_action_output("\n".join(lines[:-1]), spec=spec).values())
    assert all(
        value is None for value in parse_action_output(target + "\nBecause the lead brakes.", spec=spec).values()
    )


def test_audit_parser_requires_bounded_evidence_lines() -> None:
    """audit 模式必须逐行给出非空且不超过 14 词的证据。"""

    spec = _spec("VULNERABLE_CROSSING", STOP=True)
    target = build_action_target(spec)
    good = target + "\n" + "\n".join(f"EVIDENCE_{key}: pedestrian still inside ego path" for key in spec.output_keys)
    assert parse_action_output(good, spec=spec, audit=True) == spec_answers(spec)
    empty = target + "\n" + "\n".join(f"EVIDENCE_{key}: " for key in spec.output_keys)
    assert all(value is None for value in parse_action_output(empty, spec=spec, audit=True).values())
    long_cue = " ".join(["word"] * 15)
    too_long = target + "\n" + "\n".join(f"EVIDENCE_{key}: {long_cue}" for key in spec.output_keys)
    assert all(value is None for value in parse_action_output(too_long, spec=spec, audit=True).values())


def test_invalid_context_target_is_all_no_actions() -> None:
    """invalid 行的所有动作必须为 NO。"""

    spec = _spec("RAMP_MERGE_EXIT", **{INVALID_KEY: True})
    answers = spec_answers(spec)
    assert answers[INVALID_KEY] is True
    assert all(answers[key] is False for key in spec.output_keys if key != INVALID_KEY)


def test_prompt_fingerprint_is_stable_and_mode_sensitive() -> None:
    """同一 prompt 表面指纹稳定，且随 RGB 模式变化。"""

    assert action_prompt_sha256() == action_prompt_sha256()
    assert action_prompt_sha256(history_rgb_mode="4rgb") != action_prompt_sha256(
        history_rgb_mode="2rgb_endpoints"
    )
    assert action_prompt_sha256(audit=False) != action_prompt_sha256(audit=True)


def test_prompt_question_order_is_deterministic_per_seed() -> None:
    """同一 seed 的问题顺序可复现，不同 seed 会重排。"""

    context = CONTEXT_BY_ID["STATIC_BLOCKAGE"]
    orders = set()
    for seed in ("a", "b", "c", "d", "e", "f"):
        spec = make_prompt_spec(
            variant="all_random_order",
            answers=_no_answers(),
            seed_key=seed,
            context_id="STATIC_BLOCKAGE",
            road_structure=context.allowed_rs[0],
            goal_xy=(10.0, 0.0),
        )
        assert spec.output_keys[-1] == INVALID_KEY
        orders.add(spec.output_keys)
    assert len(orders) > 1


# ---------------------------------------------------------------------------
# INVALID 构造与配额
# ---------------------------------------------------------------------------


def test_mismatched_contexts_need_hard_geometry_evidence() -> None:
    """错配上下文只能由几何硬约束构造，不能靠场景名。"""

    assert mismatched_contexts(true_rs="R1", is_junction=False, distance_to_next_junction=60.0) == (
        "JUNCTION_RULE_CONFLICT",
        "SIGNAL_FAILURE",
        "RAMP_MERGE_EXIT",
    )
    assert mismatched_contexts(true_rs="R1", is_junction=False, distance_to_next_junction=5.0) == ()
    assert mismatched_contexts(true_rs="R3", is_junction=False, distance_to_next_junction=999.0) == (
        "JUNCTION_RULE_CONFLICT",
        "SIGNAL_FAILURE",
    )
    assert mismatched_contexts(true_rs="R4", is_junction=True, distance_to_next_junction=0.0) == (
        "RAMP_MERGE_EXIT",
    )


def test_even_quota_respects_capacity_and_total() -> None:
    """动作签名配额不能超过各桶容量，且总量守恒。"""

    quotas = even_quota_with_capacity({"a": 2, "b": 10, "c": 0}, 8)
    assert quotas["a"] <= 2
    assert quotas["c"] == 0
    assert sum(quotas.values()) == 8


def test_invalid_builder_keeps_true_rs_and_wrong_context_coverage() -> None:
    """小 INVALID 桶也必须实际抽到 R1--R5 和全部三种错误问题域。"""

    bases = []
    for index, rs in enumerate(("R1", "R2", "R3", "R4", "R5")):
        context_id = "RAMP_MERGE_EXIT" if rs == "R3" else "STATIC_BLOCKAGE"
        bases.append(
            {
                "context_id": context_id,
                "rs": rs,
                "is_junction": rs in {"R4", "R5"},
                "distance_to_next_junction": 100.0,
                "scenario": "unit",
                "route_id": f"route{index}",
                "town": "Town01",
                "split": "train",
                "frame_id": index,
                "primary_event": "R-E1",
                "event_codes": ("R-E1",),
                "action_labels": {key: False for key in ACTION_KEYS},
                "goal_x": 20.0,
                "goal_y": 0.0,
                "action_evidence": {},
                "visual_label_risk": False,
                "visual_label_risk_reasons": (),
                "history_rgb_paths": ["x.jpg"] * 4,
                "latest_rgb_path": "x.jpg",
            }
        )
    # 生产构建与运行时现在都要求十个来源 context，不再允许 builder 放过缺桶。
    bases = [{**base, "context_id": source} for base in bases for source in CONTEXT_IDS]
    rows, report = _balanced_invalid_rows(
        bases, split="train", target=32, rng=random.Random(7), require_true_rs_coverage=True
    )
    assert {row["true_rs"] for row in rows} == {"R1", "R2", "R3", "R4", "R5"}
    assert {row["context_id"] for row in rows} >= {
        "JUNCTION_RULE_CONFLICT",
        "SIGNAL_FAILURE",
        "RAMP_MERGE_EXIT",
    }
    assert report["guards"]["required_true_rs_present"]


def test_invalid_runtime_sampler_keeps_sampled_coverage() -> None:
    """train/eval 的二次均衡也不能把 R2/R5 从实际 batch 中抽没。"""

    class Row:
        def __init__(self, source: str, rs: str, asked: str):
            self.invalid_source = f"source={source}|true_rs={rs}|asked_context={asked}"
            self.true_rs = rs
            self.context_id = asked

    rows = [Row(source, rs, asked) for source in CONTEXT_IDS
            for rs in ("R1", "R2", "R3", "R4", "R5") for asked in CONTEXT_IDS]
    sampled = balanced_invalid_items(rows, target=32, rng=random.Random(8), require_coverage=True)
    assert {item.context_id for item in sampled} == set(CONTEXT_IDS)

    assert {item.true_rs for item in sampled} == {"R1", "R2", "R3", "R4", "R5"}
    assert {item.context_id for item in sampled} >= {
        "JUNCTION_RULE_CONFLICT",
        "SIGNAL_FAILURE",
        "RAMP_MERGE_EXIT",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(_THIS), "-q"]))
