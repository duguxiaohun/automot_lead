"""2026-09-07 RGB 错例暴露的时间语义、评测和索引回归。"""
import json
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3.collection_reader import iter_routes
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group, _split
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import unique_cases, invalid_subgroup_report
from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt
from qwen3vl_local.sft_new_loop_phase3.quality_guards import generation_checkpoint_guards
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import label_actions, RouteTrajectory
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts


@pytest.mark.parametrize("speeds,expected", [
    # RGB #14 / #147：anchor仍等待，随后才释放；旧label把未来释放提前。
    ([0, 0, .054, .899, 1.956, 3.642, 5.358, 6.27, 7.67], "STOP"),
    ([0, .041, .986, 2.082, 3.787, 5.334, 6.432, 8.453, 8.5], "STOP"),
    # RGB #13 / #658：增速后明显制动，隔离而非硬写RESUME/全NO。
    ([2.803, 4.33, 5.374, 4.247, 4.031, 2.69, .385, 1.164, .379], None),
    ([3.99, 5.656, 6.788, 7.836, 6.724, 2.492, 3.776, 5.079, 4.693], None),
    # 连续RGB f123-134：后面的更高速度峰不能掩盖较早的再次近停。
    ([.02406564, 2.09114739, 2.03351813, .50006618, .27486178, .67206298,
      1.98025040, 3.40226449, 2.54411220], None),
    # 已开始连续起步、当前巡航、明确减速仍保留。
    ([.9, 1.956, 3.642, 5.358, 6.27, 7.67, 7.7, 7.8, 7.8], "RESUME"),
    # 新增训练侧逐帧反证：绕障持续增速后正常回调，必须保留RESUME。
    ([5.361, 5.723, 8.106, 8.529, 10.018, 11.501, 11.099, 11.184, 10.162], "RESUME"),
    ([8] * 9, "NONE"),
    ([8, 7, 6, 5, 5, 5, 5, 5, 5], "DECELERATE"),
])
def test_rgb_reviewed_temporal_boundaries(speeds, expected):
    labels = label_actions(dict(future_speeds=speeds, future_speed_count=9))
    if expected is None:
        assert labels is None
    else:
        selected = [k for k in ("STOP", "DECELERATE", "RESUME") if labels[k]]
        assert selected == ([] if expected == "NONE" else [expected])


def test_first_borrow_then_return_is_left(tmp_path):
    # RGB #128 road146 f85=-1 -> f86=+1 -> f95=-1：第一次LEFT，不取终点车道。
    metas = {i: dict(road_id=146, lane_id=(-1 if i == 0 or i >= 10 else 1),
                    lane_type_str="Driving") for i in range(15)}
    route = RouteTrajectory(tmp_path, tuple(metas), metas, {146: -1})
    assert route.lane_change(0) == "LEFT"
    assert route.lane_change(1, horizon=12) == "RIGHT"


def _healthy_metrics():
    metrics = {"slice/invalid_exact": 1, "slice/no_action_exact": 1,
               "slice/no_action_samples": 10, "slice/valid_exact": 1,
               "same_rs_unique_routes": 2,
               "invalid_subgroup/reason/same_rs_wrong_event_exact": 1}
    for key in ("stop", "decelerate", "resume", "lane_change_left", "lane_change_right"):
        metrics.update({f"action/{key}_recall": 1, f"action/{key}_precision": 1,
                        f"action/{key}_gt_yes": 10})
    return metrics


def _guard(metrics):
    return generation_checkpoint_guards(metrics, min_invalid_exact=.8,
        min_lane_change_recall=.6, min_stop_recall=.8, min_no_action_exact=.5)


def test_guard_uses_none_not_ramp_and_blocks_collapsed_actions():
    good = _healthy_metrics()
    good["slice/ramp_merge_exit_exact"] = 0
    assert _guard(good)["all_ok"]
    for key in ("slice/no_action_exact", "action/decelerate_recall", "action/resume_recall",
                "action/stop_precision", "action/resume_precision", "action/decelerate_gt_yes"):
        assert not _guard({**good, key: 0})["all_ok"], key
    assert not _guard({**good, "same_rs_unique_routes": 1})["all_ok"]


def test_duplicate_negative_does_not_increase_independent_coverage():
    row = SimpleNamespace(scenario="s", route_id="r", frame_id=85, context_id="RAMP_MERGE_EXIT",
        true_rs="R3", prompt_road_structure="R3", invalid_reason="same_rs_wrong_event",
        invalid_source="source=RAMP_MERGE_EXIT|true_rs=R3|asked_context=RAMP_MERGE_EXIT")
    assert len(unique_cases([row] * 20)) == 1
    report = invalid_subgroup_report([row] * 20)
    assert report["same_rs_unique_routes"] == 1
    assert report["same_rs_max_case_repeat"] == 20


def test_same_version_changed_rule_source_rejects_cache():
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
        validate_action_rule, ACTION_RULE_VERSION, action_rule_sha256)
    validate_action_rule({'action_evidence': {'rule_version': ACTION_RULE_VERSION,
                                            'rule_code_sha256': action_rule_sha256()}})
    with pytest.raises(ValueError, match='source mismatch'):
        validate_action_rule({'action_evidence': {'rule_version': ACTION_RULE_VERSION,
                                                'rule_code_sha256': 'stale'}})


def test_physical_route_split_preserves_legacy_route_number():
    a = "Town12_Rep0_Town12_route15_route0_01_10_00_00_00"
    b = "Town12_Rep3_Town12_route15_route0_01_11_00_00_00"
    assert physical_route_group("s", a) == physical_route_group("s", b)
    assert physical_route_group("s", a).endswith("Town12_route15")
    assert _split("s", a, 123, .1, .05) == _split("s", b, 123, .1, .05)


def test_stream_reader_keeps_unicode_and_nested_escaped_braces(tmp_path):
    path = tmp_path / "collection.json"
    rows = [{"route_id": '路\\"{}', "annotations": [{"x": [1, 2]}]}, {"route_id": "b"}]
    path.write_text(json.dumps({"routes": rows}, ensure_ascii=False))
    assert list(iter_routes(path, chunk_size=7)) == rows


def test_prompt_has_measured_speed_and_first_crossing_not_future_trace():
    spec = make_prompt_spec(variant="all_random_order", answers={}, seed_key="s",
        context_id="POST_BYPASS_RETURN", road_structure="R1", current_speed_mps=2.125)
    text = build_action_prompt(spec=spec)
    assert "2.125 m/s" in text
    assert "FIRST crossing" in text
    assert "future_speeds" not in text


def test_wrong_junction_quarantine_is_exact_route_not_scenario():
    route = "Town04_Rep0_Town04_Scenario4_14_route0_01_10_02_24_20"
    for frame in (0, 31, 71):
        assert not mapped_contexts("VehicleTurningRoute", route, frame, "R5", "R-E5", ["R-E5"])[0]
    assert mapped_contexts("VehicleTurningRoute", "another", 31, "R5", "R-E5", ["R-E5"])[0]
    assert mapped_contexts("VehicleTurningRoute", route, 72, "R1", "U-E4", ["U-E4"])[0] == ("VULNERABLE_CROSSING",)


def test_generation_metrics_count_unique_none_and_measured_speed(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train as module
    from qwen3vl_local.sft_new_loop_phase3.prompts import build_action_target
    row = module.FrameRow(scenario="s", route_id="r", town="Town01", frame_id=10,
        true_rs="R1", prompt_road_structure="R1", context_id="LEAD_BRAKE",
        question_domain="LONGITUDINAL_YIELD", action_signature="NONE", event="U-E1", split="val",
        goal_ego_xy=(10, 0), history_rgb_paths=["a"]*4, latest_rgb_path="a", answers={},
        current_speed_mps=4.125)
    item = module._make_item(row, seed=0)
    monkeypatch.setattr(module, "_load_images", lambda paths: [])
    observed = []
    monkeypatch.setattr(module, "_kv_start_state", lambda runtime, messages: observed.append(messages))
    monkeypatch.setattr(module, "_student_generate_kv", lambda *args: (build_action_target(item.spec), None, None))
    model = SimpleNamespace(training=False, eval=lambda: None)
    bundle = SimpleNamespace(unwrap=lambda: model, processor=None, tokenizer=None, device="cpu")
    metrics = module.evaluate_generation_probe(bundle, [item, item], history_rgb_mode="4rgb", max_new_tokens=64)
    assert metrics["samples"] == 1
    assert metrics["sampled_before_dedup"] == 2
    assert metrics["slice/no_action_samples"] == 1
    assert metrics["slice/no_action_exact"] == 1
    assert "4.125 m/s" in str(observed)
