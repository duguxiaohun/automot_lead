"""不能靠换真值、删困难样本或换 RGB 获得配对提升。"""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase3.paired_eval import compare, identity, temporal_slices


def case():
    return dict(scenario="s", route_id="r", frame_id=1, context_id="STOP",
        prompt_road_structure="R1", augment_variant="all_random_order", gt={"STOP":"YES"},
        true_rs="R1", goal_ego_xy=[1, 2], context_detail="", history_rgb_mode="4rgb",
        history_rgb_selected_indices=[0,1,2,3], history_rgb_paths_used=["s/r/rgb/0001.jpg"]*4,
        history_rgb_sha256=["abc"]*4,
        action_evidence=dict(future_speeds_exact_mps=[0]*9), all_ok=False)


@pytest.mark.parametrize("field,value", [("gt", {"STOP":"NO"}),
    ("history_rgb_paths_used", ["s/r/rgb/0002.jpg"]*4),
    ("history_rgb_sha256", ["changed"]*4), ("history_rgb_sha256", None),
    ("goal_ego_xy", [2,1])])
def test_changed_targets_or_inputs_cannot_be_scored(field, value):
    a = case(); b = deepcopy(a); b[field] = value
    with pytest.raises(ValueError, match="mismatch|missing"):
        compare({identity(a):a}, {identity(a):b})


def test_no_intersection_only_scoring_and_correct_flip_counts():
    a = case(); b = deepcopy(a); b["all_ok"] = True
    with pytest.raises(ValueError, match="case set mismatch"):
        compare({identity(a):a}, {})
    report = compare({identity(a):a}, {identity(b):b})
    assert report["exact_delta"] == 1 and report["excluded_cases"] == 0
    assert report["rgb_hash_verified_cases"] == 1


def test_no_string_is_not_true_in_temporal_slices():
    row = case()
    row["history_rgb_paths_all4"] = ["a", "b", "c", "d"]
    row["gt"] = {"STOP":"NO", "DECELERATE":"NO", "RESUME":"YES", "INVALID_ACTION_CONTEXT":"NO"}
    row["action_evidence"]["future_speeds_exact_mps"] = [0,.8,2,3,4,5,6,7,8]
    assert set(temporal_slices(row)) == {"stationary_anchor_future_resume", "isolated_near_stop_in_1_5s"}
    row["gt"]["INVALID_ACTION_CONTEXT"] = "YES"
    assert temporal_slices(row) == []
