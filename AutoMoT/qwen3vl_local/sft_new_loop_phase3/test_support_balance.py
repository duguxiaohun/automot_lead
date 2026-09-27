"""稀少动作容量回流：不改变真值，不为补齐配额额外放大稀少格子。"""
import random
from collections import Counter
import pytest
from qwen3vl_local.sft_new_loop_phase3.sampling import support_aware_quota


def test_rare_capacity_returns_to_supported_action():
    assert support_aware_quota({"KEEP": 4, "STOP": 1000}, 100) == {"KEEP": 4, "STOP": 96}
    assert support_aware_quota({"KEEP": 4, "STOP": 1000}, 1500) == {"KEEP": 8, "STOP": 1492}


def test_full_pool_cycles_bound_every_action_repetition():
    for target in range(501):
        capacities = {"a": 1, "b": 4, "c": 20}
        q = support_aware_quota(capacities, target)
        assert sum(q.values()) == target
        assert all(q[k] <= n * ((target + 24) // 25) for k, n in capacities.items())
    with pytest.raises(ValueError):
        support_aware_quota({}, 1)


def test_phase3_builder_does_not_relabel_or_add_random_fallback_repeats():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario="s", route_id=f"r{i}", frame_id=i,
                 action_labels={k: k == "STOP" and i > 0 for k in ACTION_KEYS}) for i in range(101)]
    selected, audit = _sample_context_bucket(rows, context_id="LEAD_BRAKE", target=150,
                                             rng=random.Random(1), route_diverse=True)
    counts = Counter(r["frame_id"] for r in selected)
    assert len(selected) == 150 and counts[0] == 2
    assert max(counts.values()) <= 2
    assert sum(audit["signature_quota"].values()) == 150
    assert not any(rows[0]["action_labels"].values())


def test_signature_support_uses_physical_routes_and_does_not_relabel():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario='s', route_id=f'Town13_Rep{i}_1710_7_route0', frame_id=i,
                 action_labels={k: k == 'STOP' for k in ACTION_KEYS}) for i in range(3)]
    selected, audit = _sample_context_bucket(rows, context_id='LEAD_BRAKE', target=2,
                                             rng=random.Random(1), route_diverse=True)
    support = audit['signature_support']['STOP']
    assert support['frames'] == 3 and support['physical_routes'] == 1
    assert support['presentations'] == 2 and support['diagnostic_only']
    assert all(r['action_labels']['STOP'] for r in selected)


def test_compound_is_not_a_separate_quota_and_binary_evidence_survives():
    from copy import deepcopy
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario='s', route_id='same_route', frame_id=i,
                 action_labels={k: k == 'LANE_CHANGE_LEFT' or (k == 'RESUME' and i == 0)
                                for k in ACTION_KEYS}) for i in range(100)]
    original = deepcopy(rows)
    selected, audit = _sample_context_bucket(rows, context_id='RAMP_MERGE_EXIT', target=150,
                                             rng=random.Random(1), route_diverse=True)
    assert audit['primary_action_quota'] == {'LANE_CHANGE_LEFT': 150}
    assert sum(r['action_labels']['RESUME'] for r in selected) <= 2
    assert rows == original and all(r['action_labels']['LANE_CHANGE_LEFT'] for r in selected)


def test_repeated_recordings_do_not_crowd_out_other_physical_routes():
    from qwen3vl_local.sft_new_loop_phase3.sampling import route_diverse_sample, route_diversity_report
    rows = [dict(scenario='s',route_id=f'Town13_Rep{i}_1710_7_route0',frame_id=4) for i in range(20)]
    rows.append(dict(scenario='s',route_id='Town13_Rep0_9999_7_route0',frame_id=4))
    selected = route_diverse_sample(rows,target=2,rng=random.Random(4))
    assert any('9999' in r['route_id'] for r in selected)
    assert route_diversity_report(rows)['unique_routes']==2
