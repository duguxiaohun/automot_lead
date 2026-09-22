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
