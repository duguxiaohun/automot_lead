from collections import Counter
from copy import deepcopy
import pytest
from qwen3vl_local.action_prior.event_balance import equal_event_evaluation,SPECIAL_BUCKETS


def rows():
    return [dict(scenario='S',run_id=f'route{e}',frame_id=i,
        event_balance_all_special_buckets=[e],event_balance_status='special_eligible')
        for k,e in enumerate(SPECIAL_BUCKETS) for i in range(3+k)]


def test_full_evaluation_is_equal_fixed_no_replacement_and_source_unchanged():
    source=rows();before=deepcopy(source)
    chosen,report=equal_event_evaluation(source,seed=71)
    assert source==before
    assert report['per_event']==3
    assert Counter(r['_evaluation_event'] for r in chosen)==dict.fromkeys(SPECIAL_BUCKETS,3)
    assert len({(r['scenario'],r['run_id'],r['frame_id']) for r in chosen})==30
    assert equal_event_evaluation(source,seed=71)==(chosen,report)
    assert equal_event_evaluation(source,seed=71,max_samples=26)[1]['per_event']==2


def test_shared_transitions_are_not_counted_twice_in_denominator():
    source=[dict(scenario='S',run_id='shared',frame_id=i,
        event_balance_all_special_buckets=list(SPECIAL_BUCKETS)) for i in range(20)]
    chosen,report=equal_event_evaluation(source,seed=1)
    assert report['per_event']==2 and len({r['frame_id'] for r in chosen})==20
    assert all(len(r['event_balance_all_special_buckets'])==10 for r in chosen)


def test_missing_events_and_tiny_budget_are_rejected():
    with pytest.raises(ValueError,match='missing'):equal_event_evaluation(rows()[3:],seed=1)
    with pytest.raises(ValueError,match='at least ten'):equal_event_evaluation(rows(),seed=1,max_samples=9)


def test_equal_event_score_uses_assigned_bucket_without_background():
    from qwen3vl_local.action_prior.metrics import event_grouped_counts, metrics_from_counts
    selected, _ = equal_event_evaluation(rows(), seed=1)
    totals = Counter(samples=len(selected))
    for row in selected:
        row['event_balance_bucket'] = 'REGULAR_BACKGROUND'  # old training metadata must not override eval assignment
        totals.update(event_grouped_counts(row, dict(route_ade_m=2., waypoint_ade_m=4.,
            route_fde_m=3., waypoint_fde_m=5.)))
    result = metrics_from_counts(totals)
    assert result['event_balance_bucket_coverage_complete'] == 1
    assert result['event_balanced_trajectory_score'] == 3.
    assert 'group/event_balance/REGULAR_BACKGROUND/samples' not in result
    natural = Counter(samples=1)
    natural.update(event_grouped_counts(dict(event_balance_bucket=SPECIAL_BUCKETS[0]), {}))
    assert metrics_from_counts(natural)['event_balance_bucket_coverage_complete'] == 0
