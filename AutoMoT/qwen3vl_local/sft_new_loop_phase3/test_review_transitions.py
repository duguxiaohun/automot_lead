"""逐帧审计不能把历史动作、道路过渡或缺帧变成未来已确认跨线。"""
from qwen3vl_local.sft_new_loop_phase3.audit_review_transitions import lane_identity_timeline


def rows(lanes):
    return [dict(frame=i, road_id=10, section_id=0, lane_id=lane)
            for i, lane in enumerate(lanes)]


def test_first_departure_and_later_return_remain_separate():
    result = lane_identity_timeline(rows([1, 1, -1, -1, 1, 1]), 1)
    first, second = result['events']
    assert first['time_s'] == .25 and first['confirmation_time_s'] == .5
    assert not first['returns_to_anchor_identity']
    assert second['time_s'] == .75 and second['returns_to_anchor_identity']
    assert not result['direction_inferred_from_ids']


def test_crossing_at_anchor_is_history_and_last_point_is_unconfirmed():
    first, last = lane_identity_timeline(rows([2, 1, 1, 2]), 1)['events']
    assert first['in_model_history'] and first['two_sample_confirmed']
    assert not last['in_model_history'] and not last['two_sample_confirmed']
    assert last['confirmation_time_s'] is None


def test_road_transition_is_not_comparable_lane_change():
    values = rows([1, 2, 2])
    values[1]['road_id'] = values[2]['road_id'] = 11
    event = lane_identity_timeline(values, 0)['events'][0]
    assert event['two_sample_confirmed'] and not event['same_road_section']


def test_missing_frame_and_repeated_startup_do_not_invent_crossing():
    values = rows([1, 2, 2])
    assert not lane_identity_timeline([values[0], values[0], values[2]], 0)['events']
