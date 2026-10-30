"""Objective highway cut-in spans: detector, v1 merge rules, committed v2 invariants and builder override."""
import json
from pathlib import Path
import types

import pytest
from qwen3vl_local.sft_new_loop_phase2 import build_dataset, highway_ue3_audit
from qwen3vl_local.sft_new_loop_phase2 import highway_ue3_objective_spans as hw

V1 = Path(hw.__file__).with_name('highway_ue3_rgb_decisions_v1.jsonl')
V2 = Path(hw.__file__).with_name('highway_ue3_rgb_decisions_v2.jsonl')


def _track(points):
    return [dict(f=f, x=x, y=y, yaw=yaw, lane=lane, road=1, ego_lane=4, ego_road=1, lane_width=3.5) for f, x, y, yaw, lane in points]


def test_detector_right_side_entry_and_settle():
    pts = [(f, 12.0, 3.5, 0.0, 5) for f in range(0, 10)]
    pts += [(10 + i, 11.0, 3.5 - 0.35 * (i + 1), -15.0, 5 if i < 4 else 4) for i in range(9)]
    pts += [(19 + i, 10.0, 0.3 - 0.05 * i, -3.0, 4) for i in range(6)]
    d = hw.detect_span(_track(pts))
    assert d['side'] == 'right' and d['onset'] == 10 and d['settle'] == 19 and d['lane_change'] == 14
    assert hw.span_end(d) == 19


def test_detector_adjacent_lane_merge_never_fires():
    pts = [(f, 5.0 + 0.1 * f, -5.3 + 0.04 * f, 7.0 - 0.1 * f, -4) for f in range(0, 50)]
    d = hw.detect_span(_track(pts))
    assert d['side'] == 'left' and d['onset'] is None and hw.span_end(d) is None


def test_detector_without_settle_caps_span():
    pts = [(f, 12.0, 3.5, 0.0, 5) for f in range(0, 10)] + [(10 + i, 11.0, 3.0 - 0.1 * i, -12.0, 5) for i in range(30)]
    d = hw.detect_span(_track(pts))
    assert d['onset'] == 10 and d['settle'] is None and hw.span_end(d) == 30


def test_merge_keeps_test_rows_and_downgrades_superseded_positives(tmp_path):
    trace = {'Town13_Rep0_1586_9_route0_01_08_19_01_08': {'side': 'left', 'onset': None},
             'Town13_Rep0_1586_11_route0_01_09_19_15_55': {'side': 'left', 'onset': None},
             'Town13_Rep0_1283_10_route0_01_08_20_13_30': {'side': 'right', 'onset': 110}}
    proposal = {'rows': [hw._row('Town12_Rep0_2944_0_route0_01_10_10_49_40', 98, 111, 'YES', 'train', 'cue', 'note', true_rs_override='R3'),
                         hw._row('Town13_Rep0_1283_10_route0_01_08_20_13_30', 110, 120, 'YES', 'val', 'cue', 'note', true_rs_override='R3')],
                'trace': trace, 'unlabeled_test_windows': []}
    with pytest.raises(ValueError):
        hw.merge_with_v1(V1, {**proposal, 'rows': proposal['rows'][:1]})  # 1283_10 has an onset but no objective rows
    rows = hw.merge_with_v1(V1, proposal)
    v1 = [json.loads(l) for l in V1.read_text().splitlines() if l.strip()]
    srt = lambda rs: sorted((r for r in rs if r['expected_split'] == 'test'), key=lambda r: r['start_frame'])
    assert srt(rows) == srt(v1)
    downgraded = [r for r in rows if r.get('superseded_v1_answer') == 'YES']
    assert {r['route_id'][:18] for r in downgraded} == {'Town13_Rep0_1586_9', 'Town13_Rep0_1586_1'} and all(r['ue3_answer'] == 'NO' for r in downgraded)
    assert [r for r in rows if '1283_10' in r['route_id']] == [proposal['rows'][1]]  # val route comes from the objective proposal, not v1
    out = tmp_path / 'merged.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    positives, report = highway_ue3_audit.load_highway_ue3_decisions(out)
    assert report['true_rs_override_spans'] == 2 and len(positives) == 14 + 11 + 21


def test_committed_v2_invariants():
    positives, report = highway_ue3_audit.load_highway_ue3_decisions(V2)
    rows = [json.loads(l) for l in V2.read_text().splitlines() if l.strip()]
    v1 = [json.loads(l) for l in V1.read_text().splitlines() if l.strip()]
    srt = lambda rs: sorted((r for r in rs if r['expected_split'] == 'test'), key=lambda r: r['start_frame'])
    assert srt(rows) == srt(v1)
    yes = [r for r in rows if r['ue3_answer'] == 'YES']
    assert all(r.get('true_rs_override') == 'R3' for r in yes if r['expected_split'] != 'test')
    assert not any('1586' in r['route_id'] for r in yes)
    assert len({r['route_id'] for r in yes if r['expected_split'] == 'train'}) >= 40
    assert len({r['route_id'][:6] for r in yes if r['expected_split'] == 'train'}) == 2  # Town12 + Town13
    assert report['split_frame_counts']['train/YES'] >= 500 and report['split_frame_counts']['val/YES'] >= 40
    for r in yes:
        if r['expected_split'] != 'test':
            o = r['objective']
            assert abs(o['y_at_onset']) < hw.ONSET_MAX_ABS_Y and abs(o['yaw_at_onset']) >= hw.ONSET_MIN_YAW and r['start_frame'] == o['onset']
    for r in rows:
        assert r['expected_split'] == hw.route_split(r['route_id'], seed=20260819, test_ratio=0.10, val_ratio=0.05)


@pytest.mark.parametrize('bad', [('R4', 'YES'), ('R3', 'NO')])
def test_loader_rejects_illegal_override(tmp_path, bad):
    row = dict(scenario='HighwayCutIn', route_id='r', start_frame=1, end_frame=2, ue3_answer=bad[1], expected_split='train',
               rgb_review='x', visible_cue='y', true_rs_override=bad[0])
    p = tmp_path / 'd.jsonl'; p.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError):
        highway_ue3_audit.load_highway_ue3_decisions(p)


def _collection(tmp_path, rs, event):
    data = tmp_path / 'lead'; coll = tmp_path / 'coll'; coll.mkdir()
    route = 'Town12_Rep0_1_0_route0_01_01_00_00_00'
    for f in range(12):
        p = data / 'HighwayCutIn' / route / 'rgb' / f'{f:04d}.jpg'; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'jpg')
    anns = [dict(frame_id=f, primary_road_structure=rs, primary_event=event, frame_event_annotation={'events': [event]}, evidence={'xml_town': 'Town12'}) for f in range(12)]
    (coll / 'HighwayCutIn_result.json').write_text(json.dumps({'routes': [dict(route_id=route, status='ok', annotations=anns)]}))
    return data, coll, route


@pytest.mark.parametrize('rs,event', [('R4', 'R-E4'), ('R3', 'R-E1')])
def test_builder_applies_true_rs_override_only_on_decision_frames(tmp_path, rs, event):
    data, coll, route = _collection(tmp_path, rs, event)
    split = hw.route_split(route, seed=20260819, test_ratio=0.10, val_ratio=0.05)
    decisions = {('HighwayCutIn', route, f): dict(expected_split=split, true_rs_override='R3') for f in (6, 7)}
    args = types.SimpleNamespace(collection_dir=str(coll), data_root=str(data), scenarios='all', progress_every_routes=0, max_routes=0,
                                 split_seed=20260819, test_ratio=0.10, val_ratio=0.05, include_visual_risk=False)
    from collections import Counter
    stats = Counter()
    bases = {b['frame_id']: b for b in build_dataset.iter_base_frames(args, stats, highway_ue3_overrides=decisions, matched_highway_ue3_overrides=set())}
    assert bases[6]['rs'] == 'R3' and bases[6]['primary_event'] == 'R-E1' and bases[6]['event_codes'] == ('R-E1',) and bases[6]['target_event_class'] == 'UE3'
    assert bases[5]['rs'] == rs and bases[5]['primary_event'] == event and bases[5]['target_event_class'] == 'RE'
    assert stats['highway_ue3_true_rs_override_applied'] == (2 if rs == 'R4' else 0)


def test_builder_refuses_override_on_unusual_source_event(tmp_path):
    data, coll, route = _collection(tmp_path, 'R4', 'U-E6')
    split = hw.route_split(route, seed=20260819, test_ratio=0.10, val_ratio=0.05)
    decisions = {('HighwayCutIn', route, 6): dict(expected_split=split, true_rs_override='R3')}
    args = types.SimpleNamespace(collection_dir=str(coll), data_root=str(data), scenarios='all', progress_every_routes=0, max_routes=0,
                                 split_seed=20260819, test_ratio=0.10, val_ratio=0.05, include_visual_risk=False)
    with pytest.raises(ValueError):
        list(build_dataset.iter_base_frames(args, None, highway_ue3_overrides=decisions, matched_highway_ue3_overrides=set()))
