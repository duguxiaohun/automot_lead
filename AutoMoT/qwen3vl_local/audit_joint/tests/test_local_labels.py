import json
from pathlib import Path

import pytest

from qwen3vl_local.audit_joint import label_quarantine as q
from qwen3vl_local.audit_joint.label_inventory import presence_status, verify_replay
from qwen3vl_local.audit_joint.local_replay import validate_request
from qwen3vl_local.sft_new_loop_phase4.identity import file_sha


def risk(onset=44):
    return dict(scenario='scene', route_id='route', first_affected_frame=onset, risk_id='risk')


@pytest.mark.parametrize('frame,expected', [(35, []), (36, ['risk']), (43, ['risk']),
                                           (44, ['risk']), (47, ['risk']), (48, [])])
def test_phase3_checks_future_label_and_input_envelope(frame, expected):
    row = dict(scenario='scene', route_id='route', frame_id=frame, context_id='JUNCTION_RULE_CONFLICT')
    assert q.phase3_risks(row, [risk()]) == expected


def test_lateral_horizon_and_route_identity():
    row = dict(scenario='scene', route_id='route', frame_id=31, context_id='RAMP_MERGE_EXIT')
    assert q.phase3_risks(row, [risk()]) == ['risk']
    row['route_id'] = 'different'
    assert q.phase3_risks(row, [risk()]) == []


def test_confirmed_current_wait_is_not_erased_by_distant_future_defect():
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
    evidence = dict(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256(),
                    future_speeds_exact_mps=[0.] * 9, brake=True, throttle=0.)
    row = dict(scenario='scene', route_id='route', frame_id=36,
               context_id='JUNCTION_RULE_CONFLICT', action_evidence=evidence)
    assert q.phase3_risks(row, [risk()]) == []
    row['frame_id'] = 43
    assert q.phase3_risks(row, [risk()]) == ['risk']
    row.update(frame_id=36, context_id='RAMP_MERGE_EXIT')
    assert q.phase3_risks(row, [risk()]) == ['risk']  # lateral evidence still crosses the defect


def test_phase4_future_discontinuity_cannot_revoke_earlier_yes():
    row = dict(scenario='scene', route_id='route', history_frames=[37, 39, 41, 43], rule_target='YES')
    assert q.phase4_risks(row, [risk()]) == []
    assert row['rule_target'] == 'YES'
    row['history_frames'] = [42, 46]  # unsampled gap also belongs to the causal envelope
    assert q.phase4_risks(row, [risk()]) == ['risk']


def test_phase4_older_stop_evidence_is_not_lost():
    row = dict(scenario='scene', route_id='route', history_frames=[60, 62],
               control_evidence={'stop': {'observations': [{'sources': [{'frame_id': 40}]}]}})
    assert q.phase4_risks(row, [risk()]) == ['risk']
    assert q.phase4_risks({'teacher_provenance': {'question': row}}, [risk()]) == ['risk']
    with pytest.raises(ValueError, match='missing causal'):
        q.phase4_risks(dict(scenario='scene', route_id='route'), [risk()])


def test_quarantine_requires_all_six_bound_sources(tmp_path):
    sources = []
    for frame in (43, 44):
        for kind in ('rgb', 'metas', 'bboxes'):
            rel = f'scene/route/{kind}/{frame:04d}.{"jpg" if kind == "rgb" else "pkl"}'
            p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
            sources.append(dict(path=rel, frame_id=frame, kind=kind, sha256=file_sha(p)))
    record = dict(scenario='scene', route_id='route', actor_id=7, last_unaffected_frame=43,
                  first_affected_frame=44, kind='confirmed_visible_actor_disappearance',
                  independent_approval=False, reason='reviewed before/after', sources=sources)
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps(dict(policy='local_source_quarantine_v1', records=[record])))
    assert len(q.load(tmp_path, ledger)) == 1
    (tmp_path / sources[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'): q.load(tmp_path, ledger)
    record['sources'] = sources[:-1]
    ledger.write_text(json.dumps(dict(policy='local_source_quarantine_v1', records=[record])))
    with pytest.raises(ValueError, match='before/after'): q.load(tmp_path, ledger)


def test_abstention_is_not_an_explicit_event_negative():
    row = dict(phase4=[], phase3_events=['U-E7'], phase4_events=[])
    assert presence_status(row) == 'phase4_abstention_not_event_absence'
    row['phase4'] = [{'event': 'U-E1'}]; row['phase4_events'] = ['U-E1']
    assert presence_status(row) == 'event_presence_disagreement_candidate'


def test_request_rejects_sparse_timeline_and_holdout(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import dataset
    monkeypatch.setattr(dataset, 'source_route', lambda r: 'group')
    monkeypatch.setattr(dataset, 'groups', lambda: set())
    monkeypatch.setattr(dataset, 'holdout_reservations', lambda: {})
    monkeypatch.setattr(dataset, 'split_for', lambda *a, **k: 'train')
    rgb = tmp_path / 'scene/route/rgb'; rgb.mkdir(parents=True)
    for f in range(3): (rgb / f'{f:04d}.jpg').write_bytes(b'fixture')
    r = dict(scenario='scene', route_id='route', physical_group='group', split='train', rgb_frames=[0, 1, 2])
    request = dict(data_root=str(tmp_path), routes=[r])
    assert validate_request(request) == request
    r['rgb_frames'] = [0, 2]
    with pytest.raises(ValueError, match='every RGB'): validate_request(request)
    r['rgb_frames'] = [0, 1, 2]; r['split'] = 'val'
    monkeypatch.setattr(dataset, 'split_for', lambda *a, **k: 'val')
    with pytest.raises(ValueError, match='training routes only'): validate_request(request)


def test_incomplete_replay_cannot_be_curated(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='in_progress')))
    with pytest.raises(ValueError, match='incomplete replay'): verify_replay(tmp_path)


def test_route_scoped_source_verification_does_not_require_unrelated_assets(tmp_path):
    # A dataset subset may omit both ledger routes. Relevant routes must still
    # fail closed if any of their registered evidence is absent.
    assert q.load(tmp_path, route=('unrelated', 'route')) == []
    record=json.loads(q.LEDGER.read_text())['records'][0]
    with pytest.raises(FileNotFoundError):
        q.load(tmp_path, route=(record['scenario'],record['route_id']))


def test_no_risk_does_not_reinterpret_other_candidate_evidence():
    assert q.phase3_risks({}, []) == []
