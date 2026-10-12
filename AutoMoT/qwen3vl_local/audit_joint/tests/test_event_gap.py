import json
from copy import deepcopy

import pytest

from qwen3vl_local.audit_joint import event_gap


def record(split='train', target='UNKNOWN', reason='passed_behind'):
    ident = split + '/instance'
    q = dict(episode=dict(event='U-E5', instance_id=ident), rgb_mode=4, edge='proceed',
             slice='readiness', rule_target=target, state_rule_target=target,
             frame_id=10, question_id='q', facts=dict(corridor_clear=None),
             reasons=['decisive_actor_not_visible'], ego_speed=0., input_sources=[])
    return dict(scenario='s', route_id=split, split=split, frame_id=11, questions=[q],
                instance_ends=[dict(instance_id=ident, reason=reason, completed=False)])


def test_unknown_and_retirement_do_not_create_positive_or_heldout_examples():
    rows = [record(), record('val', 'YES'), record('test', 'NO')]
    rows[0]['questions'].append(dict(rows[0]['questions'][0], rgb_mode=2))
    result = event_gap.summarize(rows, 'U-E5')
    assert result['counts']['train']['questions/proceed/readiness/UNKNOWN'] == 1
    assert 'questions/proceed/readiness/YES' not in result['counts']['train']
    assert len(result['train_examples']) == 1
    assert result['counts']['val']['questions/proceed/readiness/YES'] == 1
    assert result['labels_changed'] is False


def test_examples_bounded_and_unended_instances_remain_censored():
    rows = []
    for n in range(50):
        row = record(reason=str(n))
        row['questions'][0]['episode']['instance_id'] = str(n)
        row['instance_ends'][0]['instance_id'] = str(n)
        rows.append(row)
    rows[-1]['instance_ends'] = []
    result = event_gap.summarize(rows, 'U-E5')
    assert len(result['train_examples']) == 12
    assert result['counts']['train']['instances_without_end_in_recording'] == 1


def test_inspection_uses_native_receipts_and_rejects_drift(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import make_bundle
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool
    from qwen3vl_local.audit_joint.io import file_sha
    pool, rows, path, _ = make_bundle(tmp_path)
    # Synthetic inventory bypass only; actual native route/proof/SHA checks run.
    monkeypatch.setattr(candidate_pool, 'validate', lambda x: x)
    p = tmp_path/'pool.json'; p.write_text(json.dumps(pool))
    index = tmp_path/'index.json'; expected = file_sha(index)
    result = event_gap.inspect(p, index, expected_index_sha256=expected, event='U-E1')
    assert result['selected_routes'] == 1
    assert result['bindings']['inspected_routes'][0]['sha256'] == file_sha(path)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='route changed'):
        event_gap.inspect(p, index, expected_index_sha256=expected, event='U-E1')
    with pytest.raises(ValueError, match='handoff SHA'):
        event_gap.inspect(p, index, expected_index_sha256='0'*64, event='U-E1')


def test_cleared_frame_after_end_is_not_retrospective_release():
    row = record(target='NO')
    after = deepcopy(row)
    after.update(frame_id=12, questions=[], instance_ends=[])
    result = event_gap.summarize([row, after], 'U-E5')
    assert result['counts']['train']['questions/proceed/readiness/NO'] == 1
    assert 'questions/proceed/readiness/YES' not in result['counts']['train']


def test_examples_spread_across_physical_groups_and_keep_all_counts():
    a, b, c = record(), record(), record()
    a['physical_group'] = b['physical_group'] = 'same_recording'
    c['physical_group'] = 'different_recording'
    for n, row in enumerate((a, b, c)):
        row['questions'][0]['episode']['instance_id'] = str(n)
        row['instance_ends'][0]['instance_id'] = str(n)
    result = event_gap.summarize([a, b, c], 'U-E5')
    assert result['counts']['train']['instances'] == 3
    assert [x['instance_end']['instance_id'] for x in result['train_examples']] == ['0', '2']


def test_raw_diagnostic_failure_keeps_verified_replay_counts(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import make_bundle, seal
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay, teacher_rules
    from qwen3vl_local.audit_joint.io import file_sha, digest
    pool, rows, path, header = make_bundle(tmp_path)
    monkeypatch.setattr(candidate_pool, 'validate', lambda x: x)
    q = rows[1]['questions'][0]
    q['episode']['event'] = 'U-E5';q['rgb_mode'] = 4
    q['history_frames'] = [4, 6, 8, 10]
    q['input_sources'] = [s for s in q['causal_sources'] if s['kind'] == 'rgb' and s['frame_id'] in q['history_frames']]
    q['input_sha256'] = digest(q['input_sources'])
    q['rule_class'] = teacher_rules.rule_class('U-E5', 'proceed', 4)
    q['instance_sources'] = q['causal_sources'][:3]
    seal(q)
    rows[2]['instance_ends'] = [dict(instance_id=q['episode']['instance_id'], reason='finished', completed=True)]
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    receipt = teacher_replay.inspect_route(path, pool['routes'][0], header)
    index = dict(**header, routes=[receipt]);index['sha256'] = digest(index)
    ip = tmp_path/'index.json';ip.write_text(json.dumps(index))
    pp = tmp_path/'pool.json';pp.write_text(json.dumps(pool))
    result = event_gap.inspect(pp, ip, expected_index_sha256=file_sha(ip), data_root=tmp_path)
    assert result['status'] == 'incomplete_raw_diagnostic'
    assert result['raw_errors']
    assert result['counts']['train']['questions/proceed/readiness/YES'] == 1
    assert result['bindings']['inspected_routes']
