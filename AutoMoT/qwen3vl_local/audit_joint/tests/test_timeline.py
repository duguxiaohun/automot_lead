"""G1 因果、身份、删失及未匹配边界；不以合成题认证老师质量。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.audit_joint import timeline as t
from qwen3vl_local.audit_joint.io import digest, file_sha, write_json, write_rows


def route(frames=(10, 11, 12, 13), rep='Rep0'):
    return dict(scenario='scene', route_id=rep, physical_group='scene/physical',
                split='train', rgb_frames=list(frames))


def source(frame):
    return dict(kind='rgb', frame_id=frame, path=f'scene/Rep0/rgb/{frame:04d}.jpg', sha256='a' * 64)


def left(frame=10, contexts=('LEAD_BRAKE',), rep='Rep0'):
    return dict(scenario='scene', route_id=rep, physical_group='scene/physical', frame_id=frame,
                contexts=list(contexts), source_annotation_present=True, mapping={},
                annotation_repair=dict(repaired_rs='R1'), evidence={'raw': 'kept'},
                rgb_identity={'sha256': 'a' * 64})


def question(frame=10, answer='YES', instance='instance1', mode=2):
    from qwen3vl_local.sft_new_loop_phase4.controller import Episode
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import EP_FIELDS
    from qwen3vl_local.sft_new_loop_phase4.route_context import episode_edge
    ep = Episode('U-E1', instance)
    offsets = [-4, 0] if mode == 2 else [-6, -4, -2, 0]
    sources = [source(frame + d) for d in offsets]
    q = dict(scenario='scene', route_id='Rep0', physical_group='scene/physical', frame_id=frame,
             episode={k: getattr(ep, k) for k in EP_FIELDS}, edge='proceed', rgb_mode=mode,
             rule_target=answer, state_rule_target=answer, slice='readiness',
             facts={c: answer == 'YES' for c in episode_edge(ep, 'proceed').criteria},
             input_sources=sources, causal_sources=sources, instance_sources=[],
             history_frames=[s['frame_id'] for s in sources])
    q['question_id'] = digest(q)
    return q


def right(frame=10, questions=None):
    return dict(scenario='scene', route_id='Rep0', physical_group='scene/physical', frame_id=frame,
                questions=questions if questions is not None else [question(frame)], sources=[source(frame)],
                reasons=[], traces=[], instance_ends=[])


def admission(q, answer=None):
    return dict(question=q, admission_target=answer or q['rule_target'], admission_reason='fixture')


def current_source(request_path):
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
    from qwen3vl_local.sft_new_loop_phase4 import taxonomy, route_context
    return dict(request_sha256=file_sha(request_path), phase3_mapping_sha256=mapping_contract_hash(),
                producer=dict(sources={Path(m.__file__).name: file_sha(m.__file__)
                                       for m in (taxonomy, route_context)}))


def receipt(root):
    path = root / 'receipt.json'
    data = dict(files={str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=file_sha(p))
                       for p in root.rglob('*') if p.is_file() and p != path})
    path.write_text(json.dumps(data))


@pytest.fixture
def replay(tmp_path):
    root = tmp_path / 'replay'; root.mkdir()
    r = route()
    write_json(root / 'request.json', dict(routes=[r]))
    write_json(root / 'source.json', current_source(root / 'request.json'))
    raw = [right(f, [question(f, a, mode=m) for m in (2, 4)])
           for f, a in zip(r['rgb_frames'], ['YES', 'YES', 'UNKNOWN', 'NO'])]
    write_rows(root / 'phase3_timeline.jsonl', [left(f) for f in r['rgb_frames']])
    write_rows(root / 'phase4_questions.jsonl', [admission(q) for row in raw for q in row['questions']])
    write_rows(root / 'raw.jsonl', [dict(kind='diagnostic_header', route=r), *raw])
    write_json(root / 'summary.json', dict(status='complete', total_frames=4,
        routes=[dict(scenario='scene', route_id='Rep0', raw_file='raw.jsonl', sha256=file_sha(root / 'raw.jsonl'))]))
    receipt(root)
    return root


def test_full_outer_keeps_missing_sides_and_future_tail():
    joined = list(t.join_frames(route(), [left(10), left(12)], [right(10), right(11)], {'LEAD_BRAKE': 'U-E1'}))
    assert [x['frame_id'] for x in joined] == [10, 11, 12, 13]
    assert joined[1]['phase3_record_present'] is False
    assert joined[2]['phase4_record_present'] is False
    assert joined[3]['events'][0]['presence_status'] == 'any_unknown'
    assert joined[0]['events'][0]['match_status'] == 'unmatched_participant_evidence'
    assert all(x['events'][0]['comparable'] is False for x in joined)


def test_question_absence_not_event_negative_even_with_other_event():
    r = right(); r['questions'][0]['episode']['event'] = 'U-E7'
    frame = next(t.join_frames(route(), [left()], [r], {'LEAD_BRAKE': 'U-E1', 'SIGNAL_FAILURE': 'U-E7'}))
    lead, signal = frame['events']
    assert lead['phase4_presence'] is None
    assert lead['presence_status'] == 'any_unknown'
    assert signal['presence_status'] == 'only_phase4'


@pytest.mark.parametrize('change', ['annotation', 'mapping', 'rs'])
def test_unknown_upstream_is_not_an_explicit_negative(change):
    l = left(contexts=())
    if change == 'annotation': l['source_annotation_present'] = False
    if change == 'mapping': l['mapping'] = None
    if change == 'rs': l['annotation_repair']['repaired_rs'] = 'UNKNOWN'
    event = next(t.join_frames(route(), [l], [right()], {'LEAD_BRAKE': 'U-E1'}))['events'][0]
    assert event['phase3_presence'] is None and event['presence_status'] == 'any_unknown'


def test_multiple_modes_and_edges_do_not_invent_multiple_instances():
    qs = [question(mode=2), question(mode=4)]
    event = next(t.join_frames(route(), [left()], [right(questions=qs)], {'LEAD_BRAKE': 'U-E1'}))['events'][0]
    assert event['candidate_instance_ids'] == ['instance1']
    qs.append(question(instance='instance2'))
    event = next(t.join_frames(route(), [left()], [right(questions=qs)], {'LEAD_BRAKE': 'U-E1'}))['events'][0]
    assert event['match_status'] == 'ambiguous_match' and event['confirmed_instance_id'] is None


@pytest.mark.parametrize('records', [[left(rep='Rep1')], [left(9)], [left(), left()]])
def test_recording_frame_and_duplicate_rejection(records):
    with pytest.raises(ValueError, match='identity'):
        list(t.join_frames(route(), records, [], {'LEAD_BRAKE': 'U-E1'}))


def test_same_frame_source_sha_disagreement_fails():
    r = right(); r['sources'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='RGB identity'):
        list(t.join_frames(route(), [left()], [r], {'LEAD_BRAKE': 'U-E1'}))


def collect_intervals(records):
    result = []; tracker = t.ConditionIntervals(route(), result.append)
    for frame, record in records:
        admitted = {q['question_id']: admission(q) for q in record.get('questions', [])} if record else {}
        tracker.observe(frame, record, admitted)
    tracker.finish()
    return result


def test_yes_no_yes_unknown_are_distinct_opportunities():
    result = collect_intervals([(f, right(f, [question(f, a)]))
                                for f, a in zip(range(10, 15), ['YES', 'YES', 'NO', 'YES', 'UNKNOWN'])])
    assert [(r['target'], r['frame_count']) for r in result] == [('YES', 2), ('NO', 1), ('YES', 1), ('UNKNOWN', 1)]
    assert result[0]['end_reason'] == 'target_changed' and not result[0]['right_censored']
    assert result[2]['end_reason'] == 'became_unknown' and result[2]['right_censored']
    assert result[-1]['end_reason'] == 'route_end'
    assert all(r['permission_granted'] is None and r['label_error'] is None for r in result)


@pytest.mark.parametrize('kind,reason', [('gap', 'observation_gap'), ('missing', 'missing_record'),
                                       ('not_asked', 'question_not_observed'), ('end', 'instance_end:identity_lost')])
def test_censoring_is_not_no_or_completion(kind, reason):
    next_record = right(11, [])
    if kind == 'end': next_record['instance_ends'] = [dict(instance_id='instance1', reason='identity_lost')]
    records = [(10, right()), (12 if kind == 'gap' else 11, None if kind == 'missing' else next_record)]
    result = collect_intervals(records)
    assert len(result) == 1 and result[0]['target'] == 'YES'
    assert result[0]['end_reason'] == reason and result[0]['right_censored']


def test_context_and_slice_change_break_intervals():
    for change in ('slice', 'target_corridor', 'instance_id'):
        q = question(11)
        if change == 'slice': q['slice'] = 'catchup'
        else: q['episode'][change] = 'different'
        result = collect_intervals([(10, right()), (11, right(11, [q]))])
        assert len(result) == 2 and result[0]['right_censored']


def test_offline_receipt_not_permission_or_deployment_execution():
    r = right()
    r['traces'] = [dict(instance_id='instance1', execution_receipt=dict(
        source='causal_tracker', successor_observed=True, edges=['proceed']), observed_state_sync={'state': 'PROCEED'})]
    item = collect_intervals([(10, r)])[0]
    assert item['first_offline_execution_observed_frame'] == 10
    assert item['first_offline_state_sync_frame'] == 10
    assert item['permission_granted'] is None and item['deployment_execution_frame'] is None
    assert item['model_response_frame'] is None


def test_native_mapping_and_end_to_end_receipts(replay, tmp_path):
    before = file_sha(replay / 'receipt.json')
    out = tmp_path / 'out'; result = t.run(replay, out)
    assert result['counts']['requested_frames'] == 4
    assert result['counts']['raw_questions'] == 8
    assert result['confirmed_instance_matches'] == result['comparable_propositions'] == 0
    assert result['g1_status'] == 'in_progress' and result['final_supervision_coverage'] is None
    assert file_sha(replay / 'receipt.json') == before
    for name, proof in json.loads((out / 'receipt.json').read_text())['files'].items():
        assert file_sha(out / name) == proof['sha256']
    with pytest.raises(FileExistsError): t.run(replay, out)


def test_receipt_rejects_drift_and_parent_symlink(replay, tmp_path):
    with (replay / 'phase3_timeline.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='drift'): t.run(replay, tmp_path / 'out')
    link = tmp_path / 'link'; link.symlink_to(replay, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'): t.bound_file(tmp_path, 'link/source.json')


def test_partial_timeline_reports_missing_without_inventing_negatives(replay, tmp_path):
    p = replay / 'phase3_timeline.jsonl'
    p.write_text(''.join(p.read_text().splitlines(keepends=True)[1:]))
    receipt(replay)
    result = t.run(replay, tmp_path / 'out')
    assert result['missing_records'] == 1 and result['counts']['both_records'] == 3
    assert result['event_presence_counts']['U-E1/any_unknown'] == 1


def test_mapping_drift_refused_before_output(replay, tmp_path):
    p = replay / 'source.json'; s = json.loads(p.read_text())
    s['producer']['sources']['taxonomy.py'] = '0' * 64; p.write_text(json.dumps(s)); receipt(replay)
    with pytest.raises(ValueError, match='matching source'): t.run(replay, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_question_flat_raw_disagreement_preserves_failure(replay, tmp_path):
    p = replay / 'phase4_questions.jsonl'; data = list(t.rows(p)); data[0]['question']['rule_target'] = 'NO'
    p.write_text(''.join(json.dumps(x) + '\n' for x in data)); receipt(replay)
    with pytest.raises(ValueError, match='raw/admission'): t.run(replay, tmp_path / 'out')
    assert json.loads((tmp_path / 'out/progress.json').read_text())['status'] == 'failed'
    assert not (tmp_path / 'out/receipt.json').exists()


def test_future_evidence_rejected_even_after_rehash():
    q = question(); q['causal_sources'] = [source(11)]
    q['question_id'] = digest({k: v for k, v in q.items() if k != 'question_id'})
    with pytest.raises(ValueError, match='future'): t.validate_question(q, route(), 10)


def test_stop_history_future_evidence_rejected():
    q = question(); q['control_evidence'] = {'stop': {'observations': [{'sources': [source(11)]}]}}
    q['question_id'] = digest({k: v for k, v in q.items() if k != 'question_id'})
    with pytest.raises(ValueError, match='nested causal'): t.validate_question(q, route(), 10)


def test_duplicate_question_in_same_frame_rejected():
    q = question()
    with pytest.raises(ValueError, match='duplicate instance'):
        collect_intervals([(10, right(questions=[q, q]))])


def test_mid_run_input_drift_cannot_write_success_receipt(replay, tmp_path, monkeypatch):
    original = t.verify_inputs; calls = []
    def verify(root):
        if calls:
            with (root / 'phase3_timeline.jsonl').open('a') as stream:
                stream.write('{}\n')
        calls.append(True)
        return original(root)
    monkeypatch.setattr(t, 'verify_inputs', verify)
    with pytest.raises(ValueError, match='drift'): t.run(replay, tmp_path / 'out')
    assert not (tmp_path / 'out/receipt.json').exists()
    assert json.loads((tmp_path / 'out/progress.json').read_text())['status'] == 'failed'


def test_raw_gate_and_final_supervision_remain_separate():
    q = question(); v = t.question_view(q, admission(q, 'UNKNOWN'))
    assert v['raw_rule_target'] == 'YES' and v['gate_target'] == 'UNKNOWN'
    assert v['final_dataset_target'] is None and v['finite_input_judgment'] is None
    assert v['exact_student_prompt'] is None and v['offsets'] == [-4, 0]


@pytest.mark.parametrize('change,message', [('state', 'inapplicable'), ('source', 'recording/path')])
def test_rehashed_inapplicable_or_other_recording_question_rejected(change, message):
    q = question()
    if change == 'state': q['episode']['state'] = 'DONE'
    else: q['input_sources'][0]['path'] = 'scene/Rep1/rgb/0006.jpg'
    q['question_id'] = digest({k: v for k, v in q.items() if k != 'question_id'})
    with pytest.raises(ValueError, match=message): t.validate_question(q, route(), 10)
