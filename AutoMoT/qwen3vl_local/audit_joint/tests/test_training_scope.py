from dataclasses import replace

import pytest

from qwen3vl_local.audit_joint.training_scope import balance, phase3_work_balance, phase4_schedule, phase4_support


def test_missing_event_cannot_pass_equality_or_create_eval_capacity():
    result = balance([('a', 'q1', 'g'), ('b', 'q2', 'g2')], ['a', 'b', 'c'])
    assert result['missing_events'] == ['c']
    assert not result['equal_presentations']
    assert result['max_equal_eval_questions_per_event_without_replacement'] == 0


def test_repeats_do_not_become_eval_support_or_physical_groups():
    result = balance([('a', 'q1', 'g')] * 2 + [('b', 'q2', 'h'), ('b', 'q3', 'h')], ['a', 'b'])
    assert result['equal_presentations'] and not result['equal_unique_questions']
    assert result['events']['a']['unique_questions'] == 1
    assert result['events']['b']['physical_groups'] == 1
    assert result['max_equal_eval_questions_per_event_without_replacement'] == 1


@pytest.mark.parametrize('records', [[('x', 'q', 'g')], [('a', '', 'g')],
                                    [('a', 'q', 'g'), ('b', 'q', 'h')]])
def test_invalid_identity_rejected(records):
    with pytest.raises(ValueError):
        balance(records, ['a', 'b'])


@pytest.mark.parametrize('mode', ['binary', 'choice'])
def test_native_phase3_dedup_can_break_event_equality(mode):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import unique_cases
    rows = candidate_rows()
    rows.append(replace(rows[0], frame_id=10000))
    work, _ = train._validation_work(rows, target_per_bin=64, seed=17, action_output_mode=mode)
    assert phase3_work_balance(work)['equal_presentations']
    actual = phase3_work_balance(unique_cases(work))
    assert not actual['equal_presentations']
    assert not phase3_work_balance(evaluate._balanced_cases(rows, cases_per_bin=64, seed=17, action_output_mode=mode))['equal_presentations']
    assert sorted(c['unique_questions'] for c in actual['events'].values()) == [1] * 9 + [2]


def test_native_phase4_schedule_audits_all_expected_events():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    rows = [row(i, event) for event in EVENTS for i in range(3)]
    report = phase4_schedule(rows, 2)
    assert report['world1_world4_order_equal']
    for epoch in report['worlds']['4']:
        assert epoch['equal_presentations']
        assert epoch['max_question_repeat'] == 342
    incomplete = phase4_schedule([r for r in rows if r['episode']['event'] != 'U-E7'], 1)
    assert incomplete['worlds']['4'][0]['missing_events'] == ['U-E7']
    assert not incomplete['worlds']['4'][0]['equal_presentations']


def test_transferred_teacher_is_counted_but_manual_is_not_automatic():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    a = row(1, 'U-E7')
    a.update(label_basis='transferred_rgb4_reference', split='train', edge='complete', slice='readiness',
             student_reference={'original_label_basis': 'weak_rule_teacher'})
    a['episode'].update(state='PROCEED', longitudinal='FOLLOW')
    b = {**a, 'id': 'manual', 'label_basis': 'per_frame_conditions', 'student_reference': {}}
    result = phase4_support({'train': [a, b], 'val': [], 'test': []})
    assert result['automatic_train']['events']['U-E7']['unique_questions'] == 1
    assert result['automatic_readiness_support']['train/U-E7/default/complete']['YES'] == 1
    assert result['splits']['train']['events']['U-E7']['unique_questions'] == 2


def test_empty_splits_never_pass():
    result = phase4_support({'train': [], 'val': [], 'test': []})
    assert not result['splits']['val']['equal_unique_questions']
    assert len(result['splits']['val']['missing_events']) == 10


def test_fixed_equal_selection_is_order_independent_and_without_replacement():
    from qwen3vl_local.audit_joint.training_scope import equal_eval_selection
    rows = [('a', f'a{i}', 'long') for i in range(20)] + [('a', 'rare', 'short')]
    rows += [('b', 'b1', 'x'), ('b', 'b2', 'y')]
    result = equal_eval_selection(rows + rows[:3], ['a', 'b'])
    assert result == equal_eval_selection(list(reversed(rows)), ['a', 'b'])
    assert len(result['selected_ids']) == len(set(result['selected_ids'])) == 4
    assert 'rare' in result['selected_ids']
    assert result['distribution']['equal_unique_questions']
    assert result['distribution']['events']['a']['physical_groups'] == 2


def test_equal_selection_does_not_emit_partial_subset_when_event_missing():
    from qwen3vl_local.audit_joint.training_scope import equal_eval_selection
    result = equal_eval_selection([('a', 'q', 'g')], ['a', 'b'])
    assert result['status'] == 'blocked_missing_events'
    assert result['selected_ids'] == [] and result['per_event'] == 0


def test_priority_support_excludes_catchup_and_manual_and_counts_groups():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    rows = []
    for i, basis, phase in [(1, 'weak_rule_teacher', 'readiness'),
                            (2, 'weak_rule_teacher', 'readiness'),
                            (3, 'per_frame_conditions', 'readiness'),
                            (4, 'weak_rule_teacher', 'catchup')]:
        r = row(i, 'U-E7')
        r.update(label_basis=basis, split='train', edge='complete', slice=phase,
                 target='YES', physical_group='one_group')
        r['episode'].update(state='PROCEED', longitudinal='FOLLOW')
        rows.append(r)
    result = phase4_support({'train': rows, 'val': [], 'test': []})['priority_automatic_support']
    assert result['U-E7/complete/YES'] == dict(rows=2, unique_questions=2, physical_groups=1,
                                             status='present_not_certified')
    assert result['U-E6/proceed/YES']['status'] == 'missing_in_supplied_pool'


@pytest.fixture
def bound_scope(tmp_path, monkeypatch):
    import json
    from qwen3vl_local.audit_joint import training_scope as scope
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    root = tmp_path / 'dataset'
    root.mkdir()
    manifest = dict(files={'train.jsonl': 'unused_by_mock'}, rgb_mode=4, trainable=False)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'train.jsonl').write_text('original\n')
    (root / 'review_queue.jsonl').write_text('')
    monkeypatch.setattr(native, 'load_dataset', lambda path: ({'train': [], 'val': [], 'test': []}, manifest))
    monkeypatch.setattr(scope, 'phase4_schedule', lambda *args: {})
    return scope, root


@pytest.mark.parametrize('kind', ['bytes', 'delete', 'same_bytes_symlink', 'view_marker'])
def test_run_detects_actual_input_drift_after_loading(bound_scope, monkeypatch, kind):
    scope, root = bound_scope
    def drift(*args):
        path = root / 'train.jsonl'
        if kind == 'bytes':
            path.write_text('modified\n')
        elif kind == 'delete':
            path.unlink()
        elif kind == 'same_bytes_symlink':
            target = root / 'replacement.jsonl'
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        else:
            (root / 'student_view.json').write_text('{}')
        return {}
    monkeypatch.setattr(scope, 'phase4_schedule', drift)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        scope.run(root)


@pytest.mark.parametrize('state', ['missing', 'incomplete'])
def test_missing_phase3_preserves_phase4_measurements(bound_scope, state):
    scope, root = bound_scope
    p3 = root.parent / 'p3' / 'frame_index.jsonl'
    p3.parent.mkdir()
    if state == 'incomplete':
        p3.write_text('{}\n')
    result = scope.run(root, phase3_path=p3)
    assert result['phase3']['status'] == state
    assert result['phase4']['4']['support']['splits']['val']['equal_event_selection']['selected_ids'] == []
    assert result['input_bindings']
    assert not result['training_authorized_by_report']


def test_phase3_same_bytes_redirect_during_audit_rejected(bound_scope, monkeypatch):
    scope, root = bound_scope
    p3 = root.parent / 'p3' / 'frame_index.jsonl'
    p3.parent.mkdir()
    p3.write_text('{}\n')
    p3.with_name('manifest.json').write_text('{}')
    def redirect(*args):
        other = p3.with_name('other.jsonl')
        other.write_bytes(p3.read_bytes())
        p3.unlink()
        p3.symlink_to(other)
        return {'status': 'measured'}
    monkeypatch.setattr(scope, 'phase3_index', redirect)
    with pytest.raises(ValueError, match='Phase3 input binding'):
        scope.run(root, phase3_path=p3)


def test_native_metrics_require_all_events_for_full_macro():
    from qwen3vl_local.sft_new_loop_phase4.evaluate import metrics
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    records = [dict(event=e, edge='proceed', slice='readiness', target='YES', prediction='YES') for e in EVENTS]
    partial = metrics(records[:-1])
    assert partial['event_macro_accuracy'] is None
    assert partial['event_macro_accuracy_observed'] == 1
    assert metrics(records)['event_macro_accuracy'] == 1


@pytest.mark.parametrize('inside', ['parent', 'phase3'])
def test_output_cannot_modify_view_parent_or_phase3(tmp_path, inside):
    import json
    from qwen3vl_local.audit_joint.training_scope import check_output
    view, parent, p3 = [tmp_path / name for name in ('view', 'parent', 'phase3')]
    view.mkdir()
    (view / 'student_view.json').write_text(json.dumps({'parent': {'logical_path': str(parent)}}))
    with pytest.raises(ValueError, match='outside source'):
        check_output(tmp_path / inside / 'audit', view, p3 / 'frame_index.jsonl')
    check_output(tmp_path / 'output', view, p3 / 'frame_index.jsonl')
