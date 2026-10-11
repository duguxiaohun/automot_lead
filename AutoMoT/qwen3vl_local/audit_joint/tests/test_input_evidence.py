"""学生输入/老师因果包络盘点，不使用合成证据认定可观察性。"""
from copy import deepcopy
import json

import pytest

from qwen3vl_local.audit_joint import input_evidence as audit
from qwen3vl_local.audit_joint.io import digest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key
from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import short_row
from qwen3vl_local.sft_new_loop_phase4.route_prompts import prompt


def source(frame, kind='rgb'):
    return dict(frame_id=frame, kind=kind, path=f'Scene/Rep0/{kind}/{frame:04d}.{"jpg" if kind == "rgb" else "pkl"}', sha256='a' * 64)


def row(weak=True):
    ep = dict(event='U-E1', instance_id='instance', state='YIELD', longitudinal='HOLD')
    obs = dict(frame_id=10, history_frames=[4, 6, 8, 10], speed_mps=0.)
    r = dict(id='row', scenario='Scene', route_id='Rep0', physical_group='Scene/physical',
        split='train', slice='readiness', episode=ep, edge='proceed', observation=obs,
        label_basis='weak_rule_teacher' if weak else 'reviewed_transition_band',
        target='YES', images=[source(f)['path'] for f in obs['history_frames']],
        image_sha256=['a' * 64] * 4, image_rgb_sha256=['b' * 64] * 4)
    r['prompt_sha256'] = digest(prompt(Episode(**ep), r['edge'], obs))
    r['model_input_sha256'] = model_input_key(r)
    if weak:
        q = dict(scenario=r['scenario'], route_id=r['route_id'], frame_id=10, edge='proceed', episode=ep,
            question_id='native-question', rule_target='YES', facts=dict(corridor_clear=True, priority_satisfied=None),
            causal_sources=[source(f, k) for f in range(4, 11) for k in ('rgb', 'metas', 'bboxes')],
            control_evidence={'stop': {'observations': [{'sources': [source(2, 'metas'), source(2)]}]}},
            instance_sources=[source(1, 'bboxes')])
        r['teacher_provenance'] = dict(question=q)
    return r


def test_exact_student_views_keep_full_teacher_history_without_review_claim():
    r = row(); before = deepcopy(r)
    two, four = audit.inspect_pair(r, short_row(r))
    assert r == before
    assert two['input']['history_frames'] == [8, 10]
    assert four['input']['history_frames'] == [4, 6, 8, 10]
    assert two['evidence']['earliest_source_frame'] == 1
    assert two['evidence']['observed_until'] == 10
    assert two['evidence']['control_sources_before_visible_start'] is True
    assert two['evidence']['teacher_rgb_frames_not_shown'] == [2, 4, 5, 6, 7, 9]
    assert four['evidence']['teacher_rgb_frames_not_shown'] == [2, 5, 7, 9]
    assert two['reference_target'] == four['reference_target'] == 'YES'
    assert all(x['evidence']['finite_input_judgment'] is None and not x['independent_approval'] for x in (two, four))
    assert two['student_label_basis'] == 'transferred_rgb4_reference'
    assert two['original_label_basis'] == four['original_label_basis'] == 'weak_rule_teacher'
    assert 'Use the chronological' in two['input']['system']


def test_manual_references_keep_missing_teacher_provenance_separate():
    r = row(False); records = audit.inspect_pair(r, short_row(r))
    assert all(x['facts'] is None and x['evidence'] is None for x in records)
    summary = audit.summarize(records)
    assert summary['counts']['train/rgb2/teacher_provenance_not_available'] == 1
    assert not summary['criterion_cells']
    assert all(x['observability_status'] == 'not_adjudicated' for x in records)


@pytest.mark.parametrize('field', ['model_input_sha256', 'image_rgb_sha256'])
def test_original_model_input_drift_is_not_hidden_by_short_transform(field):
    r = row()
    r[field] = 'c' * 64 if field == 'model_input_sha256' else ['c' * 64] * 4
    with pytest.raises(ValueError, match='model input identity'):
        audit.inspect_pair(r, short_row(r))


def test_duplicate_sources_are_deduplicated_not_extra_support():
    r = row(); q = r['teacher_provenance']['question']
    q['instance_sources'] += deepcopy(q['causal_sources'])
    result = audit.evidence_envelope(q, r)
    assert result['source_count'] == 24


@pytest.mark.parametrize('change,match', [('future', 'future'), ('sha', 'contradictory'),
                                        ('rgb', 'not bound'), ('route', 'semantic identity'), ('empty', 'missing')])
def test_invalid_evidence_cannot_be_counted(change, match):
    r = row(); q = r['teacher_provenance']['question']
    if change == 'future': q['control_evidence']['stop']['observations'][0]['sources'].append(source(11))
    if change == 'sha': q['instance_sources'].append(dict(source(10), sha256='c' * 64))
    if change == 'rgb': r['image_sha256'][-1] = 'c' * 64
    if change == 'route': q['route_id'] = 'Rep1'
    if change == 'empty':
        q['causal_sources'] = []; q['control_evidence'] = {}; q['instance_sources'] = []
    with pytest.raises(ValueError, match=match): audit.evidence_envelope(q, r)


def test_missing_weak_proof_fails_but_does_not_change_target():
    r = row(); r.pop('teacher_provenance')
    with pytest.raises(ValueError, match='missing teacher'): audit.inspect_pair(r, short_row(r))
    assert r['target'] == 'YES'


@pytest.mark.parametrize('field', ['target', 'images', 'student_reference'])
def test_short_view_changes_cannot_be_hidden(field):
    r = row(); small = short_row(r); small[field] = 'foreign'
    with pytest.raises(ValueError, match='official transform'): audit.inspect_pair(r, small)


def test_criteria_count_questions_and_unique_physical_groups_separately():
    r = row(); items = audit.inspect_pair(r, short_row(r))
    report = audit.summarize(items + deepcopy(items))
    assert report['counts']['train/rgb2/rows'] == 2
    for cell in report['criterion_cells'].values():
        assert cell['counts']['rows'] == 2 and cell['physical_groups'] == 1
    assert len(report['criterion_cells']) == 4


def test_failed_loader_preserves_failure_report_and_does_not_certify(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
    view = tmp_path / 'view'; view.mkdir()
    (view / 'student_view.json').write_text(json.dumps({'parent': {'logical_path': str(tmp_path / 'parent')}}))
    def fail(*args, **kwargs):
        raise ValueError('stale source contract')
    monkeypatch.setattr(data, 'load_dataset', fail)
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='stale'): audit.run(view, out)
    assert json.loads((out / 'progress.json').read_text())['status'] == 'failed'
    assert not (out / 'receipt.json').exists()
    with pytest.raises(FileExistsError): audit.run(view, out)
