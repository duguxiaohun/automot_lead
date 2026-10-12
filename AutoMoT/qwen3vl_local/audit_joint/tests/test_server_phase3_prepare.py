import json
import zipfile

import pytest

from qwen3vl_local.audit_joint import server_phase3_prepare as tool
from qwen3vl_local.audit_joint.io import file_sha


def test_native_plans_equal_events_and_preserve_short_view(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    rows = candidate_rows()
    monkeypatch.setattr(train, '_read_rows', lambda *a, **kw: rows)
    result = tool.sampling_plans(tmp_path / 'index.jsonl', tmp_path)
    assert result['views']['2rgb_short']['frame_offsets'] == [-2, 0]
    assert result['views']['4rgb']['frame_offsets'] == [-3, -2, -1, 0]
    for report in result['modes'].values():
        assert report['world1_world4_order_equal']
        assert {c['presentations'] for c in report['train']['events'].values()} == {1024}
        for name in ('val_loss', 'val_generation', 'test'):
            assert report[name]['distribution']['equal_unique_questions']
            ids = report[name]['plan']['selected_ids']
            assert len(set(ids)) == len(ids)


def test_missing_event_rejected_by_native_sampling(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    rows = candidate_rows()
    absent = rows[0].context_id
    monkeypatch.setattr(train, '_read_rows', lambda *a, **kw: [r for r in rows if r.context_id != absent])
    with pytest.raises(ValueError, match='missing'):
        tool.sampling_plans(tmp_path / 'index.jsonl', tmp_path)


def test_real_index_loader_and_native_plans_without_reader_substitution(tmp_path):
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    index = fixture(tmp_path, 'valid')
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    template = next(r for r in rows if r['invalid_action_context'])
    for split in ('train', 'val', 'test'):
        for i, context in enumerate(CONTEXT_IDS):
            for rs in ('R1', 'R2', 'R3', 'R4', 'R5'):
                rows.append(dict(template, split=split, route_id=f'{split}_{i}_{rs}',
                    context_id=context, true_rs=rs, invalid_reason='wrong_road_structure',
                    invalid_source=f'source={context}|true_rs={rs}|asked_context={context}'))
    index.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = tool.sampling_plans(index, tmp_path)
    assert result['status'] == 'event_equal'
    assert all(r['test']['plan']['per_event'] == 1 for r in result['modes'].values())


@pytest.mark.parametrize('defect', ['source', 'collection', 'existing'])
def test_build_fails_before_any_expensive_work(tmp_path, monkeypatch, defect):
    from qwen3vl_local.audit_joint import server_readiness
    monkeypatch.setattr(server_readiness, 'source_check',
                        lambda *a: dict(status='mismatch' if defect == 'source' else 'matched'))
    monkeypatch.setattr(tool, 'execute', lambda *a: pytest.fail('must not build'))
    if defect == 'existing':
        (tmp_path / 'x_result.json').write_text('{}')
        (tmp_path / 'data').mkdir()
        (tmp_path / 'data/keep.txt').write_text('untouched')
    assert tool.build(tmp_path, tmp_path, tmp_path, 1) == 2
    report = json.loads((tmp_path / 'prepare_result.json').read_text())
    assert report['status'] == 'blocked' and not report['training_started']
    assert tool.package(tmp_path, 2) == 2
    if defect == 'existing':
        assert (tmp_path / 'data/keep.txt').read_text() == 'untouched'


def test_interrupted_worker_and_corrupt_json_still_deliver_verified_small_package(tmp_path):
    (tmp_path / 'prepare_result.json').write_text('{')
    (tmp_path / 'prepare.log').write_text('worker was killed\n')
    (tmp_path / 'adapter_model.safetensors').write_bytes(b'not for upload')
    assert tool.package(tmp_path, 75) == 2
    receipt = json.loads((tmp_path / 'phase3_prepare_package.json').read_text())
    path = tmp_path / 'phase3_prepare_report.zip'
    assert receipt['zip_sha256'] == file_sha(path) and receipt['bytes'] < 30_000_000
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == set(receipt['files'])
        assert 'adapter_model.safetensors' not in z.namelist()
        report = json.loads(z.read('handoff.json'))
        assert report['status'] == 'incomplete' and report['errors']


def test_exit_zero_without_all_reports_is_not_success(tmp_path):
    (tmp_path / 'prepare_result.json').write_text('{"status":"cpu_data_prepared"}')
    assert tool.package(tmp_path, 0) == 2


def test_package_complete_reports_and_oversized_diagnostic(tmp_path):
    for name in tool.JSON_NAMES:
        (tmp_path / name).write_text('{}')
    (tmp_path / 'prepare_result.json').write_text('{"status":"cpu_data_prepared"}')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/manifest.json').write_text('{}')
    assert tool.package(tmp_path, 0) == 0
    for name in ('phase3_prepare_report.zip', 'phase3_prepare_package.json'):
        (tmp_path / name).unlink()
    (tmp_path / 'sampling_plans.json').write_text(' ' * 2_000_001)
    assert tool.package(tmp_path, 0) == 2
    with zipfile.ZipFile(tmp_path / 'phase3_prepare_report.zip') as z:
        assert json.loads(z.read('handoff.json'))['errors']
