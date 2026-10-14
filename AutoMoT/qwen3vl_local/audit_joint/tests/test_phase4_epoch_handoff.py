import hashlib
import json
import zipfile

import pytest

from qwen3vl_local.audit_joint import phase4_epoch_handoff as tool
from qwen3vl_local.sft_new_loop_phase4.rgb_short.diagnostics import sample_binding
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


@pytest.fixture
def plan():
    samples = [dict(position=i, rank=i % 4, local_index=i // 4, id=f'q{i}') for i in range(10240)]
    audit = dict(event_counts=dict.fromkeys(EVENTS, 1024), ordered_samples=sample_binding(samples, 4))
    steps = [[dict(row, epoch=0, loss=0.5) for row in samples[r::4]] for r in range(4)]
    return samples, audit, steps


def unpack(output):
    receipt = json.loads((output / 'phase4_epoch_package.json').read_text())
    path = output / 'phase4_epoch_report.zip'
    assert path.stat().st_size <= tool.LIMIT
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['zip_sha256']
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} == receipt['files']
        return {n: json.loads(z.read(n)) for n in z.namelist()}


def test_exact_completed_plan(plan):
    report = tool.check_consumption(*plan)
    assert report['status'] == 'matched_recorded_plan'
    assert report['event_counts_from_recorded_plan'] == dict.fromkeys(EVENTS, 1024)


def test_native_sampler_and_sample_records_with_repeated_ids():
    from qwen3vl_local.sft_new_loop_phase4.sampling import plan as native_plan
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.diagnostics import sample_records
    rows = [dict(id=f'{e}-{i}', episode={'event': e}, target='NO', edge='proceed',
                 slice='readiness', physical_group=f'{e}-{i}', scenario=e, route_id=str(i),
                 observation=dict(frame_id=4, speed_mps=0), label_basis='weak_rule_teacher')
            for e in EVENTS for i in range(3)]
    indices, audit = native_plan(rows, epoch=0, policy='phase3_balanced', world_size=4,
                                 expected_events=EVENTS, budget=10240)
    samples = sample_records(rows, indices, 4)
    audit['ordered_samples'] = sample_binding(samples, 4)
    steps = [[dict(s, epoch=0, loss=0.4) for s in samples[r::4]] for r in range(4)]
    assert tool.check_consumption(samples, audit, steps)['status'] == 'matched_recorded_plan'


def test_many_malformed_lines_keep_diagnostics_bounded(tmp_path):
    run = tmp_path / 'run_first_epoch_rgb2'
    run.mkdir()
    (run / 'epoch_000_rank0_steps.jsonl').write_text('{\n' * 1000)
    result = tool.collect(tmp_path)
    data = unpack(tmp_path / 'handoff')['collection_receipt.json']
    assert result['errors'] == data['error_count'] == 1000
    assert len(data['errors']) == 100


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'id', 'epoch', 'loss', 'hash', 'quota'])
def test_consumption_rejects_equal_totals_with_wrong_evidence(plan, defect):
    samples, audit, steps = plan
    if defect == 'missing':
        steps[3].pop()
    elif defect == 'duplicate':
        steps[3][-1] = steps[3][0]
    elif defect == 'order':
        steps[0].reverse()
    elif defect in ('id', 'epoch', 'loss'):
        steps[0][0][defect] = {'id': 'wrong', 'epoch': 1, 'loss': float('nan')}[defect]
    elif defect == 'hash':
        samples[0]['id'] = 'changed'
    else:
        audit['event_counts']['U-E5'] = 0
    assert tool.check_consumption(samples, audit, steps)['status'] == 'incomplete_or_inconsistent'


def test_missing_and_truncated_evidence_still_returns_verified_archive(tmp_path):
    run = tmp_path / 'run_first_epoch_rgb2'
    run.mkdir()
    (run / 'epoch_000_runtime.json').write_text('{"ranks":')
    (run / 'epoch_000_rank0_steps.jsonl').write_text('{"id":"readable"}\n{"id":')
    (tmp_path / 'admission_2_broken.json').write_text('[')
    (run / 'adapter_model.safetensors').write_bytes(b'private weights')
    result = tool.collect(tmp_path)
    assert result['status'] == 'incomplete' and result['errors'] == 3
    data = unpack(tmp_path / 'handoff')
    assert data['rgb2/epoch_000_rank0_steps.jsonl'] == [{'id': 'readable'}]
    receipt = data['collection_receipt.json']
    assert not receipt['training_accepted']
    assert any(e.get('line') == 2 for e in receipt['errors'])
    assert any('runtime' in p for p in receipt['original_files'])
    assert not any('safetensors' in n for n in data)
    with pytest.raises(FileExistsError):
        tool.collect(tmp_path)


def test_projection_applies_only_to_metadata(tmp_path):
    run = tmp_path / 'run_first_epoch_rgb2'
    run.mkdir()
    rows = [{'id': i} for i in range(20001)]
    (run / 'run.json').write_text(json.dumps({'train_ids': list(range(20001))}))
    (run / 'epoch_000_cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    tool.collect(tmp_path)
    data = unpack(tmp_path / 'handoff')
    assert data['rgb2/epoch_000_cases.jsonl'] == rows
    assert data['rgb2/projected/run.json']['train_ids']['count'] == 20001


def test_budget_excess_is_explicit_in_incomplete_package(tmp_path, monkeypatch):
    run = tmp_path / 'run_first_epoch_rgb2'
    run.mkdir()
    (run / 'epoch_000_cases.jsonl').write_text(json.dumps({'raw': 'x' * 10000}) + '\n')
    monkeypatch.setattr(tool, 'LIMIT', 2_001_000)
    result = tool.collect(tmp_path)
    assert result['status'] == 'incomplete'
    data = unpack(tmp_path / 'handoff')
    assert 'rgb2/epoch_000_cases.jsonl' not in data
    assert data['collection_receipt.json']['errors'][0]['error'] == 'package_budget_exceeded'


def test_logical_link_retarget_is_recorded_as_drift(tmp_path, monkeypatch):
    run = tmp_path / 'run_first_epoch_rgb2'
    run.mkdir()
    old, new = tmp_path / 'old.json', tmp_path / 'new.json'
    old.write_text('{}')
    new.write_text('{}')
    link = run / 'epoch_000_runtime.json'
    link.symlink_to(old)
    original = tool.check_consumption

    def check(*args):
        if link.resolve() == old:
            link.unlink()
            link.symlink_to(new)
        return original(*args)

    monkeypatch.setattr(tool, 'check_consumption', check)
    tool.collect(tmp_path)
    data = unpack(tmp_path / 'handoff')
    assert any('changed during collection' in e['error'] for e in data['collection_receipt.json']['errors'])


def test_complete_synthetic_evidence_never_claims_training_acceptance(tmp_path, plan):
    samples, audit, steps = plan
    for mode in (2, 4):
        run = tmp_path / f'run_first_epoch_rgb{mode}'
        for name in (*tool.NAMES, *tool.METADATA):
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            value = (samples if name == 'epoch_000_samples.jsonl' else
                     steps[int(name.split('rank')[1][0])] if '_steps' in name else
                     audit if name == 'epoch_000_sampling.json' else
                     [] if name.endswith('.jsonl') else {})
            path.write_text(''.join(json.dumps(r) + '\n' for r in value)
                            if name.endswith('.jsonl') else json.dumps(value))
        (tmp_path / f'rgb{mode}.log').write_text('synthetic, no GPU')
    (tmp_path / 'admission_2.json').write_text(json.dumps(dict(
        unsupported_capabilities=['U-E5/proceed/readiness/YES'])))
    result = tool.collect(tmp_path)
    assert result['status'] == 'collected_not_semantically_verified'
    data = unpack(tmp_path / 'handoff')
    assert not data['collection_receipt.json']['training_accepted']
    assert data['admission_2.json']['unsupported_capabilities']
    assert result['consumption']['4']['status'] == 'matched_recorded_plan'
