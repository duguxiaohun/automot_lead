"""Saved-result verification uses real receipts and scoring, without a model."""
import importlib
import json
from pathlib import Path

import pytest

from qwen3vl_local.audit_joint import phase3_compare as entry
from qwen3vl_local.audit_joint.phase3_completion import digest, file_sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_study(app, method, monkeypatch):
    package = entry.package(method)
    module = importlib.import_module('qwen3vl_local.audit_joint.'+package)
    dataset = importlib.import_module(module.__name__+'.dataset')
    regions = importlib.import_module(module.__name__+'.regions')
    objective = importlib.import_module(module.__name__+'.objective')
    metrics = importlib.import_module(module.__name__+'.metrics')
    from qwen3vl_local.audit_joint.geometry_study.tests.test_study import record
    variant = 'six' if method == 'single' else method
    r = record(variant=variant); r['split'] = 'test'
    root = entry.root_for(app, method); run = root/'runs'/f'{variant}_1280'
    members = {}
    for v in regions.VARIANTS:
        row = record(variant=v); row['split'] = 'test'; members[v] = [row]
    data = root/'data.json'; write(data, dict(records_by_variant=members))
    # Dataset parsing/provenance is covered by the native preparation suite.
    # Keep this fixture small while retaining exact bytes/members and real metrics.
    def load(path, *, variant, check_sources, expected_sha256):
        assert not check_sources and file_sha(path) == expected_sha256
        return dict(records=json.loads(Path(path).read_text())['records_by_variant'][variant])
    monkeypatch.setattr(dataset, 'load', load)
    admission = dict(status='admitted', fixed_validation_ids=['val'])
    admission['sha256'] = digest(admission)
    projection = dict(plans=[dict(selected=12800, optimizer_steps=3200,
        ordered_ids=[f'train/{i}' for i in range(12800)]) for _ in range(3)])
    projection['sha256'] = digest(projection)
    for name, value in [('admission', admission), ('plan', projection)]:
        write(root/(name+'.json'), value); write(run/(name+'.json'), value)
    ready = dict(status='ready', optimizer_steps=[3200]*3, dataset=str(data), dataset_sha256=file_sha(data),
        admission=str(root/'admission.json'), admission_sha256=admission['sha256'],
        plan=str(root/'plan.json'), plan_sha256=projection['sha256'])
    write(root/'ready_1280.json', ready)
    binding = dict(comparison_policy=entry.policy.contract(), code=dataset.code_identity(),
        contract=regions.contract(variant), dataset_sha256=ready['dataset_sha256'],
        admission_sha256=admission['sha256'], shared_plan_sha256=projection['sha256'])
    receipts = {}
    for epoch, plan in enumerate(projection['plans'], 1):
        folder = run/f'epoch_{epoch}'; folder.mkdir(parents=True)
        (folder/'state.pt').write_bytes(b'synthetic checkpoint bytes, not model weights')
        write(folder/'receipt.json', dict(binding=binding, epoch=epoch, step=epoch*3200,
            status='epoch_complete', state_sha256=file_sha(folder/'state.pt')))
        receipts[folder.name] = file_sha(folder/'receipt.json')
        write(run/f'epoch_{epoch}.json', dict(plan=plan, result=dict(steps=3200, status='epoch_complete',
            actual_exposure={k: epoch for k in plan['ordered_ids']})))
    write(run/'training_complete.json', dict(total_steps=9600, selected_checkpoint='epoch_3',
        selection='final_completed_epoch', binding=binding, checkpoint_receipts=receipts))
    pred = dict(token_ids=[5 if variant == 'six' else 4]*4, format_valid=[True]*4,
        contract=regions.contract(variant), sample_binding=objective.observation_binding(r), source='predicted')
    cases = [dict(record=r, prediction=pred)]
    write(run/'test/cases.json', cases); write(run/'test/metrics.json', metrics.evaluate(cases))
    return run


@pytest.mark.parametrize('method', ['single', 'a', 'b', 'c'])
def test_saved_study_completion_checks_real_metrics_and_weights(tmp_path, monkeypatch, method):
    run = completed_study(tmp_path, method, monkeypatch)
    before = {str(p): file_sha(p) for p in run.rglob('*') if p.is_file()}
    result = entry.verify_completion(tmp_path, method)
    assert result['training_verification']['test_cases'] == 1
    assert before == {str(p): file_sha(p) for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['missing_weights', 'weights', 'receipt', 'consumption',
    'empty_metrics', 'metrics', 'missing_case', 'wrong_record', 'wrong_variant', 'source', 'plan'])
def test_verify_rejects_damaged_study_without_success_receipt(tmp_path, monkeypatch, damage):
    run = completed_study(tmp_path, 'single', monkeypatch)
    if damage == 'missing_weights': (run/'epoch_3/state.pt').unlink()
    elif damage == 'weights': (run/'epoch_3/state.pt').write_bytes(b'changed')
    elif damage == 'receipt': write(run/'epoch_3/receipt.json', {})
    elif damage == 'consumption':
        p = run/'epoch_3.json'; value = json.loads(p.read_text())
        value['result']['actual_exposure']['train/0'] -= 1; write(p, value)
    elif damage in ('empty_metrics', 'metrics'):
        p = run/'test/metrics.json'; value = json.loads(p.read_text())
        value = {} if damage == 'empty_metrics' else dict(value, macro_recall=-1)
        write(p, value)
    elif damage in ('missing_case', 'wrong_record', 'wrong_variant'):
        p = run/'test/cases.json'; value = json.loads(p.read_text())
        if damage == 'missing_case': value = []
        elif damage == 'wrong_record': value[0]['record']['targets'][0]['valid'] = False
        else: value[0]['record']['variant'] = 'a'
        write(p, value)
    elif damage == 'source':
        p = run/'training_complete.json'; value = json.loads(p.read_text())
        value['binding']['code'] = {}; write(p, value)
    else: write(run/'plan.json', {})
    monkeypatch.setattr(entry, 'APP', tmp_path)
    monkeypatch.setattr(entry, 'preflight', lambda *a: pytest.fail('GPU preflight'))
    monkeypatch.setattr(entry, 'commands', lambda *a, **k: pytest.fail('training command'))
    monkeypatch.setattr('sys.argv', ['compare', 'verify', 'single'])
    with pytest.raises((ValueError, KeyError, FileNotFoundError)): entry.main()
    assert not (tmp_path/'checkpoints/phase3_compare/single_completed.json').exists()
