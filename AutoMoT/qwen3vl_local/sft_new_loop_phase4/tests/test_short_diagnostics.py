import json
from types import SimpleNamespace

import pytest
import torch
from qwen3vl_local.sft_new_loop_phase4.rgb_short import diagnostics as d


def test_constant_no_is_not_reported_as_good_learning():
    cases = [dict(target=t, prediction='NO', event='U-E1') for t in ['YES', 'NO', 'NO']]
    result = d.diagnostic_report(cases)['overall']
    assert result['recall'] == {'YES': 0, 'NO': 1}
    assert result['balanced_reference_agreement'] == .5
    assert result['majority_reference_baseline'] == 2/3
    assert result['constant_prediction'] == 'NO'
    assert 'zero_yes_recall' in result['warnings']


def test_empty_missing_class_and_malformed_are_explicit():
    assert d.binary_diagnostics([])['balanced_reference_agreement'] is None
    result = d.binary_diagnostics([dict(target='YES', prediction='MALFORMED')])
    assert result['recall'] == {'YES': 0, 'NO': None}
    assert result['balanced_reference_agreement'] is None


def test_order_binding_detects_permutations_with_same_summary():
    rows = [dict(id=str(i), target='YES', label_basis='weak_rule_teacher') for i in range(4)]
    a = d.sample_records(rows, [0, 1, 2, 3], 2)
    b = d.sample_records(rows, [1, 0, 2, 3], 2)
    assert d.sample_binding(a, 2) != d.sample_binding(b, 2)
    assert [r['rank'] for r in a] == [0, 1, 0, 1]


def test_reload_comparison_rejects_missing_changed_reference_and_prediction():
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.reload_check import compare_cases
    row = dict(raw='NO', prediction='NO', paired_identity={'source': 1})
    assert compare_cases({'a': row}, {'a': row})['status'] == 'verified'
    with pytest.raises(ValueError, match='case set'):
        compare_cases({'a': row}, {})
    with pytest.raises(ValueError, match='identity'):
        compare_cases({'a': row}, {'a': {**row, 'paired_identity': {'source': 2}}})
    assert compare_cases({'a': row}, {'a': {**row, 'prediction': 'YES'}})['status'] == 'prediction_mismatch'


def test_handoff_includes_all_new_evidence_but_not_weights(tmp_path):
    from qwen3vl_local.audit_joint.handoff import result_files
    names = ['epoch_000_rank0_steps.jsonl', 'epoch_000_samples.jsonl', 'epoch_000_runtime.json',
             'epoch_000_checkpoint.json', 'checkpoint_reload.json']
    for name in names + ['adapter_model.safetensors', 'training_state.pt']:
        (tmp_path/name).write_text('{}')
    included, omitted = result_files(tmp_path)
    assert set(included) == set(names)
    assert {r['path'] for r in omitted} == {'adapter_model.safetensors', 'training_state.pt'}


def test_cpu_training_writes_actual_steps_order_runtime_and_checkpoint(tmp_path, monkeypatch):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import train
    from qwen3vl_local.qwen35 import adapters
    from qwen3vl_local.sft_new_loop_phase4.identity import file_sha
    rows = [dict(id='first',target='YES',label_basis='weak_rule_teacher'),
            dict(id='second',target='NO',label_basis='reviewed_transition_band')]
    manifest = dict(training_admission={'trainable': True, 'evaluation_scope':'test'},
                    coverage={'ready':False},student_view={'selected_ids':{}},observation_contract={})
    monkeypatch.setattr(train,'load_dataset',lambda *a,**kw:({'train':rows,'val':rows},manifest))
    monkeypatch.setattr(train,'plan',lambda *a,**kw:([0,1,0],{'next_history':{},'unique_questions':2}))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(0.5))
        def forward(self, **kw):
            return SimpleNamespace(loss=(self.weight-1).square())
    class Processor:
        def save_pretrained(self, folder):
            (folder/'processor_config.json').write_text('{}')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(train,'load_bundle',lambda *a,**kw:SimpleNamespace(model=Model(),processor=Processor()))
    monkeypatch.setattr(train,'encode',lambda *a,**kw:{'input_ids':torch.ones(1,3,dtype=torch.long)})
    monkeypatch.setattr(train,'evaluate_rows',lambda *a,**kw:({'selection_agreement':.5},[]))
    def save(model, folder):
        folder.mkdir()
        torch.save(model.state_dict(),folder/'adapter_model.bin')
    monkeypatch.setattr(adapters,'save_adapter',save)
    monkeypatch.delenv('WORLD_SIZE',raising=False)
    monkeypatch.delenv('RANK',raising=False)
    monkeypatch.delenv('LOCAL_RANK',raising=False)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--dataset',str(tmp_path),'--output-dir',str(output),
        '--epochs','1','--accumulation','2','--sampling-policy','legacy_ring'])
    train.main()
    steps=[json.loads(s) for s in (output/'epoch_000_rank0_steps.jsonl').read_text().splitlines()]
    assert [r['id'] for r in steps]==['first','second','first']
    assert [r['optimizer_steps'] for r in steps]==[0,1,2]
    assert steps[-1]['loss'] < steps[0]['loss']
    runtime=json.loads((output/'epoch_000_runtime.json').read_text())
    assert runtime['ranks'][0]['optimizer_steps']==2 and runtime['ranks'][0]['microbatches']==3
    assert runtime['ranks'][0]['peak_allocated_bytes'] is None
    assert runtime['validation_status']=='complete'
    checkpoint=json.loads((output/'epoch_000_checkpoint.json').read_text())
    assert checkpoint['files']['adapter_model.bin']==file_sha(output/'epoch_000/adapter_model.bin')
    assert checkpoint['training_state']['sha256']==file_sha(output/'epoch_000/training_state.pt')
    records=[json.loads(s) for s in (output/'epoch_000_samples.jsonl').read_text().splitlines()]
    sampling=json.loads((output/'epoch_000_sampling.json').read_text())
    assert sampling['ordered_samples']==d.sample_binding(records,1)


@pytest.mark.parametrize('damage', [None, 'weight', 'optimizer', 'prediction', 'load', 'validation', 'interrupt'])
def test_reload_cli_checks_saved_bytes_before_loader_and_records_comparison(tmp_path, monkeypatch, damage):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import reload_check as module
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract, observation_contract
    from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row
    from qwen3vl_local.sft_new_loop_phase4.paired_eval import case_identity
    from qwen3vl_local.sft_new_loop_phase4.identity import digest, file_sha, write_json
    row=source_row(tmp_path)
    row['split']='val'
    manifest={'observation_contract':observation_contract(4)}
    run=dict(source=contract(),dataset=digest(manifest),observation_contract=manifest['observation_contract'],training={'max_length':8192})
    folder=tmp_path/'run'; adapter=folder/'epoch_000'; adapter.mkdir(parents=True)
    write_json(folder/'run.json',run);write_json(adapter/'phase4_contract.json',run)
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture weights')
    (adapter/'training_state.pt').write_bytes(b'fixture optimizer')
    write_json(folder/'latest.json',dict(adapter='epoch_000',epoch=0))
    write_json(folder/'epoch_000_checkpoint.json',dict(epoch=0,adapter='epoch_000',files=module.adapter_identity(adapter),
        training_state=dict(bytes=(adapter/'training_state.pt').stat().st_size,sha256=file_sha(adapter/'training_state.pt'))))
    case=dict(id=row['id'],target=row['target'],prediction='YES',raw='YES',event=row['episode']['event'],
        edge=row['edge'],slice=row['slice'],reference_kind=row.get('reference_kind','reviewed_rgb'),paired_identity=case_identity(row))
    (folder/'epoch_000_cases.jsonl').write_text(json.dumps(case)+'\n')
    calls=[]
    monkeypatch.setattr(module,'load_dataset',lambda *a,**kw:({'val':[row]},manifest))
    monkeypatch.setattr(module,'load_for_inference',lambda *a: calls.append(a))
    monkeypatch.setattr(module,'evaluate_rows',lambda *a: ({'count':1},[{**case,'prediction':'NO' if damage=='prediction' else 'YES'}]))
    if damage == 'load':
        def failed_load(*args):
            calls.append(args)
            raise RuntimeError('fixture model load failure')
        monkeypatch.setattr(module, 'load_for_inference', failed_load)
    generated = []
    if damage in ('validation', 'interrupt'):
        # Exercise the real validation loop: the second generation fails after one result.
        from copy import deepcopy
        from qwen3vl_local.sft_new_loop_phase4.rgb_short import evaluate
        other = deepcopy(row)
        other['id'] += '-second'
        second_case = {**case, 'id':other['id'], 'paired_identity':case_identity(other)}
        (folder/'epoch_000_cases.jsonl').write_text(json.dumps(case)+'\n'+json.dumps(second_case)+'\n')
        monkeypatch.setattr(module, 'load_dataset', lambda *a, **kw: ({'val':[row, other]}, manifest))
        def generate(*args):
            generated.append(args[1]['id'])
            if len(generated) == 2:
                if damage == 'interrupt':
                    raise KeyboardInterrupt('fixture interrupted')
                raise RuntimeError('fixture second generation failure')
            return 'YES', 'YES'
        monkeypatch.setattr(evaluate, 'generate', generate)
        monkeypatch.setattr(module, 'evaluate_rows', evaluate.evaluate_rows)
    monkeypatch.setattr(sys,'argv',['reload','--run',str(folder),'--dataset',str(tmp_path),'--data-root',str(tmp_path),
        '--model-dir',str(tmp_path),'--device','cpu'])
    if damage in ('weight','optimizer'):
        target=adapter/('adapter_model.safetensors' if damage=='weight' else 'training_state.pt')
        target.write_bytes(b'changed')
    latest_before = (folder/'latest.json').read_bytes()
    assert module.main() == (130 if damage == 'interrupt' else 2 if damage else 0)
    report_path = folder/'reload_check/checkpoint_reload.json'
    report = json.loads(report_path.read_text())
    assert (folder/'latest.json').read_bytes() == latest_before
    failures = {'weight':'checkpoint_assets', 'optimizer':'optimizer_bytes', 'load':'model_load',
                'validation':'validation', 'interrupt':'validation'}
    if damage in failures:
        assert report['status'] == 'failed'
        assert report['stage'] == failures[damage]
        assert report['error']['message']
        assert report['error']['type'] == ('KeyboardInterrupt' if damage == 'interrupt' else
                                           'ValueError' if damage in ('weight','optimizer') else 'RuntimeError')
        assert report['checkpoint_assets_verified'] == (damage != 'weight')
        assert report['optimizer_bytes_verified'] == (damage not in ('weight','optimizer'))
        assert not (folder/'reload_check/metrics.json').exists()
    else:
        assert report['status'] == ('prediction_mismatch' if damage == 'prediction' else 'verified')
        assert report['stage'] == 'complete'
    assert len(calls) == (0 if damage in ('weight', 'optimizer') else 1)
    if damage in ('validation', 'interrupt'):
        assert generated == [row['id'], other['id']]
    # A failed or successful prior attempt is immutable on retry.
    before = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        module.main()
    assert report_path.read_bytes() == before
    assert_reload_report_in_archive(tmp_path, folder, before)


def assert_reload_report_in_archive(root, run, expected):
    import zipfile
    from qwen3vl_local.audit_joint import SCHEMA
    from qwen3vl_local.audit_joint.handoff import pack, verify_package
    from qwen3vl_local.audit_joint.io import file_sha, write_json
    capture = root/'capture'
    capture.mkdir()
    for name, value in [('request.json', {}), ('source_manifest.json', {}),
                        ('split_cross_audit.json', {'conflicts': []}),
                        ('baseline_manifest.json', dict(reproducible_baseline_ready=False,
                         blockers=[{'kind':'execution','status':'not_run'}], stages={'E0':'not_run'}, artifacts=[]))]:
        write_json(capture/name, value)
    (capture/'exposure_ledger.jsonl').write_text('')
    write_json(capture/'receipt.json', dict(schema=SCHEMA, files={
        p.name:file_sha(p) for p in capture.iterdir()}))
    packed = pack(capture, results=[('reload', run)])
    assert verify_package(packed['archive'])['status'] == 'verified'
    with zipfile.ZipFile(packed['archive']) as archive:
        assert archive.read('results/reload/reload_check/checkpoint_reload.json') == expected
        assert not any(n.endswith(('.safetensors', '.pt')) for n in archive.namelist())


def test_reload_missing_run_retains_initial_failure(tmp_path, monkeypatch):
    import sys
    from qwen3vl_local.sft_new_loop_phase4.rgb_short import reload_check as module
    monkeypatch.setattr(sys, 'argv', ['reload', '--run', str(tmp_path/'absent'),
        '--dataset', str(tmp_path), '--data-root', str(tmp_path), '--model-dir', str(tmp_path), '--device', 'cpu'])
    assert module.main() == 2
    report = json.loads((tmp_path/'absent/reload_check/checkpoint_reload.json').read_text())
    assert report['status'] == 'failed' and report['stage'] == 'run_contract'
    assert report['error']['type'] == 'FileNotFoundError'
    assert 'adapter' not in report and 'epoch' not in report
