"""Fresh orchestration with real regression/packing and substituted GPU workers."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest
from qwen3vl_local.audit_joint.abc_delivery import write
from qwen3vl_local.audit_joint.phase3_abc_refine import __main__ as cli, runtime, bindings
from qwen3vl_local.audit_joint.geometry_abc_v2 import train
from qwen3vl_local.audit_joint.phase3_abc_refine.tests.test_review_fixes import suite


def args_for(root,state,tmp_path):
    return SimpleNamespace(operation='all',run_dir=str(root),output=str(root),
        model_dir=state['model_binding']['path'],baseline_cases_root=str(tmp_path/'baseline_runs'),
        baseline_dataset=state['baseline_dataset']['path'])


@pytest.mark.parametrize('loss',[False,True])
def test_all_without_any_old_weights_trains_three_and_delivers(tmp_path,monkeypatch,loss):
    from qwen3vl_local.action_prior import launch
    root,state,model,records=suite(tmp_path,monkeypatch)
    # Baselines contain JSON only. No original adapters, replay or receipts.
    assert all(p.suffix=='.json' for p in (tmp_path/'baseline_runs').rglob('*') if p.is_file())
    saved={};calls=[]
    for v in 'abc':
        saved[v]=json.loads((root/f'train_{v}/training_complete.json').read_text())
        (root/f'train_{v}/training_complete.json').unlink()
    if loss:
        p=root/'eval_a/cases.json';cases=json.loads(p.read_text())
        cases[0]['prediction']['token_ids'][0]=(cases[0]['prediction']['token_ids'][0]+1)%9
        write(p,cases)
        from qwen3vl_local.audit_joint.geometry_abc_v2 import metrics
        with runtime.activated():write(root/'eval_a/metrics.json',metrics.evaluate(cases))
    def run(command,log):
        assert 'replay' not in command and '--checkpoint' not in command
        assert 'train' in command
        v=command[command.index('--variant')+1];calls.append(v)
        write(root/f'train_{v}/training_complete.json',saved[v])
    monkeypatch.setattr(launch,'ensure_gpu',lambda n:4)
    monkeypatch.setattr(launch,'run_logged',run)
    # suite supplies the native records; this check models the full-source load.
    load=cli.dataset.load
    monkeypatch.setattr(cli.dataset,'load',lambda path,variant='a',**kw:load(path,variant=variant,**kw))
    assert cli.workflow(args_for(root,state,tmp_path))==(2 if loss else 0)
    assert calls==list('abc')
    final=json.loads((root/'pipeline.json').read_text())
    assert final['status']==('tradeoff_review_required' if loss else 'completed')
    assert 'error' not in final and 'baseline_adapters' not in final
    with zipfile.ZipFile(final['delivery']['package']) as z:
        assert json.loads(z.read('pipeline.json'))['status']==final['status']
        assert json.loads(z.read('a/regression.json'))['old_correct_losses'] or not loss
        assert 'baseline_rows' in json.loads(z.read('a/cases.json'))


def test_old_failure_directory_rejected_without_rewriting(tmp_path):
    root=tmp_path/'old';root.mkdir()
    write(root/'pipeline.json',dict(status='failed',baseline_adapters={'a':'old'},groups={}))
    before=(root/'pipeline.json').read_bytes()
    with pytest.raises(ValueError,match='legacy_or_changed_workflow'):
        cli.workflow(SimpleNamespace(operation='all',run_dir=str(root),output=str(root)))
    assert (root/'pipeline.json').read_bytes()==before
    assert not (root/'failure_delivery').exists()


def test_guard_restores_native_function_even_after_failure():
    original=train.restore
    with pytest.raises(ValueError,match='prior_adapter_restore_forbidden'):
        with runtime.fresh_training():train.restore(None,'old_adapter')
    assert train.restore is original


@pytest.mark.parametrize('name',['adapter_config.json','adapter_model.safetensors'])
def test_adapter_model_directory_rejected_before_snapshot(tmp_path,name,monkeypatch):
    (tmp_path/name).write_text('old')
    from qwen3vl_local.audit_joint import phase3_suite
    monkeypatch.setattr(phase3_suite,'model_snapshot',lambda *a:pytest.fail('must reject before model read'))
    with pytest.raises(ValueError,match='original_base_model_required'):bindings.model_identity(tmp_path)


def test_failure_package_keeps_reports_logs_and_original_error(tmp_path,monkeypatch):
    root,state,model,records=suite(tmp_path,monkeypatch)
    (root/'logs').mkdir();(root/'logs/replay_a.log').write_text('x'*65000+'actual rejection')
    write(root/'replay/a/regression.json',{'old_correct_losses':['id']})
    state.update(status='failed',error='injected original failure');write(root/'pipeline.json',state)
    receipt=cli.failure_delivery(root,state)
    with zipfile.ZipFile(receipt['package']) as z:
        assert 'replay/a/regression.json' in z.namelist()
        assert 'processor.json' in z.namelist()
        assert z.read('log_tails/replay_a.log').endswith(b'actual rejection')
        assert len(z.read('log_tails/replay_a.log'))==64000
        assert json.loads(z.read('manifest.json'))['summary']['final_result'] is False


@pytest.mark.parametrize('operation',['train','probe','all'])
def test_prior_checkpoint_rejected_before_model_or_gpu(tmp_path,monkeypatch,operation):
    import sys
    monkeypatch.setattr(sys,'argv',['refine',operation,'--output',str(tmp_path),'--checkpoint','prior'])
    with pytest.raises((ValueError,SystemExit)):cli.main()
    assert not (tmp_path/'pipeline.json').exists()


def test_removed_replay_has_clear_cli_error_without_traceback(tmp_path,monkeypatch,capsys):
    import sys
    monkeypatch.setattr(sys,'argv',['refine','replay','--output',str(tmp_path),'--checkpoint','prior'])
    with pytest.raises(SystemExit) as exc:cli.main()
    assert exc.value.code==2
    error=capsys.readouterr().err
    assert 'legacy adapter replay disabled' in error and 'Traceback' not in error
