"""Control-flow tests; workers are explicit substitutes, not GPU validation."""
import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint.abc_delivery import write,sha
from qwen3vl_local.action_expert_ablation.geometry_abc import __main__ as cli,audit,preflight
from qwen3vl_local.action_prior import launch


def fixture(tmp_path,monkeypatch,*,ready=True,fail_probe=False):
    data=tmp_path/'data';write(data/'status.json',{'data_ready':True})
    root=tmp_path/'run';root.mkdir();calls=[]
    gate=dict(data_ready=True,plan_ready=True,resources={'ready':ready,'errors':[] if ready else ['missing weights']})
    monkeypatch.setattr(preflight,'check',lambda *a,**kw:dict(gate))
    monkeypatch.setattr(preflight,'processor_check',lambda *a,**kw:write(a[3],{'full_pool':True}))
    monkeypatch.setattr(audit,'drill',lambda *a,**kw:None)
    monkeypatch.setattr(audit,'build',lambda *a,**kw:{'package':'test-substitute.zip'})
    monkeypatch.setattr(launch,'ensure_gpu',lambda n:4)
    def run(command,log):
        operation=command[command.index('--operation')+1];out=Path(command[command.index('--output-dir')+1])
        calls.append((operation,out.name))
        if operation=='probe':
            if fail_probe and len(calls)==3:raise RuntimeError('probe gradient failure')
            write(out/'probe_complete.json',{'admission_sha256':sha(root/'admission.json')})
        elif operation=='train':write(out/'training_complete.json',{})
        elif operation=='evaluate':
            write(out/'metrics.json',{});write(out/'condition_sensitivity.json',{})
        else:raise AssertionError(operation)
    monkeypatch.setattr(launch,'run_logged',run)
    args=['all','--run-dir',str(root),'--data-dir',str(data)]
    return args,root,calls


def test_all_six_short_probes_precede_first_long_training(tmp_path,monkeypatch):
    args,root,calls=fixture(tmp_path,monkeypatch)
    cli.main(args)
    assert [op for op,_ in calls]==['probe']*6+['train','evaluate']*6
    assert json.loads((root/'pipeline.json').read_text())['status']=='completed'


def test_probe_failure_keeps_long_training_off_and_packs_failure(tmp_path,monkeypatch):
    args,root,calls=fixture(tmp_path,monkeypatch,fail_probe=True)
    with pytest.raises(RuntimeError,match='probe gradient'):cli.main(args)
    assert [op for op,_ in calls]==['probe']*3
    assert (root/'failure_delivery'/'diagnostic.zip').is_file()
    assert json.loads((root/'pipeline.json').read_text())['status']=='failed'


def test_missing_resources_do_not_claim_runtime_ready(tmp_path,monkeypatch):
    args,root,calls=fixture(tmp_path,monkeypatch,ready=False)
    with pytest.raises(SystemExit) as exc:cli.main(args)
    assert exc.value.code==2 and not calls
    state=json.loads((root/'pipeline.json').read_text())
    assert state['status']=='resources_missing' and not state['admission']['processor_ready']
