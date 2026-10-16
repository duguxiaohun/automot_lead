import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
from qwen3vl_local.audit_joint import phase4_first_epoch as tool
from qwen3vl_local.audit_joint.tests.test_phase4_epoch_handoff import unpack


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    prepared=tmp_path/'prepared';model=tmp_path/'model';data=tmp_path/'data';data.mkdir()
    for p in (prepared/'data4/manifest.json',prepared/'references4/references.json',
              prepared/'references4/val.jsonl',prepared/'references4/test.jsonl',
              prepared/'first_epoch_budget.json',model/'config.json'):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    monkeypatch.setattr(tool,'source_check',lambda *a:dict(status='matched'))
    monkeypatch.setattr(tool,'budget_check',lambda *a:dict(sufficient=True))
    return prepared,tmp_path/'output',data,model


@pytest.mark.parametrize('failure',[None,'view_build','rgb2','rgb4'])
def test_sequential_gates_and_failure_handoff(inputs,monkeypatch,failure):
    calls=[]
    def execute(command, log, env):
        stage=log.stem;calls.append((stage,command,env));log.write_text('fixture '+stage)
        return 17 if stage==failure else 0
    monkeypatch.setattr(tool,'execute',execute)
    result=tool.run(*inputs)
    expected=['view_build','rgb2','rgb4'];expected=expected if failure is None else expected[:expected.index(failure)+1]
    assert [r[0] for r in calls]==expected
    assert result==(17 if failure else 2)  # Synthetic runner produces no GPU evidence.
    receipt=json.loads((inputs[1]/'workflow.json').read_text())
    assert receipt['command_exit_code']==(17 if failure else 0)
    assert receipt['training_accepted'] is False
    members=unpack(inputs[1]/'handoff')
    assert members['workflow.json']==receipt
    assert 'fixture view_build' in members['view_build_log_tail.json']['tail']
    assert members['collection_receipt.json']['status']=='incomplete'
    for stage,command,env in calls:
        if stage.startswith('rgb'):
            assert env['RGB_MODE']==stage[-1] and env['RUN_TAG']=='first_epoch'
            assert env['OUTPUT_DIR']==str(inputs[1]) and env['BATCH_SIZE']=='1'
            assert env['UPDATE_SAMPLES']=='32' and env['ACCUMULATION']=='8'
            assert env['AUTOMATIC_COVERAGE_POLICY']=='existing_data_development'
            assert command==['bash','qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh']
    assert '--dataset4' in calls[0][1] and 'teacher_replay' not in ' '.join(calls[0][1])


@pytest.mark.parametrize('defect',['source','missing','budget','budget_type','interrupted'])
def test_preflight_failures_and_interrupt_retain_diagnostics(inputs,monkeypatch,defect):
    calls=[]
    def execute(*args):
        calls.append(1);raise tool.Interrupted(signal.SIGTERM)
    monkeypatch.setattr(tool,'execute',execute)
    if defect=='source':monkeypatch.setattr(tool,'source_check',lambda *a:dict(status='mismatch'))
    if defect=='missing':(inputs[0]/'references4/val.jsonl').unlink()
    if defect=='budget_type':(inputs[0]/'first_epoch_budget.json').write_text('[]')
    if defect=='budget':monkeypatch.setattr(tool,'budget_check',lambda *a:dict(sufficient=False))
    result=tool.run(*inputs)
    assert result==(143 if defect=='interrupted' else 2)
    assert bool(calls)==(defect=='interrupted')
    members=unpack(inputs[1]/'handoff')
    assert members['workflow.json']['status']==('interrupted' if defect=='interrupted' else 'failed')
    assert members['collection_receipt.json']['status']=='incomplete'


def test_reuse_view_keeps_inputs_and_never_skips_auto_admission(inputs,monkeypatch,tmp_path):
    view=tmp_path/'current_view';view.mkdir()
    original=json.dumps(dict(parent=dict(logical_path=str(inputs[0]/'data4'))))
    (view/'student_view.json').write_text(original)
    calls=[]
    def execute(command,log,env):
        calls.append((log.stem,env['DATASET']));return 2
    monkeypatch.setattr(tool,'execute',execute)
    assert tool.run(*inputs,view=view)==2
    assert calls==[('rgb2',str(view))]
    assert (view/'student_view.json').read_text()==original


def test_existing_or_nested_outputs_refused(inputs,monkeypatch):
    monkeypatch.setattr(tool,'execute',lambda *a:pytest.fail('must not launch'))
    prepared,out,data,model=inputs
    out.mkdir();(out/'keep').write_text('keep')
    with pytest.raises(FileExistsError):tool.run(*inputs)
    with pytest.raises(ValueError,match='outside'):tool.run(prepared,prepared/'nested',data,model)
    assert (out/'keep').read_text()=='keep' and not (prepared/'nested').exists()


def test_execute_signal_cleans_owned_guard_worker(tmp_path):
    pidfile=tmp_path/'worker.pid';driver=tmp_path/'driver.py';log=tmp_path/'stage.log'
    driver.write_text('''import os, signal, sys
from pathlib import Path
from qwen3vl_local.audit_joint.phase4_first_epoch import execute, Interrupted
signal.signal(signal.SIGTERM, lambda n,f: (_ for _ in ()).throw(Interrupted(n)))
try:
 execute([sys.executable,'-c',
 'from qwen3vl_local.audit_joint import guard;guard.core_pattern=lambda: chr(99)+chr(111)+chr(114)+chr(101);raise SystemExit(guard.main())',
 '--space-path',sys.argv[1],
 '--min-free-gib','0','--',sys.executable,'-c',
 'import os,time;from pathlib import Path;Path('+repr(sys.argv[2])+').write_text(str(os.getpid()));time.sleep(60)'],Path(sys.argv[3]),dict(os.environ))
except Interrupted:
 sys.exit(143)
''')
    env=dict(os.environ,PYTHONPATH=str(tool.APP))
    proc=subprocess.Popen([sys.executable,str(driver),str(tmp_path),str(pidfile),str(log)],env=env)
    worker=None
    try:
        deadline=time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline and proc.poll() is None:time.sleep(.05)
        assert pidfile.exists(),log.read_text() if log.exists() else 'no log'
        worker=int(pidfile.read_text());proc.terminate();assert proc.wait(timeout=20)==143
        with pytest.raises(ProcessLookupError):os.kill(worker,0)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        if worker:
            try:os.kill(worker,signal.SIGKILL)
            except ProcessLookupError:pass


def test_default_gpu_selection_is_delegated_and_explicit_pin_is_preserved(inputs,monkeypatch):
    monkeypatch.delenv('GPU_IDS',raising=False)
    seen=[]
    monkeypatch.setattr(tool,'execute',lambda command,log,env:seen.append(env['GPU_IDS']) or 2)
    assert tool.run(*inputs)==2 and seen==['']
    prepared,out,data,model=inputs
    monkeypatch.setenv('GPU_IDS','3,2,1,0')
    assert tool.run(prepared,out.with_name('pinned'),data,model)==2 and seen[-1]=='3,2,1,0'


def test_wrong_parent_view_rejected_before_training(inputs,monkeypatch,tmp_path):
    view=tmp_path/'wrong_view';view.mkdir()
    (view/'student_view.json').write_text(json.dumps(dict(parent=dict(logical_path=str(tmp_path/'other')))))
    monkeypatch.setattr(tool,'execute',lambda *a:pytest.fail('wrong parent must not train'))
    assert tool.run(*inputs,view=view)==2
    assert 'requested native dataset' in json.loads((inputs[1]/'workflow.json').read_text())['error']


@pytest.mark.parametrize('fallback,two,four,expected',[
    (1,2,1,{'2':(2,4),'4':(1,8)}), (2,None,1,{'2':(2,4),'4':(1,8)}),
    (1,4,2,{'2':(4,2),'4':(2,4)}), (2,None,None,{'2':(2,4),'4':(2,4)})])
def test_mode_batches_flow_into_admission_training_and_handoff(inputs,monkeypatch,fallback,two,four,expected):
    calls=[]
    # Ignore stale environment values from earlier shell experiments.
    monkeypatch.setenv('BATCH_SIZE','8');monkeypatch.setenv('ACCUMULATION','1')
    def execute(command,log,env):
        log.write_text('fixture')
        if log.stem.startswith('rgb'):calls.append((env['RGB_MODE'],env['BATCH_SIZE'],env['ACCUMULATION'],env['UPDATE_SAMPLES']))
        return 0
    monkeypatch.setattr(tool,'execute',execute)
    assert tool.run(*inputs,batch_size=fallback,batch_size_rgb2=two,batch_size_rgb4=four)==2
    assert calls==[(mode,str(batch),str(accum),'32') for mode,(batch,accum) in expected.items()]
    receipt=unpack(inputs[1]/'handoff')['workflow.json']
    for mode,(batch,accum) in expected.items():
        config=dict(batch_size=batch,accumulation=accum,update_samples=32)
        assert receipt['mode_configuration'][mode]==config
        assert next(s for s in receipt['stages'] if s['stage']=='rgb'+mode)['training_configuration']==config
    assert receipt['batch_size']==(expected['2'][0] if expected['2']==expected['4'] else None)


def test_invalid_mode_batch_refused_before_output(inputs,monkeypatch):
    monkeypatch.setattr(tool,'execute',lambda *a:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='batch'):
        tool.run(*inputs,batch_size_rgb2=3)
    assert not inputs[1].exists()


def test_cli_passes_independent_batch_overrides(inputs,monkeypatch):
    captured={}
    monkeypatch.setattr(tool,'run',lambda *a,**kw:captured.update(kw) or 0)
    monkeypatch.setattr(sys,'argv',['phase4_first_epoch','--prepared',str(inputs[0]),
        '--output',str(inputs[1]),'--batch-size','4','--batch-size-rgb2','2','--batch-size-rgb4','1'])
    assert tool.main()==0
    assert (captured['batch_size'],captured['batch_size_rgb2'],captured['batch_size_rgb4'])==(4,2,1)


@pytest.mark.parametrize('content',['[]','{"parent": null}'])
def test_malformed_reused_view_still_produces_diagnostic_package(inputs,monkeypatch,tmp_path,content):
    view=tmp_path/'bad_view';view.mkdir();(view/'student_view.json').write_text(content)
    monkeypatch.setattr(tool,'execute',lambda *a:pytest.fail('must not launch'))
    assert tool.run(*inputs,view=view)==2
    assert 'valid parent binding' in unpack(inputs[1]/'handoff')['workflow.json']['error']
