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
            assert env['OUTPUT_DIR']==str(inputs[1]) and env['BATCH_SIZE']=='2'
            assert env['UPDATE_SAMPLES']=='32' and env['ACCUMULATION']=='4'
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


def test_split_view_and_budget_with_no_old_budget(inputs, monkeypatch, tmp_path):
    continuation = tmp_path/'continuation'
    view = continuation/'view'
    view.mkdir(parents=True)
    (view/'student_view.json').write_text(json.dumps({'parent': {'logical_path': str(inputs[0]/'data4')}}))
    budget = continuation/'first_epoch_budget.json'
    budget.write_text('{"fresh_budget": true}')
    (inputs[0]/'first_epoch_budget.json').unlink()
    before = {p: p.read_bytes() for root in (inputs[0], continuation) for p in root.rglob('*') if p.is_file()}
    seen = []
    def check(request, output):
        assert request == {'fresh_budget': True} and output == inputs[1]
        return {'sufficient': True}
    monkeypatch.setattr(tool, 'budget_check', check)
    def execute(command, log, env):
        seen.append((log.stem, env))
        return 0
    monkeypatch.setattr(tool, 'execute', execute)
    assert tool.run(*inputs, view=view, budget=budget) == 2  # No fabricated GPU results.
    assert [stage for stage, _ in seen] == ['rgb2', 'rgb4']
    for _, env in seen:
        assert env['DATASET'] == str(view)
        assert env['TEACHER_REFERENCES'] == str(inputs[0]/'references4')
        assert env['FIRST_EPOCH_BUDGET'] == str(budget)
    workflow = unpack(inputs[1]/'handoff')['workflow.json']
    assert workflow['budget_request_path'] == str(budget)
    assert workflow['budget_request_sha256'] == tool.file_sha(budget)
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize('content', [None, '[]', '{broken'])
def test_explicit_bad_budget_never_falls_back_to_old_budget(inputs, monkeypatch, tmp_path, content):
    budget = tmp_path/'fresh_budget.json'
    if content is not None:
        budget.write_text(content)
    monkeypatch.setattr(tool, 'execute', lambda *a: pytest.fail('must not launch'))
    assert tool.run(*inputs, budget=budget) == 2
    workflow = unpack(inputs[1]/'handoff')['workflow.json']
    assert workflow['status'] == 'failed' and workflow['budget_request_path'] == str(budget)


def test_budget_cannot_be_nested_in_new_output(inputs):
    with pytest.raises(ValueError, match='budget input must be outside'):
        tool.run(*inputs, budget=inputs[1]/'first_epoch_budget.json')
    assert not inputs[1].exists()


def test_known_old_view_upgrades_once_before_training_with_shared_preparation(inputs,monkeypatch,tmp_path):
    from qwen3vl_local.audit_joint import phase4_view_upgrade
    view=tmp_path/'old_view';view.mkdir()
    descriptor=dict(parent=dict(logical_path=str(inputs[0]/'data4')),source={'old':True})
    original=json.dumps(descriptor);(view/'student_view.json').write_text(original)
    monkeypatch.setattr(phase4_view_upgrade,'inspect_view',lambda v:(descriptor,{},['data.py']))
    monkeypatch.setenv('P4_PRODUCTION_CACHE','verified/routes')
    calls=[]
    def execute(command,log,env):
        calls.append((log.stem,command,env));log.write_text('fixture')
        return 0 if log.stem=='view_upgrade' else 2
    monkeypatch.setattr(tool,'execute',execute)
    assert tool.run(*inputs,view=view)==2
    assert [x[0] for x in calls]==['view_upgrade','rgb2']
    args=calls[0][1]
    assert args[args.index('--production-cache')+1]=='verified/routes'
    assert calls[1][2]['DATASET']==str(inputs[1]/'view_upgrade/view')
    assert calls[1][2]['P4_VALIDATED_DATA']==str(inputs[1]/'validated_data')
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
        '--output',str(inputs[1]),'--budget','fresh/budget.json','--batch-size','4','--batch-size-rgb2','2','--batch-size-rgb4','1'])
    assert tool.main()==0
    assert (captured['batch_size'],captured['batch_size_rgb2'],captured['batch_size_rgb4'])==(4,2,1)
    assert captured['budget']==Path('fresh/budget.json')


def test_cli_default_is_selected_b2(inputs,monkeypatch):
    captured={}
    monkeypatch.setattr(tool,'run',lambda *a,**kw:captured.update(kw) or 0)
    monkeypatch.setattr(sys,'argv',['phase4_first_epoch','--prepared',str(inputs[0]),
                                  '--output',str(inputs[1])])
    assert tool.main()==0 and captured['batch_size']==2


@pytest.mark.parametrize('override',[{}, {'BATCH_SIZE':'1'}, {'ACCUMULATION':'8'},
                                    {'UPDATE_SAMPLES':'16'}, {'GRADIENT_CHECKPOINTING':'0'},
                                    {'EPOCHS':'2'}, {'RESUME_RGB2':'old checkpoint'}])
def test_fixed_b2_shell_forwards_profile_or_rejects_conflicts(tmp_path,override):
    stub=tmp_path/'python';captured=tmp_path/'arguments.json'
    stub.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\n'
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(dict(args=sys.argv[1:],'
        'env={k:os.environ.get(k) for k in ("BATCH_SIZE","ACCUMULATION","UPDATE_SAMPLES","GPU_IDS")})))\n')
    stub.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in
         ('BATCH_SIZE','ACCUMULATION','UPDATE_SAMPLES','GRADIENT_CHECKPOINTING',
          'EPOCHS','RESUME_RGB2','RESUME_RGB4')}
    env.update(PYTHON=str(stub),CAPTURE=str(captured),PREPARED='prepared bundle',
               DATASET='existing view',P4_PRODUCTION_CACHE='route cache',
               OUTPUT_DIR=str(tmp_path/'new output'),GPU_IDS='7,6,5,4',FIRST_EPOCH_BUDGET='fresh budget/request.json',**override)
    result=subprocess.run(['bash',str(tool.APP/'qwen3vl_local/audit_joint/train_phase4_batch.sh')],
                          env=env,text=True,capture_output=True)
    if override:
        assert result.returncode==2 and not captured.exists()
    else:
        assert result.returncode==0
        record=json.loads(captured.read_text());args=record['args']
        assert args[:2]==['-m','qwen3vl_local.audit_joint.phase4_first_epoch']
        for key,value in [('--prepared','prepared bundle'),('--view','existing view'),('--budget','fresh budget/request.json'),
                          ('--batch-size','2'),('--batch-size-rgb2','2'),('--batch-size-rgb4','2')]:
            assert args[args.index(key)+1]==value
        assert record['env']==dict(BATCH_SIZE='2',ACCUMULATION='4',UPDATE_SAMPLES='32',GPU_IDS='7,6,5,4')
    assert not (tmp_path/'new output').exists()  # No actual training or view preparation.


@pytest.mark.parametrize('content',['[]','{"parent": null}'])
def test_malformed_reused_view_still_produces_diagnostic_package(inputs,monkeypatch,tmp_path,content):
    view=tmp_path/'bad_view';view.mkdir();(view/'student_view.json').write_text(content)
    monkeypatch.setattr(tool,'execute',lambda *a:pytest.fail('must not launch'))
    assert tool.run(*inputs,view=view)==2
    assert 'valid parent binding' in unpack(inputs[1]/'handoff')['workflow.json']['error']
