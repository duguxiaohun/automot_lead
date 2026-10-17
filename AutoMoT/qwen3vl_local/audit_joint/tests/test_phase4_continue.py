import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase4_continue as tool
from qwen3vl_local.audit_joint.tests.test_phase4_epoch_handoff import plan,unpack
from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract
from qwen3vl_local.sft_new_loop_phase4.identity import write_json


@pytest.fixture
def completed(tmp_path,plan):
    root=tmp_path/'run';run=root/'run_first_epoch_rgb2';(run/'epoch_000').mkdir(parents=True)
    view=root/'view';view.mkdir();prepared=tmp_path/'prepared'
    (root/'validated_data').mkdir()
    for name in ('rgb2.json','rgb2.receipt.json'):(root/'validated_data'/name).write_text('{}')
    state=dict(stage='rgb2',command_exit_code=2,status='failed',dataset=str(view),prepared=str(prepared),stages=[])
    write_json(root/'workflow.json',state)
    write_json(view/'student_view.json',dict(source=contract(),parent={'logical_path':str(prepared/'data4')}))
    ids=[f'v{i}' for i in range(370)]
    record=dict(source=contract(),world_size=4,observation_contract={'rgb_mode':2},training={'evaluation':{'plans':{'val':{'selected_ids':ids}}}})
    for path in (run/'run.json',run/'epoch_000/phase4_contract.json'):write_json(path,record)
    (run/'epoch_000/adapter_model.safetensors').write_bytes(b'fixture')
    samples,sampling,ranks=plan;write_json(run/'epoch_000_sampling.json',sampling)
    data={'epoch_000_samples.jsonl':samples,'epoch_000_cases.jsonl':[dict(id=i) for i in ids]}
    data.update({f'epoch_000_rank{i}_steps.jsonl':rank for i,rank in enumerate(ranks)})
    for name,rows in data.items():(run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root


@pytest.mark.parametrize('damage',['none','weight','source','steps','val','existing'])
def test_preflight_requires_real_completed_epoch(completed,damage):
    root=completed;run=root/'run_first_epoch_rgb2'
    if damage=='weight':(run/'epoch_000/adapter_model.safetensors').unlink()
    elif damage=='source':
        p=run/'run.json';r=tool.read(p);r['source']['policy']='wrong';write_json(p,r)
    elif damage=='steps':(run/'epoch_000_rank3_steps.jsonl').write_text('')
    elif damage=='val':(run/'epoch_000_cases.jsonl').write_text('')
    elif damage=='existing':(root/'run_first_epoch_rgb4').mkdir()
    if damage=='none':assert tool.preflight(root)[3]==run
    else:
        with pytest.raises((ValueError,FileExistsError)):tool.preflight(root)


@pytest.mark.parametrize('failure',[None,'snapshot_check','rgb2_reload','rgb2_test','rgb4','raise_after_reload'])
def test_continuation_never_retrains_rgb2_and_stops_at_failure(completed,monkeypatch,failure):
    calls=[];original=(completed/'workflow.json').read_bytes()
    monkeypatch.setattr(tool.workflow,'source_check',lambda *a:{'status':'matched'})
    monkeypatch.setattr(tool,'collect',lambda *a:{'status':'collected_not_semantically_verified'})
    def execute(command,log,env):
        stage=log.stem.replace('continuation_','');calls.append((stage,command,env))
        assert env['PYTHON'].endswith('python_dispatch.sh')
        if stage=='rgb2_test' and failure=='raise_after_reload':raise OSError('fixture')
        return 2 if failure==stage else 0
    monkeypatch.setattr(tool.workflow,'execute',execute)
    code=tool.run(completed,Path('lead_data'),Path('model'))
    assert code==(0 if failure is None else 2)
    assert [c[0] for c in calls]==['snapshot_check']+([] if failure=='snapshot_check' else ['rgb2_reload'] if failure=='rgb2_reload' else
        ['rgb2_reload','rgb2_test'] if failure in ('rgb2_test','raise_after_reload') else ['rgb2_reload','rgb2_test','rgb4'])
    assert all(c[2]['RGB_MODE']=='4' for c in calls)
    assert 'run_automatic.sh' not in ' '.join(calls[0][1])
    assert tool.read(completed/'workflow_before_continuation.json')==json.loads(original)
    assert tool.read(completed/'workflow.json')['command_exit_code']==code


def test_collector_preserves_continuation_failure(tmp_path):
    write_json(tmp_path/'continuation.json',dict(status='failed',command_exit_code=2))
    (tmp_path/'continuation_rgb2_reload.log').write_text('fixture failure')
    tool.collect(tmp_path,tmp_path/'new_handoff')
    members=unpack(tmp_path/'new_handoff')
    assert members['continuation.json']['status']=='failed'
    assert members['continuation_rgb2_reload_log_tail.json']['tail']=='fixture failure'


def make_reload(root):
    run=root/'run_first_epoch_rgb2';record=tool.read(run/'run.json');record['dataset']='fixture-dataset'
    for p in (run/'run.json',run/'epoch_000/phase4_contract.json'):write_json(p,record)
    adapter=run/'epoch_000';state=adapter/'training_state.pt';state.write_bytes(b'fixture optimizer')
    write_json(run/'epoch_000_checkpoint.json',dict(epoch=0,adapter='epoch_000',
        files={p.name:tool.file_sha(p) for p in adapter.iterdir() if p!=state},
        training_state=dict(bytes=state.stat().st_size,sha256=tool.file_sha(state))))
    folder=run/'reload_check';folder.mkdir();(folder/'cases.jsonl').write_bytes((run/'epoch_000_cases.jsonl').read_bytes())
    write_json(folder/'checkpoint_reload.json',dict(status='verified',stage='complete',epoch=0,
        world_size=4,count=370,expected_count=370,prediction_mismatch_ids=[],raw_text_differences=0,
        checkpoint_assets_verified=True,optimizer_bytes_verified=True,source=record['source'],dataset=record['dataset'],
        expected_cases_sha256=tool.file_sha(run/'epoch_000_cases.jsonl'),
        checkpoint_receipt_sha256=tool.file_sha(run/'epoch_000_checkpoint.json')))
    workflow=tool.read(root/'workflow.json');workflow['command_exit_code']=1;write_json(root/'workflow.json',workflow)
    return run


@pytest.mark.parametrize('damage',[None,'weight','optimizer','cases','receipt','missing_snapshot'])
def test_verified_reload_reuse_binds_real_assets(completed,damage):
    run=make_reload(completed)
    paths={'weight':run/'epoch_000/adapter_model.safetensors','optimizer':run/'epoch_000/training_state.pt',
        'cases':run/'reload_check/cases.jsonl','receipt':run/'epoch_000_checkpoint.json'}
    if damage in paths:paths[damage].write_text('{}')
    if damage=='missing_snapshot':(completed/'validated_data/rgb2.receipt.json').unlink()
    if damage is None:assert tool.preflight(completed)[3]==run
    else:
        with pytest.raises((ValueError,KeyError)):tool.preflight(completed)


def test_reuse_reload_does_not_run_train_or_reload_again(completed,monkeypatch):
    make_reload(completed);calls=[]
    monkeypatch.setattr(tool.workflow,'source_check',lambda *a:{'status':'matched'})
    monkeypatch.setattr(tool,'collect',lambda *a:{'status':'collected_not_semantically_verified'})
    def execute(command,log,env):calls.append((log.stem,command));return 0
    monkeypatch.setattr(tool.workflow,'execute',execute)
    assert tool.run(completed,Path('lead_data'),Path('model'))==0
    assert [c[0] for c in calls]==['continuation_snapshot_check','continuation_rgb2_test','rgb4']
    assert tool.read(completed/'continuation.json')['rgb2_reload_reused'] is True
    assert all('reload_check' not in c[1] for c in calls)


@pytest.mark.parametrize('pin',['0','0,1,2','0,1,2,2','0,1,2,x','0,00,1,2'])
def test_invalid_gpu_pin_leaves_run_untouched(completed,monkeypatch,pin):
    monkeypatch.setenv('GPU_IDS',pin)
    monkeypatch.setattr(tool,'preflight',lambda root:pytest.fail('GPU pin must be checked before preflight'))
    before={p.relative_to(completed):p.read_bytes() for p in completed.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='four distinct'):
        tool.run(completed,Path('lead_data'),Path('model'))
    assert before=={p.relative_to(completed):p.read_bytes() for p in completed.rglob('*') if p.is_file()}


def test_dangling_continuation_output_is_not_overwritten(completed):
    target=completed/'continuation.json';target.symlink_to(completed/'missing.json')
    with pytest.raises(FileExistsError):tool.preflight(completed)
    assert target.is_symlink() and not (completed/'missing.json').exists()


def test_directory_lock_blocks_alias_and_releases_after_process_exit(tmp_path):
    import subprocess
    import sys
    root=tmp_path/'run';root.mkdir();alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    child=subprocess.Popen([sys.executable,'-c',
        'from pathlib import Path; import sys; from qwen3vl_local.audit_joint.phase4_continue import continuation_lock; '
        'lock=continuation_lock(Path(sys.argv[1])); lock.__enter__(); print("locked",flush=True); sys.stdin.read()',str(root)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        with pytest.raises(RuntimeError,match='already active'):
            with tool.continuation_lock(alias):pytest.fail('concurrent continuation admitted')
    finally:
        child.terminate();child.wait(timeout=10)
    with tool.continuation_lock(alias):pass
    assert list(root.iterdir())==[]


def test_run_holds_lock_through_collection(completed,monkeypatch):
    make_reload(completed)
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setattr(tool.workflow,'source_check',lambda *a:{'status':'matched'})
    def verify_locked(*args):
        with pytest.raises(RuntimeError,match='already active'):
            with tool.continuation_lock(completed):pass
        return 0
    monkeypatch.setattr(tool.workflow,'execute',verify_locked)
    def collect(*args):
        verify_locked();return {'status':'collected_not_semantically_verified'}
    monkeypatch.setattr(tool,'collect',collect)
    assert tool.run(completed,Path('lead_data'),Path('model'))==0
    with tool.continuation_lock(completed):pass
