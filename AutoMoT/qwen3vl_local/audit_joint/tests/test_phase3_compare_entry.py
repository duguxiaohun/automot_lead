import json
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import phase3_compare as entry

def test_cpu_all_prepares_abc_once_stops_on_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'APP',tmp_path)
    monkeypatch.setattr('sys.argv',['compare','cpu','all'])
    from qwen3vl_local.audit_joint import phase3_maps
    monkeypatch.setattr(phase3_maps,'resolve',lambda *a:(tmp_path/'maps',{}))
    commands=[]
    monkeypatch.setattr(entry.subprocess,'run',lambda cmd,**kw:commands.append(cmd) or SimpleNamespace(returncode=0))
    assert entry.main()==0 and len(commands)==3
    assert 'qwen3vl_local.audit_joint.phase3_base_bounded' in commands[0] and commands[0][-1]=='cpu'
    assert any('single_v3.cpu' in arg for arg in commands[1])
    commands.clear()
    monkeypatch.setattr(entry.subprocess,'run',lambda cmd,**kw:commands.append(cmd) or SimpleNamespace(returncode=2))
    assert entry.main()==2 and len(commands)==1

def test_train_all_preflights_before_launch_and_clears_custom_run_name(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'APP',tmp_path)
    monkeypatch.setattr('sys.argv',['compare','train','all'])
    seen=[]
    monkeypatch.setattr(entry,'preflight',lambda app,methods:seen.append(('check',methods)) or '0,1,2,3')
    monkeypatch.setenv('STUDY_RUN_NAME','collapse');monkeypatch.setenv('STUDY_PER_EVENT','1024')
    def run(cmd,**kw):
        assert seen[0][0]=='check' and 'STUDY_RUN_NAME' not in kw['env']
        assert kw['env']['STUDY_PER_EVENT']=='1280' and kw['env']['GPU_IDS']=='0,1,2,3'
        seen.append(('run',cmd));return SimpleNamespace(returncode=7)
    monkeypatch.setattr(entry.subprocess,'run',run)
    assert entry.main()==7 and len(seen)==2

def test_status_and_contract_do_not_launch_or_create_directory(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(entry,'APP',tmp_path)
    monkeypatch.setattr(entry.subprocess,'run',lambda *a,**k:pytest.fail('read only'))
    for operation in ('status','contract'):
        monkeypatch.setattr('sys.argv',['compare',operation]);assert entry.main()==0
        json.loads(capsys.readouterr().out)
    assert not (tmp_path/'checkpoints').exists()

def test_collision_in_any_later_arm_blocks_before_ready_or_gpu(tmp_path,monkeypatch):
    # Check uses current order; isolate a later output collision through all-arm scan.
    root=entry.root_for(tmp_path,'a')/'runs/c_1280';root.mkdir(parents=True)
    from qwen3vl_local.audit_joint.geometry_abc_v2 import prepare
    monkeypatch.setattr(prepare,'load_ready',lambda *a,**k:dict(optimizer_steps=[3200]*3))
    monkeypatch.setattr(entry,'selected_gpus',lambda *a:pytest.fail('GPU must not be queried'))
    with pytest.raises(FileExistsError):entry.preflight(tmp_path,['a','b','c'])

def test_completed_receipt_requires_actual_final_budget(tmp_path):
    from qwen3vl_local.audit_joint import phase3_comparison_policy as policy
    run=entry.root_for(tmp_path,'single')/'runs/six_1280';(run/'test').mkdir(parents=True)
    (run/'test/metrics.json').write_text('{}')
    state=dict(total_steps=9600,selected_checkpoint='epoch_3',selection='final_completed_epoch',
        binding=dict(comparison_policy=policy.contract()))
    p=run/'training_complete.json';p.write_text(json.dumps(state))
    # A completion claim plus an empty metrics file is not training evidence.
    with pytest.raises(FileNotFoundError):entry.verify_completion(tmp_path,'single')
    state['selected_checkpoint']='epoch_1';p.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='comparison_contract_mismatch'):entry.verify_completion(tmp_path,'single')

def test_cpu_maps_geometry_and_exposure_forwarding(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint import phase3_maps
    monkeypatch.setattr(entry,'APP',tmp_path);seen=[]
    def resolve(app,view,explicit):
        assert explicit==tmp_path/'custom';return tmp_path/'view',{'source':'explicit'}
    monkeypatch.setattr(phase3_maps,'resolve',resolve)
    monkeypatch.setattr(entry.subprocess,'run',lambda cmd,**k:seen.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr('sys.argv',['compare','cpu','single','--maps',str(tmp_path/'custom'),
        '--geometry',str(tmp_path/'parent.json'),'--exposure-manifest',str(tmp_path/'known.json')])
    assert entry.main()==0
    cmd=seen[0];assert cmd[cmd.index('--maps')+1]==str(tmp_path/'view')
    assert cmd[cmd.index('--exposure-manifest')+1]==str(tmp_path/'known.json')
    assert cmd[cmd.index('--geometry')+1]==str(tmp_path/'parent.json')


def test_missing_current_ready_is_explained_before_large_load_or_gpu(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_single_v3 import prepare
    monkeypatch.setattr(prepare,'load_ready',lambda *a,**kw:pytest.fail('large load'))
    monkeypatch.setattr(entry,'selected_gpus',lambda *a:pytest.fail('GPU'))
    with pytest.raises(ValueError,match='phase3_compare.sh cpu single') as error:
        entry.preflight(tmp_path,['base','single'])
    assert 'phase3_base_bounded_v3' in str(error.value)
    assert 'phase3_single_v3' in str(error.value)


def test_base_probe_is_explicit_and_passes_the_model_gpu_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'APP',tmp_path);seen=[]
    monkeypatch.setattr(entry,'preflight',lambda *a:seen.append('check') or '0,1,2,3')
    def run(cmd,**kw):
        assert '--probe-only' in cmd and kw['env']['GPU_IDS']=='0,1,2,3'
        seen.append('probe');return SimpleNamespace(returncode=0)
    monkeypatch.setattr(entry.subprocess,'run',run)
    monkeypatch.setattr('sys.argv',['compare','probe','base'])
    assert entry.main()==0 and seen==['check','probe']
    assert not (tmp_path/'checkpoints/phase3_compare/base_completed.json').exists()


def test_existing_probe_rejects_before_expensive_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(entry,'APP',tmp_path)
    (entry.root_for(tmp_path,'base')/'run_base_bounded_v3_probe').mkdir(parents=True)
    monkeypatch.setattr(entry,'preflight',lambda *a:pytest.fail('expensive preflight'))
    monkeypatch.setattr('sys.argv',['compare','probe','base'])
    with pytest.raises(FileExistsError,match='Existing probe preserved'):entry.main()


def test_rejected_capacity_explains_cause_before_gpu(tmp_path,monkeypatch):
    root=tmp_path/'checkpoints/phase3_base_bounded_v3';root.mkdir(parents=True)
    (root/'ready_1280.json').write_text(json.dumps(dict(status='rejected',
        error='insufficient_capacity: DYNAMIC_CUTIN',admission='capacity.json')))
    monkeypatch.setattr(entry,'selected_gpus',lambda *a:pytest.fail('GPU'))
    with pytest.raises(ValueError,match='insufficient_capacity: DYNAMIC_CUTIN') as exc:
        entry.preflight(tmp_path,['base'])
    assert 'capacity.json' in str(exc.value) and 'unchanged inputs' in str(exc.value)
