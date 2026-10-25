import copy
import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint.abc_delivery import sha,write
from qwen3vl_local.action_expert_ablation.geometry_abc import recovery,contracts,worker


def fixture(tmp_path):
    old=tmp_path/'old';current=contracts.source_identity()
    policy=json.loads(Path(recovery.__file__).with_name('processor_compatibility.json').read_text())
    saved=copy.deepcopy(current)
    for key,value in policy['changes'].items():
        before=value['before'][0]
        if before is None:saved.pop(key,None)
        else:saved[key]=before
    gate=dict(code=saved,data_ready=True,plan_ready=True,raw_sources_verified=True,history_ready=True,development_ready=True,
        dataset='dataset',splits={'train':'1','val':'2','test':'3'},manifest_sha256='manifest',plan_sha256='plan',resources={'qwen':{'base':'sha'},'bev':'sha'})
    processor=dict(schema='abc_full_pool_processor_v1',code=saved,dataset='dataset',splits=gate['splits'],full_pool=True,no_truncation=True,
        checked=9,per_variant={v:3 for v in 'abc'},extrema={v:{s:{m:dict(id=s,variant=v,split=s,tokens=20,visual_tokens=4)
        for m in ('tokens','visual_tokens')} for s in ('train','val','test')} for v in 'abc'})
    write(old/'processor.json',processor);gate['processor_sha256']=sha(old/'processor.json');write(old/'admission.json',gate)
    new=dict(gate,code=current,record_counts={'train':1,'val':1,'test':1})
    return old,new,tmp_path/'new/processor.json'


def test_reviewed_source_change_reuses_full_processor_preserves_old(tmp_path):
    old,new,out=fixture(tmp_path)
    original={p.name:p.read_bytes() for p in old.iterdir()}
    assert recovery.reuse_processor(old,new,out)
    assert {p.name:p.read_bytes() for p in old.iterdir()}==original
    r=json.loads(out.read_text());assert r['code']==contracts.source_identity() and r['reused_processor']['sha256']==sha(old/'processor.json')


@pytest.mark.parametrize('bad',['prompt','model','processor_model','data','hash','counts','extrema','unknown_worker'])
def test_incompatible_or_incomplete_processor_recomputed(tmp_path,bad):
    old,new,out=fixture(tmp_path);p=old/'processor.json';r=json.loads(p.read_text());g=json.loads((old/'admission.json').read_text())
    if bad=='prompt':r['code']['qwen3vl_local/audit_joint/geometry_abc_v2/objective.py']='different'
    elif bad=='unknown_worker':r['code']['qwen3vl_local/action_expert_ablation/geometry_abc/worker.py']='unknown'
    elif bad=='model':new['resources']={'qwen':{'base':'different'}}
    elif bad=='processor_model':r['model']={'base':'different'}
    elif bad=='data':new['dataset']='different'
    elif bad=='counts':r['checked']=8
    elif bad=='extrema':r['extrema']['a']['test']['tokens']['split']='train'
    write(p,r)
    if bad!='hash':g['processor_sha256']=sha(p);write(old/'admission.json',g)
    else:p.write_text(p.read_text()+' ')
    assert not recovery.reuse_processor(old,new,out)
    assert not out.exists() and json.loads(out.with_name('recovery.json').read_text())['status']=='recompute_required'


def test_partial_training_never_restarted_automatically(tmp_path):
    old,new,out=fixture(tmp_path);(old/'a_bev_only').mkdir()
    with pytest.raises(ValueError,match='training_artifacts'):recovery.reuse_processor(old,new,out)


def test_worker_source_rejection_before_runner_or_distributed(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    monkeypatch.setattr(common,'leadmot_train',lambda:pytest.fail('runner must not load'))
    write(tmp_path/'admission.json',{'code':{'stale.py':'old'}})
    with pytest.raises(ValueError,match='admission_source_changed') as exc:
        worker.main(['--abc-variant','a','--branch','bev_only','--suite-root',str(tmp_path)])
    assert 'stale.py' in str(exc.value)


def test_outer_rejects_before_gpu_and_packs_differences(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import __main__ as cli
    from qwen3vl_local.action_prior import launch
    monkeypatch.setattr(launch,'ensure_gpu',lambda *a:pytest.fail('GPU must not be selected'))
    root=tmp_path/'run';write(root/'admission.json',dict(processor_ready=True,code={'stale.py':'old'}))
    with pytest.raises(ValueError,match='admission_source_changed'):
        cli.main(['probe','--run-dir',str(root),'--data-dir',str(tmp_path/'data'),'--data-root',str(tmp_path/'raw')])
    import zipfile
    with zipfile.ZipFile(root/'failure_delivery/diagnostic.zip') as z:
        assert 'source_mismatch.json' in z.namelist() and 'admission.json' in z.namelist()


def test_policy_binds_current_targets_and_rejects_arbitrary_new_source(tmp_path):
    current=contracts.source_identity();policy=json.loads(Path(recovery.__file__).with_name('processor_compatibility.json').read_text())
    for key,rule in policy['changes'].items():assert current[key]==rule['after']
    old,new,out=fixture(tmp_path);saved=json.loads((old/'processor.json').read_text())['code']
    changed=dict(current);changed['new_unreviewed.py']='hash'
    assert not recovery.compatible(saved,changed)


def test_recovery_all_reuses_processor_then_runs_all_six_groups(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import __main__ as cli,audit,preflight
    from qwen3vl_local.action_prior import launch
    old,gate,unused=fixture(tmp_path);new=tmp_path/'new';new.mkdir()
    dataset=tmp_path/'data';write(dataset/'status.json',dict(data_ready=True))
    gate=dict(gate,resources=dict(gate['resources'],ready=True,errors=[]))
    # Match the previously validated resource record exactly.
    saved=json.loads((old/'admission.json').read_text());saved['resources']=gate['resources'];write(old/'admission.json',saved)
    original={p.name:p.read_bytes() for p in old.iterdir()}
    monkeypatch.setattr(launch,'prepare_run_directory',lambda path:new)
    monkeypatch.setattr(preflight,'check',lambda *a,**kw:copy.deepcopy(gate))
    monkeypatch.setattr(preflight,'processor_check',lambda *a,**kw:pytest.fail('compatible expensive scan must be reused'))
    monkeypatch.setattr(audit,'drill',lambda *a:None)
    monkeypatch.setattr(audit,'build',lambda *a:dict(package='fixture.zip'))
    monkeypatch.setattr(launch,'ensure_gpu',lambda n:4)
    calls=[]
    def run(command,log):
        operation=command[command.index('--operation')+1];out=Path(command[command.index('--output-dir')+1]);calls.append(operation)
        if operation=='probe':write(out/'probe_complete.json',{'admission_sha256':sha(new/'admission.json')})
        elif operation=='train':write(out/'training_complete.json',{})
        else:write(out/'metrics.json',{});write(out/'condition_sensitivity.json',{})
    monkeypatch.setattr(launch,'run_logged',run)
    cli.main(['all','--recover-from',str(old),'--data-dir',str(dataset),'--output',str(tmp_path/'new_output')])
    assert calls==['probe']*6+['train','evaluate']*6
    assert {p.name:p.read_bytes() for p in old.iterdir()}==original
    assert json.loads((new/'pipeline.json').read_text())['status']=='completed'


def test_preflight_source_drift_is_rejected_before_a_receipt_is_published(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import preflight,data
    directory=tmp_path/'data';write(directory/'manifest.json',dict(identity='id',splits={}))
    write(directory/'status.json',dict(data_ready=True,plan_ready=True))
    manifest=json.loads((directory/'manifest.json').read_text())
    monkeypatch.setattr(preflight,'resources',lambda *a:dict(errors=[]))
    monkeypatch.setattr(data,'load',lambda *a,**kw:({'train':[],'val':[],'test':[]},manifest))
    monkeypatch.setattr(preflight,'verify_plan',lambda *a:dict(sha256='plan'))
    values=iter([{'file':'before'},{'file':'after'}]);monkeypatch.setattr(contracts,'source_identity',lambda:next(values))
    target=tmp_path/'admission.json'
    with pytest.raises(ValueError,match='source_changed_during_admission'):
        preflight.check(directory,'model','bev','raw',target,raw=False)
    assert not target.exists()


def test_old_incomplete_admission_falls_back_to_fresh_processor(tmp_path):
    old,new,out=fixture(tmp_path);(old/'admission.json').unlink()
    assert recovery.inspect_old(old)[1]=={}
    assert not recovery.reuse_processor(old,new,out)
    assert not out.exists()


def test_incompatible_data_code_selects_separate_version_without_editing_old(tmp_path):
    directory=tmp_path/'dataset';write(directory/'manifest.json',dict(code={'old':'code'}))
    before=(directory/'manifest.json').read_bytes()
    new=recovery.data_directory(directory)
    assert new!=directory and not new.exists()
    assert (directory/'manifest.json').read_bytes()==before
    assert recovery.data_directory(directory)==new
    write(directory/'manifest.json',dict(code=contracts.source_identity(data_only=True)))
    assert recovery.data_directory(directory)==directory


def test_processor_model_changed_between_admission_and_scan_is_rejected(tmp_path):
    from qwen3vl_local.action_expert_ablation.geometry_abc.probe_inputs import verify_processor
    p=tmp_path/'processor.json';r=dict(code={},dataset='id',splits={},full_pool=True,no_truncation=True,model={'weights':'new'})
    write(p,r);gate=dict(code={},dataset='id',splits={},processor_sha256=sha(p),resources={'qwen':{'weights':'old'}})
    with pytest.raises(ValueError,match='processor_base_model_changed'):verify_processor(p,gate)
