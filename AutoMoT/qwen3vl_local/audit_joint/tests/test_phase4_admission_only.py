import json
import pytest
from qwen3vl_local.audit_joint import phase4_admission_only as a
from qwen3vl_local.audit_joint import first_epoch_budget, rule_baseline
from qwen3vl_local.sft_new_loop_phase4.rgb_short import data
from qwen3vl_local.sft_new_loop_phase4 import teacher_replay, dataset, teacher_evaluation


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';out=tmp_path/'out';raw=tmp_path/'raw';model=tmp_path/'model'
    for p in (prepared/'data4',prepared/'references4',prepared/'view',raw,model):p.mkdir(parents=True)
    for p in (prepared/'data4/manifest.json',prepared/'references4/references.json',model/'config.json'):p.write_text('{}')
    (prepared/'view/student_view.json').write_text(json.dumps(dict(parent=dict(logical_path=str(prepared/'data4')))))
    monkeypatch.setattr(first_epoch_budget,'profile',lambda _:dict(parameters=10))
    monkeypatch.setattr(first_epoch_budget,'request',lambda *args,**kwargs:dict(test_budget=True))
    def forbidden(*args,**kwargs):pytest.fail('Must not replay or rebuild existing data')
    for obj,name in [(teacher_replay,'produce'),(dataset,'build'),(teacher_evaluation,'export'),(data,'create')]:
        monkeypatch.setattr(obj,name,forbidden)
    return prepared,out,raw,model


def test_reuse_runs_b2_admission_without_building_and_preserves_old_files(fixture,monkeypatch):
    prepared,out,raw,model=fixture
    before={str(p):p.read_bytes() for p in prepared.rglob('*') if p.is_file()}
    a.inputs(*fixture);out.mkdir()
    def check(view,refs,**kw):
        assert view==prepared/'view' and refs==prepared/'references4'
        assert kw['batch_size']==2 and kw['data_root']==raw
        return dict(status='admitted_existing_data_development',errors=[],unsupported_capabilities=['U-E5/proceed/YES'])
    monkeypatch.setattr(rule_baseline,'check',check)
    assert a.finish(*fixture)==0
    assert before=={str(p):p.read_bytes() for p in prepared.rglob('*') if p.is_file()}
    report=json.loads((out/'build_result.json').read_text())
    assert not report['training_started'] and not report['native_replay_started']


@pytest.mark.parametrize('defect',['missing_data','missing_refs','missing_view','output_overlap','existing_output'])
def test_missing_or_overlapping_assets_never_launch(fixture,defect):
    prepared,out,raw,model=fixture
    if defect.startswith('missing_'):
        path={'missing_data':'data4/manifest.json','missing_refs':'references4/references.json','missing_view':'view/student_view.json'}[defect]
        (prepared/path).unlink()
    elif defect=='output_overlap':out=prepared/'new'
    else:out.mkdir()
    with pytest.raises(ValueError):a.inputs(prepared,out,raw,model)


def test_wrong_view_parent_rejected_before_expensive_check(fixture,monkeypatch):
    prepared,out,raw,model=fixture;out.mkdir()
    (prepared/'view/student_view.json').write_text(json.dumps(dict(parent=dict(logical_path=str(raw)))))
    monkeypatch.setattr(rule_baseline,'check',lambda *args,**kw:pytest.fail('must reject parent first'))
    assert a.finish(*fixture)==2
    assert 'does not belong' in json.loads((out/'build_result.json').read_text())['errors'][0]


def test_only_explicit_view_build_uses_new_output_and_refuses_drift(fixture,monkeypatch):
    prepared,out,raw,model=fixture;out.mkdir()
    def create(parent,target,**kw):
        assert parent==prepared/'data4' and target==out/'view'
        target.mkdir();(target/'student_view.json').write_text(json.dumps(dict(parent=dict(logical_path=str(parent)))))
    monkeypatch.setattr(data,'create',create)
    def check(*args,**kw):
        (prepared/'references4/references.json').write_text('{"changed":true}')
        return dict(status='admitted_existing_data_development',errors=[],unsupported_capabilities=[])
    monkeypatch.setattr(rule_baseline,'check',check)
    assert a.finish(*fixture,complete_view=True)==2
    assert json.loads((out/'automatic_admission.json').read_text())['status']=='blocked'


@pytest.mark.parametrize('source_status',['matched','mismatch'])
def test_cli_uses_absolute_reserve_and_packages_failure(fixture,monkeypatch,capfd,source_status):
    import sys
    from qwen3vl_local.audit_joint import guard, server_finish, server_readiness, build_storage
    prepared,out,raw,model=fixture
    monkeypatch.setattr(sys,'argv',['admission','--prepared',str(prepared),'--output',str(out),
                                    '--data-root',str(raw),'--model-dir',str(model)])
    monkeypatch.setattr(server_readiness,'source_check',lambda *args:dict(status=source_status))
    # A large filesystem must not turn nearly all free space into a reserve.
    monkeypatch.setattr(build_storage.shutil,'disk_usage',lambda p:type('Disk',(),dict(free=3*1024**4))())
    calls=[]
    def run(command,where,reserve):
        assert source_status=='matched'
        assert where==out and reserve==20*1024**3
        assert '--worker' in command and '--complete-view' not in command
        calls.append(command)
        import os
        os.write(2,b'retained low-space diagnostic\n')
        return 75
    monkeypatch.setattr(guard,'run',run)
    packages=[]
    monkeypatch.setattr(server_finish,'package',lambda where,code:packages.append((where,code)))
    expected=75 if source_status=='matched' else 2
    assert a.main()==expected and packages==[(out,expected)]
    assert bool(calls)==(source_status=='matched')
    if source_status=='matched':
        assert 'retained low-space diagnostic' in (out/'build.log').read_text()
    else:
        assert not (out/'build.log').exists()
        assert 'Published sources mismatch' in (out/'build_result.json').read_text()
