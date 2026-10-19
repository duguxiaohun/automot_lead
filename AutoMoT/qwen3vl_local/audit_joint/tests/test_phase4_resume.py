import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase4_resume as r


@pytest.fixture
def paths(tmp_path):
    old,out,raw,model=[tmp_path/n for n in ('existing','new','raw','model')]
    for p in (old/'inputs/production',raw,model):p.mkdir(parents=True)
    (old/'inputs/candidates.json').write_text('{}')
    (model/'config.json').write_text('{}')
    return old,out,raw,model


def put(path):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')


@pytest.mark.parametrize('stage',[0,1,2])
def test_stage_routes_only_explicit_local_assets_without_mutation(paths,stage):
    old,out,raw,model=paths
    if stage>=1:
        put(old/'data4/manifest.json');put(old/'references4/references.json')
    if stage==2:put(old/'view/student_view.json')
    before={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    plan=r.plan(*paths)
    assert plan['mode']==('verify_complete_replay_then_finish_build',
                        'complete_view_then_admission','admission_only')[stage]
    assert not plan['training'] and not plan['recompute_routes'] and not out.exists()
    assert before=={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    if stage==0:assert plan['environment']['RESUME_FROM']==str(old)
    else:
        assert ('--complete-view' in plan['command'])==(stage==1)
        assert 'server_phase4_existing.sh' not in ' '.join(plan['command'])


@pytest.mark.parametrize('defect',['bad_json','bad_view','missing_old','missing_production','overlap','alias_overlap','existing_output','dangling_metadata'])
def test_invalid_asset_never_falls_back_to_replay(paths,defect):
    old,out,raw,model=paths
    if defect=='bad_json':put(old/'data4/manifest.json');(old/'data4/manifest.json').write_text('{')
    elif defect=='bad_view':put(old/'view/student_view.json');(old/'view/student_view.json').write_text('[]')
    elif defect=='missing_old':old=old/'absent'
    elif defect=='missing_production':(old/'inputs/production').rmdir()
    elif defect=='overlap':out=old/'new'
    elif defect=='alias_overlap':
        alias=out.parent/'alias';alias.symlink_to(old,target_is_directory=True);out=alias/'new'
    elif defect=='existing_output':out.mkdir()
    else:
        (old/'data4').mkdir();(old/'data4/manifest.json').symlink_to(old/'absent')
    with pytest.raises((ValueError,OSError)):r.plan(old,out,raw,model)


def test_main_rejects_lock_and_plan_only_does_not_launch(paths,monkeypatch):
    import os,fcntl,sys
    from qwen3vl_local.audit_joint import server_readiness
    monkeypatch.setattr(sys,'argv',['resume',*sum(([flag,str(p)] for flag,p in
        zip(('--prepared','--output','--data-root','--model-dir'),paths)),[]),'--plan-only'])
    monkeypatch.setattr(server_readiness,'source_check',lambda *args:dict(status='matched'))
    monkeypatch.setattr(r.subprocess,'call',lambda *args,**kw:pytest.fail('plan must not run'))
    assert r.main()==0
    fd=os.open(paths[0],os.O_RDONLY|os.O_DIRECTORY)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert r.main()==2
    finally:os.close(fd)


def test_main_preserves_child_failure_and_uses_app_cwd(paths,monkeypatch):
    import sys
    from qwen3vl_local.audit_joint import server_readiness
    monkeypatch.setattr(sys,'argv',['resume',*sum(([flag,str(p)] for flag,p in
        zip(('--prepared','--output','--data-root','--model-dir'),paths)),[])])
    monkeypatch.setattr(server_readiness,'source_check',lambda *args:dict(status='matched'))
    def call(command,**kw):
        assert kw['cwd']==r.APP and kw['env']['RESUME_FROM']==str(paths[0])
        assert '--reuse-complete-replay' not in command # Fixed existing shell owns strict verification.
        return 75
    monkeypatch.setattr(r.subprocess,'call',call)
    assert r.main()==75
