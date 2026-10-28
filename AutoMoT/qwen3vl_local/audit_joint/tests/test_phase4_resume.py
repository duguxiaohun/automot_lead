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


def test_current_release_matches_real_sources_and_preserves_historical_contract():
    import hashlib
    from qwen3vl_local.audit_joint import phase4_resume,phase4_admission_only,phase4_first_epoch,label_release_check
    from qwen3vl_local.audit_joint.server_readiness import source_check
    app=phase4_resume.APP;release=phase4_resume.RELEASE
    old=release.with_name('release_sources_v89_split_budget_20261019.json')
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='6cef74ad9ba24c33105ea91d7049fd0ae2423e68201787b0bac5c556e60bba51'
    assert source_check(app,release)['status']=='matched'
    old_spec=json.loads(old.read_text());new_spec=json.loads(release.read_text())
    assert new_spec['parent_sha256']==hashlib.sha256(old.read_bytes()).hexdigest()
    assert set(old_spec['files']).issubset(new_spec['files'])
    for name,sha in old_spec['files'].items():
        if name.startswith('qwen3vl_local/sft_new_loop_phase4/'):
            assert new_spec['files'][name]==sha
    for module in (phase4_admission_only,phase4_first_epoch,label_release_check):
        assert module.RELEASE==release
    for name in ('server_phase4_existing.sh','server_finish.sh'):
        text=(release.parent/name).read_text()
        assert release.name in text and old.name not in text


def test_real_child_shell_source_gates_use_current_release(tmp_path):
    import os,subprocess,sys
    from qwen3vl_local.audit_joint import phase4_resume
    for name,var in [('server_phase4_existing.sh','P4_EXISTING_OUT'),('server_finish.sh','FINISH_OUT')]:
        text=(phase4_resume.RELEASE.parent/name).read_text()
        script=text.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
        out=tmp_path/name;out.mkdir()
        subprocess.run([sys.executable,'-c',script],cwd=phase4_resume.APP,
                       env=dict(os.environ,**{var:str(out)}),check=True,capture_output=True,text=True)
        assert json.loads((out/'source_check.json').read_text())['status']=='matched'
