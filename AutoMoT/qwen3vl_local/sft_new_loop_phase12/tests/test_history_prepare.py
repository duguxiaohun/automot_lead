import io
import json
import sys
from pathlib import Path
import tarfile

import pytest
from qwen3vl_local.sft_new_loop_phase12 import history, history_prepare as prepare
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from test_history import source_fixture


def fixture(app):
    app.mkdir()
    paths=[]
    for phase in (1,2):
        for mode in ('2rgb_endpoints','4rgb'):
            parent=app/'checkpoints'/f'phase{phase}_{mode}_audit_bundle'
            parent.mkdir(parents=True)
            path=source_fixture(parent/'lora_production')
            obj=proof.read(path);obj.update(phase=phase,history_rgb_mode=mode)
            if phase==2:obj['items'][0]['spec']={'seed_key':'fixed'}
            proof.write(path,obj)
            casepath=path.parent/'cases.jsonl';case=json.loads(casepath.read_text())
            if mode=='2rgb_endpoints':case['history_rgb_paths_used']=case['history_rgb_paths_used'][::3]
            casepath.write_text(json.dumps(case)+'\n')
            complete=proof.read(path.parent/'evaluation_complete.json')
            complete.update(phase=phase,inputs_sha256=proof.sha(path),case_files_sha256={'cases.jsonl':proof.sha(casepath)})
            proof.write(path.parent/'evaluation_complete.json',complete)
            paths.append(path)
    original=app/'original.json'
    release=history.build(paths,[],original)
    contract=history.load(original,expected_sha=release['contract_sha256'])
    release['sources']=[{k:s[k] for k in ('source_name','phase','mode','primary','inputs_sha256','completion_sha256','case_shards')}
                        for s in contract['sources']]
    proof.write(history.RELEASE,release)
    return original,paths,app/'checkpoints/phase12_history_20261030/contract.json'


@pytest.mark.parametrize('layout',['native','bundle','shared_checkpoints'])
def test_rebuild_exact_release_without_upload_or_model(tmp_path,layout):
    app=tmp_path/'AutoMoT';original,paths,target=fixture(app)
    if layout=='bundle':
        for path in paths:
            root=path.parent.parent
            with tarfile.open(root.with_suffix('.tar.gz'),'w:gz') as tar:
                tar.add(root,arcname=root.name)
            for name in ('evaluation_inputs.json','evaluation_complete.json','cases.jsonl'):(path.parent/name).unlink()
    elif layout=='native':
        # Original eval directories do not have audit-bundle names.
        for i,path in enumerate(paths):path.parent.parent.rename(path.parent.parent.with_name(f'native_eval_{i}'))
    else:
        (app/'checkpoints').rename(tmp_path/'assets')
        (app/'checkpoints').symlink_to(tmp_path/'assets',target_is_directory=True)
    result=prepare.ensure(target,app)
    assert target.read_bytes()==original.read_bytes()
    assert len(result['sources'])==4
    assert prepare.ensure(target,app)==result


@pytest.mark.parametrize('bad',['missing','wrong_shard','wrong_completion','existing_changed'])
def test_missing_or_changed_history_never_creates_partial_contract(tmp_path,bad):
    app=tmp_path/'AutoMoT';original,paths,target=fixture(app)
    path=paths[0]
    if bad=='missing':path.unlink()
    elif bad=='wrong_shard':(path.parent/'cases.jsonl').write_text('{}\n')
    elif bad=='wrong_completion':(path.parent/'evaluation_complete.json').write_text('{}')
    else:
        target.parent.mkdir();target.write_text('wrong');before=target.read_bytes()
    with pytest.raises((ValueError,KeyError)):
        prepare.ensure(target,app)
    if bad=='existing_changed':assert target.read_bytes()==before
    else:assert not target.exists()


@pytest.mark.parametrize('kind',['valid','wrong_sha','symlink','duplicate'])
def test_local_contract_archive_is_verified_and_never_bulk_extracted(tmp_path,kind):
    app=tmp_path/'AutoMoT';original,paths,target=fixture(app)
    payload=original.read_bytes() if kind!='wrong_sha' else b'wrong'
    with tarfile.open(app/prepare.ARCHIVE,'w:gz') as tar:
        member=tarfile.TarInfo(prepare.MEMBER);member.size=len(payload)
        if kind=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd';member.size=0
        tar.addfile(member,io.BytesIO(payload))
        if kind=='duplicate':tar.addfile(member,io.BytesIO(payload))
        extra=tarfile.TarInfo('../outside');extra.size=3;tar.addfile(extra,io.BytesIO(b'bad'))
    if kind=='valid':
        prepare.ensure(target,app)
        assert target.read_bytes()==original.read_bytes()
    else:
        with pytest.raises(ValueError):prepare.ensure(target,app)
        assert not target.exists()
    assert not (tmp_path/'outside').exists()


def test_discovery_does_not_follow_nested_symlink_or_frozen_trees(tmp_path):
    outside=tmp_path/'outside';outside.mkdir();(outside/'evaluation_inputs.json').write_text('{}')
    root=tmp_path/'checkpoints';root.mkdir();(root/'linked').symlink_to(outside,target_is_directory=True)
    (root/'frozen_runtime').mkdir();(root/'frozen_runtime/evaluation_inputs.json').write_text('{}')
    assert list(prepare.discover(root))==[]


def test_publish_race_never_overwrites_an_existing_contract(tmp_path):
    app=tmp_path/'AutoMoT';original,paths,target=fixture(app)
    prepare.publish(original.read_bytes(),target)
    before=target.stat()
    prepare.publish(original.read_bytes(),target)
    assert target.stat().st_ino==before.st_ino


def test_discovery_is_bounded(tmp_path):
    (tmp_path/'child').mkdir()
    with pytest.raises(ValueError,match='limit reached'):list(prepare.discover(tmp_path,timeout=-1))


def test_incomplete_diagnostic_copy_does_not_hide_verified_source(tmp_path):
    app=tmp_path/'AutoMoT';original,paths,target=fixture(app)
    diagnostic=app/'checkpoints/000_diagnostic';diagnostic.mkdir()
    (diagnostic/'evaluation_inputs.json').write_bytes(paths[0].read_bytes())
    prepare.ensure(target,app)
    assert target.read_bytes()==original.read_bytes()


@pytest.mark.parametrize('phase',['phase1','fusion'])
def test_launch_prepares_default_history_before_reserving_run(tmp_path,monkeypatch,phase):
    if phase=='phase1':
        from qwen3vl_local.audit_joint.phase12_retrain import launch
        argv=['cpu','1']
    else:
        from qwen3vl_local.sft_new_loop_phase12 import launch
        argv=['cpu']
    monkeypatch.setattr(launch,'APP',tmp_path)
    def refuse(target,app):
        assert target==tmp_path/'checkpoints/phase12_history_20261030/contract.json'
        assert app==tmp_path
        raise ValueError('explicit fixture missing historical evidence')
    monkeypatch.setattr(prepare,'ensure',refuse)
    run=tmp_path/'run';monkeypatch.setattr(sys,'argv',['launch',*argv,'--run-dir',str(run)])
    with pytest.raises(ValueError,match='explicit fixture'):launch.main()
    assert not run.exists()


def test_independent_phase2_does_not_require_phase1_history(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase12_retrain import launch,snapshot
    monkeypatch.setattr(launch,'APP',tmp_path)
    def wrong(*a,**kw):raise AssertionError('Phase2 must not prepare Phase1 history')
    def reached(*a,**kw):raise ValueError('Phase2 reached snapshot without historical inputs')
    monkeypatch.setattr(prepare,'ensure',wrong);monkeypatch.setattr(snapshot,'prepare',reached)
    with pytest.raises(ValueError,match='Phase2 reached snapshot'):
        launch.main(['cpu','2','--run-dir',str(tmp_path/'run'),'--budget-profile','phase2_10k'])
