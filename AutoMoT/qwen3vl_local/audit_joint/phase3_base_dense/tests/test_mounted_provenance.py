"""External route mounts remain bound to logical identities and actual sources."""
from pathlib import Path
import json
import shutil
import pytest
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
from qwen3vl_local.audit_joint.phase3_base_dense import provenance,runtime
from qwen3vl_local.audit_joint import phase3_base_trial as trial


def mounted(tmp_path,monkeypatch,kind):
    job,original,extra=raw_fixture(tmp_path,monkeypatch)
    root=Path(job['data_root']);external=tmp_path/'external';external.mkdir()
    sample=extra[0];route=root/sample['scenario']/sample['route_id']
    if kind=='scenario':source=route.parent
    elif kind=='route':source=route
    elif kind=='metas':source=route/'metas'
    elif kind=='rgb':source=route/'rgb'
    else:source=route/'metas/0012.pkl'
    target=external/source.name
    shutil.move(str(source),str(target));source.symlink_to(target,target_is_directory=target.is_dir())
    return job,original,extra,source,target


@pytest.mark.parametrize('kind',['scenario','route','metas','rgb','meta_file'])
def test_external_mounts_full_recompute_and_cached_verification(tmp_path,monkeypatch,kind):
    job,_,_,source,target=mounted(tmp_path,monkeypatch,kind)
    pool=runtime.load_pool(job,recompute_aux=True)
    assert pool.aux_added==60
    summary=provenance.verify_cached(job)
    assert summary['rows_by_split']==dict(train=20,val=20,test=20)
    assert summary['external_route_mounts']>0 if kind in ('scenario','route') else summary['external_route_mounts']==0
    proof=trial.read(Path(job['output'])/'dense_aux_provenance.json')
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in proof['sources'])
    assert any(Path(r['resolved_path']).is_relative_to(tmp_path/'external') for r in proof['sources'].values())
    assert source.is_symlink() and source.resolve()==target


@pytest.mark.parametrize('kind',['scenario','route','metas','rgb','meta_file'])
def test_retarget_even_to_identical_bytes_rejected(tmp_path,monkeypatch,kind):
    job,_,_,source,target=mounted(tmp_path,monkeypatch,kind)
    runtime.load_pool(job,recompute_aux=True)
    replacement=target.with_name(target.name+'_copy')
    if target.is_dir():shutil.copytree(target,replacement)
    else:shutil.copy2(target,replacement)
    source.unlink();source.symlink_to(replacement,target_is_directory=replacement.is_dir())
    with pytest.raises(ValueError,match='target changed|source changed'):provenance.verify_cached(job)


def test_external_meta_content_and_membership_change_rejected(tmp_path,monkeypatch):
    job,_,_,source,target=mounted(tmp_path,monkeypatch,'route')
    runtime.load_pool(job,recompute_aux=True)
    (target/'metas/new.pkl').write_bytes(b'new')
    with pytest.raises(ValueError,match='membership changed'):provenance.verify_cached(job)
    (target/'metas/new.pkl').unlink()
    p=target/'metas/0012.pkl';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source changed'):provenance.verify_cached(job)


def test_mount_change_during_recomputation_rejected(tmp_path,monkeypatch):
    job,_,_,source,target=mounted(tmp_path,monkeypatch,'route')
    replacement=target.with_name('replacement');shutil.copytree(target,replacement)
    original=provenance.rule.load_route_trajectory
    def load(run):
        trajectory=original(run)
        if Path(run)==source:
            source.unlink();source.symlink_to(replacement,target_is_directory=True)
        return trajectory
    monkeypatch.setattr(provenance.rule,'load_route_trajectory',load)
    with pytest.raises(ValueError,match='target changed|source changed'):provenance.recompute(job)


def test_logical_traversal_and_aliased_routes_rejected(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='logical auxiliary path'):provenance.logical_path(tmp_path,'../elsewhere')
    with pytest.raises(ValueError,match='logical auxiliary path'):provenance.logical_path(tmp_path,'/absolute')
    job,_,extra,source,target=mounted(tmp_path,monkeypatch,'route')
    row=extra[0];alias=Path(job['data_root'])/row['scenario']/'different_route'
    alias.symlink_to(target,target_is_directory=True)
    p=Path(job['aux_index']);new=dict(row,route_id=alias.name)
    with p.open('a') as f:f.write(json.dumps(new)+'\n')
    job['aux_binding']['index_sha256']=trial.file_sha(p)
    with pytest.raises(ValueError,match='alias one physical route'):provenance.recompute(job)
