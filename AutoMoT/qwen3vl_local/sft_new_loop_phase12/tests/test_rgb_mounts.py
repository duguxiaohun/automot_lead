from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase12 import data,history
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase2 import train as native_train,eval as native_eval
from qwen3vl_local.sft_new_loop_phase12.tests.test_missing_labels import fixture


@pytest.mark.parametrize('layout',['cr02','cee'])
@pytest.mark.parametrize('level',['root','scenario','route','rgb','file'])
def test_nested_mount_recovery_history_and_native_readers(tmp_path,monkeypatch,layout,level):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    data.build(p1,p2,rgb,tmp_path/'plain')
    project=tmp_path/layout/'AutoMoT';project.mkdir(parents=True)
    # cr02 uses an ordinary lead_data directory containing external links;
    # cee/root-linked layouts must continue working too.
    root=project/'lead_data'
    if level=='root':root.symlink_to(rgb,target_is_directory=True)
    else:
        rgb.rename(root)
        member={'scenario':'s','route':'s/route','rgb':'s/route/rgb','file':base['history_rgb_paths'][0]}[level]
        logical=root/member;disk=tmp_path/'datashare';disk.mkdir()
        target=disk/logical.name;logical.rename(target)
        logical.symlink_to(target,target_is_directory=target.is_dir())
    original=p1.read_bytes(),p2.read_bytes()
    data.build(p1,p2,root,tmp_path/'mounted')
    assert (tmp_path/'plain/frame_index.jsonl').read_bytes()==(tmp_path/'mounted/frame_index.jsonl').read_bytes()
    assert (p1.read_bytes(),p2.read_bytes())==original
    hashes=proof.read(tmp_path/'plain/rgb_sha256.json')
    contract={'sources':[{'image_sha256':hashes}]}
    assert history.verify_images(contract,root)==hashes
    for name,sha in hashes.items():
        assert proof.sha(data.resolve_rgb(root,name))==sha
        for reader in (native_train,native_eval):
            assert proof.sha(reader._resolve_rgb_path(name,root))==sha
        assert data.resolve_rgb(root,'/old/server/AutoMoT/lead_data/'+name)==data.resolve_rgb(root,name)
    # Physical links never waive frozen content identity.
    (root/base['history_rgb_paths'][0]).write_bytes(b'changed image')
    with pytest.raises(ValueError,match='RGB differs from local audit'):
        data.build(p1,p2,root,tmp_path/'changed')
    with pytest.raises(ValueError,match='Historical RGB changed'):
        history.verify_images(contract,root)


@pytest.mark.parametrize('member',['../s/route/rgb/x.jpg','/old/../s/route/rgb/x.jpg',
    '/old/../lead_data/s/route/rgb/x.jpg','s/route/not_rgb/x.jpg','s/route/rgb'])
def test_invalid_logical_member_still_rejected(tmp_path,member):
    with pytest.raises(ValueError,match='Invalid RGB member'):data.resolve_rgb(tmp_path,member)


def test_broken_nested_mount_reports_actual_target(tmp_path):
    root=tmp_path/'lead_data';root.mkdir();target=tmp_path/'missing_disk'
    (root/'s').symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='Missing RGB') as error:
        data.resolve_rgb(root,'s/route/rgb/0000.jpg')
    assert str(root/'s/route/rgb/0000.jpg') in str(error.value)
    assert str(target/'route/rgb/0000.jpg') in str(error.value)
    assert 'symlinks=' in str(error.value)
