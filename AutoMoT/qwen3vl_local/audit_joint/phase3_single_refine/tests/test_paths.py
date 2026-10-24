"""Real external scene symlinks; original source/admission mocked only in CPU fixture."""
import json
from pathlib import Path
import pytest
from PIL import Image
from ..paths import image_path
from ..audit import sha
from .. import preflight
from ..identity import digest


@pytest.mark.parametrize('name',['../rgb.jpg','scene/../../rgb.jpg','/tmp/rgb.jpg','','.','scene\\rgb.jpg'])
def test_reject_nonrelative_dataset_names(tmp_path,name):
    with pytest.raises(ValueError,match='invalid_dataset_image_path'):image_path(tmp_path,name)


def test_external_scene_and_root_mounts_and_retarget(tmp_path):
    root=tmp_path/'lead_data';root.mkdir();external=tmp_path/'external';external.mkdir()
    (external/'rgb.jpg').write_bytes(b'first');(root/'Scene').symlink_to(external,target_is_directory=True)
    alias=tmp_path/'root_alias';alias.symlink_to(root,target_is_directory=True)
    p=image_path(alias,'Scene/rgb.jpg');before=sha(p)
    assert p==alias/'Scene/rgb.jpg' and p.resolve()==external/'rgb.jpg'
    other=tmp_path/'other';other.mkdir();(other/'rgb.jpg').write_bytes(b'changed')
    (root/'Scene').unlink();(root/'Scene').symlink_to(other,target_is_directory=True)
    assert sha(p)!=before


def test_broken_mount_or_directory_rejected(tmp_path):
    (tmp_path/'broken').symlink_to(tmp_path/'missing')
    with pytest.raises(FileNotFoundError):image_path(tmp_path,'broken/rgb.jpg')
    (tmp_path/'folder').mkdir()
    with pytest.raises(ValueError,match='not_file'):image_path(tmp_path,'folder')


def test_cpu_prepare_and_reload_mounted_rgb_then_reject_content_change(tmp_path,monkeypatch):
    root=tmp_path/'data';root.mkdir();external=tmp_path/'disk';external.mkdir()
    Image.new('RGB',(32,32),'red').save(external/'rgb.jpg')
    (root/'Scene').symlink_to(external,target_is_directory=True)
    records=[dict(id='anchor',split='test',image='Scene/rgb.jpg')]
    artifact=dict(records=records,data_root=str(root))
    ds=tmp_path/'dataset.json';ds.write_text('{}')
    native=tmp_path/'native.json';native.write_text(json.dumps(dict(status='ready',dataset=str(ds),dataset_sha256=sha(ds))))
    def bound(x):return dict(x,sha256=digest(x))
    admission=bound(dict(status='ready'))
    plan=bound(dict(plans=[dict(optimizer_steps=3200,ordered_ids=['anchor'])]*3))
    monkeypatch.setattr(preflight,'require_received_source',lambda:None)
    monkeypatch.setattr(preflight,'verify_snapshot',lambda:None)
    monkeypatch.setattr(preflight.dataset,'load',lambda *_a,**_kw:artifact)
    monkeypatch.setattr(preflight.admission,'inspect',lambda _:admission)
    monkeypatch.setattr(preflight.admission,'require',lambda x:x)
    monkeypatch.setattr(preflight.planning,'project',lambda *_a,**_kw:plan)
    out=tmp_path/'output'
    assert preflight.prepare(native,out)['status']=='ready'
    assert preflight.load_ready(out,check_images=True)['status']=='ready'
    Image.new('RGB',(32,32),'blue').save(external/'rgb.jpg')
    with pytest.raises(ValueError,match='preflight_RGB_changed'):preflight.load_ready(out,check_images=True)
