import json
from pathlib import Path
import sys
import pytest
from PIL import Image
from qwen3vl_local.audit_joint import phase4_gpu_smoke as s
from qwen3vl_local.sft_new_loop_phase4 import identity
from qwen3vl_local.sft_new_loop_phase4.rgb_short import data


def fixture(tmp_path,monkeypatch):
    native=tmp_path/'native';native.mkdir();root=tmp_path/'rgb';root.mkdir()
    (native/'manifest.json').write_text(json.dumps({'contract':{},'rgb_mode':4}))
    paths=[]
    for frame in range(4,11):
        name=f'{frame:04d}.jpg';Image.new('RGB',(4,4),(frame,0,0)).save(root/name)
    paths=[f'{n:04d}.jpg' for n in (4,6,8,10)]
    row=dict(id='one',split='train',target='YES',images=paths,
             image_sha256=[identity.file_sha(root/p) for p in paths],observation={'frame_id':10})
    (native/'train.jsonl').write_text(json.dumps(row)+'\nTHIS SECOND LINE MUST NOT BE READ')
    monkeypatch.setattr(identity,'check_contract',lambda *a:None)
    monkeypatch.setattr(data,'short_row',lambda r:dict(r,images=r['images'][-2:]))
    monkeypatch.setattr(data,'four_row',lambda r,ids:dict(r,images=list(ids)))
    return native,root,row


@pytest.mark.parametrize('mode,frames',[(2,[8,10]),(4,[7,8,9,10])])
def test_reads_only_one_row_and_required_rgb(tmp_path,monkeypatch,mode,frames):
    native,root,row=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(data,'load_dataset',lambda *a,**kw:pytest.fail('must not load full pool'))
    actual=s.example(native,root,mode)
    assert actual['images']==[f'{f:04d}.jpg' for f in frames]
    assert actual['id']=='one'


def test_changed_original_image_rejected(tmp_path,monkeypatch):
    native,root,row=fixture(tmp_path,monkeypatch)
    Image.new('RGB',(4,4),(99,0,0)).save(root/'0010.jpg')
    with pytest.raises(ValueError,match='SHA'):s.example(native,root,2)


@pytest.mark.parametrize('mode',[2,4])
@pytest.mark.parametrize('layout',['route','files'])
def test_external_rgb_links_match_formal_loader(tmp_path,monkeypatch,mode,layout):
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.model import load_images
    native,disk,row=fixture(tmp_path,monkeypatch)
    root=tmp_path/'lead_data';root.mkdir()
    if layout=='route':
        (root/'route').symlink_to(disk,target_is_directory=True)
        row['images']=['route/'+p for p in row['images']]
        (native/'train.jsonl').write_text(json.dumps(row))
    else:
        for path in disk.iterdir():(root/path.name).symlink_to(path)
    actual=s.example(native,root,mode)
    # Fixture transformers omit full metadata; compare real encoder RGB loading.
    actual['image_sha256']=[identity.file_sha(root/p) for p in actual['images']]
    actual['observation']['history_frames']=[int(Path(p).stem) for p in actual['images']]
    assert len(load_images(actual,root))==mode
    Image.new('RGB',(4,4),(99,0,0)).save(disk/'0010.jpg')
    with pytest.raises(ValueError,match='SHA'):s.example(native,root,mode)


@pytest.mark.parametrize('bad',['../outside/0010.jpg','/tmp/0010.jpg','route/../0010.jpg'])
@pytest.mark.parametrize('mode',[2,4])
def test_diagnostic_rejects_lexical_escape(tmp_path,monkeypatch,bad,mode):
    native,root,row=fixture(tmp_path,monkeypatch)
    row['images'][-1]=bad
    (native/'train.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='invalid relative dataset path'):
        s.example(native,root,mode)


def test_val_row_rejected(tmp_path,monkeypatch):
    native,root,row=fixture(tmp_path,monkeypatch);row['split']='val'
    (native/'train.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='non-training'):s.example(native,root,2)


def test_transport_needs_no_model_or_dataset_and_timeout_is_bounded(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(s,'run_owned',lambda *args:calls.append(args) or 124)
    monkeypatch.setattr(sys,'argv',['smoke','--check','transport','--gpus','0,1,2,3','--timeout','60','--output',str(tmp_path/'out')])
    with pytest.raises(SystemExit) as ex:s.main()
    assert ex.value.code==124 and len(calls)==1
    cmd,log,env,seconds=calls[0]
    assert '--nproc_per_node=4' in cmd and seconds==60
    result=json.loads((tmp_path/'out/summary.json').read_text())
    assert result['status']=='failed' and not result['full_pool_admitted']


def test_missing_rank_never_reports_success(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'run_owned',lambda *args:0)
    monkeypatch.setattr(sys,'argv',['smoke','--check','transport','--gpus','0,1','--output',str(tmp_path/'out')])
    with pytest.raises(SystemExit) as ex:s.main()
    assert ex.value.code==2
