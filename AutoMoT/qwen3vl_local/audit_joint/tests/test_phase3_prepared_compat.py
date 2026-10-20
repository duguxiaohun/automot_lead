"""Only the exact released producer with unchanged native generation may be reused."""
import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint import phase3_suite_data as data,phase3_base_trial as trial


def compat():return trial.read(Path(data.__file__).with_name('phase3_suite_data_compat_1103ece79.json'))


def test_known_published_producer_accepted_with_unchanged_generation():
    data.verify_producer_release(compat()['producer_release_sha256'],'current')
    data.verify_producer_release('same','same')
    with pytest.raises(ValueError,match='not compatible'):data.verify_producer_release('unknown','current')


def test_native_rule_or_prompt_change_rejected(monkeypatch):
    original=trial.file_sha
    name='qwen3vl_local/sft_new_loop_phase3/trajectory_action.py'
    monkeypatch.setattr(trial,'file_sha',lambda p:'changed' if str(p).endswith(name) else original(p))
    with pytest.raises(ValueError,match='dependency changed'):
        data.verify_producer_release(compat()['producer_release_sha256'],'current')


def test_builder_change_rejected_without_modifying_old_receipt(monkeypatch):
    original=Path.read_text
    def changed(path,*a,**kw):
        text=original(path,*a,**kw)
        return text.replace('wanted=set(); routes={}','wanted=set(); routes={}; changed=True') if path==Path(data.__file__) else text
    monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(ValueError,match='generation logic changed: requests'):
        data.verify_producer_release(compat()['producer_release_sha256'],'current')


def test_old_prepared_receipt_and_files_read_only(tmp_path,monkeypatch):
    root=tmp_path/'shared';root.mkdir();data_root=tmp_path/'raw';data_root.mkdir()
    files={}
    for name in ('base/frame_index.jsonl','base/manifest.json','auxiliary/frame_index.jsonl','auxiliary/manifest.json'):
        p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(name);files[name]=trial.file_sha(p)
    monkeypatch.setattr(trial,'reference',lambda:dict(index_sha256=files['base/frame_index.jsonl']))
    receipt=dict(policy=data.POLICY,release_sha256=compat()['producer_release_sha256'],
        offset_frames=[0,12,24],data_root=str(data_root),files=files)
    p=root/'shared_data_receipt.json';p.write_text(json.dumps(receipt));before=p.read_bytes()
    assert data.verify(root,data_root=data_root,offsets=[0,12,24],release='current')==receipt
    assert p.read_bytes()==before
    (root/'auxiliary/frame_index.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='data changed'):data.verify(root,data_root=data_root,offsets=[0,12,24],release='current')
