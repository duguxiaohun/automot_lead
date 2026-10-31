import json
from pathlib import Path
import pytest
from qwen3vl_local.audit_joint.phase12_retrain import history_holdout as holdout
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def fixture(tmp_path, monkeypatch):
    contract=tmp_path/'contract.json'
    proof.write(contract,dict(schema=holdout.history.SCHEMA,sources=[dict(phase=1,records=[
        dict(key=['s','Town01_Rep0_123_route0_01_11_05_46_46',2])])],holdout_groups=[['s','Town01_123']]))
    release=tmp_path/'release.json';proof.write(release,dict(contract_sha256=proof.sha(contract)))
    monkeypatch.setattr(holdout.history,'RELEASE',release)
    source=tmp_path/'source';source.mkdir()
    rows=[dict(scenario='s',route_id=route,frame_id=i,split=split,answers={'RS1':True})
        for i,(route,split) in enumerate([
            ('Town01_Rep1_123_route0_01_08_13_44_23','train'),
            ('Town01_Rep2_123_route0_01_08_13_44_23','val'),
            ('Town01_Rep0_123_route0_01_11_05_46_46','test'),
            ('unrelated_train','train'),('unrelated_val','val')])]
    (source/'frame_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for name in ('manifest.json','training_revision.json','preservation.json'):proof.write(source/name,{})
    proof.write(source/'manifest.json',dict(focus_bin_availability={'old':99},fresh_training_revision={'old':True}))
    return source/'frame_index.jsonl',contract,tmp_path/'output'


def test_excludes_repeated_routes_preserves_original_test_and_all_retained_bytes(tmp_path,monkeypatch):
    index,contract,out=fixture(tmp_path,monkeypatch)
    before=index.read_bytes();r=holdout.derive(index,contract,out)
    assert index.read_bytes()==before
    assert r['excluded_counts']=={'train':1,'val':1}
    assert (out/'frame_index.jsonl').read_bytes()==b''.join(before.splitlines(keepends=True)[2:])
    assert (out/'historical_exclusions.jsonl').read_bytes()==b''.join(before.splitlines(keepends=True)[:2])
    assert holdout.verify(out/'frame_index.jsonl')==r
    assert proof.read(out/'manifest.json')['counts']=={'frames/test':1,'frames/train':1,'frames/val':1}
    manifest=proof.read(out/'manifest.json')
    assert 'focus_bin_availability' not in manifest
    assert manifest['pre_isolation_statistics']['focus_bin_availability']=={'old':99}
    assert manifest['fresh_training_revision']==proof.read(out/'training_revision.json')
    with pytest.raises(ValueError):holdout.derive(index,contract,out)


@pytest.mark.parametrize('bad',['contract','index','excluded','test','missing'])
def test_isolation_evidence_cannot_be_bypassed(tmp_path,monkeypatch,bad):
    index,contract,out=fixture(tmp_path,monkeypatch);holdout.derive(index,contract,out)
    name={'contract':'historical_contract.json','index':'frame_index.jsonl',
        'excluded':'historical_exclusions.jsonl','test':'historical_isolation.json','missing':'historical_isolation.json'}[bad]
    path=out/name
    if bad=='missing':path.unlink()
    elif bad=='test':
        r=proof.read(path);r['test_sha256']='wrong';proof.write(path,r)
    else:path.write_text(path.read_text()+' ')
    with pytest.raises((ValueError,FileNotFoundError)):holdout.verify(out/'frame_index.jsonl')


def test_wrong_contract_rejected_before_output(tmp_path,monkeypatch):
    index,contract,out=fixture(tmp_path,monkeypatch)
    contract.write_text(contract.read_text()+' ')
    with pytest.raises(ValueError):holdout.derive(index,contract,out)
    assert not out.exists()


def test_launcher_rejects_old_ready_without_historical_isolation(tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')  # Historical ready intentionally lacks a layout.
    from qwen3vl_local.audit_joint.phase12_retrain import launch
    index,contract,out=fixture(tmp_path,monkeypatch)
    version=tmp_path/'version'
    monkeypatch.setattr(launch.runtime,'verify',lambda *args: {})
    monkeypatch.setattr(launch.proof,'data_identity',lambda *args: {})
    proof.write(tmp_path/'ready.json',dict(phase=1,version=str(version),contract=launch.CONTRACT,
        files={},index=str(index),data={}))
    with pytest.raises(FileNotFoundError):launch.verify_ready(tmp_path,1,version)


def test_missing_contract_refuses_launcher_before_run_creation(tmp_path):
    from qwen3vl_local.audit_joint.phase12_retrain import launch
    out=tmp_path/'run'
    with pytest.raises(ValueError,match='historical contract'):
        launch.main(['cpu','1','--run-dir',str(out),'--history-contract',str(tmp_path/'absent')])
    assert not out.exists()
