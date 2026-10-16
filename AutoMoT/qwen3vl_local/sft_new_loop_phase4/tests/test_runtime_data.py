import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from qwen3vl_local.sft_new_loop_phase4.rgb_short import runtime_data as runtime,data
from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import native_view


def fixture(tmp_path,monkeypatch):
    view=tmp_path/'view';view.mkdir();parent=tmp_path/'native';parent.mkdir()
    (view/'student_view.json').write_text(json.dumps(dict(parent=dict(logical_path=str(parent)))))
    state=dict(source='valid',input='original')
    monkeypatch.setattr(runtime,'inputs',lambda v,m:dict(state,mode=m))
    monkeypatch.setenv('P4_VALIDATED_DATA',str(tmp_path/'prepared'))
    monkeypatch.delenv('WORLD_SIZE',raising=False)
    calls=[]
    def loader(*a,**kw):
        calls.append(kw)
        return {'train':[{'id':'one'}],'val':[],'test':[]},{'training_admission':dict(trainable=False,errors=['empty eval'])}
    return view,state,calls,loader


def test_prepare_once_then_workers_reuse_and_flags_still_apply(tmp_path,monkeypatch):
    view,state,calls,loader=fixture(tmp_path,monkeypatch)
    expected=runtime.load(view,2,loader)
    for rank in range(4):
        monkeypatch.setenv('WORLD_SIZE','4');monkeypatch.setenv('RANK',str(rank))
        assert runtime.load(view,2,loader)==expected
    assert len(calls)==1
    with pytest.raises(ValueError,match='prerequisites'):
        runtime.load(view,2,loader,require_trainable=True)
    with pytest.raises(ValueError,match='complete-coverage'):
        runtime.load(view,2,loader,require_complete_coverage=True)
    with pytest.raises(ValueError,match='missing'):
        runtime.load(view,4,loader)
    assert len(calls)==1


@pytest.mark.parametrize('damage',['source','input','payload','receipt'])
def test_snapshot_reuse_rejects_changes(tmp_path,monkeypatch,damage):
    view,state,calls,loader=fixture(tmp_path,monkeypatch);runtime.load(view,2,loader)
    if damage in state:state[damage]='changed'
    else:
        suffix='json' if damage=='payload' else 'receipt.json'
        (tmp_path/'prepared'/('rgb2.'+suffix)).write_text('{}')
    with pytest.raises((ValueError,KeyError)):runtime.load(view,2,loader)
    assert len(calls)==1


def test_failed_prepare_never_publishes_receipt(tmp_path,monkeypatch):
    view,state,calls,loader=fixture(tmp_path,monkeypatch)
    def changed(*a,**kw):
        result=loader(*a,**kw);state['input']='changed';return result
    with pytest.raises(ValueError,match='changed'):runtime.load(view,2,changed)
    assert not (tmp_path/'prepared/rgb2.receipt.json').exists()


def test_real_view_four_worker_processes_do_not_validate_native(native_view,tmp_path,monkeypatch):
    parent,view=native_view;data.create(parent,view)
    monkeypatch.setenv('P4_VALIDATED_DATA',str(tmp_path/'snapshot'))
    expected=data.load_dataset(view,rgb_mode=2)
    script='''
import sys
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase4 import dataset as native
def forbidden(*a,**kw):raise AssertionError('worker repeated native validation')
native.load_dataset=forbidden
from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
rows,manifest=load_dataset(Path(sys.argv[1]),rgb_mode=2)
assert sum(map(len,rows.values()))==int(sys.argv[2])
'''
    processes=[]
    for rank in range(4):
        env=dict(os.environ,WORLD_SIZE='4',RANK=str(rank))
        processes.append(subprocess.Popen([sys.executable,'-c',script,str(view),str(sum(map(len,expected[0].values())))],
                                         env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
    for proc in processes:
        stdout,stderr=proc.communicate(timeout=30)
        assert proc.returncode==0,stderr
        assert 'native_validation=not_repeated' in stdout


def test_native_adapter_restores_after_failure(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import dataset as native
    original=native.load_dataset
    with pytest.raises(ValueError):
        with runtime.native_read():raise ValueError('failed')
    assert native.load_dataset is original


def test_direct_ddp_refuses_repeated_native_load(tmp_path,monkeypatch):
    monkeypatch.delenv('P4_VALIDATED_DATA',raising=False);monkeypatch.setenv('WORLD_SIZE','4')
    with pytest.raises(ValueError,match='CPU-prepared'):
        runtime.load(tmp_path,2,lambda *a,**k:pytest.fail('must not load independently on ranks'))


def test_disk_reserve_failure_never_publishes_snapshot(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    view,state,calls,loader=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    with pytest.raises(ValueError,match='disk reserve'):runtime.load(view,2,loader)
    assert not (tmp_path/'prepared/rgb2.receipt.json').exists()
