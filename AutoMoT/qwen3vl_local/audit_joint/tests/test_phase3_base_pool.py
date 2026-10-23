import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3vl_local.audit_joint import phase3_base_pool as pool, phase3_native_runtime as runtime
from qwen3vl_local.sft_new_loop_phase3 import build_dataset as native
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS


def bucket(split, count):
    return [dict(split=split,scenario='Synthetic',route_id=f'Town01_{split}_{i//10}',frame_id=i,
                 action_labels=dict.fromkeys(ACTION_KEYS,False)) for i in range(count)]


def test_full_train_pool_removes_truncation_preserves_exact_native_holdouts():
    original=native._sample_context_bucket
    def select(rows):
        return native._sample_context_bucket(rows,context_id='DYNAMIC_CUTIN',target=12,
                                             rng=random.Random(24),route_diverse=True)
    expected={s:select(bucket(s,40)) for s in ('train','val','test')}
    assert len(expected['train'][0])==12
    with pool.full_train_selection() as observed:
        train,report=select(bucket('train',40))
        assert len(train)==40 and {x['frame_id'] for x in train}==set(range(40))
        assert report['full_candidates']==40 and not report['epoch_quota_applied']
        assert observed['DYNAMIC_CUTIN']==dict(legacy_target=12,full_candidates=40)
        for split in ('val','test'):assert select(bucket(split,40))==expected[split]
    assert native._sample_context_bucket is original


def test_selection_restores_on_failure_and_rejects_mixed_splits():
    original=native._sample_context_bucket
    with pytest.raises(ValueError,match='mixed_split'):
        with pool.full_train_selection():
            native._sample_context_bucket(bucket('train',1)+bucket('test',1),context_id='DYNAMIC_CUTIN',
                target=1,rng=random.Random(0),route_diverse=True)
    assert native._sample_context_bucket is original


def test_real_cpu_projection_reports_eligible_capacity_without_a_model(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    j,root,_=common_fixture(tmp_path,monkeypatch)
    report=pool.project(root/'native/frame_index.jsonl',Path(j['data_root']),root/'reference.json')
    assert report['status']=='rejected' and 'capacity' in report['error']
    assert len(report['supply'])==10 and not report['model_inputs_checked'] and not report['GPU_executed']
    assert not report['actual_exposure_advanced']


def test_missing_ready_rejects_before_model_or_data(tmp_path):
    with pytest.raises(ValueError,match='cpu base first'):pool.load_ready(tmp_path)


def test_cpu_compatibility_keeps_application_paths_and_no_model_requirement(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'APP',tmp_path/'app')
    monkeypatch.setattr(runtime,'prepare',lambda _:tmp_path/'runtime')
    def call(cmd,**kwargs):
        assert 'phase3_base_pool' in cmd[3] and 'cpu' in cmd
        assert '--model-dir' not in cmd
        assert Path(cmd[cmd.index('--root')+1])==tmp_path/'app/checkpoints/phase3_base_pool_v1'
        assert kwargs['cwd']==tmp_path/'runtime'
        return 0
    monkeypatch.setattr(runtime.subprocess,'call',call)
    assert runtime.execute('phase3_base_pool',['cpu'])==0


def test_failed_projection_preserves_report_and_rejects_ready(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    j,common,_=common_fixture(tmp_path,monkeypatch)
    manifest=common/'native/manifest.json'
    value=json.loads(manifest.read_text());value['base_pool_contract']=pool.contract()
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(pool.common,'ensure',lambda *a,**kw:(common/'native/frame_index.jsonl',common/'reference.json'))
    with pytest.raises(ValueError,match='base_pool_capacity_rejected'):
        pool.ensure(tmp_path/'pool',j['data_root'],tmp_path/'collection',lambda *a:pytest.fail('build'))
    ready=json.loads((tmp_path/'pool/ready_1280.json').read_text())
    assert ready['status']=='rejected' and Path(ready['admission']).is_file()
    assert pool.trial.read(ready['admission'])['supply']


def test_coverage_launcher_uses_full_pool_for_all_workers(tmp_path,monkeypatch):
    import sys
    from qwen3vl_local.audit_joint.tests.test_phase3_common_data import common_fixture
    from qwen3vl_local.audit_joint import phase3_launch as launch
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    j,root,_=common_fixture(tmp_path,monkeypatch)
    before=(root/'native/frame_index.jsonl').read_bytes()
    seen=[]
    def ensure(target,*args,**kwargs):
        assert target==root.parent/'phase3_base_pool_v1'
        seen.append('full_pool')
        return root/'native/frame_index.jsonl',root/'reference.json'
    monkeypatch.setattr(pool,'ensure',ensure)
    monkeypatch.setattr(pool.common,'ensure',lambda *a,**kw:pytest.fail('legacy common build'))
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(pool.trial,'gpu_ids',lambda *a:'0,1,2,3')
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    def run(cmd,log,*args,**kwargs):
        assert 'qwen3vl_local.audit_joint.phase3_base_coverage' in cmd
        assert str(root/'reference.json') in cmd
        log.write_text('ValueError: insufficient_capacity: DYNAMIC_CUTIN')
        seen.append(log.stem)
        return dict(exit_code=2,log=str(log),seconds=0)
    monkeypatch.setattr(pool.trial,'run_stage',run)
    monkeypatch.setattr(sys,'argv',['launch','--method','base','--base-budget','coverage_v2',
        '--data-root',j['data_root'],'--model-dir',j['model_dir'],
        '--common-data-dir',str(root),'--output',str(tmp_path/'run')])
    assert launch.main()==2 and seen==['full_pool','prepare']
    assert (root/'native/frame_index.jsonl').read_bytes()==before
