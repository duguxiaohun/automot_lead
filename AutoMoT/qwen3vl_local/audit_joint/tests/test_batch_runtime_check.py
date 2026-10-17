import json
import sys
import pytest
from qwen3vl_local.audit_joint import batch_runtime_check as check


@pytest.mark.parametrize('code',[0,124,7])
def test_cpu_bounded_exit_and_no_gpu(monkeypatch,tmp_path,code):
    calls=[]
    monkeypatch.setattr(check,'source_check',lambda *a:{'status':'matched'})
    def run(args,log,env,timeout):
        calls.append((args,env,timeout));return code
    monkeypatch.setattr(check,'run_owned',run)
    out=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['check','--output',str(out),'--timeout','30'])
    with pytest.raises(SystemExit) as exc:check.main()
    assert exc.value.code==code
    args,env,timeout=calls[0]
    assert timeout==30 and env['CUDA_VISIBLE_DEVICES']==''
    assert 'pytest' in args and '--worker' not in args
    status=json.loads((out/'status.json').read_text())
    assert status['completed']==(code==0) and not status['full_pool_admitted']


def test_source_mismatch_stops_before_output(monkeypatch,tmp_path):
    monkeypatch.setattr(check,'source_check',lambda *a:{'status':'mismatch'})
    out=tmp_path/'new'
    monkeypatch.setattr(sys,'argv',['check','--output',str(out)])
    with pytest.raises(SystemExit):check.main()
    assert not out.exists()


def test_upgrade_known_v73_view_but_not_missing_current_binding(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from qwen3vl_local.audit_joint import phase4_view_upgrade as upgrade
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import contract,POLICY,ROOT
    release=json.loads(Path(upgrade.__file__).with_name('release_sources_v73_epoch_recovery_20261017.json').read_text())['files']
    current=contract();old=deepcopy(current)
    for name in old['sources']:
        key=str((ROOT/name).resolve().relative_to(ROOT.parents[2]))
        if key in release:old['sources'][name]=release[key]
    del old['sources']['../../batch_runtime.py']
    path=tmp_path/'student_view.json'
    path.write_text(json.dumps(dict(schema=POLICY,source=old)))
    _,normalized,changes=upgrade.inspect_view(tmp_path)
    assert normalized==current and '../../batch_runtime.py (added)' in changes
    damaged=deepcopy(current);del damaged['sources']['../../batch_runtime.py']
    path.write_text(json.dumps(dict(schema=POLICY,source=damaged)))
    with pytest.raises(ValueError,match='missing batch runtime'):upgrade.inspect_view(tmp_path)


@pytest.mark.parametrize('key,value',[('EVAL_BATCH_SIZE','2'),('REUSE_FROZEN_RGB','1')])
def test_resume_rejects_runtime_configuration_change(monkeypatch,tmp_path,key,value):
    from qwen3vl_local.audit_joint.tests.test_phase4_training import make_checkpoint
    from qwen3vl_local.audit_joint.phase4_training import checkpoint
    path=make_checkpoint(tmp_path,2)
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError,match='configuration'):checkpoint(path,2,3)


@pytest.mark.parametrize('valid',[False,True])
def test_gpu_requires_both_modes_and_pins_one_card(monkeypatch,tmp_path,valid):
    from qwen3vl_local.sft_new_loop_phase4 import devices
    monkeypatch.setattr(check,'source_check',lambda *a:{'status':'matched'})
    monkeypatch.setattr(devices,'select_gpus',lambda n:'3' if n==1 else 'bad')
    native=tmp_path/'native';native.mkdir();(native/'train.jsonl').write_text('{}')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    out=tmp_path/'out'
    def run(args,log,env,timeout):
        assert env['CUDA_VISIBLE_DEVICES']=='3' and timeout==7
        assert '--worker' in args and '--native' in args
        (out/'result.json').write_text(json.dumps(dict(status='correctness_passed',
            modes=[dict(rgb_mode=m) for m in ([2,4] if valid else [2])])))
        return 0
    monkeypatch.setattr(check,'run_owned',run)
    monkeypatch.setattr(sys,'argv',['check','--check','gpu','--native',str(native),
        '--model-dir',str(model),'--output',str(out),'--timeout','7'])
    with pytest.raises(SystemExit) as exc:check.main()
    assert exc.value.code==(0 if valid else 2)

    status=json.loads((out/'status.json').read_text())
    assert status['returncode']==exc.value.code and status['worker_returncode']==0
