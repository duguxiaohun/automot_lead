import json
from pathlib import Path
import tarfile

import pytest

from qwen3vl_local.audit_joint.phase12_retrain.collect_capacity import collect


def fixture(tmp_path, layout):
    app = tmp_path / layout / 'AutoMoT'
    app.mkdir(parents=True)
    assets = tmp_path / 'assets'; assets.mkdir()
    (app / 'checkpoints').symlink_to(assets, target_is_directory=True)
    run = assets / 'failed'; run.mkdir()
    data = assets / 'explicit_parent'; data.mkdir()
    index = data / 'frame_index.jsonl'
    index.write_text(json.dumps(dict(scenario='S',route_id='R',frame_id=1,split='train'))+'\n')
    (data / 'manifest.json').write_text('{}')
    version = assets / 'runtime/v1'; version.mkdir(parents=True)
    (version / 'manifest.json').write_text('{}')
    for name in ('sft_new_loop_phase2/train.py', 'sft_new_loop_phase2/invalid_balance.py',
                 'sft_new_loop_phase1/training_policy.py', 'sft_new_loop_phase1/exposure_sampling.py'):
        p = version / 'app/qwen3vl_local' / name
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text('# frozen source\n')
    (run/'status.json').write_text(json.dumps(dict(phase=2,status='failed',stage='cpu_admission',version=str(version))))
    (run/'cpu_admission.json').write_text('{}')
    (run/'cpu_admission.log').write_text('Coverage quota exceeds compatible frame capacity')
    (run/'cpu').mkdir()
    (run/'cpu/2rgb_endpoints_sampling.json').write_text(json.dumps(dict(command=['python','--index',str(index),'--action-index','auto'])))
    return app,run,index


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_collect_metadata_only_and_effective_hints(tmp_path,layout):
    app,run,index=fixture(tmp_path,layout)
    legacy=app/'checkpoints/sft_new_loop_phase3_data_v23/frame_index.jsonl'
    legacy.parent.mkdir()
    legacy.write_text(json.dumps(dict(scenario='S',route_id='R',frame_id=1,split='train',primary_action='STOP'))+'\n')
    before=index.read_bytes(); out=tmp_path/'bundle.tar.gz'
    receipt=collect(app,run,out)
    assert receipt['model_loaded'] is False and receipt['rgb_read'] is False
    with tarfile.open(out) as tar:
        assert tar.extractfile('data/frame_index.jsonl').read()==before
        hints=json.load(tar.extractfile('effective_action_hints.json'))
        assert hints['values']==[['S','R',1,'STOP']]
        assert not any('safetensors' in p or p.endswith('.jpg') for p in tar.getnames())
    assert index.read_bytes()==before
    with pytest.raises(FileExistsError):collect(app,run,out)


@pytest.mark.parametrize('bad',['training','not_capacity','missing_source'])
def test_reject_wrong_failure_or_missing_evidence(tmp_path,bad):
    app,run,index=fixture(tmp_path,'cr02')
    if bad=='training':
        state=json.loads((run/'status.json').read_text());state['stage']='fresh_train_eval'
        (run/'status.json').write_text(json.dumps(state))
    elif bad=='not_capacity':(run/'cpu_admission.log').write_text('different failure')
    else:index.unlink()
    out=tmp_path/'bundle.tar.gz'
    with pytest.raises(ValueError):collect(app,run,out)
    assert not out.exists()
