import json
from pathlib import Path
import tarfile
import pytest
from qwen3vl_local.audit_joint.phase1_visual_refine import remote as r


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_discover_current_machine_assets_and_symlink(tmp_path,layout):
    app=tmp_path/layout/'AutoMoT';app.mkdir(parents=True)
    assets=tmp_path/'results';assets.mkdir();(app/'checkpoints').symlink_to(assets,target_is_directory=True)
    adapter=assets/'run/best_generation';adapter.mkdir(parents=True);(adapter/'phase12_adapter_identity.json').write_text('{}')
    folder=assets/'pipeline/4rgb/eval_review/lora_production';folder.mkdir(parents=True)
    inputs=folder/'evaluation_inputs.json'
    inputs.write_text(json.dumps({'phase':1,'history_rgb_mode':'4rgb','arguments':{'adapter_dir':'/other/server/AutoMoT/checkpoints/run/best_generation'}}))
    found,local=r.discover(app/'checkpoints/pipeline','4rgb',app)
    assert found.resolve()==inputs and local.resolve()==adapter
    with pytest.raises(FileNotFoundError):r.discover(app/'checkpoints/pipeline','2rgb_endpoints',app)
    (adapter/'phase12_adapter_identity.json').unlink()
    with pytest.raises(FileNotFoundError,match='Original local adapter'):r.discover(app/'checkpoints/pipeline','4rgb',app)


def test_gpu_automatic_and_explicit_pin(monkeypatch):
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**kw:'0, 20000, 80\n1, 500, 0\n2, 500, 50\n')
    assert r.choose_gpu()=='1'
    monkeypatch.setenv('GPU_IDS','2');assert r.choose_gpu()=='2'
    monkeypatch.setenv('GPU_IDS','0,1')
    with pytest.raises(ValueError,match='one GPU'):r.choose_gpu()


def test_package_all_raw_outputs_and_refuse_tamper(tmp_path):
    out=tmp_path/'run';out.mkdir();inputs=tmp_path/'inputs.json';inputs.write_text('{}')
    for name in ('baseline.jsonl','candidate.jsonl','comparison.json','preparation.json'):(out/name).write_text('{}\n')
    done=dict(status='completed',source_sha256={},outputs={n:r.sha(out/n) for n in ('baseline.jsonl','candidate.jsonl')},
              comparison_sha256=r.sha(out/'comparison.json'),frozen_input_sha256=r.sha(inputs))
    (out/'completion.json').write_text(json.dumps(done));archive=r.package(out,inputs)
    with tarfile.open(archive) as tar:
        assert {'baseline.jsonl','candidate.jsonl','evaluation_inputs.json','completion.json'}<=set(tar.getnames())
    (out/'candidate.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='binding mismatch'):r.package(out,inputs)


def test_review_required_preserves_both_modes_and_returns_two(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase1_visual_refine import experiment
    model=tmp_path/'model';model.mkdir();(model/'model.safetensors.index.json').write_text('{}')
    monkeypatch.setattr(r,'discover',lambda p,m:(tmp_path/(m+'.json'),tmp_path/m))
    monkeypatch.setattr(r,'admit',lambda *a,**kw:(tmp_path,{'status':'fixture_rgb_ready'}))
    monkeypatch.setattr(r,'choose_gpu',lambda:'0');monkeypatch.setenv('GPU_IDS','0');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    calls=[]
    def replay(*args):
        calls.append(Path(args[-1]).name);Path(args[-1]).mkdir()
        return {'status':'review_required' if len(calls)==1 else 'no_observed_regression'}
    monkeypatch.setattr(experiment,'replay',replay);monkeypatch.setattr(r,'package',lambda out,inputs:out.with_suffix('.tar.gz'))
    monkeypatch.setattr(r.sys,'argv',['remote','all','--model-dir',str(model),'--data-root',str(tmp_path),'--output-root',str(tmp_path/'outputs')])
    with pytest.raises(SystemExit) as exc:r.main()
    assert exc.value.code==2 and calls==list(r.MODES)
    status=json.loads(next((tmp_path/'outputs').glob('*/status.json')).read_text());assert status['status']=='review_required'


def test_missing_second_mode_rgb_stops_before_gpu_or_output(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir();(model/'model.safetensors.index.json').write_text('{}')
    monkeypatch.setattr(r,'discover',lambda p,m:(tmp_path/(m+'.json'),tmp_path/m))
    def admit(inputs,*args,**kwargs):
        if inputs.name=='4rgb.json':raise FileNotFoundError('fixture missing 4rgb')
        return tmp_path,{'status':'fixture_rgb_ready'}
    monkeypatch.setattr(r,'admit',admit)
    monkeypatch.setattr(r,'choose_gpu',lambda:pytest.fail('GPU selection must follow all RGB checks'))
    monkeypatch.setattr(r.sys,'argv',['remote','all','--model-dir',str(model),'--output-root',str(tmp_path/'outputs')])
    with pytest.raises(FileNotFoundError,match='missing 4rgb'):r.main()
    assert not (tmp_path/'outputs').exists()
