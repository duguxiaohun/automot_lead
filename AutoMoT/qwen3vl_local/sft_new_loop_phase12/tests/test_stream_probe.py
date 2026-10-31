"""Real tiny BF16 model smoke; synthetic processor, no 4B/CARLA certification."""
import json
from types import SimpleNamespace
import pytest
from PIL import Image
from qwen3vl_local.qwen35 import cache_policy
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_online_benchmark import setup
from qwen3vl_local.sft_new_loop_phase12 import stream_probe as probe, launch, integrity


def fixture(model,tmp_path,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    bundle,rows,_=setup(model,tmp_path,12)
    items=[SimpleNamespace(row=SimpleNamespace(question_domain=domain,
        history_rgb_paths=[rows[i]['images'][0]]*4))
        for domain in probe.prompts.P2.QUESTION_DOMAINS for i in range(2)]
    monkeypatch.setattr(probe.plan,'evaluation',lambda *a:items)
    (tmp_path/'ready.json').write_text(json.dumps(dict(data_root=str(tmp_path),text_cache_policy=cache_policy.current())))
    return bundle,items


def test_two_modes_domains_real_switch_hit_and_equal_work(model,tmp_path,monkeypatch):
    bundle,items=fixture(model,tmp_path,monkeypatch)
    receipt=probe.run(bundle,tmp_path,tmp_path/'output',max_new_tokens=2)
    assert receipt['status']=='stream_reuse_and_timing_checked_quality_pending'
    assert receipt['optimizer_updates']==0 and not receipt['trained_adapter']
    assert not receipt['source_verified_after_measurement']  # only frozen CLI may stamp this
    assert len(receipt['cases'])==4
    for case in receipt['cases']:
        assert case['status']=='passed' and case['cache']['full_hits']>=4
        assert case['summary']['cached']['generated_tokens']==case['summary']['fresh']['generated_tokens']
        assert case['summary']['fresh']['cache_classes']['miss']['count']==16
        assert case['summary']['cached']['token_limit_calls']>0
        target=tmp_path/'output'/case['mode']/case['domain']
        assert (target/'resident_text_stream.json').is_file()
        assert (target/'trial_01_fresh.json').is_file()
        rows=json.loads((target/'requests.json').read_text())
        assert rows[0]==rows[2] and rows[1]==rows[3] and rows[0]!=rows[1]
        assert set(rows[0])=={'images','context'}


def test_no_second_group_rejected_before_benchmark(model,tmp_path,monkeypatch):
    bundle,items=fixture(model,tmp_path,monkeypatch)
    monkeypatch.setattr(probe.plan,'evaluation',lambda *a:items[::2])
    with pytest.raises(ValueError,match='Two distinct'):probe.run(bundle,tmp_path,tmp_path/'output')
    assert json.loads((tmp_path/'output/result.json').read_text())['status']=='failed'


def test_duplicate_pixels_at_different_paths_do_not_count_as_cross_image(model,tmp_path,monkeypatch):
    _,items=fixture(model,tmp_path,monkeypatch)
    with Image.open(tmp_path/'0.png') as image:image.save(tmp_path/'1.png')
    with pytest.raises(ValueError,match='Two distinct'):
        probe.requests(items,'2rgb_endpoints',probe.prompts.P2.ROAD_DOMAIN)


def test_mid_benchmark_failure_preserves_raw_receipt(model,tmp_path,monkeypatch):
    bundle,_=fixture(model,tmp_path,monkeypatch)
    def fail(*a,**kw):
        kw['on_run']('cold_cached',{'status':'failed','failed_frame':1})
        raise ValueError('injected runtime failure')
    monkeypatch.setattr(probe.online_benchmark,'benchmark',fail)
    with pytest.raises(ValueError,match='injected'):probe.run(bundle,tmp_path,tmp_path/'output')
    receipt=json.loads((tmp_path/'output/result.json').read_text())
    assert receipt['status']=='failed' and receipt['cases'][0]['status']=='failed'
    assert list((tmp_path/'output').glob('*/*/cold_cached.json'))


def test_launcher_stream_probe_uses_one_idle_card_and_frozen_worker(tmp_path,monkeypatch):
    root=tmp_path/'run';(root/'cpu').mkdir(parents=True);version=tmp_path/'version'
    monkeypatch.setattr(launch,'APP',version/'app')
    monkeypatch.setattr(launch.runtime,'verify',lambda *_:None)
    monkeypatch.setattr(integrity,'verify_ready',lambda *a,**k:{})
    monkeypatch.setattr(launch,'gpu_ids',lambda count:['2'] if count==1 else pytest.fail('four GPUs'))
    (root/'cpu/guard_coverage.json').write_text(json.dumps(dict(selected=2,coverage_adequate=True)))
    calls=[]
    def stage(command,path,*,timeout,env):
        calls.append(command)
        assert env['GPU_IDS']==env['CUDA_VISIBLE_DEVICES']=='2'
        assert 'qwen3vl_local.sft_new_loop_phase12.stream_probe' in command
        assert command[-2:]==['--max-new-tokens','48']
        assert not {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'}&set(env)
        assert timeout==3600 and path.parent.parent==root/'stream_probes'
    monkeypatch.setattr(launch.data_stage,'run',stage)
    launch.worker(SimpleNamespace(command='stream-probe',run_dir=root,version=version,
        model_dir=tmp_path/'base',stream_max_new_tokens=48))
    assert len(calls)==1 and not (root/'runs').exists()
    assert json.loads((root/'status.json').read_text())['status']=='single_gpu_stream_probe_passed_training_not_started'
