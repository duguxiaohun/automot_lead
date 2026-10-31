from types import SimpleNamespace
import json
import hashlib
import copy
import subprocess
import sys
import tarfile
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import cache,cache_diagnose
from qwen3vl_local.qwen35.tests.test_backend import model


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_streamed_logits_comparison_keeps_existing_tolerance(dtype):
    torch.manual_seed(3);a=torch.randn(1,53,64,dtype=dtype);b=a.clone();b[:,18,10]+=2
    result=cache_diagnose.compare_logits(a,b)
    expected=torch.isclose(a.float(),b.float(),atol=result['atol'],rtol=result['rtol'])
    assert result['mismatches']==int((~expected).sum())==1
    assert result['elements']==a.numel() and not result['all_close']
    assert result['last_argmax_matches'] and result['max_abs']>=1.99
    a[:,0,0]=float('nan')
    result=cache_diagnose.compare_logits(a,b)
    assert result['nonfinite']==1 and not result['all_close']


@pytest.mark.parametrize('frames',[2,4])
def test_real_hybrid_full_output_diagnostic_without_updates(model,frames):
    prefix=[10]+[v for _ in range(frames) for v in (2,4,4,4,4,3)]
    pixels=torch.randn(frames*16,3*2*16*16);grid=torch.tensor([[1,4,4]]*frames)
    def inputs(suffix):
        ids=torch.tensor([prefix+suffix])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,image_grid_thw=grid)
    bundle=SimpleNamespace(model=model,device=torch.device('cpu'),
        tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda x:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())]))
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=cache_diagnose.measure(bundle,[inputs([11,12,13]),inputs([20,21,22,23])],cache,max_new_tokens=8)
    assert result['full_answers_identical'] and all(r['all_close'] for r in result['logits'])
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_real_bf16_long_suffix_regression(model):
    model.to(dtype=torch.bfloat16)
    prefix=[10]+[v for _ in range(2) for v in (2,4,4,4,4,3)]
    ids=torch.tensor([prefix+[11+i%200 for i in range(4993)]])
    item=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(32,3*2*16*16,dtype=torch.bfloat16),image_grid_thw=torch.tensor([[1,4,4]]*2))
    with torch.no_grad():
        n,prefix_cache=cache.prefill_prefix(model,item,item)
        split=cache.fork(model,item,n,prefix_cache)
        full=model(**item,use_cache=False)
        report=cache_diagnose.compare_logits(split.logits,full.logits[:,n:])
    assert report['all_close'] and report['last_argmax_matches']


@pytest.mark.parametrize('fails',[False,True])
@pytest.mark.parametrize('layout',['images_before_text_legacy','text_before_images_v1'])
def test_diagnostic_launcher_preserves_source_and_packages_partial_evidence(tmp_path,monkeypatch,fails,layout):
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    root=source_fixture(tmp_path,layout);before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    output=tmp_path/'diagnostic';monkeypatch.setenv('RANK','0');monkeypatch.setenv('WORLD_SIZE','4')
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setattr(sys,'argv',['diagnose','--source-run',str(root),'--output',str(output)])
    monkeypatch.setattr(cache_diagnose.subprocess,'check_output',lambda *a,**k:'0,40000,0\n1,0,0\n')
    def stage(command,status,timeout,env):
        assert env['CUDA_VISIBLE_DEVICES']==env['GPU_IDS']=='1'
        assert env['AUTOMOT_TEXT_PREFIX']==('1' if layout=='text_before_images_v1' else '0')
        assert env['AUTOMOT_TEXT_PREFIX_CACHE']==env['AUTOMOT_TEXT_PREFIX']
        assert 'RANK' not in env and 'WORLD_SIZE' not in env
        assert timeout==1800 and '--worker' in command
        assert command[command.index('--source-run')+1]==str(root)
        status.write_text('{}');status.with_suffix('.log').write_text('original diagnostic error' if fails else 'done')
        if fails:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(data_stage,'run',stage)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):cache_diagnose.main()
    else:cache_diagnose.main()
    assert {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}==before
    with tarfile.open(str(output)+'.tar.gz') as archive:
        assert set(archive.getnames())=={'worker.json','worker.log','cache_diagnose.py','cache_checks.py'}


def source_fixture(tmp_path,layout='images_before_text_legacy'):
    root=tmp_path/'failed_run';root.mkdir()
    version=tmp_path/'frozen';(version/'app').mkdir(parents=True)
    (root/'execution.json').write_text(json.dumps(dict(version=str(version))))
    (root/'ready.json').write_text(json.dumps(dict(version=str(version),input_layout=layout)))
    folder=root/'runs/2rgb_endpoints';folder.mkdir(parents=True)
    (folder/'training_inputs.json').write_text(json.dumps(dict(prompt=dict(input_layout=layout),
        ready_sha256=hashlib.sha256((root/'ready.json').read_bytes()).hexdigest(),base_model=str(tmp_path/'base'))))
    return root


@pytest.mark.parametrize('layout',['images_before_text_legacy','text_before_images_v1'])
def test_cpu_policy_reads_saved_layout_without_gpu_or_caller_env(tmp_path,monkeypatch,capsys,layout):
    root=source_fixture(tmp_path,layout)
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0' if layout=='text_before_images_v1' else '1')
    monkeypatch.setattr(sys,'argv',['diagnose','--source-run',str(root),'--check-only'])
    def forbidden(*a,**k):raise AssertionError('CPU check selected a GPU')
    monkeypatch.setattr(cache_diagnose.subprocess,'check_output',forbidden)
    cache_diagnose.main();report=json.loads(capsys.readouterr().out)
    assert report['policy']['input_layout']==layout and not report['inference_executed']
    assert report['policy']['cache_switch_source']=='default_not_recorded_in_old_run'


@pytest.mark.parametrize('problem',['ready_sha','layout','version','missing_frozen','probe_layout'])
def test_policy_rejects_changed_or_missing_sources(tmp_path,problem):
    root=source_fixture(tmp_path);config=root/'runs/2rgb_endpoints/training_inputs.json'
    if problem in ('ready_sha','layout'):
        value=json.loads(config.read_text())
        if problem=='ready_sha':value['ready_sha256']='bad'
        else:value['prompt']['input_layout']='text_before_images_v1'
        config.write_text(json.dumps(value))
    elif problem=='version':(root/'execution.json').write_text(json.dumps(dict(version=str(tmp_path/'other'))))
    elif problem=='missing_frozen':(tmp_path/'frozen/app').rmdir()
    else:(config.parent/'gpu_prefix_probe.json').write_text(json.dumps(dict(input_layout='text_before_images_v1')))
    with pytest.raises((ValueError,FileNotFoundError)):cache_diagnose.source_policy(root,'2rgb_endpoints')


@pytest.mark.parametrize('layout',['0','1'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_both_layouts_precision_controls_keep_bank_and_trace_identical(model,monkeypatch,layout,dtype):
    from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
    from qwen3vl_local.qwen35 import text_prefix as tp
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX',layout)
    model.to(dtype=dtype)
    a=inputs(prefix=65);b={k:v.clone() for k,v in a.items()};b['input_ids'][0,-1]=23
    if layout=='1':b['input_ids'][0,50]=155
    bundle=SimpleNamespace(model=model,device=torch.device('cpu'),
        tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda _:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())]))
    bank=tp.bank(bundle);before=copy.deepcopy(bank.report())
    report=cache_diagnose.measure(bundle,[a,b],cache,max_new_tokens=4,warmup_inputs=([a,b],))
    assert report['full_answers_identical'] and report['independent_repeat_identical'] and report['cached_repeat_identical']
    assert report['priming_cases']==1
    trace=cache_diagnose.trace_first_divergence(bundle,[a,b],cache,2,4,warmup_inputs=([a,b],))
    assert trace['status'] in ('identical_to_stop','identical_to_limit')
    assert tp.bank(bundle) is bank and bank.report()==before


@pytest.mark.parametrize('eos',[False,True])
def test_trace_reports_first_disagreement_with_common_history(monkeypatch,eos):
    from qwen3vl_local.sft_v3 import train as native
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')
    ids=torch.tensor([[10,11]])
    def logits(route,step):
        out=torch.zeros(1,16)
        token=6+step if step<2 or route=='cached' else 9
        out[0,token]=2
        return out
    def output(route):return SimpleNamespace(logits=logits(route,0)[:,None,:],past_key_values=route,rope_deltas=None)
    b=SimpleNamespace(model=lambda **kw:output('full'),processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.tolist())]),
        tokenizer=SimpleNamespace(eos_token_id=9 if eos else 1,convert_tokens_to_ids=lambda _:1))
    monkeypatch.setattr(cache_diagnose,'_cached_output',lambda *a:output('cached'))
    def advance(bundle,state,token):
        joined=torch.cat((state.decoded_input_ids,token),1)
        return native.KVState(joined,joined,torch.ones_like(joined),state.past_key_values,None,
                              logits(state.past_key_values,joined.shape[1]-2)),None
    monkeypatch.setattr(native,'_append_token_ids',advance)
    result=cache_diagnose.trace_first_divergence(b,[dict(input_ids=ids,attention_mask=torch.ones_like(ids))],cache,1,8)
    assert result['status']=='diverged' and result['token_index']==2
    assert result['common_token_ids']==[6,7] and result['cached_token_id']==8 and result['independent_token_id']==9
    assert result['independent_is_eos']==eos and result['cached']['margin']==2


def test_output_evidence_is_retained_before_extra_diagnostics_fail(model,monkeypatch):
    from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');data=[inputs(),inputs()]
    b=SimpleNamespace(model=model,device=torch.device('cpu'),tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda _:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.tolist())]))
    original=cache._pair
    def changed(*a):
        result=original(*a)
        if not a[-1]:result['branches'][0]['token_ids'][0]+=1
        return result
    monkeypatch.setattr(cache,'_pair',changed)
    monkeypatch.setattr(cache_diagnose,'trace_first_divergence',lambda *a:(_ for _ in ()).throw(RuntimeError('trace OOM fixture')))
    retained=[]
    with pytest.raises(RuntimeError,match='trace OOM'):
        cache_diagnose.measure(b,data,cache,max_new_tokens=2,on_evidence=lambda r:retained.append(copy.deepcopy(r)))
    assert retained and retained[0]['first_difference']['token_index']==0
    assert retained[0]['shared']['branches'] and retained[0]['independent']['branches']


def test_precision_summary_does_not_certify_or_hide_nondeterminism():
    good=dict(full_answers_identical=True,cached_repeat_identical=True,independent_repeat_identical=True)
    bad={**good,'full_answers_identical':False}
    assert 'precision_sensitive' in cache_diagnose.precision_summary(bad,good)
    assert 'fp32' in cache_diagnose.precision_summary(bad,bad)
    assert 'not_reproduced' in cache_diagnose.precision_summary(good,good)
    assert 'unstable' in cache_diagnose.precision_summary({**bad,'independent_repeat_identical':False},good)


@pytest.mark.parametrize('destination',['run','frozen'])
def test_output_cannot_overwrite_old_run_or_frozen_source(tmp_path,monkeypatch,destination):
    root=source_fixture(tmp_path);output=(root if destination=='run' else tmp_path/'frozen')/'diagnosis'
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setattr(sys,'argv',['diagnose','--source-run',str(root),'--output',str(output)])
    monkeypatch.setattr(cache_diagnose.subprocess,'check_output',lambda *a,**k:'0,0,0\n')
    with pytest.raises(ValueError,match='outside original'):cache_diagnose.main()
    assert not output.exists()


def test_text_cache_disabled_in_saved_probe_is_preserved(tmp_path):
    root=source_fixture(tmp_path,'text_before_images_v1')
    (root/'runs/2rgb_endpoints/gpu_prefix_probe.json').write_text(json.dumps(dict(
        input_layout='text_before_images_v1',text_cache_enabled=False)))
    policy=cache_diagnose.source_policy(root,'2rgb_endpoints')
    assert not policy['text_cache_enabled'] and policy['cache_switch_source']=='probe_receipt'
