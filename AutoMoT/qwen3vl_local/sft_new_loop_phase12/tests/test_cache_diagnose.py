from types import SimpleNamespace
import json
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
def test_diagnostic_launcher_preserves_source_and_packages_partial_evidence(tmp_path,monkeypatch,fails):
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    root=tmp_path/'failed_run';root.mkdir();source=root/'execution.json'
    source.write_text(json.dumps({'version':str(tmp_path/'frozen')}));before=source.read_bytes()
    output=tmp_path/'diagnostic';monkeypatch.setenv('RANK','0');monkeypatch.setenv('WORLD_SIZE','4')
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setattr(sys,'argv',['diagnose','--source-run',str(root),'--output',str(output)])
    monkeypatch.setattr(cache_diagnose.subprocess,'check_output',lambda *a,**k:'0,40000,0\n1,0,0\n')
    def stage(command,status,timeout,env):
        assert env['CUDA_VISIBLE_DEVICES']==env['GPU_IDS']=='1'
        assert 'RANK' not in env and 'WORLD_SIZE' not in env
        assert timeout==1800 and '--worker' in command
        assert command[command.index('--source-run')+1]==str(root)
        status.write_text('{}');status.with_suffix('.log').write_text('original diagnostic error' if fails else 'done')
        if fails:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(data_stage,'run',stage)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):cache_diagnose.main()
    else:cache_diagnose.main()
    assert source.read_bytes()==before and sorted(p.name for p in root.iterdir())==['execution.json']
    with tarfile.open(str(output)+'.tar.gz') as archive:
        assert set(archive.getnames())=={'worker.json','worker.log','cache_diagnose.py'}
