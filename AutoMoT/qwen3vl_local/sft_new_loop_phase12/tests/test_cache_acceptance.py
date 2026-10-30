import copy
import json
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.sft_new_loop_phase12 import cache, cache_checks, audit
from qwen3vl_local.qwen35.tests.test_backend import model
from test_data_delivery import evidence


def bundle_for(model, monkeypatch):
    prefix=[10,2,4,4,4,4,3,2,4,4,4,4,3]
    pixels=torch.randn(32,3*2*16*16);grid=torch.tensor([[1,4,4]]*2)
    def encode(bundle,images,text):
        ids=torch.tensor([prefix+([11,12,13] if text=='a' else [21,22,23,24])])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=pixels,image_grid_thw=grid)
    monkeypatch.setattr(cache,'encode',encode)
    return SimpleNamespace(model=model,device=torch.device('cpu'),
        tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda x:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())]))


def test_prompt_logit_difference_is_recorded_not_falsely_passed(model,monkeypatch):
    bundle=bundle_for(model,monkeypatch);original=model.forward
    def forward(**kwargs):
        out=original(**kwargs)
        if kwargs.get('use_cache') is False:
            out.logits=out.logits.clone();out.logits[:,-2,0]+=8.3125
        return out
    monkeypatch.setattr(model,'forward',forward)
    report=cache.generate(bundle,[],['a','b'],max_new_tokens=8,check_full=True)
    assert report['full_forward_equivalence_checked']
    assert report['full_forward_equivalence_passed'] is False
    assert report['full_answer_equivalence_checked']
    assert all(b['full_comparison']['mismatches']>0 for b in report['branches'])
    cache_checks.verify_output_evidence(report,report['branches'],require_logits=True)
    report['full_forward_equivalence_passed']=True
    with pytest.raises(ValueError,match='misreported'):
        cache_checks.verify_output_evidence(report,report['branches'],require_logits=True)


@pytest.mark.parametrize('problem',['token','stop','limit'])
def test_output_difference_still_blocks_generation(model,monkeypatch,problem):
    bundle=bundle_for(model,monkeypatch);original=cache._pair
    def pair(bundle,inputs,max_new_tokens,shared):
        result=original(bundle,inputs,max_new_tokens,shared)
        if not shared:
            branch=result['branches'][0]
            if problem=='token':branch['token_ids'][0]+=1
            elif problem=='stop':branch['stop_token_id']=999
            else:branch['reached_token_limit']=not branch['reached_token_limit']
        return result
    monkeypatch.setattr(cache,'_pair',pair)
    with pytest.raises(ValueError):cache.generate(bundle,[],['a','b'],max_new_tokens=8,check_full=True)


@pytest.mark.parametrize('problem',['nan','decision','shape'])
def test_invalid_or_decision_changing_logits_are_not_diagnostic_only(problem):
    a=torch.zeros(1,3,8);a[:,:,-1]=1;b=a.clone()
    if problem=='nan':b[:,0,0]=float('nan')
    elif problem=='decision':b[:,-1,0]=2
    else:b=b[:,:-1]
    with pytest.raises(ValueError):cache_checks.logits(a,b)


@pytest.mark.parametrize('location',['probe','validation','test'])
@pytest.mark.parametrize('problem',['protocol','parity','logits','tokens'])
def test_auditor_rejects_missing_or_false_cache_evidence(tmp_path,location,problem):
    evidence(tmp_path)
    if location=='probe':
        path=tmp_path/'runs/4rgb/gpu_prefix_probe.json';obj=json.loads(path.read_text())
        perf=obj['reports'][0];branches=perf['branches']
    else:
        path=tmp_path/('runs/4rgb/val_step_1000/cases.jsonl' if location=='validation' else 'evaluation/4rgb/fusion_test/cases.jsonl')
        obj=json.loads(path.read_text());perf=obj['performance'];branches=[obj['phase1'],obj['phase2']]
    if problem=='protocol':perf.pop('cache_acceptance_protocol')
    elif problem=='parity':perf['full_answer_equivalence_checked']=False
    elif problem=='logits':branches[0]['full_comparison']['mismatches']=1
    else:perf['independent_branches'][0]['token_ids']=[999]
    path.write_text(json.dumps(obj)+'\n')
    with pytest.raises((ValueError,KeyError)):audit.verify_evaluations(tmp_path,'4rgb')


def test_training_then_single_gpu_evaluation_pin_same_physical_ids(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import launch,integrity
    root=tmp_path/'run';root.mkdir();version=tmp_path/'version'
    monkeypatch.setattr(launch,'APP',version/'app')
    monkeypatch.setattr(launch.runtime,'verify',lambda *_:None)
    monkeypatch.setattr(integrity,'verify_ready',lambda *a,**k:{})
    monkeypatch.setattr(integrity,'verify_training',lambda *a,**k:{})
    monkeypatch.setattr(launch,'gpu_ids',lambda:['0','1','2','3'])
    (root/'cpu').mkdir();(root/'cpu/guard_coverage.json').write_text(json.dumps(dict(selected=2,coverage_adequate=True)))
    calls=[]
    def stage(command,path,*,timeout,env):
        name=path.stem;calls.append(name)
        if name.endswith('_train'):
            assert env['GPU_IDS']==env['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
            assert '--max_restarts=0' in command
        if name.endswith('_test'):
            assert env['GPU_IDS']==env['CUDA_VISIBLE_DEVICES']=='0'
            (root/(name.removesuffix('_test')+'_audit_bundle.tar.gz')).write_bytes(b'fixture')
    monkeypatch.setattr(launch.data_stage,'run',stage)
    launch.worker(SimpleNamespace(command='train',run_dir=root,version=version,model_dir=tmp_path/'base'))
    assert calls==['model_check','2rgb_endpoints_train','2rgb_endpoints_test','4rgb_train','4rgb_test']
