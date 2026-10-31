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


def text_bundle(model,monkeypatch):
    from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','1')
    b=bundle_for(model,monkeypatch);item=inputs(prefix=65)
    def encode(bundle,images,text):
        result={k:v.clone() for k,v in item.items()}
        result['input_ids'][0,60]=111 if text=='a' else 112
        if images:result['pixel_values']=-result['pixel_values']
        return result
    monkeypatch.setattr(cache,'encode',encode)
    return b


def test_text_contract_accepts_full_path_decision_change_but_audits_it(model,monkeypatch):
    b=text_bundle(model,monkeypatch);original=cache._pair
    def changed(*args):
        result=original(*args)
        if args[-1] is False:
            result['branches'][0]['token_ids'][0]+=1
            result['branches'][0]['text']+='control difference'
        return result
    monkeypatch.setattr(cache,'_pair',changed)
    forward=model.forward
    def different_logits(**kw):
        out=forward(**kw)
        if kw.get('use_cache') is False:
            out.logits=out.logits.clone();out.logits[:,-1,0]+=100
        return out
    monkeypatch.setattr(model,'forward',different_logits)
    report=cache.generate(b,[],['a','b'],max_new_tokens=3,check_full=True,benchmark_repeats=2,other_images=['alternate'])
    assert report['cache_acceptance_protocol']==cache_checks.TEXT_PROTOCOL
    assert not report['full_answer_equivalence_passed'] and report['full_answer_difference']['arm']==1
    assert not report['branches'][0]['full_comparison']['next_token_matches']
    assert report['same_partition_equivalence_checked']
    assert len(report['text_reuse_structure'])==2
    assert b._text_prefix_cache.stats['requests']==2  # all checks/replays isolated
    cache_checks.verify_output_evidence(report,report['branches'],require_logits=True)
    for field,value in [('cache_acceptance_protocol',cache_checks.PROTOCOL),
                        ('full_answer_equivalence_passed',True),('same_partition_equivalence_checked',False)]:
        altered=copy.deepcopy(report);altered[field]=value
        with pytest.raises(ValueError):cache_checks.verify_output_evidence(altered,altered['branches'],require_logits=True)
    altered=copy.deepcopy(report)
    altered['recomputed_split_branches'][0]['token_ids'][0]+=1
    with pytest.raises(cache_checks.CacheParityError):cache_checks.verify_output_evidence(altered,altered['branches'])
    altered=copy.deepcopy(report)
    state=next(iter(altered['text_reuse_structure'][0]['comparisons']['warm_vs_cold']['full_final_states'].values()))
    state['max_abs']=1
    with pytest.raises(ValueError,match='state changed'):
        cache_checks.verify_output_evidence(altered,altered['branches'])


def test_text_generation_rejects_changed_reused_output(model,monkeypatch):
    b=text_bundle(model,monkeypatch);original=cache._pair
    def corrupt(*args):
        result=original(*args)
        if args[-1] is True:result['branches'][0]['token_ids'][0]+=1
        return result
    monkeypatch.setattr(cache,'_pair',corrupt)
    with pytest.raises(cache_checks.CacheParityError):
        cache.generate(b,[],['a','b'],max_new_tokens=3,verify_outputs=True)


def test_no_retained_prefix_cannot_claim_structural_acceptance(model,monkeypatch):
    b=text_bundle(model,monkeypatch)
    b._text_prefix_cache=cache.text_prefix.TextPrefixCache(max_bytes=1)
    with pytest.raises(ValueError,match='cannot coexist'):
        cache.generate(b,[],['a','b'],max_new_tokens=1,check_full=True,other_images=['alternate'])


def test_full_control_metrics_distinguish_decisions_from_invalid_outputs():
    from qwen3vl_local.sft_new_loop_phase12.evaluate import full_control_metrics
    def arm(value,limit=False):
        return dict(text=str(value),token_ids=[int(bool(value))],stop_token_id=None if limit else 1,
                    reached_token_limit=limit,gt={'X':True},parsed={'X':value})
    case=dict(scenario='fixture',phase1=arm(True),phase2=arm(None,True),
              full_forward_control=dict(phase1=arm(False),phase2=arm(True)))
    report=full_control_metrics([case])
    assert report['changes']['phase1']['valid_decision_changes']==1
    assert report['changes']['phase1']['cached_correct_full_wrong']==1
    assert report['changes']['phase2']['valid_decision_changes']==0
    assert report['changes']['phase2']['invalid_or_truncated']==1


def test_single_gpu_probe_executes_frozen_code_without_training(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import launch,integrity
    root=tmp_path/'run';root.mkdir();version=tmp_path/'version'
    monkeypatch.setattr(launch,'APP',version/'app')
    monkeypatch.setattr(launch.runtime,'verify',lambda *_:None)
    monkeypatch.setattr(integrity,'verify_ready',lambda *a,**k:{})
    monkeypatch.setattr(launch,'gpu_ids',lambda count:['2'] if count==1 else pytest.fail('Requested four GPUs'))
    (root/'cpu').mkdir();(root/'cpu/guard_coverage.json').write_text(json.dumps(dict(selected=2,coverage_adequate=True)))
    calls=[]
    def stage(command,path,*,timeout,env):
        calls.append(command)
        assert env['GPU_IDS']==env['CUDA_VISIBLE_DEVICES']=='2'
        assert '--probe-only' in command and '--probe-output' in command
        assert timeout==3600 and 'torch.distributed.run' not in command
        assert path.parent.parent==root/'probes'
    monkeypatch.setattr(launch.data_stage,'run',stage)
    launch.worker(SimpleNamespace(command='probe',run_dir=root,version=version,model_dir=tmp_path/'base'))
    assert len(calls)==1 and not (root/'runs').exists()
    assert json.loads((root/'status.json').read_text())['status']=='single_gpu_probe_passed_training_not_started'


def test_single_gpu_selection_rejects_busy_and_multiple_ids(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import launch
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:'0, 1024, 0\n1, 0, 0\n2, 0, 0\n')
    monkeypatch.delenv('GPU_IDS',raising=False)
    assert launch.gpu_ids(count=1)==['1']
    for setting in ('0','1,2'):
        monkeypatch.setenv('GPU_IDS',setting)
        with pytest.raises(ValueError):launch.gpu_ids(count=1)


@pytest.mark.parametrize('enabled,recorded',[('0',cache_checks.TEXT_PROTOCOL),('1',cache_checks.PROTOCOL),('1',None)])
def test_cpu_ready_prevents_cache_acceptance_switch(tmp_path,monkeypatch,enabled,recorded):
    from qwen3vl_local.sft_new_loop_phase12 import integrity,CONTRACT
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE',enabled)
    ready=dict(contract=CONTRACT,input_layout='text_before_images_v1')
    if recorded is not None:ready['cache_acceptance_protocol']=recorded
    (tmp_path/'ready.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='cache acceptance differs'):integrity.verify_ready(tmp_path)
