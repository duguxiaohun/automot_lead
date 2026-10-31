"""Real tiny Qwen: resident capacity, actual alternating images and decode isolation."""
import copy
import json
import pytest
import torch
from qwen3vl_local.qwen35 import text_prefix as tp
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
from qwen3vl_local.qwen35.tests.test_online_prefix import bundle
from qwen3vl_local.sft_new_loop_phase12 import cache,cache_structure as structure


def requests():
    first=[inputs(prefix=65),inputs(prefix=70)]
    first[1]['input_ids'][0,50]=199
    other=[inputs(prefix=65,grid=6),inputs(prefix=70,grid=6)]
    other[1]['input_ids'][0,50]=199
    return first,other


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_two_resident_questions_cross_images_and_return(model,monkeypatch,dtype):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');model.to(dtype=dtype)
    b=bundle(model);original=tp.bank(b);a,other=requests()
    for item in a+other:item['pixel_values']=item['pixel_values'].to(dtype)
    result=structure.measure_stream(b,a,other,cache,max_new_tokens=3)
    assert result['status']=='passed' and result['required_entries']==2
    assert [(s['question'],s['image']) for s in result['steps']]==[(0,0),(1,0),(0,1),(1,1),(0,0),(1,0)]
    assert [s['cache']['full_hit'] for s in result['steps']]==[False,False,True,True,True,True]
    assert result['cache']['hits']==4 and result['cache']['misses']==2 and result['cache']['evictions']==0
    assert result['required_bytes']==result['cache']['bytes']<=result['limits']['max_bytes']
    assert tp.bank(b) is original and original.stats['requests']==0
    structure.verify_stream(result)
    for mutation in ('missing','hit','bytes','state','fingerprint','answer'):
        bad=copy.deepcopy(result)
        if mutation=='missing':bad['steps'].pop()
        elif mutation=='hit':bad['steps'][2]['cache']['full_hit']=False
        elif mutation=='bytes':bad['required_bytes']+=1
        elif mutation=='state':bad['steps'][2]['comparison']['next_logits']['max_abs']=1
        elif mutation=='fingerprint':bad['steps'][2]['prefix_fingerprint']='changed'
        else:bad['steps'][2]['answer']['text']+='changed'
        with pytest.raises(ValueError):structure.verify_stream(bad)


def test_individual_fit_but_pair_thrash_cannot_pass(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');a,other=requests()
    sizes=[]
    for item in a:
        bank=tp.TextPrefixCache();bank.prefill(model,item);sizes.append(bank.last['prefix_bytes'])
    budget=max(sizes)+1
    assert budget<sum(sizes)
    bank=tp.TextPrefixCache(max_bytes=budget)
    for item in a+other:bank.prefill(model,item)
    assert bank.stats['requests']==4 and bank.stats['hits']==0 and bank.stats['evictions']==3
    b=bundle(model);b._text_prefix_cache=tp.TextPrefixCache(max_bytes=budget)
    with pytest.raises(ValueError,match='cannot coexist') as caught:
        structure.measure_stream(b,a,other,cache,max_new_tokens=1)
    evidence=caught.value.evidence
    assert evidence['status']=='failed' and evidence['required_bytes']==sum(sizes)
    assert b._text_prefix_cache.stats['requests']==0


@pytest.mark.parametrize('kind',['entry','tokens','same_image','missing_image'])
def test_inadequate_stream_controls_rejected(model,monkeypatch,kind):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b=bundle(model);a,other=requests()
    if kind=='entry':b._text_prefix_cache=tp.TextPrefixCache(max_entries=1)
    elif kind=='tokens':b._text_prefix_cache=tp.TextPrefixCache(max_tokens=64)
    elif kind=='same_image':other=a
    else:other=[]
    with pytest.raises(ValueError):structure.measure_stream(b,a,other,cache,max_new_tokens=1)


def test_decode_corrupting_retained_prefix_is_detected(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b=bundle(model);a,other=requests();decode=cache._decode
    def corrupt(bundle,item,out,budget):
        answer=decode(bundle,item,out,budget)
        bank=tp.bank(bundle)
        if bank.stats['requests']==3:
            next(iter(bank.entries.values()))[0].recurrent_states[0].add_(1)
        return answer
    monkeypatch.setattr(cache,'_decode',corrupt)
    with pytest.raises(ValueError):structure.measure_stream(b,a,other,cache,max_new_tokens=1)


def test_default_capacity_and_explicit_environment(monkeypatch):
    for name in ('ENTRIES','MIB','TOKENS'):monkeypatch.delenv('AUTOMOT_TEXT_PREFIX_CACHE_'+name,raising=False)
    assert tp.cache_limits()==dict(max_entries=2,max_bytes=1024**3,max_tokens=8192)
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_MIB','512')
    assert tp.TextPrefixCache().max_bytes==512*1024**2
    assert tp.TextPrefixCache(max_bytes=123).max_bytes==123
    for value in ('0','-1','1.5','nan',' 2',''):
        monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_MIB',value)
        with pytest.raises(ValueError):tp.TextPrefixCache()


def test_cpu_ready_capacity_cannot_change_after_admission(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import integrity,cache_checks,CONTRACT
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    ready=dict(contract=CONTRACT,input_layout=tp.identity(),cache_acceptance_protocol=cache_checks.protocol(),text_cache_limits=tp.cache_limits())
    (tmp_path/'ready.json').write_text(json.dumps(ready))
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_MIB','8')
    with pytest.raises(ValueError,match='cache limits differ'):integrity.verify_ready(tmp_path)


def test_same_image_measure_cannot_substitute_for_cross_image_gate(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b=bundle(model)
    report=structure.measure(b,inputs(),cache)
    structure.verify_reuse(report)
    with pytest.raises(ValueError,match='Missing cross-image'):structure.verify_reuse(report,require_cross=True)


def test_frozen_diagnostic_restores_capacity_without_inheriting_shell():
    from qwen3vl_local.sft_new_loop_phase12.cache_diagnose import restore_capacity,capacity_environment
    env={'AUTOMOT_TEXT_PREFIX_CACHE_MIB':'1','AUTOMOT_TEXT_PREFIX_CACHE_TOKENS':'4','unrelated':'keep'}
    restore_capacity(env,dict(text_cache_limits=dict(max_entries=2,max_bytes=512*1024**2,max_tokens=8192)))
    assert env==dict(AUTOMOT_TEXT_PREFIX_CACHE_MIB='512',AUTOMOT_TEXT_PREFIX_CACHE_TOKENS='8192',
                     AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES='2',unrelated='keep')
    restore_capacity(env,{})
    assert env=={'unrelated':'keep'}
    with pytest.raises(ValueError):capacity_environment(dict(max_entries=2,max_bytes=123,max_tokens=8192))
