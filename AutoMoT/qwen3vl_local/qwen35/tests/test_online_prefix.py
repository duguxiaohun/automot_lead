"""CPU regression: production cache isolation, question windows and online routing."""
import copy
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.qwen35 import text_prefix as tp, question_schedule as qs, online
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_text_prefix import inputs
from qwen3vl_local.sft_new_loop_phase12 import cache, evaluate


def bundle(model):
    return SimpleNamespace(model=model, device=torch.device('cpu'),
        tokenizer=SimpleNamespace(eos_token_id=1,convert_tokens_to_ids=lambda _:1),
        processor=SimpleNamespace(batch_decode=lambda ids,**kw:[str(ids.flatten().tolist())]))


def test_production_stream_includes_cold_start_and_isolates_replays(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    b=bundle(model)
    data={key:inputs() for key in ('a','b','c','d')}
    for number,item in enumerate(data.values()):item['input_ids'][0,50]=150+number
    monkeypatch.setattr(cache,'encode',lambda bundle,images,text:dict(data[text],pixel_values=-data[text]['pixel_values']) if images else data[text])
    cases=[]
    for number,pair in enumerate((('a','b'),('a','b'),('c','d'),('c','d'))):
        result=cache.generate(b,[],pair,max_new_tokens=4,check_full=True,
                              verify_outputs=True,benchmark_repeats=2,other_images=["alternate"])
        cases.append(dict(performance=result,phase1=result['branches'][0],phase2=result['branches'][1]))
        stats=tp.bank(b).stats
        assert stats['requests']==(number+1)*2
        assert stats['full_hits']==(0,2,2,4)[number]
        assert stats['misses']==(2,2,4,4)[number]
        assert [arm['text_cache']['full_hit'] for arm in result['branches']]==[bool(number%2)]*2
        assert result['paired_benchmark']['warmup_pairs_per_method']==0
        cache.verify_benchmark(result['paired_benchmark'],device='cpu',dtype='torch.float32',max_new_tokens=4)
    summary=evaluate.stream_summary(cases)
    assert summary['requests']==8 and summary['cache']['misses']==4
    assert summary['shared_seconds']==sum(c['performance']['cached_pair_seconds'] for c in cases)
    bad=copy.deepcopy(cases);bad[0]['performance']['stream_benchmark']['cache_before']['requests']=1
    with pytest.raises(ValueError,match='cold'):evaluate.stream_summary(bad)
    bad=copy.deepcopy(cases);bad[1]['performance']['stream_benchmark']['cache_before']['requests']+=2
    with pytest.raises(ValueError,match='continuity'):evaluate.stream_summary(bad)


def test_isolated_cache_restored_even_on_error(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b=bundle(model);original=tp.bank(b)
    with pytest.raises(RuntimeError):
        with cache.isolated_text_cache(b):
            tp.prefill(b,inputs());raise RuntimeError('probe failed')
    assert tp.bank(b) is original and not original.entries


def test_windows_are_rank_local_reproducible_and_rotate():
    a=qs.QuestionSchedule(window_frames=4,slots=2,rank=1)
    b=qs.QuestionSchedule(window_frames=4,slots=2,rank=1)
    rows=[a.next() for _ in range(8)]
    assert rows==[b.next() for _ in range(8)]
    assert rows[0]['seed']==rows[2]['seed'] and rows[1]['seed']==rows[3]['seed']
    assert rows[0]['seed']!=rows[4]['seed']
    assert rows[0]['seed']!=qs.QuestionSchedule(window_frames=4,slots=2,rank=2).next()['seed']
    with pytest.raises(ValueError):qs.QuestionSchedule(window_frames=0)
    with pytest.raises(ValueError):qs.QuestionSchedule(slots=3)


@pytest.mark.parametrize('phase',[1,2,3])
def test_native_online_windows_keep_text_and_refresh_images_and_context(monkeypatch,phase):
    from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
    from qwen3vl_local.sft_new_loop_phase3 import prompts as p3
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('AUTOMOT_QUESTION_WINDOW','2')
    b=SimpleNamespace(model=SimpleNamespace(eval=lambda:None));seen=[]
    monkeypatch.setattr(online,'_kv_start_state',lambda bundle,messages:seen.append(messages))
    monkeypatch.setattr(online,'_student_generate_kv',lambda *a:('invalid fixture',None,torch.tensor([1])))
    from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
    from qwen3vl_local.sft_new_loop_phase3 import prompt_contract
    module={1:p1,2:p2,3:p3}[phase]
    meta=dict(prompt_name=module.PROMPT_NAME,history_rgb_mode='4rgb',history_rgb_count=4,
              history_rgb_selected_indices=[0,1,2,3],input_layout=tp.identity())
    if phase==3:
        meta.update(action_output_mode='binary',prompt_variant='baseline',
                    production_prompt_sha256=prompt_contract.action_prompt_sha256())
    else:
        meta['production_prompt_sha256']=(p1.phase1_prompt_sha256 if phase==1 else p2.event_prompt_sha256)()
    online.bind_prompt_contract(b,phase,meta)
    kwargs=dict(domain=p2.ROAD_DOMAIN)
    if phase==3:
        context=next(iter(p3.CONTEXT_BY_ID.values()))
        kwargs.update(context_id=context.context_id,road_structure=next(iter(p3.ROAD_STRUCTURE_TEXT)),current_speed_mps=5)
    rows=[online.predict_phase(b,phase,[i]*4,**kwargs) for i in range(3)]
    assert rows[0]['schedule']['seed']==rows[1]['schedule']['seed']!=rows[2]['schedule']['seed']
    assert seen[0][1]['content'][0]==seen[1][1]['content'][0]
    assert seen[0][1]['content'][1]['image']==0 and seen[1][1]['content'][1]['image']==1
    if phase==3:
        # Never retain a stale speed/context just to improve prefix hits.
        kwargs['current_speed_mps']=12
        online.predict_phase(b,phase,[3]*4,**kwargs)
        assert seen[2][1]['content'][0]!=seen[3][1]['content'][0]


def test_fused_predict_automatic_window_and_explicit_fixed_seed(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import infer, prompts
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('AUTOMOT_QUESTION_WINDOW','2')
    b=SimpleNamespace(model=SimpleNamespace(eval=lambda:None));seen=[]
    def generate(bundle,images,texts):
        seen.append(texts);return dict(branches=[dict(text=''),dict(text='')])
    monkeypatch.setattr(infer.cache,'generate',generate)
    infer.bind_online_contract(b,dict(contract=infer.CONTRACT,mode='4rgb',prompt=prompts.identity('4rgb')))
    rows=[infer.predict(b,[i]*4,'4rgb',prompts.P2.ROAD_DOMAIN) for i in range(3)]
    assert seen[0]==seen[1] and rows[0]['schedule']['window']==0 and rows[2]['schedule']['window']==1
    fixed=infer.predict(b,[1]*4,'4rgb',prompts.P2.ROAD_DOMAIN,schedule=False)
    assert fixed['schedule'] is None
    with pytest.raises(ValueError,match='two cache entries'):
        infer.predict(b,[1]*4,'4rgb',prompts.P2.ROAD_DOMAIN,schedule=qs.QuestionSchedule(slots=2))


def test_prior_engine_routes_only_adapter_requests_and_retains_no_history(model,monkeypatch):
    from qwen3vl_local.action_prior.runtime import PriorEngine
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    seen=[];full=[]
    engine=SimpleNamespace(model=model,processor=SimpleNamespace(batch_decode=lambda *a,**k:['answer']),
        build_messages=lambda system,prompt,images:[dict(role='system',content=system),
            dict(role='user',content=[*[dict(type='image',image=i) for i in images],dict(type='text',text=prompt)])],
        apply_chat_template=lambda messages:seen.append(messages),prepare_inputs=lambda *a:inputs(),
        prefill=lambda item:full.append(True) or model(**item,use_cache=True),decode=lambda *a:torch.tensor([[1]]))
    runtime=PriorEngine.__new__(PriorEngine)
    runtime.engine=engine;runtime.online_text_prefix=True;runtime._active_prior_phase='phase1';runtime._phase_prefix_bundles={}
    for i in range(2):runtime.generate_messages('system','followup',[i],history=[('first',f'answer{i}')])
    retained=tp.bank(runtime._phase_prefix_bundles['phase1'])
    assert retained.stats['full_hits']==1 and all(len(k)==65 for k in retained.entries)
    assert seen[0][1]['content'][0]['text']=='first' and seen[1][2]['content']=='answer1'
    assert seen[1][1]['content'][1]['image']==1 and not full
    runtime._active_prior_phase='base';runtime.generate_messages('system','analysis',[3])
    assert full==[True] and seen[-1][1]['content'][0]['type']=='image'
    assert retained.stats['requests']==2


def test_prior_engine_rejects_old_layout_before_loading(monkeypatch):
    from qwen3vl_local.action_prior.runtime import PriorEngine
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    with pytest.raises(ValueError,match='adapters trained'):
        PriorEngine(None,dict(phase1=dict(metadata={})),online_text_prefix=True)


def test_per_adapter_banks_survive_base_and_other_adapter_switch(model,monkeypatch):
    from peft import LoraConfig,get_peft_model
    from qwen3vl_local.action_prior.runtime import PriorEngine
    from qwen3vl_local.qwen35.tests.test_text_prefix import check
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    adapters=get_peft_model(model,config,adapter_name='phase1')
    adapters.add_adapter('phase2',config);adapters.eval().requires_grad_(False)
    with torch.no_grad():
        for name,p in adapters.named_parameters():
            if 'lora_B' in name:p.normal_(0,.03)
    runtime=PriorEngine.__new__(PriorEngine)
    runtime.engine=SimpleNamespace(model=adapters.get_base_model());runtime.adapters=adapters
    runtime._active_prior_phase=None
    banks={phase:tp.TextPrefixCache() for phase in ('phase1','phase2')}
    for _ in range(2):
        for phase in ('phase1','phase2','base'):
            with runtime.mode(phase):
                if phase=='base':
                    with torch.no_grad():runtime.engine.model(**inputs(),use_cache=True)
                else:check(runtime.engine.model,banks[phase],inputs())
    for bank in banks.values():
        assert bank.stats['full_hits']==1 and bank.stats['invalidations']==0
