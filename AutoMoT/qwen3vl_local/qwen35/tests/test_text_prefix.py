"""真实小型 hybrid 模型验证跨图片 cache，不替代远端 4B 速度/质量验收。"""
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35 import text_prefix as tp
from qwen3vl_local.mrope_utils import qwen3vl_incremental_forward


def inputs(frames=2, prefix=65, grid=4, tail=(21, 22)):
    """不同图像网格产生不同的 M-RoPE delta。"""
    ids=torch.tensor([[10+i%100 for i in range(prefix)] +
        [v for _ in range(frames) for v in [2]+[4]*(grid*grid//4)+[3]]+list(tail)])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        mm_token_type_ids=(ids==4).long(),image_grid_thw=torch.tensor([[1,grid,grid]]*frames),
        pixel_values=torch.randn(frames*grid*grid,3*2*16*16))


def check(model,cache,item):
    """比较后缀 logits、完整 cache 下的后续解码与 EOS 选择。"""
    with torch.no_grad():
        out=cache.prefill(model,item);n=cache.last['prefix_tokens']
        full=model(**item,use_cache=True,return_dict=True)
        torch.testing.assert_close(out.logits,full.logits[:,n:],atol=3e-5,rtol=3e-4)
        torch.testing.assert_close(out.rope_deltas,full.rope_deltas)
        mask=item['attention_mask']
        for step in range(6):
            a=out.logits[:,-1].argmax(-1)[:,None];b=full.logits[:,-1].argmax(-1)[:,None]
            assert torch.equal(a,b)
            mask=torch.cat((mask,torch.ones_like(a)),1)
            def advance(x,token):
                return qwen3vl_incremental_forward(model,feed_ids=token,attention_mask=mask,
                    past_key_values=x.past_key_values,prefix_len=mask.shape[1]-1,rope_deltas=x.rope_deltas)
            out,full=advance(out,a),advance(full,b)
    return out


@pytest.mark.parametrize('frames',[1,2,4])
@pytest.mark.parametrize('prefix',[3,65])
def test_new_images_grids_and_decode_keep_prefix_immutable(model,frames,prefix):
    cache=tp.TextPrefixCache();check(model,cache,inputs(frames,prefix))
    retained=next(iter(cache.entries.values()))[0]
    frozen={name:[None if x is None else x.clone() for x in getattr(retained,name)]
            for name in ('key_cache','value_cache','conv_states','recurrent_states')}
    check(model,cache,inputs(frames,prefix,grid=6))
    assert cache.stats['hits']==1 and cache.stats['misses']==1
    for name,values in frozen.items():
        for old,new in zip(values,getattr(retained,name)):
            if old is not None:torch.testing.assert_close(old,new,atol=0,rtol=0)


def test_lora_weight_update_and_adapter_switch_invalidates(model):
    from peft import LoraConfig,get_peft_model
    adapted=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,
        target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name,p in adapted.named_parameters():
            if 'lora_B' in name:p.normal_(0,.03)
    cache=tp.TextPrefixCache();check(adapted,cache,inputs())
    with torch.no_grad():
        next(p for n,p in adapted.named_parameters() if 'lora_B' in n).add_(.01)
    check(adapted,cache,inputs());assert cache.stats['invalidations']==1
    with adapted.disable_adapter():check(adapted,cache,inputs())
    check(adapted,cache,inputs());assert cache.stats['invalidations']==3


def test_question_switch_rebuilds_complete_question(model):
    cache=tp.TextPrefixCache(max_entries=2)
    for token in (155,156,157,157,157):
        item=inputs(prefix=65);item['input_ids'][0,50]=token;check(model,cache,item)
        assert cache.last['prefix_tokens']==65
    assert cache.stats['misses']==3 and cache.stats['full_hits']==2
    assert cache.stats['computed_text_prefix_tokens']==195


def test_shorter_text_is_promoted_without_mutation(model):
    cache=tp.TextPrefixCache();check(model,cache,inputs(prefix=50))
    retained=next(iter(cache.entries.values()))[0]
    check(model,cache,inputs(prefix=65))
    assert retained.get_seq_length()==50
    assert cache.last['reused_tokens']==0 and cache.last['computed_tokens']==65
    assert not cache.last['hit']
    assert not cache.last['full_hit'] and cache.stats['promotions']==1
    check(model,cache,inputs(prefix=65,grid=6))
    assert cache.last['full_hit'] and cache.last['computed_tokens']==0


def test_token_limit_is_not_reported_as_complete_question_hit(model):
    cache=tp.TextPrefixCache(max_tokens=50)
    check(model,cache,inputs());check(model,cache,inputs())
    assert cache.last['hit'] and not cache.last['full_hit']
    assert cache.stats['full_hits']==0 and cache.stats['partial_hits']==1


def test_capacity_eviction_and_training_refusal(model):
    cache=tp.TextPrefixCache(max_entries=1)
    check(model,cache,inputs(prefix=3));check(model,cache,inputs(prefix=2))
    assert len(cache.entries)==1 and cache.stats['evictions']==1
    cache=tp.TextPrefixCache(max_bytes=1);check(model,cache,inputs())
    assert not cache.entries and not cache.last['stored']
    model.train()
    with pytest.raises(ValueError,match='inference-only'):cache.prefill(model,inputs())


def test_reject_padding_and_stale_or_visual_input(model):
    cache=tp.TextPrefixCache();item=inputs();item['attention_mask'][0,0]=0
    with pytest.raises(ValueError,match='unpadded'):cache.prefill(model,item)
    item=inputs();item['position_ids']=torch.zeros(3,1,item['input_ids'].shape[1])
    with pytest.raises(ValueError,match='Unsupported'):cache.prefill(model,item)
    item=inputs();item['input_ids'][0,2]=4
    with pytest.raises(ValueError,match='Visual tokens'):cache.prefill(model,item)


def test_layout_and_hash_contract_all_phases(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
    from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
    from qwen3vl_local.sft_new_loop_phase3 import prompts as p3
    from qwen3vl_local.sft_new_loop_phase12 import prompts as p12
    functions=[p1.phase1_prompt_sha256,p2.event_prompt_sha256,p3.action_prompt_sha256]
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0');old=[f() for f in functions]
    old_fused=p12.identity('4rgb')
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');new=[f() for f in functions]
    assert all(a!=b for a,b in zip(old,new))
    assert p12.identity('4rgb')!=old_fused
    assert [p['type'] for p in tp.content(['a','b'],'question')]==['text','image','image']
    assert p12.messages(['image'],'question')[1]['content'][0]['text']=='question'
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0');assert [f() for f in functions]==old


def test_fused_path_and_native_start_use_persistent_text_cache(model,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase12 import cache as fused
    from qwen3vl_local.sft_v3.train import _kv_start_state
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    item=inputs(prefix=65)
    class Processor:
        def apply_chat_template(self,*a,**kw):return 'fixture'
        def __call__(self,*a,**kw):return item
    bundle=SimpleNamespace(model=model,processor=Processor(),device=torch.device('cpu'))
    with torch.no_grad():
        _kv_start_state(bundle,[]);item=inputs(prefix=65);_kv_start_state(bundle,[])
    assert tp.bank(bundle).stats['hits']==1
    monkeypatch.setattr(fused,'_decode',lambda *args:dict(text='fixture',token_ids=[1],
        generated_tokens=1,stop_token_id=2,reached_token_limit=False))
    result=fused.generate(bundle,[object()],['question1','question2'],check_full=True,verify_outputs=True)
    assert result['full_answer_equivalence_checked']
    assert result['cache_scope']=='process_local_cross_image_text_prefix'
    assert tp.bank(bundle).stats['hits']>=3


def test_two_fixed_questions_both_hit_across_images(model):
    """两项容量须真正保存两题，不退化成只保留共同 system。"""
    cache=tp.TextPrefixCache()
    for i in range(4):
        item=inputs(prefix=65);item['input_ids'][0,50]=150+i%2
        check(model,cache,item)
    assert cache.stats['hits']==2 and cache.stats['misses']==2
    assert sorted(map(len,cache.entries))==[65,65]


def test_all_native_message_builders_keep_targets_and_candidate_text(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase1 import train as t1, eval as e1
    from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
    from qwen3vl_local.sft_new_loop_phase3 import prompts as p3,prompt_contract as pc
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    assert t1._messages(['a'],'Q','YES')[1]['content'][0]['text']=='Q'
    assert t1._messages(['a'],'Q','YES')[-1]['content']=='YES'
    assert e1._messages(['a'],'Q')==t1._generation_messages(['a'],'Q')
    s2=p2.make_prompt_spec(variant='all_random_order',answers={**{k:False for k in p2.ANSWER_KEYS},'DOMAIN_ROAD_CORRIDOR':True},seed_key='fixed')
    msg=p2.build_event_messages(images=['a'],spec=s2,target='NO')
    assert [x['type'] for x in msg[1]['content']]==['text','image']
    assert msg[-1]['content']=='NO'
    s3=p3.make_prompt_spec(variant='all_random_order',answers={k:False for k in p3.ANSWER_KEYS},
        seed_key='fixed',context_id='LEAD_BRAKE',road_structure='R1',goal_xy=(40.,-2.))
    for variant in pc.PROMPT_VARIANTS:
        msg=pc.build_action_messages(images=['a'],spec=s3,prompt_variant=variant,target='KEEP')
        assert msg[1]['content'][0]['text']==pc.build_action_prompt(spec=s3,prompt_variant=variant)
        assert msg[1]['content'][1]==dict(type='image',image='a')
        assert msg[-1]['content']=='KEEP'


def test_ordered_stream_benchmark_uses_native_decode_and_real_misses(model,monkeypatch):
    monkeypatch.setenv("AUTOMOT_TEXT_PREFIX","1")
    class Tokenizer:
        eos_token_id=1
        def convert_tokens_to_ids(self,text):return 1
    class Processor:
        def batch_decode(self,ids,**kw):return [','.join(map(str,row.tolist())) for row in ids]
    bundle=SimpleNamespace(model=model,processor=Processor(),tokenizer=Tokenizer(),device=torch.device('cpu'))
    result=tp.benchmark(bundle,[inputs(prefix=3),inputs(prefix=3,grid=6)],max_new_tokens=4,repeats=1)
    assert result['complete_output_equivalence'] and not result['quality_certified']
    assert result['cold']['cache']['misses']==1 and result['cold']['cache']['hits']==1
    assert result['trials'][0]['cached']['cache']['misses']==1


def test_cache_disabled_retains_new_layout(model,monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','0')
    bundle=SimpleNamespace(model=model);item=inputs()
    with torch.no_grad():out=tp.prefill(bundle,item)
    assert out.logits.shape[1]==item['input_ids'].shape[1]
    assert not hasattr(bundle,'_text_prefix_cache') and tp.content(['a'],'Q')[0]['type']=='text'


def test_autocast_mode_cannot_reuse_different_precision_state(model):
    cache=tp.TextPrefixCache();cache.prefill(model,inputs())
    with torch.autocast('cpu',dtype=torch.bfloat16):cache.prefill(model,inputs())
    cache.prefill(model,inputs())
    assert cache.stats['invalidations']==2 and cache.stats['hits']==0


def test_phase3_view_preserves_index_and_original_manifest(tmp_path,monkeypatch):
    import json
    from qwen3vl_local.qwen35 import text_prefix_data as data
    from qwen3vl_local.sft_new_loop_phase3.prompts import PROMPT_NAME
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')
    parent=tmp_path/'parent';parent.mkdir();index=parent/'frame_index.jsonl'
    raw=b'{"split":"test","answer":false}\n';index.write_bytes(raw)
    manifest=dict(prompt_contract=dict(prompt_name=PROMPT_NAME,production_prompt_sha256=data.hashes()),
                  other_source_contract={'keep':'unchanged'})
    mp=parent/'manifest.json';mp.write_text(json.dumps(manifest));original=mp.read_bytes()
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    view=data.prepare(index,tmp_path/'new')
    assert view.read_bytes()==raw and mp.read_bytes()==original
    saved=json.loads(view.with_name('manifest.json').read_text());data.verify(view,saved)
    assert saved['prompt_contract']['production_prompt_sha256']==data.hashes()
    assert data.prepare(view,tmp_path/'unused')==view and not (tmp_path/'unused').exists()
    saved['other_source_contract']['keep']='changed'
    with pytest.raises(ValueError,match='more than'):data.verify(view,saved)
    saved=json.loads(view.with_name('manifest.json').read_text());view.write_bytes(raw+b'\n')
    with pytest.raises(ValueError,match='changed'):data.verify(view,saved)


def test_new_layout_cpu_ready_rejects_old_mode(tmp_path,monkeypatch):
    import json
    from qwen3vl_local.sft_new_loop_phase12 import integrity,CONTRACT
    from qwen3vl_local.audit_joint.phase12_retrain import launch
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    (tmp_path/'ready.json').write_text(json.dumps(dict(contract=CONTRACT)))
    with pytest.raises(ValueError,match='layout differs'):integrity.verify_ready(tmp_path)
    (tmp_path/'ready.json').write_text(json.dumps(dict(contract=launch.CONTRACT,phase=1,version='source')))
    with pytest.raises(ValueError,match='layout differs'):launch.verify_ready(tmp_path,1,'source')


def test_short_prefix_history_does_not_change_canonical_bf16_state(model):
    model.to(dtype=torch.bfloat16)
    small=inputs(prefix=50);large=inputs(prefix=65)
    for item in (small,large):item['pixel_values']=item['pixel_values'].bfloat16()
    cache=tp.TextPrefixCache();cache.prefill(model,small)
    promoted=cache.prefill(model,large)
    fresh=tp.TextPrefixCache().prefill(model,large)
    assert torch.equal(promoted.logits,fresh.logits)
    for name in ('key_cache','value_cache','conv_states','recurrent_states'):
        for a,b in zip(getattr(promoted.past_key_values,name),getattr(fresh.past_key_values,name)):
            if a is not None:assert torch.equal(a,b)
