"""Rendering/contract tests; fake generation is not evidence of 4B model quality."""
import copy
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.qwen35 import online, text_prefix
from qwen3vl_local.sft_new_loop_phase3 import prompts as p3, prompt_contract as pc
from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME


def metadata(mode='4rgb', output='choice', variant=CANDIDATE_NAME):
    return dict(prompt_variant=variant,prompt_name=pc.prompt_name(variant),action_output_mode=output,
        history_rgb_mode=mode,history_rgb_count=4 if mode=='4rgb' else 2,
        history_rgb_selected_indices=[0,1,2,3] if mode=='4rgb' else [0,3],input_layout=text_prefix.identity(),
        production_prompt_sha256=pc.action_prompt_sha256(history_rgb_mode=mode,action_output_mode=output,prompt_variant=variant))


def make_bundle():return SimpleNamespace(model=SimpleNamespace(eval=lambda:None))


@pytest.mark.parametrize('mode',['4rgb','2rgb_endpoints'])
@pytest.mark.parametrize('output',['choice','binary'])
@pytest.mark.parametrize('variant',[CANDIDATE_NAME,'baseline'])
def test_actual_online_messages_match_native_training_contract(monkeypatch,mode,output,variant):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    b=make_bundle();meta=metadata(mode,output,variant);online.bind_prompt_contract(b,3,meta)
    seen=[]
    monkeypatch.setattr(online,'_kv_start_state',lambda bundle,messages:seen.append(messages))
    context=next(iter(p3.CONTEXT_BY_ID));road=next(iter(p3.ROAD_STRUCTURE_TEXT))
    spec=p3.make_prompt_spec(variant='all_random_order',answers={k:False for k in (*p3.ACTION_KEYS,p3.INVALID_KEY)},
        seed_key='fixed',context_id=context,road_structure=road,action_output_mode=output,
        current_speed_mps=5,goal_xy=(10,1))
    target=p3.build_action_target(spec)
    monkeypatch.setattr(online,'_student_generate_kv',lambda *a:(target,None,torch.tensor([1])))
    images=list(range(meta['history_rgb_count']))
    result=online.predict_phase(b,3,images,context_id=context,road_structure=road,current_speed_mps=5,
                                goal_xy=(10,1),seed='fixed',schedule=False)
    expected=pc.build_action_messages(images=images,spec=spec,history_rgb_mode=mode,prompt_variant=variant)
    assert seen==[expected]
    assert result['answers']==p3.parse_action_output(target,spec=spec)
    assert result['prompt_contract']['action_output_mode']==output
    if variant==CANDIDATE_NAME:
        assert expected!=p3.build_action_messages(images=images,spec=spec,history_rgb_mode=mode)
    # Caller mutation cannot change the binding after validation.
    meta['prompt_variant']='wrong'
    assert b._online_prompt_contract.prompt_variant==variant


@pytest.mark.parametrize('override',[dict(prompt_variant='baseline'),dict(action_output_mode='binary'),dict(history_rgb_mode='2rgb_endpoints')])
def test_override_conflicts_rejected_before_model_execution(monkeypatch,override):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    b=make_bundle();online.bind_prompt_contract(b,3,metadata())
    with pytest.raises(ValueError,match='training contract'):online.predict_phase(b,3,[0]*4,**override)


@pytest.mark.parametrize('field,value',[('production_prompt_sha256','0'*64),('prompt_name','old'),
    ('prompt_variant','baseline'),('input_layout','images_before_text_legacy'),('history_rgb_count',2),
    ('history_rgb_selected_indices',[0,3]),('action_output_mode','binary')])
def test_metadata_tampering_is_not_silently_normalized(monkeypatch,field,value):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');meta=metadata();meta[field]=value
    with pytest.raises(ValueError):online.bind_prompt_contract(make_bundle(),3,meta)


def test_missing_binding_replaced_model_and_disabled_adapter_rejected(monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b=make_bundle()
    with pytest.raises(ValueError,match='Bind'):online.predict_phase(b,3,[0]*4)
    online.bind_prompt_contract(b,3,metadata());b.model=SimpleNamespace(eval=lambda:None)
    with pytest.raises(ValueError,match='model/active adapter'):online.predict_phase(b,3,[0]*4)
    class Adapter(torch.nn.Module):
        active_adapters=['phase3'];disable_adapters=False
    b.model=Adapter();online.bind_prompt_contract(b,3,metadata());b.model.disable_adapters=True
    with pytest.raises(ValueError,match='model/active adapter'):online.predict_phase(b,3,[0]*4)


def test_factory_loads_and_binds_the_validated_native_config(tmp_path,monkeypatch):
    from qwen3vl_local.qwen35 import backend,adapters,identity
    from qwen3vl_local.sft_new_loop_phase3 import eval as native_eval
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');meta=metadata();calls=[]
    model=SimpleNamespace(to=lambda device:model,eval=lambda:model)
    def validate(path,**kwargs):calls.append(('validate',path));return path
    monkeypatch.setattr(adapters,'validate_adapter',validate)
    monkeypatch.setattr(identity,'base_asset_hashes',lambda path:{'fixture':'only'})
    monkeypatch.setattr(native_eval,'_validate_action_adapter',lambda *a:copy.deepcopy(meta))
    monkeypatch.setattr(backend.LocalModel,'from_pretrained',lambda *a,**k:model)
    monkeypatch.setattr(adapters.LocalPeftModel,'from_pretrained',lambda base,path,**kw:calls.append(('load',path)) or model)
    monkeypatch.setattr(backend.LocalProcessor,'from_pretrained',lambda *a:SimpleNamespace(tokenizer='fixture'))
    b=online.load_online_bundle(tmp_path/'base',tmp_path/'adapter',3,device='cpu')
    assert str(calls[0][1])==calls[1][1]==str(tmp_path/'adapter')
    assert b._online_prompt_contract.prompt_variant==CANDIDATE_NAME
    assert b._online_prompt_contract.action_output_mode=='choice'
