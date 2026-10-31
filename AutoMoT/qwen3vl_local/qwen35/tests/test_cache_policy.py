"""Deployment settings survive artifact copying; drift rejected before inference."""
import copy
import importlib
import json
from types import SimpleNamespace
import pytest
import torch
from qwen3vl_local.qwen35 import cache_policy as policy,text_prefix as tp,online
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.qwen35.tests.test_online_benchmark import setup
from qwen3vl_local.qwen35.tests.test_text_prefix import inputs


@pytest.mark.parametrize('phase',[1,2,3,12])
@pytest.mark.parametrize('setting,value',[('ENTRIES','1'),('MIB','8'),('TOKENS','64'),('','0')])
@pytest.mark.parametrize('warm',[False,True])
def test_official_online_rejects_environment_drift_before_scheduling(model,tmp_path,monkeypatch,phase,setting,value,warm):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    b,rows,_=setup(model,tmp_path,phase)
    if warm:tp.prefill(b,inputs(prefix=80))
    calls=[]
    monkeypatch.setattr(model,'forward',lambda **kw:calls.append(kw) or pytest.fail('Model executed after drift'))
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE'+('_'+setting if setting else ''),value)
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import infer
        invoke=lambda:infer.predict(b,[None]*2,'2rgb_endpoints',rows[0]['context']['domain'])
    else:invoke=lambda:online.predict_phase(b,phase,[None]*2,**rows[0]['context'])
    with pytest.raises(ValueError,match='deployment policy'):invoke()
    assert not calls and not getattr(b,'_question_schedules',{})


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_missing_policy_and_modified_bank_rejected(model,tmp_path,monkeypatch,phase):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');b,rows,cfg=setup(model,tmp_path,phase)
    missing=copy.deepcopy(cfg);missing.pop(policy.FIELD)
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import infer
        bind=lambda:infer.bind_online_contract(SimpleNamespace(model=model),missing)
        bound=lambda:infer.bound_online_contract(b)
    else:
        bind=lambda:online.bind_prompt_contract(SimpleNamespace(model=model),phase,missing)
        bound=lambda:online.bound_contract(b,phase)
    with pytest.raises(ValueError,match='lacks text_cache_policy'):bind()
    tp.bank(b).max_tokens=64
    with pytest.raises(ValueError,match='Resident cache limits'):bound()


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_native_writers_embed_startup_policy_without_ready_dependency(tmp_path,monkeypatch,phase):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_MIB','768')
    from qwen3vl_local.qwen35 import adapters
    monkeypatch.setattr(adapters,'save_adapter',lambda *a,**kw:None)
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    monkeypatch.setattr(proof,'stamp_adapter',lambda *a:None)
    b=SimpleNamespace(model=torch.nn.Linear(2,2),lora_target_modules=['fixture']);b.unwrap=lambda:b.model
    policy.start_training(b);expected=policy.current()
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import train,cache_checks
        monkeypatch.setattr(train,'save_adapter',lambda *a,**kw:None)
        (tmp_path/'ready.json').write_text(json.dumps(dict(text_cache_policy=policy.current(),text_cache_limits=tp.cache_limits(),cache_acceptance_protocol=cache_checks.protocol())))
        train.save(b,tmp_path/'final',tmp_path,'4rgb',1)
        file=tmp_path/'final/fusion_config.json'
    else:
        train=importlib.import_module(f'qwen3vl_local.sft_new_loop_phase{phase}.train')
        monkeypatch.setattr('sys.argv',['train','--device','cpu'])
        args=train.parse_args()
        train._save_adapter(b,tmp_path,args,step=1)
        file=tmp_path/f'final/sft_new_loop_phase{phase}_adapter_config.json'
    cfg=json.loads(file.read_text());assert cfg[policy.FIELD]==expected
    portable=tmp_path/'copied';portable.mkdir()
    (portable/file.name).write_text(file.read_text())
    assert policy.adapter_policy(portable)==(True,expected)  # no ready, training path or model present
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_TOKENS','64')
    with pytest.raises(ValueError,match='deployment policy'):policy.adapter_policy(portable)
    with pytest.raises(ValueError,match='deployment policy'):policy.saved_fields(b)


def test_direct_native_prefill_and_actual_bank_cannot_bypass_loaded_policy(model,monkeypatch):
    model.to(dtype=torch.bfloat16)
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');policy.bind(model,policy.current())
    b=SimpleNamespace(model=model);tp.prefill(b,inputs())
    tp.bank(b).max_entries=1
    with pytest.raises(ValueError,match='Resident cache limits'):tp.prefill(b,inputs())
    with pytest.raises(ValueError,match='Resident cache limits'):tp.TextPrefixCache(max_tokens=64).prefill(model,inputs())
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','0')
    with pytest.raises(ValueError,match='deployment policy'):tp.prefill(b,inputs())


def test_prior_engine_rejects_policy_drift_before_render_or_cache_reuse(monkeypatch):
    from qwen3vl_local.action_prior.runtime import PriorEngine
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');engine=PriorEngine.__new__(PriorEngine)
    engine.online_text_prefix=True;engine._phase_prefix_bundles={}
    engine.contract={p:{'metadata':{policy.FIELD:policy.current()}} for p in ('phase1','phase2')}
    engine._prior_cache_policies={p:policy.serialize(policy.current()) for p in ('phase1','phase2')}
    engine._check_cache_policy()
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES','1')
    with pytest.raises(ValueError,match='deployment policy'):engine.generate_messages('system','question',[])


def test_legacy_metadata_never_infers_new_layout_policy(monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0');assert policy.from_metadata({}) is None
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    with pytest.raises(ValueError,match='input layout'):policy.from_metadata({})
    with pytest.raises(ValueError,match='lacks text_cache_policy'):policy.from_metadata({'input_layout':tp.identity()})


@pytest.mark.parametrize('phase',[1,2,3,12])
def test_online_load_rejects_drift_before_allocating_weights(model,tmp_path,monkeypatch,phase):
    from qwen3vl_local.qwen35 import backend,adapters,identity
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');_,_,cfg=setup(model,tmp_path,phase)
    monkeypatch.setattr(backend.LocalModel,'from_pretrained',lambda *a,**kw:pytest.fail('Allocated weights before policy check'))
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_TOKENS','64')
    if phase==12:
        from qwen3vl_local.sft_new_loop_phase12 import infer
        (tmp_path/'fusion_config.json').write_text(json.dumps(cfg))
        monkeypatch.setattr(infer.proof,'verify_adapter',lambda *a:None)
        invoke=lambda:infer.load_online_bundle(tmp_path/'base',tmp_path,device='cpu')
    else:
        from qwen3vl_local.action_prior import contracts
        from qwen3vl_local.sft_new_loop_phase3 import eval as p3
        monkeypatch.setattr(contracts,'inspect_adapter',lambda *a:{'metadata':cfg})
        monkeypatch.setattr(p3,'_validate_action_adapter',lambda *a:cfg)
        monkeypatch.setattr(adapters,'validate_adapter',lambda *a,**kw:None)
        monkeypatch.setattr(identity,'base_asset_hashes',lambda *a:{})
        invoke=lambda:online.load_online_bundle(tmp_path/'base',tmp_path,phase,device='cpu')
    with pytest.raises(ValueError,match='deployment policy'):invoke()


def test_real_native_lora_load_binds_policy_on_model(model,tmp_path,monkeypatch):
    from peft import LoraConfig,get_peft_model,TaskType
    from qwen3vl_local.qwen35.backend import LocalModel
    from qwen3vl_local.qwen35.adapters import save_adapter,LocalPeftModel
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    model.save_pretrained(tmp_path/'base');base=LocalModel.from_pretrained(tmp_path/'base',dtype=torch.bfloat16)
    adapted=get_peft_model(base,LoraConfig(r=2,target_modules=['q_proj'],task_type=TaskType.CAUSAL_LM))
    target=tmp_path/'adapter';save_adapter(adapted,target)
    (target/'sft_new_loop_phase1_adapter_config.json').write_text(json.dumps(
        dict(input_layout=tp.identity(),text_cache_policy=policy.current())))
    loaded=LocalPeftModel.from_pretrained(adapted.unload(),target).eval()
    assert getattr(loaded.get_base_model(),policy.ATTR)==policy.serialize(policy.current())
    bundle=SimpleNamespace(model=loaded)
    tp.prefill(bundle,inputs());tp.prefill(bundle,inputs())
    assert tp.bank(bundle).stats['full_hits']==1
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_TOKENS','64')
    with pytest.raises(ValueError,match='deployment policy'):tp.prefill(bundle,inputs())
