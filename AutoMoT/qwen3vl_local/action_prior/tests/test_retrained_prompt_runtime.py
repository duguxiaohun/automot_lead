"""New trained prompt revisions: strict restoration and executable prior rechecks."""
from qwen3vl_local.qwen35 import cache_policy
import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_contracts import fixture_adapter
from qwen3vl_local.action_prior import prompt_versions as versions, priors
from qwen3vl_local.action_prior.contracts import inspect_adapter
from qwen3vl_local.audit_joint.phase12_retrain import snapshot, refinement
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
from qwen3vl_local.qwen35 import text_prefix


def trained_module(phase,revision):
    native=p1 if phase==1 else p2;source=Path(native.__file__).read_text()
    if phase==1:
        source=snapshot.training_prompt(source)
        if revision=='visual':source=refinement.phase1_prompt(source)
    else:source=(refinement.phase2_prompt_v7 if revision=='evidence' else refinement.phase2_prompt)(source)
    name=f'{__name__}.training_{phase}_{revision}'
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(source,name,'exec'),module.__dict__)
    return module,source


def metadata(module,phase,mode='4rgb'):
    return dict(prompt_name=module.PROMPT_NAME,history_rgb_mode=mode,input_layout=text_prefix.identity(),text_cache_policy=cache_policy.current(),
        history_rgb_count=4 if mode=='4rgb' else 2,history_rgb_selected_indices=[0,1,2,3] if mode=='4rgb' else [0,3],
        production_prompt_sha256=(module.phase1_prompt_sha256 if phase==1 else module.event_prompt_sha256)(history_rgb_mode=mode))


@pytest.mark.parametrize('phase,revision',[(1,'legacy'),(1,'visual'),(2,'visual'),(2,'evidence')])
@pytest.mark.parametrize('mode',['4rgb','2rgb_endpoints'])
@pytest.mark.parametrize('layout',['0','1'])
def test_restore_matches_actual_training_source_and_keeps_sha_guard(monkeypatch,phase,revision,mode,layout):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX',layout)
    trained,source=trained_module(phase,revision);meta=metadata(trained,phase,mode)
    recovered=versions.prompt_module(phase,meta)
    assert versions.prompt_source(recovered)==source.encode()
    meta['production_prompt_sha256']='0'*64
    with pytest.raises(ValueError,match='saved prompt hash'):versions.prompt_module(phase,meta)


@pytest.mark.parametrize('phase,revision',[(1,'visual'),(2,'evidence')])
def test_real_adapter_inspection_accepts_retrained_metadata(tmp_path,monkeypatch,phase,revision):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    path=fixture_adapter(tmp_path/'run',phase)
    file=path/f'sft_new_loop_phase{phase}_adapter_config.json';cfg=json.loads(file.read_text())
    trained,_=trained_module(phase,revision);cfg.update(metadata(trained,phase));file.write_text(json.dumps(cfg))
    result=inspect_adapter(path,phase,tmp_path/'base')
    assert result['metadata']['prompt_name']==trained.PROMPT_NAME
    cfg['production_prompt_sha256']='0'*64;file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='saved prompt hash'):inspect_adapter(path,phase,tmp_path/'base')


@pytest.mark.parametrize('mode',['history','independent','compare'])
@pytest.mark.parametrize('problem',[None,'disagreement','malformed','multiple_rs'])
def test_direct8_rechecks_execute_the_trained_surface_and_gate_errors(monkeypatch,mode,problem):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    one,_=trained_module(1,'visual');two,_=trained_module(2,'evidence')
    one=versions.prompt_module(1,metadata(one,1));two=versions.prompt_module(2,metadata(two,2))
    orders=[]
    def ask(phase,spec,history):
        module=one if phase==1 else two
        prompt=(module.build_phase1_prompt if phase==1 else module.build_event_prompt)(spec=spec,history_rgb_mode='4rgb')
        assert module.PROMPT_NAME in prompt
        if phase==1:
            assert spec.variant=='all_random_order' and set(spec.output_keys)==set(one.ANSWER_KEYS)
            assert not {'GROUP','DETAIL','RS_HIGHWAY'} & set(spec.output_keys)
            orders.append(tuple(spec.output_keys))
        values={k:'YES' if k in ('RS1','HIGHWAY') and phase==1 else 'NO' for k in spec.output_keys}
        if phase==1 and problem=='multiple_rs':values['RS2']='YES'
        if phase==1 and len(orders)>1 and problem=='disagreement':values['RS1']='NO'
        text='\n'.join(f'{k}: {values[k]}' for k in spec.output_keys)
        if phase==1 and len(orders)>1 and problem=='malformed':text+='\nextra'
        return text,prompt
    result=priors.collect_priors(ask,'frame',recheck_mode=mode,phase1_module=one,event_module=two)
    assert orders[0]!=orders[1]
    assert result['phase1_recheck_protocol']=='direct8_reordered_consistency_v1'
    assert result['conditions']['RS_HIGHWAY'] is None and result['unavailable_fields']==['RS_HIGHWAY']
    assert result['conditions']['ROAD_STRUCTURE']==('R1' if problem is None else None)
    if problem=='multiple_rs':assert result['invalid']['RS1']=='multiple_rs_yes'
    assert len([c for c in result['calls'] if c['phase']==2])==(6 if mode=='compare' else 4)
    assert len(orders)==(3 if mode=='compare' else 2)


def test_runtime_uses_restored_modules_for_both_spec_and_render(monkeypatch):
    from qwen3vl_local.action_prior.runtime import PriorEngine
    from qwen3vl_local.qwen35.question_schedule import QuestionSchedule
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    one,_=trained_module(1,'visual');two,_=trained_module(2,'evidence')
    runtime=PriorEngine.__new__(PriorEngine)
    runtime.contract=dict(identity='fixture',phase1=dict(metadata=metadata(one,1)),phase2=dict(metadata=metadata(two,2)))
    runtime.rgb_frame_count=4;runtime.labels=None;runtime.online_text_prefix=True
    runtime._online_question_schedule=QuestionSchedule();runtime._phase_prefix_bundles={}
    runtime._prior_cache_policies={p:cache_policy.serialize(cache_policy.current()) for p in ('phase1','phase2')}
    runtime.high_level_action_prior=False;runtime.generate_analysis=False;runtime.analysis_review=False
    runtime.recheck_mode='history';runtime.text_cache=None
    runtime.mode=lambda name:contextlib.nullcontext()
    # Exercise the real collect_priors + runtime ask/render. Model answers are
    # deliberate synthetic NOs; this test is about the exact invocation contract.
    requested=[]
    def generate(system,prompt,images,history):
        requested.append(prompt)
        module=one if system==one.SYSTEM_PROMPT else two
        assert module.PROMPT_NAME in prompt
        # Strict output lines in the rendered OUTPUT_FORMAT block.
        import re
        keys=re.findall(r'^([A-Z][A-Z0-9_]*): <YES or NO>$',prompt,re.M)
        assert keys
        return '\n'.join(f'{key}: NO' for key in keys),None
    runtime.generate_messages=generate
    runtime.engine=SimpleNamespace(
        build_messages=lambda system,text,images:[dict(role='system',content=system),dict(role='user',content=text)],
        processor=SimpleNamespace(apply_chat_template=lambda *a,**k:'base fixture'),
        prepare_inputs=lambda *a:dict(input_ids=torch.ones(1,3,dtype=torch.long)),
        prefill=lambda *a:SimpleNamespace(past_key_values='base',rope_deltas=torch.tensor([[0]])))
    runtime.condition([0]*4,'continue','sample')
    assert len(requested)==6
    assert runtime.last_audit['phase1_recheck_protocol']=='direct8_reordered_consistency_v1'
    assert runtime.last_audit['unavailable_fields']==['RS_HIGHWAY']


@pytest.mark.parametrize('phase,revision',[(1,'visual'),(2,'evidence')])
def test_online_phase12_uses_recovered_training_renderer(monkeypatch,phase,revision):
    from qwen3vl_local.qwen35 import online
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    trained,_=trained_module(phase,revision);meta=metadata(trained,phase)
    b=SimpleNamespace(model=SimpleNamespace(eval=lambda:None));online.bind_prompt_contract(b,phase,meta)
    seen=[]
    monkeypatch.setattr(online,'_kv_start_state',lambda bundle,messages:seen.append(messages))
    monkeypatch.setattr(online,'_student_generate_kv',lambda *a:('',None,torch.tensor([],dtype=torch.long)))
    kwargs=dict(domain=trained.ROAD_DOMAIN) if phase==2 else {}
    online.predict_phase(b,phase,[0]*4,seed='fixture',schedule=False,**kwargs)
    answers={trained.DOMAIN_ANSWER_KEYS[trained.ROAD_DOMAIN]:True} if phase==2 else {}
    spec=trained.make_prompt_spec(variant='all_random_order',answers=answers,seed_key='fixture')
    if phase==2:expected=trained.build_event_messages(images=[0]*4,spec=spec,history_rgb_mode='4rgb')
    else:expected=[dict(role='system',content=trained.SYSTEM_PROMPT),dict(role='user',content=text_prefix.content(
        [0]*4,trained.build_phase1_prompt(spec=spec,history_rgb_mode='4rgb')))]
    assert seen==[expected]


def test_new_direct8_protocol_is_distinct_from_legacy(monkeypatch):
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1');trained,_=trained_module(1,'visual')
    assert priors.protocol_for_phase1(versions.prompt_module(1,metadata(trained,1)))==priors.DIRECT8_PROTOCOL_VERSION
    assert priors.protocol_for_phase1(p1)==priors.PROTOCOL_VERSION
