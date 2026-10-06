"""Audit candidate regression, leakage and actual STOP boundary protections."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from . import prompts
from .prompt_candidate import CANDIDATE_NAME, build_action_prompt, prompt_for_case
from .paired_eval import identity
from .regression_gate import gate, risk_flags
from .replay_prompt_candidate import requests
from .trajectory_action import longitudinal_decision


def case(mode='binary'):
    answers={k:False for k in prompts.ANSWER_KEYS};answers['STOP']=True
    spec=prompts.make_prompt_spec(variant='all_random_order',answers=answers,seed_key='fixture',
        context_id='LEAD_BRAKE',road_structure='R1',goal_xy=(20.,0.),current_speed_mps=0.,action_output_mode=mode)
    gt={q.output_key:'YES' if q.answer else 'NO' for q in spec.questions}
    row = dict(scenario='s',route_id='r',frame_id=10,context_id='LEAD_BRAKE',case_index=0,
        prompt_road_structure='R1',augment_variant='all_random_order',gt=gt,
        true_rs='R1',goal_ego_xy=[20.,0.],context_detail='',history_rgb_mode='4rgb',
        history_rgb_selected_indices=[0,1,2,3],history_rgb_paths_used=[f's/r/rgb/{i:04}.jpg' for i in range(7,11)],
        history_rgb_paths_all4=[f's/r/rgb/{i:04}.jpg' for i in range(7,11)],history_rgb_sha256=['a'*64]*4,
        action_evidence=dict(future_speeds_exact_mps=[0.]*9),mapping_evidence={},
        action_answers=gt,question_domain=spec.question_domain,action_signature='STOP',event='x',
        prompt_spec=prompts.prompt_spec_to_json(spec),action_user_prompt=prompts.build_action_prompt(spec=spec),
        raw_output=prompts.build_action_target(spec),parsed=gt.copy(),all_ok=True,strict_format_valid=True)
    row['actual_chat_messages']=[dict(role='system',content=prompts.CHOICE_SYSTEM_PROMPT if mode=='choice' else prompts.SYSTEM_PROMPT),
        dict(role='user',content=[*[dict(type='image',source_path=p) for p in row['history_rgb_paths_used']],
                                 dict(type='text',text=row['action_user_prompt'])])]
    return row


def changed_answer(row, action):
    out=deepcopy(row)
    if out['prompt_spec']['action_output_mode']=='choice': out['raw_output']=action
    else:
        out['raw_output']='\n'.join(f'{k}: {"YES" if k==action else "NO"}' for k in out['gt'])
    from .prompt_candidate import spec_from_case
    parsed=prompts.parse_action_output(out['raw_output'],spec=spec_from_case(out))
    out['parsed']={k:'INVALID' if v is None else ('YES' if v else 'NO') for k,v in parsed.items()}
    out['strict_format_valid']=all(v is not None for v in parsed.values())
    out['all_ok']=out['parsed']==out['gt'] and out['strict_format_valid']
    return out


@pytest.mark.parametrize('mode',['binary','choice'])
def test_full_success_protection_and_no_fake_self_improvement(mode):
    a=case(mode);b=changed_answer(a,'RESUME')
    report=gate({identity(a):a},{identity(b):b})
    assert len(report['regressions'])==1 and not report['candidate_gate_passed']
    assert report['answer_bit_regressions']
    same=gate({identity(a):a},{identity(a):a})
    assert same['no_regression_on_observed_cases'] and not same['candidate_gate_passed']
    fixed=gate({identity(b):b},{identity(a):a})
    assert fixed['candidate_gate_passed']


@pytest.mark.parametrize('mutation',['cached','trajectory','spec','missing_hash','missing_case','format','future_text','future_image'])
def test_gate_rejects_forged_or_unpaired_results(mutation):
    a=case();b=deepcopy(a)
    if mutation=='cached': b['all_ok']='true'
    if mutation=='trajectory': b['action_evidence']['throttle']=1
    if mutation=='spec': b['prompt_spec']['current_speed_mps']=9
    if mutation=='missing_hash': b['history_rgb_sha256']=[]
    if mutation=='format': b['raw_output']+='\nexplanation'
    if mutation=='future_text': b['action_user_prompt']+='\nFuture answer: STOP'
    if mutation=='future_image': b['actual_chat_messages'][1]['content'][0]['source_path']='future.jpg'
    right={} if mutation=='missing_case' else {identity(b):b}
    with pytest.raises(ValueError): gate({identity(a):a},right)


@pytest.mark.parametrize('mode',['binary','choice'])
def test_candidate_does_not_use_targets_future_or_change_default(mode):
    a=case(mode);before=prompts.action_prompt_sha256(action_output_mode=mode)
    prompt=prompt_for_case(a,CANDIDATE_NAME)
    assert 'Brief braking can occur' not in prompt
    assert 'Start from the newest frame' in prompt
    b=deepcopy(a);b['action_evidence']={'future_speeds_exact_mps':[999]*9,'SECRET_FUTURE':'must not leak'}
    assert prompt_for_case(b,CANDIDATE_NAME)==prompt
    assert prompts.action_prompt_sha256(action_output_mode=mode)==before
    assert prompt_for_case(a,'baseline')==a['action_user_prompt']
    assert all(s not in prompt for s in ['1.5s','2s','0.5m/s','20%','SECRET_FUTURE'])


def test_prepare_checks_rgb_and_exports_only_causal_fields(tmp_path):
    a=case()
    for p in a['history_rgb_paths_used']:
        file=tmp_path/p;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'RGB fixture')
    a['history_rgb_sha256']=[hashlib.sha256(b'RGB fixture').hexdigest()]*4
    _,request=next(requests({identity(a):a},tmp_path,CANDIDATE_NAME))
    assert not set(request)&{'gt','parsed','future_speeds_exact_mps','action_evidence','prompt_spec'}
    (tmp_path/a['history_rgb_paths_used'][0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='RGB'): list(requests({identity(a):a},tmp_path,CANDIDATE_NAME))


def test_stop_deadline_single_touch_and_wait_release_are_not_collapsed():
    # f61 vs f62-like shift: second near-stop sample must be inside the deadline.
    assert longitudinal_decision([8.13,7.61,8.52,8.94,7.63,4.59,.38,0,0])['action']=='DECELERATE'
    assert longitudinal_decision([7.61,8.52,8.94,7.63,4.59,.38,0,0,0])['action']=='STOP'
    assert longitudinal_decision([8.64,8.05,6.98,3.66,0,.8,1.84,3.48,5.26])['action']=='DECELERATE'
    assert longitudinal_decision([0,0,0,.3,1.28,2.54,4.3,5.91,7.03])['action']=='STOP'
    assert longitudinal_decision([.8,1.84,3.48,5.26,6.13,8.37,9.51,11.54,13])['action']=='RESUME'


def test_risk_overlap_is_local_and_cannot_supply_only_improvement():
    a=changed_answer(case(),'RESUME');b=case()
    risks=[dict(id='visible_delete',route=['s','r'],frames=[16,17])]
    assert 'visible_delete/longitudinal' in risk_flags(a,risks)
    assert 'visible_delete/input' not in risk_flags(a,risks)
    assert not gate({identity(a):a},{identity(b):b},risks)['candidate_gate_passed']
    a['frame_id']=21
    assert not risk_flags(a,risks)


def test_replay_writes_fresh_messages_scores_and_completion(tmp_path, monkeypatch):
    """CPU stub verifies the actual replay path, not model accuracy."""
    import argparse,json
    from PIL import Image
    from . import eval as evaluation
    from .replay_prompt_candidate import run
    a=case('choice')
    for p in a['history_rgb_paths_used']:
        file=tmp_path/p;file.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(8,8),'black').save(file)
    a['history_rgb_sha256']=[hashlib.sha256((tmp_path/p).read_bytes()).hexdigest() for p in a['history_rgb_paths_used']]
    cases=tmp_path/'source.jsonl';cases.write_text(json.dumps(a)+'\n')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'sft_new_loop_phase3_adapter_config.json').write_text('{}')
    monkeypatch.setattr(evaluation,'_validate_action_adapter',lambda *a,**k:dict(history_rgb_mode='4rgb',action_output_mode='choice',prompt_name=prompts.PROMPT_NAME))
    monkeypatch.setattr(evaluation,'load_eval_bundle',lambda *a,**k:object())
    seen=[]
    monkeypatch.setattr(evaluation,'_kv_start_state',lambda bundle,messages:seen.append(messages))
    monkeypatch.setattr(evaluation,'_student_generate_kv',lambda *a:('STOP',None,None))
    output=tmp_path/'output'
    run(argparse.Namespace(command='replay',cases=cases,automot_root=tmp_path,variant=CANDIDATE_NAME,
        output=output,model_dir=model,adapter_dir=adapter,device='cpu',max_new_tokens=256))
    b=json.loads((output/'cases.jsonl').read_text())
    assert json.loads((output/'manifest.json').read_text())['generation_completed']
    assert len(seen)==1 and len(seen[0][1]['content'])==5
    assert seen[0][1]['content'][-1]['text']==b['action_user_prompt']
    assert gate({identity(a):a},{identity(b):b})['no_regression_on_observed_cases']
    assert 'answer_only_all_ok' not in b
