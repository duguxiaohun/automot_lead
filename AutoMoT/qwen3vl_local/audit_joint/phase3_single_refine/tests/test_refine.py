import copy
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.audit_joint.phase3_single_v3.tests.test_single import record
from qwen3vl_local.audit_joint.phase3_single_v3 import objective as original
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as old_rule
from .. import objective,prompt,regression,experiment
from ..audit import verify_snapshot,sha
from ..frozen import trajectory_action as rule,targets


def r():
    return record(variant='six')

def rows(record,ids):return [dict(id=record['id'],token_ids=ids)]

def test_snapshot_and_exact_original_prompt():
    assert len(verify_snapshot()['received_binding']['code'])==162
    a=r()
    assert objective.messages(a,'RGB')==original.messages(a,'RGB')
    assert objective.observation_binding(a)==original.observation_binding(a)

@pytest.mark.parametrize('variant',prompt.VARIANTS)
def test_no_future_evidence_or_target_leaks(variant):
    a=r();b=copy.deepcopy(a);b.update(targets='SECRET',six_audit='SECRET',sampling_mass='SECRET',split='test')
    assert objective.messages(a,'RGB',prompt_variant=variant)==objective.messages(b,'RGB',prompt_variant=variant)

def test_base_changes_do_not_change_snapshot_rule_or_text(monkeypatch):
    monkeypatch.setattr(original,'instruction',lambda *_:'CHANGED BASE')
    monkeypatch.setattr(old_rule,'longitudinal_decision',lambda *_args,**_kw:dict(action='BROKEN'))
    assert 'CHANGED BASE' not in objective.instruction('six')
    assert rule.longitudinal_decision([0]*9)['action']=='STOP'

def test_clarification_matches_actual_history_compression():
    points=[dict(action='STOP'),dict(action='RESUME')]
    assert targets.compress_history(points,0)=='RESUME'
    assert 'LATER origin' in prompt.instruction('clarified')
    assert 'lane direction uses ego heading at T' not in prompt.instruction('clarified')
    assert 'later origin at 0s' in objective.messages(r(),'RGB',prompt_variant='clarified')[1]['content'][1]['text']

@pytest.mark.parametrize('brake,throttle',[(None,None),(True,0),(False,.5)])
def test_500_random_windows_equal_frozen_source(brake,throttle):
    gen=random.Random(314)
    for _ in range(500):
        speeds=[max(0,gen.gauss(5,5)) for _ in range(9)]
        assert rule.longitudinal_decision(speeds,brake=brake,throttle=throttle)==old_rule.longitudinal_decision(speeds,brake=brake,throttle=throttle)

@pytest.mark.parametrize('speeds,expected',[
    ([8,8,8,8,8,8,0,0,0],'DECELERATE'), # second low sample outside immediate window
    ([8,8,8,8,8,0,0,0,0],'STOP'),
    ([8,8,8,8,8,8,8,8,6],'DECELERATE'),
    ([8,8,8,8,8,8,8,8,10],'NONE'),
    ([8,8,8,8,8,8,8,10,10],'RESUME'),
    ([0]*9,'STOP'),
])
def test_boundaries(speeds,expected):assert rule.longitudinal_decision(speeds)['action']==expected

def test_released_control_distinguishes_wait_from_immediate_pullaway():
    s=[0,.1,.3,.7,1.3,1.5,1.8,2,2.5]
    assert rule.longitudinal_decision(s,brake=True,throttle=0)['action']=='STOP'
    assert rule.longitudinal_decision(s,brake=False,throttle=.5)['action']=='RESUME'

def test_net_gain_cannot_hide_old_correct_losses():
    a=r();base=rows(a,[5,0,0,0]);new=rows(a,[0,5,5,5]);result=regression.compare([a],base,new)
    assert len(result['lost_correct'])==1 and len(result['repaired'])==3
    assert result['status']=='tradeoff_review_required'

def test_self_control_has_no_claim_of_training_readiness():
    a=r();p=rows(a,[5]*4);q=regression.compare([a],p,p)
    assert q['status']=='no_detected_regression' and not q['automatic_training_admission']

@pytest.mark.parametrize('ids',[[5,5,5],[5,5,5,6],[5,5,5,True]])
def test_reject_malformed_predictions(ids):
    a=r()
    with pytest.raises(ValueError):regression.compare([a],rows(a,[5]*4),rows(a,ids))

def test_reject_missing_duplicate_and_outside_domain():
    a=r();p=rows(a,[5]*4)
    for bad in ([],p+p,rows(a,[3]*4)):
        with pytest.raises(ValueError):regression.compare([a],p,bad)

@pytest.mark.parametrize('variant',prompt.VARIANTS)
def test_real_qwen_pack_backward_and_clean_replay(processor,model,variant):
    b=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu');a=r();im=Image.new('RGB',(32,32))
    n=len(b.tokenizer);packed=objective.pack(b,a,im,prompt_variant=variant)
    assert len(packed['positions'])==4 and len(packed['inputs']['image_grid_thw'])==1
    loss=objective.loss_from_logits(objective.readout(b,packed),a['targets'],packed['choices']);loss.backward()
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()
    p1=objective.predict(b,a,im,prompt_variant=variant)
    objective.predict(b,a,im,prompt_variant='clarified' if variant=='original' else 'original')
    p2=objective.predict(b,a,im,prompt_variant=variant)
    assert p1['probabilities']==p2['probabilities'] and len(b.tokenizer)==n
    assert all(m.rope_deltas is None for m in model.modules() if hasattr(m,'rope_deltas'))

def test_context_limit_never_truncates(processor,model):
    b=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    model.config.text_config.max_position_embeddings=32
    with pytest.raises(ValueError,match='input_budget'):objective.pack(b,r(),Image.new('RGB',(32,32)),prompt_variant='clarified')

def test_six_way_orchestration_keeps_targets_and_image_controls(tmp_path,monkeypatch):
    a=r();a['image']='rgb.jpg';Image.new('RGB',(32,32),'white').save(tmp_path/a['image']);seen=[]
    def predict(bundle,rec,image,*,prompt_variant):
        seen.append((prompt_variant,rec['observation']['history'][-1]['valid'],image.getpixel((0,0))))
        assert rec['targets']==a['targets'];return dict(token_ids=[5]*4)
    monkeypatch.setattr(objective,'predict',predict)
    results=experiment.paired_run(None,[a],[dict(id=a['id'],image=a['image'],sha256=sha(tmp_path/a['image']))],tmp_path,io.StringIO())
    assert len(results)==6 and len(seen)==6
    assert [x[1] for x in seen]==[True,True,False,False,True,True]
    assert seen[-1][2]==(0,0,0) and a['observation']['history'][-1]['valid']

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_single_v3 import dataset
    from ..audit import write
    a=r();a.update(split='val',image='rgb.jpg');test=copy.deepcopy(a);test.update(split='test',id='test-only')
    ds=tmp_path/'dataset.json';write(ds,dict(records=[a,test]))
    data=tmp_path/'data';data.mkdir();Image.new('RGB',(32,32),'white').save(data/'rgb.jpg')
    run=tmp_path/'run';(run/'epoch_3').mkdir(parents=True);(run/'epoch_3/state.pt').write_bytes(b'synthetic-weight')
    binding=copy.deepcopy(verify_snapshot()['received_binding']);binding.update(dataset_sha256=sha(ds),fixed_validation_ids=[a['id']])
    write(run/'epoch_3/receipt.json',dict(binding=binding,step=9600,status='epoch_complete',state_sha256=sha(run/'epoch_3/state.pt')))
    write(run/'training_complete.json',dict(binding=binding,selected_checkpoint='epoch_3',total_steps=9600,selection='final_completed_epoch',checkpoint_receipts={'epoch_3':sha(run/'epoch_3/receipt.json')}))
    ready=tmp_path/'ready.json';write(ready,dict(status='ready',dataset=str(ds),dataset_sha256=sha(ds)))
    # This fixture tests orchestration, not native full-pool eligibility or GPU loading.
    monkeypatch.setattr(dataset,'load',lambda *_a,**_k:{})
    plan=tmp_path/'plan';experiment.prepare(ready,run,data,plan)
    return plan,data,run,ready

def test_prepare_preserves_entire_val_and_excludes_test(prepared):
    plan,data,run,_=prepared;m,records,images=experiment.load_plan(plan,data,run)
    assert m['members']==1 and records[0]['split']=='val' and m['training_admitted'] is False
    assert m['sources'] and len(images)==1

@pytest.mark.parametrize('what',['rgb','state','records','source'])
def test_changed_replay_inputs_rejected(prepared,what,monkeypatch):
    plan,data,run,_=prepared
    if what=='rgb':Image.new('RGB',(32,32),'red').save(data/'rgb.jpg')
    elif what=='state':(run/'epoch_3/state.pt').write_bytes(b'changed')
    elif what=='records':(plan/'records.json').write_text('[]')
    else:monkeypatch.setattr(experiment,'code',lambda:{'changed':True})
    with pytest.raises(ValueError):experiment.load_plan(plan,data,run)

def test_prepare_rejects_cohort_not_used_by_checkpoint(prepared,tmp_path):
    plan,data,run,ready=prepared
    cp=run/'training_complete.json';c=json.loads(cp.read_text());c['binding']['dataset_sha256']='different';cp.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='mismatch'):experiment.prepare(ready,run,data,tmp_path/'newplan')

def test_mixed_only_has_no_fabricated_pure_accuracy():
    a=r()
    for t in a['targets']:t.update(name='MIXED',target=[.5,0,0,0,0,.5],raw=[.5,0,0,0,0,.5],token_id=-1)
    p=rows(a,[0]*4);result=regression.compare([a],p,p)
    assert result['summary']['new']['recall'] is None
    assert result['counts']['mixed/all']['n']==4

@pytest.mark.parametrize('speed,brake',[(0,True),(8,False)])
def test_frozen_quarantine_keeps_own_rule_identity(speed,brake):
    from ..frozen import label_quarantine as frozen_quarantine
    from qwen3vl_local.audit_joint import label_quarantine as old_quarantine
    evidence=dict(rule_version=rule.ACTION_RULE_VERSION,rule_code_sha256=rule.action_rule_sha256(),
                  future_speeds_exact_mps=[speed]*9,brake=brake,throttle=0 if brake else .5)
    row=dict(frame_id=20,context_id='LEAD_BRAKE',scenario='S',route_id='R',action_evidence=evidence)
    records=[dict(scenario='S',route_id='R',first_affected_frame=24,risk_id='risk')]
    old=copy.deepcopy(row);old['action_evidence']['rule_code_sha256']=old_rule.action_rule_sha256()
    assert frozen_quarantine.phase3_risks(row,records)==old_quarantine.phase3_risks(old,records)
    with pytest.raises(ValueError,match='source mismatch'):
        frozen_quarantine.phase3_risks(old,records)

def test_train_guards_cover_distinct_groups_without_test_selection():
    buckets={}
    for i in reversed(range(20)):
        a=r();a.update(id=f'{i:03d}',split='train',physical_group=f'g{i//2}')
        experiment.guard_candidates(buckets,a)
    a=r();a.update(id='-test',split='test',physical_group='test')
    experiment.guard_candidates(buckets,a)
    chosen=experiment.guard_records(buckets)
    assert [x['id'] for x in chosen]==['000','002','004','006']
    assert len({x['physical_group'] for x in chosen})==4
    assert all(x['split']=='train' for x in chosen)

def test_prepare_guards_are_separate_from_fixed_val(prepared,tmp_path,monkeypatch):
    from ..audit import write
    plan,data,run,ready=prepared
    rr=json.loads(ready.read_text());ds=Path(rr['dataset']);payload=json.loads(ds.read_text())
    train=copy.deepcopy(payload['records'][0]);train.update(id='train-guard',split='train')
    payload['records'].append(train);ds.write_text(json.dumps(payload));rr['dataset_sha256']=sha(ds);ready.write_text(json.dumps(rr))
    receipt=json.loads((run/'epoch_3/receipt.json').read_text());receipt['binding']['dataset_sha256']=sha(ds)
    (run/'epoch_3/receipt.json').write_text(json.dumps(receipt))
    completion=json.loads((run/'training_complete.json').read_text());completion['binding']=receipt['binding']
    completion['checkpoint_receipts']['epoch_3']=sha(run/'epoch_3/receipt.json');(run/'training_complete.json').write_text(json.dumps(completion))
    second=tmp_path/'with_guards';experiment.prepare(ready,run,data,second)
    m,records,images=experiment.load_plan(second,data,run)
    assert m['members']==1 and m['guard_members']==1
    assert json.loads((second/'guards.json').read_text())[0]['id']=='train-guard'
    assert records[0]['split']=='val'

def test_event_mask_does_not_hide_raw_format_decline():
    a=r();old=rows(a,[5]*4);new=rows(a,[5]*4)
    old[0]['prediction']=dict(format_valid=[True]*4,raw_class_ids=[5]*4)
    new[0]['prediction']=dict(format_valid=[False]*4,raw_class_ids=[-1]*4)
    report=regression.compare([a],old,new)
    assert not report['lost_correct']
    assert report['status']=='tradeoff_review_required'
    assert 'diagnostic/format_valid' in report['declining_slices']
