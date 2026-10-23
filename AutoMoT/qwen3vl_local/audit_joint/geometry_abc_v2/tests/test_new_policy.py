import copy
from types import SimpleNamespace
import pytest
import torch
from .. import objective,admission,metrics,regions
from .test_study import record,prediction
from .test_admission import cohort
from ... import phase3_comparison_policy as policy,phase3_compare as entry

def test_region_loss_ignores_other_vocabulary_and_masked_queries():
    choices=[2,4,6,8,10,12,14,16,18];r=record()
    r['targets'][1]['valid']=False
    x=torch.randn(4,30,requires_grad=True)
    actual=objective.loss_from_logits(x,r['targets'],choices)
    valid=[0,2,3];q=torch.tensor([r['targets'][i]['token_id'] for i in valid])
    expected=torch.nn.functional.cross_entropy(x[valid][:,choices],q)
    torch.testing.assert_close(actual,expected)
    changed=x.detach().clone();changed[:,[i for i in range(30) if i not in choices]]=10000
    changed[1]=float('nan')
    torch.testing.assert_close(actual,objective.loss_from_logits(changed,r['targets'],choices))
    actual.backward()
    assert x.grad[1].abs().sum()==0
    assert x.grad[:,[i for i in range(30) if i not in choices]].abs().sum()==0
    with pytest.raises(ValueError):objective.loss_from_logits(x,r['targets'],[2]*9)

def test_prediction_has_normalized_region_probability_and_separate_raw(monkeypatch):
    logits=torch.zeros(4,15);logits[:,14]=100;logits[:,2]=2
    model=torch.nn.Linear(1,1);model.train()
    monkeypatch.setattr(objective,'pack',lambda *a,**k:dict(choices=list(range(9))))
    monkeypatch.setattr(objective,'readout',lambda *a:logits)
    p=objective.predict(SimpleNamespace(model=model),record(),None)
    assert p['token_ids']==[2]*4 and p['raw_class_ids']==[-1]*4
    torch.testing.assert_close(torch.tensor(p['probabilities']).sum(-1),torch.ones(4))
    assert p['probability_space']=='nine_regions' and model.training
    assert max(p['vocabulary_region_mass'])<1e-30

def test_all_validation_members_including_rare_late_ids():
    a=cohort(val_per_event=30);ids=admission.validation_ids(a['records'])
    assert len(ids)==300 and ids==admission.validation_ids(list(reversed(a['records'])))
    assert len(admission.inspect(a)['fixed_validation_ids'])==300

def test_sparse_metrics_explain_instability_without_selecting():
    r=record();r['targets'][0]['token_id']=0
    result=metrics.evaluate([dict(record=r,prediction=prediction(r,[4]*4))])
    s=result['validation_stability']
    assert s['leave_one_group_out_undefined']==1 and s['supported_subset_macro_recall'] is None
    assert len(s['sparse_classes'])==9

def test_schedule_matches_native_and_entry_routes(tmp_path):
    from qwen3vl_local.sft_v2.train import make_scheduler as native
    p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=1.)
    schedule=native(o,9600,2500)
    for step in (0,1,2499,2500,6000,9599,9600):
        assert policy.lr_multiplier(step,9600,2500)==schedule.lr_lambdas[0](step)
    assert entry.commands(tmp_path,'train','single')[0][-1]=='six'
    assert 'phase3_single_v3' in entry.commands(tmp_path,'cpu','single')[0][-1]
    assert entry.commands(tmp_path,'train','base')[0][1].endswith('phase3_base_bounded.sh')
    assert entry.package('a')==entry.package('c')=='geometry_abc_v2'
    with pytest.raises(ValueError):policy.require(per_event=1024,epochs=3,seed=20260904,learning_rate=1e-5)

def test_formal_three_epoch_budget_and_repeat_caps_for_both_planners():
    from .. import planning
    from ...phase3_single_v2 import planning as single_planning
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    rows=[]
    for e in CONTEXT_IDS:
        for f in range(1400):
            rows.append(dict(id=f'{e}/{f}',event=e,split='train',scenario='Synthetic',
                route_id=f'Town00_{e}',physical_group=f'Synthetic/Town00_{e}',frame=f,
                targets=[dict(valid=True)],sampling_mass={'class/0':1.}))
    plan=planning.project(rows,per_event=1280,world_size=4,epochs=3,seed=20260904)
    assert [p['optimizer_steps'] for p in plan['plans']]==[3200]*3
    assert all(len(p['ordered_ids'])==12800 and p['max_frame_repeat']==1 and p['max_run_frame_repeat']<=3 for p in plan['plans'])
    assert single_planning.project(rows,per_event=1280,world_size=4,epochs=3,seed=20260904)==plan

def test_boundary_candidates_use_only_train_without_changing_labels():
    from ..boundary_audit import inspect
    train=record();train['targets'][0].update(radius_m=3.5,lateral_m=2.5)
    val=copy.deepcopy(train);val['split']='val'
    for t in val['targets']:t.update(radius_m=500,lateral_m=-300)
    before=copy.deepcopy(train)
    report=inspect([train,val])
    assert report==inspect([train]) and train==before
    assert not report['thresholds_changed'] and not report['RGB_review_completed']
