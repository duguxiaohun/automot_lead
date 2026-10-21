import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qwen3vl_local.audit_joint.geometry_v1 import codec, sources, dataset, planning, objective, expert


class Lanes:
    hashes={}
    def at(self,town,position,ego_yaw):
        return codec.reference(0.,ego_yaw)


def meta(frame=8, speed=3., yaw=0.):
    matrix=np.eye(4);c,s=math.cos(yaw),math.sin(yaw)
    matrix[:2,:2]=[[c,-s],[s,c]]; matrix[0,3]=frame*.25*speed
    f=np.zeros((41,3)); f[:,0]=np.arange(41)*.05*speed
    return dict(ego_matrix=matrix,speed=speed,dataset_information=dict(data_save_freq=5),
        future_positions=f,next_target_points=[[30,0,0],[35,0,0],[90,0,0]])


def record(frame=8, event='UE', split='train'):
    metas={i:meta(i) for i in range(0,17)}
    route=dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split=split)
    r=dataset.record(route,dict(frame=frame,event=event,road_structure='STRAIGHT'),metas,Lanes())
    r['physical_group']='Synthetic/Town00_Test'
    return r


def test_nine_regions_boundaries_and_adaptive_radius():
    assert codec.tolerance(0)==.5 and codec.tolerance(3)==pytest.approx(.6) and codec.tolerance(10)==1
    assert codec.classify([0,-2],1,2,0)['name']=='CENTER_REFERENCE'
    assert codec.classify([0,2],1,2,0)['name']=='CENTER_REFERENCE'
    assert codec.classify([4.8,0],2,3,0)['name']=='CENTER_REFERENCE'
    assert codec.classify([7.2,0],2,3,0)['name']=='CENTER_REFERENCE'
    # Exact binary-representable edges must land in the reference band.
    assert codec.classify([8,0],2,5,0)['name']=='CENTER_REFERENCE'
    assert codec.classify([12,0],2,5,0)['name']=='CENTER_REFERENCE'
    names=set()
    for y in (-3,0,3):
        for radius in (7,10,13):
            names.add(codec.classify([math.sqrt(radius**2-y*y),y],2,5,0)['name'])
    assert names==set(codec.NAMES)
    assert codec.classify([100000,0],2,0,0)['name']=='CENTER_FAR'
    assert codec.classify([0,0],2,0,0)['name']=='CENTER_REFERENCE'
    assert codec.classify([0,0],2,-.1,0)['start_speed_mps']==0


@pytest.mark.parametrize('v',[float('nan'),float('inf'),-.10001])
def test_invalid_speed_is_never_zero(v):
    with pytest.raises(ValueError):codec.classify([1,0],1,v,0)


def test_lane_orientation_and_frozen_future_reference():
    assert codec.reference(math.pi,0)['relative_yaw']==pytest.approx(0)
    assert codec.reference(2.,0)['relative_yaw']==pytest.approx(2-math.pi)
    m=meta();state=sources.observed(m,'Town00',Lanes())
    targets=sources.future(m,state,anchor=8)
    assert [t['native_tick'] for t in targets]==[10,20,30,40]
    assert all(t['name']=='CENTER_REFERENCE' for t in targets)
    m['future_positions'][30:]=np.nan
    assert [t['valid'] for t in sources.future(m,state,anchor=8)]==[True,True,False,False]


def test_history_causal_overlap_no_fill_and_no_future_fields():
    metas={i:meta(i) for i in range(17)}
    before=sources.history(metas,8,'Town00',Lanes())
    assert [t['start_frame'] for t in before]==[0,2,4,6]
    assert all(t['end_frame']==8 for t in before)
    for i,m in metas.items():
        m['future_positions']='poisoned';m['future_speeds']='poisoned'
        if i>8:m['ego_matrix']='poisoned';m['speed']='poisoned'
    assert sources.history(metas,8,'Town00',Lanes())==before
    del metas[1]
    after=sources.history(metas,8,'Town00',Lanes())
    assert [t['valid'] for t in after]==[False,True,True,True]
    assert sources.history(metas,8,'Town00',Lanes(),discontinuities=[7])[3]['valid'] is False


def test_prompt_has_no_future_label_dependency_or_event_mask():
    r=record(); original=objective.messages(r,'image')
    r['targets']='poisoned';r['anchor_action']='STOP'
    assert objective.messages(r,'image')==original
    prompt=str(original)
    assert 'I=RIGHT_FAR' in prompt and 'All nine answers' in prompt
    assert 'endpoint at -2s' in prompt and 'past position minus CURRENT origin' in prompt
    assert 'allowed actions' not in prompt


def test_full_vocabulary_loss_and_missing_query_mask():
    r=record(); logits=torch.randn(4,12,requires_grad=True)
    r['targets'][1]=codec.unavailable('gap',1)
    loss=objective.loss_from_logits(logits,r['targets'],list(range(9)))
    loss.backward()
    assert logits.grad[1].abs().sum()==0 and logits.grad[0,10]!=0
    with pytest.raises(ValueError):objective.loss_from_logits(logits,[codec.unavailable('gap',t) for t in codec.HORIZONS],list(range(9)))


def test_planning_no_old_action_dependency_bounded_and_actual_history():
    rows=[record(i) for i in (8,9,10,11)]
    for r in rows:r['action_labels']=None
    work,report=planning.plan(rows,per_event=4,world_size=2,max_repeat=2)
    assert len(work)==4 and len({r['id'] for r in work})==4 and not report['actual_exposure_advanced']
    exposure={rows[0]['id']:100}
    work,_=planning.plan(rows,per_event=2,world_size=2,exposure=exposure)
    assert rows[0]['id'] not in [r['id'] for r in work] and exposure=={rows[0]['id']:100}
    with pytest.raises(ValueError,match='capacity'):planning.plan(rows,per_event=10,world_size=2,max_repeat=2)
    with pytest.raises(ValueError,match='budget'):planning.plan(rows,per_event=4,world_size=2,max_epoch_steps=1)
    leak=copy.deepcopy(rows[0]);leak['id']='other';leak['split']='test'
    with pytest.raises(ValueError,match='split_leakage'):planning.plan(rows+[leak],per_event=2,world_size=2)


def test_predicted_expert_never_uses_oracle_fallback():
    r=record()
    with pytest.raises(ValueError,match='predicted_tokens_required'):expert.condition(r)
    predicted=dict(token_ids=[0,1,2,8],source='predicted',contract=codec.contract(),reference=r['observation']['reference'],sample_binding=objective.observation_binding(r))
    assert expert.condition(r,prediction=predicted)['geometry_token_ids'].tolist()==[[0,1,2,8]]
    assert expert.condition(r,mode='none')['geometry_token_ids'].tolist()==[[9]*4]
    r['targets'][0]=codec.unavailable('gap',.5)
    with pytest.raises(ValueError,match='complete'):expert.condition(r,mode='oracle')


@pytest.mark.parametrize('flow',[False,True])
def test_real_expert_forward_backward_time_reference_and_reload(flow):
    from qwen3vl_local.leadmot.config import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.leadmot.decoder import LeadMoTPlanningDecoder
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder,FlowMatchingConfig
    cfg=LeadMoTPlanningDecoderConfig(hidden_size=16,num_kv_heads=2,head_dim=8,num_heads=2,
        num_layers=2,rope_type='none',use_bev=False,use_geometry_tokens=True)
    make=lambda:ConditionalFlowMatchingDecoder(cfg,FlowMatchingConfig(trajectory_heads=2,trajectory_layers=1)) if flow else LeadMoTPlanningDecoder(cfg)
    torch.manual_seed(8);model=make()
    kwargs=dict(pooled_kv=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(2)],
        bev=None,speed=torch.ones(1),target_point=torch.ones(1,2),target_point_next=torch.ones(1,2),final_goal=torch.ones(1,2))
    if flow:kwargs.update(flow_state=torch.randn(1,18,2),flow_time=torch.tensor([.4]),sample_trajectory=False)
    name='flow_velocity' if flow else 'pred_future_waypoints'
    cond=dict(geometry_token_ids=torch.tensor([[0,1,2,8]]),geometry_reference_yaw=torch.tensor([.2]))
    a=model(**kwargs,**cond)[name]
    b=model(**kwargs,**dict(cond,geometry_token_ids=cond['geometry_token_ids'].flip(1)))[name]
    c=model(**kwargs,**dict(cond,geometry_reference_yaw=torch.tensor([-.3])))[name]
    assert not torch.equal(a,b) and not torch.equal(a,c)
    (a.square().mean()+b.square().mean()).backward()
    module=model.conditioner if flow else model
    assert module.geometry_embedding.token.weight.grad.abs().sum()>0
    assert module.geometry_embedding.time[0].weight.grad.abs().sum()>0
    assert cfg.slice_layout()['geometry']==(0,4)
    model.eval();other=make();other.load_state_dict(model.state_dict(),strict=True);other.eval()
    assert torch.equal(model(**kwargs,**cond)[name],other(**kwargs,**cond)[name])
    with pytest.raises(ValueError):model(**kwargs,geometry_token_ids=torch.tensor([1]),geometry_reference_yaw=torch.tensor([0.]))
    with pytest.raises(ValueError):model(**kwargs)
    if flow:
        inputs={k:v for k,v in kwargs.items() if not k.startswith('flow_') and k!='sample_trajectory'}
        r=record();prediction=dict(token_ids=[0,1,2,8],source='predicted',contract=codec.contract(),reference=r['observation']['reference'],sample_binding=objective.observation_binding(r))
        loss=expert.training_loss(model,inputs,r,torch.zeros(1,10,2),torch.zeros(1,8,2),prediction=prediction)
        loss.backward();assert torch.isfinite(loss)
        compared=expert.compare_conditions(model,inputs,r,prediction,noise=torch.randn(1,18,2))
        assert set(compared)=={'none','oracle','predicted'} and all(p.shape==(1,8,2) for p in compared.values())
