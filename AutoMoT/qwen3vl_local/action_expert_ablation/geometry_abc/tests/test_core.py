import copy
import json
from pathlib import Path
import pytest
import torch
from qwen3vl_local.action_expert_ablation.geometry_abc import planning,model,contracts,runtime
from qwen3vl_local.audit_joint.abc_delivery import pack,sha
from qwen3vl_local.audit_joint.geometry_v1 import codec
from qwen3vl_local.audit_joint.geometry_abc_v2.dataset import combine
from qwen3vl_local.audit_joint.phase3_abc_refine import prompt,regression


def test_condition_response_gate_distinguishes_no_response_and_constant_labels():
    from qwen3vl_local.action_expert_ablation.geometry_abc.diagnostics import summarize
    def case(labels,replace,permute):
        return dict(labels=labels,results={k:{'mean_waypoint_shift_from_oracle_m':v}
            for k,v in [('replace_regions',replace),('permute_horizons',permute)]})
    assert summarize([case([1,2,3,4],0.,0.)])['status']=='condition_use_review_required'
    assert summarize([case([1,1,1,1],.1,0.)])['status']=='observed_condition_response'
    assert not summarize([case([1,1,1,1],.1,0.)])['permutation_supported']
    assert summarize([case([1,2,3,4],.1,0.)])['status']=='condition_use_review_required'
    assert summarize([case([1,2,3,4],.1,.1)])['status']=='observed_condition_response'


def record(i=0,event='A',split='train'):
    obs=dict(speed_mps=0.,reference=dict(relative_yaw=0.,source=codec.REFERENCE_SOURCE),
             navigation_xy=[[1.,0.],[2.,0.],[3.,0.]],history=[codec.classify([0.,0.],t,0.,0.) for t in codec.LOOKBACKS])
    return combine(dict(id=f'{event}{i}',scenario='s',route_id='r',frame=i,image='s/r/rgb/0000.jpg',
        physical_group='s/r',split=split,event=event,road_structure='R1',observation=obs,
        targets=[codec.classify([.49,0.],t,0.,0.) for t in codec.HORIZONS]))


def tiny(variant='a'):
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.flow_matching import FlowMatchingConfig
    cfg=LeadMoTPlanningDecoderConfig(hidden_size=32,num_kv_heads=2,head_dim=16,num_heads=2,
        num_layers=2,rope_type='none',use_bev=False,use_geometry_tokens=True,dropout=0.)
    fc=FlowMatchingConfig(time_embed_dim=8,trajectory_layers=1,trajectory_heads=2,sample_steps=2)
    return model.ABCFlowDecoder(cfg,fc,variant=variant)


def inputs(m,r,variant='a',prefix=0):
    return dict(**model.tensors(r,variant,'cpu'),**runtime.status(r,'cpu',torch.float32),bev=None,
        pooled_kv=[(torch.zeros(1,2,prefix,16),torch.zeros(1,2,prefix,16)) for _ in range(2)],rope_position_offset=0)


@pytest.mark.parametrize('variant',list('abc'))
@pytest.mark.parametrize('prefix',[0,3])
def test_real_fm_history_future_grad_reload_and_same_noise(tmp_path,variant,prefix):
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    torch.manual_seed(17);m=tiny(variant);r=record();kw=inputs(m,r,variant,prefix)
    flow=make_training_flow(torch.randn(1,10,2),torch.randn(1,8,2),m.flow_config)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    out=m(**kw,flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
    loss=flow_matching_loss(out,flow['flow_target_velocity'],10,.5,1.)[0];loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.conditioner.geometry_embedding.role.weight.grad.norm(dim=1).min()>0
    assert all(p.weight.grad.norm()>0 for p in m.history_kv)
    optimizer.step();m.eval();noise=torch.randn(1,18,2)
    first=m(**kw,flow_sample_noise=noise)['pred_future_waypoints']
    changed=dict(kw,geometry_token_ids=(kw['geometry_token_ids']+1)%9)
    assert not torch.allclose(first,m(**changed,flow_sample_noise=noise)['pred_future_waypoints'])
    changed=dict(kw,history_token_ids=(kw['history_token_ids']+1)%9)
    assert not torch.allclose(first,m(**changed,flow_sample_noise=noise)['pred_future_waypoints'])
    path=tmp_path/'state.pt';torch.save(m.state_dict(),path)
    restored=tiny(variant);restored.load_state_dict(torch.load(path,weights_only=True));restored.eval()
    assert torch.equal(first,restored(**kw,flow_sample_noise=noise)['pred_future_waypoints'])
    with pytest.raises(ValueError,match='contract'):tiny('b' if variant=='a' else 'a').load_state_dict(m.state_dict(),strict=False)


def test_history_missing_is_not_near_and_future_not_in_text():
    r=record();r['representations']['a']['history'][0]=codec.unavailable('gap',2.)
    cond=model.tensors(r,'a','cpu');assert cond['history_token_ids'][0,0]==9
    assert not cond['history_mask'][0,0]
    before=runtime.observation_text(r,'a');r['representations']['a']['targets'][0]['token_id']=8
    assert runtime.observation_text(r,'a')==before
    assert 'ACTION_QUERY' not in before[1]


def test_joint_plan_shared_capacity_and_all_epochs():
    rows=[record(i,e) for i in range(5) for e in ('A','B')]
    p=planning.project(rows,quotas={'A':7,'B':8},epochs=7,cap=3,world=1,accumulation=4)
    assert p['total_presentations']==105 and p['total_updates']==28
    assert all(x['max_physical_repeat']==3 for x in p['plans'])
    assert p==planning.project(rows,quotas={'A':7,'B':8},epochs=7,cap=3,world=1,accumulation=4)
    with pytest.raises(ValueError,match='joint_physical'):
        planning.project(rows,quotas={'A':8,'B':8},epochs=1,cap=3,world=1)


def test_no_cross_split_or_duplicate_plan():
    with pytest.raises(ValueError,match='split_leakage'):
        planning.project([record(0),record(1,split='test')],quotas={'A':1},world=1)
    with pytest.raises(ValueError,match='duplicate_member'):
        planning.project([record(),record()],quotas={'A':1},world=1)


def test_prompt_causal_exact_zero_speed_and_nonmonotonic_example():
    r=record();r['variant']='a'
    text=prompt.verify_no_target_leak(r)[1]['content'][-1]['text']
    assert 'inner=0.0 m, outer=0.25 m' in text
    assert 'inner=0.0 m, outer=1.0 m' in text
    assert [t['token_id'] for t in r['targets']]==[5,4,4,4]
    for v in ('b','c'):
        r['variant']=v
        assert prompt.messages(r,None)==prompt.ORIGINAL_MESSAGES(r,None)


def test_any_old_correct_loss_requires_review_even_net_gain():
    a=[];b=[]
    for i,(old,new) in enumerate(((4,5),(5,4),(5,4))):
        r=record(i);r['targets']=[dict(valid=True,token_id=4)]*4
        a.append(dict(record=r,prediction=dict(token_ids=[old]*4)))
        b.append(dict(record=r,prediction=dict(token_ids=[new]*4)))
    result=regression.compare(a,b)
    assert result['status']=='tradeoff_review_required' and len(result['old_correct_losses'])==4
    assert len(result['new_correct'])==8
    assert regression.compare(a,a)['status']=='observed_cohort_protection_passed'


def test_pack_core_complete_atomic_and_size_refusal(tmp_path):
    import os,zipfile
    core=tmp_path/'all_cases.jsonl';core.write_text(''.join(json.dumps({'id':i})+'\n' for i in range(1000)))
    output=tmp_path/'delivery'/'audit.zip'
    receipt=pack(output,{'cases.jsonl':core},summary={'cases':1000})
    before=sha(output)
    with zipfile.ZipFile(output) as z:assert len(z.read('cases.jsonl').splitlines())==1000
    large=tmp_path/'large.json';large.write_bytes(os.urandom(10000))
    with pytest.raises(ValueError,match='core_package'):
        pack(output,{'large.json':large},limit=1000)
    assert sha(output)==before and receipt['bytes']<30_000_000
    with pytest.raises(ValueError,match='unsafe'):pack(output,{'../bad':core})
