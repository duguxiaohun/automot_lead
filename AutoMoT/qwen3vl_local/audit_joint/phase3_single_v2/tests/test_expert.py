import pytest
import torch
from .. import expert, regions
from .test_single import record, prediction


@pytest.mark.parametrize('variant',regions.VARIANTS)
def test_native_flow_expert_gradient_and_identical_noise_conditions(variant,tmp_path):
    from qwen3vl_local.leadmot.config import LeadMoTPlanningDecoderConfig
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder,FlowMatchingConfig
    cfg=LeadMoTPlanningDecoderConfig(hidden_size=16,num_kv_heads=2,head_dim=8,num_heads=2,
        num_layers=2,rope_type='none',use_bev=False,use_geometry_tokens=True)
    model=expert.RepresentationExpert(ConditionalFlowMatchingDecoder(
        cfg,FlowMatchingConfig(trajectory_heads=2,trajectory_layers=1)),variant=variant)
    inputs=dict(pooled_kv=[(torch.randn(1,2,3,8),torch.randn(1,2,3,8)) for _ in range(2)],
        bev=None,speed=torch.ones(1),target_point=torch.ones(1,2),target_point_next=torch.ones(1,2),final_goal=torch.ones(1,2))
    r=record(variant=variant);pred=prediction(r,[0,1,2,len(regions.names(variant))-1])
    loss=expert.training_loss(model,inputs,r,torch.zeros(1,10,2),torch.zeros(1,8,2),
                              expected_variant=variant,prediction=pred)
    loss.backward()
    assert torch.isfinite(loss) and model.decoder.conditioner.geometry_embedding.token.weight.grad.abs().sum()>0
    outputs=expert.compare_conditions(model,inputs,r,pred,expected_variant=variant,noise=torch.randn(1,18,2))
    assert set(outputs)=={'none','oracle','predicted'} and all(x.shape==(1,8,2) for x in outputs.values())
    expert.save_checkpoint(tmp_path/'native',model,expected_variant=variant)
    reloaded=expert.RepresentationExpert(ConditionalFlowMatchingDecoder(
        cfg,FlowMatchingConfig(trajectory_heads=2,trajectory_layers=1)),variant=variant)
    expert.load_checkpoint(tmp_path/'native',reloaded,expected_variant=variant)
    noise=torch.randn(1,18,2)
    a=expert.compare_conditions(model,inputs,r,pred,expected_variant=variant,noise=noise)
    b=expert.compare_conditions(reloaded,inputs,r,pred,expected_variant=variant,noise=noise)
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_continuous_comparison_rejects_different_members_or_ground_truth():
    import copy
    row=dict(id='one',target_waypoints=[[0.,0.]]*8,none=[[1.,0.]]*8,
             predicted=[[.5,0.]]*8,oracle=[[0.,0.]]*8)
    arms={v:[copy.deepcopy(row)] for v in regions.VARIANTS}
    result=expert.matched_metrics(arms)
    assert result['six']['predicted']['ADE_m']==.5 and result['six']['oracle']['FDE_m']==0
    arms['six'].append(copy.deepcopy(row))
    with pytest.raises(ValueError,match='duplicate_trajectory_members'):expert.matched_metrics(arms)


def tiny_decoder():
    from types import SimpleNamespace
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(use_geometry_tokens=True)
            self.weight=torch.nn.Parameter(torch.randn(2,2))
        def forward(self,**kwargs):return self.weight
    return Decoder()


def test_unbound_expert_rejected_before_forward_or_flow():
    model=tiny_decoder();r=record()
    with pytest.raises(ValueError,match='representation_unbound'):
        expert.training_loss(model,{},r,None,None,expected_variant='six')
    with pytest.raises(ValueError,match='representation_unbound'):
        expert.compare_conditions(model,{},r,None,expected_variant='six',noise=None)


@pytest.mark.parametrize('source',regions.VARIANTS)
@pytest.mark.parametrize('target',regions.VARIANTS)
def test_checkpoint_and_state_dict_bind_variant_before_copy(tmp_path,source,target):
    source_model=expert.RepresentationExpert(tiny_decoder(),variant=source)
    target_model=expert.RepresentationExpert(tiny_decoder(),variant=target)
    path=tmp_path/'weights';expert.save_checkpoint(path,source_model,expected_variant=source)
    before=target_model.decoder.weight.detach().clone()
    if source!=target:
        with pytest.raises(ValueError,match='checkpoint_representation_mismatch'):
            expert.load_checkpoint(path,target_model,expected_variant=target)
        # Cannot bypass contract checks with the native state-dict API or non-strict load.
        with pytest.raises(ValueError,match='checkpoint_representation_mismatch'):
            target_model.load_state_dict(source_model.state_dict(),strict=False)
        assert torch.equal(before,target_model.decoder.weight)
        with pytest.raises(ValueError,match='model_representation_mismatch'):
            expert.training_loss(source_model,{},record(variant=target),None,None,expected_variant=target)
        with pytest.raises(ValueError,match='model_representation_mismatch'):
            source_model(representation_contract=regions.contract(target))
        with pytest.raises(ValueError,match='model_representation_mismatch'):
            expert.compare_conditions(source_model,{},record(variant=target),None,expected_variant=target,noise=None)
    else:
        expert.load_checkpoint(path,target_model,expected_variant=target)
        assert torch.equal(source_model.decoder.weight,target_model.decoder.weight)
        assert torch.equal(target_model(representation_contract=regions.contract(target)),source_model.decoder.weight)
    with pytest.raises(ValueError,match='checkpoint_representation_mismatch'):
        target_model.load_state_dict({'decoder.weight':torch.zeros(2,2)},strict=False)


def test_expert_checkpoint_missing_receipt_or_tampered_bytes_rejected(tmp_path):
    model=expert.RepresentationExpert(tiny_decoder(),variant='six')
    path=tmp_path/'weights';expert.save_checkpoint(path,model,expected_variant='six')
    with pytest.raises(FileExistsError):expert.save_checkpoint(path,model,expected_variant='six')
    with (path/'weights.pt').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='weights_changed'):expert.load_checkpoint(path,model,expected_variant='six')
    (path/'receipt.json').unlink()
    with pytest.raises(FileNotFoundError):expert.load_checkpoint(path,model,expected_variant='six')
