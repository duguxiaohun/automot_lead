"""Explicit representation-bound expert conditions; unchanged continuous targets."""
import copy
import json
from pathlib import Path
import torch
from . import regions, objective
from ..geometry_v1.expert import paired_trajectory_metrics


class RepresentationExpert(torch.nn.Module):
    """Bind a newly initialized expert BEFORE training or loading study weights.

    The contract is part of state_dict, including ordinary torch checkpoints.
    Unbound/legacy states cannot be relabelled by loading with strict=False.
    This wrapper is not a migration or certification of existing unbound weights.
    """
    def __init__(self, decoder, *, variant):
        super().__init__()
        if isinstance(decoder, RepresentationExpert): raise ValueError('expert_already_bound')
        if not decoder.config.use_geometry_tokens: raise ValueError('four_token_expert_required')
        self.decoder = decoder
        self._representation_contract = copy.deepcopy(regions.contract(variant))

    @property
    def config(self): return self.decoder.config

    @property
    def flow_config(self): return self.decoder.flow_config

    def get_extra_state(self):
        return dict(schema='study_expert_representation_v1', contract=copy.deepcopy(self._representation_contract))

    def set_extra_state(self, state):
        if state != self.get_extra_state(): raise ValueError('expert_checkpoint_representation_mismatch')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Also runs when this expert is nested inside another torch module.
        self.set_extra_state(state_dict.get(prefix + '_extra_state'))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, *, representation_contract, **kwargs):
        if representation_contract != self._representation_contract:
            raise ValueError('expert_model_representation_mismatch')
        return self.decoder(**kwargs)


def require_model(model, expected_variant):
    base=model.module if hasattr(model,'module') else model
    if not isinstance(base, RepresentationExpert): raise ValueError('expert_model_representation_unbound')
    if base.get_extra_state()['contract'] != regions.contract(expected_variant):
        raise ValueError('expert_model_representation_mismatch')
    return base


def save_checkpoint(path, model, *, expected_variant):
    """Immutable weights checkpoint; no optimizer/RNG resume claim."""
    from . import dataset
    base=require_model(model, expected_variant)
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    torch.save(base.state_dict(), path/'weights.pt.pending')
    (path/'weights.pt.pending').rename(path/'weights.pt')
    dataset.write(path/'receipt.json',dict(binding=base.get_extra_state(),
        weights_sha256=dataset.sources.file_sha(path/'weights.pt')))


def load_checkpoint(path, model, *, expected_variant):
    from . import dataset
    base=require_model(model, expected_variant);path=Path(path)
    receipt=json.loads((path/'receipt.json').read_text())
    base.set_extra_state(receipt['binding'])
    if dataset.sources.file_sha(path/'weights.pt') != receipt['weights_sha256']:
        raise ValueError('expert_checkpoint_weights_changed')
    state=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
    base.load_state_dict(state,strict=True)
    return model


def condition(record, *, expected_variant, mode='predicted', prediction=None, device='cpu'):
    if expected_variant != record['variant']: raise ValueError('expert_representation_mismatch')
    count=len(regions.names(expected_variant))
    if mode=='none': ids=[9]*4
    elif mode=='oracle':
        if not all(t['valid'] and t['token_id']>=0 for t in record['targets']):
            raise ValueError('hard_oracle_requires_four_pure_targets_no_mixed_argmax')
        ids=[t['token_id'] for t in record['targets']]
    elif mode=='predicted':
        if prediction is None or prediction.get('source')!='predicted': raise ValueError('prediction_required_no_oracle_fallback')
        if prediction.get('contract')!=regions.contract(expected_variant) or prediction.get('sample_binding')!=objective.observation_binding(record):
            raise ValueError('expert_prediction_binding_mismatch')
        ids=prediction['token_ids']
    else: raise ValueError('invalid_condition_mode')
    if len(ids)!=4 or any(type(i) is not int or not 0<=i<(10 if mode=='none' else count) for i in ids):
        raise ValueError('invalid_expert_tokens')
    if mode != 'none' and any(i not in objective.allowed_indices(record) for i in ids):
        raise ValueError('expert_tokens_outside_event_domain')
    return dict(geometry_token_ids=torch.tensor([ids],dtype=torch.long,device=device),
        geometry_reference_yaw=torch.tensor([record['observation']['reference']['relative_yaw']],device=device))


def matched_metrics(arms):
    if set(arms)!=set(regions.VARIANTS): raise ValueError('four_representation_arms_required')
    reference=arms['a']; ids=[r['id'] for r in reference]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate_trajectory_members')
    for arm in arms.values():
        if [r['id'] for r in arm]!=ids: raise ValueError('trajectory_cohort_or_order_mismatch')
        for a,b in zip(reference,arm):
            if a['target_waypoints']!=b['target_waypoints']: raise ValueError('continuous_targets_differ')
    return {v:paired_trajectory_metrics(arm) for v,arm in arms.items()}


def training_loss(model, observed_inputs, record, gt_route, gt_waypoints, *,
                  expected_variant, mode='predicted', prediction=None, generator=None):
    from qwen3vl_local.action_prior.flow_matching import make_training_flow, flow_matching_loss
    base=require_model(model,expected_variant)
    if not base.config.use_geometry_tokens: raise ValueError('four_token_expert_required')
    if any(k in observed_inputs for k in ('action_token_id','geometry_token_ids','geometry_reference_yaw')):
        raise ValueError('duplicate_or_legacy_expert_condition')
    cond=condition(record,expected_variant=expected_variant,mode=mode,prediction=prediction,device=gt_route.device)
    flow=make_training_flow(gt_route,gt_waypoints,base.flow_config,generator)
    output=model(**observed_inputs,**cond,representation_contract=regions.contract(expected_variant),
        flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
    return flow_matching_loss(output,flow['flow_target_velocity'],base.config.num_route_queries,1.,1.)[0]


@torch.no_grad()
def compare_conditions(model, observed_inputs, record, prediction, *, expected_variant, noise):
    require_model(model,expected_variant)
    was_training=model.training;model.eval()
    try:
        return {mode:model(**observed_inputs,**condition(record,expected_variant=expected_variant,
                    mode=mode,prediction=prediction,device=noise.device),
                    representation_contract=regions.contract(expected_variant),
                    flow_sample_noise=noise.clone(),sample_trajectory=True)['pred_future_waypoints']
                for mode in ('none','oracle','predicted')}
    finally:model.train(was_training)
