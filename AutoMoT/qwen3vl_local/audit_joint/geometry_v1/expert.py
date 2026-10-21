"""Timed [B,4] geometry conditioning for the existing continuous waypoint expert."""
import torch
from torch import nn

from . import codec
from .identity import observation_binding


class GeometryEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Index 9 is explicit no-prior, never a target class or guessed STOP.
        self.token = nn.Embedding(10, hidden_size)
        self.time = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.reference = nn.Linear(2, hidden_size, bias=False)
        self.register_buffer('horizons', torch.tensor(codec.HORIZONS).reshape(1,4,1))
        nn.init.normal_(self.token.weight, std=.02)

    def forward(self, ids, yaw, *, batch_size):
        if ids is None or ids.shape != (batch_size,4) or ids.dtype != torch.long:
            raise ValueError('geometry_token_ids must be explicit [B,4] int64')
        if yaw is None or yaw.shape != (batch_size,) or not yaw.is_floating_point() or not torch.isfinite(yaw).all():
            raise ValueError('geometry_reference_yaw must be finite [B] radians')
        if (ids<0).any() or (ids>9).any(): raise ValueError('geometry IDs must be 0..8 or explicit UNCOND=9')
        if ((ids==9).any(1) != (ids==9).all(1)).any():
            raise ValueError('no-prior ablation requires all four UNCOND slots')
        device, dtype = self.token.weight.device, self.token.weight.dtype
        ids=ids.to(device); yaw=yaw.to(device=device,dtype=dtype)
        direction = torch.stack([yaw.cos(),yaw.sin()],-1)
        return self.token(ids)+self.time(self.horizons.to(dtype=dtype))+self.reference(direction).unsqueeze(1)


def condition(record, *, mode='predicted', prediction=None, device='cpu'):
    """Oracle requires explicit mode; predicted mode never falls back to labels."""
    reference = record['observation']['reference']
    if reference['source'] != codec.REFERENCE_SOURCE: raise ValueError('reference_source_mismatch')
    if mode == 'none': ids = [9]*4
    elif mode == 'oracle':
        if len(record['targets'])!=4 or not all(t['valid'] for t in record['targets']):
            raise ValueError('oracle_requires_four_complete_endpoints')
        ids = [t['token_id'] for t in record['targets']]
    elif mode == 'predicted':
        if prediction is None or prediction.get('source') != 'predicted':
            raise ValueError('predicted_tokens_required_no_oracle_fallback')
        if prediction.get('contract') != codec.contract() or prediction.get('reference') != reference:
            raise ValueError('prediction_reference_or_contract_mismatch')
        if prediction.get('sample_binding') != observation_binding(record):
            raise ValueError('prediction_sample_mismatch')
        ids = prediction['token_ids']
    else: raise ValueError('unknown_geometry_condition_mode')
    if len(ids)!=4 or any(type(i) is not int or not 0<=i<=(9 if mode=='none' else 8) for i in ids):
        raise ValueError('invalid_geometry_sequence')
    return dict(geometry_token_ids=torch.tensor([ids],dtype=torch.long,device=device),
                geometry_reference_yaw=torch.tensor([reference['relative_yaw']],dtype=torch.float32,device=device))


def paired_trajectory_metrics(cases):
    """Identical cases under none/oracle/predicted; no separate resampling."""
    result = {}
    if not cases: raise ValueError('empty_trajectory_evaluation')
    for mode in ('none','oracle','predicted'):
        errors=[]; stop_errors=[]
        for c in cases:
            target=torch.as_tensor(c['target_waypoints'],dtype=torch.float64)
            pred=torch.as_tensor(c[mode],dtype=torch.float64)
            if target.shape!=(8,2) or pred.shape!=target.shape or not torch.isfinite(target).all() or not torch.isfinite(pred).all():
                raise ValueError('eight_finite_waypoints_required')
            e=torch.linalg.vector_norm(pred-target,dim=-1); errors.append(e)
            if c.get('endpoint_speed_mps') is not None and c['endpoint_speed_mps']<=.2:
                stop_errors.append(float(e[-1]))
        e=torch.stack(errors)
        result[mode]=dict(cases=len(cases),ADE_m=float(e.mean()),FDE_m=float(e[:,-1].mean()),
            stop_cases=len(stop_errors),stop_endpoint_error_m=sum(stop_errors)/len(stop_errors) if stop_errors else None)
    return result


def training_loss(model, observed_inputs, record, gt_route, gt_waypoints, *,
                  mode='predicted', prediction=None, generator=None):
    """Use the unchanged continuous FM target; regions never become center points.

    observed_inputs is the existing expert's causal KV/BEV/speed/navigation dict.
    Caller owns optimizer/DDP/budget; this function is the geometry loss adapter.
    """
    from qwen3vl_local.action_prior.flow_matching import make_training_flow, flow_matching_loss
    base=model.module if hasattr(model,'module') else model
    if not base.config.use_geometry_tokens:raise ValueError('geometry_expert_required')
    if any(k in observed_inputs for k in ('action_token_id','geometry_token_ids','geometry_reference_yaw')):
        raise ValueError('duplicate_or_legacy_expert_condition')
    cond=condition(record,mode=mode,prediction=prediction,device=gt_route.device)
    flow=make_training_flow(gt_route,gt_waypoints,base.flow_config,generator)
    outputs=model(**observed_inputs,**cond,flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
    return flow_matching_loss(outputs,flow['flow_target_velocity'],base.config.num_route_queries,1.,1.)[0]


@torch.no_grad()
def compare_conditions(model, observed_inputs, record, prediction, *, noise):
    """Three interventions on one checkpoint and identical ODE noise.

    This is a conditional sensitivity comparison; independent training of each
    arm is still required to claim a retraining ablation benefit.
    """
    was_training=model.training;model.eval()
    try:
        return {mode:model(**observed_inputs,
                    **condition(record,mode=mode,prediction=prediction,device=noise.device),
                    flow_sample_noise=noise.clone(),sample_trajectory=True)['pred_future_waypoints']
                for mode in ('none','oracle','predicted')}
    finally:model.train(was_training)
