"""Real FM decoder with four future slots and four causal history KV slots."""
import copy
import torch
from torch import nn
from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder
from .contracts import contract


class TemporalRegions(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.token=nn.Embedding(10,hidden)
        self.role=nn.Embedding(2,hidden)
        self.time=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.reference=nn.Linear(2,hidden,bias=False)
        nn.init.normal_(self.token.weight,std=.02)

    def encode(self, ids, yaw, *, history=False, mask=None):
        if ids.ndim!=2 or ids.shape[1]!=4 or ids.dtype!=torch.long or ((ids<0)|(ids>9)).any():
            raise ValueError('four_valid_region_ids_required')
        if yaw.shape!=(ids.shape[0],) or not torch.isfinite(yaw).all(): raise ValueError('invalid_reference_yaw')
        if history:
            if mask is None or mask.dtype!=torch.bool or mask.shape!=ids.shape:
                raise ValueError('explicit_history_mask_required')
            if ((ids==9)!=~mask).any(): raise ValueError('missing_history_must_be_UNKNOWN')
        elif ((ids==9).any(1)!=(ids==9).all(1)).any():
            raise ValueError('future_UNCOND_requires_all_four_slots')
        device=self.token.weight.device;dtype=self.token.weight.dtype
        ids=ids.to(device);yaw=yaw.to(device=device,dtype=dtype)
        times=[-2.,-1.5,-1.,-.5] if history else [.5,1.,1.5,2.]
        t=torch.tensor(times,device=device,dtype=dtype).reshape(1,4,1)
        direction=torch.stack((yaw.cos(),yaw.sin()),-1)
        # UNKNOWN retains its own embedding and slot time; it is never a spatial region.
        return self.token(ids)+self.role.weight[int(history)].reshape(1,1,-1)+self.time(t)+self.reference(direction)[:,None]

    def forward(self, ids, yaw, *, batch_size):
        if ids.shape[0]!=batch_size: raise ValueError('region_batch_mismatch')
        return self.encode(ids,yaw)


class ABCFlowDecoder(ConditionalFlowMatchingDecoder):
    def __init__(self, config, flow_config, *, variant):
        if not config.use_geometry_tokens or config.use_high_level_action_token:
            raise ValueError('ABC_requires_geometry_without_semantic_action')
        super().__init__(config,flow_config)
        self.conditioner.geometry_embedding=TemporalRegions(config.hidden_size)
        self.history_kv=nn.ModuleList(nn.Linear(config.hidden_size,2*config.hidden_size)
                                      for _ in range(config.num_layers))
        self._contract=contract(variant)

    def get_extra_state(self): return copy.deepcopy(self._contract)

    def set_extra_state(self,state):
        if state!=self._contract: raise ValueError('ABC_checkpoint_contract_mismatch')

    def _load_from_state_dict(self,state,prefix,*args,**kwargs):
        self.set_extra_state(state.get(prefix+'_extra_state'))
        return super()._load_from_state_dict(state,prefix,*args,**kwargs)

    def _conditioning(self, *, history_token_ids, history_mask, representation_contract, **kwargs):
        if representation_contract!=self._contract: raise ValueError('ABC_representation_mismatch')
        history=self.conditioner.geometry_embedding.encode(history_token_ids,
            kwargs['geometry_reference_yaw'],history=True,mask=history_mask)
        caches=kwargs['pooled_kv']
        if len(caches)!=len(self.history_kv): raise ValueError('prefix_layer_count_mismatch')
        pooled=[]
        for projection,(k,v) in zip(self.history_kv,caches):
            h=projection(history).reshape(history.shape[0],4,2,self.config.num_kv_heads,self.config.head_dim)
            hk,hv=h[:,:,0].transpose(1,2),h[:,:,1].transpose(1,2)
            pooled.append((torch.cat((k,hk.to(k.dtype)),dim=2),torch.cat((v,hv.to(v.dtype)),dim=2)))
        kwargs['pooled_kv']=pooled
        offset=kwargs.get('rope_position_offset')
        kwargs['rope_position_offset']=(0 if offset is None else offset)+4
        return super()._conditioning(**kwargs)


def tensors(record,variant,device):
    view=record['representations'][variant];obs=record['observation']
    if len(view['targets'])!=4 or not all(t['valid'] and 0<=t['token_id']<9 for t in view['targets']):
        raise ValueError('oracle_requires_four_complete_targets')
    history=view['history']
    if len(history)!=4: raise ValueError('four_history_slots_required')
    return dict(geometry_token_ids=torch.tensor([[t['token_id'] for t in view['targets']]],device=device),
        geometry_reference_yaw=torch.tensor([obs['reference']['relative_yaw']],device=device,dtype=torch.float32),
        history_token_ids=torch.tensor([[t['token_id'] if t['valid'] else 9 for t in history]],device=device),
        history_mask=torch.tensor([[bool(t['valid']) for t in history]],device=device),
        representation_contract=contract(variant))


def embedding_diagnostics(model):
    weight=model.conditioner.geometry_embedding.token.weight.detach().float()
    x=weight[:9];norm=x.norm(dim=1);unit=x/norm[:,None].clamp_min(1e-12)
    singular=torch.linalg.svdvals(x);p=singular/singular.sum().clamp_min(1e-12)
    grad=model.conditioner.geometry_embedding.token.weight.grad
    return dict(norms=norm.cpu().tolist(),cosine=(unit@unit.T).cpu().tolist(),
        effective_rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum())),
        gradient_norms=grad.float().norm(dim=1).cpu().tolist() if grad is not None else None)
