"""Reuse frozen BEV/Qwen extraction, replacing status values at its boundary."""
import copy
import time
import torch
from qwen3vl_local.action_expert_ablation import common
from qwen3vl_local.audit_joint.geometry_abc_v2 import objective
from qwen3vl_local.action_prior.precision import decoder_forward
from .model import tensors


def observation_text(record,variant):
    r=copy.deepcopy(record);r['variant']=variant
    r['observation']['history']=r['representations'][variant]['history']
    messages=objective.messages(r,None)
    text=messages[1]['content'][-1]['text']
    # Region semantics and causal observation are frozen v2. No future oracle
    # answers or query markers are inserted into the language prefix.
    text=text.split('Endpoint at +0.5s from CURRENT origin')[0].rstrip()
    return objective.instruction(variant),text


def status(record,device,dtype):
    obs=record['observation'];nav=torch.tensor(obs['navigation_xy'],device=device,dtype=dtype)
    speed=torch.tensor([obs['speed_mps']],device=device,dtype=dtype)
    if nav.shape!=(3,2) or not torch.isfinite(nav).all() or not torch.isfinite(speed).all():
        raise ValueError('invalid_phase3_status')
    return dict(speed=speed,target_point=nav[0:1],target_point_next=nav[1:2],final_goal=nav[2:3])


def sample(record,data_root):
    from pathlib import Path
    return dict(scenario=record['scenario'],run_id=record['route_id'],route_id=record['route_id'],
        route_dir=str(Path(data_root)/record['scenario']/record['route_id']),anchor=record['frame'],
        rgb_frame_count=1,rgb_frame_step=1,bev_frame_count=1,bev_frame_step=1,
        frame_interval_s=.25,tp_mode='route_lookahead',use_final_goal=True,
        future_waypoint_indices=[5,10,15,20,25,30,35,40],abc_record=record)


class Runtime:
    def __init__(self,args,device,branch,variant):
        self.inner=common.make_runtime(args,device,branch)
        self.device=device;self.branch=branch;self.variant=variant;self.last_audit=None;self.intervention=None;self.args=args

    def forward_sample(self,sample,decoder,decoder_config,decoder_dtype,**kwargs):
        timing=not getattr(decoder,'training',True)
        if timing and self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        record=sample['abc_record'];condition=tensors(record,self.variant,self.device)
        self.last_processor_shape=None
        if kwargs.get('clip') is None:
            from .loader import Dataset
            prepared=Dataset([sample],self.args)[0]
            if prepared.get('_error'):raise ValueError(prepared['_error'])
            kwargs['clip']=prepared['clip']
        if self.intervention=='replace_regions':condition['geometry_token_ids']=(condition['geometry_token_ids']+1)%9
        elif self.intervention=='permute_horizons':condition['geometry_token_ids']=condition['geometry_token_ids'].flip(1)
        elif self.intervention=='no_future':condition['geometry_token_ids']=torch.full_like(condition['geometry_token_ids'],9)
        elif isinstance(self.intervention,dict):
            role=self.intervention['role'];slot=self.intervention['slot']
            if role not in ('future','history') or type(slot) is not int or not 0<=slot<4:
                raise ValueError('invalid_slot_intervention')
            key='geometry_token_ids' if role=='future' else 'history_token_ids'
            ids=condition[key].clone();ids[:,slot]=torch.where(ids[:,slot]==9,0,(ids[:,slot]+1)%9);condition[key]=ids
            if role=='history':
                condition['history_mask']=condition['history_mask'].clone();condition['history_mask'][:,slot]=True
        elif self.intervention is not None:raise ValueError('unknown_condition_intervention')
        current=status(record,self.device,decoder_dtype)
        def wrapped(**inputs):
            self.last_decoder_shapes=dict(bev=list(inputs['bev'].shape) if inputs.get('bev') is not None else None,
                prefix_kv=[list(k.shape) for k,v in inputs['pooled_kv']])
            if inputs.pop('action_token_id',None) is not None:raise ValueError('semantic_condition_forbidden')
            inputs.update(current);inputs.update(condition)
            return decoder_forward(decoder,inputs,decoder_dtype,self.device)
        runner=self.inner.runner
        saved=None
        if self.branch=='qwen_simple':
            from qwen3vl_local.action_prior.image_condition import current_base_prefill
            system,text=observation_text(record,self.variant)
            saved=runner._run_leadmot_qwen_prefill
            def prefill(rgb_pil_list,user_prompt):
                if getattr(getattr(self,'args',None),'operation',None)!='probe':
                    return current_base_prefill(runner,rgb_pil_list,text,system)
                from .probe_inputs import shape
                engine=runner.leadmot_qwen_engine;prepare=engine.prepare_inputs
                def observed(*a,**kw):
                    packed=prepare(*a,**kw);self.last_processor_shape=shape(packed,engine.processor);return packed
                engine.prepare_inputs=observed
                try:return current_base_prefill(runner,rgb_pil_list,text,system)
                finally:engine.prepare_inputs=prepare
            runner._run_leadmot_qwen_prefill=prefill
        try:
            outputs=self.inner.forward_sample(sample,wrapped,decoder_config,decoder_dtype,**kwargs)
        finally:
            if saved is not None:runner._run_leadmot_qwen_prefill=saved
        outputs['input_status']=torch.cat([current['speed'][:,None],current['target_point'],
                                         current['target_point_next'],current['final_goal']],dim=1).detach()
        self.last_audit=dict(samples=1,condition_source='oracle',variant=self.variant,branch=self.branch)
        if timing:
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            self.last_audit.update(forward_seconds=time.perf_counter()-started,
                                   first_eval_forward=getattr(self,'eval_calls',0)==0)
            self.eval_calls=getattr(self,'eval_calls',0)+1
        return outputs
