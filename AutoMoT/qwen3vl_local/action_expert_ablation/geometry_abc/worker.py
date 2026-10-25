"""Dedicated workers reuse the original optimizer/DDP/EMA/FM training loop."""
import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import os
import time
from qwen3vl_local.audit_joint.abc_delivery import digest,sha,write
from . import contracts,data,model as abc_model,runtime as abc_runtime


def hooks():
    from qwen3vl_local.action_prior.training_core import MetricHooks
    def counts(runtime,sample,metrics):
        r=sample['abc_record'];out=Counter(samples=1)
        proof=sample.get('consumed_sources',{})
        if set(proof)!=set(('metas','rgb','lidar')) or any(len(v)!=64 for v in proof.values()):
            raise ValueError('missing_consumed_source_proof')
        out['verified_exposure/'+r['id']]=1
        tracker=getattr(runtime,'gradient_evidence',None)
        if tracker is not None and getattr(runtime,'training_evidence_active',False):tracker.observe(r,runtime.variant)
        out['exposure/'+r['id']]=1
        for prefix in ('event/'+r['event'],'physical/'+r['physical_group']):
            out[prefix+'/samples']+=1
            for k,v in metrics.items():out[prefix+'/'+k]+=v
        for variant in 'abc':
            for h,t in enumerate(r['representations'][variant]['targets']):
                out[f'region/{variant}/{h}/{t["token_id"]}']+=1
        return out
    def summarize(counts):
        n=counts['samples']
        if not n:raise ValueError('no_evaluated_samples')
        result={k:v/n for k,v in counts.items() if '/' not in k and k!='samples'}
        result['samples']=n
        for scope in ('event','physical'):
            groups=defaultdict(dict)
            for key,value in counts.items():
                if key.startswith(scope+'/'):
                    group,metric=key.rsplit('/',1);groups[group][metric]=value
            for group,values in groups.items():
                for k,v in values.items():result[group+'/'+k]=v if k=='samples' else v/values['samples']
            metrics={k for values in groups.values() for k in values if k!='samples'}
            for k in metrics:
                values=[v[k]/v['samples'] for group,v in groups.items()
                        if k in v and (scope!='event' or group!='event/'+contracts.BACKGROUND)]
                if values:result[scope+'_macro/'+k]=sum(values)/len(values)
        return result
    def case(runtime,sample):
        r=sample['abc_record']
        return dict(id=r['id'],event=r['event'],physical_group=r['physical_group'],
            speed_mps=r['observation']['speed_mps'],region_ids=[t['token_id'] for t in r['representations'][runtime.variant]['targets']],
            condition_source='oracle',variant=runtime.variant,branch=runtime.branch,
            consumed_sources=sample['consumed_sources'],
            geometry_projection=sample['audit_geometry'],inference=runtime.last_audit)
    return MetricHooks(sample_counts=counts,summarize=summarize,case_record=case,
        format_metrics=lambda x:f"FM={x['loss']:.6f}",epoch_audit=True)


def validation_members(rows,seed,max_samples=0):
    from qwen3vl_local.action_prior.event_balance import equal_event_evaluation
    from qwen3vl_local.action_prior.build_event_balance_index import CONTEXT_TO_BUCKET
    merged={};lookup={}
    for r in rows:
        rec=r['abc_record'];e=rec['event']
        if e==contracts.BACKGROUND:continue
        key=(r['scenario'],r['run_id'],r['anchor']);bucket=CONTEXT_TO_BUCKET[e]
        lookup[key,bucket]=r
        if key not in merged:merged[key]=dict(r,event_balance_all_special_buckets=[])
        merged[key]['event_balance_all_special_buckets'].append(bucket)
    selected,plan=equal_event_evaluation(list(merged.values()),seed=seed,max_samples=max_samples)
    return [lookup[(r['scenario'],r['run_id'],r['anchor']),r['_evaluation_event']] for r in selected],plan


def evaluate(runtime,decoder,config,rows,args,dtype,rank,world,max_samples=0,dump_dir=None):
    from qwen3vl_local.action_expert_ablation.common import leadmot_train
    from qwen3vl_local.action_prior.training_core import evaluate as shared
    local=copy.copy(args);local.evaluation_distribution='natural'
    selected=rows
    if getattr(args,'abc_eval_scope','val')=='val':selected,_=validation_members(rows,args.seed+71,max_samples)
    from .loader import Helpers
    active=getattr(runtime,'training_evidence_active',False);runtime.training_evidence_active=False
    try:
        return shared(runtime,decoder,config,selected,local,dtype,rank,world,0,dump_dir,
                      old=Helpers(leadmot_train()),hooks=hooks())
    finally:runtime.training_evidence_active=active


@contextmanager
def planned_sampler(projection,seed):
    from qwen3vl_local.action_prior import event_balance
    original=event_balance.build_balanced_epoch
    def build(rows,*,mode,total,seed: int,**kwargs):
        epoch=seed-base_seed
        if mode!='abc_regions' or not 0<=epoch<len(projection['plans']):raise ValueError('wrong_ABC_epoch_request')
        p=projection['plans'][epoch]
        if total!=p['samples']:raise ValueError('ABC_budget_changed')
        lookup={r['abc_record']['id']:r for r in rows}
        return [lookup[k] for k in p['ordered_ids']],{k:v for k,v in p.items() if k!='ordered_ids'}
    base_seed=seed;event_balance.build_balanced_epoch=build
    try:yield
    finally:event_balance.build_balanced_epoch=original


def parser():
    from qwen3vl_local.action_expert_ablation.common import parser as original
    # Copy the original arguments, but not its semantic-only parse override.
    p=argparse.ArgumentParser(parents=[original('qwen_simple')],add_help=False,allow_abbrev=False)
    p.add_argument('--abc-variant',choices=list('abc'),required=True)
    p.add_argument('--branch',choices=['bev_only','qwen_simple'],required=True)
    p.add_argument('--operation',choices=['train','probe','evaluate'],default='train')
    p.add_argument('--checkpoint',default='')
    p.add_argument('--suite-root',required=True)
    p.set_defaults(rgb_frame_count=1,high_level_action_token=False,action_token_separation_weight=0.,
                   sampling_mode='abc_regions',evaluation_distribution='natural',event_balance_max_frame_repeats=11)
    return p


def validate_args(args):
    expected=dict(num_epochs=7,event_balanced_epoch_samples=116256,grad_accum_steps=16,
                  rgb_frame_count=1,bev_frame_count=1,route_points=10,waypoint_points=8,
                  high_level_action_token=False,action_token_separation_weight=0.,
                  event_balance_max_frame_repeats=11,seed=2026,max_train_steps=0)
    expected.update(sampling_mode='abc_regions',frame_interval_s=.25,rgb_frame_step=1,bev_frame_step=1)
    # The original parser uses None for cap to infer semantic sampling defaults.
    if args.event_balance_max_frame_repeats is None:args.event_balance_max_frame_repeats=11
    for k,v in expected.items():
        if getattr(args,k)!=v:raise ValueError('fixed_ABC_budget_or_input_changed: '+k)
    if args.resume:raise ValueError('implicit_resume_not_supported_use_original_validated_run')
    for name in ('qwen_adapter_dir','init_from_ckpt','phase1_adapter','phase2_adapter','lora_bundle'):
        if getattr(args,name,None):raise ValueError('prior_task_weights_forbidden: '+name)
    if args.operation!='evaluate' and args.checkpoint:
        raise ValueError('checkpoint_only_allowed_for_this_run_evaluation')
    if not args.use_bev or not args.use_final_goal or args.use_subgoal:raise ValueError('ABC_observation_switch_mismatch')
    if args.generate_analysis or args.event_balanced_scene_priors:raise ValueError('ABC_no_generated_or_semantic_scene_priors')
    args.condition_mode=args.branch


def numerical_config(args):
    keys=('optimizer','lr_scheduler','learning_rate','weight_decay','warmup_ratio','grad_accum_steps',
          'max_grad_norm','val_steps','val_max_samples','seed','best_selection_metric','decoder_dtype',
          'qwen_dtype','smooth_route','leadmot_rope_type','decoder_dropout','route_loss_weight',
          'waypoint_loss_weight','loss_type','flow_route_coordinate_scale_m','flow_waypoint_coordinate_scale_m',
          'flow_time_embed_dim','flow_sample_steps','flow_trajectory_layers','flow_trajectory_heads','ema_decay')
    return {key:getattr(args,key) for key in keys}


def main(argv=None):
    args=parser().parse_args(argv);validate_args(args)
    # Reject version mismatch before importing the runner or allocating NCCL/GPU resources.
    from .recovery import require_current
    require_current(json.loads((Path(args.suite_root)/'admission.json').read_text()))
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_prior import training_core as core
    from qwen3vl_local.action_prior.flow_matching import FlowMatchingConfig,make_training_flow,flow_matching_loss
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    import torch
    import torch.distributed as dist
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[name]='1'
    from .loader import Helpers,Dataset
    old=Helpers(common.leadmot_train());rank,local,world=old._init_distributed()
    if world!=4:raise ValueError('ABC_requires_four_ranks_including_probe')
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    try:
        suite=Path(args.suite_root)
        gate=json.loads((suite/'admission.json').read_text())
        if not all(gate.get(k) for k in ('data_ready','plan_ready','package_ready','processor_ready','raw_sources_verified','history_ready','development_ready')):
            raise ValueError('all_CPU_gates_required_before_probe_or_training')
        require_current(gate)
        from .preflight import resources
        resource=[resources(args.model_dir,args.lead_bev_ckpt) if rank==0 else None]
        dist.broadcast_object_list(resource,src=0)
        if resource[0]!=gate['resources'] or not resource[0]['ready']:raise ValueError('model_resources_changed')
        from .consumption import checked_bytes
        manifest_path=Path(args.data_dir)/'manifest.json'
        admitted_manifest=json.loads(checked_bytes(manifest_path,gate['manifest_sha256']))
        rows,manifest=data.load(args.data_dir,data_root=args.data_root,check_raw=False)
        if manifest!=admitted_manifest:raise ValueError('manifest_changed_after_admission')
        from .preflight import verify_development
        verify_development(manifest)
        if gate['dataset']!=manifest['identity'] or gate['splits']!=manifest['splits']:raise ValueError('admitted_dataset_changed')
        args.abc_dataset_identity=manifest['identity']
        args.abc_manifest_sha256=gate['manifest_sha256']
        from .probe_inputs import verify_processor,selection,verify_probe_inputs,verify_actual_shape
        processor=verify_processor(suite/'processor.json',gate)
        flat=[r for values in rows.values() for r in values]
        probe_selections={v:selection(flat,processor,v) for v in 'abc'}
        del flat
        if args.operation=='train':
            from .audit import GROUPS
            for group in GROUPS:
                p=suite/'probes'/group/'probe_complete.json'
                receipt=json.loads(p.read_text())
                if receipt['admission_sha256']!=sha(suite/'admission.json'):raise ValueError('six_current_probes_required')
                variant,branch=group.split('_',1);proof=receipt['binding']
                verify_probe_inputs(receipt,probe_selections[variant],gate['processor_sha256'],p.parent)
                if (proof['representation']!=contracts.contract(variant) or proof['branch']!=branch
                    or proof['source']!=gate['code'] or proof['dataset']!=gate['dataset']
                    or proof['resources']!=gate['resources'] or not receipt.get('checkpoint_reload_verified')
                    or receipt['checkpoint_sha256']!=sha(p.with_name('probe.pt'))):
                    raise ValueError('six_probe_contract_or_checkpoint_changed')
        projection=json.loads((Path(args.data_dir)/'plan.json').read_text())
        from .preflight import plan_digest
        stored=projection['sha256']
        if plan_digest(projection)!=stored or gate['plan_sha256']!=stored:raise ValueError('plan_digest_changed')
        if projection['total_updates']!=12719 or projection['total_presentations']!=813792:raise ValueError('wrong_ABC_plan')
        rows={s:[abc_runtime.sample(r,args.data_root) for r in values] for s,values in rows.items()}
        out=Path(args.output_dir)
        config=LeadMoTPlanningDecoderConfig(num_route_queries=10,num_waypoint_queries=8,
            rope_type=args.leadmot_rope_type,dropout=args.decoder_dropout,use_bev=True,
            use_geometry_tokens=True,use_high_level_action_token=False,use_final_goal=True,use_subgoal=False)
        fc=FlowMatchingConfig(route_coordinate_scale_m=args.flow_route_coordinate_scale_m,
            waypoint_coordinate_scale_m=args.flow_waypoint_coordinate_scale_m,time_embed_dim=args.flow_time_embed_dim,
            sample_steps=args.flow_sample_steps,trajectory_layers=args.flow_trajectory_layers,trajectory_heads=args.flow_trajectory_heads)
        torch.manual_seed(args.seed)
        model=abc_model.ABCFlowDecoder(config,fc,variant=args.abc_variant).to(device)
        dtype=old._dtype(args.decoder_dtype)
        runtime=abc_runtime.Runtime(args,device,args.branch,args.abc_variant)
        binding=dict(representation=contracts.contract(args.abc_variant),branch=args.branch,
            consumption_sources=dict(manifest_sha256=gate['manifest_sha256'],protocol='same_hashed_bytes_decode_v1'),
            source=contracts.source_identity(),dataset=manifest['identity'],splits=manifest['splits'],plan=stored,
            resources=resource[0],numerical_config=numerical_config(args))
        if args.operation=='evaluate':
            from qwen3vl_local.action_prior.progress import current
            if current():current().configure(out,'evaluate')
            completion=json.loads((Path(args.checkpoint).parent/'training_complete.json').read_text())
            if (completion['step']!=12719 or completion['best_sha256']!=sha(args.checkpoint)
                or completion['latest_sha256']!=sha(Path(args.checkpoint).parent/'latest.pt')):
                raise ValueError('test_requires_completed_seven_epoch_budget')
            state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if state['schema']!='abc_trajectory_checkpoint_v1' or state['condition_contract']!=binding:
                raise ValueError('ABC_checkpoint_binding_changed')
            model.load_state_dict(state['decoder'],strict=True)
            ema=old._DecoderEMA(model,args.ema_decay);ema.load_state_dict(state['ema_state_dict'])
            args.abc_eval_scope='test'
            with ema.apply_to(model):
                result=evaluate(runtime,model,config,rows['test'],args,dtype,rank,world,0,out/'cases')
                from .diagnostics import sensitivity
                sensitivity(runtime,model,config,rows['val'],args,dtype,rank,world,out/'condition_sensitivity.json')
            if rank==0:
                diagnostics=json.loads((out/'condition_sensitivity.json').read_text())
                diagnostics.update(checkpoint_sha256=sha(args.checkpoint),binding=binding)
                write(out/'condition_sensitivity.json',diagnostics)
                write(out/'metrics.json',dict(metrics=result,checkpoint_sha256=sha(args.checkpoint),binding=binding))
            return
        if rank==0:out.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        from qwen3vl_local.action_prior.progress import current
        if current():current().configure(out,args.operation)
        decoder=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local])
        if args.operation=='probe':
            probe_plan=dict(samples_per_epoch=116256,optimizer_steps_per_epoch=1817,actual_step_limit=12719,per_event=9688)
            optimizer,scheduler,ema=core.make_optimization(args,model,probe_plan,old)
            input_selection=probe_selections[args.abc_variant]
            selected=input_selection['backward'][rank]
            chosen=next(r for r in rows['train'] if r['abc_record']['id']==selected['id'])
            if chosen['abc_record']['split']!='train':raise ValueError('backward_probe_train_only')
            prepared=Dataset([chosen],args)[0]
            if prepared.get('_error'):raise ValueError(prepared['_error'])
            gt_r=prepared['gt_route'][None].to(device);gt_w=prepared['gt_waypoints'][None].to(device)
            flow=make_training_flow(gt_r,gt_w,fc)
            result=runtime.forward_sample(chosen,decoder,config,dtype,clip=prepared['clip'],
                flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
            verify_actual_shape(runtime.last_processor_shape,selected,args.branch)
            loss=flow_matching_loss(result,flow['flow_target_velocity'],10,args.route_loss_weight,args.waypoint_loss_weight)[0]
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('probe_nonfinite_or_disconnected_gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm,error_if_nonfinite=True)
            optimizer.step();scheduler.step();ema.update(model)
            input_proof=dict(id=selected['id'],split='train',processor=runtime.last_processor_shape,
                rgb_shape=list(prepared['clip']['rgb'].shape),lidar_points=[len(x) for x in prepared['clip']['lidar_points']],
                decoder_shapes=runtime.last_decoder_shapes)
            holdout_proofs=[];model.eval()
            try:
                for item in input_selection['holdout_forward_only'][rank::world]:
                    sample=next(r for r in rows[item['split']] if r['abc_record']['id']==item['id'])
                    if sample['abc_record']['split']=='train':raise ValueError('holdout_probe_member_changed')
                    prepared_holdout=Dataset([sample],args)[0]
                    if prepared_holdout.get('_error'):raise ValueError(prepared_holdout['_error'])
                    with torch.no_grad():
                        prediction=runtime.forward_sample(sample,model,config,dtype,clip=prepared_holdout['clip'],sample_trajectory=True)
                    if not torch.isfinite(prediction['pred_future_waypoints']).all():raise ValueError('nonfinite_holdout_forward')
                    verify_actual_shape(runtime.last_processor_shape,item,args.branch)
                    holdout_proofs.append(dict(item,updates=0,decoder_shapes=runtime.last_decoder_shapes))
            finally:model.train()
            write(out/f'rank{rank}.json',dict(loss=float(loss),binding=binding,embedding=abc_model.embedding_diagnostics(model),
                input=input_proof,holdout_forward_only=holdout_proofs))
            core.save_training_checkpoint(out/'probe.pt',model,optimizer,scheduler,ema,args,binding,manifest['splits'],
                {'epoch':0,'micro':1},1,math.inf,rank,world,
                schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
            dist.barrier()
            saved=torch.load(out/'probe.pt',map_location='cpu',weights_only=False)
            if saved['condition_contract']!=binding or saved['step']!=1:raise ValueError('probe_checkpoint_binding_changed')
            model.load_state_dict(saved['decoder'],strict=True);ema.load_state_dict(saved['ema_state_dict'])
            del saved
            dist.barrier()
            if rank==0:
                receipt=dict(binding=binding,ranks=4,formal_training=False,input_selection=input_selection,
                    processor_sha256=gate['processor_sha256'],rank_receipts={f'rank{i}.json':sha(out/f'rank{i}.json') for i in range(world)},
                    admission_sha256=sha(suite/'admission.json'),checkpoint_reload_verified=True,checkpoint_sha256=sha(out/'probe.pt'))
                verify_probe_inputs(receipt,input_selection,gate['processor_sha256'],out)
                write(out/'probe_complete.json',receipt)
            return
        plan=dict(samples_per_epoch=116256,optimizer_steps_per_epoch=1817,actual_step_limit=12719,per_event=9688)
        optimizer,scheduler,ema=core.make_optimization(args,model,plan,old)
        from .gradients import Evidence
        gradients=Evidence(model,optimizer,rank);runtime.gradient_evidence=gradients;runtime.training_evidence_active=True
        if rank==0:
            write(out/'config.json',vars(args));write(out/'condition_contract.json',binding);write(out/'training_plan.json',plan)
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(out/'tb') if rank==0 else None
        def save(path,cursor,step,best):
            core.save_training_checkpoint(path,model,optimizer,scheduler,ema,args,binding,manifest['splits'],
                cursor,step,best,rank,world,schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
            write(out/'token_diagnostics'/f'step_{step:08d}_rank{rank}.json',gradients.snapshot(step))
        training_started=time.monotonic()
        with planned_sampler(projection,args.seed):
            stopped=core.run_training_loop(args=args,rows=rows,plan=plan,runtime=runtime,model=model,decoder=decoder,
                config=config,flow_config=fc,optimizer=optimizer,scheduler=scheduler,ema=ema,
                cursor={'epoch':0,'micro':0},step=0,best=math.inf,device=device,dtype=dtype,
                rank=rank,world=world,out=out,writer=writer,old=old,hooks=hooks(),evaluate_fn=evaluate,checkpoint=save)
        if stopped:raise core.GracefulTerminationExit(stopped)
        gradients.close()
        torch.cuda.synchronize(device)
        performance=dict(rank=rank,wall_seconds=time.monotonic()-training_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        performances=[None]*world;dist.all_gather_object(performances,performance)
        if rank==0:
            write(out/'performance'/'ranks.json',dict(ranks=performances,scope='full training loop including validation and checkpoint',
                actual_presentations=813792,global_presentations_per_second=813792/max(p['wall_seconds'] for p in performances)))
            receipts={}
            for epoch,p in enumerate(projection['plans'],1):
                paths=list((out/'epoch_audit').glob(f'epoch_{epoch:03d}_*.json'))
                if len(paths)!=1:raise ValueError('missing_actual_epoch_exposure')
                counts=json.loads(paths[0].read_text());actual={k.removeprefix('exposure/'):v for k,v in counts.items() if k.startswith('exposure/')}
                if actual!=dict(Counter(p['ordered_ids'])):raise ValueError('actual_exposure_differs_from_plan')
                verified={k.removeprefix('verified_exposure/'):v for k,v in counts.items() if k.startswith('verified_exposure/')}
                if verified!=actual:raise ValueError('actual_consumption_source_proof_missing')
                receipts[str(epoch)]=sha(paths[0])
            latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
            if not core.budget_complete(latest['step'],plan,latest['cursor']):raise ValueError('incomplete_budget')
            write(out/'training_complete.json',dict(binding=binding,step=latest['step'],epochs=receipts,
                token_diagnostics={p.name:sha(p) for p in sorted((out/'token_diagnostics').glob('*.json'))},
                latest_sha256=sha(out/'latest.pt'),best_sha256=sha(out/'best.pt')))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    from qwen3vl_local.action_prior.progress import observed
    observed(main)()
