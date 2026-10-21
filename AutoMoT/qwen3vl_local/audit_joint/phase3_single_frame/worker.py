"""Explicit A-only worker: CPU prepare, four-rank probe/train, fixed val/test.

Every operation is explicit. This entry never invokes the discontinued B arm.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import os
import torch
from . import dataset, planning, objective, runtime
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base
from qwen3vl_local.audit_joint.phase3_suite import model_snapshot, check_model_snapshot
from qwen3vl_local.audit_joint.phase3_base_dense import baseline, evaluation
from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool
from qwen3vl_local.sft_new_loop_phase3 import train


def planned(job,artifact):
    rows=[r for split in ('train','val','test') for r in train._read_rows(Path(job['index']),split,data_root=Path(job['data_root']))]
    args=runtime.make_args(job,artifact);report={};needed={};history=None
    for epoch in range(3):
        work,audit=planning.plan(rows,artifact['records'],args,epoch=epoch,history=history,strict=False)
        report[f'epoch_{epoch}']=audit;history=audit['next_history']
        if audit['errors']:raise ValueError(str(audit['errors']))
        if len(work)!=10240:raise ValueError('Single-frame epoch budget changed')
        coverage=planning.evaluation_coverage(artifact['records'][dataset.identity(x.row)] for x in work)
        if any(n<2 for n in coverage['distinct_targets'].values()):
            raise ValueError(f'train epoch {epoch} needs two distinct supported pairs per event/window: {coverage}')
        report[f'epoch_{epoch}']['window_coverage']=coverage
        for item in work:needed[dataset.identity(item.row)]='train'
    native=Pool(rows)
    for split in ('val','test'):
        selected=evaluation.selected(native,split)
        ids=[dataset.identity(item.row) for item in selected]
        coverage=planning.evaluation_coverage(artifact['records'][i] for i in ids)
        if any(n<2 for n in coverage['distinct_targets'].values()):
            raise ValueError(f'{split} needs two distinct supported pairs per event/window: {coverage}')
        report[split]=dict(coverage=coverage,ordered_ids=ids)
        for key in ids:needed[key]='evaluation'
    loss_ids=[dataset.identity(x.row) for x in evaluation.selected(native,'val','loss')]
    for key in loss_ids:needed[key]='evaluation'
    report['fixed_loss_ids']=loss_ids
    _,report['fixed_loss_admission']=runtime.loss_members(evaluation.selected(native,'val','loss'),artifact['records'])
    loss_coverage=planning.evaluation_coverage(artifact['records'][i] for i in loss_ids)
    report['fixed_loss_coverage']=loss_coverage
    for event in train.CONTEXT_IDS:
        if not sum(n for k,n in loss_coverage['valid'].items() if k.startswith(event+'/')):
            raise ValueError('Fixed loss evaluation has no supervised window for '+event)
    return report,needed


def preflight(job,artifact,processor):
    # The native trainer repeats these checks. Do them before any GPU probe,
    # rather than discovering an incompatible manifest after lengthy packing.
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index, check_model
    native_args=runtime.make_args(job,artifact)
    native_admission=dict(index=check_index(job['index'],action_output_mode='choice',
        prompt_variant=native_args.prompt_variant),model=check_model(job['model_dir']))
    report,needed=planned(job,artifact)
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    first_record=artifact['records'][next(iter(needed))]
    tokenizer_check=objective.tokenizer_admission(bundle,first_record,train._load_images([first_record['image']])[0])
    shapes={}
    for key,usage in needed.items():
        rec=artifact['records'][key]
        image=train._load_images([rec['image']])[0]
        variants=runtime.EVALUATION_VARIANTS if usage=='evaluation' else {'full':('filled','actual')}
        for variant,(hm,im) in variants.items():
            view,picture=runtime.evaluation_input(rec,image,hm,im)
            packed=objective.pack(bundle,view,picture)
            shape_key=key if usage=='train' else key+':'+variant
            shapes[shape_key]=dict(record_id=key,history_mode=hm,image_mode=im,usage=usage,
                tokens=packed['tokens'],positions=packed['positions'],
                grid=packed['inputs']['image_grid_thw'].tolist(),
                patches=int(packed['inputs']['image_grid_thw'].prod()))
    probes=[]
    for usage in ('train','evaluation'):
        # One image and exactly four logit rows: token length/vision patches
        # determine the variable workload; retain a Pareto frontier per role.
        candidates={}
        for key,s in shapes.items():
            if s['usage']==usage:candidates.setdefault((s['tokens'],s['patches']),key)
        for cost,key in candidates.items():
            if not any(other!=cost and all(a>=b for a,b in zip(other,cost)) for other in candidates):probes.append(key)
    recovery=check_resume(job,artifact) if job.get('resume') else None
    result=dict(status='cpu_admitted',native_admission=native_admission,plans=report,shapes=shapes,probe_ids=probes,resume=recovery,
                action_token_identity=tokenizer_check,
                contract=runtime.make_args(job,artifact).single_frame_contract)
    trial.write(Path(job['output'])/'single_frame_preflight.json',result)
    return result


def bundle(job,device):
    return train.load_model_with_lora(Path(job['model_dir']),device=device,
        lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',
        strict_vision_scope=True,gradient_checkpointing=True)


def probe_periodic_loss(job,artifact,admitted,bundle,*,rank,world,device):
    """Exercise the exact native periodic-validation path before any training."""
    runtime.install(artifact,Path(job['output']))
    rows=train._read_rows(Path(job['index']),'val',data_root=Path(job['data_root']))
    work=evaluation.selected(Pool(rows),'val','loss')
    if [dataset.identity(x.row) for x in work]!=admitted['plans']['fixed_loss_ids']:
        raise ValueError('Periodic probe fixed loss members changed')
    _,coverage=runtime.loss_members(work,artifact['records'])
    if coverage!=admitted['plans']['fixed_loss_admission']:
        raise ValueError('Periodic probe target coverage changed')
    shard=train._split_work_for_rank(work,rank=rank,world_size=world)
    args=runtime.make_args(job,artifact)
    metrics=train.evaluate_loss(bundle,shard,history_rgb_mode=args.history_rgb_mode,
        prompt_variant=args.prompt_variant,max_length=args.max_length,format_loss_weight=args.format_loss_weight,
        device=device,world_size=world,batch_size=1)
    if (metrics['requested_samples']!=coverage['requested'] or metrics['samples']!=coverage['eligible'] or
            metrics['unlabeled_samples']!=len(coverage['unlabeled_ids'])):
        raise ValueError('Periodic probe aggregate differs from CPU coverage')
    return metrics


def probe(job,artifact,admitted):
    if os.environ.get('WORLD_SIZE')!='4':raise ValueError('Probe requires exactly four ranks')
    rank,local,world=train.setup_distributed()
    device=torch.device('cuda',local)
    try:
        b=bundle(job,device)
        b.model=torch.nn.parallel.DistributedDataParallel(b.model,device_ids=[local],find_unused_parameters=False)
        optimizer=torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95),weight_decay=0.)
        torch.cuda.reset_peak_memory_stats(device)
        results=[]
        for key in admitted['probe_ids']:
            expected=admitted['shapes'][key];rec=artifact['records'][expected['record_id']]
            view,picture=runtime.evaluation_input(rec,train._load_images([rec['image']])[0],
                expected['history_mode'],expected['image_mode'])
            packed=objective.pack(b,view,picture)
            if (packed['tokens']!=expected['tokens'] or packed['positions']!=expected['positions'] or
                    packed['inputs']['image_grid_thw'].tolist()!=expected['grid']):
                raise ValueError('Probe shape changed since CPU admission')
            if expected['usage']=='evaluation':
                b.model.eval()
                with torch.no_grad():objective.old.readout_logits(b.model,packed,device)
                results.append(dict(id=key,phase='evaluation_forward_only'))
            else:
                b.model.train()
                for _ in range(2):
                    value,_=objective.loss(b,packed)
                    if not torch.isfinite(value):raise ValueError('Nonfinite probe loss')
                    value.backward()
                    for name,p in b.model.named_parameters():
                        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                            raise ValueError('Missing/nonfinite probe gradient: '+name)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                results.append(dict(id=key,phase='training_two_optimizer_steps'))
        periodic=probe_periodic_loss(job,artifact,admitted,b,rank=rank,world=world,device=device)
        torch.cuda.synchronize(device)
        free,total=torch.cuda.mem_get_info(device);peak=torch.cuda.max_memory_reserved(device)
        result=dict(rank=rank,free_bytes=free,total_bytes=total,peak_reserved_bytes=peak,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),results=results,periodic_loss=periodic,
                    passed=free>=8*1024**3 and peak<=.85*total)
        ranks=[None]*world;torch.distributed.all_gather_object(ranks,result)
        passed=all(r['passed'] for r in ranks)
        if rank==0:trial.write(Path(job['output'])/'single_frame_probe.json',
            dict(status='passed' if passed else 'insufficient_headroom',ranks=ranks,world_size=4,
                 preflight_sha256=trial.file_sha(Path(job['output'])/'single_frame_preflight.json'),
                 contract=admitted['contract'],model_saved=False))
        if not passed:raise ValueError('Insufficient headroom; no automatic batch/prompt changes')
    finally:train.cleanup_distributed()


def check_resume(job,artifact,*,world_size=4,steps_per_epoch=2560):
    """Construct the same CPU binding the native trainer checks before loading."""
    from qwen3vl_local.sft_new_loop_phase3 import training_state, training_schedule
    args=runtime.make_args(job,artifact)
    args.training_index_sha256=job['index_sha256']
    args.learning_rate_schedule=training_schedule.resolve_warmup(args.num_epochs*steps_per_epoch,steps=args.warmup_steps,ratio=args.warmup_ratio)
    args.warmup_steps=args.learning_rate_schedule['warmup_updates']
    base.install_reference(trial.reference())
    args.validation_sampling={}
    if args.eval_steps>0:
        rows=train._read_rows(Path(job['index']),'val',data_root=Path(job['data_root']))
        _,audit=train._validation_work(rows,equal_events=True,history_rgb_mode=args.history_rgb_mode,
            target_per_bin=args.eval_balance_count,seed=args.seed+1009,
            auto_increase=args.auto_eval_balance_count,require_invalid_coverage=args.require_invalid_coverage,
            action_output_mode=args.action_output_mode)
        args.eval_balance_count=audit['effective'];args.validation_sampling={'loss':audit}
    expected=training_state.binding(args,world_size,steps_per_epoch)
    state=training_state.read(Path(job['resume']),expected)
    if state['next_epoch']!=job['resume_epoch']:raise ValueError('Resume epoch differs from job')
    rows=train._read_rows(Path(job['index']),'train',data_root=Path(job['data_root']))
    planning.plan(rows,artifact['records'],args,epoch=state['next_epoch'],history=state['sampling_history'],world_size=world_size)
    return dict(checkpoint=job['resume'],receipt_sha256=trial.file_sha(Path(job['resume'])/'receipt.json'),
                next_epoch=state['next_epoch'],global_step=state['global_step'],remaining_epochs=3-state['next_epoch'],
                optimizer_scheduler_rng_verified=True,boundary='completed_epoch_only')


def evaluate_variants(b,records,output,split):
    """Full inputs and three matched inference interventions, all on fixed IDs."""
    output=Path(output);cases={};summaries={}
    for name,(hm,im) in runtime.EVALUATION_VARIANTS.items():
        cases[name],summaries[name]=runtime.predict(b,records,history_mode=hm,image_mode=im)
        suffix='' if name=='full' else '_'+name
        trial.write(output/f'single_frame_{split}{suffix}_cases.json',cases[name])
        if name!='full':trial.write(output/f'single_frame_{split}{suffix}_metrics.json',summaries[name])
    metrics=summaries['full']
    metrics['paired_input_ablations']={name:objective.paired_comparison(cases['full'],rows)
        for name,rows in cases.items() if name!='full'}
    metrics['history_ablation']='completed: no_history and native_history on identical anchors/windows'
    metrics['image_ablation']='completed: same-size black RGB, retaining history and all text; not a retraining comparison'
    return metrics


def evaluate(job,artifact,admitted,split):
    from qwen3vl_local.sft_new_loop_phase3 import training_state
    root=Path(job['output']);checkpoint=root/'train/training_states/epoch_0003'
    receipt=trial.read(checkpoint/'receipt.json')
    if receipt['files']!=training_state.files(checkpoint):raise ValueError('Checkpoint files changed')
    state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
    if (state['binding']!=receipt['binding'] or state['global_step']!=7680 or state['next_epoch']!=3 or
            state['binding']['configuration']['single_frame_contract']!=admitted['contract']):
        raise ValueError('Checkpoint does not match the completed single-frame protocol')
    b=bundle(job,torch.device('cuda:0'))
    params={n:p for n,p in b.model.named_parameters() if p.requires_grad}
    if set(params)!=set(state['trainable']):raise ValueError('Trainable parameter topology changed')
    with torch.no_grad():
        for n,p in params.items():p.copy_(state['trainable'][n].to(device=p.device,dtype=p.dtype))
    records=[artifact['records'][i] for i in admitted['plans'][split]['ordered_ids']]
    metrics=evaluate_variants(b,records,root,split)
    metrics['coverage']=admitted['plans'][split]['coverage']
    metrics['history_policy']=artifact['config']['history_policy']
    metrics['post_observation_history_evidence_allowed']=True
    trial.write(root/f'single_frame_{split}_metrics.json',metrics)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('operation',choices=('prepare','probe','train','evaluate'))
    p.add_argument('--job',type=Path)
    for name in ('index','data-root','model-dir','dataset','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--resume',type=Path,help='prepare only: explicit completed epoch 1/2 checkpoint; use a new output')
    a=p.parse_args()
    if a.operation=='prepare':
        if not all(getattr(a,n) for n in ('index','data_root','model_dir','dataset','output')):
            p.error('prepare requires index/data-root/model-dir/dataset/output')
        if a.output.exists():raise ValueError('Output exists; use a new explicit directory')
        if a.output.resolve().is_relative_to(a.dataset.resolve().parent):raise ValueError('Output overlaps dataset')
        cfg=SimpleNamespace(index=a.index,data_root=a.data_root,model_dir=a.model_dir,
                            resume=a.resume,stage='complete',experiment='base')
        job=trial.build_job(cfg,a.output.resolve())
        job.update(dataset=str(a.dataset.resolve(strict=True)),model_snapshot=model_snapshot(a.model_dir.resolve()))
        artifact=dataset.load(job['dataset'],index=job['index'])
        if artifact['data_root']!=job['data_root']:raise ValueError('Dataset raw root mismatch')
        if a.resume:check_resume(job,artifact)
        a.output.mkdir(parents=True)
        trial.write(a.output/'single_frame_job.json',job)
        from qwen3vl_local.qwen35.backend import LocalProcessor
        preflight(job,artifact,LocalProcessor.from_pretrained(job['model_dir'],local_files_only=True))
        return
    if not a.job:p.error('--job is required')
    if a.resume:p.error('--resume is only accepted by prepare; use the prepared resume job')
    job=trial.read(a.job);base.validate_job(job)
    check_model_snapshot(Path(job['model_dir']),job['model_snapshot'])
    artifact=dataset.load(job['dataset'],index=job['index'])
    root=Path(job['output']);admitted=trial.read(root/'single_frame_preflight.json')
    if admitted['contract']!=runtime.make_args(job,artifact).single_frame_contract:
        raise ValueError('CPU admission contract changed')
    if a.operation=='probe':probe(job,artifact,admitted)
    elif a.operation=='train':
        if os.environ.get('WORLD_SIZE')!='4':raise ValueError('Training requires four GPUs, B1/accum1, 7680 updates')
        checked=trial.read(root/'single_frame_probe.json')
        if (checked['status']!='passed' or checked['world_size']!=4 or checked['contract']!=admitted['contract'] or
                checked['preflight_sha256']!=trial.file_sha(root/'single_frame_preflight.json')):
            raise ValueError('Matching successful four-rank probe required')
        if job.get('resume') and check_resume(job,artifact)!=admitted.get('resume'):
            raise ValueError('Resume checkpoint changed after admission')
        # Coordinate the output guard before any rank creates train/. A separate
        # per-rank exists() check would race with rank 0 creating the directory.
        rank,local,world=train.setup_distributed()
        exists=torch.tensor(int((root/'train').exists()) if rank==0 else 0,device=torch.device('cuda',local))
        torch.distributed.broadcast(exists,src=0)
        if exists.item():
            train.cleanup_distributed()
            raise ValueError('Existing training output; no overwrite or implicit restart')
        train.setup_distributed=lambda *args,**kwargs:(rank,local,world)
        try:
            runtime.install(artifact,root)
            train.train(runtime.make_args(job,artifact))
        finally:
            if torch.distributed.is_initialized():train.cleanup_distributed()
    else:
        if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Evaluation is a single-process operation')
        evaluate(job,artifact,admitted,a.split)


if __name__=='__main__':main()
