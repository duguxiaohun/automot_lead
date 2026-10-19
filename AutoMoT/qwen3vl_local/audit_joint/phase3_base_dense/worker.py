"""Dense experiment worker: bound labels, full-plan admission and four-rank probe."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import torch
from . import runtime, objectives, baseline, planning, shapes, evaluation
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base


def validate(job):
    base.validate_job(job)
    if job['mode'] not in ('future','stream','windows','windows_prefix','windows_bank'):
        raise ValueError('Unknown dense mode')
    if job.get('forecast_contract')!=planning.options(job):
        raise ValueError('Dense time/coverage/supervision contract changed')
    if job.get('aux_index'):
        p=Path(job['aux_index'])
        actual=dict(index_sha256=trial.file_sha(p),manifest_sha256=trial.file_sha(p.with_name('manifest.json')))
        if actual!=job['aux_binding']:raise ValueError('Auxiliary labels/manifest changed')
    elif job.get('aux_binding'):raise ValueError('Auxiliary label path missing')
    if job.get('initialization')=='qwen':
        if job.get('init_adapter') or job.get('init_adapter_files'):
            raise ValueError('Qwen initialization contains an adapter')
        return
    if job.get('initialization','base_adapter')!='base_adapter':
        raise ValueError('Unknown initialization')
    init=Path(job['init_adapter'])
    actual={str(p.relative_to(init)):trial.file_sha(p) for p in sorted(init.rglob('*')) if p.is_file()}
    if actual!=job['init_adapter_files']:raise ValueError('Initialization adapter changed')


def planned_epochs(job,pool,args):
    start=job.get('resume_epoch',0)
    history=dict(policy=baseline.POLICY,next_epoch=start) if start else None
    plans={};records={}
    for epoch in range(start,3):
        work,audit,items=planning.make_plan(pool,args,job,epoch=epoch,world_size=4,history=history)
        if len(work)!=10240:raise ValueError('Dense event/action schedule changed')
        plans[str(epoch)]=audit;records.update(items);history=audit['next_history']
    return plans,records


def preflight(job,pool):
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index,check_model
    from qwen3vl_local.qwen35.backend import LocalProcessor
    root=Path(job['output'])
    try:
        check_index(job['index'],action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
        check_model(job['model_dir'])
        args=runtime.make_args(job)
        if args.resume:
            from qwen3vl_local.sft_new_loop_phase3 import training_state
            from qwen3vl_local.sft_new_loop_phase3.training_schedule import resolve_warmup
            base.install_reference(trial.reference())
            val=train._read_rows(Path(args.index),'val',data_root=Path(args.data_root))
            _,loss_audit=train._validation_work(val,equal_events=True,history_rgb_mode='4rgb',
                target_per_bin=16,seed=args.seed+1009,auto_increase=args.auto_eval_balance_count,
                require_invalid_coverage=args.require_invalid_coverage,action_output_mode='choice')
            args.validation_sampling={'loss':loss_audit}
            args.training_index_sha256=job['index_sha256']
            args.learning_rate_schedule=resolve_warmup(7680,steps=2000,ratio=args.warmup_ratio)
            restored=training_state.read(Path(args.resume),training_state.binding(args,4,2560))
            if (restored['next_epoch']!=job['resume_epoch'] or
                    restored['sampling_history']!=dict(policy=baseline.POLICY,next_epoch=job['resume_epoch'])):
                raise ValueError('Dense resume epoch/history mismatch')
        plans,records=planned_epochs(job,pool,args)
        eval_plan,eval_records=evaluation.build(job,pool)
        trial.write(root/'dense_evaluation_plan.json',eval_plan)
        records.update(eval_records)
        trial.write(root/'dense_coverage.json',dict(status='planned_not_consumed',contract=runtime.contract(job),
            epoch_plans=plans,auxiliary_labels_added=pool.aux_added,
            update_budget=7680,labels_rebuilt=False,RGB_copied=False))
        processor=LocalProcessor.from_pretrained(job['model_dir'],local_files_only=True)
        objectives.choice_ids(processor.tokenizer)
        report=shapes.scan(processor,records,job.get('max_batch_tokens',16384))
        report['planned_epochs_zero_based']=list(map(int,plans))
        trial.write(root/'dense_input_shapes.json',report)
    except Exception as exc:
        trial.write(root/'dense_admission_failure.json',dict(status='blocked',error=f'{type(exc).__name__}: {exc}',
            original_index=job['index'],aux_index=job.get('aux_index'),training_started=False))
        raise


def probe_record(bundle,record,packed,optimizer):
    lengths=[p['tokens'] for p in packed] if isinstance(packed,list) else [packed['tokens']]
    windows=isinstance(packed,list)
    variants=(False,True) if windows else (False,)
    results=[]
    for source_slots in variants:
        policy=('eight_source_slots' if source_slots else 'actual_reuse') if windows else 'no_reuse_nonwindow'
        if record.get('usage')=='evaluation':
            # Holdout allocation checks never compute CE, gradients or updates.
            was_training=bundle.model.training
            bundle.model.eval()
            try:
                with torch.no_grad():
                    if windows:_,visual=objectives.window_logits(bundle,packed,source_slots=source_slots)
                    else:objectives.readout_logits(bundle.model,packed,bundle.device);visual=None
            finally:bundle.model.train(was_training)
            results.append(dict(tokens=sum(lengths),phase='evaluation_forward_only',
                                vision_policy=policy,visual_reuse=visual))
            continue
        for _ in range(2):
            loss,stats=(objectives.window_loss(bundle,packed,source_slots=source_slots) if windows
                        else objectives.readout_loss(bundle.model,packed,bundle.device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss in distributed memory probe')
            loss.backward()
            for name,p in bundle.model.named_parameters():
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise ValueError('Missing/nonfinite probe gradient: '+name)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            results.append(dict(loss=float(loss.detach()),tokens=stats['tokens'],phase='train_optimizer_step',
                                vision_policy=policy,visual_reuse=stats.get('visual_reuse')))
    return results


def probe(job,pool):
    if not torch.cuda.is_available() or os.environ.get('WORLD_SIZE')!='4':
        raise ValueError('Memory/backward probe requires the same four CUDA/DDP ranks as training')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=10))
    try:
        device=torch.device('cuda',local_rank)
        bundle=runtime.create_bundle(job,device)
        bundle.model=torch.nn.parallel.DistributedDataParallel(bundle.model,device_ids=[local_rank],find_unused_parameters=False)
        bundle.model.train()
        _,records=planned_epochs(job,pool,runtime.make_args(job))
        eval_plan,eval_records=evaluation.build(job,pool)
        if eval_plan!=trial.read(Path(job['output'])/'dense_evaluation_plan.json'):
            raise ValueError('Fixed evaluation plan changed since admission')
        records.update(eval_records)
        audit=trial.read(Path(job['output'])/'dense_input_shapes.json')
        if set(audit['shapes'])!=set(records):raise ValueError('Shape audit no longer matches planned anchors')
        optimizer=torch.optim.AdamW([p for p in bundle.model.parameters() if p.requires_grad],
                                   lr=1e-5,weight_decay=0.,betas=(.9,.95))
        torch.cuda.reset_peak_memory_stats(device)
        results=[]
        for identity in audit['probe_ids']:
            record=records[identity]
            packed=objectives.pack(bundle,record,train._load_images(record['images']),max_batch_tokens=job.get('max_batch_tokens',16384))
            lengths=[p['tokens'] for p in packed] if isinstance(packed,list) else [packed['tokens']]
            positions=[p['positions'] for p in packed] if isinstance(packed,list) else [packed['positions']]
            if (lengths!=audit['shapes'][identity]['sequence_tokens'] or
                    positions!=audit['shapes'][identity]['query_positions']):
                raise ValueError('Processor tokens/query positions differ from full-plan shape audit')
            measured=probe_record(bundle,record,packed,optimizer)
            for result in measured:
                visual=result.get('visual_reuse') or {}
                for key in ('model_logit_rows','language_tokens_computed','live_cache_token_slots'):
                    if key in visual and visual[key]!=audit['shapes'][identity][key]:
                        raise ValueError('Actual cache geometry differs from admitted shape: '+key)
            results.extend(dict(id=identity,**r) for r in measured)
        torch.cuda.synchronize(device)
        free,total=torch.cuda.mem_get_info(device);peak=torch.cuda.max_memory_reserved(device)
        local=dict(rank=rank,GPU=torch.cuda.get_device_name(device),free_bytes=free,total_bytes=total,
            peak_reserved_bytes=peak,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            optimizer_steps=sum(r['phase']=='train_optimizer_step' for r in results),results=results,passed=free>=8*1024**3 and peak<=.85*total)
        gathered=[None]*4;torch.distributed.all_gather_object(gathered,local)
        passed=all(r['passed'] for r in gathered)
        if rank==0:
            trial.write(Path(job['output'])/'dense_probe.json',dict(status='passed' if passed else 'insufficient_headroom',
                world_size=4,ranks=gathered,probe_ids=audit['probe_ids'],model_saved=False,
                shape_audit_sha256=trial.file_sha(Path(job['output'])/'dense_input_shapes.json'),
                contract_sha256=baseline.digest(runtime.contract(job)),
                scope='Training shapes: four-rank DDP/AdamW/backward; fixed evaluation shapes: forward only, no scores; windows additionally probe eight distinct source slots with twenty presentations; no future OOM guarantee'))
        if not passed:raise ValueError('Insufficient GPU headroom; no automatic batch/context changes')
    finally:torch.distributed.destroy_process_group()


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--operation',choices=['preflight','probe','train','summarize','test'],required=True)
    a=p.parse_args();job=trial.read(a.job);validate(job)
    if a.operation in ('summarize','test'):
        if a.operation=='test' and job['stage']!='complete':raise ValueError('No test-based selection in quick stage')
        if int(os.environ.get('RANK','0'))==0:
            trial.verify_training(job)
            runtime.evaluate(job,'test' if a.operation=='test' else 'val')
        return
    try:
        pool=runtime.load_pool(job,recompute_aux=a.operation=='preflight')
    except Exception as exc:
        trial.write(Path(job['output'])/'dense_admission_failure.json',dict(status='blocked',
            stage='auxiliary_raw_provenance',error=f'{type(exc).__name__}: {exc}',training_started=False))
        raise
    if a.operation=='preflight':preflight(job,pool)
    elif a.operation=='probe':probe(job,pool)
    else:
        if os.environ.get('WORLD_SIZE')!='4':raise ValueError('Dense controlled schedule requires four GPUs')
        root=Path(job['output']);probe_report=trial.read(root/'dense_probe.json')
        if (probe_report['status']!='passed' or probe_report['world_size']!=4
                or probe_report['shape_audit_sha256']!=trial.file_sha(root/'dense_input_shapes.json')
                or probe_report['contract_sha256']!=baseline.digest(runtime.contract(job))):
            raise ValueError('Dense training requires the matching successful four-rank memory probe')
        runtime.install(job,pool,trial.read(root/'dense_input_shapes.json'))
        train.train(runtime.make_args(job))


if __name__=='__main__':main()
