"""Dense experiment worker; source binding, coverage admission and GPU probe."""
import argparse
from collections import Counter
import os
from pathlib import Path
import torch
from . import runtime, objectives, baseline
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_base_trial_worker as base


def validate(job):
    base.validate_job(job)
    if job['mode'] not in ('future','stream','windows'):
        raise ValueError('Unknown dense mode')
    init=Path(job['init_adapter'])
    actual={str(p.relative_to(init)):trial.file_sha(p) for p in sorted(init.rglob('*')) if p.is_file()}
    if actual!=job['init_adapter_files']:
        raise ValueError('Initialization adapter changed')


def preflight(job,pool):
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index,check_model
    check_index(job['index'],action_output_mode='choice',prompt_variant=trial.reference()['prompt_variant'])
    check_model(job['model_dir'])
    rows,coverage=pool.eligible(job['mode'],'train')
    args=runtime.make_args(job)
    work,audit=baseline.independent_work(rows,args,epoch=0,world_size=4)
    if len(work)!=10240:
        raise ValueError('Dense event/action schedule changed')
    counts=Counter()
    for item in work:
        rec=pool.record(item.row,job['mode'])
        for i,action in enumerate(rec['actions']):
            counts[f'{item.row.context_id}/{i}/{action or "MISSING"}']+=1
    if job['mode']=='future' and not any(k.split('/')[1]!='0' and not k.endswith('/MISSING') for k in counts):
        raise ValueError('No future native labels: cannot claim denser supervision; supply a reviewed denser index')
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
        if restored['next_epoch']!=job['resume_epoch']:
            raise ValueError('Dense resume epoch mismatch')
        baseline.independent_work(rows,args,epoch=restored['next_epoch'],world_size=4,
                                  history=restored['sampling_history'])
    trial.write(Path(job['output'])/'dense_coverage.json',dict(contract=runtime.contract(job),
        coverage=coverage,actual_training_query_exposure=counts,sampling=audit,
        update_budget=7680,labels_rebuilt=False,RGB_copied=False))


def probe(job,pool):
    if not torch.cuda.is_available():
        raise ValueError('GPU memory/backward probe requires a local CUDA device')
    rows,_=pool.eligible(job['mode'],'train')
    # Processor caps all images; inspect the longest encoded example among one
    # candidate per event, rather than pretending image count is a memory bound.
    bundle=runtime.create_bundle(job,torch.device('cuda:0'))
    bundle.model.train()
    examples={}
    for row in rows:
        examples.setdefault(row.context_id,row)
    candidates=[]
    for row in examples.values():
        record=pool.record(row,job['mode'])
        p=objectives.pack(bundle,record,train._load_images(record['images']))
        size=sum(x['tokens'] for x in p) if isinstance(p,list) else p['tokens']
        candidates.append((size,p))
    packed=max(candidates,key=lambda x:x[0])[1]
    optimizer=torch.optim.AdamW([p for p in bundle.model.parameters() if p.requires_grad],lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    results=[]
    for _ in range(2):
        loss,stats=(objectives.window_loss(bundle,packed) if isinstance(packed,list)
                    else objectives.readout_loss(bundle.model,packed,bundle.device))
        loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        results.append(dict(loss=float(loss.detach()),**stats))
    torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info()
    peak=torch.cuda.max_memory_reserved()
    passed=free>=8*1024**3 and peak<=.85*total
    trial.write(Path(job['output'])/'dense_probe.json',dict(status='passed' if passed else 'insufficient_headroom',
        GPU=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,peak_reserved_bytes=peak,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_steps=2,results=results,
        model_saved=False,scope='local GPU measured samples; not a guarantee for all images or other jobs'))
    if not passed:
        raise ValueError('Insufficient GPU headroom; no automatic batch/context changes')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--operation',choices=['preflight','probe','train','summarize','test'],required=True)
    a=p.parse_args();job=trial.read(a.job);validate(job)
    if a.operation in ('summarize','test'):
        if a.operation=='test' and job['stage']!='complete':
            raise ValueError('No test-based selection in quick stage')
        # Coordinator's test command uses four processes; this evaluator uses
        # one device and the other ranks deliberately do no inference.
        if int(os.environ.get('RANK','0'))==0:
            trial.verify_training(job)
            runtime.evaluate(job,'test' if a.operation=='test' else 'val')
        return
    pool=runtime.load_pool(job)
    if a.operation=='preflight':preflight(job,pool)
    elif a.operation=='probe':probe(job,pool)
    else:
        if os.environ.get('WORLD_SIZE')!='4':
            raise ValueError('Dense controlled schedule requires four GPUs')
        runtime.install(job,pool)
        train.train(runtime.make_args(job))


if __name__=='__main__':
    main()
