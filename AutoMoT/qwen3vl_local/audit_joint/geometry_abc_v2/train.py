"""Explicit bounded B1 training; no automatic model download or legacy resume.

DDP ranks consume equal complete steps. Only successful optimizer steps advance
exposure; epoch checkpoints include optimizer/scheduler/RNG and contract binding.
"""
from collections import Counter
from pathlib import Path
import math
import os
import time
import torch
import torch.distributed as dist

from . import dataset, objective, planning, admission


def _unwrap(model):
    return model.module if hasattr(model,'module') else model


def save_checkpoint(path, bundle, optimizer, scheduler, *, binding, epoch, step, exposure, status):
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    model=_unwrap(bundle.model)
    state=dict(trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),binding=binding,
        epoch=epoch,step=step,exposure=dict(exposure),status=status,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary=path/'state.pt.tmp'; torch.save(state,temporary); temporary.rename(path/'state.pt')
    if hasattr(model,'save_pretrained'): model.save_pretrained(path/'adapter')
    dataset.write(path/'receipt.json',dict(binding=binding,epoch=epoch,step=step,status=status,
        state_sha256=dataset.sources.file_sha(path/'state.pt')))


def train_epoch(bundle, work, optimizer, scheduler, pack_record, *, rank=0, world=1,
                exposure=None, max_epoch_hours=10., timing_steps=100):
    """Reusable real loop, CPU-testable. No padding, skipped loss or fake exposure."""
    if len(work)%world: raise ValueError('unequal_rank_work')
    if not math.isfinite(max_epoch_hours) or max_epoch_hours<=0 or timing_steps<=0:
        raise ValueError('invalid_time_budget')
    if world>1 and (not dist.is_initialized() or dist.get_world_size()!=world):
        raise ValueError('distributed_group_required')
    bundle.model.train(); exposure=Counter(exposure or {}); local=Counter(); losses=[]
    start=time.monotonic(); steps=len(work)//world; stopped=False
    for i,r in enumerate(work[rank::world]):
        error=None
        try: packed=pack_record(r)
        except Exception as exc: error=f'{type(exc).__name__}: {exc}'
        bad=torch.tensor(int(error is not None),device=bundle.device)
        if world>1: dist.all_reduce(bad,op=dist.ReduceOp.MAX)
        if bad.item(): raise ValueError('rank input packing failed: '+str(error))
        optimizer.zero_grad(set_to_none=True)
        logits=objective.readout(bundle,packed)
        loss=objective.loss_from_logits(logits,r['targets'],packed['choices'])
        finite=torch.tensor(int(torch.isfinite(loss)),device=bundle.device)
        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise ValueError('nonfinite_loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(bundle.model.parameters(),1.)
        finite=torch.tensor(int(torch.isfinite(norm)),device=bundle.device)
        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item(): raise ValueError('nonfinite_gradients')
        optimizer.step(); scheduler.step(); local[r['id']]+=1; losses.append(float(loss.detach()))
        if rank==0 and ((i+1)%100==0 or i+1==steps):
            print(f'[study train] update={i+1}/{steps} loss={float(loss.detach()):.6f}',flush=True)
        elapsed=time.monotonic()-start
        stop=elapsed>max_epoch_hours*3600 or (i+1>=timing_steps and elapsed/(i+1)*steps>max_epoch_hours*3600)
        decision=torch.tensor(int(stop),device=bundle.device)
        if world>1: dist.all_reduce(decision,op=dist.ReduceOp.MAX)
        if decision.item(): stopped=True; break
    if world>1:
        gathered=[None]*world; dist.all_gather_object(gathered,dict(local))
        for counts in gathered: exposure.update(counts)
    else: exposure.update(local)
    return dict(steps=len(losses),loss=sum(losses)/len(losses) if losses else None,
        status='budget_stop' if stopped else 'epoch_complete',actual_exposure=dict(exposure),
        elapsed_seconds=time.monotonic()-start)


def training_members(artifact):
    report = artifact.get('admission')
    if report is None: report = admission.inspect(artifact)
    admission.require(report)
    fixed = admission.validation_ids(artifact['records'])
    if fixed != report['fixed_validation_ids']: raise ValueError('fixed_validation_members_changed')
    members = {r['id']: r for r in artifact['records']}
    return [members[k] for k in fixed]


def run(args):
    from ..phase3_comparison_policy import require
    require(per_event=args.per_event,epochs=args.epochs,seed=args.seed,learning_rate=args.learning_rate)
    # Single-node torchrun: raw source verification belongs to rank 0, while
    # every worker independently binds the verified dataset JSON bytes.
    from . import startup
    budget=startup.Budget(args.max_startup_hours)
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    if int(os.environ.get('GROUP_WORLD_SIZE', '1')) != 1:
        raise ValueError('geometry_requires_single_node_shared_sources')
    rank,local,world=native.setup_distributed(max(3600, math.ceil(budget.limit)+600))
    output=None
    try:
        if world != 4: raise ValueError('comparison_requires_four_ranks')
        artifact,dataset_sha=startup.load_dataset(args.dataset,rank=rank,world=world,budget=budget,variant=args.variant)
        validation=startup.all_ranks(lambda: training_members(artifact),world=world)
        device=torch.device('cuda',local)
        projection=startup.all_ranks(lambda: budget.run('all_epoch_capacity_plan', lambda: planning.project(
            artifact['records'],per_event=args.per_event,world_size=world,epochs=args.epochs,seed=args.seed)),world=world)
        report=projection['plans'][0]
        members={r['id']:r for r in artifact['records']}
        work=[members[k] for k in report['ordered_ids']]
        output=Path(args.output).resolve()
        def prepare_output():
            protected=[Path(artifact['data_root']),Path(args.dataset).resolve().parent,Path(args.model_dir).resolve()]
            if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
                raise ValueError('output_overlaps_input')
            output.mkdir(parents=True,exist_ok=False)
            dataset.write(output/'plan.json',projection)
            dataset.write(output/'admission.json',artifact['admission'])
        startup.primary(prepare_output,rank=rank,world=world)
        from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
        def model_identity_check():
            check_model(args.model_dir)
            root=Path(args.model_dir)
            return {str(p.relative_to(root)):dataset.sources.file_sha(p,check_budget=budget.check)
                    for p in sorted(root.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(root).parts}
        model_identity=startup.primary(lambda: budget.run('model_source_verification_rank0',model_identity_check),rank=rank,world=world)
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        bundle=startup.all_ranks(lambda: budget.run('model_load', lambda: native.load_model_with_lora(
            Path(args.model_dir),device=device,lora_rank=16,lora_alpha=32,lora_dropout=.05,
            lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)),world=world)
        cache={}
        def pack_record(r):
            if r['id'] in cache: return cache[r['id']]
            with Image.open(Path(artifact['data_root'])/r['image']) as image:
                return objective.pack(bundle,r,image.convert('RGB'))
        probes=startup.probe_records(work,validation)
        def probe():
            objective.choice_ids(bundle.tokenizer,args.variant)
            for r in probes[rank::world]:
                budget.check(); cache[r['id']]=pack_record(r)
        startup.all_ranks(lambda: budget.run('bounded_prompt_probe',probe),world=world)
        if world>1:
            bundle.model=startup.all_ranks(lambda: budget.run('ddp_setup', lambda:
                torch.nn.parallel.DistributedDataParallel(bundle.model,device_ids=[local],find_unused_parameters=False)),world=world)
        startup.all_ranks(budget.check,world=world)
        if rank==0:
            dataset.write(output/'startup.json',dict(budget.report(),source_verification_ranks=[0],
                processor_probe_total=len(probes),probe_ids=[r['id'] for r in probes],
                all_pool_processor_pass=False,remaining_inputs_checked_on_consumption=True))
        optimizer=torch.optim.AdamW([p for p in bundle.model.parameters() if p.requires_grad],lr=args.learning_rate,
                                   betas=(.9,.95),weight_decay=0.)
        total=report['optimizer_steps']*args.epochs
        warmup=min(max(1,round(total*2000/7680)),max(1,total-1))
        from ..phase3_comparison_policy import make_scheduler
        scheduler=make_scheduler(optimizer,total,warmup)
        binding=dict(contract=artifact['contract'],code=artifact['code'],model=model_identity,
            dataset_sha256=dataset_sha,world_size=world,epochs=args.epochs,
            per_event=args.per_event,seed=args.seed,learning_rate=args.learning_rate)
        binding['shared_plan_sha256']=projection['sha256']
        binding['warmup_steps']=warmup
        from ..phase3_comparison_policy import contract as comparison_contract
        binding['comparison_policy']=comparison_contract()
        binding['fixed_validation_ids']=[r['id'] for r in validation]
        binding['admission_sha256']=artifact['admission']['sha256']
        exposure={}; step=0; best_score=None; best_epoch=None
        for epoch in range(args.epochs):
            if epoch:
                work,report=startup.all_ranks(lambda: planning.plan(artifact['records'],per_event=args.per_event,world_size=world,
                    epochs=args.epochs,seed=args.seed+epoch,exposure=exposure),world=world)
                if report != projection['plans'][epoch]: raise ValueError('actual_exposure_diverged_from_shared_complete_epoch_plan')
            result=train_epoch(bundle,work,optimizer,scheduler,pack_record,rank=rank,world=world,
                               exposure=exposure,max_epoch_hours=args.max_epoch_hours)
            if result['status']=='epoch_complete' and result['steps']!=report['optimizer_steps']:
                raise ValueError('completed_epoch_budget_mismatch')
            exposure=result['actual_exposure'];step+=result['steps']
            def save_epoch():
                dataset.write(output/f'epoch_{epoch+1}.json',dict(plan=report,result=result))
                save_checkpoint(output/f'epoch_{epoch+1}',bundle,optimizer,scheduler,binding=binding,
                    epoch=epoch+1,step=step,exposure=exposure,status=result['status'])
            startup.primary(save_epoch,rank=rank,world=world)
            if result['status']=='budget_stop':raise RuntimeError('study training stopped at measured time budget; checkpoint saved')
            def validate_epoch():
                from . import metrics
                from types import SimpleNamespace
                evaluation_bundle=SimpleNamespace(model=_unwrap(bundle.model),processor=bundle.processor,
                    tokenizer=bundle.tokenizer,device=device)
                cases=[]
                for r in validation:
                    with Image.open(Path(artifact['data_root'])/r['image']) as image:
                        prediction=objective.predict(evaluation_bundle,r,image.convert('RGB'))
                    cases.append(dict(record=r,prediction=prediction))
                result=metrics.evaluate(cases)
                dataset.write(output/f'validation_{epoch+1}.json',result)
                score=result['macro_recall']
                # Final-epoch policy does not select on a sparse macro score.
                return score
            score=startup.primary(validate_epoch,rank=rank,world=world)
            if score is not None and (best_score is None or score>best_score): best_score=score;best_epoch=epoch+1
        startup.primary(lambda: dataset.write(output/'training_complete.json',dict(
            binding=binding,total_steps=step,best_epoch=best_epoch,best_macro_recall=best_score,
            selected_checkpoint=f'epoch_{args.epochs}',selection='final_completed_epoch',
            checkpoint_receipts={f'epoch_{i}':dataset.sources.file_sha(output/f'epoch_{i}'/'receipt.json')
                                 for i in range(1,args.epochs+1)})),rank=rank,world=world)
    finally:
        native.cleanup_distributed()


def restore(bundle, checkpoint, *, artifact, dataset_path):
    """Strict geometry-only weight loading; never interpret old A-F adapters as A-I."""
    import json
    from ..geometry_v1.sources import file_sha
    checkpoint=Path(checkpoint)
    receipt=json.loads((checkpoint/'receipt.json').read_text())
    if file_sha(checkpoint/'state.pt')!=receipt['state_sha256']:raise ValueError('checkpoint_changed')
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    binding=state['binding']
    if (binding!=receipt['binding'] or binding['contract']!=artifact['contract'] or
            binding['code']!=artifact['code'] or binding['dataset_sha256']!=file_sha(dataset_path)):
        raise ValueError('checkpoint_geometry_contract_mismatch')
    if 'admission' in artifact:
        verify_admission_binding(artifact,binding)
    params={n:p for n,p in _unwrap(bundle.model).named_parameters() if p.requires_grad}
    if set(params)!=set(state['trainable']):raise ValueError('checkpoint_parameter_topology_mismatch')
    with torch.no_grad():
        for n,p in params.items():
            if p.shape!=state['trainable'][n].shape:raise ValueError('checkpoint_parameter_shape_mismatch')
            p.copy_(state['trainable'][n].to(device=p.device,dtype=p.dtype))
    return state


def verify_admission_binding(artifact, binding):
    validation=training_members(artifact)
    if (binding.get('admission_sha256')!=artifact['admission']['sha256']
            or binding.get('fixed_validation_ids')!=[r['id'] for r in validation]):
        raise ValueError('checkpoint_admission_mismatch')


def evaluate(args):
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    from qwen3vl_local.audit_joint.phase3_suite import model_snapshot
    from . import metrics
    import json
    artifact=dataset.load(args.dataset,variant=args.variant)
    receipt=json.loads((Path(args.checkpoint)/'receipt.json').read_text())
    verify_admission_binding(artifact,receipt['binding'])
    check_model(args.model_dir)
    identity={k:v['sha256'] for k,v in model_snapshot(Path(args.model_dir)).items()}
    if receipt['binding']['model']!=identity:raise ValueError('base_model_mismatch')
    if args.split=='test':
        completion=json.loads((Path(args.checkpoint).parent/'training_complete.json').read_text())
        if (completion.get('selection')!='final_completed_epoch' or completion.get('total_steps')!=9600
            or completion.get('selected_checkpoint')!='epoch_3' or receipt['step']!=9600
            or receipt['status']!='epoch_complete' or completion['binding']!=receipt['binding'] or
            completion['selected_checkpoint']!=Path(args.checkpoint).name or
            completion['checkpoint_receipts'].get(Path(args.checkpoint).name)!=dataset.sources.file_sha(Path(args.checkpoint)/'receipt.json')):
            raise ValueError('test_requires_completed_budget_and_final_checkpoint')
    if args.ablations and args.split!='val': raise ValueError('interventions_require_validation_split')
    output=Path(args.output).resolve()
    protected=[Path(artifact['data_root']),Path(args.dataset).resolve().parent,
               Path(args.checkpoint).resolve(),Path(args.model_dir).resolve()]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):raise ValueError('output_overlaps_input')
    output.mkdir(parents=True,exist_ok=False)
    bundle=native.load_model_with_lora(Path(args.model_dir),device=torch.device('cuda:0'),
        lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=False)
    restore(bundle,args.checkpoint,artifact=artifact,dataset_path=args.dataset)
    cases=[]
    for r in planning.evaluation(artifact['records'],args.split):
        with Image.open(Path(artifact['data_root'])/r['image']) as image:
            prediction=objective.predict(bundle,r,image.convert('RGB'))
        cases.append(dict(record=r,prediction=prediction))
    dataset.write(output/'cases.json',cases)
    dataset.write(output/'metrics.json',metrics.evaluate(cases))
    if args.ablations:
        for mode in ('history_unknown','history_permuted','image_black'):
            paired=[]
            for case in cases:
                r=metrics.intervention(case['record'],mode)
                with Image.open(Path(artifact['data_root'])/r['image']) as image:
                    rgb=image.convert('RGB')
                if mode=='image_black': rgb=Image.new('RGB',rgb.size)
                prediction=objective.predict(bundle,r,rgb)
                prediction.update(source='diagnostic_predicted',intervention=mode)
                paired.append(dict(record=r,prediction=prediction))
            dataset.write(output/(mode+'.json'),dict(metrics=metrics.evaluate(paired),
                paired=metrics.paired_summary(cases,paired),cases=paired,
                intervention=mode,perturbed_input=True))
