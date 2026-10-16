"""Fixed-sample minibatch timing, including simultaneous DDP; no checkpoints saved."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from .batch_probe import nonempty_path, validate_paths
from .benchmark_runtime import progress, trace_waits, run_owned, prepare_phase4, read_prepared
from qwen3vl_local.sequence_batch import digest, plan, phase4_records


def write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def schedule(records, *, per_event, update_samples, world, batch, mode, seed):
    """Same membership and optimizer-update groups for every B/world/RGB mode."""
    if (min(per_event,update_samples,world,batch)<1 or per_event%update_samples
            or update_samples%(world*batch) or mode not in (2,4)):
        raise ValueError('samples-per-event must divide into update-samples, which must divide into world * batch')
    if any(r['split']!='train' for r in records):
        raise ValueError('benchmark only accepts training samples')
    indices,audit=plan(records,per_event=per_event,seed=seed,world_size=1,
                       batch_size=update_samples,stride=1)
    updates=[]
    for start in range(0,len(indices),update_samples):
        block=indices[start:start+update_samples]
        # Retain identical membership per update; use each mode's temporal stride.
        block.sort(key=lambda i:(records[i]['sequence'], records[i]['frame']%(2 if mode==2 else 1),
                                 records[i]['frame'],records[i]['id']))
        updates.append(block)
    ids=[[records[i]['id'] for i in block] for block in updates]
    return updates,dict(event_counts=audit['event_counts'],presentations=len(indices),
        unique_questions=len(set(indices)),sample_membership_sha256=digest(sorted(x for block in ids for x in block)),
        update_membership_sha256=digest([sorted(block) for block in ids]),
        traversal_sha256=digest(ids),updates=len(updates),update_samples=update_samples,
        accumulation=update_samples//(world*batch),anchor_stride=2 if mode==2 else 1,
        pool_sha256=digest(sorted(records,key=lambda r:r['id'])))


def rank_microbatches(block, rank, world, batch):
    return [block[k+rank*batch:k+(rank+1)*batch] for k in range(0,len(block),world*batch)]


def warmup_schedule(updates, records, rounds):
    by_event={}
    for block in updates:
        events={records[i]['event'] for i in block}
        if len(events)!=1:raise ValueError('optimizer update crosses event quota blocks')
        by_event.setdefault(next(iter(events)),[]).append(block)
    return [blocks[r%len(blocks)] for r in range(rounds) for _,blocks in sorted(by_event.items())]


def execute(model, optimizer, params, updates, *, rank, world, batch, prepare, loss_for,
            device, timing, synchronize, barrier, emit, memory=None, inputs_on_device=False):
    """Common measured loop; CPU/Gloo tests exercise actual DDP synchronization."""
    import torch
    totals=Counter();presentations=0
    barrier();started=time.perf_counter()
    for update_index,block in enumerate(updates):
        micros=rank_microbatches(block,rank,world,batch)
        optimizer.zero_grad(set_to_none=True)
        for micro_index,indices in enumerate(micros):
            stages={};stage_memory={}
            def stage(name,fn):
                before=time.perf_counter();value=fn()
                if timing=='detailed':
                    synchronize()
                    if memory is not None:stage_memory[name]=memory()
                stages[name]=time.perf_counter()-before
                return value
            before=time.perf_counter()
            inputs,meta=stage('encode_batch_device' if inputs_on_device else 'prepare_cpu',lambda:prepare(indices))
            if not inputs_on_device:
                inputs=stage('host_to_device',lambda:{k:v.to(device) for k,v in inputs.items()})
            update=micro_index+1==len(micros)
            with (model.no_sync() if world>1 and not update else nullcontext()):
                loss=stage('forward_loss',lambda:loss_for(model,inputs,indices))
                # .item() can synchronize, so charge it explicitly to validation.
                value=stage('loss_check',lambda:float(loss.detach()))
                if not __import__('math').isfinite(value):raise FloatingPointError('nonfinite benchmark loss')
                stage('backward_with_ddp' if update and world>1 else 'backward',lambda:(loss/len(micros)).backward())
            if update:
                def step():
                    torch.nn.utils.clip_grad_norm_(params,1.)
                    optimizer.step()
                stage('clip_optimizer',step)
            # Always close the step before timing/logging. DDP backward includes
            # communication; do not label it as isolated network or kernel time.
            synchronize()
            elapsed=time.perf_counter()-before
            record=dict(update=update_index,micro=micro_index,rank=rank,indices=indices,
                optimizer_update=update,loss=value,seconds=elapsed,**meta)
            record['stages_seconds' if timing=='detailed' else 'host_submission_seconds']=stages
            if stage_memory:record['stage_cuda_bytes']=stage_memory
            emit(record)
            totals.update(stages);presentations+=len(indices)
            del loss,inputs
    barrier()
    result=dict(seconds=time.perf_counter()-started,presentations=presentations,optimizer_updates=len(updates))
    result['stage_seconds' if timing=='detailed' else 'host_submission_seconds']=dict(totals)
    return result


def load_pool(a,mode):
    if a.phase=='3':
        from qwen3vl_local.sft_new_loop_phase3 import train as t
        from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
        from qwen3vl_local.sft_new_loop_phase3.invalid_balance import unique_cases,case_identity,physical_group_for_item
        check_index(a.index,action_output_mode='choice',prompt_variant=a.prompt_variant)
        items=[t._make_item(r,seed=a.seed,action_output_mode='choice')
               for r in t._read_rows(a.index,'train',data_root=a.data_root)]
        items=unique_cases([i for i in items if t.choice_rejection_reason(i.spec) is None])
        records=[dict(id=digest(case_identity(i)),event=i.row.context_id,
            sequence=json.dumps([i.row.scenario,i.row.route_id,i.row.context_id,i.row.invalid_reason]),
            frame=i.row.frame_id,action=t.sampling_action(i.row.answers,i.row.context_id),
            split=i.row.split,group=physical_group_for_item(i)) for i in items]
        expected=t.CONTEXT_IDS;manifest=None
    else:
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
        from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
        data,manifest=load_dataset(a.dataset,rgb_mode=mode)
        items=[r for r in data['train'] if r['target'] in ('YES','NO')]
        records=phase4_records(items);expected=EVENTS
    if {r['event'] for r in records}!=set(expected):raise ValueError('benchmark requires all ten training events')
    return items,records,manifest


def callbacks(a,bundle,items,mode,manifest):
    import torch
    from types import SimpleNamespace
    from qwen3vl_local.sequence_batch import collate
    # Phase3 prepares on CPU and transfers after collation, like its trainer.
    # Phase4 must use the actual trainer's per-example transfer + collation path.
    cpu=SimpleNamespace(**vars(bundle));cpu.device=torch.device('cpu')
    if a.phase=='4':bundle.observation_contract=manifest['observation_contract']
    def prepare(indices):
        if a.phase=='4':
            from qwen3vl_local.sft_new_loop_phase4.rgb_short.model import encode_batch
            inputs=encode_batch(bundle,[items[i] for i in indices],a.data_root,max_length=a.max_length)
            return inputs,dict(padded_tokens=inputs['input_ids'].numel(),max_tokens=inputs['input_ids'].shape[1])
        cache={};rows=[]
        for i in indices:
            if a.phase=='3':
                from qwen3vl_local.sft_new_loop_phase3 import train as t
                item=items[i];history='2rgb_short' if mode==2 else '4rgb'
                images=t._load_images(t.select_history_rgb_paths(item.row.history_rgb_paths,history),cache=cache)
                packed=t._build_inputs(cpu,images=images,history_rgb_mode=history,prompt_variant=a.prompt_variant,
                    target=t.build_action_target(item.spec),spec=item.spec,max_length=a.max_length,format_loss_weight=.25)
                if packed is None:raise ValueError('sample exceeds token limit; no truncation')
                row={k:packed[k].unsqueeze(0) for k in ('input_ids','attention_mask','labels','loss_weights','component_ids')}
                row.update(packed['vision']);rows.append(row)
        inputs=collate(rows,bundle.tokenizer.pad_token_id)
        return inputs,dict(tokens=int(inputs['attention_mask'].sum()),padded_tokens=inputs['input_ids'].numel(),
                           max_tokens=inputs['input_ids'].shape[1],unique_decoded_images=len(cache))
    def loss_for(model,inputs,indices):
        if a.phase=='4':return model(**inputs,use_cache=False).loss
        from qwen3vl_local.sft_new_loop_phase3.train import _loss_from_logits
        labels=inputs.pop('labels');weights=inputs.pop('loss_weights');comp=inputs.pop('component_ids')
        logits=model(**inputs,use_cache=False,return_dict=True).logits
        return torch.stack([_loss_from_logits(logits[j:j+1],labels[j:j+1],weights[j:j+1],
            comp[j:j+1],items[i].spec)[0] for j,i in enumerate(indices)]).mean()
    return prepare,loss_for


def worker(a):
    progress('worker imports START')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from qwen3vl_local.sft_v2.train import load_model_with_lora
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=len(a.gpus.split(',')):raise ValueError('worker world does not match requested GPU list')
    if not torch.cuda.is_available():raise RuntimeError('real CUDA devices required; CPU is not a capacity measurement')
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    progress('distributed_init START')
    if world>1:dist.init_process_group('nccl',timeout=timedelta(seconds=a.ddp_timeout))
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    mode=a.modes[0];batch=a.batches[0]
    progress('data_selection START')
    if a.phase=='4':
        if not a.prepared_samples:raise ValueError('Phase4 workers require parent-validated prepared samples')
        items,records,manifest,updates,selection=read_prepared(a)
    else:
        items,records,manifest=load_pool(a,mode)
        updates,selection=schedule(records,per_event=a.samples_per_event,update_samples=a.update_samples,
            world=world,batch=batch,mode=mode,seed=a.seed)
    progress(f'data_selection DONE rows={len(items)}')
    if world>1:
        selections=[None]*world;dist.all_gather_object(selections,selection)
        if any(s!=selection for s in selections):raise ValueError('rank selection mismatch')
    progress('model_load START')
    bundle=load_model_with_lora(a.model_dir,device=device,lora_rank=a.lora_rank,lora_alpha=a.lora_alpha,
        lora_dropout=.05,lora_vision_scope=a.vision_scope,strict_vision_scope=True,
        gradient_checkpointing=a.gradient_checkpointing)
    progress('model_load DONE; DDP_setup START')
    params=[p for p in bundle.model.parameters() if p.requires_grad]
    if world>1:bundle.model=DistributedDataParallel(bundle.model,device_ids=[local],find_unused_parameters=False)
    optimizer=torch.optim.AdamW(params,lr=1e-5 if a.phase=='3' else 2e-5,
        weight_decay=0. if a.phase=='3' else .01,betas=(.9,.95) if a.phase=='3' else (.9,.999))
    prepare,loss_for=callbacks(a,bundle,items,mode,manifest)
    bundle.model.train();torch.cuda.synchronize()
    free,total=torch.cuda.mem_get_info();baseline=torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    barrier=dist.barrier if world>1 else lambda:None
    sync=lambda:torch.cuda.synchronize(device)
    memory=lambda:dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                       cumulative_peak_allocated=torch.cuda.max_memory_allocated(device),
                       cumulative_peak_reserved=torch.cuda.max_memory_reserved(device))
    warm_blocks=warmup_schedule(updates,records,a.warmup_updates)
    progress('warmup START')
    warmup=execute(bundle.model,optimizer,params,warm_blocks,
        rank=rank,world=world,batch=batch,prepare=prepare,loss_for=loss_for,device=device,
        timing=a.timing,synchronize=sync,barrier=barrier,emit=lambda r:progress(f'warmup update {r["update"]+1}/{len(warm_blocks)}') if rank==0 and r['optimizer_update'] else None,inputs_on_device=a.phase=='4')
    warmup['global_event_counts']=dict(Counter(records[i]['event'] for block in warm_blocks for i in block))
    progress('warmup DONE; measurement START')
    log=a.output/f'rank{rank}_steps.jsonl'
    def emit(record):
        record['ids']=[records[i]['id'] for i in record['indices']]
        record['events']=[records[i]['event'] for i in record['indices']]
        with log.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        if rank==0 and record['optimizer_update']:
            print(f"[rank0] completed measured update {record['update']+1}/{len(updates)} loss={record['loss']:.5f}",flush=True)
    measured=execute(bundle.model,optimizer,params,updates,rank=rank,world=world,batch=batch,
        prepare=prepare,loss_for=loss_for,device=device,timing=a.timing,synchronize=sync,barrier=barrier,emit=emit,memory=memory,
        inputs_on_device=a.phase=='4')
    peak_a=torch.cuda.max_memory_allocated();peak_r=torch.cuda.max_memory_reserved();other=max(0,total-free-baseline)
    import importlib.metadata
    raw=bundle.model.module if world>1 else bundle.model
    config=getattr(raw,'config',None)
    text_config=getattr(config,'text_config',config)
    result=dict(status='measured',rank=rank,world=world,gpu=a.gpus.split(',')[local],
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
            transformers=importlib.metadata.version('transformers'),torch_cpu_threads=torch.get_num_threads(),
            attention_implementation=getattr(text_config,'_attn_implementation',None),
            allocator_environment=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            trainable_parameters=sum(p.numel() for p in params)),
        gpu_name=torch.cuda.get_device_name(device),phase=a.phase,rgb_mode=mode,batch_size=batch,
        timing=a.timing,selection=selection,warmup=warmup,**measured,
        peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r,total_bytes=total,
        other_process_bytes_at_start=other,fits_with_headroom=peak_r+other<=total*(1-a.headroom_fraction),
        configuration=dict(seed=a.seed,gradient_checkpointing=a.gradient_checkpointing,
            lora_rank=a.lora_rank,lora_alpha=a.lora_alpha,vision_scope=a.vision_scope,
            max_length=a.max_length,prompt_variant=a.prompt_variant,warmup_updates=a.warmup_updates,
            input_pipeline='phase4_encode_batch_device' if a.phase=='4' else 'phase3_cpu_collate_transfer'),
        scope='fixed training examples, fixed samples/update; synchronized timings include launch/wait overhead; DDP backward includes communication; no checkpoints saved')
    write(a.output/f'rank{rank}.json',result)
    if world>1:dist.destroy_process_group()


def collect(folder,world,returncode):
    """Require all successful ranks plus complete, exact ordered consumption."""
    ranks=[]
    for rank in range(world):
        path=folder/f'rank{rank}.json'
        if path.is_file():ranks.append(json.loads(path.read_text()))
    result=dict(status='error',returncode=returncode,ranks=ranks)
    if returncode or len(ranks)!=world or any(r['status']!='measured' for r in ranks):return result
    selection=ranks[0]['selection'];all_steps=[]
    for rank,r in enumerate(ranks):
        if r['rank']!=rank or r['selection']!=selection:raise ValueError('rank result/selection mismatch')
        steps=[json.loads(s) for s in (folder/f'rank{rank}_steps.jsonl').read_text().splitlines()]
        expected=selection['updates']*selection['accumulation']
        if len(steps)!=expected:raise ValueError('incomplete rank step records')
        for k,step in enumerate(steps):
            update,micro=divmod(k,selection['accumulation'])
            if (step['rank']!=rank or step['update']!=update or step['micro']!=micro
                or step['optimizer_update']!=(micro+1==selection['accumulation'])
                or len(step['ids'])!=r['batch_size']):raise ValueError('invalid consumed batch/update record')
        all_steps.extend(steps)
    ids_by_update=[];ordered_updates=[];events=Counter()
    for update in range(selection['updates']):
        rows=[s for s in all_steps if s['update']==update]
        ordered=[i for s in sorted(rows,key=lambda s:(s['micro'],s['rank'])) for i in s['ids']]
        ordered_updates.append(ordered)
        ids_by_update.append(sorted(ordered))
        events.update(e for s in rows for e in s['events'])
    if (digest(ordered_updates)!=selection['traversal_sha256'] or dict(events)!=selection['event_counts'] or digest(ids_by_update)!=selection['update_membership_sha256']
        or sum(r['presentations'] for r in ranks)!=selection['presentations']
        or any(r['optimizer_updates']!=selection['updates'] for r in ranks)):
        raise ValueError('actual global consumption differs from fixed plan')
    seconds=max(r['seconds'] for r in ranks)
    stage_totals=Counter()
    for r in ranks:stage_totals.update(r.get('stage_seconds',{}))
    stage_mean={k:v/world for k,v in stage_totals.items()}
    return dict(result,status='measured',selection=selection,seconds=seconds,
        mean_rank_stage_seconds=stage_mean,
        largest_measured_stage=max(stage_mean,key=stage_mean.get) if stage_mean else None,
        samples_per_second=selection['presentations']/seconds,
        fits_with_headroom=all(r['fits_with_headroom'] for r in ranks),
        max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in ranks),
        max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in ranks))


def summarize(trials,repeats):
    recommendations={};groups={}
    for row in trials:groups.setdefault((row['rgb_mode'],row['batch_size']),[]).append(row)
    bindings={}
    for (mode,batch),rows in groups.items():
        good=[r for r in rows if r['status']=='measured']
        for row in good:
            s=row['selection'];binding=(s['sample_membership_sha256'],s['update_membership_sha256'])
            if mode in bindings and bindings[mode]!=binding:raise ValueError('cannot compare different benchmark examples/update groups')
            bindings[mode]=binding
        if len(good)==repeats and all(r['fits_with_headroom'] for r in good):
            speeds=[r['samples_per_second'] for r in good]
            value=dict(batch_size=batch,median_samples_per_second=statistics.median(speeds),
                       min_samples_per_second=min(speeds),max_samples_per_second=max(speeds),repeats=repeats)
            old=recommendations.get(str(mode))
            if old is None or value['median_samples_per_second']>old['median_samples_per_second']:
                recommendations[str(mode)]=value
    return dict(trials=trials,fastest_measured_candidates=recommendations,
        note='Fastest tested median with headroom on every rank/repeat, not a convergence or full-pool guarantee. Detailed mode synchronizes stages; use end_to_end to confirm deployment throughput.')


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--phase',choices=('3','4'),required=True)
    for name in ('index','dataset','data-root','model-dir','output'):
        p.add_argument('--'+name,type=nonempty_path,required=name in ('data-root','model-dir','output'))
    p.add_argument('--gpus',default='0,1,2,3',help='Physical GPUs used simultaneously by torchrun; one GPU also supported')
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--modes',type=int,nargs='+',choices=(2,4),default=[2,4])
    p.add_argument('--max-length',type=int,default=8192)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--lora-alpha',type=int,default=32)
    p.add_argument('--vision-scope',choices=('off','merger','last4','all'),default='off')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--prompt-variant',default='baseline')
    p.add_argument('--headroom-fraction',type=float,default=.15)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--samples-per-event',type=int,default=16)
    p.add_argument('--update-samples',type=int,default=16)
    p.add_argument('--warmup-updates',type=int,default=1,help='Warmup optimizer updates PER EVENT, excluded from timed sample counts')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260904)
    p.add_argument('--timing',choices=('detailed','end_to_end'),default='detailed')
    p.add_argument('--prepare-timeout',type=int,default=120,help='CPU preparation limit per RGB mode in seconds')
    p.add_argument('--prepare-only',action='store_true',help='Phase4 bounded CPU diagnosis; never starts CUDA workers')
    p.add_argument('--prepare-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--prepared-samples',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--prepared-sha256',help=argparse.SUPPRESS)
    p.add_argument('--continue-on-error',action='store_true',help='Explicitly continue trial matrix after failures; default stops immediately')
    p.add_argument('--ddp-timeout',type=int,default=180)
    p.add_argument('--trial-timeout',type=int,default=1800)
    return p


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    p=parser();a=p.parse_args()
    if a.gradient_checkpointing is None:a.gradient_checkpointing=a.phase=='4'
    try:
        validate_paths(a)
        gpu_ids=[g.strip() for g in a.gpus.split(',')]
        if any(not g.isdigit() for g in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):raise ValueError('GPU IDs must be distinct numbers')
        a.gpus=','.join(gpu_ids)
        if (min(a.samples_per_event,a.update_samples,a.repeats,a.warmup_updates,a.ddp_timeout,a.trial_timeout,
                a.max_length,a.lora_rank,a.lora_alpha,a.prepare_timeout,*a.batches)<1 or not 0<a.headroom_fraction<1
                or a.samples_per_event%a.update_samples or any(a.update_samples%(len(gpu_ids)*b) for b in a.batches)):
            raise ValueError('invalid budget: samples-per-event divisible by update-samples; update-samples divisible by GPU count * every batch')
    except ValueError as exc:p.error(str(exc))
    if (a.prepare_only or a.prepare_worker) and a.phase!='4':p.error('CPU preparation is Phase4 only')
    if a.prepare_worker:
        trace_waits();prepare_phase4(a);return
    if a.worker:
        trace_waits()
        rank=int(os.environ.get('RANK',0))
        try:worker(a)
        except Exception as exc:
            write(a.output/f'rank{rank}.json',dict(status='error',error_type=type(exc).__name__,error=str(exc),rank=rank))
            traceback.print_exc();raise
        return
    a.output.mkdir(parents=True,exist_ok=False);trials=[]
    import hashlib
    request={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    paths=[Path(__file__),Path(__file__).with_name('batch_probe.py'),
           Path(__file__).with_name('benchmark_runtime.py'),a.model_dir/'config.json',
           a.index.with_name('manifest.json') if a.phase=='3' else a.dataset/'student_view.json']
    request['source_input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write(a.output/'request.json',request)
    for mode in dict.fromkeys(a.modes):
        prepared = None
        prepared_sha = None
        if a.phase=='4':
            prepared = a.output/f'prepared_rgb{mode}.json'
            command=[sys.executable,'-u','-m','qwen3vl_local.audit_joint.batch_benchmark',
                     '--prepare-worker','--phase','4','--output',str(prepared),'--modes',str(mode),
                     '--batches',str(a.batches[0]),'--gpus',a.gpus]
            for name in ('data_root','model_dir','dataset','samples_per_event','update_samples','seed'):
                command.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
            print(f'[benchmark] CPU preparation RGB={mode}; limit={a.prepare_timeout}s; no GPU workers',flush=True)
            code=run_owned(command,a.output/f'prepare_rgb{mode}.log',
                           dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1'),a.prepare_timeout)
            preparation=dict(rgb_mode=mode,returncode=code,status='prepared' if code==0 else 'failed',
                             log=str(a.output/f'prepare_rgb{mode}.log'))
            write(a.output/f'prepare_rgb{mode}.json',preparation)
            if code:
                write(a.output/'summary.json',dict(status='preparation_failed',preparation=preparation,
                    trials=trials,fastest_measured_candidates={}))
                raise SystemExit(code if code in (130,143) else 2)
            prepared_sha=hashlib.sha256(prepared.read_bytes()).hexdigest()
            if a.prepare_only:continue
        for repeat in range(a.repeats):
            batches=sorted(set(a.batches),reverse=bool(repeat%2))
            for batch in batches:
                folder=a.output/f'rgb{mode}_batch{batch}_repeat{repeat}';folder.mkdir()
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpu_ids)}',
                     '--monitor-interval=1','-m','qwen3vl_local.audit_joint.batch_benchmark','--worker',
                     '--phase',a.phase,'--output',str(folder),'--modes',str(mode),'--batches',str(batch),'--gpus',a.gpus]
                for name in ('data_root','model_dir','index','dataset','max_length','lora_rank','lora_alpha','vision_scope',
                             'prompt_variant','headroom_fraction','samples_per_event','update_samples','warmup_updates',
                             'repeats','seed','timing','ddp_timeout','trial_timeout'):
                    value=getattr(a,name)
                    if value is not None:cmd.extend(['--'+name.replace('_','-'),str(value)])
                cmd.append('--gradient-checkpointing' if a.gradient_checkpointing else '--no-gradient-checkpointing')
                if prepared is not None:
                    cmd.extend(['--prepared-samples',str(prepared),'--prepared-sha256',prepared_sha])
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','1'))
                print(f'[benchmark] RGB={mode} batch/card={batch} GPUs={a.gpus} repeat={repeat}',flush=True)
                code=run_owned(cmd,folder/'worker.log',env,a.trial_timeout)
                try:result=collect(folder,len(gpu_ids),code)
                except (ValueError,OSError,KeyError,TypeError) as exc:result=dict(status='error',error=str(exc),returncode=code)
                if code==124:result['error']='trial wall-clock timeout; inspect stage log and periodic stacks'
                result.update(rgb_mode=mode,batch_size=batch,repeat=repeat,folder=str(folder))
                trials.append(result);write(folder/'trial.json',result)
                write(a.output/'summary.json',summarize(trials,a.repeats))
                print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','selection')}) ,flush=True)
                if code in (130,143):raise SystemExit(code)
                if result['status']!='measured' and not a.continue_on_error:
                    raise SystemExit(2)
    if a.prepare_only:
        write(a.output/'summary.json',dict(status='prepared_cpu_only',trials=[],fastest_measured_candidates={}))
    if any(r['status']!='measured' for r in trials):raise SystemExit(2)


if __name__=='__main__':main()
