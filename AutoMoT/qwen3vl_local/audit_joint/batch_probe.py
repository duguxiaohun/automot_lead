"""Measure real training minibatch CUDA peaks, in an isolated process per trial.

No adapter/checkpoint is saved. Each trial includes forward, backward, clipping,
AdamW state allocation and subsequent updates. Run with the training environment.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--phase',choices=('3','4'),required=True)
    p.add_argument('--index',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0',help='Visible physical GPU IDs, tested sequentially')
    p.add_argument('--batches',type=int,nargs='+',default=list(range(1,9)))
    p.add_argument('--modes',type=int,nargs='+',choices=(2,4),default=[2,4])
    p.add_argument('--steps-per-event',type=int,default=2)
    p.add_argument('--max-length',type=int,default=8192)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--lora-alpha',type=int,default=32)
    p.add_argument('--vision-scope',choices=('off','merger','last4','all'),default='off')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--prompt-variant',default='baseline')
    p.add_argument('--headroom-fraction',type=float,default=.15)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p


def worker(a):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for measured capacity; no CPU or linear extrapolation')
    from qwen3vl_local.sequence_batch import digest
    mode=a.modes[0];batch_size=a.batches[0]
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if a.phase=='3':
        from qwen3vl_local.sft_new_loop_phase3 import train as t
        from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
        check_index(a.index,action_output_mode='choice',prompt_variant=a.prompt_variant)
        rows=t._read_rows(a.index,'train',data_root=a.data_root)
        items=[t._make_item(r,seed=20260904,action_output_mode='choice') for r in rows]
        items=[i for i in items if t.choice_rejection_reason(i.spec) is None]
        pools=defaultdict(list)
        for item in items:pools[item.row.context_id].append(item)
        if set(pools)!=set(t.CONTEXT_IDS):raise ValueError('missing training events')
        bundle=t.load_model_with_lora(a.model_dir,device=device,lora_rank=a.lora_rank,
            lora_alpha=a.lora_alpha,lora_dropout=.05,lora_vision_scope=a.vision_scope,
            strict_vision_scope=True,gradient_checkpointing=a.gradient_checkpointing)
        history_mode='2rgb_short' if mode==2 else '4rgb'
        def loss_for(batch):
            packed=[];image_cache={}
            for item in batch:
                images=t._load_images(t.select_history_rgb_paths(item.row.history_rgb_paths,history_mode),cache=image_cache)
                packed.append(t._build_inputs(bundle,images=images,history_rgb_mode=history_mode,
                    prompt_variant=a.prompt_variant,target=t.build_action_target(item.spec),spec=item.spec,
                    max_length=a.max_length,format_loss_weight=.25))
            if any(p is None for p in packed):raise ValueError('sample exceeds token limit')
            return t._loss_batch(bundle,packed,[i.spec for i in batch])[0],max(len(p['input_ids']) for p in packed)
        identity=lambda i:[i.row.scenario,i.row.route_id,i.row.frame_id,i.row.context_id]
    else:
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import load_dataset
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.model import load_bundle,encode_batch
        from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
        data,manifest=load_dataset(a.dataset,rgb_mode=mode)
        pools=defaultdict(list)
        for row in data['train']:
            if row['target'] in ('YES','NO'):pools[row['episode']['event']].append(row)
        if set(pools)!=set(EVENTS):raise ValueError('missing training events')
        bundle=load_bundle(a.model_dir,device,rank=a.lora_rank,alpha=a.lora_alpha,
            vision_scope=a.vision_scope,checkpointing=a.gradient_checkpointing)
        bundle.observation_contract=manifest['observation_contract']
        def loss_for(batch):
            inputs=encode_batch(bundle,batch,a.data_root,max_length=a.max_length)
            return bundle.model(**inputs,use_cache=False).loss,inputs['input_ids'].shape[1]
        identity=lambda r:r['id']
    # Fixed training-only sample selection; report IDs so coverage limits are explicit.
    for e in pools:pools[e].sort(key=lambda r:digest(identity(r)))
    params=[p for p in bundle.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-5)
    bundle.model.train()
    torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
    baseline_allocated=torch.cuda.memory_allocated();baseline_reserved=torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    records=[];start=time.perf_counter()
    for iteration in range(a.steps_per_event):
        for event,pool in sorted(pools.items()):
            batch=[pool[(iteration*batch_size+k)%len(pool)] for k in range(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            before=time.perf_counter();loss,tokens=loss_for(batch)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite probe loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            torch.cuda.synchronize()
            records.append(dict(event=event,iteration=iteration,ids=[identity(r) for r in batch],
                tokens=int(tokens),loss=float(loss.detach()),seconds=time.perf_counter()-before))
    peak_a=torch.cuda.max_memory_allocated();peak_r=torch.cuda.max_memory_reserved()
    other=max(0,total-free-baseline_reserved)
    return dict(status='measured',phase=a.phase,rgb_mode=mode,batch_size=batch_size,
        gpu_name=torch.cuda.get_device_name(),total_bytes=total,free_after_load_bytes=free,
        baseline_allocated_bytes=baseline_allocated,peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r,
        other_process_bytes_at_start=other,seconds=time.perf_counter()-start,
        fits_with_headroom=peak_r+other<=total*(1-a.headroom_fraction),
        headroom_fraction=a.headroom_fraction,records=records,
        configuration=dict(gradient_checkpointing=a.gradient_checkpointing,lora_rank=a.lora_rank,
            lora_alpha=a.lora_alpha,vision_scope=a.vision_scope,max_length=a.max_length,prompt_variant=a.prompt_variant),
        event_counts=dict.fromkeys(pools,a.steps_per_event*batch_size),
        scope='single-GPU real train steps including AdamW; sampled train inputs only; DDP/full-pool peak not certified')


def main():
    a=parser().parse_args()
    if a.gradient_checkpointing is None:a.gradient_checkpointing=a.phase=='4'
    if (min(a.batches+[a.steps_per_event,a.max_length])<1 or not 0<a.headroom_fraction<1
            or (a.phase=='3' and a.index is None) or (a.phase=='4' and a.dataset is None)):
        raise ValueError('invalid probe inputs')
    if a.worker:
        try:result=worker(a)
        except Exception as exc:
            import torch
            result=dict(status='oom' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'error',
                        error_type=type(exc).__name__,error=str(exc),phase=a.phase,rgb_mode=a.modes[0],batch_size=a.batches[0])
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        return
    gpu_ids=[g.strip() for g in a.gpus.split(',')]
    if any(not g.isdigit() for g in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError('GPU IDs must be distinct numeric IDs')
    a.gpus=','.join(gpu_ids)
    a.modes=list(dict.fromkeys(a.modes))
    a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for gpu in a.gpus.split(','):
        if not gpu.strip().isdigit():raise ValueError('GPU IDs must be numeric')
        for mode in a.modes:
            for batch in sorted(set(a.batches)):
                output=a.output/f'gpu{gpu}_rgb{mode}_batch{batch}.json'
                cmd=[sys.executable,'-m',__spec__.name,'--worker','--phase',a.phase,
                    '--data-root',str(a.data_root),'--model-dir',str(a.model_dir),'--output',str(output),
                    '--modes',str(mode),'--batches',str(batch),'--steps-per-event',str(a.steps_per_event),
                    '--max-length',str(a.max_length),'--lora-rank',str(a.lora_rank),'--lora-alpha',str(a.lora_alpha),
                    '--vision-scope',a.vision_scope,'--prompt-variant',a.prompt_variant,
                    '--headroom-fraction',str(a.headroom_fraction),
                    '--gradient-checkpointing' if a.gradient_checkpointing else '--no-gradient-checkpointing']
                for key in ('index','dataset'):
                    if getattr(a,key) is not None:cmd.extend(['--'+key,str(getattr(a,key))])
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TOKENIZERS_PARALLELISM='false')
                print(f'[probe] GPU={gpu} phase={a.phase} RGB={mode} per-card batch={batch}',flush=True)
                with output.with_suffix('.log').open('w') as log:
                    code=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                result=json.loads(output.read_text()) if output.exists() else dict(status='error',returncode=code)
                result.update(gpu=gpu,rgb_mode=mode,batch_size=batch);results.append(result)
                (a.output/'summary.json').write_text(json.dumps(dict(trials=results),indent=2)+'\n')
                print(json.dumps({k:v for k,v in result.items() if k not in ('records',)}),flush=True)
                if result['status'] in ('oom','error'):break
    recommended={str(mode):min((max((r['batch_size'] for r in results if r['gpu']==gpu and r['rgb_mode']==mode
        and r['status']=='measured' and r['fits_with_headroom']),default=0) for gpu in a.gpus.split(',')),default=0)
        for mode in a.modes}
    (a.output/'summary.json').write_text(json.dumps(dict(trials=results,
        conservative_measured_candidates= recommended,
        note='Candidate is the largest TESTED batch fitting on EVERY tested GPU with headroom, not maximum capacity. Confirm with DDP before full training.'),indent=2)+'\n')
    if any(r['status']=='error' for r in results):raise SystemExit(2)

if __name__=='__main__':main()
