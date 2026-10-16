"""Bounded CUDA/DDP and one-example 4B execution diagnosis, NOT data admission.

Never loads the full pool, creates a student view, or saves model weights.
The model check deliberately repeats one train example across ranks: this is a
hardware/encoder/backward check, not event-balanced training or a batch ranking.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

from .batch_benchmark import write
from .benchmark_runtime import progress, run_owned, trace_waits


def example(native, root, mode):
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import short_row, four_row
    from qwen3vl_local.sft_new_loop_phase4.identity import check_contract, file_sha
    from qwen3vl_local.sft_new_loop_phase4.input_identity import rgb_content_sha
    from qwen3vl_local.sft_new_loop_phase4.data_paths import data_path
    manifest=json.loads((native/'manifest.json').read_text())
    check_contract(manifest['contract'])
    if manifest['rgb_mode']!=4:raise ValueError('native RGB4 required')
    # Bounded prefix, not a hidden full-pool scan or an admitted training subset.
    selected=None
    with (native/'train.jsonl').open('rb') as stream:
        for _ in range(100):
            line=stream.readline(8*1024*1024+1)
            if not line:break
            if len(line)>8*1024*1024:raise ValueError('oversized diagnostic row')
            if not line.strip():continue
            row=json.loads(line)
            if row['split']!='train':raise ValueError('non-training row in diagnostic input')
            if row['target'] in ('YES','NO'):
                selected=row;break
    if selected is None:raise ValueError('no YES/NO train example in first 100 lines')
    original=dict(zip(selected['images'],selected['image_sha256']))
    row=short_row(selected)
    if mode==4:
        frame=selected['observation']['frame_id'];last=Path(selected['images'][-1])
        paths=[last.with_name(f'{f:04d}'+last.suffix).as_posix() for f in range(frame-3,frame+1)]
    else:paths=row['images']
    identities={}
    for rel in paths:
        # Match the formal encoder's logical dataset namespace: routes/assets
        # may be symlinks to other disks. data_path still rejects traversal.
        path=data_path(root,rel)
        before=file_sha(path)
        if rel in original and before!=original[rel]:raise ValueError('native RGB SHA mismatch')
        with Image.open(path) as image:rgb=rgb_content_sha(image.convert('RGB'))
        if file_sha(path)!=before:raise ValueError('RGB changed while reading')
        identities[rel]=dict(sha256=before,rgb_sha256=rgb)
    if mode==4:row=four_row(selected,identities)
    return row


def worker(a):
    trace_waits()
    progress('torch import START')
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    progress(f'rank={rank} CUDA allocation and matrix multiply START')
    x=torch.ones((128,128),device=device);value=x@x
    if not torch.all(value==128):raise ValueError('CUDA matrix result mismatch')
    progress(f'rank={rank} NCCL init/all_reduce START')
    dist.init_process_group('nccl',timeout=timedelta(seconds=30))
    total=torch.tensor(float(rank+1),device=device);dist.all_reduce(total)
    if total.item()!=world*(world+1)/2:raise ValueError('NCCL sum mismatch')
    result=dict(status='transport_passed',rank=rank,world=world,gpu_name=torch.cuda.get_device_name(local),
                torch=torch.__version__,cuda=torch.version.cuda,full_pool_admitted=False,
                scope='runtime smoke only; no batch capacity, event balance, quality or training admission claim')
    if a.check=='model':
        from qwen3vl_local.sft_v2.train import load_model_with_lora
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.contract import observation_contract
        from qwen3vl_local.sft_new_loop_phase4.rgb_short.model import encode_batch
        progress(f'rank={rank} one real train example START')
        row=example(a.native,a.data_root,a.rgb_mode)
        ids=[None]*world;dist.all_gather_object(ids,row['id'])
        if len(set(ids))!=1:raise ValueError('diagnostic row differs across ranks')
        progress(f'rank={rank} model load START')
        bundle=load_model_with_lora(a.model_dir,device=device,lora_rank=16,lora_alpha=32,
            lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
        bundle.observation_contract=observation_contract(a.rgb_mode)
        bundle.model=torch.nn.parallel.DistributedDataParallel(bundle.model,device_ids=[local],find_unused_parameters=False)
        bundle.model.train()
        params=[p for p in bundle.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        progress(f'rank={rank} formal encode_batch START')
        inputs=encode_batch(bundle,[row],a.data_root,max_length=8192)
        progress(f'rank={rank} forward START')
        loss=bundle.model(**inputs,use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError('nonfinite smoke loss')
        progress(f'rank={rank} backward/DDP START')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite smoke gradients')
        progress(f'rank={rank} AdamW step START')
        optimizer.step();torch.cuda.synchronize()
        result.update(status='model_step_passed',rgb_mode=a.rgb_mode,batch_per_rank=1,
                      distinct_examples=1,repeated_across_ranks=True,example_id=row['id'],
                      loss=float(loss.detach()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(a.output/f'rank{rank}.json',result)
    progress(f'rank={rank} {result["status"]}')
    dist.destroy_process_group()


def main():
    from .guard import no_core
    no_core()
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--check',choices=('transport','model'),default='transport')
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--native',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('lead_data'))
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--rgb-mode',type=int,choices=(2,4),default=2)
    p.add_argument('--timeout',type=int,default=60)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();cards=a.gpus.split(',')
    if a.timeout<1 or any(not c.isdigit() for c in cards) or len(cards)!=len(set(cards)):p.error('positive timeout and distinct numeric GPUs required')
    if a.check=='model' and (a.native is None or not (a.native/'train.jsonl').is_file() or not (a.model_dir/'config.json').is_file()):p.error('model check needs native train.jsonl and model config.json')
    if a.worker:worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(cards)}',
         '--monitor-interval=1','-m','qwen3vl_local.audit_joint.phase4_gpu_smoke','--worker',
         '--check',a.check,'--gpus',a.gpus,'--output',str(a.output),'--rgb-mode',str(a.rgb_mode),
         '--data-root',str(a.data_root),'--model-dir',str(a.model_dir)]
    if a.native:cmd+=['--native',str(a.native)]
    progress(f'output={a.output}; check={a.check}; wall limit={a.timeout}s')
    code=run_owned(cmd,a.output/'worker.log',dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,
        OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),a.timeout)
    ranks=[]
    for rank in range(len(cards)):
        path=a.output/f'rank{rank}.json'
        if path.is_file():ranks.append(json.loads(path.read_text()))
    expected='model_step_passed' if a.check=='model' else 'transport_passed'
    passed=code==0 and len(ranks)==len(cards) and all(r['rank']==i and r['world']==len(cards) and r['status']==expected for i,r in enumerate(ranks))
    write(a.output/'summary.json',dict(status=expected if passed else 'failed',returncode=code,
        full_pool_admitted=False,ranks=ranks,scope='runtime diagnosis only; never a training or benchmark recommendation'))
    raise SystemExit(0 if passed else code or 2)


if __name__=='__main__':main()
