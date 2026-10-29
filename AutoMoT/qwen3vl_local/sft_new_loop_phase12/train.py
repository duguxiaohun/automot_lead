"""Two full-gradient objectives, one shared LoRA and one optimizer update per pair."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from qwen3vl_local.sft_v2.train import load_model_with_lora, make_scheduler
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.qwen35.adapters import save_adapter
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from . import EPOCHS,WORLD,UPDATES,CONTRACT
from . import data,plan,prompts,objective,evaluate,integrity,cache,validation


def save(bundle,path,root,mode,step):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    save_adapter(bundle.unwrap(),path,safe_serialization=True)
    proof.write(path/'fusion_config.json',dict(contract=CONTRACT,mode=mode,step=step,
        prompt=prompts.identity(mode),budget=integrity.budget(),ready_sha256=proof.sha(Path(root)/'ready.json')))
    proof.stamp_adapter(path)


def validate(bundle,index,data_root,mode,output,step,root,best):
    bare=bundle.model;bundle.model=bundle.unwrap()
    candidate=output/f'val_step_{step}'
    try:
        evaluate.run(bundle,index,data_root,mode,'val',candidate)
        validation.run(bundle,root,mode,candidate/'weak_guard')
    finally:bundle.model=bare
    review=validation.decision(candidate,output/f"val_step_{best['step']}" if best else None,
                               proof.read(root/'cpu/guard_coverage.json'))
    score=review['score'];promote=best is None or (review['eligible'] and score>best['score'])
    proof.write(candidate/'selection_review.json',dict(**review,promoted=promote))
    if promote:
        name=f'best_step_{step}';save(bundle,output/name,root,mode,step)
        best=dict(score=score,step=step,adapter=name,selection='validation old-correct veto; weak recall, min task exact, mean, weak F1; test never selects')
        proof.write(output/'selection.json',best)
    bundle.model.train();return best


def probe(bundle,index,data_root,mode,output,facts):
    """Check both question domains and full-vs-prefix logits before any update."""
    bundle.model.eval();seen=set();reports=[]
    for item in plan.evaluation(index,data_root,'val'):
        if item.row.question_domain in seen:continue
        seen.add(item.row.question_domain)
        answers=facts[plan.identity(item.row)];first,second=prompts.specs(item,answers)
        texts=prompts.texts(first,second,mode);targets=prompts.targets(answers,first,second)
        images=plan.native._load_images([item.row.history_rgb_paths[i] for i in history_rgb_indices(mode)])
        try:
            for b,spec in enumerate((first,second)):
                objective.pack(bundle,images,texts[b],targets[b],spec)
            reports.append(cache.generate(bundle,images,texts,check_full=True))
        finally:
            for im in images:im.close()
    if len(seen)!=2:raise ValueError('Both question domains required for GPU pre-training probe')
    proof.write(output/'gpu_prefix_probe.json',dict(status='passed_before_optimizer_updates',reports=reports))
    bundle.model.train()


def run(args):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=WORLD:raise ValueError('Fusion budget requires four ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    root=args.run_dir.resolve();output=root/'runs'/args.mode
    try:
        ready=integrity.verify_ready(root,images=rank==0,execution=True);index=root/'data/frame_index.jsonl'
        if rank==0:output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        torch.manual_seed(plan.SEED);torch.cuda.manual_seed_all(plan.SEED)
        facts=data.labels(index);rows=plan.native._read_rows(index,'train',data_root=Path(ready['data_root']))
        sampler_args=plan.arguments(output,ready['action_index'])
        # Exercise all six epochs before model allocation on every rank.
        work=plan.work(rows,sampler_args,0);expected=proof.read(root/'cpu/plan.json')
        if plan.summarize(work,facts)!=expected['epochs'][0]:raise ValueError('Training selection differs from CPU admission')
        if rank==0:
            proof.write(output/'training_inputs.json',dict(contract=CONTRACT,budget=integrity.budget(),
                prompt=prompts.identity(args.mode),ready_sha256=proof.sha(root/'ready.json'),
                base_model=str(args.model_dir),rank_world=WORLD,initialization='fresh_base_no_adapter'))
            inputs=proof.read(output/'training_inputs.json');inputs['environment']=proof.environment()
            proof.write(output/'training_inputs.json',inputs)
            proof.write(output/'training_budget.json',integrity.budget())
        bundle=load_model_with_lora(args.model_dir,device=device,lora_rank=16,lora_alpha=32,lora_dropout=.05,
            lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
        if rank==0:probe(bundle,index,ready['data_root'],args.mode,output,facts)
        dist.barrier()
        bundle.model=DDP(bundle.model,device_ids=[local],output_device=local,find_unused_parameters=False)
        params=[p for p in bundle.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.,betas=(.9,.95))
        scheduler=make_scheduler(optimizer,total_steps=UPDATES,warmup_steps=2000)
        step=0;best=None;started=time.perf_counter();bundle.model.train()
        for epoch in range(EPOCHS):
            epoch_start=time.perf_counter()
            if epoch:work=plan.work(rows,sampler_args,epoch)
            if plan.summarize(work,facts)!=expected['epochs'][epoch]:raise ValueError('Epoch membership changed')
            for item in work[rank::WORLD]:
                answers=facts[plan.identity(item.row)];first,second=prompts.specs(item,answers)
                texts=prompts.texts(first,second,args.mode);targets=prompts.targets(answers,first,second)
                images=plan.native._load_images([item.row.history_rgb_paths[i] for i in history_rgb_indices(args.mode)])
                optimizer.zero_grad(set_to_none=True);losses=[]
                try:
                    for branch,spec in enumerate((first,second)):
                        # First branch's gradients synchronize together with the second branch.
                        with bundle.model.no_sync() if branch==0 else nullcontext():
                            packed=objective.pack(bundle,images,texts[branch],targets[branch],spec)
                            loss,_=objective.loss(bundle,packed,spec)
                            proof.finite_loss(loss,distributed=dist)
                            (loss*.5).backward();losses.append(float(loss.detach()))
                            del packed,loss
                finally:
                    for im in images:im.close()
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                proof.finite_loss(norm,distributed=dist)
                if norm.item()==0:raise ValueError('Zero gradient; refusing untrained optimizer step')
                optimizer.step();scheduler.step();step+=1
                sampler_args._exposure.record(item,valid=True);sampler_args._exposure.optimizer_applied()
                if step%20==0 and rank==0:
                    print(f'[fusion-train:{args.mode}] {step}/{UPDATES} loss_p1={losses[0]:.5f} loss_p2={losses[1]:.5f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
                if time.perf_counter()-epoch_start>86400:raise TimeoutError('Fusion epoch exceeded 24 hour budget')
                if step%1000==0 or step==UPDATES:
                    dist.barrier()
                    if rank==0:best=validate(bundle,index,ready['data_root'],args.mode,output,step,root,best)
                    dist.barrier()
            sampler_args._exposure.finish_epoch(output,rank=rank,step=step,epoch=epoch+1,distributed=dist,planned=len(work))
        if step!=UPDATES:raise ValueError('Incorrect optimizer updates')
        sampler_args._exposure.finish_run(output,rank=rank,step=step,distributed=dist)
        proof.write(output/'exposure'/f'optimizer_{rank}.json',dict(optimizer_updates=step,loop_steps=step))
        proof.write(output/'exposure'/f'branches_{rank}.json',dict(phase1=step,phase2=step,optimizer_updates=step))
        dist.barrier()
        if rank==0:
            save(bundle,output/'final',root,args.mode,step)
            integrity.verify_ready(root)
            integrity.finish_training(output,root,args.mode,best['adapter'])
            print('[TRAINED] shared LoRA; fresh test reload and quality audit pending',flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--mode',choices=('2rgb_endpoints','4rgb'),required=True)
    run(p.parse_args())

if __name__=='__main__':main()
