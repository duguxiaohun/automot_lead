"""Phase4条件LoRA SFT，支持单卡/DDP、完整验证、最佳/最终保存和epoch边界恢复。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
from .dataset import load_dataset,dump_rows
from .evaluate import evaluate_rows
from .identity import ROOT,contract,digest,write_json
from .model import load_bundle,encode
from .sampling import plan


def accumulation_size(index,total,accumulation):
    return min(accumulation,total-(index//accumulation)*accumulation)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--model-dir',type=Path,default=ROOT.parents[1]/'checkpoints/Qwen3.5-4B')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=7)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--cap',type=int,default=8)
    p.add_argument('--epoch-samples',type=int)
    p.add_argument('--seed',type=int,default=20260929)
    p.add_argument('--max-length',type=int,default=8192)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--lora-alpha',type=int,default=32)
    p.add_argument('--vision-scope',choices=('off','merger','last4','all'),default='off')
    p.add_argument('--sampling-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.accumulation,a.cap,a.lora_rank,a.lora_alpha,a.max_length)<1 or a.lr<=0:
        p.error('invalid training parameters')
    data,manifest=load_dataset(a.dataset,require_ready=not a.sampling_only)
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if a.sampling_only:
        for epoch in range(a.epochs):
            _,audit=plan(data['train'],epoch=epoch,seed=a.seed,cap=a.cap,budget=a.epoch_samples,world_size=world)
            if rank==0:
                print(json.dumps(audit),flush=True)
        return
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from qwen3vl_local.qwen35.adapters import save_adapter,validate_adapter
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device)
    if world>1:
        dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    config={k:getattr(a,k) for k in ('lr','accumulation','cap','epoch_samples','seed','max_length','lora_rank','lora_alpha','vision_scope')}
    run_contract=dict(source=contract(),dataset=digest(manifest),training=config,world_size=world,
                      observation_contract=manifest['observation_contract'])
    if rank==0:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        write_json(a.output_dir/'run.json',run_contract)
    if world>1:
        dist.barrier()
    bundle=load_bundle(a.model_dir,device,rank=a.lora_rank,alpha=a.lora_alpha,vision_scope=a.vision_scope)
    bundle.observation_contract=manifest['observation_contract']
    optimizer=torch.optim.AdamW([v for v in bundle.model.parameters() if v.requires_grad],lr=a.lr)
    first,best=0,float('-inf')
    if a.resume:
        if json.loads((a.resume/'phase4_contract.json').read_text())!=run_contract:
            raise ValueError('resume source/dataset/training/world-size mismatch')
        validate_adapter(a.resume,bundle.model)
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        set_peft_model_state_dict(bundle.model,load_peft_weights(str(a.resume),device=str(device)))
        state=torch.load(a.resume/'training_state.pt',map_location=device,weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        first,best=state['next_epoch'],state['best']
        if rank==0:
            inherited=json.loads((a.resume.parent/'best.json').read_text())
            previous_best=(a.resume.parent/inherited['adapter']).resolve()
            if json.loads((previous_best/'phase4_contract.json').read_text())!=run_contract:
                raise ValueError('inherited best checkpoint contract mismatch')
            validate_adapter(previous_best,bundle.model)
            inherited['adapter']=os.path.relpath(previous_best,a.output_dir.resolve())
            inherited['inherited']=True
            write_json(a.output_dir/'best.json',inherited)
        if first>=a.epochs:
            raise ValueError('resume has already reached requested epoch count')
    if world>1:
        bundle.model=DistributedDataParallel(bundle.model,device_ids=[local] if device.type=='cuda' else None,
                                             find_unused_parameters=False)
    for epoch in range(first,a.epochs):
        # Epoch seed makes epoch-boundary recovery independent of prior validation RNG use.
        torch.manual_seed(a.seed+epoch*world+rank)
        random.seed(a.seed+epoch*world+rank)
        indices,audit=plan(data['train'],epoch=epoch,seed=a.seed,cap=a.cap,budget=a.epoch_samples,world_size=world)
        local_indices=indices[rank::world]
        if rank==0:
            write_json(a.output_dir/f'epoch_{epoch:03d}_sampling.json',audit)
        bundle.model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.
        for j,index in enumerate(local_indices):
            inputs=encode(bundle,data['train'][index],a.data_root,max_length=a.max_length)
            size=accumulation_size(j,len(local_indices),a.accumulation)
            update=(j+1)%a.accumulation==0 or j+1==len(local_indices)
            cm=bundle.model.no_sync() if world>1 and not update else nullcontext()
            with cm:
                loss=bundle.model(**inputs,use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                (loss/size).backward()
            total_loss+=float(loss.detach())
            if update:
                torch.nn.utils.clip_grad_norm_(bundle.model.parameters(),1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if rank==0 and j%20==0:
                print(f'[train] epoch={epoch} row={j+1}/{len(local_indices)} loss={float(loss):.5f}',flush=True)
        if world>1:
            dist.barrier()
        if rank==0:
            raw=bundle.model.module if world>1 else bundle.model
            raw.eval()
            report,cases=evaluate_rows(bundle,data['val'],a.data_root,a.max_length)
            # Macro event/edge/slice score; full val support is checked before training.
            score=sum(g['accuracy'] for g in report['groups'].values())/len(report['groups'])
            improved=score>best
            best=max(best,score)
            report.update(epoch=epoch,macro_score=score,split='val',dataset=digest(manifest),
                          expected_count=len(data['val']),train_loss_rank0=total_loss/len(local_indices))
            write_json(a.output_dir/f'epoch_{epoch:03d}_validation.json',report)
            dump_rows(a.output_dir/f'epoch_{epoch:03d}_cases.jsonl',cases)
            folder=a.output_dir/f'epoch_{epoch:03d}'
            save_adapter(raw,folder)
            bundle.processor.save_pretrained(folder)
            write_json(folder/'phase4_contract.json',run_contract)
            torch.save(dict(optimizer=optimizer.state_dict(),next_epoch=epoch+1,best=best),folder/'training_state.pt')
            write_json(a.output_dir/'latest.json',dict(adapter=folder.name,epoch=epoch))
            if improved:
                write_json(a.output_dir/'best.json',dict(adapter=folder.name,score=score,epoch=epoch))
            if epoch+1==a.epochs:
                write_json(a.output_dir/'final_generation.json',report)
        if world>1:
            dist.barrier()
    if world>1:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
