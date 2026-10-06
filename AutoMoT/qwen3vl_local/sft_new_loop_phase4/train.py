"""Phase4条件LoRA SFT，支持单卡/DDP、完整验证、最佳/最终保存和epoch边界恢复。"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from .dataset import load_dataset,dump_rows
from .evaluate import evaluate_rows
from .identity import ROOT,contract,digest,file_sha,write_json
from .model import load_bundle,encode
from .sampling import plan


def accumulation_size(index,total,accumulation):
    return min(accumulation,total-(index//accumulation)*accumulation)


def adapter_identity(folder):
    # training_state contains optimizer tensors and the best reference itself.
    # Bind all inference/selection assets, without a recursive self-hash.
    return {p.relative_to(folder).as_posix():file_sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='training_state.pt'}


def best_reference(folder, selected, score, epoch):
    return dict(adapter=os.path.relpath(selected.resolve(),folder.resolve()),
                score=score,epoch=epoch,files=adapter_identity(selected))


def restore_best(resume, state, run_contract):
    """Use the selected epoch's immutable history, never its parent's best.json."""
    record=state.get('best_checkpoint')
    first=state.get('next_epoch')
    if not isinstance(record,dict):
        raise ValueError('resume lacks checkpoint-local best history; use original source or a new run')
    if (type(first) is not int or first<1 or type(record.get('epoch')) is not int
            or not 0<=record['epoch']<first or type(record.get('score')) not in (int,float)
            or not math.isfinite(record['score']) or not 0<=record['score']<=1
            or state.get('best')!=record['score']
            or not isinstance(record.get('adapter'),str) or not record['adapter']
            or not isinstance(record.get('files'),dict) or not record['files']):
        raise ValueError('inconsistent checkpoint-local best history')
    selected=(resume/record['adapter']).resolve()
    if not selected.is_dir() or adapter_identity(selected)!=record['files']:
        raise ValueError('checkpoint-local best assets missing or changed')
    if json.loads((selected/'phase4_contract.json').read_text())!=run_contract:
        raise ValueError('inherited best checkpoint contract mismatch')
    selection=json.loads((selected/'selection.json').read_text())
    if selection!=dict(epoch=record['epoch'],score=record['score']):
        raise ValueError('checkpoint-local best score/epoch mismatch')
    return first,record['score'],record['epoch'],selected


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--paired-with',type=Path,help='validated opposite RGB mode; train only their causal intersection')
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
    p.add_argument('--sampling-policy',choices=('full_event_equal','event_equal','event_edge_answer','event_weighted','legacy_ring'),default=None)
    p.add_argument('--max-question-repeat',type=int,default=4)
    p.add_argument('--require-complete-coverage',action='store_true',
                   help='Optional exhaustive coverage acceptance, not needed to start training')
    a=p.parse_args()
    if min(a.epochs,a.accumulation,a.cap,a.lora_rank,a.lora_alpha,a.max_length)<1 or a.lr<=0:
        p.error('invalid training parameters')
    data,manifest=load_dataset(a.dataset,require_trainable=not a.sampling_only,
                               require_complete_coverage=a.require_complete_coverage)
    pairing=None
    if a.paired_with is not None:
        from .paired_training import load_view
        data,pairing=load_view(data,manifest,a.paired_with)
        if a.require_complete_coverage:
            from .dataset import coverage_report
            if not coverage_report(sum(data.values(),[]))['ready']:raise ValueError('paired training view lacks exhaustive coverage')
    admission=pairing['training_admission'] if pairing is not None else manifest['training_admission']
    if a.sampling_policy is None:a.sampling_policy='event_weighted' if admission.get('weak_supervision_rows') else 'event_equal'
    if a.sampling_policy=='full_event_equal':
        from .full_sampling import validate_dataset_scope
        validate_dataset_scope(manifest,pairing)
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if a.sampling_only:
        if pairing is not None and rank==0:print(json.dumps(dict(paired_training=pairing)),flush=True)
        sampling_history=None
        for epoch in range(a.epochs):
            _,audit=plan(data['train'],epoch=epoch,seed=a.seed,cap=a.cap,budget=a.epoch_samples,world_size=world,policy=a.sampling_policy,history=sampling_history,max_question_repeat=a.max_question_repeat)
            sampling_history=audit.pop('next_history',None)
            if rank==0:
                print(json.dumps(audit),flush=True)
        return
    # Capacity is independent of learned weights; reject before CUDA/model/output creation.
    plan(data['train'],epoch=0,seed=a.seed,cap=a.cap,budget=a.epoch_samples,world_size=world,policy=a.sampling_policy,max_question_repeat=a.max_question_repeat)
    if pairing is not None:
        from .dataset import coverage_report
        coverage=coverage_report(sum(data.values(),[]))
    else:coverage=manifest['coverage']
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
    config={k:getattr(a,k) for k in ('lr','accumulation','cap','epoch_samples','seed','max_length','lora_rank','lora_alpha','vision_scope','require_complete_coverage','sampling_policy','max_question_repeat')}
    config['paired_training']=pairing
    config['selection_metric']='event_macro_accuracy_observed_v1'
    run_contract=dict(source=contract(),dataset=digest(manifest),training=config,world_size=world,
                      observation_contract=manifest['observation_contract'])
    if rank==0:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        write_json(a.output_dir/'run.json',run_contract)
        write_json(a.output_dir/'data_admission.json',dict(training=admission,coverage=coverage,paired_training=pairing))
    if world>1:
        dist.barrier()
    bundle=load_bundle(a.model_dir,device,rank=a.lora_rank,alpha=a.lora_alpha,vision_scope=a.vision_scope)
    bundle.observation_contract=manifest['observation_contract']
    optimizer=torch.optim.AdamW([v for v in bundle.model.parameters() if v.requires_grad],lr=a.lr)
    first,best=0,float('-inf')
    sampling_history=None
    best_epoch,best_path=None,None
    if a.resume:
        if json.loads((a.resume/'phase4_contract.json').read_text())!=run_contract:
            raise ValueError('resume source/dataset/training/world-size mismatch')
        validate_adapter(a.resume,bundle.model)
        state=torch.load(a.resume/'training_state.pt',map_location=device,weights_only=True)
        first,best,best_epoch,best_path=restore_best(a.resume,state,run_contract)
        sampling_history=state.get('sampling_history')
        if a.sampling_policy in ('full_event_equal','event_equal','event_edge_answer','event_weighted') and sampling_history is None:
            raise ValueError('resume lacks actual sampling exposure history')
        validate_adapter(best_path,bundle.model)
        if first>=a.epochs:
            raise ValueError('resume has already reached requested epoch count')
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        set_peft_model_state_dict(bundle.model,load_peft_weights(str(a.resume),device=str(device)))
        optimizer.load_state_dict(state['optimizer'])
        if rank==0:
            write_json(a.output_dir/'best.json',dict(
                adapter=os.path.relpath(best_path,a.output_dir.resolve()),
                score=best,epoch=best_epoch,inherited=True))
    if world>1:
        bundle.model=DistributedDataParallel(bundle.model,device_ids=[local] if device.type=='cuda' else None,
                                             find_unused_parameters=False)
    for epoch in range(first,a.epochs):
        # Epoch seed makes epoch-boundary recovery independent of prior validation RNG use.
        torch.manual_seed(a.seed+epoch*world+rank)
        random.seed(a.seed+epoch*world+rank)
        indices,audit=plan(data['train'],epoch=epoch,seed=a.seed,cap=a.cap,budget=a.epoch_samples,world_size=world,policy=a.sampling_policy,history=sampling_history,max_question_repeat=a.max_question_repeat)
        next_sampling_history=audit.pop('next_history',None)
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
            # Each observed event has one vote, independent of its subgroup count.
            # The fixed ten-event score remains null while any event is missing.
            score=report['event_macro_accuracy_observed']
            if not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('invalid validation selection score')
            improved=score>best
            best=max(best,score)
            report.update(epoch=epoch,macro_score=score,split='val',dataset=digest(manifest),
                          coverage_complete=coverage['ready'],
                          evaluation_scope=admission['evaluation_scope'],
                          expected_count=len(data['val']),train_loss_rank0=total_loss/len(local_indices))
            write_json(a.output_dir/f'epoch_{epoch:03d}_validation.json',report)
            dump_rows(a.output_dir/f'epoch_{epoch:03d}_cases.jsonl',cases)
            folder=a.output_dir/f'epoch_{epoch:03d}'
            save_adapter(raw,folder)
            bundle.processor.save_pretrained(folder)
            write_json(folder/'phase4_contract.json',run_contract)
            write_json(folder/'selection.json',dict(epoch=epoch,score=score))
            if improved:
                best_path,best_epoch=folder,epoch
            selected=best_reference(folder,best_path,best,best_epoch)
            torch.save(dict(optimizer=optimizer.state_dict(),next_epoch=epoch+1,best=best,
                            best_checkpoint=selected,sampling_history=next_sampling_history),folder/'training_state.pt')
            write_json(a.output_dir/'latest.json',dict(adapter=folder.name,epoch=epoch))
            if improved:
                write_json(a.output_dir/'best.json',dict(adapter=folder.name,score=score,epoch=epoch))
            if epoch+1==a.epochs:
                write_json(a.output_dir/'final_generation.json',report)
        if world>1:
            dist.barrier()
        sampling_history=next_sampling_history
    if world>1:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
