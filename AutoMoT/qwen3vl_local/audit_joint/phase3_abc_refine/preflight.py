"""Exhaustive real processor packing and separate four-rank backward probe."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from ..abc_delivery import sha,write
from ..geometry_abc_v2 import dataset,objective
from . import runtime,bindings,probe_inputs


def processor_check(dataset_path,model_dir,output):
    from PIL import Image
    from qwen3vl_local.qwen35.backend import AutoProcessor
    import time
    started=time.monotonic();model=bindings.model_identity(model_dir)
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    counts={};maximum=0;longest=None;longest_by_variant={};extrema={}
    with runtime.activated():
        for variant in 'abc':
            artifact=dataset.load(dataset_path,variant=variant,check_sources=False)
            counts[variant]=0
            for r in artifact['records']:
                with Image.open(Path(artifact['data_root'])/r['image']) as im:rgb=im.convert('RGB')
                packed=objective.pack(bundle,r,rgb)
                probe_inputs.update(extrema,r,variant,packed)
                counts[variant]+=1
                if packed['tokens']>longest_by_variant.get(variant,{}).get('tokens',-1):
                    longest_by_variant[variant]=dict(id=r['id'],tokens=packed['tokens'])
                if packed['tokens']>maximum:maximum=packed['tokens'];longest=dict(id=r['id'],variant=variant)
                if counts[variant]%1000==0:print(f'[refine processor] {variant}: {counts[variant]}',flush=True)
    if bindings.model_identity(model_dir)!=model:raise ValueError('processor_model_changed_during_check')
    result=dict(dataset_sha256=sha(dataset_path),code=runtime.identity(),model=model,
        full_pool=True,counts=counts,max_tokens=maximum,longest=longest,longest_by_variant=longest_by_variant,
        extrema=extrema,no_truncation=True,seconds=time.monotonic()-started)
    write(output,result);return result


def probe(args):
    import torch
    import torch.distributed as dist
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase3 import train as native
    from ..geometry_abc_v2 import train
    rank,local,world=native.setup_distributed(3600)
    try:
        if world!=4:raise ValueError('four_rank_probe_required')
        model=bindings.model_identity(args.model_dir)
        processor=bindings.verify_processor(args.processor_report,args.dataset,model)
        with runtime.activated():
            artifact=dataset.load(args.dataset,variant=args.variant,check_sources=False)
            device=torch.device('cuda',local)
            bundle=native.load_model_with_lora(Path(args.model_dir),device=device,lora_rank=16,lora_alpha=32,
                lora_dropout=.05,lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
            bundle.model=torch.nn.parallel.DistributedDataParallel(bundle.model,device_ids=[local],find_unused_parameters=False)
            selection=probe_inputs.selection(artifact['records'],processor,args.variant)
            lookup={r['id']:r for r in artifact['records']}
            work=[lookup[item['id']] for item in selection['backward']]
            if any(r['split']!='train' for r in work):raise ValueError('backward_probe_train_only')
            expected={item['id']:item for item in selection['backward']+selection['holdout_forward_only']}
            def pack_record(r):
                with Image.open(Path(artifact['data_root'])/r['image']) as image:packed=objective.pack(bundle,r,image.convert('RGB'))
                probe_inputs.verify_shape(packed,expected[r['id']]);return packed
            optimizer=torch.optim.AdamW([p for p in bundle.model.parameters() if p.requires_grad],lr=1e-5)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
            output=Path(args.output)
            if rank==0:output.mkdir(parents=True,exist_ok=False)
            dist.barrier()
            result=train.train_epoch(bundle,work,optimizer,scheduler,pack_record,rank=rank,world=world)
            if result['steps']!=1 or result['status']!='epoch_complete':raise ValueError('probe_incomplete')
            # Use the unwrapped model: different ranks may have different counts
            # of holdout forwards, so these must not enter DDP collectives.
            raw_bundle=SimpleNamespace(**vars(bundle));raw_bundle.model=train._unwrap(bundle.model)
            heldout=probe_inputs.holdout_forward(raw_bundle,artifact['records'],selection['holdout_forward_only'][rank::world],
                                                 pack_record,objective.readout)
            write(output/f'rank{rank}.json',dict(backward=selection['backward'][rank],holdout_forward_only=heldout))
            binding=dict(contract=artifact['contract'],code=artifact['code'],dataset_sha256=sha(args.dataset),model=model)
            if 'admission' in artifact:
                binding.update(admission_sha256=artifact['admission']['sha256'],
                               fixed_validation_ids=[r['id'] for r in train.training_members(artifact)])
            if rank==0:
                train.save_checkpoint(output/'checkpoint',bundle,optimizer,scheduler,
                    binding=binding,
                    epoch=0,step=1,exposure=result['actual_exposure'],status='probe_only')
            dist.barrier()
            reload_probe(bundle,output/'checkpoint',artifact,args.dataset)
            if bindings.model_identity(args.model_dir)!=model:raise ValueError('probe_base_model_changed')
            dist.barrier()
            if rank==0:
                write(output/'probe_complete.json',dict(dataset_sha256=sha(args.dataset),code=runtime.identity(),
                    model=model,variant=args.variant,contract=runtime.contract(args.variant),binding=binding,
                    processor_sha256=sha(args.processor_report),checkpoint_reload_verified=True,
                    checkpoint_sha256=sha(output/'checkpoint/state.pt'),receipt_sha256=sha(output/'checkpoint/receipt.json'),
                    probe_ids=[r['id'] for r in work],input_selection=selection,
                    rank_receipts={f'rank{i}.json':sha(output/f'rank{i}.json') for i in range(world)},
                    longest_input_id=processor['extrema'][args.variant]['train']['tokens']['id'],result=result,formal_training=False))
    finally:native.cleanup_distributed()


def reload_probe(bundle,checkpoint,artifact,dataset_path):
    """Exercise the native restore path, then compare every saved trainable tensor."""
    import torch
    from ..geometry_abc_v2 import train
    state=train.restore(bundle,checkpoint,artifact=artifact,dataset_path=dataset_path)
    for name,p in train._unwrap(bundle.model).named_parameters():
        if p.requires_grad and not torch.equal(p.detach().cpu(),state['trainable'][name]):
            raise ValueError('probe_reload_tensor_mismatch')
    return state
