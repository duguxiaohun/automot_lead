"""Immutable local epoch-boundary recovery, including exposure and optimizer state.

A resume continues the original total-epoch/LR plan into a fresh output folder.
Receipts detect incomplete/changed local checkpoints; they are not attestations.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import shutil

POLICY = 'phase3_epoch_training_state_v1'
SLOTS = ('best_val', 'best_generation', 'fallback_generation')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def binding(args, world, steps_per_epoch):
    root=Path(__file__).resolve().parents[1]
    paths={p for name in ('sft_new_loop_phase3','sft_v2','qwen35')
           for p in (root/name).rglob('*.py') if not p.name.startswith('test_') and 'tests' not in p.parts}
    paths.update(root/name for name in ('sequence_batch.py','batch_runtime.py','image_io.py','mrope_utils.py','cache_utils.py'))
    ignored={'output_dir','resume','save_training_state','stop_after_epoch','no_tb','log_steps','device','ddp_timeout_seconds'}
    config={k:v for k,v in vars(args).items() if k not in ignored}
    # Local base assets are immutable for recovery. Stat binding avoids rehashing
    # multi-GB model shards on each rank; metadata contents are hashed as well.
    base=Path(args.model_dir).resolve(strict=True)
    assets={}
    for p in sorted(base.rglob('*')):
        if p.is_file():
            s=p.stat()
            assets[str(p.relative_to(base))]=dict(resolved=str(p.resolve()),device=s.st_dev,inode=s.st_ino,
                bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,
                sha256=sha(p) if p.suffix in ('.json','.jinja','.txt','.model') else None)
    result=dict(policy=POLICY,world_size=world,steps_per_epoch=steps_per_epoch,configuration=config,
                sources={str(p.relative_to(root)):sha(p) for p in sorted(paths)},base_assets=assets)
    return json.loads(json.dumps(result,sort_keys=True))


def rng_state(device):
    import torch
    return dict(python=random.getstate(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None)


def restore_rng(state,device):
    import torch
    random.setstate(state['python']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def files(folder):
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name!='receipt.json'}


def save(folder,output,model,optimizer,scheduler,*,binding,state,rng):
    import torch
    folder=Path(folder);output=Path(output);pending=folder.with_name(folder.name+'.pending')
    if folder.exists() or pending.exists():raise FileExistsError(folder)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('epoch checkpoint requires cleared gradients at an optimizer boundary')
    folder.parent.mkdir(parents=True,exist_ok=True)
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
        if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
        return 0
    selection_bytes=sum(p.stat().st_size for slot in SLOTS for p in (output/slot).rglob('*') if p.is_file())
    needed=2*(tensor_bytes(optimizer.state_dict())+sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad))+selection_bytes+64*1024**2
    if shutil.disk_usage(folder.parent).free<20*1024**3+needed:raise ValueError('insufficient checkpoint disk reserve')
    pending.mkdir()
    try:
        selected=pending/'selection';selected.mkdir()
        for slot in SLOTS:
            if (output/slot).exists():shutil.copytree(output/slot,selected/slot)
            if (output/(slot+'.json')).exists():shutil.copyfile(output/(slot+'.json'),selected/(slot+'.json'))
        value=dict(state,binding=binding,rng=rng,
                   trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        torch.save(value,pending/'training_state.pt')
        (pending/'receipt.json').write_text(json.dumps(dict(policy=POLICY,binding=binding,files=files(pending)),sort_keys=True)+'\n')
        pending.rename(folder)
        pointer=output/'latest_training_state.pending'
        pointer.write_text(json.dumps(dict(checkpoint=str(folder.resolve()),next_epoch=state['next_epoch'],global_step=state['global_step']))+'\n')
        os.replace(pointer,output/'latest_training_state.json')
    except BaseException:
        shutil.rmtree(pending,ignore_errors=True)
        raise


def read(folder,expected):
    import torch
    folder=Path(folder);receipt=folder/'receipt.json';before=sha(receipt)
    record=json.loads(receipt.read_text())
    if record.get('policy')!=POLICY or record.get('binding')!=expected:
        raise ValueError('resume source/data/configuration/base/world-size mismatch')
    if record.get('files')!=files(folder):raise ValueError('resume checkpoint files changed or incomplete')
    state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True)
    epoch=state.get('next_epoch');history=state.get('sampling_history')
    if (state.get('binding')!=expected or type(epoch) is not int or not 0<epoch<expected['configuration']['num_epochs']
            or state.get('global_step')!=epoch*expected['steps_per_epoch']
            or not isinstance(history,dict) or history.get('next_epoch')!=epoch
            or len(state.get('rng',[]))!=expected['world_size']):
        raise ValueError('resume epoch/step/history incomplete or requested plan already completed')
    if sha(receipt)!=before or files(folder)!=record['files']:raise ValueError('resume changed during read')
    return state


def restore(state,model,optimizer,scheduler,device,rank):
    import torch
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    saved=state['trainable']
    if params.keys()!=saved.keys() or any(params[n].shape!=saved[n].shape or params[n].dtype!=saved[n].dtype for n in params):
        raise ValueError('resume trainable parameter identity mismatch')
    with torch.no_grad():
        for name,p in params.items():p.copy_(saved[name].to(p.device))
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'][rank],device)


def restore_selection(folder,output):
    source=Path(folder)/'selection';output=Path(output)
    for slot in SLOTS:
        for name in (slot,slot+'.json'):
            p=source/name;target=output/name
            if p.exists():
                if target.exists():raise FileExistsError(target)
                if p.is_dir():shutil.copytree(p,target)
                else:shutil.copyfile(p,target)
