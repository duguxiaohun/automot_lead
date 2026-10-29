"""Fresh-only execution receipts; no mutable latest pointers or implicit resume."""
import json
from pathlib import Path
from qwen3vl_local import runtime_version as runtime
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from . import CONTRACT, UPDATES, WORLD
from .data import resolve_rgb


def verify_ready(root,*,images=True,execution=False):
    root=Path(root);ready=proof.read(root/'ready.json')
    if ready['contract']!=CONTRACT:raise ValueError('Wrong fusion contract')
    if execution and Path(__file__).resolve().parents[2]!=(Path(ready['version'])/'app').resolve():
        raise ValueError('Training/evaluation must use the exact CPU-admitted frozen source')
    runtime.verify(ready['version'])
    for name,expected in ready['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or proof.sha(root/p)!=expected:raise ValueError('Prepared evidence changed: '+name)
    if images:
        rgb=proof.read(root/'data/rgb_sha256.json')
        for filename in ('reference_rgb_sha256.json','guard_rgb_sha256.json'):
            for name,expected in proof.read(root/'cpu'/filename).items():
                if name in rgb and rgb[name]!=expected:raise ValueError('Inconsistent image bindings')
                rgb[name]=expected
        for name,expected in rgb.items():
            if proof.sha(resolve_rgb(ready['data_root'],name))!=expected:raise ValueError('RGB changed: '+name)
    return ready


def budget():
    return dict(world_size=WORLD,total_steps=UPDATES,optimizer_updates=UPDATES,grad_accum=1,
                samples_per_epoch_global=7168,steps_per_epoch=1792,warmup_updates=2000,epochs=6)


def finish_training(output,ready,mode,selected):
    output=Path(output);exposure=proof.validate_consumption(output,budget())
    for rank in range(WORLD):
        r=proof.read(output/'exposure'/f'branches_{rank}.json')
        if r!=dict(phase1=UPDATES,phase2=UPDATES,optimizer_updates=UPDATES):raise ValueError('Incomplete task supervision')
    adapters={name:proof.verify_adapter(output/name) for name in ('final',selected)}
    paths=[p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl') and 'adapter' not in p.name
           and not any(part.startswith('best_step_') or part=='final' for part in p.relative_to(output).parts)]
    result=dict(contract=CONTRACT,status='trained_quality_pending',mode=mode,budget=budget(),
        ready_sha256=proof.sha(Path(ready)/'ready.json'),selected=selected,adapters=adapters,exposure=exposure,
        files={str(p.relative_to(output)):proof.sha(p) for p in paths})
    proof.write(output/'training_complete.json',result);return result


def verify_training(output,root,mode):
    output=Path(output);r=proof.read(output/'training_complete.json')
    if r['contract']!=CONTRACT or r['status']!='trained_quality_pending' or r['mode']!=mode or r['budget']!=budget():
        raise ValueError('Wrong fusion training completion')
    if r['ready_sha256']!=proof.sha(Path(root)/'ready.json'):raise ValueError('Training/ready mismatch')
    selection=proof.read(output/'selection.json')
    if r['selected']!=selection['adapter'] or r['selected']!=f"best_step_{selection['step']}":
        raise ValueError('Completion adapter differs from guarded validation selection')
    config=proof.read(output/r['selected']/'fusion_config.json')
    if config['step']!=selection['step'] or config['mode']!=mode:raise ValueError('Wrong selected checkpoint metadata')
    for name,expected in r['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or proof.sha(output/name)!=expected:
            raise ValueError('Training evidence changed: '+name)
    if proof.validate_consumption(output,budget())!=r['exposure']:raise ValueError('Training consumption mismatch')
    for name,expected in r['adapters'].items():
        if Path(name).name!=name or proof.verify_adapter(output/name)!=expected:raise ValueError('Adapter changed')
    return r
