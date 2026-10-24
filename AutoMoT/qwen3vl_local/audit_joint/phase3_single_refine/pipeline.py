"""CPU -> GPU probe -> train -> test -> verify -> <=30 MB delivery.

Completed stages are revalidated and reused. Failed training is never implicitly
resumed; a failed evaluation may be retried without repeating successful training.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .audit import APP,sha
from .state import atomic,lock
from . import preflight


def read(path):return json.loads(Path(path).read_text())

def selected_gpus(count=4):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    available={str(int(i)):int(free) for i,free in (line.split(',') for line in text.splitlines())}
    chosen=os.environ.get('GPU_IDS')
    ids=chosen.split(',') if chosen else sorted(available,key=lambda i:(-available[i],int(i)))[:count]
    if count==1:ids=ids[:1]
    if len(ids)!=count or len(set(ids))!=count or any(i not in available for i in ids):raise ValueError(f'{count}_distinct_available_GPUs_required')
    if min(available[i] for i in ids)<10000:raise ValueError('selected_GPU_free_memory_below_10000_MiB')
    return ids


def execute(command,log,env=None):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    print('LOG='+str(log.resolve()),flush=True)
    # Stream to terminal and file; no background daemon or hidden nohup child.
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=APP,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            result=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    if result:raise RuntimeError(f'child_exit_{result}: {log}')


def worker(operation,root,ready,model,ids,output,*,checkpoint=None):
    env=dict(os.environ,PYTHONPATH=str(APP),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=','.join(ids))
    # Do not inherit a previous torchrun/elastic job from the invoking shell.
    for key in tuple(env):
        if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE',
                   'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):
            env.pop(key,None)
    module='qwen3vl_local.audit_joint.phase3_single_refine.training'
    cmd=[sys.executable,'-B']
    if operation in ('probe','train'):
        cmd+=['-m','torch.distributed.run','--standalone','--max_restarts=0','--nproc_per_node=4','--module',module]
    else:cmd+=['-m',module]
    cmd +=[operation,'--dataset',ready['dataset'],'--model-dir',str(model),'--output',str(output)]
    if checkpoint:cmd+=['--checkpoint',str(checkpoint),'--split','test']
    execute(cmd,root/'logs'/(operation+'.log'),env)


def model_identity(model):
    from ..phase3_suite import model_snapshot
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    check_model(model)
    return {p:v['sha256'] for p,v in model_snapshot(model).items()}


def check_probe(root,ready,model_hashes):
    probe=root/'probe';r=read(probe/'probe_complete.json');b=r['binding']
    from .training.dataset import code_identity
    if (b['dataset_sha256']!=ready['dataset_sha256'] or b['contract']!=ready['contract'] or b['code']!=code_identity()
        or b['model']!=model_hashes or b['shared_plan_sha256']!=ready['plan_sha256']
        or r['result']['status']!='epoch_complete' or r['formal_training_steps']!=0
        or sha(probe/'probe_checkpoint/state.pt')!=r['checkpoint_sha256']
        or sha(probe/'input_scan.json')!=r['input_scan_sha256']):raise ValueError('probe_binding_mismatch')
    return r


def training_complete(root,ready):
    """Before continuing test, check all checkpoint/exposure receipts, not just a flag."""
    from collections import Counter
    from .training.dataset import code_identity
    run=root/'run';r=read(run/'training_complete.json');b=r['binding']
    if (r.get('total_steps')!=9600 or r.get('selected_checkpoint')!='epoch_3' or r.get('selection')!='final_completed_epoch'
        or b['dataset_sha256']!=ready['dataset_sha256'] or b['contract']!=ready['contract'] or b['code']!=code_identity()
        or b['shared_plan_sha256']!=ready['plan_sha256'] or b['admission_sha256']!=ready['admission_sha256']):raise ValueError('training_completion_binding_mismatch')
    if read(run/'plan.json')!=read(ready['plan']) or read(run/'admission.json')!=read(ready['admission']):raise ValueError('run_plan_or_admission_changed')
    exposure=Counter()
    for epoch,plan in enumerate(read(ready['plan'])['plans'],1):
        folder=run/f'epoch_{epoch}';receipt=read(folder/'receipt.json');actual=read(run/f'epoch_{epoch}.json')
        exposure.update(plan['ordered_ids'])
        if (sha(folder/'receipt.json')!=r['checkpoint_receipts'][folder.name] or receipt['binding']!=b
            or receipt['status']!='epoch_complete' or receipt['step']!=epoch*3200
            or sha(folder/'state.pt')!=receipt['state_sha256'] or actual['plan']!=plan
            or actual['result']['status']!='epoch_complete' or actual['result']['steps']!=3200
            or actual['result']['actual_exposure']!=dict(exposure)):raise ValueError('checkpoint_or_exposure_changed')
    return r


def status(root):
    result={}
    for name in ('state','ready_1280','delivery'):
        p=root/(name+'.json');result[name]=read(p) if p.exists() else None
    p=root/'run/progress.json';result['progress']=read(p) if p.exists() else None
    # Avoid flooding the terminal with every source/plan binding.
    if result['ready_1280']:result['ready_1280']={k:v for k,v in result['ready_1280'].items() if k in ('status','error','dataset','optimizer_steps')}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def run(operation,root,native_ready,model):
    if operation=='status':return status(root)
    with lock(root):
        stage='initializing'
        preparing=False
        def update(name,**kw):
            nonlocal stage
            stage=name;atomic(root/'state.json',dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
        try:
            if operation=='cpu' and any((root/name).exists() for name in ('run','probe')):
                raise ValueError('cpu_reprepare_refused_existing_run_or_probe; preserve its bound ready')
            if operation in ('all','check','probe','train') and not (root/'run/training_complete.json').is_file():
                if (root/'run').exists():raise ValueError('incomplete_training_preserved_no_implicit_resume')
                if (root/'probe').exists() and not (root/'probe/probe_complete.json').is_file():
                    raise ValueError('incomplete_probe_preserved_choose_new_root_or_review')
            if operation in ('cpu','all'):
                update('cpu')
                if not native_ready.is_file():
                    if native_ready.name!='ready_1280.json':raise ValueError('native_ready_must_be_ready_1280.json')
                    # Map parsing is required only when preparing the native pool.
                    import importlib
                    try:importlib.import_module('carla')
                    except ImportError as exc:raise RuntimeError('native preparation requires carla in '+sys.executable) from exc
                    execute([sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase3_single_v3.cpu',
                             '--study-root',str(native_ready.parent)],root/'logs/native_cpu.log')
                pointer=root/'ready_1280.json'
                existing=read(pointer) if pointer.is_file() else None
                started=any((root/name).exists() for name in ('run','probe'))
                reusable=existing and (existing.get('status') not in ('rejected','preparing') or started)
                if reusable and operation=='all':
                    ready=preflight.load_ready(root,check_images=True)
                else:
                    if started:raise ValueError('missing_ready_for_existing_run_or_probe')
                    if existing:
                        history=root/'cpu_attempts';history.mkdir(exist_ok=True)
                        shutil.copy2(pointer,history/(str(time.time_ns())+'_ready.json'))
                    preparing=True
                    atomic(root/'ready_1280.json',dict(status='preparing'))
                    ready=preflight.prepare(native_ready,root)
                    preparing=False
                if operation=='cpu':
                    atomic(root/'state.json',dict(status='cpu_complete',stage='ready',GPU_executed=False));return ready
            else:ready=preflight.load_ready(root,check_images=operation in ('check','probe','train'))
            if operation=='package':
                return finish(root,ready,update)
            if (operation in ('all','train','evaluate') and (root/'run/training_complete.json').is_file()
                    and (root/'run/test/cases.json').is_file() and (root/'run/test/metrics.json').is_file()
                    and (root/'run/validation_complete.json').is_file()):
                return finish(root,ready,update)
            update('preflight_model_and_gpu')
            model_hashes=model_identity(model)
            # Reserve space for three optimizer checkpoints, the probe and adapters.
            if shutil.disk_usage(root).free<12_000_000_000:raise ValueError('free_disk_below_12_GB')
            if operation in ('all','train','probe','check') and not (root/'run/training_complete.json').is_file():ids=selected_gpus()
            else:ids=[]
            if operation=='check':
                atomic(root/'state.json',dict(status='preflight_passed',stage='checked',gpu_ids=ids));return ready
            if operation in ('all','train','probe') and not (root/'run/training_complete.json').is_file():
                update('probe')
                if (root/'probe/probe_complete.json').is_file():check_probe(root,ready,model_hashes)
                else:
                    if (root/'probe').exists():raise ValueError('incomplete_probe_preserved_choose_new_root_or_review')
                    worker('probe',root,ready,model,ids,root/'probe');check_probe(root,ready,model_hashes)
                if operation=='probe':
                    atomic(root/'state.json',dict(status='probe_complete',stage='probe'));return ready
                update('training')
                if (root/'run').exists():raise ValueError('incomplete_training_preserved_no_implicit_resume')
                worker('train',root,ready,model,ids,root/'run')
            update('verify_training')
            completion=training_complete(root,ready)
            if completion['binding']['model']!=model_hashes:raise ValueError('model_changed_since_training')
            update('validation')
            if not (root/'run/validation_complete.json').is_file():
                if not ids:ids=selected_gpus(1)
                worker('validate',root,ready,model,ids[:1],root/'run')
            update('test')
            test=root/'run/test'
            if not (test/'cases.json').is_file() or not (test/'metrics.json').is_file():
                if test.exists():
                    # Preserve failed test evidence, retry only evaluation.
                    test.rename(test.with_name('test_failed_'+str(time.time_ns())))
                if not ids:ids=selected_gpus(1)
                worker('evaluate',root,ready,model,ids[:1],test,checkpoint=root/'run/epoch_3')
            return finish(root,ready,update)
        except BaseException as exc:
            atomic(root/'state.json',dict(status='failed',stage=stage,error=f'{type(exc).__name__}: {exc}',updated_unix=time.time()))
            if preparing:atomic(root/'ready_1280.json',dict(status='rejected',error=f'{type(exc).__name__}: {exc}'))
            raise


def finish(root,ready,update):
    from ..phase3_completion import verify_study
    from .delivery import package
    update('completion_verification')
    training_complete(root,ready)
    verification=verify_study(root,root/'run','phase3_single_refine.training','six')
    # Recompute full validation and exposed guard metrics too.
    from .training.validation import verify
    from .training import dataset
    from .experiment import guard_candidates,guard_records
    artifact=dataset.load(ready['dataset'],variant='six',check_sources=False,expected_sha256=ready['dataset_sha256'])
    members={r['id']:r for r in artifact['records']}
    validation=[members[k] for k in artifact['admission']['fixed_validation_ids']]
    buckets={}
    for r in artifact['records']:guard_candidates(buckets,r)
    guards=guard_records(buckets)
    for epoch in range(1,4):
        for prefix in ('validation','guard'):
            verify(root/'run'/f'{prefix}_{epoch}_results',validation if prefix=='validation' else guards,root/'run'/f'epoch_{epoch}')
    update('package')
    receipt=package(root,root/'run',verification)
    atomic(root/'state.json',dict(status='completed',stage='done',audit_package=receipt['package'],audit_bytes=receipt['bytes'],training_quality_approved=False))
    return receipt


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('operation',choices=('cpu','check','probe','train','evaluate','package','all','status'))
    p.add_argument('--root',type=Path,default=APP/'checkpoints/phase3_single_refine_train_v1')
    p.add_argument('--native-ready',type=Path,default=APP/'checkpoints/phase3_single_v3/ready_1280.json')
    p.add_argument('--model',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    args=p.parse_args();root=args.root.resolve();model=args.model.resolve()
    for protected in ((APP/'lead_data').resolve(),model,args.native_ready.resolve().parent):
        if root.is_relative_to(protected) or protected.is_relative_to(root):p.error('output_overlaps_inputs')
    run(args.operation,root,args.native_ready.resolve(),model)

if __name__=='__main__':main()
