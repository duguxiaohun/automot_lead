"""Frozen CPU preparation -> fresh shared LoRA -> reload/test -> complete audit bundle."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from qwen3vl_local import runtime_version as runtime
from qwen3vl_local.sft_new_loop_phase1 import data_stage,local_assets,run_integrity as proof
from . import CONTRACT,MODES

APP=Path(__file__).resolve().parents[2]


def prepare_data(args):
    from qwen3vl_local.audit_joint.phase12_retrain import data as old_data,refinement,phase2_data,REVIEW_PATH
    from . import data,plan,reference,validation,history
    root=args.run_dir
    historical=history.load(args.history_contract)
    for phase,parent in ((1,args.phase1_data),(2,args.phase2_data)):
        index=parent/'frame_index.jsonl'
        action,_=local_assets.ensure_data(index,phase,'auto')
        if action=='build':
            subprocess.run([sys.executable,'-B','-m',f'qwen3vl_local.sft_new_loop_phase{phase}.build_dataset',
                '--output-dir',str(parent),'--data-root',str(args.data_root),'--collection-dir',str(args.collection_dir)],check=True)
    old_data.derive(args.phase1_data/'frame_index.jsonl',APP/REVIEW_PATH,args.data_root,root/'phase1_previous/data')
    refinement.derive(root/'phase1_previous/data/frame_index.jsonl',args.data_root,root/'phase1/data')
    phase2_data.derive(args.phase2_data/'frame_index.jsonl',args.data_root,root/'phase2/data')
    data.build(root/'phase1/data/frame_index.jsonl',root/'phase2/data/frame_index.jsonl',args.data_root,root/'data',
               historical_groups=history.groups(historical))
    plan.prepare(root/'data/frame_index.jsonl',args.data_root,root/'cpu',args.action_index)
    shutil.copyfile(args.history_contract,root/'cpu/historical_contract.json')
    reference.prepare(root,args.data_root)
    validation.prepare(root,args.data_root,historical_groups=history.groups(historical))
    files={str(p.relative_to(root)):proof.sha(p) for sub in ('data','cpu','phase1_previous','phase1','phase2')
           for p in (root/sub).rglob('*') if p.is_file()}
    proof.write(root/'ready.json',dict(contract=CONTRACT,version=str(args.version),data_root=str(args.data_root),
        action_index=args.action_index,files=files,status='cpu_ready_model_training_not_started'))
    print('[CPU_READY] '+str(root),flush=True)


def gpu_ids():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    available={}
    for line in raw.strip().splitlines():
        index,memory,util=[x.strip() for x in line.split(',')]
        available[index]=(float(memory),float(util))
    requested=os.environ.get('GPU_IDS','')
    ids=requested.split(',') if requested else [k for k,v in available.items() if v[0]<256 and v[1]<=5][:4]
    if len(ids)!=4 or len(set(ids))!=4 or any(x not in available for x in ids):raise ValueError('Four distinct idle physical GPUs required')
    if any(available[x][0]>=256 or available[x][1]>5 for x in ids):raise ValueError('Requested GPU is busy; no process or compute-mode changes allowed')
    return ids


def worker(args):
    from . import integrity
    root=args.run_dir;state=dict(contract=CONTRACT,status='starting');runtime.verify(args.version)
    if (args.version/'app').resolve()!=APP:raise ValueError('Worker must execute frozen source')
    env={**os.environ,'PYTHONPATH':str(APP),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(name,None)
    def stage(name,cmd,timeout,extra=None):
        state.update(status='running',stage=name);proof.write(root/'status.json',state)
        return data_stage.run(cmd,root/(name+'.json'),timeout=timeout,env={**env,**(extra or {})})
    try:
        if args.command in ('all','train'):
            stage('model_check',[sys.executable,'-B','-m','qwen3vl_local.qwen35.preflight','--model-dir',str(args.model_dir)],7200)
        if args.command in ('all','cpu'):
            command=[sys.executable,'-B','-m','qwen3vl_local.sft_new_loop_phase12','prepare',
                '--run-dir',str(root),'--version',str(args.version),'--phase1-data',str(args.phase1_data),
                '--phase2-data',str(args.phase2_data),'--data-root',str(args.data_root),
                '--collection-dir',str(args.collection_dir),'--action-index',args.action_index,
                '--history-contract',str(args.history_contract)]
            stage('cpu_prepare',command,14400)
        integrity.verify_ready(root,execution=True)
        guard=proof.read(root/'cpu/guard_coverage.json')
        print(f"[COVERAGE] weak validation={guard['selected']} cases; adequate={guard['coverage_adequate']}; "
              "training sampler unchanged; highway validation NOT certified; historical quality pending",flush=True)
        if args.command=='cpu':
            state.update(status='cpu_ready_model_training_not_started');return
        ids=gpu_ids()
        for mode in MODES:
            stage(mode+'_train',[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=4','--max_restarts=0','--log-dir',str(root/'torchrun'/mode),'--tee','3',
                '-m','qwen3vl_local.sft_new_loop_phase12.train',
                '--run-dir',str(root),'--model-dir',str(args.model_dir),'--mode',mode],432000,
                {'CUDA_VISIBLE_DEVICES':','.join(ids),'GPU_IDS':','.join(ids)})
            integrity.verify_training(root/'runs'/mode,root,mode)
            stage(mode+'_test',[sys.executable,'-B','-m','qwen3vl_local.sft_new_loop_phase12.finish',
                '--run-dir',str(root),'--model-dir',str(args.model_dir),'--mode',mode],86400,
                {'CUDA_VISIBLE_DEVICES':ids[0],'GPU_IDS':ids[0]})
        integrity.verify_ready(root);runtime.verify(args.version)
        bundles=list(root.glob('*_audit_bundle.tar.gz'))
        if len(bundles)!=2:raise ValueError('Missing fused two-mode audit bundles')
        state.update(status='trained_evaluated_packaged_quality_pending',bundles={p.name:proof.sha(p) for p in bundles})
        for p in bundles:print('[BUNDLE] '+str(p),flush=True)
        print('[DONE] Shared Phase1+2 LoRA: training/evaluation/package finished; quality pending audit',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:proof.write(root/'status.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('all','cpu','train','prepare'))
    p.add_argument('--run-dir',type=Path);p.add_argument('--version',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--phase1-data',type=Path,default=APP/'checkpoints/sft_new_loop_phase1_data')
    p.add_argument('--phase2-data',type=Path,default=APP/'checkpoints/sft_new_loop_phase2_data')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--collection-dir',type=Path,default=APP/'keyframe_filter/collection_output')
    p.add_argument('--model-dir',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--action-index',default='auto')
    p.add_argument('--history-contract',type=Path,default=APP/'checkpoints/phase12_history_20261030/contract.json')
    args=p.parse_args()
    for name in ('phase1_data','phase2_data','data_root','collection_dir','model_dir','history_contract'):
        setattr(args,name,getattr(args,name).resolve())
    if args.run_dir is None:
        if args.command in ('train','prepare'):p.error('Exact --run-dir required')
        args.run_dir=APP/'checkpoints/phase12_shared'/f'run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    args.run_dir=args.run_dir.resolve()
    if args.command=='prepare':
        if not args.version or (args.version/'app').resolve()!=APP:raise ValueError('Prepare requires frozen worker')
        return prepare_data(args)
    if args.worker:return worker(args)
    if args.command in ('all','cpu'):
        from . import history
        from .history_prepare import ensure
        default_history=(APP/'checkpoints/phase12_history_20261030/contract.json').resolve()
        if args.history_contract==default_history:ensure(args.history_contract,APP)
        else:history.load(args.history_contract)
    if args.command in ('all','cpu'):args.run_dir.mkdir(parents=True,exist_ok=False)
    if not args.run_dir.is_dir():raise ValueError('Missing CPU-ready directory')
    with (args.run_dir/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':
            execution=proof.read(args.run_dir/'execution.json');version=Path(execution['version'])
            if execution['origin']!=str(APP):raise ValueError('CPU-ready run belongs to another machine')
            if (args.run_dir/'runs').exists() or list(args.run_dir.glob('*_train.json')):
                raise ValueError('Training already attempted; no implicit restart or overwrite')
        else:
            version=runtime.prepare(APP,APP/'checkpoints/phase12_shared_runtime')
            proof.write(args.run_dir/'execution.json',dict(origin=str(APP),version=str(version),contract=CONTRACT))
        cmd=[sys.executable,'-B','-m','qwen3vl_local.sft_new_loop_phase12',args.command,'--worker',
            '--version',str(version),'--run-dir',str(args.run_dir),'--phase1-data',str(args.phase1_data),
            '--phase2-data',str(args.phase2_data),'--data-root',str(args.data_root),'--collection-dir',str(args.collection_dir),
            '--model-dir',str(args.model_dir),'--action-index',args.action_index,'--history-contract',str(args.history_contract)]
        subprocess.run(cmd,cwd=version/'app',env={**os.environ,'PYTHONPATH':str(version/'app')},check=True)
        runtime.verify(version)

if __name__=='__main__':main()
