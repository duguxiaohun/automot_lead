"""Fresh training is mandatory; cpu/train are resumable only at the CPU boundary."""
import argparse
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from qwen3vl_local import runtime_version as runtime
from qwen3vl_local.sft_new_loop_phase1 import data_stage, local_assets, run_integrity as proof
from . import CONTRACT, REVIEW_PATH
from .budget import PROFILES, profile

APP=Path(__file__).resolve().parents[3]
MODES=('2rgb_endpoints','4rgb')


def environment(app,phase,root,index,data_root,model,collection,budget_profile='bounded'):
    """Ignore stale shell knobs; only the explicitly selected budget applies."""
    budget=profile(phase,budget_profile)
    env=dict(os.environ)
    preserve={k:env[k] for k in ('GPU_IDS','ACTION_INDEX') if k in env}
    # Native script options must not silently turn fresh train into eval/plan-only,
    # change sampling/budget, or redirect one stage into another output directory.
    for stage in (1,2):
        for script in (Path(app)/f'qwen3vl_local/sft_new_loop_phase{stage}').glob('*.sh'):
            for key in re.findall(r'\$\{([A-Z][A-Z0-9_]*)',script.read_text()):
                if not key.startswith(('NCCL_','TORCH_')):env.pop(key,None)
    for key in ('WORLD_SIZE','RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','ADAPTER_DIR','RESUME','RESUME_FROM_CHECKPOINT'):
        env.pop(key,None)
    env.update(preserve)
    if env.get('GPU_IDS'):
        ids=env['GPU_IDS'].split(',')
        if len(ids)!=4 or len(set(ids))!=4 or any(not x.isdigit() for x in ids):
            raise ValueError('This release requires four distinct physical GPUs via GPU_IDS, or automatic four-GPU selection')
    env.update(PYTHONPATH=str(app),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        MODEL_DIR=str(model),DATA_ROOT=str(data_root),COLLECTION_DIR=str(collection),INDEX=str(index),
        DATA_OUTPUT_DIR=str(Path(index).parent),HISTORY_RGB_MODES=' '.join(MODES),
        DDP_GPU_COUNT='4',NPROC_PER_NODE='4',TRAIN_MODE='ddp',GRAD_ACCUM='1',
        RUN_BUILD='0',RUN_TRAIN='1',PLAN_ONLY='0',RUN_EVAL_SH='1',RUN_BASE_EVAL='0',RUN_LORA_EVAL='0',
        RUN_AUDIT_PROMPT_EVAL='1',RUN_BALANCED_EVAL='1',RUN_VISUAL_AUDIT='0',
        TRAIN_NUM_EPOCHS='3',NUM_EPOCHS='3',TRAIN_MAX_STEPS='0',MAX_STEPS='0',
        TRAIN_FOCUS_BALANCE_COUNT='768' if phase==1 else '1024',
        FOCUS_BALANCE_COUNT='768' if phase==1 else '1024',
        MAX_TRAIN_FRAME_REPEAT='10',TRAIN_MAX_FRAME_REPEAT='10',
        EPOCH_STEP_LIMIT='4096',TOTAL_STEP_LIMIT='12288',WARMUP_STEPS='-1',
        PIPELINE_TIMESTAMP=root.name,PIPELINE_ROOT=str(root/'pipeline'),
        FINAL_EVAL_SPLIT='test',EVAL_SPLIT='val')
    env.update(TRAIN_FOCUS_BALANCE_COUNT=str(budget['quota']),FOCUS_BALANCE_COUNT=str(budget['quota']),
               EPOCH_STEP_LIMIT=str(budget['epoch_step_limit']),TOTAL_STEP_LIMIT=str(budget['total_step_limit']),
               EPOCH_HOURS_LIMIT=str(budget['epoch_hours_limit']))
    return env


def verify_ready(root,phase,version):
    ready=proof.read(root/'ready.json')
    if ready['phase']!=phase or ready['version']!=str(version) or ready['contract']!=CONTRACT:
        raise ValueError('CPU ready belongs to another phase/version')
    runtime.verify(version)
    for path,sha in ready['files'].items():
        if proof.sha(path)!=sha:raise ValueError('CPU evidence changed: '+path)
    if proof.data_identity(ready['index'])!=ready['data']:raise ValueError('Prepared data changed')
    return ready


def worker(args):
    root=args.run_dir.resolve();version=args.version.resolve();phase=args.phase
    record=runtime.verify(version)
    revision=getattr(args,'revision','legacy_20261027')
    if record.get('training_revision',{}).get('revision','legacy_20261027')!=revision:
        raise ValueError('Requested revision differs from the frozen source')
    if record.get('training_revision',{}).get('phase')!=phase or (version/'app').resolve()!=APP:
        raise ValueError('Worker must run inside its phase-specific verified source snapshot')
    state=dict(contract=CONTRACT,phase=phase,status='starting',version=str(version))
    statepath=root/'status.json'
    parent=args.parent_data.resolve()/'frame_index.jsonl'
    index=root/'data/frame_index.jsonl'
    budget=profile(phase,getattr(args,'budget_profile','bounded'))
    env=environment(APP,phase,root,index,args.data_root,args.model_dir,args.collection_dir,budget['name'])
    os.chdir(APP)
    def stage(name,cmd,stage_env=env,timeout=None):
        state.update(status='running',stage=name);proof.write(statepath,state)
        return data_stage.run(cmd,root/(name+'.json'),timeout=timeout or args.cpu_timeout,env=stage_env)
    try:
        if args.command in ('all','train'):
            stage('model_check',[sys.executable,'-B','-m','qwen3vl_local.qwen35.preflight','--model-dir',str(args.model_dir)])
        if args.command in ('all','cpu'):
            action,_=local_assets.ensure_data(parent,phase,'auto')
            if action=='build':
                # Explicit CPU stage, heartbeat + timeout. Native builder only,
                # preserving existing labels/split rules before reviewed derivation.
                stage('parent_build',[sys.executable,'-B','-m',f'qwen3vl_local.sft_new_loop_phase{phase}.build_dataset',
                    '--output-dir',str(parent.parent),'--data-root',str(args.data_root),
                    '--collection-dir',str(args.collection_dir)])
            if phase==1:
                initial_data=root/'parent_revision/data' if revision=='visual_20261029' else index.parent
                stage('train_label_derivation',[sys.executable,'-B','-m',
                    'qwen3vl_local.audit_joint.phase12_retrain.launch','derive',
                    '--parent',str(parent),'--reviews',str(APP/REVIEW_PATH),
                    '--data-root',str(args.data_root),'--output',str(initial_data)])
                if revision=='visual_20261029':
                    stage('visual_label_derivation',[sys.executable,'-B','-m',
                        'qwen3vl_local.audit_joint.phase12_retrain.refinement',
                        '--parent',str(initial_data/'frame_index.jsonl'),
                        '--data-root',str(args.data_root),'--output',str(index.parent)])
            else:
                stage('train_label_derivation',[sys.executable,'-B','-m',
                    'qwen3vl_local.audit_joint.phase12_retrain.phase2_data',
                    '--parent',str(parent),'--data-root',str(args.data_root),'--output',str(index.parent)])
            prep_env={**env,'DATA_AUDIT_DIR':str(root/'cpu'),'RUN_BUILD':'0',
                      'DATA_STAGE_TIMEOUT_SECONDS':str(args.cpu_timeout)}
            stage('cpu_admission',['bash',f'qwen3vl_local/sft_new_loop_phase{phase}/prepare_data.sh'],
                  stage_env=prep_env,timeout=args.cpu_timeout*6)
            cpu=proof.read(root/'cpu/status.json')
            if cpu['status']!='ready':raise ValueError('Native CPU admission did not pass')
            files={str(p):proof.sha(p) for p in (root/'cpu').rglob('*') if p.is_file()}
            files.update({str(p):proof.sha(p) for p in [root/'selected_train_reviews.json',index.parent/'training_revision.json',index.parent/'preservation.json']})
            ready=dict(contract=CONTRACT,phase=phase,version=str(version),index=str(index),
                       data=proof.data_identity(index),files=files,mode_order=MODES,
                       expected_updates_per_mode=budget['expected_updates'],budget=budget,revision=revision)
            proof.write(root/'ready.json',ready)
        ready=verify_ready(root,phase,version)
        if ready.get('budget',profile(phase))!=budget:
            raise ValueError('CPU-ready budget differs from requested training budget')
        if args.command=='cpu':
            state.update(status='cpu_ready_model_training_not_started');proof.write(statepath,state)
            print('[CPU_READY] '+str(root),flush=True);return
        if ready['index']!=str(index):raise ValueError('Training dataset differs from prepared dataset')
        # A completed/failed GPU stage is never auto-restarted. Native output
        # reservation also refuses old weights, even in a different outer run.
        stage('fresh_train_eval',['bash',f'qwen3vl_local/sft_new_loop_phase{phase}/run_full_pipeline.sh'],
              timeout=args.train_timeout)
        verify_ready(root,phase,version)
        bundles=sorted((root/'pipeline').rglob('*_audit_bundle.tar.gz'))
        if len(bundles)<2:raise ValueError('Missing two-mode audit bundles after training')
        state.update(status='trained_evaluated_packaged_quality_pending',
                     bundles={str(p):proof.sha(p) for p in bundles},
                     note='Training completed; model quality requires audit of new weights against baseline.')
        proof.write(root/'completion_pending.json',state)
        # Small companion contains training revision provenance; raw cases remain
        # in the native per-mode bundles (each has its own size/integrity gate).
        files=[root/'completion_pending.json',root/'ready.json',version/'manifest.json']
        files += [root/'selected_train_reviews.json',root/'data/training_revision.json',root/'data/preservation.json']
        with tarfile.open(root/'training_revision_audit.tar.gz','w:gz') as tar:
            for p in files:tar.add(p,arcname='completion.json' if p.name=='completion_pending.json' else p.name,recursive=False)
        (root/'completion_pending.json').rename(root/'completion.json')
        for p in bundles:print('[BUNDLE] '+str(p),flush=True)
        print('[BUNDLE] '+str(root/'training_revision_audit.tar.gz'),flush=True)
        print(f'[DONE] Phase{phase}: fresh training, evaluation and packaging complete; quality pending audit',flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:proof.write(statepath,state)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('all','cpu','train'))
    parser.add_argument('phase',type=int,choices=(1,2))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--parent-data',type=Path)
    parser.add_argument('--data-root',type=Path,default=APP/'lead_data')
    parser.add_argument('--collection-dir',type=Path,default=APP/'keyframe_filter/collection_output')
    parser.add_argument('--model-dir',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    parser.add_argument('--cpu-timeout',type=float,default=7200)
    parser.add_argument('--train-timeout',type=float)
    parser.add_argument('--budget-profile',choices=PROFILES,default='bounded',
                        help='phase1_40k: three complete epochs, 40320 updates per mode; same labels/prompt/sampler')
    parser.add_argument('--revision',choices=('legacy_20261027','visual_20261029'),default='legacy_20261027')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--version',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    budget=profile(args.phase,args.budget_profile)
    if args.train_timeout is None:args.train_timeout=budget['train_timeout']
    if any(not math.isfinite(t) or t<=0 for t in (args.cpu_timeout,args.train_timeout)):
        parser.error('Timeouts must be finite and positive')
    if args.command=='train' and args.run_dir is None:parser.error('train requires the exact CPU-ready --run-dir')
    if args.run_dir is None:
        args.run_dir=APP/'checkpoints/phase12_retrain'/f'p{args.phase}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    args.run_dir=args.run_dir.resolve()
    args.parent_data=(args.parent_data or APP/f'checkpoints/sft_new_loop_phase{args.phase}_data').resolve()
    for name in ('data_root','collection_dir','model_dir'):setattr(args,name,getattr(args,name).resolve())
    if args.worker:return worker(args)
    if args.command!='train':args.run_dir.mkdir(parents=True,exist_ok=False)
    elif not args.run_dir.is_dir():raise ValueError('Missing CPU-ready run directory; no implicit recovery')
    with (args.run_dir/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='train':
            execution=proof.read(args.run_dir/'execution.json');version=Path(execution['version'])
            if execution['origin']!=str(APP) or execution['phase']!=args.phase:raise ValueError('Run belongs to another machine or phase')
            if execution.get('revision','legacy_20261027')!=args.revision:raise ValueError('CPU-ready revision differs from requested training revision')
            ready=verify_ready(args.run_dir,args.phase,version)
            if ready.get('budget',profile(args.phase))!=budget:
                raise ValueError('CPU-ready budget differs from requested training budget')
            if (args.run_dir/'fresh_train_eval.json').exists():raise ValueError('Training already attempted; preserve output, choose a new run')
        else:
            from .snapshot import prepare
            version=prepare(APP,args.phase,args.revision)
            proof.write(args.run_dir/'execution.json',dict(origin=str(APP),phase=args.phase,version=str(version),contract=CONTRACT,budget=budget,revision=args.revision))
        tree=version/'app';runtime.verify(version)
        cmd=[sys.executable,'-B','-m','qwen3vl_local.audit_joint.phase12_retrain',args.command,str(args.phase),
             '--worker','--version',str(version),'--run-dir',str(args.run_dir),'--parent-data',str(args.parent_data),
             '--data-root',str(args.data_root),'--collection-dir',str(args.collection_dir),
             '--model-dir',str(args.model_dir),'--cpu-timeout',str(args.cpu_timeout),'--train-timeout',str(args.train_timeout),
             '--budget-profile',args.budget_profile,'--revision',args.revision]
        print(f'[START] Phase{args.phase} {args.command}; run={args.run_dir}; version={version.name}',flush=True)
        subprocess.run(cmd,cwd=tree,env={**os.environ,'PYTHONPATH':str(tree),'PYTHONDONTWRITEBYTECODE':'1'},check=True)
        runtime.verify(version)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='derive':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--parent',required=True)
        p.add_argument('--reviews',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
        a=p.parse_args()
        from .data import derive
        print(derive(a.parent,a.reviews,a.data_root,a.output),flush=True)
    else:main()
