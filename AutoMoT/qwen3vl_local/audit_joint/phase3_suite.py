"""One shared dataset, three independent three-epoch four-RGB choice experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import sys
from types import SimpleNamespace
import zipfile

from . import phase3_base_trial as trial, phase3_suite_data as data

POLICY='phase3_base_a_b_same_qwen_three_epochs_v1'
ARMS={'base':'base','a_future':'future','b_windows_bank':'windows_bank'}


def environment(ids):
    env=dict(os.environ,PYTHONPATH=str(trial.APP),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=ids,GPU_IDS=ids,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TORCH_NCCL_TRACE_BUFFER_SIZE='200000',TORCH_NCCL_DUMP_ON_TIMEOUT='1')
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
    return env


def model_snapshot(root):
    """Bind local weights/tokenizer and recheck their bytes before every stage."""
    result={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '.cache' in p.relative_to(root).parts:continue
        sha=trial.file_sha(p);s=p.stat()
        result[str(p.relative_to(root))]=dict(sha256=sha,
            stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
    if not result:raise ValueError('Empty model directory')
    return result


def check_model_snapshot(root,snapshot):
    current={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts}
    if current!=set(snapshot):raise ValueError('Initial model file membership changed')
    for name,expected in snapshot.items():
        s=(root/name).stat()
        if ([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=expected['stat']
                or trial.file_sha(root/name)!=expected['sha256']):
            raise ValueError('Initial model changed during suite: '+name)


def jobs(args,prepared,output,contract):
    result={}
    for arm,mode in ARMS.items():
        cfg=SimpleNamespace(index=prepared/'base/frame_index.jsonl',data_root=args.data_root,
            model_dir=args.model_dir,resume=None,stage='complete',experiment=mode,
            dense_initialization='qwen',init_adapter=None,
            aux_index=prepared/'auxiliary/frame_index.jsonl' if mode!='base' else None,
            supervision='all',forecast_span_seconds=args.forecast_span_seconds,
            forecast_step_seconds=args.forecast_step_seconds,min_future_coverage=.8,
            min_eval_coverage=.8,max_batch_tokens=32768)
        job=trial.build_job(cfg,output/arm)
        job['suite_binding']=dict(policy=POLICY,arm=arm,contract_sha256=trial.digest(contract),
            initialization='same_qwen_no_trained_adapter',data_receipt_sha256=trial.file_sha(prepared/'shared_data_receipt.json'))
        result[arm]=job
    return result


def command(job,operation):
    worker='qwen3vl_local.audit_joint.'+('phase3_base_trial_worker' if job['mode']=='base' else 'phase3_base_dense.worker')
    cmd=[sys.executable]
    if operation in ('train','probe','test'):
        cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=4','--max_restarts=0','--module',worker]
    else:cmd+=['-m',worker]
    return cmd+['--job',str(Path(job['output'])/'job.json'),'--operation',operation]


def execute(all_jobs,run,states,*,prepare_only=False,on_update=lambda:None):
    """All CPU and GPU admissions precede any training; first failure stops suite."""
    schedule=[(a,'preflight') for a in ARMS]
    if not prepare_only:
        schedule += [(a,'probe') for a in ARMS if a!='base']
        schedule += [(a,op) for a in ARMS for op in ('train','summarize','test')]
    for arm,operation in schedule:
        job=all_jobs[arm];state=states[arm]
        state.update(status='running',active_stage=operation);on_update()
        try:
            code=run(arm,operation,command(job,operation))
            if not code and operation=='train':state['training']=trial.verify_training(job)
        except BaseException as exc:
            state.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}')
            on_update();raise
        if code:
            state.update(status='failed_or_interrupted',exit_code=code)
            on_update();return code
        statuses=dict(preflight='prepared_cpu_no_training' if prepare_only else 'cpu_admitted',
            probe='gpu_admitted',train='trained_pending_evaluation',summarize='validated_pending_test',
            test='planned_training_completed_quality_pending')
        state.update(status=statuses[operation],active_stage=None)
        on_update()
    return 0


def feedback(output,states,state):
    """Three independently checksummed audit ZIPs plus a small suite summary."""
    dest=output/'handoff';dest.mkdir(exist_ok=True)
    for arm in ARMS:
        folder=output/arm;folder.mkdir(exist_ok=True)
        trial.package(folder,states[arm])
        for suffix in ('report.zip','package.json'):
            shutil.copyfile(folder/'handoff'/('phase3_base_'+suffix),dest/(arm+'_'+suffix))
    names=['suite_workflow.json','suite_contract.json','comparison.json','data_build.log']
    files={}
    for n in names:
        p=output/n
        if p.is_file():
            if p.stat().st_size>4*1024**2:
                if p.suffix!='.log':raise ValueError('Oversized suite report')
                with p.open('rb') as f:f.seek(max(0,p.stat().st_size-60000));files[n]=f.read()
            else:files[n]=p.read_bytes()
    trial.write(output/'suite_workflow.json',state)
    files['suite_workflow.json']=(output/'suite_workflow.json').read_bytes()
    with zipfile.ZipFile(dest/'suite_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
    trial.write(dest/'suite_package.json',dict(status=state['status'],
        archive_sha256=trial.file_sha(dest/'suite_report.zip'),files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}))


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=trial.APP/'lead_data')
    p.add_argument('--model-dir',type=Path,default=trial.APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--collection-dir',type=Path,default=trial.APP/'keyframe_filter/collection_output')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared-data',type=Path,help='Reuse an explicit prior suite shared_data directory; never regenerate')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--gpu-ids',default=os.environ.get('GPU_IDS',''))
    p.add_argument('--forecast-span-seconds',type=float,default=6.)
    p.add_argument('--forecast-step-seconds',default='auto')
    p.add_argument('--timeout-seconds',type=int,default=172800)
    p.add_argument('--reserve-gib',type=int,default=20)
    return p


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,resource.getrlimit(resource.RLIMIT_CORE)[1]))
    args=parser().parse_args();output=args.output.resolve()
    if args.output.exists() or args.output.is_symlink():raise ValueError('Use a new output directory')
    if args.timeout_seconds<=0 or args.reserve_gib<20:raise ValueError('Positive timeout and >=20 GiB reserve required')
    for key in ('index','data_root','model_dir'):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    if not args.prepared_data:args.collection_dir=args.collection_dir.resolve(strict=True)
    for protected in (args.index.parent,args.data_root,args.model_dir,args.collection_dir):
        if output.is_relative_to(protected.resolve()):raise ValueError('Output overlaps input assets')
    if args.prepared_data and output.is_relative_to(args.prepared_data.resolve(strict=True)):
        raise ValueError('Output overlaps shared data')
    if trial.file_sha(args.index)!=trial.reference()['index_sha256']:raise ValueError('Frozen historical index required')
    from .phase3_base_dense.planning import options
    offsets=options(vars(args))['offset_frames']
    release=trial.verified_sources()
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    check_model(args.model_dir)
    ids='' if args.prepare_only else trial.gpu_ids(args.gpu_ids)
    output.mkdir(parents=True)
    state=dict(policy=POLICY,status='preparing',hostname=socket.gethostname(),
        server_label=os.environ.get('AUTOMOT_SERVER','unspecified'),output=str(output),stages=[])
    states={a:dict(policy=POLICY,status='not_started',arm=a,mode=m,stages=[]) for a,m in ARMS.items()}
    code=0
    def update():
        trial.write(output/'suite_workflow.json',state)
        for a,s in states.items():trial.write(output/a/'workflow.json',s)
    try:
        if shutil.disk_usage(output).free<(args.reserve_gib+1)*1024**3:raise ValueError('Insufficient free space')
        snapshot=model_snapshot(args.model_dir)
        trial.write(output/'model_snapshot.json',snapshot)
        prepared=args.prepared_data.resolve(strict=True) if args.prepared_data else output/'shared_data'
        if not args.prepared_data:
            data_job=dict(index=str(args.index),data_root=str(args.data_root),collection_dir=str(args.collection_dir),
                output=str(prepared),offset_frames=offsets,release_sha256=release)
            trial.write(output/'data_job.json',data_job)
            result=trial.run_stage([sys.executable,'-m','qwen3vl_local.audit_joint.phase3_suite_data',
                '--job',str(output/'data_job.json')],output/'data_build.log',environment(''),
                timeout=args.timeout_seconds,reserve_bytes=args.reserve_gib*1024**3)
            state['stages'].append(dict(stage='shared_data',**result))
            if result['exit_code']:code=result['exit_code'];raise RuntimeError('Shared data generation failed')
        receipt=data.verify(prepared,data_root=args.data_root,offsets=offsets,release=release)
        contract=dict(policy=POLICY,profile=trial.profile(),arms=ARMS,initialization='same_qwen_no_trained_adapter',
            model_dir=str(args.model_dir),model_files_sha256={k:v['sha256'] for k,v in snapshot.items()},
            reference_sha256=trial.file_sha(trial.REFERENCE),release_sha256=release,
            shared_data=str(prepared),shared_data_receipt_sha256=trial.file_sha(prepared/'shared_data_receipt.json'),
            forecast_offset_frames=offsets,supervision='all',
            historical_best=dict(mode='4rgb_choice',val_correct=242,val_total=320,test_correct=274,test_total=380),
            interpretation='three methods, not an isolated density ablation; prompts/encoding/architecture differ; no test selection')
        trial.write(output/'suite_contract.json',contract)
        all_jobs=jobs(args,prepared,output,contract)
        for arm,job in all_jobs.items():
            folder=Path(job['output']);folder.mkdir(exist_ok=True)
            trial.write(folder/'job.json',job);trial.write(folder/'suite_contract.json',contract)
            trial.write(folder/'shared_data_receipt.json',receipt)
        def run(arm,op,cmd):
            check_model_snapshot(args.model_dir,snapshot)
            data.verify(prepared,data_root=args.data_root,offsets=offsets,release=release)
            state.update(status='running',active_arm=arm,active_stage=op);update()
            env=environment(ids if op in ('train','probe','test','summarize') else '')
            env['TORCH_NCCL_DEBUG_INFO_TEMP_FILE']=str(output/arm/'nccl_rank_')
            result=trial.run_stage(cmd,output/arm/(op+'.log'),env,timeout=args.timeout_seconds,
                reserve_bytes=args.reserve_gib*1024**3)
            states[arm]['stages'].append(dict(stage=op,**result))
            return result['exit_code']
        code=execute(all_jobs,run,states,prepare_only=args.prepare_only,on_update=update)
        if code:raise RuntimeError('Suite stopped on failed stage; completed runs preserved')
        if not args.prepare_only:
            from .phase3_suite_compare import compare
            trial.write(output/'comparison.json',compare(all_jobs))
        state.update(status='prepared_cpu_no_training' if args.prepare_only else 'three_methods_completed_quality_pending',
            active_arm=None,active_stage=None)
    except BaseException as exc:
        code=code or 2;state.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}')
        print(state['error'],file=sys.stderr,flush=True)
    finally:
        update()
        try:feedback(output,states,state)
        except Exception as exc:
            print('Suite packaging failed: '+str(exc),file=sys.stderr);code=code or 2
    print(f'[suite] exit={code} feedback={output / "handoff"}',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
