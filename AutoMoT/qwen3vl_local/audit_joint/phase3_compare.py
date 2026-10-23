"""Canonical base/single/a/b/c entry. Frozen historical launchers stay separate."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import fcntl
from . import phase3_comparison_policy as policy
from .geometry_study_abc import selected_gpus

APP=Path(__file__).resolve().parents[2]
METHODS=('base','single','a','b','c')

def package(method):
    if method not in METHODS or method=='base': raise ValueError('method_has_no_four_query_package')
    return 'phase3_single_v3' if method=='single' else 'geometry_abc_v2'

def root_for(app,method):
    return app/'checkpoints'/('phase3_compare_base' if method=='base' else package(method))

def read(path): return json.loads(Path(path).read_text())

def commands(app,operation,method,*,maps=None,geometry=None,exposure_manifests=()):
    if method=='base':
        if operation not in ('cpu','train','probe'): raise ValueError('base_command_requires_cpu_train_or_probe')
        if operation=='cpu':
            return [[sys.executable,'-u','-m','qwen3vl_local.audit_joint.phase3_base_bounded','cpu']]
        command=['bash',str(app/'qwen3vl_local/audit_joint/phase3_base_bounded.sh'),
                 '--output',str(root_for(app,method))]
        return [command+(['--probe-only'] if operation=='probe' else [])]
    module=package(method)
    if operation=='cpu':
        cmd=[sys.executable,'-u','-m',f'qwen3vl_local.audit_joint.{module}.cpu']
        if maps is not None:cmd+=['--maps',str(maps)]
        if geometry is not None:cmd+=['--geometry',str(geometry)]
        if method=='single':
            for path in exposure_manifests:cmd+=['--exposure-manifest',str(path)]
        return [cmd]
    return [['bash',str(app/f'qwen3vl_local/audit_joint/{module}/remote.sh'),'train',
             'six' if method=='single' else method]]

def methods_for(value):
    return list(METHODS) if value=='all' else ['a','b','c'] if value=='abc' else [value]

def status(app,method):
    root=root_for(app,method)
    if method=='base':
        states={}
        pointer=app/'checkpoints/phase3_base_bounded_v3/ready_1280.json'
        states['data']=read(pointer) if pointer.is_file() else dict(status='not_prepared')
        for name,suffix in (('cpu','_plan'),('probe','_probe'),('training','')):
            p=root/f'run_base_bounded_v3{suffix}/launch_workflow.json'
            state=read(p) if p.is_file() else dict(status='not_started')
            states[name]={k:state[k] for k in ('status','active_stage','output') if k in state}
        return states
    p=root/'ready_1280.json';c=root/'cpu_status.json'
    ready=read(p) if p.is_file() else dict(status='not_prepared')
    run=root/'runs'/f'{"six" if method=="single" else method}_1280'
    return dict(ready={k:ready[k] for k in ('status','error','dataset','optimizer_steps') if k in ready},
        cpu=read(c) if c.is_file() else None,training_complete=(run/'training_complete.json').is_file(),
        test_complete=(run/'test/metrics.json').is_file())

def preflight(app,methods):
    # Reject every collision before reading large datasets or inspecting GPUs.
    for method in methods:
        root=root_for(app,method)
        run=(root/'run_base_bounded_v3' if method=='base' else
             root/'runs'/f'{"six" if method=="single" else method}_1280')
        if run.exists(): raise FileExistsError(f'Existing run preserved; no implicit resume: {run}')
    missing=[]
    for method in methods:
        root=(app/'checkpoints/phase3_base_bounded_v3' if method=='base' else root_for(app,method))
        pointer=root/'ready_1280.json'
        state=read(pointer) if pointer.is_file() else {}
        if state.get('status')=='rejected':
            missing.append(f'{method}: CPU admission rejected: {state.get("error", "see admission report")}; '
                f'inspect {state.get("admission", pointer)}; unchanged inputs will not resolve capacity deficits')
        elif state.get('status')!='ready':
            missing.append(f'{method}: {pointer}; run phase3_compare.sh cpu {method}')
    if missing:raise ValueError('Current CPU admission required before any GPU launch: '+'; '.join(missing))
    checked=set()
    for method in methods:
        if method=='base':
            subprocess.run([sys.executable,'-m','qwen3vl_local.audit_joint.phase3_base_bounded','check'],
                cwd=app,env=dict(os.environ,PYTHON=sys.executable),check=True)
            continue
        name=package(method)
        if name in checked:continue
        mod=importlib.import_module(f'qwen3vl_local.audit_joint.{name}.prepare')
        ready=mod.load_ready(root_for(app,method)/'ready_1280.json',per_event=1280,world_size=4)
        if ready['optimizer_steps']!=[3200]*3: raise ValueError('comparison_budget_mismatch')
        checked.add(name)
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    check_model(app/'checkpoints/Qwen3.5-4B')
    return selected_gpus(os.environ.get('GPU_IDS'))


def verify_completion(app,method):
    root=root_for(app,method)
    if method=='base':
        from . import phase3_native_runtime
        run=root/'run_base_bounded_v3'
        state=read(run/'launch_workflow.json')
        if state['status']!='completed_quality_pending': raise ValueError('base_workflow_incomplete')
        job_path=(run/'experiment/job.json').resolve(strict=True)
        if Path(read(job_path)['output']).resolve()!=job_path.parent:
            raise ValueError('base_completion_job_output_mismatch')
        # The main tree includes newer LeadMoT geometry sources. Re-enter the
        # same content-validated native runtime used by the launcher, without
        # modifying the admitted adapter or accepting a different source SHA.
        runtime=phase3_native_runtime.prepare(app/'checkpoints',app=app)
        code=('import json,sys; from pathlib import Path; '
              'from qwen3vl_local.audit_joint import phase3_base_bounded as base; '
              'job=json.loads(Path(sys.argv[1]).read_text()); '
              'print(json.dumps(base.verify_training(job)))')
        result=subprocess.run([sys.executable,'-B','-c',code,str(job_path)],cwd=runtime,
            env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1',
                     AUTOMOT_PHASE3_VERIFIED_RUNTIME='1'),
            stdout=subprocess.PIPE,text=True,check=True)
        verified=json.loads(result.stdout)
        if verified['optimizer_updates']!=9600 or verified['verified_rank_presentations']!=38400:
            raise ValueError('base_completion_budget_mismatch')
        return dict(policy=policy.contract(),method=method,optimizer_steps=9600,
            evidence=str(run/'launch_workflow.json'),verified_runtime=str(runtime),
            training_verification=verified)
    run=root/'runs'/f'{"six" if method=="single" else method}_1280'
    from .phase3_completion import verify_study
    verified=verify_study(root,run,package(method),'six' if method=='single' else method)
    return dict(policy=policy.contract(),method=method,optimizer_steps=9600,
        evidence=str(run/'training_complete.json'),training_verification=verified)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('operation',choices=('cpu','check','probe','train','verify','status','contract'))
    p.add_argument('method',choices=(*METHODS,'abc','all'),nargs='?',default='all')
    p.add_argument('--maps',type=Path,help='Complete flat XODR or CARLA Maps root; CPU only')
    p.add_argument('--geometry',type=Path,help='Explicit validated parent dataset; CPU only')
    p.add_argument('--exposure-manifest',type=Path,action='append',default=[],help='Additional known exposed groups for single v3')
    args=p.parse_args();methods=methods_for(args.method)
    if args.operation=='probe' and methods!=['base']:p.error('explicit probe currently applies to base only')
    if (args.maps or args.geometry or args.exposure_manifest) and args.operation!='cpu':p.error('data overrides require cpu')
    if args.exposure_manifest and 'single' not in methods:p.error('exposure manifest applies to single')
    if (args.maps or args.geometry) and methods==['base']:p.error('maps/geometry apply to single or abc')
    if args.operation=='contract': print(json.dumps(policy.contract(),ensure_ascii=False,indent=2));return 0
    if args.operation=='status':
        print(json.dumps({m:status(APP,m) for m in methods},ensure_ascii=False,indent=2));return 0
    if args.operation=='check':
        print(json.dumps(dict(status='preflight_passed',gpu_ids=preflight(APP,methods),
            policy=policy.contract(),base_capacity='bound CPU admission verified; training worker rechecks projections')));return 0
    if args.operation=='probe':
        probe=root_for(APP,'base')/'run_base_bounded_v3_probe'
        if probe.exists():raise FileExistsError(f'Existing probe preserved; no implicit overwrite: {probe}')
    lockroot=APP/'checkpoints/phase3_compare';lockroot.mkdir(parents=True,exist_ok=True)
    with (lockroot/'.operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,PYTHON=sys.executable)
        env.pop('STUDY_RUN_NAME',None);env['STUDY_PER_EVENT']='1280'
        if args.operation in ('train','probe'):env['GPU_IDS']=preflight(APP,methods)
        prepared=set();resolved_maps=None
        for method in methods:
            if args.operation=='verify':
                # Recovery of outer completion only: never prepare data, query
                # GPUs, resume training or overwrite any experiment artifacts.
                from .versioned_data import write
                write(lockroot/f'{method}_completed.json',verify_completion(APP,method))
                print(json.dumps(dict(method=method,status='completion_verified')),flush=True)
                continue
            key='base' if method=='base' else package(method)
            if args.operation=='cpu' and key in prepared:continue
            if args.operation=='cpu' and method!='base' and resolved_maps is None:
                from .phase3_maps import resolve
                resolved_maps,map_proof=resolve(APP,lockroot/'maps',args.maps)
                from .versioned_data import write
                write(lockroot/'map_resolution.json',map_proof)
            for cmd in commands(APP,args.operation,method,maps=resolved_maps,
                    geometry=args.geometry,exposure_manifests=args.exposure_manifest):
                print(json.dumps(dict(method=method,operation=args.operation,command=cmd)),flush=True)
                result=subprocess.run(cmd,cwd=APP,env=env)
                if result.returncode:return result.returncode
            if args.operation=='train':
                from .versioned_data import write
                write(lockroot/f'{method}_completed.json',verify_completion(APP,method))
            prepared.add(key)
    return 0

if __name__=='__main__': raise SystemExit(main())
