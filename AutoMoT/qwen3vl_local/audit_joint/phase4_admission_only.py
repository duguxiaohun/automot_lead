"""Finish admission against existing data; never replay or compile a native pool."""
import argparse
import json
from pathlib import Path
import sys

from .io import file_sha
from qwen3vl_local.sft_new_loop_phase4.identity import write_json
from .rule_baseline import ADMITTED

APP = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).with_name('release_sources_v90_data_dependencies_20261022.json')


def inputs(prepared, output, data_root, model_dir, complete_view=False):
    from .local_paths import check_inputs
    result=check_inputs(data_root,model_dir,output)
    prepared=Path(prepared).absolute();output=Path(output).absolute()
    if output.resolve().is_relative_to(prepared.resolve()) or prepared.resolve().is_relative_to(output.resolve()):
        result['errors'].append('OUTPUT_DIR must be separate from PREPARED_DIR')
    for name in ('data4/manifest.json','references4/references.json'):
        if not (prepared/name).is_file():result['errors'].append('Required completed asset missing: '+str(prepared/name))
    if not complete_view and not (prepared/'view/student_view.json').is_file():
        result['errors'].append('Existing student_view.json missing; use --complete-view only to build a fresh view')
    if result['errors']:raise ValueError('; '.join(result['errors']))
    return result


def finish(prepared, output, data_root, model_dir, complete_view=False):
    from .first_epoch_budget import profile, request
    from .rule_baseline import check
    from .server_context import context
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.data import create
    prepared,output,data_root,model_dir=map(lambda p:Path(p).absolute(),(prepared,output,data_root,model_dir))
    state=dict(status='building',stage='existing_metadata',stages={},errors=[],training_started=False,
               native_replay_started=False,native_compilation_started=False,
               coverage_policy='existing_data_development',unsupported_capabilities=[],
               server_context=context(prepared=prepared,output=output,data_root=data_root,model_dir=model_dir))
    def stage(name):
        state['stage']=name;write_json(output/'build_result.json',state)
        print(json.dumps(dict(stage=name)),flush=True)
    try:
        # These metadata identities bind both source inputs and the refreshed budget.
        paths={'native':prepared/'data4/manifest.json','references':prepared/'references4/references.json'}
        before={k:file_sha(p) for k,p in paths.items()}
        view=output/'view' if complete_view else prepared/'view'
        if complete_view:
            stage('student_view_only')
            create(prepared/'data4',view,data_root=data_root)
        descriptor=json.loads((view/'student_view.json').read_text())
        if Path(descriptor['parent']['logical_path']).resolve() != (prepared/'data4').resolve():
            raise ValueError('Student view does not belong to PREPARED_DIR/data4')
        paths['view']=view/'student_view.json';before['view']=file_sha(paths['view'])
        state['existing_assets']={k:dict(path=str(p),sha256=before[k]) for k,p in paths.items()}
        stage('refresh_first_epoch_budget')
        measured=profile(model_dir);write_json(output/'model_budget_profile.json',measured)
        budget=request(measured,output,stored_evidence=before)
        write_json(output/'first_epoch_budget.json',budget)
        stage('automatic_admission')
        admission=check(view,prepared/'references4',budget=budget,space_path=output,
                        coverage_policy='existing_data_development',data_root=data_root,batch_size=2)
        if {k:file_sha(p) for k,p in paths.items()}!=before:
            admission['status']='blocked';admission['errors'].append('Existing input metadata changed during admission')
        write_json(output/'automatic_admission.json',admission)
        state.update(status=admission['status'],stage='finished_cpu',errors=admission['errors'],
                     unsupported_capabilities=admission['unsupported_capabilities'])
    except Exception as exc:
        state.update(status='blocked',errors=[f'{type(exc).__name__}: {exc}'])
    write_json(output/'build_result.json',state)
    return 0 if state['status'] in ADMITTED else 2


def main():
    from .build_storage import plan
    from .guard import run as guarded
    from .server_finish import package
    from .server_readiness import source_check
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--model-dir',type=Path,default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--complete-view',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:return finish(a.prepared,a.output,a.data_root,a.model_dir,a.complete_view)
    inputs(a.prepared,a.output,a.data_root,a.model_dir,a.complete_view)
    a.output.mkdir(parents=True,exist_ok=False)
    code=2
    try:
        sources=source_check(APP,RELEASE)
        sources['admission_runner_sha256']=file_sha(Path(__file__))
        write_json(a.output/'source_check.json',sources)
        if sources['status']!='matched':raise ValueError('Published sources mismatch')
        storage=plan(a.output);write_json(a.output/'build_storage.json',storage)
        if not storage['can_start']:raise ValueError('Available space is at/below 20 GiB reserve')
        # Keep worker/guard diagnostics in build.log for the bounded handoff.
        import os
        command=[sys.executable,'-m','qwen3vl_local.audit_joint.phase4_admission_only','--worker',
                 '--prepared',str(a.prepared.absolute()),'--output',str(a.output.absolute()),
                 '--data-root',str(a.data_root.absolute()),'--model-dir',str(a.model_dir.absolute())]
        if a.complete_view:command.append('--complete-view')
        saved=[os.dup(1),os.dup(2)]
        try:
            with (a.output/'build.log').open('xb') as log:
                sys.stdout.flush();sys.stderr.flush();os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
                code=guarded(command,a.output,storage['reserve_bytes'])
        finally:
            sys.stdout.flush();sys.stderr.flush();os.dup2(saved[0],1);os.dup2(saved[1],2)
            for fd in saved:os.close(fd)
    except Exception as exc:
        write_json(a.output/'build_result.json',dict(status='blocked',stage='launch',errors=[f'{type(exc).__name__}: {exc}']))
    package(a.output,code)
    print(json.dumps(dict(exit_code=code,handoff=str(a.output/'handoff'))),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
