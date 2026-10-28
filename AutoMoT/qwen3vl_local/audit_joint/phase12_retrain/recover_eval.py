"""Recover completed Phase2 two-image training, then finish evaluation and four-image work.

Old source/admissions/weights are immutable. A separately recorded torchrun entry
supplement only forces adapter deserialization to CPU before native placement.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

APP=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def validate_completed(proof, run, index, mode):
    complete=proof.verify_training(run)
    inputs=proof.read(run/'training_inputs.json')
    if complete['phase']!=2 or complete['budget']['optimizer_updates']!=5376:
        raise ValueError('Recovery requires completed native Phase2 5376-update training')
    if not proof.same_data(inputs['data'],proof.data_identity(index)):
        raise ValueError('Training belongs to different data')
    slot=next((x for x in ('best_generation','final','fallback_generation') if x in complete['adapters']),None)
    if not slot:raise ValueError('No certified adapter slot')
    adapter=run/slot
    cfg=read(adapter/'sft_new_loop_phase2_adapter_config.json')
    if cfg['history_rgb_mode']!=mode:raise ValueError('Adapter RGB mode mismatch')
    return adapter,complete


def torchrun_shim(python, entry):
    # The native shell uses `torchrun ... -- native_script ...`. Change only that
    # script entry, not rank arguments, numerical options, source files or weights.
    return ('#!/bin/sh\nexec '+shlex.quote(python)+' -c '+shlex.quote(
        'import sys; from torch.distributed.run import main; '
        'a=sys.argv[1:]; i=a.index("--")+1; '
        'a.insert(i,'+repr(str(entry))+'); main(a)')+' "$@"\n')


def mode_environment(environment, tree, old, out, info, mode, adapter=None, train_output=None):
    env=environment(tree,2,out,Path(info['index']),Path(info['data_root']),Path(info['model_dir']),tree/'keyframe_filter/collection_output')
    # Each native launcher reserves its entire root and refuses an existing root.
    env.update(HISTORY_RGB_MODES=mode,PIPELINE_TIMESTAMP=old.name,PIPELINE_ROOT=str(out/'pipeline'/mode),
               RUN_TRAIN='0' if adapter else '1',GPU_IDS=os.environ.get('GPU_IDS','0,1,2,3'),
               PYTHONPATH=str(tree),PHASE12_RECOVERY_TREE=str(tree),
               PHASE12_RECOVERY_ADAPTER_SHA=os.environ['PHASE12_RECOVERY_ADAPTER_SHA'],
               PHASE12_RECOVERY_TRACE=str(out/'traces'))
    env['PATH']=str(out/'bin')+os.pathsep+env['PATH']
    if adapter:env['ADAPTER_DIR']=str(adapter)
    if train_output:env['TRAIN_OUTPUT_DIR']=str(train_output)
    return env


def worker(old,out,version,profile):
    # This file is executed externally with PYTHONPATH pointing only at old tree.
    from qwen3vl_local import runtime_version as runtime
    from qwen3vl_local.audit_joint.phase12_retrain.launch import verify_ready,environment
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof, data_stage
    from qwen3vl_local.sft_new_loop_phase1.delivery import verify_package
    tree=version/'app'
    if proof.ROOT.resolve()!=tree.resolve():raise ValueError('Recovery must import original frozen source')
    runtime.verify(version)
    for name,digest in profile.items():
        if proof.sha(tree/name)!=digest:raise ValueError('Unsupported frozen recovery source: '+name)
    ready=verify_ready(old,2,version)
    failed=read(old/'pipeline/2rgb_endpoints/eval_review/lora_production/evaluation_inputs.json')
    info=failed['arguments']
    if Path(info['index']).resolve()!=Path(ready['index']).resolve():raise ValueError('Failed eval index differs from admission')
    if info['split']!='test':raise ValueError('Recovery requires original test evaluation')
    two=Path(info['adapter_dir']).resolve().parent
    expected_two=(tree/'checkpoints/sft_new_loop_phase2_retrain_runs/run_direct_event_format_supervised_2rgb_endpoints'/old.name).resolve()
    four=(tree/'checkpoints/sft_new_loop_phase2_retrain_runs/run_direct_event_format_supervised_4rgb'/old.name).resolve()
    if two!=expected_two:raise ValueError('Failed adapter is outside the original two-image run')
    two_adapter,two_complete=validate_completed(proof,two,ready['index'],'2rgb_endpoints')
    if proof.verify_adapter(Path(info['adapter_dir']))!=failed['adapter_identity']:
        raise ValueError('Failed eval selected adapter changed')
    if two_adapter!=Path(info['adapter_dir']).resolve():raise ValueError('Selected adapter changed since failed evaluation')
    four_adapter=None
    if four.exists():
        four_adapter,_=validate_completed(proof,four,ready['index'],'4rgb')
    (out/'bin').mkdir();(out/'traces').mkdir()
    shim=out/'bin/torchrun';shim.write_text(torchrun_shim(sys.executable,out/'code/cpu_adapter_entry.py'));shim.chmod(0o755)
    os.environ['PHASE12_RECOVERY_ADAPTER_SHA']=profile['qwen3vl_local/qwen35/adapters.py']
    proof.write(out/'recovery_plan.json',dict(old_run=str(old),version=str(version),data=ready['data'],
        two_train=str(two),two_adapter=str(two_adapter),two_training_complete_sha256=proof.sha(two/'training_complete.json'),
        four_train=str(four),four_action='reuse_completed' if four_adapter else 'fresh_train',
        patch='adapter deserialization torch_device=cpu; native model.to(rank device) unchanged',source_profile=profile))
    os.chdir(tree)
    for mode,adapter in [('2rgb_endpoints',two_adapter),('4rgb',four_adapter)]:
        # Detect source/data/weight changes before every stage, not just once.
        verify_ready(old,2,version)
        validate_completed(proof,two,ready['index'],'2rgb_endpoints')
        if mode=='4rgb' and adapter is None and four.exists():raise ValueError('Four-image output appeared during recovery')
        env=mode_environment(environment,tree,old,out,info,mode,adapter, four if mode=='4rgb' else None)
        data_stage.run(['bash','qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh'],
            out/f'{mode}_pipeline.json',timeout=172800,env=env)
        train_root=two if mode=='2rgb_endpoints' else four
        validate_completed(proof,train_root,ready['index'],mode)
        bundles=list((out/'pipeline'/mode).rglob('*_audit_bundle.tar.gz'))
        if not bundles:raise ValueError('Missing native audit package for '+mode)
        for bundle in bundles:
            verify_package(bundle.with_name(bundle.name.removesuffix('.tar.gz')))
            print('[BUNDLE] '+str(bundle),flush=True)
    verify_ready(old,2,version)
    validate_completed(proof,two,ready['index'],'2rgb_endpoints')
    validate_completed(proof,four,ready['index'],'4rgb')
    traces=list((out/'traces').glob('*.jsonl'))
    if not traces:raise ValueError('Missing adapter CPU-read recovery traces')
    for p in traces:
        if any(json.loads(line)['torch_device']!='cpu' for line in p.read_text().splitlines()):raise ValueError('Invalid adapter read trace')
    payloads=[out/'recovery_plan.json',*sorted((out/'code').glob('*')),*traces,
              old/'execution.json',old/'ready.json',old/'data/training_revision.json',old/'selected_train_reviews.json']
    record=dict(status='completed_quality_pending',original_run=str(old),old_source_unchanged=True,
        two_training_repeated=False,four_training_complete_sha256=proof.sha(four/'training_complete.json'),
        native_bundles={str(p):proof.sha(p) for p in (out/'pipeline').rglob('*_audit_bundle.tar.gz')},
        supplemental_files={str(p):proof.sha(p) for p in payloads})
    proof.write(out/'completion.json',record)
    with tarfile.open(out/'recovery_audit.tar.gz','w:gz') as tar:
        tar.add(out/'completion.json',arcname='completion.json')
        for p in payloads:
            name=str(p.relative_to(out)) if p.is_relative_to(out) else 'original/'+str(p.relative_to(old))
            tar.add(p,arcname=name,recursive=False)
    print('[BUNDLE] '+str(out/'recovery_audit.tar.gz'),flush=True)
    print('[DONE] Phase2 recovery: both modes trained, evaluated and packaged; quality pending audit',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--version',type=Path)
    a=p.parse_args();old=a.run_dir.resolve()
    if a.worker:
        return worker(old,a.output.resolve(),a.version.resolve(),read(HERE/'recovery_profile.json'))
    from qwen3vl_local import runtime_version as runtime
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    execution=read(old/'execution.json');version=Path(execution['version']).resolve()
    if execution['phase']!=2 or Path(execution['origin']).resolve()!=APP:
        raise ValueError('Recovery requires this machine original Phase2 run')
    state=read(old/'status.json')
    if state.get('status')!='failed' or state.get('stage')!='fresh_train_eval':raise ValueError('Not the supported failed training/evaluation stage')
    runtime.verify(version)
    with (old/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=old/('recovery_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        (out/'code').mkdir(parents=True,exist_ok=False)
        for name in ('recover_eval.py','cpu_adapter_entry.py','recovery_profile.json'):
            shutil.copy2(HERE/name,out/'code'/name)
        env={**os.environ,'PYTHONPATH':str(version/'app'),'PYTHONDONTWRITEBYTECODE':'1',
             'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')}
        print('[RECOVERY] '+str(out),flush=True)
        try:
            subprocess.run([sys.executable,'-B',str(out/'code/recover_eval.py'),'--worker','--run-dir',str(old),
                            '--output',str(out),'--version',str(version)],cwd=version/'app',env=env,check=True)
            runtime.verify(version)
        except BaseException as exc:
            proof.write(out/'failed.json',dict(error=str(exc),old_run_preserved=True));raise


if __name__=='__main__':main()
