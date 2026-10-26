"""Finish the exact admitted ABC run after its post-training group-cleanup bug.

External, SHA-pinned recovery supplement. Never edits a frozen source tree,
changes admissions, migrates weights, or resumes an incomplete optimizer run.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

CORE='qwen3vl_local/action_prior/training_core.py'
WORKER='qwen3vl_local/action_expert_ablation/geometry_abc/worker.py'
SUPPORTED={CORE:'d38a8af004d698a0240de58e47073655ef8e1e0f54e772afa5b4d73cecf76e64',
           WORKER:'fe79427e44c66bf80324c4998c3209bb6a2640b4e465234a1954b42385558017'}
SCHEMA='abc_post_training_completion_recovery_v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def pinned_loop(core,**kw):
    """Original wrapper lifecycle except group destruction stays with worker.

    The original numerical _run_training_loop and every passed argument remain
    unchanged. This supplement is recorded separately, not disguised as old code.
    """
    coordinator=core.TerminationCoordinator()
    core._archive_stale_termination(kw['out'],kw['rank'],kw['world'])
    try:
        with coordinator:
            core._ACTIVE_TERMINATION=coordinator
            return core._run_training_loop(**kw,termination=coordinator)
    finally:
        coordinator.close_iterators()
        core._ACTIVE_TERMINATION=None
        if kw['writer']:kw['writer'].close()


def source_gate(root):
    from qwen3vl_local.action_expert_ablation.geometry_abc import contracts
    gate=read(root/'admission.json');current=contracts.source_identity()
    if current!=gate['code']:raise ValueError('frozen_source_differs_from_original_admission')
    if any(current.get(k)!=v for k,v in SUPPORTED.items()):
        raise ValueError('unreviewed_recovery_source_version')
    for k in ('data_ready','plan_ready','processor_ready','raw_sources_verified','history_ready','development_ready','package_ready'):
        if not gate.get(k):raise ValueError('incomplete_original_admission: '+k)
    return gate


def paths(root):
    config=read(root/'a_bev_only/config.json')
    if Path(config['suite_root']).resolve()!=root:raise ValueError('run_root_moved_requires_separate_review')
    return config


def verify_environment(root,gate,config):
    from qwen3vl_local.action_expert_ablation.geometry_abc import preflight,probe_inputs,indexed_rows
    directory=Path(config['data_dir'])
    if preflight.resources(config['model_dir'],config['lead_bev_ckpt'])!=gate['resources']:
        raise ValueError('original_model_resources_changed')
    # Full split hashes and route receipts; actual consumed sensors are rehashed
    # by the unchanged native loader during evaluation/future training.
    rows,manifest=indexed_rows.load(directory,data_root=config['data_root'],manifest_sha256=gate['manifest_sha256'])
    if manifest['identity']!=gate['dataset'] or manifest['splits']!=gate['splits']:
        raise ValueError('original_dataset_changed')
    preflight.verify_development(manifest)
    projection=read(directory/'plan.json')
    if preflight.plan_digest(projection)!=gate['plan_sha256'] or projection['sha256']!=gate['plan_sha256']:
        raise ValueError('original_plan_changed')
    processor=probe_inputs.verify_processor(root/'processor.json',gate)
    selections={v:probe_inputs.selection((r for rs in rows.values() for r in rs.probe_records()),processor,v) for v in 'abc'}
    from qwen3vl_local.action_expert_ablation.geometry_abc import audit,contracts
    for group in audit.GROUPS:
        p=root/'probes'/group/'probe_complete.json';receipt=read(p);v,b=group.split('_',1);binding=receipt['binding']
        probe_inputs.verify_probe_inputs(receipt,selections[v],gate['processor_sha256'],p.parent)
        if (receipt['admission_sha256']!=sha(root/'admission.json') or not receipt.get('checkpoint_reload_verified')
            or not receipt.get('post_checkpoint_backward_verified') or receipt['checkpoint_sha256']!=sha(p.with_name('probe.pt'))
            or binding['source']!=gate['code'] or binding['dataset']!=gate['dataset'] or binding['resources']!=gate['resources']
            or binding['representation']!=contracts.contract(v) or binding['branch']!=b):
            raise ValueError('original_probe_changed: '+group)
    return manifest,projection


def evidence_inventory(run):
    paths=[run/n for n in ('config.json','condition_contract.json','training_plan.json','latest.pt','best.pt','validation/final.json','training_audit/latest.json')]
    for folder in ('epoch_audit','token_diagnostics','performance','validation','training_audit/epochs'):
        paths.extend(sorted((run/folder).glob('*.json')))
    return {str(p.relative_to(run)):sha(p) for p in sorted(set(paths))}


def validate_completed(run,manifest,projection,gate,*,publish=False):
    """CPU-only: certify a finished optimizer budget, never advance training."""
    from qwen3vl_local.action_expert_ablation.geometry_abc import audit,indexed_rows,worker
    from qwen3vl_local.action_prior import training_core as core
    from types import SimpleNamespace
    before=evidence_inventory(run)
    latest=indexed_rows.checkpoint(run/'latest.pt');best=indexed_rows.checkpoint(run/'best.pt')
    binding=read(run/'condition_contract.json');config=read(run/'config.json');v,b=run.name.split('_',1)
    if (binding['source']!=gate['code'] or binding['dataset']!=gate['dataset'] or binding['resources']!=gate['resources']
        or binding['plan']!=gate['plan_sha256'] or binding['splits']!=gate['splits']):raise ValueError('completed_binding_changed')
    for state in (latest,best):
        if (state['schema']!='abc_trajectory_checkpoint_v1' or state['condition_contract']!=binding
            or state['dataset_hashes']!=manifest['splits'] or state['world_size']!=4
            or len(state['rng_by_rank'])!=4 or state['args']!=config
            or worker.numerical_config(SimpleNamespace(**state['args']))!=binding['numerical_config']):
            raise ValueError('checkpoint_identity_or_configuration_changed')
    if latest['step']!=12719 or latest['cursor'].get('epoch')!=7 or latest['cursor'].get('micro')!=0 or not core.budget_complete(latest['step'],{'actual_step_limit':12719},latest['cursor']):
        raise ValueError('incomplete_training_cannot_be_finalized')
    final=read(run/'validation/final.json');audit_state=read(run/'training_audit/latest.json')
    if (final['optimizer_step']!=12719 or final['checkpoint']!='latest.pt' or final['weight_view']!='ema'
        or final['split']!='val' or final['metrics']['samples']<=0 or audit_state['optimizer_step']!=12719
        or audit_state['validation_pending'] or audit_state['validation']!=final['metrics']):
        raise ValueError('final_validation_incomplete_or_changed')
    validations=[read(p) for p in sorted((run/'validation').glob('*_step*.json'))]
    scored=[x for x in validations if x.get('validation_full_epoch') or x.get('validation_cycle') or x['optimizer_step']==12719]
    if not scored:raise ValueError('missing_selected_validation')
    # Original strict improvement policy: earliest best score wins.
    selected=min(scored,key=lambda x:(core.best_selection_score(x,SimpleNamespace(**config)),x['optimizer_step']))
    score=core.best_selection_score(selected,SimpleNamespace(**config))
    if best['step']!=selected['optimizer_step'] or best['best_val']!=score or latest['best_val']!=score or final['best_selection_score']!=score:
        raise ValueError('best_checkpoint_selection_changed')
    receipts={}
    for epoch in range(1,8):
        candidates=list((run/'epoch_audit').glob(f'epoch_{epoch:03d}_*.json'))
        if len(candidates)!=1:raise ValueError('missing_epoch_consumption')
        receipts[str(epoch)]=sha(candidates[0])
    candidate=dict(binding=binding,step=12719,epochs=receipts,
        token_diagnostics={p.name:sha(p) for p in sorted((run/'token_diagnostics').glob('*.json'))},
        latest_sha256=before['latest.pt'],best_sha256=before['best.pt'])
    # Native full verifier sees a temporary view. Do not publish a success
    # marker until original exposure, slots, gradients and weights all pass.
    with tempfile.TemporaryDirectory(prefix='abc_completion_verify_') as temporary:
        view=Path(temporary)
        for p in run.iterdir():
            if p.name!='training_complete.json':(view/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
        write(view/'training_complete.json',candidate)
        audit.verify_training(view,manifest,projection,v,b,manifest_sha256=gate['manifest_sha256'])
    del latest,best
    if evidence_inventory(run)!=before:raise ValueError('completion_inputs_changed_during_verification')
    target=run/'training_complete.json'
    if target.exists():
        if read(target)!=candidate:raise ValueError('existing_completion_differs')
    elif publish:write(target,candidate)
    return dict(group=run.name,step=12719,validated=True,published=publish,inputs=before,
        per_rank_performance='recorded' if (run/'performance/ranks.json').is_file() else 'missing after original all_gather failure; no values fabricated',
        optimizer_updates_added=0)


def worker_command(root,config,group,task,script):
    v,b=group.split('_',1);out=root/group if task=='train' else root/group/'test'
    args=['--operation',task,'--abc-variant',v,'--branch',b,'--suite-root',str(root),
          '--data-dir',config['data_dir'],'--data-root',config['data_root'],'--output-dir',str(out),
          '--model-dir',config['model_dir'],'--lead-bev-ckpt',config['lead_bev_ckpt']]
    if task=='evaluate':args+=['--checkpoint',str(root/group/'best.pt')]
    return [sys.executable,'-B','-m','torch.distributed.run','--standalone','--max_restarts=0','--nproc_per_node=4',
            str(script),'--internal','worker','--run-dir',str(root),'--',*args]


def run_inside(root,mode,worker_args,check_only):
    from qwen3vl_local.audit_joint.abc_delivery import lock
    gate=source_gate(root);config=paths(root)
    recovery=root/'completion_recovery';script=Path(__file__).resolve()
    proof=read(recovery/'supplement.json')
    if proof['script_sha256']!=sha(script) or proof['admission_sha256']!=sha(root/'admission.json'):
        raise ValueError('recovery_supplement_changed')
    if mode=='worker':
        from qwen3vl_local.action_prior import training_core as core
        from qwen3vl_local.action_expert_ablation.geometry_abc import worker
        from qwen3vl_local.action_prior.progress import observed
        original=core.run_training_loop
        core.run_training_loop=lambda **kw:pinned_loop(core,**kw)
        try:return observed(worker.main)(worker_args)
        finally:core.run_training_loop=original
    from qwen3vl_local.action_expert_ablation.geometry_abc import audit
    from qwen3vl_local.action_prior.host_memory import run_logged
    with lock(root):
        state=dict(schema=SCHEMA,status='verifying',groups={});write(recovery/'status.json',state)
        try:
            manifest,projection=verify_environment(root,gate,config)
            # Inspect every occupied group before starting any new long job.
            for group in audit.GROUPS:
                run=root/group
                if run.exists():
                    report=validate_completed(run,manifest,projection,gate,publish=not check_only)
                    write(recovery/(group+'_completion.json'),report)
            if check_only:
                state['status']='check_passed';write(recovery/'status.json',state);print(json.dumps(state));return
            for group in audit.GROUPS:
                run=root/group
                if not run.exists():
                    state.update(status='training',active=group);write(recovery/'status.json',state)
                    run_logged(worker_command(root,config,group,'train',script),recovery/'logs'/(group+'_train.log'))
                report=validate_completed(run,manifest,projection,gate,publish=True)
                write(recovery/(group+'_completion.json'),report)
                if not all((run/'test'/name).exists() for name in ('metrics.json','condition_sensitivity.json')):
                    # Never silently overwrite a partial evaluation; preserve
                    # it for review and use a fresh native output directory.
                    if (run/'test').exists():
                        import time
                        (run/'test').rename(run/f'test_incomplete_before_recovery_{time.time_ns()}')
                    state.update(status='evaluating',active=group);write(recovery/'status.json',state)
                    run_logged(worker_command(root,config,group,'evaluate',script),recovery/'logs'/(group+'_test.log'))
                state['groups'][group]='test_complete';write(recovery/'status.json',state)
            # Original native audit, with explicit supplemental provenance in ZIP.
            original_pack=audit.pack
            def pack(path,required,*a,**kw):
                required=dict(required)
                for p in sorted(recovery.glob('*.json')):required['completion_recovery/'+p.name]=p
                for p in sorted((recovery/'memory').glob('*.json')):required['completion_recovery/memory/'+p.name]=p
                required['completion_recovery/runner.py']=script
                return original_pack(path,required,*a,**kw)
            audit.pack=pack
            try:result=audit.build(root,config['data_dir'])
            finally:audit.pack=original_pack
            review=result.get('quality_status')=='condition_use_review_required'
            state.update(status='condition_use_review_required' if review else 'completed',delivery=result);write(recovery/'status.json',state)
            print(json.dumps(dict(result_directory=str(root),**result),ensure_ascii=False))
            return 2 if review else 0
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');write(recovery/'status.json',state)
            # Preserve the original failure package and pipeline verbatim.
            from qwen3vl_local.audit_joint.abc_delivery import pack
            required={p.name:p for p in recovery.glob('*.json')}
            for name in ('admission.json','execution_version.json','pipeline.json'):
                if (root/name).is_file():required['original/'+name]=root/name
            for path in (recovery/'memory').glob('*.json'):required['memory/'+path.name]=path
            for log in (recovery/'logs').glob('*.log'):
                with log.open('rb') as f:f.seek(max(0,log.stat().st_size-64000));tail=f.read()
                target=recovery/'failure_delivery/log_tails'/log.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tail)
                required['log_tails/'+log.name]=target
            for path in root.glob('**/progress/*.json'):required[str(path.relative_to(root))]=path
            try:pack(recovery/'failure_delivery/diagnostic.zip',required,summary=dict(final_results=False,error=state['error']))
            except Exception as packing_error:print(f'[completion failure package] {packing_error}',file=sys.stderr)
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');p.add_argument('--internal',choices=['suite','worker'])
    p.add_argument('worker_args',nargs=argparse.REMAINDER);args=p.parse_args(argv)
    root=args.run_dir.resolve(strict=True)
    if args.internal:
        worker_args=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
        raise SystemExit(run_inside(root,args.internal,worker_args,args.check_only) or 0)
    app=Path(__file__).resolve().parent.parent
    if not root.is_relative_to(app/'checkpoints'):raise ValueError('run_must_belong_to_this_server_AutoMoT_checkpoints')
    saved=read(root/'execution_version.json');directory=Path(saved['directory']).resolve(strict=True)
    if not directory.is_relative_to(app/'checkpoints/runtime_versions'):raise ValueError('frozen_version_not_on_current_server')
    spec=importlib.util.spec_from_file_location('recovery_runtime_version',app/'qwen3vl_local/runtime_version.py')
    version=importlib.util.module_from_spec(spec);spec.loader.exec_module(version);record=version.verify(directory)
    if Path(record['origin']).resolve()!=app:raise ValueError('snapshot_belongs_to_other_server')
    if any(record['files'].get(k)!=v for k,v in SUPPORTED.items()):raise ValueError('unsupported_frozen_recovery_version')
    recovery=root/'completion_recovery';recovery.mkdir(exist_ok=True)
    target=recovery/'runner.py';current=Path(__file__).read_bytes()
    if target.exists() and target.read_bytes()!=current:raise ValueError('existing_recovery_supplement_version_changed')
    if not target.exists():target.write_bytes(current)
    supplement=dict(schema=SCHEMA,script_sha256=sha(target),admission_sha256=sha(root/'admission.json'),
        frozen_version=directory.name,source_overrides=SUPPORTED,
        change='default process group destroyed by worker after performance and completion, not inner loop',
        numerical_loop_unchanged=True,scope='original six groups; completed optimizer runs only; no weight migration')
    path=recovery/'supplement.json'
    if path.exists() and read(path)!=supplement:raise ValueError('existing_supplement_contract_changed')
    if not path.exists():write(path,supplement)
    tree=directory/'app'
    env=dict(os.environ,PYTHONPATH=str(tree),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        QWEN_EXECUTION_VERSION=str(directory),QWEN_EXECUTION_ORIGIN=str(app))
    command=[sys.executable,'-B',str(target),'--internal','suite','--run-dir',str(root)]
    if args.check_only:command+=['--check-only']
    process=subprocess.Popen(command,cwd=tree,env=env,start_new_session=True)
    previous={}
    def forward(signum,frame):
        if process.poll() is None:
            try:os.killpg(process.pid,signum)
            except ProcessLookupError:pass
    for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,forward)
    try:code=process.wait()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
        version.verify(directory)
    raise SystemExit(code if code>=0 else 128-code)


if __name__=='__main__':main()
