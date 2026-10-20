"""New four-RGB choice experiment: independent event/action sampling, 7680 optimizer updates.

The quick stage stops at epoch one without shortening the three-epoch schedule.
No old model evaluation, data building, automatic retries or test-based selection.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

from .io import file_sha, digest
from .phase3_continue import write, read, gpu_ids

APP = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name('phase3_sequence_reference.json')
RELEASE = Path(__file__).with_name('phase3_base_trial_sources.json')
POLICY = 'phase3_independent_base_7680_v1'


def profile():
    return dict(world_size=4, batch_size=1, grad_accum=1, num_epochs=3,
        focus_balance_count=1024, presentations_per_epoch=10240,
        steps_per_epoch=2560, optimizer_updates=7680, total_presentations=30720,
        warmup_steps=2000, learning_rate=1e-5, anchor_stride=1,
        history_rgb_mode='4rgb', action_output_mode='choice',
        sampling_policy='event_action_independent_v1',
        augmentation='historical_per_epoch_seed',
        data_rebuilt=False, old_models_evaluated=False)


def verified_sources():
    spec = read(RELEASE)
    if spec.get('schema') != 'phase3_base_trial_release_v1' or spec.get('policy') != POLICY:
        raise ValueError('Unsupported experiment release')
    for name, expected in spec['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or file_sha(APP/p) != expected:
            raise ValueError('Published trial source mismatch: '+name)
    return file_sha(RELEASE)


def reference():
    ref = read(REFERENCE)
    if ref['schema'] != 'phase3_sequence_trial_reference_v1':
        raise ValueError('Unsupported frozen reference')
    for kind, size in [('loss', 160), ('generation', 320)]:
        ids = ref['validation'][kind]['selected_ids']
        if len(ids) != size or len(set(ids)) != size:
            raise ValueError('Malformed frozen validation IDs')
    return ref


def select_reference(work, ref, *, seed, limit, stride=1, split='val'):
    """Use frozen IDs in their saved order, never fill missing members."""
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    from qwen3vl_local.sequence_batch import digest as item_digest
    if stride != 1:
        raise ValueError('Four-RGB trial requires stride 1')
    if split == 'test':
        plan = ref['test_plan']
        if seed != ref['seed'] or limit != 0:
            raise ValueError('Test must use the complete frozen plan')
    else:
        plans = [v for v in ref['validation'].values()
                 if v['seed'] == seed and v['per_event'] == limit]
        if len(plans) != 1:
            raise ValueError('Unknown validation seed/budget; no implicit reselection')
        plan = plans[0]
    by_id = {}
    for item in work:
        if item.row.split != split:
            raise ValueError('Frozen evaluation split mismatch')
        key = item_digest(case_identity(item))
        if key in by_id:
            raise ValueError('Duplicate evaluation identity')
        by_id[key] = item
    ids = plan['selected_ids']
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError('Frozen evaluation members missing or duplicated')
    selected = [by_id[i] for i in ids]
    counts = Counter(x.row.context_id for x in selected)
    if counts != dict.fromkeys(CONTEXT_IDS, plan['per_event']):
        raise ValueError('Frozen event quotas changed')
    return selected, dict(policy='phase3_frozen_historical_members_v1', seed=seed,
        per_event=plan['per_event'], limit=limit, selected_ids=ids,
        ordered_ids_sha256=item_digest(ids), membership_sha256=item_digest(sorted(ids)),
        pool_sha256=item_digest(sorted(by_id)), invalid_unique_cases=0,
        invalid_policy='choice excludes INVALID', available=dict(Counter(x.row.context_id for x in work)),
        reference_sha256=file_sha(REFERENCE), scope='fixed historical development members; no new selection')


def build_job(args, output):
    ref = reference()
    index, data, model = [p.resolve(strict=True) for p in (args.index, args.data_root, args.model_dir)]
    if file_sha(index) != ref['index_sha256']:
        raise ValueError('Index differs from reviewed baseline; no rebuild or SHA override')
    if not data.is_dir() or not (model/'config.json').is_file():
        raise ValueError('Local data/model missing')
    for protected in (index.parent, data, model):
        if output.resolve().is_relative_to(protected):
            raise ValueError('Output must be separate from data and model')
    resume = str(args.resume.resolve(strict=True)) if args.resume else None
    resume_epoch = 0
    if resume:
        match = re.fullmatch(r'epoch_000([12])', Path(resume).name)
        if not match:
            raise ValueError('Resume requires a complete epoch 1 or 2 checkpoint')
        resume_epoch = int(match[1])
        receipt = read(Path(resume)/'receipt.json')
        cfg = receipt['binding']['configuration']
        if cfg.get('base_trial', {}).get('policy') != POLICY:
            raise ValueError('Resume requires this 7680-update experiment, not an older run')
        if output.resolve().is_relative_to(Path(resume).parents[2]):
            raise ValueError('Resume output must be outside the previous experiment')
        if args.stage == 'quick':
            raise ValueError('Use --stage complete to resume beyond the first epoch')
    job = dict(policy=POLICY, profile=profile(), reference_sha256=file_sha(REFERENCE),
        release_sha256=verified_sources(), index=str(index), index_sha256=ref['index_sha256'],
        data_root=str(data), model_dir=str(model), output=str(output), resume=resume, resume_epoch=resume_epoch,
        stop_after_epoch=1 if args.stage == 'quick' else None,
        stage=args.stage, old_models_evaluated=False, data_rebuilt=False)
    mode=getattr(args,'experiment','base')
    job['mode']=mode
    if mode!='base':
        from .phase3_base_dense.planning import options
        for name,default in [('forecast_protocol','legacy'),('trajectory_coverage_policy','masked'),('supervision','all'),('forecast_span_seconds',6.),('forecast_step_seconds','auto'),
                             ('min_future_coverage',.8),('min_eval_coverage',.8),('max_batch_tokens',16384)]:
            job[name]=getattr(args,name,default)
        job['forecast_contract']=options(job)
        targets=getattr(args,'trajectory_targets',None)
        job['trajectory_targets']=str(targets.resolve(strict=True)) if targets else None
        if targets and output.resolve().is_relative_to(targets.resolve().parent):
            raise ValueError('Output overlaps trajectory target artifacts')
        if not .5<=job['min_eval_coverage']<=1:raise ValueError('min-eval-coverage must be in [0.5,1]')
        if not 8192<=job['max_batch_tokens']<=32768:
            raise ValueError('max-batch-tokens must be in [8192,32768]')
        aux=getattr(args,'aux_index',None)
        job['aux_index']=str(aux.resolve(strict=True)) if aux else None
        job['aux_binding']=None
        if aux:
            manifest=aux.resolve().with_name('manifest.json')
            if not manifest.is_file():raise ValueError('Auxiliary index requires its native manifest.json')
            if output.resolve().is_relative_to(aux.resolve().parent):raise ValueError('Output overlaps auxiliary labels')
            job['aux_binding']=dict(index_sha256=file_sha(aux),manifest_sha256=file_sha(manifest))
        initialization=getattr(args,'dense_initialization','base_adapter')
        if initialization not in ('qwen','base_adapter'):
            raise ValueError('Unknown dense initialization')
        job['initialization']=initialization
        if initialization=='qwen':
            if getattr(args,'init_adapter',None):
                raise ValueError('Qwen initialization cannot also load an adapter')
            job.update(init_adapter=None,init_adapter_files={})
        else:
            if not getattr(args,'init_adapter',None):
                raise ValueError('Dense routes require an explicit restored base adapter or --dense-initialization qwen')
            init=args.init_adapter.resolve(strict=True)
            cfg=read(init/'sft_new_loop_phase3_adapter_config.json')
            if (cfg.get('sampling_policy')!='event_action_independent_v1'
                    or cfg.get('global_step')!=7680 or cfg.get('dense_contract')):
                raise ValueError('Initialize from the completed independent 7680-update base, not sequence/dense weights')
            job.update(init_adapter=str(init),init_adapter_files={str(p.relative_to(init)):file_sha(p)
                for p in sorted(init.rglob('*')) if p.is_file()})
    elif getattr(args,'init_adapter',None):
        raise ValueError('The restored base starts from Qwen; init-adapter is only for dense routes')
    elif getattr(args,'aux_index',None) or getattr(args,'supervision','all')!='all':
        raise ValueError('Auxiliary labels/supervision controls are only for dense experiments')
    return job


def run_stage(command, log, env, *, timeout, reserve_bytes):
    """Own process group, preserve child errors, stop on absolute disk reserve."""
    started = time.monotonic()
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=APP, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        forced = None
        try:
            while child.poll() is None:
                if time.monotonic()-started > timeout:
                    forced = 124
                    break
                if shutil.disk_usage(log.parent).free < reserve_bytes:
                    forced = 75
                    break
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f'[running] {log.stem} pid={child.pid} elapsed={time.monotonic()-started:.0f}s log={log}', flush=True)
            if forced is not None:
                print(f'[stop] {log.stem} reason={forced} (75=disk reserve, 124=timeout)', flush=True)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    code = forced if forced is not None else child.returncode
    return dict(exit_code=code if code >= 0 else 128-code,
                seconds=time.monotonic()-started, log=str(log))


def verify_training(job):
    root = Path(job['output'])/'train'
    manifest = read(root/'train_run_manifest.json')
    schedule = manifest['learning_rate_schedule']
    if (manifest['total_steps_global'] != 7680 or manifest['batch_size'] != 1
            or manifest['world_size'] != 4 or manifest['train_work_global'] != 10240
            or schedule['total_optimizer_updates'] != 7680 or schedule['warmup_updates'] != 2000):
        raise ValueError('Training did not use the 7680-update plan')
    endpoint = 1 if job['stage'] == 'quick' else 3
    start = 0
    if job['resume']:
        # The native reader checks tensor state, sources, optimizer and config.
        start = job['resume_epoch']
    consumed = []
    for epoch in range(start+1, endpoint+1):
        record = read(root/f'balance/consumed_epoch_{epoch:04d}.json')
        from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
        expected = dict.fromkeys(CONTEXT_IDS, 1024)
        expected['INVALID'] = 0
        if record['status'] != 'verified' or record['actual'] != expected:
            raise ValueError('Actual event consumption mismatch')
        if job.get('mode','base')!='base':
            from .phase3_base_dense.baseline import digest
            import math
            plan=read(Path(job['output'])/'dense_coverage.json')['epoch_plans'][str(epoch-1)]
            queries=read(root/f'balance/query_consumed_epoch_{epoch:04d}.json')
            if (queries['status']!='verified_after_backward' or queries['plan_sha256']!=digest(plan)
                    or queries['actual_query_exposure']!=plan['planned_query_exposure']
                    or queries['anchor_presentations']!=10240 or queries['ranks']!=4
                    or set(queries['actual_query_loss_mass'])!=set(plan['planned_query_loss_mass'])
                    or any(not math.isclose(queries['actual_query_loss_mass'][k],v,rel_tol=1e-5,abs_tol=1e-4)
                           for k,v in plan['planned_query_loss_mass'].items())):
                raise ValueError('Actual dense query consumption mismatch')
        consumed.append(epoch)
    if job['stage'] == 'quick':
        progress = read(root/'training_progress.json')
        if progress['global_step'] != 2560 or progress['planned_epochs'] != 3:
            raise ValueError('Quick stage changed the full schedule')
        checkpoint = root/'training_states/epoch_0001'
        if not (checkpoint/'receipt.json').is_file():
            raise ValueError('Quick stage lacks complete resume checkpoint')
    else:
        cfg = read(root/'final/sft_new_loop_phase3_adapter_config.json')
        if cfg['global_step'] != 7680:
            raise ValueError('Final adapter has wrong update count')
    return dict(completed_epochs=endpoint, optimizer_updates=endpoint*2560,
                newly_verified_epochs=consumed, planned_optimizer_updates=7680,
                conclusion='execution only; model quality is assessed separately')


def package(output, state):
    write(output/'workflow.json', state)
    files = {}
    names = {'workflow.json', 'job.json', 'preflight.json', 'validation_summary.json',
             'suite_contract.json', 'shared_data_receipt.json', 'metrics.json', 'training_progress.json', 'train_run_manifest.json',
             'train_eval_metrics.jsonl', 'final_generation.json', 'dense_val_cases.json',
             'dense_test_cases.json', 'dense_val_summary.json', 'dense_test_summary.json',
             'dense_val_sequence_metrics.json', 'dense_test_sequence_metrics.json',
             'dense_coverage.json', 'dense_probe.json', 'dense_input_shapes.json', 'dense_admission_failure.json',
             'trajectory_targets_summary.json','trajectory_train_coverage_epoch0.json',
             'trajectory_train_coverage_epoch1.json','trajectory_train_coverage_epoch2.json',
             'trajectory_eval_coverage_val.json','trajectory_eval_coverage_test.json',
             'dense_aux_summary.json','dense_evaluation_plan.json','dense_val_transfer_summary.json',
             'dense_test_transfer_summary.json','dense_val_transfer_cases.json','dense_test_transfer_cases.json'}
    for p in sorted(output.rglob('*')):
        if not p.is_file() or 'handoff' in p.relative_to(output).parts or 'training_states' in p.parts:
            continue
        relative = str(p.relative_to(output))
        if (p.name in names or p.parent.name == 'balance' and p.name.startswith(('epoch_', 'consumed_', 'plan_', 'query_consumed_'))
                or re.fullmatch(r'epoch_\d+_rank\d+_batches\.jsonl',p.name)):
            limit_mib=(32 if p.name in ('dense_coverage.json','dense_input_shapes.json') else
                       8 if re.fullmatch(r'epoch_\d+_rank\d+_batches\.jsonl',p.name) else 4)
            if p.stat().st_size > limit_mib*1024**2:
                raise ValueError('Oversized report: '+relative)
            files[relative] = p.read_bytes()
        elif p.name in ('generation_val_cases.jsonl', 'final_generation_val_cases.jsonl') or (p.parent.name == 'test' and p.name.startswith('cases') and p.suffix == '.jsonl'):
            if p.stat().st_size > 32*1024**2:
                raise ValueError('Oversized validation cases')
            files[relative] = p.read_bytes()
        elif p.suffix == '.log':
            with p.open('rb') as stream:
                stream.seek(max(0, p.stat().st_size-60000))
                files[relative] = stream.read()
    if sum(map(len, files.values())) > 128*1024**2:
        raise ValueError('Handoff exceeds uncompressed budget')
    dest = output/'handoff'; dest.mkdir(exist_ok=True)
    archive = dest/'phase3_base_report.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    if archive.stat().st_size > 30*1024**2:
        raise ValueError('Handoff exceeds 30 MiB')
    write(dest/'phase3_base_package.json', dict(status=state['status'],
        bytes=archive.stat().st_size, archive_sha256=file_sha(archive),
        files={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}))
    print('[handoff] '+str(dest), flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=APP/'lead_data')
    p.add_argument('--model-dir', type=Path, default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stage', choices=['quick', 'complete'], default='quick')
    p.add_argument('--resume', type=Path)
    p.add_argument('--experiment', choices=['base','future','stream','windows','windows_prefix','windows_bank'], default='base')
    p.add_argument('--init-adapter', type=Path)
    p.add_argument('--dense-initialization', choices=['base_adapter','qwen'], default='base_adapter')
    p.add_argument('--aux-index', type=Path, help='Separately validated native labels; matching base routes/splits only, raw meta recomputation required')
    p.add_argument('--supervision', choices=['all','single'], default='all')
    p.add_argument('--forecast-protocol',choices=('legacy','trajectory'),default='legacy')
    p.add_argument('--trajectory-coverage-policy',choices=('masked','strict'),default='masked')
    p.add_argument('--trajectory-targets',type=Path,help='Reuse a verified trajectory_targets.json from this release')
    p.add_argument('--forecast-span-seconds', type=float, default=6.)
    p.add_argument('--forecast-step-seconds', default='auto')
    p.add_argument('--min-future-coverage',type=float,default=.8)
    p.add_argument('--min-eval-coverage',type=float,default=.8)
    p.add_argument('--max-batch-tokens',type=int,default=16384)
    p.add_argument('--prepare-only',action='store_true',help='Run CPU admission and full planned input shape audit; no GPU selection or training')
    p.add_argument('--gpu-ids', default=os.environ.get('GPU_IDS', ''))
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--timeout-seconds', type=int, default=86400)
    p.add_argument('--reserve-gib', type=int, default=20)
    args = p.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists; use a fresh directory')
    if args.timeout_seconds <= 0 or args.reserve_gib < 20:
        raise ValueError('Positive timeout and >=20 GiB reserve required')
    ids = '' if args.prepare_only or args.plan_only else gpu_ids(args.gpu_ids)
    job = build_job(args, output)
    if args.plan_only:
        print(json.dumps(job, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True)
    state = dict(policy=POLICY, status='preflight', root=str(APP), output=str(output),
        hostname=socket.gethostname(), server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'),
        profile=profile(), stages=[], data_rebuilt=False, old_models_evaluated=False)
    lock_fd = None
    code = 0
    try:
        if job['resume']:
            lock_fd = os.open(job['resume'], os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(output).free < (args.reserve_gib+1)*1024**3:
            raise ValueError('Insufficient free space above the reserve')
        write(output/'job.json', job)
        env = dict(os.environ, PYTHONPATH=str(APP), PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=ids, GPU_IDS=ids, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            TORCH_NCCL_TRACE_BUFFER_SIZE='200000', TORCH_NCCL_DUMP_ON_TIMEOUT='1',
            TORCH_NCCL_DEBUG_INFO_TEMP_FILE=str(output/'nccl_rank_'))
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        worker = 'qwen3vl_local.audit_joint.phase3_base_trial_worker'
        if job['mode']!='base':
            worker='qwen3vl_local.audit_joint.phase3_base_dense.worker'
        commands = [('preflight', [sys.executable, '-m', worker, '--job', str(output/'job.json'), '--operation', 'preflight']),
            ('train', [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                '--max_restarts=0', '--module', worker, '--job', str(output/'job.json'), '--operation', 'train']),
            ('summarize', [sys.executable, '-m', worker, '--job', str(output/'job.json'), '--operation', 'summarize'])]
        if job['mode']!='base':
            commands.insert(1,('probe',[sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=4','--max_restarts=0','--module',worker,'--job',str(output/'job.json'),'--operation','probe']))
        if args.prepare_only:commands=commands[:1]
        for name, command in commands:
            state['active_stage'] = name
            write(output/'workflow.json', state)
            result = run_stage(command, output/f'{name}.log', env,
                timeout=args.timeout_seconds if name in ('train','preflight','probe') else min(args.timeout_seconds,3600),
                reserve_bytes=args.reserve_gib*1024**3)
            state['stages'].append(dict(stage=name, **result))
            if result['exit_code']:
                code = result['exit_code']
                raise RuntimeError(f'{name} exited {code}')
        if args.prepare_only:
            state.update(status='prepared_cpu_no_training',active_stage=None)
        else:
            state['training'] = verify_training(job)
            if job['stage'] == 'complete':
                result = run_stage([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=4', '--max_restarts=0', '--module', worker, '--job', str(output/'job.json'),
                    '--operation', 'test'], output/'test.log', env, timeout=min(args.timeout_seconds, 3600),
                    reserve_bytes=args.reserve_gib*1024**3)
                state['stages'].append(dict(stage='test', **result))
                if result['exit_code']:
                    code = result['exit_code']
                    raise RuntimeError(f'Test exited {code}')
            state['status'] = 'quick_epoch_completed_quality_pending' if job['stage'] == 'quick' else 'planned_training_completed_quality_pending'
            state['active_stage'] = None
    except BaseException as exc:
        state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        print(state['error'], file=sys.stderr, flush=True)
        code = code or 2
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            package(output, state)
        except Exception as exc:
            print('Handoff failed: '+str(exc), file=sys.stderr)
            code = code or 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
