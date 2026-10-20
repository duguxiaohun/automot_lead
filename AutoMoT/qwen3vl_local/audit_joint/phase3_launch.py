"""Separate, guarded base or single-frame A runs; never start scheme B."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import os
from pathlib import Path
import resource
import re
import shutil
import socket
import sys
from types import SimpleNamespace
import zipfile

from . import phase3_base_trial as trial
from .phase3_suite import environment, model_snapshot, check_model_snapshot


def command(module, *args, distributed=False):
    prefix = [sys.executable, '-m']
    if distributed:
        # The explicit separator also protects future application options from
        # torchrun's argparse abbreviation (--run/--resume etc.).
        prefix += ['torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '--max_restarts=0', '--module', '--']
    return prefix+[module, *map(str, args)]


def schedule(method, root, *, prepare_only=False):
    job = root/'experiment'/('job.json' if method == 'base' else 'single_frame_job.json')
    if method == 'base':
        own = 'qwen3vl_local.audit_joint.phase3_launch_worker'
        native = 'qwen3vl_local.audit_joint.phase3_base_trial_worker'
        stages = [('prepare', command(own, 'prepare', '--job', job)),
                  ('probe', command(own, 'probe', '--job', job, distributed=True)),
                  ('train', command(native, '--operation', 'train', '--job', job, distributed=True)),
                  ('summarize', command(own, 'summarize', '--job', job)),
                  ('test', command(own, 'test', '--job', job, distributed=True))]
    else:
        own = 'qwen3vl_local.audit_joint.phase3_single_frame.worker'
        stages = [('probe', command(own, 'probe', '--job', job, distributed=True)),
                  ('train', command(own, 'train', '--job', job, distributed=True)),
                  ('val', command(own, 'evaluate', '--split', 'val', '--job', job)),
                  ('test', command(own, 'evaluate', '--split', 'test', '--job', job))]
    return stages[:1] if prepare_only and method == 'base' else [] if prepare_only else stages


def launch_sources():
    names = ('phase3_launch.py', 'phase3_launch.sh', 'phase3_launch_worker.py',
             'phase3_common_data.py', 'phase3_common_runtime.py')
    return {n: trial.file_sha(Path(__file__).with_name(n)) for n in names}


def package(root, state):
    """Only reports/cases and bounded log tails, never datasets or weights."""
    trial.write(root/'launch_workflow.json', state)
    paths = [root/'launch_workflow.json', *root.glob('*.log')]
    experiment = root/'experiment'
    for p in experiment.rglob('*'):
        rel = p.relative_to(experiment)
        if not p.is_file() or any(part in ('training_states', 'final', 'tb', 'handoff') for part in rel.parts):
            continue
        if p.suffix in ('.json', '.jsonl', '.log'):
            paths.append(p)
    sizes = {p: min(p.stat().st_size, 60000) if p.suffix == '.log' else p.stat().st_size for p in paths}
    if any(n > 64*1024**2 for n in sizes.values()) or sum(sizes.values()) > 128*1024**2:
        raise ValueError('Report exceeds handoff budget; retain original reports, do not rerun training')
    dest = root/'handoff'; dest.mkdir(exist_ok=True)
    pending = dest/'phase3_report.zip.pending'; archive = dest/'phase3_report.zip'
    hashes = {}
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
        for p, size in sizes.items():
            with p.open('rb') as f:
                if p.suffix == '.log': f.seek(max(0, p.stat().st_size-size))
                raw = f.read()
            relative = str(p.relative_to(root)); z.writestr(relative, raw)
            hashes[relative] = hashlib.sha256(raw).hexdigest()
    if pending.stat().st_size > 30*1024**2:
        raise ValueError('Compressed handoff exceeds 30 MiB; original reports retained')
    pending.replace(archive)
    trial.write(dest/'phase3_package.json', dict(status=state['status'], method=state['method'],
        bytes=archive.stat().st_size, archive_sha256=trial.file_sha(archive), files=hashes))
    print('[handoff] '+str(dest), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--method', choices=('base', 'single_frame'), required=True)
    p.add_argument('--index', type=Path, help='Optional historical index; omitted builds/reuses the fixed common dataset')
    p.add_argument('--common-data-dir', type=Path, default=trial.APP/'checkpoints/phase3_common_v1')
    p.add_argument('--collection-dir', type=Path, default=trial.APP/'keyframe_filter/collection_output')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--data-root', type=Path, default=trial.APP/'lead_data')
    p.add_argument('--model-dir', type=Path, default=trial.APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--output', type=Path, required=True, help='Base directory; creates a fixed run_METHOD below it')
    p.add_argument('--run-tag', default=None, help='Explicit semantic name for a repeat/resume; no timestamps or overwrite')
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--build-data', action='store_true', help='A only: derive labels once from the frozen index')
    choice.add_argument('--dataset', type=Path, help='A only: reuse explicit current-source dataset.json')
    p.add_argument('--resume', type=Path, help='Complete epoch 1/2; new output, same dataset and source contract')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--gpu-ids', default=os.environ.get('GPU_IDS', ''))
    p.add_argument('--timeout-seconds', type=int, default=172800)
    p.add_argument('--reserve-gib', type=int, default=20)
    return p


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    a = parser().parse_args()
    if (a.method == 'base' and (a.build_data or a.dataset)) or (a.method == 'single_frame' and a.index is not None and not (a.build_data or a.dataset)):
        raise ValueError('base uses the original index; single_frame requires --build-data or --dataset')
    if a.resume and (a.build_data or (a.method=='single_frame' and not a.dataset)):
        raise ValueError('Resume requires the original bound --dataset, never rebuild it')
    if a.timeout_seconds <= 0 or a.reserve_gib < 20:
        raise ValueError('Positive timeout and >=20 GiB reserve required')
    automatic_index = a.index is None
    if automatic_index:
        a.common_data_dir = a.common_data_dir.resolve()
        a.index = a.common_data_dir/'native/frame_index.jsonl'
        if a.resume and not (a.common_data_dir/'common_receipt.json').is_file():
            raise ValueError('Resume requires the existing bound common dataset; no automatic generation')
    else:
        a.index = a.index.resolve(strict=True)
    for key in ('data_root', 'model_dir'):
        setattr(a, key, getattr(a, key).resolve(strict=True))
    if a.dataset: a.dataset = a.dataset.resolve(strict=True)
    if a.resume: a.resume = a.resume.resolve(strict=True)
    base_dir = a.output.resolve()
    protected = [a.common_data_dir if automatic_index else a.index.parent, a.data_root, a.model_dir]
    if a.dataset: protected.append(a.dataset.parent)
    if a.resume: protected.append(a.resume.parents[2])
    if any(base_dir.is_relative_to(p) or p.is_relative_to(base_dir) for p in protected):
        raise ValueError('Output base overlaps input or previous run; use a separate directory')
    tag = a.run_tag or ('base_independent_v1' if a.method=='base' else 'single_frame_nearest_v1')
    if a.prepare_only: tag += '_plan'
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag): raise ValueError('Invalid run tag')
    latest = base_dir/'latest'
    if latest.exists() and not latest.is_symlink():
        raise ValueError('latest is not a symlink; preserve it and choose another output base')
    root = base_dir/('run_'+tag)
    root.mkdir(parents=True, exist_ok=False)
    state = dict(method=a.method, status='preflight', output=str(root), hostname=socket.gethostname(),
        server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'), stages=[],
        launch_sources=launch_sources(), old_models_evaluated=False, scheme_two_executed=False)
    print('[run] '+str(root), flush=True)
    code = 0; lock_fd = None; contexts = ExitStack(); reference = None
    try:
        if a.resume:
            lock_fd = os.open(a.resume, os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(root).free < (a.reserve_gib+1)*1024**3:
            raise ValueError('Insufficient space above the disk reserve')
        release = trial.verified_sources()
        from qwen3vl_local.sft_new_loop_phase3.preflight import check_index, check_model
        check_model(a.model_dir)
        ids = '' if a.prepare_only else trial.gpu_ids(a.gpu_ids)
        env = environment(ids)
        env['HF_HOME'] = str(base_dir/'.hf_cache')
        snapshot = model_snapshot(a.model_dir)
        def run(name, cmd, log=None):
            nonlocal code
            if launch_sources() != state['launch_sources'] or trial.verified_sources() != release:
                raise ValueError('Launch sources changed during the run')
            check_model_snapshot(a.model_dir, snapshot)
            state.update(active_stage=name); trial.write(root/'launch_workflow.json', state)
            result = trial.run_stage(cmd, log or root/(name+'.log'), env,
                timeout=a.timeout_seconds, reserve_bytes=a.reserve_gib*1024**3)
            state['stages'].append(dict(stage=name, **result))
            trial.write(root/'launch_workflow.json', state)
            if result['exit_code']:
                code = result['exit_code']; raise RuntimeError(f'{name} exited {code}; see {result["log"]}')
        if automatic_index:
            from . import phase3_common_data as common, phase3_common_runtime as common_runtime
            a.index, reference = common.ensure(a.common_data_dir,a.data_root,a.collection_dir,run,
                workers=a.workers,single_frame=a.method=='single_frame' and not a.dataset)
            contexts.enter_context(common_runtime.activate(reference,a.data_root))
            state['common_data'] = dict(root=str(a.common_data_dir),reference_sha256=trial.file_sha(reference),
                historical_scores_comparable=trial.reference().get('historical_scores_comparable',True))
            if a.method=='single_frame' and not a.dataset:
                a.dataset=a.common_data_dir/'single_frame/dataset.json'; a.build_data=False
        if trial.file_sha(a.index) != trial.reference()['index_sha256']:
            raise ValueError('Explicit index differs from frozen reference; omit --index for a new common dataset')
        check_index(a.index, action_output_mode='choice', prompt_variant=trial.reference()['prompt_variant'])
        def worker_command(cmd):
            return common_runtime.wrap(cmd,reference,a.data_root) if reference else cmd
        exp = root/'experiment'
        if a.method == 'base':
            job = trial.build_job(SimpleNamespace(index=a.index, data_root=a.data_root, model_dir=a.model_dir,
                resume=a.resume, stage='complete', experiment='base'), exp)
            job['model_snapshot'] = snapshot
            exp.mkdir(); trial.write(exp/'job.json', job)
        else:
            data_path = a.dataset
            if a.build_data:
                run('derive_data', command('qwen3vl_local.audit_joint.phase3_single_frame', 'build',
                    '--index', a.index, '--data-root', a.data_root, '--output', root/'data'))
                data_path = root/'data/dataset.json'
            cmd = command('qwen3vl_local.audit_joint.phase3_single_frame.worker', 'prepare',
                '--index', a.index, '--data-root', a.data_root, '--model-dir', a.model_dir,
                '--dataset', data_path, '--output', exp)
            if a.resume: cmd += ['--resume', str(a.resume)]
            run('prepare', worker_command(cmd))
            job = trial.read(exp/'single_frame_job.json')
        if reference:
            trial.write(exp/'common_reference.json',trial.reference())
            trial.write(exp/'common_receipt.json',trial.read(reference.parent/'common_receipt.json'))
        for name, cmd in schedule(a.method, root, prepare_only=a.prepare_only):
            run(name, worker_command(cmd))
            if name == 'train':
                state['training'] = trial.verify_training(job)
        state.update(status='prepared_cpu_no_training' if a.prepare_only else 'completed_quality_pending', active_stage=None)
        if not a.prepare_only:
            pending = base_dir/('.latest_'+tag)
            # A concurrent completed run may update a symlink, never a real directory.
            if latest.exists() and not latest.is_symlink():
                state['latest_warning']='latest became a non-symlink; completed run retained at explicit path'
            else:
                pending.symlink_to(root.name); pending.replace(latest)
    except BaseException as exc:
        state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        print(state['error'], file=sys.stderr, flush=True); code = code or 2
    finally:
        contexts.close()
        if lock_fd is not None: os.close(lock_fd)
        try: package(root, state)
        except Exception as exc:
            print('Handoff failed: '+str(exc), file=sys.stderr); code = code or 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
