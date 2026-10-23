"""Operational entry for geometry-only a/b/c; independent from six-action eligibility."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

APP = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def selected_gpus(explicit=None, query=subprocess.check_output):
    rows = query(['nvidia-smi', '--query-gpu=index,memory.free',
                  '--format=csv,noheader,nounits'], text=True)
    available = [tuple(map(int, line.split(','))) for line in rows.splitlines() if line.strip()]
    ids = explicit.split(',') if explicit else [str(i) for i, _ in sorted(available, key=lambda p: (-p[1], p[0]))[:4]]
    if (len(ids) != 4 or len(set(ids)) != 4 or not all(i.isdigit() for i in ids)
            or not set(map(int, ids)).issubset({i for i, _ in available})):
        raise ValueError('Matched a/b/c training requires four distinct available GPUs')
    return ','.join(ids)


def ready(app, quota):
    from .geometry_abc.prepare import load_ready
    return load_ready(app/f'checkpoints/geometry_abc_v1/ready_{quota}.json',
                      per_event=quota, world_size=4)


def preflight(app, quota, variants):
    root = app/'checkpoints/geometry_abc_v1'
    for variant in variants:
        for path in (root/f'runs/{variant}_{quota}', root/f'operations/{variant}_{quota}.log'):
            if path.exists():
                raise FileExistsError(f'Existing run/log preserved, no implicit resume: {path}')
    result = ready(app, quota)
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_model
    check_model(app/'checkpoints/Qwen3.5-4B')
    gpu_ids = selected_gpus(os.environ.get('GPU_IDS'))
    return result, gpu_ids


def run_training(app, quota, variants, launch=subprocess.run):
    root = app/'checkpoints/geometry_abc_v1'
    root.mkdir(parents=True, exist_ok=True)
    # One a/b/c launcher per study root; status remains read-only and unlocked.
    with (root/'.abc_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, gpu_ids = preflight(app, quota, variants)
        from .versioned_data import write
        operations = root/'operations'
        operations.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHON=sys.executable, GPU_IDS=gpu_ids, STUDY_PER_EVENT=str(quota))
        # Do not let a caller's custom single-run name collapse a/b/c outputs.
        env.pop('STUDY_RUN_NAME', None)
        print(json.dumps(dict(stage='preflight_passed',variants=variants, gpu_ids=gpu_ids,
            dataset=result['dataset'], steps_per_variant=sum(result['optimizer_steps']))), flush=True)
        for variant in variants:
            log = operations/f'{variant}_{quota}.log'
            status = operations/f'{variant}_{quota}.json'
            command = ['bash', str(app/'qwen3vl_local/audit_joint/geometry_abc/remote.sh'), 'train', variant]
            state = dict(variant=variant, per_event=quota, status='running',
                         dataset=result['dataset'], dataset_sha256=result['dataset_sha256'],
                         gpu_ids=gpu_ids, log=str(log), command=command)
            with log.open('x') as stream:
                write(status, state)
                print(f'[{variant}] train + validation-selected checkpoint test; log={log}', flush=True)
                try:
                    process = launch(command, cwd=app, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    code = process.returncode
                    if code == 0:
                        run = root/f'runs/{variant}_{quota}'
                        completion = read(run/'training_complete.json')
                        read(run/'test/metrics.json')
                        if (completion['binding']['contract']['variant'] != variant
                                or completion['binding']['dataset_sha256'] != result['dataset_sha256']
                                or completion['total_steps'] != sum(result['optimizer_steps'])):
                            raise ValueError('completed_run_does_not_match_preflight')
                    state.update(status='completed' if code == 0 else 'failed', exit_code=code)
                    write(status, state)
                    if code: return code
                except BaseException as exc:
                    write(status, dict(state, status='failed', error=f'{type(exc).__name__}: {exc}'))
                    raise
    return 0


def status_report(app, quota):
    root = app/'checkpoints/geometry_abc_v1'
    pointer = root/f'ready_{quota}.json'
    result = dict(ready=read(pointer) if pointer.is_file() else {'status': 'not_prepared'}, runs={})
    cpu = root/'cpu_status.json'
    if cpu.is_file(): result['cpu'] = read(cpu)
    for variant in ('a', 'b', 'c'):
        run = root/f'runs/{variant}_{quota}'
        operation = root/f'operations/{variant}_{quota}.json'
        epochs = []
        for path in sorted(run.glob('epoch_*.json')):
            r = read(path)['result']
            epochs.append(dict(epoch=path.stem, status=r['status'], steps=r['steps'], loss=r['loss'],
                               elapsed_minutes=round(r['elapsed_seconds']/60, 2)))
        completion = run/'training_complete.json'
        log = root/f'operations/{variant}_{quota}.log'
        last = ''
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 4096))
                lines = stream.read().decode(errors='replace').splitlines()
                last = lines[-1] if lines else ''
        result['runs'][variant] = dict(operation=read(operation) if operation.is_file() else None,
            startup_complete=(run/'startup.json').is_file(), epochs=epochs,
            selected_checkpoint=read(completion)['selected_checkpoint'] if completion.is_file() else None,
            test_complete=(run/'test/metrics.json').is_file(), last_log_line=last)
    # Avoid printing full generation manifests and ordered training IDs.
    result['ready'] = {k: result['ready'][k] for k in ('status', 'error', 'dataset', 'optimizer_steps') if k in result['ready']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('operation', choices=('cpu', 'check', 'train', 'status'))
    parser.add_argument('variant', nargs='?', choices=('a', 'b', 'c', 'all'), default='all')
    parser.add_argument('--per-event', choices=(1024, 1280), type=int, default=1280)
    args = parser.parse_args()
    variants = ['a', 'b', 'c'] if args.variant == 'all' else [args.variant]
    if args.operation == 'status':
        print(json.dumps(status_report(APP, args.per_event), ensure_ascii=False, indent=2)); return 0
    if args.operation == 'cpu':
        return subprocess.run([sys.executable, '-u', '-m',
            'qwen3vl_local.audit_joint.geometry_abc.cpu',
            '--per-event', str(args.per_event)], cwd=APP).returncode
    if args.operation == 'check':
        result, gpus = preflight(APP, args.per_event, variants)
        print(json.dumps(dict(status='preflight_passed', dataset=result['dataset'], gpu_ids=gpus,
            variants=variants, optimizer_steps=result['optimizer_steps'], model_forward_tested=False))); return 0
    return run_training(APP, args.per_event, variants)


if __name__ == '__main__':
    raise SystemExit(main())
