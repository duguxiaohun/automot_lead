"""Continue an explicitly selected local build at its earliest missing stage.

No latest-directory search, cross-server lookup, native replay or GPU training.
Existing files are evidence to validate, not evidence of successful admission.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .local_paths import check_inputs

APP = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).with_name('release_sources_v88_resume_lifecycle_20261019.json')


def metadata(path):
    if not path.is_file() or path.stat().st_size > 64_000_000:
        raise ValueError(f'Metadata missing, not a file or oversized: {path}')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Metadata must be an object: {path}')
    return value


def plan(prepared, output, data_root, model_dir):
    prepared, output, data_root, model_dir = [Path(p).absolute() for p in
                                             (prepared, output, data_root, model_dir)]
    if not prepared.is_dir():
        raise ValueError(f'Existing local build directory missing: {prepared}')
    errors = check_inputs(data_root, model_dir, output)['errors']
    if (output.resolve().is_relative_to(prepared.resolve()) or
            prepared.resolve().is_relative_to(output.resolve())):
        errors.append('Output and existing build must be separate')
    if errors:
        raise ValueError('; '.join(errors))
    native = prepared/'data4/manifest.json'
    refs = prepared/'references4/references.json'
    view = prepared/'view/student_view.json'
    # Never downgrade a corrupt/present asset into a missing one and rebuild it.
    for p in (native, refs, view):
        if p.exists() or p.is_symlink():
            metadata(p)
    env = dict(DATA_ROOT=str(data_root), MODEL_DIR=str(model_dir),
               OUTPUT_DIR=str(output), PYTHON=sys.executable)
    if native.is_file() and refs.is_file():
        mode = 'admission_only' if view.is_file() else 'complete_view_then_admission'
        cmd = [sys.executable, '-m', 'qwen3vl_local.audit_joint.phase4_admission_only',
               '--prepared', str(prepared), '--output', str(output),
               '--data-root', str(data_root), '--model-dir', str(model_dir)]
        if not view.is_file():
            cmd.append('--complete-view')
    else:
        metadata(prepared/'inputs/candidates.json')
        if not (prepared/'inputs/production').is_dir():
            raise ValueError('Existing production directory missing; no automatic replay fallback')
        mode = 'verify_complete_replay_then_finish_build'
        cmd = ['bash', str(APP/'qwen3vl_local/audit_joint/server_phase4_existing.sh')]
        env['RESUME_FROM'] = str(prepared)
    return dict(mode=mode, prepared=str(prepared), output=str(output),
                command=cmd, environment=env, training=False, recompute_routes=False,
                scope='stage routing only; child verifies contracts, complete replay and event-equal admission')


def main():
    from .server_readiness import source_check
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=APP/'lead_data')
    p.add_argument('--model-dir', type=Path, default=APP/'checkpoints/Qwen3.5-4B')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    try:
        # Directory lock coordinates this entry without creating or deleting files.
        fd = os.open(a.prepared, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selected = plan(a.prepared, a.output, a.data_root, a.model_dir)
            sources = source_check(APP, RELEASE)
            if sources['status'] != 'matched':
                raise ValueError('Published source mismatch; synchronize before continuing')
            print(json.dumps(dict(resume_plan=selected, source_status=sources['status']),
                             ensure_ascii=False), flush=True)
            if a.plan_only:
                return 0
            env = dict(os.environ, **selected['environment'])
            # Avoid inherited thread oversubscription during CPU-only admission.
            env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            return subprocess.call(selected['command'], cwd=APP, env=env)
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        print(json.dumps(dict(status='blocked', training=False,
                              error=f'{type(exc).__name__}: {exc}')), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
