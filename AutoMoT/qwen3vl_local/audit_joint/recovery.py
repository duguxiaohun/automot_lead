"""Read-only Phase4 replay reuse diagnostics. Never build, rename or remove artifacts."""
import fcntl
import json
import os
from pathlib import Path
import shutil

from .io import file_sha


def tree_usage(root):
    root = Path(root)
    result = dict(path=str(root), exists=root.exists(), files=0, logical_bytes=0,
                  allocated_bytes=0, skipped_links=[])
    if not root.exists():
        return result
    def unreadable(error):
        raise error
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=unreadable):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                result['skipped_links'].append(str(path))
                dirs.remove(name)
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                result['skipped_links'].append(str(path))
                continue
            info = path.stat()
            result['files'] += 1
            result['logical_bytes'] += info.st_size
            result['allocated_bytes'] += info.st_blocks * 512
    return result


def check_recovery(base, data_root, annotations, min_free_bytes=10 * 1024 ** 3):
    """Hold an existing native build lock without creating files in the source tree."""
    base = Path(base).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    if min_free_bytes < 0:
        raise ValueError('negative space reserve')
    lock = base / '.build.lock'
    if not lock.exists():
        return _check(base, data_root, annotations, min_free_bytes, 'absent; stop all writers before checking')
    with lock.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise ValueError('Phase4 builder is active; wait for it to finish') from ex
        return _check(base, data_root, annotations, min_free_bytes, 'existing_native_build_lock_held')


def _check(base, data_root, annotations, reserve, lock_status):
    from qwen3vl_local.sft_new_loop_phase4 import full_pipeline as pipeline
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay, teacher_approval
    report = dict(schema='joint_phase4_recovery_check_v1', base=str(base),
                  lock_status=lock_status, checks={}, training_ready=False,
                  scope='read-only reuse diagnostics; no build, RGB revalidation, label approval or training')
    checks = report['checks']
    def check(name, operation):
        try:
            result = operation()
            checks[name] = dict(status='verified', result=result)
            return result
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration, AttributeError) as ex:
            checks[name] = dict(status='invalid', error=f'{type(ex).__name__}: {ex}')
            return None
    def read(path):
        return json.loads(Path(path).read_text())

    def read_request():
        value = read(base / 'build_request.json')
        if not isinstance(value, dict) or not isinstance(value.get('external', {}), dict):
            raise ValueError('invalid build request structure')
        if any(not isinstance(record, dict) for record in value.get('external', {}).values()):
            raise ValueError('invalid external dependency record')
        return value
    request = check('build_request_read', read_request)
    external = (request or {}).get('external', {})
    def dependency(name, fallback):
        value = external.get(name, {}).get('path')
        if value is None:
            return base / fallback
        if not Path(value).is_absolute():
            raise ValueError(f'external {name} has no absolute path')
        return Path(value)
    def request_check():
        if request is None:
            raise ValueError('missing readable build request')
        if not Path(data_root).is_dir():
            raise ValueError('DATA_ROOT missing')
        kwargs = {key: dependency(key, '') for key in external}
        if set(kwargs) - {'candidate_pool', 'production_index', 'teacher_registry'}:
            raise ValueError('unknown external build dependency')
        if 'teacher_registry' in kwargs:
            kwargs['registry'] = kwargs.pop('teacher_registry')
        expected = pipeline.request_identity(data_root, annotations, **kwargs)
        if request != expected:
            raise ValueError('build source/configuration changed; old artifacts must remain unchanged')
        return dict(sha256=file_sha(base / 'build_request.json'))
    check('request_compatibility', request_check)
    pool = None
    def pool_check():
        nonlocal pool
        path = dependency('candidate_pool', 'candidates.json')
        pool = candidate_pool.validate(read(path))
        return dict(path=str(path), sha256=file_sha(path), routes=len(pool['routes']))
    check('candidate_pool', pool_check)
    def replay_check():
        if pool is None:
            raise ValueError('candidate pool failed validation')
        result = teacher_replay.accounting(pool, dependency('production_index', 'full_production/index.json'))
        if not result['complete_causal_production']:
            raise ValueError('incomplete route/frame accounting')
        return result
    check('replay_accounting', replay_check)
    def registry_check():
        path = dependency('teacher_registry', 'weak_registry.json')
        teacher_approval.validate_registry(read(path))
        return dict(path=str(path), sha256=file_sha(path))
    check('teacher_registry', registry_check)
    def ready_check():
        if checks['request_compatibility']['status'] != 'verified':
            raise ValueError('request compatibility not established')
        ready = pipeline.validate_ready(base, request)
        return dict(pairs=ready['pairing']['pairs'], sha256=file_sha(base / 'pipeline_ready.json'))
    check('paired_pipeline_receipt', ready_check)

    report['datasets'] = {}
    rebuild_modes = []
    for mode in (2, 4):
        path = base / f'data{mode}'
        usage = tree_usage(path)
        manifest = path / 'manifest.json'
        usage['manifest_exists'] = manifest.is_file()
        usage['native_next_action'] = ('validate_existing_pair' if manifest.is_file() else
                                       'retain_by_rename_then_recompile' if path.exists() else 'compile_new')
        report['datasets'][f'data{mode}'] = usage
        if not manifest.is_file():
            rebuild_modes.append(mode)
    def production_inventory():
        index = dependency('production_index', 'full_production/index.json')
        usage = tree_usage(index.parent)
        usage['referenced_copy_bytes'] = None
        if checks['replay_accounting']['status'] == 'verified':
            names = {receipt['artifact'] for receipt in read(index)['routes']}
            if any(Path(name).name != name for name in names):
                raise ValueError('route artifact escapes production bundle')
            usage['referenced_copy_bytes'] = index.stat().st_size + sum(
                (index.parent / name).stat().st_size for name in names)
        return usage
    production = check('production_inventory', production_inventory)
    free = shutil.disk_usage(base).free
    copy_bytes = production.get('referenced_copy_bytes') if production else None
    minimum = copy_bytes * len(rebuild_modes) if copy_bytes is not None else None
    report['storage'] = dict(free_bytes=free, reserve_bytes=reserve, production=production,
                             compile_modes=rebuild_modes, additional_production_copy_bytes_lower_bound=minimum,
                             lower_bound_and_reserve_fit=None if minimum is None else free >= minimum + reserve,
                             total_required_bytes=None, sufficient_for_build=None,
                             caveat='lower bound only: retained partial datasets remain; JSONL, metadata, '
                                    'temporary files, skipped symlinks and concurrent disk use are not budgeted')
    report['replay_reusable_under_current_request'] = all(
        checks[name]['status'] == 'verified' for name in
        ('request_compatibility', 'candidate_pool', 'replay_accounting', 'teacher_registry'))
    report['status'] = 'diagnosed' if report['replay_reusable_under_current_request'] else 'blocked'
    return report
