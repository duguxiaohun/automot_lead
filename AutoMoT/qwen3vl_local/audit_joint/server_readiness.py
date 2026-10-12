"""Read-only server inventory and bounded handoff; no build, download or training.

Legacy metadata is reported as unverified declarations, never loaded through a
new producer contract. Checkpoint weights and row indexes are not copied/read.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from .io import file_sha

LIMIT = 30_000_000
METADATA_LIMIT = 2_000_000
LOG_LIMIT = 2_000_000
NAMES = {'manifest.json', 'pipeline_ready.json', 'view.json', 'student_view.json', 'references.json',
         'build_request.json', 'sft_new_loop_phase3_adapter_config.json',
         'effective_pool_audit.manifest.json', 'effective_pool_manifest.json', 'candidates.json', 'index.json'}
MARKERS = {'frame_index.jsonl', 'train.jsonl', 'val.jsonl', 'test.jsonl',
           'adapter_model.safetensors', 'effective_pool_audit.jsonl'}
FIELDS = ('counts', 'rgb_mode', 'trainable', 'complete_coverage', 'policy',
          'prompt_variant', 'observation_contract', 'production_index_sha256',
          'complete', 'complete_causal_production', 'processed_routes', 'candidate_routes')
LOG_NAMES = {'tests.log', 'preflight.log', 'commands.json'}
PYTHON_ROOTS = ('qwen3vl_local/audit_joint', 'qwen3vl_local/sft_new_loop_phase3',
                'qwen3vl_local/sft_new_loop_phase4', 'qwen3vl_local/qwen35')


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def scan(root, *, exclude=None, max_records=2000):
    root = Path(root).absolute()
    if not root.is_dir():
        return dict(status='missing', root=str(root), records=[], errors=[])
    rows, errors = [], []
    def error(exc):
        if len(errors) < 20:
            errors.append(str(exc))
    truncated = False
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=error):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__', '.cache')
                         and (exclude is None or (parent/d).absolute() != Path(exclude).absolute()))
        for name in sorted(files):
            if name not in NAMES | MARKERS:
                continue
            if len(rows) >= max_records:
                truncated = True
                break
            path = parent/name
            record = dict(logical_path=str(path), metadata_verified=False)
            try:
                record.update(resolved_path=str(path.resolve(strict=True)), bytes=path.stat().st_size)
                if path.is_symlink():
                    record['status'] = 'symlink_not_read'
                elif name in MARKERS:
                    record['status'] = 'exists_contents_not_read'
                elif record['bytes'] > METADATA_LIMIT:
                    record['status'] = 'oversized_metadata_not_read'
                else:
                    raw = path.read_bytes()
                    record['sha256'] = hashlib.sha256(raw).hexdigest()
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ValueError('metadata root is not an object')
                    # Bounded summaries: do not copy full contracts, rows or arbitrary fields.
                    record['declared'] = {k:v for k in FIELDS if k in data
                                          and len(json.dumps(v := data[k])) <= 16000}
                    contract = data.get('contract', {})
                    record['declared_task'] = contract.get('task') if isinstance(contract, dict) else None
                    routes = data.get('routes')
                    if isinstance(routes, list):
                        record['declared_route_count'] = len(routes)
                        record['declared_route_splits'] = dict(Counter(str(r.get('split', 'unknown'))
                            for r in routes if isinstance(r, dict)))
                    record['status'] = 'parsed_unverified_metadata'
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record.update(status='unreadable_or_invalid', error=str(exc))
            rows.append(record)
        if truncated:
            break
    return dict(status='incomplete' if truncated or errors else 'inventoried', root=str(root),
                records=rows, errors=errors, truncated=truncated,
                scope='declared metadata and file presence only; no dataset/input/weight certification; directory symlinks not followed')


def source_check(app, expected):
    spec = json.loads(Path(expected).read_text())
    rows = []
    for name, sha in spec['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe expected source path')
        path = Path(app)/relative
        try:
            actual = file_sha(path)
            status = 'matched' if actual == sha else 'changed'
        except OSError:
            actual, status = None, 'missing'
        rows.append(dict(path=name, expected=sha, actual=actual, status=status))
    extras, errors = [], []
    # Overlay synchronization deliberately preserves local files. Report them;
    # run_tests selects published tests and rejects imports from unlisted files.
    roots = [name for name in PYTHON_ROOTS if any(p.startswith(name+'/') for p in spec['files'])]
    for name in roots:
        for directory, dirs, files in os.walk(Path(app)/name, followlinks=False,
                onerror=lambda exc: errors.append(str(exc))):
            dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', 'probe_output', '.git'))
            for filename in sorted(files):
                path = Path(directory)/filename
                relative = path.relative_to(app).as_posix()
                if path.suffix in ('.py', '.pyc') and relative not in spec['files']:
                    extras.append(relative)
    return dict(status='matched' if rows and all(r['status']=='matched' for r in rows) and not errors else 'mismatch',
                expected_manifest_sha256=file_sha(expected), files=rows,
                unexpected_python_files=sorted(extras), scan_errors=errors,
                python_scan_roots=roots,
                scope='release file hashes plus unexpected Python files in declared runtime packages; probe_output/cache excluded, directory links not followed; not authentication of Git remote or all external imports')


def run_tests(app, expected):
    """Run published CPU tests in an overlay checkout without discovering local tests.

    Additional source files are preserved, but must not supply imported modules
    inside the four checked runtime packages. External dependencies retain the
    existing manifest's scope; this is not a complete environment certificate.
    """
    import importlib.abc
    import importlib.machinery
    app = Path(app).absolute()
    report = source_check(app, expected)
    if report['status'] != 'matched':
        print(json.dumps(dict(stage='published_tests', status='source_mismatch')))
        return 2
    allowed = {r['path'] for r in report['files']}
    roots = report['python_scan_roots']
    test_roots = ('qwen3vl_local/audit_joint/tests/',
                  'qwen3vl_local/sft_new_loop_phase4/tests/',
                  'qwen3vl_local/sft_new_loop_phase3/')
    tests = sorted(p for p in allowed if p.startswith(test_roots)
                   and Path(p).name.startswith('test_') and p.endswith('.py'))
    if not tests:
        raise ValueError('release manifest contains no CPU tests')

    def check_origin(origin, fullname=''):
        if not origin or origin in ('built-in', 'frozen'):
            return
        path = Path(os.path.abspath(origin))
        try:
            name = path.relative_to(app).as_posix()
        except ValueError:
            if any(fullname == root.replace('/', '.') or
                   fullname.startswith(root.replace('/', '.') + '.') for root in roots):
                raise ImportError('checked runtime module loaded outside application: ' + fullname)
            return
        if any(name.startswith(root + '/') for root in roots) and name not in allowed:
            raise ImportError('unlisted local runtime module: ' + name)

    class PublishedImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                check_origin(spec.origin, fullname)
            return None  # Normal import machinery and pytest rewriting still run.

    for name, module in tuple(sys.modules.items()):
        check_origin(getattr(module, '__file__', None), name)
    # Do not let retained pytest.ini/conftest, user addopts or entrypoint plugins
    # expand the release's test scope. These tests require no conftest/plugin.
    os.environ.pop('PYTEST_ADDOPTS', None)
    os.environ.pop('PYTEST_PLUGINS', None)
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    sys.path.insert(0, str(app))
    guard = PublishedImports()
    sys.meta_path.insert(0, guard)
    print(json.dumps(dict(stage='published_tests', policy='manifest_tests_guarded_imports_v1',
        test_files=tests, extra_files_preserved=report['unexpected_python_files']), ensure_ascii=False), flush=True)
    try:
        import pytest
        code = int(pytest.main(['-q', '-rs', '-c', os.devnull, '--noconftest',
                               '--rootdir', str(app), *[str(app/p) for p in tests]]))
        for name, module in tuple(sys.modules.items()):
            check_origin(getattr(module, '__file__', None), name)
        if source_check(app, expected)['status'] != 'matched':
            raise ValueError('release sources changed during tests')
        return code
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(str(app))


def collect(checkpoints, output, *, app, expected, logs=()):
    output = Path(output).absolute()
    package = output.with_name(output.name + '.zip')
    if output.exists() or package.exists():
        raise FileExistsError('use a fresh output and ZIP name')
    # Explicit log allowlist: never accept a checkpoint/weights directory.
    paths = [Path(p) for p in logs]
    if len({p.name for p in paths}) != len(paths) or any(p.name not in LOG_NAMES for p in paths):
        raise ValueError('logs must have unique names: tests.log, preflight.log, commands.json')
    output.mkdir(parents=True)
    report = scan(checkpoints, exclude=output)
    source = source_check(app, expected)
    disk = shutil.disk_usage(output)
    write(output/'inventory.json', report)
    write(output/'source_check.json', source)
    log_records = []
    for p in paths:
        item = dict(path=str(p.absolute()), name=p.name)
        try:
            size = p.stat().st_size
            with p.open('rb') as stream:
                stream.seek(max(0, size-LOG_LIMIT))
                raw = stream.read(LOG_LIMIT)
            (output/p.name).write_bytes(raw)
            item.update(status='tail_only' if size > LOG_LIMIT else 'copied', source_bytes=size,
                        captured_bytes=len(raw), truncated=size>LOG_LIMIT)
        except OSError as exc:
            item.update(status='missing_or_unreadable', error=str(exc))
        log_records.append(item)
    write(output/'summary.json', dict(status='diagnostic_only', training_ready=None,
        source_status=source['status'], inventory_status=report['status'],
        free_bytes=disk.free, total_bytes=disk.total, logs=log_records,
        scope='CPU inventory, not G0 or training admission; no data build/training/delete performed'))
    files = {p.name:dict(bytes=p.stat().st_size, sha256=file_sha(p)) for p in output.iterdir() if p.is_file()}
    write(output/'receipt.json', dict(schema='server_readiness_v1', files=files))
    if sum(p.stat().st_size for p in output.iterdir()) > LIMIT-100000:
        raise ValueError('diagnostics exceed 30 MB; no ZIP emitted')
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            z.write(p, p.name)
    if package.stat().st_size > LIMIT:
        raise ValueError('ZIP exceeds 30 MB')
    verify(package)
    return dict(package=str(package), bytes=package.stat().st_size, sha256=file_sha(package),
                verified=True, source_status=source['status'], training_ready=None)


def verify(path):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('oversized ZIP')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > LIMIT:
            raise ValueError('duplicate/oversized ZIP entries')
        if any(Path(n).name != n or '\\' in n for n in names):
            raise ValueError('invalid ZIP names')
        receipt = json.loads(z.read('receipt.json'))
        if receipt['schema'] != 'server_readiness_v1' or set(names) != set(receipt['files']) | {'receipt.json'}:
            raise ValueError('ZIP file set mismatch')
        for name, record in receipt['files'].items():
            raw = z.read(name)
            if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('ZIP content mismatch: ' + name)
    return dict(status='verified', files=len(names), scope='package byte integrity only')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    p.add_argument('--output', type=Path)
    p.add_argument('--log', action='append', type=Path, default=[])
    p.add_argument('--verify', type=Path)
    p.add_argument('--run-tests', action='store_true', help='Published CPU tests only; preserve overlay files')
    a = p.parse_args()
    if a.run_tests:
        app = Path(__file__).resolve().parents[2]
        raise SystemExit(run_tests(app, Path(__file__).with_name('release_sources_v50_20261011.json')))
    if a.verify:
        result = verify(a.verify)
    else:
        if a.output is None:
            p.error('--output required unless --verify')
        app = Path(__file__).resolve().parents[2]
        result = collect(a.checkpoints, a.output, app=app,
            expected=Path(__file__).with_name('release_sources_v50_20261011.json'), logs=a.log)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
