"""Storage accounting and external references; never delete existing artifacts."""
import json
import heapq
import os
from collections import Counter
from pathlib import Path
import shutil

from .io import file_sha, snapshot_file, source_files

GIB = 1024 ** 3
METADATA_LIMIT = 2 * 1024 ** 2


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def copied_artifact(path, mode):
    return mode == 'full' or (path.suffix == '.json' and path.stat().st_size <= METADATA_LIMIT)


def plan(config, output, mode='references', min_free_bytes=2 * GIB):
    if mode not in ('references', 'full') or min_free_bytes < 0:
        raise ValueError('invalid storage mode or free-space reserve')
    copies = set(source_files(config['source_roots']))
    references = set()
    for spec in config['artifacts']:
        path = Path(spec['path']).absolute()
        if path.is_file():
            (copies if copied_artifact(path, mode) else references).add(path)
    for spec in config['split_sources']:
        path = Path(spec['path']).absolute()
        if path.is_file():
            (copies if mode == 'full' else references).add(path)
    copy_bytes = sum(p.stat().st_size for p in copies)
    reference_bytes = sum(p.stat().st_size for p in references - copies)
    parent = existing_parent(output)
    free = shutil.disk_usage(parent).free
    # Planning allowance, not a promise about arbitrary input/report sizes.
    allowance = 128 * 1024 ** 2
    required = copy_bytes + allowance + min_free_bytes
    return dict(mode=mode, filesystem_path=str(parent), free_bytes=free,
                copy_bytes_upper_bound=copy_bytes, referenced_input_bytes=reference_bytes,
                report_and_zip_allowance_bytes=allowance, min_free_bytes=min_free_bytes,
                required_free_bytes=required, sufficient=free >= required,
                full_pipeline_build_included=False,
                scope='G0 copy estimate only; not a filesystem quota or a full-build size prediction.')


def capture_input(path, output, *, copy):
    if copy:
        return dict(snapshot_file(path, output), storage='snapshot')
    path = Path(path).resolve()
    return dict(sha256=file_sha(path), bytes=path.stat().st_size,
                resolved_path=str(path), storage='external_reference',
                note='Original must remain at this path for re-verification; bytes are not frozen in this capture.')


def verify_inputs(capture):
    from .baseline import verify_bundle
    verify_bundle(capture)
    root = Path(capture)
    baseline = json.loads((root / 'baseline_manifest.json').read_text())
    splits = json.loads((root / 'split_cross_audit.json').read_text())
    records = baseline['artifacts'] + [r.get('snapshot', {}) for r in splits['sources']]
    results = []
    for item in records:
        if item.get('storage') != 'external_reference':
            continue
        path = Path(item['resolved_path'])
        try:
            valid = path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256']
            result = dict(path=str(path), status='verified' if valid else 'changed')
        except (OSError, RuntimeError) as error:
            result = dict(path=str(path), status='unavailable', error=str(error))
        results.append(result)
    model = baseline.get('model', {})
    model_report = dict(status='unavailable_at_capture', checked_files=0, files=[], added=[], missing=[])
    if model.get('status') == 'hashed' and model.get('assets'):
        from qwen3vl_local.qwen35.identity import base_asset_hashes
        directory = Path(model['path'])
        expected = model['assets']
        try:
            current = base_asset_hashes(directory)
            model_report.update(added=sorted(current.keys() - expected.keys()),
                                missing=sorted(expected.keys() - current.keys()))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            current = {}
            model_report['inventory_error'] = str(error)
        for name, sha in expected.items():
            path = directory / name
            try:
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('unsafe recorded model asset path')
                actual = current[name] if name in current else file_sha(path)
                row = dict(path=str(path), status='verified' if actual == sha else 'changed')
            except (OSError, ValueError, RuntimeError) as error:
                row = dict(path=str(path), status='unavailable', error=str(error))
                if name not in model_report['missing']:
                    model_report['missing'].append(name)
            model_report['files'].append(row)
        model_report['checked_files'] = len(expected)
        model_report['status'] = ('verified' if not model_report.get('inventory_error')
                                  and not model_report['added'] and not model_report['missing']
                                  and all(r['status'] == 'verified' for r in model_report['files']) else 'failed')
    failed = any(r['status'] != 'verified' for r in results) or model_report['status'] == 'failed'
    return dict(status='failed' if failed else ('verified' if model_report['status'] == 'verified' else 'incomplete'),
                references=results, model=model_report, checked_files=len(results) + model_report['checked_files'],
                scope='External input and recorded base assets, including asset-set drift; no experiment approval.')


def inventory(root, core_dir=None, top=20):
    """Metadata-only inventory; inspect just ELF headers of core-named files."""
    root = Path(root).resolve()
    if not root.is_dir() or top < 1:
        raise ValueError('existing directory and positive top count required')
    totals, largest, cores, errors = Counter(), [], [], []
    file_count = core_count = 0
    def failure(error):
        if len(errors) < 20:
            errors.append(str(error))
    def examine(path, account):
        nonlocal file_count, core_count
        try:
            if path.is_symlink() or not path.is_file():
                return
            stat = path.stat()
            allocated = stat.st_blocks * 512
            if account:
                file_count += 1
                relative = path.relative_to(root)
                totals[relative.parts[0] if len(relative.parts) > 1 else '(root files)'] += allocated
                heapq.heappush(largest, (allocated, stat.st_size, str(path)))
                if len(largest) > top:
                    heapq.heappop(largest)
            if path.name == 'core' or path.name.startswith('core.'):
                with path.open('rb') as stream:
                    header = stream.read(18)
                is_core = (len(header) == 18 and header[:4] == b'\x7fELF' and header[5] in (1, 2)
                           and int.from_bytes(header[16:18], 'little' if header[5] == 1 else 'big') == 4)
                if is_core:
                    core_count += 1
                    heapq.heappush(cores, (allocated, stat.st_size, str(path)))
                    if len(cores) > top:
                        heapq.heappop(cores)
        except OSError as error:
            failure(error)
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=failure):
        parent = Path(directory)
        dirs[:] = [name for name in dirs if not (parent / name).is_symlink()]
        for name in names:
            examine(parent / name, True)
    if core_dir is not None:
        directory = Path(core_dir).resolve()
        if not directory.is_relative_to(root):
            for path in directory.iterdir():
                if path.name == 'core' or path.name.startswith('core.'):
                    examine(path, False)
    rows = lambda heap: [dict(path=p, allocated_bytes=a, apparent_bytes=s) for a, s, p in sorted(heap, reverse=True)]
    return dict(root=str(root), free_bytes=shutil.disk_usage(root).free, files=file_count,
                allocated_bytes=sum(totals.values()), directories=dict(totals.most_common()),
                largest_files=rows(largest), confirmed_elf_core_count=core_count,
                largest_elf_cores=rows(cores), read_errors_first_20=errors,
                scope='Read-only; no links followed, no deletion. Hard links counted per path. '
                      'Extra core_dir checks direct children only. Does not account for deleted open files.')
