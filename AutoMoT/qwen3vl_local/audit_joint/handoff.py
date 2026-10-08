"""Bounded, offline audit handoff. Capture receipts and old results stay immutable."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from .baseline import verify_bundle
from .io import file_sha

LIMIT = 30_000_000
SCHEMA = 'joint_audit_handoff_v1'
REPORTS = ('request.json', 'baseline_manifest.json', 'source_manifest.json',
           'split_cross_audit.json', 'exposure_ledger.jsonl', 'receipt.json')
# Explicit result names, never a recursive "all JSON" copy of a checkpoint tree.
RESULT_NAMES = {
    'run.json', 'train_run_manifest.json', 'pipeline_manifest.json', 'pipeline_ready.json',
    'metrics.json', 'data_admission.json', 'best.json', 'latest.json', 'best_val.json',
    'best_generation.json', 'fallback_generation.json', 'generation_selection_status.json',
    'final_generation.json', 'phase4_contract.json', 'selection.json', 'adapter_config.json',
    'sft_new_loop_phase3_adapter_config.json', 'qwen35_backend.json', 'qwen35_base_assets.json',
    'paired_base_lora.json', 'visual_audit_manifest.json', 'paired_eval.json',
    'training_plan.json', 'benchmark_report.json', 'preflight.json', 'host_preflight.json',
    'run_manifest.json', 'model_contract.json', 'condition_contract.json', 'train_log.jsonl',
    'train_metrics.jsonl', 'train_eval_metrics.jsonl', 'train_balance.json',
    'generation_val_cases.jsonl', 'final_generation_val_cases.jsonl',
    'route_results.csv', 'scenario_results.csv', 'ability_results.csv',
}
RESULT_PATTERN = re.compile(r'(cases(?:_rank\d+)?\.jsonl|epoch_\d+_(sampling|validation)\.json|'
                            r'epoch_\d+_cases\.jsonl|epoch_\d+\.json)\Z')


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError(f'unsafe archive path: {name}')
    return name


def result_files(root):
    included, omitted = {}, []
    def unreadable(error):
        raise error
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=unreadable):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if path.is_symlink():
                omitted.append(dict(path=path.relative_to(root).as_posix(), reason='symlink_directory'))
                dirs.remove(name)
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                omitted.append(dict(path=relative, reason='symlink'))
            elif name in RESULT_NAMES or RESULT_PATTERN.fullmatch(name):
                included[safe_name(relative)] = path
            else:
                omitted.append(dict(path=relative, reason='outside_result_allowlist'))
    return included, sorted(omitted, key=lambda item: item['path'])


def pack(capture, output=None, results=(), max_bytes=LIMIT):
    if not 4096 <= max_bytes <= LIMIT:
        raise ValueError('archive budget must be 4096..30,000,000 bytes')
    capture = Path(capture).resolve()
    output = Path(output or capture.with_name(capture.name + '.audit.zip')).absolute()
    if output.resolve().is_relative_to(capture):
        raise ValueError('archive must be outside immutable capture directory')
    if os.path.lexists(output):
        raise FileExistsError(output)
    verify_bundle(capture)
    baseline = json.loads((capture / 'baseline_manifest.json').read_text())
    sources = json.loads((capture / 'source_manifest.json').read_text())
    files = {f'g0/{name}': capture / name for name in REPORTS}
    expected = {name: file_sha(path) for name, path in files.items()}
    omitted = []

    def snapshot(name, record):
        safe_name(name)
        path = capture / safe_name(record['blob'])
        if not path.resolve().is_relative_to(capture) or file_sha(path) != record['sha256']:
            raise ValueError(f'snapshot mismatch: {name}')
        files[name] = path
        expected[name] = record['sha256']

    for name, record in sources.items():
        snapshot('source/' + safe_name(name), record)
    for number, record in enumerate(baseline['artifacts']):
        if 'blob' not in record:
            if record.get('storage') == 'external_reference':
                omitted.append(dict(kind='artifact', id=record['id'], sha256=record['sha256'],
                                    bytes=record['bytes'], reason='external_reference_not_copied'))
            continue
        # Only small configuration/manifest evidence, never adapter weight blobs.
        if Path(record['path']).suffix == '.json':
            snapshot(f'artifacts/{number:04d}/' + Path(record['path']).name, record)
        else:
            omitted.append(dict(kind='artifact', id=record['id'], sha256=record['sha256'],
                                bytes=record['bytes'], reason='not_json_metadata'))
    result_inventory = {}
    roots = []
    for label, directory in results:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in result_inventory:
            raise ValueError('result labels must be unique letters/digits/underscore/hyphen')
        root = Path(directory).resolve()
        if not root.is_dir() or root in roots:
            raise ValueError(f'missing or duplicate result directory: {root}')
        if output.resolve().is_relative_to(root):
            raise ValueError('archive must be outside result directories')
        selected, excluded = result_files(root)
        if not selected:
            raise ValueError(f'no recognized audit results: {root}')
        roots.append(root)
        result_inventory[label] = dict(root=str(root), included=sorted(selected), omitted=excluded,
                                       semantic_validation='not_performed; inspect original run status and coverage')
        for name, path in selected.items():
            key = f'results/{label}/{name}'
            files[key] = path
            expected[key] = file_sha(path)

    manifest = dict(schema=SCHEMA, max_bytes=max_bytes, capture=str(capture),
                    capture_receipt_sha256=expected['g0/receipt.json'],
                    baseline_ready=baseline['reproducible_baseline_ready'],
                    storage=baseline.get('storage', {'mode': 'legacy_full'}),
                    external_assets_required=True,
                    blockers=baseline['blockers'], stages=baseline['stages'],
                    results=result_inventory, omitted_artifacts=omitted,
                    excluded_categories=['raw RGB/video', 'model/optimizer weights', 'full dataset/index blobs'],
                    source_bytes_included=True, files={},
                    scope='Complete G0 reports and selected result files; not the full local snapshot. '
                          'No new training, evaluation, visual review or stage approval is performed.',
                    packer_sha256=file_sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.audit-pack-', dir=output.parent) as scratch:
        pending = Path(scratch) / 'audit.zip'
        # Source bytes are useful but optional. Every report/result remains complete.
        for include_sources in (True, False):
            manifest['source_bytes_included'] = include_sources
            manifest['source_omission_reason'] = None if include_sources else 'compressed_size_budget; full source SHA manifest retained'
            manifest['files'] = {}
            with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for name, path in sorted(files.items()):
                    if not include_sources and name.startswith('source/'):
                        continue
                    sha = hashlib.sha256()
                    size = 0
                    with path.open('rb') as stream, archive.open(name, 'w', force_zip64=True) as entry:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            sha.update(chunk)
                            size += len(chunk)
                            entry.write(chunk)
                    if sha.hexdigest() != expected[name] or file_sha(path) != expected[name]:
                        raise ValueError(f'audit input changed during packaging: {name}')
                    manifest['files'][name] = dict(sha256=sha.hexdigest(), bytes=size)
                archive.writestr('handoff_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            if pending.stat().st_size <= max_bytes:
                break
        else:
            raise ValueError('complete reports/results exceed 30 MB budget; no archive published, '
                             'no cases/metrics truncated; package separate explicitly named runs')
        # Check that a running training/eval job did not append new result files.
        for label, record in result_inventory.items():
            selected, excluded = result_files(Path(record['root']))
            if sorted(selected) != record['included'] or excluded != record['omitted']:
                raise ValueError(f'result inventory changed during packaging: {label}')
        for name, path in files.items():
            if file_sha(path) != expected[name]:
                raise ValueError(f'audit input changed during packaging: {name}')
        verify_package(pending)
        os.link(pending, output)  # Atomic no-clobber publication, even with concurrent packers.
    return dict(archive=str(output), bytes=output.stat().st_size, sha256=file_sha(output),
                source_bytes_included=manifest['source_bytes_included'],
                baseline_ready=manifest['baseline_ready'], stages=manifest['stages'])


def verify_package(path):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('archive exceeds 30,000,000 bytes')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        for name in names:
            safe_name(name)
        manifest = json.loads(archive.read('handoff_manifest.json'))
        if manifest.get('schema') != SCHEMA:
            raise ValueError('unknown handoff schema')
        if not 4096 <= manifest['max_bytes'] <= LIMIT or path.stat().st_size > manifest['max_bytes']:
            raise ValueError('archive exceeds declared budget')
        if set(names) != set(manifest['files']) | {'handoff_manifest.json'}:
            raise ValueError('handoff inventory changed')
        for name, record in manifest['files'].items():
            sha, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                    size += len(chunk)
            if size != record['bytes'] or sha.hexdigest() != record['sha256']:
                raise ValueError(f'handoff hash mismatch: {name}')
    return dict(status='verified', files=len(manifest['files']), bytes=path.stat().st_size,
                scope='archive bytes only, not experiment approval')
