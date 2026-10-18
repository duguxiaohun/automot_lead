"""Read-only Phase3 recovery inventory. No torch, inference, deletion or resume.

Run with this machine's explicit AutoMoT root. Discovery never selects a latest
run, follows directory symlinks, or declares old outputs safe to delete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def read_json(path):
    try:
        if path.stat().st_size > 8 * 1024**2:
            return {'inspection_error': 'metadata exceeds 8 MiB'}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {'inspection_error': 'expected object'}
    except (OSError, ValueError) as exc:
        return {'inspection_error': str(exc)}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after) or fields(after) != fields(path.stat()):
        raise ValueError('file changed during inspection: ' + str(path))
    return h.hexdigest()


def tail(path, limit=12000):
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size-limit))
            data = stream.read(limit)
        if size > limit:
            data = data.partition(b'\n')[2]
        return data.decode('utf-8', errors='replace')
    except OSError as exc:
        return str(exc)


def inspect_run(root, run, verify=False):
    manifest = read_json(run/'train_run_manifest.json')
    keep_keys = ('git', 'history_rgb_mode', 'action_output_mode', 'epoch_recovery_enabled',
                 'index', 'data_root', 'model_dir', 'resume_from', 'total_steps_global',
                 'batch_size', 'world_size', 'sampling_policy', 'run_log')
    result = dict(run=str(run), manifest={k: manifest[k] for k in keep_keys if k in manifest},
                  latest_pointer=read_json(run/'latest_training_state.json'),
                  checkpoints=[], rank_last_records={}, inspection_only=True,
                  resume_authorized=False, delete_authorized=False)
    result['protected_paths'] = [str(run), str(run.parent)]
    for key in ('index', 'data_root', 'model_dir', 'resume_from'):
        if manifest.get(key):
            path = Path(manifest[key])
            path = path if path.is_absolute() else root/path
            if key == 'index':
                path = path.parent
            result['protected_paths'].append(str(path.resolve()))
    result['pipeline_manifest'] = read_json(run.parent/'pipeline_manifest.json')
    result['packaging_manifest'] = read_json(run.parent/'packaging_manifest.json')
    if 'inspection_error' in manifest:
        result['manifest_error'] = manifest['inspection_error']
    for path in sorted(run.glob('epoch_*_rank*_batches.jsonl')):
        lines = tail(path).splitlines()
        result['rank_last_records'][path.name] = lines[-2:]
    for folder in sorted((run/'training_states').glob('epoch_*')):
        if folder.is_symlink() or not folder.is_dir():
            continue
        receipt = read_json(folder/'receipt.json')
        binding = receipt.get('binding', {})
        item = dict(path=str(folder), pending=folder.name.endswith('.pending'),
                    receipt_error=receipt.get('inspection_error'),
                    policy=receipt.get('policy'), configuration=binding.get('configuration'),
                    steps_per_epoch=binding.get('steps_per_epoch'),
                    world_size=binding.get('world_size'), source_differences=[],
                    missing_files=[], hashes_checked=verify, file_hash_errors=[])
        for name, expected in binding.get('sources', {}).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                item['source_differences'].append({'path': name, 'error': 'invalid relative path'})
                continue
            path = root/'qwen3vl_local'/relative
            try:
                actual = digest(path)
            except (OSError, ValueError) as exc:
                actual = str(exc)
            if actual != expected:
                item['source_differences'].append(dict(path=name, expected=expected, actual=actual))
        for name, expected in receipt.get('files', {}).items():
            relative = Path(name)
            path = folder/relative
            if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(folder.resolve()):
                item['file_hash_errors'].append({'path': name, 'error': 'outside checkpoint'})
                continue
            if not path.is_file():
                item['missing_files'].append(name)
            elif verify:
                try:
                    if digest(path) != expected:
                        item['file_hash_errors'].append({'path': name, 'error': 'SHA mismatch'})
                except (OSError, ValueError) as exc:
                    item['file_hash_errors'].append({'path': name, 'error': str(exc)})
        item['validation_scope'] = 'Receipt-listed files only; production resume must validate full file set, tensors, data, base assets, configuration and RNG.'
        result['checkpoints'].append(item)
    for path in (run/'train.log', run.parent/'pipeline.log'):
        if path.is_file():
            result.setdefault('log_tails', {})[str(path)] = tail(path)
    return result


def collect(root, run=None, verify=False, max_dirs=10000):
    root = root.resolve(strict=True)
    cp = root/'checkpoints'
    usage = shutil.disk_usage(cp)
    result = dict(schema='phase3_recovery_inspection_v1', hostname=socket.gethostname(),
                  server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'),
                  project_root=str(root), disk=dict(total=usage.total, used=usage.used, free=usage.free),
                  scope='Read-only; no newest-run selection, tensor loading, training or deletion.',
                  deletion_candidates=[], deletion_inventory_complete=False,
                  checkpoint_entries=[], runs=[], discovery_errors=[])
    try:
        result['current_git_head'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=10,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        result['current_git_head'] = None
    for path in sorted(cp.iterdir()):
        if 'phase3' in path.name.lower() or 'phase4' in path.name.lower():
            result['checkpoint_entries'].append(dict(path=str(path), symlink=path.is_symlink(),
                link_target=os.readlink(path) if path.is_symlink() else None,
                deletion_status='unreviewed_keep', size_status='not_recursively_measured'))
    if run is not None:
        selected = run if run.is_absolute() else root/run
        selected = selected.resolve(strict=True)
        if (selected/'pipeline_manifest.json').is_file() and not (selected/'train_run_manifest.json').is_file():
            configured = read_json(selected/'pipeline_manifest.json').get('run_root')
            if not configured:
                raise ValueError('pipeline has no recorded run_root')
            selected = Path(configured)
            selected = (selected if selected.is_absolute() else root/selected).resolve(strict=True)
        result['runs'].append(inspect_run(root, selected, verify))
    else:
        count = 0
        for directory, dirs, files in os.walk(cp, followlinks=False,
                onerror=lambda exc: result['discovery_errors'].append(str(exc))):
            count += 1
            if count > max_dirs:
                result['discovery_truncated'] = True
                break
            path = Path(directory)
            depth = len(path.relative_to(cp).parts)
            dirs[:] = sorted(d for d in dirs if not (path/d).is_symlink()
                             and depth < 4 and (depth > 0 or 'phase3' in d.lower()))
            if 'train_run_manifest.json' in files:
                result['runs'].append(inspect_run(root, path, verify))
                dirs[:] = []
        result['discovery_depth_limit'] = 4
    return result


def disk_inventory(entries):
    paths = [entry['path'] for entry in entries if not entry['symlink']]
    if not paths:
        return {'status': 'empty'}
    # GNU du reads metadata, does not follow symlinks or cross mounted filesystems.
    # Size is a disk-use estimate, never a proof that a directory can be deleted.
    try:
        value = subprocess.run(['du', '-x', '-s', '-B1', '--files0-from=-'],
            input='\0'.join(paths)+'\0', text=True, capture_output=True, timeout=60)
        return dict(status='measured' if value.returncode == 0 else 'partial',
                    sizes=value.stdout, errors=value.stderr,
                    scope='One filesystem; shared hardlinks counted once. No deletion approved.')
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'incomplete', 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help="This machine's AutoMoT root")
    parser.add_argument('--run', type=Path, action='append', help='Explicit train or pipeline folder; repeat for protected runs')
    parser.add_argument('--disk-usage', action='store_true', help='Read Phase3/4 directory sizes, up to 60 seconds; never delete')
    parser.add_argument('--verify-checkpoint-files', action='store_true', help='Read receipt-listed checkpoint bytes for SHA; does not certify resumability')
    args = parser.parse_args()
    if args.verify_checkpoint_files and args.run is None:
        parser.error('--verify-checkpoint-files requires an explicit --run')
    runs = args.run or [None]
    report = collect(args.root, runs[0], args.verify_checkpoint_files)
    for run in runs[1:]:
        report['runs'].extend(collect(args.root, run, args.verify_checkpoint_files)['runs'])
    if args.disk_usage:
        report['directory_disk_usage'] = disk_inventory(report['checkpoint_entries'])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
