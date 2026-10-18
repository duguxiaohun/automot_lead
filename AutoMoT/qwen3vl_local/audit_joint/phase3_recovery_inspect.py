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
import signal
import socket
import subprocess
import sys
import time
import zipfile


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


def checkpoint_metadata(root, folder, verify):
    """Reject malformed receipts and compare exact members without torch.load."""
    receipt_path = folder/'receipt.json'
    receipt = read_json(receipt_path)
    binding = receipt.get('binding')
    members = receipt.get('files')
    item = dict(path=str(folder), pending=folder.name.endswith('.pending'),
                receipt_error=receipt.get('inspection_error'), policy=receipt.get('policy'),
                source_differences=[], missing_files=[], extra_files=[],
                hashes_checked=False, file_hash_errors=[], metadata_errors=[])
    if not isinstance(binding, dict) or not isinstance(binding.get('sources'), dict) or not isinstance(members, dict):
        item['metadata_errors'].append('receipt binding/sources/files must be objects')
        return item
    if receipt.get('policy') != 'phase3_epoch_training_state_v1':
        item['metadata_errors'].append('unknown checkpoint policy')
    if 'training_state.pt' not in members:
        item['metadata_errors'].append('training_state.pt not bound by receipt')
    before = digest(receipt_path)
    item.update(configuration=binding.get('configuration'), steps_per_epoch=binding.get('steps_per_epoch'),
                world_size=binding.get('world_size'), receipt_sha256=before)
    for name, expected in binding['sources'].items():
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
    actual_members = set()
    for directory, dirs, files in os.walk(folder, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            if (base/name).is_symlink():
                item['metadata_errors'].append('symlink directory: '+str((base/name).relative_to(folder)))
                dirs.remove(name)
        for name in files:
            path = base/name
            if path != receipt_path:
                actual_members.add(str(path.relative_to(folder)))
    item['extra_files'] = sorted(actual_members-set(members))
    item['hashes_checked'] = verify
    for name, expected in members.items():
        relative = Path(name)
        path = folder/relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(folder.resolve()):
            item['file_hash_errors'].append({'path': name, 'error': 'outside checkpoint'})
            continue
        if path.is_symlink():
            item['file_hash_errors'].append({'path': name, 'error': 'symlink checkpoint member'})
        elif not path.is_file():
            item['missing_files'].append(name)
        elif verify:
            try:
                if digest(path) != expected:
                    item['file_hash_errors'].append({'path': name, 'error': 'SHA mismatch'})
            except (OSError, ValueError) as exc:
                item['file_hash_errors'].append({'path': name, 'error': str(exc)})
    if digest(receipt_path) != before:
        item['metadata_errors'].append('receipt changed during inspection')
    item['validation_scope'] = 'No tensor loading; production resume still must validate data, base assets, configuration, optimizer and RNG.'
    return item


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
    pointer_path = result['latest_pointer'].get('checkpoint')
    if isinstance(pointer_path, str) and pointer_path:
        target = Path(pointer_path)
        result['protected_paths'].append(str((target if target.is_absolute() else root/target).resolve()))
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
    # Metadata-only checkpoint candidates, not inferred training success.
    result['adapter_slots'] = {}
    for slot in ('best_generation', 'final', 'fallback_generation'):
        path = run/slot/'sft_new_loop_phase3_adapter_config.json'
        if path.is_file():
            config = read_json(path)
            result['adapter_slots'][slot] = {k: config.get(k) for k in
                ('global_step', 'num_epochs', 'history_rgb_mode', 'action_output_mode', 'saved_at')}
    result['evaluation_metrics'] = {}
    for mode in ('base_production', 'lora_production', 'lora_audit'):
        path = run.parent/'eval'/mode/'metrics.json'
        if path.is_file():
            result['evaluation_metrics'][mode] = read_json(path)
    for folder in sorted((run/'training_states').glob('epoch_*')):
        if folder.is_symlink() or not folder.is_dir():
            continue
        try:
            result['checkpoints'].append(checkpoint_metadata(root, folder, verify))
        except (OSError, ValueError, TypeError) as exc:
            result['checkpoints'].append(dict(path=str(folder), inspection_error=str(exc)))
    for path in (run/'train.log', run.parent/'pipeline.log'):
        if path.is_file():
            result.setdefault('log_tails', {})[str(path)] = tail(path, 96000)
    return result


def process_inventory(root):
    """Read relevant PIDs/cmdlines only; never inspect environment or send signals."""
    rows = []
    errors = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry/'cmdline').read_bytes()[:16384]
            args = raw.decode('utf-8', errors='replace').split('\0')
            if any('phase3_recovery_inspect.py' in arg for arg in args):
                continue
            if not any('sft_new_loop_phase3' in arg or 'torchrun' in arg for arg in args):
                continue
            cwd = os.readlink(entry/'cwd')
            if str(root) not in cwd and not any(str(root) in arg for arg in args):
                continue
            rows.append(dict(pid=int(entry.name), cwd=cwd, argv=args[:40]))
        except OSError:
            errors += 1
    return dict(processes=rows, inaccessible_or_exited=errors,
                scope='Best-effort local process snapshot, not an absence-of-writers guarantee.')


def bounded_run(root, run, verify=False, timeout=120):
    """Isolate slow/missing mounts; kill only this collector's child session."""
    command = [sys.executable, '-S', '-B', str(Path(__file__).resolve()), '--root', str(root), '--run', str(run)]
    if verify:
        command.append('--verify-checkpoint-files')
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return dict(requested_run=str(run), inspection_error='inspection timeout', timeout_seconds=timeout)
    if proc.returncode:
        return dict(requested_run=str(run), inspection_error='inspection process failed',
                    exit_code=proc.returncode, stderr=stderr[-12000:])
    try:
        return json.loads(stdout)
    except ValueError:
        return dict(requested_run=str(run), inspection_error='invalid inspection output', stderr=stderr[-12000:])


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
            if path.is_dir() and not path.is_symlink() and ('pipeline' in path.name or path.name.endswith('_runs')):
                children = sorted(path.iterdir())
                result.setdefault('run_directory_entries', []).extend(dict(path=str(p),
                    symlink=p.is_symlink(), deletion_status='unreviewed_keep') for p in children[:1000])
                if len(children) > 1000:
                    result.setdefault('run_directory_listing_truncated', []).append(str(path))
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


def package_inspection(root, runs, output, *, verify=False, disk_usage=False, timeout=120):
    """Write only a fresh diagnostics directory, even when some inputs fail."""
    root = root.absolute()
    output = output.absolute()
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('diagnostic output must be outside AutoMoT; use a fresh /tmp directory')
    for run in runs:
        path = run if run.is_absolute() else root/run
        if output.resolve().is_relative_to(path.resolve()):
            raise ValueError('diagnostic output overlaps a requested run')
    if output.exists() or output.is_symlink():
        raise FileExistsError('refusing existing diagnostic output: '+str(output))
    output.mkdir(parents=True)
    started = time.time()
    report = dict(schema='phase3_recovery_handoff_v2', hostname=socket.gethostname(),
        server_label=os.environ.get('AUTOMOT_SERVER', 'unspecified'), project_root=str(root),
        requested_runs=[str(p) for p in runs], runs=[], errors=[],
        protected_paths=[str((p if p.is_absolute() else root/p).resolve()) for p in runs],
        resume_authorized=False, deletion_candidates=[], deletion_inventory_complete=False,
        collector_sha256=digest(Path(__file__)), collected_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        scope='Metadata/log inspection only. No model, dataset, source, checkpoint or process mutations.')
    for number, run in enumerate(runs, 1):
        print(f'[inspect {number}/{len(runs)}] {run}; limit={timeout}s', file=sys.stderr, flush=True)
        try:
            partial = bounded_run(root, run, verify, timeout)
        except (OSError, ValueError) as exc:
            partial = {'requested_run': str(run), 'inspection_error': str(exc)}
        if 'inspection_error' in partial:
            report['errors'].append(partial)
            continue
        report['runs'].extend(partial.get('runs', []))
        for key in ('current_git_head', 'disk', 'checkpoint_entries', 'discovery_errors',
                    'run_directory_entries', 'run_directory_listing_truncated'):
            if key in partial:
                report[key] = partial[key]
    for run in report['runs']:
        report['protected_paths'].extend(run.get('protected_paths', []))
        if run.get('manifest_error'):
            report['errors'].append(dict(run=run['run'], inspection_error=run['manifest_error']))
    report['protected_paths'] = sorted(set(report['protected_paths']))
    for entry in report.get('checkpoint_entries', []) + report.get('run_directory_entries', []):
        path = Path(entry['path']).resolve()
        entry['overlaps_protected_paths'] = any(path.is_relative_to(Path(p)) or Path(p).is_relative_to(path)
            for p in report['protected_paths'])
    try:
        report['process_snapshot'] = process_inventory(root)
    except OSError as exc:
        report['process_snapshot'] = {'inspection_error': str(exc)}
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        report['gpu_processes'] = dict(exit_code=gpu.returncode, stdout=gpu.stdout[-12000:], stderr=gpu.stderr[-2000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['gpu_processes'] = {'inspection_error': str(exc)}
    if disk_usage:
        print('[inspect storage] read-only du; limit=60s', file=sys.stderr, flush=True)
        report['directory_disk_usage'] = disk_inventory(report.get('checkpoint_entries', []))
    report['elapsed_seconds'] = round(time.time()-started, 3)
    report['status'] = 'collected_with_errors' if report['errors'] else 'collected_for_review'
    data = (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode()
    if len(data) > 24*1024**2:
        # Keep the full local report instead of silently trimming evidence.
        (output/'phase3_recovery_report.json').write_bytes(data)
        raise ValueError('report exceeds 24 MiB; full local JSON preserved, no success package')
    (output/'phase3_recovery_report.json').write_bytes(data)
    archive = output/'phase3_recovery_report.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('phase3_recovery_report.json', data)
    receipt = dict(schema='phase3_recovery_package_v1', status=report['status'],
        hostname=report['hostname'], project_root=str(root), collector_sha256=report['collector_sha256'],
        archive=archive.name, archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
        members={'phase3_recovery_report.json': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}},
        contents='Metadata, bounded log tails, source hashes, process/disk observations; no weights, RGB or dataset rows.',
        resume_authorized=False, deletion_authorized=False)
    (output/'phase3_recovery_package.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], output=str(output),
        feedback_files=[str(archive), str(output/'phase3_recovery_package.json')]), ensure_ascii=False), flush=True)
    return 2 if report['errors'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help="This machine's AutoMoT root")
    parser.add_argument('--run', type=Path, action='append', help='Explicit train or pipeline folder; repeat for protected runs')
    parser.add_argument('--disk-usage', action='store_true', help='Read Phase3/4 directory sizes, up to 60 seconds; never delete')
    parser.add_argument('--verify-checkpoint-files', action='store_true', help='Read receipt-listed checkpoint bytes for SHA; does not certify resumability')
    parser.add_argument('--output', type=Path, help='Fresh diagnostics directory outside AutoMoT; writes report ZIP and SHA receipt')
    parser.add_argument('--run-timeout', type=int, default=120, help='Per-run diagnostic child limit with --output, in seconds')
    args = parser.parse_args()
    if args.verify_checkpoint_files and args.run is None:
        parser.error('--verify-checkpoint-files requires an explicit --run')
    if not 1 <= args.run_timeout <= 1800:
        parser.error('--run-timeout must be between 1 and 1800 seconds')
    if args.output:
        if not args.run:
            parser.error('--output requires explicit --run paths')
        return package_inspection(args.root, args.run, args.output, verify=args.verify_checkpoint_files,
                                  disk_usage=args.disk_usage, timeout=args.run_timeout)
    runs = args.run or [None]
    report = collect(args.root, runs[0], args.verify_checkpoint_files)
    for run in runs[1:]:
        report['runs'].extend(collect(args.root, run, args.verify_checkpoint_files)['runs'])
    if args.disk_usage:
        report['directory_disk_usage'] = disk_inventory(report['checkpoint_entries'])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
