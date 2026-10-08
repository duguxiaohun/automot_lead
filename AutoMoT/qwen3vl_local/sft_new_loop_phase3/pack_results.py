"""Package completed Phase3 evaluations without a model, GPU, or new inference."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {(r, a) for r in ('4rgb', '2rgb_endpoints') for a in ('binary', 'choice')}
TEXT = {'.json', '.jsonl', '.md', '.txt', '.csv'}
IMAGES = {'.jpg', '.jpeg', '.png'}
CONFIG = 'sft_new_loop_phase3_adapter_config.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def eval_root(path):
    path = Path(path).expanduser().resolve()
    return path/'eval' if (path/'eval/lora_production').is_dir() else path


def inspect_eval(directory):
    metrics = read(directory/'metrics.json')
    case_files = sorted(directory.glob('cases*.jsonl'))
    rows = []
    for path in case_files:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows or len(rows) != metrics.get('total_cases'):
        raise ValueError(f'{directory}: incomplete/duplicate case files: {len(rows)} rows, metrics={metrics.get("total_cases")}')
    ids = [row['case_index'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{directory}: duplicate case_index')
    group = (metrics['history_rgb_mode'], metrics['action_output_mode'])
    if group not in GROUPS:
        raise ValueError(f'{directory}: unknown group {group}')
    for row in rows:
        if type(row.get('all_ok')) is not bool:
            raise ValueError(f'{directory}: missing all_ok in case {row.get("case_index")}')
        if row.get('history_rgb_mode') != group[0] or row.get('prompt_spec', {}).get('action_output_mode') != group[1]:
            raise ValueError(f'{directory}: mixed case input/output modes')
    return metrics, dict(cases=len(rows), errors=sum(not row['all_ok'] for row in rows),
                        successes=sum(row['all_ok'] for row in rows))


def discover(root):
    """Choose the most recently finished pipeline per mode; report the exact selection."""
    selected = {}
    for manifest in sorted(root.glob('*/pipeline_manifest.json')):
        state = read(manifest)
        if not state.get('status', '').startswith('completed'):
            continue
        directory = manifest.parent/'eval'
        metric_file = directory/'lora_production/metrics.json'
        if not metric_file.is_file():
            continue
        metrics = read(metric_file)
        group = (metrics.get('history_rgb_mode'), metrics.get('action_output_mode'))
        if group not in GROUPS:
            continue
        key = (metric_file.stat().st_mtime_ns, str(directory))
        if group not in selected or key > selected[group][0]:
            selected[group] = (key, directory)
    missing = GROUPS - selected.keys()
    if missing:
        raise ValueError(f'Missing completed groups under {root}: {sorted(missing)}. '
                         'Pass explicit pipeline/eval directories to package available runs.')
    return [selected[group][1] for group in sorted(GROUPS)]


def package(directory, destination):
    directory = eval_root(directory)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing archive: {destination}')
    files, reports = {}, {}
    group = None
    adapter = None
    for name in ('lora_production', 'base_production', 'lora_audit'):
        source = directory/name
        if not source.exists() and name != 'lora_production':
            continue
        metrics, report = inspect_eval(source)
        actual = (metrics['history_rgb_mode'], metrics['action_output_mode'])
        if group is not None and actual != group:
            raise ValueError(f'{source}: group differs from LoRA production')
        group = actual
        if name == 'lora_production':
            adapter = metrics.get('adapter_dir')
        reports[name] = report
        # All JSONL rows and saved RGB are included; no 500-line or example-count truncation.
        for path in sorted(source.rglob('*')):
            if path.is_file() and path.suffix.lower() in TEXT | IMAGES:
                files[f'{name}/{path.relative_to(source).as_posix()}'] = path
        report['saved_error_case_records'] = len(list((source/'error_cases').rglob('case.json')))
        report['saved_error_rgb_files'] = sum(p.is_file() and p.suffix.lower() in IMAGES
                                             for p in (source/'error_cases').rglob('*'))
    for name in ('paired_base_lora.json', 'visual_audit_manifest.json'):
        if (directory/name).is_file(): files[name] = directory/name
    if (directory.parent/'pipeline_manifest.json').is_file():
        files['pipeline_manifest.json'] = directory.parent/'pipeline_manifest.json'
    if adapter:
        adapter_path = Path(adapter)
        if not adapter_path.is_absolute(): adapter_path = ROOT/adapter_path
        for name in (CONFIG, 'adapter_config.json'):
            if (adapter_path/name).is_file(): files[f'adapter_metadata/{name}'] = adapter_path/name
        if (adapter_path.parent/'train_run_manifest.json').is_file():
            files['adapter_metadata/train_run_manifest.json'] = adapter_path.parent/'train_run_manifest.json'
    identities = {name:dict(bytes=p.stat().st_size, sha256=digest(p)) for name,p in files.items()}
    manifest = dict(schema='phase3_posthoc_package_v1', source_eval=str(directory), group=list(group),
        evaluations=reports, files=identities, inference_performed=False, weights_included=False,
        rgb_policy='All already-saved evaluation RGB; success RGB may be absent. No new RGB copied or visual review performed.',
        packaging_source_sha256=digest(Path(__file__)))
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed package leaves the source untouched and never publishes a partial archive.
    with tempfile.TemporaryDirectory(prefix='.pack-', dir=destination.parent) as scratch:
        temp = Path(scratch)/'result.tar.gz'
        with tarfile.open(temp, 'w:gz', dereference=True) as archive:
            for name, path in files.items():
                archive.add(path, arcname=f'audit_bundle/{name}', recursive=False)
            item = tarfile.TarInfo('audit_bundle/bundle_manifest.json'); item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
        if any(digest(path) != identities[name]['sha256'] for name,path in files.items()):
            raise ValueError('Evaluation files changed while packaging; wait for evaluation to finish')
        # Atomic no-overwrite publication on the same filesystem.
        import os
        os.link(temp, destination)
    return dict(archive=str(destination), bytes=destination.stat().st_size,
                group=list(group), evaluations=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('directories', nargs='*', type=Path, help='pipeline run or eval directories; default: latest completed run per mode')
    parser.add_argument('--pipeline-root', type=Path, default=ROOT/'checkpoints/sft_new_loop_phase3_pipeline')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'checkpoints/phase3_audit_exports')
    parser.add_argument('--receipt', type=Path, help='write an explicit source/archive mapping; must not exist')
    args = parser.parse_args()
    directories = [eval_root(p) for p in args.directories] if args.directories else discover(args.pipeline_root)
    if len(set(directories)) != len(directories): parser.error('duplicate input directories')
    # Validate every selected result before starting the first archive.
    for directory in directories:
        metrics, _ = inspect_eval(directory/'lora_production')
        group = (metrics['history_rgb_mode'], metrics['action_output_mode'])
        for name in ('base_production', 'lora_audit'):
            if (directory/name).exists():
                other, _ = inspect_eval(directory/name)
                if (other['history_rgb_mode'], other['action_output_mode']) != group:
                    raise ValueError(f'{directory/name}: mode differs from LoRA production')
        print(f'[pack-results] group={group[0]}_{group[1]} source={directory}', flush=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    receipt = args.receipt or args.output_dir/f'packaging_{stamp}.json'
    if receipt.exists(): raise FileExistsError(f'Refusing existing receipt: {receipt}')
    reports = []
    for number, directory in enumerate(directories, 1):
        metrics = read(directory/'lora_production/metrics.json')
        tag = f'{metrics["history_rgb_mode"]}_{metrics["action_output_mode"]}'
        output = args.output_dir/f'sft_new_loop_phase3_{stamp}_{number}_{tag}_audit_bundle.tar.gz'
        report = package(directory, output)
        report['source_eval'] = str(directory)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(dict(status='completed', all_four_groups={tuple(r['group']) for r in reports} == GROUPS,
                       packages=reports), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'[pack-results] completed {len(reports)} package(s); receipt={receipt}', flush=True)


if __name__ == '__main__':
    main()
