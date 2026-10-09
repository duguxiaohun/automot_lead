"""Fresh-process adapter reload and full-validation prediction comparison."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from ..identity import digest, file_sha, write_json
from ..dataset import dump_rows
from ..paired_eval import load, case_identity
from .contract import check_contract
from .data import load_dataset
from .evaluate import evaluate_rows
from .model import load_for_inference
from .train import adapter_identity


def compare_cases(expected, actual):
    if expected.keys() != actual.keys() or not expected:
        raise ValueError('reload case set differs or is empty')
    mismatches = []
    for key, row in actual.items():
        before = expected[key]
        if row['paired_identity'] != before['paired_identity']:
            raise ValueError('reload causal input/reference identity differs')
        if row['prediction'] != before['prediction']:
            mismatches.append(key)
    return dict(status='verified' if not mismatches else 'prediction_mismatch',
                count=len(actual), prediction_mismatch_ids=mismatches,
                raw_text_differences=sum(actual[k]['raw'] != expected[k]['raw'] for k in actual))


def run_check(a, output, report, stage):
    stage('run_contract')
    run = json.loads((a.run / 'run.json').read_text())
    check_contract(run['source'])
    report.update(source=run['source'], dataset=run['dataset'])
    stage('checkpoint_identity')
    latest = json.loads((a.run / 'latest.json').read_text())
    epoch = latest['epoch']
    if type(epoch) is not int or epoch < 0 or latest['adapter'] != f'epoch_{epoch:03d}':
        raise ValueError('invalid latest checkpoint identity')
    folder = a.run / latest['adapter']
    report.update(epoch=epoch, adapter=str(folder))
    if json.loads((folder / 'phase4_contract.json').read_text()) != run:
        raise ValueError('checkpoint/run contract differs')
    checkpoint_path = a.run / f'epoch_{epoch:03d}_checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    report['checkpoint_receipt_sha256'] = file_sha(checkpoint_path)
    stage('checkpoint_assets')
    if checkpoint['epoch'] != epoch or checkpoint['adapter'] != latest['adapter'] or checkpoint['files'] != adapter_identity(folder):
        raise ValueError('saved checkpoint assets changed')
    report['checkpoint_assets_verified'] = True
    stage('optimizer_bytes')
    state = folder / 'training_state.pt'
    if checkpoint['training_state'] != dict(bytes=state.stat().st_size, sha256=file_sha(state)):
        raise ValueError('saved optimizer state changed')
    report['optimizer_bytes_verified'] = True
    stage('dataset')
    data, manifest = load_dataset(a.dataset, rgb_mode=run['observation_contract']['rgb_mode'])
    if digest(manifest) != run['dataset']:
        raise ValueError('reload dataset differs')
    stage('expected_cases')
    expected_path = a.run / f'epoch_{epoch:03d}_cases.jsonl'
    expected, _ = load(expected_path)
    if {r['id']: case_identity(r) for r in data['val']} != {k:r['paired_identity'] for k,r in expected.items()}:
        raise ValueError('saved validation cases differ from complete dataset')
    report.update(expected_cases_sha256=file_sha(expected_path), expected_count=len(expected))
    stage('device_selection')
    device = a.device
    if device == 'cuda':
        from ..devices import select_gpus
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = select_gpus(1)
        device = 'cuda:0'
    report['device'] = device
    stage('model_load')
    bundle = load_for_inference(a.model_dir, folder, device)
    stage('validation')
    metrics, cases = evaluate_rows(bundle, data['val'], a.data_root, run['training']['max_length'])
    stage('comparison')
    dump_rows(output / 'cases.jsonl', cases)
    actual, _ = load(output / 'cases.jsonl')
    comparison = compare_cases(expected, actual)
    stage('metrics_write')
    write_json(output / 'metrics.json', metrics)
    report.update(comparison, stage='complete')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('run', 'dataset', 'data-root', 'model-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--output', type=Path, help='Default: RUN/reload_check; existing reports are preserved')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    output = a.output or a.run / 'reload_check'
    # Reserve the destination before validation; never overwrite a prior attempt.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='in_progress', stage='initializing', run=str(a.run),
                  dataset_path=str(a.dataset), model_dir=str(a.model_dir),
                  data_root=str(a.data_root), device=a.device,
                  checkpoint_assets_verified=False, optimizer_bytes_verified=False,
                  scope='fresh adapter load and full val predictions; optimizer bytes checked, optimizer resume not executed')

    def save():
        report['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'checkpoint_reload.json', report)

    def stage(name):
        report['stage'] = name
        save()

    save()
    try:
        run_check(a, output, report, stage)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        save()
        traceback.print_exc()
        print(json.dumps(report, ensure_ascii=False))
        return 130 if isinstance(error, KeyboardInterrupt) else 2
    save()
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status'] == 'verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
