"""Read-only historical pool inventory, not native admission or view migration.

Reads the recorded training JSONL once, retaining only sampling fields. It does
not load RGB, replay teachers, create a cache, or authorize training. In
particular, a matching JSONL SHA does not certify historical labels as current.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from .io import file_sha
from .server_context import context
from .server_readiness import source_check
from .rule_baseline import critical_support
from qwen3vl_local.sft_new_loop_phase4.identity import contract_differences, write_json
from qwen3vl_local.sft_new_loop_phase4.rgb_short.batch_sampling import plan
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS

APP = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).with_name('release_sources_v86_build_recovery_20261018.json')
FIELDS = ('id', 'episode', 'scenario', 'route_id', 'physical_group', 'split',
          'observation', 'edge', 'target', 'slice', 'label_basis')


def signature(value):
    return tuple(getattr(value, k) for k in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def inventory(dataset):
    dataset = Path(dataset).absolute()
    manifest_path = dataset/'manifest.json'
    manifest_sha = file_sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError('manifest changed while reading')
    expected = manifest['files']['train.jsonl']
    path = dataset/'train.jsonl'
    resolved = path.resolve(strict=True)
    rows, groups, counts, ids = [], {}, Counter(), set()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        before = signature(os.fstat(stream.fileno()))
        for number, line in enumerate(stream, 1):
            hasher.update(line)
            row = json.loads(line)
            if row['id'] in ids:
                raise ValueError('duplicate historical question ID')
            ids.add(row['id'])
            if row['split'] != 'train' or row['target'] not in ('YES', 'NO'):
                raise ValueError(f'invalid train row at line {number}')
            event = row['episode']['event']
            if event not in EVENTS:
                raise ValueError('unknown event: '+str(event))
            rows.append({k: row[k] for k in FIELDS})
            counts[event] += 1
            groups.setdefault(event, set()).add(row['physical_group'])
            if number % 25000 == 0:
                print(json.dumps(dict(stage='read_historical_train', rows=number)), flush=True)
        after = signature(os.fstat(stream.fileno()))
    if before != after or after != signature(path.stat()) or path.resolve() != resolved:
        raise ValueError('train input changed while reading')
    if hasher.hexdigest() != expected:
        raise ValueError('historical train SHA mismatch')
    if manifest.get('counts', {}).get('train') != len(rows):
        raise ValueError('manifest/train count mismatch')
    differences = contract_differences(manifest.get('contract'))
    result = dict(manifest_sha256=manifest_sha, train_sha256=expected,
        contract_status='different' if differences else 'matched', contract_differences=differences,
        rows=len(rows), event_counts=dict(counts),
        label_basis_counts=dict(Counter(r['label_basis'] for r in rows)),
        event_groups={e: len(g) for e, g in groups.items()},
        missing_events=sorted(set(EVENTS)-set(counts)), **critical_support(rows),
        scope='Historical native train JSONL only. Before student filtering; no RGB, '
              'production proof, holdout, cross-task isolation or label-quality certification.')
    if result['missing_events']:
        result['sampling_status'] = 'blocked_missing_events'
    else:
        result['sampling_status'] = 'simulated_on_historical_rows'
        result['plans'] = {}
        for mode, stride in ((2, 2), (4, 1)):
            _, audit = plan(rows, epoch=0, world_size=4, batch_size=2,
                            expected_events=EVENTS, anchor_stride=stride)
            # Keep aggregate evidence, not the large per-row history or proofs.
            result['plans'][str(mode)] = {k: audit[k] for k in
                ('policy', 'event_counts', 'ordered_ids_sha256', 'unique_questions',
                 'max_question_repeat', 'action_minimums', 'unsupported_critical_cells')}
            print(json.dumps(dict(stage='historical_sampling_complete', rgb_mode=mode)), flush=True)
    if file_sha(manifest_path) != manifest_sha or signature(path.stat()) != after or path.resolve() != resolved:
        raise ValueError('input changed during sampling')
    return result


def run(dataset, output):
    dataset, output = Path(dataset).absolute(), Path(output).absolute()
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError('output must be outside dataset')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status='blocked', training_authorized=False,
        server_context=context(dataset=dataset, output=output), errors=[])
    try:
        result['sources'] = source_check(APP, RELEASE)
        if result['sources']['status'] != 'matched':
            raise ValueError('release source mismatch')
        result['inventory'] = inventory(dataset)
        result['status'] = 'diagnostic_complete'
        result['next_step'] = ('plan producer update; do not relabel old assets by changing SHA'
            if result['inventory']['contract_status'] == 'different' else
            'native admission still required before new training')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.monotonic()-started
    report = output/'label_release_report.json'
    write_json(report, result)
    archive = output/'label_release_report.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report, report.name)
    if archive.stat().st_size > 30_000_000:
        raise ValueError('diagnostic ZIP exceeds 30 MB; not signed complete')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or hashlib.sha256(bundle.read(report.name)).hexdigest() != file_sha(report):
            raise ValueError('diagnostic ZIP verification failed')
    write_json(output/'label_release_package.json', dict(status='verified',
        report_status=result['status'], training_authorized=False,
        archive=archive.name, bytes=archive.stat().st_size, sha256=file_sha(archive),
        members={report.name: file_sha(report)}))
    print(json.dumps(dict(status=result['status'], training_authorized=False,
                         output=str(output), seconds=result['seconds'])), flush=True)
    return 0 if result['status'] == 'diagnostic_complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='native data4 directory on THIS server')
    parser.add_argument('--output', type=Path, required=True, help='new small diagnostic directory')
    args = parser.parse_args()
    # The real hostname and local paths are recorded by context(). A caller's
    # optional server label must not decide whether this machine may run.
    raise SystemExit(run(args.dataset, args.output))


if __name__ == '__main__':
    main()
