"""Fresh Phase3 CPU build, native label checks and event-equal sampling handoff."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .io import digest, file_sha, write_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'qwen3vl_local/sft_new_loop_phase3'
RELEASE = Path(__file__).with_name('release_sources_phase3_prepare_20261012.json')
SEED = 20260904
JSON_NAMES = ('source_check.json', 'prepare_result.json', 'input_bindings.json',
              'index_preflight.json', 'sampling_plans.json', 'temporal_slices.json',
              *(f'{name}_{mode}.json' for name in ('audit_rebuilt_index', 'audit_raw_index')
                for mode in ('binary', 'choice')))


def sampling_plans(index, data_root):
    """Same native samplers/seeds as train.sh defaults and explicit test seed."""
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
    from qwen3vl_local.sft_new_loop_phase3.evaluation_balance import select_work
    from qwen3vl_local.sft_new_loop_phase3.history_rgb import history_rgb_contract
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    from qwen3vl_local.sft_new_loop_phase3.prompt_contract import action_prompt_sha256
    from .training_scope import phase3_work_balance

    def summarize(work, evaluation=False):
        report = phase3_work_balance(work)
        ids = [case_identity(item) for item in work]
        if not report['equal_presentations']:
            raise ValueError('native sampling lacks ten equal events')
        if evaluation and len({digest(i) for i in ids}) != len(ids):
            raise ValueError('evaluation repeats question identities')
        report.update(ordered_cases_sha256=digest(ids), total_cases=len(work))
        return report

    rows = {s: train._read_rows(Path(index), s, data_root=Path(data_root))
            for s in ('train', 'val', 'test')}
    modes = {}
    for mode in ('binary', 'choice'):
        work = train._balanced_work(rows['train'], target_per_bin=1024, seed=SEED,
                                   route_diverse=True, action_output_mode=mode)
        report = {'train': summarize(work)}
        if any(c['presentations'] != 1024 for c in report['train']['events'].values()):
            raise ValueError('train quota differs from 1024 per event')
        # Verify native DDP partition reconstructs the global sequence exactly.
        shards = [train._split_work_for_rank(work, rank=r, world_size=4) for r in range(4)]
        rebuilt = [shards[i % 4][i // 4] for i in range(len(work))]
        if [case_identity(i) for i in rebuilt] != [case_identity(i) for i in work]:
            raise ValueError('four-rank sampling order mismatch')
        report['world1_world4_order_equal'] = True
        for name, quota, seed in (('val_loss', 16, SEED + 1009), ('val_generation', 32, SEED)):
            selected, plan = train._validation_work(rows['val'], target_per_bin=quota,
                seed=seed, action_output_mode=mode, equal_events=True)
            report[name] = dict(distribution=summarize(selected, True), plan=plan)
        natural = evaluate._balanced_cases(rows['test'], cases_per_bin=0, seed=SEED,
                                          action_output_mode=mode)
        selected, plan = select_work(natural, seed=SEED)
        report['test'] = dict(distribution=summarize(selected, True), plan=plan)
        modes[mode] = report
    return dict(status='event_equal', seed=SEED, modes=modes, prompt_variant=CANDIDATE_NAME,
                prompt_sha256={m: {v: action_prompt_sha256(prompt_variant=CANDIDATE_NAME,
                    action_output_mode=m, history_rgb_mode=v) for v in ('4rgb', '2rgb_short')}
                    for m in modes},
                views={m: history_rgb_contract(m) for m in ('4rgb', '2rgb_short')},
                scope='CPU native plans shared by both RGB views; INVALID reported separately; '
                      'no RGB decoding, training, GPU throughput or label-accuracy certification')


def execute(output, name, argv):
    print(f'[phase3-prepare] {name}', flush=True)
    with (output / f'{name}.log').open('w') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'[phase3-prepare] running; log={output / (name + ".log")}', flush=True)
    if code:
        raise RuntimeError(f'{name} exited {code}; see {name}.log')


def build(output, data_root, collection_dir, workers):
    from .server_readiness import source_check
    from .server_phase3_index import inspect
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding
    result = dict(status='blocked', training_started=False, gpu_used=False)
    try:
        sources = source_check(ROOT, RELEASE)
        write_json(output / 'source_check.json', sources)
        if sources['status'] != 'matched':
            raise ValueError('source mismatch; synchronize this release')
        if workers < 1:
            raise ValueError('workers must be positive')
        if not data_root.is_dir():
            raise ValueError(f'data root missing: {data_root}')
        collection = sorted(collection_dir.glob('*_result.json'))
        if not collection:
            raise ValueError(f'no formal collection results: {collection_dir}; do not substitute a review cache')
        index = output / 'data/frame_index.jsonl'
        if index.parent.exists():
            raise ValueError('data directory already exists; use a fresh output')
        inputs = dict(collection=[binding(p) for p in collection],
                      data_root=dict(logical_path=str(data_root), resolved_path=str(data_root.resolve())))
        write_json(output / 'input_bindings.json', inputs)
        def command(name, *args):
            return [sys.executable, '-m', 'qwen3vl_local.sft_new_loop_phase3.' + name, *map(str, args)]
        execute(output, 'build', command('build_dataset', '--collection-dir', collection_dir,
            '--data-root', data_root, '--output-dir', index.parent,
            '--review-root', output / 'no_local_review',
            '--development-groups-extra', SOURCE / 'development_route_groups_rgb_stage_20261006.json',
            '--workers', workers, '--split-seed', 20260920, '--test-ratio', .10, '--val-ratio', .05,
            '--require-invalid-true-rs-coverage'))
        index_bindings = {k: binding(p) for k, p in (
            ('index', index), ('manifest', index.with_name('manifest.json')))}
        result['bindings'] = index_bindings
        admission, _ = inspect(index, CANDIDATE_NAME, 'choice')
        write_json(output / 'index_preflight.json', admission)
        if admission['status'] != 'index_admitted':
            raise ValueError(admission.get('error', 'index admission failed'))
        for mode in ('binary', 'choice'):
            for name in ('audit_rebuilt_index', 'audit_raw_index'):
                argv = command(name, '--index', index, '--data-root', data_root,
                    '--action-output-mode', mode, '--output', output / f'{name}_{mode}.json')
                if name == 'audit_raw_index':
                    argv += ['--workers', str(workers)]
                execute(output, f'{name}_{mode}', argv)
        execute(output, 'temporal_slices', command('audit_temporal_slices', '--index', index,
                                                  '--output', output / 'temporal_slices.json'))
        write_json(output / 'sampling_plans.json', sampling_plans(index, data_root))
        for item in [*inputs['collection'], *index_bindings.values()]:
            check_binding(item)
        if sorted(collection_dir.glob('*_result.json')) != collection:
            raise ValueError('collection input set changed during build')
        if str(data_root.resolve()) != inputs['data_root']['resolved_path']:
            raise ValueError('data root redirected during build')
        if source_check(ROOT, RELEASE)['status'] != 'matched':
            raise ValueError('source changed during build')
        result['status'] = 'cpu_data_prepared'
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['scope'] = 'Phase3 automatic index and CPU admission; first-epoch budget/GPU and joint isolation remain separate'
    write_json(output / 'prepare_result.json', result)
    return 0 if result['status'] == 'cpu_data_prepared' else 2


def package(output, exit_code):
    from .server_pilot import package as pack
    members, errors = {}, []
    paths = [(n, output / n) for n in JSON_NAMES]
    paths.append(('manifest.json', output / 'data/manifest.json'))
    for name, path in paths:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 2_000_000:
                raise ValueError('diagnostic exceeds 2MB; retained on server')
            members[name] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    for name in ('prepare', 'build', 'temporal_slices', *(f'{n}_{m}'
                 for n in ('audit_rebuilt_index', 'audit_raw_index') for m in ('binary', 'choice'))):
        path = output / f'{name}.log'
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 60_000))
                members[f'{name}_log_tail.json'] = dict(sha256=file_sha(path),
                    bytes=path.stat().st_size, tail=stream.read().decode(errors='replace'))
    success = (exit_code == 0 and not errors
               and members.get('prepare_result.json', {}).get('status') == 'cpu_data_prepared'
               and all(n in members for n in JSON_NAMES) and 'manifest.json' in members)
    members['handoff.json'] = dict(status='cpu_data_prepared' if success else 'incomplete',
        command_exit_code=exit_code, errors=errors, training_started=False)
    pack(output, members)
    (output / 'pilot_report.zip').rename(output / 'phase3_prepare_report.zip')
    (output / 'pilot_package.json').rename(output / 'phase3_prepare_package.json')
    return 0 if success else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('build', 'package'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('lead_data'))
    p.add_argument('--collection-dir', type=Path, default=Path('keyframe_filter/collection_output'))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--exit-code', type=int, default=2)
    a = p.parse_args()
    output = a.output.absolute()
    return (build(output, a.data_root.absolute(), a.collection_dir.absolute(), a.workers)
            if a.command == 'build' else package(output, a.exit_code))


if __name__ == '__main__':
    raise SystemExit(main())
