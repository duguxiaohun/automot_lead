"""Two sequential one-epoch choice baselines from an admitted Phase3 index."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from .io import digest, file_sha, write_json

ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).with_name('release_sources_phase3_epoch_recovery_20261013.json')
MODES = ('2rgb_short', '4rgb')
INDEX_SHA = '925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99'


# This recovery is deliberately bound to the handoff actually reviewed locally.
REVIEWED_EPOCH_SHA = '5e28b686dedc62b941161fdbcdc9a22b977f1bb2d59122d9156d0a5a1266172d'
CASE_FIELDS = ('scenario', 'route_id', 'frame_id', 'context_id', 'prompt_spec', 'history_rgb_paths_used',
               'history_rgb_sha256', 'raw_output', 'parsed', 'gt', 'all_ok', 'format_valid', 'strict_format_valid')


def update_state(output, state):
    """Replace only this run's mutable progress; immutable evidence stays exclusive."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output,
                                     prefix='.epoch_state_', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, output / 'epoch_result.json')
    finally:
        tmp.unlink(missing_ok=True)


def report_value(path):
    if path.suffix == '.jsonl':
        return [{k: r[k] for k in CASE_FIELDS if k in r}
                for r in (json.loads(line) for line in path.read_text().splitlines())]
    return json.loads(path.read_text())


def verified_reuse(prior, evidence, accepted):
    """Reuse the reviewed completed short-RGB epoch, never an unfinished optimizer run."""
    import hashlib
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding
    archive, receipt = prior / 'phase3_epoch_report.zip', prior / 'phase3_epoch_package.json'
    r = json.loads(receipt.read_text())
    if file_sha(archive) != REVIEWED_EPOCH_SHA or r['zip_sha256'] != REVIEWED_EPOCH_SHA:
        raise ValueError('reuse requires the reviewed Phase3 epoch ZIP')
    if archive.stat().st_size != r['bytes']:
        raise ValueError('reuse ZIP size mismatch')
    with zipfile.ZipFile(archive) as z:
        if (len(z.namelist()) != len(set(z.namelist())) or z.testzip()
                or sum(i.file_size for i in z.infolist()) > 30_000_000):
            raise ValueError('invalid reuse package')
        if {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} != r['files']:
            raise ValueError('reuse member SHA mismatch')
        members = {n: json.loads(z.read(n)) for n in z.namelist()}
    admission = members['admission.json']
    if admission['prepared'] != evidence or admission['sampling']['modes']['choice'] != accepted['modes']['choice']:
        raise ValueError('reuse preparation or sampling mismatch')
    # Production code must remain identical to the version that trained this adapter.
    for record in members['source_check.json']['files']:
        if record['path'].startswith(('qwen3vl_local/sft_new_loop_phase3/', 'qwen3vl_local/qwen35/')):
            if file_sha(ROOT / record['path']) != record['actual']:
                raise ValueError('reuse training/contract source changed: ' + record['path'])
    checks = [binding(archive), binding(receipt)]
    for n, value in members.items():
        if n.startswith('2rgb_short/train/'):
            path = prior / n
            if report_value(path) != value:
                raise ValueError('reuse training report changed: ' + n)
            checks.append(binding(path))
    assets = members['2rgb_short/adapter_bindings.json']
    adapter = prior / '2rgb_short/train/final'
    if {Path(v['logical_path']) for v in assets} != {p.absolute() for p in adapter.iterdir() if p.is_file()}:
        raise ValueError('reuse adapter asset set/path mismatch')
    checks += assets + members['validation_input_bindings.json']
    for v in checks:
        check_binding(v)
    consumption = members['2rgb_short/train/balance/consumed_epoch_0001.json']
    from qwen3vl_local.sft_new_loop_phase3.train import CONTEXT_IDS
    if consumption['status'] != 'verified' or any(consumption['actual'].get(k) != 1024 for k in CONTEXT_IDS):
        raise ValueError('reuse epoch is incomplete')
    return dict(status='verified', mode='2rgb_short', prior_root=str(prior),
                train_path=str(prior / '2rgb_short/train'), bindings=checks,
                scope='completed adapter reuse; no optimizer resume, no repeat two-RGB training')


def evaluation_argv(index, data_root, model_dir, adapter, output, split, quota):
    # LoRA RGB/output/prompt contracts are read from the persisted adapter config.
    return [sys.executable, '-m', 'qwen3vl_local.sft_new_loop_phase3.eval',
            '--index', str(index), '--data-root', str(data_root), '--model-dir', str(model_dir),
            '--adapter-dir', str(adapter), '--output-dir', str(output),
            '--split', split, '--cases-per-bin', str(quota), '--seed', '20260904',
            '--max-new-tokens', '64', '--evaluation-distribution', 'equal_events',
            '--no-merge-lora', '--no-timestamp-output', '--no-save-error-rgb', '--no-save-all-rgb']


def verified_preparation(prepared):
    """Consume the existing successful handoff; never rebuild its dataset."""
    import hashlib
    from qwen3vl_local.sft_new_loop_phase4.production_storage import check_binding
    receipt = json.loads((prepared / 'phase3_prepare_package.json').read_text())
    archive = prepared / 'phase3_prepare_report.zip'
    if file_sha(archive) != receipt['zip_sha256'] or archive.stat().st_size != receipt['bytes']:
        raise ValueError('prepare ZIP identity mismatch')
    with zipfile.ZipFile(archive) as z:
        if (z.testzip() or len(z.namelist()) != len(set(z.namelist()))
                or sum(i.file_size for i in z.infolist()) > 30_000_000):
            raise ValueError('invalid or oversized preparation package')
        if {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} != receipt['files']:
            raise ValueError('prepare member SHA mismatch')
        result = json.loads(z.read('prepare_result.json'))
        handoff = json.loads(z.read('handoff.json'))
        plans = json.loads(z.read('sampling_plans.json'))
    if result['status'] != 'cpu_data_prepared' or handoff['status'] != 'cpu_data_prepared':
        raise ValueError('prepare did not complete')
    if result['bindings']['index']['sha256'] != INDEX_SHA:
        raise ValueError('index is not the reviewed 20261013 formal pool')
    for value in result['bindings'].values():
        check_binding(value)
    return result, plans


def training_env(index, data_root, model_dir, output, mode):
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    # Shell options must not inherit a smoke cap, a different output, or best-save policy.
    from qwen3vl_local.sft_new_loop_phase3.four_runs import SETTINGS
    env = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'OUTPUT_DIR', 'RUN_LOG', 'MODE',
                'MASTER_PORT', 'CUDA_VISIBLE_DEVICES', 'RUN_TIMESTAMP'):
        env.pop(key, None)
    env.update(SETTINGS)
    env.update(INDEX=str(index), DATA_ROOT=str(data_root), MODEL_DIR=str(model_dir),
        OUTPUT_DIR=str(output), PROMPT_VARIANT=CANDIDATE_NAME, HISTORY_RGB_MODE=mode,
        ACTION_OUTPUT_MODE='choice', NUM_EPOCHS='1', MAX_STEPS='0', MAX_FRAMES='0',
        MAX_EVAL_FRAMES='0', EVAL_SPLIT='val', FOCUS_BALANCE_COUNT='1024',
        SEED='20260904', NPROC_PER_NODE='4', DDP_GPU_COUNT='4',
        SAVE_STEPS='0', SAVE_BEST_VAL='0', SAVE_BEST_GENERATION='0', SAVE_FINAL='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''))
    if env.get('GPU_IDS'):
        ids = env['GPU_IDS'].split(',')
        if len(ids) != 4 or len(set(ids)) != 4 or any(not i.isdigit() for i in ids):
            raise ValueError('this fixed first epoch requires four unique numeric GPU_IDS, or automatic selection')
    return env


def compare_reload(before, after):
    """Compare every saved validation case, reparsing raw output with frozen specs."""
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import spec_from_case
    from qwen3vl_local.sft_new_loop_phase3.prompts import parse_action_output

    def load(path):
        cases = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            gt = {q['output_key']: 'YES' if q['answer'] else 'NO' for q in r['prompt_spec']['questions']}
            spec = spec_from_case(dict(r, gt=gt))
            parsed = parse_action_output(r['raw_output'], spec=spec)
            saved = r['parsed']
            if any(type(v) is not bool and v is not None and v not in ('YES', 'NO', 'INVALID') for v in saved.values()):
                raise ValueError('invalid recorded prediction encoding')
            normalized = {k: v if type(v) is bool or v is None else
                          {'YES': True, 'NO': False, 'INVALID': None}.get(v) for k, v in saved.items()}
            if parsed != normalized:
                raise ValueError('raw output contradicts recorded prediction')
            key = digest([*[r[k] for k in ('scenario', 'route_id', 'frame_id', 'context_id')], r['prompt_spec']])
            if key in cases:
                raise ValueError('duplicate validation case')
            cases[key] = (parsed, r['history_rgb_paths_used'])
        return cases
    a, b = load(before), load(after)
    if not a or a.keys() != b.keys():
        raise ValueError('reload validation case set/spec mismatch')
    if any(a[k][1] != b[k][1] for k in a):
        raise ValueError('reload input paths differ')
    mismatches = [k for k in a if a[k][0] != b[k][0]]
    return dict(status='verified' if not mismatches else 'prediction_mismatch', cases=len(a),
                prediction_mismatch_ids=mismatches,
                before_sha256=file_sha(before), after_sha256=file_sha(after),
                scope='unmerged adapter, identical fixed validation questions and parser; not model accuracy approval')


def run(prepared, output, data_root, model_dir, reuse_epoch=None):
    # This coordinator is not a DDP worker; only train.sh owns rank variables.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    from .server_readiness import source_check
    from .server_phase3_prepare import execute, sampling_plans
    from .first_epoch_budget import profile, request
    from .training_budget import plan
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
    state = dict(status='blocked', completed_modes=[], training_started=False)
    try:
        source = source_check(ROOT, RELEASE)
        write_json(output / 'source_check.json', source)
        if source['status'] != 'matched':
            raise ValueError('source mismatch; synchronize epoch release')
        evidence, accepted = verified_preparation(prepared)
        index = Path(evidence['bindings']['index']['logical_path'])
        check_index(index, 'choice', prompt_variant=CANDIDATE_NAME)
        actual = sampling_plans(index, data_root)
        if actual['modes']['choice'] != accepted['modes']['choice'] or actual['prompt_sha256'] != accepted['prompt_sha256']:
            raise ValueError('native sampling or prompt differs from accepted preparation')
        write_json(output / 'admission.json', dict(prepared=evidence, sampling=actual))
        # Before/after RGB bindings cover the validation subset used for reload.
        from qwen3vl_local.sft_new_loop_phase3 import train
        rows = train._read_rows(index, 'val', data_root=data_root)
        work, _ = train._validation_work(rows, target_per_bin=32, seed=20260904,
                                         action_output_mode='choice', equal_events=True)
        pixels = [binding(Path(p)) for p in sorted({p for i in work for p in i.row.history_rgb_paths})]
        write_json(output / 'validation_input_bindings.json', pixels)
        budget = request(profile(model_dir), output, stored_evidence=evidence)
        budget['components']['phase3_index']['evidence'] = 'existing reviewed index; no build or data copy'
        budget['components']['optimizer_per_checkpoint'] = dict(upper_bound_bytes=0,
            evidence='native Phase3 saves adapters only; no optimizer or resumable training state')
        budget['limits'] = 'two sequential choice modes, one epoch, rank16 vision-off, final-only adapters; no RGB copies'
        budget = plan(budget)
        write_json(output / 'budget.json', budget)
        if budget['sufficient'] is not True:
            raise ValueError('first-epoch disk budget does not fit')
        reuse = verified_reuse(reuse_epoch, evidence, accepted) if reuse_epoch else None
        if reuse:
            write_json(output / 'reuse.json', reuse)
        for mode in MODES:
            mode_root = output / mode
            mode_root.mkdir()
            state.update(active_mode=mode, active_stage='train')
            if reuse and mode == reuse['mode']:
                (mode_root / 'train').symlink_to(reuse['train_path'], target_is_directory=True)
                state['reused_modes'] = [mode]
            else:
                env = training_env(index, data_root, model_dir, mode_root / 'train', mode)
                state.update(status='training', active_mode=mode, training_started=True)
                update_state(output, state)
                # Use the existing GPU launcher; preserve its automatic GPU selection.
                argv = ['bash', str(ROOT / 'qwen3vl_local/sft_new_loop_phase3/train.sh'), 'ddp']
                with (output / f'{mode}_train.log').open('w') as log:
                    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        try:
                            code = process.wait(timeout=30); break
                        except subprocess.TimeoutExpired:
                            print(f'[phase3-epoch] training {mode}; log={output / (mode + "_train.log")}', flush=True)
                if code:
                    raise RuntimeError(f'{mode} training exited {code}')
            consumed = json.loads((mode_root / 'train/balance/consumed_epoch_0001.json').read_text())
            if consumed['status'] != 'verified' or any(consumed['actual'].get(k) != 1024 for k in train.CONTEXT_IDS):
                raise ValueError('actual training was not one complete equal-event epoch')
            adapter = mode_root / 'train/final'
            from qwen3vl_local.sft_new_loop_phase3.eval import _validate_action_adapter
            cfg = _validate_action_adapter(adapter, model_dir, action_output_mode='choice')
            if cfg['history_rgb_mode'] != mode or cfg.get('prompt_variant') != CANDIDATE_NAME:
                raise ValueError('saved adapter differs from requested first-epoch contract')
            adapter_bindings = [binding(p) for p in sorted(adapter.iterdir()) if p.is_file()]
            write_json(mode_root / 'adapter_bindings.json', adapter_bindings)
            for split, quota, label in (('val', 32, 'reload_val'), ('test', 0, 'test')):
                for p in pixels:
                    check_binding(p)
                state.update(status='evaluating', active_stage=label)
                update_state(output, state)
                execute(output, f'{mode}_{label}', evaluation_argv(
                    index, data_root, model_dir, adapter, mode_root / label, split, quota))
                evaluated = json.loads((mode_root / label / 'evaluation_plan.json').read_text())['plan']
                expected = accepted['modes']['choice']['val_generation' if split == 'val' else 'test']['plan']
                if evaluated['selected_ids'] != expected['selected_ids']:
                    raise ValueError(f'{label} did not consume the fixed accepted evaluation list')
                if split == 'val':
                    compared = compare_reload(mode_root / 'train/final_generation_val_cases.jsonl',
                                              mode_root / 'reload_val/cases.jsonl')
                    write_json(mode_root / 'reload_comparison.json', compared)
                    if compared['status'] != 'verified':
                        raise ValueError('save/reload prediction mismatch; inspect comparison')
            for value in [*evidence['bindings'].values(), *pixels, *adapter_bindings,
                          *(reuse['bindings'] if reuse else [])]:
                check_binding(value)
            state['completed_modes'].append(mode)
        if source_check(ROOT, RELEASE)['status'] != 'matched':
            raise ValueError('source drift during first epoch')
        state.update(status='first_epoch_verified', active_stage='complete')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        state.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
    update_state(output, state)
    return 0 if state['status'] == 'first_epoch_verified' else 2


def package(output, exit_code):
    from .server_pilot import package as pack
    members, errors = {}, []
    paths = [output / n for n in ('source_check.json', 'admission.json', 'budget.json',
                                  'epoch_result.json', 'validation_input_bindings.json')]
    for mode in MODES:
        paths += [output / mode / n for n in ('adapter_bindings.json', 'reload_comparison.json',
            'train/train_run_manifest.json', 'train/train_balance.json', 'train/final_generation.json',
            'train/balance/consumed_epoch_0001.json', 'train/balance/epoch_0001.json',
            'train/final/sft_new_loop_phase3_adapter_config.json',
            'reload_val/metrics.json', 'reload_val/evaluation_plan.json', 'test/metrics.json', 'test/evaluation_plan.json',
            'train/final_generation_val_cases.jsonl', 'reload_val/cases.jsonl', 'test/cases.jsonl')]
    if (output / 'reuse.json').is_file():
        paths.append(output / 'reuse.json')
    for path in paths:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 8_000_000:
                raise ValueError('report exceeds 8MB; retained on server')
            value = report_value(path)
            members[str(path.relative_to(output))] = value
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    for name in ('epoch', *(f'{m}_{s}' for m in MODES for s in ('train', 'reload_val', 'test'))):
        path = output / f'{name}.log'
        if path.is_file():
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 40_000))
                members[f'{name}_log_tail.json'] = dict(sha256=file_sha(path), tail=f.read().decode(errors='replace'))
    success = (exit_code == 0 and not errors
               and members.get('epoch_result.json', {}).get('status') == 'first_epoch_verified'
               and all(str(p.relative_to(output)) in members for p in paths))
    members['handoff.json'] = dict(status='first_epoch_verified' if success else 'incomplete',
                                  command_exit_code=exit_code, errors=errors)
    pack(output, members)
    (output / 'pilot_report.zip').rename(output / 'phase3_epoch_report.zip')
    (output / 'pilot_package.json').rename(output / 'phase3_epoch_package.json')
    return 0 if success else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'package'))
    p.add_argument('--prepared', type=Path)
    p.add_argument('--reuse-epoch', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=ROOT / 'lead_data')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'checkpoints/Qwen3.5-4B')
    p.add_argument('--exit-code', type=int, default=2)
    a = p.parse_args()
    if a.command == 'run' and a.prepared is None:
        p.error('--prepared required')
    return (run(a.prepared.absolute(), a.output.absolute(), a.data_root.absolute(), a.model_dir.absolute(),
                a.reuse_epoch.absolute() if a.reuse_epoch else None)
            if a.command == 'run' else package(a.output.absolute(), a.exit_code))


if __name__ == '__main__':
    raise SystemExit(main())
