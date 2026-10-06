"""Prepare/check/run four isolated Phase3 RGB-stage experiments, sequentially."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'qwen3vl_local.sft_new_loop_phase3'
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from qwen3vl_local.sft_new_loop_phase3.prompt_candidate import CANDIDATE_NAME
from qwen3vl_local.sft_new_loop_phase3.prompt_contract import PROMPT_VARIANTS, action_prompt_sha256

GROUPS = (
    ('4rgb', 'binary', '20260928_220255'),
    ('2rgb_endpoints', 'binary', '20260929_064331'),
    ('4rgb', 'choice', '20260929_121830'),
    ('2rgb_endpoints', 'choice', '20260929_181319'),
)
# Only these explicit settings are inherited; unrelated pipeline/run/skip overrides cannot truncate a suite.
SETTINGS = dict(NUM_EPOCHS='3', FOCUS_BALANCE_COUNT='1024', INVALID_FOCUS_MULTIPLIER='2.0',
    LEARNING_RATE='1e-5', GRAD_ACCUM='1', WARMUP_STEPS='2000', SEED='20260904',
    EVAL_STEPS='2000', GENERATION_EVAL_STEPS='2000', EVAL_BALANCE_COUNT='16',
    GENERATION_EVAL_BALANCE_COUNT='32', GENERATION_EVAL_SAMPLING_SEED='20260904',
    GENERATION_EVAL_MAX_NEW_TOKENS='64', MAX_LENGTH='8192', SAVE_STEPS='20000',
    LORA_RANK='16', LORA_ALPHA='32', LORA_DROPOUT='0.05', LORA_VISION_SCOPE='off',
    DDP_TIMEOUT_SECONDS='3600', WEIGHT_DECAY='0.0', MAX_GRAD_NORM='1.0',
    FORMAT_LOSS_WEIGHT='0.25', TRAIN_ROUTE_DIVERSE='1', GENERATION_EVAL_ROUTE_DIVERSE='1',
    AUTO_EVAL_BALANCE_COUNT='1', GENERATION_EVAL_MIN_VALID_RATE='1.0',
    GENERATION_EVAL_MIN_INVALID_EXACT='0.80', GENERATION_EVAL_MIN_LANE_CHANGE_RECALL='0.60',
    GENERATION_EVAL_MIN_STOP_RECALL='0.80', GENERATION_EVAL_MIN_NO_ACTION_EXACT='0.50')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def groups(args):
    return [g for g in GROUPS if not args.groups or f'{g[0]}_{g[1]}' in args.groups]


def environment(args, rgb, output):
    # Preserve runtime paths/CUDA libraries, but reset all shell knobs used by this workflow.
    env = os.environ.copy()
    for key in ('OUTPUT_DIR', 'RUN_LOG', 'RUN_NAME', 'RUN_TIMESTAMP', 'MODE', 'MASTER_PORT',
                'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update({k: os.environ.get(k, v) for k, v in SETTINGS.items()})
    env.update(INDEX=str(args.index), DATA_ROOT=str(args.data_root), MODEL_DIR=str(args.model_dir),
        PROMPT_VARIANT=args.prompt_variant, HISTORY_RGB_MODE=rgb, ACTION_OUTPUT_MODE=output,
        GPU_IDS=args.gpus, NPROC_PER_NODE=str(len(args.gpus.split(','))),
        MAX_STEPS='0', MAX_FRAMES='0', MAX_EVAL_FRAMES='0', EVAL_SPLIT='val',
        SAVE_BEST_VAL='1', SAVE_BEST_GENERATION='1', SAVE_FINAL='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        # Shell launchers use python/torchrun; force the active interpreter's environment first.
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    return env


def plan(args):
    result = []
    for rgb, output, old in groups(args):
        env = environment(args, rgb, output)
        result.append(dict(group=f'{rgb}_{output}', prompt_variant=args.prompt_variant,
            prompt_sha256=action_prompt_sha256(prompt_variant=args.prompt_variant,
                history_rgb_mode=rgb, action_output_mode=output),
            train_dir=str(args.output / f'{rgb}_{output}' / 'train'),
            settings={k: env[k] for k in SETTINGS},
            original_bundle=str(args.audit_root / f'sft_new_loop_phase3_{old}_{rgb}_{output}_audit_bundle')))
    return dict(schema='phase3_four_runs_v1', groups=result, index=str(args.index),
        model_dir=str(args.model_dir), gpus=args.gpus, sequential=True,
        full_training=True, automatic_promotion=False, old_audit_is_development_only=True)


def command(args, argv, log_name, *, env=None, allowed=(0,)):
    print('[four-runs] ' + ' '.join(map(str, argv)), flush=True)
    with (args.output / log_name).open('w') as log:
        process = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'[four-runs] running; log={args.output / log_name}', flush=True)
    if code not in allowed:
        raise RuntimeError(f'command failed ({code}); see {args.output / log_name}')
    return code


def module(name, *args):
    return [sys.executable, '-m', f'{PACKAGE}.{name}', *args]


def prepare(args):
    if not args.index.exists():
        if args.index.name != 'frame_index.jsonl':
            raise ValueError('missing custom index: automatic build requires filename frame_index.jsonl')
        if args.index.parent.exists():
            raise ValueError('partial/existing data directory without index; choose a fresh data directory')
        command(args, module('build_dataset', '--collection-dir', args.collection_dir,
            '--data-root', args.data_root, '--output-dir', args.index.parent,
            '--development-groups-extra', SOURCE / 'development_route_groups_rgb_stage_20261006.json',
            '--workers', args.workers, '--split-seed', '20260920', '--test-ratio', '0.10',
            '--val-ratio', '0.05', '--require-invalid-true-rs-coverage'), 'build.log')
    if not args.index.with_name('manifest.json').is_file():
        raise ValueError('full build manifest required; bare/custom partial index is not admitted')
    from qwen3vl_local.sft_new_loop_phase3.candidate_data import validate_candidate_index
    validate_candidate_index(args.index)
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
    write(args.output / 'index_preflight.json', check_index(args.index))
    for output in ('binary', 'choice'):
        for name in ('audit_rebuilt_index', 'audit_raw_index'):
            argv = module(name, '--index', args.index, '--data-root', args.data_root,
                '--action-output-mode', output, '--output', args.output / f'{name}_{output}.json')
            if name == 'audit_raw_index': argv.extend(['--workers', args.workers])
            command(args, argv, f'{name}_{output}.log')
    command(args, module('audit_temporal_slices', '--index', args.index,
        '--output', args.output / 'temporal_slices.json'), 'temporal_slices.log')
    for rgb, output, _ in groups(args):
        command(args, ['bash', SOURCE / 'train.sh', 'sampling'], f'sampling_{rgb}_{output}.log',
            env=environment(args, rgb, output))
    return dict(index_sha256=sha(args.index), manifest_sha256=sha(args.index.with_name('manifest.json')))


def runtime_check(args):
    from qwen3vl_local.qwen35.preflight import check
    report = check(args.model_dir)
    import torch
    gpu_ids = args.gpus.split(',')
    if not torch.cuda.is_available(): report['errors'].append('CUDA unavailable')
    elif any(int(i) >= torch.cuda.device_count() for i in gpu_ids):
        report['errors'].append('requested GPU index is unavailable to this process')
    report['ready'] = not report['errors']
    # Full vendor provenance stays in the report, never upload it.
    write(args.output / 'runtime_preflight.json', report)
    if not report['ready']: raise RuntimeError('runtime/model/GPU preflight failed; see runtime_preflight.json')


def audit_inputs_check(args):
    from qwen3vl_local.sft_new_loop_phase3.paired_eval import load_cases
    from qwen3vl_local.sft_new_loop_phase3.replay_prompt_candidate import requests
    for group in plan(args)['groups']:
        cases, _ = load_cases(Path(group['original_bundle']) / 'lora_production')
        # Actual SHA and original v23 prompt identity checked before any GPU training.
        list(requests(cases, args.automot_root, args.prompt_variant))


def historical_dataset_comparison(args):
    current = json.loads(args.index.with_name('manifest.json').read_text())
    results = []
    for group in plan(args)['groups']:
        path = Path(group['original_bundle']) / 'dataset_metadata/manifest.json'
        if not path.is_file():
            results.append(dict(group=group['group'], available=False)); continue
        old = json.loads(path.read_text())
        results.append(dict(group=group['group'], available=True, old_manifest_sha256=sha(path),
            old_counts=old['counts'], new_counts=current['counts'],
            counts_equal=old['counts']==current['counts'],
            old_split_coverage_sha256=hashlib.sha256(json.dumps(old.get('split_coverage'),sort_keys=True).encode()).hexdigest(),
            new_split_coverage_sha256=hashlib.sha256(json.dumps(current.get('split_coverage'),sort_keys=True).encode()).hexdigest(),
            interpretation='New exposure isolation changes the pool. Historical score differences do not isolate prompt effects.'))
    return results


def run(args):
    if args.command == 'plan':
        print(json.dumps(plan(args), ensure_ascii=False, indent=2)); return
    args.output.mkdir(parents=True, exist_ok=False)
    state = plan(args)
    state.update(status='preparing', completed_groups=[])
    source_hashes = {p.name: sha(p) for p in SOURCE.iterdir() if p.suffix in ('.py', '.sh', '.json')}
    state['source_sha256'] = source_hashes
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for name in source_hashes:
        (snapshot / name).write_bytes((SOURCE / name).read_bytes())
    write(args.output / 'suite.json', state)
    try:
        if args.command in ('check', 'run'): runtime_check(args)
        state['data_identity'] = prepare(args)
        state['historical_dataset_comparison'] = historical_dataset_comparison(args)
        if args.command in ('check', 'run'): audit_inputs_check(args)
        if args.command != 'run':
            state['status'] = 'data_prepared' if args.command == 'prepare' else 'preflight_passed'
            return
        for group in state['groups']:
            if any(sha(SOURCE / name) != value for name, value in source_hashes.items()):
                raise ValueError('source changed during suite; use a new suite after reviewing changes')
            if sha(args.index) != state['data_identity']['index_sha256']:
                raise ValueError('training index changed during suite')
            tag = group['group']; rgb, output = next((r, o) for r, o, _ in GROUPS if f'{r}_{o}' == tag)
            env = environment(args, rgb, output)
            train_dir = Path(group['train_dir'])
            if train_dir.exists(): raise ValueError(f'refusing existing train directory: {train_dir}')
            env['OUTPUT_DIR'] = str(train_dir)
            state.update(status='training', active_group=tag); write(args.output / 'suite.json', state)
            mode = 'single' if len(args.gpus.split(',')) == 1 else 'ddp'
            command(args, ['bash', SOURCE / 'train.sh', mode], f'train_{tag}.log', env=env)
            env.update(ADAPTER_DIR=str(train_dir), OUTPUT_ROOT=str(train_dir.parent / 'eval'),
                SPLIT='test', REQUIRE_INVALID_COVERAGE='1', CASES_PER_BIN='0', ROUTE_DIVERSE_SAMPLING='0', MAX_EVAL_FRAMES='0',
                MAX_NEW_TOKENS='256', RUN_BASE_EVAL='1', RUN_AUDIT_PROMPT_EVAL='auto',
                EXCLUDE_CASES_JSONL='', EXPECTED_EXCLUDED_CASES='0', EXPECTED_TOTAL_CASES='0',
                TIMESTAMP=args.output.name, BUNDLE_BASENAME=f'{args.output.name}_{tag}_audit_bundle')
            command(args, ['bash', SOURCE / 'eval.sh'], f'eval_{tag}.log', env=env)
            from qwen3vl_local.sft_new_loop_phase3.eval import _resolve_adapter_dir
            adapter, _ = _resolve_adapter_dir(train_dir)
            replay_dir = train_dir.parent / 'old_cases_replay'
            command(args, module('replay_prompt_candidate', 'replay', '--cases',
                Path(group['original_bundle']) / 'lora_production', '--output', replay_dir,
                '--variant', args.prompt_variant, '--automot-root', args.automot_root,
                '--model-dir', args.model_dir, '--adapter-dir', adapter, '--device', 'cuda:0'),
                f'replay_{tag}.log', env={**env, 'CUDA_VISIBLE_DEVICES': args.gpus})
            gate_path = train_dir.parent / 'regression_gate.json'
            gate_code = command(args, module('regression_gate', '--left',
                Path(group['original_bundle']) / 'lora_production', '--right', replay_dir,
                '--risks', SOURCE / 'rgb_stage_risks_20261006.json', '--output', gate_path),
                f'regression_{tag}.log', allowed=(0, 2))
            # A regression blocks promotion, not the other independent experiments.
            state['completed_groups'].append(dict(group=tag, regression_gate_passed=gate_code == 0,
                regression_report=str(gate_path)))
            write(args.output / 'suite.json', state)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=str(exc)); raise
    finally:
        write(args.output / 'suite.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'prepare', 'check', 'run'))
    p.add_argument('--model-dir', type=Path, default=ROOT / 'checkpoints/Qwen3.5-4B')
    p.add_argument('--index', type=Path, default=ROOT / 'checkpoints/sft_new_loop_phase3_data_rgb_stage_20261006_isolated/frame_index.jsonl')
    p.add_argument('--data-root', type=Path, default=ROOT / 'lead_data')
    p.add_argument('--collection-dir', type=Path, default=ROOT / 'keyframe_filter/collection_output')
    p.add_argument('--automot-root', type=Path, default=ROOT)
    p.add_argument('--audit-root', type=Path, default=ROOT / 'checkpoints')
    p.add_argument('--output', type=Path, default=ROOT / 'checkpoints/phase3_four_runs' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--gpus', default=os.environ.get('GPU_IDS', '0,1,2,3'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--prompt-variant', choices=PROMPT_VARIANTS, default=CANDIDATE_NAME)
    p.add_argument('--groups', nargs='+', choices=[f'{r}_{o}' for r,o,_ in GROUPS])
    args = p.parse_args()
    ids = args.gpus.split(',')
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids): p.error('--gpus requires unique numeric GPU indices')
    for name in ('model_dir', 'index', 'data_root', 'collection_dir', 'audit_root', 'automot_root', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    run(args)


if __name__ == '__main__':
    main()
