"""Contracts for the env-based single-group pipeline; never promotes an adapter."""
import argparse
import json
import importlib
import os
from pathlib import Path
import subprocess
import sys

from .four_runs import GROUPS, ROOT, SOURCE, module, sha, write
from .prompt_candidate import CANDIDATE_NAME
from .prompt_contract import prompt_name, adapter_prompt_variant
from .history_rgb import HISTORY_RGB_MODES, validate_adapter_history


def original_cases(rgb, output, audit_root):
    if rgb == '2rgb_short':
        raise ValueError('2rgb_short has no historical same-input audit; use RUN_REGRESSION=0 for this new experiment')
    stamp = next(t for r, o, t in GROUPS if (r, o) == (rgb, output))
    return Path(audit_root) / f'sft_new_loop_phase3_{stamp}_{rgb}_{output}_audit_bundle/lora_production'


def adapter_contract(root, rgb, output, variant):
    # Match eval.py's exact/best/final/fallback resolution order without importing torch.
    root = Path(root)
    for path in (root, root/'best_generation', root/'final', root/'fallback_generation'):
        cfg_path = path/'sft_new_loop_phase3_adapter_config.json'
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text())
        if (cfg.get('history_rgb_mode'), cfg.get('action_output_mode', 'binary'),
                adapter_prompt_variant(cfg)) != (rgb, output, variant):
            raise ValueError('adapter RGB/output/prompt differs from requested pipeline group')
        validate_adapter_history(cfg)
        return path.resolve()
    raise FileNotFoundError(f'no Phase3 adapter under {root}')


class HistoricalInputsMissing(FileNotFoundError):
    """An actionable prerequisite error, not a model or training failure."""


def validate_history(cases_path, variant, automot_root):
    cases_path = Path(cases_path)
    if not cases_path.exists() or (cases_path.is_dir() and not any(cases_path.glob('cases*.jsonl'))):
        archive = cases_path.parent.with_name(cases_path.parent.name + '.tar.gz')
        archive_hint = (f'发现压缩包 {archive.resolve()}，请先解压并保留包内目录结构。'
                        if archive.is_file() else '请将对应旧审计包放到此处，或用 AUDIT_ROOT 指向四个包的共同父目录。')
        raise HistoricalInputsMissing(
            f'缺少历史回归 case 文件：{cases_path.resolve()}/cases*.jsonl\n'
            '当前 RUN_REGRESSION=1；尚未开始本次训练。历史审计包是本地产物，不随 Git 源码同步。\n'
            f'{archive_hint}\n'
            '还需保留 case 引用的原始 RGB；REPLAY_AUTOMOT_ROOT 可指定它们的根目录。\n'
            '若明确只做训练和新 test 评测，可设置 RUN_REGRESSION=0；这会跳过旧成功题回归，不能视为无退化验收。')
    from .paired_eval import load_cases
    from .replay_prompt_candidate import requests
    cases, hashes = load_cases(cases_path)
    if not cases:
        raise ValueError('historical regression cases are empty')
    # SHA verification is not a new visual audit.
    count = sum(1 for _ in requests(cases, automot_root, variant))
    return dict(cases=count, original_successes=sum(r['all_ok'] for r in cases.values()),
                source_sha256=hashes, path=str(cases_path))


def data_identity(index, variant, output):
    from .preflight import check_index
    if variant == CANDIDATE_NAME:
        from .candidate_data import validate_candidate_index
        validate_candidate_index(index)
    manifest = json.loads(index.with_name('manifest.json').read_text())
    if manifest.get('source_scope') != 'collection_results':
        raise ValueError('full pipeline requires a collection_results build')
    report = check_index(index, action_output_mode=output)
    return dict(index_sha256=sha(index), manifest_sha256=sha(index.with_name('manifest.json')),
                preflight=report)


def source_identity():
    return {p.name: sha(p) for p in SOURCE.iterdir() if p.suffix in ('.py', '.sh', '.json')}


def assert_binding(index, state):
    expected = state['data_identity']
    if sha(index) != expected['index_sha256'] or sha(index.with_name('manifest.json')) != expected['manifest_sha256']:
        raise ValueError('index/manifest changed during pipeline; start a fresh run')
    if source_identity() != state['source_sha256']:
        raise ValueError('Phase3 source changed during pipeline; start a fresh run')


def replay_gpu(env):
    if env.get('GPU_IDS'):
        return env['GPU_IDS'].split(',')[0]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = [tuple(int(v.strip()) for v in row.split(',')) for row in rows.splitlines() if row.strip()]
    if not devices:
        raise RuntimeError('no GPU available for historical replay')
    return str(min(devices, key=lambda d: (d[1], d[2], d[0]))[0])


def replay_and_gate(cases, adapter, pipeline, variant, model, automot_root):
    replay = pipeline/'old_cases_replay'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=replay_gpu(os.environ))
    subprocess.run(module('replay_prompt_candidate', 'replay', '--cases', cases,
        '--output', replay, '--variant', variant, '--automot-root', automot_root,
        '--model-dir', model, '--adapter-dir', adapter, '--device', 'cuda:0'), env=env, check=True)
    report = pipeline/'regression_gate.json'
    result = subprocess.run(module('regression_gate', '--left', cases, '--right', replay,
        '--risks', SOURCE/'rgb_stage_risks_20261006.json', '--output', report))
    # Only a completed, parsed quality failure may continue the next independent run.
    if result.returncode not in (0, 2):
        raise RuntimeError(f'historical regression process failed: {result.returncode}')
    payload = json.loads(report.read_text())
    if payload['candidate_gate_passed'] is not (result.returncode == 0):
        raise ValueError('regression report disagrees with process status')
    return payload['candidate_gate_passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('stage', choices=('start', 'bind', 'verify', 'adapter', 'finish'))
    args = parser.parse_args()
    env = os.environ
    pipeline = Path(env['PIPELINE_ROOT'])
    index = Path(env['INDEX'])
    rgb, output, variant = (env[k] for k in ('HISTORY_RGB_MODE', 'ACTION_OUTPUT_MODE', 'PROMPT_VARIANT'))
    prompt_name(variant)
    if rgb not in HISTORY_RGB_MODES or output not in ('binary', 'choice'):
        raise ValueError('unknown RGB/output group')
    state_path = pipeline/'pipeline_manifest.json'
    if args.stage == 'start':
        state = dict(group=f'{rgb}_{output}', prompt_variant=variant, status='preflight',
            automatic_promotion=False, audits_enabled=env.get('RUN_AUDITS', '0') == '1',
            historical_regression_enabled=env['RUN_REGRESSION'] == '1', source_sha256=source_identity(),
            index=str(index.resolve()), run_root=env['RUN_ROOT'])
        write(state_path, state)
        if env['SKIP_TRAIN'] != '1' and Path(env['RUN_ROOT']).exists():
            raise ValueError('training output already exists; use a fresh RUN_ROOT')
        if env['SKIP_EVAL'] != '1' and env['RUN_REGRESSION'] == '1':
            cases = original_cases(rgb, output, env['AUDIT_ROOT'])
            state['historical_input'] = validate_history(cases, variant, Path(env['REPLAY_AUTOMOT_ROOT']))
            write(state_path, state)
        if env['SKIP_TRAIN'] == '1' and env['SKIP_EVAL'] != '1':
            adapter_contract(env['RUN_ROOT'], rgb, output, variant)
        if env['SKIP_TRAIN'] != '1' or env['SKIP_EVAL'] != '1':
            from qwen3vl_local.qwen35.preflight import check
            runtime = check(env['MODEL_DIR'])
            # Mirror launchers: explicit GPU_IDS wins, otherwise ignore inherited CUDA masks.
            if env.get('GPU_IDS'):
                ids = env['GPU_IDS'].split(',')
                if any(not i.isdigit() for i in ids) or len(ids) != len(set(ids)):
                    raise ValueError('GPU_IDS must contain distinct nonnegative GPU indices')
                os.environ['CUDA_VISIBLE_DEVICES'] = env['GPU_IDS']
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            for dependency in ('peft', 'torch'):
                try:
                    importlib.import_module(dependency)
                except ImportError as exc:
                    runtime['errors'].append(str(exc))
            if 'torch' in sys.modules:
                torch = sys.modules['torch']
                want = 1 if env.get('TRAIN_MODE') == 'single' else int(env.get('DDP_GPU_COUNT', env.get('NPROC_PER_NODE', '4')))
                if env['SKIP_TRAIN'] == '1': want = 0
                if env['SKIP_EVAL'] != '1': want = max(want, int(env.get('EVAL_GPU_COUNT', '4')))
                if env.get('GPU_IDS'): want = len(env['GPU_IDS'].split(','))
                runtime['requested_gpu_count'] = want
                runtime['available_gpu_count'] = torch.cuda.device_count()
                if not torch.cuda.is_available() or torch.cuda.device_count() < want:
                    runtime['errors'].append(f'CUDA devices insufficient: need {want}, found {torch.cuda.device_count()}')
            runtime['ready'] = not runtime['errors']
            write(pipeline/'runtime_preflight.json', runtime)
            if not runtime['ready']:
                raise RuntimeError('local model/runtime incomplete; see runtime_preflight.json')
        return
    state = json.loads(state_path.read_text())
    if args.stage == 'bind':
        state['data_identity'] = data_identity(index, variant, output)
        state['status'] = 'data_checked'
        write(state_path, state)
        return
    assert_binding(index, state)
    if args.stage == 'verify':
        return
    if args.stage == 'adapter':
        print(adapter_contract(env['RUN_ROOT'], rgb, output, variant))
        return
    if env['SKIP_EVAL'] == '1':
        state['status'] = 'data_prepared' if env['SKIP_TRAIN'] == '1' else 'trained_without_test'
    else:
        state['status'] = 'completed'
        state['regression_gate_passed'] = None
        if env['RUN_REGRESSION'] == '1':
            adapter = adapter_contract(env['RUN_ROOT'], rgb, output, variant)
            cases = original_cases(rgb, output, env['AUDIT_ROOT'])
            if validate_history(cases, variant, Path(env['REPLAY_AUTOMOT_ROOT'])) != state['historical_input']:
                raise ValueError('historical cases changed since preflight')
            passed = replay_and_gate(cases, adapter, pipeline, variant, env['MODEL_DIR'], env['REPLAY_AUTOMOT_ROOT'])
            state['regression_gate_passed'] = passed
            if not passed:
                state['status'] = 'completed_regression_failed'
                print('[phase3-pipeline] regression FAILED: review regression_gate.json; next experiment may continue; no promotion')
        else:
            state['status'] = 'completed_without_historical_regression'
    assert_binding(index, state)
    write(state_path, state)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = Path(os.environ['PIPELINE_ROOT'])/'pipeline_manifest.json'
        if path.is_file():
            state = json.loads(path.read_text())
            state.update(status='failed', error=str(exc))
            write(path, state)
        if isinstance(exc, HistoricalInputsMissing):
            print(f'[phase3-pipeline] {exc}', file=sys.stderr)
            raise SystemExit(2) from None
        raise
