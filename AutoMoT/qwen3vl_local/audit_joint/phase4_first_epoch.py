"""Reuse a completed native pool; build a current view, run two modes and collect evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import file_sha
from .guard import no_core
from .phase4_epoch_handoff import collect
from .rule_baseline import budget_check
from .server_readiness import source_check
from qwen3vl_local.sft_new_loop_phase4.identity import write_json

APP = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).with_name('release_sources_v60_batch_profiles_20261015.json')


class Interrupted(BaseException):
    def __init__(self, number):
        self.code = 128 + number


def stop(process):
    # Signal the owned shell and its guards together. Give guards time to clean
    # their separate torchrun groups, even if the shell exits before them.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def execute(command, log, env):
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, cwd=APP, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=no_core)
        try:
            code = process.wait()
            return code if code >= 0 else 128 - code
        except BaseException:
            # Further signals must not abandon cleanup half way through.
            previous = {n: signal.signal(n, signal.SIG_IGN) for n in (signal.SIGINT, signal.SIGTERM)}
            try:
                stop(process)
            finally:
                for n, handler in previous.items():
                    signal.signal(n, handler)
            raise


def run(prepared, output, data_root, model_dir, *, view=None, batch_size=1, gpus=None):
    prepared, output, data_root, model_dir = [Path(p).absolute() for p in (prepared, output, data_root, model_dir)]
    view = Path(view).absolute() if view is not None else None
    if batch_size not in (1, 2, 4, 8):
        raise ValueError('four-card first epoch supports per-rank batch 1, 2, 4 or 8')
    gpus = os.environ.get('GPU_IDS', '') if gpus is None else gpus
    cards = gpus.split(',') if gpus else []
    if gpus and (len(cards) != 4 or len(set(cards)) != 4 or any(not c.isdigit() for c in cards)):
        raise ValueError('specify four distinct numeric GPU IDs')
    for source in (prepared, data_root, model_dir, *((view,) if view else ())):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('output must be outside all input directories')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='preflight', stages=[], prepared=str(prepared),
                 dataset=str(view or output/'student/view'), batch_size=batch_size, gpu_ids=gpus or 'auto',
                 update_samples=32, accumulation=8//batch_size,
                 training_accepted=False, unsupported_capabilities=['U-E5/proceed/YES'])
    receipt = output/'workflow.json'
    code = 2
    def save():
        write_json(receipt, state)
    def on_signal(number, frame):
        raise Interrupted(number)
    previous = {n: signal.signal(n, on_signal) for n in (signal.SIGINT, signal.SIGTERM)}
    try:
        save()
        sources = source_check(APP, RELEASE)
        write_json(output/'source_check.json', sources)
        if sources['status'] != 'matched':
            raise ValueError('source mismatch; synchronize the declared release')
        for path in (prepared/'data4/manifest.json', prepared/'references4/references.json',
                     prepared/'references4/val.jsonl', prepared/'references4/test.jsonl',
                     prepared/'first_epoch_budget.json', model_dir/'config.json'):
            if not path.is_file():
                raise ValueError(f'required input missing: {path}')
        if not data_root.is_dir():
            raise ValueError('data root is missing')
        if view is not None:
            descriptor = json.loads((view/'student_view.json').read_text())
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get('parent'), dict):
                raise ValueError('reused view has no valid parent binding')
            parent = descriptor.get('parent', {}).get('logical_path')
            if not parent or Path(parent).resolve() != (prepared/'data4').resolve():
                raise ValueError('reused view does not belong to the requested native dataset')
        budget_path = prepared/'first_epoch_budget.json'
        request = json.loads(budget_path.read_text())
        if not isinstance(request, dict):
            raise ValueError('budget request must be an object')
        budget = budget_check(request, output)
        write_json(output/'launch_budget.json', budget)
        if budget['sufficient'] is not True:
            raise ValueError('first-epoch disk budget incomplete or insufficient')
        state['budget_request_sha256'] = file_sha(budget_path)
        env = dict(os.environ, PYTHON=sys.executable, PYTHONUNBUFFERED='1',
                   DATASET=state['dataset'], DATA_ROOT=str(data_root), MODEL_DIR=str(model_dir),
                   TEACHER_REFERENCES=str(prepared/'references4'),
                   FIRST_EPOCH_BUDGET=str(budget_path), OUTPUT_DIR=str(output),
                   RUN_TAG='first_epoch', GPU_IDS=gpus, BATCH_SIZE=str(batch_size),
                   UPDATE_SAMPLES='32', ACCUMULATION=str(8//batch_size),
                   AUTOMATIC_COVERAGE_POLICY='existing_data_development')
        commands = []
        if view is None:
            commands.append(('view_build', [sys.executable, '-m', 'qwen3vl_local.audit_joint.guard',
                '--space-path', str(output), '--min-free-gib', '20', '--', sys.executable, '-m',
                'qwen3vl_local.sft_new_loop_phase4.rgb_short.build', '--dataset4', str(prepared/'data4'),
                '--data-root', str(data_root), '--output', str(output/'student')]))
        else:
            state['reused_view'] = str(view)  # Native admission validates its current contract before GPU.
        for mode in (2, 4):
            commands.append((f'rgb{mode}', ['bash', 'qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh']))
        for stage, command in commands:
            state['stage'] = stage
            record = dict(stage=stage, status='running', log=f'{stage}.log')
            state['stages'].append(record)
            save()
            print(f'[phase4] {stage}: {output / record["log"]}', flush=True)
            started = time.monotonic()
            code = execute(command, output/record['log'], dict(env, RGB_MODE=stage[-1]) if stage.startswith('rgb') else env)
            record.update(exit_code=code, elapsed_seconds=time.monotonic()-started,
                          status='finished' if code == 0 else 'failed')
            save()
            if code:
                break
        state['status'] = 'completed_execution_not_audited' if code == 0 else 'failed'
    except Interrupted as ex:
        code = ex.code
        state.update(status='interrupted', error=f'signal exit {code}')
    except (OSError, ValueError, KeyError, TypeError) as ex:
        code = 2
        state.update(status='failed', error=f'{type(ex).__name__}: {ex}')
    finally:
        state['command_exit_code'] = code
        save()
        for n, handler in previous.items():
            signal.signal(n, handler)
    try:
        packed = collect(output)
        write_json(output/'handoff_result.json', packed)
        if packed['status'] != 'collected_not_semantically_verified' and code == 0:
            code = 2
    except (OSError, ValueError, RuntimeError) as ex:
        write_json(output/'handoff_error.json', dict(error=f'{type(ex).__name__}: {ex}'))
        if code == 0:
            code = 2
    print(f'[phase4] exit={code}; status={state["status"]}; evidence={output}/handoff', flush=True)
    return code


def main():
    no_core()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('lead_data'))
    p.add_argument('--model-dir', type=Path, default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--view', type=Path, help='Optional existing current view; admission still validates it')
    p.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8), default=1)
    a = p.parse_args()
    try:
        return run(a.prepared, a.output, a.data_root, a.model_dir, view=a.view, batch_size=a.batch_size)
    except (OSError, ValueError) as ex:
        p.error(str(ex))


if __name__ == '__main__':
    raise SystemExit(main())
