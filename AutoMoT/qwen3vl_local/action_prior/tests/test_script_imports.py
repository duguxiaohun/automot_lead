"""Fresh-process entrypoint regressions without inheriting pytest's import paths."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('working_directory', ['repository', 'elsewhere'])
def test_probe_parent_can_pack_after_successful_child(tmp_path, working_directory):
    """Exercise the late parent import; only GPU evaluation and plotting are substituted."""
    runner = tmp_path / 'isolated_launcher.py'
    runner.write_text('''
from pathlib import Path
import json, runpy, subprocess, sys
launcher, output = sys.argv[1:]
# Match `python /path/launch.py`: the parent starts with the script directory,
# not AutoMoT; the environment has no inherited PYTHONPATH.
sys.path[0] = str(Path(launcher).parent)
scope = runpy.run_path(launcher, run_name='launcher_under_test')
main = scope['main']
real_run = subprocess.run
def evaluated(command, log):
    directory = Path(output)
    directory.mkdir()
    (directory / 'cases').mkdir()
    (directory / 'metrics.json').write_text(json.dumps({'samples': 24}))
    # The child sees the launcher's exported PYTHONPATH even before the fix.
    real_run([sys.executable, '-m', 'qwen3vl_local.action_prior.audit_bundle',
              '--root', output], check=True)
    (directory / 'audit.zip').unlink()
    (directory / 'child_finished').touch()
def plotted(command, **kwargs):
    assert command[1:3] == ['-m', 'qwen3vl_local.action_prior.plot_probe']
    return subprocess.CompletedProcess(command, 0)
main.__globals__['ensure_gpu'] = lambda count: 1
main.__globals__['run_logged'] = evaluated
subprocess.run = plotted
sys.argv = [launcher, 'probe', '--checkpoint', str(Path(output).parent / 'best.pt'),
            '--output-dir', output]
main()
''')
    output = tmp_path / 'probe result'
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(runner), str(ROOT / 'qwen3vl_local/action_prior/launch.py'), str(output)],
        cwd=ROOT if working_directory == 'repository' else tmp_path,
        env=env, capture_output=True, text=True,
    )
    assert (output / 'child_finished').is_file(), result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(output / 'audit.zip') as archive:
        assert 'metrics.json' in archive.namelist()


@pytest.mark.parametrize('entrypoint', [
    'action_prior/launch.py', 'action_expert_ablation/launch.py',
    'action_prior/prepare_event_balance.py',
    'action_prior/build_event_balance_index.py',
    'sft_new_loop_phase3/build_dataset.py', 'sft_new_loop_phase3/preflight.py',
    'sft_new_loop_phase3/visual_audit.py', 'sft_new_loop_phase3/audit_raw_index.py',
    'sft_new_loop_phase3/audit_rebuilt_index.py', 'sft_new_loop_phase3/audit_temporal_slices.py',
    'sft_new_loop_phase3/paired_eval.py',
    'action_prior/phase3_releases/v23_io1/build_dataset.py',
])
def test_direct_entrypoints_without_pythonpath(tmp_path, entrypoint):
    """Actual CLI startup from outside the repository; no model or dataset loading."""
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT / 'qwen3vl_local' / entrypoint), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()
