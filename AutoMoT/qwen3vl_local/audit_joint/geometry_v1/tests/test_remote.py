"""Real shell and map preparation; expensive build/model calls are substitutes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def app(tmp_path):
    root = tmp_path/'app with spaces'
    entry = root/'qwen3vl_local/audit_joint/geometry_v1'
    entry.mkdir(parents=True)
    shutil.copy(Path(__file__).parents[1]/'remote.sh', entry/'remote.sh')
    (root/'.venv/bin').mkdir(parents=True)
    (root/'.venv/bin/python').symlink_to(sys.executable)
    for rel in ('lead_data', 'checkpoints/Qwen3.5-4B', 'keyframe_filter/collection_output'):
        (root/rel).mkdir(parents=True)
    # Deliberately link CARLA outside the app, as on the user's machines.
    carla = tmp_path/'installed CARLA'
    maps = carla/'CarlaUE4/Content/Carla/Maps'
    for town in [f'Town{i:02d}' for i in range(1, 8)] + ['Town10HD', 'Town11', 'Town12', 'Town13', 'Town15']:
        folder = maps/'OpenDrive' if town in [f'Town{i:02d}' for i in range(1, 8)] + ['Town10HD'] else maps/town/'OpenDrive'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/f'{town}.xodr').write_text(f'<OpenDRIVE name="{town}"/>')
    (root/'CARLA_0915').symlink_to(carla, target_is_directory=True)
    for name in ('carla', 'numpy', 'torch', 'transformers', 'peft'):
        (root/f'{name}.py').write_text('')
    (root/'PIL').mkdir()
    (root/'PIL/__init__.py').write_text('Image = object()\n')
    preflight = root/'qwen3vl_local/sft_new_loop_phase3'
    preflight.mkdir()
    (preflight/'preflight.py').write_text('def check_model(path): pass\n')
    (entry/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|%s\\n' "$1" "${GPU_IDS:-auto}" "$*" >> "$GEOMETRY_ROOT/calls"
[[ ${FAIL_STAGE:-} != "$1" ]] || exit 7
''')
    # The current CPU shell routes through the versioned preparation profiler.
    # Substitute that expensive dependency at its actual module boundary; keep
    # the real shell, project map lookup and propagation of failures under test.
    (entry.parent/'geometry_prepare_profile.py').write_text('''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
root=Path(os.environ['GEOMETRY_ROOT'])
assert args[args.index('--timing-report')+1]==str(root/'prepare_timings.jsonl')
assert args[args.index('--maps')+1]==str(root/'maps')
ready=Path(args[args.index('--receipt')+1])
if ready.exists():
    with (root/'calls').open('a') as f:f.write('reuse|cpu|verified_existing\\n')
else:
    for stage in ('build','plan'):
        with (root/'calls').open('a') as f:f.write(stage+'|cpu|native_preparation_substitute\\n')
        if os.environ.get('FAIL_STAGE')==stage:sys.exit(7)
    ready.write_text(json.dumps(dict(dataset=str(root/'versioned/data/dataset.json'))))
''')
    return root


def run(app, mode, **extra):
    env = dict(os.environ)
    for key in ('PYTHON', 'PYTHONPATH', 'GPU_IDS', 'GEOMETRY_EVAL_GPU'):
        env.pop(key, None)
    env.update(extra)
    return subprocess.run(['bash', str(app/'qwen3vl_local/audit_joint/geometry_v1/remote.sh'), mode],
                          cwd=app.parent, env=env, text=True, capture_output=True)


def test_project_symlinks_and_fresh_output(app):
    result = run(app, 'cpu')
    assert result.returncode == 0, result.stderr
    out = app/'checkpoints/geometry_v1'
    assets = json.loads((out/'assets.json').read_text())
    assert len(assets['maps']) == 12
    assert str(app/'.venv/bin/python') in result.stdout
    for town, item in assets['maps'].items():
        link = out/'maps'/f'{town}.xodr'
        assert link.is_symlink() and str(link.resolve()) == item['source']
    assert (out/'cpu.done').is_file()
    assert [line.split('|')[0] for line in (out/'calls').read_text().splitlines()] == ['build', 'plan']
    assert run(app, 'cpu').returncode == 0
    assert [line.split('|')[0] for line in (out/'calls').read_text().splitlines()] == ['build','plan','reuse']


@pytest.mark.parametrize('problem', ['missing', 'conflicting'])
def test_bad_map_before_output_creation(app, problem):
    maps = app/'CARLA_0915/CarlaUE4/Content/Carla/Maps'
    if problem == 'missing':
        (maps/'Town12/OpenDrive/Town12.xodr').unlink()
    else:
        (maps/'OpenDrive/Town12.xodr').write_text('different')
    assert run(app, 'cpu').returncode != 0
    assert not (app/'checkpoints/geometry_v1').exists()


@pytest.mark.parametrize('stage', ['build', 'plan'])
def test_cpu_failure_no_done(app, stage):
    assert run(app, 'cpu', FAIL_STAGE=stage).returncode == 7
    out = app/'checkpoints/geometry_v1'
    assert not (out/'cpu.done').exists()
    assert (out/'calls').read_text().splitlines()[-1].startswith(stage+'|')


def test_train_requires_cpu_and_stops_on_failure(app):
    assert run(app, 'train').returncode != 0
    assert run(app, 'cpu').returncode == 0
    assert run(app, 'train', GPU_IDS='0,1').returncode != 0
    assert run(app, 'train', FAIL_STAGE='train').returncode == 7
    out = app/'checkpoints/geometry_v1'
    assert not (out/'train_test.done').exists()
    assert not any(line.startswith('evaluate|') for line in (out/'calls').read_text().splitlines())


@pytest.mark.parametrize('devices,expected', [({}, 'auto'), ({'GPU_IDS': '3,2,1,0'}, '3')])
def test_train_then_evaluate(app, devices, expected):
    assert run(app, 'cpu').returncode == 0
    result = run(app, 'train', **devices)
    assert result.returncode == 0, result.stderr
    out = app/'checkpoints/geometry_v1'
    assert (out/'train_test.done').exists()
    assert (out/'calls').read_text().splitlines()[-1].startswith(f'evaluate|{expected}|')
    assert '--dataset '+str(out/'versioned/data/dataset.json') in (out/'calls').read_text()
    (out/'run').mkdir()
    assert run(app, 'train', **devices).returncode != 0
