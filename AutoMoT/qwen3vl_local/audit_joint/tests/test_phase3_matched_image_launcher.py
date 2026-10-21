import json
import os
from pathlib import Path
import subprocess

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / 'phase3_matched_image_eval.sh'


@pytest.mark.parametrize('mode', ['pin', 'auto', 'preflight', 'bad_pin', 'failure'])
def test_launcher_selection_forwarding_and_exit_status(tmp_path, mode):
    capture = tmp_path / 'capture.json'
    fake_python = tmp_path / 'python'
    fake_python.write_text('''#!/usr/bin/env python3
import json,os,sys
if sys.argv[1] == '-c':
    exec(sys.argv[2])
else:
    json.dump(dict(args=sys.argv[1:],gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cwd=os.getcwd(),offline=os.environ['HF_HUB_OFFLINE'],rank=os.environ.get('RANK')),
        open(os.environ['CAPTURE'],'w'))
    sys.exit(int(os.environ.get('TEST_EXIT','0')))
''')
    fake_python.chmod(0o755)
    smi = tmp_path / 'nvidia-smi'
    smi.write_text('#!/bin/sh\n' + (
        "printf '0, 100, 0\\n2, 10, 20\\n1, 10, 0\\n'\n" if mode == 'auto' else 'exit 99\n'))
    smi.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
        PYTHON=str(fake_python), CAPTURE=str(capture), GPU_IDS='',
        CUDA_VISIBLE_DEVICES='9', RANK='3', TEST_EXIT='7' if mode == 'failure' else '0')
    if mode in ('pin', 'failure'): env['GPU_IDS'] = '2'
    if mode == 'bad_pin': env['GPU_IDS'] = '0,1'
    args = ['--job', 'old job.json', '--output', 'new output']
    if mode == 'preflight': args += ['--preflight-only']
    result = subprocess.run(['bash', str(LAUNCHER), *args], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == {'bad_pin': 2, 'failure': 7}.get(mode, 0), result.stderr
    if mode == 'bad_pin':
        assert not capture.exists()
        return
    got = json.loads(capture.read_text())
    assert got['args'][-len(args):] == args
    assert got['gpu'] == {'auto': '1', 'preflight': '9'}.get(mode, '2')
    assert got['offline'] == '1' and got['rank'] is None
    assert Path(got['cwd']) == LAUNCHER.parents[2]
    for name in got['args'][3:6]:
        assert (Path(got['cwd'])/name).is_file()
