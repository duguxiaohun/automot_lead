import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen3vl_local.audit_joint import local_paths, label_release_check


@pytest.mark.parametrize('layout', ['home/cruser1/lda/AutoMoT', 'mnt/workspace/cee3sgh/AutoMoT'])
@pytest.mark.parametrize('label', [None, 'cr02', 'cee'])
def test_paths_follow_local_assets_not_server_label(tmp_path, monkeypatch, layout, label):
    app = tmp_path/layout
    raw = app/'raw'; model = app/'checkpoints/model'; raw.mkdir(parents=True); model.mkdir(parents=True)
    (model/'config.json').write_text('{}')
    if label is None: monkeypatch.delenv('AUTOMOT_SERVER', raising=False)
    else: monkeypatch.setenv('AUTOMOT_SERVER', label)
    result = local_paths.check_inputs(raw, model, app/'checkpoints/new/output')
    assert result['status'] == 'local_paths_present'
    assert result['server_context']['paths']['data_root']['resolved'] == str(raw)
    assert result['server_context']['server_label'] == (label or 'unspecified')
    assert not (app/'checkpoints/new').exists()


@pytest.mark.parametrize('defect', ['missing_raw', 'missing_model', 'in_raw', 'in_model', 'existing', 'dangling', 'alias'])
def test_refuse_wrong_local_assets_or_overwriting_input(tmp_path, defect):
    raw = tmp_path/'raw'; model = tmp_path/'model'; raw.mkdir(); model.mkdir()
    (model/'config.json').write_text('{}'); out = tmp_path/'out'
    if defect == 'missing_raw': raw = tmp_path/'other-machine/raw'
    elif defect == 'missing_model': model = tmp_path/'other-machine/model'
    elif defect == 'in_raw': out = raw/'out'
    elif defect == 'in_model': out = model/'out'
    elif defect == 'existing': out.mkdir()
    elif defect == 'dangling': out.symlink_to(tmp_path/'missing')
    elif defect == 'alias':
        alias=tmp_path/'alias';alias.symlink_to(raw, target_is_directory=True);out=alias/'out'
    result = local_paths.check_inputs(raw, model, out)
    assert result['status'] == 'blocked' and result['errors']
    assert not (tmp_path/'raw/out').exists()


def test_readonly_cli_does_not_require_a_machine_selection(tmp_path, monkeypatch):
    monkeypatch.delenv('AUTOMOT_SERVER', raising=False)
    monkeypatch.setattr(sys, 'argv', ['label_release_check', '--dataset', str(tmp_path/'data4'), '--output', str(tmp_path/'out')])
    calls=[]
    monkeypatch.setattr(label_release_check, 'run', lambda *a: calls.append(a) or 0)
    with pytest.raises(SystemExit) as caught: label_release_check.main()
    assert caught.value.code == 0 and calls == [(tmp_path/'data4',tmp_path/'out')]


def test_build_shell_from_another_cwd_refuses_missing_assets_before_output(tmp_path):
    app = Path(local_paths.__file__).resolve().parents[2]
    output = tmp_path/'outputs/new'
    env = {k:v for k,v in os.environ.items() if k not in ('AUTOMOT_SERVER','PYTHONPATH')}
    env.update(PYTHON=sys.executable, DATA_ROOT=str(tmp_path/'absent-raw'),
               MODEL_DIR=str(tmp_path/'absent-model'), OUTPUT_DIR=str(output))
    result = subprocess.run(['bash',str(app/'qwen3vl_local/audit_joint/server_phase4_existing.sh')],
                            cwd=tmp_path,env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode == 2, result.stderr
    report=json.loads(result.stdout)
    assert len(report['errors']) == 2 and report['server_context']['project_root'] == str(app)
    assert not output.exists() and not output.parent.exists()
