import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from qwen3vl_local.audit_joint import __main__ as cli
from qwen3vl_local.audit_joint.baseline import git_state, run, verify_bundle
from qwen3vl_local.audit_joint.handoff import pack, verify_package
from qwen3vl_local.audit_joint.layout import project_layout


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], stderr=subprocess.PIPE, text=True).strip()


def fixture_repo(tmp_path, nested, with_git=True):
    root = tmp_path / 'repo'
    application = root / 'AutoMoT' if nested else root
    package = application / 'qwen3vl_local/audit_joint'
    package.mkdir(parents=True)
    actual = Path(cli.__file__).parent
    for source in actual.glob('*.py'):
        shutil.copyfile(source, package / source.name)
    for name in ('qwen3vl_local/sft_new_loop_phase3', 'qwen3vl_local/sft_new_loop_phase4',
                 'lead_video_tools', 'keyframe_filter'):
        (application / name).mkdir()
    if with_git:
        git(root, 'init')
        git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '--allow-empty', '-m', 'fixture')
    return root, application, package / '__main__.py'


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('unrelated_cwd', [False, True])
def test_actual_cli_prepare_discovers_git_and_application_without_cwd(tmp_path, nested, unrelated_cwd):
    root, application, module = fixture_repo(tmp_path, nested)
    cwd = tmp_path / 'unrelated' if unrelated_cwd else application
    cwd.mkdir(exist_ok=True)
    output = tmp_path / 'request.json'
    env = dict(os.environ, PYTHONPATH=str(application))
    subprocess.run([sys.executable, '-m', 'qwen3vl_local.audit_joint', 'prepare',
                    '--phase3-prompt-variant', 'baseline', '--output', str(output)],
                   cwd=cwd, env=env, capture_output=True, text=True, check=True)
    config = json.loads(output.read_text())
    assert config['project_root'] == str(root)
    assert config['application_root'] == str(application)
    assert config['source_roots'] == [str(application / p) for p in ('qwen3vl_local', 'lead_video_tools', 'keyframe_filter')]
    assert config['model_dir'] == str(application / 'checkpoints/Qwen3.5-4B')
    assert all(s['data_root'] == str(application / 'lead_data') for s in config['split_sources'])
    assert next(s for s in config['split_sources'] if s['id'] == 'phase3_full_index')['path'].startswith(str(application / 'checkpoints'))
    # Real Git discovery and state, including blocked run and ZIP, not mocked.
    capture = tmp_path / 'capture'
    with patch.object(sys, 'argv', ['audit', 'run', '--config', str(output), '--output', str(capture)]), \
            patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}):
        assert cli.main() == 2
    saved = json.loads((capture / 'baseline_manifest.json').read_text())
    assert saved['git']['availability'] == 'available'
    assert saved['git']['head'] == git(root, 'rev-parse', 'HEAD')
    assert not any(b['kind'] == 'git_identity' for b in saved['blockers'])
    assert verify_bundle(capture)['status'] == 'verified'
    assert verify_package(capture.with_suffix('.audit.zip'))['status'] == 'verified'


@pytest.mark.parametrize('nested', [False, True])
def test_explicit_project_root_supports_both_layouts(tmp_path, nested):
    root, application, _ = fixture_repo(tmp_path, nested)
    assert project_layout(root) == (root, application)
    assert project_layout(application) == (application, application)


def test_ambiguous_and_invalid_explicit_roots_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='exactly one application layout'):
        project_layout(tmp_path)
    (tmp_path / 'qwen3vl_local').mkdir()
    (tmp_path / 'AutoMoT/qwen3vl_local').mkdir(parents=True)
    with pytest.raises(ValueError, match='exactly one application layout'):
        project_layout(tmp_path)


def test_git_worktree_file_is_supported(tmp_path):
    root, _, _ = fixture_repo(tmp_path, False)
    worktree = tmp_path / 'worktree'
    git(root, 'worktree', 'add', '--detach', str(worktree), 'HEAD')
    module = worktree / 'qwen3vl_local/audit_joint/__main__.py'
    module.parent.mkdir(parents=True)
    module.touch()
    assert (worktree / '.git').is_file()
    assert project_layout(module_file=module) == (worktree, worktree)
    assert git_state(worktree)['availability'] == 'available'


def test_source_copy_without_git_still_produces_blocked_diagnostics(tmp_path):
    root, application, module = fixture_repo(tmp_path, False, with_git=False)
    assert project_layout(module_file=module) == (application, application)
    config = dict(project_root=str(root), source_roots=[str(application / 'qwen3vl_local')],
                  artifacts=[], split_sources=[], model_dir=str(root / 'absent'), phase3_prompt_variant='baseline')
    with patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}):
        manifest, _ = run(config, tmp_path / 'capture')
    assert manifest['git']['availability'] == 'unavailable'
    assert any(b['kind'] == 'git_identity' for b in manifest['blockers'])
    assert verify_bundle(tmp_path / 'capture')['status'] == 'verified'
    assert verify_package(pack(tmp_path / 'capture')['archive'])['status'] == 'verified'


def test_old_parent_root_request_keeps_git_failure_as_blocker(tmp_path):
    _, application, _ = fixture_repo(tmp_path, False)
    config = dict(project_root=str(tmp_path), source_roots=[str(application / 'qwen3vl_local')],
                  artifacts=[], split_sources=[], model_dir=str(application / 'absent'), phase3_prompt_variant='baseline')
    with patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}):
        manifest, _ = run(config, tmp_path / 'capture')
    assert manifest['git']['availability'] == 'unavailable'
    assert any(b['kind'] == 'git_identity' for b in manifest['blockers'])
    assert verify_bundle(tmp_path / 'capture')['status'] == 'verified'


@pytest.mark.parametrize('failure', [FileNotFoundError('git unavailable'),
                                    subprocess.TimeoutExpired(['git'], 30, stderr=b'fixture timeout')])
def test_git_command_failures_are_json_serializable_blockers(tmp_path, failure):
    with patch('qwen3vl_local.audit_joint.baseline.subprocess.check_output', side_effect=failure):
        result = git_state(tmp_path)
    assert result['availability'] == 'unavailable'
    assert result['error']
    json.dumps(result)


def test_empty_git_repository_has_no_fabricated_head(tmp_path):
    git(tmp_path, 'init')
    state = git_state(tmp_path)
    assert state['availability'] == 'unavailable'
    assert 'head' not in state
