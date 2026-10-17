"""Missing production sources stay fatal; contract tests use an explicit isolated tree."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from qwen3vl_local.action_expert_ablation import common, launch, source_check


@pytest.mark.parametrize('variant', ['qwen_simple', 'bev_only'])
def test_missing_sources_rejected_before_weights_or_hashing(variant, ablation_source_tree):
    missing = source_check.EXTERNAL_SOURCES
    for name in missing:
        (ablation_source_tree/name).unlink()
    # No args: accessing model/weights would raise AttributeError first.
    with pytest.raises(FileNotFoundError) as error:
        common.build_contract(SimpleNamespace(), variant)
    assert all(name in str(error.value) for name in missing)
    with pytest.raises(FileNotFoundError):
        common._execution_fingerprint(ablation_source_tree, common.contract_source_paths(variant))


@pytest.mark.parametrize('mode', ['train','eval','preflight'])
def test_launcher_rejects_before_gpu_output_or_subprocess(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['launch',mode,'--variant','bev_only'])
    def forbidden(*args,**kwargs):raise AssertionError('GPU/output/subprocess reached')
    monkeypatch.setattr(launch,'ensure_gpu',forbidden)
    monkeypatch.setattr(launch,'prepare_run_directory',forbidden)
    monkeypatch.setattr(launch,'run_logged',forbidden)
    monkeypatch.setattr(launch.subprocess,'run',forbidden)
    with pytest.raises(FileNotFoundError,match='required sources missing'):
        launch.main()
    assert not list(tmp_path.iterdir())


def test_external_content_is_still_bound_and_removal_is_fatal(ablation_source_tree):
    paths=common.contract_source_paths('bev_only')
    before=common._execution_fingerprint(ablation_source_tree,paths)
    for name in source_check.EXTERNAL_SOURCES:
        path=ablation_source_tree/name
        path.write_text(path.read_text()+'# changed fixture\n')
        after=common._execution_fingerprint(ablation_source_tree,paths)
        assert before['code'][name]!=after['code'][name]
        before=after
    (ablation_source_tree/source_check.EXTERNAL_SOURCES[0]).unlink()
    with pytest.raises(FileNotFoundError):common._execution_fingerprint(ablation_source_tree,paths)


def test_source_cli_is_stdlib_only_and_missing_has_nonzero_exit(tmp_path):
    # -S removes site packages: the precheck cannot require torch or select GPUs.
    command=[sys.executable,'-S',str(Path(source_check.__file__)),'--root',str(tmp_path)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==2
    report=json.loads(result.stdout)
    assert report['status']=='missing_or_unreadable_sources'
    assert all(not item['exists'] for item in report['files'])
    for name in source_check.EXTERNAL_SOURCES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==0
    report=json.loads(result.stdout)
    assert report['status']=='sources_present' and all(item['sha256'] for item in report['files'])
    assert 'not import, model, GPU or end-to-end acceptance' in report['scope']
