"""数据文件原子发布的故障注入；不依赖 torch 或训练机挂载。"""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import filesystem as fs


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(fs, 'ESTALE_DELAYS', (0, 0, 0, 0))


@pytest.mark.parametrize('committed', [False, True])
@pytest.mark.parametrize('stale_read', [False, True])
def test_file_publication_handles_uncertain_rename(tmp_path, monkeypatch, committed, stale_read):
    source, target = tmp_path / 'index.tmp', tmp_path / 'index'
    source.write_bytes(b'new complete index')
    target.write_bytes(b'old complete index')
    replace, sha256 = Path.replace, fs._sha256
    faults, reads = [], []

    def unstable_replace(path, destination):
        if not faults:
            faults.append(path)
            if committed:
                replace(path, destination)
            raise OSError(errno.ESTALE, 'rename result uncertain')
        return replace(path, destination)

    def unstable_read(path):
        if stale_read and path == target and not reads:
            reads.append(path)
            raise OSError(errno.ESTALE, 'target lookup stale')
        return sha256(path)

    monkeypatch.setattr(Path, 'replace', unstable_replace)
    monkeypatch.setattr(fs, '_sha256', unstable_read)
    fs.publish_file(source, target)
    assert target.read_bytes() == b'new complete index'
    assert not source.exists()


@pytest.mark.parametrize('code', [errno.ESTALE, errno.EIO, errno.EACCES, errno.ENOSPC, errno.EXDEV])
def test_persistent_error_keeps_old_target_and_complete_temporary(tmp_path, monkeypatch, code):
    source, target = tmp_path / 'index.tmp', tmp_path / 'index'
    source.write_bytes(b'complete new index')
    target.write_bytes(b'complete old index')
    calls = []

    def fail(*args):
        calls.append(1)
        raise OSError(code, 'injected failure')

    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError) as error:
        fs.publish_file(source, target)
    assert error.value.errno == code
    assert len(calls) == (5 if code == errno.ESTALE else 1)
    assert source.read_bytes() == b'complete new index'
    assert target.read_bytes() == b'complete old index'


def test_missing_source_does_not_accept_wrong_target(tmp_path, monkeypatch):
    source, target = tmp_path / 'index.tmp', tmp_path / 'index'
    source.write_bytes(b'new')
    target.write_bytes(b'different')

    def lose_source(*args):
        source.unlink(missing_ok=True)
        raise FileNotFoundError(errno.ENOENT, 'source disappeared')

    monkeypatch.setattr(Path, 'replace', lose_source)
    with pytest.raises(FileNotFoundError):
        fs.publish_file(source, target)
    assert target.read_bytes() == b'different'


def test_atomic_metadata_restarts_partial_write_without_truncating_target(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('old')
    original = Path.write_text
    attempts = []

    def write(target, text, **kwargs):
        assert path.read_text() == 'old'
        attempts.append(target)
        if len(attempts) == 1:
            original(target, 'partial')
            raise OSError(errno.ESTALE, 'partial temporary write')
        return original(target, text, **kwargs)

    monkeypatch.setattr(Path, 'write_text', write)
    fs.atomic_write_text(path, '{"ready":true}')
    assert path.read_text() == '{"ready":true}'
    assert len(attempts) == 2 and attempts[0] == attempts[1]


@pytest.mark.parametrize('committed', [False, True])
def test_phase3_parallel_scan_publication(tmp_path, monkeypatch, committed):
    from collections import Counter
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset, parallel_scan
    rows = [{'scenario': 'S', 'frame_id': i} for i in range(3)]
    scans = Counter()

    def frames(*args, **kwargs):
        scans['calls'] += 1
        return iter(rows)

    original, faults = Path.replace, []

    def replace(source, target):
        if Path(target).name == 'S.jsonl' and not faults:
            faults.append(target)
            if committed:
                original(source, target)
            raise OSError(errno.ESTALE, 'parallel scan publication')
        return original(source, target)

    monkeypatch.setattr(build_dataset, 'iter_base_frames', frames)
    monkeypatch.setattr(Path, 'replace', replace)
    info = parallel_scan.scan_one(({'output_dir': str(tmp_path)}, 'S'))
    assert info['candidates'] == 3 and scans['calls'] == 1
    assert len(Path(info['path']).read_text().splitlines()) == 3
    assert Path(info['path']).with_suffix('.summary.json').is_file()


@pytest.mark.parametrize('variant', ['bev_only', 'qwen_simple'])
@pytest.mark.parametrize('sampling', ['event-balanced', 'action-balanced'])
def test_real_ablation_pipeline_calls_shared_preparer(tmp_path, variant, sampling):
    """执行用户的实际 shell 入口，仅用记录器代替重型 Python 构建/训练。"""
    data, run, binary = (tmp_path / name for name in ('data', 'run', 'bin'))
    for path in (data, run, binary):
        path.mkdir()
    for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        (data / name).write_text('{}\n')
    (run / 'best.pt').write_bytes(b'fixture')
    stub = binary / 'python'
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1].endswith('prepare_event_balance.py'):
    print(os.environ['AUTO_SOURCE'])
''')
    stub.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    full = str(tmp_path / 'full map.jsonl')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('EVENT_', 'ACTION_', 'HIGH_LEVEL_', 'DATA_', 'OUTPUT_', 'RESUME', 'RUN_TAG'))}
    env.update(PATH=str(binary) + os.pathsep + env['PATH'], CALL_LOG=str(log), AUTO_SOURCE=full,
               DATA_DIR=str(data), OUTPUT_DIR=str(run), NO_RUN_SUBDIR='1')
    result = subprocess.run(
        ['bash', f'qwen3vl_local/action_expert_ablation/{variant}/run_full_pipeline.sh', '--' + sampling],
        cwd=Path(__file__).resolve().parents[3], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    prep, = [call for call in calls if call[0].endswith('prepare_event_balance.py')]
    assert prep[prep.index('--action-data-dir') + 1] == str(data)
    training, evaluation = [call for call in calls if call[0].endswith('launch.py')]
    for call in (training, evaluation):
        assert call[call.index('--event-balance-index') + 1] == full
