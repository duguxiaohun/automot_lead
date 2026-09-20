"""自动数据准备：替换重型数据扫描，保留真实产物校验、内容身份、锁和原子发布。"""
import json
import errno
import multiprocessing
from pathlib import Path
import shutil
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import prepare_event_balance as preparation
from qwen3vl_local.action_prior import event_balance as balance
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.sft_new_loop_phase3 import source_mapping
from qwen3vl_local.sft_new_loop_phase3.build_dataset import FRAME_INDEX_FORMAT


@pytest.fixture
def prepared_sources(tmp_path, monkeypatch):
    """用真实 manifest/index 格式模拟 builder 的最小完整输出。"""
    data, collection, cache = (tmp_path / name for name in ('data', 'collection', 'cache'))
    data.mkdir()
    collection.mkdir()
    (collection / 'Scenario_result.json').write_text('{}')
    for split in ('train', 'val', 'test'):
        (data / f'{split}.jsonl').write_text('{}\n')
    state = {'mapping': 'a' * 64, 'fail_full': False, 'calls': []}
    monkeypatch.setattr(preparation, 'mapping_contract_hash', lambda: state['mapping'])
    monkeypatch.setattr(source_mapping, 'mapping_contract_hash', lambda: state['mapping'])

    def builder(script, arguments):
        """只物化小索引，生产 helper 仍执行自己的校验与发布步骤。"""
        args = dict(zip(arguments[::2], arguments[1::2]))
        out = Path(args['--output-dir'])
        out.mkdir(parents=True)
        state['calls'].append(Path(script).name)
        if Path(script).name == 'build_dataset.py':
            (out / 'frame_index.jsonl').write_text('{}\n')
            row = dict(scenario='S', route_id='R', frame_id=1, context_id='LEAD_BRAKE',
                       mapping_contract_hash=state['mapping'])
            (out / 'candidate_frames.jsonl').write_text(json.dumps(row) + '\n')
            (out / 'candidate_counts.json').write_text(json.dumps({'train/UE1': 1}))
            (out / 'manifest.json').write_text(json.dumps(dict(
                format=FRAME_INDEX_FORMAT,
                frame_index=str(out / 'frame_index.jsonl'), mapping_contract_hash=state['mapping'])))
        else:
            if state['fail_full']:
                (out / '.partial').write_text('interrupted')
                raise RuntimeError('injected full-map build failure')
            row = dict(schema=balance.FULL_INDEX_SCHEMA, mapping_contract_hash=state['mapping'],
                       scenario='S', route_id='R', frame_id=1, special_buckets=['UE1'],
                       eligible_buckets=['UE1'], scene_contexts=['UE1'], status=balance.SPECIAL_ELIGIBLE)
            index = out / 'full_event_mapping.jsonl'
            index.write_text(json.dumps(row) + '\n')
            (out / 'manifest.json').write_text(json.dumps(dict(
                schema=balance.FULL_MANIFEST_SCHEMA, index_schema=balance.FULL_INDEX_SCHEMA,
                mapping_policy=balance.EVENT_BALANCE_MAPPING_POLICY, index_file=index.name,
                index_sha256=file_hash(index), mapping_contract_hash=state['mapping'],
                candidate_sha256=file_hash(args['--candidate-index']),
                action_dataset_hashes={split: file_hash(data / f'{split}.jsonl') for split in ('train', 'val', 'test')},
            )))
    monkeypatch.setattr(preparation, 'run_builder', builder)
    return data, collection, cache, state


def test_prepare_builds_reuses_and_rebuilds_for_changed_sources(prepared_sources, capsys):
    data, collection, cache, state = prepared_sources
    first = preparation.prepare('raw', data, collection, cache)
    assert first.is_file()
    assert state['calls'] == ['build_dataset.py', 'build_event_balance_index.py']
    assert preparation.prepare('raw', data, collection, cache) == first
    assert len(state['calls']) == 2
    (data / 'train.jsonl').write_text('{"changed":true}\n')
    second = preparation.prepare('raw', data, collection, cache)
    assert second != first and first.is_file()
    assert state['calls'][-1] == 'build_event_balance_index.py' and len(state['calls']) == 3
    state['mapping'] = 'b' * 64
    third = preparation.prepare('raw', data, collection, cache)
    assert third not in (first, second)
    assert state['calls'][-2:] == ['build_dataset.py', 'build_event_balance_index.py']
    (collection / 'Scenario_result.json').write_text('{"changed":true}')
    assert preparation.prepare('raw', data, collection, cache) != third
    assert state['calls'][-2:] == ['build_dataset.py', 'build_event_balance_index.py']
    captured = capsys.readouterr()
    assert captured.out == ''
    assert '[prepare] published:' in captured.err and '[prepare] reuse cache:' in captured.err


def test_real_phase3_writer_passes_action_publication_and_reuse(prepared_sources, monkeypatch):
    """只替换原始数据读取；真实Phase3均衡、manifest写入与Action发布必须相容。"""
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as phase3, same_rs_invalid
    from qwen3vl_local.sft_new_loop_phase3.test_build_invalid_quota import candidates
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import longitudinal_decision
    data, collection, cache, state = prepared_sources
    bases, _ = candidates()
    for base in bases:
        base['split'] = 'train'
        base['mapping_contract_hash'] = state['mapping']
        base['action_evidence'].update(
            longitudinal_decision=longitudinal_decision([8, 7, 6, 5, 5, 5, 5, 5, 5]),
            lateral_observation_complete=True, lane_change_direction='')
    monkeypatch.setattr(phase3, 'mapping_contract_hash', lambda: state['mapping'])
    monkeypatch.setattr(phase3, 'iter_base_frames', lambda *a, **k: iter(bases))
    monkeypatch.setattr(same_rs_invalid, 'reviewed_invalid_rows', lambda *a: [])
    monkeypatch.setattr(phase3, 'load_review_coverage', lambda **k: (
        {'synthetic': {'Town01': {'completed_routes': 1}}}, 'synthetic-input-only'))
    original_builder = preparation.run_builder

    def builder(script, arguments):
        """Phase3调用真实构建函数，full-map继续用已有的小数据模拟。"""
        if Path(script).name != 'build_dataset.py':
            return original_builder(script, arguments)
        state['calls'].append('build_dataset.py')
        with monkeypatch.context() as patch:
            patch.setattr(sys, 'argv', [str(script), *map(str, arguments),
                                      '--val-ratio', '0', '--test-ratio', '0'])
            phase3.build_dataset(phase3.parse_args())

    monkeypatch.setattr(preparation, 'run_builder', builder)
    full_path = preparation.prepare('raw', data, collection, cache)
    candidate_dir = next(cache.glob('phase3_*'))
    manifest = json.loads((candidate_dir / 'manifest.json').read_text())
    assert manifest['format'] == FRAME_INDEX_FORMAT
    assert Path(manifest['frame_index']) == candidate_dir / 'frame_index.jsonl'
    assert len(preparation._candidate_membership(candidate_dir / 'candidate_frames.jsonl', state['mapping'])) == len(bases)
    assert preparation.prepare('raw', data, collection, cache) == full_path
    assert state['calls'] == ['build_dataset.py', 'build_event_balance_index.py']


@pytest.mark.parametrize('field,value,message', [
    ('format', 'sft_new_loop_phase3_frame_index_v3_current_phase', 'format mismatch'),
    ('format', 'unknown_future_format', 'format mismatch'),
    ('mapping_contract_hash', 'old-hash', 'mapping hash mismatch'),
    ('frame_index', 'other.jsonl', 'invalid frame-index artifact'),
])
def test_candidate_contract_errors_are_specific(prepared_sources, field, value, message):
    """共用新格式不代表放行旧/未知schema、旧映射或错误的产物文件。"""
    data, collection, cache, state = prepared_sources
    preparation.prepare('raw', data, collection, cache)
    directory = next(cache.glob('phase3_*'))
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        preparation._candidate_membership(directory / 'candidate_frames.jsonl', state['mapping'])


@pytest.mark.parametrize('script_name', ['build_dataset.py', 'build_event_balance_index.py'])
def test_invalid_fresh_output_is_never_published(prepared_sources, monkeypatch, script_name):
    """新构建器输出非法内容应中止，不能套旧缓存的自动修复循环。"""
    data, collection, cache, state = prepared_sources
    original_builder = preparation.run_builder

    def builder(script, arguments):
        """先构建完整小产物，再破坏当前阶段索引来触发正式校验。"""
        original_builder(script, arguments)
        if Path(script).name == script_name:
            args = dict(zip(arguments[::2], arguments[1::2]))
            name = 'candidate_frames.jsonl' if script_name == 'build_dataset.py' else 'full_event_mapping.jsonl'
            (Path(args['--output-dir']) / name).write_text('{}\n')

    monkeypatch.setattr(preparation, 'run_builder', builder)
    with pytest.raises(ValueError):
        preparation.prepare('raw', data, collection, cache)
    assert state['calls'].count(script_name) == 1
    prefix = 'phase3_' if script_name == 'build_dataset.py' else 'full_'
    assert not list(cache.glob(prefix + '*'))
    assert not list(cache.glob('.invalid-*'))
    assert not list(cache.glob('.candidate-*')) and not list(cache.glob('.full-*'))


def test_prepare_failure_does_not_publish_partial_output_and_can_retry(prepared_sources):
    data, collection, cache, state = prepared_sources
    state['fail_full'] = True
    with pytest.raises(RuntimeError, match='injected'):
        preparation.prepare('raw', data, collection, cache)
    assert not list(cache.glob('full_*'))
    assert not list(cache.glob('.full-*'))
    state['fail_full'] = False
    assert preparation.prepare('raw', data, collection, cache).is_file()
    assert state['calls'].count('build_dataset.py') == 1


@pytest.mark.parametrize('stage', ['candidate', 'full'])
@pytest.mark.parametrize('damage', ['row', 'manifest_syntax', 'manifest_shape', 'missing', 'counts_shape'])
def test_prepare_preserves_corrupt_cache_and_rebuilds(prepared_sources, stage, damage):
    """残缺自动缓存不能卡死重跑，原文件保留在隔离目录供排查。"""
    data, collection, cache, state = prepared_sources
    first = preparation.prepare('raw', data, collection, cache)
    corrupt = first if stage == 'full' else next(cache.glob('phase3_*')) / 'candidate_frames.jsonl'
    if damage.startswith('manifest'):
        corrupt = corrupt.with_name('manifest.json')
    if damage == 'counts_shape' and stage == 'candidate':
        corrupt = corrupt.with_name('candidate_counts.json')
    broken = 'null\n' if damage in ('manifest_shape', 'counts_shape') else '{}\n'
    if damage == 'manifest_syntax':
        broken = '{"incomplete":\n'
    if damage == 'missing':
        corrupt.unlink()
    else:
        corrupt.write_text(broken)
    assert preparation.prepare('raw', data, collection, cache) == first
    preserved = list(cache.glob('.invalid-*'))
    assert len(preserved) == 1
    if damage == 'missing':
        assert not (preserved[0] / corrupt.name).exists()
    else:
        assert (preserved[0] / corrupt.name).read_text() == broken
    assert state['calls'].count('build_dataset.py') == (2 if stage == 'candidate' else 1)
    assert state['calls'].count('build_event_balance_index.py') == (2 if stage == 'full' else 1)
    calls = list(state['calls'])
    assert preparation.prepare('raw', data, collection, cache) == first
    assert state['calls'] == calls


@pytest.mark.parametrize('prefix', ['phase3_', 'full_'])
@pytest.mark.parametrize('conflict', ['valid', 'partial', 'different'])
@pytest.mark.parametrize('error_number', [errno.EEXIST, errno.ENOTEMPTY])
def test_publication_collision(prepared_sources, monkeypatch, prefix, conflict, error_number):
    """模拟检查之后、rename 之前的外部发布，覆盖两种系统目录冲突 errno。"""
    data, collection, cache, state = prepared_sources
    original_rename = Path.rename
    injected = []

    def race(source, target):
        """在真实产物校验之后注入完整或残缺目标，保持其余 rename 正常。"""
        target = Path(target)
        if target.name.startswith(prefix) and not injected:
            injected.append(target)
            if conflict == 'partial':
                target.mkdir()
                (target / '.partial').write_text('interrupted external builder')
            else:
                shutil.copytree(source, target)
                if conflict == 'different':
                    name = 'candidate_frames.jsonl' if prefix == 'phase3_' else 'full_event_mapping.jsonl'
                    index = target / name
                    row = json.loads(index.read_text())
                    row['frame_id'] = 2
                    index.write_text(json.dumps(row) + '\n')
                    if prefix == 'full_':
                        manifest = target / 'manifest.json'
                        value = json.loads(manifest.read_text())
                        value['index_sha256'] = file_hash(index)
                        manifest.write_text(json.dumps(value))
            raise OSError(error_number, 'injected directory publication conflict', str(target))
        return original_rename(source, target)

    monkeypatch.setattr(Path, 'rename', race)
    if conflict == 'different':
        with pytest.raises(ValueError, match='valid cache content differs'):
            preparation.prepare('raw', data, collection, cache)
    else:
        assert preparation.prepare('raw', data, collection, cache).is_file()
        assert state['calls'] == ['build_dataset.py', 'build_event_balance_index.py']
    assert len(injected) == 1 and injected[0].is_dir()
    assert len(list(cache.glob('.invalid-*'))) == (1 if conflict == 'partial' else 0)
    assert not list(cache.glob('.candidate-*')) and not list(cache.glob('.full-*'))


@pytest.mark.parametrize('error_number', [errno.EACCES, errno.ENOSPC, errno.EIO])
def test_publication_io_errors_are_not_cache_hits(prepared_sources, monkeypatch, error_number):
    """权限、空间和 I/O 错误必须原样上报，不能吞掉后继续训练。"""
    data, collection, cache, _ = prepared_sources

    def fail_rename(source, target):
        """仅模拟原子发布失败，不改动任何已有目录。"""
        raise OSError(error_number, 'injected I/O failure')

    monkeypatch.setattr(Path, 'rename', fail_rename)
    with pytest.raises(OSError) as error:
        preparation.prepare('raw', data, collection, cache)
    assert error.value.errno == error_number
    assert not list(cache.glob('phase3_*')) and not list(cache.glob('full_*'))


def test_full_map_must_match_current_candidate(prepared_sources):
    """合法格式但绑定其它候选的 full map 也必须保留并重建。"""
    data, collection, cache, state = prepared_sources
    path = preparation.prepare('raw', data, collection, cache)
    manifest = path.with_name('manifest.json')
    value = json.loads(manifest.read_text())
    value['candidate_sha256'] = '0' * 64
    manifest.write_text(json.dumps(value))
    assert preparation.prepare('raw', data, collection, cache) == path
    assert state['calls'].count('build_event_balance_index.py') == 2
    assert len(list(cache.glob('.invalid-full_*'))) == 1


@pytest.mark.parametrize('terminate_first', [False, True])
def test_concurrent_preparation_builds_once(prepared_sources, monkeypatch, terminate_first):
    """真实 flock 竞争：正常等待复用，持锁者被终止后自动释放并重建。"""
    data, collection, cache, _ = prepared_sources
    context = multiprocessing.get_context('fork')
    building, release, waiting = (context.Event() for _ in range(3))
    calls, results = context.Queue(), context.Queue()
    original_builder, original_flock = preparation.run_builder, preparation.fcntl.flock

    def builder(script, arguments):
        """首个构建器暂挂到竞争者明确进入等锁分支。"""
        calls.put(Path(script).name)
        if Path(script).name == 'build_dataset.py':
            first_builder = not building.is_set()
            building.set()
            if terminate_first and first_builder:
                # 不在共享 Event.wait 内杀进程，避免损坏测试同步原语自身的锁。
                signal.pause()
            assert release.wait(10)
        original_builder(script, arguments)

    def flock(handle, operation):
        """保留真实内核锁，只观察非阻塞竞争失败。"""
        try:
            return original_flock(handle, operation)
        except BlockingIOError:
            waiting.set()
            raise

    def worker():
        """通过队列回传跨进程路径或异常供主测试断言。"""
        try:
            results.put(('ok', str(preparation.prepare('raw', data, collection, cache))))
        except Exception as exc:
            results.put(('error', repr(exc)))

    monkeypatch.setattr(preparation, 'run_builder', builder)
    monkeypatch.setattr(preparation.fcntl, 'flock', flock)
    cache.mkdir()
    # 残留锁文件本身不表示仍被持锁，不能靠删除文件来“解锁”。
    (cache / '.prepare.lock').write_text('previous process exited')
    processes = [context.Process(target=worker) for _ in range(2)]
    try:
        processes[0].start()
        assert building.wait(10)
        processes[1].start()
        assert waiting.wait(10)
        if terminate_first:
            processes[0].terminate()
            processes[0].join(timeout=5)
            assert processes[0].exitcode == -15
        release.set()
        output = [results.get(timeout=10) for _ in range(1 if terminate_first else 2)]
        assert all(item == output[0] and item[0] == 'ok' for item in output)
        for process in processes[1:] if terminate_first else processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        expected = ['build_dataset.py'] * (2 if terminate_first else 1) + ['build_event_balance_index.py']
        assert [calls.get(timeout=2) for _ in expected] == expected
        assert len(list(cache.glob('phase3_*'))) == len(list(cache.glob('full_*'))) == 1
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        calls.close()
        results.close()
