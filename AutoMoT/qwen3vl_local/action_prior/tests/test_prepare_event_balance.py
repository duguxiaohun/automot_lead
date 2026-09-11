"""自动数据准备：替换重型数据扫描，保留真实产物校验、内容身份、锁和原子发布。"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior import prepare_event_balance as preparation
from qwen3vl_local.action_prior import event_balance as balance
from qwen3vl_local.action_prior.contracts import file_hash
from qwen3vl_local.sft_new_loop_phase3 import source_mapping


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
                format='sft_new_loop_phase3_frame_index_v3_current_phase',
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


def test_prepare_builds_reuses_and_rebuilds_for_changed_sources(prepared_sources):
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


def test_prepare_rejects_corrupt_cached_mapping(prepared_sources):
    data, collection, cache, _ = prepared_sources
    first = preparation.prepare('raw', data, collection, cache)
    first.write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        preparation.prepare('raw', data, collection, cache)
