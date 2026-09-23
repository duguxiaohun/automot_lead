"""实际训练池新增帧必须经过RGB与原始meta核验，重复帧不重复读取meta。"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from qwen3vl_local.sft_new_loop_phase3 import audit_raw_index as raw, audit_rebuilt_index as rebuilt
from qwen3vl_local.sft_new_loop_phase3 import prompts
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.test_rgb_review_20260916 import index_rows


def write_pool(index, rows):
    path = index.with_name('train_sampling_pool.jsonl')
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = dict(training_pool=dict(file=path.name, rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    prompt_contract=dict(prompt_name=prompts.PROMPT_NAME, production_prompt_sha256={
                        mode: {rgb: prompts.action_prompt_sha256(audit=False, history_rgb_mode=rgb, action_output_mode=mode)
                               for rgb in ('4rgb', '2rgb_endpoints')} for mode in ('binary', 'choice')}))
    index.with_name('manifest.json').write_text(json.dumps(manifest))


@pytest.fixture
def data(tmp_path, monkeypatch):
    rows = index_rows(0)
    for row in rows:
        row.update(true_rs='R1', goal_ego_xy=[10, 0], context_detail='',
                   action_signature='INVALID' if row['invalid_action_context'] else 'KEEP')
        if row['invalid_action_context']:
            row['invalid_reason'] = 'same_rs_wrong_event'
        row['history_rgb_paths'] = [f"{row['route_id']}/{row['frame_id']}.jpg"] * 4
        row['latest_rgb_path'] = row['history_rgb_paths'][0]
        row['action_evidence'].update(future_speeds_exact_mps=[3.] * 9, future_speeds_mps=[3.] * 9)
    index = tmp_path / 'frame_index.jsonl'
    index.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    new = copy.deepcopy(rows[0])
    new.update(route_id='pool_only', frame_id=100, history_rgb_paths=['new.jpg'] * 4, latest_rgb_path='new.jpg')
    variant = copy.deepcopy(new)
    variant['context_id'] = CONTEXT_IDS[1]
    variant.update(choice_annotation(variant['answers'], variant['context_id'], variant['action_evidence']))
    pool = [r for r in rows if r['split'] == 'train'] + [new, copy.deepcopy(new), variant]
    write_pool(index, pool)
    for r in rows + pool:
        for name in r['history_rgb_paths']:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'RGB fixture; existence audit does not decode pixels')
    loads, signal_calls = [], []
    def load(directory):
        loads.append(str(directory))
        def signals(frame):
            signal_calls.append((str(directory), frame))
            return dict(future_speeds=[3.] * 9, future_speed_count=9, lane_change_direction='',
                        lateral_observation_complete=True)
        return SimpleNamespace(signals=signals)
    monkeypatch.setattr(raw, 'is_abnormal_lead_route', lambda *args: (False, ''))
    monkeypatch.setattr(raw, 'load_route_trajectory', load)
    return SimpleNamespace(index=index, root=tmp_path, pool=pool, loads=loads, calls=signal_calls)


@pytest.mark.parametrize('module', [rebuilt, raw])
def test_actual_pool_and_holdout_coverage_with_frame_io_dedup(data, module):
    report = module.audit(data.index, data.root, include_training_pool=True)
    coverage = report['input_coverage']
    assert coverage['scope'] == 'training_pool_plus_original_holdout'
    assert set(coverage['sources']) == {'training_pool', 'index_val', 'index_test'}
    assert coverage['sources']['training_pool'] == dict(rows=14, unique_frames=12, unique_cases=13)
    assert coverage['sources']['index_val']['unique_frames'] == 11
    assert coverage['sources']['index_test']['unique_frames'] == 11
    assert coverage['unique_frames'] == 34 and coverage['identical_rows_reused'] == 1
    assert coverage['row_variants_verified'] == 35
    if module is raw:
        assert report['raw_rows_verified'] == 35 and report['raw_frames_verified'] == 34
        assert max(Counter(data.calls).values()) == 1
    else:
        assert report['rgb_files_checked'] == 34
        assert report['rgb_files_by_source'] == dict(training_pool=12, index_val=11, index_test=11)


def test_missing_rgb_only_in_training_pool_fails_before_training(data):
    (data.root / 'new.jpg').unlink()
    with pytest.raises(FileNotFoundError, match='new.jpg'):
        rebuilt.audit(data.index, data.root, include_training_pool=True)


def test_raw_evidence_conflict_only_in_training_pool_is_not_hidden_by_dedup(data):
    # 同一case的第一个副本正确，第二个副本被改坏；不能按帧或case丢弃冲突证据。
    data.pool[-2]['action_evidence']['future_speeds_exact_mps'][0] = 3.01
    write_pool(data.index, data.pool)
    with pytest.raises(ValueError, match='raw speed mismatch: pool_only/100'):
        raw.audit(data.index, data.root, include_training_pool=True)


@pytest.mark.parametrize('module', [rebuilt, raw])
def test_declared_pool_is_automatic_and_explicit_pool_cannot_fall_back(data, module):
    assert module.audit(data.index, data.root)['input_coverage']['sources']['training_pool']['rows'] == 14
    path = data.index.with_name('manifest.json')
    manifest = json.loads(path.read_text())
    manifest.pop('training_pool')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hashed training_pool'):
        module.audit(data.index, data.root, include_training_pool=True)


def test_pipeline_requires_pool_for_both_pretraining_audits():
    source = Path(__file__).with_name('run_full_pipeline.sh').read_text()
    for script in ('audit_rebuilt_index.py', 'audit_raw_index.py'):
        command = source.split(script, 1)[1].split('\n\n', 1)[0]
        assert '--include-training-pool' in command
