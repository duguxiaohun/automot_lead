"""无模型验证预检阶段可观测性及选帧不再全池计算SHA256。"""
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qwen3vl_local.action_prior.comparison_progress import PreflightProgress
from qwen3vl_local.action_prior import comparison_cases as cc


def test_preflight_heartbeat_and_success(tmp_path, capsys):
    progress = PreflightProgress(tmp_path, interval=.01)
    with progress.stage('CPU label loading'):
        deadline = time.monotonic()+2
        while '[preflight] running' not in (tmp_path/'preflight.log').read_text():
            assert time.monotonic() < deadline
            time.sleep(.01)
        current = cc.read_json(tmp_path/'preflight.json')
        assert current['status'] == 'running' and current['location']
    result = cc.read_json(tmp_path/'preflight.json')
    assert result['status'] == 'done' and len(result['completed_stages']) == 1
    assert not any(t.name=='comparison-preflight-progress' for t in threading.enumerate())
    assert 'RSS=' in capsys.readouterr().out


def test_preflight_failure_preserved_and_thread_stopped(tmp_path):
    progress = PreflightProgress(tmp_path, interval=.01)
    with pytest.raises(ValueError, match='bad contract'):
        with progress.stage('contract validation'):
            raise ValueError('bad contract')
    assert cc.read_json(tmp_path/'preflight.json')['status'] == 'failed'
    assert not any(t.name=='comparison-preflight-progress' for t in threading.enumerate())


def test_selection_hashes_only_chosen_cases(monkeypatch):
    rows = [dict(scenario='scene', run_id='run', route_group='group', anchor=i, split='test',
                 action_token=dict(name='STOP', reason='complete_phase3_evidence', version='v1'),
                 event_balance_status='special_eligible', event_balance_all_special_buckets=['UE1']) for i in range(10000)]
    original = cc.case_id
    calls = []
    def counted(row):
        calls.append(row['anchor'])
        return original(row)
    monkeypatch.setattr(cc, 'case_id', counted)
    picked, groups, coverage = cc.select_cases(rows, per_category=3)
    assert len(calls) == 6
    expected = {cid for style in groups.values() for cases in style.values() for cid in cases}
    assert {original(r) for r in picked} == expected
    assert coverage['action']['STOP']['available_frames'] == 10000


def test_prepare_reports_stages_and_keeps_disabled_token_off(tmp_path, monkeypatch):
    from types import SimpleNamespace, ModuleType
    from qwen3vl_local.action_prior import compare_checkpoints as main
    from qwen3vl_local.action_prior import comparison_runtime as runtime
    from qwen3vl_local.action_prior import config, action_token
    from qwen3vl_local.action_prior.contracts import file_hash
    torch = ModuleType('torch')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    mapping = tmp_path/'full.jsonl'
    mapping.write_text('same mapping')
    states = {}
    for i in range(2):
        path = tmp_path/f'run{i}'/'best.pt'
        path.parent.mkdir()
        path.write_text(f'fake checkpoint {i}')
        states[str(path.parent)] = (path, dict(args={}, dataset_hashes=dict(train='t', val='v', test='x')), dict(step=1, score=.1))
    monkeypatch.setattr(main, 'select_checkpoint', lambda run, load: states[run])
    def restored(state, checkpoint, overrides):
        enabled = checkpoint.parent.name == 'run1'
        args = dict(high_level_action_token=enabled, rgb_frame_count=4, seed=12, event_balance_index=str(mapping))
        args.update(dict.fromkeys(('route_points', 'waypoint_points', 'smooth_route', 'frame_interval_s',
            'flow_route_coordinate_scale_m', 'flow_waypoint_coordinate_scale_m', 'flow_sample_steps',
            'route_loss_weight', 'waypoint_loss_weight', 'tp_mode', 'target_point_lookahead_s',
            'next_target_point_lookahead_s', 'tp_min_lookahead_m'), 1))
        return SimpleNamespace(**args), 'bev_only'
    monkeypatch.setattr(runtime, 'restore_args', restored)
    monkeypatch.setattr(runtime, 'check_contract', lambda *args: dict(identity='contract'))
    def tokens(rows):
        for row in rows:
            row.update(action_token=dict(name='STOP', reason='complete_phase3_evidence', version='v1'), action_token_id=2)
    def events(rows):
        for row in rows:
            row.update(event_balance_status='special_eligible', event_balance_all_special_buckets=['UE1'])
    source = SimpleNamespace(full=SimpleNamespace(annotate=events, source=SimpleNamespace(sha256=file_hash(mapping))), annotate=tokens, identity={})
    monkeypatch.setattr(action_token, 'token_source', lambda args: source)
    def read_rows(args, split):
        rows = [dict(scenario='scene', run_id=split, route_group=split, anchor=i, split=split) for i in range(3)]
        events(rows)
        if args.high_level_action_token:
            tokens(rows)
        return rows
    monkeypatch.setattr(config, 'read_rows', read_rows)
    fields = ('data_root', 'data_dir', 'model_dir', 'lead_bev_ckpt', 'event_balance_index', 'high_level_action_index',
              'prior_labels', 'phase1_training_index', 'phase2_training_index')
    cli = SimpleNamespace(**dict.fromkeys(fields, ''), runs=list(states), names=None, label_index='', seed=2026,
                          workers=0, cases_per_category=1, category_counts={}, camera_configuration={}, min_frame_gap=1)
    out = tmp_path/'result'
    manifest, jobs = main.prepare(cli, out)
    for split in ('train', 'test'):
        a, b = [cc.read_json(job['rows'][split]) for job in jobs]
        assert {cc.identity(r) for r in a} == {cc.identity(r) for r in b}
        assert all('action_token' not in r for r in a)
        assert all(r['action_token']['name']=='STOP' for r in b)
    stages = cc.read_json(out/'preflight.json')['completed_stages']
    assert len(stages)==17 and all(s['status']=='done' for s in stages)
    assert (out/'_plan/job_01.json').is_file()
