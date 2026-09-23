"""稀少动作容量回流：不改变真值，不为补齐配额额外放大稀少格子。"""
import random
from collections import Counter
import pytest
from qwen3vl_local.sft_new_loop_phase3.sampling import support_aware_quota


def test_rare_capacity_returns_to_supported_action():
    assert support_aware_quota({"KEEP": 4, "STOP": 1000}, 100) == {"KEEP": 4, "STOP": 96}
    assert support_aware_quota({"KEEP": 4, "STOP": 1000}, 1500) == {"KEEP": 8, "STOP": 1492}


def test_full_pool_cycles_bound_every_action_repetition():
    for target in range(501):
        capacities = {"a": 1, "b": 4, "c": 20}
        q = support_aware_quota(capacities, target)
        assert sum(q.values()) == target
        assert all(q[k] <= n * ((target + 24) // 25) for k, n in capacities.items())
    with pytest.raises(ValueError):
        support_aware_quota({}, 1)


def test_phase3_builder_does_not_relabel_or_add_random_fallback_repeats():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario="s", route_id=f"r{i}", frame_id=i,
                 action_labels={k: k == "STOP" and i > 0 for k in ACTION_KEYS}) for i in range(101)]
    selected, audit = _sample_context_bucket(rows, context_id="LEAD_BRAKE", target=150,
                                             rng=random.Random(1), route_diverse=True)
    counts = Counter(r["frame_id"] for r in selected)
    assert len(selected) == 150 and counts[0] == 2
    assert max(counts.values()) <= 2
    assert sum(audit["signature_quota"].values()) == 150
    assert not any(rows[0]["action_labels"].values())


def test_signature_support_uses_physical_routes_and_does_not_relabel():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario='s', route_id=f'Town13_Rep{i}_1710_7_route0', frame_id=i,
                 action_labels={k: k == 'STOP' for k in ACTION_KEYS}) for i in range(3)]
    selected, audit = _sample_context_bucket(rows, context_id='LEAD_BRAKE', target=2,
                                             rng=random.Random(1), route_diverse=True)
    support = audit['signature_support']['STOP']
    assert support['frames'] == 3 and support['physical_routes'] == 1
    assert support['presentations'] == 2 and support['diagnostic_only']
    assert all(r['action_labels']['STOP'] for r in selected)


def test_compound_is_not_a_separate_quota_and_binary_evidence_survives():
    from copy import deepcopy
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _sample_context_bucket
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    rows = [dict(scenario='s', route_id='same_route', frame_id=i,
                 action_labels={k: k == 'LANE_CHANGE_LEFT' or (k == 'RESUME' and i == 0)
                                for k in ACTION_KEYS}) for i in range(100)]
    original = deepcopy(rows)
    selected, audit = _sample_context_bucket(rows, context_id='RAMP_MERGE_EXIT', target=150,
                                             rng=random.Random(1), route_diverse=True)
    assert audit['primary_action_quota'] == {'LANE_CHANGE_LEFT': 150}
    assert sum(r['action_labels']['RESUME'] for r in selected) <= 2
    assert rows == original and all(r['action_labels']['LANE_CHANGE_LEFT'] for r in selected)


def test_repeated_recordings_do_not_crowd_out_other_physical_routes():
    from qwen3vl_local.sft_new_loop_phase3.sampling import route_diverse_sample, route_diversity_report
    rows = [dict(scenario='s',route_id=f'Town13_Rep{i}_1710_7_route0',frame_id=4) for i in range(20)]
    rows.append(dict(scenario='s',route_id='Town13_Rep0_9999_7_route0',frame_id=4))
    selected = route_diverse_sample(rows,target=2,rng=random.Random(4))
    assert any('9999' in r['route_id'] for r in selected)
    assert route_diversity_report(rows)['unique_routes']==2


def test_hierarchical_route_event_action_sampling():
    from qwen3vl_local.sft_new_loop_phase3.sampling import (
        route_event_action_sample,
        support_aware_quota,
        route_diverse_cursor_sample,
        hierarchical_event_action_epoch_sample,
    )
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
    from qwen3vl_local.sft_new_loop_phase3.train import _make_item, FrameRow

    # 1. 验证 support_aware_quota 在 smooth_cap 模式下的平滑提权与容量回流
    # KEEP 4 帧 (极小动作), STOP 1000 帧 (极大动作), 预算 100
    q = support_aware_quota({"KEEP": 4, "STOP": 1000}, 100, repeat_cap=2, smooth_power=0.5, mode="smooth_cap")
    assert sum(q.values()) == 100
    assert q["KEEP"] <= 8 and q["KEEP"] > 0
    assert q["STOP"] == 100 - q["KEEP"]

    # 验证问题 3: 超容量时必须明确报错 ValueError，绝不允许突破 repeat_cap 硬上限
    # 2 帧，上限 2 (最大容量 4)，请求 5 帧 -> 必须 raise ValueError
    with pytest.raises(ValueError, match="smooth_cap quota infeasible"):
        support_aware_quota({"A": 1, "B": 1}, 5, repeat_cap=2, mode="smooth_cap")

    # 2. 验证问题 4: 跨轮游标滑动保证不放回与 100% 覆盖
    # 10 帧池，两轮各取 5 帧，即使外部种子不同，由于同一周期内主队列稳定，两轮必须精确覆盖全部 10 帧 (0 重复)
    pool_10 = [dict(scenario="s", route_id=f"r_{i}", frame_id=i) for i in range(10)]
    s1, c1 = route_diverse_cursor_sample(pool_10, target=5, cursor=0, master_seed="test_pool")
    assert len(s1) == 5
    assert c1 == 5
    s2, c2 = route_diverse_cursor_sample(pool_10, target=5, cursor=c1, master_seed="test_pool")
    assert len(s2) == 5
    assert c2 == 10
    union_frames = {r["frame_id"] for r in s1} | {r["frame_id"] for r in s2}
    assert len(union_frames) == 10  # 100% 完整覆盖，无任何放回重复！

    # 3. 验证问题 6: 兼容真实 Phase3 对象 (WorkItem / FrameRow / dict)
    dummy_answers = {k: (k == "STOP") for k in ACTION_KEYS}
    dummy_row = FrameRow(
        scenario="s",
        route_id="Town01_route0",
        town="Town01",
        frame_id=1,
        true_rs="R1",
        prompt_road_structure="R1",
        context_id="LEAD_BRAKE",
        question_domain="LONGITUDINAL",
        action_signature="STOP",
        event="LEAD_BRAKE",
        split="train",
        goal_ego_xy=(0.0, 10.0),
        history_rgb_paths=["rgb.jpg"],
        latest_rgb_path="rgb.jpg",
        answers=dummy_answers,
    )
    dummy_work_item = _make_item(dummy_row, seed=0, action_output_mode="binary")
    dummy_choice_item = _make_item(dummy_row, seed=0, action_output_mode="choice")

    # 传 FrameRow
    sel_row, _ = route_event_action_sample([dummy_row], context_id="LEAD_BRAKE", target=1, rng=random.Random(1))
    assert len(sel_row) == 1
    # 传 WorkItem (binary)
    sel_item, _ = route_event_action_sample([dummy_work_item], context_id="LEAD_BRAKE", target=1, rng=random.Random(1))
    assert len(sel_item) == 1
    # 传 WorkItem (choice)
    sel_choice, _ = route_event_action_sample([dummy_choice_item], context_id="LEAD_BRAKE", target=1, rng=random.Random(1))
    assert len(sel_choice) == 1

    # 4. 验证问题 5: 跨 Event 联合采样严格保证全 Epoch 单帧上限
    # 构造支持两个事件的共享帧
    shared_frame_1 = dict(scenario="s", route_id="r1", frame_id=101, action_labels={k: (k == "STOP") for k in ACTION_KEYS})
    shared_frame_2 = dict(scenario="s", route_id="r2", frame_id=102, action_labels={k: (k == "STOP") for k in ACTION_KEYS})
    event_items = {
        "LEAD_BRAKE": [shared_frame_1, shared_frame_2],
        "OBSTACLE_CROSS_TWO_WAY": [shared_frame_1, shared_frame_2],
    }
    # 若 repeat_cap=2, 两个事件各自要 4 帧 (总共需 8 帧，而两帧总容量只有 4 帧) -> 必须抛出容量不足
    with pytest.raises(ValueError, match="insufficient shared frame capacity"):
        hierarchical_event_action_epoch_sample(
            event_items,
            {"LEAD_BRAKE": 4, "OBSTACLE_CROSS_TWO_WAY": 4},
            repeat_cap=2,
            master_seed=42,
        )

    # 在合法配额下 (如各要 2 帧，总共 4 帧正好等于 2*2):
    joint_sel, joint_audit = hierarchical_event_action_epoch_sample(
        event_items,
        {"LEAD_BRAKE": 2, "OBSTACLE_CROSS_TWO_WAY": 2},
        repeat_cap=2,
        master_seed=42,
    )
    assert len(joint_sel) == 4
    assert joint_audit["max_frame_repeat"] <= 2


@pytest.mark.parametrize('cursor,cap,target', [(5, 1, 10), (9, 2, 20), (99, 1, 10)])
def test_epoch_boundary_cap_and_canonical_input(cursor, cap, target):
    from qwen3vl_local.sft_new_loop_phase3.sampling import route_event_action_sample
    rows = [dict(scenario='s', route_id='r', frame_id=i, action_token={'name': 'STOP'}) for i in range(10)]
    kw = dict(context_id='E', target=target, repeat_cap=cap,
              cursor_state={'E/STOP': cursor}, master_seed=3)
    first, audit = route_event_action_sample(rows, rng=random.Random(1), **kw)
    second, _ = route_event_action_sample(rows[::-1], rng=random.Random(2), **kw)
    assert Counter(r['frame_id'] for r in first) == Counter(r['frame_id'] for r in second)
    assert max(Counter(r['frame_id'] for r in first).values()) <= cap
    assert audit['next_cursors']['E/STOP'] == cursor + target


def test_joint_reassigns_shared_frames_and_returns_action_overflow():
    from qwen3vl_local.sft_new_loop_phase3.sampling import hierarchical_event_action_epoch_sample as sample
    shared = dict(scenario='s', route_id='r', frame_id=0, action_token={'name': 'STOP'})
    own = dict(shared, frame_id=1, action_token={'name': 'KEEP'})
    for seed in range(10):
        # A 的初始 STOP 配额与 B 冲突，必须改分 A/KEEP；不允许贪心拒绝可行预算。
        chosen, audit = sample({'A': [shared, own], 'B': [shared]}, {'A': 1, 'B': 1},
                               repeat_cap=1, master_seed=seed)
        assert len(chosen) == 2 and audit['max_frame_repeat'] == 1
        assert dict(zip((c[0] for c in audit['selected_cells']), (r['frame_id'] for r in chosen))) == {'A': 1, 'B': 0}
        assert audit['event_audits']['A']['sampled_actions'] == {'KEEP': 1, 'STOP': 0}


def test_joint_feasibility_matches_exhaustive_assignment():
    import itertools
    from qwen3vl_local.sft_new_loop_phase3.sampling import hierarchical_event_action_epoch_sample as sample
    rng = random.Random(11)
    rows = [dict(scenario='s', route_id='r', frame_id=i, action_token={'name': 'STOP'}) for i in range(4)]
    for _ in range(60):
        pools = {e: [r for r in rows if rng.random() < .6] for e in ('A', 'B', 'C')}
        targets = {e: rng.randrange(3) for e in pools}
        cap = rng.randrange(1, 3)
        slots = [e for e, n in targets.items() for _ in range(n)]
        solutions = [assignment for assignment in itertools.product(*[[r['frame_id'] for r in pools[e]] for e in slots])
                     if max(Counter(assignment).values(), default=0) <= cap]
        if not solutions:
            with pytest.raises(ValueError):
                sample(pools, targets, repeat_cap=cap)
        else:
            chosen, audit = sample(pools, targets, repeat_cap=cap)
            assert len(chosen) == sum(targets.values())
            assert audit['unique_frames'] == max(len(set(a)) for a in solutions)


def test_queue_constructed_once_per_membership_pool(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import sampling
    original = sampling.deterministic_cycle_order
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sampling, 'deterministic_cycle_order', counted)
    rows = [dict(scenario='s', route_id='r', frame_id=i, action_token={'name': 'STOP'}) for i in range(100)]
    sampling.hierarchical_event_action_epoch_sample({'A': rows}, {'A': 150}, repeat_cap=2)
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['binary', 'choice'])
def test_real_phase3_training_covers_full_pool_across_epochs(mode):
    from dataclasses import replace
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    base = candidate_rows()
    positives = [r for r in base if not r.invalid_source]
    rows = [replace(row, frame_id=i + 10000) for row in positives for i in range(100)]
    if mode == 'binary':
        rows += [r for r in base if r.invalid_source]
    cursors, seen, history = {}, set(), {}
    for epoch in range(2):
        audit = {}
        work, cursors = train._balanced_work(
            rows if epoch == 0 else rows[::-1], target_per_bin=50, seed=epoch * 1000003,
            master_seed=17, action_output_mode=mode, mode='smooth_cap', repeat_cap=1,
            cursor_state=cursors, return_cursors=True, sampling_audit=audit, route_diverse=True,
            pool_history=history,
        )
        assert audit['pool_history_start'] == history
        history = audit['next_pool_history']
        assert all(value['presentations'] == cursors[key] for key, value in history.items())
        counts = Counter((w.row.route_id, w.row.frame_id) for w in work)
        assert max(counts.values()) == 1
        seen.update((w.row.context_id, w.row.frame_id) for w in work if not w.row.invalid_source)
    assert len(seen) == 1000


def test_reserved_invalid_uses_real_frame_capacity():
    from qwen3vl_local.sft_new_loop_phase3.sampling import hierarchical_event_action_epoch_sample as sample, _item_identity
    rows = [dict(scenario='s', route_id='r', frame_id=i, action_token={'name': 'STOP'}) for i in range(2)]
    chosen, audit = sample({'A': rows}, {'A': 1}, repeat_cap=1,
                           initial_frame_usage={_item_identity(rows[0]): 1})
    assert chosen[0]['frame_id'] == 1 and audit['max_frame_repeat'] == 1
    with pytest.raises(ValueError, match='insufficient shared frame capacity'):
        sample({'A': rows}, {'A': 2}, repeat_cap=1, initial_frame_usage={_item_identity(rows[0]): 1})


@pytest.mark.parametrize('kwargs', [dict(repeat_cap=0), dict(smooth_power=float('nan')),
                                   dict(smooth_power=-1), dict(smooth_power=2), dict(mode='typo')])
def test_invalid_sampling_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        support_aware_quota({'A': 3}, 2, **kwargs)


def test_phase3_invalid_and_positive_are_allocated_jointly():
    from dataclasses import replace
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
    rows = candidate_rows(0)
    positives = [r for r in rows if not r.invalid_source]
    shared = positives[0]
    negatives = []
    for r in rows:
        if r.invalid_source:
            negatives.extend([replace(r, route_id=shared.route_id, frame_id=shared.frame_id),
                              replace(r, route_id='alternative', frame_id=999)])
    for seed in range(8):
        work = train._balanced_work(
            positives + negatives, target_per_bin=1, invalid_multiplier=1,
            require_invalid_coverage=False, seed=seed, mode='smooth_cap', repeat_cap=1,
        )
        assert max(Counter((w.row.route_id, w.row.frame_id) for w in work).values()) == 1
        assert len(work) == 11
        assert next(w for w in work if w.row.invalid_source).row.route_id == 'alternative'


def test_full_training_pool_load_hash_and_validation_isolation(tmp_path):
    import json
    from collections import defaultdict
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder, train
    from qwen3vl_local.sft_new_loop_phase3.test_build_invalid_quota import candidates
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
    from qwen3vl_local.sft_new_loop_phase3.preflight import training_pool_path
    bases, _ = candidates(10)
    buckets = defaultdict(list)
    for base in bases:
        base.update(split='train')
        base['action_evidence'].update(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256(),
                                      longitudinal_decision=dict(eligible=True, action='DECELERATE'),
                                      lateral_observation_complete=True, lane_change_direction='')
        buckets[base['context_id']].append(base)
    info = builder._write_training_pool(buckets, [], tmp_path, 5)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(training_pool=info)))
    full_path = training_pool_path(tmp_path / 'frame_index.jsonl')
    # 原均衡索引只有一个样本，新训练池必须能读出全部100个，验证仍读原文件。
    one = json.loads(full_path.read_text().splitlines()[0])
    index = tmp_path / 'frame_index.jsonl'
    index.write_text(json.dumps(dict(one, split='val')) + '\n')
    loaded = train._read_rows(index, split='train', training_pool=True)
    assert len(loaded) == info['positive_rows'] == 100
    assert len(train._read_rows(index, split='val')) == 1
    with pytest.raises(ValueError, match='train-only'):
        train._read_rows(index, split='val', training_pool=True)
    full_path.write_text(full_path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        train._read_rows(index, split='train', training_pool=True)


def test_adapter_and_run_manifest_record_actual_sampling_settings(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3 import train
    monkeypatch.setattr(sys, 'argv', ['train.py', '--sampling-repeat-cap', '2', '--sampling-smooth-power', '.7'])
    args = train.parse_args()
    args.training_pool_identity = dict(sha256='test-hash', rows=100)
    assert args.sampling_policy == 'smooth_cap'
    train._write_run_metadata(None, tmp_path, args, world_size=1, train_rows=100,
                              train_work_global=50, train_work_rank=50, eval_work_rank=1,
                              generation_eval_global=1, total_steps=5)
    # Save a stub adapter through the real metadata writer, without model weights.
    bundle = SimpleNamespace(unwrap=lambda: SimpleNamespace(save_pretrained=lambda path: None),
                             lora_target_modules=['q_proj'])
    train._save_adapter(bundle, tmp_path, args, step=5, name='final')
    run = json.loads((tmp_path / 'train_run_manifest.json').read_text())
    adapter = json.loads((tmp_path / 'final/sft_new_loop_phase3_adapter_config.json').read_text())
    assert run['sampling_config'] == adapter['sampling_config']
    assert run['sampling_config']['repeat_cap'] == 2
    assert run['sampling_config']['smooth_power'] == .7
    assert run['sampling_config']['training_pool_identity']['sha256'] == 'test-hash'


def test_fairness_never_overrides_primary_quota_or_unique_frame_objectives():
    """给不该获优先权的池极大等待债务，仍不能牺牲动作配额/唯一帧数。"""
    from qwen3vl_local.sft_new_loop_phase3.sampling import hierarchical_event_action_epoch_sample as sample
    rows = [dict(scenario='s', route_id='r', frame_id=i, action_token={'name': 'STOP'}) for i in range(3)]
    rows.append(dict(scenario='s', route_id='r', frame_id=3, action_token={'name': 'KEEP'}))
    cursor = {'A/STOP': 1000000, 'A/KEEP': 0}
    history = {'A/STOP': dict(presentations=1000000, skipped_epochs=0),
               'A/KEEP': dict(presentations=0, skipped_epochs=1000000)}
    _, audit = sample({'A': rows}, {'A': 4}, repeat_cap=2, cursor_state=cursor, pool_history=history)
    # sqrt配额STOP3 KEEP1恰能4帧全用，不能为了未曝光KEEP重复它、牺牲STOP。
    assert audit['action_quota_overflow'] == 0 and audit['unique_frames'] == 4
    assert audit['event_audits']['A']['sampled_actions'] == {'KEEP': 1, 'STOP': 3}
    with pytest.raises(ValueError, match='history/cursor mismatch'):
        sample({'A': rows}, {'A': 4}, repeat_cap=2, pool_history=history)
