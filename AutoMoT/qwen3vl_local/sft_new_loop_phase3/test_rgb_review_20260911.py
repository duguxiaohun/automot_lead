"""收到4RGB结果后，逐帧证据对应的时间边界、短prompt和开发集隔离回归。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.audit_temporal_slices import timing_diagnostics
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    SYSTEM_PROMPT, build_action_messages, build_action_prompt, build_action_target,
    make_prompt_spec, parse_action_output)
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts


def test_stop_first_sample_at_deadline_is_not_confirmation():
    # #590：首次近零1.5s，第二次1.75s；#699则1.25s/1.5s均已近零。
    late = timing_diagnostics([7.962, 8.006, 8.729, 8.058, 7.139, 4.395, 0, 0, 0])
    inside = timing_diagnostics([6.729, 1.869, 1.086, 1.013, .559, .153, .004, .003, .365])
    assert late['stop_start_s'] == 1.5 and late['stop_confirmed_s'] == 1.75
    assert late['stop_pair_crosses_1_5s_boundary']
    assert inside['stop_confirmed_s'] == 1.5
    assert not inside['stop_pair_crosses_1_5s_boundary']


def test_small_real_drop_and_isolated_control_pulse_are_separate():
    # #88：8.749→7.557的实际下降不够20%；#422先单点降速再持续加速。
    small = timing_diagnostics([8.749, 7.557, 9, 10, 11, 12, 13, 14, 14.21])
    pulse = timing_diagnostics([6.015, 2.618, 4.420, 5.380, 6.892, 8.295, 9.044, 8.311, 7.588])
    assert small['subthreshold_drop_present'] and small['first_drop_s'] is None
    assert pulse['first_drop_s'] == .25
    # 第二点4.420也超过减速幅度，不能误称只有一帧达到阈值。
    assert not pulse['first_drop_single_sample']
    with pytest.raises(ValueError):
        timing_diagnostics([0] * 8)


@pytest.mark.parametrize('context_id', CONTEXT_BY_ID)
def test_compact_prompt_preserves_schema_and_causal_inputs(context_id):
    context = CONTEXT_BY_ID[context_id]
    spec = make_prompt_spec(variant='all_random_order', answers={}, seed_key='review',
        context_id=context_id, road_structure=context.allowed_rs[0],
        goal_xy=(42, -3), current_speed_mps=8.125)
    prompt = build_action_prompt(spec=spec)
    assert len(SYSTEM_PROMPT.split()) <= 20
    assert len((SYSTEM_PROMPT + ' ' + prompt).split()) <= 400
    assert '1.5 seconds' in prompt and 'two consecutive' in prompt
    assert 'max(1.2 m/s, 20%)' in prompt
    assert '8.125 m/s' in prompt and 'y=-3.0 m' in prompt
    assert 'negative LEFT, positive RIGHT' in prompt
    assert '[VISUAL_CHECK_ORDER]' not in prompt and '[QUESTIONS]' not in prompt
    assert 'future_speeds' not in prompt and 'lane_change_direction' not in prompt
    assert 'Slow or wait while' not in prompt and 'must give' not in prompt
    if 'LANE_CHANGE_LEFT' not in spec.output_keys:
        assert 'LANE_CHANGE_LEFT' not in prompt and 'Steering input' not in prompt
    else:
        assert 'FIRST crossing' in prompt and 'crossings already in the input' in prompt
    target = build_action_target(spec)
    assert all(v is False for v in parse_action_output(target, spec=spec).values())
    evidence = '\n'.join('EVIDENCE_' + k + ': unclear' for k in spec.output_keys)
    assert all(v is False for v in parse_action_output(target+'\n'+evidence, spec=spec, audit=True).values())
    messages = build_action_messages(images=['a','b','c','d'], spec=spec)
    assert [x['image'] for x in messages[1]['content'] if x['type']=='image'] == ['a','b','c','d']
    endpoint = build_action_prompt(spec=spec, history_rgb_mode='2rgb_endpoints')
    assert 'two endpoint frames' in endpoint and 't-0.50' not in endpoint


def test_quarantine_only_visually_reviewed_span_and_route():
    route = 'Town04_Rep0_Town04_Scenario4_119_route0_01_08_23_44_07'
    for frame in (2, 5, 17, 21, 33):
        assert mapped_contexts('VehicleTurningRoute', route, frame, 'R5', 'R-E5', ['R-E5'])[0] == ()
    assert mapped_contexts('VehicleTurningRoute', route, 34, 'R5', 'R-E5', ['R-E5'])[0]
    assert mapped_contexts('VehicleTurningRoute', 'unreviewed', 5, 'R5', 'R-E5', ['R-E5'])[0]
    route = 'Town13_Rep0_1777_0_route0_01_10_22_42_02'
    assert mapped_contexts('InvadingTurn', route, 78, 'R2', 'U-E5', ['U-E5'])[0] == ()
    assert mapped_contexts('InvadingTurn', route, 79, 'R2', 'U-E5', ['U-E5'])[0]


def test_new_review_routes_and_repetitions_are_train_only():
    notes = Path(__file__).with_name('EVAL_RGB_REVIEW_20260911.jsonl')
    for line in notes.read_text().splitlines():
        row = json.loads(line)
        assert _split(row['scenario'], row['route_id'], 20260819, .1, .05) == 'train'
        assert _split(row['scenario'], row['route_id'].replace('_Rep0_', '_Rep99_'), 1, .99, .01) == 'train'


def test_blind_negative_labels_are_separate_from_error_driven_development():
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group, development_route_groups
    from qwen3vl_local.sft_new_loop_phase3.prompts import action_prompt_sha256
    path = Path(__file__).with_name('same_rs_invalid_review_v1.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    blind = [r for r in rows if r.get('review_purpose') == 'label_only_for_new_holdout_after_prompt_frozen']
    assert len(blind) == 3
    groups = set()
    for row in blind:
        group = physical_route_group(row['scenario'], row['route_id'])
        groups.add(group)
        assert group not in development_route_groups()
        assert row['model_outputs_inspected'] is False
        assert row['frozen_prompt_sha256'] == action_prompt_sha256()
        assert _split(row['scenario'], row['route_id'], 20260911, .1, .05) == 'val'
        assert len(row['original_rgb_frames']) == len(row['input_rgb_sha256']) == 4
    assert len(groups) == 3
