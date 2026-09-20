"""自动错误RS组合与稀疏人工负例不阻塞训练的无torch回归。"""
from collections import Counter
import json
import random
from types import SimpleNamespace

import pytest

from qwen3vl_local.sft_new_loop_phase3 import build_dataset as builder, preflight
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS, CONTEXT_IDS, CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (
    balanced_invalid_items, invalid_subgroup_report, mismatched_road_contexts,
    _sample_prompt_roads,
)
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation, PRIMARY_CHOICE_VERSION
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import longitudinal_decision
from qwen3vl_local.sft_new_loop_phase3.test_build_invalid_quota import candidates
from qwen3vl_local.sft_new_loop_phase3.test_repair_contract import _healthy_metrics, _guard
from qwen3vl_local.sft_new_loop_phase3.quality_guards import generation_checkpoint_score


@pytest.mark.parametrize('rs,junction,distance,roads', [
    ('R1', False, 50., {'R3', 'R4', 'R5'}),
    ('R2', False, 50., {'R3', 'R4', 'R5'}),
    ('R3', False, 50., {'R4', 'R5'}),
    ('R4', True, 0., {'R3'}), ('R5', True, 0., {'R3'}),
    ('R1', False, 24., set()), ('R2', True, 0., set()),
    ('R3', True, 0., set()), ('UNKNOWN', False, 50., set()),
])
def test_all_safe_rs_event_pairs_and_no_unproven_pairs(rs, junction, distance, roads):
    pairs = mismatched_road_contexts(true_rs=rs, is_junction=junction,
                                     distance_to_next_junction=distance)
    assert set(pairs) == {(ctx, fake) for ctx in CONTEXT_IDS for fake in roads
                          if fake in CONTEXT_BY_ID[ctx].allowed_rs}
    assert len(pairs) == len(set(pairs))
    assert all(fake != rs for _, fake in pairs)


def test_prompt_road_sampling_balances_seed_and_unequal_candidate_counts():
    items = [SimpleNamespace(prompt_road_structure=rs, frame_id=i)
             for rs, count in [('R3', 1), ('R4', 90), ('R5', 10)] for i in range(count)]
    for seed in range(20):
        initial = [items[0]]
        sampled = initial + _sample_prompt_roads(items, 11, random.Random(seed), initial)
        assert Counter(r.prompt_road_structure for r in sampled) == {'R3': 4, 'R4': 4, 'R5': 4}


@pytest.mark.parametrize('manual', [0, 1])
def test_build_then_runtime_sampling_retains_required_coverage_with_sparse_manual(manual):
    bases, reviewed = candidates()
    # 五个不同问题来自同一物理路线，复现远端稀疏覆盖。
    reviewed = [{**r, 'route_id': 'one_reviewed_route'} for r in reviewed] if manual else []
    rows, report = builder._balanced_invalid_rows(bases, split='val', target=30,
        rng=random.Random(3), same_rs_rows=reviewed)
    assert report['same_rs_support']['status'] == 'insufficient_support'
    assert report['same_rs_unique_routes'] == manual
    assert all(report['guards'].values())
    assert set(report['true_rs']['counts']) == {'R1', 'R2', 'R3', 'R4', 'R5'}
    assert set(report['asked_context']['counts']) == set(CONTEXT_IDS)
    assert report['candidate_true_prompt_rs_context']['counts']
    for seed in range(10):
        sampled = balanced_invalid_items([SimpleNamespace(**r) for r in rows],
                                         target=64, rng=random.Random(seed))
        audit = invalid_subgroup_report(sampled)
        assert all(audit['guards'].values())
        assert audit['same_rs_max_case_repeat'] <= 1
        assert audit['same_rs_unique_routes'] == manual
        automatic_sources = {r.invalid_source.split('|')[0].removeprefix('source=') for r in sampled
                             if r.invalid_reason == 'wrong_road_structure'}
        assert automatic_sources == set(CONTEXT_IDS)


def _minimal_index(path, manual):
    """三个split十类有效样本与合成负例，用真实构建/动作合同组装。"""
    bases, reviewed = candidates()
    rows = []
    for split in ('train', 'val', 'test'):
        for context in CONTEXT_IDS:
            base = next(b for b in bases if b['context_id'] == context)
            base = {**base, 'split': split, 'route_id': f'{split}_{context}',
                    'rs': CONTEXT_BY_ID[context].allowed_rs[0]}
            base['action_labels'] = dict.fromkeys(ACTION_KEYS, False)
            base['action_evidence'] = {**base['action_evidence'],
                'longitudinal_decision': longitudinal_decision([8.] * 9),
                'lateral_observation_complete': True, 'lane_change_direction': ''}
            row = builder._make_row(base=base, context_id=context, invalid=False)
            from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
            row['action_evidence'].update(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256())
            row.update(choice_annotation(base['action_labels'], context, base['action_evidence']))
            rows.append(row)
        negatives, _ = builder._balanced_invalid_rows(
            [{**b, 'split': split, 'route_id': f'{split}_{b["route_id"]}'} for b in bases],
            split=split, target=30, rng=random.Random(1))
        for row in negatives:
            row['action_evidence'].update(rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256())
            row.update(primary_action_version=PRIMARY_CHOICE_VERSION, primary_action=None,
                       keep_scope=None, primary_action_evidence_status='invalid_context')
        rows.extend(negatives)
        if manual:
            row = {**negatives[0], 'invalid_reason': 'same_rs_wrong_event', 'route_id': f'{split}_manual'}
            rows.append(row)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return rows


@pytest.mark.parametrize('manual', [0, 1])
def test_binary_preflight_reports_sparse_subgroup_and_still_rejects_leakage(tmp_path, manual):
    path = tmp_path / 'index.jsonl'
    rows = _minimal_index(path, manual)
    report = preflight.check_index(path, 'binary')
    assert report['same_rs_evaluation']['val']['status'] == 'insufficient_support'
    assert report['same_rs_physical_routes']['val'] == manual
    rows.append({**rows[0], 'split': 'val'})
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='physical route leakage'):
        preflight.check_index(path, 'binary')


def test_sparse_manual_accuracy_does_not_select_checkpoint_or_claim_full_evaluation():
    good = {**_healthy_metrics(), 'same_rs_unique_routes': 1}
    kwargs = dict(min_invalid_exact=.8, min_lane_change_recall=.6, min_stop_recall=.8, min_no_action_exact=.5)
    bad = {**good, 'invalid_subgroup/reason/same_rs_wrong_event_exact': 0.}
    assert generation_checkpoint_score(good, **kwargs) == generation_checkpoint_score(bad, **kwargs)
    assert _guard(bad)['all_ok'] and not _guard(bad)['evaluation_complete']
    assert _guard(bad)['same_rs_evaluation']['passed'] is None
    assert not _guard({**bad, 'same_rs_unique_routes': 2})['all_ok']
