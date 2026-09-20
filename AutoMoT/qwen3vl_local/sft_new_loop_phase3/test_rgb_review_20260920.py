"""本轮精确隔离、旧评测曝光隔离及主要动作审计的回归。"""
import json
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import ACTION_KEYS
from qwen3vl_local.sft_new_loop_phase3.prepare_error_review import compare_action_labels
from qwen3vl_local.sft_new_loop_phase3.primary_action import primary_answers
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts

ROOT = Path(__file__).parent
DECISIONS = [json.loads(l) for l in (ROOT/'mapping_rgb_decisions_20260920.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('decision', DECISIONS)
def test_quarantine_is_exact_and_does_not_manufacture_negative_answers(decision):
    d = decision
    # 隔离整个损坏输入/错误前提；边界之外及其他采集仍走原始映射。
    for frame in range(d['start_frame'], d['end_frame']+1):
        contexts, evidence = mapped_contexts(d['scenario'], d['route_id'], frame, 'R3', 'R-E3', ['R-E3'])
        assert contexts == () and evidence['rgb_quarantine'] == d
        assert 'answers' not in evidence
    for frame in (d['start_frame']-1, d['end_frame']+1):
        assert mapped_contexts(d['scenario'], d['route_id'], frame, 'R3', 'R-E3', ['R-E3'])[0]
    assert mapped_contexts(d['scenario'], d['route_id']+'_different_run', d['start_frame'],
                           'R3', 'R-E3', ['R-E3'])[0]


def test_exposed_physical_routes_never_reenter_holdout_via_rep_or_timestamp():
    groups = json.loads((ROOT/'development_route_groups_20260920.json').read_text())['groups']
    assert len(groups) == 346
    for group in groups:
        scenario, stem = group.split('/', 1)
        town, rest = stem.split('_', 1)
        route = f'{town}_Rep9_{rest}_route0_02_02_02_02_02'
        for seed in (20260916, 20260920):
            assert _split(scenario, route, seed, .1, .05) == 'train'


@pytest.mark.parametrize('mode', ['choice', 'binary'])
@pytest.mark.parametrize('speed', ['STOP', 'RESUME', 'DECELERATE'])
def test_review_distinguishes_raw_compound_labels_from_primary_target(mode, speed):
    labels = {k: k in (speed, 'LANE_CHANGE_LEFT') for k in ACTION_KEYS}
    encode = lambda x: {**{k: 'YES' if v else 'NO' for k, v in x.items()}, 'INVALID_ACTION_CONTEXT': 'NO'}
    raw = encode(labels)
    gt = encode(primary_answers(labels)) if mode == 'choice' else raw.copy()
    row = dict(action_answers=raw, gt=gt, prompt_spec={'action_output_mode': mode})
    assert compare_action_labels(row, labels) == ([], [])
    row['gt'] = {**gt, 'LANE_CHANGE_RIGHT': 'YES'}
    assert compare_action_labels(row, labels) == ([], ['LANE_CHANGE_RIGHT'])
    row['action_answers'] = {**raw, speed: 'NO'}
    assert compare_action_labels(row, labels) == ([speed], ['LANE_CHANGE_RIGHT'])


def test_review_manifest_keeps_visual_uncertainty_distinct_from_none():
    notes = [json.loads(l) for l in (ROOT/'rgb_review_notes_20260920.jsonl').read_text().splitlines()]
    assert len(notes) == 44
    assert all(len(n['viewed_frames']) == 17 and n['input_sha256_match'] and n['raw_speed_match']
               and not n['label_mismatches'] and not n['target_mismatches'] for n in notes)
    dark = next(n for n in notes if n['case_index'] == 793)
    assert dark['decision'] == 'KEEP_WITH_UNCERTAINTY'
    assert dark['gt']['LANE_CHANGE_LEFT'] == 'YES'
