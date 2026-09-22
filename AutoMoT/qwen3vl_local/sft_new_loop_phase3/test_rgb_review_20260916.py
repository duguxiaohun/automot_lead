"""逐帧审计后的覆盖、物理路线独立性、精确隔离和全量评测回归。"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import random
import sys

import pytest

from qwen3vl_local.sft_new_loop_phase3 import preflight, train, eval as evaluation
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _split, _stable_unit, physical_route_group
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS, ACTION_KEYS
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation, PRIMARY_CHOICE_VERSION
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import longitudinal_decision
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (
    invalid_subgroup_report, require_same_rs_support, balanced_invalid_items,
)
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash, mapped_contexts
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import ACTION_RULE_VERSION, action_rule_sha256
from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows


def index_rows(same_count=2):
    """最小完整索引，合同真实；不同 Rep 复采故意用同一个物理路线。"""
    rows = []
    for split in ('train', 'val', 'test'):
        for i, context in enumerate((*CONTEXT_IDS, 'INVALID')):
            rows.append(dict(scenario='synthetic', route_id=f'{split}_{i}', split=split,
                frame_id=i, current_speed_mps=3, context_id=CONTEXT_IDS[0] if context=='INVALID' else context,
                prompt_road_structure='R1', invalid_action_context=context=='INVALID',
                invalid_reason='wrong_road_structure' if context=='INVALID' else '',
                mapping_contract_hash=mapping_contract_hash(), action_evidence=dict(
                    rule_version=ACTION_RULE_VERSION, rule_code_sha256=action_rule_sha256())))
        for i in range(same_count):
            rows.append(dict(rows[-1], route_id=f'Town01_Rep{i}_{split}_route_{i}_route0_01_01_01_01_01',
                             frame_id=20+i, invalid_reason='same_rs_wrong_event'))
    for row in rows:
        row['answers'] = {**dict.fromkeys(ACTION_KEYS, False), 'KEEP': not row['invalid_action_context'], 'INVALID_ACTION_CONTEXT': row['invalid_action_context']}
        row['action_evidence'].update(longitudinal_decision=longitudinal_decision([3.0]*9),
                                      lateral_observation_complete=True, lane_change_direction='')
        row.update(dict(primary_action_version=PRIMARY_CHOICE_VERSION, primary_action=None,
                        keep_scope=None, primary_action_evidence_status='invalid_context')
                   if row['invalid_action_context'] else
                   choice_annotation(row['answers'], row['context_id'], row['action_evidence']))
    return rows


def write_index(tmp_path, rows):
    path = tmp_path/'index.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_preflight_reports_sparse_independent_support_without_blocking_training(tmp_path):
    rows = index_rows(1)
    # 再加不同 Rep/时间戳和不同问题，也仍只有一条物理路线。
    val = next(r for r in rows if r['split']=='val' and r['invalid_reason']=='same_rs_wrong_event')
    rows.append(dict(val, route_id=val['route_id'].replace('_Rep0_', '_Rep9_').replace('01_01_01_01_01','02_02_02_02_02'), frame_id=29))
    path = write_index(tmp_path, rows)
    report = preflight.check_index(path)
    assert report['same_rs_evaluation']['val']['status'] == 'insufficient_support'
    assert preflight.check_index(path, 'choice')['same_rs_physical_routes']['val'] == 1
    good = preflight.check_index(write_index(tmp_path,index_rows(2)))
    assert good['same_rs_physical_routes']==dict(train=2,val=2,test=2)


def test_missing_required_context_fails_before_model_check_and_nccl(monkeypatch,tmp_path):
    path = write_index(tmp_path,[r for r in index_rows(0)
                               if r['split'] != 'val' or r['context_id'] != CONTEXT_IDS[0]])
    monkeypatch.setattr(sys,'argv',['train.py','--index',str(path),'--action-output-mode','binary'])
    monkeypatch.setattr(preflight,'check_model',lambda *a: pytest.fail('model touched'))
    monkeypatch.setattr(train,'setup_distributed',lambda *a: pytest.fail('NCCL touched'))
    with pytest.raises(ValueError,match='missing contexts'):
        train.train(train.parse_args())


def test_metric_and_sampled_support_collapse_repeated_physical_route():
    same = next(r for r in candidate_rows() if r.invalid_reason=='same_rs_wrong_event')
    route='Town01_Rep0_route_42_route0_01_01_01_01_01'
    rows=[replace(same,route_id=route,frame_id=1),replace(same,route_id=route.replace('_Rep0_','_Rep7_'),frame_id=2)]
    assert invalid_subgroup_report(rows)['same_rs_unique_routes']==1
    with pytest.raises(ValueError,match='sampled same_rs_wrong_event has 1'):
        require_same_rs_support(rows,stage='generation validation')
    rows.append(replace(same,route_id=route.replace('route_42','route_43')))
    require_same_rs_support(rows,stage='generation validation')


def test_real_review_decisions_stay_blind_disjoint_and_sampled():
    decisions=[json.loads(l) for l in Path(__file__).with_name('same_rs_invalid_review_20260916.jsonl').read_text().splitlines()]
    counts={'val':set(),'test':set()}
    for d in decisions:
        assert d['frozen_prompt_sha256_kind'] == 'source_file_sha256:prompts.py'
        # 历史盲审记录保留当时源码身份；后续prompt修改不能倒写审计历史。
        assert d['frozen_prompt_sha256'] == '1b31876e83bb39bed3e76090559ee27e29836eaa020f74ddca6c4516653cd27a'
        # 验证当时的盲划分；20260920已曝光路线现应迁入train，不能倒写历史记录。
        group=physical_route_group(d['scenario'],d['route_id'])
        unit=_stable_unit(f'20260916:{group}')
        split='test' if unit < .1 else 'val' if unit < .15 else 'train'
        assert split==d['review_split'] and not d['model_outputs_inspected']
        counts[split].add(physical_route_group(d['scenario'],d['route_id']))
    assert {k:len(v) for k,v in counts.items()}=={'val':2,'test':4}
    for split in counts:
        rows=[r for r in candidate_rows() if r.invalid_reason=='wrong_road_structure']
        for d in decisions:
            if d['review_split']!=split:continue
            for asked in d['asked_contexts']:
                rows.append(SimpleNamespace(scenario=d['scenario'],route_id=d['route_id'],frame_id=d['frame_id'],
                    true_rs=d['true_rs'],context_id=asked,prompt_road_structure=d['true_rs'],
                    invalid_reason='same_rs_wrong_event',
                    invalid_source=f"source={d['source_context']}|true_rs={d['true_rs']}|asked_context={asked}"))
        for seed in range(12):
            selected=balanced_invalid_items(rows,target=64,rng=random.Random(seed))
            require_same_rs_support(selected,stage='reviewed holdout')
            assert invalid_subgroup_report(selected)['same_rs_max_case_repeat']==1


def test_exact_event_withdrawal_does_not_relabel_unseen_frames_or_other_events():
    route='Town07_Rep0_Town07_Scenario3_6_route0_01_11_05_19_17'
    for frame in (121,124):
        assert not mapped_contexts('DynamicObjectCrossing',route,frame,'R1','U-E3',['U-E3'])[0]
        assert mapped_contexts('DynamicObjectCrossing',route,frame,'R1','U-E4',['U-E4'])[0]
    for frame in (120,125):
        assert mapped_contexts('DynamicObjectCrossing',route,frame,'R1','U-E3',['U-E3'])[0]
    groups=json.loads(Path(__file__).with_name('development_route_groups_20260916.json').read_text())['groups']
    for group in groups:
        scenario,route=group.split('/',1)
        assert _split(scenario,route,20260916,.1,.05)=='train'


def test_default_eval_is_full_coverage(monkeypatch):
    monkeypatch.setattr(sys,'argv',['eval.py'])
    args=evaluation.parse_args()
    assert args.cases_per_bin==0
    assert 'data_v23' in args.index
    rows=candidate_rows()
    selected=evaluation._balanced_cases(rows,cases_per_bin=args.cases_per_bin,seed=3)
    assert len(selected)==len(rows)
    assert 'CASES_PER_BIN:-0' in Path(__file__).with_name('eval.sh').read_text()


@pytest.mark.parametrize('input_frames', [(7, 10), (7, 8, 9, 10)])
def test_review_panels_preserve_actual_input_boundary(tmp_path, monkeypatch, input_frames):
    """两个未来帧不能因固定四列布局被误记成 2RGB 模型输入。"""
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase3 import prepare_error_review as review
    run = tmp_path/'data'/'synthetic'/'route'
    (run/'rgb').mkdir(parents=True)
    (run/'metas').mkdir()
    metas = {}
    for f in range(7, 24):
        Image.new('RGB', (12, 4), 'gray').save(run/'rgb'/f'{f:04d}.jpg')
        (run/'metas'/f'{f:04d}.pkl').write_bytes(b'fixture')
        metas[f] = dict(speed=0)
    signals = dict(future_speeds=[0]*13)
    monkeypatch.setattr(review, 'is_abnormal_lead_route', lambda *a: (False, {}))
    monkeypatch.setattr(review, 'load_route_trajectory', lambda *a: SimpleNamespace(
        metas=metas, signals=lambda f: signals))
    monkeypatch.setattr(review, 'label_actions', lambda s: {'STOP': True})
    inputs = [run/'rgb'/f'{f:04d}.jpg' for f in input_frames]
    case = dict(case_index=0, scenario='synthetic', route_id='route', frame_id=10,
        history_rgb_paths_used=list(map(str, inputs)), history_rgb_sha256=list(map(review.digest, inputs)),
        action_evidence=dict(future_speeds_exact_mps=signals['future_speeds']),
        gt=dict(STOP='YES', INVALID_ACTION_CONTEXT='NO'), parsed=dict(STOP='YES'),
        all_ok=True, context_id='LEAD_BRAKE', context_detail='', prompt_road_structure='R1')
    bundle = tmp_path/'bundle'
    (bundle/'lora_production').mkdir(parents=True)
    (bundle/'lora_production'/'cases.jsonl').write_text(json.dumps(case)+'\n')
    output = tmp_path/'review'
    review.prepare(bundle, tmp_path/'data', output, only_ids=[0])
    frames = json.loads((output/'evidence.json').read_text())['cases'][0]['frames']
    assert [f['frame'] for f in frames if f['input']] == list(input_frames)
    assert [f['frame'] for f in frames if not f['input']] == list(range(11, 24))
    assert [f['frame'] for f in frames if f['confirmation_only']] == [23]


def test_reviewed_negative_rejects_changed_rgb(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import same_rs_invalid as negatives
    decision = json.loads(negatives.NEW_DECISIONS.read_text().splitlines()[0])
    old, new = tmp_path/'old.jsonl', tmp_path/'new.jsonl'
    old.write_text('')
    new.write_text(json.dumps(decision)+'\n')
    monkeypatch.setattr(negatives, 'DECISIONS', old)
    monkeypatch.setattr(negatives, 'NEW_DECISIONS', new)
    monkeypatch.setattr(negatives, 'is_abnormal_lead_route', lambda *a: (False, {}))
    run = tmp_path/decision['scenario']/decision['route_id']
    (run/'rgb').mkdir(parents=True)
    for frame in decision['original_rgb_frames']:
        (run/'rgb'/f'{frame:04d}.jpg').write_bytes(b'replaced RGB')
    monkeypatch.setattr(negatives, 'load_route_trajectory', lambda *a: pytest.fail('changed RGB reached meta loading'))
    with pytest.raises(ValueError, match='changed RGB in same-RS decision'):
        negatives.reviewed_invalid_rows(SimpleNamespace(data_root=tmp_path),
            {(decision['scenario'], decision['route_id'])})


def test_rebuilt_audit_reports_physical_support_separately_from_runs(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import audit_rebuilt_index as rebuilt
    monkeypatch.setattr(rebuilt, 'check_index', lambda p, **kw: {})
    row = dict(scenario='synthetic', split='val', frame_id=1, context_id='STATIC_BLOCKAGE',
        prompt_road_structure='R3', invalid_reason='same_rs_wrong_event', action_signature='INVALID',
        invalid_action_context=True, history_rgb_paths=[], context_detail='', current_speed_mps=3,
        goal_ego_xy=[10, 0], answers=dict(DECELERATE=False, STOP=False, RESUME=False,
            LANE_CHANGE_LEFT=False, LANE_CHANGE_RIGHT=False, INVALID_ACTION_CONTEXT=True))
    rows=[dict(row, route_id=f'Town01_Rep{rep}_route_{number}_route0_01_01_01_01_01')
          for rep, number in ((0,42),(1,42),(0,43))]
    result=rebuilt.audit(write_index(tmp_path, rows), tmp_path)['same_rs']['val']
    assert result['unique_runs']==3 and result['unique_routes']==2


@pytest.mark.parametrize('module_name', ['audit_rebuilt_index', 'audit_raw_index'])
def test_pipeline_audit_cli_forwards_choice_exemption(tmp_path, monkeypatch, module_name):
    import importlib
    module = importlib.import_module('qwen3vl_local.sft_new_loop_phase3.'+module_name)
    index = write_index(tmp_path, [])
    observed = []
    def check(path, action_output_mode):
        observed.append(action_output_mode)
        return {}
    monkeypatch.setattr(module, 'check_index', check)
    monkeypatch.setattr(sys, 'argv', [module_name, '--index', str(index), '--data-root', str(tmp_path),
        '--output', str(tmp_path/'report.json'), '--action-output-mode', 'choice'])
    module.main()
    assert observed == ['choice']
