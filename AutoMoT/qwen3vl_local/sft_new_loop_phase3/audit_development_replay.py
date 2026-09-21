"""回读开发索引的每行原始证据与四种提示词；不替代生产 preflight 的三 split 验收。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    load_route_trajectory, label_actions, action_evidence, validate_action_rule,
)
from qwen3vl_local.sft_new_loop_phase3.source_mapping import validate_mapping_contract
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import validate_choice_row
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.action_review import build_action_review
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group, development_route_groups
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import mismatched_road_contexts
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    make_prompt_spec, build_action_prompt, build_action_target, parse_action_output, spec_answers,
)


def audit(index, data_root):
    rows = [json.loads(line) for line in Path(index).read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    counts, groups = Counter(), {}
    for row in rows:
        validate_action_rule(row)
        validate_mapping_contract(row)
        validate_choice_row(row)
        group = physical_route_group(row['scenario'], row['route_id'])
        if group in development_route_groups() and row['split'] != 'train':
            raise ValueError('development route leaked into holdout')
        if group in groups and groups[group] != row['split']:
            raise ValueError('physical route split conflict')
        groups[group] = row['split']
        grouped[row['scenario'],row['route_id']].append(row)
    for (scenario, route_id), items in grouped.items():
        trajectory = load_route_trajectory(Path(data_root)/scenario/route_id)
        if trajectory is None:
            raise ValueError(f'missing raw trajectory: {scenario}/{route_id}')
        for row in items:
            frame = row['frame_id']
            signals = trajectory.signals(frame)
            if signals is None or action_evidence(signals) != row['action_evidence']:
                raise ValueError(f'raw evidence mismatch: {scenario}/{route_id}/{frame}')
            if row['current_speed_mps'] != round(signals['speed'],3) or row['goal_ego_xy'] != [round(signals['goal_x'],3),round(signals['goal_y'],3)]:
                raise ValueError('raw input speed or navigation mismatch')
            labels = label_actions(signals)
            if row.get('invalid_reason') == 'wrong_road_structure':
                allowed = mismatched_road_contexts(true_rs=row['true_rs'],
                    is_junction=signals['is_junction'],
                    distance_to_next_junction=signals['distance_to_next_junction'])
                if (row['context_id'],row['prompt_road_structure']) not in allowed:
                    raise ValueError('unsupported synthetic invalid context')
            if not row['invalid_action_context']:
                if labels is None or any(labels[k] != row['answers'][k]
                        for k in CONTEXT_BY_ID[row['context_id']].action_keys):
                    raise ValueError('raw labels mismatch')
                if {**build_action_review(trajectory,frame,row['context_id'],labels,signals),
                    'applies_to_prompt_context': True} != row['action_review']:
                    raise ValueError('raw review mismatch')
                counts['confirmed_pullaway_rows'] += signals is not None and row['action_evidence']['longitudinal_decision']['confirmed_pullaway']
            for path in row['history_rgb_paths']:
                if not (Path(data_root)/path).is_file():
                    raise FileNotFoundError(path)
            for mode in ('binary','choice'):
                if row['invalid_action_context'] and mode == 'choice':
                    continue
                for rgb in ('4rgb','2rgb_endpoints'):
                    spec=make_prompt_spec(variant='all_random_order',answers=row['answers'],
                        seed_key='development-replay',context_id=row['context_id'],
                        road_structure=row['prompt_road_structure'],goal_xy=row['goal_ego_xy'],
                        context_detail=row['context_detail'],current_speed_mps=row['current_speed_mps'],
                        action_output_mode=mode)
                    prompt=build_action_prompt(spec=spec,history_rgb_mode=rgb)
                    for hidden in ('future_speeds','anchor_controls','throttle','gain_confirmed_s'):
                        if hidden in prompt:raise ValueError('offline evidence leaked into prompt')
                    target=build_action_target(spec)
                    if parse_action_output(target,spec=spec)!=spec_answers(spec):
                        raise ValueError('target/parser mismatch')
                    counts['prompt_target_replays']+=1
            counts['rows']+=1
    return dict(scope='development_row_replay_not_production_preflight',
                production_preflight_performed=False, manual_rgb_review_performed=False,
                counts=dict(counts), routes=len(grouped), splits=sorted(set(groups.values())),
                raw_evidence_mismatches=0, raw_label_mismatches=0, raw_review_mismatches=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True,type=Path)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=audit(args.index,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
