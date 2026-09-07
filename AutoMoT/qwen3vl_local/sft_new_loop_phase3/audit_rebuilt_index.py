"""检查重建索引的标签/输入一致性与独立覆盖，生成可复现CPU审计报告。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import label_actions
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt


def audit(index, data_root):
    """逐行验证被问动作，按split报告独立case/负例/动作；不伪造视觉审核。"""
    report = check_index(index)
    signatures = defaultdict(Counter)
    repeats = defaultdict(Counter)
    negatives = defaultdict(set)
    positive = defaultdict(Counter)
    checked_paths = set()
    near_boundary = Counter()
    boundary_cases = []
    for line in Path(index).open():
        row = json.loads(line)
        split = row['split']
        key = (row['scenario'], row['route_id'], row['frame_id'], row['context_id'],
               row['prompt_road_structure'], row['invalid_reason'])
        repeats[split][key] += 1
        signatures[split][row['action_signature']] += 1
        if row['invalid_reason'] == 'same_rs_wrong_event':
            negatives[split].add((row['scenario'], row['route_id'], row['frame_id'], row['context_id']))
        for path in row['history_rgb_paths']:
            full = Path(data_root) / path
            if full not in checked_paths and not full.is_file():
                raise FileNotFoundError(full)
            checked_paths.add(full)
        if not row['invalid_action_context']:
            evidence = row['action_evidence']
            exact_speeds = evidence['future_speeds_exact_mps']
            exact_labels = label_actions(dict(future_speeds=exact_speeds,
                future_speed_count=len(exact_speeds), lane_change_direction=evidence['lane_change_direction']))
            asked = CONTEXT_BY_ID[row['context_id']].action_keys
            if exact_labels is None or any(exact_labels[k] != row['answers'][k] for k in asked):
                raise ValueError(f'exact action evidence mismatch: {key}')
            # 展示用3位小数可能改变边界判断；真值以完整精度速度为准。
            labels = label_actions(dict(future_speeds=evidence['future_speeds_mps'],
                future_speed_count=len(evidence['future_speeds_mps']),
                lane_change_direction=evidence['lane_change_direction']))
            if labels is None or any(labels[k] != row['answers'][k] for k in asked):
                near_boundary[split] += 1
                boundary_cases.append(dict(scenario=row['scenario'], route_id=row['route_id'],
                    frame_id=row['frame_id'], context_id=row['context_id'], split=split,
                    answers=row['answers'], rounded_speeds=evidence['future_speeds_mps']))
            for k in asked:
                positive[split][k + ('/YES' if row['answers'][k] else '/NO')] += 1
        spec = make_prompt_spec(variant='all_random_order', answers=row['answers'], seed_key='audit',
            context_id=row['context_id'], road_structure=row['prompt_road_structure'],
            goal_xy=row['goal_ego_xy'], context_detail=row['context_detail'],
            current_speed_mps=row['current_speed_mps'])
        prompt = build_action_prompt(spec=spec)
        if 'future_speeds_mps' in prompt or 'lane_change_direction' in prompt:
            raise AssertionError('offline evidence leaked into prompt')
    report.update(manual_rgb_confirmation_by_this_program=False, rgb_files_checked=len(checked_paths),
        exact_speed_label_mismatches=0,
        signature_counts={k: dict(v) for k, v in signatures.items()},
        asked_action_counts={k: dict(v) for k, v in positive.items()},
        rounded_evidence_boundary_recheck=dict(near_boundary),
        rounded_evidence_boundary_cases=boundary_cases,
        max_input_repeat={k: max(v.values()) for k, v in repeats.items()},
        same_rs={k: dict(unique_cases=len(v), unique_routes=len({r[:2] for r in v}),
                         contexts=sorted({r[-1] for r in v})) for k, v in negatives.items()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.index, args.data_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('signature_counts', 'asked_action_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
