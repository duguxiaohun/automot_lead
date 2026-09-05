"""导出已接入主构建器的同 RS 人工负例；此复用诊断导出不作独立holdout。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _make_row
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from qwen3vl_local.sft_new_loop_phase3.prompts import make_prompt_spec, build_action_prompt, build_action_target
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route


def build(candidates: Path, decisions: Path, data_root: Path, output: Path):
    """只复用已过滤候选，要求全部所看 history 与真正输入一致，不从事件缺标造负例。"""
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3.same_rs_invalid import reviewed_invalid_rows, DECISIONS
    if decisions.resolve() != DECISIONS.resolve():
        raise ValueError('Use the versioned reviewed decisions; alternate files are not production evidence')
    split_by_route = {}
    for line in candidates.open():
        base = json.loads(line)
        validate_action_rule(base)
        key = (base['scenario'], base['route_id'])
        if key in split_by_route and split_by_route[key] != base['split']:
            raise ValueError(f'route split leakage: {key}')
        split_by_route[key] = base['split']
    args = SimpleNamespace(data_root=data_root, split_seed=0, test_ratio=0, val_ratio=0)
    rows = reviewed_invalid_rows(args, set(split_by_route))
    for row in rows:
        row['source_split'] = split_by_route[(row['scenario'], row['route_id'])]
        row['split'] = 'diagnostic'
        row['diagnostic_only'] = True
        spec = make_prompt_spec(variant='all_random_order', answers=row['answers'],
            seed_key=f"challenge:{row['route_id']}:{row['frame_id']}:{row['context_id']}",
            context_id=row['context_id'], road_structure=row['true_rs'], goal_xy=row['goal_ego_xy'])
        row['prompt'] = build_action_prompt(spec=spec)
        row['target'] = build_action_target(spec)
    output.mkdir(parents=True, exist_ok=True)
    (output/'frame_index.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    summary = dict(rows=len(rows), source_routes=len({(r['scenario'],r['route_id']) for r in rows}),
        covered_contexts=sorted({r['context_id'] for r in rows}),
        missing_contexts=sorted(set(CONTEXT_IDS)-{r['context_id'] for r in rows}),
        training_integration=True, independent_holdout=False,
        note="Challenge reuses reviewed production candidates; evaluate split-specific production index for holdout metrics")
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',required=True,type=Path)
    p.add_argument('--decisions',type=Path,default=Path(__file__).with_name('same_rs_invalid_review_v1.jsonl'))
    p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[2]/'lead_data')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(build(a.candidates,a.decisions,a.data_root,a.output),ensure_ascii=False))
