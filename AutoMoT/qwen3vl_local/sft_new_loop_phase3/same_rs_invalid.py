"""显式逐帧 RGB 决定构造同 RS 错事件负例；不从未标注推断事件不存在。"""
import json
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, ACTION_KEYS
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, action_evidence
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route

DECISIONS = Path(__file__).with_name('same_rs_invalid_review_v1.jsonl')


def reviewed_invalid_rows(args, scanned_routes):
    """只使用本次实际扫描路线；保留原 route split，并重读 meta/实际四帧路径。

    source_context 是负例来源桶，不声称该事件真实发生。明确隔离的原 R-E3
    正例仍能通过人工决定成为负例，但绝不重回 valid 池。
    """
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _make_row, _split, _history
    root = Path(args.data_root).resolve()
    rows = []
    for line in DECISIONS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        key = (d['scenario'], d['route_id'])
        if key not in scanned_routes:
            continue
        if d['decision'] != 'SAME_RS_EVENT_MISMATCH' or d['scope'] != 'training_and_evaluation_route_split':
            continue
        run = root / key[0] / key[1]
        if is_abnormal_lead_route(run, key[0])[0]:
            continue
        history = _history(run, d['frame_id'])
        if history is None or any(int(Path(path).stem) not in d['original_rgb_frames'] for path in history):
            raise ValueError(f'unreviewed history in same-RS decision: {key}/{d["frame_id"]}')
        trajectory = load_route_trajectory(run)
        signals = trajectory.signals(d['frame_id']) if trajectory is not None else None
        if signals is None or not signals['goal_available']:
            raise ValueError(f'missing same-RS anchor meta: {key}/{d["frame_id"]}')
        # run 目录自身可能是 symlink；逻辑 Scenario/run_id 路径才是可迁移数据合同。
        paths = [str(Path(key[0]) / key[1] / 'rgb' / Path(path).name) for path in history]
        source = d['source_context']
        if source not in CONTEXT_BY_ID:
            raise ValueError(f'unknown negative provenance: {source}')
        base = dict(scenario=key[0], route_id=key[1], town=key[1].split('_')[0],
                    frame_id=d['frame_id'], rs=d['true_rs'], invalid_prompt_rs=d['true_rs'],
                    split=_split(*key, args.split_seed, args.test_ratio, args.val_ratio),
                    primary_event='RGB_REVIEWED_NEGATIVE', event_codes=[], context_id=source,
                    action_labels={k: False for k in ACTION_KEYS}, action_evidence=action_evidence(signals),
                    goal_x=signals['goal_x'], goal_y=signals['goal_y'],
                    mapping_evidence={'same_rs_rgb_review': d, 'source_context_is_provenance_only': True},
                    visual_label_risk=False, visual_label_risk_reasons=[],
                    history_rgb_paths=paths, latest_rgb_path=paths[-1])
        for asked in d['asked_contexts']:
            if d['true_rs'] not in CONTEXT_BY_ID[asked].allowed_rs:
                raise ValueError(f'same-RS decision is actually wrong-RS: {asked}/{d["true_rs"]}')
            row = _make_row(base=base, context_id=asked, invalid=True,
                            invalid_source=f'source={source}|true_rs={d["true_rs"]}|asked_context={asked}')
            row['invalid_reason'] = 'same_rs_wrong_event'
            rows.append(row)
    return rows
