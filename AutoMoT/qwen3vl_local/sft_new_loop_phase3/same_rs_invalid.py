"""显式逐帧 RGB 决定构造同 RS 错事件负例；不从未标注推断事件不存在。"""
import json
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, ACTION_KEYS
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, action_evidence
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route

DECISIONS = Path(__file__).with_name('same_rs_invalid_review_v1.jsonl')


def reviewed_invalid_rows(args, scanned_routes):
    """使用本次扫描路线；正式全量扫描也保留显式核验过的纯负例路线。

    source_context 是负例来源桶，不声称该事件真实发生。明确隔离的原 R-E3
    正例仍能通过人工决定成为负例，但绝不重回 valid 池。
    """
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import _make_row, _split, _history
    root = Path(args.data_root).resolve()
    rows = []
    # 正式全量扫描后重新平衡时，纯负例route可能不出现在正候选缓存。
    # 只复用同目录全量manifest和逐字匹配的人工决定；不把任意小样本cache扩大为全量。
    cached_reviewed = set()
    cache = getattr(args, 'candidate_cache', '')
    if cache:
        parent = Path(cache).parent
        manifest_path = parent / 'manifest.json'
        negative_path = parent / 'same_rs_invalid_candidates.jsonl'
        if manifest_path.is_file() and negative_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('source_scope') == 'collection_results':
                for line in negative_path.read_text().splitlines():
                    old = json.loads(line)
                    review = old.get('mapping_evidence', {}).get('same_rs_rgb_review')
                    if review:
                        cached_reviewed.add(json.dumps(review, sort_keys=True))
    for line in DECISIONS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        key = (d['scenario'], d['route_id'])
        if key not in scanned_routes and json.dumps(d, sort_keys=True) not in cached_reviewed:
            # 一条route没有可用正动作窗口，不等于人工确认的负例失效。
            # 只在完整collection扫描且人工已核验该源route时允许；小样本/cache仍严格限定范围。
            selected = getattr(args, 'scenarios', '')
            full_source = (not getattr(args, 'candidate_cache', '')
                           and not getattr(args, 'use_review_cache', False)
                           and not getattr(args, 'max_routes', 0)
                           and not getattr(args, 'max_routes_per_scenario', 0))
            allowed = selected == 'all' or key[0] in selected.split(',')
            source = Path(getattr(args, 'collection_dir', '')) / f'{key[0]}_result.json'
            if not (full_source and allowed and d.get('collection_route_verified') is True
                    and source.is_file()):
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
