"""复用逐帧取证表，报告车道身份变化的起点与确认点；不自动改标签。"""
import argparse
import json
from pathlib import Path


def lane_identity_timeline(frames, anchor):
    """仅记录相邻帧事实；lane ID 符号不能直接转换为自车左右。

    road/section 变化单独记录，缺帧不跨越连接；末帧没有后继即未确认。
    回到 anchor 身份不意味着一定完成视觉归位，仍须原图复核。
    """
    by_frame = {int(row['frame']): row for row in frames}

    def identity(row):
        return tuple(row.get(k) for k in ('road_id', 'section_id', 'lane_id'))

    events = []
    initial = identity(by_frame[anchor]) if anchor in by_frame else None
    for frame, row in sorted(by_frame.items()):
        before = by_frame.get(frame - 1)
        if before is None or identity(before) == identity(row):
            continue
        old, new = identity(before), identity(row)
        following = by_frame.get(frame + 1)
        confirmed = following is not None and identity(following) == new
        comparable = all(v is not None for v in (*old, *new)) and old[:2] == new[:2]
        events.append(dict(frame=frame, time_s=(frame-anchor)*.25,
            from_identity=list(old), to_identity=list(new),
            same_road_section=comparable,
            two_sample_confirmed=confirmed,
            confirmation_time_s=(frame+1-anchor)*.25 if confirmed else None,
            in_model_history=frame <= anchor,
            returns_to_anchor_identity=frame > anchor and comparable and new == initial))
    return dict(events=events, visual_crossing_confirmed=False,
        direction_inferred_from_ids=False,
        note='Raw identity changes only; inspect RGB and the authoritative lateral guard before assigning ego direction.')


def main():
    """不生成重复RGB，直接读取已有 evidence.json 的逐帧元信息。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cases = []
    for source in args.evidence:
        for row in json.loads(source.read_text())['cases']:
            cases.append(dict(case_index=row['case_index'], scenario=row['scenario'],
                route_id=row['route_id'], frame_id=row['frame_id'], evidence_source=str(source),
                **lane_identity_timeline(row['frames'], row['frame_id'])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(cases=cases, labels_modified=False), ensure_ascii=False, indent=2)+'\n')
    print(f'lane identity timelines: {len(cases)} cases; labels unchanged')


if __name__ == '__main__':
    main()
