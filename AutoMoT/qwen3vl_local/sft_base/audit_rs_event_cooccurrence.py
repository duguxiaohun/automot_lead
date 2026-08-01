"""审计 collection_output 里的 RS x EVENT 共现矩阵。

用法（从 AutoMoT/ 目录）：
  python qwen3vl_local/sft_base/audit_rs_event_cooccurrence.py \
    --collection-dir keyframe_filter/collection_output \
    --output-json checkpoints/sft_base_data/rs_event_cooccurrence.json

脚本不加载模型，只读取标注 JSON，用于检查 `EVENT_CANDIDATES_BY_RS` 是否漏掉真实
数据里已经出现的 RS/UE 组合，同时统计 RS/R-E regular 子类分布与多 regular 标签
比例。RE 展开后，R3 是否真的有足够 R-E1/R-E2/R-E3 区分度要先看这里。

默认阈值是严格方案 A：count >= 20 且占该 RS 帧数 rate >= 0.1%。脚本同时报告
missing（数据显著存在但静态表没有）和 spurious（静态表有但数据低于阈值或为 0），
避免只看 audit examples 单向补表。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, Mapping

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base.labels import (  # noqa: E402
    EVENT_CANDIDATES_BY_RS,
    RS_LABELS,
    is_regular_event,
    resolve_event_target,
    resolve_rs_target,
)
from qwen3vl_local.sft_base.build_dataset import _skip_sets  # noqa: E402


def _iter_result_paths(collection_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """遍历 collection_output 下的 result JSON。"""

    for path in sorted(collection_dir.glob("*_result.json")):
        if path.name == "noScenarios_result.json":
            continue
        yield path


def _iter_route_frames(result: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """按 build_dataset 口径过滤 route 后产出 frame annotation。"""

    routes = result.get("routes") or result.get("route_results") or []
    if isinstance(routes, Mapping):
        routes = routes.values()
    abnormal_skips, missing_skips = _skip_sets(result)
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        route_id = str(route.get("route_id") or route.get("run_id") or "")
        if route_id in abnormal_skips or route_id in missing_skips:
            continue
        if route.get("status") not in {None, "success"}:
            continue
        frames = route.get("frames") or route.get("frame_annotations") or route.get("annotations") or []
        if isinstance(frames, Mapping):
            frames = frames.values()
        for frame in frames:
            if isinstance(frame, Mapping):
                yield route_id, frame


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 RS x UE 共现审计。"""

    collection_dir = pathlib.Path(args.collection_dir)
    min_count = max(1, int(args.min_count))
    min_rate = max(0.0, float(args.min_rate))
    rs_totals = {rs: 0 for rs in RS_LABELS}
    ue_totals = {rs: {code: 0 for code in EVENT_CANDIDATES_BY_RS.get(rs, []) if str(code).startswith("U-E")} for rs in RS_LABELS}
    all_ue_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    all_regular_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    multi_regular_frames_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    regular_frame_total_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    skipped = 0
    total_frames = 0
    abnormal_frames = 0
    result_files = 0
    skipped_routes_abnormal_or_missing = 0
    skipped_routes_failed = 0
    for path in _iter_result_paths(collection_dir):
        result_files += 1
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        abnormal_skips, missing_skips = _skip_sets(result)
        skipped_routes_abnormal_or_missing += len(abnormal_skips) + len(missing_skips)
        routes = result.get("routes") or result.get("route_results") or []
        if isinstance(routes, Mapping):
            routes = routes.values()
        skipped_routes_failed += sum(1 for route in routes if isinstance(route, Mapping) and route.get("status") not in {None, "success"})
        for _route_id, frame in _iter_route_frames(result):
            try:
                rs = resolve_rs_target(frame).label
                event = resolve_event_target(frame)
            except Exception:
                skipped += 1
                continue
            if rs not in RS_LABELS:
                skipped += 1
                continue
            total_frames += 1
            rs_totals[rs] += 1
            regular_codes = [code for code in event.raw_events if is_regular_event(code)]
            if regular_codes:
                regular_frame_total_by_rs[rs] += 1
            if len(set(regular_codes)) > 1:
                multi_regular_frames_by_rs[rs] += 1
            for code in regular_codes:
                all_regular_by_rs[rs][code] = all_regular_by_rs[rs].get(code, 0) + 1
            if not event.abnormal:
                continue
            abnormal_frames += 1
            label = str(event.label)
            all_ue_by_rs[rs][label] = all_ue_by_rs[rs].get(label, 0) + 1
            if label in ue_totals[rs]:
                ue_totals[rs][label] += 1

    static_sets = {rs: {code for code in EVENT_CANDIDATES_BY_RS.get(rs, []) if str(code).startswith("U-E")} for rs in RS_LABELS}
    missing = []
    low_rate_missing = []
    for rs in RS_LABELS:
        for ue, count in sorted(all_ue_by_rs[rs].items()):
            rate = count / max(1, rs_totals[rs])
            row = {
                "rs": rs,
                "event": ue,
                "count": count,
                "rs_frame_rate": rate,
                "in_static_table": ue in static_sets[rs],
            }
            if ue not in static_sets[rs]:
                if count >= min_count and rate >= min_rate:
                    missing.append(row)
                else:
                    low_rate_missing.append(row)
    spurious = []
    for rs in RS_LABELS:
        if rs_totals[rs] <= 0:
            continue
        observed = all_ue_by_rs.get(rs, {})
        for ue in sorted(static_sets[rs]):
            count = int(observed.get(ue, 0))
            rate = count / max(1, rs_totals[rs])
            if count < min_count or rate < min_rate:
                spurious.append(
                    {
                        "rs": rs,
                        "event": ue,
                        "count": count,
                        "rs_frame_rate": rate,
                        "in_static_table": True,
                    }
                )
    report = {
        "collection_dir": str(collection_dir),
        "result_files": result_files,
        "total_frames": total_frames,
        "abnormal_frames": abnormal_frames,
        "skipped_items": skipped,
        "skipped_routes_abnormal_or_missing": skipped_routes_abnormal_or_missing,
        "skipped_routes_failed": skipped_routes_failed,
        "min_count": min_count,
        "min_rate": min_rate,
        "rs_totals": rs_totals,
        "ue_by_rs": all_ue_by_rs,
        "regular_by_rs": all_regular_by_rs,
        "regular_frame_total_by_rs": regular_frame_total_by_rs,
        "multi_regular_frames_by_rs": multi_regular_frames_by_rs,
        "multi_regular_frame_rate_by_rs": {
            rs: multi_regular_frames_by_rs[rs] / max(1, regular_frame_total_by_rs[rs])
            for rs in RS_LABELS
        },
        "static_ue_by_rs": {rs: sorted(static_sets[rs]) for rs in RS_LABELS},
        "missing_combinations": missing,
        "low_rate_missing_combinations": low_rate_missing,
        "spurious_combinations": spurious,
        "candidate_count_after_static_table": {
            rs: len(EVENT_CANDIDATES_BY_RS.get(rs, []))
            for rs in RS_LABELS
        },
    }
    return report


def main() -> None:
    """CLI 入口。"""

    p = argparse.ArgumentParser(description="Audit RS x UE co-occurrence against static EVENT_CANDIDATES_BY_RS")
    p.add_argument("--collection-dir", type=str, default="keyframe_filter/collection_output")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--min-rate", type=float, default=0.001)
    args = p.parse_args()
    report = audit(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
