"""审计 collection_output 里的 RS x EVENT 共现矩阵。

用法（从 AutoMoT/ 目录）：
  python qwen3vl_local/sft_base_simple/audit_rs_event_cooccurrence.py \
    --collection-dir keyframe_filter/collection_output \
    --output-json checkpoints/sft_base_simple_data/rs_event_cooccurrence.json

脚本不加载模型，只读取标注 JSON，用于检查 `EVENT_CANDIDATES_BY_RS` 是否漏掉真实
数据里已经出现的 RS/UE 或 RS/R-E 组合，同时统计 regular 子类分布、多 regular
标签比例，以及 raw regular 被 RS canonical 映射的 scenario/route 归因。RE 展开后，
R3 是否真的有足够 R-E1/R-E2/R-E3 区分度，以及 R5+R-E4 这类冲突是否集中，都先看这里。

默认阈值是严格方案 A：count >= 20 且占该 RS 帧数 rate >= 0.1%。脚本同时报告
missing（数据显著存在但静态表没有）和 spurious（静态表有但数据低于阈值或为 0），
避免只看 audit examples 单向补表。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base_simple.labels import (  # noqa: E402
    EVENT_CANDIDATES_BY_RS,
    RS_LABELS,
    canonical_regular_event_for_rs,
    is_regular_event,
    resolve_event_target,
    resolve_rs_target,
)


def _skip_sets(result: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """读取 build_dataset 同款异常时长/数据缺失 route skip 集合。

    audit 只需要这两个顶层名单，内联 helper 可以避免轻量测试为了 import
    build_dataset 而依赖 numpy。
    """

    abnormal = {
        str(x.get("route_id") or x.get("run_id"))
        for x in result.get("abnormal_duration_skipped", [])
        if x.get("route_id") or x.get("run_id")
    }
    missing = {str(x.get("route_id") or x.get("run_id")) for x in result.get("data_missing_skipped", []) if x.get("route_id") or x.get("run_id")}
    return abnormal, missing


def _iter_result_paths(collection_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """遍历 collection_output 下的 result JSON。"""

    for path in sorted(collection_dir.glob("*_result.json")):
        if path.name == "noScenarios_result.json":
            continue
        yield path


def _iter_route_frames(result: Mapping[str, Any], scenario: str) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
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
                yield scenario, route_id, frame


def _split_static_sets() -> tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    """把静态 EVENT 表拆成 UE 与 regular 两套集合。"""

    static_ue = {
        rs: {code for code in EVENT_CANDIDATES_BY_RS.get(rs, []) if str(code).startswith("U-E")}
        for rs in RS_LABELS
    }
    static_regular = {
        rs: {code for code in EVENT_CANDIDATES_BY_RS.get(rs, []) if is_regular_event(str(code))}
        for rs in RS_LABELS
    }
    return static_ue, static_regular


def _combo_rows(
    *,
    observed_by_rs: Mapping[str, Mapping[str, int]],
    static_sets: Mapping[str, set[str]],
    rs_totals: Mapping[str, int],
    min_count: int,
    min_rate: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """对某一族 EVENT 做 missing / low-rate missing / spurious 双向审计。"""

    missing: list[Dict[str, Any]] = []
    low_rate_missing: list[Dict[str, Any]] = []
    spurious: list[Dict[str, Any]] = []
    for rs in RS_LABELS:
        for event, count in sorted((observed_by_rs.get(rs) or {}).items()):
            rate = count / max(1, int(rs_totals.get(rs, 0)))
            row = {
                "rs": rs,
                "event": event,
                "count": int(count),
                "rs_frame_rate": rate,
                "in_static_table": event in static_sets.get(rs, set()),
            }
            if event not in static_sets.get(rs, set()):
                if count >= min_count and rate >= min_rate:
                    missing.append(row)
                else:
                    low_rate_missing.append(row)
        if int(rs_totals.get(rs, 0)) <= 0:
            continue
        for event in sorted(static_sets.get(rs, set())):
            count = int((observed_by_rs.get(rs) or {}).get(event, 0))
            rate = count / max(1, int(rs_totals.get(rs, 0)))
            if count < min_count or rate < min_rate:
                spurious.append(
                    {
                        "rs": rs,
                        "event": event,
                        "count": count,
                        "rs_frame_rate": rate,
                        "in_static_table": True,
                    }
                )
    return missing, low_rate_missing, spurious


def _top_counter_rows(counter: Counter[str], limit: int) -> list[Dict[str, Any]]:
    """把 Counter 转成 JSON 友好的 top-k rows。"""

    return [{"key": key, "count": int(count)} for key, count in counter.most_common(max(0, int(limit)))]


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 RS x UE 共现审计。"""

    collection_dir = pathlib.Path(args.collection_dir)
    min_count = max(1, int(args.min_count))
    min_rate = max(0.0, float(args.min_rate))
    top_k = max(0, int(args.top_k))
    static_ue_sets, static_regular_sets = _split_static_sets()
    rs_totals = {rs: 0 for rs in RS_LABELS}
    all_ue_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    all_regular_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    raw_regular_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    pure_raw_regular_by_rs: Dict[str, Dict[str, int]] = {rs: {} for rs in RS_LABELS}
    multi_regular_annotation_frames_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    multi_pure_regular_frames_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    frames_with_regular_annotation_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    pure_regular_frames_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    regular_static_mismatch_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    raw_regular_remap_by_rs: Dict[str, int] = {rs: 0 for rs in RS_LABELS}
    raw_regular_remap_combo_counts: Counter[str] = Counter()
    raw_regular_remap_scenario_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    raw_regular_remap_route_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    focus_combo_counts: Counter[str] = Counter()
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
        scenario = str(result.get("scenario") or path.name[: -len("_result.json")])
        abnormal_skips, missing_skips = _skip_sets(result)
        skipped_routes_abnormal_or_missing += len(abnormal_skips) + len(missing_skips)
        routes = result.get("routes") or result.get("route_results") or []
        if isinstance(routes, Mapping):
            routes = routes.values()
        skipped_routes_failed += sum(1 for route in routes if isinstance(route, Mapping) and route.get("status") not in {None, "success"})
        for frame_scenario, route_id, frame in _iter_route_frames(result, scenario):
            try:
                rs = resolve_rs_target(frame).label
                event = resolve_event_target(frame, rs_label=rs)
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
                frames_with_regular_annotation_by_rs[rs] += 1
            if len(set(regular_codes)) > 1:
                multi_regular_annotation_frames_by_rs[rs] += 1
            for code in regular_codes:
                raw_regular_by_rs[rs][code] = raw_regular_by_rs[rs].get(code, 0) + 1
            if not event.abnormal:
                label = str(event.label)
                if is_regular_event(label):
                    pure_regular_frames_by_rs[rs] += 1
                    if len(set(regular_codes)) > 1:
                        multi_pure_regular_frames_by_rs[rs] += 1
                    raw_event = str(event.event_code)
                    pure_raw_regular_by_rs[rs][raw_event] = pure_raw_regular_by_rs[rs].get(raw_event, 0) + 1
                    mapped_event = canonical_regular_event_for_rs(rs, raw_event)
                    if raw_event != mapped_event:
                        assert raw_event != mapped_event
                        raw_regular_remap_by_rs[rs] += 1
                        combo_key = f"{rs}:{raw_event}"
                        route_key = f"{frame_scenario}/{route_id}"
                        raw_regular_remap_combo_counts[combo_key] += 1
                        raw_regular_remap_scenario_counts[combo_key][frame_scenario] += 1
                        raw_regular_remap_route_counts[combo_key][route_key] += 1
                        if args.focus_combo and combo_key == str(args.focus_combo):
                            focus_combo_counts[route_key] += 1
                    all_regular_by_rs[rs][label] = all_regular_by_rs[rs].get(label, 0) + 1
                    if label not in static_regular_sets.get(rs, set()):
                        regular_static_mismatch_by_rs[rs] += 1
                continue
            abnormal_frames += 1
            label = str(event.label)
            all_ue_by_rs[rs][label] = all_ue_by_rs[rs].get(label, 0) + 1

    missing_ue, low_rate_missing_ue, spurious_ue = _combo_rows(
        observed_by_rs=all_ue_by_rs,
        static_sets=static_ue_sets,
        rs_totals=rs_totals,
        min_count=min_count,
        min_rate=min_rate,
    )
    missing_regular, low_rate_missing_regular, spurious_regular = _combo_rows(
        observed_by_rs=all_regular_by_rs,
        static_sets=static_regular_sets,
        rs_totals=rs_totals,
        min_count=min_count,
        min_rate=min_rate,
    )
    raw_regular_remap_breakdown = []
    for combo_key, count in raw_regular_remap_combo_counts.most_common():
        rs, event = combo_key.split(":", 1)
        raw_regular_remap_breakdown.append(
            {
                "rs": rs,
                "raw_event": event,
                "mapped_event": canonical_regular_event_for_rs(rs, event),
                "count": int(count),
                "rs_frame_rate": int(count) / max(1, rs_totals.get(rs, 0)),
                "top_scenarios": _top_counter_rows(raw_regular_remap_scenario_counts[combo_key], top_k),
                "top_routes": _top_counter_rows(raw_regular_remap_route_counts[combo_key], top_k),
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
        "raw_regular_by_rs": raw_regular_by_rs,
        "pure_raw_regular_by_rs": pure_raw_regular_by_rs,
        "frames_with_regular_annotation_by_rs": frames_with_regular_annotation_by_rs,
        "pure_regular_frames_by_rs": pure_regular_frames_by_rs,
        # 兼容旧 JSON consumer：从本版本起该字段改为最终 GT 为 regular 的帧数。
        "regular_frame_total_by_rs": pure_regular_frames_by_rs,
        "multi_regular_annotation_frames_by_rs": multi_regular_annotation_frames_by_rs,
        "multi_pure_regular_frames_by_rs": multi_pure_regular_frames_by_rs,
        "multi_regular_frames_by_rs": multi_pure_regular_frames_by_rs,
        "multi_regular_annotation_frame_rate_by_rs": {
            rs: multi_regular_annotation_frames_by_rs[rs] / max(1, frames_with_regular_annotation_by_rs[rs])
            for rs in RS_LABELS
        },
        "multi_pure_regular_frame_rate_by_rs": {
            rs: multi_pure_regular_frames_by_rs[rs] / max(1, pure_regular_frames_by_rs[rs])
            for rs in RS_LABELS
        },
        "multi_regular_frame_rate_by_rs": {
            rs: multi_pure_regular_frames_by_rs[rs] / max(1, pure_regular_frames_by_rs[rs])
            for rs in RS_LABELS
        },
        "static_ue_by_rs": {rs: sorted(static_ue_sets[rs]) for rs in RS_LABELS},
        "static_regular_by_rs": {rs: sorted(static_regular_sets[rs]) for rs in RS_LABELS},
        "missing_ue_combinations": missing_ue,
        "low_rate_missing_ue_combinations": low_rate_missing_ue,
        "spurious_ue_combinations": spurious_ue,
        "missing_regular_combinations": missing_regular,
        "low_rate_missing_regular_combinations": low_rate_missing_regular,
        "spurious_regular_combinations": spurious_regular,
        # 旧字段名保留为 UE-only，兼容之前的 shell/json 解析。
        "missing_combinations": missing_ue,
        "low_rate_missing_combinations": low_rate_missing_ue,
        "spurious_combinations": spurious_ue,
        "missing_combinations_all": missing_ue + missing_regular,
        "low_rate_missing_combinations_all": low_rate_missing_ue + low_rate_missing_regular,
        "spurious_combinations_all": spurious_ue + spurious_regular,
        "regular_static_mismatch_total": sum(regular_static_mismatch_by_rs.values()),
        "regular_static_mismatch_rate": sum(regular_static_mismatch_by_rs.values()) / max(1, total_frames),
        "regular_static_mismatch_by_rs": regular_static_mismatch_by_rs,
        "regular_static_mismatch_rate_by_rs": {
            rs: regular_static_mismatch_by_rs[rs] / max(1, rs_totals[rs])
            for rs in RS_LABELS
        },
        "raw_regular_remap_total": sum(raw_regular_remap_by_rs.values()),
        "raw_regular_remap_rate": sum(raw_regular_remap_by_rs.values()) / max(1, total_frames),
        "raw_regular_remap_rate_over_pure_regular": sum(raw_regular_remap_by_rs.values()) / max(1, sum(pure_regular_frames_by_rs.values())),
        "raw_regular_remap_by_rs": raw_regular_remap_by_rs,
        "raw_regular_remap_rate_by_rs": {
            rs: raw_regular_remap_by_rs[rs] / max(1, rs_totals[rs])
            for rs in RS_LABELS
        },
        "raw_regular_remap_rate_over_pure_regular_by_rs": {
            rs: raw_regular_remap_by_rs[rs] / max(1, pure_regular_frames_by_rs[rs])
            for rs in RS_LABELS
        },
        "raw_regular_remap_breakdown": raw_regular_remap_breakdown,
        "focus_combo": str(args.focus_combo or ""),
        "focus_combo_top_routes": _top_counter_rows(focus_combo_counts, top_k),
        "candidate_count_after_static_table": {
            rs: len(EVENT_CANDIDATES_BY_RS.get(rs, []))
            for rs in RS_LABELS
        },
    }
    return report


def main() -> None:
    """CLI 入口。"""

    p = argparse.ArgumentParser(description="Audit RS x EVENT co-occurrence against static EVENT_CANDIDATES_BY_RS")
    p.add_argument("--collection-dir", type=str, default="keyframe_filter/collection_output")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--min-rate", type=float, default=0.001)
    p.add_argument("--top-k", type=int, default=20, help="Top scenario/route rows kept for regular mismatch attribution")
    p.add_argument(
        "--focus-combo",
        type=str,
        default="R5:R-E4",
        help="Optional RS:EVENT combo whose top routes are copied to focus_combo_top_routes",
    )
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



