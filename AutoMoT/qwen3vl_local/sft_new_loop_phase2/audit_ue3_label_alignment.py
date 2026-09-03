#!/usr/bin/env python3
"""把 UE3 逐帧 RGB 决策与源 EVENT 规则、时序 span 和数据索引联表。

本工具只做诊断，不修改 collection_output、frame_index、prompt、checkpoint 或正式指标。
RGB 结论必须来自已逐帧查看并入库的 decisions JSONL，不能从 scenario 名称反推。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream  # noqa: E402
from qwen3vl_local.sft_new_loop_phase2.sampling import route_diversity_report  # noqa: E402


Identity = Tuple[str, str, int, str]
VISIBLE_CLASS = "VISIBLE_ACTIVE"
NON_VISIBLE_CLASSES = {
    "PRE_EVENT",
    "POST_EVENT",
    "DOMAIN_CONFLICT",
    "2RGB_UNOBSERVABLE",
    "AMBIGUOUS",
}
METRIC_KEYS = (
    "dist_to_cutin_vehicle",
    "speed_reduced_by_obj_distance",
    "brake_cutin",
    "vehicle_hazard",
    "defect_conflict_vehicle",
    "hard_decel",
    "changed_route",
    "speed",
    "target_speed",
)


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSON object。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            yield row


def _identity(row: Mapping[str, Any]) -> Identity:
    """返回 RGB audit 使用的稳定 case 身份。"""

    return (
        str(row.get("scenario", "")),
        str(row.get("route_id", "")),
        int(row.get("frame_id", -1)),
        str(row.get("question_domain", "")),
    )


def _event_payload(annotation: Mapping[str, Any]) -> Mapping[str, Any]:
    """读取 collector 最终 EVENT payload，兼容旧 event_evidence。"""

    return annotation.get("frame_event_annotation") or annotation.get("event_evidence") or {}


def _ue3_spans(annotations: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int, int]]:
    """返回 route 内连续 U-E3 frame-id span。"""

    frame_ids = sorted(
        int(annotation["frame_id"])
        for annotation in annotations
        if annotation.get("frame_id") is not None
        and str(annotation.get("primary_event")) == "U-E3"
    )
    spans: List[List[int]] = []
    for frame_id in frame_ids:
        if not spans or frame_id != spans[-1][-1] + 1:
            spans.append([frame_id])
        else:
            spans[-1].append(frame_id)
    return [(span[0], span[-1], len(span)) for span in spans]


def _span_for(frame_id: int, spans: Sequence[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    """查找包含目标帧的 U-E3 span。"""

    return next((span for span in spans if span[0] <= frame_id <= span[1]), None)


def _compact_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """只保留此次 RGB 归因需要的稳定 EVENT 指标。"""

    metrics = payload.get("metrics") or {}
    return {key: metrics.get(key) for key in METRIC_KEYS}


def _load_and_validate_decisions(
    decisions_path: pathlib.Path,
    manifest_path: pathlib.Path,
) -> Dict[Identity, Dict[str, Any]]:
    """校验 RGB decisions 与 audit manifest 一一对应。"""

    manifest = {_identity(row): row for row in _iter_jsonl(manifest_path)}
    decisions: Dict[Identity, Dict[str, Any]] = {}
    for row in _iter_jsonl(decisions_path):
        key = _identity(row)
        if key in decisions:
            raise ValueError(f"duplicate RGB decision: {key}")
        decisions[key] = row
    missing = sorted(set(manifest) - set(decisions))
    extra = sorted(set(decisions) - set(manifest))
    if missing or extra:
        raise ValueError(
            f"RGB decision identity mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    return decisions


def _collect_source_cases(
    decisions: Mapping[Identity, Mapping[str, Any]],
    collection_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    """只扫描 decisions 涉及的 scenario/route，并联接源逐帧标注。"""

    wanted: Dict[str, Dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for scenario, route_id, frame_id, _ in decisions:
        wanted[scenario][route_id].add(frame_id)

    source_cases: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for scenario, route_frames in sorted(wanted.items()):
        source_path = collection_dir / f"{scenario}_result.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source annotation: {source_path}")
        remaining_routes = set(route_frames)
        for route in _iter_routes_stream(source_path):
            route_id = str(route.get("route_id") or "")
            if route_id not in remaining_routes:
                continue
            annotations = list(route.get("annotations") or [])
            by_frame = {
                int(annotation["frame_id"]): annotation
                for annotation in annotations
                if annotation.get("frame_id") is not None
            }
            spans = _ue3_spans(annotations)
            for frame_id in sorted(route_frames[route_id]):
                annotation = by_frame.get(frame_id)
                if annotation is None:
                    continue
                payload = _event_payload(annotation)
                span = _span_for(frame_id, spans)
                source_cases[(scenario, route_id, frame_id)] = {
                    "source_annotation": str(source_path),
                    "source_primary_event": str(annotation.get("primary_event") or ""),
                    "source_primary_rs": str(annotation.get("primary_road_structure") or ""),
                    "source_rules_fired": sorted(str(rule) for rule in (payload.get("rules_fired") or [])),
                    "source_metrics": _compact_metrics(payload),
                    "source_ue3_span": (
                        {
                            "start_frame": span[0],
                            "end_frame": span[1],
                            "length": span[2],
                            "offset": frame_id - span[0],
                        }
                        if span is not None
                        else None
                    ),
                }
            remaining_routes.remove(route_id)
            if not remaining_routes:
                break
        if remaining_routes:
            raise ValueError(
                f"source routes not found in {source_path}: {sorted(remaining_routes)}"
            )

    cases: List[Dict[str, Any]] = []
    missing_source: List[Identity] = []
    for identity, decision in sorted(decisions.items()):
        source = source_cases.get(identity[:3])
        if source is None:
            missing_source.append(identity)
            continue
        cases.append({**dict(decision), **source})
    if missing_source:
        raise ValueError(f"audited frames missing from source annotations: {missing_source[:10]}")
    return cases


def _number_summary(values: Sequence[float]) -> Dict[str, float]:
    """输出适合比较视觉类别的稳健数值摘要。"""

    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": min(ordered),
        "median": statistics.median(ordered),
        "max": max(ordered),
    }


def _metric_report(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """按 RGB visual class 汇总源 metric；不据此自动生成阈值。"""

    report: Dict[str, Any] = {}
    classes = sorted({str(case["visual_class"]) for case in cases})
    for visual_class in classes:
        class_cases = [case for case in cases if str(case["visual_class"]) == visual_class]
        metrics: Dict[str, Any] = {}
        for key in METRIC_KEYS:
            raw = [case["source_metrics"].get(key) for case in class_cases]
            present = [value for value in raw if value is not None]
            if present and all(isinstance(value, bool) for value in present):
                metrics[key] = {
                    "present": len(present),
                    "missing": len(raw) - len(present),
                    "true": sum(bool(value) for value in present),
                    "false": sum(not bool(value) for value in present),
                }
            elif present and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
                metrics[key] = {
                    **_number_summary([float(value) for value in present]),
                    "missing": len(raw) - len(present),
                }
            else:
                metrics[key] = {"present": len(present), "missing": len(raw) - len(present)}
        report[visual_class] = metrics
    return report


def _rules_report(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """检查同一源规则是否同时覆盖相反的逐帧 RGB 结论。"""

    by_rule: Dict[str, Counter[str]] = defaultdict(Counter)
    by_signature: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        visual_class = str(case["visual_class"])
        rules = list(case.get("source_rules_fired") or [])
        for rule in rules or ["<NO_RULE>"]:
            by_rule[str(rule)][visual_class] += 1
        signature = "+".join(rules) if rules else "<NO_RULE>"
        by_signature[signature][visual_class] += 1

    def render(groups: Mapping[str, Counter[str]]) -> Dict[str, Any]:
        return {
            key: {
                "cases": sum(counts.values()),
                "visual_class_counts": dict(sorted(counts.items())),
                "conflicts_visible_vs_nonvisible": (
                    counts.get(VISIBLE_CLASS, 0) > 0
                    and any(counts.get(name, 0) > 0 for name in NON_VISIBLE_CLASSES)
                ),
            }
            for key, counts in sorted(groups.items())
        }

    return {"by_rule": render(by_rule), "by_rule_signature": render(by_signature)}


def _index_report(index_path: pathlib.Path) -> Dict[str, Any]:
    """审计已有 Phase2 index 中 UE3 的 route 集中度与重复身份。"""

    if not index_path.is_file():
        return {"available": False, "path": str(index_path)}
    rows = [
        row
        for row in _iter_jsonl(index_path)
        if str(row.get("target_event_class")) == "UE3"
        and not bool(row.get("invalid_event_context"))
    ]
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "UNKNOWN"))].append(row)
    return {
        "available": True,
        "path": str(index_path),
        "ue3_cases": len(rows),
        "duplicate_case_identities": len(rows)
        - len({(row.get("scenario"), row.get("route_id"), row.get("frame_id")) for row in rows}),
        "by_split": {
            split: route_diversity_report(split_rows)
            for split, split_rows in sorted(by_split.items())
        },
    }


def build_report(
    *,
    audit_root: pathlib.Path,
    decisions_path: pathlib.Path,
    collection_dir: pathlib.Path,
    index_path: pathlib.Path,
    min_visible_routes_for_prompt_change: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """生成联表报告与逐例明细。"""

    decisions = _load_and_validate_decisions(decisions_path, audit_root / "manifest.jsonl")
    cases = _collect_source_cases(decisions, collection_dir)
    class_counts = Counter(str(case["visual_class"]) for case in cases)
    visible_routes = {
        (str(case["scenario"]), str(case["route_id"]))
        for case in cases
        if str(case["visual_class"]) == VISIBLE_CLASS
    }
    route_counts = Counter((str(case["scenario"]), str(case["route_id"])) for case in cases)
    rules = _rules_report(cases)
    conflicting_signatures = [
        signature
        for signature, item in rules["by_rule_signature"].items()
        if item["conflicts_visible_vs_nonvisible"]
    ]
    return (
        {
            "format": "sft_new_loop_phase2_ue3_label_alignment_audit_v1",
            "official_metric": False,
            "mutation": False,
            "contract": (
                "Diagnostic only. RGB decisions are joined to source annotations and index distribution; "
                "this report never relabels validation, changes prompt, selects checkpoints, or opens unseen."
            ),
            "audit_root": str(audit_root),
            "decisions": str(decisions_path),
            "collection_dir": str(collection_dir),
            "cases": len(cases),
            "visual_class_counts": dict(sorted(class_counts.items())),
            "audited_route_counts": {
                f"{scenario}/{route_id}": count
                for (scenario, route_id), count in sorted(
                    route_counts.items(), key=lambda pair: (-pair[1], pair[0])
                )
            },
            "visible_active_routes": len(visible_routes),
            "min_visible_routes_for_prompt_change": int(min_visible_routes_for_prompt_change),
            "prompt_evidence_route_floor_met": len(visible_routes)
            >= int(min_visible_routes_for_prompt_change),
            "rules": rules,
            "conflicting_rule_signatures": conflicting_signatures,
            "automatic_source_rule_relabel_safe": not conflicting_signatures,
            "metrics_by_visual_class": _metric_report(cases),
            "index_ue3_distribution": _index_report(index_path),
            "decision": {
                "change_prompt_now": False,
                "automatic_metric_threshold_relabel_now": False,
                "use_route_diverse_train_sampling": True,
                "preserve_legacy_val_test_sampling": True,
                "reason": (
                    "The same source rule signature covers RGB-confirmed VISIBLE_ACTIVE and non-visible "
                    "PRE/POST/DOMAIN/2RGB/AMBIGUOUS cases, while visible evidence is route-concentrated."
                ),
            },
        },
        cases,
    )


def _markdown(report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> str:
    """生成人类可读审计记录。"""

    index_report = report["index_ue3_distribution"]
    lines = [
        "# UE3 RGB × source-label alignment audit",
        "",
        f"- official metric: `{report['official_metric']}`",
        f"- mutation: `{report['mutation']}`",
        f"- audited cases: `{report['cases']}`",
        f"- visual classes: `{report['visual_class_counts']}`",
        f"- visible-active routes: `{report['visible_active_routes']}` / required evidence floor "
        f"`{report['min_visible_routes_for_prompt_change']}`",
        f"- conflicting source rule signatures: `{report['conflicting_rule_signatures']}`",
        "",
        "## Decision",
        "",
        "- Keep production prompt v3 unchanged.",
        "- Do not derive an automatic UE3 relabel threshold from these metadata fields.",
        "- Rebuild train rows with route-round-robin sampling before any new training.",
        "- Preserve legacy val/test sampling so frozen case identities remain comparable.",
        "- Do not open unseen-456 from this diagnostic report.",
        "",
        "同一源规则同时生成 RGB 清晰正例与 PRE/POST/问题域/不可观察样本，说明仅凭该规则名或"
        "单一距离阈值无法安全重标。当前代码修正只降低 train 连续 span 的 route 过度权重，不改 prompt/标签，"
        "也不改变 val/test 的 legacy sampler。",
        "",
        "## Audited cases",
        "",
        "| visual class | scenario/route | frame | source span | rules |",
        "|---|---|---:|---|---|",
    ]
    for case in cases:
        span = case.get("source_ue3_span") or {}
        span_text = (
            f"{span.get('start_frame')}-{span.get('end_frame')} (len={span.get('length')})"
            if span
            else "N/A"
        )
        route = f"{case['scenario']}/{case['route_id']}"
        rules = ", ".join(case.get("source_rules_fired") or []) or "N/A"
        lines.append(
            f"| {case['visual_class']} | {route} | {case['frame_id']} | {span_text} | {rules} |"
        )
    lines.extend(["", "## Existing index UE3 distribution", ""])
    if not index_report.get("available"):
        lines.append(f"Index not found locally: `{index_report['path']}`. Re-run on the training machine.")
    else:
        lines.extend(
            [
                f"- UE3 cases: `{index_report['ue3_cases']}`",
                f"- duplicate identities: `{index_report['duplicate_case_identities']}`",
                "",
                "| split | cases | routes | max cases/route |",
                "|---|---:|---:|---:|",
            ]
        )
        for split, item in index_report.get("by_split", {}).items():
            lines.append(
                f"| {split} | {item['cases']} | {item['unique_routes']} | {item['max_cases_per_route']} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", default="checkpoints/ue3_route_diverse_full_rgb_audit")
    parser.add_argument(
        "--decisions",
        default=str(_THIS.with_name("ue3_route_diverse_rgb_decisions_v1.jsonl")),
    )
    parser.add_argument("--collection-dir", default="keyframe_filter/collection_output")
    parser.add_argument("--index", default="checkpoints/sft_new_loop_phase2_data/frame_index.jsonl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-visible-routes-for-prompt-change", type=int, default=15)
    args = parser.parse_args()
    if int(args.min_visible_routes_for_prompt_change) <= 0:
        raise ValueError("--min-visible-routes-for-prompt-change must be positive")
    return args


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    audit_root = pathlib.Path(args.audit_root)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else audit_root / "label_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    report, cases = build_report(
        audit_root=audit_root,
        decisions_path=pathlib.Path(args.decisions),
        collection_dir=pathlib.Path(args.collection_dir),
        index_path=pathlib.Path(args.index),
        min_visible_routes_for_prompt_change=int(args.min_visible_routes_for_prompt_change),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report, cases), encoding="utf-8")
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] UE3 label alignment audit: {output_dir}")


if __name__ == "__main__":
    main()
