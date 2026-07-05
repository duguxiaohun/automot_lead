#!/usr/bin/env python3
"""Generate per-scenario ROAD_STRUCTURE research artifacts.

This script reads LEAD data, route XML, and CARLA XODR summaries, then writes
auditable evidence under keyframe_filter/collection_output/rs_research/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional on headless/minimal envs
    Image = None
    ImageDraw = None

KEYFRAME_DIR = Path(__file__).resolve().parent
AUTOMOT_ROOT = KEYFRAME_DIR.parent
if str(KEYFRAME_DIR) not in sys.path:
    sys.path.insert(0, str(KEYFRAME_DIR))
if str(AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMOT_ROOT))

from collector import (  # noqa: E402
    RouteXmlIndex,
    SCENARIO_RULE_CONFIG,
    SCENARIO_RULE_KIND,
    SCENARIO_TO_ROAD_STRUCTURE,
    _DEFAULT_CARLA_ROOT,
    _DEFAULT_LEAD_DATA_ROOT,
    _DEFAULT_XML_ROOT,
    _extract_route_num,
    load_pickle_file,
)
from quick_start import (  # noqa: E402
    _build_scenario_policy_plan,
    _parse_xodr_spatial_index,
    _sample_xml_summary,
    _summarize_xodr,
    _xodr_spatial_probe_for_xml,
)
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402


RUN_PATTERNS = [
    re.compile(r"^(?P<town>Town\d+HD|Town\d+)_Rep\d+_route_(?P<route>.+?)_route(?P<route_index>\d+)_"),
    re.compile(r"^(?P<town>Town\d+HD|Town\d+)_Rep\d+_(?P<route>.+?)_route(?P<route_index>\d+)_"),
]

META_KEYS = [
    "pos_global",
    "theta",
    "speed",
    "traffic_light_state",
    "light_hazard",
    "stop_sign_close",
    "stop_sign_hazard",
    "is_junction",
    "is_intersection",
    "junction_id",
    "dist_to_junction",
    "distance_to_next_junction",
    "lane_id",
    "ego_lane_id",
    "ego_lane_width",
    "lane_type_str",
    "lane_change_str",
    "current_active_scenario_type",
]


def _json_default(value: Any) -> Any:
    """Convert numpy/scalar/path values to JSON-friendly objects."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _run_parts(run_id: str) -> Optional[Dict[str, str]]:
    for pattern in RUN_PATTERNS:
        match = pattern.match(run_id)
        if match:
            return match.groupdict()
    return None


def _town_from_run(run_id: str) -> str:
    parts = _run_parts(run_id)
    return parts["town"] if parts else "UNKNOWN"


def _expected_xml_name(run_id: str) -> Optional[str]:
    parts = _run_parts(run_id)
    if not parts:
        return None
    return f"{parts['town']}_route_{parts['route']}.xml"


def _select_diverse(items: List[Any], count: int) -> List[Any]:
    if not items:
        return []
    if len(items) <= count:
        return list(items)
    count = max(1, count)
    idxs = {
        round(i * (len(items) - 1) / max(1, count - 1))
        for i in range(count)
    }
    selected = [items[i] for i in sorted(idxs)]
    if len(selected) < count:
        seen = {str(item) for item in selected}
        for item in items:
            if str(item) in seen:
                continue
            selected.append(item)
            seen.add(str(item))
            if len(selected) >= count:
                break
    return selected


def _point_xy(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    try:
        if len(value) < 2:
            return None
        return (float(value[0]), float(value[1]))
    except Exception:
        return None


def _scalar(value: Any) -> Any:
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)[0]
        except Exception:
            return str(value)
    if isinstance(value, (str, bool, int)):
        return value
    try:
        out = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(out):
        return str(out)
    return round(out, 4)


def _meta_files(run_dir: Path) -> List[Path]:
    metas = run_dir / "metas"
    if not metas.exists():
        return []
    return sorted(metas.glob("*.pkl"))


def _rgb_files(run_dir: Path) -> List[Path]:
    rgb = run_dir / "rgb"
    if not rgb.exists():
        return []
    return sorted(rgb.glob("*.jpg"))


def _sample_meta_paths(run_dir: Path, max_samples: int = 3) -> List[Path]:
    return _select_diverse(_meta_files(run_dir), max_samples)


def _has_readable_meta(run_dir: Path) -> bool:
    for path in _sample_meta_paths(run_dir, 3):
        try:
            meta = load_pickle_file(path)
        except Exception:
            continue
        if isinstance(meta, dict):
            return True
    return False


def _select_research_runs(runs: List[Path], count: int) -> List[Path]:
    """Prefer readable-meta runs; skip the town if no readable meta exists."""
    if not runs:
        return []
    with_meta = [run for run in runs if _has_readable_meta(run)]
    if with_meta:
        return _select_diverse(with_meta, min(count, len(with_meta)))
    return []


def _meta_frame_row(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"frame_id": path.stem, "path": str(path)}
    try:
        meta = load_pickle_file(path)
    except Exception as exc:
        row["load_error"] = str(exc)
        return row
    if not isinstance(meta, dict):
        row["load_error"] = "not_dict"
        return row

    for key in META_KEYS:
        if key in meta:
            row[key] = _scalar(meta[key])
    pos = _point_xy(meta.get("pos_global"))
    if pos:
        row["ego_xy"] = [round(pos[0], 4), round(pos[1], 4)]

    finite_dist = {}
    for key, value in meta.items():
        if not key.startswith("dist_to_") and key != "distance_to_next_junction":
            continue
        scalar = _scalar(value)
        if isinstance(scalar, float):
            finite_dist[key] = scalar
    row["finite_dist_to"] = finite_dist
    return row


def _trajectory_rows(run_dir: Path, max_points: int = 160) -> List[Dict[str, Any]]:
    files = _meta_files(run_dir)
    if not files:
        return []
    step = max(1, len(files) // max_points)
    rows = []
    for path in files[::step]:
        row = _meta_frame_row(path)
        if "ego_xy" in row:
            rows.append(row)
    if files[-1] not in files[::step]:
        row = _meta_frame_row(files[-1])
        if "ego_xy" in row:
            rows.append(row)
    return rows


def _summarize_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta_paths = _meta_files(run_dir)
    rows = [_meta_frame_row(path) for path in _sample_meta_paths(run_dir, 3)]
    traffic = Counter(str(row.get("traffic_light_state")) for row in rows if "traffic_light_state" in row)
    active = Counter(str(row.get("current_active_scenario_type")) for row in rows if "current_active_scenario_type" in row)
    return {
        "available": bool(meta_paths) and any("load_error" not in row for row in rows),
        "run_dir": str(run_dir),
        "meta_count": len(meta_paths),
        "sampled_frame_ids": [row.get("frame_id") for row in rows],
        "traffic_light_values": dict(traffic),
        "active_scenario_values": dict(active),
        "has_ego_trace": any("ego_xy" in row for row in rows),
        "sampled_frames": rows,
    }


def _copy_xml(info: Any, dst: Path) -> Optional[str]:
    if info is None or not getattr(info, "path", None):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info.path, dst)
    return str(dst)


def _make_contact_sheet(run_dir: Path, out_path: Path, max_images: int = 3) -> Dict[str, Any]:
    rgb = _rgb_files(run_dir)
    if not rgb:
        return {"available": False, "reason": "rgb_files_missing"}
    sampled = _select_diverse(rgb, max_images)
    if Image is None or ImageDraw is None:
        return {"available": False, "reason": "PIL_unavailable", "sampled": [str(p) for p in sampled]}
    thumbs = []
    for path in sampled:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((384, 128))
            canvas = Image.new("RGB", (384, 152), "white")
            canvas.paste(img, (0, 18))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), path.stem, fill=(0, 0, 0))
            thumbs.append(canvas)
        except Exception:
            continue
    if not thumbs:
        return {"available": False, "reason": "rgb_open_failed", "sampled": [str(p) for p in sampled]}
    out = Image.new("RGB", (384 * len(thumbs), 152), "white")
    for idx, img in enumerate(thumbs):
        out.paste(img, (idx * 384, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=90)
    return {"available": True, "path": str(out_path), "sampled": [str(p) for p in sampled]}


def _draw_polyline(draw: Any, points: List[Tuple[float, float]], transform, color, width=2) -> None:
    if len(points) < 2:
        return
    draw.line([transform(p) for p in points], fill=color, width=width)


def _make_trace_plot(info: Any, run_dir: Path, out_path: Path) -> Dict[str, Any]:
    if Image is None or ImageDraw is None:
        return {"available": False, "reason": "PIL_unavailable"}
    route = list(getattr(info, "waypoints", []) or [])
    triggers = list(getattr(info, "trigger_points", []) or [])
    trace_rows = _trajectory_rows(run_dir)
    trace = [tuple(row["ego_xy"]) for row in trace_rows if "ego_xy" in row]
    points = route + triggers + trace
    if not points:
        return {"available": False, "reason": "no_points"}

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 20.0
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    width, height = 900, 700

    def tx(point):
        x, y = point
        sx = (x - min_x) / max(1e-6, max_x - min_x)
        sy = (y - min_y) / max(1e-6, max_y - min_y)
        return (int(40 + sx * (width - 80)), int(height - 40 - sy * (height - 80)))

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    _draw_polyline(draw, route, tx, (40, 90, 220), width=3)
    _draw_polyline(draw, trace, tx, (220, 70, 50), width=2)
    for point in triggers:
        x, y = tx(point)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(20, 150, 80), outline=(0, 0, 0))
    for idx, point in enumerate(trace[:1] + trace[-1:]):
        x, y = tx(point)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0))
        draw.text((x + 6, y - 8), "start" if idx == 0 else "end", fill=(0, 0, 0))
    draw.text((10, 10), "blue=XML route, red=ego trace, green=XML trigger", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "available": True,
        "path": str(out_path),
        "route_points": len(route),
        "trigger_points": len(triggers),
        "ego_trace_points": len(trace),
    }


def _match_xml(index: RouteXmlIndex, scenario: str, run_id: str):
    info = index.match(scenario, run_id)
    if info is not None:
        return info
    expected = _expected_xml_name(run_id)
    if not expected:
        return None
    for item in index.by_scenario.get(scenario, []):
        if item.path.name == expected:
            return item
    return None


def _scenario_runs_by_town(lead_data_root: Path, scenario: str) -> Dict[str, List[Path]]:
    scenario_dir = lead_data_root / scenario
    by_town: Dict[str, List[Path]] = defaultdict(list)
    if not scenario_dir.exists():
        return by_town
    for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
        should_exclude, _abnormal_info = is_abnormal_lead_route(run_dir, scenario)
        if should_exclude:
            continue
        by_town[_town_from_run(run_dir.name)].append(run_dir)
    return dict(sorted(by_town.items()))


def _rule_design_text(scenario: str, entry: Dict[str, Any], sampled_runs: Dict[str, Any]) -> str:
    logic = entry.get("generated_frame_label_logic", {})
    complete = logic.get("complete_investigation_status", {})
    candidates = ", ".join(entry.get("road_candidates", []))
    kind = entry.get("rule_kind")
    towns = ", ".join(entry.get("towns", {}).keys())
    notes = logic.get("logic_validation_from_samples", [])
    frame_rules = logic.get("frame_primary_rules", [])
    xodr_usage = logic.get("xodr_contract", {}).get("usage", [])
    incomplete = complete.get("incomplete_towns", {})

    lines = [
        f"# {scenario} ROAD_STRUCTURE 调研设计",
        "",
        "## 1. Scenario 语义与候选 RS",
        "",
        f"- rule_kind: `{kind}`",
        f"- candidate_pool: `{candidates}`",
        f"- towns: `{towns}`",
        "",
        "## 2. 样本覆盖",
        "",
    ]
    for town, item in sampled_runs.items():
        lines.append(f"- {town}: " + ", ".join(item.get("sampled_run_ids", [])))
    lines.extend([
        "",
        "## 3. XML 使用",
        "",
        "XML 用于 route 粗投影、trigger 窗口、scenario tag 参数和数据源追溯；不能单独作为帧级 RS 真值。",
        "",
        "## 4. XODR 使用",
        "",
    ])
    lines.extend(f"- {item}" for item in xodr_usage)
    lines.extend([
        "",
        "## 5. Meta 使用",
        "",
        "运行时优先读取 `pos_global/theta/speed`、灯态、junction、active scenario 和 finite `dist_to_*` 字段。",
        "meta 或匹配 XML 缺失时标为 `data_missing_skip`，不进入 RS/EVENT 规则标定；summary/可视化必须写明缺失原因。",
        "",
        "## 6. RGB 人工观察结论",
        "",
        "本轮自动生成 5 个分散 id 的 contact sheet，作为人工复核入口。正式 complete 前需要人工检查 contact sheet 与边界帧。",
        "必须重点核验 `rgb/*sample_contact_sheet.jpg` 是否与 `maps/*route_trigger_ego_trace.png` 的道路结构判断一致；",
        "若 RGB 与 XML/XODR 冲突，记录到 `failure_modes.md`，对应帧降到 medium/low confidence。",
        "",
        "## 7. 自车轨迹与地图对齐",
        "",
        "每个抽样 run 已生成 route/trigger/ego trace 图。若 trace 与 XML route 长期偏离，应标记 `projection_untrusted`。",
        "用于调阈值的 run 需要满足：route projection median error <= 3m、p90 error <= 5m、trigger 到 ego trace 最近距离 <= 20m。",
        "",
        "## 8. 帧级 RS 分段逻辑",
        "",
    ])
    lines.extend(f"- {item}" for item in frame_rules)
    lines.extend([
        "",
        "## 9. 置信度规则",
        "",
        "- high: scenario prior + XML window + XODR topology + meta signal/junction/active 至少三源一致。",
        "- medium: 两源一致，或强 meta 信号成立但 XODR/RGB 支持不足。",
        "- low: 只有 scenario prior 或弱 XODR hint；必须 review。",
        "",
        "## 10. Review 规则",
        "",
        "出现 XODR 缺失、route projection error、候选分数接近、RGB 与地图冲突时必须 review；meta/XML 缺失则直接 data_missing_skip。",
        "",
        "## 11. 已知失败模式",
        "",
    ])
    if incomplete:
        for town, reasons in incomplete.items():
            lines.append(f"- {town}: incomplete because {', '.join(reasons)}")
    else:
        lines.append("- 当前自动审计未发现 town 级输入缺口；仍需人工复核 RGB 边界帧后才能最终确认。")
    lines.extend([
        "",
        "## 12. 当前样本逻辑备注",
        "",
    ])
    lines.extend(f"- {item}" for item in notes)
    lines.extend([
        "",
        "## 13. 阈值来源与代码配置",
        "",
        "`thresholds.json` 里的每个阈值都必须写明 value/unit/source/supporting_runs/reviewed_artifacts/reason。",
        "如果当前文件只有裸数值或 rule_config 默认值，则只能视为 `threshold_source=temporary_default`，不能作为最终 complete 规则。",
        "",
        "## 14. 地图/RGB 对齐验收",
        "",
        "- map_rgb_alignment_status: `not_checked`",
        "- 人工验收入口：`maps/*route_trigger_ego_trace.png` 与 `rgb/*sample_contact_sheet.jpg`。",
        "- 若发现 route/trigger/ego trace 不贴合，先修 XML/run 匹配或 projection，不要直接调 RS 阈值。",
        "",
        "## 15. 错帧回查路径",
        "",
        "按 README -> map trace -> XML/XODR -> meta frame_features -> RGB contact/boundary -> thresholds/code 的顺序回查。",
        "错帧归因必须落到 XML、XODR、meta、RGB、arbitration、threshold 中的一类或多类。",
        "",
        "## 16. 完整性状态",
        "",
        f"- auto_input_complete: `{complete.get('is_complete', False)}`",
        "- manual_map_rgb_checked: `False`",
        "- final_complete: `False` until maps/RGB boundary frames and threshold provenance are manually checked.",
        "- 自动产物完成不等于人工最终完成；RGB 边界帧人工复核前，规则仍应保留 review 通道。",
        "",
    ])
    return "\n".join(lines)


def _threshold_payload(entry: Dict[str, Any], sampled_runs: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap current rule config with the provenance fields required by the protocol."""
    supporting_runs = [
        run_id
        for item in sampled_runs.values()
        for run_id in item.get("sampled_run_ids", [])
    ]
    rule_config = entry.get("rule_config", {})
    if not rule_config:
        return {
            "_status": "no_scenario_specific_thresholds",
            "_protocol": "Add value/unit/source/supporting_runs/reviewed_artifacts/reason before marking final_complete=true.",
            "supporting_runs": supporting_runs,
        }
    payload: Dict[str, Any] = {
        "_status": "temporary_defaults_need_manual_provenance",
        "_protocol": "Each threshold must point to map/RGB/meta/XML/XODR evidence before final use.",
    }
    for key, value in rule_config.items():
        payload[key] = {
            "value": value,
            "unit": "m" if key.endswith("_m") or key.endswith("_pre_m") or key.endswith("_post_m") else "unknown",
            "source": "temporary_default_rule_config",
            "supporting_runs": supporting_runs,
            "reviewed_artifacts": [],
            "reason": "Generated from existing rule_config; fill after map/RGB/meta review.",
        }
    return payload


def _build_scenario_audit_entry(
    index: RouteXmlIndex,
    scenario: str,
    run_by_town: Dict[str, List[Path]],
    carla_root: Path,
    samples_per_town: int,
    xodr_summary_cache: Dict[str, Dict[str, Any]],
    xodr_spatial_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    infos = index.by_scenario.get(scenario, [])
    by_town_xml: Dict[str, List[Any]] = defaultdict(list)
    for info in infos:
        by_town_xml[info.town or "UNKNOWN"].append(info)

    scenario_entry = {
        "rule_kind": SCENARIO_RULE_CONFIG.get(scenario, {}).get("kind", SCENARIO_RULE_KIND.get(scenario, "default_meta_map")),
        "rule_config": SCENARIO_RULE_CONFIG.get(scenario, {}),
        "road_candidates": [rs.value for rs in SCENARIO_TO_ROAD_STRUCTURE.get(scenario, [])],
        "xml_count": len(infos),
        "samples_per_town": samples_per_town,
        "towns": {},
    }

    all_towns = sorted(set(by_town_xml) | set(run_by_town))
    for town in all_towns:
        town_infos = by_town_xml.get(town, [])
        if town not in xodr_summary_cache:
            xodr_summary_cache[town] = _summarize_xodr(town, carla_root)
        if town not in xodr_spatial_cache:
            xodr_spatial_cache[town] = _parse_xodr_spatial_index(town, carla_root)
        xodr_summary = xodr_summary_cache[town]
        spatial_index = xodr_spatial_cache[town]
        town_runs = run_by_town.get(town, [])
        readable_runs = [run for run in town_runs if _has_readable_meta(run)]
        sampled_runs = _select_research_runs(town_runs, samples_per_town)
        sampled_infos = []
        for run_dir in sampled_runs:
            info = _match_xml(index, scenario, run_dir.name)
            if info is not None:
                sampled_infos.append((info, run_dir))
        if not sampled_infos and not town_runs:
            sampled_infos = [(info, None) for info in _select_diverse(town_infos, samples_per_town)]
        tag_counter = Counter()
        samples = []
        for info, run_dir in sampled_infos:
            for tag in info.scenario_tags:
                for key in tag:
                    if key not in {"name", "type"}:
                        tag_counter[key] += 1
            sample = _sample_xml_summary(info)
            sample["xodr_spatial_probe"] = _xodr_spatial_probe_for_xml(info, spatial_index)
            if run_dir is not None:
                sample["lead_meta_probe"] = _summarize_run_meta(run_dir)
                sample["sampled_run_id"] = run_dir.name
            else:
                sample["lead_meta_probe"] = {"available": False, "reason": "lead_run_not_sampled"}
            samples.append(sample)
        wp_counts = [len(info.waypoints) for info in town_infos]
        scenario_entry["towns"][town] = {
            "xml_count": len(town_infos),
            "lead_run_count": len(run_by_town.get(town, [])),
            "readable_lead_run_count": len(readable_runs),
            "waypoint_count_min": min(wp_counts) if wp_counts else 0,
            "waypoint_count_avg": round(sum(wp_counts) / len(wp_counts), 2) if wp_counts else 0,
            "waypoint_count_max": max(wp_counts) if wp_counts else 0,
            "xodr": xodr_summary,
            "top_tag_keys": dict(tag_counter.most_common(12)),
            "sampled_xml": samples,
        }
    scenario_entry["generated_frame_label_logic"] = _build_scenario_policy_plan(scenario, scenario_entry)
    return scenario_entry


def generate_research(args: argparse.Namespace) -> Dict[str, Any]:
    lead_data_root = Path(args.lead_data_root)
    xml_root = Path(args.xml_root)
    carla_root = Path(args.carla_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    index = RouteXmlIndex(xml_root)
    scenarios = sorted(SCENARIO_TO_ROAD_STRUCTURE)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario in wanted]

    manifest = {
        "lead_data_root": str(lead_data_root),
        "xml_root": str(xml_root),
        "carla_root": str(carla_root),
        "output_root": str(output_root),
        "samples_per_town": args.samples_per_town,
        "scenarios": {},
    }
    xodr_summary_cache: Dict[str, Dict[str, Any]] = {}
    xodr_spatial_cache: Dict[str, Dict[str, Any]] = {}

    for scenario in scenarios:
        scenario_dir = output_root / scenario
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)
        run_by_town = _scenario_runs_by_town(lead_data_root, scenario)
        entry = _build_scenario_audit_entry(
            index,
            scenario,
            run_by_town,
            carla_root,
            args.samples_per_town,
            xodr_summary_cache,
            xodr_spatial_cache,
        )
        sampled_runs_summary: Dict[str, Any] = {}

        for town, runs in run_by_town.items():
            sampled_runs = _select_research_runs(runs, args.samples_per_town)
            readable_run_count = sum(1 for run in runs if _has_readable_meta(run))
            sampled_runs_summary[town] = {
                "run_count": len(runs),
                "readable_run_count": readable_run_count,
                "sampled_run_ids": [run.name for run in sampled_runs],
            }
            if town not in xodr_summary_cache:
                xodr_summary_cache[town] = _summarize_xodr(town, carla_root)
            if town not in xodr_spatial_cache:
                xodr_spatial_cache[town] = _parse_xodr_spatial_index(town, carla_root)
            xodr_summary = xodr_summary_cache[town]
            _write_json(scenario_dir / "xodr" / f"{town}__xodr_summary.json", xodr_summary)
            spatial = xodr_spatial_cache[town]
            _write_json(
                scenario_dir / "xodr" / f"{town}__junction_signal_index.json",
                {
                    "available": spatial.get("available", False),
                    "path": spatial.get("path"),
                    "road_count": len(spatial.get("roads", {})),
                    "signal_count": len(spatial.get("signals", [])),
                    "parse_error": spatial.get("parse_error"),
                },
            )

            for run_dir in sampled_runs:
                info = _match_xml(index, scenario, run_dir.name)
                run_id = run_dir.name
                xml_base = scenario_dir / "xml" / town / run_id
                meta_base = scenario_dir / "meta" / town
                rgb_base = scenario_dir / "rgb" / town
                map_base = scenario_dir / "maps" / town

                xml_summary = _sample_xml_summary(info) if info is not None else {
                    "xml": None,
                    "run_id": run_id,
                    "expected_xml": _expected_xml_name(run_id),
                    "match_error": "xml_not_matched",
                }
                if info is not None:
                    xml_summary["xodr_spatial_probe"] = _xodr_spatial_probe_for_xml(info, spatial)
                    _copy_xml(info, xml_base.with_name(f"{run_id}__route.xml"))
                _write_json(xml_base.with_name(f"{run_id}__xml_summary.json"), xml_summary)

                meta_summary = _summarize_run_meta(run_dir)
                _write_json(meta_base / f"{run_id}__meta_probe.json", meta_summary)
                _write_jsonl(meta_base / f"{run_id}__frame_features.jsonl", [_meta_frame_row(path) for path in _sample_meta_paths(run_dir, 3)])

                contact = _make_contact_sheet(run_dir, rgb_base / f"{run_id}__sample_contact_sheet.jpg")
                _write_json(rgb_base / f"{run_id}__rgb_summary.json", contact)

                trace_plot = _make_trace_plot(info, run_dir, map_base / f"{run_id}__route_trigger_ego_trace.png") if info is not None else {"available": False, "reason": "xml_not_matched"}
                _write_json(map_base / f"{run_id}__map_summary.json", trace_plot)

        town_index = {
            town: {
                "run_count": len(runs),
                "xml_count": entry.get("towns", {}).get(town, {}).get("xml_count", 0),
                "xodr_available": entry.get("towns", {}).get(town, {}).get("xodr", {}).get("available", False),
            }
            for town, runs in run_by_town.items()
        }
        _write_json(scenario_dir / "town_index.json", town_index)
        _write_json(scenario_dir / "sampled_runs.json", sampled_runs_summary)
        _write_json(scenario_dir / "scenario_audit.json", entry)

        rules_dir = scenario_dir / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        (rules_dir / "scenario_rule_design.md").write_text(
            _rule_design_text(scenario, entry, sampled_runs_summary),
            encoding="utf-8",
        )
        _write_json(rules_dir / "thresholds.json", _threshold_payload(entry, sampled_runs_summary))
        _write_json(
            rules_dir / "confidence_policy.json",
            {
                "high": "scenario prior + XML window + XODR topology + meta signal/junction/active at least three-source agreement",
                "medium": "two-source agreement or strong meta signal with partial XODR/RGB support",
                "low": "scenario prior only or weak XODR hint; review_required=true",
            },
        )
        (rules_dir / "failure_modes.md").write_text(
            "\n".join(
                [
                    f"# {scenario} failure modes",
                    "",
                    "- XML route projection error high.",
                    "- XODR missing or spatial probe too coarse.",
                    "- meta missing/empty or active scenario not recorded.",
                    "- RGB shows road structure conflicting with map prior.",
                    "- RS boundary hysteresis too short or too long.",
                    "- Candidate scores close around transition frame.",
                    "- Threshold source is temporary_default or lacks reviewed map/RGB artifacts.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (scenario_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {scenario} RS Research",
                    "",
                    f"- rule_kind: `{entry.get('rule_kind')}`",
                    f"- road_candidates: `{', '.join(entry.get('road_candidates', []))}`",
                    f"- sampled towns: `{', '.join(run_by_town.keys())}`",
                    f"- auto_input_complete: `{entry.get('generated_frame_label_logic', {}).get('complete_investigation_status', {}).get('is_complete', False)}`",
                    "- map_rgb_alignment_status: `not_checked`",
                    "- manual_final_complete: `False`",
                    "",
                    "See `rules/scenario_rule_design.md` for the current scenario-specific logic.",
                    "See `maps/`, `rgb/`, `meta/`, `xml/`, and `xodr/` for the evidence chain.",
                    "Before changing runtime thresholds, check `maps/*route_trigger_ego_trace.png` and `rgb/*sample_contact_sheet.jpg`, then fill threshold provenance in `rules/thresholds.json`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        manifest["scenarios"][scenario] = {
            "path": str(scenario_dir),
            "town_count": len(run_by_town),
            "sampled_run_count": sum(len(item.get("sampled_run_ids", [])) for item in sampled_runs_summary.values()),
            "complete": entry.get("generated_frame_label_logic", {}).get("complete_investigation_status", {}).get("is_complete", False),
        }

    _write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-data-root", default=str(_DEFAULT_LEAD_DATA_ROOT))
    parser.add_argument("--xml-root", default=str(_DEFAULT_XML_ROOT))
    parser.add_argument("--carla-root", default=str(_DEFAULT_CARLA_ROOT))
    parser.add_argument("--output-root", default=str(KEYFRAME_DIR / "collection_output" / "rs_research"))
    parser.add_argument("--samples-per-town", type=int, default=5)
    parser.add_argument("--scenario", action="append", help="Limit to one scenario; may be repeated.")
    return parser.parse_args()


def main() -> None:
    manifest = generate_research(parse_args())
    print(f"wrote {manifest['output_root']}")
    print(f"scenarios={len(manifest['scenarios'])}")
    incomplete = [name for name, item in manifest["scenarios"].items() if not item.get("complete")]
    print(f"incomplete={len(incomplete)}")
    if incomplete:
        print("incomplete_scenarios=" + ",".join(incomplete[:20]))


if __name__ == "__main__":
    main()
