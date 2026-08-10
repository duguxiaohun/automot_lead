#!/usr/bin/env python3
"""Render the original phase-1 labels over every RGB frame selected for review.

Unlike ``keyframe_filter/rs_full_frame_review.py``, this tool never recomputes
RS/EVENT and never drops a route merely because a probe meta file is unreadable.
It starts from the answer table's real ``rgb_review_samples``.  Consequently a
Town is reviewed with the exact routes that established every initial
``scenario × RS × EVENT`` row, while the rendered overlay remains the original
collection annotation that the answer table must eventually correct or split.

The generated sheets are evidence only.  ``ready_for_human_rgb_read`` means the
frames are available to inspect, not that they have been visually adjudicated.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping

from PIL import Image, ImageDraw


_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402


RS_COLORS = {"R1": (95, 116, 142), "R2": (214, 89, 76), "R3": (85, 139, 214), "R4": (50, 151, 91), "R5": (156, 105, 205)}
EVENT_COLORS = {
    "R-E1": (95, 116, 142), "R-E2": (85, 139, 214), "R-E3": (62, 148, 172),
    "R-E4": (50, 151, 91), "R-E5": (156, 105, 205), "U-E1": (210, 83, 65),
    "U-E2": (214, 89, 76), "U-E3": (220, 128, 58), "U-E4": (200, 93, 143),
    "U-E5": (176, 76, 132), "U-E6": (190, 54, 54), "U-E7": (142, 80, 196),
    "U-E8": (172, 102, 45),
}


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("scenario") or "UNKNOWN"), str(row.get("rs") or "UNKNOWN"), str(row.get("event") or "UNKNOWN"))


def _load_targets(table_path: pathlib.Path, scenarios: set[str] | None) -> tuple[dict[str, dict[str, dict[str, set[tuple[str, str, str]]]]], list[dict[str, Any]]]:
    """Return route targets indexed by scenario/town and the source rows."""

    payload = json.loads(table_path.read_text(encoding="utf-8"))
    targets: dict[str, dict[str, dict[str, set[tuple[str, str, str]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        scenario, rs, event = _key(row)
        if scenario == "noScenarios" or (scenarios is not None and scenario not in scenarios):
            continue
        rows.append(dict(row))
        group_key = (scenario, rs, event)
        for sample in row.get("rgb_review_samples", []) or []:
            town = str(sample.get("town") or "UNKNOWN")
            route_id = str(sample.get("route_id") or "")
            if route_id:
                targets[scenario][town][route_id].add(group_key)
    return targets, rows


def _annotation_labels(annotation: Mapping[str, Any]) -> tuple[str, str]:
    rs = str(annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label") or "UNKNOWN")
    event = str(annotation.get("primary_event") or (annotation.get("frame_event_annotation") or {}).get("label") or "UNKNOWN")
    return rs, event


def _render_route(
    annotations: Iterable[Mapping[str, Any]], run_dir: pathlib.Path, out_dir: pathlib.Path, *, frames_per_sheet: int, cols: int
) -> tuple[int, dict[str, int], dict[str, int], list[str]]:
    """Render every annotated RGB frame with labels exactly as stored in collection output."""

    frames: list[tuple[int, pathlib.Path, str, str]] = []
    rs_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    for annotation in annotations:
        try:
            frame_id = int(annotation.get("frame_id"))
        except (TypeError, ValueError):
            continue
        rgb = _rgb_path(run_dir, frame_id)
        if rgb is None:
            continue
        rs, event = _annotation_labels(annotation)
        frames.append((frame_id, rgb, rs, event))
        rs_counts[rs] += 1
        event_counts[event] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h, header_h, footer_h = 576, 192, 26, 24
    rows = max(1, math.ceil(frames_per_sheet / cols))
    sheets: list[str] = []
    for sheet_idx, start in enumerate(range(0, len(frames), frames_per_sheet)):
        chunk = frames[start:start + frames_per_sheet]
        sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + header_h + footer_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (frame_id, rgb_path, rs, event) in enumerate(chunk):
            x = (idx % cols) * tile_w
            y = (idx // cols) * (tile_h + header_h + footer_h)
            try:
                image = Image.open(rgb_path).convert("RGB")
                image.thumbnail((tile_w, tile_h))
                tile = Image.new("RGB", (tile_w, tile_h), "black")
                tile.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
                sheet.paste(tile, (x, y + header_h))
            except OSError:
                draw.rectangle((x, y + header_h, x + tile_w - 1, y + header_h + tile_h - 1), outline="red", width=3)
                draw.text((x + 8, y + header_h + 8), "unreadable RGB", fill="red")
            draw.rectangle((x, y, x + tile_w - 1, y + header_h - 1), fill=RS_COLORS.get(rs, (60, 60, 60)))
            draw.text((x + 6, y + 6), f"frame={frame_id}  RS={rs}", fill="white")
            draw.rectangle((x, y + header_h + tile_h, x + tile_w - 1, y + header_h + tile_h + footer_h - 1), fill=EVENT_COLORS.get(event, (50, 50, 50)))
            draw.text((x + 6, y + header_h + tile_h + 5), f"EVENT={event}", fill="white")
        path = out_dir / f"all_frames_{sheet_idx:04d}_f{chunk[0][0]}_to_f{chunk[-1][0]}.jpg"
        sheet.save(path, quality=92)
        sheets.append(str(path))
    return len(frames), dict(sorted(rs_counts.items())), dict(sorted(event_counts.items())), sheets


def build_review(*, table_path: pathlib.Path, collection_dir: pathlib.Path, data_root: pathlib.Path, output_dir: pathlib.Path, scenarios: set[str] | None, frames_per_sheet: int, cols: int) -> dict[str, Any]:
    """Write full-frame, original-label evidence for every table-referenced route."""

    targets, source_rows = _load_targets(table_path, scenarios)
    report: dict[str, Any] = {
        "format": "phase1_fullframe_rgb_label_review_v1",
        "source_answer_table": str(table_path),
        "source_annotation": "collection_output original route annotations; no runtime reclassification",
        "excluded_scenarios": ["noScenarios"],
        "manual_status": "ready_for_human_rgb_read",
        "scenarios": {},
    }
    expected_pairs = {(str(row["scenario"]), str(sample.get("town") or "UNKNOWN"), str(sample.get("route_id") or "")) for row in source_rows for sample in row.get("rgb_review_samples", []) or [] if sample.get("route_id")}
    rendered_pairs: set[tuple[str, str, str]] = set()

    for scenario in sorted(targets):
        result_path = collection_dir / f"{scenario}_result.json"
        if not result_path.exists():
            report["scenarios"][scenario] = {"status": "missing_collection_result"}
            continue
        scenario_out: dict[str, Any] = {"towns": {}, "status": "ready_for_human_rgb_read"}
        wanted_ids = {route_id for town_routes in targets[scenario].values() for route_id in town_routes}
        found_ids: set[str] = set()
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if route_id not in wanted_ids:
                continue
            found_ids.add(route_id)
            town = str(route.get("xml_town") or "UNKNOWN")
            # The table's sample town is authoritative when the collector stored no top-level town.
            if town not in targets[scenario] or route_id not in targets[scenario][town]:
                town = next((candidate for candidate, routes in targets[scenario].items() if route_id in routes), town)
            run_dir = data_root / scenario / route_id
            abnormal, abnormal_info = is_abnormal_lead_route(run_dir, scenario)
            route_out = output_dir / scenario / town / route_id
            group_keys = sorted("/".join(item) for item in targets[scenario][town][route_id])
            if abnormal or not run_dir.is_dir():
                route_summary: dict[str, Any] = {
                    "route_id": route_id, "groups": group_keys, "manual_status": "blocked_missing_or_abnormal",
                    "abnormal": bool(abnormal), "abnormal_info": abnormal_info, "run_dir_exists": run_dir.is_dir(),
                }
            else:
                count, rs_counts, event_counts, sheets = _render_route(
                    route.get("annotations", []) or [], run_dir, route_out / "sheets", frames_per_sheet=frames_per_sheet, cols=cols
                )
                route_summary = {
                    "route_id": route_id, "groups": group_keys, "rendered_frames": count,
                    "primary_rs_distribution": rs_counts, "primary_event_distribution": event_counts,
                    "full_frame_sheets": sheets, "manual_status": "ready_for_human_rgb_read",
                }
                rendered_pairs.add((scenario, town, route_id))
            town_out = scenario_out["towns"].setdefault(town, {"routes": []})
            town_out["routes"].append(route_summary)
        for town, town_routes in targets[scenario].items():
            town_out = scenario_out["towns"].setdefault(town, {"routes": []})
            expected_ids = sorted(town_routes)
            actual_ids = sorted(route["route_id"] for route in town_out["routes"])
            town_out["expected_route_ids"] = expected_ids
            town_out["found_route_ids"] = actual_ids
            town_out["three_id_requirement"] = "met" if len(actual_ids) >= 3 else "insufficient_source_routes"
            town_out["manual_status"] = "ready_for_human_rgb_read" if len(actual_ids) >= 3 else "blocked_insufficient_source_routes"
        report["scenarios"][scenario] = scenario_out
        missing = sorted(wanted_ids.difference(found_ids))
        if missing:
            scenario_out["missing_route_ids_from_collection"] = missing

    report["expected_table_sample_routes"] = len(expected_pairs)
    report["rendered_table_sample_routes"] = len(rendered_pairs)
    report["missing_table_sample_routes"] = sorted(expected_pairs.difference(rendered_pairs))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full RGB histories with original phase-1 RS/EVENT labels")
    parser.add_argument("--answer-table", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json"))
    parser.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    parser.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    parser.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_fullframe_rgb_original_labels"))
    parser.add_argument("--scenarios", default="all", help="all or comma-separated scenario names")
    parser.add_argument("--frames-per-sheet", type=int, default=12)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()
    scenarios = None if args.scenarios == "all" else {item.strip() for item in args.scenarios.split(",") if item.strip()}
    report = build_review(
        table_path=pathlib.Path(args.answer_table), collection_dir=pathlib.Path(args.collection_dir),
        data_root=pathlib.Path(args.data_root), output_dir=pathlib.Path(args.output_dir), scenarios=scenarios,
        frames_per_sheet=args.frames_per_sheet, cols=args.cols,
    )
    print(f"phase1 full-frame original-label review: scenarios={len(report['scenarios'])} rendered_routes={report['rendered_table_sample_routes']} output={args.output_dir}")


if __name__ == "__main__":
    main()
