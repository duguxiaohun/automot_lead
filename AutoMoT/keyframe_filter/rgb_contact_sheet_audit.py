#!/usr/bin/env python3
"""Create per-route RGB contact sheets for manual R4/R5 audit.

This tool is intentionally visual-first: every `rgb/*.jpg` frame of each route
is read and placed into chronological contact-sheet pages. It does not decide
R4/R5 by weak color heuristics; the sheets are evidence for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "AutoMoT") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "AutoMoT"))

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402


@dataclass
class RouteSheetResult:
    scenario: str
    route_id: str
    frame_count: int
    page_count: int
    abnormal_duration: bool
    abnormal_reason: str
    sheet_dir: str
    first_sheet: str
    status: str
    message: str = ""


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {part for chunk in value.split(",") for part in chunk.split() if part}
    return out or None


def _frame_id(path: pathlib.Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1


def _iter_routes(data_root: pathlib.Path, scenarios: set[str] | None) -> list[tuple[str, str, pathlib.Path]]:
    routes: list[tuple[str, str, pathlib.Path]] = []
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if (route_dir / "rgb").is_dir():
                routes.append((scenario, route_dir.name, route_dir))
    return routes


def _route_out_dir(output_dir: pathlib.Path, scenario: str, route_id: str) -> pathlib.Path:
    return output_dir / "route_sheets" / scenario / route_id


def _load_annotation_index(result_dir: pathlib.Path | None) -> dict[tuple[str, str], dict]:
    """Load per-route annotation summaries keyed by (scenario, route_id)."""
    if not result_dir:
        return {}
    if not result_dir.exists():
        return {}
    index: dict[tuple[str, str], dict] = {}
    for result_path in sorted(result_dir.glob("*_result.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scenario = str(data.get("scenario") or result_path.stem.replace("_result", ""))
        for route in data.get("routes", []) or []:
            route_id = str(route.get("route_id") or "")
            if not route_id:
                continue
            rs_dist = route.get("primary_rs_distribution", {}) or {}
            event_dist = route.get("primary_event_distribution", {}) or {}
            total = int(route.get("num_frames", 0) or sum(int(v or 0) for v in rs_dist.values()))
            r2_frames = int(rs_dist.get("R2", 0) or 0)
            index[(scenario, route_id)] = {
                "annotation_status": route.get("status", ""),
                "annotation_num_frames": total,
                "primary_rs_distribution": dict(sorted(rs_dist.items())),
                "primary_event_distribution": dict(sorted(event_dist.items())),
                "r2_frames": r2_frames,
                "r2_ratio": round(r2_frames / total, 6) if total > 0 else 0.0,
                "has_r2": r2_frames > 0,
                "review_required_frames": int(route.get("review_required_frames", 0) or 0),
                "review_required_ratio": float(route.get("review_required_ratio", 0.0) or 0.0),
            }
    return index


def _load_thumb(path: pathlib.Path, thumb_w: int, thumb_h: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        thumb = np.full((thumb_h, thumb_w, 3), 45, dtype=np.uint8)
        cv2.putText(thumb, "missing", (8, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return thumb
    return cv2.resize(bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)


def _draw_tile(path: pathlib.Path, thumb_w: int, thumb_h: int, label_h: int) -> np.ndarray:
    tile = np.full((thumb_h + label_h, thumb_w, 3), 245, dtype=np.uint8)
    tile[:thumb_h] = _load_thumb(path, thumb_w, thumb_h)
    cv2.rectangle(tile, (0, 0), (thumb_w - 1, 20), (0, 0, 0), -1)
    cv2.putText(tile, path.stem, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _write_page(
    page_paths: list[pathlib.Path],
    out_path: pathlib.Path,
    *,
    scenario: str,
    route_id: str,
    page_idx: int,
    page_count: int,
    total_frames: int,
    abnormal: bool,
    cols: int,
    thumb_w: int,
    thumb_h: int,
    label_h: int,
    jpeg_quality: int,
) -> None:
    rows = int(np.ceil(len(page_paths) / cols))
    header_h = 72
    tile_h = thumb_h + label_h
    sheet = np.full((header_h + rows * tile_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    title = (
        f"{scenario}/{route_id} | page {page_idx + 1}/{page_count} | "
        f"frames={total_frames} | abnormal_duration={abnormal}"
    )
    cv2.putText(sheet, title[:180], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "All RGB frames on this page are shown in chronological order.",
        (8, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    for idx, path in enumerate(page_paths):
        r = idx // cols
        c = idx % cols
        y = header_h + r * tile_h
        x = c * thumb_w
        sheet[y : y + tile_h, x : x + thumb_w] = _draw_tile(path, thumb_w, thumb_h, label_h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


def build_route_sheet(task: tuple[str, str, str, str, int, int, int, int, int, int, bool]) -> dict:
    (
        scenario,
        route_id,
        route_dir_str,
        output_dir_str,
        cols,
        thumb_w,
        thumb_h,
        label_h,
        max_frames_per_page,
        jpeg_quality,
        overwrite,
    ) = task
    route_dir = pathlib.Path(route_dir_str)
    output_dir = pathlib.Path(output_dir_str)
    rgb_dir = route_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("*.jpg"), key=_frame_id)
    out_dir = _route_out_dir(output_dir, scenario, route_id)
    is_bad, info = is_abnormal_lead_route(route_dir, scenario)
    if not frame_paths:
        return asdict(
            RouteSheetResult(
                scenario=scenario,
                route_id=route_id,
                frame_count=0,
                page_count=0,
                abnormal_duration=is_bad,
                abnormal_reason=str(info.get("reason", "")),
                sheet_dir=str(out_dir),
                first_sheet="",
                status="empty",
                message="no rgb jpg",
            )
        )
    page_count = int(np.ceil(len(frame_paths) / max_frames_per_page))
    first_sheet = out_dir / "page_000.jpg"
    if first_sheet.exists() and not overwrite:
        return asdict(
            RouteSheetResult(
                scenario=scenario,
                route_id=route_id,
                frame_count=len(frame_paths),
                page_count=page_count,
                abnormal_duration=is_bad,
                abnormal_reason=str(info.get("reason", "")),
                sheet_dir=str(out_dir),
                first_sheet=str(first_sheet),
                status="exists",
            )
        )
    for page_idx in range(page_count):
        start = page_idx * max_frames_per_page
        end = min(len(frame_paths), start + max_frames_per_page)
        _write_page(
            frame_paths[start:end],
            out_dir / f"page_{page_idx:03d}.jpg",
            scenario=scenario,
            route_id=route_id,
            page_idx=page_idx,
            page_count=page_count,
            total_frames=len(frame_paths),
            abnormal=is_bad,
            cols=cols,
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            label_h=label_h,
            jpeg_quality=jpeg_quality,
        )
    return asdict(
        RouteSheetResult(
            scenario=scenario,
            route_id=route_id,
            frame_count=len(frame_paths),
            page_count=page_count,
            abnormal_duration=is_bad,
            abnormal_reason=str(info.get("reason", "")),
            sheet_dir=str(out_dir),
            first_sheet=str(first_sheet),
            status="written",
        )
    )


def _write_indexes(output_dir: pathlib.Path, results: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "route_sheet_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = [
        "scenario",
        "route_id",
        "frame_count",
        "page_count",
        "abnormal_duration",
        "abnormal_reason",
        "annotation_status",
        "annotation_num_frames",
        "primary_rs_distribution",
        "primary_event_distribution",
        "has_r2",
        "r2_frames",
        "r2_ratio",
        "review_required_frames",
        "review_required_ratio",
        "status",
        "first_sheet",
        "sheet_dir",
        "message",
    ]
    with (output_dir / "route_sheet_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})

    by_scenario: dict[str, list[dict]] = {}
    for row in results:
        by_scenario.setdefault(row["scenario"], []).append(row)
    summary = []
    for scenario, rows in sorted(by_scenario.items()):
        summary.append(
            {
                "scenario": scenario,
                "routes": len(rows),
                "frames": sum(int(r.get("frame_count", 0) or 0) for r in rows),
                "pages": sum(int(r.get("page_count", 0) or 0) for r in rows),
                "abnormal_routes": sum(1 for r in rows if r.get("abnormal_duration")),
                "r2_routes": sum(1 for r in rows if r.get("has_r2")),
                "r2_frames": sum(int(r.get("r2_frames", 0) or 0) for r in rows),
                "first_sheet": next((r.get("first_sheet") for r in rows if r.get("first_sheet")), ""),
            }
        )
    (output_dir / "scenario_sheet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "scenario_sheet_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields2 = ["scenario", "routes", "frames", "pages", "abnormal_routes", "r2_routes", "r2_frames", "first_sheet"]
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(summary)


def _progress(done: int, total: int, start_time: float, last: str = "") -> None:
    elapsed = max(1e-6, time.time() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"[rgb-sheets] {done}/{total} ({100.0 * done / max(total, 1):5.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s {last}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    scenarios = _parse_csv(args.scenarios)
    routes = _iter_routes(data_root, scenarios)
    if args.max_routes_per_scenario > 0:
        kept = []
        counts: dict[str, int] = {}
        for scenario, route_id, route_dir in routes:
            if counts.get(scenario, 0) >= args.max_routes_per_scenario:
                continue
            kept.append((scenario, route_id, route_dir))
            counts[scenario] = counts.get(scenario, 0) + 1
        routes = kept
    tasks = [
        (
            scenario,
            route_id,
            str(route_dir),
            str(output_dir),
            args.cols,
            args.thumb_w,
            args.thumb_h,
            args.label_h,
            args.max_frames_per_page,
            args.jpeg_quality,
            args.overwrite,
        )
        for scenario, route_id, route_dir in routes
    ]
    print(f"[rgb-sheets] discovered routes={len(tasks)} output={output_dir}", flush=True)
    results: list[dict] = []
    start = time.time()
    _progress(0, len(tasks), start, "starting")
    if args.workers <= 1:
        for idx, task in enumerate(tasks, 1):
            results.append(build_route_sheet(task))
            if idx == len(tasks) or idx % max(1, args.progress_interval) == 0:
                _progress(idx, len(tasks), start, f"last={task[0]}/{task[1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_task = {pool.submit(build_route_sheet, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                results.append(future.result())
                if idx == len(tasks) or idx % max(1, args.progress_interval) == 0:
                    _progress(idx, len(tasks), start, f"last={task[0]}/{task[1]}")
    results.sort(key=lambda x: (x["scenario"], x["route_id"]))
    annotation_index = _load_annotation_index(pathlib.Path(args.annotation_result_dir) if args.annotation_result_dir else None)
    for row in results:
        ann = annotation_index.get((row["scenario"], row["route_id"]), {})
        for key, value in ann.items():
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
    _write_indexes(output_dir, results)
    print(f"[rgb-sheets] wrote manifest={output_dir / 'route_sheet_manifest.csv'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "AutoMoT" / "lead_data"))
    parser.add_argument("--output-dir", default="/tmp/automot_rgb_route_sheets_full")
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--thumb-w", type=int, default=288)
    parser.add_argument("--thumb-h", type=int, default=96)
    parser.add_argument("--label-h", type=int, default=22)
    parser.add_argument("--max-frames-per-page", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--max-routes-per-scenario", type=int, default=0)
    parser.add_argument(
        "--annotation-result-dir",
        default=None,
        help="Optional directory containing *_result.json files; route RS/event distributions are merged into the manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
