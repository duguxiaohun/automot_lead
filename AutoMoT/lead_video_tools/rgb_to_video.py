"""Convert LEAD stitched RGB frame folders into seekable MP4 videos.

The expected input layout is:

    /datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/0000.jpg

The output layout mirrors the scenario/run_id hierarchy:

    /data/lead_video/<Scenario>/<run_id>/input.mp4

Only RGB videos are generated. The online eval_carla debug/demo/grid videos need
CARLA runtime state, predicted trajectories, or live camera actors, so raw LEAD
offline data cannot fully reproduce them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


DEFAULT_DATA_ROOT = pathlib.Path("/datashare/IOL4SGH/data/data")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/data/lead_video")
DEFAULT_FPS = 4.0  # LEAD logs one frame every 5 CARLA ticks at 20Hz => 0.25s/frame.
VIDEO_NAME = "input.mp4"
SUPPORTED_VIEWS = ("input", "left", "front", "right")
VIEW_TO_FILE = {
    "input": "input.mp4",
    "left": "left.mp4",
    "front": "front.mp4",
    "right": "right.mp4",
}
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@dataclass(frozen=True)
class RouteTask:
    """One LEAD route RGB folder to convert."""

    scenario: str
    run_id: str
    route_dir: str
    rgb_dir: str
    output_dir: str
    video_path: str
    frame_count: int


@dataclass
class ConvertResult:
    """Conversion status for one route."""

    scenario: str
    run_id: str
    video_path: str
    status: str
    frame_count: int
    message: str = ""
    elapsed_s: float = 0.0
    outputs: list[str] | None = None


def _run_command(cmd: Sequence[str], timeout_s: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found in PATH")
    return path


def _find_font_file() -> str | None:
    for candidate in FONT_CANDIDATES:
        path = pathlib.Path(candidate)
        if path.exists():
            return str(path)
    return None


def _escape_drawtext_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _natural_jpgs(rgb_dir: pathlib.Path) -> list[pathlib.Path]:
    def _key(path: pathlib.Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return sys.maxsize, path.name

    return sorted(rgb_dir.glob("*.jpg"), key=_key)


def _parse_frame_stems(rgb_files: Sequence[pathlib.Path]) -> list[int] | None:
    try:
        return [int(p.stem) for p in rgb_files]
    except ValueError:
        return None


def _ffprobe_image(path: pathlib.Path) -> tuple[int, int] | None:
    """Probe one image's width/height without decoding it in Python."""

    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(path),
    ]
    proc = _run_command(cmd)
    if proc.returncode != 0:
        return None
    try:
        streams = json.loads(proc.stdout).get("streams") or []
        if not streams:
            return None
        return int(streams[0]["width"]), int(streams[0]["height"])
    except Exception:
        return None


def validate_rgb_sequence(
    rgb_dir: pathlib.Path,
    *,
    min_frames: int,
    allow_noncontiguous: bool,
) -> tuple[bool, str, list[pathlib.Path], tuple[int, int] | None]:
    """Reject obviously broken LEAD RGB routes before spending ffmpeg time."""

    rgb_files = _natural_jpgs(rgb_dir)
    if len(rgb_files) < min_frames:
        return False, f"too_few_frames:{len(rgb_files)}<{min_frames}", rgb_files, None

    stems = _parse_frame_stems(rgb_files)
    if stems is None:
        return False, "non_numeric_frame_name", rgb_files, None
    expected = list(range(len(rgb_files)))
    if stems != expected and not allow_noncontiguous:
        missing = sorted(set(range(stems[0], stems[-1] + 1)) - set(stems))[:8]
        return False, f"noncontiguous_frames:first_missing={missing}", rgb_files, None

    first_meta = _ffprobe_image(rgb_files[0])
    last_meta = _ffprobe_image(rgb_files[-1])
    if first_meta is None:
        return False, f"unreadable_first_frame:{rgb_files[0].name}", rgb_files, None
    if last_meta is None:
        return False, f"unreadable_last_frame:{rgb_files[-1].name}", rgb_files, first_meta
    if first_meta != last_meta:
        return False, f"dimension_mismatch:first={first_meta},last={last_meta}", rgb_files, first_meta
    if first_meta[0] < 3 or first_meta[1] <= 0:
        return False, f"invalid_dimensions:{first_meta}", rgb_files, first_meta
    if first_meta[0] % 3 != 0:
        return False, f"stitched_width_not_divisible_by_3:{first_meta[0]}", rgb_files, first_meta
    return True, "ok", rgb_files, first_meta


def discover_routes(
    data_root: pathlib.Path,
    scenarios: set[str] | None = None,
    run_ids: set[str] | None = None,
) -> list[tuple[str, str, pathlib.Path, pathlib.Path, int]]:
    """Find route folders with a non-empty rgb/*.jpg sequence."""

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    routes: list[tuple[str, str, pathlib.Path, pathlib.Path, int]] = []
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            run_id = route_dir.name
            if run_ids and run_id not in run_ids:
                continue
            rgb_dir = route_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            frame_count = len(_natural_jpgs(rgb_dir))
            if frame_count <= 0:
                continue
            routes.append((scenario, run_id, route_dir, rgb_dir, frame_count))
    return routes


def build_tasks(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    scenarios: set[str] | None = None,
    run_ids: set[str] | None = None,
) -> list[RouteTask]:
    tasks: list[RouteTask] = []
    for scenario, run_id, route_dir, rgb_dir, frame_count in discover_routes(data_root, scenarios, run_ids):
        output_dir = output_root / scenario / run_id
        tasks.append(
            RouteTask(
                scenario=scenario,
                run_id=run_id,
                route_dir=str(route_dir),
                rgb_dir=str(rgb_dir),
                output_dir=str(output_dir),
                video_path=str(output_dir / VIDEO_NAME),
                frame_count=frame_count,
            )
        )
    return tasks


def probe_video(video_path: pathlib.Path) -> dict:
    """Return ffprobe metadata. Missing ffprobe is treated as no metadata."""

    if not video_path.exists() or shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,duration,width,height",
        "-of", "json",
        str(video_path),
    ]
    proc = _run_command(cmd)
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    streams = data.get("streams") or []
    if not streams:
        return {}
    return streams[0]


def _read_manifest(output_dir: pathlib.Path) -> dict:
    manifest_path = output_dir / "video_meta.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_video_complete(
    video_path: pathlib.Path,
    expected_frames: int,
    fps: float,
    *,
    require_manifest: bool = True,
    expected_frame_index: bool = True,
) -> tuple[bool, str]:
    """Check whether an existing video is good enough for resume skip."""

    if not video_path.exists() or video_path.stat().st_size <= 0:
        return False, "missing_or_empty"
    if require_manifest:
        manifest = _read_manifest(video_path.parent)
        view = video_path.stem
        if not manifest:
            return False, "missing_manifest"
        if bool(manifest.get("draw_frame_index")) != bool(expected_frame_index):
            return False, "manifest_frame_index_mismatch"
        if view not in set(manifest.get("views", [])):
            return False, f"manifest_missing_view:{view}"
    meta = probe_video(video_path)
    if not meta:
        return False, "probe_failed"
    nb_frames = meta.get("nb_frames")
    if nb_frames not in (None, "N/A"):
        try:
            actual = int(nb_frames)
            if abs(actual - expected_frames) <= 1:
                return True, f"frames_ok:{actual}"
            return False, f"frame_mismatch:{actual}!={expected_frames}"
        except ValueError:
            pass
    duration = meta.get("duration")
    if duration not in (None, "N/A"):
        try:
            actual_frames = round(float(duration) * fps)
            if abs(actual_frames - expected_frames) <= 2:
                return True, f"duration_ok:{duration}"
            return False, f"duration_mismatch:{actual_frames}!={expected_frames}"
        except ValueError:
            pass
    return True, "exists_unverified"


def _drawtext_filter(view: str) -> str:
    # n is zero-based and matches LEAD's 0000.jpg frame id for normal contiguous routes.
    text = f"{view} frame %{{n}}"
    font_file = _find_font_file()
    parts = ["drawtext="]
    if font_file:
        parts.append(f"fontfile='{_escape_drawtext_path(font_file)}':")
    parts.append(
        f"text='{text}':"
        "x=12:y=12:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"
    )
    return "".join(parts)


def _view_filter(view: str, draw_frame_index: bool) -> str | None:
    filters: list[str] = []
    if view == "left":
        filters.append("crop=iw/3:ih:0:0")
    elif view == "front":
        filters.append("crop=iw/3:ih:iw/3:0")
    elif view == "right":
        filters.append("crop=iw/3:ih:2*iw/3:0")
    elif view != "input":
        raise ValueError(f"unsupported view: {view}")
    if draw_frame_index:
        filters.append(_drawtext_filter(view))
    return ",".join(filters) if filters else None


def _write_concat_list(rgb_files: Sequence[pathlib.Path], list_path: pathlib.Path, frame_duration: float) -> None:
    """Write an ffmpeg concat-demuxer list for non-contiguous frame names."""

    with list_path.open("w", encoding="utf-8", newline="\n") as f:
        for path in rgb_files:
            safe_path = str(path.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
            f.write(f"duration {frame_duration:.9f}\n")
        # ffmpeg concat demuxer needs the last file repeated to preserve duration.
        if rgb_files:
            safe_last = str(rgb_files[-1].resolve()).replace("'", "'\\''")
            f.write(f"file '{safe_last}'\n")


def _encode_one_view(
    *,
    ffmpeg: str,
    rgb_dir: pathlib.Path,
    rgb_files: Sequence[pathlib.Path],
    output_dir: pathlib.Path,
    view: str,
    fps: float,
    crf: int,
    draw_frame_index: bool,
    allow_noncontiguous: bool,
) -> tuple[bool, str, pathlib.Path]:
    video_path = output_dir / VIEW_TO_FILE[view]
    tmp_video = output_dir / f".{VIEW_TO_FILE[view]}.tmp.mp4"
    list_path: pathlib.Path | None = None
    frame_duration = 1.0 / float(fps)
    view_filter = _view_filter(view, draw_frame_index)

    stems = _parse_frame_stems(rgb_files)
    contiguous = stems == list(range(len(rgb_files)))
    if contiguous:
        input_pattern = str(rgb_dir / "%04d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-framerate", f"{fps:.6f}",
            "-start_number", "0",
            "-i", input_pattern,
            "-frames:v", str(len(rgb_files)),
        ]
    elif allow_noncontiguous:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffconcat",
            prefix="lead_rgb_",
            dir=str(output_dir),
            delete=False,
        ) as tmp:
            list_path = pathlib.Path(tmp.name)
        _write_concat_list(rgb_files, list_path, frame_duration)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-frames:v", str(len(rgb_files)),
        ]
    else:
        return False, "noncontiguous_frames", video_path

    if view_filter:
        cmd.extend(["-vf", view_filter])
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(tmp_video),
    ])
    try:
        proc = _run_command(cmd)
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            return False, msg, video_path
        os.replace(tmp_video, video_path)
        ok, reason = is_video_complete(
            video_path,
            len(rgb_files),
            fps,
            require_manifest=False,
            expected_frame_index=draw_frame_index,
        )
        return ok, reason if ok else f"postcheck_failed:{reason}", video_path
    finally:
        if list_path is not None:
            try:
                list_path.unlink()
            except FileNotFoundError:
                pass
        try:
            if tmp_video.exists():
                tmp_video.unlink()
        except FileNotFoundError:
            pass


def convert_route(
    task: RouteTask,
    fps: float,
    overwrite: bool = False,
    crf: int = 18,
    views: Sequence[str] = ("input",),
    draw_frame_index: bool = True,
    min_frames: int = 2,
    allow_noncontiguous: bool = False,
) -> ConvertResult:
    """Convert one route's stitched RGB frames to input.mp4."""

    start = time.time()
    output_dir = pathlib.Path(task.output_dir)
    video_paths = [output_dir / VIEW_TO_FILE[v] for v in views]
    complete_reasons = [
        is_video_complete(p, task.frame_count, fps, expected_frame_index=draw_frame_index)
        for p in video_paths
    ]
    if all(ok for ok, _reason in complete_reasons) and not overwrite:
        return ConvertResult(
            task.scenario,
            task.run_id,
            str(video_paths[0]),
            "skipped",
            task.frame_count,
            ";".join(reason for _ok, reason in complete_reasons),
            time.time() - start,
            [str(p) for p in video_paths],
        )

    ffmpeg = _require_tool("ffmpeg")
    rgb_dir = pathlib.Path(task.rgb_dir)
    ok, reason, rgb_files, dims = validate_rgb_sequence(
        rgb_dir,
        min_frames=min_frames,
        allow_noncontiguous=allow_noncontiguous,
    )
    if not ok:
        return ConvertResult(
            task.scenario,
            task.run_id,
            str(video_paths[0]),
            "excluded",
            len(rgb_files),
            reason,
            time.time() - start,
            [str(p) for p in video_paths],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_duration = 1.0 / float(fps)
    outputs: list[str] = []
    messages: list[str] = []
    converted_any = False
    for view in views:
        view_path = output_dir / VIEW_TO_FILE[view]
        if not overwrite:
            complete, skip_reason = is_video_complete(
                view_path,
                len(rgb_files),
                fps,
                expected_frame_index=draw_frame_index,
            )
            if complete:
                outputs.append(str(view_path))
                messages.append(f"{view}:skipped:{skip_reason}")
                continue
        view_ok, view_msg, view_path = _encode_one_view(
            ffmpeg=ffmpeg,
            rgb_dir=rgb_dir,
            rgb_files=rgb_files,
            output_dir=output_dir,
            view=view,
            fps=fps,
            crf=crf,
            draw_frame_index=draw_frame_index,
            allow_noncontiguous=allow_noncontiguous,
        )
        if not view_ok:
            return ConvertResult(
                task.scenario,
                task.run_id,
                str(view_path),
                "failed",
                len(rgb_files),
                f"{view}:{view_msg}",
                time.time() - start,
                outputs,
            )
        converted_any = True
        outputs.append(str(view_path))
        messages.append(f"{view}:{view_msg}")

    status = "converted" if converted_any else "skipped"
    manifest = {
        "scenario": task.scenario,
        "run_id": task.run_id,
        "source_rgb_dir": str(rgb_dir),
        "output_dir": str(output_dir),
        "views": list(views),
        "video_paths": outputs,
        "frame_count": len(rgb_files),
        "fps": fps,
        "seconds_per_frame": frame_duration,
        "draw_frame_index": draw_frame_index,
        "frame_index_base": 0,
        "image_width": dims[0] if dims else None,
        "image_height": dims[1] if dims else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": "ffmpeg libx264 yuv420p faststart",
        "supported_offline_views": list(SUPPORTED_VIEWS),
        "unsupported_eval_carla_views": {
            "debug.mp4": "requires model predictions and camera projection",
            "bev_debug.mp4": "requires model predictions plus LiDAR/tp/ntp overlay",
            "demo.mp4": "requires live CARLA camera actors",
            "grid.mp4": "requires demo camera frames",
        },
    }
    (output_dir / "video_meta.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ConvertResult(
        task.scenario,
        task.run_id,
        outputs[0] if outputs else str(video_paths[0]),
        status,
        len(rgb_files),
        ";".join(messages),
        time.time() - start,
        outputs,
    )


def _parse_csv(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if part:
                out.add(part)
    return out or None


def _parse_views(value: str) -> tuple[str, ...]:
    views = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    if not views:
        raise ValueError("--views must contain at least one view")
    bad = [v for v in views if v not in SUPPORTED_VIEWS]
    if bad:
        raise ValueError(f"unsupported view(s): {bad}; supported={SUPPORTED_VIEWS}")
    return views


def _write_summary(output_root: pathlib.Path, results: Iterable[ConvertResult]) -> None:
    rows = [asdict(r) for r in results]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "lead_video_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="LEAD RGB playback fps; default 4.0")
    parser.add_argument("--scenario", action="append", help="Only convert selected scenario(s); comma is supported")
    parser.add_argument("--run-id", action="append", help="Only convert selected run id(s); comma is supported")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel ffmpeg workers; 0 = auto from CPU count, keep small on shared storage")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N routes after filtering")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if input.mp4 already passes checks")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned tasks")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF; lower is higher quality/larger file")
    parser.add_argument("--views", type=str, default="input",
                        help="Comma/space separated views: input,left,front,right. Default: input")
    parser.add_argument("--no-frame-index", action="store_true",
                        help="Do not draw '<view> frame N' at the top-left corner")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="Exclude routes with fewer than this many RGB frames")
    parser.add_argument("--allow-noncontiguous", action="store_true",
                        help="Do not exclude routes with missing frame numbers; use concat fallback")
    args = parser.parse_args(argv)

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    scenarios = _parse_csv(args.scenario)
    run_ids = _parse_csv(args.run_id)
    views = _parse_views(args.views)
    tasks = build_tasks(args.data_root, args.output_root, scenarios, run_ids)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if args.workers == 0:
        cpu_count = os.cpu_count() or 1
        resolved_workers = min(max(1, cpu_count // 2), 8, max(1, len(tasks)))
    else:
        resolved_workers = max(1, int(args.workers))

    print(f"[discover] data_root={args.data_root} output_root={args.output_root}")
    print(
        f"[discover] routes={len(tasks)} fps={args.fps} workers={resolved_workers} "
        f"views={','.join(views)} frame_index={not args.no_frame_index}"
    )
    if args.dry_run:
        for task in tasks:
            planned = [str(pathlib.Path(task.output_dir) / VIEW_TO_FILE[v]) for v in views]
            print(f"[plan] {task.scenario}/{task.run_id}: {task.frame_count} frames -> {planned}")
        return 0
    if not tasks:
        print("[done] no routes found")
        return 0

    _require_tool("ffmpeg")
    _require_tool("ffprobe")
    results: list[ConvertResult] = []
    workers = resolved_workers
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            result = convert_route(
                task,
                args.fps,
                args.overwrite,
                args.crf,
                views,
                not args.no_frame_index,
                args.min_frames,
                args.allow_noncontiguous,
            )
            results.append(result)
            print(
                f"[{idx}/{len(tasks)}] {result.status} {result.scenario}/{result.run_id} "
                f"frames={result.frame_count} dt={result.elapsed_s:.1f}s {result.message}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    convert_route,
                    task,
                    args.fps,
                    args.overwrite,
                    args.crf,
                    views,
                    not args.no_frame_index,
                    args.min_frames,
                    args.allow_noncontiguous,
                )
                for task in tasks
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                print(
                    f"[{idx}/{len(tasks)}] {result.status} {result.scenario}/{result.run_id} "
                    f"frames={result.frame_count} dt={result.elapsed_s:.1f}s {result.message}",
                    flush=True,
                )

    _write_summary(args.output_root, results)
    converted = sum(1 for r in results if r.status == "converted")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    excluded = sum(1 for r in results if r.status == "excluded")
    print(f"[summary] converted={converted} skipped={skipped} excluded={excluded} failed={failed}")
    print(f"[summary] wrote {args.output_root / 'lead_video_summary.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
