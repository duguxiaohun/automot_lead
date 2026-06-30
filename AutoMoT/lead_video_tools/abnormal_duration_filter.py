"""按 LEAD RGB 帧数筛出疑似采集异常的 route。

这个脚本只做轻量目录扫描：统计 `<Scenario>/<run_id>/rgb/*.jpg` 数量，
不调用 ffprobe、不检查旧视频、不启动 ffmpeg。筛选结果可以反复给
`rgb_to_video.py --abnormal-route-list-dir ...` 复用，避免每次转视频都重新扫全库。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Sequence


DEFAULT_DATA_ROOT = pathlib.Path("/datashare/IOL4SGH/data/data")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/data/lead_video")
DEFAULT_FPS = 4.0
DEFAULT_POSSIBLE_ABNORMAL_MIN_SECONDS = 90.0
DEFAULT_CONFIRMED_ABNORMAL_MIN_SECONDS = 120.0
DEFAULT_FILTER_DIR = pathlib.Path(__file__).resolve().parent / "abnormal_duration_filter"
ABNORMAL_POSSIBLE_FILE = "abnormal_possible_90s_to_120s.txt"
ABNORMAL_CONFIRMED_FILE = "abnormal_confirmed_over_120s.txt"
ABNORMAL_SUMMARY_FILE = "abnormal_duration_summary.json"


@dataclass
class AbnormalDurationItem:
    """按采集时长筛出的疑似异常 route。"""

    severity: str
    scenario: str
    run_id: str
    frame_count: int
    duration_s: float
    source_rgb_dir: str
    video_dir: str
    plan_status: str = "frame_count_scan"
    plan_message: str = "counted_rgb_jpg_only"


def _parse_csv(values: Sequence[str] | None) -> set[str] | None:
    """解析可重复传入、也可逗号/空格分隔的 CLI 参数。"""

    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if part:
                out.add(part)
    return out or None


def duration_threshold_frames(fps: float, seconds: float) -> int:
    """把秒级阈值换算成帧数，向上取整避免低估时长。"""

    return int(seconds * fps + 0.999999)


def classify_abnormal_duration(
    frame_count: int,
    *,
    fps: float,
    possible_min_seconds: float = DEFAULT_POSSIBLE_ABNORMAL_MIN_SECONDS,
    confirmed_min_seconds: float = DEFAULT_CONFIRMED_ABNORMAL_MIN_SECONDS,
) -> str | None:
    """按视频时长把 route 分成 possible / confirmed 两类异常候选。"""

    possible_min_frames = duration_threshold_frames(fps, possible_min_seconds)
    confirmed_min_frames = duration_threshold_frames(fps, confirmed_min_seconds)
    if frame_count >= confirmed_min_frames:
        return "confirmed"
    if frame_count >= possible_min_frames:
        return "possible"
    return None


def _count_jpgs(rgb_dir: pathlib.Path) -> int:
    """快速统计 jpg 数量；不排序、不读取图片内容。"""

    try:
        return sum(1 for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    except OSError:
        return 0


def scan_abnormal_durations(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    fps: float,
    scenarios: set[str] | None = None,
    run_ids: set[str] | None = None,
    possible_min_seconds: float = DEFAULT_POSSIBLE_ABNORMAL_MIN_SECONDS,
    confirmed_min_seconds: float = DEFAULT_CONFIRMED_ABNORMAL_MIN_SECONDS,
    progress_interval: int = 10,
) -> list[AbnormalDurationItem]:
    """扫描数据根目录，返回达到异常时长阈值的 route。"""

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    items: list[AbnormalDurationItem] = []
    start = time.time()
    scenario_count = 0
    route_count = 0
    print(f"[filter] scanning data_root={data_root}", flush=True)
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        scenario_count += 1
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            run_id = route_dir.name
            if run_ids and run_id not in run_ids:
                continue
            rgb_dir = route_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            frame_count = _count_jpgs(rgb_dir)
            if frame_count <= 0:
                continue
            route_count += 1
            severity = classify_abnormal_duration(
                frame_count,
                fps=fps,
                possible_min_seconds=possible_min_seconds,
                confirmed_min_seconds=confirmed_min_seconds,
            )
            if severity is None:
                continue
            video_dir = output_root / scenario / run_id
            items.append(
                AbnormalDurationItem(
                    severity=severity,
                    scenario=scenario,
                    run_id=run_id,
                    frame_count=frame_count,
                    duration_s=frame_count / fps,
                    source_rgb_dir=str(rgb_dir),
                    video_dir=str(video_dir),
                )
            )
        if scenario_count == 1 or scenario_count % max(1, progress_interval) == 0:
            elapsed = time.time() - start
            print(
                f"[filter] scenarios={scenario_count} routes={route_count} "
                f"candidates={len(items)} last={scenario} elapsed={elapsed:.1f}s",
                flush=True,
            )
    elapsed = time.time() - start
    print(
        f"[filter] done scenarios={scenario_count} routes={route_count} "
        f"candidates={len(items)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return sorted(items, key=lambda x: (x.severity, x.scenario, x.run_id))


def _format_abnormal_duration_line(item: AbnormalDurationItem) -> str:
    """生成人工巡检友好的异常名单行。"""

    return (
        f"{item.scenario}/{item.run_id}\t"
        f"frames={item.frame_count}\t"
        f"duration_s={item.duration_s:.2f}\t"
        f"rgb={item.source_rgb_dir}\t"
        f"video_dir={item.video_dir}"
    )


def write_abnormal_duration_lists(
    output_dir: pathlib.Path,
    items: Sequence[AbnormalDurationItem],
    *,
    fps: float,
    possible_min_seconds: float = DEFAULT_POSSIBLE_ABNORMAL_MIN_SECONDS,
    confirmed_min_seconds: float = DEFAULT_CONFIRMED_ABNORMAL_MIN_SECONDS,
) -> dict[str, int | str | float]:
    """把异常名单写进一个可复用筛选目录。"""

    possible = [item for item in items if item.severity == "possible"]
    confirmed = [item for item in items if item.severity == "confirmed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    possible_path = output_dir / ABNORMAL_POSSIBLE_FILE
    confirmed_path = output_dir / ABNORMAL_CONFIRMED_FILE
    summary_path = output_dir / ABNORMAL_SUMMARY_FILE
    header = (
        "# LEAD abnormal duration candidates\n"
        f"# fps={fps:.6f} possible=[{possible_min_seconds:.1f}s,{confirmed_min_seconds:.1f}s) "
        f"confirmed=>={confirmed_min_seconds:.1f}s\n"
        "# format: Scenario/run_id\\tframes=N\\tduration_s=S\\trgb=...\\tvideo_dir=...\n"
    )
    possible_path.write_text(
        header + "".join(_format_abnormal_duration_line(item) + "\n" for item in possible),
        encoding="utf-8",
    )
    confirmed_path.write_text(
        header + "".join(_format_abnormal_duration_line(item) + "\n" for item in confirmed),
        encoding="utf-8",
    )
    summary = {
        "fps": fps,
        "possible_min_seconds": possible_min_seconds,
        "confirmed_min_seconds": confirmed_min_seconds,
        "possible_min_frames": duration_threshold_frames(fps, possible_min_seconds),
        "confirmed_min_frames": duration_threshold_frames(fps, confirmed_min_seconds),
        "possible_count": len(possible),
        "confirmed_count": len(confirmed),
        "possible_file": str(possible_path),
        "confirmed_file": str(confirmed_path),
        "items": [asdict(item) for item in possible + confirmed],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "possible_count": len(possible),
        "confirmed_count": len(confirmed),
        "possible_file": str(possible_path),
        "confirmed_file": str(confirmed_path),
        "summary_file": str(summary_path),
        "possible_min_frames": summary["possible_min_frames"],
        "confirmed_min_frames": summary["confirmed_min_frames"],
    }


def _parse_route_line(line: str) -> tuple[str, str] | None:
    """从 txt 名单的一行解析 `(scenario, run_id)`。"""

    line = line.strip()
    if not line or line.startswith("#"):
        return None
    route_key = line.split("\t", 1)[0]
    if "/" not in route_key:
        return None
    scenario, run_id = route_key.split("/", 1)
    if not scenario or not run_id:
        return None
    return scenario, run_id


def load_abnormal_route_keys(list_dir: pathlib.Path, *, kind: str = "all") -> set[tuple[str, str]]:
    """从筛选目录读取 route key 集合，供 rgb_to_video 复用。"""

    if kind not in {"possible", "confirmed", "all"}:
        raise ValueError("--abnormal-route-kind must be one of possible, confirmed, all")
    summary_path = list_dir / ABNORMAL_SUMMARY_FILE
    keys: set[tuple[str, str]] = set()
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            severity = item.get("severity")
            if kind != "all" and severity != kind:
                continue
            scenario = str(item.get("scenario") or "")
            run_id = str(item.get("run_id") or "")
            if scenario and run_id:
                keys.add((scenario, run_id))
        return keys

    files: list[str] = []
    if kind in {"possible", "all"}:
        files.append(ABNORMAL_POSSIBLE_FILE)
    if kind in {"confirmed", "all"}:
        files.append(ABNORMAL_CONFIRMED_FILE)
    for filename in files:
        path = list_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key = _parse_route_line(line)
            if key is not None:
                keys.add(key)
    return keys


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--filter-dir", type=pathlib.Path, default=DEFAULT_FILTER_DIR)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--scenario", action="append", help="Only scan selected scenario(s); comma is supported")
    parser.add_argument("--run-id", action="append", help="Only scan selected run id(s); comma is supported")
    parser.add_argument("--possible-min-seconds", type=float, default=DEFAULT_POSSIBLE_ABNORMAL_MIN_SECONDS)
    parser.add_argument("--confirmed-min-seconds", type=float, default=DEFAULT_CONFIRMED_ABNORMAL_MIN_SECONDS)
    args = parser.parse_args(argv)

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    scenarios = _parse_csv(args.scenario)
    run_ids = _parse_csv(args.run_id)
    items = scan_abnormal_durations(
        args.data_root,
        args.output_root,
        fps=args.fps,
        scenarios=scenarios,
        run_ids=run_ids,
        possible_min_seconds=args.possible_min_seconds,
        confirmed_min_seconds=args.confirmed_min_seconds,
    )
    info = write_abnormal_duration_lists(
        args.filter_dir,
        items,
        fps=args.fps,
        possible_min_seconds=args.possible_min_seconds,
        confirmed_min_seconds=args.confirmed_min_seconds,
    )
    print(
        "[filter] "
        f"possible={info['possible_count']} (>={info['possible_min_frames']} frames) "
        f"confirmed={info['confirmed_count']} (>={info['confirmed_min_frames']} frames)"
    )
    print(f"[filter] wrote {info['possible_file']}")
    print(f"[filter] wrote {info['confirmed_file']}")
    print(f"[filter] wrote {info['summary_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
