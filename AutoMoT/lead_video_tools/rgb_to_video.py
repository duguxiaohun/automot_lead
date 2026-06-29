"""把 LEAD 离线 stitched RGB 帧批量转换成可拖动的 MP4 视频。

输入目录遵循 LEAD 数据集结构：

    /datashare/IOL4SGH/data/data/<Scenario>/<run_id>/rgb/0000.jpg

输出目录镜像 scenario/run_id 层级：

    /data/lead_video/<Scenario>/<run_id>/input.mp4

本脚本只生成 raw RGB 可以可靠支持的视频：input 以及从 stitched RGB 裁出来的
left/front/right。eval_carla 里的 debug/bev_debug/demo/grid 依赖在线 CARLA 状态、
模型预测轨迹或 live camera actor，不能仅靠离线 RGB 完整复现。
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
# LEAD 采集时 CARLA 是 20Hz，每 5 tick 落盘 1 帧，因此离线 RGB 是 4Hz。
DEFAULT_FPS = 4.0
DEFAULT_PRESET = "veryfast"
DEFAULT_AUTO_WORKER_CAP = 16
DEFAULT_FFMPEG_THREADS = 1
DEFAULT_DISCOVER_PROGRESS_INTERVAL = 10
VIDEO_NAME = "input.mp4"

# 当前离线 raw RGB 能可靠支持的视角：
# - input: 原始 1152x384 三视角横拼图；
# - left/front/right: 按宽度三等分裁出的 384x384 单视角。
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
    """一条 LEAD route 的转换任务。

    这里刻意只保存字符串路径，方便 ProcessPoolExecutor 在多进程间 pickle。
    """

    scenario: str
    run_id: str
    route_dir: str
    rgb_dir: str
    output_dir: str
    video_path: str
    frame_count: int


@dataclass
class ConvertResult:
    """一条 route 转换后的结果，用于终端汇总和 summary JSON。"""

    scenario: str
    run_id: str
    video_path: str
    status: str
    frame_count: int
    message: str = ""
    elapsed_s: float = 0.0
    outputs: list[str] | None = None


@dataclass
class PlanItem:
    """正式编码前的预检查结果。

    status 只取三类：
    - already_done: 断点续跑检查通过，后续不再开 ffmpeg；
    - excluded: RGB 序列异常，本次剔除；
    - to_run: 需要真正编码。
    """

    task: RouteTask
    status: str
    message: str
    frame_count: int
    outputs: list[str]
    dims: tuple[int, int] | None = None


@dataclass
class PlanSummary:
    """正式转换前打印给用户看的计划统计。"""

    total: int = 0
    already_done: int = 0
    excluded: int = 0
    to_run: int = 0


def _run_command(cmd: Sequence[str], timeout_s: int | None = None) -> subprocess.CompletedProcess:
    """运行外部命令并捕获 stdout/stderr。

    ffmpeg/ffprobe 失败时我们要把 stderr 写进 ConvertResult，而不是让异常直接
    打断全量任务，所以这里统一 check=False。
    """

    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _require_tool(name: str) -> str:
    """确认外部工具在 PATH 中，并返回实际路径。"""

    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found in PATH")
    return path


def _find_font_file() -> str | None:
    """寻找 drawtext 可用字体文件。

    Linux 远端通常有 DejaVu；Windows 本地烟雾测试用 Arial。显式传 fontfile
    可以避免 ffmpeg drawtext 因 fontconfig 默认配置缺失而失败。
    """

    for candidate in FONT_CANDIDATES:
        path = pathlib.Path(candidate)
        if path.exists():
            return str(path)
    return None


def _escape_drawtext_path(path: str) -> str:
    """转义 ffmpeg drawtext 的 fontfile 路径。"""

    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _natural_jpgs(rgb_dir: pathlib.Path) -> list[pathlib.Path]:
    """按帧号语义排序 jpg。

    正常 LEAD 文件名是 0000.jpg、0001.jpg；若混入非数字文件名，则排到最后，
    后续 validate_rgb_sequence 会把它识别为异常。
    """

    def _key(path: pathlib.Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return sys.maxsize, path.name

    return sorted(rgb_dir.glob("*.jpg"), key=_key)


def _has_jpg(rgb_dir: pathlib.Path) -> bool:
    """快速判断 rgb 目录里是否至少有一张 jpg。

    discover 阶段只需要知道这是不是一条可候选 route，不需要统计全部帧。
    用 iterdir 找到第一张 jpg 就返回，可以避免全量数据在 discover 时把每条
    route 的所有图片都 glob+sort 一遍。
    """

    try:
        for path in rgb_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".jpg":
                return True
    except OSError:
        return False
    return False


def _parse_frame_stems(rgb_files: Sequence[pathlib.Path]) -> list[int] | None:
    """把文件 stem 解析成帧号；任一文件非数字则返回 None。"""

    try:
        return [int(p.stem) for p in rgb_files]
    except ValueError:
        return None


def _ffprobe_image(path: pathlib.Path) -> tuple[int, int] | None:
    """用 ffprobe 读取单张图片的宽高。

    不在 Python 里引 PIL/cv2，避免远端轻量批处理脚本多一个 Python 图像依赖。
    """

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
    """在编码前剔除明显异常的 RGB 序列。

    这一步服务两个目标：
    1. 避免缺帧 / 非连续命名导致 ffmpeg 静默截断；
    2. 避免尺寸异常的 stitched RGB 被错误三等分。
    """

    rgb_files = _natural_jpgs(rgb_dir)
    if len(rgb_files) < min_frames:
        return False, f"too_few_frames:{len(rgb_files)}<{min_frames}", rgb_files, None

    # 默认严格要求 0000..N 连续。用户显式传 --allow-noncontiguous 时，后续会走
    # concat fallback，但正常全量巡检建议保持严格，宁可剔除坏 route。
    stems = _parse_frame_stems(rgb_files)
    if stems is None:
        return False, "non_numeric_frame_name", rgb_files, None
    expected = list(range(len(rgb_files)))
    if stems != expected and not allow_noncontiguous:
        missing = sorted(set(range(stems[0], stems[-1] + 1)) - set(stems))[:8]
        return False, f"noncontiguous_frames:first_missing={missing}", rgb_files, None

    # 只 probe 首尾帧，成本低；若首尾尺寸一致，通常整条 route 尺寸也一致。
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
) -> list[tuple[str, str, pathlib.Path, pathlib.Path]]:
    """发现数据根目录下所有含 rgb/*.jpg 的 route。

    data_root 的层级假设为 `<Scenario>/<run_id>/rgb/*.jpg`。
    注意这里不统计帧数；帧数留到 scan/convert 阶段按需计算，避免 discover 慢到
    还没打印进度就长时间无输出。
    """

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    routes: list[tuple[str, str, pathlib.Path, pathlib.Path]] = []
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
            if not _has_jpg(rgb_dir):
                continue
            routes.append((scenario, run_id, route_dir, rgb_dir))
    return routes


def build_tasks(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    scenarios: set[str] | None = None,
    run_ids: set[str] | None = None,
    *,
    show_progress: bool = True,
    progress_interval: int = DEFAULT_DISCOVER_PROGRESS_INTERVAL,
) -> list[RouteTask]:
    """把发现到的 route 转成带输出路径的任务列表。"""

    tasks: list[RouteTask] = []
    discover_start = time.time()
    scenario_count = 0
    if show_progress:
        print(f"[discover] scanning data_root={data_root}", flush=True)
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    for scenario_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        scenario = scenario_dir.name
        if scenarios and scenario not in scenarios:
            continue
        scenario_count += 1
        route_count_before = len(tasks)
        for route_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            run_id = route_dir.name
            if run_ids and run_id not in run_ids:
                continue
            rgb_dir = route_dir / "rgb"
            if not rgb_dir.is_dir() or not _has_jpg(rgb_dir):
                continue
            output_dir = output_root / scenario / run_id
            tasks.append(
                RouteTask(
                    scenario=scenario,
                    run_id=run_id,
                    route_dir=str(route_dir),
                    rgb_dir=str(rgb_dir),
                    output_dir=str(output_dir),
                    video_path=str(output_dir / VIDEO_NAME),
                    # discover 阶段刻意不数帧，后续 scan/convert 才按需填真实帧数。
                    frame_count=0,
                )
            )
        if show_progress and (scenario_count == 1 or scenario_count % max(1, progress_interval) == 0):
            elapsed = time.time() - discover_start
            added = len(tasks) - route_count_before
            print(
                f"[discover] scenarios={scenario_count} routes={len(tasks)} "
                f"last={scenario} added={added} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if show_progress:
        elapsed = time.time() - discover_start
        print(f"[discover] done scenarios={scenario_count} routes={len(tasks)} elapsed={elapsed:.1f}s", flush=True)
    return tasks


def _build_tasks_legacy(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    scenarios: set[str] | None = None,
    run_ids: set[str] | None = None,
) -> list[RouteTask]:
    """保留给外部 import 的轻量兼容入口；当前 CLI 不调用。"""

    tasks: list[RouteTask] = []
    for scenario, run_id, route_dir, rgb_dir in discover_routes(data_root, scenarios, run_ids):
        output_dir = output_root / scenario / run_id
        tasks.append(
            RouteTask(
                scenario=scenario,
                run_id=run_id,
                route_dir=str(route_dir),
                rgb_dir=str(rgb_dir),
                output_dir=str(output_dir),
                video_path=str(output_dir / VIDEO_NAME),
                frame_count=0,
            )
        )
    return tasks


def probe_video(video_path: pathlib.Path) -> dict:
    """读取已有 mp4 的 ffprobe 元数据。

    返回空 dict 表示 probe 失败；调用方会据此决定不能断点跳过。
    """

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
    """读取 route 输出目录里的 video_meta.json。"""

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
    """判断已有视频是否可以在断点续跑时跳过。

    除了检查视频本身的帧数/时长，还检查 video_meta.json 中的 views 与
    draw_frame_index 配置，防止用户改了输出配置后旧视频被误认为可复用。
    """

    if not video_path.exists() or video_path.stat().st_size <= 0:
        return False, "missing_or_empty"
    if require_manifest:
        # manifest 是配置兼容性标记；没有 manifest 的旧产物一律重做。
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


def _manifest_frame_count(output_dir: pathlib.Path) -> int | None:
    """从 video_meta.json 读取已知帧数。

    断点重跑时，如果旧产物 manifest 完整，就可以先用 manifest 里的 frame_count
    检查视频是否可跳过；这样 completed route 不需要再扫描 `/datashare` 原始 jpg。
    """

    manifest = _read_manifest(output_dir)
    try:
        frame_count = int(manifest.get("frame_count"))
    except (TypeError, ValueError):
        return None
    return frame_count if frame_count > 0 else None


def _task_video_paths(task: RouteTask, views: Sequence[str]) -> list[pathlib.Path]:
    """返回某条任务在指定 views 下应该产出的所有视频路径。"""

    output_dir = pathlib.Path(task.output_dir)
    return [output_dir / VIEW_TO_FILE[v] for v in views]


def inspect_task_for_resume(
    task: RouteTask,
    *,
    fps: float,
    views: Sequence[str],
    draw_frame_index: bool,
    overwrite: bool,
    min_frames: int,
    allow_noncontiguous: bool,
) -> PlanItem:
    """预判一条 route 是跳过、剔除还是需要编码。

    这个函数让正式开跑前就能打印 total/already_done/excluded/to_run，
    用户不用等 ffmpeg 跑起来才知道还剩多少工作量。
    """

    video_paths = _task_video_paths(task, views)
    outputs = [str(p) for p in video_paths]
    if not overwrite:
        manifest_count = _manifest_frame_count(pathlib.Path(task.output_dir))
        if manifest_count is not None:
            # 快路径：已有 manifest 时，先尝试只 probe 输出视频；成功则完全不碰原始 jpg。
            complete_reasons = [
                is_video_complete(p, manifest_count, fps, expected_frame_index=draw_frame_index)
                for p in video_paths
            ]
            if complete_reasons and all(ok for ok, _reason in complete_reasons):
                return PlanItem(
                    task=task,
                    status="already_done",
                    message="manifest_fast_skip:" + ";".join(reason for _ok, reason in complete_reasons),
                    frame_count=manifest_count,
                    outputs=outputs,
                )
        # 所有目标视角都完整时，才认为这条 route 已完成；缺任意一路就进入 to_run。
        if task.frame_count > 0:
            complete_reasons = [
                is_video_complete(p, task.frame_count, fps, expected_frame_index=draw_frame_index)
                for p in video_paths
            ]
            if complete_reasons and all(ok for ok, _reason in complete_reasons):
                return PlanItem(
                    task=task,
                    status="already_done",
                    message=";".join(reason for _ok, reason in complete_reasons),
                    frame_count=task.frame_count,
                    outputs=outputs,
                )

    ok, reason, rgb_files, dims = validate_rgb_sequence(
        pathlib.Path(task.rgb_dir),
        min_frames=min_frames,
        allow_noncontiguous=allow_noncontiguous,
    )
    if not ok:
        return PlanItem(
            task=task,
            status="excluded",
            message=reason,
            frame_count=len(rgb_files),
            outputs=outputs,
            dims=dims,
        )
    return PlanItem(
        task=task,
        status="to_run",
        message="needs_conversion" if not overwrite else "overwrite",
        frame_count=len(rgb_files),
        outputs=outputs,
        dims=dims,
    )


def build_resume_plan(
    tasks: Sequence[RouteTask],
    *,
    fps: float,
    views: Sequence[str],
    draw_frame_index: bool,
    overwrite: bool,
    min_frames: int,
    allow_noncontiguous: bool,
    show_progress: bool = True,
) -> tuple[list[PlanItem], PlanSummary]:
    """预扫描全部任务并返回逐条计划和聚合统计。"""

    items: list[PlanItem] = []
    summary = PlanSummary(total=len(tasks))
    scan_start = time.time()
    if show_progress and tasks:
        print_progress(0, len(tasks), scan_start, prefix="scan", last="starting")
    for idx, task in enumerate(tasks, start=1):
        item = inspect_task_for_resume(
            task,
            fps=fps,
            views=views,
            draw_frame_index=draw_frame_index,
            overwrite=overwrite,
            min_frames=min_frames,
            allow_noncontiguous=allow_noncontiguous,
        )
        items.append(item)
        if item.status == "already_done":
            summary.already_done += 1
        elif item.status == "excluded":
            summary.excluded += 1
        elif item.status == "to_run":
            summary.to_run += 1
        if show_progress and (idx == len(tasks) or idx % 20 == 0):
            print_progress(
                idx,
                len(tasks),
                scan_start,
                prefix="scan",
                last=f"{item.status} {task.scenario}/{task.run_id}",
            )
    return items, summary


def _result_from_plan_item(item: PlanItem) -> ConvertResult:
    """把无需编码的 plan item 转成最终结果，保证 summary JSON 仍记录它们。"""

    status = "skipped" if item.status == "already_done" else item.status
    return ConvertResult(
        item.task.scenario,
        item.task.run_id,
        item.outputs[0] if item.outputs else item.task.video_path,
        status,
        item.frame_count,
        item.message,
        0.0,
        item.outputs,
    )


def _progress_bar(done: int, total: int, *, width: int = 28) -> str:
    """生成固定宽度的文本进度条。"""

    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_progress(
    done: int,
    total: int,
    start_time: float,
    *,
    prefix: str = "progress",
    last: str = "",
) -> None:
    """打印 route 级实时进度。

    进度只统计 to_run，不把 already_done/excluded 混进分母，这样 eta 更贴近真实编码耗时。
    """

    elapsed = max(0.0, time.time() - start_time)
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = remaining / rate if rate > 0 else 0.0
    pct = (100.0 * done / total) if total > 0 else 100.0
    print(
        f"[{prefix}] {_progress_bar(done, total)} {done}/{total} "
        f"({pct:5.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s {last}",
        flush=True,
    )


def _drawtext_filter(view: str) -> str:
    """生成 ffmpeg drawtext 滤镜，在左上角写 view 与帧号。"""

    # n 是 ffmpeg 当前输出帧的 0-based 序号；正常连续 route 下正好对应 0000.jpg 的帧号。
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
    """生成单路视角对应的 ffmpeg -vf filter。

    left/front/right 通过 crop 从 stitched RGB 中裁出；frame index 通过 drawtext 叠加。
    """

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
    """为非连续帧写 ffmpeg concat demuxer 列表。

    正常 LEAD route 走 `%04d.jpg` image sequence 更快更干净；这个函数只作为
    `--allow-noncontiguous` 的兜底。
    """

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
    preset: str,
    ffmpeg_threads: int,
    draw_frame_index: bool,
    allow_noncontiguous: bool,
) -> tuple[bool, str, pathlib.Path]:
    """编码某条 route 的单个 view。

    返回 (是否成功, 诊断信息, 视频路径)。调用方会按 views 循环调用本函数。
    """

    video_path = output_dir / VIEW_TO_FILE[view]
    tmp_video = output_dir / f".{VIEW_TO_FILE[view]}.tmp.mp4"
    list_path: pathlib.Path | None = None
    frame_duration = 1.0 / float(fps)
    view_filter = _view_filter(view, draw_frame_index)

    stems = _parse_frame_stems(rgb_files)
    contiguous = stems == list(range(len(rgb_files)))
    if contiguous:
        # 标准 LEAD 命名路径：用 image sequence，保证输出帧数与输入帧数一一对应。
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
        # 非标准路径：显式列出每张图，避免 ffmpeg 遇到缺号时提前停止。
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
    # yuv420p + faststart 确保浏览器/VLC/mpv 都容易播放和拖动进度条。
    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-threads", str(max(0, int(ffmpeg_threads))),
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
        # 先写临时文件，成功后原子替换，避免中断时留下半截目标视频。
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
        # 清理临时 concat list 与未完成视频，断点续跑时不会误判它们。
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
    preset: str = DEFAULT_PRESET,
    ffmpeg_threads: int = DEFAULT_FFMPEG_THREADS,
    views: Sequence[str] = ("input",),
    draw_frame_index: bool = True,
    min_frames: int = 2,
    allow_noncontiguous: bool = False,
) -> ConvertResult:
    """转换一条 route 的一个或多个 view。"""

    start = time.time()
    output_dir = pathlib.Path(task.output_dir)
    video_paths = [output_dir / VIEW_TO_FILE[v] for v in views]
    # convert_route 自己也保留一次断点检查，防止预扫描之后外部进程刚好补齐产物。
    if not overwrite:
        expected_frames = task.frame_count if task.frame_count > 0 else _manifest_frame_count(output_dir)
        if expected_frames is not None:
            complete_reasons = [
                is_video_complete(p, expected_frames, fps, expected_frame_index=draw_frame_index)
                for p in video_paths
            ]
            if all(ok for ok, _reason in complete_reasons):
                return ConvertResult(
                    task.scenario,
                    task.run_id,
                    str(video_paths[0]),
                    "skipped",
                    expected_frames,
                    ";".join(reason for _ok, reason in complete_reasons),
                    time.time() - start,
                    [str(p) for p in video_paths],
                )

    ffmpeg = _require_tool("ffmpeg")
    rgb_dir = pathlib.Path(task.rgb_dir)
    # 再做一次异常检查，避免多进程场景下预扫描后数据被改动。
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
            # 单 view 级别断点：四路视角里已有的路不重做，只补缺失/失配的路。
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
            preset=preset,
            ffmpeg_threads=ffmpeg_threads,
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
    # manifest 既是人工可读元信息，也是后续断点续跑的配置兼容性依据。
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
        "ffmpeg_preset": preset,
        "ffmpeg_threads": int(ffmpeg_threads),
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
    """解析可重复传入、也可逗号/空格分隔的 CLI 参数。"""

    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if part:
                out.add(part)
    return out or None


def _parse_views(value: str) -> tuple[str, ...]:
    """解析并校验 --views。"""

    views = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    if not views:
        raise ValueError("--views must contain at least one view")
    bad = [v for v in views if v not in SUPPORTED_VIEWS]
    if bad:
        raise ValueError(f"unsupported view(s): {bad}; supported={SUPPORTED_VIEWS}")
    return views


def _write_summary(output_root: pathlib.Path, results: Iterable[ConvertResult]) -> None:
    """写全局 summary，记录 skipped/excluded/converted/failed 全部 route。"""

    rows = [asdict(r) for r in results]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "lead_video_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。"""

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
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET,
                        help=f"libx264 preset; default {DEFAULT_PRESET} for faster browsing videos")
    parser.add_argument("--ffmpeg-threads", type=int, default=DEFAULT_FFMPEG_THREADS,
                        help="Threads per ffmpeg process; default 1 for route-level parallelism, 0 lets ffmpeg decide")
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
        # 自动并行策略偏保守：CPU 一半、最多 DEFAULT_AUTO_WORKER_CAP 个，避免把共享存储打满。
        cpu_count = os.cpu_count() or 1
        resolved_workers = min(max(1, cpu_count // 2), DEFAULT_AUTO_WORKER_CAP, max(1, len(tasks)))
    else:
        resolved_workers = max(1, int(args.workers))

    print(f"[discover] data_root={args.data_root} output_root={args.output_root}")
    print(
        f"[discover] routes={len(tasks)} fps={args.fps} workers={resolved_workers} "
        f"views={','.join(views)} frame_index={not args.no_frame_index}"
    )
    if not tasks:
        print("[done] no routes found")
        return 0

    # 先要求 ffprobe 可用，因为预扫描就要用它判断已有视频和异常帧。
    _require_tool("ffprobe")
    plan_items, plan_summary = build_resume_plan(
        tasks,
        fps=args.fps,
        views=views,
        draw_frame_index=not args.no_frame_index,
        overwrite=args.overwrite,
        min_frames=args.min_frames,
        allow_noncontiguous=args.allow_noncontiguous,
        show_progress=True,
    )
    print(
        "[plan] "
        f"total={plan_summary.total} "
        f"already_done={plan_summary.already_done} "
        f"excluded={plan_summary.excluded} "
        f"to_run={plan_summary.to_run}"
    )
    if args.dry_run:
        # dry-run 打印逐条计划，但不创建输出目录、不启动 ffmpeg。
        for item in plan_items:
            print(
                f"[plan:{item.status}] {item.task.scenario}/{item.task.run_id}: "
                f"{item.frame_count} frames -> {item.outputs} ({item.message})"
            )
        return 0

    results: list[ConvertResult] = [
        _result_from_plan_item(item)
        for item in plan_items
        if item.status in ("already_done", "excluded")
    ]
    to_convert = [item.task for item in plan_items if item.status == "to_run"]
    if not to_convert:
        # 即使无需编码，也写 summary，方便用户确认本次扫描结果。
        _write_summary(args.output_root, results)
        print("[progress] no routes need conversion")
        print(f"[summary] converted=0 skipped={plan_summary.already_done} excluded={plan_summary.excluded} failed=0")
        print(f"[summary] wrote {args.output_root / 'lead_video_summary.json'}")
        return 0

    # 只有确实需要编码时才要求 ffmpeg，dry-run / 全部跳过不启动编码器。
    _require_tool("ffmpeg")
    workers = min(resolved_workers, len(to_convert))
    progress_start = time.time()
    print_progress(0, len(to_convert), progress_start, last="starting")
    if workers == 1:
        for idx, task in enumerate(to_convert, start=1):
            result = convert_route(
                task,
                args.fps,
                args.overwrite,
                args.crf,
                args.preset,
                args.ffmpeg_threads,
                views,
                not args.no_frame_index,
                args.min_frames,
                args.allow_noncontiguous,
            )
            results.append(result)
            print_progress(
                idx,
                len(to_convert),
                progress_start,
                last=f"{result.status} {result.scenario}/{result.run_id}",
            )
            print(
                f"[route {idx}/{len(to_convert)}] {result.status} {result.scenario}/{result.run_id} "
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
                    args.preset,
                    args.ffmpeg_threads,
                    views,
                    not args.no_frame_index,
                    args.min_frames,
                    args.allow_noncontiguous,
                )
                for task in to_convert
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = fut.result()
                results.append(result)
                print_progress(
                    idx,
                    len(to_convert),
                    progress_start,
                    last=f"{result.status} {result.scenario}/{result.run_id}",
                )
                print(
                    f"[route {idx}/{len(to_convert)}] {result.status} {result.scenario}/{result.run_id} "
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
