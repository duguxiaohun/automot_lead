"""用本地 Qwen3-VL-Instruct 探测 ROAD_STRUCTURE / EVENTS 标注能力。

这个脚本是轻量能力探针，不是最终数据生产入口。它支持两条思路：

1. RGB route：给一个 LEAD route id 或 route 目录，按 anchor 帧读取 stitched RGB。
2. Video：给一个 mp4，按时间窗口抽帧，让 Qwen 自己判断时间段标签。

Demo（默认当前目录为远程 AutoMoT/）：

GPU 规则：

- 不指定 GPU 时，脚本自动用 nvidia-smi 选择当前最空闲的 dGPU，并把进程内设备视为 cuda:0/auto。
- 需要显式指定 GPU 时，用 ``--gpu-ids 1``，或用环境变量 ``GPU_IDS=1``。

保存布局：

``eval_json/qwen_road_event_probe/run_<tag>/<Scenario>/<run_id>/<method>/<ROAD__EVENTS>/anchor_<frame>/``

其中 ``method`` 区分单帧、短 RGB clip、整段 run、视频 clip、整段视频；
``ROAD__EVENTS`` 按 Qwen 输出分组，便于比较不同处理方式和不同事件标签。

1. 单帧 RGB：从默认 lead_data 读取 Accident route 的第 80 帧，只给当前帧。

   python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --anchor 80 --num-frames 1 --run-tag accident_single_80

   python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --anchor 80 --num-frames 1 --run-tag accident_single_80

2. 短 RGB clip：同一 route 从 anchor 往前取 4 帧 stitched RGB，按 oldest->newest 喂给 Qwen。

   python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --anchor 80 --num-frames 4 --frame-step 1 --run-tag accident_clip_80

   python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --anchor 80 --num-frames 4 --frame-step 1 --run-tag accident_clip_80

3. 整段 run_id：扫描完整 route，每隔 8 帧问一次 Qwen，并在 summary.json 合并连续同标签时间段。

   python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --whole-run --every 8 --num-frames 4 --run-tag accident_whole_run

   python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --whole-run --every 8 --num-frames 4 --run-tag accident_whole_run

4. 整段视频：扫描 lead_video 里的 input.mp4，每隔 8 帧问一次 Qwen。

   python keyframe_filter/qwen_road_event_probe.py --video-file lead_video/Accident/<run_id>/input.mp4 --scenario Accident --whole-video --every 8 --num-frames 4 --run-tag accident_whole_video

   python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --video-file lead_video/Accident/<run_id>/input.mp4 --scenario Accident --whole-video --every 8 --num-frames 4 --run-tag accident_whole_video

默认只读本地 ``checkpoints/Qwen3-VL-4B-Instruct``，并通过
``qwen3vl_local.engine.LocalQwen3VLInstructEngine`` 做图文生成。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_THIS_FILE = pathlib.Path(__file__).resolve()
_KEYFRAME_DIR = _THIS_FILE.parent
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
_DEFAULT_MODEL_DIR = _AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"
_DEFAULT_MAPPING_MD = _KEYFRAME_DIR / "ROAD_EVENT_CANDIDATE_MAPPING.md"
_DEFAULT_SAVE_ROOT = _AUTOMOT_ROOT / "eval_json" / "qwen_road_event_probe"

for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


ROAD_DEFINITION_FALLBACK: Dict[str, str] = {
    "R1": "Regular same-direction road: default bucket for ordinary lane keeping, car following, same-direction lane change, and normal drivable road.",
    "R2": "Two-way single-lane / opposite-lane borrowing road: the opposite lane or oncoming traffic is part of the immediate decision.",
    "R3": "Highway / ramp / merge / exit road: ramp, merge, split, highway cut-in, highway exit, or speed-matching with main traffic.",
    "R4": "Signalized intersection: cross intersection, T-junction, or junction area where a normal traffic light is the main rule.",
    "R5": "Unsignalized or signal-failure intersection: cross intersection, T-junction, or junction area where no usable traffic-light rule is available.",
    "R6": "Roadside parking / parking-occupied road: parked cars, parking bay exit, door opening, roadside occlusion, or parking-dominated risk.",
}

EVENT_DEFINITION_FALLBACK: Dict[str, str] = {
    "R-E1": "Car following / lane keeping.",
    "R-E2": "Goal-directed lane change or lane correction required by the route.",
    "R-E3": "Routine ramp merge, lane merge, or highway exit.",
    "R-E4": "Following normal traffic-light rules at an intersection.",
    "R-E5": "Passing an unsignalized intersection by right-of-way and safe gaps.",
    "U-E1": "Lead vehicle hard braking or sudden deceleration.",
    "U-E2": "Static obstacle blocking the ego path, such as crash vehicle, construction object, or parked vehicle.",
    "U-E3": "Dynamic vehicle cut-in or dynamic occupation of the ego path.",
    "U-E4": "Pedestrian or cyclist crossing the ego path.",
    "U-E5": "Oncoming vehicle abnormally invading the ego lane.",
    "U-E6": "Rule-violating vehicle conflict, such as another vehicle running a red light.",
    "U-E7": "Traffic-light failure or intersection rule-source failure.",
    "U-E8": "Temporary blockage ahead or blockage clearing.",
}


def _cli_value(name: str) -> Optional[str]:
    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _pick_idle_gpus(n: int = 1) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


def _maybe_set_idle_gpu_mask(gpu_ids: str = "") -> None:
    """默认自动挑 1 张空闲 GPU；用户显式指定 device/GPU_IDS 时不覆盖。"""

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"[gpu] use explicit --gpu-ids={gpu_ids}; process uses cuda:0/auto")
        return
    if os.environ.get("GPU_IDS"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_IDS"]
        print(f"[gpu] use explicit GPU_IDS={os.environ['GPU_IDS']}")
        return
    device_arg = _cli_value("--device")
    if device_arg and device_arg.strip().lower() not in ("", "auto"):
        return
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"process uses cuda:0/auto; previous={previous or '<unset>'}"
        )


@dataclass
class CandidateTables:
    road_definitions: Dict[str, str]
    scenario_to_roads: Dict[str, List[str]]
    road_to_events: Dict[str, List[str]]
    scenario_to_events: Dict[str, List[str]]


@dataclass
class ProbeCase:
    source_kind: str
    scenario: str
    run_id: str
    anchor: int
    frame_indices: List[int]
    time_range: Tuple[float, float]
    images: List[Any]


def _resolve_automot_path(value: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (_AUTOMOT_ROOT / path).resolve()


def _split_md_row(line: str) -> List[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def _section_lines(text: str, heading_prefix: str) -> List[str]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return []
    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        out.append(line)
    return out


def _parse_event_ids(text: str) -> List[str]:
    return re.findall(r"\b(?:R|U)-E\d+\b", text)


def _parse_road_ids(text: str) -> List[str]:
    return re.findall(r"\bR[1-6]\b", text)


def load_candidate_tables(mapping_md: pathlib.Path) -> CandidateTables:
    """从候选 Markdown 表解析 scenario / road / event 候选。"""

    text = pathlib.Path(mapping_md).read_text(encoding="utf-8")
    road_definitions = dict(ROAD_DEFINITION_FALLBACK)
    scenario_to_roads: Dict[str, List[str]] = {}
    road_to_events: Dict[str, List[str]] = {}
    scenario_to_events: Dict[str, List[str]] = {}

    # 只从 Markdown 读取候选关系。给 Qwen 看的自然语言定义固定使用英文
    # ROAD_DEFINITION_FALLBACK，避免把中文说明混进 prompt。

    for line in _section_lines(text, "## 3."):
        if not line.strip().startswith("|") or "---" in line:
            continue
        cells = _split_md_row(line)
        if len(cells) >= 2 and cells[0] != "Scenario":
            scenario_to_roads[cells[0]] = _parse_road_ids(cells[1])

    for line in _section_lines(text, "## 4."):
        if not line.strip().startswith("|") or "---" in line:
            continue
        cells = _split_md_row(line)
        if len(cells) >= 2 and cells[0] != "ROAD_STRUCTURE":
            roads = _parse_road_ids(cells[0])
            if roads:
                road_to_events[roads[0]] = _parse_event_ids(cells[1])

    for line in _section_lines(text, "## 5."):
        if not line.strip().startswith("|") or "---" in line:
            continue
        cells = _split_md_row(line)
        if len(cells) >= 2 and cells[0] != "Scenario":
            scenario_to_events[cells[0]] = _parse_event_ids(cells[1])

    return CandidateTables(
        road_definitions=road_definitions,
        scenario_to_roads=scenario_to_roads,
        road_to_events=road_to_events,
        scenario_to_events=scenario_to_events,
    )


def _infer_scenario_from_path(path: pathlib.Path, tables: CandidateTables) -> str:
    for part in reversed(path.resolve().parts):
        if part in tables.scenario_to_roads:
            return part
    return "unknown"


def resolve_route_dir(args: argparse.Namespace, tables: CandidateTables) -> Tuple[pathlib.Path, str, str]:
    """解析 --route-id / --scenario + --run-id / --route-dir 三种 route 输入。"""

    if args.route_dir:
        route_dir = _resolve_automot_path(args.route_dir)
        scenario = args.scenario or _infer_scenario_from_path(route_dir, tables)
        return route_dir, scenario, route_dir.name

    if args.route_id:
        norm = args.route_id.replace("\\", "/").strip("/")
        parts = norm.split("/")
        if len(parts) < 2:
            raise ValueError("--route-id must look like Scenario/run_id")
        scenario, run_id = parts[0], "/".join(parts[1:])
        return _resolve_automot_path(args.data_root) / scenario / run_id, scenario, run_id.replace("/", "_")

    if args.scenario and args.run_id:
        return _resolve_automot_path(args.data_root) / args.scenario / args.run_id, args.scenario, args.run_id

    raise ValueError("provide --route-id, --route-dir, or --scenario + --run-id")


def _open_rgb(path: pathlib.Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _route_rgb_files(route_dir: pathlib.Path) -> List[pathlib.Path]:
    rgb_dir = route_dir / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"missing rgb directory: {rgb_dir}")
    files = sorted(rgb_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no .jpg under {rgb_dir}")
    return files


def _clip_indices(anchor: int, total: int, frame_count: int, frame_step: int) -> List[int]:
    frame_count = max(1, int(frame_count))
    frame_step = max(1, int(frame_step))
    anchor = min(max(int(anchor), 0), total - 1)
    desc = [max(anchor - i * frame_step, 0) for i in range(frame_count)]
    return list(reversed(desc))


def load_route_case(
    route_dir: pathlib.Path,
    scenario: str,
    run_id: str,
    anchor: int,
    frame_count: int,
    frame_step: int,
) -> ProbeCase:
    files = _route_rgb_files(route_dir)
    indices = _clip_indices(anchor, len(files), frame_count, frame_step)
    images = []
    for idx in indices:
        expected = route_dir / "rgb" / f"{idx:04d}.jpg"
        images.append(_open_rgb(expected if expected.exists() else files[idx]))
    return ProbeCase(
        source_kind="route_rgb",
        scenario=scenario,
        run_id=run_id,
        anchor=min(max(anchor, 0), len(files) - 1),
        frame_indices=indices,
        time_range=(indices[0] * 0.25, indices[-1] * 0.25),
        images=images,
    )


def _load_video_frames(video_file: pathlib.Path, indices: Sequence[int]) -> List[Any]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open video: {video_file}")
    images = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read video frame {idx} from {video_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame).convert("RGB"))
    finally:
        cap.release()
    return images


def load_video_case(
    video_file: pathlib.Path,
    scenario: str,
    anchor: int,
    frame_count: int,
    frame_step: int,
) -> ProbeCase:
    import cv2

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open video: {video_file}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 4.0)
    cap.release()
    if total <= 0:
        raise RuntimeError(f"video has no readable frames: {video_file}")
    indices = _clip_indices(anchor, total, frame_count, frame_step)
    images = _load_video_frames(video_file, indices)
    return ProbeCase(
        source_kind="video",
        scenario=scenario,
        run_id=video_file.stem,
        anchor=min(max(anchor, 0), total - 1),
        frame_indices=indices,
        time_range=(indices[0] / fps, indices[-1] / fps),
        images=images,
    )


def load_synthetic_case(scenario: str, anchor: int, frame_count: int, frame_step: int) -> ProbeCase:
    """生成无驾驶语义的三视角占位图，只用于 dry-run 检查 prompt/落盘链路。"""

    from PIL import Image, ImageDraw

    images = []
    indices = _clip_indices(anchor, max(anchor + 1, 1), frame_count, frame_step)
    for idx in indices:
        img = Image.new("RGB", (1152, 384), (35, 35, 35))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 383, 383], fill=(120, 50, 50))
        draw.rectangle([384, 0, 767, 383], fill=(50, 110, 60))
        draw.rectangle([768, 0, 1151, 383], fill=(50, 70, 130))
        draw.text((16, 16), f"synthetic frame {idx}", fill=(255, 255, 255))
        images.append(img)
    return ProbeCase(
        source_kind="synthetic",
        scenario=scenario or "Accident",
        run_id="synthetic",
        anchor=anchor,
        frame_indices=indices,
        time_range=(indices[0] * 0.25, indices[-1] * 0.25),
        images=images,
    )


def build_timeline_anchors(total_frames: int, every: int, start: Optional[int], end: Optional[int]) -> List[int]:
    lo = max(0, 0 if start is None else int(start))
    hi = min(total_frames - 1, total_frames - 1 if end is None else int(end))
    if hi < lo:
        raise ValueError(f"empty frame range: start={start}, end={end}, total={total_frames}")
    every = max(1, int(every))
    anchors = list(range(lo, hi + 1, every))
    if anchors[-1] != hi:
        anchors.append(hi)
    return anchors


def system_prompt() -> str:
    return (
        "You are an autonomous-driving visual annotator. "
        "Classify the ego vehicle's current road-structure and events from the provided driving images. "
        "Write a brief evidence-based analysis, then return valid JSON only."
    )


def _candidate_lines(labels: Iterable[str], descriptions: Dict[str, str]) -> str:
    return "\n".join(f"- {label}: {descriptions.get(label, label)}" for label in labels)


def _event_descriptions(labels: Iterable[str]) -> str:
    return "\n".join(f"- {label}: {EVENT_DEFINITION_FALLBACK.get(label, label)}" for label in labels)


def expand_junction_candidates(
    roads: Sequence[str],
    scenario_events: Sequence[str],
    tables: CandidateTables,
) -> Tuple[List[str], List[str]]:
    """If any junction is possible, expose both R4 and R5 to Qwen.

    Some routes contain an unsignalized T-junction even when the scenario prior
    only listed a generic signalized-junction candidate. Showing both R4 and R5
    lets the model choose by visible geometry + signal evidence instead of
    falling back to R1.
    """

    out_roads = list(dict.fromkeys(str(x) for x in roads))
    has_junction = "R4" in out_roads or "R5" in out_roads
    if has_junction:
        for road in ("R4", "R5"):
            if road not in out_roads:
                out_roads.append(road)

    out_events = list(dict.fromkeys(str(x) for x in scenario_events))
    if has_junction:
        event_set = set(out_events)
        for road in ("R4", "R5"):
            event_set.update(tables.road_to_events.get(road, []))
        out_events = [event for event in sorted(event_set) if event in EVENT_DEFINITION_FALLBACK]
    return out_roads, out_events


def build_user_prompt(case: ProbeCase, tables: CandidateTables) -> str:
    raw_roads = tables.scenario_to_roads.get(case.scenario) or list(ROAD_DEFINITION_FALLBACK)
    raw_events = tables.scenario_to_events.get(case.scenario) or sorted(EVENT_DEFINITION_FALLBACK)
    roads, scenario_events = expand_junction_candidates(raw_roads, raw_events, tables)
    per_road_event_lines = []
    for road in roads:
        road_events = tables.road_to_events.get(road, scenario_events)
        allowed = [x for x in road_events if x in scenario_events] or road_events
        per_road_event_lines.append(f"- {road}: {', '.join(allowed)}")

    if len(case.images) <= 1:
        image_note = "One current RGB observation is provided."
    else:
        image_note = (
            f"{len(case.images)} RGB observations are provided oldest to newest. "
            "Use temporal changes to decide whether the label changed inside this window."
        )

    return f"""Task: label the current autonomous-driving observation.

Input source:
- source_kind: {case.source_kind}
- scenario_prior: {case.scenario}
- run_id: {case.run_id}
- anchor_frame: {case.anchor}
- frame_indices: {case.frame_indices}
- approximate_time_range_s: [{case.time_range[0]:.2f}, {case.time_range[1]:.2f}]
- image_order: {image_note}

ROAD_STRUCTURE must be exactly one of these candidates:
{_candidate_lines(roads, tables.road_definitions)}

Look for only decision-relevant visible cues:
- traffic lights and traffic signs;
- nearby vehicles, pedestrians, cyclists, and obstacles;
- lane markings and road structure;
- key factors that affect the ego vehicle's next decision.

Important distinction rules:
- R1 is the default/other bucket. Choose R1 when there is no clear visual evidence for junction, ramp/merge/highway exit, two-way single-lane borrowing, or parking-dominated risk.
- Choose R2 only when the opposite lane or oncoming traffic is part of the immediate decision.
- Choose R3 only for ramp, merge, highway, split, or exit structures.
- A junction can be recognized by road geometry, not only by traffic lights: side-road opening, T-shaped road, cross-shaped road, stop/yield line, turning pocket, cross traffic, or a wide conflict area.
- If clear junction geometry is visible, do not choose R1 just because no traffic light is visible.
- Choose R4 when junction geometry is present and a normal traffic light is visible or is the main rule.
- Choose R5 when junction geometry is present but no usable traffic light is visible, or the decision mainly depends on right-of-way/gaps.
- Choose R6 only when roadside parking, parking-space exit, parked cars, door opening, or parking occlusion dominates the decision.

EVENTS may contain one or more labels, but only from the candidates compatible with the chosen ROAD_STRUCTURE and scenario:
{chr(10).join(per_road_event_lines)}

Scenario-level event candidates:
{_event_descriptions(scenario_events)}

Return the answer only as this JSON object. Do not output markdown or extra explanation:
{{
  "ANALYSIS": "brief visible-evidence analysis in one or two short sentences",
  "ROAD_STRUCTURE": "R1",
  "EVENTS": ["R-E1"],
  "STATUS": "R-E1",
  "SUBGOAL": "R-E1",
  "SPAN_CHANGE": false,
  "CONFIDENCE": 0.0,
  "REASON": "one short sentence based only on visible evidence"
}}
"""


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    """容错解析模型 JSON；失败时仍提取 ROAD_STRUCTURE / EVENTS 行。"""

    text = (raw_text or "").strip()
    candidates = [text]
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidates.insert(0, m.group(1).strip())
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidates.insert(0, m.group(0))
    for item in candidates:
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    road_match = re.search(r"\bR[1-6]\b", text)
    events = _parse_event_ids(text)
    return {
        "ROAD_STRUCTURE": road_match.group(0) if road_match else None,
        "EVENTS": events,
        "STATUS": events[0] if events else None,
        "SUBGOAL": events[-1] if events else None,
        "ANALYSIS": text[:500],
        "SPAN_CHANGE": None,
        "CONFIDENCE": None,
        "REASON": text[:500],
        "_parse_warning": "json_parse_failed",
    }


def normalize_label_record(parsed: Dict[str, Any], case: ProbeCase, raw_text: str) -> Dict[str, Any]:
    events = parsed.get("EVENTS")
    if isinstance(events, str):
        events = _parse_event_ids(events) or [events]
    if not isinstance(events, list):
        events = []
    events = [str(x).strip() for x in events if str(x).strip()]

    return {
        "source_kind": case.source_kind,
        "scenario": case.scenario,
        "run_id": case.run_id,
        "anchor": case.anchor,
        "frame_indices": case.frame_indices,
        "time_range_s": [round(case.time_range[0], 3), round(case.time_range[1], 3)],
        "ROAD_STRUCTURE": parsed.get("ROAD_STRUCTURE"),
        "EVENTS": events,
        "STATUS": parsed.get("STATUS"),
        "SUBGOAL": parsed.get("SUBGOAL"),
        "ANALYSIS": parsed.get("ANALYSIS", ""),
        "SPAN_CHANGE": parsed.get("SPAN_CHANGE"),
        "CONFIDENCE": parsed.get("CONFIDENCE"),
        "REASON": parsed.get("REASON", ""),
        "raw_text": raw_text,
    }


def safe_path_part(value: Any, fallback: str = "unknown") -> str:
    """把 scenario/run_id/event/method 转成稳定目录名。"""

    text = str(value or "").strip()
    if not text:
        text = fallback
    text = text.replace("\\", "/").strip("/")
    text = re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", text)
    return text or fallback


def method_key(args: argparse.Namespace, *, source_kind: str, scan: bool) -> str:
    """生成处理方法目录名，便于比较单帧、clip、整段和视频。"""

    if source_kind == "route":
        base = "whole_run" if scan else ("single_rgb" if args.num_frames <= 1 else "rgb_clip")
    elif source_kind == "video":
        base = "whole_video" if scan else ("single_video_frame" if args.num_frames <= 1 else "video_clip")
    else:
        base = "synthetic_timeline" if scan else "synthetic_single"
    return safe_path_part(
        f"{base}_frames{max(1, int(args.num_frames))}_step{max(1, int(args.frame_step))}_every{max(1, int(args.every)) if scan else 'na'}"
    )


def event_key(record: Dict[str, Any]) -> str:
    """按模型输出的 ROAD_STRUCTURE + EVENTS 分组保存。"""

    road = record.get("ROAD_STRUCTURE") or "NO_ROAD"
    events = record.get("EVENTS") or []
    if events:
        event_text = "+".join(str(x) for x in events)
    else:
        event_text = "NO_EVENT"
    return safe_path_part(f"{road}__{event_text}")


def _label_key(record: Dict[str, Any]) -> Tuple[Any, Tuple[str, ...]]:
    return record.get("ROAD_STRUCTURE"), tuple(record.get("EVENTS") or [])


def merge_adjacent_spans(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    spans: List[Dict[str, Any]] = []
    cur = {
        "start_anchor": records[0]["anchor"],
        "end_anchor": records[0]["anchor"],
        "start_time_s": records[0]["time_range_s"][0],
        "end_time_s": records[0]["time_range_s"][1],
        "ROAD_STRUCTURE": records[0].get("ROAD_STRUCTURE"),
        "EVENTS": records[0].get("EVENTS") or [],
        "records": 1,
    }
    prev_key = _label_key(records[0])
    for record in records[1:]:
        key = _label_key(record)
        if key == prev_key:
            cur["end_anchor"] = record["anchor"]
            cur["end_time_s"] = record["time_range_s"][1]
            cur["records"] += 1
        else:
            spans.append(dict(cur))
            cur = {
                "start_anchor": record["anchor"],
                "end_anchor": record["anchor"],
                "start_time_s": record["time_range_s"][0],
                "end_time_s": record["time_range_s"][1],
                "ROAD_STRUCTURE": record.get("ROAD_STRUCTURE"),
                "EVENTS": record.get("EVENTS") or [],
                "records": 1,
            }
            prev_key = key
    spans.append(dict(cur))
    return spans


def case_manifest(case: ProbeCase) -> Dict[str, Any]:
    """保存给人工复盘看的输入元信息。"""

    return {
        "source_kind": case.source_kind,
        "scenario": case.scenario,
        "run_id": case.run_id,
        "anchor": case.anchor,
        "frame_indices": case.frame_indices,
        "time_range_s": [round(case.time_range[0], 3), round(case.time_range[1], 3)],
        "num_images": len(case.images),
        "image_order": "oldest_to_newest",
    }


def save_case_io(
    case: ProbeCase,
    out_dir: pathlib.Path,
    sys_prompt: str,
    user_prompt: str,
    raw_text: str,
    parsed: Dict[str, Any],
) -> None:
    """完整保存单次 Qwen 调用的输入输出，方便逐 anchor 人工检查。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_dir / "inputs"
    outputs_dir = out_dir / "outputs"
    img_dir = inputs_dir / "images"
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    (inputs_dir / "system_prompt.txt").write_text(sys_prompt, encoding="utf-8")
    (inputs_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (inputs_dir / "case.json").write_text(
        json.dumps(case_manifest(case), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for i, img in enumerate(case.images):
        if hasattr(img, "save"):
            img.save(str(img_dir / f"{i:02d}_frame_{case.frame_indices[i]:04d}.jpg"), quality=90)

    (outputs_dir / "qwen_raw_text.txt").write_text(raw_text, encoding="utf-8")
    (outputs_dir / "answer.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 根目录保留一份短入口，方便脚本/人工快速打开。
    (out_dir / "raw_text.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "parsed.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")


def run_qwen_case(
    engine: Optional[Any],
    case: ProbeCase,
    tables: CandidateTables,
    method_dir: pathlib.Path,
    dry_run: bool,
) -> Dict[str, Any]:
    sys_prompt = system_prompt()
    user = build_user_prompt(case, tables)
    if dry_run:
        raw_text = ""
        parsed = {
            "ANALYSIS": "",
            "ROAD_STRUCTURE": None,
            "EVENTS": [],
            "STATUS": None,
            "SUBGOAL": None,
            "SPAN_CHANGE": None,
            "CONFIDENCE": None,
            "REASON": "dry_run",
        }
    else:
        assert engine is not None
        raw_text, trace = engine.generate(sys_prompt, user, case.images)
        parsed = parse_model_json(raw_text)
    record = normalize_label_record(parsed, case, raw_text)
    case_dir = method_dir / event_key(record) / f"anchor_{case.anchor:04d}"
    if not dry_run:
        (case_dir / "trace").mkdir(parents=True, exist_ok=True)
        (case_dir / "trace" / "generation_trace.json").write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    save_case_io(case, case_dir, sys_prompt, user, raw_text, record)
    record["save_dir"] = str(case_dir)
    record["method_dir"] = str(method_dir)
    record["event_dir"] = str(case_dir.parent)
    return record


def build_engine(args: argparse.Namespace) -> Any:
    _maybe_set_idle_gpu_mask(args.gpu_ids)
    from qwen3vl_local.engine import LocalQwen3VLInstructEngine

    model_dir = _resolve_automot_path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"missing local Qwen checkpoint: {model_dir}")
    return LocalQwen3VLInstructEngine(
        checkpoint_dir=model_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        save_cache=False,
        cache_system_prompt=False,
    )


def _route_total_frames(route_dir: pathlib.Path) -> int:
    return len(_route_rgb_files(route_dir))


def _video_total_frames(video_file: pathlib.Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open video: {video_file}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total


def should_scan_timeline(args: argparse.Namespace, *, source_kind: str) -> bool:
    """统一判断当前输入是单 anchor 还是整段扫描。"""

    if args.whole_run and source_kind == "route":
        return True
    if args.whole_video and source_kind == "video":
        return True
    return args.mode == "timeline"


def run(args: argparse.Namespace) -> None:
    tables = load_candidate_tables(_resolve_automot_path(args.mapping_md))
    save_root = _resolve_automot_path(args.save_root)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = save_root / f"run_{run_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    engine = None if args.dry_run else build_engine(args)
    records: List[Dict[str, Any]] = []

    if args.synthetic:
        scan = should_scan_timeline(args, source_kind="synthetic")
        anchors = [args.anchor] if not scan else build_timeline_anchors(
            max(args.end or args.anchor, args.anchor) + 1,
            args.every,
            args.start,
            args.end,
        )
        scenario = args.scenario or "Accident"
        run_id = "synthetic"
        method_dir = (
            out_root
            / safe_path_part(scenario)
            / safe_path_part(run_id)
            / method_key(args, source_kind="synthetic", scan=scan)
        )
        for anchor in anchors:
            case = load_synthetic_case(scenario, anchor, args.num_frames, args.frame_step)
            records.append(run_qwen_case(engine, case, tables, method_dir, args.dry_run))
    elif args.video_file:
        video_file = _resolve_automot_path(args.video_file)
        total = _video_total_frames(video_file)
        scenario = args.scenario or _infer_scenario_from_path(video_file, tables)
        scan = should_scan_timeline(args, source_kind="video")
        anchors = (
            [args.anchor]
            if not scan
            else build_timeline_anchors(total, args.every, args.start, args.end)
        )
        run_id = video_file.stem
        method_dir = (
            out_root
            / safe_path_part(scenario)
            / safe_path_part(run_id)
            / method_key(args, source_kind="video", scan=scan)
        )
        for anchor in anchors:
            case = load_video_case(video_file, scenario, anchor, args.num_frames, args.frame_step)
            case.run_id = run_id
            records.append(run_qwen_case(engine, case, tables, method_dir, args.dry_run))
    else:
        route_dir, scenario, run_id = resolve_route_dir(args, tables)
        should_exclude, abnormal_info = is_abnormal_lead_route(route_dir, scenario)
        if should_exclude:
            raise ValueError(
                "abnormal LEAD route rejected before probe: "
                f"{scenario}/{run_id} duration_s={abnormal_info['duration_s']:.2f} "
                "(rule: duration_s > 90 unless scenario is BlockedIntersection or ControlLoss)"
            )
        total = _route_total_frames(route_dir)
        scan = should_scan_timeline(args, source_kind="route")
        anchors = (
            [args.anchor]
            if not scan
            else build_timeline_anchors(total, args.every, args.start, args.end)
        )
        method_dir = (
            out_root
            / safe_path_part(scenario)
            / safe_path_part(run_id)
            / method_key(args, source_kind="route", scan=scan)
        )
        for anchor in anchors:
            case = load_route_case(route_dir, scenario, run_id, anchor, args.num_frames, args.frame_step)
            records.append(run_qwen_case(engine, case, tables, method_dir, args.dry_run))

    jsonl_path = out_root / "labels_all.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    method_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        method_groups.setdefault(str(record.get("method_dir", out_root)), []).append(record)
    for method_dir_text, method_records in method_groups.items():
        method_path = pathlib.Path(method_dir_text)
        method_path.mkdir(parents=True, exist_ok=True)
        method_jsonl = method_path / "labels.jsonl"
        with method_jsonl.open("w", encoding="utf-8") as f:
            for record in method_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        method_summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_tag": run_tag,
            "record_count": len(method_records),
            "labels_jsonl": str(method_jsonl),
            "event_groups": sorted({event_key(record) for record in method_records}),
            "spans": merge_adjacent_spans(method_records),
        }
        (method_path / "summary.json").write_text(
            json.dumps(method_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "record_count": len(records),
        "labels_jsonl": str(jsonl_path),
        "layout": "<save-root>/run_<tag>/<scenario>/<run_id>/<method>/<ROAD__EVENTS>/anchor_<frame>/",
        "method_dirs": sorted(method_groups),
        "spans": merge_adjacent_spans(records),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"[output] {out_root}")
    print(f"[labels] {jsonl_path}")
    print(f"[records] {len(records)}")
    for span in summary["spans"]:
        print(
            f"[span] {span['start_anchor']}..{span['end_anchor']} "
            f"{span['ROAD_STRUCTURE']} {','.join(span['EVENTS'])}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    demo = """demo:
  # GPU：不指定时自动选最空闲 dGPU；显式指定时用 --gpu-ids 1 或 GPU_IDS=1。

  # 1) 单帧 RGB：从默认 lead_data 读取 Accident route 的第 80 帧，只给当前帧。
  python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --anchor 80 --num-frames 1 --run-tag accident_single_80
  python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --anchor 80 --num-frames 1 --run-tag accident_single_80

  # 2) 短 RGB clip：同一 route 从 anchor 往前取 4 帧 stitched RGB，按 oldest->newest 喂给 Qwen。
  python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --anchor 80 --num-frames 4 --frame-step 1 --run-tag accident_clip_80
  python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --anchor 80 --num-frames 4 --frame-step 1 --run-tag accident_clip_80

  # 3) 整段 run_id：扫描完整 lead_data route，每隔 8 帧问一次 Qwen。
  python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --whole-run --every 8 --num-frames 4 --run-tag accident_whole_run
  python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --route-id Accident/<run_id> --whole-run --every 8 --num-frames 4 --run-tag accident_whole_run

  # 4) 整段视频：扫描完整 lead_video input.mp4，每隔 8 帧问一次 Qwen。
  python keyframe_filter/qwen_road_event_probe.py --video-file lead_video/Accident/<run_id>/input.mp4 --scenario Accident --whole-video --every 8 --num-frames 4 --run-tag accident_whole_video
  python keyframe_filter/qwen_road_event_probe.py --gpu-ids 1 --video-file lead_video/Accident/<run_id>/input.mp4 --scenario Accident --whole-video --every 8 --num-frames 4 --run-tag accident_whole_video

  # 环境变量写法也可以。
  GPU_IDS=1 python keyframe_filter/qwen_road_event_probe.py --route-id Accident/<run_id> --anchor 80 --num-frames 1
"""
    p = argparse.ArgumentParser(
        description="Probe local Qwen3-VL road/event annotation ability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=demo,
    )
    p.add_argument("--model-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--mapping-md", default=str(_DEFAULT_MAPPING_MD))
    p.add_argument("--save-root", default=str(_DEFAULT_SAVE_ROOT))
    p.add_argument("--run-tag", default="")
    p.add_argument("--dry-run", action="store_true", help="只落 prompt/采样图片，不加载模型。")

    p.add_argument("--mode", choices=["single", "timeline"], default="single")
    p.add_argument("--data-root", default="lead_data")
    p.add_argument("--route-id", default="", help="形如 Accident/Town03_... 的 route id。")
    p.add_argument("--route-dir", default="", help="直接指定包含 rgb/*.jpg 的 route 目录。")
    p.add_argument("--scenario", default="", help="scenario 先验；route-id 会自动提供。")
    p.add_argument("--run-id", default="")
    p.add_argument("--video-file", default="", help="mp4 路径；会按窗口抽帧当作视频方案测试。")
    p.add_argument("--synthetic", action="store_true", help="生成占位图，仅用于 dry-run/入口检查。")
    p.add_argument("--whole-run", action="store_true", help="扫描完整 route/run_id；等价于 route 输入下的 --mode timeline。")
    p.add_argument("--whole-video", action="store_true", help="扫描完整视频；等价于 video 输入下的 --mode timeline。")

    p.add_argument("--anchor", type=int, default=12)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--every", type=int, default=16, help="timeline 模式每隔多少帧问一次。")
    p.add_argument("--num-frames", type=int, default=4, help="每次给 Qwen 的图像数，oldest->newest。")
    p.add_argument("--frame-step", type=int, default=1, help="同一窗口内相邻图像间隔。")

    p.add_argument("--device", default="auto")
    p.add_argument("--gpu-ids", default="", help="显式指定可见 GPU，例如 0 或 1；优先级高于 GPU_IDS。")
    p.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--max-gen-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
