"""Build GoalGen v1 jsonl datasets from keyframe timelines.

Run from ``AutoMoT/`` on the remote machine, for example:

python qwen3vl_local/goalgen/build_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /data/lead_data/data \
  --output-dir checkpoints/goalgen_v1_data

The builder mirrors the SFT v1 timeline idea, but the target is different:
for each anchor frame, STATUS is the current GT state and SUBGOAL is the next
event in the scenario sequence. The supervised image target is the future
keyframe where that SUBGOAL begins.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    SCENARIO_EVENT_SEQUENCES,
    SCENARIO_LABELS,
    get_full_sequence,
)


ACCEPTED_RUN_STATUS = {"Completed", "Perfect"}
DEFAULT_KEYFRAMES = "/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json"
DEFAULT_DATA_ROOT = "/data/lead_data/data"
RGB_FRAME_COUNT = 4
RGB_FRAME_STEP = 1


@dataclass
class RunTimeline:
    scenario: str
    run_id: str
    total_frames: int
    intervals: List[Tuple[int, int, str]]
    event_frames: Dict[str, int]


@dataclass
class GoalGenSample:
    scenario: str
    run_id: str
    anchor: int
    status: str
    subgoal: str
    target_frame: int
    history_frames: List[int]
    history_rgb_paths: List[str]
    current_rgb_path: str
    target_rgb_path: str


def next_event_in_sequence(scenario: str, status: str) -> Optional[str]:
    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return None
    if idx + 1 >= len(seq):
        return None
    return seq[idx + 1]


def build_run_timeline(run: dict) -> Optional[RunTimeline]:
    if run.get("status") not in ACCEPTED_RUN_STATUS:
        return None

    scenario = run.get("scenario")
    if scenario not in SCENARIO_EVENT_SEQUENCES:
        return None

    initial = run.get("initial")
    middle = run.get("middle", [])
    final = run.get("final")
    total_frames = run.get("diagnostics", {}).get("total_frames")
    if not initial or len(middle) != 3 or not final or total_frames is None:
        return None

    expected_seq = get_full_sequence(scenario)
    actual_seq = (
        initial["event"],
        middle[0]["event"],
        middle[1]["event"],
        middle[2]["event"],
        final["event"],
    )
    if actual_seq != expected_seq:
        return None

    boundaries = [
        (initial["frame"], middle[0]["frame"] - 1, initial["event"]),
        (middle[0]["frame"], middle[1]["frame"] - 1, middle[0]["event"]),
        (middle[1]["frame"], middle[2]["frame"] - 1, middle[1]["event"]),
        (middle[2]["frame"], final["frame"] - 1, middle[2]["event"]),
        (final["frame"], total_frames - 1, final["event"]),
    ]
    for start, end, _status in boundaries:
        if start > end:
            return None

    event_frames = {
        initial["event"]: int(initial["frame"]),
        middle[0]["event"]: int(middle[0]["frame"]),
        middle[1]["event"]: int(middle[1]["frame"]),
        middle[2]["event"]: int(middle[2]["frame"]),
        final["event"]: int(final["frame"]),
    }
    return RunTimeline(
        scenario=scenario,
        run_id=run["run_id"],
        total_frames=int(total_frames),
        intervals=boundaries,
        event_frames=event_frames,
    )


def iter_status_ranges(timeline: RunTimeline) -> Iterable[Tuple[int, int, str]]:
    for start, end, status in timeline.intervals:
        if status == "final":
            continue
        subgoal = next_event_in_sequence(timeline.scenario, status)
        if not subgoal:
            continue
        target_frame = timeline.event_frames.get(subgoal)
        if target_frame is None:
            continue
        # The valid anchors are strictly before the subgoal keyframe.
        yield start, min(end, target_frame - 1), status


def _as_posix(path: pathlib.Path) -> str:
    return str(path).replace("\\", "/")


def _load_rgb_directory(rgb_dir: pathlib.Path) -> tuple[set, List[pathlib.Path]]:
    """读一次 route/rgb/，返回 (文件名集合, 排序后文件列表)。

    NFS 上 `Path.exists()` 是单次 stat（~1ms）；7000 route × 几十 anchor × 4 帧
    历史 × 1 目标 = 数百万次会非常慢。这里改成：每个 route 进 collect_samples 时
    只调一次 `rgb_dir.glob` 然后在内存里 O(1) 查。
    """

    if not rgb_dir.exists():
        return set(), []
    files = sorted(rgb_dir.glob("*.jpg"))
    names = {p.name for p in files}
    return names, files


def resolve_rgb_path(
    route_dir: pathlib.Path,
    frame_idx: int,
    rgb_cache: Optional[tuple] = None,
) -> str:
    """把 frame_idx 解析成磁盘上的 JPG 路径字符串。

    优先用 `{frame_idx:04d}.jpg` 这个 LEAD 默认命名；找不到则按 sorted 列表 fallback。
    rgb_cache 是 (names_set, sorted_files) 元组，由 _load_rgb_directory 一次性
    准备好；提供时所有判断都走内存，没有额外 IO。不提供时退回到老的 Path.exists()
    路径（仅供单元测试或一次性查询）。
    """

    rgb_dir = route_dir / "rgb"
    name = f"{frame_idx:04d}.jpg"
    direct = rgb_dir / name

    if rgb_cache is not None:
        names, files = rgb_cache
        if name in names:
            return _as_posix(direct)
        if 0 <= frame_idx < len(files):
            return _as_posix(files[frame_idx])
        return _as_posix(direct)

    if direct.exists():
        return _as_posix(direct)

    files = sorted(rgb_dir.glob("*.jpg")) if rgb_dir.exists() else []
    if 0 <= frame_idx < len(files):
        return _as_posix(files[frame_idx])
    return _as_posix(direct)


def history_frames(anchor: int, count: int, step: int) -> List[int]:
    desc = [max(anchor - i * step, 0) for i in range(count)]
    return list(reversed(desc))


def collect_samples(
    timeline: RunTimeline,
    data_root: pathlib.Path,
    frame_stride: int,
    min_future_gap: int,
    num_frames: int,
    rgb_frame_step: int,
) -> List[GoalGenSample]:
    route_dir = data_root / timeline.scenario / timeline.run_id
    samples: List[GoalGenSample] = []

    # 每个 route 只 glob 一次 rgb/，后续 resolve_rgb_path 走内存查；NFS 上能省
    # 几个数量级的 stat 调用，对 7000 routes 的 dataset build 几乎是必须的优化。
    rgb_cache = _load_rgb_directory(route_dir / "rgb")

    for start, end, status in iter_status_ranges(timeline):
        subgoal = next_event_in_sequence(timeline.scenario, status)
        if subgoal is None:
            continue
        target_frame = timeline.event_frames[subgoal]

        min_anchor = max(start, (num_frames - 1) * rgb_frame_step)
        max_anchor = min(end, target_frame - min_future_gap)
        if min_anchor > max_anchor:
            continue

        for anchor in range(min_anchor, max_anchor + 1, frame_stride):
            frames = history_frames(anchor, num_frames, rgb_frame_step)
            hist_paths = [resolve_rgb_path(route_dir, f, rgb_cache=rgb_cache) for f in frames]
            current_path = hist_paths[-1]
            target_path = resolve_rgb_path(route_dir, target_frame, rgb_cache=rgb_cache)
            samples.append(
                GoalGenSample(
                    scenario=timeline.scenario,
                    run_id=timeline.run_id,
                    anchor=anchor,
                    status=status,
                    subgoal=subgoal,
                    target_frame=target_frame,
                    history_frames=frames,
                    history_rgb_paths=hist_paths,
                    current_rgb_path=current_path,
                    target_rgb_path=target_path,
                )
            )
    return samples


def choose_samples(
    samples: List[GoalGenSample],
    target_total: int,
    rng: random.Random,
) -> List[GoalGenSample]:
    if target_total <= 0 or len(samples) <= target_total:
        chosen = list(samples)
        rng.shuffle(chosen)
        return chosen

    buckets: Dict[str, List[GoalGenSample]] = defaultdict(list)
    for sample in samples:
        buckets[f"{sample.status}->{sample.subgoal}"].append(sample)

    chosen: List[GoalGenSample] = []
    chosen_ids = set()
    per_bucket = max(1, target_total // max(1, len(buckets)))
    for bucket_samples in buckets.values():
        picked = (
            rng.sample(bucket_samples, per_bucket)
            if len(bucket_samples) > per_bucket
            else list(bucket_samples)
        )
        for sample in picked:
            chosen.append(sample)
            chosen_ids.add(id(sample))

    if len(chosen) < target_total:
        remaining = [s for s in samples if id(s) not in chosen_ids]
        need = target_total - len(chosen)
        chosen.extend(rng.sample(remaining, min(need, len(remaining))))
    elif len(chosen) > target_total:
        chosen = rng.sample(chosen, target_total)

    rng.shuffle(chosen)
    return chosen


def sample_to_json(sample: GoalGenSample) -> dict:
    seq = get_full_sequence(sample.scenario)
    completed_until = seq.index(sample.status) + 1 if sample.status in seq else 1
    return {
        "scenario": sample.scenario,
        "scenario_label": SCENARIO_LABELS.get(sample.scenario, sample.scenario),
        "run_id": sample.run_id,
        "anchor": sample.anchor,
        "status": sample.status,
        "subgoal": sample.subgoal,
        "target_event": sample.subgoal,
        "target_frame": sample.target_frame,
        "history_frames": sample.history_frames,
        "history_rgb_paths": sample.history_rgb_paths,
        "current_rgb_path": sample.current_rgb_path,
        "target_rgb_path": sample.target_rgb_path,
        "memory": {
            "scenario": sample.scenario,
            "scenario_label": SCENARIO_LABELS.get(sample.scenario, sample.scenario),
            "event_sequence": seq,
            "status": sample.status,
            "subgoal": sample.subgoal,
            "completed_events": list(seq[:completed_until]),
        },
    }


def split_train_val(
    samples_by_run: Dict[str, List[dict]],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    run_ids = sorted(samples_by_run.keys())
    rng.shuffle(run_ids)
    num_val = max(1, int(len(run_ids) * val_ratio)) if run_ids else 0
    val_runs = set(run_ids[:num_val])

    train: List[dict] = []
    val: List[dict] = []
    for run_id, samples in samples_by_run.items():
        (val if run_id in val_runs else train).extend(samples)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GoalGen v1 jsonl dataset")
    parser.add_argument("--keyframes", default=DEFAULT_KEYFRAMES)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints" / "goalgen_v1_data"))
    parser.add_argument("--samples-per-scenario", type=int, default=1000,
                        help="0 means keep all valid anchors; default keeps a large balanced subset.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-future-gap", type=int, default=1,
                        help="Require target_frame - anchor >= this many frames.")
    parser.add_argument("--num-frames", type=int, default=RGB_FRAME_COUNT)
    parser.add_argument("--rgb-frame-step", type=int, default=RGB_FRAME_STEP)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keyframes_path = pathlib.Path(args.keyframes)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] keyframes={keyframes_path}")
    with keyframes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])

    timelines_by_scenario: Dict[str, List[RunTimeline]] = defaultdict(list)
    skipped = Counter()
    for run in runs:
        timeline = build_run_timeline(run)
        if timeline is None:
            skipped[run.get("status", "Unknown")] += 1
            continue
        timelines_by_scenario[timeline.scenario].append(timeline)

    if args.dry_run:
        keep_scenarios = sorted(timelines_by_scenario)[:3]
        timelines_by_scenario = {
            scenario: timelines_by_scenario[scenario][:5]
            for scenario in keep_scenarios
        }

    print(
        f"[filter] kept_runs={sum(len(v) for v in timelines_by_scenario.values())} "
        f"skipped={dict(skipped)}"
    )

    samples_by_run: Dict[str, List[dict]] = defaultdict(list)
    stats: Dict[str, dict] = {}
    data_root = pathlib.Path(args.data_root)
    target_per_scenario = 50 if args.dry_run else args.samples_per_scenario

    for scenario, timelines in sorted(timelines_by_scenario.items()):
        candidates: List[GoalGenSample] = []
        for timeline in timelines:
            candidates.extend(
                collect_samples(
                    timeline=timeline,
                    data_root=data_root,
                    frame_stride=max(1, args.frame_stride),
                    min_future_gap=max(1, args.min_future_gap),
                    num_frames=args.num_frames,
                    rgb_frame_step=args.rgb_frame_step,
                )
            )
        chosen = choose_samples(candidates, target_per_scenario, rng)
        by_transition = Counter(f"{s.status}->{s.subgoal}" for s in chosen)
        for sample in chosen:
            samples_by_run[sample.run_id].append(sample_to_json(sample))

        stats[scenario] = {
            "runs": len(timelines),
            "candidates": len(candidates),
            "chosen": len(chosen),
            "chosen_by_transition": dict(sorted(by_transition.items())),
        }
        print(
            f"[scenario] {scenario:42s} runs={len(timelines):4d} "
            f"candidates={len(candidates):7d} chosen={len(chosen):5d}"
        )

    train, val = split_train_val(samples_by_run, args.val_ratio, rng)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "stats.json"

    with train_path.open("w", encoding="utf-8") as f:
        for sample in train:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for sample in val:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "train_size": len(train),
                "val_size": len(val),
                "scenario_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[write] {train_path}")
    print(f"[write] {val_path}")
    print(f"[write] {stats_path}")


if __name__ == "__main__":
    main()
