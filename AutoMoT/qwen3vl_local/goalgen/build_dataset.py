"""从关键帧时间线构建 GoalGen v1/v2 的 jsonl 数据集。

本脚本应在远端机器的 ``AutoMoT/`` 目录下运行，例如：

python qwen3vl_local/goalgen/build_dataset.py \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --output-dir checkpoints/goalgen_v1_data

构建逻辑沿用 SFT 的事件时间线思路，但监督目标不同：
对每个锚点帧，STATUS 是该帧的真值状态，SUBGOAL 是场景事件链里的下一个事件；
图像监督目标则是这个 SUBGOAL 开始发生时的未来关键帧。
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
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402


ACCEPTED_RUN_STATUS = {"Completed", "Perfect"}
DEFAULT_KEYFRAMES = "lead_data/keyframes_all_scenarios.json"
DEFAULT_DATA_ROOT = "lead_data"
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


def is_middle_transition_pair(scenario: str, status: str, subgoal: str) -> bool:
    """v2 只允许 full sequence 里的 middle[0]->middle[1] 与 middle[1]->middle[2]。"""

    seq = get_full_sequence(scenario)
    return any(
        idx + 1 < len(seq) and status == seq[idx] and subgoal == seq[idx + 1]
        for idx in (1, 2)
    )


def build_run_timeline(run: dict) -> Optional[RunTimeline]:
    # 只接收 Completed / Perfect 的 run：失败 run 的 event 链可能在中途断掉，
    # 用这种 run 的"过去 → 子目标"对训练是错的（agent 实际没到达 SUBGOAL）。
    if run.get("status") not in ACCEPTED_RUN_STATUS:
        return None

    scenario = run.get("scenario")
    if scenario not in SCENARIO_EVENT_SEQUENCES:
        return None

    initial = run.get("initial")
    middle = run.get("middle", [])
    final = run.get("final")
    total_frames = run.get("diagnostics", {}).get("total_frames")
    # LEAD 约定每个场景必有 1 个 initial + 3 个 middle + 1 个 final = 5 个事件；
    # middle 数量不等于 3 说明这条 run 已经偏离标准链，跳过比硬补更安全。
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
    # 严格全等检查：事件**顺序**必须与场景模板逐字一致。LEAD 偶尔会出现"中途事件名
    # 被替换或漏掉"的脏 run，名字对得上但顺序错位也算坏数据，扔掉。
    if actual_seq != expected_seq:
        return None

    # 把"每个 status 占据的连续帧区间"摊平。后一个事件起点 - 1 当作前一个 status 区间末尾，
    # 这样区间间两两不重叠也无缝隙；final 区间一直延伸到 total_frames - 1。
    boundaries = [
        (initial["frame"], middle[0]["frame"] - 1, initial["event"]),
        (middle[0]["frame"], middle[1]["frame"] - 1, middle[0]["event"]),
        (middle[1]["frame"], middle[2]["frame"] - 1, middle[1]["event"]),
        (middle[2]["frame"], final["frame"] - 1, middle[2]["event"]),
        (final["frame"], total_frames - 1, final["event"]),
    ]
    # 任何一个 status 区间长度 <= 0 都说明事件帧顺序乱了（例如 middle[0].frame >= middle[1].frame）；
    # 这是 LEAD 时间链异常，整条 run 都不能用。
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


def iter_status_ranges(
    timeline: RunTimeline,
    mode: str = "v1",
) -> Iterable[Tuple[int, int, str]]:
    for start, end, status in timeline.intervals:
        if status == "final":
            continue
        subgoal = next_event_in_sequence(timeline.scenario, status)
        if not subgoal:
            continue
        # v2 模式只保留三个 middle 子目标之间的两段转换：
        # middle[0]→middle[1]、middle[1]→middle[2]。
        # 排除 status == "initial"（起手 transition）和 subgoal == "final"（收尾 transition）；
        # 这两类样本被认为在生成训练中信号弱（initial 视觉上没有任何"任务进度"信息、
        # final 子目标视觉上常常是"减速/停车"，对未来关键帧生成几乎不携带方向信息）。
        if mode == "v2" and (status == "initial" or subgoal == "final"):
            continue
        target_frame = timeline.event_frames.get(subgoal)
        if target_frame is None:
            continue
        # 合法锚点必须严格早于子目标关键帧；等于目标帧会退化成"预测当前图"。
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
    mode: str = "v1",
) -> List[GoalGenSample]:
    route_dir = data_root / timeline.scenario / timeline.run_id
    samples: List[GoalGenSample] = []

    # 每个 route 只 glob 一次 rgb/，后续 resolve_rgb_path 走内存查；NFS 上能省
    # 几个数量级的 stat 调用，对 7000 routes 的 dataset build 几乎是必须的优化。
    rgb_cache = _load_rgb_directory(route_dir / "rgb")

    for start, end, status in iter_status_ranges(timeline, mode=mode):
        subgoal = next_event_in_sequence(timeline.scenario, status)
        if subgoal is None:
            continue
        target_frame = timeline.event_frames[subgoal]

        # min_anchor 两边夹：
        # - 不早于 status 区间 start（anchor 必须落在当前 status 内，否则 STATUS 真值就错了）；
        # - 至少有 (num_frames-1) * rgb_frame_step 帧的过去，否则 history_frames 会被 max(., 0) 截断
        #   出现重复帧，下游 VAE 编出来的 z_history 退化成"贴图叠加"，破坏时序信息。
        min_anchor = max(start, (num_frames - 1) * rgb_frame_step)
        # max_anchor：
        # - 不超过 status 区间 end（同上：anchor 不能跨到下一个 status）；
        # - 距离 target_frame 至少 min_future_gap 帧（避免目标帧 = 当前帧的退化样本，
        #   那种样本对生成模型完全没价值，会把模型推向"恒等映射"陷阱）。
        max_anchor = min(end, target_frame - min_future_gap)
        if min_anchor > max_anchor:
            # status 区间太短或者太靠近 target_frame 时整段不可用；跳过该 status 而不是补救。
            continue

        for anchor in range(min_anchor, max_anchor + 1, frame_stride):
            frames = history_frames(anchor, num_frames, rgb_frame_step)
            # 复用 route 级 rgb_cache 字典查文件，避免每 anchor × 历史帧数 次 stat 调用；
            # 7000 run × 几十 anchor × 4 历史帧 × 2 (history + target) 是百万次量级，必须缓存。
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
    # target_total <= 0 表示"不限量、全部要"（CLI `--samples-per-scenario 0`）；
    # 候选数本身不足 target_total 时也不上采样，避免引入重复样本污染训练分布。
    if target_total <= 0 or len(samples) <= target_total:
        chosen = list(samples)
        rng.shuffle(chosen)
        return chosen

    # 按 (status, subgoal) 转移对分桶，做分层抽样。每个 status->subgoal 是一种"状态变换类型"，
    # 不分桶直接随机会让某些少见 transition（例如 rare scenario 的 final 段）数量过低，
    # 导致 DiT 在那类转移上欠拟合。
    buckets: Dict[str, List[GoalGenSample]] = defaultdict(list)
    for sample in samples:
        buckets[f"{sample.status}->{sample.subgoal}"].append(sample)

    chosen: List[GoalGenSample] = []
    # 用 id() 作为去重 key 而不是把 GoalGenSample 加进 set：dataclass 默认 unhashable，
    # 且我们只需要"对象身份"等价而非"内容"等价，id() 在 list 元素未释放时唯一。
    chosen_ids = set()
    per_bucket = max(1, target_total // max(1, len(buckets)))
    for bucket_samples in buckets.values():
        # 单桶足额就采 per_bucket；不足就全收。比例不均时这里会"先把所有少数桶吃完，
        # 大桶按 per_bucket 截断"，下面 if 分支再补到 target_total。
        picked = (
            rng.sample(bucket_samples, per_bucket)
            if len(bucket_samples) > per_bucket
            else list(bucket_samples)
        )
        for sample in picked:
            chosen.append(sample)
            chosen_ids.add(id(sample))

    if len(chosen) < target_total:
        # 凑不够 target_total（少数桶太少）→ 从所有未选样本里随机补；
        # 此时已无法保证桶均匀，但样本总量优先级更高。
        remaining = [s for s in samples if id(s) not in chosen_ids]
        need = target_total - len(chosen)
        chosen.extend(rng.sample(remaining, min(need, len(remaining))))
    elif len(chosen) > target_total:
        # per_bucket 向下取整 + 大桶塞满，可能轻微超 target；这里随机砍掉多余的。
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
    # 按 run_id 切分，不按样本切：同一 run 的相邻 anchor 共享大量视觉上下文，
    # 如果同一 run 的样本既出现在 train 又出现在 val，验证集泄漏会让 loss 假性下降。
    # sorted 给定 run_ids 顺序是确定的，再加 rng.shuffle 让切分本身可复现。
    run_ids = sorted(samples_by_run.keys())
    rng.shuffle(run_ids)
    # max(1, ...) 兜底：val_ratio 太小 + run 数太少时仍保证至少 1 个 val run，
    # 避免空 val.jsonl 让下游 eval 脚本崩溃。
    num_val = max(1, int(len(run_ids) * val_ratio)) if run_ids else 0
    val_runs = set(run_ids[:num_val])

    train: List[dict] = []
    val: List[dict] = []
    for run_id, samples in samples_by_run.items():
        (val if run_id in val_runs else train).extend(samples)
    # 再 shuffle 一次让 train / val 内部样本顺序与原 run 顺序无关；
    # 训练器自己每个 epoch 还会重新洗牌，这里 shuffle 主要是为了让"按行读取的 dry run"
    # 也能看到混合分布而不是连续同 run。
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 GoalGen v1/v2 的 jsonl 数据集")
    parser.add_argument(
        "--mode",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "v1：保留全部 4 类 status→subgoal 转换（含 initial→middle[0] 与 middle[2]→final）；"
            "v2：只保留三个 middle 子目标之间的两段转换 middle[0]→middle[1] / middle[1]→middle[2]，"
            "即排除当前状态是 initial、以及子目标是 final 的样本。"
        ),
    )
    parser.add_argument("--keyframes", default=DEFAULT_KEYFRAMES)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "默认 None：根据 --mode 自动落在 checkpoints/goalgen_v1_data 或 goalgen_v2_data；"
            "显式指定时尊重用户路径。"
        ),
    )
    parser.add_argument("--samples-per-scenario", type=int, default=0,
                        help="0 表示保留所有合法锚点；默认保留较大的均衡子集。")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-future-gap", type=int, default=1,
                        help="要求 target_frame - anchor 至少达到这个帧数。")
    parser.add_argument("--num-frames", type=int, default=RGB_FRAME_COUNT)
    parser.add_argument("--rgb-frame-step", type=int, default=RGB_FRAME_STEP)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keyframes_path = pathlib.Path(args.keyframes)
    # --output-dir 没显式给 → 按 mode 默认走 goalgen_v1_data / goalgen_v2_data，
    # 避免 v2 误覆盖 v1 已经构建好的 jsonl。
    if args.output_dir is None:
        default_name = "goalgen_v2_data" if args.mode == "v2" else "goalgen_v1_data"
        args.output_dir = str(_AUTOMOT_ROOT / "checkpoints" / default_name)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mode] {args.mode}（输出目录={output_dir}）")

    print(f"[load] 关键帧={keyframes_path}")
    with keyframes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])

    timelines_by_scenario: Dict[str, List[RunTimeline]] = defaultdict(list)
    skipped = Counter()
    data_root = pathlib.Path(args.data_root)
    for run in runs:
        timeline = build_run_timeline(run)
        if timeline is None:
            skipped[run.get("status", "Unknown")] += 1
            continue
        route_dir = data_root / timeline.scenario / timeline.run_id
        should_exclude, _abnormal_info = is_abnormal_lead_route(route_dir, timeline.scenario)
        if should_exclude:
            skipped["abnormal_duration_over_90s"] += 1
            continue
        timelines_by_scenario[timeline.scenario].append(timeline)

    if args.dry_run:
        keep_scenarios = sorted(timelines_by_scenario)[:3]
        timelines_by_scenario = {
            scenario: timelines_by_scenario[scenario][:5]
            for scenario in keep_scenarios
        }

    print(
        f"[filter] 保留 run 数={sum(len(v) for v in timelines_by_scenario.values())} "
        f"跳过={dict(skipped)}"
    )

    samples_by_run: Dict[str, List[dict]] = defaultdict(list)
    stats: Dict[str, dict] = {}
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
                    mode=args.mode,
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
            f"[scenario] {scenario:42s} run 数={len(timelines):4d} "
            f"候选={len(candidates):7d} 选中={len(chosen):5d}"
        )

    train, val = split_train_val(samples_by_run, args.val_ratio, rng)
    if args.mode == "v2":
        invalid = [
            (
                sample.get("scenario", ""),
                sample.get("status", ""),
                sample.get("subgoal", ""),
                sample.get("run_id", ""),
                sample.get("anchor", ""),
            )
            for sample in (train + val)
            if not is_middle_transition_pair(
                sample.get("scenario", ""),
                sample.get("status", ""),
                sample.get("subgoal", ""),
            )
        ]
        if invalid:
            preview = ", ".join(
                f"{scenario}:{status}->{subgoal}@{run_id}:{anchor}"
                for scenario, status, subgoal, run_id, anchor in invalid[:5]
            )
            raise RuntimeError(
                "GoalGen v2 数据只能包含 middle 子目标之间的转换，"
                f"但构建结果含 initial/final 或非法 pair：{preview}"
            )
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
                "abnormal_duration_rule": "exclude duration_s > 90 unless scenario is BlockedIntersection or ControlLoss",
                "skipped_runs": dict(skipped),
                "train_size": len(train),
                "val_size": len(val),
                "scenario_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[write] 写入训练集 {train_path}")
    print(f"[write] 写入验证集 {val_path}")
    print(f"[write] 写入统计 {stats_path}")


if __name__ == "__main__":
    main()
