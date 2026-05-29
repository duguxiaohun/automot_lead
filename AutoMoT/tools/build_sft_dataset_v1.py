"""SFT v1 数据集生成脚本 — 为 Qwen3-VL-4B-Instruct LoRA 微调准备样本。

设计目标见 tools/SFT_V1_PLAN.md。本脚本纯 CPU、不需要 GPU，可以在本地或远程跑。

核心流程：
1. 读 keyframes_all_scenarios.json，过滤 status ∈ {"Completed","Perfect"} 的 run。
2. 对每条 run，根据 initial / middle / final 帧号构造 status_timeline。
3. stratified 采样：推进类（跨 GT 转换帧）+ 各 status 段保持类，按目标配比下采样。
4. 对每个采样帧拼 messages（system + user + assistant），写 train / val jsonl。

注意：assistant 的 STATUS 段是当前帧 GT，user 的 MEMORY 块是上一帧 GT —— 防 leak。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 把 AutoMoT 加入 sys.path，复用 prompt_pipeline 里的 system prompt / memory / 状态机。
# 本文件位于 AutoMoT/tools/，parents[1]=AutoMoT/，parents[2]=automot_lead 仓库根。
_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    SCENARIO_EVENT_SEQUENCES,
    SCENARIO_LABELS,
    build_system_prompt,
    build_user_prompt,
    get_full_sequence,
)


# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

# 上一帧距当前帧的间隔 K。LEAD 数据约 4 Hz，K=4 ≈ 1 秒前。详见 PLAN §1。
DEFAULT_K_FRAMES = 4

# GT 转换帧前后丢弃的 buffer 半径，避免标注噪声。
DEFAULT_BOUNDARY_BUFFER = 2

# 每帧 image 采样的步长 / 数量，需要与 runner 默认值（image_io.load_lead_rgb_clip）一致。
RGB_FRAME_STEP = 1
RGB_FRAME_COUNT = 4

# 训练时只接受这两个 run status。其它（如 "Failed" / "Crashed"）GT 标注不可信。
ACCEPTED_RUN_STATUS = {"Completed", "Perfect"}

# 占位 ANALYSIS。v1 训练时这段 token 会被 swift loss_scale 设权重 0。
PLACEHOLDER_ANALYSIS = "Observations recorded."


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RunTimeline:
    """单条 run 的状态时间轴。"""

    scenario: str
    run_id: str
    total_frames: int
    seconds_per_frame: float
    # 按 [start_frame, end_frame, status] 区间表示，end_frame 含。
    intervals: List[Tuple[int, int, str]]
    # 所有 GT 转换帧（middle[i].frame 和 final.frame）。
    transition_frames: List[int]


@dataclass
class SampleRecord:
    """一条训练样本的中间表示，最终序列化为 jsonl 一行。"""

    scenario: str
    run_id: str
    anchor: int
    prev_anchor: int
    memory_in_status: str
    target_status: str
    is_transition_sample: bool


# ---------------------------------------------------------------------------
# Timeline 构造
# ---------------------------------------------------------------------------

def build_run_timeline(run: dict) -> Optional[RunTimeline]:
    """从 keyframes_all_scenarios.json 的一条 run 构造时间轴。

    返回 None 表示该 run 不可用（status 不在白名单 / 缺字段 / 序列不闭合）。
    """

    if run.get("status") not in ACCEPTED_RUN_STATUS:
        return None

    scenario = run.get("scenario")
    if scenario not in SCENARIO_EVENT_SEQUENCES:
        # keyframes 中可能出现 prompt_pipeline 里没注册的场景（如 42 vs 41 差异）。
        return None

    initial = run.get("initial")
    middle = run.get("middle", [])
    final = run.get("final")
    diag = run.get("diagnostics", {})
    if not initial or not final or len(middle) != 3:
        return None

    total_frames = diag.get("total_frames")
    if total_frames is None:
        return None
    seconds_per_frame = diag.get("seconds_per_frame", 0.2513)

    # 期望事件名顺序与 prompt_pipeline.get_full_sequence 完全一致。
    expected_seq = get_full_sequence(scenario)
    actual_seq = (
        initial["event"],
        middle[0]["event"], middle[1]["event"], middle[2]["event"],
        final["event"],
    )
    if actual_seq != expected_seq:
        # 状态机不一致的 run 不能用，否则 target_subgoal 推导会乱。
        return None

    # 闭区间映射。每段从该事件 frame 开始（含），到下一事件 frame - 1 结束。
    boundaries = [
        (initial["frame"],   middle[0]["frame"] - 1, initial["event"]),
        (middle[0]["frame"], middle[1]["frame"] - 1, middle[0]["event"]),
        (middle[1]["frame"], middle[2]["frame"] - 1, middle[1]["event"]),
        (middle[2]["frame"], final["frame"]   - 1, middle[2]["event"]),
        (final["frame"],     total_frames     - 1, final["event"]),
    ]
    # 检查区间合法。出现 start > end 说明 GT 帧号倒序，弃用。
    for s, e, _ in boundaries:
        if s > e:
            return None

    transition_frames = [
        middle[0]["frame"], middle[1]["frame"], middle[2]["frame"], final["frame"]
    ]

    return RunTimeline(
        scenario=scenario,
        run_id=run["run_id"],
        total_frames=total_frames,
        seconds_per_frame=seconds_per_frame,
        intervals=boundaries,
        transition_frames=transition_frames,
    )


def lookup_status(timeline: RunTimeline, frame: int) -> Optional[str]:
    """查表：frame 落在哪个 status 区间。越界返回 None。"""
    if frame < 0 or frame >= timeline.total_frames:
        return None
    for s, e, status in timeline.intervals:
        if s <= frame <= e:
            return status
    return None


def is_near_transition(timeline: RunTimeline, frame: int, buffer: int) -> bool:
    """判断 frame 是否落在任一 GT 转换帧的 ±buffer 内。"""
    for t in timeline.transition_frames:
        if abs(frame - t) <= buffer:
            return True
    return False


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def collect_candidates(
    timeline: RunTimeline,
    k_frames: int,
    buffer: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """枚举 run 内所有合法候选样本，返回 (保持类, 推进类)。"""

    keep_samples: List[SampleRecord] = []
    advance_samples: List[SampleRecord] = []

    # anchor 最小值取 k_frames + RGB_FRAME_COUNT - 1，保证 prev_anchor >= 0 且
    # image_clip 不越界。
    min_anchor = max(k_frames, RGB_FRAME_COUNT - 1)
    for anchor in range(min_anchor, timeline.total_frames):
        prev_anchor = anchor - k_frames
        if prev_anchor < 0:
            continue

        curr_status = lookup_status(timeline, anchor)
        prev_status = lookup_status(timeline, prev_anchor)
        if curr_status is None or prev_status is None:
            continue

        # 推进类 = 跨过至少一个 GT 转换帧。判定：prev 与 curr 不在同一 status 区间。
        is_transition = (prev_status != curr_status)

        if not is_transition:
            # 保持类样本：避开转换帧 ±buffer。
            if is_near_transition(timeline, anchor, buffer):
                continue
            keep_samples.append(SampleRecord(
                scenario=timeline.scenario,
                run_id=timeline.run_id,
                anchor=anchor,
                prev_anchor=prev_anchor,
                memory_in_status=prev_status,
                target_status=curr_status,
                is_transition_sample=False,
            ))
        else:
            # 推进类样本：anchor 应正好等于某个 GT 转换帧（这样 STATUS 变化最干净）。
            # 不强求 anchor == transition_frame，但要求 prev_anchor 仍在上一段。
            if anchor not in timeline.transition_frames:
                # 跨转换但 anchor 不在转换点上的样本——丢弃以减少 GT 噪声。
                continue
            advance_samples.append(SampleRecord(
                scenario=timeline.scenario,
                run_id=timeline.run_id,
                anchor=anchor,
                prev_anchor=prev_anchor,
                memory_in_status=prev_status,
                target_status=curr_status,
                is_transition_sample=True,
            ))

    return keep_samples, advance_samples


def stratify_scenario(
    keeps: List[SampleRecord],
    advances: List[SampleRecord],
    target_total: int,
    advance_ratio: float,
    rng: random.Random,
) -> List[SampleRecord]:
    """按 advance_ratio 比例下采样到 target_total。

    推进类样本天然稀少，能取多少取多少；保持类按段平衡下采样。
    """

    target_advance = int(target_total * advance_ratio)
    target_keep = target_total - target_advance

    # 推进类全收（或下采样到 target_advance）。
    if len(advances) > target_advance:
        chosen_adv = rng.sample(advances, target_advance)
    else:
        chosen_adv = list(advances)

    # 保持类按 memory_in_status 分桶，平衡采样。
    buckets: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in keeps:
        buckets[s.memory_in_status].append(s)

    chosen_keep: List[SampleRecord] = []
    if buckets:
        per_bucket = max(1, target_keep // len(buckets))
        for status, samples in buckets.items():
            if len(samples) > per_bucket:
                chosen_keep.extend(rng.sample(samples, per_bucket))
            else:
                chosen_keep.extend(samples)
        # 补齐到 target_keep。
        if len(chosen_keep) < target_keep:
            remaining = [s for s in keeps if s not in chosen_keep]
            need = target_keep - len(chosen_keep)
            if remaining:
                chosen_keep.extend(rng.sample(remaining, min(need, len(remaining))))

    return chosen_adv + chosen_keep


# ---------------------------------------------------------------------------
# Message 拼装
# ---------------------------------------------------------------------------

def build_image_paths(
    run_dir_template: str,
    scenario: str,
    run_id: str,
    anchor: int,
) -> List[str]:
    """构造 4 张 RGB 图绝对路径，按 oldest → newest 排序。

    与 image_io.load_lead_rgb_clip 的采样规则保持一致：
    desc = [max(anchor - i*step, 0) for i in range(count)]
    返回时反转为 oldest → newest。
    """

    desc_frames = [max(anchor - i * RGB_FRAME_STEP, 0) for i in range(RGB_FRAME_COUNT)]
    ordered = list(reversed(desc_frames))
    run_dir = run_dir_template.format(scenario=scenario, run_id=run_id)
    return [f"{run_dir}/rgb/{f:04d}.jpg" for f in ordered]


def _next_event_in_seq(scenario: str, status: str) -> str:
    """在场景状态机里查 status 的下一事件。final 自指。"""
    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return "final"
    return seq[idx + 1] if idx + 1 < len(seq) else "final"


def build_messages(
    sample: SampleRecord,
    image_paths: List[str],
) -> Dict:
    """拼 messages 列表 + images，最终成为 jsonl 一行。"""

    scenario = sample.scenario

    # 构造 memory_in：status = prev 帧 GT，subgoal 由状态机推导。
    memory_subgoal = _next_event_in_seq(scenario, sample.memory_in_status)
    seq = get_full_sequence(scenario)
    # completed_events：从 initial 累加到 prev 帧 status。
    try:
        completed_until = seq.index(sample.memory_in_status) + 1
        completed = list(seq[:completed_until])
    except ValueError:
        completed = ["initial"]

    memory_in = DrivingMemory(
        scenario=scenario,
        scenario_label=SCENARIO_LABELS.get(scenario, scenario),
        event_sequence=seq,
        status=sample.memory_in_status,
        subgoal=memory_subgoal,
        completed_events=completed,
    )

    # 与 runner 一致的图片说明文本。
    image_description = (
        f"The {len(image_paths)} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )

    system_content = build_system_prompt()
    user_text = build_user_prompt(memory_in, image_description=image_description)
    # ms-swift 多模态 user content 用 <image> 占位符注入图片。每张图一个 <image>。
    user_content = "".join("<image>" for _ in image_paths) + "\n" + user_text

    target_subgoal = _next_event_in_seq(scenario, sample.target_status)
    assistant_content = (
        f"ANALYSIS: {PLACEHOLDER_ANALYSIS}\n"
        f"STATUS: {sample.target_status}\n"
        f"SUBGOAL: {target_subgoal}"
    )

    return {
        "scenario": sample.scenario,
        "run_id": sample.run_id,
        "anchor": sample.anchor,
        "prev_anchor": sample.prev_anchor,
        "images": image_paths,
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "is_transition_sample": sample.is_transition_sample,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def split_train_val(
    samples_by_run: Dict[str, List[Dict]],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict]]:
    """按 run_id 划分 train / val。防止 frame 级 leak。"""

    run_ids = sorted(samples_by_run.keys())
    rng.shuffle(run_ids)
    num_val = max(1, int(len(run_ids) * val_ratio))
    val_runs = set(run_ids[:num_val])

    train, val = [], []
    for rid, samples in samples_by_run.items():
        target = val if rid in val_runs else train
        target.extend(samples)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser(description="SFT v1 dataset builder")
    parser.add_argument("--keyframes", type=str,
                        default=str(_PROJECT_ROOT / "keyframes_all_scenarios.json"))
    parser.add_argument("--data-root", type=str,
                        default="/data/lead_data/data",
                        help="LEAD 数据根目录。每个 scenario 是子目录。")
    parser.add_argument("--output-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v1_data"))
    parser.add_argument("--samples-per-scenario", type=int, default=200)
    parser.add_argument("--advance-ratio", type=float, default=0.25)
    parser.add_argument("--k-frames", type=int, default=DEFAULT_K_FRAMES)
    parser.add_argument("--boundary-buffer", type=int, default=DEFAULT_BOUNDARY_BUFFER)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--dry-run", action="store_true",
                        help="只处理前 3 个场景的前 5 个 run，快速验证流水线。")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keyframes_path = pathlib.Path(args.keyframes)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] keyframes from {keyframes_path}")
    with open(keyframes_path, "r", encoding="utf-8") as f:
        kf = json.load(f)
    runs = kf.get("runs", [])
    print(f"[load] {len(runs)} total runs in keyframes")

    # 按 scenario 分桶并构造时间轴。
    timelines_by_scenario: Dict[str, List[RunTimeline]] = defaultdict(list)
    n_skipped = Counter()
    for run in runs:
        tl = build_run_timeline(run)
        if tl is None:
            n_skipped[run.get("status", "Unknown")] += 1
            continue
        timelines_by_scenario[tl.scenario].append(tl)
    print(f"[filter] kept {sum(len(v) for v in timelines_by_scenario.values())} runs; "
          f"skipped by status: {dict(n_skipped)}")

    # dry-run 模式只取前 3 场景前 5 run。
    if args.dry_run:
        keep_scenarios = sorted(timelines_by_scenario.keys())[:3]
        timelines_by_scenario = {
            k: timelines_by_scenario[k][:5] for k in keep_scenarios
        }
        print(f"[dry-run] reduced to {len(timelines_by_scenario)} scenarios")

    run_dir_template = args.data_root + "/{scenario}/{run_id}"

    # 对每个场景：收集候选 → stratify → 拼 messages。
    samples_by_run: Dict[str, List[Dict]] = defaultdict(list)
    scenario_stats: Dict[str, Dict[str, int]] = {}

    for scenario, timelines in sorted(timelines_by_scenario.items()):
        all_keeps: List[SampleRecord] = []
        all_advs: List[SampleRecord] = []
        for tl in timelines:
            keeps, advs = collect_candidates(tl, args.k_frames, args.boundary_buffer)
            all_keeps.extend(keeps)
            all_advs.extend(advs)

        target = args.samples_per_scenario if not args.dry_run else 20
        chosen = stratify_scenario(
            all_keeps, all_advs,
            target_total=target,
            advance_ratio=args.advance_ratio,
            rng=rng,
        )

        for s in chosen:
            image_paths = build_image_paths(
                run_dir_template, s.scenario, s.run_id, s.anchor
            )
            sample_dict = build_messages(s, image_paths)
            samples_by_run[s.run_id].append(sample_dict)

        n_adv = sum(1 for s in chosen if s.is_transition_sample)
        scenario_stats[scenario] = {
            "candidates_keep": len(all_keeps),
            "candidates_advance": len(all_advs),
            "chosen_total": len(chosen),
            "chosen_advance": n_adv,
            "chosen_keep": len(chosen) - n_adv,
        }
        print(f"[stratify] {scenario:42s} keep={len(all_keeps):5d} adv={len(all_advs):4d}"
              f" -> chosen={len(chosen)} (adv={n_adv})")

    # 按 run_id 划 train / val。
    train_samples, val_samples = split_train_val(
        samples_by_run, args.val_ratio, rng,
    )
    print(f"[split] train={len(train_samples)}  val={len(val_samples)}")

    # 写盘。
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "stats.json"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "scenario_stats": scenario_stats,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "transition_in_train": sum(s["is_transition_sample"] for s in train_samples),
            "transition_in_val": sum(s["is_transition_sample"] for s in val_samples),
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] {train_path}")
    print(f"[write] {val_path}")
    print(f"[write] {stats_path}")


if __name__ == "__main__":
    main()
