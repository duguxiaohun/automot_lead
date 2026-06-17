"""构建 SFT v2 串行选择题 jsonl 数据。

本脚本复用旧 ``qwen3vl_local/sft`` 的 LEAD keyframe 时间线、keep/advance
候选采样逻辑，但把监督拆成两个显式阶段：

1. ``images + scene prompt -> SCENE``
2. ``selected-scene prompt -> STATUS/SUBGOAL``

注意这里完全不再生成 ANALYSIS、teacher cache 或 pending placeholder。每一行数据
都保存 ``stage_messages.scene`` 和 ``stage_messages.status``，训练时再拼成一条
multi-turn 对话并只对值 token 计算 loss。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft.build_dataset import (  # noqa: E402
    DEFAULT_BOUNDARY_BUFFER,
    DEFAULT_K_FRAMES,
    RGB_FRAME_COUNT,
    build_image_paths,
    build_run_timeline,
    collect_candidates,
    stratify_scenario,
)
from qwen3vl_local.sft_v2.prompts import (  # noqa: E402
    DATASET_VERSION,
    SCENE_SYSTEM_PROMPT,
    SCENARIO_EVENT_SEQUENCES,
    STATUS_SYSTEM_PROMPT,
    build_scene_user_prompt,
    build_status_user_prompt,
    format_scene_assistant,
    format_status_assistant,
    get_full_sequence,
    next_event,
)


def build_messages(sample, image_paths: List[str]) -> Dict:
    """把旧 SFT 的 SampleRecord 转成 SFT v2 两阶段 row。

    第一阶段携带图像占位符和场景选择 prompt；第二阶段作为后续 text-only user turn，
    默认使用真实场景的 EVENT_SEQUENCE。wrong-scene 增强会在 split 之后只改 train row
    的第二阶段 prompt/target。
    """

    # previous hint 来自 anchor-K 的状态，用来维持“不要无证据提前推进状态”的记忆语义。
    previous_subgoal = next_event(sample.scenario, sample.memory_in_status)
    # target_subgoal 是当前 GT status 在同一状态机中的下一阶段。
    target_subgoal = next_event(sample.scenario, sample.target_status)
    scene_user_text = build_scene_user_prompt(
        image_count=len(image_paths),
    )
    status_user_text = build_status_user_prompt(
        image_count=len(image_paths),
        selected_scene=sample.scenario,
        previous_status=sample.memory_in_status,
        previous_subgoal=previous_subgoal,
    )
    # jsonl 里的 <image> 只是便于肉眼检查；训练/eval 会把 PIL 图片结构化传给 processor。
    image_prefix = "".join("<image>" for _ in image_paths) + "\n"
    scene_assistant = format_scene_assistant(sample.scenario)
    status_assistant = format_status_assistant(sample.target_status, target_subgoal)
    return {
        "scenario": sample.scenario,
        "run_id": sample.run_id,
        "anchor": sample.anchor,
        "prev_anchor": sample.prev_anchor,
        "images": image_paths,
        # `messages` mirrors the first stage for lightweight inspection.
        # 训练和 eval 使用下面显式的两阶段 messages；messages 字段只保留第一阶段便于检查。
        "messages": [
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            {"role": "user", "content": image_prefix + scene_user_text},
            {"role": "assistant", "content": scene_assistant},
        ],
        "stage_messages": {
            "scene": [
                {"role": "system", "content": SCENE_SYSTEM_PROMPT},
                {"role": "user", "content": image_prefix + scene_user_text},
                {"role": "assistant", "content": scene_assistant},
            ],
            "status": [
                {"role": "system", "content": STATUS_SYSTEM_PROMPT},
                {"role": "user", "content": status_user_text},
                {"role": "assistant", "content": status_assistant},
            ],
        },
        "is_transition_sample": sample.is_transition_sample,
        "dataset_version": DATASET_VERSION,
        "choice_meta": {
            "target_scene": sample.scenario,
            "selected_scene": sample.scenario,
            "status_scene_matches_target": True,
            "target_status": sample.target_status,
            "target_subgoal": target_subgoal,
            "memory_in_status": sample.memory_in_status,
            "memory_in_subgoal": previous_subgoal,
            "transition": "advance" if sample.is_transition_sample else "keep",
        },
    }


def split_train_val(
    samples_by_run: Dict[str, List[Dict]],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict]]:
    """按 run_id 划分 train/val，避免相邻帧泄漏。

    同一条 CARLA run 内的样本通常时间上高度相邻，如果随机按帧切分，val 会看到和 train
    几乎连续的图像。这里按 run 粒度切分更接近真实泛化评估。
    """

    run_ids = sorted(samples_by_run.keys())
    rng.shuffle(run_ids)
    num_val = max(1, int(len(run_ids) * val_ratio))
    val_runs = set(run_ids[:num_val])
    train, val = [], []
    for rid, rows in samples_by_run.items():
        (val if rid in val_runs else train).extend(rows)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def apply_wrong_scene_augmentation(rows: List[Dict], ratio: float, rng: random.Random) -> int:
    """对 train row 做 wrong-scene 第二阶段增强。

    增强目标是模拟 eval 中“第一阶段预测了一个合法但错误的 scene，第二阶段仍要沿该
    predicted scene 继续判断”的情况。关键点：

    - 只改第二阶段 selected scene，不改第一阶段 GT SCENE 监督。
    - previous hint 和 STATUS/SUBGOAL target 都按“状态机相位”映射到 selected scene。
    - 因此第二阶段 prompt 和 label 始终内部一致，不会训练模型输出非法 event。
    """

    if ratio <= 0:
        return 0
    scenarios = sorted(SCENARIO_EVENT_SEQUENCES)
    n_aug = 0
    for row in rows:
        if rng.random() >= ratio:
            continue
        target_scene = row["choice_meta"]["target_scene"]
        choices = [s for s in scenarios if s != target_scene]
        if not choices:
            continue
        selected_scene = rng.choice(choices)
        meta = row["choice_meta"]

        # 用真实场景中的 target_status 找到当前处于 initial/middle/final 的哪个相位，
        # 再投影到 wrong selected scene 的同一相位。这样模型学到的是“按已选场景状态机
        # 继续判断”，而不是背真实场景的 event 名。
        source_seq = list(get_full_sequence(target_scene))
        target_seq = list(get_full_sequence(selected_scene))
        try:
            phase_idx = source_seq.index(meta["target_status"])
        except ValueError:
            phase_idx = 0
        phase_idx = min(max(phase_idx, 0), len(target_seq) - 1)
        selected_status = target_seq[phase_idx]
        selected_subgoal = next_event(selected_scene, selected_status)

        # previous hint 也必须做同相位映射，否则 prompt 中的 hint 会引用真实场景的
        # event，和 selected scene EVENT_SEQUENCE 冲突。
        try:
            memory_phase_idx = source_seq.index(meta["memory_in_status"])
        except ValueError:
            memory_phase_idx = phase_idx
        memory_phase_idx = min(max(memory_phase_idx, 0), len(target_seq) - 1)
        selected_memory_status = target_seq[memory_phase_idx]
        selected_memory_subgoal = next_event(selected_scene, selected_memory_status)

        # 重写第二阶段 user prompt 和 assistant target。第一阶段 scene target 保持 GT，
        # 因为增强只用来训练第二阶段在“给定某个 selected scene”后的选择行为。
        status_user_text = build_status_user_prompt(
            image_count=len(row.get("images", [])),
            selected_scene=selected_scene,
            previous_status=selected_memory_status,
            previous_subgoal=selected_memory_subgoal,
        )
        row["stage_messages"]["status"][1]["content"] = status_user_text
        row["stage_messages"]["status"][2]["content"] = format_status_assistant(selected_status, selected_subgoal)
        meta["selected_scene"] = selected_scene
        meta["status_scene_matches_target"] = False
        meta["selected_scene_phase_index"] = phase_idx
        meta["selected_scene_status_target"] = selected_status
        meta["selected_scene_subgoal_target"] = selected_subgoal
        meta["selected_scene_memory_phase_index"] = memory_phase_idx
        meta["selected_scene_memory_status"] = selected_memory_status
        meta["selected_scene_memory_subgoal"] = selected_memory_subgoal
        meta["wrong_scene_augmented"] = True
        n_aug += 1
    return n_aug


def main() -> None:
    """命令行入口：读取 keyframes，采样候选，写出 train/val jsonl。"""

    parser = argparse.ArgumentParser(description="Build SFT v2 serial-choice data")
    parser.add_argument("--keyframes", type=str, default="lead_data/keyframes_all_scenarios.json")
    parser.add_argument("--data-root", type=str, default="lead_data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft_v2_data")
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=0,
        help="<=0 means keep all valid candidates per scenario (default).",
    )
    parser.add_argument("--advance-ratio", type=float, default=0.35)
    parser.add_argument("--k-frames", type=int, default=DEFAULT_K_FRAMES)
    parser.add_argument("--boundary-buffer", type=int, default=DEFAULT_BOUNDARY_BUFFER)
    parser.add_argument(
        "--wrong-scene-ratio",
        type=float,
        default=0.15,
        help="Fraction of train rows whose stage-2 selected scene is intentionally replaced.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.keyframes, "r", encoding="utf-8") as f:
        keyframes = json.load(f)
    runs = keyframes.get("runs", [])
    timelines_by_scenario = defaultdict(list)
    skipped = Counter()
    for run in runs:
        timeline = build_run_timeline(run)
        if timeline is None:
            # 只保留可解析且状态合法的 run；跳过原因按原 run status 统计，方便检查数据质量。
            skipped[run.get("status", "Unknown")] += 1
            continue
        timelines_by_scenario[timeline.scenario].append(timeline)
    print(f"[load] runs={len(runs)} kept={sum(len(v) for v in timelines_by_scenario.values())} skipped={dict(skipped)}")

    if args.dry_run:
        scenarios = sorted(timelines_by_scenario.keys())[:3]
        timelines_by_scenario = {s: timelines_by_scenario[s][:5] for s in scenarios}
        print(f"[dry-run] scenarios={scenarios}")

    run_dir_template = args.data_root + "/{scenario}/{run_id}"
    samples_by_run: Dict[str, List[Dict]] = defaultdict(list)
    scenario_stats: Dict[str, Dict[str, int]] = {}

    for scenario, timelines in sorted(timelines_by_scenario.items()):
        keeps, advances = [], []
        for timeline in timelines:
            # collect_candidates 复用旧 SFT 逻辑：keep 样本表示状态保持，advance 样本表示
            # 靠近状态边界并进入下一阶段。
            k, a = collect_candidates(timeline, args.k_frames, args.boundary_buffer)
            keeps.extend(k)
            advances.extend(a)
        if args.dry_run:
            chosen = stratify_scenario(
                keeps,
                advances,
                target_total=20,
                advance_ratio=args.advance_ratio,
                rng=rng,
            )
        elif args.samples_per_scenario <= 0:
            # 用户当前默认希望全量采样；<=0 表示不做 per-scenario 下采样。
            chosen = list(advances) + list(keeps)
            rng.shuffle(chosen)
        else:
            chosen = stratify_scenario(
                keeps,
                advances,
                target_total=args.samples_per_scenario,
                advance_ratio=args.advance_ratio,
                rng=rng,
            )
        for sample in chosen:
            image_paths = build_image_paths(run_dir_template, sample.scenario, sample.run_id, sample.anchor)
            row = build_messages(sample, image_paths)
            samples_by_run[sample.run_id].append(row)
        n_adv = sum(1 for s in chosen if s.is_transition_sample)
        scenario_stats[scenario] = {
            "candidates_keep": len(keeps),
            "candidates_advance": len(advances),
            "chosen_total": len(chosen),
            "chosen_advance": n_adv,
            "chosen_keep": len(chosen) - n_adv,
        }
        print(f"[sample] {scenario:42s} keep={len(keeps):5d} adv={len(advances):4d} -> chosen={len(chosen)}")

    train_rows, val_rows = split_train_val(samples_by_run, args.val_ratio, rng)
    # wrong-scene 增强只施加到 train，val 保持真实串行评估分布。
    wrong_scene_augmented = apply_wrong_scene_augmentation(train_rows, args.wrong_scene_ratio, rng)
    if wrong_scene_augmented:
        print(f"[augment] wrong_scene train rows={wrong_scene_augmented}/{len(train_rows)}")
    for name, rows in (("train", train_rows), ("val", val_rows)):
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {path} rows={len(rows)}")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_version": DATASET_VERSION,
            "config": vars(args),
            "rgb_frame_count": RGB_FRAME_COUNT,
            "scenario_stats": scenario_stats,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "wrong_scene_augmented_train": wrong_scene_augmented,
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
