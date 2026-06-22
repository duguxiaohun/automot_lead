"""SFT v3 episode index 构建脚本。

与 v1/v2 不同，v3 **不生成 per-frame 训练样本 jsonl**。
本脚本只产出 episode_index.jsonl，每行对应一个 sub-scenario：

  {
    "run_id": "...",
    "scenario": "...",
    "anchors": [f0, f1, f2, f3, f4],  # [init, sub1, sub2, sub3, end]
    "delta": 7,
    "frame_range": [f1 - delta, f3],
    "gt_scene": "...",
    "gt_event_sequence": ["initial", "e1", "e2", "e3", "final"],
    "run_dir": "/abs/path/to/run",
    "split": "train" | "val"
  }

训练时 DataLoader 每次吐一条 episode；训练 loop 读 episode 后按外循环帧号即时读 RGB。

数据契约：
  - run.status 不在 {"Completed", "Perfect"} 时跳过
  - scenario 不在 SCENARIO_LABELS 中时跳过
  - 其余可用 run 必须严格是一条 3-middle episode：
    initial + middle[3] + final，anchors 严格递增，事件序列与
    prompt_pipeline.get_full_sequence 完全一致；否则直接报错退出

典型用法（从 AutoMoT/ 目录运行）：
  python qwen3vl_local/sft_v3/build_dataset.py \\
    --keyframes lead_data/keyframes_all_scenarios.json \\
    --data-root lead_data \\
    --output-dir checkpoints/sft_v3_data \\
    --val-ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
from qwen3vl_local.sft.build_dataset import (  # noqa: E402
    ACCEPTED_RUN_STATUS,
    RGB_FRAME_COUNT,
    RGB_FRAME_STEP,
)

# ────────────────────────────────────────────────────
# δ 计算
# ────────────────────────────────────────────────────

DELTA_CAP = 10  # 帧数上限
DELTA_NARROW_WARN_THRESHOLD = 1  # 只对 δ=0 的退化窗口打 warning，避免 dry-run 刷屏


def compute_delta(
    anchors: List[int],
    *,
    run_id: str | None = None,
    scenario: str | None = None,
    warn: bool = True,
) -> int:
    """按 PLAN §2.2 公式计算 δ，并对 anchor 间距过窄的 episode 打 warning。

    δ = min(anchor[1]-anchor[0], anchor[2]-anchor[1]) // 2，再封顶 DELTA_CAP。
    C1 修正：不设置下限，允许 δ=0。δ=0 时 Phase A 退化为 anchor[1] 单帧，
    但这是公式本身表达的数据事实，不能静默改成 1 后越过 anchor[1]。只有
    raw_delta==0 时打印 warning，避免 raw_delta=1/2/3 的合法短窗口刷屏。
    """

    d01 = anchors[1] - anchors[0]
    d12 = anchors[2] - anchors[1]
    raw_delta = min(d01, d12) // 2
    delta = min(raw_delta, DELTA_CAP)
    if warn and raw_delta < DELTA_NARROW_WARN_THRESHOLD:
        tag = ""
        if run_id is not None or scenario is not None:
            tag = f" (run={run_id} scenario={scenario})"
        print(
            f"[build_dataset][warn] narrow anchor spacing{tag}: d01={d01}, d12={d12}, "
            f"raw_delta={raw_delta}, capped delta={delta}; Phase A 覆盖区间将仅为 "
            f"±{delta} 帧（δ=0 时只覆盖 anchor[1] 单帧）",
            file=sys.stderr,
        )
    return delta


# ────────────────────────────────────────────────────
# Episode 构建
# ────────────────────────────────────────────────────

def build_episode(
    run: dict,
    run_dir_base: str,
) -> Optional[dict]:
    """从 keyframes_all_scenarios.json 的一条 run 构造 episode 字典。

    返回 None 表示该 run 不可用。
    """

    if run.get("status") not in ACCEPTED_RUN_STATUS:
        return None

    scenario = run.get("scenario")
    if not scenario or scenario not in SCENARIO_LABELS:
        return None

    run_id = str(run.get("run_id", "<missing_run_id>"))
    initial = run.get("initial")
    middle = run.get("middle", [])
    final_kf = run.get("final")
    diag = run.get("diagnostics", {})

    if not initial or not final_kf or len(middle) != 3:
        raise ValueError(
            f"run {run_id} / scenario {scenario} is not a strict 3-middle episode: "
            f"initial={bool(initial)} middle_count={len(middle) if isinstance(middle, list) else 'non-list'} "
            f"final={bool(final_kf)}"
        )

    total_frames = diag.get("total_frames")
    if total_frames is None:
        raise ValueError(f"run {run_id} / scenario {scenario} missing diagnostics.total_frames")

    # 事件序列一致性检查
    expected_seq = tuple(get_full_sequence(scenario))
    actual_seq = (
        initial["event"],
        middle[0]["event"], middle[1]["event"], middle[2]["event"],
        final_kf["event"],
    )
    if actual_seq != expected_seq:
        raise ValueError(
            f"run {run_id} / scenario {scenario} event sequence mismatch: "
            f"actual={actual_seq}, expected={expected_seq}"
        )

    # 提取 5 个 anchor 帧号
    anchors = [
        int(initial["frame"]),
        int(middle[0]["frame"]),
        int(middle[1]["frame"]),
        int(middle[2]["frame"]),
        int(final_kf["frame"]),
    ]

    # 严格递增检查
    for i in range(len(anchors) - 1):
        if anchors[i] >= anchors[i + 1]:
            raise ValueError(f"run {run_id} / scenario {scenario} anchors are not strictly increasing: {anchors}")

    # 训练窗口必须能容纳至少 1 帧
    delta = compute_delta(anchors, run_id=run_id, scenario=scenario)
    frame_start = anchors[1] - delta
    frame_end = anchors[3]
    if frame_start > frame_end:
        raise ValueError(
            f"run {run_id} / scenario {scenario} invalid training frame range: "
            f"start={frame_start}, end={frame_end}, anchors={anchors}, delta={delta}"
        )

    # 起始帧要留出历史 RGB 窗口
    min_start = RGB_FRAME_COUNT - 1
    if frame_start < min_start:
        frame_start = min_start

    run_dir = pathlib.Path(run_dir_base.format(scenario=scenario, run_id=run_id))

    return {
        "run_id": run_id,
        "scenario": scenario,
        "anchors": anchors,
        "delta": delta,
        "frame_range": [frame_start, frame_end],
        "gt_scene": scenario,
        "gt_event_sequence": list(expected_seq),
        "run_dir": str(run_dir),
        "total_frames": int(total_frames),
    }


def load_keyframe_runs(path: pathlib.Path) -> List[dict]:
    """Load keyframe runs from either the current metadata dict or legacy list.

    The checked-in/reference ``keyframes_all_scenarios.json`` has a top-level
    object with a ``runs`` list. Older intermediate dumps were already a list,
    so keep that form as a compatibility fallback.
    """

    with open(path, "r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    if isinstance(payload, list):
        runs = payload
    elif isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        runs = payload["runs"]
    else:
        raise ValueError(
            f"Unsupported keyframes schema in {path}: expected list or object with a runs list"
        )
    bad = [type(x).__name__ for x in runs[:5] if not isinstance(x, dict)]
    if bad:
        raise ValueError(f"keyframes runs must be objects; first bad entry types: {bad}")
    return runs


# ────────────────────────────────────────────────────
# train/val 划分
# ────────────────────────────────────────────────────

def split_train_val(
    episodes: List[dict],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    """按 run_id 粒度划分 train/val，避免相邻帧泄漏。

    v3 的训练样本是一整段时间序列，如果同一个 run 的相邻帧分别进入 train/val，
    eval 会高估 memory 恢复能力。所以这里先按 run_id 聚合，再整体划分。
    """

    runs_by_id: Dict[str, List[dict]] = defaultdict(list)
    for ep in episodes:
        runs_by_id[ep["run_id"]].append(ep)

    run_ids = sorted(runs_by_id.keys())
    rng.shuffle(run_ids)
    num_val = max(1, int(len(run_ids) * val_ratio))
    val_runs = set(run_ids[:num_val])

    train, val = [], []
    for rid, eps in runs_by_id.items():
        if rid in val_runs:
            val.extend(eps)
        else:
            train.extend(eps)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析 episode index 构建参数。

    注意本脚本只写 `train.jsonl` / `val.jsonl` 的 episode 元数据，不写 per-frame
    样本，也不读取 RGB 图片；真正的图像懒加载发生在 `train.py` 的外循环里。
    """

    p = argparse.ArgumentParser(description="SFT v3 episode index builder")
    p.add_argument("--keyframes", type=str, default="lead_data/keyframes_all_scenarios.json",
                   help="keyframes_all_scenarios.json 路径")
    p.add_argument("--data-root", type=str, default="lead_data",
                   help="LEAD 数据根目录，run_dir 以此为 base")
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v3_data",
                   help="输出目录，写 train.jsonl / val.jsonl / stats.json")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="按 run_id 划分 val 集比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-episode-length", type=int, default=5,
                   help="训练窗口帧数下限（frame_range 长度），过短的 episode 过滤掉")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印统计信息，不写文件")
    return p.parse_args()


def main() -> None:
    """构建 SFT v3 episode index 主入口。

    流程：
    1. 读取 keyframes_all_scenarios.json；
    2. 对每个 run 做严格 3-middle episode 契约检查；
    3. 计算 delta 与训练帧窗口 `[f1-delta, f3]`；
    4. 按 run_id 划分 train/val；
    5. 写 jsonl 和 stats。
    """

    args = parse_args()

    kf_path = pathlib.Path(args.keyframes)
    if not kf_path.exists():
        raise FileNotFoundError(f"keyframes 文件不存在: {kf_path}")

    all_runs = load_keyframe_runs(kf_path)

    # run_dir 模板：data_root/<scenario>/<run_id>
    run_dir_template = str(pathlib.Path(args.data_root) / "{scenario}" / "{run_id}")

    episodes: List[dict] = []
    counters: Counter = Counter()
    skip_reasons: Counter = Counter()

    for run in all_runs:
        ep = build_episode(run, run_dir_template)
        if ep is None:
            skip_reasons["filtered"] += 1
            continue
        # 过滤训练窗口过短的 episode
        frame_len = ep["frame_range"][1] - ep["frame_range"][0] + 1
        if frame_len < args.min_episode_length:
            skip_reasons["too_short"] += 1
            continue
        episodes.append(ep)
        counters[ep["scenario"]] += 1

    print(f"[build_dataset] 有效 episode 总数: {len(episodes)}")
    print(f"[build_dataset] 跳过: {dict(skip_reasons)}")
    print("[build_dataset] 按场景分布:")
    for sc, cnt in sorted(counters.items()):
        print(f"  {sc}: {cnt}")

    if args.dry_run:
        print("[build_dataset] dry-run 模式，不写文件。")
        return

    rng = random.Random(args.seed)
    train_eps, val_eps = split_train_val(episodes, args.val_ratio, rng)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 train.jsonl
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for ep in train_eps:
            ep_out = dict(ep, split="train")
            f.write(json.dumps(ep_out, ensure_ascii=False) + "\n")
    print(f"[build_dataset] train: {len(train_eps)} episodes -> {train_path}")

    # 写 val.jsonl
    val_path = out_dir / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for ep in val_eps:
            ep_out = dict(ep, split="val")
            f.write(json.dumps(ep_out, ensure_ascii=False) + "\n")
    print(f"[build_dataset] val:   {len(val_eps)} episodes -> {val_path}")

    # 写 stats.json
    stats = {
        "total": len(episodes),
        "train": len(train_eps),
        "val": len(val_eps),
        "by_scenario": dict(counters),
        "skip_reasons": dict(skip_reasons),
        "delta_cap": DELTA_CAP,
        "rgb_frame_count": RGB_FRAME_COUNT,
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[build_dataset] stats -> {stats_path}")


if __name__ == "__main__":
    main()
