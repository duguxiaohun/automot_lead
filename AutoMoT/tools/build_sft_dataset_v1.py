"""SFT v1 数据集生成脚本 — 为 Qwen3-VL-4B-Instruct LoRA 微调准备样本。

设计目标见 tools/SFT_V1_PLAN.md。本脚本纯 CPU、不需要 GPU，可以在本地或远程跑。

核心流程：
1. 读 keyframes_all_scenarios.json，过滤 status ∈ {"Completed","Perfect"} 的 run。
2. 对每条 run，根据 initial / middle / final 帧号构造 status_timeline。
3. stratified 采样：推进类（跨 GT 转换帧）+ 各 status 段保持类，按目标配比下采样。
4. 对每个采样帧拼 messages（system + user + assistant），写 train / val jsonl。

注意：assistant 的 STATUS 段是当前帧 GT，user 的 MEMORY 块是上一帧 GT —— 防 leak。

典型用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：

```bash
# 远程真实数据环境：生成完整训练 / 验证 jsonl
python tools/build_sft_dataset_v1.py \
  --keyframes /datashare/IOL4SGH/data/data/keyframes_all_scenarios.json \
  --data-root /data/lead_data/data \
  --output-dir checkpoints/sft_v1_data

# 本地或远程快速检查：只取少量场景和 run，验证 jsonl schema 是否能生成
python tools/build_sft_dataset_v1.py --dry-run --output-dir /tmp/sft_v1_dry
```

输出文件：
- train.jsonl：ms-swift 的训练集，每行一个多模态 chat 样本。
- val.jsonl：按 run_id 切开的验证集，避免同一 route 的相邻帧同时出现在 train/val。
- stats.json：每个场景保留/推进样本数量，排查类别不均衡时先看它。

重要边界：
- 本脚本只生成图片路径，不读取图片内容；真正图片读取发生在 ms-swift 训练侧或 eval_sft_v1.py。
- 这里的 `images` 路径必须和 runner 看到的 RGB clip 顺序一致：oldest -> newest。
- v1 的 ANALYSIS 是占位文本；训练脚本用 loss_scale 把该段 loss 置 0，主监督信号是 STATUS。
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

# 保持类样本避开 GT 转换帧附近的 buffer 半径，避免边界标注噪声。
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

    keyframes_all_scenarios.json 只给出 5 个关键帧：
    initial、middle[0]、middle[1]、middle[2]、final。训练时需要每个 anchor frame
    都有一个 GT STATUS，所以这里把关键帧扩展成闭区间时间轴。例如：

        [initial.frame, middle0.frame - 1] -> initial
        [middle0.frame, middle1.frame - 1] -> middle0.event

    这样 anchor 正好等于 middle0.frame 时，GT 就已经切到 middle0.event。
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
    # 如果 keyframes 里的事件名和 prompt_pipeline 状态机不一致，宁可丢弃该 run，
    # 否则 target_subgoal 会由另一套序列推导，训练信号会自相矛盾。
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
    """枚举 run 内所有合法候选样本，返回 (保持类, 推进类)。

    `memory_in_status` 来自 prev_anchor，`target_status` 来自 anchor。
    这样训练样本模拟真实在线推理：模型看到上一轮 memory，再根据当前最新帧决定
    是否推进 STATUS。

    两类样本的语义：
    - keep：prev_anchor 和 anchor 落在同一状态区间，模型应该保持 STATUS。
    - advance：prev_anchor 在上一状态，anchor 已落在新状态窗口，模型应该推进一步。

    v1 的核心痛点是“模型太早推进”，所以 keep 样本会避开转换帧附近 buffer；
    但转换后的 [f_t, f_t + K - 1] 会自然成为推进窗口，给模型补足“过界后应推进”的监督。
    """

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
            # 推进类样本：只要 prev_anchor 还在上一段、anchor 已经落到新段，就应该推进。
            # 对单个转换帧 f_t 和 K=4，这会产生 [f_t, f_t+3] 的推进窗口。
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

    # 推进类天然少：每条 run 最多约 4 * K 个转换窗口样本。
    # 如果数量不够目标比例，就全收；不要为了比例复制样本，避免过拟合某些 route。
    if len(advances) > target_advance:
        chosen_adv = rng.sample(advances, target_advance)
    else:
        chosen_adv = list(advances)

    # 保持类数量通常远多于推进类，而且不同 status 段长度差异很大。
    # 按 memory_in_status 分桶可以防止长 initial/final 段把中间状态淹没。
    buckets: Dict[str, List[SampleRecord]] = defaultdict(list)
    for s in keeps:
        buckets[s.memory_in_status].append(s)

    chosen_keep: List[SampleRecord] = []
    # ---- 为什么用 id() set 而不是 `s in chosen_keep` ----
    # SampleRecord 是 @dataclass，Python 自动给它生成 __eq__：两个实例只要所有
    # 字段值都相等就视为相等。这意味着：
    #   * `if s not in chosen_keep` 会按"值"判等，而不是按"是不是同一个对象"；
    #   * 如果某一天 collect_candidates() 为同一 (run_id, anchor) 产生
    #     两个独立的 SampleRecord 实例（比如多次调用合并、或加新字段后
    #     夫妻字段都一样），值相等的 record 会被错误剔除，导致 remaining 漏掉
    #     真正"未被选中的对象"。
    # 这里 (run_id, anchor) 实际唯一所以现在不出错，但用对象 id 做集合判重
    # 不依赖 dataclass __eq__ 行为，未来加字段 / 改字段也不会塌。
    # 另外好处：id() set 查询 O(1)，比 `in list` O(n) 在补齐循环里也更快。
    chosen_ids: set = set()
    if buckets:
        per_bucket = max(1, target_keep // len(buckets))
        for status, samples in buckets.items():
            if len(samples) > per_bucket:
                picked = rng.sample(samples, per_bucket)
            else:
                picked = list(samples)
            for s in picked:
                chosen_keep.append(s)
                chosen_ids.add(id(s))
        # 补齐到 target_keep。
        if len(chosen_keep) < target_keep:
            remaining = [s for s in keeps if id(s) not in chosen_ids]
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
    run_dir = pathlib.Path(run_dir_template.format(scenario=scenario, run_id=run_id))
    rgb_dir = run_dir / "rgb"

    # 与 qwen3vl_local.image_io.load_lead_rgb_clip 保持同一套容错。
    #
    # 为什么不能简单写死 f"{idx:04d}.jpg"：
    # - keyframes 的 frame 是 0-based 索引；
    # - 某些 LEAD route 的落盘文件可能从 0001.jpg 起步；
    # - runner 读取时已经有 fallback 到 sorted(rgb/*.jpg)[idx]。
    #
    # 训练数据如果不采用同样 fallback，LoRA 训练可能会看到不存在的图片路径，
    # 或者和 runner 的 anchor 对齐错一帧。
    if not rgb_dir.exists():
        return [str(rgb_dir / f"{f:04d}.jpg") for f in ordered]

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        return [str(rgb_dir / f"{f:04d}.jpg") for f in ordered]

    resolved: List[str] = []
    for idx in ordered:
        rgb_path = rgb_dir / f"{idx:04d}.jpg"
        if rgb_path.exists():
            resolved.append(str(rgb_path))
        elif 0 <= idx < len(rgb_files):
            resolved.append(str(rgb_files[idx]))
        else:
            resolved.append(str(rgb_path))
    return resolved


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
    """拼 messages 列表 + images，最终成为 jsonl 一行。

    ms-swift 多模态 SFT 常用格式是：
    - 顶层 `images` 保存图片路径列表；
    - user.content 里用 `<image>` 占位符标记图片插入位置；
    - assistant.content 是需要计算 loss 的回答文本。

    注意这里的 user prompt 复用 qwen3vl_local.prompt_pipeline，保证训练时看到的
    system/user 指令和 standalone runner 尽量一致。
    """

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
    # eval_sft_v1.py 会在复原 prompt 时去掉这些占位符，再交给本项目的 structured
    # image message 路径；因此 `<image>` 只服务于训练框架，不进入本地 engine 的 user_prompt。
    user_content = "".join("<image>" for _ in image_paths) + "\n" + user_text

    target_subgoal = _next_event_in_seq(scenario, sample.target_status)
    assistant_content = (
        # v1 暂不学习自然语言解释。这个固定句子只是保留三段输出格式，
        # 训练脚本会用 loss_scale 把 ANALYSIS 段权重置 0。
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
    """按 run_id 划分 train / val。防止 frame 级 leak。

    同一 route 内相邻帧高度相似。如果按 sample 随机切分，训练集和验证集会包含
    同一段驾驶过程的相邻 anchor，验证指标会虚高。按 run_id 划分虽然会让 val
    数量略不均衡，但更能反映泛化到新 route 的状态识别能力。
    """

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
    # samples-per-scenario / advance-ratio：
    # 200/0.25 是 v1 早期版本；ckpt-8100 那次实测 STATUS 答对但 EOS 信号被刷崩，
    # 主因之一是数据太少导致同一 sample 被反复看几十遍。
    # 800/0.35 是当前默认：全集从 ~4000 涨到 ~14000，推进类比例提到 35%，
    # 配合 sft_v1_train.sh 里 num_epochs=2 / lr=5e-5，等效 batch=32 时
    # 总 step ≈ 900，与 PLAN §6 估算量级回到一致。
    parser.add_argument("--samples-per-scenario", type=int, default=800)
    parser.add_argument("--advance-ratio", type=float, default=0.35)
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
    #
    # 这里先在场景内部平衡，再把所有场景合并。这样训练集不会被样本数最多的场景
    # 主导，模型更容易学到“每个 scenario 都有自己的 EVENT_SEQUENCE”。
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
