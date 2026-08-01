"""SFT base 训练入口：RS/EVENT 两问直接 token 监督。

和 sft_v5 相同：
- 数据来自 route-level sequence index；
- Q1 判 RS，Q2 判 EVENT；
- 使用相同 history RGB、EVENT 候选随机化、EGO_TO_GOAL_XY 与 memory 状态。

和 sft_v5 不同：
- 没有 OPSD；
- 没有 student rollout；
- 没有 privileged teacher / teacher logits；
- 没有 CoT，assistant target 只包含选项行。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_base import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_base.labels import (  # noqa: E402
    RS_DESCRIPTIONS,
    EventTarget,
    RSTarget,
    event_in_candidates,
    is_regular_event,
    is_unusual,
    resolve_event_target,
)
from qwen3vl_local.sft_base.memory_curriculum import (  # noqa: E402
    RouteMemoryCorruptor,
    maybe_corrupt_memory as _maybe_corrupt_memory,
    resample_event_memory_for_q2,
)
from qwen3vl_local.sft_base.prompts import (  # noqa: E402
    SYSTEM_PROMPT_BASE,
    Memory,
    build_q1_prompt,
    build_q1_target,
    build_q2_prompt,
    build_q2_target,
    refresh_memory_goal,
    loss_weights_q1,
    loss_weights_q2,
    reset_memory_for_frame,
    target_spans_q1,
    target_spans_q2,
    update_memory_after_q1,
    update_memory_after_q2,
)
from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _assert_inside_assistant_turn,
    _find_subsequence,
    _is_vision_module_name,
    collate_passthrough,
    cleanup_distributed,
    load_model_with_lora,
    make_scheduler,
    setup_distributed,
    validate_safety_args,
)


@dataclass
class FrameRow:
    """训练时使用的单帧轻量对象。"""

    frame_id: int
    history_rgb_paths: List[str]
    weather_text: str
    ego_to_goal_xy: Optional[Tuple[float, float]]
    rs_label: str
    event_label: str
    event_code: str
    abnormal: bool
    # Q2 本帧候选与展示顺序，直接来自 build_dataset 的 `event_candidates_ordered`。
    event_candidates: List[str]
    regular_event_codes: List[str]
    raw: Dict[str, Any]


@dataclass
class SequenceRow:
    """一条 route sequence。"""

    scenario: str
    route_id: str
    split: str
    frames: List[FrameRow]


class RouteSequenceDataset(Dataset):
    """读取 build_dataset.py 生成的 sequence_index.jsonl。

    Dataset 只保留路径和标签元数据，不在 __init__ 里读 RGB。真正的图像读取放在
    run_batch/evaluate_loss 内部逐帧执行，避免 DDP/DataLoader worker 在启动阶段把
    大量 PIL 对象常驻内存。
    """

    def __init__(self, path: pathlib.Path, *, max_routes: int = 0, max_frames_per_route: int = 0):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"sequence index not found: {self.path}")
        rows: List[SequenceRow] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("dataset_version") != DATASET_VERSION:
                    raise ValueError(
                        f"{self.path}: dataset_version mismatch. expected {DATASET_VERSION}, "
                        f"got {obj.get('dataset_version')!r}. Rebuild the sequence index with current build_dataset.py."
                    )
                frames: List[FrameRow] = []
                for fr in obj.get("frames", []):
                    if "event_candidates_ordered" not in fr:
                        # 旧 schema 用 `event_option_map` 存 {字母: 标签}。token 协议下字母
                        # 已完全废弃，这里不做兼容读取：静默降级会让 Q2 候选顺序或候选集合
                        # 悄悄变化，指标看起来正常但训练分布已经错了。
                        raise ValueError(
                            f"{self.path}: frame missing 'event_candidates_ordered'. "
                            "This index was built by the old letter-choice schema and is no longer "
                            "supported. Rebuild it with the current build_dataset.py."
                        )
                    frames.append(
                        FrameRow(
                            frame_id=int(fr["frame_id"]),
                            history_rgb_paths=[str(x) for x in fr.get("history_rgb_paths", [])],
                            weather_text=str(fr.get("weather_text", "")),
                            ego_to_goal_xy=_parse_goal_xy(fr.get("ego_to_goal_xy")),
                            rs_label=str(fr.get("rs_label", "R1")),
                            event_label=str(fr.get("event_label", "R-E1")),
                            event_code=str(fr.get("event_code", "R-E1")),
                            abnormal=bool(fr.get("abnormal", False)),
                            event_candidates=[str(x) for x in (fr.get("event_candidates_ordered") or [])],
                            regular_event_codes=[str(x) for x in fr.get("regular_event_codes", [])],
                            raw=fr,
                        )
                    )
                if max_frames_per_route > 0:
                    frames = frames[:max_frames_per_route]
                if not frames:
                    continue
                rows.append(
                    SequenceRow(
                        scenario=str(obj.get("scenario", "")),
                        route_id=str(obj.get("route_id", "")),
                        split=str(obj.get("split", "")),
                        frames=frames,
                    )
                )
                if max_routes > 0 and len(rows) >= max_routes:
                    break
        self.rows = rows
        event_label_counts: Dict[str, int] = {}
        rs_event_label_counts: Dict[str, Dict[str, int]] = {}
        for row in rows:
            for frame in row.frames:
                event_label_counts[frame.event_label] = event_label_counts.get(frame.event_label, 0) + 1
                rs_counts = rs_event_label_counts.setdefault(frame.rs_label, {})
                rs_counts[frame.event_label] = rs_counts.get(frame.event_label, 0) + 1
        self.event_label_counts = event_label_counts
        self.rs_event_label_counts = rs_event_label_counts

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SequenceRow:
        return self.rows[idx]


def _parse_goal_xy(value: Any) -> Optional[Tuple[float, float]]:
    """把 dataset 里的 `ego_to_goal_xy` 容错解析成二元组。"""

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _load_images(paths: List[str]) -> List[Image.Image]:
    """读取 4 帧 RGB history。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _transition_positions(frames: List[FrameRow], *, radius: int) -> set[int]:
    """找出 RS/EVENT 发生变化的帧及其邻域，用于重点重复训练。

    真正需要看图的样本主要集中在 regular->UE、UE->regular 和 RS 改变的附近；整段 UE
    过采样会大量重复“抄 memory 也能对”的帧，所以这里单独放大转折邻域。
    """

    positions: set[int] = set()
    rad = max(0, int(radius))
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        cur = frames[idx]
        changed = prev.rs_label != cur.rs_label or prev.event_label != cur.event_label or bool(prev.abnormal) != bool(cur.abnormal)
        if not changed:
            continue
        lo = max(0, idx - rad)
        hi = min(len(frames) - 1, idx + rad)
        positions.update(range(lo, hi + 1))
    return positions


def _rs_target_from_frame(frame: FrameRow) -> RSTarget:
    """把 FrameRow 还原成 RSTarget。"""

    return RSTarget(
        label=frame.rs_label,
        description=RS_DESCRIPTIONS.get(frame.rs_label, ""),
        confidence=float(frame.raw.get("rs_confidence", 0.0) or 0.0),
        secondary=tuple(frame.raw.get("rs_secondary") or []),
        candidates={str(k): float(v) for k, v in (frame.raw.get("rs_candidates") or {}).items()},
    )


def _event_target_from_frame(frame: FrameRow, student_event: Optional[str] = None) -> EventTarget:
    """从 frame row 解析 EventTarget，保持 v5 多标签容错逻辑。"""

    raw = dict(frame.raw)
    if "events" not in raw:
        raw["events"] = raw.get("event_labels_raw") or [frame.event_code]
    raw["primary_event"] = raw.get("event_code_raw") or frame.event_code
    return resolve_event_target(raw, student_event=student_event, rs_label=frame.rs_label)


def _messages(images: List[Image.Image], q1_prompt: str, q1_target: str, q2_prompt: Optional[str], q2_target: Optional[str]) -> List[Dict[str, Any]]:
    """构造完整 teacher-forced chat。

    Q2 是同一条多轮对话里的第二个 user turn，不重新喂图片；这与 eval 的 KV 复用
    协议保持一致。训练虽然一次性 forward 整段 chat，但 loss 只会落在 assistant
    答案值 token 上。
    """

    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": q1_prompt})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": content},
        {"role": "assistant", "content": q1_target},
    ]
    if q2_prompt is not None and q2_target is not None:
        messages.extend(
            [
                {"role": "user", "content": q2_prompt},
                {"role": "assistant", "content": q2_target},
            ]
        )
    return messages


def _value_token_ids(bundle: Any, assistant_text: str, span_fn: Any, weights_by_name: Mapping[str, float]) -> Tuple[List[int], List[float]]:
    """对 assistant turn 生成 token 级 loss 权重。

    这里先用字符 span 找到 `RS:`/`EVENT:` 后面的值，再通过 tokenizer
    offset 映射回 token。格式 token、换行和 label 名称权重都为 0，模型只被监督
    输出选项值本身，避免把直接 baseline 变成格式复读训练。
    """

    enc = bundle.tokenizer(assistant_text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    spans = span_fn(assistant_text)
    weights: List[float] = [0.0 for _ in token_ids]
    for name, (lo, hi) in spans.items():
        weight = float(weights_by_name.get(name, 0.0))
        if weight <= 0:
            continue
        for i, (a, b) in enumerate(offsets):
            if a < hi and b > lo:
                weights[i] = max(weights[i], weight)
    return token_ids, weights


def _build_inputs(
    bundle: Any,
    *,
    images: List[Image.Image],
    q1_prompt: str,
    q1_target: str,
    q1_loss_weights: Optional[Mapping[str, float]],
    q2_prompt: Optional[str],
    q2_target: Optional[str],
    q2_loss_weights: Optional[Mapping[str, float]],
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """构造模型输入，并只在答案值 token 上打 loss。

    processor 会把图片 token、system、user、assistant 全部展开到同一串 input_ids。
    后面通过 `_find_subsequence` 把每个 assistant target 定位回展开后的 token 序列；
    Q2 target 可能和 Q1 target 文本很短且重复，所以第二个 assistant turn 使用
    `prefer_last=True`，再由 `_assert_inside_assistant_turn` 确认没有打到 user prompt。
    """

    messages = _messages(images, q1_prompt, q1_target, q2_prompt, q2_target)
    chat_text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = bundle.processor(text=[chat_text], images=images if images else None, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"][0]
    if int(input_ids.shape[0]) > int(max_length):
        return None
    labels = input_ids.clone()
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    expanded_ids = [int(x) for x in input_ids.tolist()]
    asst_header_ids = list(bundle.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])

    assistant_specs = [
        (q1_target, target_spans_q1, q1_loss_weights or loss_weights_q1(), False),
    ]
    if q2_target is not None:
        assistant_specs.append((q2_target, target_spans_q2, q2_loss_weights or loss_weights_q2(), True))
    cursor = 0
    for turn_idx, (assistant_text, span_fn, span_weights, prefer_last) in enumerate(assistant_specs):
        assistant_ids, value_weights = _value_token_ids(bundle, assistant_text, span_fn, span_weights)
        pos = _find_subsequence(expanded_ids, assistant_ids, cursor, last=prefer_last)
        _assert_inside_assistant_turn(expanded_ids, pos, asst_header_ids, turn_idx)
        for j, weight in enumerate(value_weights):
            if weight > 0:
                weights[pos + j] = float(weight)
        cursor = pos + len(assistant_ids)

    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "loss_weights": weights,
        "vision": extra,
        "chat_text": chat_text,
    }


def _loss_one_sample(bundle: Any, packed: Mapping[str, Any]) -> torch.Tensor:
    """单样本 weighted CE。"""

    kwargs: Dict[str, Any] = {
        "input_ids": packed["input_ids"].unsqueeze(0).to(bundle.device),
        "attention_mask": packed["attention_mask"].unsqueeze(0).to(bundle.device),
    }
    labels = packed["labels"].unsqueeze(0).to(bundle.device)
    weights = packed["loss_weights"].unsqueeze(0).to(bundle.device)
    for k, v in packed["vision"].items():
        kwargs[k] = v.to(bundle.device) if isinstance(v, torch.Tensor) else v
    out = bundle.model(**kwargs, use_cache=False, return_dict=True)
    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    active = shift_weights.gt(0)
    if not bool(active.any()):
        return shift_logits.sum() * 0.0
    per_tok = F.cross_entropy(shift_logits[active], shift_labels[active], reduction="none")
    return (per_tok * shift_weights[active]).sum() / shift_weights[active].sum().clamp_min(1e-6)


def _event_loss_weights(
    event_target: EventTarget,
    *,
    rs_label: str,
    n_candidates: int,
    ue_event_loss_weight: float,
    re_event_loss_weight: float,
    single_candidate_re_scale: float,
    event_label_counts: Optional[Mapping[str, int]] = None,
    rs_event_label_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Dict[str, float]:
    """按 EVENT 类型和候选难度返回 Q2 token loss 权重。

    UE 按 RS 条件发生率做 inverse-sqrt 缩放，避免模型学成“R2 就猜 UE、
    R4/R5 就猜 regular”的先验；regular 内部也按全局频次做 inverse-sqrt，
    让 LANE_CHANGE/HIGHWAY_MANEUVER 等长尾 regular 不被 LANE_FOLLOWING 淹没。
    """

    if event_target.abnormal:
        weight = float(ue_event_loss_weight)
        all_counts = {str(k): int(v) for k, v in (event_label_counts or {}).items() if int(v) > 0}
        total = sum(all_counts.values())
        ue_total = sum(v for k, v in all_counts.items() if is_unusual(k))
        rs_counts = {str(k): int(v) for k, v in (rs_event_label_counts or {}).get(str(rs_label), {}).items() if int(v) > 0}
        rs_total = sum(rs_counts.values())
        rs_ue_total = sum(v for k, v in rs_counts.items() if is_unusual(k))
        if total > 0 and ue_total > 0 and rs_total > 0 and rs_ue_total > 0:
            global_rate = float(ue_total) / float(total)
            rs_rate = float(rs_ue_total) / float(rs_total)
            weight *= math.sqrt(max(global_rate, 1e-6) / max(rs_rate, 1e-6))
    elif int(n_candidates) <= 1:
        weight = float(re_event_loss_weight) * max(0.0, float(single_candidate_re_scale))
    else:
        weight = float(re_event_loss_weight)
        regular_counts = [
            int(v)
            for k, v in (event_label_counts or {}).items()
            if is_regular_event(str(k)) and int(v) > 0
        ]
        current = int((event_label_counts or {}).get(str(event_target.label), 0) or 0)
        if regular_counts and current > 0:
            regular_counts = sorted(regular_counts)
            median = float(regular_counts[len(regular_counts) // 2])
            weight *= math.sqrt(max(median, 1.0) / float(current))
    return {"event": max(0.0, weight)}


def _ue_repeat_for_frame(
    frame: FrameRow,
    *,
    event_label_counts: Optional[Mapping[str, int]],
    ue_frame_repeat: int,
    ue_repeat_mode: str,
    ue_repeat_max: int,
) -> int:
    """按 UE 子类频次计算重复次数。

    fixed 模式保留旧行为；inverse_sqrt 模式在旧 UE_FRAME_REPEAT 基础上按
    sqrt(median/count[label]) 调整，长尾 UE 子类得到更多重复，常见 UE 可降到 1。
    """

    if not bool(frame.abnormal):
        return 1
    base = max(1, int(ue_frame_repeat))
    mode = str(ue_repeat_mode).lower()
    if mode != "inverse_sqrt":
        return base
    counts = {str(k): int(v) for k, v in (event_label_counts or {}).items() if str(k).startswith("U-E") and int(v) > 0}
    current = counts.get(str(frame.event_label), 0)
    if current <= 0 or not counts:
        return base
    values = sorted(counts.values())
    median = float(values[len(values) // 2])
    repeat = int(round(float(base) * math.sqrt(max(median, 1.0) / float(current))))
    return max(1, min(max(1, int(ue_repeat_max)), repeat))


def _regular_repeat_for_frame(
    frame: FrameRow,
    *,
    event_label_counts: Optional[Mapping[str, int]],
    regular_frame_repeat: int,
    regular_repeat_mode: str,
    regular_repeat_max: int,
) -> int:
    """按 regular 子类频次计算重复次数。

    UE 已有独立 repeat；这里只处理最终 GT 为 regular 的帧。默认 inverse_sqrt 会
    让 R-E2/R-E3 这类长尾 regular 得到额外曝光，但常见 R-E1/R-E4/R-E5 仍保持
    约 1 倍，避免把总训练量粗暴放大。
    """

    if bool(frame.abnormal) or not is_regular_event(str(frame.event_label)):
        return 1
    base = max(1, int(regular_frame_repeat))
    mode = str(regular_repeat_mode).lower()
    if mode != "inverse_sqrt":
        return base
    counts = {str(k): int(v) for k, v in (event_label_counts or {}).items() if is_regular_event(str(k)) and int(v) > 0}
    current = counts.get(str(frame.event_label), 0)
    if current <= 0 or not counts:
        return base
    values = sorted(counts.values())
    median = float(values[len(values) // 2])
    repeat = int(round(float(base) * math.sqrt(max(median, 1.0) / float(current))))
    return max(1, min(max(1, int(regular_repeat_max)), repeat))


def _frame_training_pack(
    bundle: Any,
    frame: FrameRow,
    memory: Memory,
    images: List[Image.Image],
    max_length: int,
    *,
    route_id: str,
    memory_noise_seed: int,
    ue_event_loss_weight: float,
    re_event_loss_weight: float,
    single_candidate_re_scale: float,
    event_label_counts: Optional[Mapping[str, int]],
    rs_event_label_counts: Optional[Mapping[str, Mapping[str, int]]],
    event_wrong_prob: float,
    event_unknown_prob: float,
    clean_event_memory_label: str,
    q2_memory_override: Optional[Memory] = None,
) -> Tuple[Optional[Dict[str, Any]], Memory, bool]:
    """把一个 frame 转成 teacher-forced 训练输入。

    这是 sft_base 和 v5 on-policy 最大的区别：本函数直接使用 GT RS/EVENT
    构造 Q1/Q2 target，并用 GT answer 更新下一步 memory。这样得到的是“干净直接
    监督 baseline”，不是 student rollout 后的自维护 memory 分布。
    """

    rs_target = _rs_target_from_frame(frame)
    event_target = _event_target_from_frame(frame)
    q1_prompt = build_q1_prompt(memory, choice_seed=f"rs::{frame.frame_id}")
    q1_target = build_q1_target(rs_target=rs_target, event_target=event_target)
    q1_loss_weights = loss_weights_q1()
    memory_after_q1 = update_memory_after_q1(memory, student_rs_label=rs_target.label)
    if q2_memory_override is not None:
        q2_memory = q2_memory_override.copy()
    else:
        q2_memory = resample_event_memory_for_q2(
            memory_after_q1,
            frame=frame,
            route_id=route_id,
            seed=int(memory_noise_seed),
            event_wrong_prob=float(event_wrong_prob),
            event_unknown_prob=float(event_unknown_prob),
            keep_event_label=str(clean_event_memory_label),
        )

    q2_prompt: Optional[str] = None
    q2_target: Optional[str] = None
    q2_included = event_in_candidates(event_target.label, frame.event_candidates)
    if q2_included:
        q2_prompt = build_q2_prompt(
            q2_memory,
            candidates=frame.event_candidates,
            regular_event_codes=frame.regular_event_codes,
        )
        q2_target = build_q2_target(
            q2_memory,
            candidates=frame.event_candidates,
            event_target=event_target,
            regular_event_codes=frame.regular_event_codes,
        )
        q2_loss_weights = _event_loss_weights(
            event_target,
            rs_label=frame.rs_label,
            n_candidates=len(frame.event_candidates),
            ue_event_loss_weight=ue_event_loss_weight,
            re_event_loss_weight=re_event_loss_weight,
            single_candidate_re_scale=single_candidate_re_scale,
            event_label_counts=event_label_counts,
            rs_event_label_counts=rs_event_label_counts,
        )
    else:
        q2_loss_weights = None
    # 如果 EVENT 真值不在候选表里，跳过 Q2 监督但仍训练 Q1；单候选帧也训练，
    # 只是单候选 regular 会在 loss 权重里降权，避免送分题淹没硬负样本。
    packed = _build_inputs(
        bundle,
        images=images,
        q1_prompt=q1_prompt,
        q1_target=q1_target,
        q1_loss_weights=q1_loss_weights,
        q2_prompt=q2_prompt,
        q2_target=q2_target,
        q2_loss_weights=q2_loss_weights,
        max_length=max_length,
    )
    next_memory = update_memory_after_q2(memory_after_q1, student_event_label=event_target.label)
    return packed, next_memory, q2_included


@dataclass
class StepStats:
    loss_sum: float = 0.0
    n_samples: int = 0
    n_frames: int = 0
    n_q2: int = 0
    n_q2_ue: int = 0
    n_q2_re: int = 0
    n_skipped: int = 0
    q2_ue_weight_sum: float = 0.0
    q2_re_weight_sum: float = 0.0


def run_batch(
    bundle: Any,
    batch: List[SequenceRow],
    *,
    max_length: int,
    loss_scale: float,
    sync_grads: bool,
    frames_per_sync: int = 0,
    ue_frame_repeat: int = 1,
    transition_frame_repeat: int = 1,
    transition_frame_window: int = 1,
    ue_event_loss_weight: float = 4.0,
    re_event_loss_weight: float = 1.0,
    single_candidate_re_scale: float = 0.1,
    event_label_counts: Optional[Mapping[str, int]] = None,
    rs_event_label_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
    ue_repeat_mode: str = "inverse_sqrt",
    ue_repeat_max: int = 8,
    regular_frame_repeat: int = 1,
    regular_repeat_mode: str = "inverse_sqrt",
    regular_repeat_max: int = 6,
    first_frame_memory_unknown: bool = True,
    memory_rs_wrong_prob: float = 0.30,
    memory_rs_unknown_prob: float = 0.40,
    memory_event_wrong_prob: float = 0.35,
    memory_event_unknown_prob: float = 0.35,
    rs_wrong_event_unknown_prob: float = 0.25,
    memory_dropout_prob: float = 0.15,
    memory_noise_seed: int = 20260724,
    memory_perturbation_mode: str = "route_state",
    memory_perturb_duration_min: int = 3,
    memory_perturb_duration_max: int = 5,
) -> StepStats:
    """逐 route/frame 跑 teacher-forced CE，并累积梯度。

    函数分两段做事：先按 route 顺序生成每帧的 memory snapshot，再逐帧读图 forward。
    第一段每帧刷新 EGO_TO_GOAL_XY，然后用 GT answer teacher-forced 更新离散 memory；
    第二段只负责训练 loss。这样即使某帧图片读取失败，也不会改变同 route 后续帧的
    teacher-forced memory 轨迹。

    多卡时每个 rank 拿到的是完整 route，route 帧数差异会很大。如果只在整条 route
    结束后同步，短 route 的 rank 可能在 NCCL all-reduce 里等长 route 超过 10 分钟。
    `frames_per_sync` 会把长 route 切成固定帧数 heartbeat：各 rank 按相同 collective
    顺序同步累计梯度，已跑完本地 chunk 的 rank 继续带着已有平均梯度参与，避免超时。
    """

    stats = StepStats()
    work: List[Tuple[str, FrameRow, Memory, Optional[Memory], str]] = []
    for route in batch:
        memory: Optional[Memory] = None
        route_corruptor: Optional[RouteMemoryCorruptor] = None
        if str(memory_perturbation_mode) == "route_state":
            route_corruptor = RouteMemoryCorruptor(
                route_id=route.route_id,
                seed=int(memory_noise_seed),
                first_frame_unknown=bool(first_frame_memory_unknown),
                rs_wrong_prob=float(memory_rs_wrong_prob),
                rs_unknown_prob=float(memory_rs_unknown_prob),
                event_wrong_prob=float(memory_event_wrong_prob),
                event_unknown_prob=float(memory_event_unknown_prob),
                rs_wrong_event_unknown_prob=float(rs_wrong_event_unknown_prob),
                memory_dropout_prob=float(memory_dropout_prob),
                duration_min=int(memory_perturb_duration_min),
                duration_max=int(memory_perturb_duration_max),
            )
        transition_pos = _transition_positions(list(route.frames), radius=int(transition_frame_window))
        for frame_pos, frame in enumerate(route.frames):
            rs_target = _rs_target_from_frame(frame)
            if memory is None:
                memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
            else:
                memory = refresh_memory_goal(memory, frame.ego_to_goal_xy)
            # 存 copy 而不是引用：后面会继续更新 memory，当前帧 prompt 必须看到
            # “提问前”的状态，而不是被后续帧改写后的状态。
            if route_corruptor is not None:
                prompt_memory = route_corruptor.corrupt(memory, frame=frame, frame_pos=frame_pos)
            else:
                prompt_memory = _maybe_corrupt_memory(
                    memory,
                    frame=frame,
                    route_id=route.route_id,
                    frame_pos=frame_pos,
                    seed=int(memory_noise_seed),
                    first_frame_unknown=bool(first_frame_memory_unknown),
                    rs_wrong_prob=float(memory_rs_wrong_prob),
                    rs_unknown_prob=float(memory_rs_unknown_prob),
                    event_wrong_prob=float(memory_event_wrong_prob),
                    event_unknown_prob=float(memory_event_unknown_prob),
                    rs_wrong_event_unknown_prob=float(rs_wrong_event_unknown_prob),
                    memory_dropout_prob=float(memory_dropout_prob),
                )
            repeat = _ue_repeat_for_frame(
                frame,
                event_label_counts=event_label_counts,
                ue_frame_repeat=int(ue_frame_repeat),
                ue_repeat_mode=str(ue_repeat_mode),
                ue_repeat_max=int(ue_repeat_max),
            )
            if not bool(frame.abnormal):
                repeat = max(
                    repeat,
                    _regular_repeat_for_frame(
                        frame,
                        event_label_counts=event_label_counts,
                        regular_frame_repeat=int(regular_frame_repeat),
                        regular_repeat_mode=str(regular_repeat_mode),
                        regular_repeat_max=int(regular_repeat_max),
                    ),
                )
            if frame_pos in transition_pos:
                repeat = max(repeat, max(1, int(transition_frame_repeat)))
            clean_event_memory_label = "UNKNOWN" if frame_pos == 0 else str(memory.event_label)
            q2_memory_override: Optional[Memory] = None
            if route_corruptor is not None:
                q2_seed_memory = update_memory_after_q1(prompt_memory, student_rs_label=rs_target.label)
                q2_memory_override = route_corruptor.resample_event_for_q2(
                    q2_seed_memory,
                    frame=frame,
                    keep_event_label=clean_event_memory_label,
                )
            for _ in range(repeat):
                work.append(
                    (
                        route.route_id,
                        frame,
                        prompt_memory.copy(),
                        q2_memory_override.copy() if q2_memory_override is not None else None,
                        clean_event_memory_label,
                    )
                )
            event_target = _event_target_from_frame(frame)
            memory_after_q1 = update_memory_after_q1(memory, student_rs_label=rs_target.label)
            memory = update_memory_after_q2(memory_after_q1, student_event_label=event_target.label)
    if not work:
        if sync_grads and int(frames_per_sync) > 0:
            while True:
                _sync_trainable_grads(bundle)
                if _ddp_sum_float(0.0, bundle.device) <= 0.0:
                    break
        elif sync_grads:
            _sync_trainable_grads(bundle)
        return stats

    if sync_grads and int(frames_per_sync) > 0:
        chunk_size = max(1, int(frames_per_sync))
        offset = 0
        while True:
            chunk = work[offset : offset + chunk_size]
            offset += len(chunk)
            chunk_stats = _run_work_items(
                bundle,
                chunk,
                max_length=max_length,
                loss_scale=loss_scale,
                loss_normalizer=max(1, len(work)),
                ue_event_loss_weight=float(ue_event_loss_weight),
                re_event_loss_weight=float(re_event_loss_weight),
                single_candidate_re_scale=float(single_candidate_re_scale),
                event_label_counts=event_label_counts,
                rs_event_label_counts=rs_event_label_counts,
                memory_noise_seed=int(memory_noise_seed),
                event_wrong_prob=float(memory_event_wrong_prob),
                event_unknown_prob=float(memory_event_unknown_prob),
            )
            _merge_stats(stats, chunk_stats)
            _sync_trainable_grads(bundle)
            if _ddp_sum_float(1.0 if offset < len(work) else 0.0, bundle.device) <= 0.0:
                break
        return stats

    chunk_stats = _run_work_items(
        bundle,
        work,
        max_length=max_length,
        loss_scale=loss_scale,
        loss_normalizer=max(1, len(work)),
        ue_event_loss_weight=float(ue_event_loss_weight),
        re_event_loss_weight=float(re_event_loss_weight),
        single_candidate_re_scale=float(single_candidate_re_scale),
        event_label_counts=event_label_counts,
        rs_event_label_counts=rs_event_label_counts,
        memory_noise_seed=int(memory_noise_seed),
        event_wrong_prob=float(memory_event_wrong_prob),
        event_unknown_prob=float(memory_event_unknown_prob),
    )
    _merge_stats(stats, chunk_stats)
    if sync_grads:
        _sync_trainable_grads(bundle)
    return stats


def _merge_stats(dst: StepStats, src: StepStats) -> None:
    """把 chunk 统计累加到 batch 统计。"""

    dst.loss_sum += src.loss_sum
    dst.n_samples += src.n_samples
    dst.n_frames += src.n_frames
    dst.n_q2 += src.n_q2
    dst.n_q2_ue += src.n_q2_ue
    dst.n_q2_re += src.n_q2_re
    dst.n_skipped += src.n_skipped
    dst.q2_ue_weight_sum += src.q2_ue_weight_sum
    dst.q2_re_weight_sum += src.q2_re_weight_sum


def _run_work_items(
    bundle: Any,
    work: List[Tuple[str, FrameRow, Memory, Optional[Memory], str]],
    *,
    max_length: int,
    loss_scale: float,
    loss_normalizer: int,
    ue_event_loss_weight: float,
    re_event_loss_weight: float,
    single_candidate_re_scale: float,
    event_label_counts: Optional[Mapping[str, int]],
    rs_event_label_counts: Optional[Mapping[str, Mapping[str, int]]],
    memory_noise_seed: int,
    event_wrong_prob: float,
    event_unknown_prob: float,
) -> StepStats:
    """执行一段 frame work，不在函数内部发起 DDP collective。"""

    stats = StepStats()
    for route_id, frame, memory, q2_memory_override, clean_event_memory_label in work:
        try:
            images = _load_images(frame.history_rgb_paths)
        except (FileNotFoundError, OSError) as exc:
            print(f"[warn] image load failed {route_id} frame={frame.frame_id}: {exc}")
            stats.n_skipped += 1
            continue
        packed, _next_memory, q2_included = _frame_training_pack(
            bundle,
            frame,
            memory,
            images,
            max_length,
            route_id=route_id,
            memory_noise_seed=int(memory_noise_seed),
            ue_event_loss_weight=float(ue_event_loss_weight),
            re_event_loss_weight=float(re_event_loss_weight),
            single_candidate_re_scale=float(single_candidate_re_scale),
            event_label_counts=event_label_counts,
            rs_event_label_counts=rs_event_label_counts,
            event_wrong_prob=float(event_wrong_prob),
            event_unknown_prob=float(event_unknown_prob),
            clean_event_memory_label=clean_event_memory_label,
            q2_memory_override=q2_memory_override,
        )
        if packed is None:
            stats.n_skipped += 1
            continue
        sync_ctx = bundle.model.no_sync() if hasattr(bundle.model, "no_sync") else nullcontext()
        with sync_ctx:
            # DDP no_sync 必须同时包住 forward 和 backward。这里每帧都只在本 rank
            # 累积梯度，micro-batch 末尾由 _sync_trainable_grads 手动 all-reduce 一次，
            # 所以不同 rank 的 route/frame 数量不同也不会造成 collective 次数漂移。
            bundle.model.train()
            loss = _loss_one_sample(bundle, packed)
            (loss / float(max(1, loss_normalizer)) / max(loss_scale, 1.0)).backward()
        stats.loss_sum += float(loss.detach().item())
        stats.n_samples += 1
        stats.n_frames += 1
        stats.n_q2 += int(q2_included)
        stats.n_q2_ue += int(q2_included and frame.abnormal)
        stats.n_q2_re += int(q2_included and not frame.abnormal)
        if q2_included:
            weight = _event_loss_weights(
                _event_target_from_frame(frame),
                rs_label=frame.rs_label,
                n_candidates=len(frame.event_candidates),
                ue_event_loss_weight=ue_event_loss_weight,
                re_event_loss_weight=re_event_loss_weight,
                single_candidate_re_scale=single_candidate_re_scale,
                event_label_counts=event_label_counts,
                rs_event_label_counts=rs_event_label_counts,
            ).get("event", 0.0)
            if frame.abnormal:
                stats.q2_ue_weight_sum += float(weight)
            else:
                stats.q2_re_weight_sum += float(weight)
    return stats


def _sync_trainable_grads(bundle: Any) -> None:
    """DDP 下手动同步所有 LoRA 可训练参数梯度。

    即使某个 rank 当前 micro-batch 没有有效 frame，也会为 trainable 参数补零梯度并
    参与同一组 all-reduce，避免 rank 间 collective 数量或参数顺序不一致。
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    model = bundle.unwrap() if hasattr(bundle, "unwrap") else bundle.model
    world = float(dist.get_world_size())
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world)


@torch.no_grad()
def evaluate_loss(
    bundle: Any,
    loader: DataLoader,
    *,
    max_length: int,
    max_samples: int,
    ue_event_loss_weight: float,
    re_event_loss_weight: float,
    single_candidate_re_scale: float,
    event_label_counts: Optional[Mapping[str, int]] = None,
    rs_event_label_counts: Optional[Mapping[str, Mapping[str, int]]] = None,
    memory_noise_seed: int = 20260724,
    first_frame_memory_unknown: bool = True,
    memory_rs_wrong_prob: float = 0.30,
    memory_rs_unknown_prob: float = 0.40,
    event_wrong_prob: float = 0.35,
    event_unknown_prob: float = 0.35,
    rs_wrong_event_unknown_prob: float = 0.25,
    memory_dropout_prob: float = 0.15,
    memory_perturbation_mode: str = "route_state",
    memory_perturb_duration_min: int = 3,
    memory_perturb_duration_max: int = 5,
) -> Dict[str, float]:
    """计算 teacher-forced 验证 loss，并使用训练同款 memory 扰动。"""

    bundle.model.eval()
    losses: List[float] = []
    skipped = 0
    q2_count = 0
    frame_count = 0
    for batch in loader:
        for route in batch:
            memory: Optional[Memory] = None
            route_corruptor: Optional[RouteMemoryCorruptor] = None
            if str(memory_perturbation_mode) == "route_state":
                route_corruptor = RouteMemoryCorruptor(
                    route_id=route.route_id,
                    seed=int(memory_noise_seed),
                    first_frame_unknown=bool(first_frame_memory_unknown),
                    rs_wrong_prob=float(memory_rs_wrong_prob),
                    rs_unknown_prob=float(memory_rs_unknown_prob),
                    event_wrong_prob=float(event_wrong_prob),
                    event_unknown_prob=float(event_unknown_prob),
                    rs_wrong_event_unknown_prob=float(rs_wrong_event_unknown_prob),
                    memory_dropout_prob=float(memory_dropout_prob),
                    duration_min=int(memory_perturb_duration_min),
                    duration_max=int(memory_perturb_duration_max),
                )
            for frame_pos, frame in enumerate(route.frames):
                if max_samples > 0 and len(losses) >= max_samples:
                    break
                rs_target = _rs_target_from_frame(frame)
                if memory is None:
                    memory = reset_memory_for_frame(rs_target, ego_to_goal_xy=frame.ego_to_goal_xy)
                else:
                    memory = refresh_memory_goal(memory, frame.ego_to_goal_xy)
                try:
                    images = _load_images(frame.history_rgb_paths)
                    clean_event_memory_label = "UNKNOWN" if frame_pos == 0 else str(memory.event_label)
                    if route_corruptor is not None:
                        prompt_memory = route_corruptor.corrupt(memory, frame=frame, frame_pos=frame_pos)
                        q2_seed_memory = update_memory_after_q1(prompt_memory, student_rs_label=rs_target.label)
                        q2_memory_override = route_corruptor.resample_event_for_q2(
                            q2_seed_memory,
                            frame=frame,
                            keep_event_label=clean_event_memory_label,
                        )
                    else:
                        prompt_memory = _maybe_corrupt_memory(
                            memory,
                            frame=frame,
                            route_id=route.route_id,
                            frame_pos=frame_pos,
                            seed=int(memory_noise_seed),
                            first_frame_unknown=bool(first_frame_memory_unknown),
                            rs_wrong_prob=float(memory_rs_wrong_prob),
                            rs_unknown_prob=float(memory_rs_unknown_prob),
                            event_wrong_prob=float(event_wrong_prob),
                            event_unknown_prob=float(event_unknown_prob),
                            rs_wrong_event_unknown_prob=float(rs_wrong_event_unknown_prob),
                            memory_dropout_prob=float(memory_dropout_prob),
                        )
                        q2_memory_override = None
                    packed, _next_memory, q2_included = _frame_training_pack(
                        bundle,
                        frame,
                        prompt_memory,
                        images,
                        max_length,
                        route_id=route.route_id,
                        memory_noise_seed=int(memory_noise_seed),
                        ue_event_loss_weight=float(ue_event_loss_weight),
                        re_event_loss_weight=float(re_event_loss_weight),
                        single_candidate_re_scale=float(single_candidate_re_scale),
                        event_label_counts=event_label_counts,
                        rs_event_label_counts=rs_event_label_counts,
                        event_wrong_prob=float(event_wrong_prob),
                        event_unknown_prob=float(event_unknown_prob),
                        clean_event_memory_label=clean_event_memory_label,
                        q2_memory_override=q2_memory_override,
                    )
                    event_target = _event_target_from_frame(frame)
                    memory_after_q1 = update_memory_after_q1(memory, student_rs_label=rs_target.label)
                    memory = update_memory_after_q2(memory_after_q1, student_event_label=event_target.label)
                    if packed is None:
                        skipped += 1
                        continue
                    losses.append(float(_loss_one_sample(bundle, packed).item()))
                    q2_count += int(q2_included)
                    frame_count += 1
                except (FileNotFoundError, OSError):
                    skipped += 1
            if max_samples > 0 and len(losses) >= max_samples:
                break
        if max_samples > 0 and len(losses) >= max_samples:
            break
    values = torch.tensor([sum(losses), len(losses), skipped, q2_count, frame_count], device=bundle.device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "loss": float(values[0].item()) / max(float(values[1].item()), 1.0),
        "samples": float(values[1].item()),
        "skipped": float(values[2].item()),
        "q2_rate": float(values[3].item()) / max(float(values[4].item()), 1.0),
    }


def _trainable_param_groups(bundle: Any, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter], List[nn.Parameter]]:
    """按语言/视觉 LoRA 分组。"""

    language: List[nn.Parameter] = []
    vision: List[nn.Parameter] = []
    for name, param in bundle.unwrap().named_parameters():
        if not param.requires_grad:
            continue
        if _is_vision_module_name(name):
            vision.append(param)
        else:
            language.append(param)
    groups: List[Dict[str, Any]] = []
    if language:
        groups.append({"params": language, "lr": float(args.learning_rate), "name": "language"})
    if vision:
        groups.append({"params": vision, "lr": float(args.learning_rate) * float(args.vision_lr_scale), "name": "vision"})
    return groups, language, vision


def _ddp_sum_float(value: float, device: torch.device) -> float:
    """DDP sum 一个标量；单进程直接返回。"""

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _vision_param_norm(vision_params: List[nn.Parameter]) -> float:
    """计算视觉 LoRA 参数范数。"""

    if not vision_params:
        return 0.0
    with torch.no_grad():
        return math.sqrt(sum(float(p.detach().float().norm().item()) ** 2 for p in vision_params))


def _max_across_ranks(values: List[float], device: torch.device) -> List[float]:
    """DDP 下对若干标量取 rank 间最大值。"""

    if not (dist.is_available() and dist.is_initialized()):
        return [float(v) for v in values]
    tensor = torch.tensor(values, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(v) for v in tensor.cpu().tolist()]


def _vision_guard_reason(
    *,
    vision_params: List[nn.Parameter],
    vis_grad_norm: float,
    args: argparse.Namespace,
    device: torch.device,
    bad_steps: int,
) -> Tuple[int, Optional[str], float, float]:
    """检查视觉 LoRA fuse guard，返回更新后的 bad_steps 和可选熔断原因。

    视觉 LoRA 默认只训 merger，但它仍然可能把视觉桥接层冲坏。guard 用 rank 间最大
    grad/param norm 判断异常，连续超过 patience 才熔断；熔断时保存 emergency adapter，
    并跳过 final，避免后续 eval 误拿异常权重。
    """

    if not bool(args.vision_guard_enabled) or not vision_params:
        return 0, None, float(vis_grad_norm), 0.0
    vis_param_norm = _vision_param_norm(vision_params)
    vis_grad_norm, vis_param_norm = _max_across_ranks([float(vis_grad_norm), float(vis_param_norm)], device)
    bad_grad = (not math.isfinite(vis_grad_norm)) or vis_grad_norm > float(args.vision_guard_grad_norm_max)
    bad_param = (not math.isfinite(vis_param_norm)) or vis_param_norm > float(args.vision_guard_param_norm_max)
    next_bad_steps = bad_steps + 1 if (bad_grad or bad_param) else 0
    if next_bad_steps < int(args.vision_guard_patience):
        return next_bad_steps, None, vis_grad_norm, vis_param_norm
    reason = (
        "vision fuse triggered: "
        f"grad_norm={vis_grad_norm:.4f} (max={float(args.vision_guard_grad_norm_max):.4f}), "
        f"param_norm={vis_param_norm:.4f} (max={float(args.vision_guard_param_norm_max):.4f}), "
        f"bad_steps={next_bad_steps}"
    )
    return next_bad_steps, reason, vis_grad_norm, vis_param_norm


def _save_adapter(path: pathlib.Path, bundle: Any, args: argparse.Namespace) -> None:
    """保存 adapter 与 sft_base 自描述配置。"""

    path.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(path))
    try:
        bundle.processor.save_pretrained(str(path))
    except Exception as exc:
        print(f"[warn] save processor skipped: {exc}")
    target_modules = list(getattr(bundle, "lora_target_modules", []))
    vision_targets = [name for name in target_modules if _is_vision_module_name(name)]
    payload = {
        "schema_version": 1,
        "route": "sft_base_token_choice",
        "dataset_version": DATASET_VERSION,
        "base_model_dir": str(args.model_dir),
        "base_model_mutated": False,
        "lora_vision": bool(vision_targets),
        "lora_vision_scope": getattr(bundle, "lora_vision_scope", str(args.lora_vision_scope)),
        "target_modules": target_modules,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "vision_lr_scale": float(args.vision_lr_scale),
        "language_clip_norm": float(args.language_clip_norm),
        "vision_clip_norm": float(args.vision_clip_norm),
        "max_length": int(args.max_length),
        "frames_per_sync": int(args.frames_per_sync),
        "ue_event_loss_weight": float(args.ue_event_loss_weight),
        "re_event_loss_weight": float(args.re_event_loss_weight),
        "single_candidate_re_scale": float(args.single_candidate_re_scale),
        "rs_conditional_ue_loss": True,
        "regular_event_loss_reweight": "inverse_sqrt",
        "ue_frame_repeat": int(args.ue_frame_repeat),
        "ue_repeat_mode": str(args.ue_repeat_mode),
        "ue_repeat_max": int(args.ue_repeat_max),
        "regular_frame_repeat": int(args.regular_frame_repeat),
        "regular_repeat_mode": str(args.regular_repeat_mode),
        "regular_repeat_max": int(args.regular_repeat_max),
        "transition_frame_repeat": int(args.transition_frame_repeat),
        "transition_frame_window": int(args.transition_frame_window),
        "first_frame_memory_unknown": bool(args.first_frame_memory_unknown),
        "memory_rs_wrong_prob": float(args.memory_rs_wrong_prob),
        "memory_rs_unknown_prob": float(args.memory_rs_unknown_prob),
        "memory_event_wrong_prob": float(args.memory_event_wrong_prob),
        "memory_event_unknown_prob": float(args.memory_event_unknown_prob),
        "rs_wrong_event_unknown_prob": float(args.rs_wrong_event_unknown_prob),
        "memory_dropout_prob": float(args.memory_dropout_prob),
        "memory_perturbation_mode": str(args.memory_perturbation_mode),
        "memory_perturb_duration_min": int(args.memory_perturb_duration_min),
        "memory_perturb_duration_max": int(args.memory_perturb_duration_max),
    }
    (path / "sft_base_adapter_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    p = argparse.ArgumentParser(description="Train SFT base direct RS/EVENT token LoRA")
    p.add_argument("--train-index", type=str, required=True)
    p.add_argument("--val-index", type=str, default=None)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="last4", choices=["off", "merger", "last4", "all"])
    p.add_argument("--lora-vision", action="store_true", help="legacy alias for --lora-vision-scope=all")
    p.add_argument("--vision-lr-scale", type=float, default=0.2)
    p.add_argument("--max-vision-lr-scale", type=float, default=0.25)
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--language-clip-norm", type=float, default=1.0)
    p.add_argument("--vision-clip-norm", type=float, default=0.3)
    p.add_argument("--vision-guard-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision-guard-grad-norm-max", type=float, default=10.0)
    p.add_argument("--vision-guard-param-norm-max", type=float, default=200.0)
    p.add_argument("--vision-guard-patience", type=int, default=3)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--frames-per-sync", type=int, default=64)
    p.add_argument("--ue-event-loss-weight", type=float, default=4.0)
    p.add_argument("--re-event-loss-weight", type=float, default=1.0)
    p.add_argument("--single-candidate-re-scale", type=float, default=0.1)
    p.add_argument("--ue-frame-repeat", type=int, default=2)
    p.add_argument("--ue-repeat-mode", choices=["fixed", "inverse_sqrt"], default="inverse_sqrt")
    p.add_argument("--ue-repeat-max", type=int, default=8)
    p.add_argument("--regular-frame-repeat", type=int, default=1)
    p.add_argument("--regular-repeat-mode", choices=["fixed", "inverse_sqrt"], default="inverse_sqrt")
    p.add_argument("--regular-repeat-max", type=int, default=6)
    p.add_argument("--transition-frame-repeat", type=int, default=4)
    p.add_argument("--transition-frame-window", type=int, default=3)
    p.add_argument("--first-frame-memory-unknown", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory-rs-wrong-prob", type=float, default=0.30)
    p.add_argument("--memory-rs-unknown-prob", type=float, default=0.40)
    p.add_argument("--memory-event-wrong-prob", type=float, default=0.35)
    p.add_argument("--memory-event-unknown-prob", type=float, default=0.35)
    p.add_argument("--rs-wrong-event-unknown-prob", type=float, default=0.25)
    p.add_argument("--memory-dropout-prob", type=float, default=0.15)
    p.add_argument("--memory-perturbation-mode", choices=["frame", "route_state"], default="route_state")
    p.add_argument("--memory-perturb-duration-min", type=int, default=3)
    p.add_argument("--memory-perturb-duration-max", type=int, default=5)
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=20260724)
    return p.parse_args()


def main() -> None:
    """训练主入口。"""

    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(int(args.seed) + rank)
    torch.manual_seed(int(args.seed) + rank)

    if args.lora_vision and str(args.lora_vision_scope).lower() == "off":
        args.lora_vision_scope = "all"
    validate_safety_args(args)

    train_ds = RouteSequenceDataset(pathlib.Path(args.train_index), max_routes=int(args.max_routes), max_frames_per_route=int(args.max_frames_per_route))
    val_ds = RouteSequenceDataset(pathlib.Path(args.val_index), max_routes=int(args.max_routes), max_frames_per_route=int(args.max_frames_per_route)) if args.val_index else None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds else None
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    train_loader = DataLoader(train_ds, batch_size=int(args.per_device_batch_size), sampler=train_sampler, shuffle=shuffle, num_workers=int(args.num_workers), collate_fn=collate_passthrough)
    val_loader = DataLoader(val_ds, batch_size=int(args.per_device_batch_size), sampler=val_sampler, shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_passthrough) if val_ds else None

    if args.check:
        print(f"[check] routes={len(train_ds)} first={train_ds.rows[0].scenario + '/' + train_ds.rows[0].route_id if train_ds.rows else 'NA'}")
        cleanup_distributed()
        return

    output_dir = pathlib.Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] output_dir={output_dir} rank={rank} world={world_size} device={device} vision_scope={args.lora_vision_scope}")
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_vision_scope=str(args.lora_vision_scope),
        strict_vision_scope=bool(args.strict_vision_scope),
        gradient_checkpointing=not bool(args.no_grad_checkpoint),
    )
    if world_size > 1:
        bundle.model = torch.nn.parallel.DistributedDataParallel(
            bundle.model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    groups, language_params, vision_params = _trainable_param_groups(bundle, args)
    optimizer = torch.optim.AdamW(groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, int(args.grad_accum))))
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else steps_per_epoch * int(args.num_epochs)
    scheduler = make_scheduler(optimizer, total_steps, int(total_steps * float(args.warmup_ratio)))
    tb = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE else None

    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    accum_samples = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    fuse_stopped = False
    guard_bad_steps = 0

    def finish_optimizer_step(reason: str, last_stats: StepStats) -> None:
        """完成一次梯度同步后的 optimizer step。

        进入这里之前，run_batch 已经在需要同步的 micro-batch 上调用
        `_sync_trainable_grads`，所以这里只做全局 sample/loss 统计、分组裁剪、
        视觉 guard 和 optimizer/scheduler step。tail step 也走同一条路径，避免最后
        不满 grad_accum 的梯度留在显存里没有更新。
        """

        nonlocal global_step, accum_loss, accum_samples, stop, fuse_stopped, guard_bad_steps
        global_samples = _ddp_sum_float(float(accum_samples), device)
        global_loss_sum = _ddp_sum_float(float(accum_loss), device)
        if global_samples <= 0:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_samples = 0
            return
        if language_params:
            lang_norm = torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm))
        else:
            lang_norm = torch.tensor(0.0)
        if vision_params:
            vis_norm = torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm))
        else:
            vis_norm = torch.tensor(0.0)
        guard_bad_steps, guard_reason, vis_norm_value, vis_param_norm = _vision_guard_reason(
            vision_params=vision_params,
            vis_grad_norm=float(vis_norm),
            args=args,
            device=device,
            bad_steps=guard_bad_steps,
        )
        if guard_reason:
            # 熔断后不写 normal final：目录名和 fuse_reason.txt 能强提醒这是异常中止产物。
            optimizer.zero_grad(set_to_none=True)
            stop = True
            fuse_stopped = True
            if rank == 0:
                emergency_dir = output_dir / f"fuse_stop_step_{global_step + 1}"
                _save_adapter(emergency_dir, bundle, args)
                (emergency_dir / "fuse_reason.txt").write_text(guard_reason + "\n", encoding="utf-8")
                print(f"[fuse-stop] {guard_reason}")
                print(f"[fuse-stop] emergency adapter -> {emergency_dir}")
            accum_loss = 0.0
            accum_samples = 0
            return
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        loss_avg = global_loss_sum / max(global_samples, 1.0)
        if rank == 0 and (global_step == 1 or global_step % int(args.logging_steps) == 0 or reason == "tail"):
            elapsed = (time.time() - start) / 60.0
            q2_rate = last_stats.n_q2 / max(1, last_stats.n_frames)
            q2_ue_rate = last_stats.n_q2_ue / max(1, last_stats.n_q2)
            q2_weight_total = last_stats.q2_ue_weight_sum + last_stats.q2_re_weight_sum
            q2_ue_weight_share = last_stats.q2_ue_weight_sum / max(1e-6, q2_weight_total)
            print(
                f"[train] epoch={epoch} step={global_step}/{total_steps} loss={loss_avg:.4f} "
                f"samples={int(global_samples)} q2_rate={q2_rate:.3f} q2_ue_rate={q2_ue_rate:.3f} "
                f"q2_ue_weight_share={q2_ue_weight_share:.3f} "
                f"|g|_lang={float(lang_norm):.3f} |g|_vis={vis_norm_value:.3f} "
                f"|w|_vis={vis_param_norm:.3f} guard_bad_steps={guard_bad_steps} "
                f"{reason}=1 elapsed={elapsed:.1f}m"
            )
            if tb is not None:
                tb.add_scalar("train/loss", loss_avg, global_step)
                tb.add_scalar("train/q2_rate_last_batch", q2_rate, global_step)
                tb.add_scalar("train/q2_ue_rate_last_batch", q2_ue_rate, global_step)
                tb.add_scalar("train/q2_ue_weight_share_last_batch", q2_ue_weight_share, global_step)
                tb.add_scalar("train/q2_ue_weight_sum_last_batch", last_stats.q2_ue_weight_sum, global_step)
                tb.add_scalar("train/q2_re_weight_sum_last_batch", last_stats.q2_re_weight_sum, global_step)
                tb.add_scalar("train/grad_norm/language", float(lang_norm), global_step)
                if vision_params:
                    tb.add_scalar("train/grad_norm/vision", vis_norm_value, global_step)
                    tb.add_scalar("train/param_norm/lora_vision", vis_param_norm, global_step)
                    tb.add_scalar("train/vision_guard_bad_steps", float(guard_bad_steps), global_step)
        accum_loss = 0.0
        accum_samples = 0
        if rank == 0 and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
            _save_adapter(output_dir / f"checkpoint-{global_step}", bundle, args)
        if val_loader is not None and int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
            metrics = evaluate_loss(
                bundle,
                val_loader,
                max_length=int(args.max_length),
                max_samples=int(args.max_eval_samples),
                ue_event_loss_weight=float(args.ue_event_loss_weight),
                re_event_loss_weight=float(args.re_event_loss_weight),
                single_candidate_re_scale=float(args.single_candidate_re_scale),
                event_label_counts=getattr(val_ds, "event_label_counts", None),
                rs_event_label_counts=getattr(val_ds, "rs_event_label_counts", None),
                memory_noise_seed=int(args.seed),
                first_frame_memory_unknown=bool(args.first_frame_memory_unknown),
                memory_rs_wrong_prob=float(args.memory_rs_wrong_prob),
                memory_rs_unknown_prob=float(args.memory_rs_unknown_prob),
                event_wrong_prob=float(args.memory_event_wrong_prob),
                event_unknown_prob=float(args.memory_event_unknown_prob),
                rs_wrong_event_unknown_prob=float(args.rs_wrong_event_unknown_prob),
                memory_dropout_prob=float(args.memory_dropout_prob),
                memory_perturbation_mode=str(args.memory_perturbation_mode),
                memory_perturb_duration_min=int(args.memory_perturb_duration_min),
                memory_perturb_duration_max=int(args.memory_perturb_duration_max),
            )
            if rank == 0:
                print(f"[eval@{global_step}] {metrics}")
                if tb is not None:
                    for key, value in metrics.items():
                        tb.add_scalar(f"val/{key}", value, global_step)
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            stop = True

    for epoch in range(int(args.num_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        last_stats = StepStats()
        for batch in train_loader:
            # 只有到 grad_accum 边界时才做一次手动梯度同步；边界前的 micro-batch
            # 都在 no_sync 下本地累积，避免每帧 backward 都触发 DDP collective。
            sync_this = (micro_step + 1) % max(1, int(args.grad_accum)) == 0
            stats = run_batch(
                bundle,
                batch,
                max_length=int(args.max_length),
                loss_scale=float(max(1, int(args.grad_accum))),
                sync_grads=sync_this,
                frames_per_sync=int(args.frames_per_sync),
                ue_frame_repeat=int(args.ue_frame_repeat),
                event_label_counts=getattr(train_ds, "event_label_counts", None),
                rs_event_label_counts=getattr(train_ds, "rs_event_label_counts", None),
                ue_repeat_mode=str(args.ue_repeat_mode),
                ue_repeat_max=int(args.ue_repeat_max),
                regular_frame_repeat=int(args.regular_frame_repeat),
                regular_repeat_mode=str(args.regular_repeat_mode),
                regular_repeat_max=int(args.regular_repeat_max),
                transition_frame_repeat=int(args.transition_frame_repeat),
                transition_frame_window=int(args.transition_frame_window),
                ue_event_loss_weight=float(args.ue_event_loss_weight),
                re_event_loss_weight=float(args.re_event_loss_weight),
                single_candidate_re_scale=float(args.single_candidate_re_scale),
                first_frame_memory_unknown=bool(args.first_frame_memory_unknown),
                memory_rs_wrong_prob=float(args.memory_rs_wrong_prob),
                memory_rs_unknown_prob=float(args.memory_rs_unknown_prob),
                memory_event_wrong_prob=float(args.memory_event_wrong_prob),
                memory_event_unknown_prob=float(args.memory_event_unknown_prob),
                rs_wrong_event_unknown_prob=float(args.rs_wrong_event_unknown_prob),
                memory_dropout_prob=float(args.memory_dropout_prob),
                memory_noise_seed=int(args.seed) + rank,
                memory_perturbation_mode=str(args.memory_perturbation_mode),
                memory_perturb_duration_min=int(args.memory_perturb_duration_min),
                memory_perturb_duration_max=int(args.memory_perturb_duration_max),
            )
            last_stats = stats
            accum_loss += stats.loss_sum
            accum_samples += stats.n_samples
            micro_step += 1
            if not sync_this:
                continue
            finish_optimizer_step("grad_accum", stats)
            if stop:
                break
        if micro_step % max(1, int(args.grad_accum)) != 0 and not stop:
            # epoch 尾部不足 grad_accum 的残余梯度也必须显式 all-reduce 后再 step；
            # 否则最后几个 micro-batch 只更新本 rank，本地 LoRA 会在保存前分叉。
            _sync_trainable_grads(bundle)
            finish_optimizer_step("tail", last_stats)
        if stop:
            break

    if rank == 0 and not fuse_stopped:
        _save_adapter(output_dir / "final", bundle, args)
        print(f"[done] final adapter -> {output_dir / 'final'}")
    elif rank == 0 and fuse_stopped:
        print("[done] skipped final adapter because vision fuse guard stopped training early")
    if tb is not None:
        tb.flush()
        tb.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
