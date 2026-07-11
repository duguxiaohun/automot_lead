"""SFT v5 训练入口：RS/EVENT 两问 OPSD + true DDP sequence padding。

核心训练流程：

1. DataLoader 每次读取若干条 route sequence；
2. collate 阶段只做本 rank local padding，主训练进程再 all-reduce 得到 global T；
3. 每个有效 frame 先让 student 自由回答 Q1；
4. Q1 RS 正确才进入 Q2，否则本帧结束，下一有效帧恢复 GT RS + RE；
5. teacher 关闭 LoRA，读取 privileged prompt，在同一批 student rollout token 上给
   full-vocabulary logits；
6. student/teacher logits 做 forward-KL，梯度只回到 LoRA student。

`--check` 模式不加载模型，只检查 dataset / DDP padding / prompt / memory 状态机。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _is_vision_module_name,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.sft_v3.train import (  # noqa: E402
    _append_token_ids_with_logits,
    _append_user_turn,
    _clone_kv_state,
    _kv_start_state,
    _student_generate_kv,
    _teacher_eval_context,
    _teacher_start_state,
)
from qwen3vl_local.sft_v5 import DATASET_VERSION  # noqa: E402
from qwen3vl_local.sft_v5.labels import (  # noqa: E402
    RS_LABEL_TO_OPTION,
    RS_OPTION_DESCRIPTIONS,
    EventTarget,
    RSTarget,
    option_for_event,
    resolve_event_target,
)
from qwen3vl_local.sft_v5.prompts import (  # noqa: E402
    SYSTEM_PROMPT_V5,
    Memory,
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    loss_weights_q1,
    loss_weights_q2,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    target_spans_q1,
    target_spans_q2,
    update_memory_after_q1,
    update_memory_after_q2,
)


@dataclass
class FrameRow:
    """训练时使用的单帧轻量对象。"""

    frame_id: int
    history_rgb_paths: List[str]
    weather_text: str
    rs_label: str
    rs_option: str
    event_label: str
    event_code: str
    abnormal: bool
    event_option_map: Dict[str, str]
    regular_event_codes: List[str]
    raw: Dict[str, Any]


@dataclass
class SequenceRow:
    """一条 route sequence。"""

    scenario: str
    route_id: str
    run_dir: str
    split: str
    frames: List[FrameRow]


class RouteSequenceDataset(Dataset):
    """读取 build_dataset.py 生成的 sequence_index.jsonl。"""

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
                frames: List[FrameRow] = []
                for fr in obj.get("frames", []):
                    # build_dataset 已经把原始 annotation 压成训练需要的最小字段；
                    # raw 仍完整保留 frame row，供多标签 EVENT 动态真值和 probe 审计回查。
                    frames.append(
                        FrameRow(
                            frame_id=int(fr["frame_id"]),
                            history_rgb_paths=[str(x) for x in fr.get("history_rgb_paths", [])],
                            weather_text=str(fr.get("weather_text", "")),
                            rs_label=str(fr.get("rs_label", "R1")),
                            rs_option=str(fr.get("rs_option", "A")),
                            event_label=str(fr.get("event_label", "RE")),
                            event_code=str(fr.get("event_code", "R-E1")),
                            abnormal=bool(fr.get("abnormal", False)),
                            event_option_map={str(k): str(v) for k, v in (fr.get("event_option_map") or {}).items()},
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
                        run_dir=str(obj.get("run_dir", "")),
                        split=str(obj.get("split", "")),
                        frames=frames,
                    )
                )
                if max_routes > 0 and len(rows) >= max_routes:
                    break
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SequenceRow:
        return self.rows[idx]


def _global_max_int(value: int) -> int:
    """DDP 下对一个 int 做 all_reduce max；单进程直接返回。"""

    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())
    return int(value)


def collate_route_sequences(batch: List[SequenceRow]) -> Dict[str, Any]:
    """按本 rank 的最长 sequence 做 local padding。

    这里不读图片，只把 frame 对象 pad 成 None。真正的图像读取放在训练 step 中，
    避免 DataLoader worker 触碰 PIL/CUDA 状态。分布式 global max 在主训练进程里
    计算，避免 `num_workers>0` 时在 worker 进程触碰 distributed runtime。
    """

    local_max = max((len(row.frames) for row in batch), default=0)
    padded: List[List[Optional[FrameRow]]] = []
    valid: List[List[bool]] = []
    for row in batch:
        frames: List[Optional[FrameRow]] = list(row.frames)
        mask = [True] * len(frames)
        while len(frames) < local_max:
            frames.append(None)
            mask.append(False)
        padded.append(frames)
        valid.append(mask)
    return {
        "routes": batch,
        "frame_rows": padded,
        "valid_mask": torch.tensor(valid, dtype=torch.bool) if valid else torch.zeros((0, local_max), dtype=torch.bool),
        "max_T_local": local_max,
    }


def pad_batch_to_global_length(batch: Dict[str, Any]) -> Dict[str, Any]:
    """主训练进程把 local padded batch 右侧补齐到所有 rank 的 global max_T。"""

    local_max = int(batch.get("max_T_local", 0))
    global_max = _global_max_int(local_max)
    if global_max == local_max:
        batch["max_T_global"] = global_max
        return batch
    for frames in batch["frame_rows"]:
        # 只在主训练进程里补到 global max_T；padding frame 是 None，后续 loop 会跳过，
        # 不读图、不进 Qwen、不产生 loss。这样不同 rank 的 sequence 长度能对齐 collective，
        # 但不会把 padding 当成训练样本。
        while len(frames) < global_max:
            frames.append(None)
    valid = batch["valid_mask"]
    if valid.shape[1] < global_max:
        pad = torch.zeros((valid.shape[0], global_max - valid.shape[1]), dtype=torch.bool)
        batch["valid_mask"] = torch.cat([valid, pad], dim=1)
    batch["max_T_global"] = global_max
    return batch


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torch.distributed。"""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=6))
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """清理 DDP process group。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_images(paths: List[str]) -> List[Image.Image]:
    """读取 4 帧 RGB history；缺文件时抛错，由外层 frame skip 记录。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _messages(images: List[Image.Image], user_prompt: str) -> List[Dict[str, Any]]:
    """构造 Qwen structured chat messages。"""

    # Qwen3-VL processor 需要 structured message：4 张历史图先放，再放同一个 user prompt。
    # system prompt 固定为 v5 协议，确保 train/eval/probe 的图文输入完全一致。
    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V5},
        {"role": "user", "content": content},
    ]


def _loss_positions(bundle: Any, text: str, span_fn: Any, weights: Mapping[str, float]) -> Dict[str, List[int]]:
    """把字符 span 映射到 token 下标。"""

    enc = bundle.tokenizer(text or "", return_offsets_mapping=True, add_special_tokens=False)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    spans = span_fn(text or "")
    out: Dict[str, List[int]] = {key: [] for key in weights}
    for key, (lo, hi) in spans.items():
        if key not in out:
            continue
        # 字符 span 与 token offset 相交即可纳入 loss。这样即使 tokenizer 把 "A - ..."
        # 切成多个 token，也能完整监督离散值所在整段。
        out[key] = [i for i, (a, b) in enumerate(offsets) if a < hi and b > lo]
    return out


def _kl_positions(student_logits: torch.Tensor, teacher_logits: torch.Tensor, positions: List[int], *, temperature: float) -> torch.Tensor:
    """选定 token 位置上的 forward-KL。"""

    if not positions:
        return student_logits.sum() * 0.0
    idx = torch.tensor(positions, device=student_logits.device, dtype=torch.long)
    s = student_logits[:, idx, :].reshape(-1, student_logits.shape[-1])
    t = teacher_logits[:, idx, :].reshape(-1, teacher_logits.shape[-1]).detach()
    temp = max(float(temperature), 1e-6)
    return F.kl_div(
        F.log_softmax(s / temp, dim=-1),
        F.softmax(t / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)


def _opsd_loss(
    bundle: Any,
    *,
    student_state: Any,
    teacher_state: Any,
    rollout_text: str,
    rollout_ids: torch.Tensor,
    span_fn: Any,
    weights: Mapping[str, float],
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """在同一批 student rollout token 上比较 teacher/student logits。"""

    zero = student_state.next_logits.sum() * 0.0
    if rollout_ids.numel() == 0:
        return zero, {key: 0.0 for key in weights}
    with _teacher_eval_context(bundle):
        # teacher 使用同一个 base Qwen，但 adapter 被临时禁用，并吃 privileged prompt。
        # logits 只 detach 作目标分布；反向梯度只流向当前启用 LoRA 的 student。
        _, teacher_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(teacher_state), rollout_ids)
    _, student_logits, _ = _append_token_ids_with_logits(bundle, _clone_kv_state(student_state), rollout_ids)
    positions = _loss_positions(bundle, rollout_text, span_fn, weights)
    total = zero
    parts: Dict[str, float] = {}
    for key, weight in weights.items():
        loss = _kl_positions(student_logits, teacher_logits, positions.get(key, []), temperature=temperature)
        total = total + float(weight) * loss
        parts[key] = float(loss.detach().item()) if loss.numel() else 0.0
    return total, parts


def _rs_target_from_frame(frame: FrameRow) -> RSTarget:
    """把 FrameRow 还原成 RSTarget。"""

    return RSTarget(
        label=frame.rs_label,
        option=RS_LABEL_TO_OPTION.get(frame.rs_label, frame.rs_option),
        description=RS_OPTION_DESCRIPTIONS.get(RS_LABEL_TO_OPTION.get(frame.rs_label, frame.rs_option), ""),
        confidence=float(frame.raw.get("rs_confidence", 0.0) or 0.0),
        secondary=tuple(frame.raw.get("rs_secondary") or []),
        candidates={str(k): float(v) for k, v in (frame.raw.get("rs_candidates") or {}).items()},
    )


def _event_target_from_frame(frame: FrameRow, student_event: Optional[str] = None) -> EventTarget:
    """从压缩 frame row 解析 EventTarget。

    frame.raw 已保留 event_labels_raw，resolve_event_target 会按 v5 的多标签规则处理。
    """

    raw = dict(frame.raw)
    if "events" not in raw:
        raw["events"] = raw.get("event_labels_raw") or [frame.event_code]
    return resolve_event_target(raw, student_event=student_event)


def _run_frame(
    bundle: Any,
    memory: Memory,
    frame: FrameRow,
    *,
    max_new_tokens_q1: int,
    max_new_tokens_q2: int,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, Any], Memory, bool]:
    """运行单帧 Q1/Q2，返回 loss、统计、更新后的 memory、是否需要下一帧 reset。"""

    images = _load_images(frame.history_rgb_paths)
    rs_target = _rs_target_from_frame(frame)
    event_target_static = _event_target_from_frame(frame)

    # ---- Q1: student rollout ----
    q1_prompt = build_q1_student_prompt(memory)
    with torch.no_grad():
        # student 先自由生成 Q1；OPSD 的监督不是 teacher-forced token，而是随后在
        # 这批 student 自己采样出的 token 上比较 teacher/student 分布。
        q1_student_state = _kv_start_state(bundle, _messages(images, q1_prompt))
        q1_text, q1_after, q1_ids = _student_generate_kv(bundle, q1_student_state, max_new_tokens_q1)
    q1_parsed = parse_q1_output(q1_text)
    q1_teacher_prompt = build_q1_teacher_prompt(
        memory,
        rs_target=rs_target,
        event_target=event_target_static,
        weather_text=frame.weather_text,
    )
    q1_teacher_state = _teacher_start_state(bundle, _messages(images, q1_teacher_prompt))
    q1_loss, q1_parts = _opsd_loss(
        bundle,
        student_state=q1_student_state,
        teacher_state=q1_teacher_state,
        rollout_text=q1_text,
        rollout_ids=q1_ids,
        span_fn=target_spans_q1,
        weights=loss_weights_q1(),
        temperature=temperature,
    )

    student_rs = q1_parsed.get("rs_label")
    q1_rs_correct = student_rs == frame.rs_label
    q1_abnormal = q1_parsed.get("abnormal") == "YES" if q1_parsed.get("abnormal") else None
    memory_after_q1 = update_memory_after_q1(memory, student_rs_label=student_rs, student_abnormal=q1_abnormal)
    stats: Dict[str, Any] = {
        "q1_rs_correct": q1_rs_correct,
        "q1_abnormal_correct": q1_abnormal == frame.abnormal if q1_abnormal is not None else False,
        "q2_triggered": False,
        "candidate_mismatch": False,
        "q1_rollout_tokens": int(q1_ids.numel()),
        "q2_rollout_tokens": 0,
        "q1_parts": q1_parts,
    }
    if not q1_rs_correct:
        # Q1 的 RS 是 Q2 候选池的上层条件。RS 错时继续问 Q2 会把错误道路结构传下去，
        # 所以本帧立即截断；下一帧由外层恢复 GT RS + RE。
        return q1_loss, stats, memory_after_q1, True

    # ---- Q2: 只有 RS 正确才进入 ----
    q2_prompt = build_q2_student_prompt(
        memory_after_q1,
        option_map=frame.event_option_map,
        q1_abnormal=bool(q1_abnormal),
        regular_event_codes=frame.regular_event_codes,
    )
    with torch.no_grad():
        q2_student_state = _append_user_turn(bundle, q1_after, q2_prompt)
        q2_text, _q2_after, q2_ids = _student_generate_kv(bundle, q2_student_state, max_new_tokens_q2)
    q2_parsed = parse_q2_output(q2_text, frame.event_option_map)
    event_target = _event_target_from_frame(frame, student_event=q2_parsed.get("event_label"))
    # EVENT 支持“单标签训练、双标签容错”：如果 raw label 有多个 UE/RE，而 student
    # 选中了其中之一，teacher target 会接受这个 student_event 作为解释目标。
    target_option = option_for_event(event_target.label, frame.event_option_map)
    stats["candidate_mismatch"] = target_option is None
    q2_teacher_prompt = build_q2_teacher_prompt(
        memory_after_q1,
        option_map=frame.event_option_map,
        q1_abnormal=bool(q1_abnormal),
        event_target=event_target,
        regular_event_codes=frame.regular_event_codes,
    )
    q2_teacher_state = _teacher_start_state(bundle, _messages(images, q2_teacher_prompt))
    q2_loss, q2_parts = _opsd_loss(
        bundle,
        student_state=q2_student_state,
        teacher_state=q2_teacher_state,
        rollout_text=q2_text,
        rollout_ids=q2_ids,
        span_fn=target_spans_q2,
        weights=loss_weights_q2(),
        temperature=temperature,
    )
    student_event = q2_parsed.get("event_label")
    memory_after_q2 = update_memory_after_q2(memory_after_q1, student_event_label=student_event)
    q2_invalid = student_event is None
    stats.update(
        {
            "q2_triggered": True,
            "q2_event_correct": student_event == event_target.label,
            "q2_invalid_output": q2_invalid,
            "q2_rollout_tokens": int(q2_ids.numel()),
            "q2_parts": q2_parts,
        }
    )
    return q1_loss + q2_loss, stats, memory_after_q2, q2_invalid


def _trainable_param_groups(bundle: Any, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter], List[nn.Parameter]]:
    """按语言/视觉 LoRA 分组，复用 v2/v3 的视觉保险口径。"""

    language: List[nn.Parameter] = []
    vision: List[nn.Parameter] = []
    for name, param in bundle.model.named_parameters():
        if not param.requires_grad:
            continue
        # 视觉 LoRA 默认关闭；如果用户显式开启，视觉参数单独用较小 LR 和较低 clip norm，
        # 防止少量 RS/EVENT 监督把 Qwen 视觉表征冲坏。
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


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace) -> None:
    """保存 LoRA adapter 与 v5 元数据。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    model = bundle.unwrap() if hasattr(bundle, "unwrap") else bundle.model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
    meta = {
        "dataset_version": DATASET_VERSION,
        "lora_vision_scope": getattr(bundle, "lora_vision_scope", "off"),
        "lora_target_modules": getattr(bundle, "lora_target_modules", []),
        "max_new_tokens_q1": int(args.max_new_tokens_q1),
        "max_new_tokens_q2": int(args.max_new_tokens_q2),
        "temperature": float(args.temperature),
    }
    with open(output_dir / "sft_v5_adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def run_check(loader: DataLoader, *, max_batches: int = 2) -> None:
    """轻量检查 dataset 和 DDP padding，不加载模型。"""

    for batch_idx, batch in enumerate(loader):
        batch = pad_batch_to_global_length(batch)
        routes = batch["routes"]
        max_t = int(batch["max_T_global"])
        valid = batch["valid_mask"]
        print(f"[check] batch={batch_idx} routes={len(routes)} max_T_global={max_t} valid={int(valid.sum().item())}")
        for row in routes[:2]:
            first = row.frames[0]
            mem = reset_memory_for_frame(_rs_target_from_frame(first))
            print(
                f"  route={row.scenario}/{row.route_id} frames={len(row.frames)} "
                f"first_rs={first.rs_label} q2_options={first.event_option_map} mem={mem.rs_label}/{mem.event_label}"
            )
        if batch_idx + 1 >= max_batches:
            break


_TRAIN_WINDOW_KEYS = (
    "batches",
    "frames",
    "loss_sum",
    "q1_rs_correct",
    "q1_abnormal_correct",
    "q2_triggered",
    "q2_event_correct",
    "q2_invalid_output",
    "candidate_mismatch",
    "reset_next",
    "q1_rollout_tokens",
    "q2_rollout_tokens",
    "valid_slots",
    "padding_slots",
    "max_T_local_sum",
    "max_T_global_sum",
)


def _new_train_window_stats() -> Dict[str, float]:
    """创建一个 logging window 内的 on-policy 采样/训练统计容器。"""

    return {key: 0.0 for key in _TRAIN_WINDOW_KEYS}


def _add_batch_shape_stats(stats: Dict[str, float], batch: Mapping[str, Any]) -> None:
    """记录本 rank 当前 batch 的 padding 形状。

    这些数字能直接审计 DDP padding 是否按预期运行：valid_slots 是真实 frame 数，
    padding_slots 是补齐到 global max_T 后额外占位的 None frame 数。
    """

    routes = batch.get("routes") or []
    valid = batch.get("valid_mask")
    valid_count = int(valid.sum().item()) if isinstance(valid, torch.Tensor) else 0
    max_t_global = int(batch.get("max_T_global", 0))
    max_t_local = int(batch.get("max_T_local", 0))
    total_slots = len(routes) * max_t_global
    stats["batches"] += 1.0
    stats["valid_slots"] += float(valid_count)
    stats["padding_slots"] += float(max(0, total_slots - valid_count))
    stats["max_T_local_sum"] += float(max_t_local)
    stats["max_T_global_sum"] += float(max_t_global)


def _add_frame_rollout_stats(stats: Dict[str, float], frame_stats: Mapping[str, Any], *, need_reset: bool) -> None:
    """累计一个有效 frame 的 on-policy rollout 统计。"""

    stats["frames"] += 1.0
    stats["q1_rs_correct"] += float(bool(frame_stats.get("q1_rs_correct", False)))
    stats["q1_abnormal_correct"] += float(bool(frame_stats.get("q1_abnormal_correct", False)))
    stats["q2_triggered"] += float(bool(frame_stats.get("q2_triggered", False)))
    stats["q2_event_correct"] += float(bool(frame_stats.get("q2_event_correct", False)))
    stats["q2_invalid_output"] += float(bool(frame_stats.get("q2_invalid_output", False)))
    stats["candidate_mismatch"] += float(bool(frame_stats.get("candidate_mismatch", False)))
    stats["reset_next"] += float(bool(need_reset))
    stats["q1_rollout_tokens"] += float(int(frame_stats.get("q1_rollout_tokens", 0) or 0))
    stats["q2_rollout_tokens"] += float(int(frame_stats.get("q2_rollout_tokens", 0) or 0))


def _ddp_sum_train_stats(stats: Mapping[str, float]) -> Dict[str, float]:
    """把各 rank 的 logging window 统计按 sum 聚合到所有 rank。"""

    values = [float(stats.get(key, 0.0)) for key in _TRAIN_WINDOW_KEYS]
    if not (dist.is_available() and dist.is_initialized()):
        return dict(zip(_TRAIN_WINDOW_KEYS, values))
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(_TRAIN_WINDOW_KEYS, tensor.cpu().tolist())}


def _format_train_window(stats: Mapping[str, float]) -> str:
    """把聚合后的 on-policy 统计格式化成一行日志。"""

    frames = max(1.0, float(stats.get("frames", 0.0)))
    q2 = max(1.0, float(stats.get("q2_triggered", 0.0)))
    batches = max(1.0, float(stats.get("batches", 0.0)))
    valid_slots = max(1.0, float(stats.get("valid_slots", 0.0)))
    rollout_tokens = float(stats.get("q1_rollout_tokens", 0.0)) + float(stats.get("q2_rollout_tokens", 0.0))
    return (
        f"loss/frame={float(stats.get('loss_sum', 0.0)) / frames:.4f} "
        f"frames={int(stats.get('frames', 0.0))} "
        f"q2_rate={float(stats.get('q2_triggered', 0.0)) / frames:.3f} "
        f"rs_acc={float(stats.get('q1_rs_correct', 0.0)) / frames:.3f} "
        f"abn_acc={float(stats.get('q1_abnormal_correct', 0.0)) / frames:.3f} "
        f"event_acc={float(stats.get('q2_event_correct', 0.0)) / q2:.3f} "
        f"invalid={int(stats.get('q2_invalid_output', 0.0))} "
        f"reset={int(stats.get('reset_next', 0.0))} "
        f"tok/frame={rollout_tokens / frames:.1f} "
        f"pad_rate={float(stats.get('padding_slots', 0.0)) / (valid_slots + float(stats.get('padding_slots', 0.0))):.3f} "
        f"maxT={float(stats.get('max_T_global_sum', 0.0)) / batches:.1f}"
    )


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    p = argparse.ArgumentParser(description="Train SFT v5 RS/EVENT OPSD")
    p.add_argument("--train-index", type=str, required=True)
    p.add_argument("--val-index", type=str, default=None)
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="off", choices=["off", "merger", "last4", "all"])
    p.add_argument("--vision-lr-scale", type=float, default=0.1)
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--language-clip-norm", type=float, default=1.0)
    p.add_argument("--vision-clip-norm", type=float, default=0.3)
    p.add_argument("--max-new-tokens-q1", type=int, default=256)
    p.add_argument("--max-new-tokens-q2", type=int, default=192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--max-frames-per-route", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=20260711)
    return p.parse_args()


def main() -> None:
    """训练主入口。"""

    args = parse_args()
    torch.manual_seed(int(args.seed))
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds = RouteSequenceDataset(
        pathlib.Path(args.train_index),
        max_routes=int(args.max_routes),
        max_frames_per_route=int(args.max_frames_per_route),
    )
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        # v5 这里使用 true DDP 的普通 DistributedSampler；和 v3 local-SGD/work-stealing 不同。
        # 不同 rank 拿到的 route 长度可能不同，所以后面还要 global max_T 对齐。
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed))
    else:
        sampler = None
    loader = DataLoader(
        train_ds,
        batch_size=int(args.per_device_batch_size),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(args.num_workers),
        collate_fn=collate_route_sequences,
    )
    if args.check:
        run_check(loader)
        cleanup_distributed()
        return

    output_dir = pathlib.Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
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
            find_unused_parameters=True,
        )
    groups, language_params, vision_params = _trainable_param_groups(bundle, args)
    optimizer = torch.optim.AdamW(groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    total_steps = max(1, math.ceil(len(loader) * int(args.num_epochs) / max(1, int(args.grad_accum))))
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(total_steps * float(args.warmup_ratio)),
    )
    tb = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE else None

    global_step = 0
    micro_step = 0
    window_stats = _new_train_window_stats()
    start = time.time()
    bundle.model.train()
    for epoch in range(int(args.num_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = pad_batch_to_global_length(batch)
            _add_batch_shape_stats(window_stats, batch)
            batch_loss = None
            frame_count = 0
            routes: List[SequenceRow] = batch["routes"]
            frame_rows: List[List[Optional[FrameRow]]] = batch["frame_rows"]
            # 每条 route 各自维护 memory/reset；batch 内 route 之间互不影响。
            # reset_next=True 表示上一个有效帧 RS 错或 Q2 非法，下一帧开头恢复 GT RS + RE。
            reset_next = [False for _ in routes]
            memories: List[Optional[Memory]] = [None for _ in routes]
            for t in range(int(batch["max_T_global"])):
                for b, route in enumerate(routes):
                    frame = frame_rows[b][t]
                    if frame is None:
                        # global padding 位置不参与任何计算，确保 DDP 对齐只影响 loop 长度，
                        # 不影响训练样本数量和 loss。
                        continue
                    rs_target = _rs_target_from_frame(frame)
                    if memories[b] is None or reset_next[b]:
                        memories[b] = reset_memory_for_frame(rs_target)
                        reset_next[b] = False
                    assert memories[b] is not None
                    try:
                        loss, stats, next_mem, need_reset = _run_frame(
                            bundle,
                            memories[b],
                            frame,
                            max_new_tokens_q1=int(args.max_new_tokens_q1),
                            max_new_tokens_q2=int(args.max_new_tokens_q2),
                            temperature=float(args.temperature),
                        )
                    except FileNotFoundError as exc:
                        if rank == 0:
                            print(f"[warn] skip missing image route={route.route_id} frame={frame.frame_id}: {exc}")
                        continue
                    batch_loss = loss if batch_loss is None else batch_loss + loss
                    frame_count += 1
                    _add_frame_rollout_stats(window_stats, stats, need_reset=bool(need_reset))
                    memories[b] = next_mem
                    reset_next[b] = bool(need_reset)
            if batch_loss is None or frame_count == 0:
                continue
            batch_loss = batch_loss / float(frame_count)
            window_stats["loss_sum"] += float(batch_loss.detach().item()) * float(frame_count)
            loss_scaled = batch_loss / float(max(1, int(args.grad_accum)))
            sync_this_step = (micro_step + 1) % max(1, int(args.grad_accum)) == 0
            sync_context = (
                bundle.model.no_sync()
                if (world_size > 1 and not sync_this_step and hasattr(bundle.model, "no_sync"))
                else nullcontext()
            )
            with sync_context:
                # 梯度累积中间 micro step 不做 DDP all-reduce，最后一个 micro step 再同步，
                # 避免每帧/每小 batch 都触发昂贵 collective。
                loss_scaled.backward()
            micro_step += 1
            if micro_step % max(1, int(args.grad_accum)) == 0:
                if language_params:
                    torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm))
                if vision_params:
                    torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                log_every = max(1, int(args.logging_steps))
                if global_step % log_every == 0:
                    reduced_stats = _ddp_sum_train_stats(window_stats)
                    # 所有 rank 都必须在 collective 后清零窗口；只有 rank0 负责输出人可读日志。
                    window_stats = _new_train_window_stats()
                    elapsed = time.time() - start
                    if rank == 0:
                        log_line = _format_train_window(reduced_stats)
                        print(f"[train] step={global_step} epoch={epoch} {log_line} elapsed={elapsed:.1f}s")
                        if tb is not None:
                            frames = max(1.0, reduced_stats["frames"])
                            q2 = max(1.0, reduced_stats["q2_triggered"])
                            valid = max(1.0, reduced_stats["valid_slots"])
                            pad_total = valid + reduced_stats["padding_slots"]
                            tb.add_scalar("train/loss_frame", reduced_stats["loss_sum"] / frames, global_step)
                            tb.add_scalar("train/q2_trigger_rate", reduced_stats["q2_triggered"] / frames, global_step)
                            tb.add_scalar("train/q1_rs_acc_window", reduced_stats["q1_rs_correct"] / frames, global_step)
                            tb.add_scalar("train/q1_abnormal_acc_window", reduced_stats["q1_abnormal_correct"] / frames, global_step)
                            tb.add_scalar("train/q2_event_acc_window", reduced_stats["q2_event_correct"] / q2, global_step)
                            tb.add_scalar("train/q2_invalid_output", reduced_stats["q2_invalid_output"], global_step)
                            tb.add_scalar("train/reset_next", reduced_stats["reset_next"], global_step)
                            tb.add_scalar("train/rollout_tokens_per_frame", (reduced_stats["q1_rollout_tokens"] + reduced_stats["q2_rollout_tokens"]) / frames, global_step)
                            tb.add_scalar("ddp/padding_rate", reduced_stats["padding_slots"] / max(1.0, pad_total), global_step)
                            tb.add_scalar("ddp/max_T_global_avg", reduced_stats["max_T_global_sum"] / max(1.0, reduced_stats["batches"]), global_step)
                if rank == 0 and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                    _save_adapter(bundle, output_dir / f"checkpoint-{global_step}", args)
                if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                    break
        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break
    if rank == 0:
        _save_adapter(bundle, output_dir / "final", args)
    if tb is not None:
        tb.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
