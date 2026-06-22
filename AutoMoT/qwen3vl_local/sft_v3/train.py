"""SFT v3 训练入口：sequence memory + 三步 OPD 蒸馏。

实现对应 ``SFT_V3_PLAN.md``：

- 一条 episode 是一个 sub-scenario 时间序列；
- 学生用自己的 step2/step3 输出维护纯文本 memory；
- 每帧依次跑 step1 纯视觉分析、step2 场景判断、可选 step3 状态/子目标判断；
- teacher 与 student 共享同一份 base Qwen，teacher 通过 ``disable_adapter`` 关闭 LoRA；
- 训练只保存 LoRA adapter delta，base checkpoint 始终只读。

本文件里最容易读错的是 KV cache 处理：训练时 student 会先“自由生成”用于更新
memory，然后再把 teacher 生成的 target 用 teacher-forced 方式 append 到同一个
prompt state 上计算 CE。自由生成决定下一帧 memory；teacher-forced target 决定梯度。
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import pathlib
import pickle
import random
import re
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import nn
from torch.distributed.algorithms.join import Join
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.prompt_pipeline import get_full_sequence
from qwen3vl_local.sft_v2.train import (
    _is_vision_module_name,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.engine import _clone_cache
from qwen3vl_local.sft_v3.prompts import (
    DEFAULT_W_ANALYSIS,
    DEFAULT_W_SCENE,
    DEFAULT_W_STATUS,
    DEFAULT_W_SUBGOAL,
    SYSTEM_PROMPT_V3,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step2_teacher_target,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    build_step3_teacher_target,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
    force_memory_to_gt_scene,
    should_trigger_step3,
    init_memory,
    parse_output,
    target_spans_scene,
    target_spans_status,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_event,
)

RGB_FRAME_COUNT = 4
RGB_FRAME_STEP = 1


@dataclass
class EpisodeRow:
    """一条 sub-scenario episode 的轻量元数据。

    ``frame_start/frame_end`` 已经是训练窗口 `[anchor1-delta, anchor3]`，训练和 eval
    都不再扫描 anchor0 或 anchor4 之后的帧，除非 eval 显式使用诊断模式。
    """

    run_id: str
    scenario: str
    anchors: List[int]
    delta: int
    frame_start: int
    frame_end: int
    gt_scene: str
    gt_event_sequence: List[str]
    run_dir: str
    split: str


class EpisodeDataset(Dataset):
    """读取 `build_dataset.py` 生成的 episode jsonl。

    Dataset 只保留元数据，不预读图片。这样 DataLoader 不会持有大量 PIL 对象，也避免
    worker 进程过早触碰 Qwen/CUDA 状态。
    """

    def __init__(self, path: pathlib.Path):
        """加载 episode 索引文件，并把每一行解析成轻量的 `EpisodeRow`。"""

        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"episode jsonl not found: {self.path}")
        self.rows: List[EpisodeRow] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                frame_range = obj.get("frame_range") or [0, -1]
                self.rows.append(
                    EpisodeRow(
                        run_id=str(obj["run_id"]),
                        scenario=str(obj["scenario"]),
                        anchors=[int(x) for x in obj["anchors"]],
                        delta=int(obj["delta"]),
                        frame_start=int(frame_range[0]),
                        frame_end=int(frame_range[1]),
                        gt_scene=str(obj.get("gt_scene", obj["scenario"])),
                        gt_event_sequence=[str(x) for x in obj.get("gt_event_sequence", [])],
                        run_dir=str(obj["run_dir"]),
                        split=str(obj.get("split", "train")),
                    )
                )

    def __len__(self) -> int:
        """返回 episode 数量。"""

        return len(self.rows)

    def __getitem__(self, idx: int) -> EpisodeRow:
        """返回单条 episode 元数据。"""

        return self.rows[idx]


def collate_episode(batch: List[EpisodeRow]) -> List[EpisodeRow]:
    """DataLoader collate：保持 episode 对象列表，不做 tensor stack。"""

    return batch


collate_passthrough = collate_episode


def _build_rgb_paths(run_dir: pathlib.Path, anchor: int) -> List[str]:
    """构造某个 anchor 帧对应的 4 帧历史 RGB 路径。

    这里采用 left-pad 风格：当 anchor-i 小于 0 时回退到 0。路径优先按 `0000.jpg`
    这类标准命名查找；若文件名不完全匹配，再用排序后的 jpg 列表兜底。
    """

    frames = [max(anchor - i * RGB_FRAME_STEP, 0) for i in range(RGB_FRAME_COUNT)]
    ordered = list(reversed(frames))
    rgb_dir = run_dir / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.jpg")) if rgb_dir.exists() else []
    paths: List[str] = []
    for idx in ordered:
        direct = rgb_dir / f"{idx:04d}.jpg"
        if direct.exists():
            paths.append(str(direct))
        elif 0 <= idx < len(rgb_files):
            paths.append(str(rgb_files[idx]))
        else:
            paths.append(str(direct))
    return paths


def _load_images(image_paths: List[str]) -> List[Image.Image]:
    """按路径读取 RGB PIL 图像。"""

    return [Image.open(p).convert("RGB") for p in image_paths]


def _inverse_conversion_2d(point: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    """把 world-frame 点转换到当前 ego frame，与 LeadMoT runner 同款公式。"""

    pt = np.asarray(point, dtype=np.float32).reshape(2)
    tr = np.asarray(translation, dtype=np.float32).reshape(2)
    delta = pt - tr
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    return np.asarray([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]], dtype=np.float32)


def _extract_final_goal_ego_from_meta(meta: Dict[str, Any]) -> Tuple[float, float]:
    """按 LeadMoT 口径取 LEAD 剩余 command route 终点并转 ego frame。

    final_goal 必须来自 ``meta["next_target_points"][-1]``，这是 LEAD 采集器保存的
    剩余 command route 末端；不要退回 ``meta["route"][-1]``，后者只是局部监督片段。
    """

    next_points = np.asarray(meta.get("next_target_points", []), dtype=np.float32)
    if next_points.size == 0:
        raise KeyError("meta missing next_target_points")
    next_points = next_points.reshape(-1, next_points.shape[-1])
    if next_points.shape[-1] < 2:
        raise ValueError(f"next_target_points last dim < 2: shape={next_points.shape}")
    if "pos_global" not in meta:
        raise KeyError("meta missing pos_global")
    if "theta" not in meta:
        raise KeyError("meta missing theta")
    pos_xy = np.asarray(meta["pos_global"], dtype=np.float32).reshape(-1)[:2]
    theta = float(np.asarray(meta["theta"], dtype=np.float32).reshape(-1)[0])
    fg = _inverse_conversion_2d(next_points[-1, :2], pos_xy, theta)
    return float(fg[0]), float(fg[1])


def _load_goal_xy(run_dir: pathlib.Path, frame: int) -> Tuple[float, float]:
    """读取当前帧 ego frame 下的 final destination 坐标。

    严格走 LEAD meta.pkl 里的 ``next_target_points[-1]``——这是与 LeadMoT final_goal
    对齐的来源（PROJECT_CONTEXT 与 PLAN §3.2 都明文要求）。meta 不存在、加载失败或
    缺关键字段都直接抛 RuntimeError，禁止静默 fallback 到 measurements / (0, 0)：
    一旦 memory 里塞了错误 ego_to_goal_xy，"目的地在车体左前 → 大概率左转场景"
    这条核心消歧信号就被污染了，错的训练样本会无声地拉模型偏走。
    """

    meta_path = run_dir / "metas" / f"{frame:04d}.pkl"
    if not meta_path.exists():
        raise RuntimeError(
            f"missing meta.pkl for ego_to_goal_xy: {meta_path} — v3 不允许 fallback，"
            f"请检查数据完备性 (run_dir={run_dir}, frame={frame})"
        )
    try:
        with lzma.open(meta_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load meta.pkl for ego_to_goal_xy: {meta_path} (run_dir={run_dir}, frame={frame})"
        ) from exc
    try:
        return _extract_final_goal_ego_from_meta(meta)
    except Exception as exc:
        raise RuntimeError(
            f"failed to extract final_goal from meta.pkl: {meta_path} (run_dir={run_dir}, frame={frame})"
        ) from exc


def _prefetch_goal_xy_for_next_frame(
    memory: Any,
    run_dir: pathlib.Path,
    next_frame: int,
    frame_end: int,
) -> None:
    """在帧末为下一帧预取 ego_to_goal_xy。

    C3 收敛后的口径：当前帧 prompt 使用进入本帧前已经写入 memory 的坐标；本帧结束、
    status/subgoal 更新完成后，再读取下一帧 meta 并写入 memory。这样实现与 PLAN
    §3.3 “进入下一帧前重算 ego_to_goal_xy”的字面语义一致。
    """

    if next_frame <= frame_end:
        gx, gy = _load_goal_xy(run_dir, next_frame)
        memory.ego_to_goal_x = gx
        memory.ego_to_goal_y = gy


def _gt_status_subgoal(ep: EpisodeRow, frame: int) -> Tuple[str, str]:
    """根据 frame 所在 anchor 区间计算 GT status/subgoal。

    status 是当前阶段事件；subgoal 是当前阶段之后的下一事件。比如处在 subgoal1 到
    subgoal2 之间时，status=seq[1]，subgoal=seq[2]。
    """

    seq = list(ep.gt_event_sequence) if ep.gt_event_sequence else list(get_full_sequence(ep.gt_scene))
    if len(seq) < 5:
        seq = list(get_full_sequence(ep.gt_scene))
    _, f1, f2, f3, f4 = ep.anchors
    if frame < f1:
        status = seq[0]
    elif frame < f2:
        status = seq[1]
    elif frame < f3:
        status = seq[2]
    elif frame < f4:
        status = seq[3]
    else:
        status = seq[4]
    try:
        idx = seq.index(status)
    except ValueError:
        idx = 0
    subgoal = seq[idx + 1] if idx + 1 < len(seq) else seq[-1]
    return status, subgoal


def _is_phase_a(ep: EpisodeRow, frame: int) -> bool:
    """判断某帧是否位于 Phase A `[f1-delta, f1+delta]`。"""

    f1 = ep.anchors[1]
    return (f1 - ep.delta) <= frame <= (f1 + ep.delta)


def _build_messages_with_images(
    *,
    user_text: str,
    images: List[Image.Image],
    prev_turns: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """构造 Qwen chat messages：system + 首个带 4 张图的 user turn。

    step1 是唯一带图的 user turn；step2/step3 后续只通过 KV cache 追加文本 user turn，
    不重复传图，避免重 prefill。
    """

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_V3}]
    content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content})
    if prev_turns:
        messages.extend(prev_turns)
    return messages


def _collect_images_from_messages(messages: List[Dict[str, Any]]) -> Optional[List[Image.Image]]:
    """从 structured chat message 中收集 PIL 图像，供 processor 编码。"""

    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image = item.get("image")
                    if isinstance(image, Image.Image):
                        images.append(image)
    return images or None


_LABEL_LINE_RE = re.compile(r"^\s*(SCENE|STATUS|SUBGOAL)\s*:", re.MULTILINE | re.IGNORECASE)


def _analysis_char_end(text: str) -> int:
    """返回分析段结束位置，即第一条 `SCENE/STATUS/SUBGOAL:` 标签行之前。"""

    match = _LABEL_LINE_RE.search(text)
    return match.start() if match else len(text)


def _analysis_before_labels(text: str) -> str:
    """截取标签行之前的分析文本；空分析时给一个兜底句。"""

    end = _analysis_char_end(text or "")
    analysis = (text or "")[:end].strip()
    return analysis or "I observe the current driving scene from the images."


@dataclass
class KVState:
    """一条不断增长的 Qwen chat 序列的 KV cache 状态。

    - ``decoded_input_ids``：当前完整 token 序列；
    - ``cache_input_ids``：已经进入 past_key_values 的前缀；
    - ``next_logits``：预测下一个 token 的 logits，用来计算第一个 append token 的 CE；
    - ``rope_deltas``：Qwen3-VL remote-code 需要的多模态 RoPE 偏移。
    """

    decoded_input_ids: torch.Tensor
    cache_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Any
    rope_deltas: Any
    next_logits: torch.Tensor


def _clone_kv_state(state: KVState) -> KVState:
    """复制 KVState，避免 teacher/student 分支共享同一个 mutable cache。"""

    return KVState(
        decoded_input_ids=state.decoded_input_ids.detach().clone(),
        cache_input_ids=state.cache_input_ids.detach().clone(),
        attention_mask=state.attention_mask.detach().clone(),
        past_key_values=_clone_cache(state.past_key_values),
        rope_deltas=state.rope_deltas.detach().clone() if hasattr(state.rope_deltas, "detach") else state.rope_deltas,
        next_logits=state.next_logits.detach().clone(),
    )


def _kv_start_state(bundle: Any, messages: List[Dict[str, Any]]) -> KVState:
    """对 system + 图像 + step1 user 做一次 prefill，得到初始 KVState。"""

    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(
        text=[text],
        images=_collect_images_from_messages(messages),
        return_tensors="pt",
        padding=True,
    )
    kwargs: Dict[str, Any] = {
        "input_ids": inputs["input_ids"].to(bundle.device),
        "attention_mask": inputs["attention_mask"].to(bundle.device),
    }
    for key, value in inputs.items():
        if key in ("input_ids", "attention_mask"):
            continue
        kwargs[key] = value.to(bundle.device) if isinstance(value, torch.Tensor) else value
    outputs = bundle.model(**kwargs, use_cache=True, return_dict=True)
    input_ids = kwargs["input_ids"]
    attention_mask = kwargs["attention_mask"]
    return KVState(
        decoded_input_ids=input_ids,
        cache_input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=getattr(outputs, "rope_deltas", None),
        next_logits=outputs.logits[:, -1, :],
    )


def _append_token_ids(
    bundle: Any,
    state: KVState,
    suffix_ids: torch.Tensor,
    *,
    assistant_text: str = "",
    span_fn: Any | None = None,
    analysis_enabled: bool = False,
) -> Tuple[KVState, Dict[str, torch.Tensor]]:
    """把一段 token 追加到 KVState，并可同时计算 assistant target 的分项 loss。

    关键点：
    1. ``state.next_logits`` 对应第一个追加 token 的预测；
    2. 后续 token 的预测来自本次 forward 的 ``outputs.logits[:, :-1]``；
    3. ``analysis_enabled=False`` 时分析 token loss 置 0，但 scene/status/subgoal 值
       token loss 仍照算，用于 GT 泄露过滤。
    """

    if suffix_ids.ndim == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    prefix_ids = state.cache_input_ids
    decoded_ids = state.decoded_input_ids.to(prefix_ids.device)
    pending_ids = decoded_ids[:, prefix_ids.shape[1] :]
    feed_ids = torch.cat([pending_ids, suffix_ids], dim=1) if pending_ids.numel() else suffix_ids
    zero = state.next_logits.sum() * 0.0
    parts = {"analysis": zero, "scene": zero, "status": zero, "subgoal": zero}
    if feed_ids.shape[1] == 0:
        return state, parts

    old_attention = state.attention_mask.to(prefix_ids.device)
    attention_mask = torch.cat(
        [old_attention, torch.ones_like(feed_ids, device=old_attention.device)],
        dim=1,
    )
    decoded_input_ids = torch.cat([prefix_ids, feed_ids], dim=1)
    prefix_len = int(prefix_ids.shape[1])
    cache_position = torch.arange(prefix_len, prefix_len + feed_ids.shape[1], device=prefix_ids.device)
    model_inputs = bundle.unwrap().prepare_inputs_for_generation(
        decoded_input_ids,
        past_key_values=state.past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )
    if state.rope_deltas is not None and "rope_deltas" not in model_inputs:
        model_inputs["rope_deltas"] = state.rope_deltas
    outputs = bundle.model(**model_inputs, return_dict=True)

    pred_logits = torch.cat([state.next_logits.unsqueeze(1), outputs.logits[:, :-1, :]], dim=1)
    pending_len = int(pending_ids.shape[1])

    def token_loss(feed_positions: List[int]) -> torch.Tensor:
        """对 feed_ids 中指定位置计算平均 CE。"""

        if not feed_positions:
            return zero
        idx = torch.tensor(feed_positions, device=pred_logits.device, dtype=torch.long)
        labels = feed_ids[:, idx].reshape(-1)
        logits = pred_logits[:, idx, :].reshape(-1, pred_logits.shape[-1])
        return F.cross_entropy(logits, labels, reduction="mean")

    if assistant_text and span_fn is not None:
        enc = bundle.tokenizer(assistant_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        spans = span_fn(assistant_text)
        analysis_end = _analysis_char_end(assistant_text)
        if analysis_enabled:
            positions = [
                pending_len + j
                for j, (lo, hi) in enumerate(offsets)
                if hi > 0 and lo < analysis_end and assistant_text[lo:hi].strip()
            ]
            parts["analysis"] = token_loss(positions)
        for key, (span_lo, span_hi) in spans.items():
            positions = [
                pending_len + j
                for j, (lo, hi) in enumerate(offsets)
                if lo < span_hi and hi > span_lo
            ]
            if key in parts:
                parts[key] = token_loss(positions)

    new_state = KVState(
        decoded_input_ids=decoded_input_ids,
        cache_input_ids=decoded_input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        rope_deltas=getattr(outputs, "rope_deltas", state.rope_deltas),
        next_logits=outputs.logits[:, -1, :],
    )
    return new_state, parts


def _append_text(
    bundle: Any,
    state: KVState,
    text: str,
    *,
    span_fn: Any | None = None,
    analysis_enabled: bool = False,
) -> Tuple[KVState, Dict[str, torch.Tensor]]:
    """把文本 tokenized 后追加到 KVState。"""

    enc = bundle.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return _append_token_ids(
        bundle,
        state,
        enc["input_ids"],
        assistant_text=text,
        span_fn=span_fn,
        analysis_enabled=analysis_enabled,
    )


def _render_user_suffix(bundle: Any, user_text: str) -> str:
    """把后续 user turn 渲染成 chat template 文本片段。"""

    suffix = bundle.processor.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return suffix if suffix.startswith("\n") else "\n" + suffix


def _append_user_turn(bundle: Any, state: KVState, user_text: str) -> KVState:
    """在已有 assistant 输出后追加一个新的 user turn。

    如果上一轮没有 pending assistant token，需要先补 `<|im_end|>` 关闭 assistant turn；
    如果上一轮自由生成产生了 pending token，则一起送入 cache，保证后续 KV 与完整聊天
    历史一致。
    """

    pending_len = int(state.decoded_input_ids.shape[1] - state.cache_input_ids.shape[1])
    close_assistant = "" if pending_len > 0 else "<|im_end|>"
    new_state, _ = _append_text(bundle, state, close_assistant + _render_user_suffix(bundle, user_text))
    return new_state


def _eos_token_ids(bundle: Any) -> set[int]:
    """收集模型可用的 EOS / `<|im_end|>` token id。"""

    ids: set[int] = set()
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))
    im_end = bundle.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.add(int(im_end))
    return ids


_REPETITION_PENALTY = 1.05


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """对 ``seen_ids`` 中出现过的 token 施加 HF 风格 repetition penalty。

    与 transformers 的 RepetitionPenaltyLogitsProcessor 等价：正分除以 penalty、
    负分乘以 penalty，降低重复 token 的概率。teacher 端 greedy 生成 80 token 的开销
    可忽略，主要避免 step1 出现 "I see I see ..." 之类的复读，污染 L_A1 的目标分布。
    """

    if penalty == 1.0 or seen_ids.numel() == 0:
        return logits
    out = logits.clone()
    scores = out.index_select(-1, seen_ids)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    out.index_copy_(-1, seen_ids, scores)
    return out


def _kv_generate_text(
    bundle: Any,
    state: KVState,
    max_new_tokens: int,
    *,
    repetition_penalty: float = _REPETITION_PENALTY,
) -> Tuple[str, KVState]:
    """从已有 KVState 贪心生成一段 assistant 文本，并返回生成后的状态。

    框架明确要求 ``do_sample=False`` + ``repetition_penalty=1.05``；这里 argmax 已等价
    于 do_sample=False，再额外按 repetition_penalty 对前文 token 做惩罚。前文范围
    取 ``state.decoded_input_ids`` ∪ 已生成 token，与 HF generate 默认行为一致。
    """

    generated: List[torch.Tensor] = []
    eos_ids = _eos_token_ids(bundle)
    cur = state
    device = cur.next_logits.device
    seen_unique = torch.unique(cur.decoded_input_ids.reshape(-1).to(device))
    for _ in range(max_new_tokens):
        logits = _apply_repetition_penalty(cur.next_logits, seen_unique, repetition_penalty)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token)
        token_id = int(next_token.reshape(-1)[0].item())
        if token_id in eos_ids:
            break
        cur, _ = _append_token_ids(bundle, cur, next_token)
        # 把新 token 也纳入 repetition 集合；torch.unique 在已排序集合上插入仍是 O(N)
        seen_unique = torch.unique(torch.cat([seen_unique, next_token.reshape(-1).to(device)], dim=0))
    if not generated:
        return "", cur
    ids = torch.cat(generated, dim=1)
    text = bundle.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text, cur


def _assistant_loss_from_state(
    bundle: Any,
    state: KVState,
    assistant_target: str,
    span_fn: Any,
    *,
    analysis_enabled: bool,
) -> Dict[str, torch.Tensor]:
    """从某个 prompt state 出发，对 assistant target 做 teacher-forced loss。"""

    _, parts = _append_text(
        bundle,
        state,
        assistant_target,
        span_fn=span_fn,
        analysis_enabled=analysis_enabled,
    )
    return parts


def _teacher_start_state(bundle: Any, messages: List[Dict[str, Any]]) -> KVState:
    """teacher prefill 包装：只读 no_grad。"""

    with torch.no_grad():
        return _kv_start_state(bundle, messages)


def _student_start_state(bundle: Any, messages: List[Dict[str, Any]]) -> KVState:
    """student prefill 包装：当前实现也 no_grad，训练梯度只来自 LoRA 文本续写段。"""

    with torch.no_grad():
        return _kv_start_state(bundle, messages)


def _teacher_generate_kv(bundle: Any, state: KVState, max_new_tokens: int) -> Tuple[str, KVState]:
    """teacher 贪心生成包装：关闭梯度。"""

    with torch.no_grad():
        return _kv_generate_text(bundle, state, max_new_tokens)


def _student_generate_kv(bundle: Any, state: KVState, max_new_tokens: int) -> Tuple[str, KVState]:
    """student 自由生成包装：关闭梯度，只用于更新 memory。"""

    with torch.no_grad():
        return _kv_generate_text(bundle, state, max_new_tokens)


@dataclass
class FrameLossPack:
    """单帧三步内循环产生的 loss 与诊断标志。"""

    a1: torch.Tensor
    a2: torch.Tensor
    s2: torch.Tensor
    a3: torch.Tensor
    s3_status: torch.Tensor
    s3_subgoal: torch.Tensor
    step3_ran: bool
    scene_flip: bool
    leak2: bool
    leak3: bool
    phase_a: bool


def iter_episode_loss_packs(
    bundle: Any,
    ep: EpisodeRow,
    *,
    max_length: int = 8192,
    outer_stride: int = 1,
) -> Iterable[FrameLossPack]:
    """逐帧执行一条 episode，并 yield 每帧 loss pack。

    这是 v3 训练的核心状态机：
    1. 帧开头根据 Phase 决定是否强制 memory.scene=GT；
    2. step1 只看图，student 对齐 teacher 的视觉分析；
    3. step2 用学生当前 memory 判断场景，学生自由生成结果更新 memory；
    4. 若 step2 后 memory.scene==GT scene，才进入 step3 训练 status/subgoal；
    5. 自由生成决定 memory 走向，teacher-forced target 决定梯度。

    KV cache 只在单帧三步内复用，不跨帧；跨帧连续性只来自纯文本 memory。
    """

    del max_length  # processor truncation is intentionally not enabled for Qwen image chats.
    run_dir = pathlib.Path(ep.run_dir)
    gx, gy = _load_goal_xy(run_dir, ep.frame_start)
    memory = init_memory(
        run_id=ep.run_id,
        sub_scenario_id=f"{ep.run_id}:{ep.anchors[1]}",
        ego_to_goal_x=gx,
        ego_to_goal_y=gy,
        gt_scene=ep.gt_scene,
    )

    stride = max(1, outer_stride)
    for frame in range(ep.frame_start, ep.frame_end + 1, stride):
        phase_a = _is_phase_a(ep, frame)
        if not phase_a:
            # Phase B：每帧开始先把 scene 修回 GT，再学习“对的别改”。
            memory = force_memory_to_gt_scene(memory, gt_scene=ep.gt_scene)

        image_paths = _build_rgb_paths(run_dir, frame)
        try:
            images = _load_images(image_paths)
        except Exception:
            _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)
            continue

        gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

        step1_user = build_step1_user_prompt(len(images))
        step1_msgs = _build_messages_with_images(user_text=step1_user, images=images)
        step2_teacher_user = build_step2_teacher_prompt(memory, ep.gt_scene)
        step2_student_user = build_step2_student_prompt(memory)

        # Teacher 分支：关闭 LoRA，用 base Qwen 生成 hindsight 分析。
        teacher_model = bundle.unwrap()
        teacher_was_training = bool(teacher_model.training)
        teacher_model.eval()
        with teacher_model.disable_adapter():
            teacher_step1_prompt_state = _teacher_start_state(bundle, step1_msgs)
            teacher_step1, teacher_step1_state = _teacher_generate_kv(
                bundle, _clone_kv_state(teacher_step1_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP1
            )
            teacher_step1 = teacher_step1 or "I observe the current driving scene from the images."
            with torch.no_grad():
                teacher_step2_prompt_state = _append_user_turn(bundle, teacher_step1_state, step2_teacher_user)
            raw_teacher_step2, teacher_step2_state = _teacher_generate_kv(
                bundle, _clone_kv_state(teacher_step2_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP2
            )
        if teacher_was_training:
            teacher_model.train()

        # Student 自由生成分支：只用于更新 memory，不直接反传。
        student_was_training = bool(bundle.model.training)
        bundle.model.eval()
        student_step1_prompt_state = _student_start_state(bundle, step1_msgs)
        student_step1, student_step1_state = _student_generate_kv(
            bundle, _clone_kv_state(student_step1_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP1
        )
        student_step1 = student_step1 or teacher_step1
        with torch.no_grad():
            student_step2_prompt_state = _append_user_turn(bundle, student_step1_state, step2_student_user)
        raw_student_step2, student_step2_state = _student_generate_kv(
            bundle, _clone_kv_state(student_step2_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP2
        )
        if student_was_training:
            bundle.model.train()

        # 真正的 step1 梯度：student prompt state + teacher 文本 target。
        step1_parts = _assistant_loss_from_state(
            bundle,
            _clone_kv_state(student_step1_prompt_state),
            teacher_step1,
            lambda _text: {},
            analysis_enabled=True,
        )
        analysis2 = _analysis_before_labels(raw_teacher_step2)
        leak2 = check_gt_leak_scene(analysis2, ep.gt_scene)
        target2 = build_step2_teacher_target(analysis2, ep.gt_scene)
        step2_parts = _assistant_loss_from_state(
            bundle,
            _clone_kv_state(student_step2_prompt_state),
            target2,
            target_spans_scene,
            analysis_enabled=not leak2,
        )
        pred2 = parse_output(raw_student_step2)
        old_scene = memory.scene
        memory = update_memory_after_step2(memory, student_scene=pred2.get("scene"))
        scene_flip = memory.scene != old_scene

        zero = step1_parts["analysis"] * 0.0
        a3 = zero
        s3_status = zero
        s3_subgoal = zero
        leak3 = False
        step3_ran = should_trigger_step3(memory_scene_after_step2=memory.scene, gt_scene=ep.gt_scene)

        if step3_ran:
            # 只有场景已经正确时才训 status/subgoal；错误场景下 event 字典不同，跳过。
            step3_teacher_user = build_step3_teacher_prompt(memory, gt_status, gt_subgoal)
            teacher_was_training = bool(teacher_model.training)
            teacher_model.eval()
            with teacher_model.disable_adapter():
                with torch.no_grad():
                    teacher_step3_prompt_state = _append_user_turn(bundle, teacher_step2_state, step3_teacher_user)
                raw_teacher_step3, _teacher_step3_state = _teacher_generate_kv(
                    bundle, _clone_kv_state(teacher_step3_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP3
                )
            if teacher_was_training:
                teacher_model.train()
            analysis3 = _analysis_before_labels(raw_teacher_step3)
            leak3 = check_gt_leak_status_subgoal(analysis3, gt_status, gt_subgoal)
            target3 = build_step3_teacher_target(analysis3, gt_status, gt_subgoal)

            step3_student_user = build_step3_student_prompt(memory)
            student_was_training = bool(bundle.model.training)
            bundle.model.eval()
            with torch.no_grad():
                student_step3_prompt_state = _append_user_turn(bundle, student_step2_state, step3_student_user)
            raw_student_step3, _student_step3_state = _student_generate_kv(
                bundle, _clone_kv_state(student_step3_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP3
            )
            if student_was_training:
                bundle.model.train()
            step3_parts = _assistant_loss_from_state(
                bundle,
                _clone_kv_state(student_step3_prompt_state),
                target3,
                target_spans_status,
                analysis_enabled=not leak3,
            )
            a3 = step3_parts["analysis"]
            s3_status = step3_parts["status"]
            s3_subgoal = step3_parts["subgoal"]

            pred3 = parse_output(raw_student_step3)
            pred_status = pred3.get("status") if validate_event(memory.scene, pred3.get("status")) else None
            pred_subgoal = pred3.get("subgoal") if validate_event(memory.scene, pred3.get("subgoal")) else None
            memory = update_memory_after_step3(
                memory,
                student_status=pred_status,
                student_subgoal=pred_subgoal,
            )

        pack = FrameLossPack(
            a1=step1_parts["analysis"],
            a2=step2_parts["analysis"],
            s2=step2_parts["scene"],
            a3=a3,
            s3_status=s3_status,
            s3_subgoal=s3_subgoal,
            step3_ran=step3_ran,
            scene_flip=scene_flip,
            leak2=leak2,
            leak3=leak3,
            phase_a=phase_a,
        )
        _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)
        yield pack


def compute_episode_losses(
    bundle: Any,
    ep: EpisodeRow,
    *,
    max_length: int = 8192,
    outer_stride: int = 1,
) -> Tuple[List[FrameLossPack], int, int]:
    """收集一条 episode 的所有 frame loss，并返回漏帧统计。

    漏帧通常来自 RGB 路径不存在或图片读取失败；训练日志保留这个值，方便排查数据根
    目录或 run_dir 组织是否正确。
    """

    packs = list(
        iter_episode_loss_packs(
            bundle,
            ep,
            max_length=max_length,
            outer_stride=outer_stride,
        )
    )
    expected = len(range(ep.frame_start, ep.frame_end + 1, max(1, outer_stride)))
    return packs, max(0, expected - len(packs)), expected


def _safe_ratio(num: float, den: float) -> float:
    """安全除法，分母为 0 时返回 0。"""

    return num / den if den > 0 else 0.0


def _to_float(value: torch.Tensor) -> float:
    """把标量 tensor 转成 Python float，用于日志和 TensorBoard。"""

    return float(value.detach().item())


def _save_adapter_config(path: pathlib.Path, bundle: Any, args: argparse.Namespace) -> None:
    """保存 v3 adapter 的附加配置。

    PEFT 自带配置不知道 v3 的 loss 权重、outer stride 和视觉 LoRA 保险参数；这里写
    `sft_v3_adapter_config.json`，让 eval/probe 和后续审计能识别 checkpoint 口径。
    """

    targets = list(bundle.lora_target_modules)
    vision_targets = [x for x in targets if _is_vision_module_name(x)]
    payload = {
        "schema_version": 1,
        "route": "sft_v3_sequence",
        "base_model_dir": str(args.model_dir),
        "base_model_mutated": False,
        "lora_vision_scope": bundle.lora_vision_scope,
        "lora_vision": bool(vision_targets),
        "target_modules": targets,
        "strict_vision_scope": bool(args.strict_vision_scope),
        "loss_weights": {
            "a1": float(args.w_a1),
            "a2": float(args.w_a2),
            "a3": float(args.w_a3),
            "s2": float(args.w_s2),
            "s3_status": float(args.w_s3_status),
            "s3_subgoal": float(args.w_s3_subgoal),
        },
        "outer_stride": int(args.outer_stride),
        "phase_b_force_gt_scene": True,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "sft_v3_adapter_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


save_sft_v3_adapter_config = _save_adapter_config


@torch.no_grad()
def evaluate_quick(bundle: Any, loader: DataLoader, args: argparse.Namespace) -> Dict[str, float]:
    """训练中的轻量 validation，只算 teacher-forced quick loss。

    它不是完整自由生成 eval，不报告 scene recovery 等 memory 指标；完整指标请训练后
    单独运行 `eval.py`。DDP 下该路径被禁用。
    """

    was_training = bool(bundle.model.training)
    bundle.model.eval()
    total = 0.0
    frames = 0
    episodes = 0
    for batch in loader:
        for ep in batch:
            packs, _, _ = compute_episode_losses(
                bundle,
                ep,
                max_length=args.max_length,
                outer_stride=args.outer_stride,
            )
            for pack in packs:
                loss = (
                    args.w_a1 * pack.a1
                    + args.w_a2 * pack.a2
                    + args.w_s2 * pack.s2
                    + args.w_a3 * pack.a3
                    + args.w_s3_status * pack.s3_status
                    + args.w_s3_subgoal * pack.s3_subgoal
                )
                total += _to_float(loss)
                frames += 1
            episodes += 1
            if args.max_eval_episodes > 0 and episodes >= args.max_eval_episodes:
                break
        if args.max_eval_episodes > 0 and episodes >= args.max_eval_episodes:
            break
    if was_training:
        bundle.model.train()
    return {
        "loss": total / max(frames, 1),
        "episodes": float(episodes),
        "frames": float(frames),
    }


def parse_args() -> argparse.Namespace:
    """解析训练参数；通常由 `train.sh` 负责填默认值。"""

    parser = argparse.ArgumentParser(description="Train SFT v3 sequence LoRA")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--outer-stride", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-vision-scope", choices=["off", "merger", "last4", "all"], default="off")
    parser.add_argument("--lora-vision", action="store_true")
    parser.add_argument("--vision-lr-scale", type=float, default=0.1)
    parser.add_argument("--max-vision-lr-scale", type=float, default=0.25)
    parser.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-clip-norm", type=float, default=1.0)
    parser.add_argument("--vision-clip-norm", type=float, default=0.3)
    parser.add_argument("--vision-guard-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-guard-grad-norm-max", type=float, default=10.0)
    parser.add_argument("--vision-guard-param-norm-max", type=float, default=200.0)
    parser.add_argument("--vision-guard-patience", type=int, default=3)
    parser.add_argument("--w-a1", type=float, default=DEFAULT_W_ANALYSIS)
    parser.add_argument("--w-a2", type=float, default=DEFAULT_W_ANALYSIS)
    parser.add_argument("--w-a3", type=float, default=DEFAULT_W_ANALYSIS)
    parser.add_argument("--w-s2", type=float, default=DEFAULT_W_SCENE)
    parser.add_argument("--w-s3-status", type=float, default=DEFAULT_W_STATUS)
    parser.add_argument("--w-s3-subgoal", type=float, default=DEFAULT_W_SUBGOAL)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """训练参数一致性检查。

    DDP + eval_steps>0 会直接报错，避免用户以为多卡训练中已经跑了 in-loop eval。
    """

    if args.outer_stride <= 0:
        raise ValueError("--outer-stride must be > 0")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.vision_lr_scale < 0:
        raise ValueError("--vision-lr-scale must be >= 0")
    if args.vision_lr_scale > args.max_vision_lr_scale:
        raise ValueError("--vision-lr-scale exceeds --max-vision-lr-scale")
    if args.vision_guard_patience <= 0:
        raise ValueError("--vision-guard-patience must be > 0")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and args.eval_steps > 0:
        raise ValueError(
            "SFT v3 does not run in-loop eval under DDP. Set --eval-steps 0 and run "
            "qwen3vl_local/sft_v3/eval.py after training."
        )


def _param_norm(params: List[nn.Parameter]) -> float:
    """计算参数组 L2 norm，用于视觉 LoRA 熔断监控。"""

    with torch.no_grad():
        return math.sqrt(sum(float(p.detach().float().norm().item()) ** 2 for p in params))


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torch.distributed，返回 rank/world_size/local_rank。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """销毁 torch.distributed 进程组。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """判断当前进程是否 rank0。"""

    return rank == 0


def _barrier() -> None:
    """DDP barrier；单进程时为空操作。"""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce_sum(values: List[float], device: torch.device) -> List[float]:
    """对一组 float 做 DDP sum all-reduce。"""

    if not (dist.is_available() and dist.is_initialized()):
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _all_reduce_max_int(value: int, device: torch.device) -> int:
    """对整数做 DDP max all-reduce，用于同步 schedule 步数。"""

    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _sync_bool_or(value: bool, device: torch.device) -> bool:
    """跨 rank 同步布尔 OR；任一 rank 触发熔断则全体停止。"""

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


def _shard_episode_dataset(ds: EpisodeDataset, *, rank: int, world_size: int) -> None:
    """按 episode 做 rank 分片，避免 DistributedSampler padding 复制 episode。"""

    if world_size > 1:
        ds.rows = ds.rows[rank::world_size]


def main() -> None:
    """SFT v3 训练主入口。

    主循环按 episode -> frame 两层推进。默认每帧 backward/step 一次，避免显存随
    episode 长度增长；如果用户显式设置 `grad_accum>1`，只累计梯度，memory 仍按
    学生自由生成逐帧实时更新。
    """

    args = parse_args()
    validate_args(args)
    rank, world_size, local_rank = setup_distributed()
    if world_size > 1 and args.grad_accum != 1:
        raise ValueError("SFT v3 DDP requires --grad-accum 1 so all ranks step once per frame.")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if args.lora_vision and args.lora_vision_scope == "off":
        args.lora_vision_scope = "all"

    output_dir = pathlib.Path(args.output_dir)
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] output_dir={output_dir} rank={rank} world_size={world_size} local_rank={local_rank}")
    _barrier()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # SFT v3 的 loss forward 依赖 prefill 出来的 past_key_values（prepare_inputs_for_generation
    # 会把 input_ids 切成 suffix-only），而 HF 在 training=True + gradient_checkpointing=True
    # 时会强行把 use_cache 改 False、past_key_values 改 None。如果让 grad_checkpointing 打开，
    # loss forward 拿到的是没有 prefix 上下文的 suffix-only 序列，loss 静默错位、训练学不到东西。
    # test_kv_reuse.py 验证 KV 复用时也强制要求 gradient_checkpointing=False。
    # 因此这里硬关：`--no-grad-checkpoint` 保留只是兼容 CLI，不再实际控制行为。
    if not args.no_grad_checkpoint and is_rank0(rank):
        print("[warn] sft_v3 forces gradient_checkpointing=False (incompatible with KV-reuse loss design)")
    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_vision_scope=args.lora_vision_scope,
        strict_vision_scope=bool(args.strict_vision_scope),
        gradient_checkpointing=False,
    )
    if world_size > 1:
        # broadcast_buffers=False：Qwen3-VL 的 RoPE inv_freq 等 buffer 由 config 确定性
        # 算出，每个 rank 完全一致，不需要 DDP 同步。新版 torch 的 _sync_buffers 在
        # find_unused_parameters=True + 没有真正 unused param 时会让 _find_common_rank
        # 返回非法 authoritative_rank，进 _broadcast_coalesced 时直接 TypeError。
        # find_unused_parameters=False：LoRA 训练下 trainable 参数都进计算图，警告也
        # 明示"did not find any unused parameters"，关掉省去多余的 autograd 图遍历。
        bundle.model = torch.nn.parallel.DistributedDataParallel(
            bundle.model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    train_ds = EpisodeDataset(pathlib.Path(args.train_jsonl))
    val_ds = EpisodeDataset(pathlib.Path(args.val_jsonl)) if args.val_jsonl else None
    train_total = len(train_ds.rows)
    val_total = len(val_ds.rows) if val_ds is not None else 0
    _shard_episode_dataset(train_ds, rank=rank, world_size=world_size)
    if val_ds is not None:
        _shard_episode_dataset(val_ds, rank=rank, world_size=world_size)
    if is_rank0(rank):
        print(f"[data] train_episodes={train_total} val_episodes={val_total} per_rank_batch={args.per_device_batch_size}")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_episode,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_episode)
        if val_ds is not None
        else None
    )

    language_params: List[nn.Parameter] = []
    vision_params: List[nn.Parameter] = []
    # 与 v2 一致，把 LoRA 参数按语言侧/视觉侧分组，视觉侧用更小 LR 和独立裁剪。
    for name, param in bundle.unwrap().named_parameters():
        if not param.requires_grad:
            continue
        if _is_vision_module_name(name):
            vision_params.append(param)
        else:
            language_params.append(param)
    groups: List[Dict[str, Any]] = []
    if language_params:
        groups.append({"params": language_params, "lr": args.learning_rate, "name": "language"})
    if vision_params:
        groups.append(
            {
                "params": vision_params,
                "lr": args.learning_rate * float(args.vision_lr_scale),
                "name": "vision",
            }
        )
    if not groups:
        raise RuntimeError("no trainable LoRA params found")
    if is_rank0(rank):
        print(
            f"[opt] language_params={len(language_params)} vision_params={len(vision_params)} "
            f"grad_accum={args.grad_accum}"
        )

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # schedule 用“每帧一个优化单位”估算；DDP 下取各 rank 最大帧数，避免某些 rank
    # 先耗尽数据后 scheduler 步数不一致。
    estimated_frames = sum(
        len(range(ep.frame_start, ep.frame_end + 1, max(1, args.outer_stride))) for ep in train_ds.rows
    )
    estimated_frames_for_schedule = _all_reduce_max_int(estimated_frames, device)
    steps_per_epoch = max(1, math.ceil(estimated_frames_for_schedule / max(1, args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * max(1, args.num_epochs)
    if args.check:
        total_steps = min(total_steps, 2)
        args.max_steps = total_steps
    scheduler = make_scheduler(optimizer, total_steps, int(total_steps * args.warmup_ratio))
    tb = SummaryWriter(log_dir=str(output_dir / "tb")) if (is_rank0(rank) and _TB_AVAILABLE) else None
    global_step = 0
    accum_steps = 0
    accum_loss = 0.0
    stats = {
        "frames": 0,
        "step3": 0,
        "flip": 0,
        "leak2": 0,
        "leak3": 0,
        "phase_a": 0,
        "loss_a1": 0.0,
        "loss_a2": 0.0,
        "loss_a3": 0.0,
        "loss_s2": 0.0,
        "loss_s3_status": 0.0,
        "loss_s3_subgoal": 0.0,
    }
    guard_bad_steps = 0
    fuse_stopped = False
    saved: List[pathlib.Path] = []
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    join_context = Join([bundle.model]) if world_size > 1 else nullcontext()
    with join_context:
        for epoch in range(args.num_epochs):
            for batch in train_loader:
                for ep in batch:
                    if fuse_stopped or (args.max_steps > 0 and global_step >= args.max_steps):
                        break
                    for pack in iter_episode_loss_packs(
                        bundle,
                        ep,
                        max_length=args.max_length,
                        outer_stride=args.outer_stride,
                    ):
                        if fuse_stopped or (args.max_steps > 0 and global_step >= args.max_steps):
                            break
                        loss = (
                            args.w_a1 * pack.a1
                            + args.w_a2 * pack.a2
                            + args.w_s2 * pack.s2
                            + args.w_a3 * pack.a3
                            + args.w_s3_status * pack.s3_status
                            + args.w_s3_subgoal * pack.s3_subgoal
                        )
                        # 每帧结束就反传；这正对应“memory 已经被本帧学生输出更新，
                        # 下一帧继续用新 memory”的在线递推训练口径。
                        (loss / args.grad_accum).backward()
                        accum_loss += _to_float(loss)
                        accum_steps += 1
                        stats["frames"] += 1
                        stats["step3"] += int(pack.step3_ran)
                        stats["flip"] += int(pack.scene_flip)
                        stats["leak2"] += int(pack.leak2)
                        stats["leak3"] += int(pack.leak3)
                        stats["phase_a"] += int(pack.phase_a)
                        stats["loss_a1"] += _to_float(pack.a1)
                        stats["loss_a2"] += _to_float(pack.a2)
                        stats["loss_a3"] += _to_float(pack.a3)
                        stats["loss_s2"] += _to_float(pack.s2)
                        stats["loss_s3_status"] += _to_float(pack.s3_status)
                        stats["loss_s3_subgoal"] += _to_float(pack.s3_subgoal)

                        if accum_steps < args.grad_accum:
                            continue

                        lang_norm = (
                            torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm))
                            if language_params
                            else torch.tensor(0.0, device=device)
                        )
                        vis_norm = (
                            torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm))
                            if vision_params
                            else torch.tensor(0.0, device=device)
                        )
                        vis_grad = float(vis_norm)
                        vis_param_norm = _param_norm(vision_params) if vision_params else 0.0
                        local_fuse = False
                        if args.vision_guard_enabled and vision_params:
                            bad = (
                                (not math.isfinite(vis_grad))
                                or vis_grad > float(args.vision_guard_grad_norm_max)
                                or (not math.isfinite(vis_param_norm))
                                or vis_param_norm > float(args.vision_guard_param_norm_max)
                            )
                            guard_bad_steps = guard_bad_steps + 1 if bad else 0
                            local_fuse = guard_bad_steps >= args.vision_guard_patience
                        if local_fuse:
                            fuse_stopped = True
                            if is_rank0(rank):
                                emergency = output_dir / f"fuse_stop_step_{global_step + 1}"
                                emergency.mkdir(parents=True, exist_ok=True)
                                bundle.unwrap().save_pretrained(str(emergency))
                                _save_adapter_config(emergency, bundle, args)
                                (emergency / "fuse_reason.txt").write_text(
                                    (
                                        "vision fuse triggered\n"
                                        f"grad={vis_grad:.4f}\n"
                                        f"param={vis_param_norm:.4f}\n"
                                        f"bad_steps={guard_bad_steps}\n"
                                    ),
                                    encoding="utf-8",
                                )
                            break

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        if is_rank0(rank) and (global_step == 1 or global_step % max(1, args.logging_steps) == 0):
                            frames = max(stats["frames"], 1)
                            loss_scalar = accum_loss / max(accum_steps, 1)
                            lrs = scheduler.get_last_lr()
                            print(
                                f"[train] epoch={epoch} step={global_step}/{total_steps} "
                                f"loss={loss_scalar:.4f} "
                                f"lr={float(lrs[0]) if lrs else 0.0:.2e} "
                                f"step3={_safe_ratio(stats['step3'], frames):.3f} "
                                f"flip={_safe_ratio(stats['flip'], frames):.3f} "
                                f"leak2={_safe_ratio(stats['leak2'], frames):.3f} "
                                f"leak3={_safe_ratio(stats['leak3'], max(stats['step3'], 1)):.3f} "
                                f"phase_a={_safe_ratio(stats['phase_a'], frames):.3f} "
                                f"elapsed={(time.time() - start_time) / 60.0:.1f}m"
                            )
                            if tb is not None:
                                language_param_norm = _param_norm(language_params) if language_params else 0.0
                                tb.add_scalar("train/loss_total", loss_scalar, global_step)
                                tb.add_scalar("train/loss/a1", stats["loss_a1"] / frames, global_step)
                                tb.add_scalar("train/loss/a2", stats["loss_a2"] / frames, global_step)
                                tb.add_scalar("train/loss/a3", stats["loss_a3"] / frames, global_step)
                                tb.add_scalar("train/loss/s2", stats["loss_s2"] / frames, global_step)
                                tb.add_scalar("train/loss/s3_status", stats["loss_s3_status"] / frames, global_step)
                                tb.add_scalar("train/loss/s3_subgoal", stats["loss_s3_subgoal"] / frames, global_step)
                                tb.add_scalar("train/loss_weight/a1", float(args.w_a1), global_step)
                                tb.add_scalar("train/loss_weight/a2", float(args.w_a2), global_step)
                                tb.add_scalar("train/loss_weight/a3", float(args.w_a3), global_step)
                                tb.add_scalar("train/loss_weight/s2", float(args.w_s2), global_step)
                                tb.add_scalar("train/loss_weight/s3_status", float(args.w_s3_status), global_step)
                                tb.add_scalar("train/loss_weight/s3_subgoal", float(args.w_s3_subgoal), global_step)
                                tb.add_scalar("train/lr", float(lrs[0]) if lrs else 0.0, global_step)
                                if len(lrs) > 1:
                                    tb.add_scalar("train/lr_vision", float(lrs[1]), global_step)
                                tb.add_scalar("train/step3_trigger_rate", _safe_ratio(stats["step3"], frames), global_step)
                                tb.add_scalar("train/scene_flip_rate", _safe_ratio(stats["flip"], frames), global_step)
                                tb.add_scalar("train/phase_a_frame_frac", _safe_ratio(stats["phase_a"], frames), global_step)
                                tb.add_scalar("train/gt_leak_skip_rate/step2", _safe_ratio(stats["leak2"], frames), global_step)
                                tb.add_scalar(
                                    "train/gt_leak_skip_rate/step3",
                                    _safe_ratio(stats["leak3"], max(stats["step3"], 1)),
                                    global_step,
                                )
                                tb.add_scalar("train/grad_norm/language", float(lang_norm), global_step)
                                tb.add_scalar("train/grad_norm/vision", float(vis_norm), global_step)
                                tb.add_scalar("train/param_norm/lora_language", language_param_norm, global_step)
                                tb.add_scalar("train/param_norm/lora_vision", vis_param_norm, global_step)
                                tb.add_scalar("train/vision_guard_bad_steps", float(guard_bad_steps), global_step)

                        accum_steps = 0
                        accum_loss = 0.0
                        for key in stats:
                            stats[key] = 0

                        if (
                            world_size <= 1
                            and args.eval_steps > 0
                            and val_loader is not None
                            and global_step % args.eval_steps == 0
                        ):
                            metrics = evaluate_quick(bundle, val_loader, args)
                            if is_rank0(rank):
                                print(f"[eval@{global_step}] {metrics}")
                                if tb is not None:
                                    for key, value in metrics.items():
                                        tb.add_scalar(f"val/{key}", float(value), global_step)

                        if (
                            (not args.check)
                            and args.save_steps > 0
                            and global_step % args.save_steps == 0
                            and is_rank0(rank)
                        ):
                            ckpt = output_dir / f"checkpoint-{global_step}"
                            bundle.unwrap().save_pretrained(str(ckpt))
                            _save_adapter_config(ckpt, bundle, args)
                            saved.append(ckpt)
                            if args.save_total_limit > 0 and len(saved) > args.save_total_limit:
                                old = saved.pop(0)
                                shutil.rmtree(old, ignore_errors=True)

                    # 单卡时 episode 尾部不足 grad_accum 的帧也更新一次。DDP 下不能在
                    # episode 边界做不对齐的 tail step；默认 grad_accum=1，每帧同步更新。
                    if (
                        world_size <= 1
                        and accum_steps > 0
                        and not fuse_stopped
                        and not (args.max_steps > 0 and global_step >= args.max_steps)
                    ):
                        lang_norm = (
                            torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm))
                            if language_params
                            else torch.tensor(0.0, device=device)
                        )
                        vis_norm = (
                            torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm))
                            if vision_params
                            else torch.tensor(0.0, device=device)
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if is_rank0(rank) and (global_step == 1 or global_step % max(1, args.logging_steps) == 0):
                            print(
                                f"[train] epoch={epoch} step={global_step}/{total_steps} "
                                f"loss={accum_loss / max(accum_steps, 1):.4f} tail=1 "
                                f"|g|_lang={float(lang_norm):.3f} |g|_vis={float(vis_norm):.3f} "
                                f"elapsed={(time.time() - start_time) / 60.0:.1f}m"
                            )
                        accum_steps = 0
                        accum_loss = 0.0
                        for key in stats:
                            stats[key] = 0

                    if fuse_stopped or (args.max_steps > 0 and global_step >= args.max_steps):
                        break
                if fuse_stopped or (args.max_steps > 0 and global_step >= args.max_steps):
                    break
            if fuse_stopped or (args.max_steps > 0 and global_step >= args.max_steps):
                break

    if tb is not None:
        tb.flush()
        tb.close()

    if is_rank0(rank) and not args.check and not fuse_stopped:
        final_dir = output_dir / "final"
        bundle.unwrap().save_pretrained(str(final_dir))
        _save_adapter_config(final_dir, bundle, args)
        try:
            bundle.processor.save_pretrained(str(final_dir))
        except Exception as exc:
            print(f"[warn] save processor skipped: {exc}")
        print(f"[done] final adapter -> {final_dir}")
    elif is_rank0(rank) and fuse_stopped:
        print("[done] stopped by vision fuse guard; final save skipped")
    _barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
