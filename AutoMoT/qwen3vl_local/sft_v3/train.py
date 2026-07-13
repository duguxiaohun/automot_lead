"""SFT v3 训练入口：sequence memory + 三步 OPSD 蒸馏。

实现对应 ``SFT_V3_PLAN.md``：

- 一条 episode 是一个 sub-scenario 时间序列；
- 学生用自己的 step2/step3 输出维护纯文本 memory；
- 每帧依次跑 step1 纯视觉分析、step2 场景判断、可选 step3 状态/子目标判断；
- teacher 与 student 共享同一份 base Qwen，teacher 通过 ``disable_adapter`` 关闭 LoRA；
- 训练只保存 LoRA adapter delta，base checkpoint 始终只读。

本文件里最容易读错的是 KV cache 处理：训练时 student 先“自由生成”得到 on-policy
rollout 并更新 memory；privileged teacher 只提供同一批 student token 上的 full-vocab
logit distribution。OPSD loss 当前是 forward-KL 分布匹配，不再把 teacher 文本当硬标签 CE。
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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
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
from qwen3vl_local.mrope_utils import qwen3vl_incremental_forward
from qwen3vl_local.sft_v3.prompts import (
    DEFAULT_W_ANALYSIS,
    DEFAULT_W_ROAD_STRUCTURE,
    DEFAULT_W_SCENE,
    DEFAULT_W_STATUS,
    DEFAULT_W_SUBGOAL,
    SYSTEM_PROMPT_V3,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    canonicalize_scene,
    build_step1_teacher_prompt,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
    force_memory_to_gt_chain,
    get_step_system_prompt,
    get_road_structure,
    should_trigger_step3,
    should_trigger_step2,
    init_memory,
    parse_output,
    target_spans_road_structure,
    target_spans_scene,
    target_spans_status,
    update_memory_after_step1,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_road_structure,
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
    raw_gt_scene: str
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
                raw_gt_scene = str(obj.get("raw_gt_scene", obj.get("scenario", "")))
                gt_scene = str(canonicalize_scene(obj.get("gt_scene", raw_gt_scene)))
                self.rows.append(
                    EpisodeRow(
                        run_id=str(obj["run_id"]),
                        scenario=str(obj["scenario"]),
                        raw_gt_scene=raw_gt_scene,
                        anchors=[int(x) for x in obj["anchors"]],
                        delta=int(obj["delta"]),
                        frame_start=int(frame_range[0]),
                        frame_end=int(frame_range[1]),
                        gt_scene=gt_scene,
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
    system_prompt: str = SYSTEM_PROMPT_V3,
    prev_turns: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """构造 Qwen chat messages：system + 首个带 4 张图的 user turn。

    v3 不维护自己的 prompt 文本；调用方按 v4 的 step 语义传入
    ``get_step_system_prompt("STEPx")``。student 的 step2/3 仍从 step1
    KV 追加文本 user turn，以保持 on-policy rollout；privileged teacher
    每个 step 用对应的 v4 system prompt 独立 prefill 后，在同一段 student
    step 输出 token 上给 forward-KL 分布监督。
    """

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
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
    - ``next_logits``：预测下一个 token 的 logits；OPSD 路径用它对齐第一个 append token 的分布；
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


def _disable_adapter_context(bundle: Any) -> Any:
    """返回关闭 LoRA adapter 的上下文。

    OPSD 的 teacher 必须是 privileged frozen/base Qwen，而不是带 LoRA 的当前
    student。PEFT 模型提供 ``disable_adapter()``；没有该方法时退化为 no-op，
    便于无 LoRA 的轻量测试继续运行。
    """

    model = bundle.unwrap() if hasattr(bundle, "unwrap") else bundle.model
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


@contextmanager
def _teacher_eval_context(bundle: Any) -> Iterable[None]:
    """teacher 侧统一上下文：关闭 LoRA、切 eval、禁止梯度。

    OPSD 的 teacher distribution 必须是稳定的 privileged base 分布。如果只用
    ``torch.no_grad()`` 而不切 ``eval()``，训练态 dropout 仍会随机扰动 teacher logits；
    如果只切 ``eval()`` 而不 ``disable_adapter()``，teacher 又会被当前 LoRA student
    污染。这里把三件事绑在一起，避免不同 step 手写上下文时漏掉其中一项。
    """

    model = bundle.model
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad(), _disable_adapter_context(bundle):
        try:
            yield
        finally:
            if was_training:
                model.train()


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
       token loss 仍照算。当前 v4 的 GT leak hook 是 legacy no-op，这个开关只作为
       兼容旧实验和未来重新启用过滤时的安全阀。
    """

    if suffix_ids.ndim == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    prefix_ids = state.cache_input_ids
    decoded_ids = state.decoded_input_ids.to(prefix_ids.device)
    pending_ids = decoded_ids[:, prefix_ids.shape[1] :]
    feed_ids = torch.cat([pending_ids, suffix_ids], dim=1) if pending_ids.numel() else suffix_ids
    zero = state.next_logits.sum() * 0.0
    parts = {"analysis": zero, "road_structure": zero, "scene": zero, "status": zero, "subgoal": zero}
    if feed_ids.shape[1] == 0:
        return state, parts

    old_attention = state.attention_mask.to(prefix_ids.device)
    attention_mask = torch.cat(
        [old_attention, torch.ones_like(feed_ids, device=old_attention.device)],
        dim=1,
    )
    prefix_len = int(prefix_ids.shape[1])
    # 与 sft_v4 同款修复：不要经过 PeftModelForCausalLM.prepare_inputs_for_generation。
    # PEFT wrapper 会裁掉 cache_position，导致 Qwen3-VL decode 阶段 M-RoPE 位置退化为 0；
    # 本地 helper 用本条 KVState 的 rope_deltas 显式复算 position_ids，再直接 forward。
    outputs = qwen3vl_incremental_forward(
        bundle.model,
        feed_ids=feed_ids,
        attention_mask=attention_mask,
        past_key_values=state.past_key_values,
        prefix_len=prefix_len,
        rope_deltas=state.rope_deltas,
    )
    decoded_input_ids = torch.cat([prefix_ids, feed_ids], dim=1)

    pred_logits = torch.cat([state.next_logits.unsqueeze(1), outputs.logits[:, :-1, :]], dim=1)
    pending_len = int(pending_ids.shape[1])

    def token_loss(feed_positions: List[int]) -> torch.Tensor:
        """旧 CE 计算器，仅供 append_user/兼容路径复用；OPSD 主损失不调用它。"""

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
        # 增量追加纯文本不会改变图文 prefix 的 M-RoPE delta。这里必须沿用输入
        # state.rope_deltas，而不是读取 outputs.rope_deltas：部分 Qwen/Transformers
        # 版本会把模型对象上一次 batched prefill 的旧 rope_deltas 带出来，active batch
        # 缩小时会把 batch=2 的 delta 写回 batch=1 的 KVState，导致后续 decode expand
        # 报 `Target sizes: [1, -1]. Tensor sizes: [2, 1]`。
        rope_deltas=state.rope_deltas,
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


def _append_token_ids_with_logits(
    bundle: Any,
    state: KVState,
    suffix_ids: torch.Tensor,
) -> Tuple[KVState, torch.Tensor, torch.Tensor]:
    """追加一段 token，并返回这些追加 token 对应的 next-token logits。

    OPSD 不再把 teacher 生成文本当 hard CE 标签。训练时先让 student 自由生成一段
    rollout 文本，然后把**同一段 token**分别追加到 student prompt state 和 privileged
    teacher prompt state 上，比较两边对这些 token 的 full-vocabulary next-token
    distribution。返回的 ``pred_logits[:, j, :]`` 对应 ``feed_ids[:, j]`` 这个 token
    被预测时的分布，后续会按 target span 只抽 analysis / label 值 token 位置算 forward-KL。
    """

    if suffix_ids.ndim == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    suffix_ids = suffix_ids.to(state.cache_input_ids.device)
    prefix_ids = state.cache_input_ids
    decoded_ids = state.decoded_input_ids.to(prefix_ids.device)
    pending_ids = decoded_ids[:, prefix_ids.shape[1] :]
    # decoded_input_ids 可能已经包含“已解码但尚未写入 cache”的 pending token。
    # 增量 forward 必须先补齐 pending，再追加本次 suffix；否则 cache_input_ids、
    # attention_mask 和 M-RoPE 位置会错位，teacher/student logits 会比较在不同上下文上。
    feed_ids = torch.cat([pending_ids, suffix_ids], dim=1) if pending_ids.numel() else suffix_ids
    if feed_ids.shape[1] == 0:
        empty = state.next_logits.new_zeros((1, 0, state.next_logits.shape[-1]))
        return state, empty, feed_ids

    old_attention = state.attention_mask.to(prefix_ids.device)
    attention_mask = torch.cat(
        [old_attention, torch.ones_like(feed_ids, device=old_attention.device)],
        dim=1,
    )
    prefix_len = int(prefix_ids.shape[1])
    outputs = qwen3vl_incremental_forward(
        bundle.model,
        feed_ids=feed_ids,
        attention_mask=attention_mask,
        past_key_values=state.past_key_values,
        prefix_len=prefix_len,
        rope_deltas=state.rope_deltas,
    )
    # 第一个追加 token 的预测分布来自旧 state.next_logits；后续 token 的预测分布来自
    # 本次 forward 的 logits 前 T-1 位。这个拼接保证 pred_logits 与 feed_ids 逐位对齐。
    pred_logits = torch.cat([state.next_logits.unsqueeze(1), outputs.logits[:, :-1, :]], dim=1)
    decoded_input_ids = torch.cat([prefix_ids, feed_ids], dim=1)
    new_state = KVState(
        decoded_input_ids=decoded_input_ids,
        cache_input_ids=decoded_input_ids,
        attention_mask=attention_mask,
        past_key_values=outputs.past_key_values,
        # 与 _append_token_ids 保持一致：纯文本增量不会改变多模态 rope delta，
        # 继续使用输入 state，避免 batched rollout 后 Qwen 输出 stale delta 污染 KVState。
        rope_deltas=state.rope_deltas,
        next_logits=outputs.logits[:, -1, :],
    )
    return new_state, pred_logits, feed_ids


def _append_text_with_logits(bundle: Any, state: KVState, text: str) -> Tuple[KVState, torch.Tensor, torch.Tensor]:
    """把文本转成 token 后追加到 KV，并暴露每个追加 token 的预测分布。"""

    enc = bundle.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return _append_token_ids_with_logits(bundle, state, enc["input_ids"])


def _loss_positions_for_text(
    bundle: Any,
    assistant_text: str,
    span_fn: Any | None,
    *,
    analysis_enabled: bool,
    include_keys: Iterable[str],
) -> Dict[str, List[int]]:
    """把 analysis/label 的字符 span 映射到 token 下标，供 OPSD forward-KL 选择位置。

    prompt 模块负责告诉我们 ``SCENE: xxx`` 这类值 token 的字符区间；这里再用 tokenizer
    offset mapping 转成 token 下标。analysis 部分只取 label 行之前的非空 token，避免格式
    token、换行和 prompt 侧重复字符串参与监督。
    """

    enc = bundle.tokenizer(assistant_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    out: Dict[str, List[int]] = {"analysis": []}
    for key in include_keys:
        out.setdefault(str(key), [])
    if analysis_enabled:
        analysis_end = _analysis_char_end(assistant_text)
        out["analysis"] = [
            j for j, (lo, hi) in enumerate(offsets)
            if hi > 0 and lo < analysis_end and assistant_text[lo:hi].strip()
        ]
    if span_fn is not None:
        spans = span_fn(assistant_text)
        for key, (span_lo, span_hi) in spans.items():
            if key not in out:
                continue
            out[key] = [
                j for j, (lo, hi) in enumerate(offsets)
                if lo < span_hi and hi > span_lo
            ]
    return out


def _zero_from_state(state: KVState) -> torch.Tensor:
    """在模型设备上构造可反传的 0，便于跳过某个 loss 分量时保持图结构一致。"""

    return state.next_logits.sum() * 0.0


def _kl_for_positions(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    positions: List[int],
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """在选中的追加 token 位置上计算 full-vocabulary OPSD forward-KL。

    teacher logits 会 detach，只作为 privileged 分布目标；student logits 保持梯度。
    temperature 按蒸馏常规乘回 ``T^2``，避免调高温度后梯度尺度突然变小。
    """

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


def _opsd_loss_from_states(
    bundle: Any,
    student_state: KVState,
    teacher_state: KVState,
    student_rollout: str,
    span_fn: Any | None,
    *,
    analysis_enabled: bool,
    include_keys: Iterable[str],
    temperature: float = 1.0,
    rollout_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """在同一段 on-policy student rollout 上匹配 teacher/student logits。

    这是 v3 OPSD 的核心：student 文本先生成并用于推进 Memory；teacher 不生成 hard
    target，而是在 ``disable_adapter()`` 的 base Qwen 上读 privileged prompt，对这段
    student 文本逐 token 打分。当前实现使用单向 forward-KL
    ``KL(teacher || student)``：teacher 分布 detach，梯度只回到 student LoRA。这样
    loss 优化的是“student 在自己真实输出轨迹上应更像 privileged teacher 的分布偏好”，
    不是离线老师文本的逐字模仿；如果以后要改成 JSD，必须显式加 loss type 开关并同步
    v3/v4 文档。
    """

    zero = _zero_from_state(student_state)
    parts = {"analysis": zero, "road_structure": zero, "scene": zero, "status": zero, "subgoal": zero}
    # 保留 student rollout 的原始文本，不做 strip。文本只用于解析/定位 target span；
    # 真正送入 teacher/student 打分的优先是自由生成时已经进入 KV 的 token ids。
    # 这样 OPSD 监督的是“推动 memory 和后续 KV 的同一批 token”，而不是 batch_decode
    # 后再 tokenizer 一次得到的近似 tokenization。
    text = student_rollout or ""
    token_ids = rollout_token_ids
    if token_ids is not None:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.shape[1] == 0:
            token_ids = None
    if token_ids is None and not text.strip():
        return parts
    # teacher 分支必须在 no_grad + disable_adapter 下运行，否则 LoRA student 会给自己当
    # teacher，OPSD 退化成自蒸馏；这里用 clone 避免 scoring 过程污染外层 KVState。
    with _teacher_eval_context(bundle):
        if token_ids is not None:
            _, teacher_logits, _ = _append_token_ids_with_logits(
                bundle, _clone_kv_state(teacher_state), token_ids
            )
        else:
            _, teacher_logits, _ = _append_text_with_logits(bundle, _clone_kv_state(teacher_state), text)
    # student 分支保留梯度，但同样只在 clone 上 append；外层 state 继续代表 prompt state，
    # 便于后续 step2/step3 自己按 rollout 链式推进。
    if token_ids is not None:
        _, student_logits, _ = _append_token_ids_with_logits(
            bundle, _clone_kv_state(student_state), token_ids
        )
    else:
        _, student_logits, _ = _append_text_with_logits(bundle, _clone_kv_state(student_state), text)
    positions = _loss_positions_for_text(
        bundle,
        text,
        span_fn,
        analysis_enabled=analysis_enabled,
        include_keys=include_keys,
    )
    for key, pos in positions.items():
        if key in parts:
            valid_pos = [p for p in pos if p < student_logits.shape[1]]
            parts[key] = _kl_for_positions(student_logits, teacher_logits, valid_pos, temperature=temperature)
    return parts


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
) -> Tuple[str, KVState, torch.Tensor]:
    """从已有 KVState 贪心生成一段 assistant 文本，并返回生成后的状态。

    框架明确要求 ``do_sample=False`` + ``repetition_penalty=1.05``；这里 argmax 已等价
    于 do_sample=False，再额外按 repetition_penalty 对前文 token 做惩罚。前文范围
    取 ``state.decoded_input_ids`` ∪ 已生成 token，与 HF generate 默认行为一致。
    """

    cached_generated: List[torch.Tensor] = []
    eos_ids = _eos_token_ids(bundle)
    cur = state
    device = cur.next_logits.device
    seen_unique = torch.unique(cur.decoded_input_ids.reshape(-1).to(device))
    for _ in range(max_new_tokens):
        logits = _apply_repetition_penalty(cur.next_logits, seen_unique, repetition_penalty)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.reshape(-1)[0].item())
        if token_id in eos_ids:
            break
        cached_generated.append(next_token)
        cur, _ = _append_token_ids(bundle, cur, next_token)
        # 把新 token 也纳入 repetition 集合；torch.unique 在已排序集合上插入仍是 O(N)
        seen_unique = torch.unique(torch.cat([seen_unique, next_token.reshape(-1).to(device)], dim=0))
    if not cached_generated:
        empty = cur.decoded_input_ids.new_zeros((1, 0))
        return "", cur, empty
    ids = torch.cat(cached_generated, dim=1)
    # 不 strip：外层 OPSD loss 需要尽量保留生成文本的 token 边界。
    text = bundle.processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text, cur, ids


def _teacher_start_state(bundle: Any, messages: List[Dict[str, Any]]) -> KVState:
    """teacher prefill 包装：关闭 LoRA，只读 base Qwen。"""

    with _teacher_eval_context(bundle):
        return _kv_start_state(bundle, messages)


def _student_start_state(bundle: Any, messages: List[Dict[str, Any]]) -> KVState:
    """student prefill 包装：当前实现也 no_grad，训练梯度只来自 LoRA 文本续写段。"""

    with torch.no_grad():
        return _kv_start_state(bundle, messages)


def _teacher_generate_kv(bundle: Any, state: KVState, max_new_tokens: int) -> Tuple[str, KVState, torch.Tensor]:
    """teacher 贪心生成包装：关闭 LoRA 与梯度。"""

    with _teacher_eval_context(bundle):
        return _kv_generate_text(bundle, state, max_new_tokens)


def _student_generate_kv(bundle: Any, state: KVState, max_new_tokens: int) -> Tuple[str, KVState, torch.Tensor]:
    """student 自由生成包装：返回文本、KV 和真实进入 KV 的 token ids。

    OPSD 的“on-policy”要求 teacher/student 在同一批 student rollout token 上打分。
    文本解析仍用于更新 Memory，但 loss 侧优先使用这里返回的 token ids，避免
    ``batch_decode -> tokenizer`` 往返造成 token 边界漂移。
    """

    with torch.no_grad():
        return _kv_generate_text(bundle, state, max_new_tokens)


@dataclass
class FrameLossPack:
    """单帧三步内循环产生的 loss 与诊断标志。"""

    a1: torch.Tensor
    rs1: torch.Tensor
    a2: torch.Tensor
    s2: torch.Tensor
    a3: torch.Tensor
    s3_status: torch.Tensor
    s3_subgoal: torch.Tensor
    step2_ran: bool
    step3_ran: bool
    rs_flip: bool
    scene_flip: bool
    leak1: bool
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
    """用 v4 同步 prompt 跑一条 episode，并产出每帧 OPSD loss 分量。

    关键顺序是“student 先自由生成，再由 teacher 给这段 student token 打分”：
    1. student 的 step1/2/3 KV 串行追加，完全模拟在线 rollout；
    2. Memory 只由 student 解析结果推进，错了也按错 memory 继续；
    3. teacher 每个 step 用 v4 对应 system prompt + privileged user prompt 独立 prefill；
    4. loss 只比较 teacher/student 对同一批 student step token 的 full-vocab forward-KL。
    """

    del max_length
    run_dir = pathlib.Path(ep.run_dir)
    gx, gy = _load_goal_xy(run_dir, ep.frame_start)
    memory = init_memory(
        run_id=ep.run_id,
        sub_scenario_id=f"{ep.run_id}:{ep.anchors[1]}",
        ego_to_goal_x=gx,
        ego_to_goal_y=gy,
        gt_scene=ep.gt_scene,
    )

    gt_road_structure = get_road_structure(ep.gt_scene)
    stride = max(1, outer_stride)
    for frame in range(ep.frame_start, ep.frame_end + 1, stride):
        phase_a = _is_phase_a(ep, frame)
        road_structure_reset_this_frame = False
        scene_reset_this_frame = False
        if not phase_a:
            # Phase B 是弱纠偏阶段：帧首把上层 memory 拉回 GT 链，让模型学习
            # “已经正确时不要乱改”。reset 标志会参与 trigger gate，避免同一帧刚纠正
            # road/scene 后立刻继续下钻，造成 teacher 用稳定上下文而 student 刚被外力改写。
            before_rs = memory.road_structure
            before_scene = memory.scene
            memory = force_memory_to_gt_chain(
                memory, gt_road_structure=gt_road_structure, gt_scene=ep.gt_scene
            )
            road_structure_reset_this_frame = memory.road_structure != before_rs
            scene_reset_this_frame = memory.scene != before_scene

        image_paths = _build_rgb_paths(run_dir, frame)
        try:
            images = _load_images(image_paths)
        except Exception:
            _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)
            continue

        gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)
        zero_seed: Optional[torch.Tensor] = None

        # Step1：student 只读 road-only memory + 图像，先自由生成 ROAD_STRUCTURE。
        # teacher 也用 STEP1 system prompt，但 user prompt 含 privileged answer road；
        # 两边 prompt 不同，不能共享 KV，只能比较同一段 student 输出上的 logits。
        step1_student_user = build_step1_user_prompt(len(images), memory=memory)
        step1_teacher_user = build_step1_teacher_prompt(memory, gt_road_structure)
        step1_msgs_student = _build_messages_with_images(
            user_text=step1_student_user,
            images=images,
            system_prompt=get_step_system_prompt("STEP1"),
        )
        step1_msgs_teacher = _build_messages_with_images(
            user_text=step1_teacher_user,
            images=images,
            system_prompt=get_step_system_prompt("STEP1"),
        )

        student_was_training = bool(bundle.model.training)
        bundle.model.eval()
        student_step1_prompt_state = _student_start_state(bundle, step1_msgs_student)
        raw_student_step1, student_step1_state, student_step1_ids = _student_generate_kv(
            bundle, _clone_kv_state(student_step1_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP1
        )
        if student_was_training:
            bundle.model.train()
        if not raw_student_step1:
            # 空输出通常表示模型立刻给 EOS。它没有可用于 OPSD 的 student rollout
            # token，不能用带 GT 的 teacher target 兜底，否则会把 v3 悄悄退化成
            # hard-label imitation。这里跳过该帧，并在帧末正常推进 goal 坐标。
            _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)
            continue

        teacher_was_training = bool(bundle.model.training)
        bundle.model.eval()
        teacher_step1_prompt_state = _teacher_start_state(bundle, step1_msgs_teacher)
        if teacher_was_training:
            bundle.model.train()
        leak1 = False
        step1_parts = _opsd_loss_from_states(
            bundle,
            _clone_kv_state(student_step1_prompt_state),
            _clone_kv_state(teacher_step1_prompt_state),
            raw_student_step1,
            target_spans_road_structure,
            analysis_enabled=True,
            include_keys=("road_structure",),
            rollout_token_ids=student_step1_ids,
        )
        zero_seed = step1_parts["analysis"] * 0.0

        # Memory 的推进只看 student 自己的解析结果，而不是 teacher target。
        # 这正是 on-policy：如果 student 把 road 判断错，后续 step2 会在错误 bucket 里继续。
        pred1 = parse_output(raw_student_step1)
        old_rs = memory.road_structure
        memory = update_memory_after_step1(memory, student_road_structure=pred1.get("road_structure"))
        rs_flip = memory.road_structure != old_rs

        a2 = zero_seed
        s2 = zero_seed
        a3 = zero_seed
        s3_status = zero_seed
        s3_subgoal = zero_seed
        leak2 = False
        leak3 = False
        scene_flip = False
        old_scene = memory.scene
        student_step2_state: Optional[KVState] = None
        raw_student_step2 = ""

        step2_ran = should_trigger_step2(
            memory_road_structure_before_step1=old_rs,
            memory_road_structure_after_step1=memory.road_structure,
            gt_road_structure=gt_road_structure,
            road_structure_reset_this_frame=road_structure_reset_this_frame,
        )
        if step2_ran:
            # Step2：沿用 student step1 之后的 KV 追加 user turn，不重新喂图。
            # teacher 则打开 fresh STEP2 dialog，直接看到 GT road/scene，用来给
            # student 的 step2 rollout 分布打分。
            step2_student_user = build_step2_student_prompt(memory)
            with torch.no_grad():
                student_step2_prompt_state = _append_user_turn(bundle, student_step1_state, step2_student_user)
            student_was_training = bool(bundle.model.training)
            bundle.model.eval()
            raw_student_step2, student_step2_state, student_step2_ids = _student_generate_kv(
                bundle, _clone_kv_state(student_step2_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP2
            )
            if student_was_training:
                bundle.model.train()

            step2_msgs_teacher = _build_messages_with_images(
                user_text=build_step2_teacher_prompt(memory, gt_road_structure, ep.gt_scene),
                images=images,
                system_prompt=get_step_system_prompt("STEP2"),
            )
            teacher_step2_prompt_state = _teacher_start_state(bundle, step2_msgs_teacher)
            # v4 当前把 GT leak 检查保留为 legacy no-op，因此这里通常为 False。
            # 保留调用是为了让 v3/v4 prompt contract 将来若重新启用过滤时仍只改一处。
            leak2 = check_gt_leak_scene(_analysis_before_labels(raw_student_step2), ep.gt_scene)
            step2_parts = _opsd_loss_from_states(
                bundle,
                _clone_kv_state(student_step2_prompt_state),
                _clone_kv_state(teacher_step2_prompt_state),
                raw_student_step2,
                target_spans_scene,
                analysis_enabled=not leak2,
                include_keys=("scene",),
                rollout_token_ids=student_step2_ids,
            )
            a2 = step2_parts["analysis"]
            s2 = step2_parts["scene"]
            # step2 的输出决定 memory.scene。后续 step3 是否运行只看学生 memory 是否稳定
            # 命中 GT，而不是 teacher 知道答案这件事。
            pred2 = parse_output(raw_student_step2)
            old_scene = memory.scene
            memory = update_memory_after_step2(memory, student_scene=pred2.get("scene"))
            scene_flip = memory.scene != old_scene

        step3_ran = (
            step2_ran
            and should_trigger_step3(
                memory_scene_before_step2=old_scene,
                memory_scene_after_step2=memory.scene,
                gt_scene=ep.gt_scene,
                scene_reset_this_frame=scene_reset_this_frame,
            )
        )
        if step3_ran:
            assert student_step2_state is not None
            # Step3 只在 scene 已经稳定正确时运行；否则 STATUS/SUBGOAL 候选表会属于错误
            # scene，继续监督反而会把不同事件序列混在一起。
            step3_student_user = build_step3_student_prompt(memory)
            with torch.no_grad():
                student_step3_prompt_state = _append_user_turn(bundle, student_step2_state, step3_student_user)
            student_was_training = bool(bundle.model.training)
            bundle.model.eval()
            raw_student_step3, _student_step3_state, student_step3_ids = _student_generate_kv(
                bundle, _clone_kv_state(student_step3_prompt_state), TEACHER_MAX_NEW_TOKENS_STEP3
            )
            if student_was_training:
                bundle.model.train()

            step3_msgs_teacher = _build_messages_with_images(
                user_text=build_step3_teacher_prompt(memory, gt_road_structure, ep.gt_scene, gt_status, gt_subgoal),
                images=images,
                system_prompt=get_step_system_prompt("STEP3"),
            )
            teacher_step3_prompt_state = _teacher_start_state(bundle, step3_msgs_teacher)
            # 与 step2 一样，这是跟随 v4 的 legacy no-op hook；当前不会跳 analysis KL。
            leak3 = check_gt_leak_status_subgoal(_analysis_before_labels(raw_student_step3), gt_status, gt_subgoal)
            step3_parts = _opsd_loss_from_states(
                bundle,
                _clone_kv_state(student_step3_prompt_state),
                _clone_kv_state(teacher_step3_prompt_state),
                raw_student_step3,
                target_spans_status,
                analysis_enabled=not leak3,
                include_keys=("status", "subgoal"),
                rollout_token_ids=student_step3_ids,
            )
            a3 = step3_parts["analysis"]
            s3_status = step3_parts["status"]
            s3_subgoal = step3_parts["subgoal"]

            # 只接受当前 canonical scene 事件序列里的 status/subgoal，防止自由生成把别的
            # scene 事件写进 memory。
            pred3 = parse_output(raw_student_step3)
            pred_status = pred3.get("status") if validate_event(memory.scene, pred3.get("status")) else None
            pred_subgoal = pred3.get("subgoal") if validate_event(memory.scene, pred3.get("subgoal")) else None
            memory = update_memory_after_step3(
                memory,
                student_status=pred_status,
                student_subgoal=pred_subgoal,
            )

        yield FrameLossPack(
            a1=step1_parts["analysis"],
            rs1=step1_parts["road_structure"],
            a2=a2,
            s2=s2,
            a3=a3,
            s3_status=s3_status,
            s3_subgoal=s3_subgoal,
            step2_ran=step2_ran,
            step3_ran=step3_ran,
            rs_flip=rs_flip,
            scene_flip=scene_flip,
            leak1=leak1,
            leak2=leak2,
            leak3=leak3,
            phase_a=phase_a,
        )
        # 帧末预取下一帧 goal，保证下一轮 prompt 里的 EGO_TO_GOAL_XY 与即将读取的 RGB
        # 时间对齐，而不是滞后一帧。
        _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)


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
        "route": "sft_v3_sequence_opsd",
        "base_model_dir": str(args.model_dir),
        "base_model_mutated": False,
        "opsd": {
            "teacher": "shared_base_disable_adapter",
            "rollout_source": "student_free_generation",
            "loss_type": "forward_kl_teacher_to_student",
            "teacher_logits_detached": True,
            "hard_teacher_text_ce": False,
        },
        "distributed_train": {
            "mode": "work_stealing_local_sgd",
            "sync_every_episodes_per_rank": int(args.sync_every_episodes),
            "weighted_average_by_optimizer_steps": True,
            "allow_max_steps_truncation": bool(args.allow_max_steps_truncation),
        },
        "lora_vision_scope": bundle.lora_vision_scope,
        "lora_vision": bool(vision_targets),
        "target_modules": targets,
        "strict_vision_scope": bool(args.strict_vision_scope),
        "loss_weights": {
            "a1": float(args.w_a1),
            "rs1": float(args.w_rs1),
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
    """训练中的轻量 validation，只算 offline OPSD quick loss。

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
                    + args.w_rs1 * pack.rs1
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
    parser.add_argument("--w-rs1", type=float, default=DEFAULT_W_ROAD_STRUCTURE)
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
    # work-stealing + local-SGD：每 rank 目标处理 N 个 episode 后进入一个 sync round。
    # sync round 先用 TCPStore 做 CPU 侧等齐，再做 NCCL 参数平均；0 = 仅 epoch 末同步。
    # 默认 4：sft_v3 每 episode ~85 秒（14 帧 × 6 sec/帧），K=4 时每 rank 每轮约
    # 5.6 分钟，对 work-stealing 不均衡有余量；想要更松（少同步、参数漂移更大）可
    # 显式调大，比如 16；K=1 最接近同步 SGD。
    parser.add_argument("--sync-every-episodes", type=int, default=4)
    parser.add_argument(
        "--allow-max-steps-truncation",
        action="store_true",
        help="Allow --max-steps to stop inside an episode. Intended only for smoke/debug runs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """训练参数一致性检查。

    多 rank + eval_steps>0 会直接报错，避免用户以为分布式训练中已经跑了 in-loop eval。
    """

    if args.outer_stride <= 0:
        raise ValueError("--outer-stride must be > 0")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.per_device_batch_size != 1:
        raise ValueError(
            "SFT v3 work-stealing trains one episode per worker pull; "
            "--per-device-batch-size must stay 1."
        )
    if args.vision_lr_scale < 0:
        raise ValueError("--vision-lr-scale must be >= 0")
    if args.vision_lr_scale > args.max_vision_lr_scale:
        raise ValueError("--vision-lr-scale exceeds --max-vision-lr-scale")
    if args.vision_guard_patience <= 0:
        raise ValueError("--vision-guard-patience must be > 0")
    if args.sync_every_episodes < 0:
        raise ValueError("--sync-every-episodes must be >= 0 (0 = epoch-end only)")
    if args.max_steps > 0 and not (args.check or args.allow_max_steps_truncation):
        raise ValueError(
            "--max-steps can stop inside an episode and is disabled for normal training. "
            "Use --check for smoke tests or pass --allow-max-steps-truncation explicitly."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and args.eval_steps > 0:
        raise ValueError(
            "SFT v3 does not run in-loop eval under multi-rank. Set --eval-steps 0 and run "
            "qwen3vl_local/sft_v3/eval.py after training."
        )


def _param_norm(params: List[nn.Parameter]) -> float:
    """计算参数组 L2 norm，用于视觉 LoRA 熔断监控。"""

    with torch.no_grad():
        return math.sqrt(sum(float(p.detach().float().norm().item()) ** 2 for p in params))


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torch.distributed，返回 rank/world_size/local_rank。

    NCCL watchdog 默认 10 分钟，对 sft_v3 work-stealing 完全不够——每帧 teacher/student
    各 ~80 step decode，每 episode ~14 帧 × 6 sec ≈ 90 sec；一轮 K=16 时即便完美均衡
    单 rank 也要 22 分钟，更别提慢 rank 触发的延迟。这里显式把超时设到 2 小时，给
    work-stealing 留足 sync 等待时间。注意 init_process_group 的 timeout 同时也是默认
    Store 的 wait/get 超时，所以 _store_rendezvous 里的 store.wait 也跟着被放宽。
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
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
    """分布式 barrier；单进程时为空操作。"""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _sync_bool_or(value: bool, device: torch.device) -> bool:
    """跨 rank 同步布尔 OR；任一 rank 触发熔断则全体停止。"""

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


def _sync_int_sum(value: int, device: torch.device) -> int:
    """跨 rank 汇总整数计数；单进程直接返回本地值。"""

    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _get_default_store() -> Optional[Any]:
    """拿到 init_process_group 创建的默认 TCPStore；单进程时返回 None。"""

    if not (dist.is_available() and dist.is_initialized()):
        return None
    # PyTorch 没有给默认 Store 暴露很漂亮的顶层 public API；distributed_c10d 里的
    # helper 是 torchrun 场景最直接的入口。这里显式包一层报错，避免版本差异时
    # 变成难排查的 AttributeError。
    try:
        return dist.distributed_c10d._get_default_store()
    except AttributeError as exc:
        raise RuntimeError(
            "torch.distributed default store is unavailable; SFT v3 work-stealing requires "
            "a PyTorch build exposing distributed_c10d._get_default_store()."
        ) from exc


def _store_rendezvous(store: Optional[Any], key: str, *, rank: int, world_size: int) -> None:
    """用 TCPStore 做长等待 rendezvous，避免快 rank 提前进入 NCCL collective 超时。

    work-stealing 下 episode 时长差异很大：快 rank 可能先跑完整个 sync round，而慢 rank
    还在 teacher/student 内循环里。若快 rank 直接调用 ``dist.barrier`` 或 ``all_reduce``，
    NCCL watchdog 会在其他 rank 尚未进入 collective 时超时。这里先用 CPU 侧 store 等齐；
    等所有 rank 都到达后，再让后续 NCCL allreduce 快速完成。

    ``key`` 必须是单调唯一的同步点名称，例如包含 epoch/round；TCPStore key 不回收，
    复用同一个 key 会把上一轮的 arrived/done 状态带进来。
    """

    if world_size <= 1 or store is None:
        return
    arrived_key = f"{key}:arrived"
    done_key = f"{key}:done"
    if rank == 0:
        store.set(arrived_key, "0")
    store.wait([arrived_key])
    arrived = int(store.add(arrived_key, 1))
    if arrived >= world_size:
        store.set(done_key, "1")
    else:
        store.wait([done_key])


def _weighted_average_lora_params_inplace(
    params: List[nn.Parameter],
    *,
    local_weight: float,
    device: torch.device,
) -> float:
    """按本轮本地训练量加权平均 LoRA 参数，并返回全 rank 权重和。

    work-stealing 下每个 rank 抢到的 episode / frame 数不一定相同；如果简单
    ``sum/world_size``，最后一轮或慢 rank 没抢到任务时，未更新参数也会等权参与平均，
    把真正训练过的 rank 更新稀释掉。因此这里按本轮 optimizer.step 数加权；权重为 0 的
    rank 只接收平均后的参数，不贡献旧参数。
    """

    if not (dist.is_available() and dist.is_initialized()) or not params:
        return float(local_weight)
    weight = torch.tensor([float(local_weight)], dtype=torch.float32, device=device)
    dist.all_reduce(weight, op=dist.ReduceOp.SUM)
    total_weight = float(weight.item())
    if total_weight <= 0.0:
        return total_weight
    with torch.no_grad():
        for p in params:
            averaged = p.detach().float().mul(float(local_weight))
            dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
            averaged.div_(total_weight)
            p.data.copy_(averaged.to(dtype=p.dtype))
    return total_weight


def _sync_fuse_info(
    *,
    triggered: bool,
    rank: int,
    grad_norm: float,
    param_norm: float,
    bad_steps: int,
    device: torch.device,
) -> Tuple[int, float, float, int]:
    """汇总视觉熔断诊断信息，保证非 rank0 触发时 rank0 也能写 reason 文件。"""

    local_rank = rank if triggered else -1
    local_grad = float(grad_norm) if triggered and math.isfinite(float(grad_norm)) else -1.0
    local_param = float(param_norm) if triggered and math.isfinite(float(param_norm)) else -1.0
    local_bad = int(bad_steps) if triggered else 0
    if not (dist.is_available() and dist.is_initialized()):
        return local_rank, local_grad, local_param, local_bad
    payload = torch.tensor(
        [float(local_rank), local_grad, local_param, float(local_bad)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.MAX)
    return int(payload[0].item()), float(payload[1].item()), float(payload[2].item()), int(payload[3].item())


def _broadcast_lora_params_from_rank0(params: List[nn.Parameter], world_size: int) -> None:
    """把 rank0 的 LoRA 初始权重广播到所有 rank。

    local-SGD 不包 DDP，因此模型构建后不会自动同步初始参数。LoRA 初始化通常带随机性，
    如果不广播，各 rank 会从不同 adapter 起点出发，第一轮参数平均会混入初始化差异。
    """

    if world_size <= 1 or not params:
        return
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, src=0)


def _scale_pending_grads(params: List[nn.Parameter], scale: float) -> None:
    """按比例缩放尚未 step 的梯度，用于 grad_accum 尾部 flush。"""

    if abs(scale - 1.0) < 1e-12:
        return
    for p in params:
        if p.grad is not None:
            p.grad.mul_(float(scale))


def _sync_scheduler_to_step(scheduler: Any, optimizer: torch.optim.Optimizer, step: int) -> None:
    """把 LambdaLR 对齐到给定全局 step，并直接刷新 optimizer 当前 lr。

    work-stealing 下每个 rank 在一个 sync round 内做的本地 step 数可能不同；轮末用
    all-rank step 总数对齐 scheduler，避免快慢 rank 在参数平均后继续沿不同 LR 曲线走。
    这里不调用 ``scheduler.step(step)``，因为某些 rank 可能本轮没有 optimizer.step，
    直接调 scheduler.step 会触发 PyTorch 的 step-order warning。

    同时把 ``_step_count`` 也对齐到 step+1（与正常 scheduler.step() 累计次数语义一致），
    这样后续训练里 scheduler.step() 不会因为内部"opt._step_count vs sched._step_count"
    比较异常而在边界条件下打 UserWarning。
    """

    if not hasattr(scheduler, "lr_lambdas") or not hasattr(scheduler, "base_lrs"):
        return
    scheduler.last_epoch = int(step)
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = int(step) + 1
    lrs: List[float] = []
    for group, base_lr, lr_lambda in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        lr = float(base_lr) * float(lr_lambda(int(step)))
        group["lr"] = lr
        lrs.append(lr)
    scheduler._last_lr = lrs


def main() -> None:
    """SFT v3 训练主入口。

    主循环按 episode -> frame 两层推进。默认每帧 backward/step 一次，避免显存随
    episode 长度增长；如果用户显式设置 `grad_accum>1`，只累计梯度，memory 仍按
    学生自由生成逐帧实时更新。
    """

    args = parse_args()
    validate_args(args)
    rank, world_size, local_rank = setup_distributed()
    store = _get_default_store()
    # 注：work-stealing + local-SGD 下不再强制 grad_accum=1。每 rank 独立 step，无锁步。
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
    _store_rendezvous(store, "sft_v3_output_dir_ready", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # SFT v3 的 loss forward 依赖 prefill 出来的 past_key_values；后续 target token
    # 只以 suffix-only 形式追加到 cache 后计算 student logits。HF 在 training=True + gradient_checkpointing=True
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
    # work-stealing + local-SGD：不再用 DDP wrap，也不再静态分片数据集。
    # 多 rank 通过 TCPStore 原子 counter 抢同一份 train_ds.rows，每轮最多开放
    # K*world_size 个全局 episode，并在 barrier 处按本轮训练量加权平均 LoRA 参数，
    # 避免 DDP per-step 锁步死锁。
    train_ds = EpisodeDataset(pathlib.Path(args.train_jsonl))
    val_ds = EpisodeDataset(pathlib.Path(args.val_jsonl)) if args.val_jsonl else None
    train_total = len(train_ds.rows)
    val_total = len(val_ds.rows) if val_ds is not None else 0
    if train_total <= 0:
        raise ValueError(f"empty train dataset: {args.train_jsonl}")
    if is_rank0(rank):
        print(
            f"[data] train_episodes={train_total} val_episodes={val_total} "
            f"world_size={world_size} sync_every_episodes={args.sync_every_episodes}"
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
    _store_rendezvous(store, "sft_v3_lora_params_ready", rank=rank, world_size=world_size)
    _broadcast_lora_params_from_rank0(language_params + vision_params, world_size)

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # schedule 按 all-rank optimizer.step 总数估算；每个 sync round 末用 allreduce 得到
    # 全局步数并把所有 rank 的 scheduler 对齐到同一位置。grad_accum>1 时，round 尾部
    # 会 flush 不足 grad_accum 的梯度，真实 step 数可能高于 ceil(total_frames/grad_accum)；
    # 因此用“每 episode 自己 ceil 后求和”的上界，避免 LR 在最后几轮过早降为 0。
    episode_frame_counts = [
        len(range(ep.frame_start, ep.frame_end + 1, max(1, args.outer_stride))) for ep in train_ds.rows
    ]
    steps_per_epoch = max(
        1,
        sum(math.ceil(frame_count / max(1, args.grad_accum)) for frame_count in episode_frame_counts),
    )
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * max(1, args.num_epochs)
    if args.check:
        total_steps = min(total_steps, 2)
        args.max_steps = total_steps
    scheduler = make_scheduler(optimizer, total_steps, int(total_steps * args.warmup_ratio))
    tb = SummaryWriter(log_dir=str(output_dir / "tb")) if (is_rank0(rank) and _TB_AVAILABLE) else None
    # global_step 是当前 rank 自己的 optimizer.step 计数；rank0 负责日志/保存。
    # 轮末会 allreduce 得到 all_rank_steps，用来观察全局实际训练量；不要把 rank0 的
    # step 当成多卡总步数。
    global_step = 0
    synced_all_rank_steps = 0
    accum_steps = 0
    accum_loss = 0.0
    stats = {
        "frames": 0,
        "step2": 0,
        "step3": 0,
        "rs_flip": 0,
        "flip": 0,
        "leak1": 0,
        "leak2": 0,
        "leak3": 0,
        "phase_a": 0,
        "loss_a1": 0.0,
        "loss_rs1": 0.0,
        "loss_a2": 0.0,
        "loss_a3": 0.0,
        "loss_s2": 0.0,
        "loss_s3_status": 0.0,
        "loss_s3_subgoal": 0.0,
    }
    guard_bad_steps = 0
    fuse_stopped = False
    fuse_stop_saved = False
    last_fuse_grad = -1.0
    last_fuse_param_norm = -1.0
    last_fuse_bad_steps = 0
    saved: List[pathlib.Path] = []
    last_saved_step = 0
    round_local_steps = 0
    # 本 rank 在当前 sync round 内实际完成的 episode 数。它只用于日志/TB 观测负载均衡，
    # 参数平均仍按 optimizer.step 数加权，因为不同 episode 帧数差异很大。
    round_local_episodes = 0
    # 全局累计已完成 episode 数；每轮 sync 后由 round_local_episodes allreduce-sum 累加。
    total_episodes_processed = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    # ---- work-stealing + local-SGD 主循环 ----
    # 每个 epoch 先按同一 seed 得到 epoch_order，再切成若干个 sync round。
    # sync_K 表示“每 rank 目标处理 K 个 episode”，所以每轮最多开放 K*world_size 个
    # 全局 episode。K=1 时每个 rank 通常至多拿到一条 episode，更接近同步 SGD；
    # K=0 表示整轮 epoch 末才同步。
    # 每轮 rank0 把 TCPStore counter 初始化到 round_start；各 rank 只允许抢
    # [round_start, round_end) 内的 idx。这样即使多个 rank 同时空闲，也不会越过同步边界
    # 先多训下一轮 episode；轮末统一 flush 梯度、TCPStore rendezvous、LoRA 参数平均。
    sync_K = int(args.sync_every_episodes)
    round_size = train_total if sync_K <= 0 else max(1, sync_K * max(1, world_size))
    if world_size > 1 and args.max_steps > 0:
        # max_steps/check 是 debug 口径；缩小同步轮可避免多卡 smoke 一次跑完整个 K-episode round。
        round_size = 1
    rounds_per_epoch = max(1, math.ceil(train_total / max(1, round_size)))
    if is_rank0(rank):
        print(
            f"[sync] rounds_per_epoch={rounds_per_epoch} sync_K={sync_K} "
            f"round_size={round_size}"
        )

    all_lora_params: List[nn.Parameter] = language_params + vision_params
    stop_requested = False

    def should_stop_local() -> bool:
        """本 rank 达到 max_steps 后先停止抓任务；全局停止在 sync 后广播。"""

        return args.max_steps > 0 and global_step >= args.max_steps

    def flush_partial_optimizer_step(reason: str) -> None:
        """在 sync 前把未满 grad_accum 的梯度落一次 step，避免跨平均边界残留梯度。"""

        nonlocal accum_steps, accum_loss, global_step, guard_bad_steps, fuse_stopped, synced_all_rank_steps
        nonlocal round_local_steps, last_fuse_grad, last_fuse_param_norm, last_fuse_bad_steps
        if accum_steps <= 0:
            return
        if accum_steps < args.grad_accum:
            # 训练时每帧反传 loss / grad_accum；sync/epoch 尾部不足 grad_accum 时，
            # 这里把梯度恢复成当前 tail 帧的平均值，避免尾部 step 被系统性压小。
            _scale_pending_grads(language_params + vision_params, float(args.grad_accum) / float(accum_steps))
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
        if args.vision_guard_enabled and vision_params:
            bad = (
                (not math.isfinite(vis_grad))
                or vis_grad > float(args.vision_guard_grad_norm_max)
                or (not math.isfinite(vis_param_norm))
                or vis_param_norm > float(args.vision_guard_param_norm_max)
            )
            guard_bad_steps = guard_bad_steps + 1 if bad else 0
            fuse_stopped = fuse_stopped or (guard_bad_steps >= args.vision_guard_patience)
            if fuse_stopped:
                last_fuse_grad = vis_grad
                last_fuse_param_norm = vis_param_norm
                last_fuse_bad_steps = guard_bad_steps
        if not fuse_stopped:
            optimizer.step()
            scheduler.step()
            global_step += 1
            round_local_steps += 1
        optimizer.zero_grad(set_to_none=True)
        if is_rank0(rank):
            print(
                f"[flush] reason={reason} local_step={global_step}/{total_steps} "
                f"accum_frames={accum_steps} |g|_lang={float(lang_norm):.3f} "
                f"|g|_vis={float(vis_norm):.3f} all_rank_steps={synced_all_rank_steps}"
            )
        accum_steps = 0
        accum_loss = 0.0
        for key in stats:
            stats[key] = 0

    def do_sync_round(sync_key: str) -> None:
        """完成一个 local-SGD 同步轮。

        顺序很重要：先 flush 本 rank 的尾部梯度，再在 TCPStore 上等所有 rank 到齐；
        到齐后才允许进入 NCCL allreduce/broadcast 类 collective。这样慢 episode 只会
        让快 rank 等在 CPU store 上，不会占用 NCCL work 直到 watchdog timeout。
        """

        nonlocal fuse_stopped, stop_requested, synced_all_rank_steps, last_saved_step
        nonlocal round_local_steps, fuse_stop_saved, round_local_episodes, total_episodes_processed
        # 每个 rank 进入 sync round 时都打一条日志（含 local_steps / local_episodes / elapsed），
        # work-stealing 出问题时一眼就能看出哪个 rank 落后。打印走 stderr，避免被 [train]
        # 大量 stdout 行掩盖。
        enter_msg = (
            f"[sync-enter] key={sync_key} rank={rank}/{world_size} "
            f"local_steps={round_local_steps} local_eps={round_local_episodes} "
            f"global_step={global_step}/{total_steps} "
            f"elapsed={(time.time() - start_time) / 60.0:.1f}m"
        )
        print(enter_msg, file=sys.stderr, flush=True)
        flush_partial_optimizer_step("sync")
        _store_rendezvous(store, sync_key, rank=rank, world_size=world_size)
        round_weight = _weighted_average_lora_params_inplace(
            all_lora_params,
            local_weight=float(round_local_steps),
            device=device,
        )
        local_fuse_triggered = bool(fuse_stopped)
        fuse_stopped = _sync_bool_or(fuse_stopped, device)
        synced_all_rank_steps = _sync_int_sum(global_step, device)
        # episode 计数只是诊断信号：用来确认 work-stealing 是否真的把整轮数据消耗完，
        # 以及各轮负载是否符合预期；不要用它做参数平均权重。
        round_episodes_sum = _sync_int_sum(round_local_episodes, device)
        total_episodes_processed += round_episodes_sum
        _sync_scheduler_to_step(scheduler, optimizer, synced_all_rank_steps)
        fuse_rank, fuse_grad, fuse_param, fuse_bad = _sync_fuse_info(
            triggered=local_fuse_triggered,
            rank=rank,
            grad_norm=last_fuse_grad,
            param_norm=last_fuse_param_norm,
            bad_steps=last_fuse_bad_steps,
            device=device,
        )
        stop_requested = _sync_bool_or(
            stop_requested
            or fuse_stopped
            or should_stop_local()
            or (world_size > 1 and args.max_steps > 0 and synced_all_rank_steps >= args.max_steps),
            device,
        )
        if is_rank0(rank):
            print(
                f"[sync] all_rank_steps={synced_all_rank_steps} "
                f"round_weight={round_weight:.1f} round_eps={round_episodes_sum} "
                f"total_eps={total_episodes_processed} "
                f"rank0_step={global_step}/{total_steps} stop={int(stop_requested)}"
            )
            if tb is not None:
                tb.add_scalar("train/sync/round_weight", float(round_weight), synced_all_rank_steps)
                tb.add_scalar("train/sync/episodes_this_round", float(round_episodes_sum), synced_all_rank_steps)
                tb.add_scalar("train/sync/episodes_total", float(total_episodes_processed), synced_all_rank_steps)
                tb.add_scalar("train/sync/all_rank_steps", float(synced_all_rank_steps), synced_all_rank_steps)
            if fuse_stopped and not fuse_stop_saved:
                emergency = output_dir / f"fuse_stop_step_{max(synced_all_rank_steps, global_step)}"
                emergency.mkdir(parents=True, exist_ok=True)
                bundle.unwrap().save_pretrained(str(emergency))
                _save_adapter_config(emergency, bundle, args)
                (emergency / "fuse_reason.txt").write_text(
                    (
                        "vision fuse triggered\n"
                        f"trigger_rank={fuse_rank}\n"
                        f"grad={fuse_grad:.4f}\n"
                        f"param={fuse_param:.4f}\n"
                        f"bad_steps={fuse_bad}\n"
                        f"all_rank_steps={synced_all_rank_steps}\n"
                    ),
                    encoding="utf-8",
                )
                fuse_stop_saved = True
                print(f"[fuse] diagnostic adapter -> {emergency}")
            if (
                (not args.check)
                and (not fuse_stopped)
                and args.save_steps > 0
                and synced_all_rank_steps > 0
                and synced_all_rank_steps // args.save_steps > last_saved_step // args.save_steps
            ):
                ckpt = output_dir / f"checkpoint-{synced_all_rank_steps}"
                bundle.unwrap().save_pretrained(str(ckpt))
                _save_adapter_config(ckpt, bundle, args)
                saved.append(ckpt)
                last_saved_step = synced_all_rank_steps
                print(f"[save] averaged checkpoint -> {ckpt}")
                if args.save_total_limit > 0 and len(saved) > args.save_total_limit:
                    old = saved.pop(0)
                    shutil.rmtree(old, ignore_errors=True)
        round_local_steps = 0
        round_local_episodes = 0

    for epoch in range(args.num_epochs):
        if is_rank0(rank):
            print(f"[epoch {epoch}] starting train_total={train_total}")
        # 所有 rank 用同一 seed 重排 epoch_order，保证拿到同一份 idx -> ep 映射
        epoch_rng = random.Random(args.seed * 131 + epoch * 17)
        epoch_order = list(range(train_total))
        epoch_rng.shuffle(epoch_order)

        for round_idx in range(rounds_per_epoch):
            if stop_requested:
                break
            round_start = round_idx * round_size
            round_end = min(train_total, round_start + round_size)
            counter_key = f"sft_v3_epoch_{epoch}_round_{round_idx}_counter"
            local_counter = [round_start]
            if store is not None and is_rank0(rank):
                store.set(counter_key, str(round_start))
            if store is not None:
                store.wait([counter_key])  # 确保本轮 counter 已写入，再允许各 rank 开抢
            else:
                _barrier()

            while True:
                if stop_requested or fuse_stopped or should_stop_local():
                    stop_requested = True
                    break

                if store is not None:
                    new_count = int(store.add(counter_key, 1))
                    idx = new_count - 1
                else:
                    idx = local_counter[0]
                    local_counter[0] += 1

                if idx >= round_end:
                    break

                ep = train_ds.rows[epoch_order[idx]]
                round_local_episodes += 1
                # 限频 [claim] 日志：每轮第 1 条 + 每 8 条打一次（rate-limit），用 stderr
                # 避开 [train] 的高频 stdout。让 work-stealing 实际怎么分配任务变得可见——
                # 比如某 rank 一轮内连续抢到 16 条说明它在独吃，其他 rank 大概率被卡住了。
                if round_local_episodes == 1 or round_local_episodes % 8 == 0:
                    print(
                        f"[claim] rank={rank}/{world_size} round={round_idx + 1}/{rounds_per_epoch} "
                        f"idx={idx} round_eps={round_local_episodes} "
                        f"elapsed={(time.time() - start_time) / 60.0:.1f}m",
                        file=sys.stderr,
                        flush=True,
                    )
                for pack in iter_episode_loss_packs(
                    bundle,
                    ep,
                    max_length=args.max_length,
                    outer_stride=args.outer_stride,
                ):
                    if stop_requested or fuse_stopped or should_stop_local():
                        stop_requested = True
                        break
                    loss = (
                        args.w_a1 * pack.a1
                        + args.w_rs1 * pack.rs1
                        + args.w_a2 * pack.a2
                        + args.w_s2 * pack.s2
                        + args.w_a3 * pack.a3
                        + args.w_s3_status * pack.s3_status
                        + args.w_s3_subgoal * pack.s3_subgoal
                    )
                    (loss / args.grad_accum).backward()
                    accum_loss += _to_float(loss)
                    accum_steps += 1
                    stats["frames"] += 1
                    stats["step2"] += int(pack.step2_ran)
                    stats["step3"] += int(pack.step3_ran)
                    stats["rs_flip"] += int(pack.rs_flip)
                    stats["flip"] += int(pack.scene_flip)
                    stats["leak1"] += int(pack.leak1)
                    stats["leak2"] += int(pack.leak2)
                    stats["leak3"] += int(pack.leak3)
                    stats["phase_a"] += int(pack.phase_a)
                    stats["loss_a1"] += _to_float(pack.a1)
                    stats["loss_rs1"] += _to_float(pack.rs1)
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
                        # 仅本地标记；fuse_stopped 真正生效要等 sync barrier allreduce。
                        # 诊断文件统一在 sync 后由 rank0 写，确保非 rank0 触发也有记录。
                        fuse_stopped = True
                        last_fuse_grad = vis_grad
                        last_fuse_param_norm = vis_param_norm
                        last_fuse_bad_steps = guard_bad_steps
                        break

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    round_local_steps += 1

                    if is_rank0(rank) and (global_step == 1 or global_step % max(1, args.logging_steps) == 0):
                        frames = max(stats["frames"], 1)
                        loss_scalar = accum_loss / max(accum_steps, 1)
                        lrs = scheduler.get_last_lr()
                        print(
                            f"[train] epoch={epoch} round={round_idx + 1}/{rounds_per_epoch} "
                            f"step={global_step}/{total_steps} loss={loss_scalar:.4f} "
                            f"lr={float(lrs[0]) if lrs else 0.0:.2e} "
                            f"step2={_safe_ratio(stats['step2'], frames):.3f} "
                            f"step3={_safe_ratio(stats['step3'], frames):.3f} "
                            f"rs_flip={_safe_ratio(stats['rs_flip'], frames):.3f} "
                            f"flip={_safe_ratio(stats['flip'], frames):.3f} "
                            f"leak1={_safe_ratio(stats['leak1'], frames):.3f} "
                            f"leak2={_safe_ratio(stats['leak2'], frames):.3f} "
                            f"leak3={_safe_ratio(stats['leak3'], max(stats['step3'], 1)):.3f} "
                            f"phase_a={_safe_ratio(stats['phase_a'], frames):.3f} "
                            f"all_rank_steps={synced_all_rank_steps} "
                            f"elapsed={(time.time() - start_time) / 60.0:.1f}m"
                        )
                        if tb is not None:
                            language_param_norm = _param_norm(language_params) if language_params else 0.0
                            tb.add_scalar("train/loss_total", loss_scalar, global_step)
                            tb.add_scalar("train/loss/a1", stats["loss_a1"] / frames, global_step)
                            tb.add_scalar("train/loss/rs1", stats["loss_rs1"] / frames, global_step)
                            tb.add_scalar("train/loss/a2", stats["loss_a2"] / frames, global_step)
                            tb.add_scalar("train/loss/a3", stats["loss_a3"] / frames, global_step)
                            tb.add_scalar("train/loss/s2", stats["loss_s2"] / frames, global_step)
                            tb.add_scalar("train/loss/s3_status", stats["loss_s3_status"] / frames, global_step)
                            tb.add_scalar("train/loss/s3_subgoal", stats["loss_s3_subgoal"] / frames, global_step)
                            tb.add_scalar("train/loss_weight/a1", float(args.w_a1), global_step)
                            tb.add_scalar("train/loss_weight/rs1", float(args.w_rs1), global_step)
                            tb.add_scalar("train/loss_weight/a2", float(args.w_a2), global_step)
                            tb.add_scalar("train/loss_weight/a3", float(args.w_a3), global_step)
                            tb.add_scalar("train/loss_weight/s2", float(args.w_s2), global_step)
                            tb.add_scalar("train/loss_weight/s3_status", float(args.w_s3_status), global_step)
                            tb.add_scalar("train/loss_weight/s3_subgoal", float(args.w_s3_subgoal), global_step)
                            tb.add_scalar("train/lr", float(lrs[0]) if lrs else 0.0, global_step)
                            if len(lrs) > 1:
                                tb.add_scalar("train/lr_vision", float(lrs[1]), global_step)
                            tb.add_scalar("train/step2_trigger_rate", _safe_ratio(stats["step2"], frames), global_step)
                            tb.add_scalar("train/step3_trigger_rate", _safe_ratio(stats["step3"], frames), global_step)
                            tb.add_scalar("train/road_structure_flip_rate", _safe_ratio(stats["rs_flip"], frames), global_step)
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
                            tb.add_scalar("train/all_rank_steps_at_last_sync", float(synced_all_rank_steps), global_step)

                    accum_steps = 0
                    accum_loss = 0.0
                    for key in stats:
                        stats[key] = 0

                    # in-loop eval 仅单进程：多 rank 下各 rank 参数还未平均，eval 没意义
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

                # episode 处理完，继续在当前 round 抢下一个。

            do_sync_round(f"sft_v3_epoch_{epoch}_round_{round_idx}_sync")
            if stop_requested:
                break

        if stop_requested:
            break

    if tb is not None:
        tb.flush()
        tb.close()

    # 训练结束时 rank0 保存最终 adapter。do_sync_round 已经把各 rank LoRA 参数
    # allreduce 平均回同一份，rank0 保存的就是 averaged adapter。
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
    _store_rendezvous(store, "sft_v3_final_save_done", rank=rank, world_size=world_size)
    cleanup_distributed()


if __name__ == "__main__":
    main()
