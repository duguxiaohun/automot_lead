#!/usr/bin/env python3
"""训练 sft_loop_phase3 的 RS-gated EVENT 二元问题 LoRA adapter。

训练目标是当前 RS gate 下出现的 UE1/UE3/UE5/UE6/INVALID YES/NO 语义 token，
并以低权重监督字段格式和 assistant 结束符。数据构建阶段保证四个 UE 正类
1:1:1:1，并额外注入约 20% wrong-RS invalid 样本。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
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
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_loop_phase3 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase3.history_rgb import (  # noqa: E402
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODES,
    history_rgb_indices,
    history_rgb_mode_tag,
    select_history_rgb_paths,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_loop_phase3.prompts import (  # noqa: E402
    ANSWER_KEYS,
    GROUP_DEFINITIONS,
    INVALID_KEY,
    PROMPT_NAME,
    SUBSET_COUNTS,
    VARIANT_ORDER,
    VARIANT_WEIGHTS,
    PromptSpec,
    build_phase3_messages,
    build_phase3_target,
    make_prompt_spec,
    parse_phase3_output,
    phase3_prompt_sha256,
    prompt_spec_to_json,
    spec_metric_items,
)
from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _assert_inside_assistant_turn,
    _find_subsequence,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


FORMAT_COMPONENT_ID = -1


def setup_distributed() -> Tuple[int, int, int]:
    """初始化可选 torchrun DDP。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_loop_phase3 DDP requires CUDA.")
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def ddp_barrier(local_rank: int) -> None:
    """在当前 rank 绑定的 GPU 上执行 barrier，避免 NCCL 猜测设备映射。"""

    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier(device_ids=[int(local_rank)])
    except TypeError:
        dist.barrier()


def cleanup_distributed() -> None:
    """清理 torch.distributed。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _append_jsonl(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """向运行目录追加一行 JSON 指标，便于不用 TensorBoard 也能审计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _tb_tag(key: str) -> str:
    """把内部 metric key 转成稳定 TensorBoard tag。"""

    return str(key).replace(":", "/").replace(" ", "_")


def _write_scalar_dict(writer: Any, prefix: str, metrics: Mapping[str, Any], step: int) -> None:
    """批量写 TensorBoard scalar，并统一规整 tag。"""

    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{_tb_tag(key)}", float(value), int(step))


def _write_run_metadata(
    writer: Any,
    output_dir: pathlib.Path,
    args: argparse.Namespace,
    *,
    world_size: int,
    train_rows: int,
    train_work_global: int,
    train_work_rank: int,
    eval_work_rank: int,
    generation_eval_global: int,
    total_steps: int,
) -> None:
    """写入 TB 和 JSON manifest，保证训练目录、TB、日志路径互相可追。"""

    payload = {
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "output_dir": str(output_dir),
        "tb_dir": str(output_dir / "tb"),
        "run_log": os.environ.get("RUN_LOG", ""),
        "run_name": os.environ.get("RUN_NAME", ""),
        "run_timestamp": os.environ.get("RUN_TIMESTAMP", ""),
        "model_dir": str(args.model_dir),
        "index": str(args.index),
        "split": str(args.split),
        "eval_split": str(args.eval_split),
        "world_size": int(world_size),
        "history_rgb_mode": str(args.history_rgb_mode),
        "history_rgb_count": len(history_rgb_indices(args.history_rgb_mode)),
        "history_rgb_selected_indices": list(history_rgb_indices(args.history_rgb_mode)),
        "train_rows": int(train_rows),
        "train_work_global": int(train_work_global),
        "train_work_rank": int(train_work_rank),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "eval_work_rank": int(eval_work_rank),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_global": int(generation_eval_global),
        "save_best_val": bool(args.save_best_val),
        "save_best_generation": bool(args.save_best_generation),
        "save_steps": int(args.save_steps),
        "total_steps_global": int(total_steps),
        "focus_balance_count": int(args.focus_balance_count),
        "regular_focus_multiplier": float(args.regular_focus_multiplier),
        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
        "production_prompt_sha256": phase3_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
    }
    (output_dir / "train_run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if writer is None:
        return
    writer.add_text(
        "run/summary",
        "\n".join(
            [
                f"output_dir: `{payload['output_dir']}`",
                f"tb_dir: `{payload['tb_dir']}`",
                f"run_log: `{payload['run_log'] or 'not set'}`",
                f"train: rows={train_rows}, global_work={train_work_global}, rank_work={train_work_rank}",
                f"val: eval_steps={args.eval_steps}, eval_work_rank={eval_work_rank}",
                f"generation: steps={args.generation_eval_steps}, global_cases={generation_eval_global}",
                f"best_generation: `{payload['save_best_generation']}`",
                f"prompt_sha256: `{payload['production_prompt_sha256']}`",
            ]
        ),
        0,
    )
    for key in (
        "world_size",
        "history_rgb_count",
        "train_rows",
        "train_work_global",
        "train_work_rank",
        "eval_work_rank",
        "generation_eval_global",
        "total_steps_global",
        "focus_balance_count",
        "regular_focus_multiplier",
        "invalid_focus_multiplier",
    ):
        writer.add_scalar(f"setup/{key}", float(payload[key]), 0)


@dataclass
class FrameRow:
    """一帧训练样本。"""

    scenario: str
    route_id: str
    town: str
    frame_id: int
    rs: str
    event: str
    split: str
    history_rgb_paths: List[str]
    latest_rgb_path: str
    answers: Dict[str, bool]


@dataclass(frozen=True)
class WorkItem:
    """一条增强训练/验证 case。"""

    row: FrameRow
    spec: PromptSpec
    balance_key: str


def _read_rows(path: pathlib.Path, split: str, max_frames: int = 0) -> List[FrameRow]:
    """读取 frame_index.jsonl。"""

    rows: List[FrameRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row_dataset = obj.get("dataset_name")
            if row_dataset != DATASET_NAME:
                raise ValueError(f"dataset_name mismatch in {path}: {row_dataset!r}")
            if str(obj.get("split")) != str(split):
                continue
            rows.append(
                FrameRow(
                    scenario=str(obj.get("scenario")),
                    route_id=str(obj.get("route_id")),
                    town=str(obj.get("town")),
                    frame_id=int(obj.get("frame_id")),
                    rs=str(obj.get("rs")),
                    event=str(obj.get("event")),
                    split=str(obj.get("split")),
                    history_rgb_paths=[str(x) for x in obj.get("history_rgb_paths", [])],
                    latest_rgb_path=str(obj.get("latest_rgb_path")),
                    answers={key: bool(value) for key, value in (obj.get("answers") or {}).items()},
                )
            )
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {path}")
    return rows


def _work_item_seed(row: FrameRow, *parts: object) -> str:
    """返回增强 spec 的稳定种子字段。"""

    return ":".join(
        [
            row.scenario,
            row.route_id,
            str(row.frame_id),
            row.rs,
            *[str(part) for part in parts],
        ]
    )


def _answer_text(value: bool) -> str:
    """布尔转 YES/NO。"""

    return "YES" if bool(value) else "NO"


def _target_class(row: FrameRow) -> str:
    """从 answers 恢复本行 phase3 目标类。"""

    if row.answers.get("INVALID_RS_CONTEXT", False):
        return "INVALID"
    positives = [key for key in ("UE1", "UE3", "UE5", "UE6") if row.answers.get(key, False)]
    if len(positives) == 1:
        return positives[0]
    return "RE"


def _raw_focus_bin_counts(rows: Sequence[FrameRow]) -> Dict[str, int]:
    """统计 phase3 class 与答案桶计数，供 balance artifact 审计。"""

    counts: Counter[str] = Counter()
    for row in rows:
        target_class = _target_class(row)
        counts[f"class/{target_class}"] += 1
        counts[f"rs_context/{row.rs}"] += 1
        for key in ANSWER_KEYS:
            counts[f"answer/{key}:{_answer_text(row.answers.get(key, False))}"] += 1
    return dict(counts)


def _make_all_item(row: FrameRow, focus: str, *, seed: int) -> WorkItem:
    """构造 phase3 事件 gate case。"""

    spec = make_prompt_spec(
        variant="all_random_order",
        answers=row.answers,
        seed_key=_work_item_seed(row, seed, "all", focus),
        focus=focus,
    )
    return WorkItem(row=row, spec=spec, balance_key=f"all_random_order/class/{_target_class(row)}")


def _iter_candidate_items(rows: Sequence[FrameRow], *, seed: int) -> Iterable[WorkItem]:
    """逐个生成 phase3 候选；每帧只对应一个事件 gate prompt。"""

    for row in rows:
        yield _make_all_item(row, _target_class(row), seed=seed)


def _balance_target_for_key(
    key: str,
    *,
    target_per_bin: int,
    regular_multiplier: float,
    invalid_multiplier: float,
) -> int:
    """返回训练采样目标；UE 正类保持等量，RE/invalid 可单独加权。"""

    target = int(target_per_bin)
    if target <= 0:
        return target
    if key.endswith("/class/RE"):
        return max(1, int(round(float(target) * float(regular_multiplier))))
    if key.endswith("/class/INVALID"):
        return max(1, int(round(float(target) * float(invalid_multiplier))))
    return target


def _balanced_work(
    rows: Sequence[FrameRow],
    *,
    target_per_bin: int,
    seed: int,
    regular_multiplier: float = 1.0,
    invalid_multiplier: float = 1.0,
) -> List[WorkItem]:
    """按 phase3 class 构建 deterministic work list。

    数据索引已按 split 保证 UE1/UE3/UE5/UE6、RE、invalid 的目标比例；
    这里在 target_per_bin>0 时按各 class 截取/循环，用于快速 check 或固定步数训练。
    UE1/UE3/UE5/UE6 始终共享同一个目标数；RE 和 invalid 可独立倍率放大，
    用于补强 hard negative，同时保持 UE 正类 1:1:1:1。
    """

    groups: Dict[str, List[WorkItem]] = defaultdict(list)
    for item in _iter_candidate_items(rows, seed=seed):
        groups[item.balance_key].append(item)
    if not groups:
        return []
    rng = random.Random(
        f"{seed}:phase3_balance:{len(rows)}:{int(target_per_bin)}:"
        f"{float(regular_multiplier):.6f}:{float(invalid_multiplier):.6f}"
    )
    work: List[WorkItem] = []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        target = _balance_target_for_key(
            key,
            target_per_bin=int(target_per_bin),
            regular_multiplier=float(regular_multiplier),
            invalid_multiplier=float(invalid_multiplier),
        )
        if target <= 0:
            work.extend(items)
        elif len(items) >= target:
            work.extend(items[:target])
        else:
            repeated = [items[i % len(items)] for i in range(target)]
            rng.shuffle(repeated)
            work.extend(repeated)
    rng.shuffle(work)
    return work

def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    """读取当前 RGB-history 合同选择出的图片。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _messages(
    images: List[Image.Image],
    *,
    spec: PromptSpec,
    history_rgb_mode: str,
    target: str,
) -> List[Dict[str, Any]]:
    """构造 teacher-forced chat。"""

    return build_phase3_messages(
        images=images,
        spec=spec,
        audit=False,
        history_rgb_mode=history_rgb_mode,
        target=target,
    )


def _line_value_span(text: str, key: str) -> Tuple[int, int]:
    """返回某个答案行中 YES/NO 的字符 span。"""

    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(YES|NO)\b", text)
    if not match:
        raise ValueError(f"target missing {key}: {text!r}")
    return match.start(1), match.end(1)


def _target_token_weights(
    bundle: Any,
    target: str,
    *,
    spec: PromptSpec,
    format_loss_weight: float,
) -> Tuple[List[int], List[float], List[int]]:
    """映射当前输出的语义与格式 token 权重。

    YES/NO 是被问到题目的主监督，权重固定为 1。字段名、冒号和换行只承担自由生成
    的语法锚定，使用较低权重；它们绝不编码四题之间的任何逻辑关系。
    """

    enc = bundle.tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    weights = [float(format_loss_weight) for _ in token_ids]
    component_ids = [FORMAT_COMPONENT_ID for _ in token_ids]
    for component_id, key in enumerate(spec.output_keys, start=1):
        lo, hi = _line_value_span(target, key)
        for i, (a, b) in enumerate(offsets):
            if a < hi and b > lo:
                weights[i] = 1.0
                component_ids[i] = component_id
    return token_ids, weights, component_ids


def _assistant_end_token_ids(bundle: Any) -> set[int]:
    """返回 chat template 中可作为 assistant turn 结束符的 token id。"""

    ids: set[int] = set()
    eos = getattr(bundle.tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple, set)):
        ids.update(int(x) for x in eos)
    elif eos is not None:
        ids.add(int(eos))
    im_end = bundle.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.add(int(im_end))
    return ids


def _build_inputs(
    bundle: Any,
    *,
    images: List[Image.Image],
    history_rgb_mode: str,
    target: str,
    spec: PromptSpec,
    max_length: int,
    format_loss_weight: float,
) -> Optional[Dict[str, Any]]:
    """构造模型输入，主监督答案值并低权重监督四行格式与结束符。"""

    messages = _messages(images, spec=spec, history_rgb_mode=history_rgb_mode, target=target)
    chat_text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = bundle.processor(text=[chat_text], images=images, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"][0]
    if int(input_ids.shape[0]) > int(max_length):
        return None
    labels = input_ids.clone()
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    component_ids = torch.zeros_like(input_ids, dtype=torch.long)
    expanded = [int(x) for x in input_ids.tolist()]
    target_ids, token_weights, token_components = _target_token_weights(
        bundle,
        target,
        spec=spec,
        format_loss_weight=float(format_loss_weight),
    )
    pos = _find_subsequence(expanded, target_ids, 0)
    asst_header_ids = list(bundle.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])
    _assert_inside_assistant_turn(expanded, pos, asst_header_ids, 0)
    for j, weight in enumerate(token_weights):
        if weight > 0:
            weights[pos + j] = float(weight)
            component_ids[pos + j] = int(token_components[j])
    # assistant 的 `<|im_end|>` 未包含在 target 文本中；监督它能避免 LoRA 自由生成
    # 裸 YES/NO 后继续重复。只看 target 后很短的模板尾部，绝不波及 user/prompt token。
    end_ids = _assistant_end_token_ids(bundle)
    for end_pos in range(pos + len(target_ids), min(len(expanded), pos + len(target_ids) + 4)):
        if expanded[end_pos] in end_ids:
            weights[end_pos] = float(format_loss_weight)
            component_ids[end_pos] = FORMAT_COMPONENT_ID
            break
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "loss_weights": weights,
        "component_ids": component_ids,
        "vision": extra,
    }


def _loss_one(bundle: Any, packed: Mapping[str, Any], spec: PromptSpec) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算一个样本的 weighted CE。"""

    kwargs: Dict[str, Any] = {
        "input_ids": packed["input_ids"].unsqueeze(0).to(bundle.device),
        "attention_mask": packed["attention_mask"].unsqueeze(0).to(bundle.device),
    }
    labels = packed["labels"].unsqueeze(0).to(bundle.device)
    weights = packed["loss_weights"].unsqueeze(0).to(bundle.device)
    comp = packed["component_ids"].unsqueeze(0).to(bundle.device)
    for key, value in packed["vision"].items():
        kwargs[key] = value.to(bundle.device) if isinstance(value, torch.Tensor) else value
    out = bundle.model(**kwargs, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    shift_comp = comp[:, 1:].contiguous()
    active = shift_weights.gt(0)
    if not bool(active.any()):
        zero = logits.sum() * 0.0
        return zero, {"denom": 0.0}
    per_tok = F.cross_entropy(logits[active], shift_labels[active], reduction="none")
    numerator = (per_tok * shift_weights[active]).sum()
    denom = shift_weights[active].sum().clamp_min(1.0)
    loss = numerator / denom
    pred = logits.argmax(dim=-1)
    value_active = active & shift_comp.gt(0)
    format_active = active & shift_comp.eq(FORMAT_COMPONENT_ID)
    stats: Dict[str, float] = {
        "denom": float(denom.detach().item()),
        "token_acc": float(torch.equal(pred[active], shift_labels[active])),
        "value_token_acc": float(bool(value_active.any()) and torch.equal(pred[value_active], shift_labels[value_active])),
        "format_token_acc": float(bool(format_active.any()) and torch.equal(pred[format_active], shift_labels[format_active])),
    }
    for component_id, key in enumerate(spec.output_keys, start=1):
        mask = active & shift_comp.eq(component_id)
        stats[f"answer/{key.lower()}_ok"] = float(bool(mask.any() and torch.equal(pred[mask], shift_labels[mask])))
    return loss, stats


def _ddp_sync_zero_loss(bundle: Any, images: Sequence[Image.Image]) -> torch.Tensor:
    """样本过长跳过时，仍走一次真实 model forward 来同步 DDP reducer。

    直接用 trainable parameter 求和能产生零梯度，但不会触发 DDP wrapper 的 forward。
    如果某些 rank 正常 forward、某些 rank 只做参数零和，reducer 的边界仍可能不一致。
    这里用最短的图文 chat 跑过同一个 DDP model，再把 logits 归零参与 backward。
    """

    content: List[Dict[str, object]] = []
    sync_images = list(images[:1])
    for image in sync_images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": "SYNC"})
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": "Synchronize a skipped training sample."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "OK"},
    ]
    chat_text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if sync_images:
        inputs = bundle.processor(text=[chat_text], images=sync_images, return_tensors="pt", padding=True)
    else:
        inputs = bundle.processor(text=[chat_text], return_tensors="pt", padding=True)
    kwargs = {key: value.to(bundle.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    out = bundle.model(**kwargs, use_cache=False, return_dict=True)
    return out.logits.sum() * 0.0


def _optimizer_step(
    *,
    params: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    max_grad_norm: float,
) -> None:
    """统一执行一次 optimizer step，避免累积/保存路径各写一份逻辑。"""

    torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _split_work_for_rank(work: Sequence[WorkItem], *, rank: int, world_size: int) -> List[WorkItem]:
    """按 rank 切分同一个均衡全集。"""

    if int(world_size) <= 1:
        return list(work)
    shard = list(work)[int(rank) :: int(world_size)]
    if not shard:
        raise ValueError(f"rank {rank} got empty work shard; reduce WORLD_SIZE or increase balance count")
    return shard


@torch.no_grad()
def evaluate_loss(
    bundle: Any,
    work: Sequence[WorkItem],
    *,
    history_rgb_mode: str,
    max_length: int,
    format_loss_weight: float,
    device: torch.device,
    world_size: int,
) -> Dict[str, float]:
    """在独立 val split 上跑 teacher-forced loss 和四问 token accuracy。"""

    was_training = bool(bundle.model.training)
    bundle.model.eval()
    loss_sum = 0.0
    samples = 0
    skipped = 0
    token_acc_sum = 0.0
    value_token_acc_sum = 0.0
    format_token_acc_sum = 0.0
    metric_ok: Counter[str] = Counter()
    metric_count: Counter[str] = Counter()
    variant_ok: Counter[str] = Counter()
    variant_count: Counter[str] = Counter()
    invalid_gt_total = 0.0
    invalid_line_ok = 0.0
    invalid_ue_all_no_ok = 0.0
    invalid_joint_ok = 0.0
    for item in work:
        row = item.row
        spec = item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        target = build_phase3_target(spec)
        packed = _build_inputs(
            bundle,
            images=images,
            history_rgb_mode=history_rgb_mode,
            target=target,
            spec=spec,
            max_length=int(max_length),
            format_loss_weight=float(format_loss_weight),
        )
        if packed is None:
            skipped += 1
            continue
        loss, stats = _loss_one(bundle, packed, spec)
        loss_value = float(loss.detach().item())
        loss_sum += loss_value
        samples += 1
        token_acc_sum += float(stats.get("token_acc", 0.0))
        value_token_acc_sum += float(stats.get("value_token_acc", 0.0))
        format_token_acc_sum += float(stats.get("format_token_acc", 0.0))
        variant_count[spec.variant] += 1
        all_answer_ok = True
        for output_key, metric_key, _ in spec_metric_items(spec):
            ok = float(stats.get(f"answer/{output_key.lower()}_ok", 0.0))
            metric_ok[metric_key] += ok
            metric_count[metric_key] += 1
            all_answer_ok = all_answer_ok and bool(ok)
        variant_ok[spec.variant] += int(all_answer_ok)
        if bool(row.answers.get(INVALID_KEY, False)):
            invalid_gt_total += 1.0
            invalid_line = bool(stats.get(f"answer/{INVALID_KEY.lower()}_ok", 0.0))
            event_all_no = all(
                bool(stats.get(f"answer/{key.lower()}_ok", 0.0))
                for key in spec.output_keys
                if key != INVALID_KEY
            )
            invalid_line_ok += float(invalid_line)
            invalid_ue_all_no_ok += float(event_all_no)
            invalid_joint_ok += float(invalid_line and event_all_no)
    metric_names = list(ANSWER_KEYS)
    values = [
        loss_sum,
        float(samples),
        float(skipped),
        token_acc_sum,
        value_token_acc_sum,
        format_token_acc_sum,
        invalid_gt_total,
        invalid_line_ok,
        invalid_ue_all_no_ok,
        invalid_joint_ok,
    ]
    values.extend(metric_ok[name] for name in metric_names)
    values.extend(metric_count[name] for name in metric_names)
    values.extend(variant_ok[name] for name in VARIANT_ORDER)
    values.extend(variant_count[name] for name in VARIANT_ORDER)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    vals = [float(x) for x in tensor.detach().cpu().tolist()]
    total_samples = max(1.0, vals[1])
    offset = 10
    metrics: Dict[str, float] = {
        "loss": vals[0] / total_samples,
        "samples": vals[1],
        "skipped": vals[2],
        "token_acc": vals[3] / total_samples,
        "value_token_acc": vals[4] / total_samples,
        "format_token_acc": vals[5] / total_samples,
        "invalid_gt_total": vals[6],
        "invalid_line_token_ok_rate": vals[7] / max(1.0, vals[6]),
        "invalid_ue_all_no_token_ok_rate": vals[8] / max(1.0, vals[6]),
        "invalid_joint_token_ok_rate": vals[9] / max(1.0, vals[6]),
    }
    metric_ok_vals = vals[offset : offset + len(metric_names)]
    offset += len(metric_names)
    metric_count_vals = vals[offset : offset + len(metric_names)]
    offset += len(metric_names)
    for idx, key in enumerate(metric_names):
        safe = key.lower().replace(":", "_")
        metrics[f"metric/{safe}_acc"] = metric_ok_vals[idx] / max(1.0, metric_count_vals[idx])
        metrics[f"metric/{safe}_samples"] = metric_count_vals[idx]
    variant_ok_vals = vals[offset : offset + len(VARIANT_ORDER)]
    offset += len(VARIANT_ORDER)
    variant_count_vals = vals[offset : offset + len(VARIANT_ORDER)]
    for idx, key in enumerate(VARIANT_ORDER):
        metrics[f"variant/{key}_exact"] = variant_ok_vals[idx] / max(1.0, variant_count_vals[idx])
        metrics[f"variant/{key}_samples"] = variant_count_vals[idx]
    if was_training:
        bundle.model.train()
    return metrics


def _generation_messages(
    images: List[Image.Image],
    *,
    spec: PromptSpec,
    history_rgb_mode: str,
) -> List[Dict[str, Any]]:
    """构造不含 target 的生产式 chat，用于检查实际自由生成格式。"""

    return build_phase3_messages(
        images=images,
        spec=spec,
        audit=False,
        history_rgb_mode=history_rgb_mode,
        target=None,
    )


def _dynamic_answer_pattern(values: Mapping[str, Optional[str]]) -> str:
    """对当前被问到的输出行统计 ALL_NO / 单 YES / 多 YES / INVALID。"""

    if any(value not in ("YES", "NO") for value in values.values()):
        return "INVALID"
    positive = [key for key, value in values.items() if value == "YES"]
    if not positive:
        return "ALL_NO"
    if len(positive) == 1:
        return positive[0]
    return "MULTI:" + "+".join(sorted(positive))


def _update_generation_pattern_counters(
    counter: Counter[str],
    *,
    spec: PromptSpec,
    row: FrameRow,
    gt: Mapping[str, str],
    parsed: Mapping[str, Optional[str]],
    raw_output: str,
) -> None:
    """累计训练期 generation probe 的增强答案模式诊断。"""

    gt_pattern = _dynamic_answer_pattern(gt)
    pred_pattern = _dynamic_answer_pattern(parsed)
    variant = spec.variant
    counter[f"{variant}/total"] += 1
    counter[f"{variant}/gt_pattern/{gt_pattern}"] += 1
    counter[f"{variant}/pred_pattern/{pred_pattern}"] += 1
    counter[f"{variant}/pair/{gt_pattern}=>{pred_pattern}"] += 1
    counter[f"{variant}/pattern_exact"] += int(gt_pattern == pred_pattern)
    counter[f"{variant}/pred_invalid"] += int(pred_pattern == "INVALID")
    counter[f"{variant}/gt_all_no"] += int(gt_pattern == "ALL_NO")
    counter[f"{variant}/pred_all_no"] += int(pred_pattern == "ALL_NO")
    counter[f"{variant}/gt_multi_yes"] += int(gt_pattern.startswith("MULTI:"))
    counter[f"{variant}/pred_multi_yes"] += int(pred_pattern.startswith("MULTI:"))
    event_keys = [key for key in spec.output_keys if key != INVALID_KEY]
    gt_invalid = gt.get(INVALID_KEY) == "YES"
    pred_invalid_yes = parsed.get(INVALID_KEY) == "YES"
    pred_events_all_no = all(parsed.get(key) == "NO" for key in event_keys)
    if gt_invalid:
        counter[f"{variant}/invalid_gt_total"] += 1
        counter[f"{variant}/invalid_pred_line_yes"] += int(pred_invalid_yes)
        counter[f"{variant}/invalid_ue_all_no"] += int(pred_events_all_no)
        counter[f"{variant}/invalid_joint_ok"] += int(pred_invalid_yes and pred_events_all_no)
    if pred_invalid_yes:
        counter[f"{variant}/pred_invalid_yes_total"] += 1
        counter[f"{variant}/pred_invalid_yes_ue_all_no"] += int(pred_events_all_no)


def _pattern_report(counter: Counter[str]) -> Dict[str, Any]:
    """把训练期 generation probe 的答案模式 counter 整理成 eval 同款结构。"""

    out: Dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        total = int(counter.get(f"{variant}/total", 0))
        denom = max(1.0, float(total))
        out[variant] = {
            "total": total,
            "pattern_exact_accuracy": float(counter.get(f"{variant}/pattern_exact", 0)) / denom,
            "pred_invalid_rate": float(counter.get(f"{variant}/pred_invalid", 0)) / denom,
            "invalid_gt_total": int(counter.get(f"{variant}/invalid_gt_total", 0)),
            "invalid_line_yes_rate": float(counter.get(f"{variant}/invalid_pred_line_yes", 0))
            / max(1.0, float(counter.get(f"{variant}/invalid_gt_total", 0))),
            "invalid_ue_all_no_rate": float(counter.get(f"{variant}/invalid_ue_all_no", 0))
            / max(1.0, float(counter.get(f"{variant}/invalid_gt_total", 0))),
            "invalid_joint_ok_rate": float(counter.get(f"{variant}/invalid_joint_ok", 0))
            / max(1.0, float(counter.get(f"{variant}/invalid_gt_total", 0))),
            "pred_invalid_yes_total": int(counter.get(f"{variant}/pred_invalid_yes_total", 0)),
            "pred_invalid_yes_ue_all_no_rate": float(counter.get(f"{variant}/pred_invalid_yes_ue_all_no", 0))
            / max(1.0, float(counter.get(f"{variant}/pred_invalid_yes_total", 0))),
            "gt_all_no": int(counter.get(f"{variant}/gt_all_no", 0)),
            "pred_all_no": int(counter.get(f"{variant}/pred_all_no", 0)),
            "gt_multi_yes": int(counter.get(f"{variant}/gt_multi_yes", 0)),
            "pred_multi_yes": int(counter.get(f"{variant}/pred_multi_yes", 0)),
            "gt_patterns": {
                key.removeprefix(f"{variant}/gt_pattern/"): int(value)
                for key, value in sorted(counter.items())
                if key.startswith(f"{variant}/gt_pattern/")
            },
            "pred_patterns": {
                key.removeprefix(f"{variant}/pred_pattern/"): int(value)
                for key, value in sorted(counter.items())
                if key.startswith(f"{variant}/pred_pattern/")
            },
            "pattern_pairs": {
                key.removeprefix(f"{variant}/pair/"): int(value)
                for key, value in sorted(counter.items())
                if key.startswith(f"{variant}/pair/")
            },
        }
    return out


@torch.no_grad()
def evaluate_generation_probe(
    bundle: Any,
    work: Sequence[WorkItem],
    *,
    history_rgb_mode: str,
    max_new_tokens: int,
    record_path: Optional[pathlib.Path] = None,
    step: int = 0,
) -> Dict[str, float]:
    """在固定独立 val 样本上以真实 greedy generation 检查四行格式。

    这是训练选 best adapter 的防退化闸门。它不改变答案、解析或四题独立性；只用
    production prompt 复现最终 eval 的自由生成行为。
    """

    model = bundle.unwrap()
    was_training = bool(model.training)
    model.eval()
    runtime = SimpleNamespace(model=model, processor=bundle.processor, tokenizer=bundle.tokenizer, device=bundle.device)
    samples = 0.0
    valid = 0.0
    exact = 0.0
    variant_valid: Counter[str] = Counter()
    variant_exact: Counter[str] = Counter()
    variant_count: Counter[str] = Counter()
    metric_ok: Counter[str] = Counter()
    metric_count: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    records: List[Dict[str, Any]] = []
    for item in work:
        row = item.row
        spec = item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        state = _kv_start_state(runtime, _generation_messages(images, spec=spec, history_rgb_mode=history_rgb_mode))
        raw, _, _ = _student_generate_kv(runtime, state, int(max_new_tokens))
        parsed = parse_phase3_output(raw, spec=spec)
        gt = {q.output_key: bool(q.answer) for q in spec.questions}
        gt_text = {key: _answer_text(value) for key, value in gt.items()}
        parsed_text = {
            key: None if value is None else _answer_text(value)
            for key, value in parsed.items()
        }
        gt_pattern = _dynamic_answer_pattern(gt_text)
        pred_pattern = _dynamic_answer_pattern(parsed_text)
        is_valid = all(parsed[key] is not None for key in spec.output_keys)
        samples += 1.0
        valid += float(is_valid)
        variant_count[spec.variant] += 1
        variant_valid[spec.variant] += int(is_valid)
        all_ok = False
        if is_valid:
            all_ok = all(bool(parsed[key]) == gt[key] for key in spec.output_keys)
            exact += float(all_ok)
            variant_exact[spec.variant] += int(all_ok)
        for output_key, metric_key, answer in spec_metric_items(spec):
            if parsed.get(output_key) is not None:
                metric_ok[metric_key] += int(bool(parsed[output_key]) == bool(answer))
            metric_count[metric_key] += 1
        _update_generation_pattern_counters(
            pattern_counts,
            spec=spec,
            row=row,
            gt=gt_text,
            parsed=parsed_text,
            raw_output=raw,
        )
        records.append(
            {
                "step": int(step),
                "scenario": row.scenario,
                "route_id": row.route_id,
                "town": row.town,
                "frame_id": row.frame_id,
                "augment_spec": prompt_spec_to_json(spec),
                "balance_key": item.balance_key,
                "answers": row.answers,
                "parsed": parsed,
                "gt_answer_pattern": gt_pattern,
                "pred_answer_pattern": pred_pattern,
                "format_valid": is_valid,
                "all_ok": all_ok,
                "raw_output": raw,
                "history_rgb_paths_used": select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode),
            }
        )
    if was_training:
        model.train()
    metrics: Dict[str, float] = {
        "samples": samples,
        "format_valid_rate": valid / max(1.0, samples),
        "exact_accuracy": exact / max(1.0, samples),
    }
    pattern_report = _pattern_report(pattern_counts)
    metric_names = list(ANSWER_KEYS)
    for key in metric_names:
        safe = key.lower().replace(":", "_")
        metrics[f"metric/{safe}_acc"] = metric_ok[key] / max(1.0, metric_count[key])
        metrics[f"metric/{safe}_samples"] = float(metric_count[key])
    for key in VARIANT_ORDER:
        metrics[f"variant/{key}_valid"] = variant_valid[key] / max(1.0, variant_count[key])
        metrics[f"variant/{key}_exact"] = variant_exact[key] / max(1.0, variant_count[key])
        metrics[f"variant/{key}_samples"] = float(variant_count[key])
        total = max(1.0, float(pattern_counts.get(f"{key}/total", 0)))
        metrics[f"pattern/{key}_pattern_exact"] = float(pattern_counts.get(f"{key}/pattern_exact", 0)) / total
        metrics[f"pattern/{key}_pred_invalid_rate"] = float(pattern_counts.get(f"{key}/pred_invalid", 0)) / total
        metrics[f"pattern/{key}_gt_all_no_rate"] = float(pattern_counts.get(f"{key}/gt_all_no", 0)) / total
        metrics[f"pattern/{key}_pred_all_no_rate"] = float(pattern_counts.get(f"{key}/pred_all_no", 0)) / total
        metrics[f"pattern/{key}_gt_multi_yes_rate"] = float(pattern_counts.get(f"{key}/gt_multi_yes", 0)) / total
        metrics[f"pattern/{key}_pred_multi_yes_rate"] = float(pattern_counts.get(f"{key}/pred_multi_yes", 0)) / total
        invalid_total = max(1.0, float(pattern_counts.get(f"{key}/invalid_gt_total", 0)))
        pred_invalid_yes_total = max(1.0, float(pattern_counts.get(f"{key}/pred_invalid_yes_total", 0)))
        metrics[f"invalid/{key}_gt_total"] = float(pattern_counts.get(f"{key}/invalid_gt_total", 0))
        metrics[f"invalid/{key}_line_yes_rate"] = float(pattern_counts.get(f"{key}/invalid_pred_line_yes", 0)) / invalid_total
        metrics[f"invalid/{key}_ue_all_no_rate"] = float(pattern_counts.get(f"{key}/invalid_ue_all_no", 0)) / invalid_total
        metrics[f"invalid/{key}_joint_ok_rate"] = float(pattern_counts.get(f"{key}/invalid_joint_ok", 0)) / invalid_total
        metrics[f"invalid/{key}_pred_yes_ue_all_no_rate"] = (
            float(pattern_counts.get(f"{key}/pred_invalid_yes_ue_all_no", 0)) / pred_invalid_yes_total
        )
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        report_path = record_path.with_name(f"{record_path.stem}_pattern_reports{record_path.suffix}")
        with report_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": int(step),
                        "samples": int(samples),
                        "answer_pattern_diagnostics": pattern_report,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return metrics


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace, *, step: int, name: str = "final") -> pathlib.Path:
    """保存 LoRA adapter 和 Phase3 自描述配置。"""

    final_dir = output_dir / str(name)
    final_dir.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(final_dir))
    cfg = {
        "schema": "sft_loop_phase3_adapter_config",
        "route": "sft_loop_phase3_event_gate",
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "production_prompt_sha256": phase3_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
        "augment_variants": list(VARIANT_ORDER),
        "augment_variant_weights": dict(VARIANT_WEIGHTS),
        "subset_question_counts": list(SUBSET_COUNTS),
        "hierarchical_group_ids": list(GROUP_DEFINITIONS.keys()),
        "history_rgb_mode": str(args.history_rgb_mode),
        "history_rgb_count": len(history_rgb_indices(args.history_rgb_mode)),
        "history_rgb_selected_indices": list(history_rgb_indices(args.history_rgb_mode)),
        "base_model_dir": str(args.model_dir),
        "lora_vision_scope": str(args.lora_vision_scope),
        "lora_target_modules": list(bundle.lora_target_modules),
        "answer_order": list(ANSWER_KEYS),
        "global_step": int(step),
        "num_epochs": int(args.num_epochs),
        "max_steps": int(args.max_steps),
        "focus_balance_count": int(args.focus_balance_count),
        "regular_focus_multiplier": float(args.regular_focus_multiplier),
        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
        "eval_split": str(args.eval_split),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "format_loss_weight": float(args.format_loss_weight),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_max_new_tokens": int(args.generation_eval_max_new_tokens),
        "generation_eval_min_valid_rate": float(args.generation_eval_min_valid_rate),
        "save_best_val": bool(args.save_best_val),
        "save_best_generation": bool(args.save_best_generation),
    }
    (final_dir / "sft_loop_phase3_adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_dir


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if rank == 0:
        print(
            f"[startup] world_size={world_size} device={device} index={args.index} "
            f"split={args.split} focus_balance_count={args.focus_balance_count} "
            f"regular_focus_multiplier={args.regular_focus_multiplier} "
            f"invalid_focus_multiplier={args.invalid_focus_multiplier} "
            f"eval_steps={args.eval_steps} generation_eval_steps={args.generation_eval_steps}",
            flush=True,
        )
        print("[startup] reading train rows...", flush=True)
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    if rank == 0:
        print(f"[startup] train rows loaded: {len(rows)}", flush=True)
    raw_focus_counts = _raw_focus_bin_counts(rows)
    effective_focus_target_per_raw_rs_bin = (
        int(args.focus_balance_count) if int(args.focus_balance_count) > 0 else min(raw_focus_counts.values())
    )
    # 第一个 epoch 的全局 work；后续 epoch 会以不同 seed 重建，富余桶不会固定重复
    # 同一小批样本，稀缺桶则按需要循环重采样以保持八桶严格相等。
    if rank == 0:
        print(
            f"[startup] building train work target_per_raw_rs_bin={effective_focus_target_per_raw_rs_bin} "
            "(streaming candidate sampler)...",
            flush=True,
        )
    full_work = _balanced_work(
        rows,
        target_per_bin=int(args.focus_balance_count),
        seed=int(args.seed),
        regular_multiplier=float(args.regular_focus_multiplier),
        invalid_multiplier=float(args.invalid_focus_multiplier),
    )
    if not full_work:
        raise ValueError("balanced work list is empty")
    # 训练用同一个全局 work 序列按 global step 对齐取样；rank0 仍保存第一轮
    # rank shard 供审计，每个后续 epoch 重新采样富余桶。
    work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
    output_dir = pathlib.Path(args.output_dir)

    eval_rows: List[FrameRow] = []
    full_eval_work: List[Tuple[FrameRow, str]] = []
    eval_work: List[Tuple[FrameRow, str]] = []
    full_generation_eval_work: List[Tuple[FrameRow, str]] = []
    if int(args.eval_steps) > 0 and int(args.eval_balance_count) > 0:
        try:
            if rank == 0:
                print("[startup] reading validation rows...", flush=True)
            eval_rows = _read_rows(pathlib.Path(args.index), split=str(args.eval_split), max_frames=int(args.max_eval_frames))
            if rank == 0:
                print(
                    f"[startup] validation rows loaded: {len(eval_rows)}; building validation work...",
                    flush=True,
                )
            full_eval_work = _balanced_work(eval_rows, target_per_bin=int(args.eval_balance_count), seed=int(args.seed) + 1009)
            eval_work = _split_work_for_rank(full_eval_work, rank=rank, world_size=world_size)
            if int(args.generation_eval_steps) > 0 and int(args.generation_eval_balance_count) > 0:
                if rank == 0:
                    print("[startup] building generation validation work...", flush=True)
                full_generation_eval_work = _balanced_work(
                    eval_rows,
                    target_per_bin=int(args.generation_eval_balance_count),
                    seed=int(args.seed) + 2017,
                )
        except Exception as exc:
            raise RuntimeError(
                "periodic validation was requested but its split cannot satisfy the exact eight-bin balance. "
                "Rebuild/fix the dataset instead of silently training without validation."
            ) from exc
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        val_balance = {}
        if eval_work:
            val_balance = {
                "raw_available": _raw_focus_bin_counts(eval_rows),
                "global_sampled": dict(Counter(item.balance_key for item in full_eval_work)),
                "rank0_shard": dict(Counter(item.balance_key for item in eval_work)),
            }
        (output_dir / "train_balance.json").write_text(
            json.dumps(
                {
                    "world_size": int(world_size),
                    "history_rgb_mode": str(args.history_rgb_mode),
                    "history_rgb_count": len(history_rgb_indices(args.history_rgb_mode)),
                    "history_rgb_selected_indices": list(history_rgb_indices(args.history_rgb_mode)),
                    "train": {
                        "split": str(args.split),
                        "focus_balance_count": int(args.focus_balance_count),
                        "regular_focus_multiplier": float(args.regular_focus_multiplier),
                        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
                        "effective_focus_target_per_raw_rs_bin": int(effective_focus_target_per_raw_rs_bin),
                        "resample_each_epoch": True,
                        "epoch_seed_formula": "seed + epoch * 1000003",
                        "step_schedule": "global_step aligned across ranks; item index = step_in_epoch * world_size + rank, wrapping only padded tail steps",
                        "steps_per_epoch_global": int(math.ceil(len(full_work) / max(1, int(world_size)))),
                        "raw_available": raw_focus_counts,
                        "global_sampled": dict(Counter(item.balance_key for item in full_work)),
                        "rank0_shard": dict(Counter(item.balance_key for item in work)),
                        "variant_counts": dict(Counter(item.spec.variant for item in full_work)),
                        "variant_ratio_target": dict(VARIANT_WEIGHTS),
                    },
                    "eval": {
                        "split": str(args.eval_split),
                        "eval_steps": int(args.eval_steps),
                        "eval_balance_count": int(args.eval_balance_count),
                        **val_balance,
                    },
                    "generation_eval": {
                        "steps": int(args.generation_eval_steps),
                        "balance_count": int(args.generation_eval_balance_count),
                        "max_new_tokens": int(args.generation_eval_max_new_tokens),
                        "min_format_valid_rate": float(args.generation_eval_min_valid_rate),
                        "global_sampled": dict(Counter(item.balance_key for item in full_generation_eval_work)),
                        "rank0_only": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if world_size > 1:
        ddp_barrier(local_rank)

    if rank == 0:
        print("[startup] loading Qwen + LoRA...", flush=True)
    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_vision_scope=str(args.lora_vision_scope),
        strict_vision_scope=bool(args.strict_vision_scope),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    if world_size > 1:
        bundle.model = DDP(bundle.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if rank == 0:
        print("[startup] model loaded; starting optimizer setup...", flush=True)
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
    steps_per_epoch = max(1, math.ceil(len(full_work) / max(1, int(world_size))))
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else max(1, steps_per_epoch * max(1, int(args.num_epochs)))
    total_optimizer_steps = max(1, math.ceil(total_steps / max(1, int(args.grad_accum))))
    scheduler = make_scheduler(optimizer, total_steps=total_optimizer_steps, warmup_steps=int(args.warmup_steps))
    writer = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE and not bool(args.no_tb) else None
    if rank == 0:
        _write_run_metadata(
            writer,
            output_dir,
            args,
            world_size=world_size,
            train_rows=len(rows),
            train_work_global=len(full_work),
            train_work_rank=len(work),
            eval_work_rank=len(eval_work),
            generation_eval_global=len(full_generation_eval_work),
            total_steps=total_steps,
        )

    global_step = 0
    grad_accum = max(1, int(args.grad_accum))
    accum_steps = 0
    pending_checkpoint_step: Optional[int] = None
    skipped = 0
    best_val_loss = math.inf
    best_generation_score = (-math.inf, -math.inf)
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "train_eval_metrics.jsonl"
    window_loss_sum = 0.0
    window_samples = 0
    window_variants: Counter[str] = Counter()
    window_stats: Counter[str] = Counter()
    window_metric_ok: Counter[str] = Counter()
    window_metric_count: Counter[str] = Counter()
    t0 = time.time()
    bundle.model.train()
    if rank == 0:
        print(
            f"[data] train_rows={len(rows)} train_work_global={len(full_work)} train_work_rank={len(work)} "
            f"effective_focus_target_per_raw_rs_bin={effective_focus_target_per_raw_rs_bin} resample_each_epoch=True "
            f"regular_focus_multiplier={float(args.regular_focus_multiplier):.3f} "
            f"invalid_focus_multiplier={float(args.invalid_focus_multiplier):.3f} "
            f"steps_per_epoch_global={steps_per_epoch} num_epochs={int(args.num_epochs)} max_steps={int(args.max_steps)} "
            f"total_steps_global={total_steps} eval_work_rank={len(eval_work)} "
            f"generation_eval_global={len(full_generation_eval_work)} "
            f"history_rgb_mode={args.history_rgb_mode} history_rgb_count={len(history_rgb_indices(args.history_rgb_mode))}"
        )
    epoch = 0
    while global_step < total_steps:
        if epoch > 0:
            epoch_seed = int(args.seed) + epoch * 1_000_003
            full_work = _balanced_work(
                rows,
                target_per_bin=int(args.focus_balance_count),
                seed=epoch_seed,
                regular_multiplier=float(args.regular_focus_multiplier),
                invalid_multiplier=float(args.invalid_focus_multiplier),
            )
            work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
        order_rng = random.Random(int(args.seed) + epoch * 1_000_003)
        order_rng.shuffle(full_work)
        work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
        epoch_start_step = global_step
        for step_in_epoch in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            global_item_index = step_in_epoch * max(1, int(world_size)) + int(rank)
            padded_step = global_item_index >= len(full_work)
            item = full_work[global_item_index % len(full_work)]
            row = item.row
            spec = item.spec
            images = _load_images(select_history_rgb_paths(row.history_rgb_paths, args.history_rgb_mode))
            target = build_phase3_target(spec)
            packed = _build_inputs(
                bundle,
                images=images,
                history_rgb_mode=args.history_rgb_mode,
                target=target,
                spec=spec,
                max_length=int(args.max_length),
                format_loss_weight=float(args.format_loss_weight),
            )
            if packed is None:
                skipped += 1
                loss = _ddp_sync_zero_loss(bundle, images)
                stats = {
                    "denom": 0.0,
                    "token_acc": 0.0,
                    "value_token_acc": 0.0,
                    "format_token_acc": 0.0,
                }
            else:
                loss, stats = _loss_one(bundle, packed, spec)
            loss_value = float(loss.detach().item())
            if packed is not None:
                window_loss_sum += loss_value
                window_samples += 1
                window_variants[spec.variant] += 1
                for key, value in stats.items():
                    window_stats[key] += float(value)
                for output_key, metric_key, _ in spec_metric_items(spec):
                    stat_key = f"answer/{output_key.lower()}_ok"
                    if stat_key in stats:
                        window_metric_ok[metric_key] += float(stats[stat_key])
                        window_metric_count[metric_key] += 1
            (loss / grad_accum).backward()
            accum_steps += 1
            optimizer_stepped = False
            if accum_steps >= grad_accum:
                _optimizer_step(
                    params=params,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    max_grad_norm=float(args.max_grad_norm),
                )
                accum_steps = 0
                optimizer_stepped = True
            if writer:
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), global_step)
                writer.add_scalar("train/skipped_too_long", skipped, global_step)
                writer.add_scalar("train/padded_ddp_step", float(padded_step), global_step)
                writer.add_scalar(f"train/variant/{spec.variant}", 1, global_step)
                for key, value in stats.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
            if rank == 0 and global_step % int(args.log_steps) == 0:
                window_den = max(1, int(window_samples))
                window_payload: Dict[str, Any] = {
                    "step": int(global_step),
                    "epoch": int(epoch + 1),
                    "samples": int(window_samples),
                    "loss_mean": float(window_loss_sum / window_den),
                    "last_loss": loss_value,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "skipped_too_long_total": int(skipped),
                    "variant_counts": dict(window_variants),
                    "balance_key": item.balance_key,
                }
                for variant in VARIANT_ORDER:
                    window_payload[f"variant_rate/{variant}"] = float(window_variants[variant]) / float(window_den)
                for key, value in window_stats.items():
                    window_payload[f"mean/{key}"] = float(value) / float(window_den)
                for key in list(ANSWER_KEYS):
                    denom = max(1.0, float(window_metric_count[key]))
                    window_payload[f"answer_acc/{key}"] = float(window_metric_ok[key]) / denom
                    window_payload[f"answer_samples/{key}"] = float(window_metric_count[key])
                _append_jsonl(train_metrics_path, window_payload)
                _write_scalar_dict(writer, "train_window", window_payload, global_step)
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} loss={loss_value:.4f} "
                    f"win_loss={window_payload['loss_mean']:.4f} "
                    f"variant={spec.variant} bin={item.balance_key} skipped={skipped} world={world_size} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
                window_loss_sum = 0.0
                window_samples = 0
                window_variants.clear()
                window_stats.clear()
                window_metric_ok.clear()
                window_metric_count.clear()
            global_step += 1
            saved_deferred_checkpoint = False
            if pending_checkpoint_step is not None and optimizer_stepped:
                if rank == 0:
                    ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
                    print(
                        f"[save] deferred checkpoint requested_at={pending_checkpoint_step} "
                        f"flushed_at={global_step} adapter={ckpt_dir}",
                        flush=True,
                    )
                pending_checkpoint_step = None
                saved_deferred_checkpoint = True
            if eval_work and int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
                metrics = evaluate_loss(
                    bundle,
                    eval_work,
                    history_rgb_mode=args.history_rgb_mode,
                    max_length=int(args.max_length),
                    format_loss_weight=float(args.format_loss_weight),
                    device=device,
                    world_size=world_size,
                )
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(f"val/{key}", float(value), global_step)
                if rank == 0:
                    _append_jsonl(
                        eval_metrics_path,
                        {
                            "step": int(global_step),
                            "kind": "teacher_forced",
                            "split": str(args.eval_split),
                            **{key: float(value) for key, value in metrics.items()},
                        },
                    )
                    print(
                        f"[eval] step={global_step}/{total_steps} split={args.eval_split} "
                        f"loss={metrics['loss']:.4f} value_acc={metrics['value_token_acc']:.4f} "
                        f"format_acc={metrics['format_token_acc']:.4f} "
                        f"event_gate_exact={metrics.get('variant/all_random_order_exact', 0.0):.4f}"
                    )
                generation_metrics: Optional[Dict[str, float]] = None
                run_generation_eval = bool(full_generation_eval_work) and global_step % int(args.generation_eval_steps) == 0
                if run_generation_eval and rank == 0:
                    generation_metrics = evaluate_generation_probe(
                        bundle,
                        full_generation_eval_work,
                        history_rgb_mode=args.history_rgb_mode,
                        max_new_tokens=int(args.generation_eval_max_new_tokens),
                        record_path=output_dir / "generation_val_cases.jsonl",
                        step=global_step,
                    )
                    if writer:
                        for key, value in generation_metrics.items():
                            writer.add_scalar(f"val_generation/{key}", float(value), global_step)
                    _append_jsonl(
                        eval_metrics_path,
                        {
                            "step": int(global_step),
                            "kind": "free_generation",
                            "split": str(args.eval_split),
                            **{key: float(value) for key, value in generation_metrics.items()},
                        },
                    )
                    print(
                        f"[generation-val] step={global_step}/{total_steps} "
                        f"format_valid={generation_metrics['format_valid_rate']:.4f} "
                        f"exact={generation_metrics['exact_accuracy']:.4f} "
                        f"event_gate_valid={generation_metrics.get('variant/all_random_order_valid', 0.0):.4f}"
                    )
                if run_generation_eval and world_size > 1:
                    ddp_barrier(local_rank)
                if rank == 0:
                    eligible_for_best = not bool(full_generation_eval_work) or run_generation_eval
                    format_gate_ok = (
                        eligible_for_best
                        and (
                            generation_metrics is None
                            or float(generation_metrics["format_valid_rate"]) >= float(args.generation_eval_min_valid_rate)
                        )
                    )
                    if bool(args.save_best_val) and float(metrics["loss"]) < best_val_loss and format_gate_ok:
                        best_val_loss = float(metrics["loss"])
                        best_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_val")
                        (output_dir / "best_val.json").write_text(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "val_split": str(args.eval_split),
                                    "val_loss": best_val_loss,
                                    "token_acc": float(metrics["token_acc"]),
                                    "value_token_acc": float(metrics["value_token_acc"]),
                                    "format_token_acc": float(metrics["format_token_acc"]),
                                    "generation": generation_metrics,
                                    "history_rgb_mode": str(args.history_rgb_mode),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(f"[best-val] step={global_step} loss={best_val_loss:.4f} adapter={best_dir}")
                    elif bool(args.save_best_val) and float(metrics["loss"]) < best_val_loss and run_generation_eval and not format_gate_ok:
                        print(
                            f"[best-val-skip] step={global_step} loss={metrics['loss']:.4f} "
                            f"generation format_valid={generation_metrics['format_valid_rate']:.4f} "
                            f"< required {args.generation_eval_min_valid_rate:.4f}"
                        )
                    if bool(args.save_best_generation) and run_generation_eval and format_gate_ok and generation_metrics is not None:
                        gen_score = (
                            float(generation_metrics.get("exact_accuracy", 0.0)),
                            float(generation_metrics.get("pattern/all_random_order_pattern_exact", 0.0)),
                        )
                        if gen_score > best_generation_score:
                            best_generation_score = gen_score
                            best_gen_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_generation")
                            (output_dir / "best_generation.json").write_text(
                                json.dumps(
                                    {
                                        "step": global_step,
                                        "val_split": str(args.eval_split),
                                        "selection": "max_free_generation_exact_then_all_random_order",
                                        "generation_exact_accuracy": gen_score[0],
                                        "generation_all_random_order_exact": gen_score[1],
                                        "teacher_forced_loss": float(metrics["loss"]),
                                        "teacher_forced_value_token_acc": float(metrics["value_token_acc"]),
                                        "teacher_forced_format_token_acc": float(metrics["format_token_acc"]),
                                        "generation": generation_metrics,
                                        "history_rgb_mode": str(args.history_rgb_mode),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            print(
                                f"[best-generation] step={global_step} exact={gen_score[0]:.4f} "
                                f"all_random={gen_score[1]:.4f} adapter={best_gen_dir}"
                            )
            checkpoint_due = int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0
            if checkpoint_due and not saved_deferred_checkpoint:
                if accum_steps == 0:
                    if rank == 0:
                        ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
                        print(f"[save] step={global_step} adapter={ckpt_dir}")
                else:
                    pending_checkpoint_step = int(global_step)
                    if writer:
                        writer.add_scalar("train/checkpoint_deferred_for_grad_accum", 1.0, global_step)
                    if rank == 0:
                        print(
                            f"[save-defer] step={global_step} has {accum_steps}/{grad_accum} "
                            "pending grad_accum samples; saving after next optimizer step",
                            flush=True,
                        )
            if global_step >= total_steps:
                break
        if global_step == epoch_start_step:
            raise RuntimeError("no train steps were completed in an epoch; check max_length and input data")
        epoch += 1

    if accum_steps > 0:
        _optimizer_step(
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=float(args.max_grad_norm),
        )
        if writer:
            writer.add_scalar("train/flushed_partial_grad_accum", 1.0, global_step)
        if rank == 0:
            print(f"[flush] applied partial grad_accum window at step={global_step}", flush=True)
        accum_steps = 0
        if pending_checkpoint_step is not None:
            if rank == 0:
                ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
                print(
                    f"[save] deferred checkpoint requested_at={pending_checkpoint_step} "
                    f"flushed_at=final:{global_step} adapter={ckpt_dir}",
                    flush=True,
                )
            pending_checkpoint_step = None

    if world_size > 1:
        ddp_barrier(local_rank)
    final_dir = _save_adapter(bundle, output_dir, args, step=global_step) if rank == 0 else None
    if writer:
        writer.close()
    if rank == 0:
        print(f"[done] saved adapter to {final_dir}")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Train sft_loop_phase3 random-question LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase3_data/frame_index.jsonl"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument(
        "--output-dir",
        default="",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--history-rgb-mode", choices=HISTORY_RGB_MODES, default=DEFAULT_HISTORY_RGB_MODE)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--focus-balance-count",
        type=int,
        default=0,
        help="per augment balance bin; 0 uses the smallest raw bin in full each epoch, positive values set a fixed target and repeat scarce bins when needed",
    )
    p.add_argument(
        "--regular-focus-multiplier",
        type=float,
        default=2.0,
        help="training-only multiplier for the RE/all-NO balance bin; UE positive bins remain 1:1:1:1",
    )
    p.add_argument(
        "--invalid-focus-multiplier",
        type=float,
        default=1.0,
        help="training-only multiplier for wrong-RS invalid samples",
    )
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="0 means train num_epochs over the balanced work list")
    p.add_argument("--eval-split", default="val")
    p.add_argument(
        "--eval-steps",
        type=int,
        default=2_000,
        help="validate every N optimizer steps; 0 disables periodic validation",
    )
    p.add_argument("--eval-balance-count", type=int, default=16)
    p.add_argument("--max-eval-frames", type=int, default=0)
    p.add_argument(
        "--format-loss-weight",
        type=float,
        default=0.25,
        help="low loss weight for answer field names, separators, newlines, and assistant end token; YES/NO values always use 1.0",
    )
    p.add_argument(
        "--generation-eval-steps",
        type=int,
        default=2_000,
        help="run rank0 free-generation validation every N teacher-forced validation steps; 0 disables it",
    )
    p.add_argument("--generation-eval-balance-count", type=int, default=2)
    p.add_argument("--generation-eval-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-eval-min-valid-rate", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=20_000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-vision-scope", choices=["off", "merger", "last4", "all"], default="off")
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument(
        "--save-best-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save output_dir/best_val whenever periodic validation loss improves",
    )
    p.add_argument(
        "--save-best-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save output_dir/best_generation by free-generation exact, with all_random_order exact as tie-breaker",
    )
    p.add_argument("--no-tb", action="store_true")
    args = p.parse_args()
    args.history_rgb_mode = validate_history_rgb_mode(args.history_rgb_mode)
    if not 0.0 < float(args.format_loss_weight) <= 1.0:
        raise ValueError("--format-loss-weight must be in (0, 1]")
    if int(args.generation_eval_steps) > 0:
        if int(args.eval_steps) <= 0 or int(args.eval_balance_count) <= 0:
            raise ValueError("free-generation validation requires --eval-steps and --eval-balance-count to be positive")
        if int(args.generation_eval_steps) % int(args.eval_steps) != 0:
            raise ValueError("--generation-eval-steps must be a multiple of --eval-steps")
        if int(args.generation_eval_balance_count) <= 0:
            raise ValueError("--generation-eval-balance-count must be positive when generation validation is enabled")
    if not 0.0 <= float(args.generation_eval_min_valid_rate) <= 1.0:
        raise ValueError("--generation-eval-min-valid-rate must be in [0, 1]")
    if float(args.regular_focus_multiplier) < 0.0:
        raise ValueError("--regular-focus-multiplier must be non-negative")
    if float(args.invalid_focus_multiplier) < 0.0:
        raise ValueError("--invalid-focus-multiplier must be non-negative")
    if not args.output_dir:
        args.output_dir = str(
            _AUTOMOT_ROOT
            / "checkpoints/sft_loop_phase3_runs"
            / f"run_event_gate_format_supervised_{history_rgb_mode_tag(args.history_rgb_mode)}"
        )
    return args


def main() -> None:
    """CLI 入口。"""

    train(parse_args())


if __name__ == "__main__":
    main()
