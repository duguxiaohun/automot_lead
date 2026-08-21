#!/usr/bin/env python3
"""训练 sft_loop_phase2 的四个 ROAD_STRUCTURE 二元问题 LoRA adapter。

训练目标是四行 YES/NO 的语义 token，并以低权重监督字段格式和 assistant 结束符。
采样时每帧展开成四个不可见 focus 视图，并按 `问题 x YES/NO` 八桶 exact balance；
prompt 中不出现 focus，模型每次仍回答四项。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

from qwen3vl_local.sft_loop_phase2 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase2.history_rgb import (  # noqa: E402
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODES,
    history_rgb_indices,
    history_rgb_mode_tag,
    select_history_rgb_paths,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_loop_phase2.prompts import (  # noqa: E402
    ANSWER_KEYS,
    PROMPT_NAME,
    SYSTEM_PROMPT,
    build_phase2_prompt,
    build_phase2_target,
    parse_phase2_output,
    phase2_prompt_sha256,
)
from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _assert_inside_assistant_turn,
    _find_subsequence,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


FORMAT_COMPONENT_ID = -1


def _git_metadata() -> Dict[str, Any]:
    """记录训练时的代码版本；失败时保留空字段，不影响训练。"""

    def run(args: Sequence[str]) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=_PROJECT_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except Exception:
            return None

    status = run(["status", "--short"]) or ""
    return {
        "root": str(_PROJECT_ROOT),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines()[:300],
    }


def setup_distributed() -> Tuple[int, int, int]:
    """初始化可选 torchrun DDP。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_loop_phase2 DDP requires CUDA.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """清理 torch.distributed。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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
                    answers={key: bool((obj.get("answers") or {}).get(key, False)) for key in ANSWER_KEYS},
                )
            )
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {path}")
    return rows


def _focus_key(row: FrameRow, focus: str) -> str:
    """返回八桶采样键。"""

    return f"{focus}:{'YES' if row.answers[focus] else 'NO'}"


def _expected_focus_bins() -> List[str]:
    """返回四个主任务的固定 YES/NO 八桶顺序。"""

    return [f"{key}:{value}" for key in ANSWER_KEYS for value in ("YES", "NO")]


def _assert_exact_focus_balance(work: Sequence[Tuple[FrameRow, str]], *, target: int, context: str) -> None:
    """确保完整 work list 的八桶全部存在且每桶严格等量。"""

    counts = Counter(_focus_key(row, focus) for row, focus in work)
    expected = int(target)
    invalid = {key: int(counts.get(key, 0)) for key in _expected_focus_bins() if int(counts.get(key, 0)) != expected}
    if invalid:
        raise RuntimeError(
            f"{context} violates exact 1:1:1:1 and YES/NO 1:1 balance; "
            f"expected every bin={expected}, got={dict(counts)}, invalid={invalid}"
        )


def _raw_focus_bin_counts(rows: Sequence[FrameRow]) -> Dict[str, int]:
    """统计 split 中可用的原始八桶计数，供 balance artifact 审计。"""

    counts: Counter[str] = Counter()
    for row in rows:
        for focus in ANSWER_KEYS:
            counts[_focus_key(row, focus)] += 1
    return {key: int(counts.get(key, 0)) for key in _expected_focus_bins()}


def _balanced_work(rows: Sequence[FrameRow], *, target_per_bin: int, seed: int) -> List[Tuple[FrameRow, str]]:
    """构建八桶 exact-balance work list。

    ``target_per_bin=0`` 使用最少原始桶的全量样本。目标高于某桶原始量时，
    该桶会确定性循环重采样；这让稀缺正类可以参与大规模严格均衡训练。
    """

    groups: Dict[str, List[Tuple[FrameRow, str]]] = {f"{key}:{value}": [] for key in ANSWER_KEYS for value in ("YES", "NO")}
    for row in rows:
        for focus in ANSWER_KEYS:
            groups[_focus_key(row, focus)].append((row, focus))
    raw_counts = {key: len(items) for key, items in groups.items()}
    missing = [key for key in _expected_focus_bins() if raw_counts[key] == 0]
    if missing:
        raise ValueError(
            "cannot build exact 1:1 Phase2 training work: required focus bins are empty; "
            f"missing={missing} raw_counts={raw_counts}. Rebuild/check the dataset split or reduce filtering."
        )
    nonempty = list(raw_counts.values())
    target = int(target_per_bin) if int(target_per_bin) > 0 else min(nonempty)
    target = max(1, target)
    rng = random.Random(f"{seed}:phase2_balance:{len(rows)}:{target}")
    work: List[Tuple[FrameRow, str]] = []
    for key in sorted(groups):
        items = list(groups[key])
        if not items:
            continue
        rng.shuffle(items)
        if len(items) >= target:
            work.extend(items[:target])
        else:
            repeated = [items[i % len(items)] for i in range(target)]
            rng.shuffle(repeated)
            work.extend(repeated)
    rng.shuffle(work)
    _assert_exact_focus_balance(work, target=target, context="training work")
    return work


def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    """读取当前 RGB-history 合同选择出的图片。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _messages(images: List[Image.Image], user_prompt: str, target: str) -> List[Dict[str, Any]]:
    """构造 teacher-forced chat。"""

    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "assistant", "content": target},
    ]


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
    format_loss_weight: float,
) -> Tuple[List[int], List[float], List[int]]:
    """映射四行输出的语义与格式 token 权重。

    YES/NO 是四题的主监督，权重固定为 1。字段名、冒号和换行只承担自由生成
    的语法锚定，使用较低权重；它们绝不编码四题之间的任何逻辑关系。
    """

    enc = bundle.tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    weights = [float(format_loss_weight) for _ in token_ids]
    component_ids = [FORMAT_COMPONENT_ID for _ in token_ids]
    for component_id, key in enumerate(ANSWER_KEYS, start=1):
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
    prompt: str,
    target: str,
    max_length: int,
    format_loss_weight: float,
) -> Optional[Dict[str, Any]]:
    """构造模型输入，主监督答案值并低权重监督四行格式与结束符。"""

    messages = _messages(images, prompt, target)
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


def _loss_one(bundle: Any, packed: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    for component_id, key in enumerate(ANSWER_KEYS, start=1):
        mask = active & shift_comp.eq(component_id)
        stats[f"{key.lower()}_ok"] = float(bool(mask.any() and torch.equal(pred[mask], shift_labels[mask])))
    return loss, stats


def _split_work_for_rank(work: Sequence[Tuple[FrameRow, str]], *, rank: int, world_size: int) -> List[Tuple[FrameRow, str]]:
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
    work: Sequence[Tuple[FrameRow, str]],
    *,
    prompt: str,
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
    component_ok = {key: 0.0 for key in ANSWER_KEYS}
    focus_ok = {key: 0.0 for key in ANSWER_KEYS}
    focus_count = {key: 0.0 for key in ANSWER_KEYS}
    for row, focus in work:
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        target = build_phase2_target(row.answers)
        packed = _build_inputs(
            bundle,
            images=images,
            prompt=prompt,
            target=target,
            max_length=int(max_length),
            format_loss_weight=float(format_loss_weight),
        )
        if packed is None:
            skipped += 1
            continue
        loss, stats = _loss_one(bundle, packed)
        loss_value = float(loss.detach().item())
        loss_sum += loss_value
        samples += 1
        token_acc_sum += float(stats.get("token_acc", 0.0))
        value_token_acc_sum += float(stats.get("value_token_acc", 0.0))
        format_token_acc_sum += float(stats.get("format_token_acc", 0.0))
        for key in ANSWER_KEYS:
            ok = float(stats.get(f"{key.lower()}_ok", 0.0))
            component_ok[key] += ok
            if key == focus:
                focus_ok[key] += ok
                focus_count[key] += 1.0
    values = [loss_sum, float(samples), float(skipped), token_acc_sum, value_token_acc_sum, format_token_acc_sum]
    values.extend(component_ok[key] for key in ANSWER_KEYS)
    values.extend(focus_ok[key] for key in ANSWER_KEYS)
    values.extend(focus_count[key] for key in ANSWER_KEYS)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if int(world_size) > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    vals = [float(x) for x in tensor.detach().cpu().tolist()]
    total_samples = max(1.0, vals[1])
    offset = 6
    metrics: Dict[str, float] = {
        "loss": vals[0] / total_samples,
        "samples": vals[1],
        "skipped": vals[2],
        "token_acc": vals[3] / total_samples,
        "value_token_acc": vals[4] / total_samples,
        "format_token_acc": vals[5] / total_samples,
    }
    for idx, key in enumerate(ANSWER_KEYS):
        metrics[f"{key.lower()}_acc"] = vals[offset + idx] / total_samples
    offset += len(ANSWER_KEYS)
    for idx, key in enumerate(ANSWER_KEYS):
        denom = max(1.0, vals[offset + len(ANSWER_KEYS) + idx])
        metrics[f"focus_{key.lower()}_acc"] = vals[offset + idx] / denom
        metrics[f"focus_{key.lower()}_samples"] = vals[offset + len(ANSWER_KEYS) + idx]
    if was_training:
        bundle.model.train()
    return metrics


def _generation_messages(images: List[Image.Image], user_prompt: str) -> List[Dict[str, Any]]:
    """构造不含 target 的生产式 chat，用于检查实际自由生成格式。"""

    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


@torch.no_grad()
def evaluate_generation_probe(
    bundle: Any,
    work: Sequence[Tuple[FrameRow, str]],
    *,
    prompt: str,
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
    focus_ok = {key: 0.0 for key in ANSWER_KEYS}
    focus_count = {key: 0.0 for key in ANSWER_KEYS}
    records: List[Dict[str, Any]] = []
    for row, focus in work:
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        state = _kv_start_state(runtime, _generation_messages(images, prompt))
        raw, _, _ = _student_generate_kv(runtime, state, int(max_new_tokens))
        parsed = parse_phase2_output(raw)
        is_valid = all(parsed[key] is not None for key in ANSWER_KEYS)
        samples += 1.0
        valid += float(is_valid)
        if is_valid:
            exact += float(all(bool(parsed[key]) == bool(row.answers[key]) for key in ANSWER_KEYS))
        if parsed[focus] is not None:
            focus_ok[focus] += float(bool(parsed[focus]) == bool(row.answers[focus]))
        focus_count[focus] += 1.0
        records.append(
            {
                "step": int(step),
                "scenario": row.scenario,
                "route_id": row.route_id,
                "town": row.town,
                "frame_id": row.frame_id,
                "focus": focus,
                "answers": row.answers,
                "parsed": parsed,
                "format_valid": is_valid,
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
    for key in ANSWER_KEYS:
        metrics[f"focus_{key.lower()}_acc"] = focus_ok[key] / max(1.0, focus_count[key])
        metrics[f"focus_{key.lower()}_samples"] = focus_count[key]
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace, *, step: int, name: str = "final") -> pathlib.Path:
    """保存 LoRA adapter 和 Phase2 自描述配置。"""

    final_dir = output_dir / str(name)
    final_dir.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(final_dir))
    cfg = {
        "schema": "sft_loop_phase2_adapter_config",
        "route": "sft_loop_phase2_road_structure_four_binary",
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "production_prompt_sha256": phase2_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_script": str(_THIS),
        "git": _git_metadata(),
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
        "eval_split": str(args.eval_split),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "format_loss_weight": float(args.format_loss_weight),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_max_new_tokens": int(args.generation_eval_max_new_tokens),
        "generation_eval_min_valid_rate": float(args.generation_eval_min_valid_rate),
        "save_best_val": bool(args.save_best_val),
    }
    (final_dir / "sft_loop_phase2_adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_dir


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    # 第一个 epoch 的全局 work；后续 epoch 会以不同 seed 重建，富余桶不会固定重复
    # 同一小批样本，稀缺桶则按需要循环重采样以保持八桶严格相等。
    full_work = _balanced_work(rows, target_per_bin=int(args.focus_balance_count), seed=int(args.seed))
    if not full_work:
        raise ValueError("balanced work list is empty")
    # 每个 rank 使用同一个八桶均衡全集的 rank::world_size 分片。rank0 保存第一个
    # epoch 的 balance 供审计；每个后续 epoch 重新采样富余桶。
    work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
    output_dir = pathlib.Path(args.output_dir)

    eval_rows: List[FrameRow] = []
    full_eval_work: List[Tuple[FrameRow, str]] = []
    eval_work: List[Tuple[FrameRow, str]] = []
    full_generation_eval_work: List[Tuple[FrameRow, str]] = []
    if int(args.eval_steps) > 0 and int(args.eval_balance_count) > 0:
        try:
            eval_rows = _read_rows(pathlib.Path(args.index), split=str(args.eval_split), max_frames=int(args.max_eval_frames))
            full_eval_work = _balanced_work(eval_rows, target_per_bin=int(args.eval_balance_count), seed=int(args.seed) + 1009)
            eval_work = _split_work_for_rank(full_eval_work, rank=rank, world_size=world_size)
            if int(args.generation_eval_steps) > 0 and int(args.generation_eval_balance_count) > 0:
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
                "global_sampled": dict(Counter(_focus_key(row, focus) for row, focus in full_eval_work)),
                "rank0_shard": dict(Counter(_focus_key(row, focus) for row, focus in eval_work)),
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
                        "effective_target_per_bin": len(full_work) // len(_expected_focus_bins()),
                        "resample_each_epoch": True,
                        "epoch_seed_formula": "seed + epoch * 1000003",
                        "raw_available": _raw_focus_bin_counts(rows),
                        "global_sampled": dict(Counter(_focus_key(row, focus) for row, focus in full_work)),
                        "rank0_shard": dict(Counter(_focus_key(row, focus) for row, focus in work)),
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
                        "global_sampled": dict(Counter(_focus_key(row, focus) for row, focus in full_generation_eval_work)),
                        "rank0_only": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()

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
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
    steps_per_epoch = len(work)
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else max(1, steps_per_epoch * max(1, int(args.num_epochs)))
    total_optimizer_steps = max(1, math.ceil(total_steps / max(1, int(args.grad_accum))))
    scheduler = make_scheduler(optimizer, total_steps=total_optimizer_steps, warmup_steps=int(args.warmup_steps))
    writer = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE and not bool(args.no_tb) else None
    prompt = build_phase2_prompt(audit=False, history_rgb_mode=args.history_rgb_mode)

    global_step = 0
    skipped = 0
    best_val_loss = math.inf
    t0 = time.time()
    bundle.model.train()
    if rank == 0:
        print(
            f"[data] train_rows={len(rows)} train_work_global={len(full_work)} train_work_rank={len(work)} "
            f"target_per_bin={len(full_work) // len(_expected_focus_bins())} resample_each_epoch=True "
            f"steps_per_epoch_rank={steps_per_epoch} num_epochs={int(args.num_epochs)} max_steps={int(args.max_steps)} "
            f"total_steps_rank={total_steps} eval_work_rank={len(eval_work)} "
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
            )
            work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
        rng = random.Random(int(args.seed) + epoch * 1_000_003 + rank)
        rng.shuffle(work)
        epoch_start_step = global_step
        for row, focus in work:
            images = _load_images(select_history_rgb_paths(row.history_rgb_paths, args.history_rgb_mode))
            target = build_phase2_target(row.answers)
            packed = _build_inputs(
                bundle,
                images=images,
                prompt=prompt,
                target=target,
                max_length=int(args.max_length),
                format_loss_weight=float(args.format_loss_weight),
            )
            if packed is None:
                skipped += 1
                continue
            loss, stats = _loss_one(bundle, packed)
            (loss / max(1, int(args.grad_accum))).backward()
            if (global_step + 1) % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if writer:
                writer.add_scalar("train/loss", float(loss.detach().item()), global_step)
                writer.add_scalar("train/skipped_too_long", skipped, global_step)
                writer.add_scalar(f"train/focus/{focus.lower()}", 1, global_step)
                for key, value in stats.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
            if rank == 0 and global_step % int(args.log_steps) == 0:
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} loss={float(loss.detach().item()):.4f} "
                    f"focus={focus}:{'YES' if row.answers[focus] else 'NO'} skipped={skipped} world={world_size} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            global_step += 1
            if eval_work and int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
                metrics = evaluate_loss(
                    bundle,
                    eval_work,
                    prompt=prompt,
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
                    print(
                        f"[eval] step={global_step}/{total_steps} split={args.eval_split} "
                        f"loss={metrics['loss']:.4f} value_acc={metrics['value_token_acc']:.4f} "
                        f"format_acc={metrics['format_token_acc']:.4f} "
                        f"focus_rs1={metrics.get('focus_rs1_acc', 0.0):.4f} "
                        f"focus_rs2={metrics.get('focus_rs2_acc', 0.0):.4f} "
                        f"focus_rs4={metrics.get('focus_rs4_acc', 0.0):.4f} "
                        f"focus_rs5={metrics.get('focus_rs5_acc', 0.0):.4f}"
                    )
                generation_metrics: Optional[Dict[str, float]] = None
                run_generation_eval = bool(full_generation_eval_work) and global_step % int(args.generation_eval_steps) == 0
                if run_generation_eval and rank == 0:
                    generation_metrics = evaluate_generation_probe(
                        bundle,
                        full_generation_eval_work,
                        prompt=prompt,
                        history_rgb_mode=args.history_rgb_mode,
                        max_new_tokens=int(args.generation_eval_max_new_tokens),
                        record_path=output_dir / "generation_val_cases.jsonl",
                        step=global_step,
                    )
                    if writer:
                        for key, value in generation_metrics.items():
                            writer.add_scalar(f"val_generation/{key}", float(value), global_step)
                    print(
                        f"[generation-val] step={global_step}/{total_steps} "
                        f"format_valid={generation_metrics['format_valid_rate']:.4f} "
                        f"exact={generation_metrics['exact_accuracy']:.4f} "
                        f"focus_rs1={generation_metrics.get('focus_rs1_acc', 0.0):.4f} "
                        f"focus_rs2={generation_metrics.get('focus_rs2_acc', 0.0):.4f} "
                        f"focus_rs4={generation_metrics.get('focus_rs4_acc', 0.0):.4f} "
                        f"focus_rs5={generation_metrics.get('focus_rs5_acc', 0.0):.4f}"
                    )
                if run_generation_eval and world_size > 1:
                    dist.barrier()
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
            if rank == 0 and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
                print(f"[save] step={global_step} adapter={ckpt_dir}")
            if global_step >= total_steps:
                break
        if global_step == epoch_start_step:
            raise RuntimeError("no train steps were completed in an epoch; check max_length and input data")
        epoch += 1

    if world_size > 1:
        dist.barrier()
    final_dir = _save_adapter(bundle, output_dir, args, step=global_step) if rank == 0 else None
    if writer:
        writer.close()
    if rank == 0:
        print(f"[done] saved adapter to {final_dir}")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Train sft_loop_phase2 four-question LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase2_data/frame_index.jsonl"))
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
        help="per focus YES/NO bin; 0 uses the smallest raw bin in full each epoch, positive values set a fixed target and repeat scarce bins when needed",
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
    if not args.output_dir:
        args.output_dir = str(
            _AUTOMOT_ROOT
            / "checkpoints/sft_loop_phase2_runs"
            / f"run_rs_four_binary_format_supervised_{history_rgb_mode_tag(args.history_rgb_mode)}"
        )
    return args


def main() -> None:
    """CLI 入口。"""

    train(parse_args())


if __name__ == "__main__":
    main()
