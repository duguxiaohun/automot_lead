#!/usr/bin/env python3
"""训练 sft_loop_phase1 四问 LoRA adapter。

训练目标是四行 YES/NO 的值 token。采样时每帧展开成四个不可见 focus 视图，并按
`问题 x YES/NO` 八桶 exact balance；prompt 中不出现 focus，模型每次仍回答四项。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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

from qwen3vl_local.sft_loop_phase1 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase1.prompts import (  # noqa: E402
    ANSWER_KEYS,
    PROMPT_NAME,
    SYSTEM_PROMPT,
    build_phase1_prompt,
    build_phase1_target,
)
from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _assert_inside_assistant_turn,
    _find_subsequence,
    load_model_with_lora,
    make_scheduler,
)


def setup_distributed() -> Tuple[int, int, int]:
    """初始化可选 torchrun DDP。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_loop_phase1 DDP requires CUDA.")
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


def _balanced_work(rows: Sequence[FrameRow], *, target_per_bin: int, seed: int) -> List[Tuple[FrameRow, str]]:
    """构建八桶 exact-balance work list。"""

    groups: Dict[str, List[Tuple[FrameRow, str]]] = {f"{key}:{value}": [] for key in ANSWER_KEYS for value in ("YES", "NO")}
    for row in rows:
        for focus in ANSWER_KEYS:
            groups[_focus_key(row, focus)].append((row, focus))
    nonempty = [len(v) for v in groups.values() if v]
    if not nonempty:
        return []
    target = int(target_per_bin) if int(target_per_bin) > 0 else min(nonempty)
    target = max(1, target)
    rng = random.Random(f"{seed}:phase1_balance:{len(rows)}:{target}")
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
    return work


def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    """读取 4 帧 RGB history。"""

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


def _target_value_weights(bundle: Any, target: str) -> Tuple[List[int], List[float], List[int]]:
    """把四个 YES/NO 字符 span 映射到 token loss 权重。"""

    enc = bundle.tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    weights = [0.0 for _ in token_ids]
    component_ids = [0 for _ in token_ids]
    for component_id, key in enumerate(ANSWER_KEYS, start=1):
        lo, hi = _line_value_span(target, key)
        for i, (a, b) in enumerate(offsets):
            if a < hi and b > lo:
                weights[i] = 1.0
                component_ids[i] = component_id
    return token_ids, weights, component_ids


def _build_inputs(bundle: Any, *, images: List[Image.Image], prompt: str, target: str, max_length: int) -> Optional[Dict[str, Any]]:
    """构造模型输入，只监督四个答案值 token。"""

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
    target_ids, value_weights, value_components = _target_value_weights(bundle, target)
    pos = _find_subsequence(expanded, target_ids, 0)
    asst_header_ids = list(bundle.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])
    _assert_inside_assistant_turn(expanded, pos, asst_header_ids, 0)
    for j, weight in enumerate(value_weights):
        if weight > 0:
            weights[pos + j] = float(weight)
            component_ids[pos + j] = int(value_components[j])
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
    stats: Dict[str, float] = {"denom": float(denom.detach().item()), "token_acc": float(torch.equal(pred[active], shift_labels[active]))}
    for component_id, key in enumerate(ANSWER_KEYS, start=1):
        mask = active & shift_comp.eq(component_id)
        stats[f"{key.lower()}_ok"] = float(bool(mask.any() and torch.equal(pred[mask], shift_labels[mask])))
    return loss, stats


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace, *, step: int) -> pathlib.Path:
    """保存 LoRA adapter 和 Phase1 自描述配置。"""

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(final_dir))
    cfg = {
        "schema": "sft_loop_phase1_adapter_config",
        "route": "sft_loop_phase1_four_visible_facts",
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "base_model_dir": str(args.model_dir),
        "lora_vision_scope": str(args.lora_vision_scope),
        "lora_target_modules": list(bundle.lora_target_modules),
        "answer_order": list(ANSWER_KEYS),
        "global_step": int(step),
    }
    (final_dir / "sft_loop_phase1_adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_dir


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    work = _balanced_work(rows, target_per_bin=int(args.focus_balance_count), seed=int(args.seed))
    if not work:
        raise ValueError("balanced work list is empty")
    if world_size > 1:
        # 每个 rank 使用同一个八桶均衡全集的 rank::world_size 分片。所有 rank 的
        # max_steps 相同，DDP backward 次数一致；rank0 保存全集 balance 供审计。
        work = work[rank::world_size]
        if not work:
            raise ValueError(f"rank {rank} got empty work shard; reduce WORLD_SIZE or increase focus_balance_count")
    output_dir = pathlib.Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_work = _balanced_work(rows, target_per_bin=int(args.focus_balance_count), seed=int(args.seed))
        (output_dir / "train_balance.json").write_text(
            json.dumps(
                {
                    "world_size": int(world_size),
                    "global": dict(Counter(_focus_key(row, focus) for row, focus in full_work)),
                    "rank0_shard": dict(Counter(_focus_key(row, focus) for row, focus in work)),
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
    total_steps = int(args.max_steps)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(args.warmup_steps))
    writer = SummaryWriter(str(output_dir / "tb")) if rank == 0 and _TB_AVAILABLE and not bool(args.no_tb) else None
    prompt = build_phase1_prompt(audit=False)

    rng = random.Random(int(args.seed))
    global_step = 0
    skipped = 0
    t0 = time.time()
    bundle.model.train()
    while global_step < total_steps:
        rng.shuffle(work)
        for row, focus in work:
            images = _load_images(row.history_rgb_paths)
            target = build_phase1_target(row.answers)
            packed = _build_inputs(bundle, images=images, prompt=prompt, target=target, max_length=int(args.max_length))
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
                    f"step={global_step}/{total_steps} loss={float(loss.detach().item()):.4f} "
                    f"focus={focus}:{'YES' if row.answers[focus] else 'NO'} skipped={skipped} world={world_size} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            global_step += 1
            if global_step >= total_steps:
                break

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

    p = argparse.ArgumentParser(description="Train sft_loop_phase1 four-question LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase1_data/frame_index.jsonl"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase1"))
    p.add_argument("--split", default="train")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--focus-balance-count", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=50)
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
    p.add_argument("--no-tb", action="store_true")
    return p.parse_args()


def main() -> None:
    """CLI 入口。"""

    train(parse_args())


if __name__ == "__main__":
    main()
