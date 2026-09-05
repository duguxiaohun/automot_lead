#!/usr/bin/env python3
"""训练 sft_new_loop_phase3 的单轮 high-level 动作 LoRA adapter。

训练目标是 ``DECELERATE / STOP / RESUME / LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT``
以及 ``INVALID_ACTION_CONTEXT`` 的 YES/NO 语义 token，并以低权重监督字段格式和
assistant 结束符。数据构建阶段保证九个动作上下文 1:1、每个上下文内部按动作签名
尽量均分，并额外注入约 20% 的上下文错配 invalid 样本。

输入是单轮 image+text：RGB history + Phase1/Phase2 已确定的道路结构与情境前提 +
route 目标点的 ego 相对坐标。没有任何 synthetic assistant 前缀，也不喂 RS/EVENT
code 文本。
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

from qwen3vl_local.sft_new_loop_phase3 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (  # noqa: E402
    ACTION_KEYS,
    CONTEXT_BY_ID,
    CONTEXT_IDS,
)
from qwen3vl_local.sft_new_loop_phase3.history_rgb import (  # noqa: E402
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODES,
    history_rgb_indices,
    select_history_rgb_paths,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import (  # noqa: E402
    balanced_invalid_items,
    invalid_subgroup_keys,
    invalid_subgroup_report,
)
from qwen3vl_local.sft_new_loop_phase3.prompts import (  # noqa: E402
    ANSWER_KEYS,
    INVALID_KEY,
    PROMPT_NAME,
    VARIANT_ORDER,
    VARIANT_WEIGHTS,
    PromptSpec,
    action_prompt_sha256,
    build_action_messages,
    build_action_target,
    make_prompt_spec,
    parse_action_output,
    prompt_spec_to_json,
    spec_metric_items,
)
from qwen3vl_local.sft_new_loop_phase3.sampling import (  # noqa: E402
    even_quota_with_capacity,
    route_diverse_sample,
    route_diversity_report,
)
from qwen3vl_local.sft_v2.train import (  # noqa: E402
    _assert_inside_assistant_turn,
    _find_subsequence,
    load_model_with_lora,
    make_scheduler,
)
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


FORMAT_COMPONENT_ID = -1
BALANCE_CLASSES: Tuple[str, ...] = (*CONTEXT_IDS, "INVALID")


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
            raise RuntimeError("sft_new_loop_phase3 DDP requires CUDA.")
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def ddp_barrier(local_rank: int) -> None:
    """在当前 rank 绑定的 GPU 上执行 barrier。"""

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
    """向运行目录追加一行 JSON 指标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _tb_tag(key: str) -> str:
    """把内部 metric key 转成稳定 TensorBoard tag。"""

    return str(key).replace(":", "/").replace(" ", "_")


def _write_scalar_dict(writer: Any, prefix: str, metrics: Mapping[str, Any], step: int) -> None:
    """批量写 TensorBoard scalar。"""

    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{_tb_tag(key)}", float(value), int(step))


@dataclass
class FrameRow:
    """一帧训练样本。"""

    scenario: str
    route_id: str
    town: str
    frame_id: int
    true_rs: str
    prompt_road_structure: str
    context_id: str
    question_domain: str
    action_signature: str
    event: str
    split: str
    goal_ego_xy: Tuple[float, float]
    history_rgb_paths: List[str]
    latest_rgb_path: str
    answers: Dict[str, bool]
    invalid_source: str = ""
    invalid_reason: str = ""
    context_detail: str = ""


@dataclass(frozen=True)
class WorkItem:
    """一条训练/验证 case。"""

    row: FrameRow
    spec: PromptSpec
    balance_key: str


def _resolve_rgb_path(raw: str, data_root: pathlib.Path) -> str:
    """解析相对路径，并兼容旧机器写出的 lead_data 绝对路径。"""

    path = pathlib.Path(raw).expanduser()
    root = data_root.expanduser()
    if path.is_absolute() and path.is_file():
        return str(path)
    parts = path.parts
    if "lead_data" in parts:
        candidate = root.joinpath(*parts[parts.index("lead_data") + 1 :])
        if candidate.is_file():
            return str(candidate)
    return str(root / path)


def _read_rows(
    path: pathlib.Path,
    split: str,
    max_frames: int = 0,
    data_root: Optional[pathlib.Path] = None,
) -> List[FrameRow]:
    """读取 frame_index.jsonl。"""

    root = pathlib.Path(data_root) if data_root is not None else (_AUTOMOT_ROOT / "lead_data")
    rows: List[FrameRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("dataset_name") != DATASET_NAME:
                raise ValueError(f"dataset_name mismatch in {path}: {obj.get('dataset_name')!r}")
            if str(obj.get("split")) != str(split):
                continue
            from qwen3vl_local.sft_new_loop_phase3.trajectory_action import validate_action_rule
            validate_action_rule(obj)
            from qwen3vl_local.sft_new_loop_phase3.source_mapping import validate_mapping_contract
            validate_mapping_contract(obj)
            goal = list(obj["goal_ego_xy"])
            if len(goal) != 2 or not all(math.isfinite(float(v)) for v in goal):
                raise ValueError("goal_ego_xy requires two finite coordinates")
            rows.append(
                FrameRow(
                    scenario=str(obj.get("scenario")),
                    route_id=str(obj.get("route_id")),
                    town=str(obj.get("town")),
                    frame_id=int(obj.get("frame_id")),
                    true_rs=str(obj.get("true_rs")),
                    prompt_road_structure=str(obj.get("prompt_road_structure")),
                    context_id=str(obj.get("context_id")),
                    question_domain=str(obj.get("question_domain")),
                    action_signature=str(obj.get("action_signature")),
                    event=str(obj.get("event")),
                    split=str(obj.get("split")),
                    goal_ego_xy=(float(goal[0]), float(goal[1])),
                    history_rgb_paths=[_resolve_rgb_path(str(x), root) for x in obj.get("history_rgb_paths", [])],
                    latest_rgb_path=_resolve_rgb_path(str(obj.get("latest_rgb_path")), root),
                    answers={key: bool(value) for key, value in (obj.get("answers") or {}).items()},
                    invalid_source=str(obj.get("invalid_source") or ""),
                    invalid_reason=str(obj.get("invalid_reason") or "wrong_road_structure"),
                    context_detail=str(obj.get("context_detail") or ""),
                )
            )
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {path}")
    return rows


def _work_item_seed(row: FrameRow, *parts: object) -> str:
    """返回 spec 的稳定种子字段。"""

    return ":".join(
        [row.scenario, row.route_id, str(row.frame_id), row.context_id, *[str(part) for part in parts]]
    )


def _answer_text(value: bool) -> str:
    """布尔转 YES/NO。"""

    return "YES" if bool(value) else "NO"


def _balance_class(row: FrameRow) -> str:
    """从 answers 恢复本行的平衡类别。"""

    return "INVALID" if row.answers.get(INVALID_KEY, False) else row.context_id


def _raw_focus_bin_counts(rows: Sequence[FrameRow]) -> Dict[str, int]:
    """统计上下文、动作签名、真实 RS、答案与 invalid 子类别。"""

    counts: Counter = Counter()
    for row in rows:
        balance_class = _balance_class(row)
        counts[f"class/{balance_class}"] += 1
        counts[f"question_domain/{row.question_domain}"] += 1
        counts[f"true_rs/{row.true_rs}"] += 1
        counts[f"action_signature/{row.action_signature}"] += 1
        if balance_class == "INVALID":
            for dimension, value in invalid_subgroup_keys(row).items():
                counts[f"invalid/{dimension}/{value}"] += 1
        for key in ANSWER_KEYS:
            counts[f"answer/{key}:{_answer_text(row.answers.get(key, False))}"] += 1
    return dict(counts)


def _make_item(row: FrameRow, *, seed: int) -> WorkItem:
    """构造单轮动作 case。"""

    spec = make_prompt_spec(
        variant="all_random_order",
        answers=row.answers,
        seed_key=_work_item_seed(row, seed, "action"),
        context_id=row.context_id,
        road_structure=row.prompt_road_structure,
        goal_xy=row.goal_ego_xy,
        context_detail=row.context_detail,
    )
    return WorkItem(row=row, spec=spec, balance_key=f"all_random_order/class/{_balance_class(row)}")


def _balanced_work(
    rows: Sequence[FrameRow],
    *,
    target_per_bin: int,
    seed: int,
    invalid_multiplier: float = 2.0,
    route_diverse: bool = False,
    require_invalid_coverage: bool = True,
) -> List[WorkItem]:
    """按动作上下文构建 deterministic work list；上下文内再按动作签名尽量均分。"""

    class_counts = Counter(_balance_class(row) for row in rows)
    missing = [key for key in BALANCE_CLASSES if class_counts.get(key, 0) <= 0]
    if missing:
        raise ValueError(
            "phase3 training balance requires every action context plus INVALID; "
            f"missing={missing} available={dict(sorted(class_counts.items()))}. "
            "Increase --max-frames or rebuild a complete index."
        )
    effective_target = int(target_per_bin)
    if effective_target <= 0:
        effective_target = min(int(class_counts[key]) for key in BALANCE_CLASSES)

    groups: Dict[str, List[WorkItem]] = defaultdict(list)
    for row in rows:
        item = _make_item(row, seed=seed)
        groups[item.balance_key].append(item)
    rng = random.Random(f"{seed}:new_phase3_balance:{len(rows)}:{effective_target}:{invalid_multiplier:.6f}")
    work: List[WorkItem] = []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        target = effective_target
        if key.endswith("/class/INVALID"):
            target = max(1, int(round(float(effective_target) * float(invalid_multiplier))))
            work.extend(
                balanced_invalid_items(
                    items, target=target, rng=rng, require_coverage=bool(require_invalid_coverage)
                )
            )
            continue
        by_signature: Dict[str, List[WorkItem]] = defaultdict(list)
        for item in items:
            by_signature[item.row.action_signature].append(item)
        quotas = even_quota_with_capacity({k: len(v) for k, v in by_signature.items()}, target)
        selected: List[WorkItem] = []
        for signature in sorted(quotas):
            count = int(quotas[signature])
            if count <= 0:
                continue
            bucket = by_signature[signature]
            selected.extend(
                route_diverse_sample(bucket, target=count, rng=rng)
                if route_diverse
                else (bucket[:count] if len(bucket) >= count else [bucket[i % len(bucket)] for i in range(count)])
            )
        shortfall = target - len(selected)
        if shortfall > 0:
            selected.extend(items[i % len(items)] for i in range(shortfall))
        work.extend(selected)
    rng.shuffle(work)
    return work


def _effective_class_target(rows: Sequence[FrameRow], requested: int) -> int:
    """返回全部平衡类别的共同基数。"""

    if int(requested) > 0:
        return int(requested)
    class_counts = Counter(_balance_class(row) for row in rows)
    missing = [key for key in BALANCE_CLASSES if class_counts.get(key, 0) <= 0]
    if missing:
        raise ValueError(
            f"cannot infer class target; missing={missing} available={dict(sorted(class_counts.items()))}"
        )
    return min(int(class_counts[key]) for key in BALANCE_CLASSES)


def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    """读取当前 RGB-history 合同选择出的图片。"""

    return [Image.open(path).convert("RGB") for path in paths]


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
    """映射当前输出的语义与格式 token 权重。"""

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


def _assistant_end_token_ids(bundle: Any) -> set:
    """返回 chat template 中可作为 assistant turn 结束符的 token id。"""

    ids: set = set()
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
    """构造模型输入，主监督答案值并低权重监督输出行格式与结束符。"""

    messages = build_action_messages(
        images=images,
        spec=spec,
        audit=False,
        history_rgb_mode=history_rgb_mode,
        target=target,
    )
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
        bundle, target, spec=spec, format_loss_weight=float(format_loss_weight)
    )
    pos = _find_subsequence(expanded, target_ids, 0)
    asst_header_ids = list(bundle.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"])
    _assert_inside_assistant_turn(expanded, pos, asst_header_ids, 0)
    for j, weight in enumerate(token_weights):
        if weight > 0:
            weights[pos + j] = float(weight)
            component_ids[pos + j] = int(token_components[j])
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
        "value_token_acc": float(
            bool(value_active.any()) and torch.equal(pred[value_active], shift_labels[value_active])
        ),
        "format_token_acc": float(
            bool(format_active.any()) and torch.equal(pred[format_active], shift_labels[format_active])
        ),
    }
    for component_id, key in enumerate(spec.output_keys, start=1):
        mask = active & shift_comp.eq(component_id)
        stats[f"answer/{key.lower()}_ok"] = float(bool(mask.any() and torch.equal(pred[mask], shift_labels[mask])))
    return loss, stats


def _ddp_sync_zero_loss(bundle: Any, images: Sequence[Image.Image]) -> torch.Tensor:
    """样本过长跳过时，仍走一次真实 model forward 来同步 DDP reducer。"""

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
    inputs = (
        bundle.processor(text=[chat_text], images=sync_images, return_tensors="pt", padding=True)
        if sync_images
        else bundle.processor(text=[chat_text], return_tensors="pt", padding=True)
    )
    kwargs = {
        key: value.to(bundle.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    out = bundle.model(**kwargs, use_cache=False, return_dict=True)
    return out.logits.sum() * 0.0


def _optimizer_step(
    *,
    params: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    max_grad_norm: float,
) -> None:
    """统一执行一次 optimizer step。"""

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
    """在独立 val split 上跑 teacher-forced loss 和当前问题组 token accuracy。"""

    was_training = bool(bundle.model.training)
    bundle.model.eval()
    loss_sum = 0.0
    samples = 0
    skipped = 0
    token_acc_sum = 0.0
    value_token_acc_sum = 0.0
    format_token_acc_sum = 0.0
    metric_ok: Counter = Counter()
    metric_count: Counter = Counter()
    invalid_gt_total = 0.0
    invalid_line_ok = 0.0
    invalid_actions_all_no_ok = 0.0
    invalid_joint_ok = 0.0
    invalid_subgroup_ok: Counter = Counter()
    invalid_subgroup_total: Counter = Counter()
    for item in work:
        row = item.row
        spec = item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        packed = _build_inputs(
            bundle,
            images=images,
            history_rgb_mode=history_rgb_mode,
            target=build_action_target(spec),
            spec=spec,
            max_length=int(max_length),
            format_loss_weight=float(format_loss_weight),
        )
        if packed is None:
            skipped += 1
            continue
        loss, stats = _loss_one(bundle, packed, spec)
        loss_sum += float(loss.detach().item())
        samples += 1
        token_acc_sum += float(stats.get("token_acc", 0.0))
        value_token_acc_sum += float(stats.get("value_token_acc", 0.0))
        format_token_acc_sum += float(stats.get("format_token_acc", 0.0))
        all_answer_ok = True
        for output_key, metric_key, _ in spec_metric_items(spec):
            ok = float(stats.get(f"answer/{output_key.lower()}_ok", 0.0))
            metric_ok[metric_key] += ok
            metric_count[metric_key] += 1
            all_answer_ok = all_answer_ok and bool(ok)
        if bool(row.answers.get(INVALID_KEY, False)):
            invalid_gt_total += 1.0
            invalid_line = bool(stats.get(f"answer/{INVALID_KEY.lower()}_ok", 0.0))
            actions_all_no = all(
                bool(stats.get(f"answer/{key.lower()}_ok", 0.0))
                for key in spec.output_keys
                if key != INVALID_KEY
            )
            invalid_line_ok += float(invalid_line)
            invalid_actions_all_no_ok += float(actions_all_no)
            invalid_joint_ok += float(invalid_line and actions_all_no)
            for dimension, value in invalid_subgroup_keys(row).items():
                subgroup_key = f"{dimension}/{value}"
                invalid_subgroup_total[subgroup_key] += 1
                invalid_subgroup_ok[subgroup_key] += int(all_answer_ok)
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
        invalid_actions_all_no_ok,
        invalid_joint_ok,
    ]
    values.extend(metric_ok[name] for name in metric_names)
    values.extend(metric_count[name] for name in metric_names)
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
        "invalid_actions_all_no_token_ok_rate": vals[8] / max(1.0, vals[6]),
        "invalid_joint_token_ok_rate": vals[9] / max(1.0, vals[6]),
    }
    metric_ok_vals = vals[offset : offset + len(metric_names)]
    offset += len(metric_names)
    metric_count_vals = vals[offset : offset + len(metric_names)]
    for idx, key in enumerate(metric_names):
        safe = key.lower()
        metrics[f"metric/{safe}_acc"] = metric_ok_vals[idx] / max(1.0, metric_count_vals[idx])
        metrics[f"metric/{safe}_samples"] = metric_count_vals[idx]
    subgroup_payloads: List[Any] = [{"ok": dict(invalid_subgroup_ok), "total": dict(invalid_subgroup_total)}]
    if int(world_size) > 1:
        subgroup_payloads = [None for _ in range(int(world_size))]
        dist.all_gather_object(
            subgroup_payloads, {"ok": dict(invalid_subgroup_ok), "total": dict(invalid_subgroup_total)}
        )
    merged_ok: Counter = Counter()
    merged_total: Counter = Counter()
    for payload in subgroup_payloads:
        merged_ok.update((payload or {}).get("ok", {}))
        merged_total.update((payload or {}).get("total", {}))
    for key, count in sorted(merged_total.items()):
        metrics[f"invalid_subgroup/{key}_samples"] = float(count)
        metrics[f"invalid_subgroup/{key}_exact"] = float(merged_ok.get(key, 0)) / max(1.0, float(count))
    if was_training:
        bundle.model.train()
    return metrics


def _answer_pattern(values: Mapping[str, Optional[str]]) -> str:
    """对当前被问到的输出行统计 ALL_NO / 单 YES / 多 YES / INVALID。"""

    if any(value not in ("YES", "NO") for value in values.values()):
        return "INVALID"
    positive = [key for key, value in values.items() if value == "YES"]
    if not positive:
        return "ALL_NO"
    if len(positive) == 1:
        return positive[0]
    return "MULTI:" + "+".join(sorted(positive))


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
    """在固定独立 val 样本上以真实 greedy generation 检查输出行格式与语义。"""

    model = bundle.unwrap()
    was_training = bool(model.training)
    model.eval()
    runtime = SimpleNamespace(
        model=model, processor=bundle.processor, tokenizer=bundle.tokenizer, device=bundle.device
    )
    samples = 0.0
    valid = 0.0
    exact = 0.0
    metric_ok: Counter = Counter()
    metric_count: Counter = Counter()
    slice_counts: Counter = Counter()
    action_counts: Counter = Counter()
    invalid_subgroup_counts: Counter = Counter()
    pattern_counts: Counter = Counter()
    records: List[Dict[str, Any]] = []
    for item in work:
        row = item.row
        spec = item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        state = _kv_start_state(
            runtime,
            build_action_messages(images=images, spec=spec, audit=False, history_rgb_mode=history_rgb_mode),
        )
        raw, _, _ = _student_generate_kv(runtime, state, int(max_new_tokens))
        parsed = parse_action_output(raw, spec=spec)
        gt = {q.output_key: bool(q.answer) for q in spec.questions}
        gt_text = {key: _answer_text(value) for key, value in gt.items()}
        parsed_text = {key: None if value is None else _answer_text(value) for key, value in parsed.items()}
        is_valid = all(parsed[key] is not None for key in spec.output_keys)
        samples += 1.0
        valid += float(is_valid)
        all_ok = bool(is_valid and all(bool(parsed[key]) == gt[key] for key in spec.output_keys))
        exact += float(all_ok)
        balance_class = _balance_class(row)
        slice_name = balance_class.lower()
        slice_counts[f"{slice_name}/total"] += 1
        slice_counts[f"{slice_name}/exact"] += int(all_ok)
        for key in ACTION_KEYS:
            if key not in spec.output_keys:
                continue
            if gt[key]:
                action_counts[f"{key}/gt_yes"] += 1
                action_counts[f"{key}/recall_hit"] += int(parsed.get(key) is True)
            if parsed.get(key) is True:
                action_counts[f"{key}/pred_yes"] += 1
                action_counts[f"{key}/precision_hit"] += int(gt[key])
        if balance_class == "INVALID":
            for dimension, value in invalid_subgroup_keys(row).items():
                invalid_subgroup_counts[f"{dimension}/{value}/total"] += 1
                invalid_subgroup_counts[f"{dimension}/{value}/exact"] += int(all_ok)
        for output_key, metric_key, answer in spec_metric_items(spec):
            if parsed.get(output_key) is not None:
                metric_ok[metric_key] += int(bool(parsed[output_key]) == bool(answer))
            metric_count[metric_key] += 1
        gt_pattern = _answer_pattern(gt_text)
        pred_pattern = _answer_pattern(parsed_text)
        pattern_counts["total"] += 1
        pattern_counts[f"gt_pattern/{gt_pattern}"] += 1
        pattern_counts[f"pred_pattern/{pred_pattern}"] += 1
        pattern_counts[f"pair/{gt_pattern}=>{pred_pattern}"] += 1
        pattern_counts["pattern_exact"] += int(gt_pattern == pred_pattern)
        if gt[INVALID_KEY]:
            pattern_counts["invalid_gt_total"] += 1
            actions_all_no = all(parsed.get(key) is False for key in spec.output_keys if key != INVALID_KEY)
            pattern_counts["invalid_pred_line_yes"] += int(parsed.get(INVALID_KEY) is True)
            pattern_counts["invalid_actions_all_no"] += int(actions_all_no)
            pattern_counts["invalid_joint_ok"] += int(parsed.get(INVALID_KEY) is True and actions_all_no)
        records.append(
            {
                "step": int(step),
                "scenario": row.scenario,
                "route_id": row.route_id,
                "town": row.town,
                "frame_id": row.frame_id,
                "true_rs": row.true_rs,
                "context_id": row.context_id,
                "question_domain": row.question_domain,
                "action_signature": row.action_signature,
                "invalid_source": row.invalid_source,
                "invalid_subgroups": invalid_subgroup_keys(row) if balance_class == "INVALID" else None,
                "prompt_spec": prompt_spec_to_json(spec),
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
    for name in BALANCE_CLASSES:
        key = name.lower()
        count = float(slice_counts.get(f"{key}/total", 0))
        metrics[f"slice/{key}_samples"] = count
        metrics[f"slice/{key}_exact"] = float(slice_counts.get(f"{key}/exact", 0)) / max(1.0, count)
    for key in ACTION_KEYS:
        gt_yes = float(action_counts.get(f"{key}/gt_yes", 0))
        pred_yes = float(action_counts.get(f"{key}/pred_yes", 0))
        metrics[f"action/{key.lower()}_gt_yes"] = gt_yes
        metrics[f"action/{key.lower()}_pred_yes"] = pred_yes
        metrics[f"action/{key.lower()}_recall"] = float(action_counts.get(f"{key}/recall_hit", 0)) / max(1.0, gt_yes)
        metrics[f"action/{key.lower()}_precision"] = float(
            action_counts.get(f"{key}/precision_hit", 0)
        ) / max(1.0, pred_yes)
    for key, count in sorted(invalid_subgroup_counts.items()):
        if not key.endswith("/total"):
            continue
        prefix = key.removesuffix("/total")
        metrics[f"invalid_subgroup/{prefix}_samples"] = float(count)
        metrics[f"invalid_subgroup/{prefix}_exact"] = float(
            invalid_subgroup_counts.get(f"{prefix}/exact", 0)
        ) / max(1.0, float(count))
    for key in ANSWER_KEYS:
        safe = key.lower()
        metrics[f"metric/{safe}_acc"] = metric_ok[key] / max(1.0, metric_count[key])
        metrics[f"metric/{safe}_samples"] = float(metric_count[key])
    total = max(1.0, float(pattern_counts.get("total", 0)))
    invalid_total = max(1.0, float(pattern_counts.get("invalid_gt_total", 0)))
    metrics["pattern/pattern_exact"] = float(pattern_counts.get("pattern_exact", 0)) / total
    metrics["invalid/gt_total"] = float(pattern_counts.get("invalid_gt_total", 0))
    metrics["invalid/line_yes_rate"] = float(pattern_counts.get("invalid_pred_line_yes", 0)) / invalid_total
    metrics["invalid/actions_all_no_rate"] = float(pattern_counts.get("invalid_actions_all_no", 0)) / invalid_total
    metrics["invalid/joint_ok_rate"] = float(pattern_counts.get("invalid_joint_ok", 0)) / invalid_total
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
                        "answer_patterns": {
                            key: int(value) for key, value in sorted(pattern_counts.items())
                        },
                        "invalid_subgroups": invalid_subgroup_report(work),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return metrics


def generation_checkpoint_guards(
    metrics: Mapping[str, float],
    *,
    min_invalid_exact: float,
    min_lane_change_recall: float,
    min_stop_recall: float,
    min_no_action_exact: float,
) -> Dict[str, Any]:
    """返回 checkpoint 多指标门槛的逐项审计结果。"""

    lane_change_recall = min(
        float(metrics.get("action/lane_change_left_recall", 0.0)),
        float(metrics.get("action/lane_change_right_recall", 0.0)),
    )
    values = {
        "invalid_exact": float(metrics.get("slice/invalid_exact", 0.0)),
        "lane_change_recall": lane_change_recall,
        "stop_recall": float(metrics.get("action/stop_recall", 0.0)),
        "no_action_context_exact": float(metrics.get("slice/ramp_merge_exit_exact", 0.0)),
    }
    floors = {
        "invalid_exact": float(min_invalid_exact),
        "lane_change_recall": float(min_lane_change_recall),
        "stop_recall": float(min_stop_recall),
        "no_action_context_exact": float(min_no_action_exact),
    }
    passed = {key: values[key] >= floors[key] for key in values}
    return {"all_ok": all(passed.values()), "values": values, "floors": floors, "passed": passed}


def generation_checkpoint_score(
    metrics: Mapping[str, float],
    *,
    min_invalid_exact: float,
    min_lane_change_recall: float,
    min_stop_recall: float,
    min_no_action_exact: float,
) -> Tuple[float, float, float, float, float]:
    """构造自由生成 checkpoint 选优分数；达标候选再按总 exact 选优。"""

    report = generation_checkpoint_guards(
        metrics,
        min_invalid_exact=min_invalid_exact,
        min_lane_change_recall=min_lane_change_recall,
        min_stop_recall=min_stop_recall,
        min_no_action_exact=min_no_action_exact,
    )
    exact = float(metrics.get("exact_accuracy", 0.0))
    pattern_exact = float(metrics.get("pattern/pattern_exact", 0.0))
    ratios = [
        1.0 if report["floors"][key] <= 0.0 else report["values"][key] / report["floors"][key]
        for key in report["values"]
    ]
    passed = list(report["passed"].values())
    if report["all_ok"]:
        return (1.0, exact, min(ratios), pattern_exact, float(sum(passed)))
    return (0.0, float(sum(passed)) / float(len(passed)), min(ratios), exact, pattern_exact)


def _save_adapter(
    bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace, *, step: int, name: str = "final"
) -> pathlib.Path:
    """保存 LoRA adapter 和 phase3 自描述配置。"""

    final_dir = output_dir / str(name)
    final_dir.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(final_dir))
    cfg = {
        "schema": "sft_new_loop_phase3_adapter_config",
        "route": "sft_new_loop_phase3_high_level_action",
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "production_prompt_sha256": action_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_script": str(_THIS),
        "git": _git_metadata(),
        "action_keys": list(ACTION_KEYS),
        "answer_order": list(ANSWER_KEYS),
        "context_ids": list(CONTEXT_IDS),
        "augment_variants": list(VARIANT_ORDER),
        "augment_variant_weights": dict(VARIANT_WEIGHTS),
        "history_rgb_mode": str(args.history_rgb_mode),
        "history_rgb_count": len(history_rgb_indices(args.history_rgb_mode)),
        "history_rgb_selected_indices": list(history_rgb_indices(args.history_rgb_mode)),
        "base_model_dir": str(args.model_dir),
        "data_root": str(args.data_root),
        "input_contract": (
            "single image+text user turn: RGB history, the Phase1/Phase2 road structure and situation "
            "premise, and the ego-frame route target (x forward, y negative left, y positive right)"
        ),
        "lora_vision_scope": str(args.lora_vision_scope),
        "lora_target_modules": list(bundle.lora_target_modules),
        "global_step": int(step),
        "num_epochs": int(args.num_epochs),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "focus_balance_count": int(args.focus_balance_count),
        "train_route_diverse": bool(args.train_route_diverse),
        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
        "eval_split": str(args.eval_split),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "format_loss_weight": float(args.format_loss_weight),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_route_diverse": bool(args.generation_eval_route_diverse),
        "generation_eval_sampling_seed": int(args.generation_eval_sampling_seed),
        "generation_eval_max_new_tokens": int(args.generation_eval_max_new_tokens),
        "generation_eval_min_valid_rate": float(args.generation_eval_min_valid_rate),
        "generation_eval_min_invalid_exact": float(args.generation_eval_min_invalid_exact),
        "generation_eval_min_lane_change_recall": float(args.generation_eval_min_lane_change_recall),
        "generation_eval_min_stop_recall": float(args.generation_eval_min_stop_recall),
        "generation_eval_min_no_action_exact": float(args.generation_eval_min_no_action_exact),
        "save_best_val": bool(args.save_best_val),
        "save_best_generation": bool(args.save_best_generation),
        "save_final": bool(args.save_final),
    }
    (final_dir / "sft_new_loop_phase3_adapter_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_dir


def default_pipeline_adapter(
    *,
    best_generation_available: bool,
    final_available: bool,
    fallback_generation_available: bool,
) -> Optional[str]:
    """返回自动评测的权重槽位；best 不存在时必须仍可回退 final。"""

    if best_generation_available:
        return "best_generation"
    if final_available:
        return "final"
    if fallback_generation_available:
        return "fallback_generation"
    return None


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
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_script": str(_THIS),
        "git": _git_metadata(),
        "output_dir": str(output_dir),
        "tb_dir": str(output_dir / "tb"),
        "run_log": os.environ.get("RUN_LOG", ""),
        "run_name": os.environ.get("RUN_NAME", ""),
        "run_timestamp": os.environ.get("RUN_TIMESTAMP", ""),
        "model_dir": str(args.model_dir),
        "index": str(args.index),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "eval_split": str(args.eval_split),
        "seed": int(args.seed),
        "world_size": int(world_size),
        "history_rgb_mode": str(args.history_rgb_mode),
        "history_rgb_count": len(history_rgb_indices(args.history_rgb_mode)),
        "train_rows": int(train_rows),
        "train_work_global": int(train_work_global),
        "train_work_rank": int(train_work_rank),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "eval_work_rank": int(eval_work_rank),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_global": int(generation_eval_global),
        "total_steps_global": int(total_steps),
        "focus_balance_count": int(args.focus_balance_count),
        "train_route_diverse": bool(args.train_route_diverse),
        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
        "production_prompt_sha256": action_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
    }
    (output_dir / "train_run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
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
        "invalid_focus_multiplier",
    ):
        writer.add_scalar(f"setup/{key}", float(payload[key]), 0)


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    validate_history_rgb_mode(args.history_rgb_mode)
    rank, local_rank, world_size = setup_distributed()
    device = (
        torch.device(f"cuda:{local_rank}")
        if world_size > 1
        else torch.device(
            args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    )
    if rank == 0:
        print(
            f"[startup] world_size={world_size} device={device} index={args.index} split={args.split} "
            f"focus_balance_count={args.focus_balance_count} eval_steps={args.eval_steps}",
            flush=True,
        )
    rows = _read_rows(
        pathlib.Path(args.index),
        split=str(args.split),
        max_frames=int(args.max_frames),
        data_root=pathlib.Path(args.data_root),
    )
    raw_focus_counts = _raw_focus_bin_counts(rows)
    effective_target = _effective_class_target(rows, int(args.focus_balance_count))
    full_work = _balanced_work(
        rows,
        target_per_bin=int(args.focus_balance_count),
        seed=int(args.seed),
        invalid_multiplier=float(args.invalid_focus_multiplier),
        route_diverse=bool(args.train_route_diverse),
        require_invalid_coverage=bool(args.require_invalid_coverage),
    )
    if not full_work:
        raise ValueError("balanced work list is empty")
    work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
    output_dir = pathlib.Path(args.output_dir)

    eval_rows: List[FrameRow] = []
    full_eval_work: List[WorkItem] = []
    eval_work: List[WorkItem] = []
    full_generation_eval_work: List[WorkItem] = []
    if int(args.eval_steps) > 0 and int(args.eval_balance_count) > 0:
        try:
            eval_rows = _read_rows(
                pathlib.Path(args.index),
                split=str(args.eval_split),
                max_frames=int(args.max_eval_frames),
                data_root=pathlib.Path(args.data_root),
            )
            full_eval_work = _balanced_work(
                eval_rows,
                target_per_bin=int(args.eval_balance_count),
                seed=int(args.seed) + 1009,
                require_invalid_coverage=bool(args.require_invalid_coverage),
            )
            eval_work = _split_work_for_rank(full_eval_work, rank=rank, world_size=world_size)
            if int(args.generation_eval_steps) > 0 and int(args.generation_eval_balance_count) > 0:
                full_generation_eval_work = _balanced_work(
                    eval_rows,
                    target_per_bin=int(args.generation_eval_balance_count),
                    seed=int(args.generation_eval_sampling_seed),
                    route_diverse=bool(args.generation_eval_route_diverse),
                    require_invalid_coverage=bool(args.require_invalid_coverage),
                )
        except Exception as exc:
            raise RuntimeError(
                "periodic validation was requested but its split cannot satisfy the required class balance. "
                "Rebuild/fix the dataset instead of silently training without validation."
            ) from exc

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        val_balance: Dict[str, Any] = {}
        if eval_work:
            val_balance = {
                "raw_available": _raw_focus_bin_counts(eval_rows),
                "global_sampled": dict(Counter(item.balance_key for item in full_eval_work)),
                "rank0_shard": dict(Counter(item.balance_key for item in eval_work)),
                "global_invalid_subgroups": invalid_subgroup_report(full_eval_work),
            }
        (output_dir / "train_balance.json").write_text(
            json.dumps(
                {
                    "world_size": int(world_size),
                    "history_rgb_mode": str(args.history_rgb_mode),
                    "history_rgb_selected_indices": list(history_rgb_indices(args.history_rgb_mode)),
                    "train": {
                        "split": str(args.split),
                        "focus_balance_count": int(args.focus_balance_count),
                        "invalid_focus_multiplier": float(args.invalid_focus_multiplier),
                        "route_diverse": bool(args.train_route_diverse),
                        "effective_focus_target_per_class": int(effective_target),
                        "resample_each_epoch": True,
                        "epoch_seed_formula": "seed + epoch * 1000003",
                        "steps_per_epoch_global": int(math.ceil(len(full_work) / max(1, int(world_size)))),
                        "raw_available": raw_focus_counts,
                        "global_sampled": dict(Counter(item.balance_key for item in full_work)),
                        "action_signature_counts": dict(
                            Counter(item.row.action_signature for item in full_work)
                        ),
                        "route_diversity": route_diversity_report([item.row.__dict__ for item in full_work]),
                        "rank0_shard": dict(Counter(item.balance_key for item in work)),
                        "global_invalid_subgroups": invalid_subgroup_report(full_work),
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
                        "route_diverse": bool(args.generation_eval_route_diverse),
                        "sampling_seed": int(args.generation_eval_sampling_seed),
                        "max_new_tokens": int(args.generation_eval_max_new_tokens),
                        "min_format_valid_rate": float(args.generation_eval_min_valid_rate),
                        "min_invalid_exact": float(args.generation_eval_min_invalid_exact),
                        "min_lane_change_recall": float(args.generation_eval_min_lane_change_recall),
                        "min_stop_recall": float(args.generation_eval_min_stop_recall),
                        "min_no_action_exact": float(args.generation_eval_min_no_action_exact),
                        "global_sampled": dict(
                            Counter(item.balance_key for item in full_generation_eval_work)
                        ),
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
        bundle.model = DDP(
            bundle.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay), betas=(0.9, 0.95)
    )
    steps_per_epoch = max(1, math.ceil(len(full_work) / max(1, int(world_size))))
    total_steps = (
        int(args.max_steps)
        if int(args.max_steps) > 0
        else max(1, steps_per_epoch * max(1, int(args.num_epochs)))
    )
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
    best_generation_score = (-math.inf,) * 5
    best_fallback_score = (-math.inf,) * 5
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "train_eval_metrics.jsonl"
    window_loss_sum = 0.0
    window_samples = 0
    window_stats: Counter = Counter()
    window_metric_ok: Counter = Counter()
    window_metric_count: Counter = Counter()
    t0 = time.time()
    bundle.model.train()
    epoch = 0
    while global_step < total_steps:
        if epoch > 0:
            full_work = _balanced_work(
                rows,
                target_per_bin=int(args.focus_balance_count),
                seed=int(args.seed) + epoch * 1_000_003,
                invalid_multiplier=float(args.invalid_focus_multiplier),
                route_diverse=bool(args.train_route_diverse),
                require_invalid_coverage=bool(args.require_invalid_coverage),
            )
        random.Random(int(args.seed) + epoch * 1_000_003).shuffle(full_work)
        work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
        if rank == 0:
            epoch_invalid_report = invalid_subgroup_report(full_work)
            epoch_balance_dir = output_dir / "balance"
            epoch_balance_dir.mkdir(parents=True, exist_ok=True)
            (epoch_balance_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                json.dumps(
                    {
                        "epoch": int(epoch + 1),
                        "seed": int(args.seed) + epoch * 1_000_003,
                        "class_counts": dict(Counter(_balance_class(item.row) for item in full_work)),
                        "action_signature_counts": dict(
                            Counter(item.row.action_signature for item in full_work)
                        ),
                        "route_diverse": bool(args.train_route_diverse),
                        "invalid_subgroups": epoch_invalid_report,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if writer:
                for dimension in ("source_class", "true_rs", "asked_context", "joint_signature"):
                    for value, count in epoch_invalid_report[dimension]["counts"].items():
                        writer.add_scalar(
                            f"balance/invalid/{dimension}/{_tb_tag(value)}", float(count), int(epoch + 1)
                        )
                for guard, passed in epoch_invalid_report["guards"].items():
                    writer.add_scalar(f"balance/invalid_guard/{_tb_tag(guard)}", float(bool(passed)), int(epoch + 1))
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
            packed = _build_inputs(
                bundle,
                images=images,
                history_rgb_mode=args.history_rgb_mode,
                target=build_action_target(spec),
                spec=spec,
                max_length=int(args.max_length),
                format_loss_weight=float(args.format_loss_weight),
            )
            if packed is None:
                skipped += 1
                loss = _ddp_sync_zero_loss(bundle, images)
                stats = {"denom": 0.0, "token_acc": 0.0, "value_token_acc": 0.0, "format_token_acc": 0.0}
            else:
                loss, stats = _loss_one(bundle, packed, spec)
            loss_value = float(loss.detach().item())
            if packed is not None:
                window_loss_sum += loss_value
                window_samples += 1
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
                    "balance_key": item.balance_key,
                }
                for key, value in window_stats.items():
                    window_payload[f"mean/{key}"] = float(value) / float(window_den)
                for key in ANSWER_KEYS:
                    denom = max(1.0, float(window_metric_count[key]))
                    window_payload[f"answer_acc/{key}"] = float(window_metric_ok[key]) / denom
                    window_payload[f"answer_samples/{key}"] = float(window_metric_count[key])
                _append_jsonl(train_metrics_path, window_payload)
                _write_scalar_dict(writer, "train_window", window_payload, global_step)
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} loss={loss_value:.4f} "
                    f"win_loss={window_payload['loss_mean']:.4f} bin={item.balance_key} "
                    f"skipped={skipped} world={world_size} elapsed={time.time() - t0:.1f}s"
                )
                window_loss_sum = 0.0
                window_samples = 0
                window_stats.clear()
                window_metric_ok.clear()
                window_metric_count.clear()
            global_step += 1
            saved_deferred_checkpoint = False
            if pending_checkpoint_step is not None and optimizer_stepped:
                if rank == 0:
                    ckpt_dir = _save_adapter(
                        bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}"
                    )
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
                        f"format_acc={metrics['format_token_acc']:.4f}"
                    )
                generation_metrics: Optional[Dict[str, float]] = None
                run_generation_eval = (
                    bool(full_generation_eval_work)
                    and int(args.generation_eval_steps) > 0
                    and global_step % int(args.generation_eval_steps) == 0
                )
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
                        f"exact={generation_metrics['exact_accuracy']:.4f}"
                    )
                if run_generation_eval and world_size > 1:
                    ddp_barrier(local_rank)
                if rank == 0:
                    eligible_for_best = not bool(full_generation_eval_work) or run_generation_eval
                    format_gate_ok = eligible_for_best and (
                        generation_metrics is None
                        or float(generation_metrics["format_valid_rate"]) >= float(args.generation_eval_min_valid_rate)
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
                    if (
                        bool(args.save_best_generation)
                        and run_generation_eval
                        and format_gate_ok
                        and generation_metrics is not None
                    ):
                        guard_report = generation_checkpoint_guards(
                            generation_metrics,
                            min_invalid_exact=float(args.generation_eval_min_invalid_exact),
                            min_lane_change_recall=float(args.generation_eval_min_lane_change_recall),
                            min_stop_recall=float(args.generation_eval_min_stop_recall),
                            min_no_action_exact=float(args.generation_eval_min_no_action_exact),
                        )
                        gen_score = generation_checkpoint_score(
                            generation_metrics,
                            min_invalid_exact=float(args.generation_eval_min_invalid_exact),
                            min_lane_change_recall=float(args.generation_eval_min_lane_change_recall),
                            min_stop_recall=float(args.generation_eval_min_stop_recall),
                            min_no_action_exact=float(args.generation_eval_min_no_action_exact),
                        )
                        guard_ok = bool(guard_report["all_ok"])
                        slot = "best_generation" if guard_ok else "fallback_generation"
                        previous_score = best_generation_score if guard_ok else best_fallback_score
                        if gen_score > previous_score:
                            if guard_ok:
                                best_generation_score = gen_score
                            else:
                                best_fallback_score = gen_score
                            selected_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=slot)
                            (output_dir / f"{slot}.json").write_text(
                                json.dumps(
                                    {
                                        "step": global_step,
                                        "val_split": str(args.eval_split),
                                        "selection": "multi_guard_then_max_free_generation_exact",
                                        "selection_slot": slot,
                                        "selection_score": list(gen_score),
                                        "generation_guards": guard_report,
                                        "generation_guards_ok": guard_ok,
                                        "generation_exact_accuracy": float(
                                            generation_metrics.get("exact_accuracy", 0.0)
                                        ),
                                        "teacher_forced_loss": float(metrics["loss"]),
                                        "generation": generation_metrics,
                                        "history_rgb_mode": str(args.history_rgb_mode),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            print(
                                f"[{slot}] step={global_step} guards_ok={guard_ok} "
                                f"guards={guard_report['passed']} "
                                f"exact={generation_metrics.get('exact_accuracy', 0.0):.4f} adapter={selected_dir}"
                            )
            checkpoint_due = int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0
            if checkpoint_due and not saved_deferred_checkpoint:
                if accum_steps == 0:
                    if rank == 0:
                        ckpt_dir = _save_adapter(
                            bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}"
                        )
                        print(f"[save] step={global_step} adapter={ckpt_dir}")
                else:
                    pending_checkpoint_step = int(global_step)
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
            params=params, optimizer=optimizer, scheduler=scheduler, max_grad_norm=float(args.max_grad_norm)
        )
        accum_steps = 0
        if pending_checkpoint_step is not None and rank == 0:
            _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
            pending_checkpoint_step = None

    if world_size > 1:
        ddp_barrier(local_rank)
    final_dir = (
        _save_adapter(bundle, output_dir, args, step=global_step)
        if rank == 0 and bool(args.save_final)
        else None
    )
    if rank == 0:
        best_generation_available = (
            (output_dir / "best_generation.json").is_file()
            and (output_dir / "best_generation" / "sft_new_loop_phase3_adapter_config.json").is_file()
        )
        fallback_generation_available = (
            (output_dir / "fallback_generation.json").is_file()
            and (output_dir / "fallback_generation" / "sft_new_loop_phase3_adapter_config.json").is_file()
        )
        final_available = (
            final_dir is not None and (final_dir / "sft_new_loop_phase3_adapter_config.json").is_file()
        )
        selection_status = {
            "best_generation_available": best_generation_available,
            "fallback_generation_available": fallback_generation_available,
            "final_available": final_available,
            "pipeline_adapter": default_pipeline_adapter(
                best_generation_available=best_generation_available,
                final_available=final_available,
                fallback_generation_available=fallback_generation_available,
            ),
            "production_ready": best_generation_available,
            "contract": (
                "best_generation records the configured action guards. The normal full pipeline evaluates "
                "and packages best_generation when available, otherwise final."
            ),
        }
        (output_dir / "generation_selection_status.json").write_text(
            json.dumps(selection_status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if writer:
        writer.close()
    if rank == 0:
        print(f"[done] saved adapter to {final_dir}" if final_dir is not None else "[done] final adapter disabled")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Train sft_new_loop_phase3 single-turn high-level action LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_data/frame_index.jsonl"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_runs/manual"))
    p.add_argument("--split", default="train")
    p.add_argument("--history-rgb-mode", choices=HISTORY_RGB_MODES, default=DEFAULT_HISTORY_RGB_MODE)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--focus-balance-count",
        type=int,
        default=0,
        help="per action context; 0 uses the smallest context/INVALID class",
    )
    p.add_argument(
        "--train-route-diverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="within each train class, rotate routes before taking another frame from the same route",
    )
    p.add_argument(
        "--invalid-focus-multiplier",
        type=float,
        default=2.0,
        help="training-only multiplier for mismatched-context invalid samples",
    )
    p.add_argument(
        "--require-invalid-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require INVALID rows to cover R1-R5 and every mismatched context; disable only for smoke subsets",
    )
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-split", default="val")
    p.add_argument("--eval-steps", type=int, default=2_000)
    p.add_argument("--eval-balance-count", type=int, default=16)
    p.add_argument("--max-eval-frames", type=int, default=0)
    p.add_argument("--format-loss-weight", type=float, default=0.25)
    p.add_argument("--generation-eval-steps", type=int, default=2_000)
    p.add_argument("--generation-eval-balance-count", type=int, default=32)
    p.add_argument("--generation-eval-sampling-seed", type=int, default=20260904)
    p.add_argument(
        "--generation-eval-route-diverse", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--generation-eval-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-eval-min-valid-rate", type=float, default=1.0)
    p.add_argument("--generation-eval-min-invalid-exact", type=float, default=0.80)
    p.add_argument("--generation-eval-min-lane-change-recall", type=float, default=0.60)
    p.add_argument("--generation-eval-min-stop-recall", type=float, default=0.80)
    p.add_argument("--generation-eval-min-no-action-exact", type=float, default=0.50)
    p.add_argument("--save-steps", type=int, default=20_000)
    p.add_argument("--save-best-val", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-best-generation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-vision-scope", default="off")
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--no-tb", action="store_true")
    args = p.parse_args()
    if int(args.generation_eval_steps) > 0 and int(args.eval_steps) > 0:
        if int(args.generation_eval_steps) % int(args.eval_steps) != 0:
            raise ValueError("--generation-eval-steps must be a multiple of --eval-steps")
    return args


if __name__ == "__main__":
    train(parse_args())
