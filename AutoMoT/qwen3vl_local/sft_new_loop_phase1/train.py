#!/usr/bin/env python3
"""训练 sft_new_loop_phase1 八问 LoRA adapter。

训练目标是 Phase1 四问 + Phase2 四个 RS 问题的 YES/NO 语义 token，并以低权重
监督字段格式和 assistant 结束符。采样时每帧展开成八个不可见 focus 视图，
按 `问题 x YES/NO` exact balance；prompt 中不出现 focus，模型仍输出当前 spec 的
全部行；当前 focus 行使用完整语义权重，其他主答案行使用小权重，避免同一 target 中
大量自然 NO 的非 focus 行破坏采样器声明的 1:1；RS_HIGHWAY/GROUP 派生行继续完整监督。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
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

from qwen3vl_local.sft_new_loop_phase1 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_new_loop_phase1.history_rgb import (  # noqa: E402
    DEFAULT_HISTORY_RGB_MODE,
    HISTORY_RGB_MODES,
    history_rgb_indices,
    history_rgb_mode_tag,
    select_history_rgb_paths,
    validate_history_rgb_mode,
)
from qwen3vl_local.sft_new_loop_phase1.prompts import (  # noqa: E402
    ANSWER_KEYS,
    GROUP_DEFINITIONS,
    PHASE1_ANSWER_KEYS,
    PHASE2_ANSWER_KEYS,
    PROMPT_NAME,
    SUBSET_COUNTS,
    SYSTEM_PROMPT,
    TRAIN_VARIANT_WEIGHTS,
    VARIANT_ORDER,
    VARIANT_WEIGHTS,
    PromptSpec,
    build_phase1_prompt,
    build_phase1_target,
    focus_phase,
    make_prompt_spec,
    parse_phase1_output,
    phase1_prompt_sha256,
    phase2_output_keys,
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
DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT = 0.1
SEMANTIC_SUPERVISION = "focus_plus_scaled_class_balanced_nonfocus_and_derived_v2"
GENERATION_FOCUS_METRIC_KEYS = tuple(f"focus_{key.lower()}_acc" for key in ANSWER_KEYS)


def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量；非法值直接报错，避免 DDP 超时配置静默失效。"""

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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
            raise RuntimeError("sft_new_loop_phase1 DDP requires CUDA.")
        torch.cuda.set_device(local_rank)
        timeout = datetime.timedelta(seconds=_env_int("DDP_TIMEOUT_SECONDS", 7200))
        try:
            dist.init_process_group(
                backend="nccl",
                timeout=timeout,
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        except TypeError:
            dist.init_process_group(backend="nccl", timeout=timeout)
    return rank, local_rank, world_size


def _ddp_barrier(local_rank: int) -> None:
    """执行短同步 barrier，并显式绑定本 rank GPU，避免 NCCL 推断设备。"""

    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier(device_ids=[int(local_rank)])
    except TypeError:
        dist.barrier()


def _write_sync_file(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """原子写入 rank0 同步文件，供其它 rank 在长 generation eval 时轮询。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_sync_payload(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """读取同步文件；写入中的临时状态返回 None，由调用方继续等待。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _wait_for_rank0_sync(
    done_path: pathlib.Path,
    error_path: pathlib.Path,
    *,
    timeout_seconds: int,
    sync_token: str,
) -> None:
    """等待 rank0 完成长耗时 generation eval；等待期间不占用 NCCL collective。"""

    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        done_payload = _read_sync_payload(done_path)
        if done_payload and done_payload.get("sync_token") == sync_token:
            return
        error_payload = _read_sync_payload(error_path)
        if error_payload and error_payload.get("sync_token") == sync_token:
            raise RuntimeError(f"rank0 generation eval failed before sync: {error_payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for rank0 generation eval sync file {done_path} "
                f"after {timeout_seconds} seconds; sync_token={sync_token}"
            )
        time.sleep(5.0)


def cleanup_distributed() -> None:
    """清理 torch.distributed。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _append_jsonl(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """追加 JSONL 指标，训练中断后也能审计 eval 历史。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_run_metadata(
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
    """写入训练运行 manifest，保证 checkpoint/TB/数据口径可追溯。"""

    payload = {
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_script": str(_THIS),
        "git": _git_metadata(),
        "output_dir": str(output_dir),
        "tb_dir": str(output_dir / "tb"),
        "model_dir": str(args.model_dir),
        "index": str(args.index),
        "data_root": str(args.data_root),
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
        "save_steps": int(args.save_steps),
        "total_steps_rank": int(total_steps),
        "focus_balance_count": int(args.focus_balance_count),
        "phase1_output_ordering": "deterministic_random_per_case",
        "semantic_supervision": SEMANTIC_SUPERVISION,
        "non_focus_semantic_loss_weight": float(args.non_focus_semantic_loss_weight),
        "max_train_frame_repeat": int(args.max_train_frame_repeat),
        "train_variant_weights": dict(TRAIN_VARIANT_WEIGHTS),
        "eval_variant_weights": dict(VARIANT_WEIGHTS),
        "resample_each_epoch": True,
        "epoch_seed_formula": "seed + epoch * 1000003",
        "production_prompt_sha256": phase1_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
    }
    (output_dir / "train_run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
    """一条带 Phase2 augment spec 的训练/验证 case。"""

    row: FrameRow
    focus: str
    spec: PromptSpec
    balance_key: str
    augment_balance_key: str


def _semantic_base_weights(
    item: WorkItem,
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> Dict[str, float]:
    """返回每个语义答案行在类别再平衡前的角色权重。"""

    non_focus_weight = float(non_focus_semantic_loss_weight)
    if not 0.0 <= non_focus_weight <= 1.0:
        raise ValueError("non_focus_semantic_loss_weight must be in [0, 1]")
    if item.focus not in item.spec.output_keys:
        raise RuntimeError(
            f"focus {item.focus!r} is absent from prompt output keys {item.spec.output_keys!r}"
        )
    weights: Dict[str, float] = {}
    for output_key, metric_key, _ in spec_metric_items(item.spec):
        if output_key == item.focus:
            weights[output_key] = 1.0
        elif metric_key not in ANSWER_KEYS:
            # hierarchical 的 RS_HIGHWAY/GROUP 没有独立 focus 桶，继续完整监督。
            weights[output_key] = 1.0
        elif non_focus_weight > 0.0:
            # RGB 审计显示联合输出常出现 focus 正确、副行错误；只给小权重，避免
            # 自然 NO 和局部跨阶段标签冲突重新压过均衡 focus 监督。
            weights[output_key] = non_focus_weight
    return weights


def _semantic_output_keys(
    item: WorkItem,
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> Tuple[str, ...]:
    """返回训练时施加 YES/NO 语义 loss 的输出行。

    focus 主行完整监督；其它主答案行使用可配置的小权重；hierarchical 的
    RS_HIGHWAY/GROUP 没有独立 focus 桶，因此继续完整监督。所有行随后再按
    当轮有效语义质量修正 YES/NO 数量差。
    """

    base_weights = _semantic_base_weights(item, non_focus_semantic_loss_weight)
    return tuple(key for key in item.spec.output_keys if key in base_weights)


def _semantic_class_weights(
    work: Sequence[WorkItem],
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> Dict[str, float]:
    """按 focus/non-focus 基础质量计算每个 metric 的 YES/NO 等质量权重。"""

    counts: Counter[str] = Counter()
    metrics: set[str] = set()
    for item in work:
        base_weights = _semantic_base_weights(item, non_focus_semantic_loss_weight)
        for output_key, metric_key, answer in spec_metric_items(item.spec):
            if output_key in base_weights:
                metrics.add(metric_key)
                counts[f"{metric_key}:{_answer_text(answer)}"] += float(base_weights[output_key])
    weights: Dict[str, float] = {}
    for metric_key in sorted(metrics):
        yes_key = f"{metric_key}:YES"
        no_key = f"{metric_key}:NO"
        yes = float(counts.get(yes_key, 0.0))
        no = float(counts.get(no_key, 0.0))
        if yes <= 0 or no <= 0:
            if metric_key in ANSWER_KEYS:
                raise RuntimeError(
                    f"semantic supervision requires both YES and NO for {metric_key}: YES={yes} NO={no}"
                )
            # Tiny smoke/eval work may expose only one class for a derived
            # hierarchical metric.  Keep its observed class neutral instead
            # of inventing a missing class or aborting an otherwise valid check.
            if yes > 0:
                weights[yes_key] = 1.0
            if no > 0:
                weights[no_key] = 1.0
            continue
        total = yes + no
        weights[yes_key] = total / (2.0 * yes)
        weights[no_key] = total / (2.0 * no)
    return weights


def _semantic_output_weights(
    item: WorkItem,
    class_weights: Mapping[str, float],
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> Dict[str, float]:
    """把 metric-level 类别权重映射到当前 target 的输出键。"""

    base_weights = _semantic_base_weights(item, non_focus_semantic_loss_weight)
    weights: Dict[str, float] = {}
    for output_key, metric_key, answer in spec_metric_items(item.spec):
        if output_key not in base_weights:
            continue
        label = f"{metric_key}:{_answer_text(answer)}"
        if label not in class_weights:
            raise RuntimeError(f"missing semantic class weight for {label}")
        weights[output_key] = float(base_weights[output_key]) * float(class_weights[label])
    return weights


def _resolve_rgb_path(raw: str, data_root: pathlib.Path) -> str:
    """Resolve relative RGB paths and remap old absolute lead_data paths."""

    value = str(raw)
    root = data_root.expanduser()
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        return str(root / path)
    parts = path.parts
    if "lead_data" in parts:
        idx = parts.index("lead_data")
        rel = pathlib.Path(*parts[idx + 1 :])
        remapped = root / rel
        if remapped.exists() or not path.exists():
            return str(remapped)
    return str(path)


def _read_rows(path: pathlib.Path, split: str, max_frames: int = 0, data_root: Optional[pathlib.Path] = None) -> List[FrameRow]:
    """读取 frame_index.jsonl。"""

    root = pathlib.Path(data_root) if data_root is not None else (_AUTOMOT_ROOT / "lead_data")
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
                    history_rgb_paths=[_resolve_rgb_path(str(x), root) for x in obj.get("history_rgb_paths", [])],
                    latest_rgb_path=_resolve_rgb_path(str(obj.get("latest_rgb_path")), root),
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
    """返回八个主任务的固定 YES/NO focus 桶顺序。"""

    return [f"{key}:{value}" for key in ANSWER_KEYS for value in ("YES", "NO")]


def _assert_exact_focus_balance(
    work: Sequence[WorkItem],
    *,
    target: int,
    context: str,
    keys: Sequence[str] = ANSWER_KEYS,
) -> None:
    """确保完整 work list 的所有 `问题 x YES/NO` 桶严格等量。"""

    counts = Counter(item.balance_key for item in work)
    expected = int(target)
    expected_bins = [f"{key}:{value}" for key in keys for value in ("YES", "NO")]
    invalid = {key: int(counts.get(key, 0)) for key in expected_bins if int(counts.get(key, 0)) != expected}
    if invalid:
        raise RuntimeError(
            f"{context} violates per-question YES/NO 1:1 balance; "
            f"expected every bin={expected}, got={dict(counts)}, invalid={invalid}"
        )


def _raw_focus_bin_counts(rows: Sequence[FrameRow]) -> Dict[str, int]:
    """统计 split 中可用的原始 focus 桶计数，供 balance artifact 审计。"""

    counts: Counter[str] = Counter()
    for row in rows:
        for focus in ANSWER_KEYS:
            counts[_focus_key(row, focus)] += 1
    return {key: int(counts.get(key, 0)) for key in _expected_focus_bins()}


def _metric_names() -> List[str]:
    """返回融合评测里需要长期追踪的真实问题/诊断指标名。"""

    return [*ANSWER_KEYS, "RS_HIGHWAY", *[f"GROUP:{key}" for key in GROUP_DEFINITIONS]]


def _answer_text(value: bool) -> str:
    """布尔转 YES/NO。"""

    return "YES" if bool(value) else "NO"


def _work_item_seed(row: FrameRow, *parts: object) -> str:
    """返回增强 spec 的稳定种子字段。"""

    return ":".join(
        [row.scenario, row.route_id, str(row.frame_id), row.rs, *[str(part) for part in parts]]
    )


def _canonical_phase2_answer_sets() -> List[Dict[str, bool]]:
    """返回 R1/R2/R3/R4/R5 的合法 Phase2 四问答案原型。"""

    answer_sets: List[Dict[str, bool]] = []
    for rs in ("R1", "R2", "R3", "R4", "R5"):
        answers = {key: False for key in PHASE2_ANSWER_KEYS}
        if rs != "R3":
            answers[f"RS{rs[1]}"] = True
        answer_sets.append(answers)
    return answer_sets


def _expected_all_augment_keys() -> List[str]:
    """返回 all-random-order 增强的固定 Phase2 balance key 空间。"""

    return [f"all_random_order/{key}:{value}" for key in PHASE2_ANSWER_KEYS for value in ("YES", "NO")]


def _expected_subset_augment_keys() -> List[str]:
    """返回 subset-random 增强在合法 RS 标签下可能出现的 balance key。"""

    keys: set[str] = set()
    for count in SUBSET_COUNTS:
        for subset in itertools.permutations(PHASE2_ANSWER_KEYS, int(count)):
            for answers in _canonical_phase2_answer_sets():
                items = ",".join(f"{key}:{_answer_text(answers[key])}" for key in subset)
                keys.add(f"subset_random/q{int(count)}/items/{items}")
    return sorted(keys)


def _expected_hierarchical_augment_keys() -> List[str]:
    """返回 hierarchical-probe 增强在合法 RS 标签下可能出现的 balance key。"""

    keys: set[str] = set()
    for answers in _canonical_phase2_answer_sets():
        positives = [key for key in PHASE2_ANSWER_KEYS if answers[key]]
        rs = positives[0].replace("RS", "R") if positives else "R3"
        highway = _answer_text(rs == "R3")
        for group_id, group_def in GROUP_DEFINITIONS.items():
            group_answer = _answer_text(rs in set(group_def[3]))
            for detail_key in PHASE2_ANSWER_KEYS:
                detail_answer = _answer_text(bool(answers[detail_key]))
                keys.add(
                    f"hierarchical_probe/highway:{highway}"
                    f"/group/{group_id}:{group_answer}"
                    f"/detail/{detail_key}:{detail_answer}"
                )
    return sorted(keys)


def _subset_key_count(key: str) -> int:
    """从 subset balance key 提取实际问题数量。"""

    match = re.search(r"subset_random/q([123])/", key)
    if not match:
        raise ValueError(f"malformed subset balance key: {key}")
    return int(match.group(1))


def _subset_key_labels(key: str) -> Tuple[str, ...]:
    """从 subset balance key 提取实际输出 RS×YES/NO 标签。"""

    count = _subset_key_count(key)
    labels = tuple(
        f"subset_q{count}/{rs}:{answer}"
        for rs, answer in re.findall(r"(RS[1245]):(YES|NO)", key)
    )
    if len(labels) != count:
        raise ValueError(f"malformed subset balance key labels: {key}")
    return labels


def _hierarchical_key_labels(key: str) -> Tuple[str, str, str]:
    """从 hierarchical balance key 提取 HIGHWAY/GROUP/DETAIL 三个边际标签。"""

    high = re.search(r"highway:(YES|NO)", key)
    group = re.search(r"/group/([^:]+):(YES|NO)", key)
    detail = re.search(r"/detail/([^:]+):(YES|NO)", key)
    if not (high and group and detail):
        raise ValueError(f"malformed hierarchical balance key: {key}")
    return (
        f"highway:{high.group(1)}",
        f"group/{group.group(1)}:{group.group(2)}",
        f"detail/{detail.group(1)}:{detail.group(2)}",
    )


def _subset_key_targets(keys: Sequence[str], total: int) -> Dict[str, int]:
    """按 q1/q2/q3 和实际输出 RS×YES/NO 多边际分配 subset 配额。"""

    total = int(total)
    if total <= 0:
        return {}
    chosen: Counter[str] = Counter()
    count_totals = {
        int(count): total // len(SUBSET_COUNTS) + int(idx < total % len(SUBSET_COUNTS))
        for idx, count in enumerate(SUBSET_COUNTS)
    }
    for count in SUBSET_COUNTS:
        count_keys = [key for key in sorted(keys) if _subset_key_count(key) == int(count)]
        if not count_keys:
            continue
        count_total = int(count_totals[int(count)])
        target: Counter[str] = Counter()
        asked_per_rs = (count_total * int(count)) // len(PHASE2_ANSWER_KEYS)
        yes_per_rs = min(asked_per_rs // 2, count_total // len(PHASE2_ANSWER_KEYS))
        no_per_rs = max(0, asked_per_rs - yes_per_rs)
        for rs in PHASE2_ANSWER_KEYS:
            target[f"subset_q{count}/{rs}:YES"] = yes_per_rs
            target[f"subset_q{count}/{rs}:NO"] = no_per_rs
        labels_by_key = {key: _subset_key_labels(key) for key in count_keys}
        current: Counter[str] = Counter()
        for _ in range(count_total):
            best_key = None
            best_score = -10**18
            for key, labels in labels_by_key.items():
                score = 0.0
                for label in labels:
                    deficit = target[label] - current[label]
                    weight = 100.0 / max(1.0, float(target[label]))
                    score += deficit * weight
                    if deficit <= 0:
                        score += deficit * weight * 2.0
                score -= chosen[key]
                if score > best_score:
                    best_key = key
                    best_score = score
            if best_key is None:
                break
            chosen[best_key] += 1
            for label in labels_by_key[best_key]:
                current[label] += 1
    return {key: int(value) for key, value in chosen.items() if int(value) > 0}


def _hierarchical_key_targets(keys: Sequence[str], total: int) -> Dict[str, int]:
    """用多边际贪心为 hierarchical keys 分配近似均衡配额。"""

    total = int(total)
    if total <= 0:
        return {}
    target: Counter[str] = Counter()
    target["highway:YES"] = total // 2
    target["highway:NO"] = total - target["highway:YES"]
    group_bins = [f"group/{group_id}:{answer}" for group_id in GROUP_DEFINITIONS for answer in ("YES", "NO")]
    detail_bins = [f"detail/{detail_key}:{answer}" for detail_key in PHASE2_ANSWER_KEYS for answer in ("YES", "NO")]
    for bins in (group_bins, detail_bins):
        base = total // len(bins)
        remainder = total - base * len(bins)
        for idx, name in enumerate(bins):
            target[name] = base + int(idx < remainder)
    labels_by_key = {key: _hierarchical_key_labels(key) for key in sorted(keys)}
    current: Counter[str] = Counter()
    chosen: Counter[str] = Counter()
    for _ in range(total):
        best_key = None
        best_score = -10**18
        for key, labels in labels_by_key.items():
            score = 0
            for label in labels:
                deficit = target[label] - current[label]
                weight = 30 if label.startswith("highway:") else 10
                score += deficit * weight
                if deficit <= 0:
                    score += deficit * weight * 2
            score -= chosen[key]
            if score > best_score:
                best_key = key
                best_score = score
        if best_key is None:
            break
        chosen[best_key] += 1
        for label in labels_by_key[best_key]:
            current[label] += 1
    return {key: int(value) for key, value in chosen.items() if int(value) > 0}


def _make_all_item(row: FrameRow, focus: str, *, seed: int) -> WorkItem:
    """构造 Phase2 all_random_order 候选。"""

    spec = make_prompt_spec(
        variant="all_random_order",
        answers=row.answers,
        seed_key=_work_item_seed(row, seed, "all", focus),
        focus=focus,
    )
    return WorkItem(
        row=row,
        focus=focus,
        spec=spec,
        balance_key=_focus_key(row, focus),
        augment_balance_key=f"all_random_order/{focus}:{_answer_text(row.answers[focus])}",
    )


def _make_subset_item(row: FrameRow, focus: str, count: int, *, seed: int) -> WorkItem:
    """构造 Phase2 subset_random 候选。"""

    spec = make_prompt_spec(
        variant="subset_random",
        answers=row.answers,
        seed_key=_work_item_seed(row, seed, "subset", focus, count),
        focus=focus,
        subset_count=int(count),
    )
    items = ",".join(f"{q.question_id}:{_answer_text(q.answer)}" for q in spec.phase2_spec.questions)
    return WorkItem(
        row=row,
        focus=focus,
        spec=spec,
        balance_key=_focus_key(row, focus),
        augment_balance_key=f"subset_random/q{int(count)}/items/{items}",
    )


def _make_hier_item(row: FrameRow, group_id: str, detail_key: str, *, seed: int) -> WorkItem:
    """构造 Phase2 hierarchical_probe 候选。"""

    spec = make_prompt_spec(
        variant="hierarchical_probe",
        answers=row.answers,
        seed_key=_work_item_seed(row, seed, "hier", group_id, detail_key),
        group_id=group_id,
        detail_key=detail_key,
    )
    answers = {q.output_key: bool(q.answer) for q in spec.phase2_spec.questions}
    return WorkItem(
        row=row,
        focus=detail_key,
        spec=spec,
        balance_key=_focus_key(row, detail_key),
        augment_balance_key=(
            f"hierarchical_probe/highway:{_answer_text(answers['HIGHWAY'])}"
            f"/group/{group_id}:{_answer_text(answers['GROUP'])}"
            f"/detail/{detail_key}:{_answer_text(answers['DETAIL'])}"
        ),
    )


def _augment_target_counts(total: int, weights: Mapping[str, int]) -> Dict[str, int]:
    """把 Phase2 半边总样本数按 variant 权重拆成整数配额。"""

    total = max(0, int(total))
    weight_items = [(key, max(0, int(weights.get(key, 0)))) for key in VARIANT_ORDER]
    weight_sum = sum(weight for _, weight in weight_items)
    if weight_sum <= 0:
        raise ValueError("empty augment variant weights")
    raw = [(key, total * weight / weight_sum) for key, weight in weight_items]
    counts = {key: int(value) for key, value in raw}
    remainder = total - sum(counts.values())
    for key, _ in sorted(raw, key=lambda item: item[1] - int(item[1]), reverse=True)[:remainder]:
        counts[key] += 1
    return counts


def _candidate_specs_for_row(row: FrameRow, *, seed: int, focus: str) -> Iterable[WorkItem]:
    """为一个已选 Phase1-focus row 生成所有可用 Phase2 augment spec 候选。"""

    phase2_focus_keys = (focus,) if focus in PHASE2_ANSWER_KEYS else PHASE2_ANSWER_KEYS
    for phase2_focus in phase2_focus_keys:
        item = _make_all_item(row, phase2_focus, seed=seed)
        yield WorkItem(row=row, focus=focus, spec=item.spec, balance_key=_focus_key(row, focus), augment_balance_key=item.augment_balance_key)
        for count in SUBSET_COUNTS:
            item = _make_subset_item(row, phase2_focus, int(count), seed=seed)
            yield WorkItem(row=row, focus=focus, spec=item.spec, balance_key=_focus_key(row, focus), augment_balance_key=item.augment_balance_key)
    for group_id in GROUP_DEFINITIONS:
        for detail_key in phase2_focus_keys:
            item = _make_hier_item(row, group_id, detail_key, seed=seed)
            yield WorkItem(row=row, focus=focus, spec=item.spec, balance_key=_focus_key(row, focus), augment_balance_key=item.augment_balance_key)


def _augment_key_targets(total_items: int, variant_weights: Mapping[str, int]) -> Dict[str, int]:
    """返回给定样本总数下的 Phase2 augment balance-key 目标。"""

    variant_targets = _augment_target_counts(int(total_items), variant_weights)
    keys_by_variant = {
        "all_random_order": _expected_all_augment_keys(),
        "subset_random": _expected_subset_augment_keys(),
        "hierarchical_probe": _expected_hierarchical_augment_keys(),
    }
    per_key_targets: Dict[str, int] = {}
    for variant, total in variant_targets.items():
        keys = list(keys_by_variant.get(variant, []))
        if not keys or int(total) <= 0:
            continue
        if variant == "subset_random":
            per_key_targets.update(_subset_key_targets(keys, int(total)))
            continue
        if variant == "hierarchical_probe":
            per_key_targets.update(_hierarchical_key_targets(keys, int(total)))
            continue
        base = int(total) // len(keys)
        remainder = int(total) - base * len(keys)
        for idx, key in enumerate(keys):
            value = base + int(idx < remainder)
            if value > 0:
                per_key_targets[key] = value
    return per_key_targets


def _balance_key_variant_targets(
    pairs: Sequence[Tuple[FrameRow, str]],
    *,
    variant_weights: Mapping[str, int],
    seed: int,
) -> Dict[Tuple[str, str], int]:
    """Distribute variant quotas across main focus buckets without losing totals."""

    balance_counts = Counter(_focus_key(row, focus) for row, focus in pairs)
    total = sum(balance_counts.values())
    if total <= 0:
        return {}
    variant_totals = _augment_target_counts(total, variant_weights)
    quotas: Dict[Tuple[str, str], int] = {}
    balance_keys = sorted(balance_counts)
    for variant, variant_total in variant_totals.items():
        floors: Dict[str, int] = {}
        remainders: List[Tuple[float, str, str]] = []
        floor_sum = 0
        for balance_key in balance_keys:
            raw = float(variant_total) * float(balance_counts[balance_key]) / float(total)
            floor = int(math.floor(raw))
            floors[balance_key] = floor
            floor_sum += floor
            jitter = hashlib.sha256(f"{seed}:{variant}:{balance_key}".encode("utf-8")).hexdigest()
            remainders.append((raw - floor, jitter, balance_key))
        for _, _, balance_key in sorted(remainders, reverse=True)[: max(0, int(variant_total) - floor_sum)]:
            floors[balance_key] += 1
        for balance_key, count in floors.items():
            if count > 0:
                quotas[(balance_key, variant)] = int(count)
    return quotas


def _frame_key(row: FrameRow) -> Tuple[str, str, int]:
    """Return a stable frame identity shared by sampling/repeat audits."""

    return row.scenario, row.route_id, int(row.frame_id)


def _exact_all_random_assignments(
    pairs: Sequence[Tuple[FrameRow, str]],
    *,
    targets: Mapping[str, int],
    seed: int,
) -> Tuple[List[WorkItem], set[int]]:
    """Assign the all-random slots exactly with a small RS-to-question max flow."""

    all_targets = {
        str(key): int(value)
        for key, value in targets.items()
        if str(key).startswith("all_random_order/") and int(value) > 0
    }
    if not all_targets:
        return [], set()
    slots_by_rs: Dict[str, List[int]] = defaultdict(list)
    for slot, (row, _) in enumerate(pairs):
        slots_by_rs[row.rs].append(slot)
    rng = random.Random(f"{seed}:exact_all_random:{len(pairs)}")
    for slots in slots_by_rs.values():
        rng.shuffle(slots)

    selected: List[WorkItem] = []
    used_slots: set[int] = set()

    def add_slot(slot: int, question: str, expected_key: str) -> None:
        row, main_focus = pairs[slot]
        candidate = _make_all_item(row, question, seed=seed + slot)
        item = WorkItem(
            row=row,
            focus=main_focus,
            spec=candidate.spec,
            balance_key=_focus_key(row, main_focus),
            augment_balance_key=candidate.augment_balance_key,
        )
        if item.augment_balance_key != expected_key:
            raise RuntimeError(
                f"all-random assignment mismatch: expected={expected_key} actual={item.augment_balance_key}"
            )
        selected.append(item)
        used_slots.add(slot)

    # YES is possible only on the matching one-hot RS state, so reserve it first.
    for question in PHASE2_ANSWER_KEYS:
        rs = question.replace("RS", "R")
        key = f"all_random_order/{question}:YES"
        need = int(all_targets.get(key, 0))
        available = [slot for slot in slots_by_rs.get(rs, []) if slot not in used_slots]
        if len(available) < need:
            raise RuntimeError(
                f"insufficient Phase1 rows for exact {key}: need={need} available={len(available)}; "
                "repair Phase1 secondary RS coverage without cycling rare per-focus subgroups"
            )
        for slot in available[:need]:
            add_slot(slot, question, key)

    # Allocate the remaining NO quotas with a tiny integral max-flow problem.
    remaining_by_rs = {
        rs: [slot for slot in slots if slot not in used_slots]
        for rs, slots in slots_by_rs.items()
    }
    no_demands = {
        question: int(all_targets.get(f"all_random_order/{question}:NO", 0))
        for question in PHASE2_ANSWER_KEYS
    }
    state_names = sorted(remaining_by_rs)
    question_names = list(PHASE2_ANSWER_KEYS)
    source = 0
    state_offset = 1
    question_offset = state_offset + len(state_names)
    sink = question_offset + len(question_names)
    graph: List[List[List[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(src: int, dst: int, capacity: int) -> List[int]:
        forward = [dst, len(graph[dst]), int(capacity)]
        reverse = [src, len(graph[src]), 0]
        graph[src].append(forward)
        graph[dst].append(reverse)
        return forward

    flow_edges: Dict[Tuple[str, str], List[int]] = {}
    for state_idx, rs in enumerate(state_names):
        add_edge(source, state_offset + state_idx, len(remaining_by_rs[rs]))
        for question_idx, question in enumerate(question_names):
            if rs == question.replace("RS", "R"):
                continue
            flow_edges[(rs, question)] = add_edge(
                state_offset + state_idx,
                question_offset + question_idx,
                len(remaining_by_rs[rs]),
            )
    for question_idx, question in enumerate(question_names):
        add_edge(question_offset + question_idx, sink, no_demands[question])

    total_demand = sum(no_demands.values())
    total_flow = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for dst, _, capacity in graph[node]:
                if capacity > 0 and level[dst] < 0:
                    level[dst] = level[node] + 1
                    queue.append(dst)
        if level[sink] < 0:
            break
        cursor = [0] * len(graph)

        def send(node: int, pushed: int) -> int:
            if node == sink:
                return pushed
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                dst, reverse_idx, capacity = edge
                if capacity > 0 and level[dst] == level[node] + 1:
                    amount = send(dst, min(pushed, capacity))
                    if amount > 0:
                        edge[2] -= amount
                        graph[dst][reverse_idx][2] += amount
                        return amount
                cursor[node] += 1
            return 0

        while True:
            pushed = send(source, 10**9)
            if pushed <= 0:
                break
            total_flow += pushed
    if total_flow != total_demand:
        supply = {rs: len(slots) for rs, slots in remaining_by_rs.items()}
        raise RuntimeError(
            f"cannot satisfy exact all-random NO quotas: flow={total_flow} demand={total_demand} "
            f"supply={supply}"
        )
    for rs in state_names:
        slots = list(remaining_by_rs[rs])
        cursor = 0
        for question in question_names:
            edge = flow_edges.get((rs, question))
            if edge is None:
                continue
            # The reverse residual capacity equals the realized flow.
            dst, reverse_idx, _ = edge
            count = int(graph[dst][reverse_idx][2])
            key = f"all_random_order/{question}:NO"
            for slot in slots[cursor : cursor + count]:
                add_slot(slot, question, key)
            cursor += count
    if len(selected) != sum(all_targets.values()):
        raise RuntimeError(
            f"exact all-random assignment count mismatch: selected={len(selected)} target={sum(all_targets.values())}"
        )
    return selected, used_slots


def _assign_specs_to_focus_pairs(
    pairs: Sequence[Tuple[FrameRow, str]],
    *,
    seed: int,
    variant_weights: Mapping[str, int],
    hard_focus_variant: bool = True,
) -> List[WorkItem]:
    """为已均衡的 focus pairs 选择 spec，并压低 Phase2 augment-key 偏差。"""

    targets = Counter(_augment_key_targets(len(pairs), variant_weights))
    variant_targets = Counter(_augment_target_counts(len(pairs), variant_weights))
    balance_variant_targets = Counter()
    if hard_focus_variant:
        balance_variant_targets = Counter(
            {
                f"{balance_key}|{variant}": count
                for (balance_key, variant), count in _balance_key_variant_targets(
                    pairs,
                    variant_weights=variant_weights,
                    seed=seed,
                ).items()
            }
        )
    key_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    balance_variant_counts: Counter[str] = Counter()
    out: List[WorkItem] = []
    ordered_pairs = list(pairs)
    used_slots: set[int] = set()
    if not hard_focus_variant:
        all_items, used_slots = _exact_all_random_assignments(
            ordered_pairs,
            targets=targets,
            seed=seed,
        )
        out.extend(all_items)
        for item in all_items:
            key_counts[item.augment_balance_key] += 1
            variant_counts[item.spec.variant] += 1
            balance_variant_counts[f"{item.balance_key}|{item.spec.variant}"] += 1
    for slot, (row, focus) in enumerate(ordered_pairs):
        if slot in used_slots:
            continue
        candidates = list(_candidate_specs_for_row(row, seed=seed + slot, focus=focus))
        if not candidates:
            raise RuntimeError(f"no augment candidates for focus={focus} row={row.scenario}/{row.route_id}/f{row.frame_id}")
        best: Optional[WorkItem] = None
        best_score = -10**18
        for candidate in candidates:
            variant = candidate.spec.variant
            variant_deficit = variant_targets[variant] - variant_counts[variant]
            if variant_deficit <= 0:
                continue
            balance_variant_key = f"{candidate.balance_key}|{variant}"
            balance_variant_deficit = 0
            if hard_focus_variant:
                balance_variant_deficit = balance_variant_targets[balance_variant_key] - balance_variant_counts[balance_variant_key]
                if balance_variant_deficit <= 0:
                    continue
            key = candidate.augment_balance_key
            key_deficit = targets[key] - key_counts[key]
            score = (
                1_000.0 * float(variant_deficit) / max(1.0, float(variant_targets[variant]))
                + 200.0 * float(key_deficit) / max(1.0, float(targets[key]))
                + float(targets[key])
            )
            if hard_focus_variant:
                score += 20_000.0 * float(balance_variant_deficit) / max(1.0, float(balance_variant_targets[balance_variant_key]))
            if key_deficit <= 0:
                score += 500.0 * float(key_deficit)
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            best = max(
                candidates,
                key=lambda candidate: (
                    (
                        balance_variant_targets[f"{candidate.balance_key}|{candidate.spec.variant}"]
                        - balance_variant_counts[f"{candidate.balance_key}|{candidate.spec.variant}"]
                    )
                    if hard_focus_variant
                    else 0,
                    targets[candidate.augment_balance_key] - key_counts[candidate.augment_balance_key],
                    variant_targets[candidate.spec.variant] - variant_counts[candidate.spec.variant],
                    candidate.augment_balance_key,
                ),
            )
        out.append(best)
        key_counts[best.augment_balance_key] += 1
        variant_counts[best.spec.variant] += 1
        balance_variant_counts[f"{best.balance_key}|{best.spec.variant}"] += 1
    if hard_focus_variant:
        unmet = {
            key: int(target) - int(balance_variant_counts.get(key, 0))
            for key, target in balance_variant_targets.items()
            if int(balance_variant_counts.get(key, 0)) != int(target)
        }
        if unmet:
            raise RuntimeError(f"failed to satisfy focus/label/variant quotas: {unmet}")
    return out


def _balanced_focus_pairs(
    rows: Sequence[FrameRow],
    *,
    keys: Sequence[str],
    target: int,
    seed: int,
    context: str,
    secondary_rs_minimums: Optional[Mapping[str, int]] = None,
) -> List[Tuple[FrameRow, str]]:
    """按 focus 抽取严格等量 pairs，并按全局容量修复必要的二级 RS 覆盖。"""

    groups: Dict[str, List[Tuple[FrameRow, str]]] = {
        f"{key}:{value}": [] for key in keys for value in ("YES", "NO")
    }
    for row in rows:
        for focus in keys:
            groups[_focus_key(row, focus)].append((row, focus))
    raw_counts = {key: len(items) for key, items in groups.items()}
    missing = [key for key in groups if raw_counts[key] == 0]
    if missing:
        raise ValueError(
            f"cannot build exact 1:1 {context}: required focus bins are empty; "
            f"missing={missing} raw_counts={raw_counts}. Rebuild/check the dataset split or reduce filtering."
        )
    rng = random.Random(f"{seed}:{context}:{len(rows)}:{target}")
    selected_by_group: Dict[str, List[Tuple[FrameRow, str]]] = {}
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        if len(items) >= int(target):
            selected = items[: int(target)]
        else:
            selected = [items[idx % len(items)] for idx in range(int(target))]
            rng.shuffle(selected)
        selected_by_group[key] = selected

    if secondary_rs_minimums:
        rs_counts = Counter(item[0].rs for selected in selected_by_group.values() for item in selected)
        frame_counts = Counter(_frame_key(item[0]) for selected in selected_by_group.values() for item in selected)
        for desired_rs, raw_minimum in sorted(secondary_rs_minimums.items()):
            minimum = max(0, int(raw_minimum))
            deficit = minimum - int(rs_counts[desired_rs])
            if deficit <= 0:
                continue
            group_order = sorted(
                groups,
                key=lambda group_key: sum(1 for item in groups[group_key] if item[0].rs == desired_rs),
                reverse=True,
            )
            for group_key in group_order:
                if deficit <= 0:
                    break
                selected = selected_by_group[group_key]
                selected_frames = {_frame_key(item[0]) for item in selected}
                extras = [
                    item
                    for item in groups[group_key]
                    if item[0].rs == desired_rs and _frame_key(item[0]) not in selected_frames
                ]
                extras.sort(
                    key=lambda item: (
                        frame_counts[_frame_key(item[0])],
                        hashlib.sha256(
                            f"{seed}:{group_key}:{desired_rs}:{_frame_key(item[0])}".encode("utf-8")
                        ).hexdigest(),
                    )
                )
                donor_indices = [
                    idx
                    for idx, item in enumerate(selected)
                    if item[0].rs != desired_rs
                    and int(rs_counts[item[0].rs]) > int(secondary_rs_minimums.get(item[0].rs, 0))
                ]
                donor_indices.sort(
                    key=lambda idx: (
                        -(int(rs_counts[selected[idx][0].rs]) - int(secondary_rs_minimums.get(selected[idx][0].rs, 0))),
                        -frame_counts[_frame_key(selected[idx][0])],
                    )
                )
                extra_idx = 0
                for donor_idx in donor_indices:
                    if deficit <= 0 or extra_idx >= len(extras):
                        break
                    donor = selected[donor_idx]
                    donor_rs = donor[0].rs
                    donor_minimum = int(secondary_rs_minimums.get(donor_rs, 0))
                    if donor_rs == desired_rs or int(rs_counts[donor_rs]) <= donor_minimum:
                        continue
                    extra = extras[extra_idx]
                    extra_idx += 1
                    donor_frame = _frame_key(donor[0])
                    extra_frame = _frame_key(extra[0])
                    selected[donor_idx] = extra
                    rs_counts[donor_rs] -= 1
                    rs_counts[desired_rs] += 1
                    frame_counts[donor_frame] -= 1
                    frame_counts[extra_frame] += 1
                    deficit -= 1
            if deficit > 0:
                raise RuntimeError(
                    f"cannot repair global Phase1 RS coverage for {desired_rs}: "
                    f"minimum={minimum} actual={rs_counts[desired_rs]} without cycling rare subgroups"
                )

    pairs = [item for key in sorted(selected_by_group) for item in selected_by_group[key]]
    rng.shuffle(pairs)
    return pairs


def _repeat_report(work: Sequence[WorkItem]) -> Dict[str, Any]:
    """统计均衡重采样的重复率，避免大 target 静默过度复用稀缺正样本。"""

    counts: Counter[str] = Counter(
        f"{item.row.scenario}/{item.row.route_id}/f{item.row.frame_id}" for item in work
    )
    total = len(work)
    unique = len(counts)
    return {
        "total_cases": int(total),
        "unique_frames": int(unique),
        "mean_repeat": float(total) / max(1.0, float(unique)),
        "max_repeat": int(max(counts.values(), default=0)),
        "frames_repeated": int(sum(1 for value in counts.values() if int(value) > 1)),
        "top_repeated": [
            {"frame": key, "count": int(value)}
            for key, value in counts.most_common(20)
        ],
    }


def _assert_repeat_limit(work: Sequence[WorkItem], *, max_repeat: int, context: str) -> None:
    """Refuse a sampled epoch when a small secondary subgroup is being memorized."""

    if int(max_repeat) <= 0:
        return
    report = _repeat_report(work)
    if int(report["max_repeat"]) > int(max_repeat):
        raise RuntimeError(
            f"{context} exceeds --max-train-frame-repeat={int(max_repeat)}: "
            f"max_repeat={report['max_repeat']} mean_repeat={report['mean_repeat']:.4f} "
            f"top_repeated={report['top_repeated'][:10]}"
        )


def _counter_dict(counter: Mapping[str, int]) -> Dict[str, int]:
    """稳定排序 counter，方便 JSON 审计 diff。"""

    return {str(key): int(counter[key]) for key in sorted(counter)}


def _target_deviation_report(actual: Mapping[str, int], targets: Mapping[str, int]) -> Dict[str, Any]:
    """对比实际计数与目标计数，返回偏差摘要和 top-k 明细。"""

    keys = sorted(set(actual) | set(targets))
    rows = [
        {
            "key": str(key),
            "actual": int(actual.get(key, 0)),
            "target": int(targets.get(key, 0)),
            "delta": int(actual.get(key, 0)) - int(targets.get(key, 0)),
        }
        for key in keys
    ]
    off = [row for row in rows if int(row["delta"]) != 0]
    return {
        "exact": not off,
        "keys": int(len(keys)),
        "off_target_keys": int(len(off)),
        "max_abs_delta": int(max((abs(int(row["delta"])) for row in rows), default=0)),
        "total_abs_delta": int(sum(abs(int(row["delta"])) for row in rows)),
        "underfull_top": sorted(off, key=lambda row: (int(row["delta"]), row["key"]))[:20],
        "overfull_top": sorted(off, key=lambda row: (-int(row["delta"]), row["key"]))[:20],
    }


def _work_balance_report(
    work: Sequence[WorkItem],
    *,
    rows: Sequence[FrameRow],
    split: str,
    focus_balance_count: int,
    seed: int,
    variant_weights: Mapping[str, int],
    world_size: int,
    rank_work: Optional[Sequence[WorkItem]] = None,
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> Dict[str, Any]:
    """生成一次 sampled work 的完整均衡审计。"""

    augment_actual = Counter(item.augment_balance_key for item in work)
    phase_counts = Counter(focus_phase(item.focus) for item in work)
    augment_targets: Counter[str] = Counter()
    variant_targets: Counter[str] = Counter()
    for phase in sorted(phase_counts):
        count = int(phase_counts[phase])
        augment_targets.update(_augment_key_targets(count, variant_weights))
        variant_targets.update(_augment_target_counts(count, variant_weights))
    variant_actual = Counter(item.spec.variant for item in work)
    all_random_actual = Counter(
        {key: value for key, value in augment_actual.items() if str(key).startswith("all_random_order/")}
    )
    all_random_targets = Counter(
        {key: value for key, value in augment_targets.items() if str(key).startswith("all_random_order/")}
    )
    focus_variant_actual = Counter(f"{item.balance_key}|{item.spec.variant}" for item in work)
    focus_variant_targets = Counter(
        {
            f"{balance_key}|{variant}": count
            for (balance_key, variant), count in _balance_key_variant_targets(
                [(item.row, item.focus) for item in work],
                variant_weights=variant_weights,
                seed=seed,
            ).items()
        }
    )
    phase2_items = [item for item in work if item.focus in PHASE2_ANSWER_KEYS]
    phase2_focus_variant_actual = Counter(f"{item.balance_key}|{item.spec.variant}" for item in phase2_items)
    phase2_focus_variant_targets = Counter(
        {
            f"{balance_key}|{variant}": count
            for (balance_key, variant), count in _balance_key_variant_targets(
                [(item.row, item.focus) for item in phase2_items],
                variant_weights=variant_weights,
                seed=seed,
            ).items()
        }
    )
    emitted_answer_counts: Counter[str] = Counter()
    semantic_answer_counts: Counter[str] = Counter()
    semantic_answer_base_mass: Counter[str] = Counter()
    for item in work:
        base_weights = _semantic_base_weights(item, non_focus_semantic_loss_weight)
        for output_key, metric_key, answer in spec_metric_items(item.spec):
            label = f"{metric_key}:{_answer_text(answer)}"
            emitted_answer_counts[label] += 1
            if output_key in base_weights:
                semantic_answer_counts[label] += 1
                semantic_answer_base_mass[label] += float(base_weights[output_key])
    payload: Dict[str, Any] = {
        "split": str(split),
        "seed": int(seed),
        "focus_balance_count": int(focus_balance_count),
        "world_size": int(world_size),
        "raw_available": _raw_focus_bin_counts(rows),
        "global_sampled": _counter_dict(Counter(item.balance_key for item in work)),
        "augment_global_sampled": _counter_dict(augment_actual),
        "augment_target_sampled": _counter_dict(augment_targets),
        "augment_target_deviation": _target_deviation_report(augment_actual, augment_targets),
        "all_random_order_target_deviation": _target_deviation_report(all_random_actual, all_random_targets),
        "variant_sampled": _counter_dict(variant_actual),
        "variant_target_sampled": _counter_dict(variant_targets),
        "variant_target_deviation": _target_deviation_report(variant_actual, variant_targets),
        "focus_variant_sampled": _counter_dict(focus_variant_actual),
        "focus_variant_target_sampled": _counter_dict(focus_variant_targets),
        "focus_variant_target_deviation": _target_deviation_report(focus_variant_actual, focus_variant_targets),
        "phase2_focus_variant_sampled": _counter_dict(phase2_focus_variant_actual),
        "phase2_focus_variant_target_sampled": _counter_dict(phase2_focus_variant_targets),
        "phase2_focus_variant_target_deviation": _target_deviation_report(phase2_focus_variant_actual, phase2_focus_variant_targets),
        "phase_group_sampled": _counter_dict(Counter(focus_phase(item.focus) for item in work)),
        "phase_group_answer_sampled": _counter_dict(
            Counter(f"{focus_phase(item.focus)}:{'YES' if item.row.answers[item.focus] else 'NO'}" for item in work)
        ),
        "semantic_supervision": SEMANTIC_SUPERVISION,
        "non_focus_semantic_loss_weight": float(non_focus_semantic_loss_weight),
        "emitted_answer_counts": _counter_dict(emitted_answer_counts),
        "semantic_answer_counts": _counter_dict(semantic_answer_counts),
        "semantic_answer_base_mass": {
            key: float(value) for key, value in sorted(semantic_answer_base_mass.items())
        },
        "semantic_class_weights": {
            key: float(value)
            for key, value in sorted(
                _semantic_class_weights(work, non_focus_semantic_loss_weight).items()
            )
        },
        "phase1_output_ordering": "deterministic_random_per_case",
        "phase1_output_order_sampled": _counter_dict(
            Counter("|".join(item.spec.phase1_output_keys) for item in work)
        ),
        "repeat_audit": _repeat_report(work),
    }
    if rank_work is not None:
        payload["rank0_shard"] = _counter_dict(Counter(item.balance_key for item in rank_work))
        payload["rank0_augment_shard"] = _counter_dict(Counter(item.augment_balance_key for item in rank_work))
    return payload


def _write_epoch_balance_report(
    output_dir: pathlib.Path,
    *,
    epoch: int,
    work: Sequence[WorkItem],
    rows: Sequence[FrameRow],
    split: str,
    focus_balance_count: int,
    seed: int,
    variant_weights: Mapping[str, int],
    world_size: int,
    non_focus_semantic_loss_weight: float = DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
) -> pathlib.Path:
    """每个重采样 epoch 写一份完整 train balance 审计。"""

    report = _work_balance_report(
        work,
        rows=rows,
        split=split,
        focus_balance_count=focus_balance_count,
        seed=seed,
        variant_weights=variant_weights,
        world_size=world_size,
        non_focus_semantic_loss_weight=non_focus_semantic_loss_weight,
    )
    report["epoch"] = int(epoch)
    balance_dir = output_dir / "balance"
    balance_dir.mkdir(parents=True, exist_ok=True)
    path = balance_dir / f"epoch_{int(epoch):03d}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _append_jsonl(
        balance_dir / "epochs.jsonl",
        {
            "epoch": int(epoch),
            "path": str(path.relative_to(output_dir)),
            "seed": int(seed),
            "cases": int(len(work)),
            "augment_exact": bool(report["augment_target_deviation"]["exact"]),
            "augment_off_target_keys": int(report["augment_target_deviation"]["off_target_keys"]),
            "augment_max_abs_delta": int(report["augment_target_deviation"]["max_abs_delta"]),
            "all_random_exact": bool(report["all_random_order_target_deviation"]["exact"]),
            "all_random_max_abs_delta": int(report["all_random_order_target_deviation"]["max_abs_delta"]),
            "focus_variant_exact": bool(report["focus_variant_target_deviation"]["exact"]),
            "focus_variant_max_abs_delta": int(report["focus_variant_target_deviation"]["max_abs_delta"]),
            "phase2_focus_variant_exact": bool(report["phase2_focus_variant_target_deviation"]["exact"]),
            "phase2_focus_variant_max_abs_delta": int(report["phase2_focus_variant_target_deviation"]["max_abs_delta"]),
            "max_repeat": int(report["repeat_audit"]["max_repeat"]),
            "mean_repeat": float(report["repeat_audit"]["mean_repeat"]),
        },
    )
    return path


def _balanced_work(
    rows: Sequence[FrameRow],
    *,
    target_per_bin: int,
    seed: int,
    variant_weights: Mapping[str, int],
) -> List[WorkItem]:
    """构建 Phase1 focus 半边 + Phase2 augment 半边的融合 work list。"""

    focus_counts = _raw_focus_bin_counts(rows)
    relevant_counts = [focus_counts[f"{key}:{value}"] for key in ANSWER_KEYS for value in ("YES", "NO")]
    target = int(target_per_bin) if int(target_per_bin) > 0 else min(relevant_counts)
    target = max(1, target)
    rng = random.Random(f"{seed}:phase1_balance:{len(rows)}:{target}")
    phase1_total = len(PHASE1_ANSWER_KEYS) * 2 * target
    phase1_augment_targets = _augment_key_targets(phase1_total, variant_weights)
    secondary_rs_minimums = {
        key.replace("all_random_order/RS", "R").removesuffix(":YES"): int(value)
        for key, value in phase1_augment_targets.items()
        if key.startswith("all_random_order/RS") and key.endswith(":YES")
    }
    phase1_pairs = _balanced_focus_pairs(
        rows,
        keys=PHASE1_ANSWER_KEYS,
        target=target,
        seed=seed,
        context="Phase1-focus training work",
        secondary_rs_minimums=secondary_rs_minimums,
    )
    phase1_work = _assign_specs_to_focus_pairs(
        phase1_pairs,
        seed=seed,
        variant_weights=variant_weights,
        hard_focus_variant=False,
    )
    _assert_exact_focus_balance(
        phase1_work,
        target=target,
        context="Phase1-focus training work",
        keys=PHASE1_ANSWER_KEYS,
    )
    phase2_pairs = _balanced_focus_pairs(
        rows,
        keys=PHASE2_ANSWER_KEYS,
        target=target,
        seed=seed + 17,
        context="Phase2-focus training work",
    )
    phase2_work = _assign_specs_to_focus_pairs(phase2_pairs, seed=seed + 31, variant_weights=variant_weights)
    _assert_exact_focus_balance(
        phase2_work,
        target=target,
        context="Phase2-focus training work",
        keys=PHASE2_ANSWER_KEYS,
    )
    if not phase2_work:
        raise ValueError("cannot build Phase2 focus-balanced work; no active Phase2 buckets were sampled")
    work = [*phase1_work, *phase2_work]
    variant_actual = Counter(item.spec.variant for item in work)
    variant_target = Counter()
    all_actual = Counter()
    all_target = Counter()
    for phase_work in (phase1_work, phase2_work):
        variant_target.update(_augment_target_counts(len(phase_work), variant_weights))
        phase_targets = _augment_key_targets(len(phase_work), variant_weights)
        all_target.update({key: value for key, value in phase_targets.items() if key.startswith("all_random_order/")})
    all_actual.update(
        item.augment_balance_key
        for item in work
        if item.augment_balance_key.startswith("all_random_order/")
    )
    if variant_actual != variant_target:
        raise RuntimeError(f"combined work lost exact variant quotas: actual={variant_actual} target={variant_target}")
    if all_actual != all_target:
        raise RuntimeError(f"combined work lost exact all-random quotas: actual={all_actual} target={all_target}")
    rng.shuffle(work)
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
    output_keys: Sequence[str],
    semantic_output_keys: Optional[Sequence[str]],
    semantic_output_weights: Optional[Mapping[str, float]],
    format_loss_weight: float,
) -> Tuple[List[int], List[float], List[int]]:
    """映射当前 spec 输出行的语义与格式 token 权重。

    ``semantic_output_keys=None`` 表示评估所有答案值。训练只传当前
    focus 与派生层级键，非 focus 主任务的值 token 权重为零。
    """

    enc = bundle.tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    weights = [float(format_loss_weight) for _ in token_ids]
    component_ids = [FORMAT_COMPONENT_ID for _ in token_ids]
    semantic = set(output_keys if semantic_output_keys is None else semantic_output_keys)
    unknown = semantic.difference(output_keys)
    if unknown:
        raise ValueError(f"semantic output keys are absent from target: {sorted(unknown)}")
    value_weights = {key: 1.0 for key in semantic}
    if semantic_output_weights is not None:
        unknown_weights = set(semantic_output_weights).difference(semantic)
        if unknown_weights:
            raise ValueError(f"semantic weights supplied for disabled keys: {sorted(unknown_weights)}")
        value_weights.update({key: float(value) for key, value in semantic_output_weights.items()})
    for component_id, key in enumerate(output_keys, start=1):
        lo, hi = _line_value_span(target, key)
        for i, (a, b) in enumerate(offsets):
            if a < hi and b > lo:
                weights[i] = float(value_weights[key]) if key in semantic else 0.0
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
    output_keys: Sequence[str],
    semantic_output_keys: Optional[Sequence[str]],
    semantic_output_weights: Optional[Mapping[str, float]],
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
        output_keys=output_keys,
        semantic_output_keys=semantic_output_keys,
        semantic_output_weights=semantic_output_weights,
        format_loss_weight=float(format_loss_weight),
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
        "output_keys": list(output_keys),
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
    for component_id, key in enumerate(packed.get("output_keys", []), start=1):
        mask = active & shift_comp.eq(component_id)
        stats[f"{key.lower()}_ok"] = float(bool(mask.any() and torch.equal(pred[mask], shift_labels[mask])))
    return loss, stats


def _split_work_for_rank(work: Sequence[WorkItem], *, rank: int, world_size: int) -> List[WorkItem]:
    """按 rank 切分同一个均衡全集，并 padding 到每个 rank 等长。"""

    if int(world_size) <= 1:
        return list(work)
    items = list(work)
    if not items:
        raise ValueError("empty work list")
    padded_len = int(math.ceil(len(items) / float(world_size))) * int(world_size)
    if len(items) < padded_len:
        items = [*items, *[items[idx % len(items)] for idx in range(padded_len - len(items))]]
    shard = items[int(rank) :: int(world_size)]
    if not shard:
        raise ValueError(f"rank {rank} got empty work shard; reduce WORLD_SIZE or increase balance count")
    return shard


def _dummy_ddp_forward_zero_loss(
    bundle: Any,
    *,
    image: Image.Image,
    max_length: int,
    format_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """超长样本跳过时跑一个短图文 DDP forward，并把 logits loss 归零。"""

    prompt = (
        "[OUTPUT]\n"
        "Output exactly this line and nothing else:\n"
        "HIGHWAY: <YES or NO>\n"
        "[/OUTPUT]"
    )
    target = "HIGHWAY: NO"
    packed = _build_inputs(
        bundle,
        images=[image],
        prompt=prompt,
        target=target,
        output_keys=["HIGHWAY"],
        semantic_output_keys=["HIGHWAY"],
        semantic_output_weights=None,
        max_length=max(int(max_length), 128),
        format_loss_weight=float(format_loss_weight),
    )
    if packed is None:
        raise RuntimeError(
            "dummy DDP forward also exceeded max_length; increase --max-length so skipped samples can still synchronize DDP"
        )
    loss, _ = _loss_one(bundle, packed)
    zero = loss * 0.0
    return zero, {"denom": 0.0, "token_acc": 0.0, "value_token_acc": 0.0, "format_token_acc": 0.0}


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
    """在独立 val split 上跑 teacher-forced loss 和八问 token accuracy。"""

    was_training = bool(bundle.model.training)
    bundle.model.eval()
    loss_sum = 0.0
    samples = 0
    skipped = 0
    token_acc_sum = 0.0
    value_token_acc_sum = 0.0
    format_token_acc_sum = 0.0
    metric_names = _metric_names()
    metric_ok = {key: 0.0 for key in metric_names}
    metric_count = {key: 0.0 for key in metric_names}
    focus_ok = {key: 0.0 for key in ANSWER_KEYS}
    focus_count = {key: 0.0 for key in ANSWER_KEYS}
    for item in work:
        row, focus, spec = item.row, item.focus, item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        prompt = build_phase1_prompt(spec=spec, audit=False, history_rgb_mode=history_rgb_mode)
        target = build_phase1_target(row.answers, spec=spec)
        packed = _build_inputs(
            bundle,
            images=images,
            prompt=prompt,
            target=target,
            output_keys=spec.output_keys,
            semantic_output_keys=None,
            semantic_output_weights=None,
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
        stats_by_output = {key: float(stats.get(f"{key.lower()}_ok", 0.0)) for key in spec.output_keys}
        for output_key, metric_key, _ in spec_metric_items(spec):
            if metric_key in metric_ok:
                metric_ok[metric_key] += float(stats_by_output.get(output_key, 0.0))
                metric_count[metric_key] += 1.0
        if focus in spec.output_keys:
            focus_ok[focus] += float(stats_by_output.get(focus, 0.0))
        focus_count[focus] += 1.0
    values = [loss_sum, float(samples), float(skipped), token_acc_sum, value_token_acc_sum, format_token_acc_sum]
    values.extend(metric_ok[key] for key in metric_names)
    values.extend(metric_count[key] for key in metric_names)
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
    for idx, key in enumerate(metric_names):
        safe = key.lower().replace(":", "_")
        denom = max(1.0, vals[offset + len(metric_names) + idx])
        metrics[f"metric/{safe}_acc"] = vals[offset + idx] / denom
        metrics[f"metric/{safe}_samples"] = vals[offset + len(metric_names) + idx]
    offset += len(metric_names)
    offset += len(metric_names)
    for idx, key in enumerate(ANSWER_KEYS):
        denom = max(1.0, vals[offset + len(ANSWER_KEYS) + idx])
        metrics[f"{key.lower()}_acc"] = vals[offset + idx] / denom
        metrics[f"{key.lower()}_samples"] = vals[offset + len(ANSWER_KEYS) + idx]
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


def _update_pattern_counters(
    counter: Counter[str],
    *,
    spec: PromptSpec,
    row: FrameRow,
    gt: Mapping[str, str],
    parsed: Mapping[str, Optional[str]],
    raw_output: str,
) -> None:
    """累计增强问法的答案模式与未问字段泄漏诊断。"""

    phase2_keys = phase2_output_keys(spec)
    phase2_gt = {key: gt[key] for key in phase2_keys if key in gt}
    phase2_parsed = {key: parsed.get(key) for key in phase2_keys}
    gt_pattern = _dynamic_answer_pattern(phase2_gt)
    pred_pattern = _dynamic_answer_pattern(phase2_parsed)
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
    if variant == "subset_random":
        highway_bucket = "gt_highway" if row.rs == "R3" else "gt_non_highway"
        if gt_pattern == "ALL_NO":
            counter[f"subset_random/gt_all_no/{highway_bucket}"] += 1
        if pred_pattern == "ALL_NO":
            counter[f"subset_random/pred_all_no/{highway_bucket}"] += 1
        asked = set(phase2_keys)
        leaked = []
        for key in PHASE2_ANSWER_KEYS:
            if key in asked:
                continue
            if re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(YES|NO)\b", raw_output or ""):
                leaked.append(key)
        if leaked:
            counter["subset_random/unasked_rs_line_leak"] += 1
            for key in leaked:
                counter[f"subset_random/unasked_rs_line_leak/{key}"] += 1


def _pattern_report(counter: Counter[str]) -> Dict[str, Any]:
    """把答案模式 counter 整理成 JSON 友好的结构。"""

    out: Dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        total = int(counter.get(f"{variant}/total", 0))
        denom = max(1.0, float(total))
        out[variant] = {
            "total": total,
            "pattern_exact_accuracy": float(counter.get(f"{variant}/pattern_exact", 0)) / denom,
            "pred_invalid_rate": float(counter.get(f"{variant}/pred_invalid", 0)) / denom,
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
    out["subset_random_all_no"] = {
        "gt_all_no_highway": int(counter.get("subset_random/gt_all_no/gt_highway", 0)),
        "gt_all_no_non_highway": int(counter.get("subset_random/gt_all_no/gt_non_highway", 0)),
        "pred_all_no_on_highway_gt": int(counter.get("subset_random/pred_all_no/gt_highway", 0)),
        "pred_all_no_on_non_highway_gt": int(counter.get("subset_random/pred_all_no/gt_non_highway", 0)),
    }
    out["subset_random_unasked_key_leak"] = {
        "cases": int(counter.get("subset_random/unasked_rs_line_leak", 0)),
        **{
            key.removeprefix("subset_random/unasked_rs_line_leak/"): int(value)
            for key, value in sorted(counter.items())
            if key.startswith("subset_random/unasked_rs_line_leak/")
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
    """在固定独立 val 样本上以真实 greedy generation 检查四行格式。"""

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
    focus_ok = {key: 0.0 for key in ANSWER_KEYS}
    focus_count = {key: 0.0 for key in ANSWER_KEYS}
    pattern_counts: Counter[str] = Counter()
    records: List[Dict[str, Any]] = []
    for item in work:
        row, focus, spec = item.row, item.focus, item.spec
        images = _load_images(select_history_rgb_paths(row.history_rgb_paths, history_rgb_mode))
        prompt = build_phase1_prompt(spec=spec, audit=False, history_rgb_mode=history_rgb_mode)
        state = _kv_start_state(runtime, _generation_messages(images, prompt))
        raw, _, _ = _student_generate_kv(runtime, state, int(max_new_tokens))
        parsed = parse_phase1_output(raw, spec=spec)
        expected = tuple(spec.output_keys)
        target = build_phase1_target(row.answers, spec=spec)
        gt = dict(line.split(": ", 1) for line in target.splitlines())
        is_valid = all(parsed[key] in ("YES", "NO") for key in expected)
        samples += 1.0
        valid += float(is_valid)
        variant_count[spec.variant] += 1
        variant_valid[spec.variant] += int(is_valid)
        all_ok = False
        if is_valid:
            all_ok = all(parsed[key] == gt[key] for key in expected)
            exact += float(all_ok)
            variant_exact[spec.variant] += int(all_ok)
        for output_key, metric_key, answer in spec_metric_items(spec):
            metric_count[metric_key] += 1
            if parsed.get(output_key) in ("YES", "NO"):
                metric_ok[metric_key] += int(parsed.get(output_key) == _answer_text(answer))
        if parsed[focus] in ("YES", "NO"):
            focus_ok[focus] += float(parsed[focus] == ("YES" if row.answers[focus] else "NO"))
        focus_count[focus] += 1.0
        _update_pattern_counters(
            pattern_counts,
            spec=spec,
            row=row,
            gt={key: gt[key] for key in spec.output_keys},
            parsed={key: parsed.get(key) for key in spec.output_keys},
            raw_output=raw,
        )
        records.append(
            {
                "step": int(step),
                "scenario": row.scenario,
                "route_id": row.route_id,
                "town": row.town,
                "frame_id": row.frame_id,
                "focus": focus,
                "prompt_spec": prompt_spec_to_json(spec),
                "augment_balance_key": item.augment_balance_key,
                "answers": row.answers,
                "parsed": parsed,
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
    for key in ANSWER_KEYS:
        metrics[f"focus_{key.lower()}_acc"] = focus_ok[key] / max(1.0, focus_count[key])
        metrics[f"focus_{key.lower()}_samples"] = focus_count[key]
    for key in _metric_names():
        safe = key.lower().replace(":", "_")
        metrics[f"metric/{safe}_acc"] = metric_ok[key] / max(1.0, metric_count[key])
        metrics[f"metric/{safe}_samples"] = float(metric_count[key])
    for key in VARIANT_ORDER:
        metrics[f"variant/{key}_valid"] = variant_valid[key] / max(1.0, variant_count[key])
        metrics[f"variant/{key}_exact"] = variant_exact[key] / max(1.0, variant_count[key])
        metrics[f"variant/{key}_samples"] = float(variant_count[key])
        denom = max(1.0, float(pattern_counts.get(f"{key}/total", 0)))
        metrics[f"pattern/{key}_pattern_exact"] = float(pattern_counts.get(f"{key}/pattern_exact", 0)) / denom
        metrics[f"pattern/{key}_pred_invalid_rate"] = float(pattern_counts.get(f"{key}/pred_invalid", 0)) / denom
        metrics[f"pattern/{key}_gt_all_no_rate"] = float(pattern_counts.get(f"{key}/gt_all_no", 0)) / denom
        metrics[f"pattern/{key}_pred_all_no_rate"] = float(pattern_counts.get(f"{key}/pred_all_no", 0)) / denom
        metrics[f"pattern/{key}_gt_multi_yes_rate"] = float(pattern_counts.get(f"{key}/gt_multi_yes", 0)) / denom
        metrics[f"pattern/{key}_pred_multi_yes_rate"] = float(pattern_counts.get(f"{key}/pred_multi_yes", 0)) / denom
    subset_total = max(1.0, float(pattern_counts.get("subset_random/total", 0)))
    metrics["pattern/subset_unasked_rs_line_leak_rate"] = (
        float(pattern_counts.get("subset_random/unasked_rs_line_leak", 0)) / subset_total
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
                        "answer_pattern_diagnostics": _pattern_report(pattern_counts),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return metrics


def _save_adapter(bundle: Any, output_dir: pathlib.Path, args: argparse.Namespace, *, step: int, name: str = "final") -> pathlib.Path:
    """保存 LoRA adapter 和 fused-loop 自描述配置。"""

    final_dir = output_dir / str(name)
    final_dir.mkdir(parents=True, exist_ok=True)
    bundle.unwrap().save_pretrained(str(final_dir))
    cfg = {
        "schema": "sft_new_loop_phase1_adapter_config",
        "route": "sft_new_loop_phase1_phase1_phase2_visible_facts",
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "production_prompt_sha256": phase1_prompt_sha256(audit=False, history_rgb_mode=args.history_rgb_mode),
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
        "phase1_output_ordering": "deterministic_random_per_case",
        "answer_phase": {key: focus_phase(key) for key in ANSWER_KEYS},
        "augment_variants": list(VARIANT_ORDER),
        "train_augment_variant_weights": dict(TRAIN_VARIANT_WEIGHTS),
        "eval_augment_variant_weights": dict(VARIANT_WEIGHTS),
        "subset_question_counts": list(SUBSET_COUNTS),
        "hierarchical_group_ids": list(GROUP_DEFINITIONS.keys()),
        "checkpoint_slot": str(name),
        "checkpoint_selection_policy": (
            "min_focus_accuracy_then_joint_exact_then_focus_macro"
            if str(name) == "best_generation_balanced"
            else "joint_exact_after_format_gate"
            if str(name) == "best_generation"
            else "not_generation_selected"
        ),
        "global_step": int(step),
        "num_epochs": int(args.num_epochs),
        "max_steps": int(args.max_steps),
        "focus_balance_count": int(args.focus_balance_count),
        "semantic_supervision": SEMANTIC_SUPERVISION,
        "non_focus_semantic_loss_weight": float(args.non_focus_semantic_loss_weight),
        "max_train_frame_repeat": int(args.max_train_frame_repeat),
        "eval_split": str(args.eval_split),
        "eval_steps": int(args.eval_steps),
        "eval_balance_count": int(args.eval_balance_count),
        "format_loss_weight": float(args.format_loss_weight),
        "generation_eval_steps": int(args.generation_eval_steps),
        "generation_eval_balance_count": int(args.generation_eval_balance_count),
        "generation_eval_max_new_tokens": int(args.generation_eval_max_new_tokens),
        "generation_format_valid_gate": float(args.generation_format_valid_gate),
    }
    (final_dir / "sft_new_loop_phase1_adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_dir


def _balanced_generation_selection_key(metrics: Mapping[str, float]) -> Tuple[float, float, float]:
    """按最弱 focus、联合 exact、focus 宏平均选择额外的稳健 checkpoint。

    ``best_generation`` 仍保持历史的联合 exact 主指标；这个 key 只用于额外保存
    ``best_generation_balanced``，防止八问联合 exact 掩盖单个 focus 明显塌陷。
    """

    focus_values = [float(metrics.get(key, 0.0)) for key in GENERATION_FOCUS_METRIC_KEYS]
    focus_macro = sum(focus_values) / max(1, len(focus_values))
    return min(focus_values), float(metrics.get("exact_accuracy", 0.0)), focus_macro


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = _read_rows(
        pathlib.Path(args.index),
        split=str(args.split),
        max_frames=int(args.max_frames),
        data_root=pathlib.Path(args.data_root),
    )
    full_work = _balanced_work(
        rows,
        target_per_bin=int(args.focus_balance_count),
        seed=int(args.seed),
        variant_weights=TRAIN_VARIANT_WEIGHTS,
    )
    _assert_repeat_limit(
        full_work,
        max_repeat=int(args.max_train_frame_repeat),
        context="initial train work",
    )
    if not full_work:
        raise ValueError("balanced work list is empty")
    semantic_class_weights = _semantic_class_weights(
        full_work,
        float(args.non_focus_semantic_loss_weight),
    )
    # 每个 rank 使用同一个八桶均衡全集的 rank::world_size 分片。rank0 保存全集
    # balance 供审计；训练 epoch 会在每个 rank 的 shard 内 shuffle。
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
                variant_weights=VARIANT_WEIGHTS,
            )
            eval_work = _split_work_for_rank(full_eval_work, rank=rank, world_size=world_size)
            if int(args.generation_eval_steps) > 0 and int(args.generation_eval_balance_count) > 0:
                full_generation_eval_work = _balanced_work(
                    eval_rows,
                    target_per_bin=int(args.generation_eval_balance_count),
                    seed=int(args.seed) + 2017,
                    variant_weights=VARIANT_WEIGHTS,
                )
        except Exception as exc:
            raise RuntimeError(
                "periodic validation was requested but its split cannot satisfy the exact eight-bin balance. "
                "Rebuild/fix the dataset instead of silently training without validation."
            ) from exc
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch0_balance_path = _write_epoch_balance_report(
            output_dir,
            epoch=0,
            work=full_work,
            rows=rows,
            split=str(args.split),
            focus_balance_count=int(args.focus_balance_count),
            seed=int(args.seed),
            variant_weights=TRAIN_VARIANT_WEIGHTS,
            world_size=world_size,
            non_focus_semantic_loss_weight=float(args.non_focus_semantic_loss_weight),
        )
        val_balance = {}
        if eval_work:
            val_balance = _work_balance_report(
                full_eval_work,
                rows=eval_rows,
                split=str(args.eval_split),
                focus_balance_count=int(args.eval_balance_count),
                seed=int(args.seed) + 1009,
                variant_weights=VARIANT_WEIGHTS,
                world_size=world_size,
                rank_work=eval_work,
                non_focus_semantic_loss_weight=float(args.non_focus_semantic_loss_weight),
            )
        generation_balance = {}
        if full_generation_eval_work:
            generation_balance = _work_balance_report(
                full_generation_eval_work,
                rows=eval_rows,
                split=str(args.eval_split),
                focus_balance_count=int(args.generation_eval_balance_count),
                seed=int(args.seed) + 2017,
                variant_weights=VARIANT_WEIGHTS,
                world_size=1,
                non_focus_semantic_loss_weight=float(args.non_focus_semantic_loss_weight),
            )
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
                        "resample_each_epoch": True,
                        "epoch_seed_formula": "seed + epoch * 1000003",
                        "epoch_balance_dir": "balance",
                        "epoch0_balance_path": str(epoch0_balance_path.relative_to(output_dir)),
                        **_work_balance_report(
                            full_work,
                            rows=rows,
                            split=str(args.split),
                            focus_balance_count=int(args.focus_balance_count),
                            seed=int(args.seed),
                            variant_weights=TRAIN_VARIANT_WEIGHTS,
                            world_size=world_size,
                            rank_work=work,
                            non_focus_semantic_loss_weight=float(args.non_focus_semantic_loss_weight),
                        ),
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
                        **generation_balance,
                        "rank0_only": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if world_size > 1:
        _ddp_barrier(local_rank)

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
    if rank == 0:
        _write_run_metadata(
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
    rng = random.Random(int(args.seed))
    global_step = 0
    skipped = 0
    train_window: Counter[str] = Counter()
    best_val_score = -1.0
    best_generation_score = -1.0
    best_generation_balanced_key: Optional[Tuple[float, float, float]] = None
    t0 = time.time()
    bundle.model.train()
    if rank == 0:
        print(
            f"[data] train_rows={len(rows)} train_work_global={len(full_work)} train_work_rank={len(work)} "
            f"steps_per_epoch_rank={steps_per_epoch} num_epochs={int(args.num_epochs)} max_steps={int(args.max_steps)} "
            f"total_steps_rank={total_steps} eval_work_rank={len(eval_work)} "
            f"generation_eval_global={len(full_generation_eval_work)} "
            f"history_rgb_mode={args.history_rgb_mode} history_rgb_count={len(history_rgb_indices(args.history_rgb_mode))}"
        )

    def run_scheduled_after_optimizer_step(
        *,
        teacher_trigger_step: int,
        generation_trigger_step: int,
        checkpoint_trigger_step: int,
    ) -> None:
        """在 optimizer step 后执行延迟的评测/保存，避免保存未应用梯度的 adapter。"""

        nonlocal best_val_score, best_generation_score, best_generation_balanced_key
        if eval_work and int(args.eval_steps) > 0 and teacher_trigger_step > 0:
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
                event_record = {"step": int(global_step), "type": "teacher_forced", **metrics}
                if teacher_trigger_step != global_step:
                    event_record["trigger_step"] = int(teacher_trigger_step)
                    event_record["delayed_until_optimizer_step"] = True
                _append_jsonl(output_dir / "train_eval_metrics.jsonl", event_record)
                val_score = float(metrics.get("value_token_acc", 0.0))
                if val_score > best_val_score:
                    best_val_score = val_score
                    ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_val")
                    print(f"[best-val] step={global_step} value_token_acc={val_score:.4f} adapter={ckpt_dir}")
                delay_note = "" if teacher_trigger_step == global_step else f" delayed_from={teacher_trigger_step}"
                print(
                    f"[eval] step={global_step}/{total_steps}{delay_note} split={args.eval_split} "
                    f"loss={metrics['loss']:.4f} value_acc={metrics['value_token_acc']:.4f} "
                    f"format_acc={metrics['format_token_acc']:.4f} "
                    f"focus_highway={metrics.get('focus_highway_acc', 0.0):.4f} "
                    f"focus_static_obstacle={metrics.get('focus_static_obstacle_acc', 0.0):.4f} "
                    f"focus_vulnerable={metrics.get('focus_vulnerable_acc', 0.0):.4f} "
                    f"focus_light={metrics.get('focus_traffic_light_abnormal_acc', 0.0):.4f}"
                )
        run_generation_eval = bool(full_generation_eval_work) and generation_trigger_step > 0
        if run_generation_eval:
            sync_dir = output_dir / ".dist_sync"
            sync_done = sync_dir / f"generation_eval_step_{global_step}.done.json"
            sync_error = sync_dir / f"generation_eval_step_{global_step}.error.json"
            sync_token = (
                f"{os.environ.get('MASTER_ADDR', '')}:"
                f"{os.environ.get('MASTER_PORT', '')}:"
                f"{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}:"
                f"{global_step}"
            )
            if rank == 0:
                try:
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
                    generation_record = {"step": int(global_step), "type": "generation", **generation_metrics}
                    if generation_trigger_step != global_step:
                        generation_record["trigger_step"] = int(generation_trigger_step)
                        generation_record["delayed_until_optimizer_step"] = True
                    _append_jsonl(output_dir / "train_eval_metrics.jsonl", generation_record)
                    generation_score = float(generation_metrics.get("exact_accuracy", 0.0))
                    generation_valid = float(generation_metrics.get("format_valid_rate", 0.0))
                    if generation_valid >= float(args.generation_format_valid_gate) and generation_score > best_generation_score:
                        best_generation_score = generation_score
                        ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_generation")
                        print(
                            f"[best-generation] step={global_step} exact={generation_score:.4f} "
                            f"format_valid={generation_valid:.4f} adapter={ckpt_dir}"
                        )
                    balanced_key = _balanced_generation_selection_key(generation_metrics)
                    if (
                        generation_valid >= float(args.generation_format_valid_gate)
                        and (best_generation_balanced_key is None or balanced_key > best_generation_balanced_key)
                    ):
                        best_generation_balanced_key = balanced_key
                        balanced_dir = _save_adapter(
                            bundle,
                            output_dir,
                            args,
                            step=global_step,
                            name="best_generation_balanced",
                        )
                        print(
                            f"[best-generation-balanced] step={global_step} "
                            f"min_focus={balanced_key[0]:.4f} exact={balanced_key[1]:.4f} "
                            f"focus_macro={balanced_key[2]:.4f} format_valid={generation_valid:.4f} "
                            f"adapter={balanced_dir}"
                        )
                    delay_note = "" if generation_trigger_step == global_step else f" delayed_from={generation_trigger_step}"
                    print(
                        f"[generation-val] step={global_step}/{total_steps}{delay_note} "
                        f"format_valid={generation_metrics['format_valid_rate']:.4f} "
                        f"exact={generation_metrics['exact_accuracy']:.4f} "
                        f"focus_highway={generation_metrics.get('focus_highway_acc', 0.0):.4f} "
                        f"focus_static_obstacle={generation_metrics.get('focus_static_obstacle_acc', 0.0):.4f} "
                        f"focus_vulnerable={generation_metrics.get('focus_vulnerable_acc', 0.0):.4f} "
                        f"focus_light={generation_metrics.get('focus_traffic_light_abnormal_acc', 0.0):.4f}"
                    )
                    _write_sync_file(
                        sync_done,
                        {
                            "sync_token": sync_token,
                            "step": int(global_step),
                            "trigger_step": int(generation_trigger_step),
                            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        },
                    )
                except BaseException as exc:
                    _write_sync_file(
                        sync_error,
                        {
                            "sync_token": sync_token,
                            "step": int(global_step),
                            "trigger_step": int(generation_trigger_step),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    raise
            else:
                _wait_for_rank0_sync(
                    sync_done,
                    sync_error,
                    timeout_seconds=_env_int("GENERATION_EVAL_SYNC_TIMEOUT_SECONDS", 7200),
                    sync_token=sync_token,
                )
            if world_size > 1:
                _ddp_barrier(local_rank)
        if rank == 0 and checkpoint_trigger_step > 0:
            if checkpoint_trigger_step == global_step:
                ckpt_name = f"checkpoint-{global_step}"
            else:
                ckpt_name = f"checkpoint-{checkpoint_trigger_step}-applied-{global_step}"
            ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=ckpt_name)
            delay_note = "" if checkpoint_trigger_step == global_step else f" delayed_from={checkpoint_trigger_step}"
            print(f"[save] step={global_step}{delay_note} adapter={ckpt_dir}")

    pending_teacher_eval_step = 0
    pending_generation_eval_step = 0
    pending_checkpoint_step = 0
    epoch = 0
    while global_step < total_steps:
        if epoch > 0:
            epoch_seed = int(args.seed) + epoch * 1_000_003
            full_work = _balanced_work(
                rows,
                target_per_bin=int(args.focus_balance_count),
                seed=epoch_seed,
                variant_weights=TRAIN_VARIANT_WEIGHTS,
            )
            semantic_class_weights = _semantic_class_weights(
                full_work,
                float(args.non_focus_semantic_loss_weight),
            )
            _assert_repeat_limit(
                full_work,
                max_repeat=int(args.max_train_frame_repeat),
                context=f"train epoch {epoch}",
            )
            work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
            if rank == 0:
                _write_epoch_balance_report(
                    output_dir,
                    epoch=epoch,
                    work=full_work,
                    rows=rows,
                    split=str(args.split),
                    focus_balance_count=int(args.focus_balance_count),
                    seed=epoch_seed,
                    variant_weights=TRAIN_VARIANT_WEIGHTS,
                    world_size=world_size,
                    non_focus_semantic_loss_weight=float(args.non_focus_semantic_loss_weight),
                )
        rng.shuffle(work)
        epoch_start_step = global_step
        for item in work:
            row, focus, spec = item.row, item.focus, item.spec
            images = _load_images(select_history_rgb_paths(row.history_rgb_paths, args.history_rgb_mode))
            prompt = build_phase1_prompt(spec=spec, audit=False, history_rgb_mode=args.history_rgb_mode)
            target = build_phase1_target(row.answers, spec=spec)
            packed = _build_inputs(
                bundle,
                images=images,
                prompt=prompt,
                target=target,
                output_keys=spec.output_keys,
                semantic_output_keys=_semantic_output_keys(
                    item,
                    float(args.non_focus_semantic_loss_weight),
                ),
                semantic_output_weights=_semantic_output_weights(
                    item,
                    semantic_class_weights,
                    float(args.non_focus_semantic_loss_weight),
                ),
                max_length=int(args.max_length),
                format_loss_weight=float(args.format_loss_weight),
            )
            if packed is None:
                skipped += 1
                loss, stats = _dummy_ddp_forward_zero_loss(
                    bundle,
                    image=images[-1],
                    max_length=int(args.max_length),
                    format_loss_weight=float(args.format_loss_weight),
                )
            else:
                loss, stats = _loss_one(bundle, packed)
            (loss / max(1, int(args.grad_accum))).backward()
            optimizer_stepped = False
            if (global_step + 1) % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_stepped = True
            if writer:
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                writer.add_scalar("train/loss", float(loss.detach().item()), global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("train/skipped_too_long", skipped, global_step)
                writer.add_scalar(f"train/focus/{focus.lower()}", 1, global_step)
                for key, value in stats.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
            train_window["samples"] += 1
            train_window["loss_sum"] += float(loss.detach().item())
            train_window[f"focus/{focus}"] += 1
            train_window[f"variant/{spec.variant}"] += 1
            train_window[f"augment/{item.augment_balance_key}"] += 1
            train_window["skipped"] = skipped
            for key in ("token_acc", "value_token_acc", "format_token_acc"):
                train_window[f"{key}_sum"] += float(stats.get(key, 0.0))
            if rank == 0 and global_step % int(args.log_steps) == 0:
                samples_window = max(1.0, float(train_window.get("samples", 0)))
                current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                _append_jsonl(
                    output_dir / "train_metrics.jsonl",
                    {
                        "step": int(global_step),
                        "epoch": int(epoch + 1),
                        "samples": int(train_window.get("samples", 0)),
                        "learning_rate": current_lr,
                        "loss": float(train_window.get("loss_sum", 0.0)) / samples_window,
                        "token_acc": float(train_window.get("token_acc_sum", 0.0)) / samples_window,
                        "value_token_acc": float(train_window.get("value_token_acc_sum", 0.0)) / samples_window,
                        "format_token_acc": float(train_window.get("format_token_acc_sum", 0.0)) / samples_window,
                        "skipped_total": int(skipped),
                        "focus_counts": {
                            key.removeprefix("focus/"): int(value)
                            for key, value in sorted(train_window.items())
                            if key.startswith("focus/")
                        },
                        "variant_counts": {
                            key.removeprefix("variant/"): int(value)
                            for key, value in sorted(train_window.items())
                            if key.startswith("variant/")
                        },
                        "augment_counts": {
                            key.removeprefix("augment/"): int(value)
                            for key, value in sorted(train_window.items())
                            if key.startswith("augment/")
                        },
                    },
                )
                train_window.clear()
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} loss={float(loss.detach().item()):.4f} "
                    f"lr={current_lr:.6g} "
                    f"focus={focus}:{'YES' if row.answers[focus] else 'NO'} skipped={skipped} world={world_size} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            global_step += 1
            if eval_work and int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
                pending_teacher_eval_step = global_step
            if (
                full_generation_eval_work
                and int(args.generation_eval_steps) > 0
                and global_step % int(args.generation_eval_steps) == 0
            ):
                pending_generation_eval_step = global_step
            if int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                pending_checkpoint_step = global_step
            if optimizer_stepped and (pending_teacher_eval_step or pending_generation_eval_step or pending_checkpoint_step):
                run_scheduled_after_optimizer_step(
                    teacher_trigger_step=pending_teacher_eval_step,
                    generation_trigger_step=pending_generation_eval_step,
                    checkpoint_trigger_step=pending_checkpoint_step,
                )
                pending_teacher_eval_step = 0
                pending_generation_eval_step = 0
                pending_checkpoint_step = 0
            if global_step >= total_steps:
                break
        if global_step == epoch_start_step:
            raise RuntimeError("no train steps were completed in an epoch; check max_length and input data")
        epoch += 1

    if global_step > 0 and global_step % max(1, int(args.grad_accum)) != 0:
        torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if pending_teacher_eval_step or pending_generation_eval_step or pending_checkpoint_step:
            run_scheduled_after_optimizer_step(
                teacher_trigger_step=pending_teacher_eval_step,
                generation_trigger_step=pending_generation_eval_step,
                checkpoint_trigger_step=pending_checkpoint_step,
            )
            pending_teacher_eval_step = 0
            pending_generation_eval_step = 0
            pending_checkpoint_step = 0
    if world_size > 1:
        _ddp_barrier(local_rank)
    final_dir = _save_adapter(bundle, output_dir, args, step=global_step) if rank == 0 else None
    if writer:
        writer.close()
    if rank == 0:
        print(f"[done] saved adapter to {final_dir}")
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Train sft_new_loop_phase1 fused eight-question LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_data/frame_index.jsonl"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"), help="root used to resolve relative RGB paths or remap old absolute lead_data paths")
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument(
        "--output-dir",
        default="",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--history-rgb-mode", choices=HISTORY_RGB_MODES, default=DEFAULT_HISTORY_RGB_MODE)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--focus-balance-count", type=int, default=9216)
    p.add_argument(
        "--max-train-frame-repeat",
        type=int,
        default=10,
        help="abort before model training when one sampled frame appears more than this many times; <=0 disables",
    )
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="0 means train num_epochs over the balanced work list")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--eval-steps", type=int, default=2_000)
    p.add_argument("--eval-balance-count", type=int, default=16)
    p.add_argument("--max-eval-frames", type=int, default=0)
    p.add_argument(
        "--format-loss-weight",
        type=float,
        default=0.25,
        help=(
            "low loss weight for answer field names, separators, newlines, and assistant end token; "
            "YES/NO value weights are controlled separately by semantic supervision and "
            "--non-focus-semantic-loss-weight"
        ),
    )
    p.add_argument(
        "--non-focus-semantic-loss-weight",
        type=float,
        default=DEFAULT_NON_FOCUS_SEMANTIC_LOSS_WEIGHT,
        help=(
            "semantic YES/NO loss scale for requested main-answer lines other than the balanced focus line; "
            "focus and hierarchical derived lines keep base scale 1.0"
        ),
    )
    p.add_argument(
        "--generation-eval-steps",
        type=int,
        default=2_000,
        help="run rank0 free-generation validation every N teacher-forced validation steps; 0 disables it",
    )
    p.add_argument("--generation-eval-balance-count", type=int, default=16)
    p.add_argument("--generation-eval-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-format-valid-gate", type=float, default=0.99)
    p.add_argument("--save-steps", type=int, default=20_000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=2000)
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
    args = p.parse_args()
    args.history_rgb_mode = validate_history_rgb_mode(args.history_rgb_mode)
    if not 0.0 < float(args.format_loss_weight) <= 1.0:
        raise ValueError("--format-loss-weight must be in (0, 1]")
    if not 0.0 <= float(args.non_focus_semantic_loss_weight) <= 1.0:
        raise ValueError("--non-focus-semantic-loss-weight must be in [0, 1]")
    if int(args.generation_eval_steps) > 0:
        if int(args.eval_steps) <= 0 or int(args.eval_balance_count) <= 0:
            raise ValueError("free-generation validation requires --eval-steps and --eval-balance-count to be positive")
        if int(args.generation_eval_steps) % int(args.eval_steps) != 0:
            raise ValueError("--generation-eval-steps must be a multiple of --eval-steps")
        if int(args.generation_eval_balance_count) <= 0:
            raise ValueError("--generation-eval-balance-count must be positive when generation validation is enabled")
    if not args.output_dir:
        args.output_dir = str(
            _AUTOMOT_ROOT
            / "checkpoints/sft_new_loop_phase1_runs"
            / f"run_combined_phase1_phase2_{history_rgb_mode_tag(args.history_rgb_mode)}"
        )
    return args


def main() -> None:
    """CLI 入口。"""

    train(parse_args())


if __name__ == "__main__":
    main()
