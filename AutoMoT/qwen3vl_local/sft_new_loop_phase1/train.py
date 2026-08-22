#!/usr/bin/env python3
"""训练 sft_new_loop_phase1 八问 LoRA adapter。

训练目标是 Phase1 四问 + Phase2 四个 RS 问题的 YES/NO 语义 token，并以低权重
监督字段格式和 assistant 结束符。采样时每帧展开成八个不可见 focus 视图，
按 `问题 x YES/NO` exact balance；prompt 中不出现 focus，模型每次仍回答八项。
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
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


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


def _variant_for_slot(slot: int, weights: Mapping[str, int]) -> str:
    """按 phase2 权重把 slot 映射到增强 variant。"""

    schedule: List[str] = []
    for name, weight in weights.items():
        schedule.extend([str(name)] * max(0, int(weight)))
    if not schedule:
        raise ValueError("empty variant weights")
    return schedule[int(slot) % len(schedule)]


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


def _make_spec(row: FrameRow, focus: str, *, seed: int, slot: int, weights: Mapping[str, int]) -> PromptSpec:
    """为当前 focus 构造一个融合 Phase2 augment spec。"""

    variant = _variant_for_slot(slot, weights)
    phase2_focus = focus if focus in {"RS1", "RS2", "RS4", "RS5"} else ["RS1", "RS2", "RS4", "RS5"][slot % 4]
    subset_count = SUBSET_COUNTS[slot % len(SUBSET_COUNTS)]
    group_ids = sorted(GROUP_DEFINITIONS.keys())
    group_id = group_ids[slot % len(group_ids)]
    detail_key = phase2_focus if phase2_focus in {"RS1", "RS2", "RS4", "RS5"} else ["RS1", "RS2", "RS4", "RS5"][slot % 4]
    return make_prompt_spec(
        variant=variant,
        answers=row.answers,
        seed_key=f"{seed}:{row.scenario}:{row.route_id}:{row.frame_id}:{focus}:{slot}",
        focus=phase2_focus,
        subset_count=int(subset_count),
        group_id=group_id,
        detail_key=detail_key,
    )


def _augment_balance_key(spec: PromptSpec, *, fallback_focus: str) -> str:
    """返回 Phase2 原版风格的增强 balance key，用于实际问题均衡审计。"""

    variant = spec.variant
    if variant == "all_random_order":
        focus = fallback_focus if fallback_focus in PHASE2_ANSWER_KEYS else spec.phase2_spec.questions[0].question_id
        answer = next((bool(q.answer) for q in spec.phase2_spec.questions if q.question_id == focus), False)
        return f"all_random_order/{focus}:{_answer_text(answer)}"
    if variant == "subset_random":
        items = ",".join(f"{q.question_id}:{_answer_text(q.answer)}" for q in spec.phase2_spec.questions)
        return f"subset_random/q{len(spec.phase2_spec.questions)}/items/{items}"
    if variant == "hierarchical_probe":
        answers = {q.output_key: bool(q.answer) for q in spec.phase2_spec.questions}
        group = next(q.metric_key.split(":", 1)[1] for q in spec.phase2_spec.questions if q.metric_key.startswith("GROUP:"))
        detail = next(q.question_id for q in spec.phase2_spec.questions if q.question_id in PHASE2_ANSWER_KEYS)
        return (
            f"hierarchical_probe/highway:{_answer_text(answers['HIGHWAY'])}"
            f"/group/{group}:{_answer_text(answers['GROUP'])}"
            f"/detail/{detail}:{_answer_text(answers['DETAIL'])}"
        )
    raise ValueError(f"unknown variant: {variant}")


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


def _iter_phase2_candidate_items(rows: Sequence[FrameRow], *, seed: int) -> Iterable[WorkItem]:
    """按 Phase2 原版 augment surface 流式生成候选。"""

    for row in rows:
        for focus in PHASE2_ANSWER_KEYS:
            yield _make_all_item(row, focus, seed=seed)
            for count in SUBSET_COUNTS:
                yield _make_subset_item(row, focus, int(count), seed=seed)
        for group_id in GROUP_DEFINITIONS:
            for detail_key in PHASE2_ANSWER_KEYS:
                yield _make_hier_item(row, group_id, detail_key, seed=seed)


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


def _phase2_balanced_work(
    rows: Sequence[FrameRow],
    *,
    total_items: int,
    seed: int,
    variant_weights: Mapping[str, int],
) -> List[WorkItem]:
    """按 Phase2 augment 的实际问题多边际配额抽样。"""

    rng = random.Random(f"{seed}:fused_phase2_augment_balance:{len(rows)}:{int(total_items)}")
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
    selected: Dict[str, List[WorkItem]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for item in _iter_phase2_candidate_items(rows, seed=seed):
        key = item.augment_balance_key
        per_key_target = int(per_key_targets.get(key, 0))
        if per_key_target <= 0:
            continue
        seen[key] += 1
        bucket = selected[key]
        if len(bucket) < per_key_target:
            bucket.append(item)
            continue
        replace_idx = rng.randrange(int(seen[key]))
        if replace_idx < per_key_target:
            bucket[replace_idx] = item
    work: List[WorkItem] = []
    for key in sorted(per_key_targets):
        target = int(per_key_targets[key])
        items = selected.get(key, [])
        if not items:
            continue
        if len(items) >= target:
            work.extend(items[:target])
        else:
            repeated = [items[idx % len(items)] for idx in range(target)]
            rng.shuffle(repeated)
            work.extend(repeated)
    rng.shuffle(work)
    return work


def _candidate_specs_for_row(row: FrameRow, *, seed: int, focus: str) -> Iterable[WorkItem]:
    """为一个已选 Phase1-focus row 生成所有可用 Phase2 augment spec 候选。"""

    for phase2_focus in PHASE2_ANSWER_KEYS:
        item = _make_all_item(row, phase2_focus, seed=seed)
        yield WorkItem(row=row, focus=focus, spec=item.spec, balance_key=_focus_key(row, focus), augment_balance_key=item.augment_balance_key)
        for count in SUBSET_COUNTS:
            item = _make_subset_item(row, phase2_focus, int(count), seed=seed)
            yield WorkItem(row=row, focus=focus, spec=item.spec, balance_key=_focus_key(row, focus), augment_balance_key=item.augment_balance_key)
    for group_id in GROUP_DEFINITIONS:
        for detail_key in PHASE2_ANSWER_KEYS:
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


def _assign_phase1_specs(
    pairs: Sequence[Tuple[FrameRow, str]],
    *,
    seed: int,
    variant_weights: Mapping[str, int],
) -> List[WorkItem]:
    """为 Phase1-focus pairs 贪心选择 Phase2 spec，减少实际增强问题偏斜。"""

    per_key_targets = _augment_key_targets(len(pairs), variant_weights)
    variant_targets = _augment_target_counts(len(pairs), variant_weights)
    label_targets: Counter[str] = Counter()
    all_total = int(variant_targets.get("all_random_order", 0))
    for key in PHASE2_ANSWER_KEYS:
        label_targets[f"all/{key}:YES"] = all_total // 2
        label_targets[f"all/{key}:NO"] = all_total - label_targets[f"all/{key}:YES"]
    subset_total = int(variant_targets.get("subset_random", 0))
    for count in SUBSET_COUNTS:
        count_total = subset_total // len(SUBSET_COUNTS) + int((int(count) - 1) < subset_total % len(SUBSET_COUNTS))
        asked_per_rs = (count_total * int(count)) // len(PHASE2_ANSWER_KEYS)
        yes_per_rs = min(asked_per_rs // 2, count_total // len(PHASE2_ANSWER_KEYS))
        no_per_rs = max(0, asked_per_rs - yes_per_rs)
        for rs in PHASE2_ANSWER_KEYS:
            label_targets[f"subset_q{count}/{rs}:YES"] = yes_per_rs
            label_targets[f"subset_q{count}/{rs}:NO"] = no_per_rs
    hier_total = int(variant_targets.get("hierarchical_probe", 0))
    label_targets["highway:YES"] = hier_total // 2
    label_targets["highway:NO"] = hier_total - label_targets["highway:YES"]
    for bins in (
        [f"group/{group_id}:{answer}" for group_id in GROUP_DEFINITIONS for answer in ("YES", "NO")],
        [f"detail/{detail_key}:{answer}" for detail_key in PHASE2_ANSWER_KEYS for answer in ("YES", "NO")],
    ):
        base = hier_total // len(bins) if bins else 0
        remainder = hier_total - base * len(bins)
        for idx, label in enumerate(bins):
            label_targets[label] = base + int(idx < remainder)
    key_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    out: List[WorkItem] = []
    for slot, (row, focus) in enumerate(pairs):
        candidates = list(_candidate_specs_for_row(row, seed=seed + slot, focus=focus))
        best: Optional[WorkItem] = None
        best_score = -10**18
        for candidate in candidates:
            variant = candidate.spec.variant
            if variant_counts[variant] >= variant_targets.get(variant, 0):
                continue
            key_target = int(per_key_targets.get(candidate.augment_balance_key, 0))
            key_deficit = key_target - key_counts[candidate.augment_balance_key]
            variant_deficit = variant_targets.get(variant, 0) - variant_counts[variant]
            if variant == "hierarchical_probe":
                labels = _hierarchical_key_labels(candidate.augment_balance_key)
            elif variant == "subset_random":
                labels = _subset_key_labels(candidate.augment_balance_key)
            else:
                labels = tuple(
                    f"all/{q.question_id}:{_answer_text(q.answer)}"
                    for q in candidate.spec.phase2_spec.questions
                    if q.question_id in PHASE2_ANSWER_KEYS
                )
            label_score = 0.0
            for label in labels:
                deficit = label_targets[label] - label_counts[label]
                weight = 100.0 / max(1.0, float(label_targets[label]))
                label_score += deficit * weight
                if deficit <= 0:
                    label_score += deficit * weight * 2.0
            score = label_score * 100 + key_deficit * 10 + variant_deficit
            if key_deficit <= 0:
                score += key_deficit * 20
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            best = candidates[slot % len(candidates)]
        if best.spec.variant == "hierarchical_probe":
            best_labels = _hierarchical_key_labels(best.augment_balance_key)
        elif best.spec.variant == "subset_random":
            best_labels = _subset_key_labels(best.augment_balance_key)
        else:
            best_labels = tuple(
                f"all/{q.question_id}:{_answer_text(q.answer)}"
                for q in best.spec.phase2_spec.questions
                if q.question_id in PHASE2_ANSWER_KEYS
            )
        for label in best_labels:
            label_counts[label] += 1
        key_counts[best.augment_balance_key] += 1
        variant_counts[best.spec.variant] += 1
        out.append(best)
    return out


def _balanced_work(
    rows: Sequence[FrameRow],
    *,
    target_per_bin: int,
    seed: int,
    variant_weights: Mapping[str, int],
) -> List[WorkItem]:
    """构建 Phase1 focus 半边 + Phase2 augment 半边的融合 work list。"""

    groups: Dict[str, List[Tuple[FrameRow, str]]] = {f"{key}:{value}": [] for key in PHASE1_ANSWER_KEYS for value in ("YES", "NO")}
    for row in rows:
        for focus in PHASE1_ANSWER_KEYS:
            groups[_focus_key(row, focus)].append((row, focus))
    raw_counts = {key: len(items) for key, items in groups.items()}
    expected_phase1_bins = [f"{key}:{value}" for key in PHASE1_ANSWER_KEYS for value in ("YES", "NO")]
    missing = [key for key in expected_phase1_bins if raw_counts[key] == 0]
    if missing:
        raise ValueError(
            "cannot build exact 1:1 Phase1-focus training work: required focus bins are empty; "
            f"missing={missing} raw_counts={raw_counts}. Rebuild/check the dataset split or reduce filtering."
        )
    nonempty = list(raw_counts.values())
    target = int(target_per_bin) if int(target_per_bin) > 0 else min(nonempty)
    target = max(1, target)
    rng = random.Random(f"{seed}:phase1_balance:{len(rows)}:{target}")
    phase1_pairs: List[Tuple[FrameRow, str]] = []
    for key in sorted(groups):
        items = list(groups[key])
        if not items:
            continue
        rng.shuffle(items)
        if len(items) >= target:
            selected = items[:target]
        else:
            selected = [items[i % len(items)] for i in range(target)]
            rng.shuffle(selected)
        phase1_pairs.extend(selected)
    rng.shuffle(phase1_pairs)
    phase1_work = _assign_phase1_specs(phase1_pairs, seed=seed, variant_weights=variant_weights)
    _assert_exact_focus_balance(
        phase1_work,
        target=target,
        context="Phase1-focus training work",
        keys=PHASE1_ANSWER_KEYS,
    )
    phase2_work = _phase2_balanced_work(
        rows,
        total_items=len(phase1_work),
        seed=seed,
        variant_weights=variant_weights,
    )
    if not phase2_work:
        raise ValueError("cannot build Phase2 augment-balanced work; no active augment buckets were sampled")
    work = [*phase1_work, *phase2_work]
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
    format_loss_weight: float,
) -> Tuple[List[int], List[float], List[int]]:
    """映射当前 spec 输出行的语义与格式 token 权重。"""

    enc = bundle.tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    weights = [float(format_loss_weight) for _ in token_ids]
    component_ids = [FORMAT_COMPONENT_ID for _ in token_ids]
    for component_id, key in enumerate(output_keys, start=1):
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
    output_keys: Sequence[str],
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


def _zero_loss_for_all_trainable_params(bundle: Any) -> torch.Tensor:
    """超长样本跳过时仍让 DDP 对所有可训练参数完成一次零梯度 backward。"""

    total: Optional[torch.Tensor] = None
    for param in bundle.model.parameters():
        if not param.requires_grad:
            continue
        term = param.float().sum() * 0.0
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("no trainable parameters for zero-loss DDP step")
    return total


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
    if variant == "subset_random":
        highway_bucket = "gt_highway" if row.rs == "R3" else "gt_non_highway"
        if gt_pattern == "ALL_NO":
            counter[f"subset_random/gt_all_no/{highway_bucket}"] += 1
        if pred_pattern == "ALL_NO":
            counter[f"subset_random/pred_all_no/{highway_bucket}"] += 1
        asked = set(spec.output_keys)
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
        "answer_phase": {key: focus_phase(key) for key in ANSWER_KEYS},
        "augment_variants": list(VARIANT_ORDER),
        "train_augment_variant_weights": dict(TRAIN_VARIANT_WEIGHTS),
        "eval_augment_variant_weights": dict(VARIANT_WEIGHTS),
        "subset_question_counts": list(SUBSET_COUNTS),
        "hierarchical_group_ids": list(GROUP_DEFINITIONS.keys()),
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
        "generation_format_valid_gate": float(args.generation_format_valid_gate),
    }
    (final_dir / "sft_new_loop_phase1_adapter_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_dir


def train(args: argparse.Namespace) -> None:
    """训练主流程。"""

    rank, local_rank, world_size = setup_distributed()
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    full_work = _balanced_work(
        rows,
        target_per_bin=int(args.focus_balance_count),
        seed=int(args.seed),
        variant_weights=TRAIN_VARIANT_WEIGHTS,
    )
    if not full_work:
        raise ValueError("balanced work list is empty")
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
            eval_rows = _read_rows(pathlib.Path(args.index), split=str(args.eval_split), max_frames=int(args.max_eval_frames))
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
        val_balance = {}
        if eval_work:
            val_balance = {
                "raw_available": _raw_focus_bin_counts(eval_rows),
                "global_sampled": dict(Counter(item.balance_key for item in full_eval_work)),
                "augment_global_sampled": dict(Counter(item.augment_balance_key for item in full_eval_work)),
                "variant_sampled": dict(Counter(item.spec.variant for item in full_eval_work)),
                "phase_group_sampled": dict(Counter(focus_phase(item.focus) for item in full_eval_work)),
                "phase_group_answer_sampled": dict(
                    Counter(f"{focus_phase(item.focus)}:{'YES' if item.row.answers[item.focus] else 'NO'}" for item in full_eval_work)
                ),
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
                        "resample_each_epoch": True,
                        "epoch_seed_formula": "seed + epoch * 1000003",
                        "raw_available": _raw_focus_bin_counts(rows),
                        "global_sampled": dict(Counter(item.balance_key for item in full_work)),
                        "augment_global_sampled": dict(Counter(item.augment_balance_key for item in full_work)),
                        "variant_sampled": dict(Counter(item.spec.variant for item in full_work)),
                        "phase_group_sampled": dict(Counter(focus_phase(item.focus) for item in full_work)),
                        "phase_group_answer_sampled": dict(
                            Counter(f"{focus_phase(item.focus)}:{'YES' if item.row.answers[item.focus] else 'NO'}" for item in full_work)
                        ),
                        "rank0_shard": dict(Counter(item.balance_key for item in work)),
                        "rank0_augment_shard": dict(Counter(item.augment_balance_key for item in work)),
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
                        "global_sampled": dict(Counter(item.balance_key for item in full_generation_eval_work)),
                        "augment_global_sampled": dict(Counter(item.augment_balance_key for item in full_generation_eval_work)),
                        "variant_sampled": dict(Counter(item.spec.variant for item in full_generation_eval_work)),
                        "phase_group_sampled": dict(Counter(focus_phase(item.focus) for item in full_generation_eval_work)),
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
            work = _split_work_for_rank(full_work, rank=rank, world_size=world_size)
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
                max_length=int(args.max_length),
                format_loss_weight=float(args.format_loss_weight),
            )
            if packed is None:
                skipped += 1
                loss = _zero_loss_for_all_trainable_params(bundle)
                stats = {"denom": 0.0, "token_acc": 0.0, "value_token_acc": 0.0, "format_token_acc": 0.0}
            else:
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
                _append_jsonl(
                    output_dir / "train_metrics.jsonl",
                    {
                        "step": int(global_step),
                        "epoch": int(epoch + 1),
                        "samples": int(train_window.get("samples", 0)),
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
                    },
                )
                train_window.clear()
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
                        output_dir / "train_eval_metrics.jsonl",
                        {"step": int(global_step), "type": "teacher_forced", **metrics},
                    )
                    val_score = float(metrics.get("value_token_acc", 0.0))
                    if val_score > best_val_score:
                        best_val_score = val_score
                        ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_val")
                        print(f"[best-val] step={global_step} value_token_acc={val_score:.4f} adapter={ckpt_dir}")
                    print(
                        f"[eval] step={global_step}/{total_steps} split={args.eval_split} "
                        f"loss={metrics['loss']:.4f} value_acc={metrics['value_token_acc']:.4f} "
                        f"format_acc={metrics['format_token_acc']:.4f} "
                        f"focus_highway={metrics.get('focus_highway_acc', 0.0):.4f} "
                        f"focus_static_obstacle={metrics.get('focus_static_obstacle_acc', 0.0):.4f} "
                        f"focus_vulnerable={metrics.get('focus_vulnerable_acc', 0.0):.4f} "
                        f"focus_light={metrics.get('focus_traffic_light_abnormal_acc', 0.0):.4f}"
                    )
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
                        output_dir / "train_eval_metrics.jsonl",
                        {"step": int(global_step), "type": "generation", **generation_metrics},
                    )
                    generation_score = float(generation_metrics.get("exact_accuracy", 0.0))
                    generation_valid = float(generation_metrics.get("format_valid_rate", 0.0))
                    if generation_valid >= float(args.generation_format_valid_gate) and generation_score > best_generation_score:
                        best_generation_score = generation_score
                        ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name="best_generation")
                        print(
                            f"[best-generation] step={global_step} exact={generation_score:.4f} "
                            f"format_valid={generation_valid:.4f} adapter={ckpt_dir}"
                        )
                    print(
                        f"[generation-val] step={global_step}/{total_steps} "
                        f"format_valid={generation_metrics['format_valid_rate']:.4f} "
                        f"exact={generation_metrics['exact_accuracy']:.4f} "
                        f"focus_highway={generation_metrics.get('focus_highway_acc', 0.0):.4f} "
                        f"focus_static_obstacle={generation_metrics.get('focus_static_obstacle_acc', 0.0):.4f} "
                        f"focus_vulnerable={generation_metrics.get('focus_vulnerable_acc', 0.0):.4f} "
                        f"focus_light={generation_metrics.get('focus_traffic_light_abnormal_acc', 0.0):.4f}"
                    )
                if run_generation_eval and world_size > 1:
                    dist.barrier()
            if rank == 0 and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                ckpt_dir = _save_adapter(bundle, output_dir, args, step=global_step, name=f"checkpoint-{global_step}")
                print(f"[save] step={global_step} adapter={ckpt_dir}")
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

    p = argparse.ArgumentParser(description="Train sft_new_loop_phase1 fused eight-question LoRA")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_data/frame_index.jsonl"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument(
        "--output-dir",
        default="",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--history-rgb-mode", choices=HISTORY_RGB_MODES, default=DEFAULT_HISTORY_RGB_MODE)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--focus-balance-count", type=int, default=512)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="0 means train num_epochs over the balanced work list")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--eval-steps", type=int, default=100)
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
        default=1_000,
        help="run rank0 free-generation validation every N teacher-forced validation steps; 0 disables it",
    )
    p.add_argument("--generation-eval-balance-count", type=int, default=16)
    p.add_argument("--generation-eval-max-new-tokens", type=int, default=64)
    p.add_argument("--generation-format-valid-gate", type=float, default=0.99)
    p.add_argument("--save-steps", type=int, default=500)
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
