#!/usr/bin/env python3
"""评估 sft_loop_phase3 的 base Qwen 或 LoRA adapter。

默认按 phase3 class 抽样，并保存 `cases.jsonl`、`metrics.json`、`summary.md`
和错例 RGB/输入输出，便于人工分析 prompt 失败原因。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
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


def _maybe_apply_gpu_ids() -> None:
    """在 torch 初始化前应用 GPU_IDS pin 卡约定。"""

    pinned = ",".join(part.strip() for part in os.environ.get("GPU_IDS", "").split(",") if part.strip())
    if pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        print(f"[gpu] using GPU_IDS={pinned}")


_maybe_apply_gpu_ids()

import torch
import torch.distributed as dist
from PIL import Image

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
    SYSTEM_PROMPT,
    VARIANT_ORDER,
    VARIANT_WEIGHTS,
    PromptSpec,
    build_phase2_context_assistant,
    build_phase2_context_user_prompt,
    build_phase3_messages,
    build_phase3_prompt,
    make_prompt_spec,
    parse_phase3_output,
    phase3_prompt_sha256,
    prompt_spec_to_json,
    spec_metric_items,
)
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


def setup_distributed() -> Tuple[int, int, int]:
    """初始化可选 torchrun 多卡评估。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_loop_phase3 multi-GPU eval requires CUDA.")
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


@dataclass
class FrameRow:
    """评估帧。"""

    idx: int
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
    """一条增强评估 case。"""

    row: FrameRow
    spec: PromptSpec
    balance_key: str


@dataclass
class EvalBundle:
    """模型、processor、tokenizer 和设备。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self) -> Any:
        """兼容 KV helper 对 PEFT wrapper 的访问。"""

        return getattr(self.model, "module", self.model)


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
                raise ValueError(f"dataset_name mismatch: {row_dataset!r}")
            if str(obj.get("split")) != str(split):
                continue
            rows.append(
                FrameRow(
                    idx=len(rows),
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
        raise ValueError(f"no rows for split={split!r}: {path}")
    return rows


def _work_item_seed(row: FrameRow, *parts: object) -> str:
    """返回增强 spec 的稳定种子字段。"""

    return ":".join(
        [row.scenario, row.route_id, str(row.frame_id), row.rs, *[str(part) for part in parts]]
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
    """统计 split 中可用的 class 与答案桶。"""

    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"class/{_target_class(row)}"] += 1
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


def _balanced_cases(rows: Sequence[FrameRow], *, cases_per_bin: int, seed: int) -> List[WorkItem]:
    """按 phase3 class 抽样评估 case。"""

    groups: Dict[str, List[WorkItem]] = defaultdict(list)
    for row in rows:
        item = _make_all_item(row, _target_class(row), seed=seed)
        groups.setdefault(item.balance_key, []).append(item)
    if not groups:
        raise ValueError("cannot build Phase3 evaluation cases: no class buckets available")
    rng = random.Random(f"{seed}:phase3_eval_balance:{len(rows)}:{cases_per_bin}")
    out: List[WorkItem] = []
    for key in sorted(groups):
        items = list(groups[key])
        rng.shuffle(items)
        target = int(cases_per_bin)
        if target <= 0:
            out.extend(items)
        elif len(items) >= target:
            out.extend(items[:target])
        else:
            repeated = [items[i % len(items)] for i in range(target)]
            rng.shuffle(repeated)
            out.extend(repeated)
    rng.shuffle(out)
    return out

def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    """读取 RGB history。"""

    return [Image.open(path).convert("RGB") for path in paths]


def _messages(
    images: List[Image.Image],
    *,
    spec: PromptSpec,
    audit: bool,
    history_rgb_mode: str,
) -> List[Dict[str, Any]]:
    """构造自由生成 chat。"""

    return build_phase3_messages(
        images=images,
        spec=spec,
        audit=bool(audit),
        history_rgb_mode=history_rgb_mode,
        target=None,
    )


def _validate_phase3_adapter(adapter_dir: pathlib.Path, model_dir: pathlib.Path) -> Dict[str, Any]:
    """读取 Phase3 adapter 自描述配置。"""

    cfg_path = adapter_dir / "sft_loop_phase3_adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing phase3 augment adapter config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("route") != "sft_loop_phase3_event_gate":
        raise ValueError(f"adapter route mismatch: {cfg.get('route')!r}")
    adapter_dataset = cfg.get("dataset_name")
    if adapter_dataset != DATASET_NAME:
        raise ValueError(f"adapter dataset_name mismatch: {adapter_dataset!r}")
    adapter_prompt = cfg.get("prompt_name")
    if adapter_prompt != PROMPT_NAME:
        raise ValueError(f"adapter prompt_name mismatch: {adapter_prompt!r}")
    return cfg


def _adapter_config_path(adapter_dir: pathlib.Path) -> pathlib.Path:
    """返回 Phase3 adapter 配置文件路径。"""

    return adapter_dir / "sft_loop_phase3_adapter_config.json"


def _resolve_adapter_dir(adapter_dir: pathlib.Path) -> Tuple[pathlib.Path, str]:
    """允许传 run/latest 目录，自动选择 best_generation/best_val/final。"""

    path = pathlib.Path(adapter_dir)
    checked = []
    direct = _adapter_config_path(path)
    checked.append(str(direct))
    if direct.exists():
        return path, "exact_adapter_dir"

    for child in ("best_generation", "best_val", "final"):
        candidate = path / child
        cfg_path = _adapter_config_path(candidate)
        checked.append(str(cfg_path))
        if cfg_path.exists():
            return candidate, f"run_dir_{child}"

    if path.name in {"best_generation", "best_val"}:
        for child in ("best_val", "final"):
            fallback = path.parent / child
            if fallback == path:
                continue
            cfg_path = _adapter_config_path(fallback)
            checked.append(str(cfg_path))
            if cfg_path.exists():
                return fallback, f"missing_{path.name}_fallback_{child}"

    raise FileNotFoundError(
        "cannot resolve Phase3 adapter. Pass either an adapter dir, "
        "or a run dir such as checkpoints/sft_loop_phase3_runs/latest. "
        f"Checked: {checked}"
    )


def _resolve_history_rgb_mode(
    requested_mode: Optional[str], adapter_cfg: Optional[Mapping[str, Any]]
) -> Tuple[str, str]:
    """Resolve the RGB input contract, with LoRA configuration as the authority."""

    if adapter_cfg is None:
        return validate_history_rgb_mode(requested_mode or DEFAULT_HISTORY_RGB_MODE), "base_cli"
    if requested_mode is not None:
        raise ValueError(
            "--history-rgb-mode is only for base-Qwen evaluation. LoRA evaluation reads the "
            "persisted history_rgb_mode from sft_loop_phase3_adapter_config.json."
        )
    persisted = adapter_cfg.get("history_rgb_mode", DEFAULT_HISTORY_RGB_MODE)
    source = "adapter_config" if "history_rgb_mode" in adapter_cfg else "legacy_adapter_default_4rgb"
    return validate_history_rgb_mode(str(persisted)), source


def load_eval_bundle(model_dir: pathlib.Path, adapter_dir: Optional[pathlib.Path], device: torch.device, *, merge_lora: bool) -> EvalBundle:
    """加载 base Qwen 和可选 Phase3 LoRA。"""

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    kwargs = {"local_files_only": True, "trust_remote_code": True}
    try:
        model = ModelClass.from_pretrained(str(model_dir), dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = ModelClass.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, **kwargs)
    if adapter_dir is not None:
        _validate_phase3_adapter(pathlib.Path(adapter_dir), pathlib.Path(model_dir))
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _generate(
    bundle: EvalBundle,
    images: List[Image.Image],
    *,
    spec: PromptSpec,
    audit: bool,
    history_rgb_mode: str,
    max_new_tokens: int,
) -> str:
    """单次 fresh prefill + decode。"""

    with torch.inference_mode():
        state = _kv_start_state(
            bundle,
            _messages(images, spec=spec, audit=bool(audit), history_rgb_mode=history_rgb_mode),
        )
        text, _, _ = _student_generate_kv(bundle, state, int(max_new_tokens))
    return text


def _bool_text(value: bool) -> str:
    """布尔转 YES/NO。"""

    return "YES" if bool(value) else "NO"


def _parsed_text(value: Optional[bool]) -> Optional[str]:
    """把严格 parser 的 bool 输出转换到评测使用的 YES/NO 文本域。"""

    return None if value is None else _bool_text(value)


def _binary_report(counter: Mapping[str, int]) -> Dict[str, Any]:
    """生成 YES 正类的二分类 TP/FP/TN/FN 和混淆矩阵。"""

    tp = int(counter.get("cm/YES/YES", 0))
    fp = int(counter.get("cm/NO/YES", 0))
    fn = int(counter.get("cm/YES/NO", 0)) + int(counter.get("cm/YES/INVALID", 0))
    tn = int(counter.get("cm/NO/NO", 0))
    total = int(counter.get("total", 0))
    correct = int(counter.get("correct", 0))
    precision = float(tp) / max(1.0, float(tp + fp))
    recall = float(tp) / max(1.0, float(tp + fn))
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "positive": "YES",
        "negative": "NO",
        "total": total,
        "accuracy": float(correct) / max(1.0, float(total)),
        "invalid_rate": float(counter.get("pred/INVALID", 0)) / max(1.0, float(total)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "YES": {
                "YES": int(counter.get("cm/YES/YES", 0)),
                "NO": int(counter.get("cm/YES/NO", 0)),
                "INVALID": int(counter.get("cm/YES/INVALID", 0)),
            },
            "NO": {
                "YES": int(counter.get("cm/NO/YES", 0)),
                "NO": int(counter.get("cm/NO/NO", 0)),
                "INVALID": int(counter.get("cm/NO/INVALID", 0)),
            },
        },
        "counts": dict(counter),
    }


def _update_binary_counter(counter: Counter[str], gt_value: str, pred_value: Optional[str]) -> None:
    """累计一个 YES/NO/INVALID 二分类观测。"""

    pred = pred_value if pred_value in ("YES", "NO") else "INVALID"
    counter[f"gt/{gt_value}"] += 1
    counter[f"pred/{pred}"] += 1
    counter[f"cm/{gt_value}/{pred}"] += 1
    counter["correct"] += int(pred == gt_value)
    counter["total"] += 1


def _dynamic_answer_pattern(values: Mapping[str, Optional[str]]) -> str:
    """把当前被问到的多行答案折叠成诊断 pattern。"""

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
    """累计 phase3 event-gate 的答案模式与 invalid 联合约束。"""

    del row, raw_output
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
    """把 phase3 pattern counter 整理成 metrics.json 结构。"""

    out: Dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        total = int(counter.get(f"{variant}/total", 0))
        denom = max(1.0, float(total))
        invalid_total = max(1.0, float(counter.get(f"{variant}/invalid_gt_total", 0)))
        pred_invalid_total = max(1.0, float(counter.get(f"{variant}/pred_invalid_yes_total", 0)))
        out[variant] = {
            "total": total,
            "pattern_exact_accuracy": float(counter.get(f"{variant}/pattern_exact", 0)) / denom,
            "pred_invalid_rate": float(counter.get(f"{variant}/pred_invalid", 0)) / denom,
            "invalid_gt_total": int(counter.get(f"{variant}/invalid_gt_total", 0)),
            "invalid_line_yes_rate": float(counter.get(f"{variant}/invalid_pred_line_yes", 0)) / invalid_total,
            "invalid_ue_all_no_rate": float(counter.get(f"{variant}/invalid_ue_all_no", 0)) / invalid_total,
            "invalid_joint_ok_rate": float(counter.get(f"{variant}/invalid_joint_ok", 0)) / invalid_total,
            "pred_invalid_yes_total": int(counter.get(f"{variant}/pred_invalid_yes_total", 0)),
            "pred_invalid_yes_ue_all_no_rate": float(counter.get(f"{variant}/pred_invalid_yes_ue_all_no", 0)) / pred_invalid_total,
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


def _task_case_dir(root: pathlib.Path, focus: str, case_idx: int, row: FrameRow) -> pathlib.Path:
    """按主任务组织 RGB/case 输出目录。"""

    safe_scenario = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in row.scenario)
    return root / focus / f"case_{case_idx:05d}_{safe_scenario}_f{row.frame_id}"


def _copy_case_rgb(case_dir: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """保存 case JSON 和实际送入模型的 RGB，便于审计输入合同。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rgb_dir = case_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    selected_indices = payload.get("history_rgb_selected_indices") or []
    selected_paths = payload.get("history_rgb_paths_used") or []
    for source_idx, src in zip(selected_indices, selected_paths):
        src_path = pathlib.Path(src)
        if src_path.exists():
            shutil.copy2(src_path, rgb_dir / f"history_source_{source_idx}_{src_path.name}")


def _prepare_output_dir(output_dir: pathlib.Path, *, overwrite: bool) -> None:
    """准备 eval 输出目录；overwrite 时清掉旧 rank/case/RGB 残留。"""

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output dir is not empty: {output_dir}; pass --overwrite or remove it first")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_output_dir(base_dir: pathlib.Path, *, timestamp_output: bool, rank: int, world_size: int) -> pathlib.Path:
    """可选在输出目录下追加一个由 rank0 统一生成的时间戳子目录。"""

    if not timestamp_output:
        return base_dir
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
    if world_size > 1:
        holder = [tag]
        dist.broadcast_object_list(holder, src=0)
        tag = str(holder[0])
    return base_dir / str(tag)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """评估主流程。"""

    rank, local_rank, world_size = setup_distributed()
    adapter_cfg = (
        _validate_phase3_adapter(pathlib.Path(args.adapter_dir), pathlib.Path(args.model_dir))
        if args.adapter_dir
        else None
    )
    history_rgb_mode, history_rgb_mode_source = _resolve_history_rgb_mode(
        args.history_rgb_mode, adapter_cfg
    )
    output_dir = _resolve_output_dir(
        pathlib.Path(args.output_dir),
        timestamp_output=bool(args.timestamp_output),
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    if world_size > 1:
        ddp_barrier(local_rank)
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    raw_focus_bin_availability = _raw_focus_bin_counts(rows)
    cases = _balanced_cases(rows, cases_per_bin=int(args.cases_per_bin), seed=int(args.seed))
    local_cases = cases[rank::world_size]
    device = torch.device(f"cuda:{local_rank}") if world_size > 1 else torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_eval_bundle(
        pathlib.Path(args.model_dir),
        pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
        device,
        merge_lora=bool(args.merge_lora),
    )
    total = 0
    exact = 0
    metric_names = list(ANSWER_KEYS)
    metric_counts: Dict[str, Counter[str]] = {key: Counter() for key in metric_names}
    variant_counts: Dict[str, Counter[str]] = {key: Counter() for key in VARIANT_ORDER}
    balance_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    case_path = output_dir / (f"cases_rank{rank}.jsonl" if world_size > 1 else "cases.jsonl")
    variant_case_root = output_dir / "variant_cases"
    variant_case_paths = {
        variant: variant_case_root / variant / (f"cases_rank{rank}.jsonl" if world_size > 1 else "cases.jsonl")
        for variant in VARIANT_ORDER
    }
    error_root = output_dir / "error_cases"
    rgb_root = output_dir / "rgb_cases"
    variant_fps = {}
    for variant, variant_path in variant_case_paths.items():
        variant_path.parent.mkdir(parents=True, exist_ok=True)
        variant_fps[variant] = variant_path.open("w", encoding="utf-8")
    with case_path.open("w", encoding="utf-8") as f:
        try:
            for local_idx, item in enumerate(local_cases):
                row = item.row
                spec = item.spec
                case_idx = rank + local_idx * max(1, world_size)
                used_history_rgb_paths = select_history_rgb_paths(
                    row.history_rgb_paths, history_rgb_mode
                )
                images = _load_images(used_history_rgb_paths)
                prompt = build_phase3_prompt(
                    spec=spec,
                    audit=bool(args.audit_prompt),
                    history_rgb_mode=history_rgb_mode,
                    include_inline_phase2_context=False,
                )
                phase2_user_prompt = build_phase2_context_user_prompt(spec=spec)
                phase2_assistant = build_phase2_context_assistant(spec=spec)
                raw = _generate(
                    bundle,
                    images,
                    spec=spec,
                    audit=bool(args.audit_prompt),
                    history_rgb_mode=history_rgb_mode,
                    max_new_tokens=int(args.max_new_tokens),
                )
                parsed_bool = parse_phase3_output(raw, spec=spec)
                parsed = {key: _parsed_text(parsed_bool.get(key)) for key in spec.output_keys}
                gt = {q.output_key: _bool_text(q.answer) for q in spec.questions}
                ok_by_key = {key: parsed.get(key) == gt[key] for key in spec.output_keys}
                all_ok = all(ok_by_key.values())
                total += 1
                exact += int(all_ok)
                balance_counts[item.balance_key] += 1
                variant_counts[spec.variant]["total"] += 1
                variant_counts[spec.variant]["exact"] += int(all_ok)
                variant_counts[spec.variant]["format_valid"] += int(all(value is not None for value in parsed_bool.values()))
                for output_key, metric_key, answer in spec_metric_items(spec):
                    _update_binary_counter(metric_counts[metric_key], _bool_text(answer), parsed.get(output_key))
                _update_pattern_counters(
                    pattern_counts,
                    spec=spec,
                    row=row,
                    gt=gt,
                    parsed=parsed,
                    raw_output=raw,
                )
                payload = {
                    "case_index": case_idx,
                    "augment_variant": spec.variant,
                    "augment_balance_key": item.balance_key,
                    "augment_spec": prompt_spec_to_json(spec),
                    "scenario": row.scenario,
                    "town": row.town,
                    "route_id": row.route_id,
                    "frame_id": row.frame_id,
                    "rs": row.rs,
                    "event": row.event,
                    "history_rgb_mode": history_rgb_mode,
                    "history_rgb_count": len(history_rgb_indices(history_rgb_mode)),
                    "history_rgb_selected_indices": list(history_rgb_indices(history_rgb_mode)),
                    "history_rgb_paths_used": used_history_rgb_paths,
                    "history_rgb_paths_all4": row.history_rgb_paths,
                    "latest_rgb_path": row.latest_rgb_path,
                    "phase3_answers": {key: _bool_text(row.answers.get(key, False)) for key in ANSWER_KEYS},
                    "gt": gt,
                    "parsed": parsed,
                    "ok_by_key": ok_by_key,
                    "all_ok": all_ok,
                    "raw_output": raw,
                    "phase2_user_prompt": phase2_user_prompt if bool(args.save_prompts) else None,
                    "phase2_assistant": phase2_assistant if bool(args.save_prompts) else None,
                    "phase3_user_prompt": prompt if bool(args.save_prompts) else None,
                    "actual_chat_messages": (
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    *[
                                        {"type": "image", "source_path": path}
                                        for path in used_history_rgb_paths
                                    ],
                                    {"type": "text", "text": phase2_user_prompt},
                                ],
                            },
                            {"role": "assistant", "content": phase2_assistant},
                            {"role": "user", "content": prompt},
                        ]
                        if bool(args.save_prompts)
                        else None
                    ),
                    "prompt": prompt if bool(args.save_prompts) else None,
                }
                encoded = json.dumps(payload, ensure_ascii=False)
                f.write(encoded + "\n")
                variant_fps[spec.variant].write(encoded + "\n")
                if not all_ok and bool(args.save_error_rgb):
                    _copy_case_rgb(_task_case_dir(error_root, spec.variant, case_idx, row), payload)
                if bool(args.save_all_rgb):
                    _copy_case_rgb(_task_case_dir(rgb_root, spec.variant, case_idx, row), payload)
        finally:
            for fp in variant_fps.values():
                fp.close()

    local_payload = {
        "total": total,
        "exact": exact,
        "metric_counts": {key: dict(counter) for key, counter in metric_counts.items()},
        "variant_counts": {key: dict(counter) for key, counter in variant_counts.items()},
        "balance_counts": dict(balance_counts),
        "pattern_counts": dict(pattern_counts),
        "case_path": str(case_path),
        "variant_case_paths": {key: str(path) for key, path in variant_case_paths.items()},
    }
    gathered: List[Dict[str, Any]] = [local_payload]
    if world_size > 1:
        gathered = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_payload)
    if rank != 0:
        cleanup_distributed()
        return {}

    total = sum(int(item.get("total", 0)) for item in gathered)
    exact = sum(int(item.get("exact", 0)) for item in gathered)
    metric_counts = {key: Counter() for key in metric_names}
    variant_counts = {key: Counter() for key in VARIANT_ORDER}
    balance_counts = Counter()
    pattern_counts = Counter()
    for item in gathered:
        for key in metric_names:
            metric_counts[key].update(item.get("metric_counts", {}).get(key, {}))
        for key in VARIANT_ORDER:
            variant_counts[key].update(item.get("variant_counts", {}).get(key, {}))
        balance_counts.update(item.get("balance_counts", {}))
        pattern_counts.update(item.get("pattern_counts", {}))

    per_key = {}
    for key, counter in metric_counts.items():
        per_key[key] = _binary_report(counter)
    variant_reports = {}
    for variant in VARIANT_ORDER:
        counter = variant_counts[variant]
        cases_n = int(counter.get("total", 0))
        variant_reports[variant] = {
            "cases": cases_n,
            "exact_match_accuracy": float(counter.get("exact", 0)) / max(1.0, float(cases_n)),
            "format_valid_rate": float(counter.get("format_valid", 0)) / max(1.0, float(cases_n)),
            "counts": dict(counter),
        }
    variant_total_counts = {variant: int(variant_counts[variant].get("total", 0)) for variant in VARIANT_ORDER}
    answer_pattern_report = _pattern_report(pattern_counts)
    invalid_contract_report = {
        variant: {
            "invalid_gt_total": answer_pattern_report[variant]["invalid_gt_total"],
            "invalid_line_yes_rate": answer_pattern_report[variant]["invalid_line_yes_rate"],
            "invalid_ue_all_no_rate": answer_pattern_report[variant]["invalid_ue_all_no_rate"],
            "invalid_joint_ok_rate": answer_pattern_report[variant]["invalid_joint_ok_rate"],
            "pred_invalid_yes_total": answer_pattern_report[variant]["pred_invalid_yes_total"],
            "pred_invalid_yes_ue_all_no_rate": answer_pattern_report[variant]["pred_invalid_yes_ue_all_no_rate"],
        }
        for variant in VARIANT_ORDER
    }
    metrics = {
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "prompt_mode": "audit" if bool(args.audit_prompt) else "production",
        "history_rgb_mode": history_rgb_mode,
        "history_rgb_mode_source": history_rgb_mode_source,
        "history_rgb_count": len(history_rgb_indices(history_rgb_mode)),
        "history_rgb_selected_indices": list(history_rgb_indices(history_rgb_mode)),
        "production_prompt_sha256": phase3_prompt_sha256(
            audit=False, history_rgb_mode=history_rgb_mode
        ),
        "eval_prompt_sha256": phase3_prompt_sha256(
            audit=bool(args.audit_prompt), history_rgb_mode=history_rgb_mode
        ),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "adapter_dir_resolve_source": getattr(args, "adapter_dir_resolve_source", None) if args.adapter_dir else None,
        "adapter_production_prompt_sha256": (
            adapter_cfg.get("production_prompt_sha256") if adapter_cfg is not None else None
        ),
        "adapter_prompt_matches_current_production": (
            adapter_cfg.get("production_prompt_sha256")
            == phase3_prompt_sha256(audit=False, history_rgb_mode=history_rgb_mode)
            if adapter_cfg is not None and adapter_cfg.get("production_prompt_sha256")
            else None
        ),
        "audit_prompt": bool(args.audit_prompt),
        "sampling_contract": "Phase3 eval samples one RS-gated event prompt per frame. The dataset index is built with UE1/UE3/UE5/UE6 positive classes at 1:1:1:1, RE as all-NO regular negatives, and wrong-RS invalid rows at the configured ratio.",
        "sampling_verification": {
            "raw_focus_bin_availability": raw_focus_bin_availability,
            "target_cases_per_bin": int(args.cases_per_bin),
            "variant_target_weights": dict(VARIANT_WEIGHTS),
            "sampled_variant_counts": variant_total_counts,
            "sampled_balance_keys": dict(balance_counts),
        },
        "output_dir": str(output_dir),
        "total_cases": total,
        "exact_match_accuracy": float(exact) / max(1, total),
        "per_question": per_key,
        "variant_reports": variant_reports,
        "invalid_contract": invalid_contract_report,
        "answer_pattern_diagnostics": answer_pattern_report,
        "cases_jsonl": str(case_path) if world_size == 1 else [str(item.get("case_path")) for item in gathered],
        "variant_cases_jsonl": (
            {key: str(path) for key, path in variant_case_paths.items()}
            if world_size == 1
            else {
                key: [str((output_dir / "variant_cases" / key / f"cases_rank{rank_idx}.jsonl")) for rank_idx in range(world_size)]
                for key in VARIANT_ORDER
            }
        ),
        "error_rgb_layout": "error_cases/<VARIANT>/case_<id>_<scenario>_f<frame>/rgb/history_source_<original_index>_*.jpg",
        "all_rgb_layout": "rgb_cases/<VARIANT>/case_<id>_<scenario>_f<frame>/rgb/history_source_<original_index>_*.jpg when --save-all-rgb is enabled",
        "world_size": int(world_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# sft_loop_phase3 eval",
        "",
        f"- prompt_name: `{PROMPT_NAME}`",
        f"- prompt_mode: `{'audit' if bool(args.audit_prompt) else 'production'}`",
        f"- history_rgb_mode: `{history_rgb_mode}` ({len(history_rgb_indices(history_rgb_mode))} images; original indices {list(history_rgb_indices(history_rgb_mode))})",
        f"- history_rgb_mode_source: `{history_rgb_mode_source}`",
        f"- eval_prompt_sha256: `{metrics['eval_prompt_sha256']}`",
        f"- adapter: `{args.adapter_dir or 'BASE_QWEN'}`",
        f"- adapter_dir_resolve_source: `{metrics['adapter_dir_resolve_source'] or 'n/a'}`",
        f"- adapter_production_prompt_sha256: `{metrics['adapter_production_prompt_sha256'] or ('n/a (base)' if not args.adapter_dir else 'unknown (legacy adapter)')}`",
        f"- adapter_prompt_matches_current_production: `{metrics['adapter_prompt_matches_current_production'] if args.adapter_dir else 'n/a (base)'}`",
        f"- cases: {total}",
        f"- exact_match_accuracy: {metrics['exact_match_accuracy']:.4f}",
        f"- sampling: `{metrics['sampling_contract']}`",
        f"- invalid_joint_ok_rate: {invalid_contract_report.get('all_random_order', {}).get('invalid_joint_ok_rate', 0.0):.4f}",
        f"- invalid_ue_all_no_rate: {invalid_contract_report.get('all_random_order', {}).get('invalid_ue_all_no_rate', 0.0):.4f}",
        "",
        "## Variant Metrics",
        "",
        "| variant | cases | format_valid | exact |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        report = variant_reports[variant]
        lines.append(f"| {variant} | {report['cases']} | {report['format_valid_rate']:.4f} | {report['exact_match_accuracy']:.4f} |")
    lines.extend(
        [
            "",
        "## Answer Pattern Diagnostics",
            "",
            "- Pattern diagnostics are informational and do not constrain parsing, loss, or scoring.",
        ]
    )
    lines.extend(["", "## Question Metrics", "", "| question | accuracy | precision_yes | recall_yes | f1_yes | total |", "|---|---:|---:|---:|---:|---:|"])
    for key in metric_names:
        report = per_key[key]
        lines.append(
            f"| {key} | {report['accuracy']:.4f} | {report['precision']:.4f} | "
            f"{report['recall']:.4f} | {report['f1']:.4f} | {report['total']} |"
        )
    lines.append("")
    lines.append(f"Cases: `{case_path.name}`")
    lines.append("Per-variant case JSONL files are under `variant_cases/<VARIANT>/`; Phase3 currently uses one event-gate variant.")
    lines.append("Wrong examples with RGB are under `error_cases/<VARIANT>/` when enabled.")
    lines.append("Copied RGB files are exactly the images fed to the model; their filenames retain original four-frame indices.")
    lines.append("All evaluated RGB histories are copied under `rgb_cases/<TASK>/` only when `--save-all-rgb` is enabled.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] metrics={output_dir / 'metrics.json'} cases={case_path}")
    cleanup_distributed()
    return metrics


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Evaluate base Qwen or Phase3 LoRA on balanced RS-gated event cases")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase3_data/frame_index.jsonl"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument("--adapter-dir", default="")
    p.add_argument(
        "--output-dir",
        default="",
        help="optional override; otherwise selects the fixed final base/LoRA result directory",
    )
    p.add_argument("--split", default="test")
    p.add_argument(
        "--history-rgb-mode",
        choices=HISTORY_RGB_MODES,
        default=None,
        help="base Qwen only: 4rgb uses source frames [0,1,2,3]; 2rgb_endpoints uses [0,3]. LoRA reads its checkpoint config.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--cases-per-bin", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--audit-prompt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-error-rgb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-all-rgb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timestamp-output", action=argparse.BooleanOptionalAction, default=True, help="write results under --output-dir/YYYYmmdd_HHMMSS")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()
    if args.adapter_dir:
        resolved_adapter_dir, resolve_source = _resolve_adapter_dir(pathlib.Path(args.adapter_dir))
        args.adapter_dir = str(resolved_adapter_dir)
        args.adapter_dir_resolve_source = resolve_source
    else:
        args.adapter_dir_resolve_source = "base"
    adapter_cfg = (
        _validate_phase3_adapter(pathlib.Path(args.adapter_dir), pathlib.Path(args.model_dir))
        if args.adapter_dir
        else None
    )
    history_rgb_mode, _ = _resolve_history_rgb_mode(args.history_rgb_mode, adapter_cfg)
    if not args.output_dir:
        name = "lora_event_gate_final" if args.adapter_dir else "base_event_gate_final"
        name += f"_{history_rgb_mode_tag(history_rgb_mode)}"
        if args.audit_prompt:
            name += "_audit"
        args.output_dir = str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase3_eval" / name)
    return args


def main() -> None:
    """CLI 入口。"""

    evaluate(parse_args())


if __name__ == "__main__":
    main()
