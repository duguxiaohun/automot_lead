#!/usr/bin/env python3
"""评估 sft_loop_phase1 的 base Qwen 或 LoRA adapter。

默认使用 8 桶均衡抽样，并保存 `cases.jsonl`、`metrics.json`、`summary.md` 和错例
RGB/输入输出，便于人工分析 prompt 失败原因。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import sys
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

from qwen3vl_local.sft_loop_phase1 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase1.prompts import (  # noqa: E402
    ANSWER_KEYS,
    PROMPT_NAME,
    SYSTEM_PROMPT,
    build_phase1_prompt,
    parse_phase1_output,
)
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv  # noqa: E402


def setup_distributed() -> Tuple[int, int, int]:
    """初始化可选 torchrun 多卡评估。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("sft_loop_phase1 multi-GPU eval requires CUDA.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


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
                    answers={key: bool((obj.get("answers") or {}).get(key, False)) for key in ANSWER_KEYS},
                )
            )
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise ValueError(f"no rows for split={split!r}: {path}")
    return rows


def _focus_key(row: FrameRow, focus: str) -> str:
    """返回八桶采样键。"""

    return f"{focus}:{'YES' if row.answers[focus] else 'NO'}"


def _balanced_cases(rows: Sequence[FrameRow], *, cases_per_bin: int, seed: int) -> List[Tuple[FrameRow, str]]:
    """按四问题 x YES/NO 采样评估 case。"""

    groups: Dict[str, List[Tuple[FrameRow, str]]] = {f"{key}:{value}": [] for key in ANSWER_KEYS for value in ("YES", "NO")}
    for row in rows:
        for focus in ANSWER_KEYS:
            groups[_focus_key(row, focus)].append((row, focus))
    rng = random.Random(f"{seed}:phase1_eval_balance:{len(rows)}:{cases_per_bin}")
    out: List[Tuple[FrameRow, str]] = []
    for key in sorted(groups):
        items = list(groups[key])
        if not items:
            continue
        rng.shuffle(items)
        target = max(1, int(cases_per_bin))
        if len(items) >= target:
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


def _messages(images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    """构造自由生成 chat。"""

    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _validate_phase1_adapter(adapter_dir: pathlib.Path, model_dir: pathlib.Path) -> Dict[str, Any]:
    """读取 Phase1 adapter 自描述配置。"""

    cfg_path = adapter_dir / "sft_loop_phase1_adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing phase1 adapter config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("route") != "sft_loop_phase1_four_visible_facts":
        raise ValueError(f"adapter route mismatch: {cfg.get('route')!r}")
    adapter_dataset = cfg.get("dataset_name")
    if adapter_dataset != DATASET_NAME:
        raise ValueError(f"adapter dataset_name mismatch: {adapter_dataset!r}")
    adapter_prompt = cfg.get("prompt_name")
    if adapter_prompt != PROMPT_NAME:
        raise ValueError(f"adapter prompt_name mismatch: {adapter_prompt!r}")
    return cfg


def load_eval_bundle(model_dir: pathlib.Path, adapter_dir: Optional[pathlib.Path], device: torch.device, *, merge_lora: bool) -> EvalBundle:
    """加载 base Qwen 和可选 Phase1 LoRA。"""

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
        _validate_phase1_adapter(pathlib.Path(adapter_dir), pathlib.Path(model_dir))
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    return EvalBundle(model=model, processor=processor, tokenizer=processor.tokenizer, device=device)


def _generate(bundle: EvalBundle, images: List[Image.Image], prompt: str, max_new_tokens: int) -> str:
    """单次 fresh prefill + decode。"""

    with torch.inference_mode():
        state = _kv_start_state(bundle, _messages(images, prompt))
        text, _, _ = _student_generate_kv(bundle, state, int(max_new_tokens))
    return text


def _bool_text(value: bool) -> str:
    """布尔转 YES/NO。"""

    return "YES" if bool(value) else "NO"


def _copy_error_case(case_dir: pathlib.Path, row: FrameRow, payload: Mapping[str, Any]) -> None:
    """保存错例 RGB 和 JSON，便于后续人工看图改 prompt。"""

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rgb_dir = case_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for idx, src in enumerate(row.history_rgb_paths):
        src_path = pathlib.Path(src)
        if src_path.exists():
            shutil.copy2(src_path, rgb_dir / f"history_{idx}_{src_path.name}")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """评估主流程。"""

    rank, local_rank, world_size = setup_distributed()
    output_dir = pathlib.Path(args.output_dir)
    if rank == 0 and output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"output dir is not empty: {output_dir}")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    rows = _read_rows(pathlib.Path(args.index), split=str(args.split), max_frames=int(args.max_frames))
    cases = _balanced_cases(rows, cases_per_bin=int(args.cases_per_bin), seed=int(args.seed))
    local_cases = cases[rank::world_size]
    device = torch.device(f"cuda:{local_rank}") if world_size > 1 else torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_eval_bundle(
        pathlib.Path(args.model_dir),
        pathlib.Path(args.adapter_dir) if args.adapter_dir else None,
        device,
        merge_lora=bool(args.merge_lora),
    )
    prompt = build_phase1_prompt(audit=bool(args.audit_prompt))

    total = 0
    exact = 0
    answer_counts: Dict[str, Counter[str]] = {key: Counter() for key in ANSWER_KEYS}
    focus_counts: Counter[str] = Counter()
    case_path = output_dir / (f"cases_rank{rank}.jsonl" if world_size > 1 else "cases.jsonl")
    error_root = output_dir / "error_cases"
    with case_path.open("w", encoding="utf-8") as f:
        for local_idx, (row, focus) in enumerate(local_cases):
            case_idx = rank + local_idx * max(1, world_size)
            images = _load_images(row.history_rgb_paths)
            raw = _generate(bundle, images, prompt, int(args.max_new_tokens))
            parsed = parse_phase1_output(raw)
            gt = {key: _bool_text(row.answers[key]) for key in ANSWER_KEYS}
            ok_by_key = {key: parsed.get(key) == gt[key] for key in ANSWER_KEYS}
            all_ok = all(ok_by_key.values())
            total += 1
            exact += int(all_ok)
            focus_counts[_focus_key(row, focus)] += 1
            for key in ANSWER_KEYS:
                answer_counts[key][f"gt/{gt[key]}"] += 1
                answer_counts[key][f"pred/{parsed.get(key) or 'INVALID'}"] += 1
                answer_counts[key]["correct"] += int(ok_by_key[key])
                answer_counts[key]["total"] += 1
                answer_counts[key][f"focus_correct/{focus}"] += int(focus == key and ok_by_key[key])
                answer_counts[key][f"focus_total/{focus}"] += int(focus == key)
            payload = {
                "case_index": case_idx,
                "focus_question": focus,
                "focus_bucket": _focus_key(row, focus),
                "scenario": row.scenario,
                "town": row.town,
                "route_id": row.route_id,
                "frame_id": row.frame_id,
                "rs": row.rs,
                "event": row.event,
                "history_rgb_paths": row.history_rgb_paths,
                "gt": gt,
                "parsed": parsed,
                "ok_by_key": ok_by_key,
                "all_ok": all_ok,
                "raw_output": raw,
                "prompt": prompt if bool(args.save_prompts) else None,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if not all_ok and bool(args.save_error_rgb):
                _copy_error_case(error_root / f"rank{rank}_{case_idx:05d}_{focus}_{row.scenario}_f{row.frame_id}", row, payload)

    local_payload = {
        "total": total,
        "exact": exact,
        "answer_counts": {key: dict(counter) for key, counter in answer_counts.items()},
        "focus_counts": dict(focus_counts),
        "case_path": str(case_path),
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
    answer_counts = {key: Counter() for key in ANSWER_KEYS}
    focus_counts = Counter()
    for item in gathered:
        for key in ANSWER_KEYS:
            answer_counts[key].update(item.get("answer_counts", {}).get(key, {}))
        focus_counts.update(item.get("focus_counts", {}))

    per_key = {}
    for key, counter in answer_counts.items():
        total_key = int(counter.get("total", 0))
        per_key[key] = {
            "accuracy": float(counter.get("correct", 0)) / max(1, total_key),
            "total": total_key,
            "counts": dict(counter),
        }
    metrics = {
        "dataset_name": DATASET_NAME,
        "prompt_name": PROMPT_NAME,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "audit_prompt": bool(args.audit_prompt),
        "total_cases": total,
        "exact_match_accuracy": float(exact) / max(1, total),
        "per_question": per_key,
        "focus_counts": dict(focus_counts),
        "cases_jsonl": str(case_path) if world_size == 1 else [str(item.get("case_path")) for item in gathered],
        "world_size": int(world_size),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# sft_loop_phase1 eval",
        "",
        f"- prompt_name: `{PROMPT_NAME}`",
        f"- adapter: `{args.adapter_dir or 'BASE_QWEN'}`",
        f"- cases: {total}",
        f"- exact_match_accuracy: {metrics['exact_match_accuracy']:.4f}",
        "",
        "| question | accuracy | total |",
        "|---|---:|---:|",
    ]
    for key in ANSWER_KEYS:
        lines.append(f"| {key} | {per_key[key]['accuracy']:.4f} | {per_key[key]['total']} |")
    lines.append("")
    lines.append(f"Cases: `{case_path.name}`")
    lines.append("Wrong examples with RGB are under `error_cases/` when enabled.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] metrics={output_dir / 'metrics.json'} cases={case_path}")
    cleanup_distributed()
    return metrics


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    p = argparse.ArgumentParser(description="Evaluate base Qwen or Phase1 LoRA on balanced four-question cases")
    p.add_argument("--index", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase1_data/frame_index.jsonl"))
    p.add_argument("--model-dir", default=str(_AUTOMOT_ROOT / "checkpoints/Qwen3-VL-4B-Instruct"))
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase1_eval/base_zero_shot_prompt"))
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--cases-per-bin", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--audit-prompt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-prompts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-error-rgb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=20260810)
    return p.parse_args()


def main() -> None:
    """CLI 入口。"""

    evaluate(parse_args())


if __name__ == "__main__":
    main()
