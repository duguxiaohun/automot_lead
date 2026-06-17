"""Evaluate SFT v2 LoRA by free-generating SCENE/STATUS/SUBGOAL."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from PIL import Image

from qwen3vl_local.engine import LocalQwen3VLInstructEngine
from qwen3vl_local.sft_v2.prompts import (
    STATUS_SYSTEM_PROMPT,
    build_status_user_prompt,
    extract_gt,
    parse_output,
    validate_choice,
)
from qwen3vl_local.sft_v2.train import strip_image_placeholders


def _cli_value(name: str) -> Optional[str]:
    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _pick_idle_gpus(n: int = 1) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append((int(parts[1]), int(parts[2]), parts[0]))
            except ValueError:
                pass
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


def _normalize_gpu_ids(value: str) -> str:
    return ",".join(part.strip() for part in value.split(",") if part.strip())


def _maybe_set_idle_gpu_mask() -> None:
    device = _cli_value("--device")
    if device and device.lower() not in ("", "auto"):
        return
    pinned = _normalize_gpu_ids(os.environ.get("GPU_IDS", ""))
    if pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        print(f"[gpu] using GPU_IDS={pinned}")
        return
    selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[gpu] auto CUDA_VISIBLE_DEVICES={selected}")


_maybe_set_idle_gpu_mask()


def load_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parsed_gt = extract_gt(row["messages"][2]["content"])
            choice_meta = row.get("choice_meta") or {}
            gt = {
                "scene": choice_meta.get("target_scene") or parsed_gt["scene"],
                "status": choice_meta.get("target_status") or parsed_gt["status"],
                "subgoal": choice_meta.get("target_subgoal") or parsed_gt["subgoal"],
            }
            stages = row.get("stage_messages")
            if not stages or "scene" not in stages or "status" not in stages:
                raise ValueError(f"SFT v2 row missing stage_messages.scene/status in {path}")
            scene_msgs = stages["scene"]
            images = list(row.get("images", []))
            scene_system_prompt = scene_msgs[0]["content"]
            scene_user_prompt = strip_image_placeholders(scene_msgs[1]["content"])
            rows.append({
                "sample_idx": idx,
                "scenario": row.get("scenario", gt["scene"]),
                "run_id": row.get("run_id", ""),
                "anchor": int(row.get("anchor", -1)),
                "images": images,
                "scene_system_prompt": scene_system_prompt,
                "scene_user_prompt": scene_user_prompt,
                "memory_in_status": choice_meta.get("memory_in_status", ""),
                "memory_in_subgoal": choice_meta.get("memory_in_subgoal", ""),
                "gt": gt,
                "is_transition_sample": bool(row.get("is_transition_sample", False)),
            })
    return rows


def load_images(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def _text_from_ids(engine: LocalQwen3VLInstructEngine, token_ids: Any) -> str:
    return engine.processor.batch_decode(token_ids, skip_special_tokens=True)[0].lstrip("\n ")


def _generate_full_multiturn(
    engine: LocalQwen3VLInstructEngine,
    row: Dict[str, Any],
    images: List[Image.Image],
    raw_scene: str,
    status_user_prompt: str,
) -> str:
    """Fallback: regenerate status from the full two-turn context."""

    messages = engine.build_messages(row["scene_system_prompt"], row["scene_user_prompt"], images)
    messages.extend([
        {"role": "assistant", "content": raw_scene.strip()},
        {"role": "user", "content": status_user_prompt},
    ])
    chat_text = engine.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = engine.prepare_inputs(chat_text, images)
    trace = type("_Trace", (), {
        "prefill_cache_summary": {},
        "final_cache_summary": {},
        "decode_steps": [],
        "prefill_cache_file": None,
        "final_cache_file": None,
        "system_prompt_cache_enabled": False,
        "system_prompt_cache_used": False,
        "system_prompt_cache_note": None,
        "system_prompt_cache_summary": {},
    })()
    prefill_outputs = engine.prefill(inputs)
    new_ids = engine.decode(inputs, prefill_outputs, trace)
    return _text_from_ids(engine, new_ids)


def _render_status_suffix(engine: LocalQwen3VLInstructEngine, status_user_prompt: str) -> str:
    """Render the follow-up user turn and assistant generation prompt."""

    return engine.processor.apply_chat_template(
        [{"role": "user", "content": status_user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_status_with_scene_kv(
    engine: LocalQwen3VLInstructEngine,
    row: Dict[str, Any],
    images: List[Image.Image],
    raw_scene: str,
    status_user_prompt: str,
) -> str:
    """Continue from the scene-generation KV cache and decode STATUS/SUBGOAL."""

    import torch

    state = getattr(engine, "_last_decode_state", None)
    if not state:
        return _generate_full_multiturn(engine, row, images, raw_scene, status_user_prompt)

    prefix_ids = state.get("cache_input_ids", state["decoded_input_ids"])
    prefix_len = int(prefix_ids.shape[1])
    decoded_ids = state["decoded_input_ids"].to(prefix_ids.device)
    pending_generated_ids = decoded_ids[:, prefix_len:]
    suffix_text = _render_status_suffix(engine, status_user_prompt)
    if not suffix_text.startswith("\n"):
        suffix_text = "\n" + suffix_text
    if pending_generated_ids.shape[1] == 0:
        # max_gen_tokens 用完但没命中 EOS：assistant turn 还没闭合，需要手动补 <|im_end|>。
        # 这里固定写死 <|im_end|>，不依赖 tokenizer.eos_token——后者在某些 Qwen 配置里
        # 是 <|endoftext|>，会破坏 chat template 的 turn 结构。
        suffix_text = "<|im_end|>" + suffix_text
    suffix_ids = engine.processor.tokenizer(
        suffix_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(prefix_ids.device)
    if pending_generated_ids.shape[1] > 0:
        suffix_ids = torch.cat([pending_generated_ids, suffix_ids], dim=1)
    if suffix_ids.shape[1] == 0:
        return ""
    old_attention = state["attention_mask"]
    if old_attention is None:
        old_attention = torch.ones_like(prefix_ids, device=prefix_ids.device)
    suffix_ids = suffix_ids.to(prefix_ids.device)
    attention_mask = torch.cat(
        [old_attention, torch.ones_like(suffix_ids, device=old_attention.device)],
        dim=1,
    )
    decoded_input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    cache_position = torch.arange(
        prefix_len,
        prefix_len + suffix_ids.shape[1],
        device=prefix_ids.device,
    )
    model_inputs = engine.model.prepare_inputs_for_generation(
        decoded_input_ids,
        past_key_values=state["past_key_values"],
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=True,
    )
    if state.get("rope_deltas") is not None and "rope_deltas" not in model_inputs:
        model_inputs["rope_deltas"] = state["rope_deltas"]
    outputs = engine.model(**model_inputs, return_dict=True)
    trace = type("_Trace", (), {
        "prefill_cache_summary": {},
        "final_cache_summary": {},
        "decode_steps": [],
        "prefill_cache_file": None,
        "final_cache_file": None,
        "system_prompt_cache_enabled": False,
        "system_prompt_cache_used": False,
        "system_prompt_cache_note": None,
        "system_prompt_cache_summary": {},
    })()
    inputs = {"input_ids": decoded_input_ids, "attention_mask": attention_mask}
    new_ids = engine.decode(inputs, outputs, trace)
    return _text_from_ids(engine, new_ids)


def dump_case(root: pathlib.Path, row: Dict[str, Any], raw: str, parsed: Dict[str, Optional[str]], valid: Dict[str, bool]) -> None:
    case_dir = root / f"{row['scenario']}__{row['run_id']}__{row['anchor']}"
    inputs_dir = case_dir / "inputs"
    outputs_dir = case_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "scene_system_prompt.txt").write_text(row["scene_system_prompt"], encoding="utf-8")
    (inputs_dir / "scene_user_prompt.txt").write_text(row["scene_user_prompt"], encoding="utf-8")
    if row.get("status_user_prompt") is not None:
        (inputs_dir / "status_system_prompt.txt").write_text(STATUS_SYSTEM_PROMPT, encoding="utf-8")
        (inputs_dir / "status_user_prompt.txt").write_text(row["status_user_prompt"], encoding="utf-8")
    for i, src in enumerate(row["images"]):
        try:
            shutil.copy2(src, inputs_dir / f"image_{i:02d}.jpg")
        except OSError:
            pass
    (outputs_dir / "raw_text.txt").write_text(raw, encoding="utf-8")
    (outputs_dir / "parsed.json").write_text(json.dumps({"parsed": parsed, "valid": valid}, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "gt.json").write_text(json.dumps(row["gt"], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = (
        f"# {row['scenario']} / {row['run_id']} / {row['anchor']}\n\n"
        f"- GT scene/status/subgoal: `{row['gt']['scene']}` / `{row['gt']['status']}` / `{row['gt']['subgoal']}`\n"
        f"- Pred scene/status/subgoal: `{parsed.get('scene')}` / `{parsed.get('status')}` / `{parsed.get('subgoal')}`\n"
        f"- Valid: `{valid}`\n"
    )
    (case_dir / "summary.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT v2 serial choice LoRA")
    p.add_argument("--jsonl", type=str, default="checkpoints/sft_v2_data/val.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--lora-dir", type=str, default="checkpoints/sft_v2_lora/latest/final")
    p.add_argument("--save-root", type=str, default="checkpoints/sft_v2_lora/latest")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--full-dump", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--full-dump-limit", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(pathlib.Path(args.jsonl))
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    save_root = pathlib.Path(args.save_root)
    eval_dir = save_root / "eval_v2"
    eval_dir.mkdir(parents=True, exist_ok=True)

    engine = LocalQwen3VLInstructEngine(
        pathlib.Path(args.model_dir),
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )
    engine.load()
    if args.lora_dir:
        engine.attach_lora_adapter(args.lora_dir, merge=args.merge_lora)

    predictions = []
    counters = Counter()
    per_scenario = defaultdict(Counter)
    dump_cases = args.full_dump if args.full_dump is not None else bool(args.max_samples > 0)
    dump_count = 0
    t0 = time.time()
    for row in rows:
        status_user_prompt = None
        raw_scene = ""
        raw_status = ""
        try:
            images = load_images(row["images"])
            raw_scene, _trace = engine.generate(row["scene_system_prompt"], row["scene_user_prompt"], images)
        except Exception as exc:
            parsed = {"scene": None, "status": None, "subgoal": None}
            valid = validate_choice(None, None, None)
            counters["runtime_error"] += 1
            pred = {
                "sample_idx": row["sample_idx"],
                "error": repr(exc),
                "raw_text": {"scene": raw_scene, "status": raw_status},
                "parsed": parsed,
                "valid": valid,
                "gt": row["gt"],
            }
            predictions.append(pred)
            continue

        scene_parsed = parse_output(raw_scene)
        pred_scene = scene_parsed.get("scene")
        scene_valid = validate_choice(pred_scene, None, None)["scene_valid"]
        if scene_valid:
            status_user_prompt = build_status_user_prompt(
                image_count=len(row["images"]),
                selected_scene=str(pred_scene),
                previous_status=row["memory_in_status"],
                previous_subgoal=row["memory_in_subgoal"],
            )
            try:
                raw_status = generate_status_with_scene_kv(engine, row, images, raw_scene, status_user_prompt)
                status_parsed = parse_output(raw_status)
            except Exception as exc:
                status_parsed = {"scene": None, "status": None, "subgoal": None}
                counters["runtime_error"] += 1
                raw_status = repr(exc)
        else:
            status_parsed = {"scene": None, "status": None, "subgoal": None}

        parsed = {
            "scene": pred_scene,
            "status": status_parsed.get("status"),
            "subgoal": status_parsed.get("subgoal"),
        }
        valid = validate_choice(parsed.get("scene"), parsed.get("status"), parsed.get("subgoal"))
        gt = row["gt"]
        scene_ok = parsed.get("scene") == gt["scene"]
        status_raw_ok = parsed.get("status") == gt["status"]
        subgoal_raw_ok = parsed.get("subgoal") == gt["subgoal"]
        # Serial metrics require the predicted scene to be correct.
        status_ok = scene_ok and status_raw_ok
        subgoal_ok = scene_ok and subgoal_raw_ok
        all_ok = scene_ok and status_raw_ok and subgoal_raw_ok
        counters["total"] += 1
        counters["valid_scene"] += int(scene_valid)
        counters["scene_ok"] += int(scene_ok)
        counters["status_ok"] += int(status_ok)
        counters["subgoal_ok"] += int(subgoal_ok)
        counters["status_raw_ok"] += int(status_raw_ok)
        counters["subgoal_raw_ok"] += int(subgoal_raw_ok)
        counters["all_ok"] += int(all_ok)
        # 在 scene 合法（不论是否等于 GT）的样本里看 status/subgoal 文本是否对得上 GT，
        # 用来诊断"scene 错了但 status/subgoal 文本碰巧也对"的情况，与 serial 口径分离。
        counters["status_raw_ok_valid_scene"] += int(scene_valid and status_raw_ok)
        counters["subgoal_raw_ok_valid_scene"] += int(scene_valid and subgoal_raw_ok)
        counters["invalid_scene"] += int(not valid["scene_valid"])
        counters["invalid_status_for_pred_scene"] += int(valid["scene_valid"] and not valid["status_valid_for_scene"])
        counters["invalid_subgoal_for_pred_scene"] += int(valid["scene_valid"] and not valid["subgoal_valid_for_scene"])
        counters["subgoal_not_next"] += int(valid["scene_valid"] and valid["status_valid_for_scene"] and not valid["subgoal_matches_status"])
        key = row["scenario"]
        per_scenario[key]["total"] += 1
        per_scenario[key]["valid_scene"] += int(scene_valid)
        per_scenario[key]["scene_ok"] += int(scene_ok)
        per_scenario[key]["status_ok"] += int(status_ok)
        per_scenario[key]["subgoal_ok"] += int(subgoal_ok)
        per_scenario[key]["status_raw_ok"] += int(status_raw_ok)
        per_scenario[key]["subgoal_raw_ok"] += int(subgoal_raw_ok)
        per_scenario[key]["all_ok"] += int(all_ok)
        per_scenario[key]["status_raw_ok_valid_scene"] += int(scene_valid and status_raw_ok)
        per_scenario[key]["subgoal_raw_ok_valid_scene"] += int(scene_valid and subgoal_raw_ok)

        pred = {
            "sample_idx": row["sample_idx"],
            "scenario": row["scenario"],
            "run_id": row["run_id"],
            "anchor": row["anchor"],
            "is_transition_sample": row["is_transition_sample"],
            "raw_text": {"scene": raw_scene, "status": raw_status},
            "parsed": parsed,
            "valid": valid,
            "gt": gt,
            "correct": {
                "scene": scene_ok,
                "status": status_ok,
                "subgoal": subgoal_ok,
                "status_raw": status_raw_ok,
                "subgoal_raw": subgoal_raw_ok,
                "all": all_ok,
            },
        }
        predictions.append(pred)
        if dump_cases and dump_count < args.full_dump_limit and not all_ok:
            dump_row = dict(row)
            dump_row["status_user_prompt"] = status_user_prompt
            raw_dump = json.dumps({"scene": raw_scene, "status": raw_status}, ensure_ascii=False, indent=2)
            dump_case(eval_dir / "cases", dump_row, raw_dump, parsed, valid)
            dump_count += 1

    total = max(counters["total"], 1)
    valid_total = max(counters["valid_scene"], 1)
    metrics = {
        "total": counters["total"],
        "valid_total": counters["valid_scene"],
        "scene_accuracy": counters["scene_ok"] / total,
        "status_accuracy": counters["status_ok"] / total,
        "subgoal_accuracy": counters["subgoal_ok"] / total,
        "status_raw_accuracy": counters["status_raw_ok"] / total,
        "subgoal_raw_accuracy": counters["subgoal_raw_ok"] / total,
        "all_accuracy": counters["all_ok"] / total,
        "invalid_scene_rate": counters["invalid_scene"] / total,
        "invalid_status_for_pred_scene_rate": counters["invalid_status_for_pred_scene"] / total,
        "invalid_subgoal_for_pred_scene_rate": counters["invalid_subgoal_for_pred_scene"] / total,
        "subgoal_not_next_rate": counters["subgoal_not_next"] / total,
        "status_accuracy_valid_scene": counters["status_ok"] / valid_total,
        "subgoal_accuracy_valid_scene": counters["subgoal_ok"] / valid_total,
        "all_accuracy_valid_scene": counters["all_ok"] / valid_total,
        "status_raw_accuracy_valid_scene": counters["status_raw_ok_valid_scene"] / valid_total,
        "subgoal_raw_accuracy_valid_scene": counters["subgoal_raw_ok_valid_scene"] / valid_total,
        "elapsed_sec": time.time() - t0,
        "_metric_doc": {
            "status_accuracy": "Serial accuracy: scene must be correct and STATUS must exactly match GT.",
            "status_accuracy_valid_scene": "Same serial numerator, divided by rows whose predicted scene is valid.",
            "status_raw_accuracy": "Raw STATUS exact match without requiring scene correctness; diagnostic only.",
            "status_raw_accuracy_valid_scene": "Raw STATUS match within rows whose predicted scene is valid (ignores serial scene constraint).",
            "subgoal_raw_accuracy_valid_scene": "Raw SUBGOAL match within rows whose predicted scene is valid (ignores serial scene constraint).",
            "invalid_status_for_pred_scene_rate": "Prediction picked a valid scene, but status was not in that predicted scene's event sequence.",
        },
    }
    scenario_metrics = {}
    for scenario, c in sorted(per_scenario.items()):
        n = max(c["total"], 1)
        nv = max(c["valid_scene"], 1)
        scenario_metrics[scenario] = {
            "total": c["total"],
            "valid_total": c["valid_scene"],
            "scene_accuracy": c["scene_ok"] / n,
            "status_accuracy": c["status_ok"] / n,
            "subgoal_accuracy": c["subgoal_ok"] / n,
            "status_raw_accuracy": c["status_raw_ok"] / n,
            "subgoal_raw_accuracy": c["subgoal_raw_ok"] / n,
            "all_accuracy": c["all_ok"] / n,
            "status_accuracy_valid_scene": c["status_ok"] / nv,
            "subgoal_accuracy_valid_scene": c["subgoal_ok"] / nv,
            "all_accuracy_valid_scene": c["all_ok"] / nv,
            "status_raw_accuracy_valid_scene": c["status_raw_ok_valid_scene"] / nv,
            "subgoal_raw_accuracy_valid_scene": c["subgoal_raw_ok_valid_scene"] / nv,
        }

    (eval_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "scenario_metrics.json").write_text(json.dumps(scenario_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(eval_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(eval_dir / "predictions_diff.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            if not row.get("correct", {}).get("all", False):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[write] {eval_dir}")


if __name__ == "__main__":
    main()
