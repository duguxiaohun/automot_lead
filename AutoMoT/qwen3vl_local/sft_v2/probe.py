"""Probe SFT v2 cases: save prompts, images, GT, and model output."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v2.eval import dump_case, generate_status_with_scene_kv, load_images, load_rows
from qwen3vl_local.engine import LocalQwen3VLInstructEngine
from qwen3vl_local.sft_v2.prompts import build_status_user_prompt, parse_output, validate_choice


def choose_rows(rows: List[Dict[str, Any]], scenarios: Optional[List[str]], num_per_scenario: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    allowed = set(scenarios or [])
    for row in rows:
        if allowed and row["scenario"] not in allowed:
            continue
        buckets[row["scenario"]].append(row)
    chosen = []
    for scenario, items in sorted(buckets.items()):
        rng.shuffle(items)
        chosen.extend(items[:num_per_scenario])
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe SFT v2 cases")
    p.add_argument("--jsonl", type=str, default="checkpoints/sft_v2_data/val.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--lora-dir", type=str, default="checkpoints/sft_v2_lora/latest/final")
    p.add_argument("--save-root", type=str, default="checkpoints/sft_v2_lora/latest")
    p.add_argument("--scenarios", type=str, default="")
    p.add_argument("--num-per-scenario", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(pathlib.Path(args.jsonl))
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    selected = choose_rows(rows, scenarios, args.num_per_scenario, args.seed)
    out_root = pathlib.Path(args.save_root) / "eval_cases_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    engine = LocalQwen3VLInstructEngine(
        pathlib.Path(args.model_dir),
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
    )
    engine.load()
    if args.lora_dir:
        engine.attach_lora_adapter(args.lora_dir, merge=args.merge_lora)

    log_rows = []
    for row in selected:
        images = load_images(row["images"])
        raw_scene, _trace = engine.generate(row["scene_system_prompt"], row["scene_user_prompt"], images)
        scene_parsed = parse_output(raw_scene)
        pred_scene = scene_parsed.get("scene")
        status_user_prompt = None
        raw_status = ""
        if validate_choice(pred_scene, None, None)["scene_valid"]:
            status_user_prompt = build_status_user_prompt(
                image_count=len(row["images"]),
                selected_scene=str(pred_scene),
                previous_status=row["memory_in_status"],
                previous_subgoal=row["memory_in_subgoal"],
            )
            raw_status = generate_status_with_scene_kv(engine, row, images, raw_scene, status_user_prompt)
            status_parsed = parse_output(raw_status)
        else:
            status_parsed = {"scene": None, "status": None, "subgoal": None}
        parsed = {
            "scene": pred_scene,
            "status": status_parsed.get("status"),
            "subgoal": status_parsed.get("subgoal"),
        }
        valid = validate_choice(parsed.get("scene"), parsed.get("status"), parsed.get("subgoal"))
        dump_row = dict(row)
        dump_row["status_user_prompt"] = status_user_prompt
        raw_dump = json.dumps({"scene": raw_scene, "status": raw_status}, ensure_ascii=False, indent=2)
        dump_case(out_root, dump_row, raw_dump, parsed, valid)
        log_rows.append({
            "scenario": row["scenario"],
            "run_id": row["run_id"],
            "anchor": row["anchor"],
            "gt": row["gt"],
            "parsed": parsed,
            "valid": valid,
        })
        print(f"[case] {row['scenario']} {row['run_id']} anchor={row['anchor']} pred={parsed}")
    (out_root / "manifest.json").write_text(json.dumps(log_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_root}")


if __name__ == "__main__":
    main()
