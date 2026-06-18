"""SFT v2 case-level 诊断脚本。

probe 不做全量指标统计，而是按场景抽样若干 case，保存 prompt、输入图片、GT、模型
原始输出和解析结果。它复用 eval.py 的串行协议：先生成 SCENE，scene 合法时按预测
scene 构造第二阶段 prompt，并尝试复用第一阶段 KV 继续生成 STATUS/SUBGOAL。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v2.eval import dump_case, generate_status_with_scene_kv, load_images, load_rows, remap_status_hint
from qwen3vl_local.engine import LocalQwen3VLInstructEngine
from qwen3vl_local.sft_v2.prompts import build_status_user_prompt, parse_output, validate_choice


def choose_rows(rows: List[Dict[str, Any]], scenarios: Optional[List[str]], num_per_scenario: int, seed: int) -> List[Dict[str, Any]]:
    """按场景抽样 probe case。

    ``scenarios`` 为空时遍历所有场景；每个场景最多取 ``num_per_scenario`` 条，随机种子
    固定以便复现实验。
    """

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
    """解析 probe 命令行参数。"""

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
    """probe 入口：加载模型，抽样生成，并把每个 case 落盘。"""

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
        with torch.no_grad():
            # 第一阶段生成 SCENE，同时在 engine 内保存可供第二阶段续接的 KV 状态。
            raw_scene, _trace = engine.generate(row["scene_system_prompt"], row["scene_user_prompt"], images)
        scene_parsed = parse_output(raw_scene)
        pred_scene = scene_parsed.get("scene")
        status_user_prompt = None
        raw_status = ""
        status_kv_reused = False
        if validate_choice(pred_scene, None, None)["scene_valid"]:
            # 第二阶段 prompt 必须和预测场景自洽，因此把前序提示映射到
            # 预测场景的同相位事件。
            previous_status, previous_subgoal, _previous_phase_idx = remap_status_hint(
                row["gt"]["scene"],
                row["memory_in_status"],
                str(pred_scene),
            )
            status_user_prompt = build_status_user_prompt(
                image_count=len(row["images"]),
                selected_scene=str(pred_scene),
                previous_status=previous_status,
                previous_subgoal=previous_subgoal,
            )
            with torch.no_grad():
                # 与 eval.py 相同：优先复用第一阶段 KV，返回值会记录是否真的复用成功。
                raw_status, status_kv_reused = generate_status_with_scene_kv(engine, row, images, raw_scene, status_user_prompt)
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
        # raw_dump 同时保存两阶段输出，方便肉眼检查模型是否多写解释或格式漂移。
        raw_dump = json.dumps({"scene": raw_scene, "status": raw_status}, ensure_ascii=False, indent=2)
        dump_case(out_root, dump_row, raw_dump, parsed, valid)
        log_rows.append({
            "scenario": row["scenario"],
            "run_id": row["run_id"],
            "anchor": row["anchor"],
            "gt": row["gt"],
            "parsed": parsed,
            "valid": valid,
            "status_kv_reused": status_kv_reused,
        })
        print(f"[case] {row['scenario']} {row['run_id']} anchor={row['anchor']} pred={parsed}")
    (out_root / "manifest.json").write_text(json.dumps(log_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_root}")


if __name__ == "__main__":
    main()
