#!/usr/bin/env python3
"""只根据 validation 记录选择多 seed 中的 production-ready checkpoint。"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    """读取 JSON object。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def collect_runs(run_roots: List[pathlib.Path]) -> List[Dict[str, Any]]:
    """收集每个 seed 的正式/回退选优状态，不读取 test 指标。"""

    records: List[Dict[str, Any]] = []
    for run_root in run_roots:
        root = run_root.resolve()
        run_manifest = _read_json(root / "train_run_manifest.json")
        status_path = root / "generation_selection_status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        best_path = root / "best_generation.json"
        fallback_path = root / "fallback_generation.json"
        best = _read_json(best_path) if best_path.is_file() else None
        fallback = _read_json(fallback_path) if fallback_path.is_file() else None
        if best is not None and not bool(best.get("generation_guards_ok", False)):
            raise ValueError(f"best_generation did not pass all guards: {best_path}")
        records.append(
            {
                "run_root": str(root),
                "seed": int(run_manifest.get("seed", -1)),
                "prompt_name": str(run_manifest.get("prompt_name", "")),
                "production_prompt_sha256": str(run_manifest.get("production_prompt_sha256", "")),
                "history_rgb_mode": str(run_manifest.get("history_rgb_mode", "")),
                "status": status,
                "best_generation": best,
                "fallback_generation": fallback,
            }
        )
    return records


def select_checkpoint(records: List[Dict[str, Any]], *, required_seeds: int) -> Dict[str, Any]:
    """校验实验同构性，并按 validation exact 选择通过全部门槛的 seed。"""

    if int(required_seeds) > 0 and len(records) != int(required_seeds):
        raise ValueError(f"seed run count mismatch: got={len(records)} expected={required_seeds}")
    if not records:
        raise ValueError("no seed runs supplied")
    for field in ("prompt_name", "production_prompt_sha256", "history_rgb_mode"):
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(f"seed runs disagree on {field}: {sorted(values)}")
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError(f"seeds must be present and unique: {seeds}")
    eligible = [record for record in records if record["best_generation"] is not None]
    if not eligible:
        raise ValueError(
            "no seed produced production-ready best_generation; inspect fallback_generation and stop the experiment"
        )

    def score(record: Dict[str, Any]) -> tuple[float, float, int]:
        best = record["best_generation"]
        generation = best.get("generation") or {}
        return (
            float(best.get("generation_exact_accuracy", generation.get("exact_accuracy", 0.0))),
            float(generation.get("pattern/all_random_order_pattern_exact", 0.0)),
            -int(record["seed"]),
        )

    selected = max(eligible, key=score)
    best = selected["best_generation"]
    return {
        "selection_contract": "validation-only; all configured guards pass, then max generation exact",
        "prompt_name": selected["prompt_name"],
        "production_prompt_sha256": selected["production_prompt_sha256"],
        "history_rgb_mode": selected["history_rgb_mode"],
        "selected_seed": selected["seed"],
        "selected_run_root": selected["run_root"],
        "selected_adapter_dir": str(pathlib.Path(selected["run_root"]) / "best_generation"),
        "selected_validation": best,
        "eligible_seeds": [record["seed"] for record in eligible],
        "all_runs": records,
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--required-seeds", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-adapter-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    records = collect_runs([pathlib.Path(path) for path in args.run_root])
    result = select_checkpoint(records, required_seeds=int(args.required_seeds))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print_adapter_only:
        print(result["selected_adapter_dir"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
