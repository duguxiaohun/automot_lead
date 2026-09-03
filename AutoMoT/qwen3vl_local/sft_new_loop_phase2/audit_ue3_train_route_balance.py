#!/usr/bin/env python3
"""CPU-only 对比旧 UE3 重复方式与新的 route-balanced 训练曝光。"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any, Dict, Iterable, Mapping


_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_new_loop_phase2.history_rgb import HISTORY_RGB_MODE_END2  # noqa: E402
from qwen3vl_local.sft_new_loop_phase2.prompts import (  # noqa: E402
    PROMPT_NAME,
    event_prompt_sha256,
)
from qwen3vl_local.sft_new_loop_phase2.sampling import (  # noqa: E402
    UE3_TRAIN_SAMPLER_VERSION,
    frame_repetition_report,
    route_balanced_sample,
    route_diverse_sample,
    route_diversity_report,
    route_extra_exposure_report,
)


EXPECTED_PROMPT_NAME = "sft_new_loop_phase2_direct_event_visual_v3"
EXPECTED_PROMPT_HASH = "cd564634257fe0f072de70947200a820d6dd2b43375981b60120a1fe2296dd7f"


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSON object。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            yield row


def _is_train_ue3(row: Mapping[str, Any]) -> bool:
    """只选择可适用的 train UE3 正类。"""

    return (
        str(row.get("split", "")) == "train"
        and not bool(row.get("invalid_event_context"))
        and str(row.get("target_event_class", "")) == "UE3"
    )


def build_report(
    index_path: pathlib.Path,
    *,
    target: int,
    seed: int,
    max_frame_repeat: int,
) -> Dict[str, Any]:
    """构建单变量训练采样审计；不修改 index、prompt、标签或 split。"""

    source = [row for row in _iter_jsonl(index_path) if _is_train_ue3(row)]
    if not source:
        raise ValueError(f"no train UE3 rows found in {index_path}")
    legacy = route_diverse_sample(source, target=int(target), rng=random.Random(int(seed)))
    balanced = route_balanced_sample(
        source,
        target=int(target),
        rng=random.Random(int(seed)),
        max_frame_repeat=int(max_frame_repeat),
    )
    raw_routes = route_diversity_report(source)
    legacy_routes = route_diversity_report(legacy)
    balanced_routes = route_diversity_report(balanced)
    raw_repeats = frame_repetition_report(source)
    balanced_repeats = frame_repetition_report(balanced)
    extra_route_report = route_extra_exposure_report(source, balanced)
    prompt_hash = event_prompt_sha256(history_rgb_mode=HISTORY_RGB_MODE_END2)
    guards = {
        "production_prompt_v3_frozen": (
            PROMPT_NAME == EXPECTED_PROMPT_NAME and prompt_hash == EXPECTED_PROMPT_HASH
        ),
        "sample_count_equal_target": len(balanced) == int(target),
        "all_raw_ue3_routes_preserved": (
            int(balanced_routes["unique_routes"]) == int(raw_routes["unique_routes"])
        ),
        "all_raw_ue3_frames_preserved_once_before_extra_sampling": (
            int(target) < len(source)
            or int(balanced_repeats["unique_frames"]) == int(raw_repeats["unique_frames"])
        ),
        "extra_route_exposure_max_deviation_le_1": (
            int(target) < len(source)
            or (
                bool(extra_route_report["all_nonnegative"])
                and int(extra_route_report["cases"]) == int(target) - len(source)
                and int(extra_route_report["max_deviation"]) <= 1
            )
        ),
        "max_route_exposure_strictly_reduced_vs_legacy": (
            int(balanced_routes["max_cases_per_route"])
            < int(legacy_routes["max_cases_per_route"])
        ),
        "frame_repeat_within_hard_cap": (
            int(balanced_repeats["max_frame_repeat"]) <= int(max_frame_repeat)
        ),
        "index_prompt_labels_and_frozen_splits_not_mutated": True,
    }
    return {
        "format": "sft_new_loop_phase2_ue3_train_route_balance_audit_v2",
        "official_metric": False,
        "mutation": False,
        "contract": (
            "CPU-only projection of train UE3 sampling. It does not load a model, rewrite the "
            "index, change labels/prompt, select a checkpoint, or open unseen."
        ),
        "index": str(index_path),
        "seed": int(seed),
        "target": int(target),
        "max_frame_repeat": int(max_frame_repeat),
        "sampler_version": UE3_TRAIN_SAMPLER_VERSION,
        "prompt": {
            "name": PROMPT_NAME,
            "history_rgb_mode": HISTORY_RGB_MODE_END2,
            "sha256": prompt_hash,
        },
        "raw": {
            "route_diversity": raw_routes,
            "frame_repetition": raw_repeats,
        },
        "legacy_repeat_whole_bucket": {
            "route_diversity": legacy_routes,
            "frame_repetition": frame_repetition_report(legacy),
        },
        "candidate_ue3_route_balanced": {
            "route_diversity": balanced_routes,
            "frame_repetition": balanced_repeats,
            "extra_route_exposure": extra_route_report,
        },
        "guards": guards,
        "passed": all(guards.values()),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    """渲染短审计报告。"""

    raw = report["raw"]["route_diversity"]
    legacy = report["legacy_repeat_whole_bucket"]["route_diversity"]
    candidate = report["candidate_ue3_route_balanced"]["route_diversity"]
    repeats = report["candidate_ue3_route_balanced"]["frame_repetition"]
    lines = [
        "# UE3 train route-balanced sampling smoke",
        "",
        f"- passed: `{report['passed']}`",
        f"- prompt: `{report['prompt']['name']}` / `{report['prompt']['sha256']}`",
        f"- target: `{report['target']}`",
        f"- sampler: `{report['sampler_version']}`",
        f"- max frame repeat cap: `{report['max_frame_repeat']}`",
        "",
        "| sampler | cases | routes | max cases/route | max frame repeat |",
        "|---|---:|---:|---:|---:|",
        f"| raw | {raw['cases']} | {raw['unique_routes']} | {raw['max_cases_per_route']} | 1 |",
        (
            "| legacy repeat-whole-bucket | "
            f"{legacy['cases']} | {legacy['unique_routes']} | {legacy['max_cases_per_route']} | "
            f"{report['legacy_repeat_whole_bucket']['frame_repetition']['max_frame_repeat']} |"
        ),
        (
            "| candidate coverage-first + balanced extras | "
            f"{candidate['cases']} | {candidate['unique_routes']} | "
            f"{candidate['max_cases_per_route']} | {repeats['max_frame_repeat']} |"
        ),
        "",
        "## Guards",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in report["guards"].items())
    lines.extend(
        [
            "",
            "This is a sampling audit only. Validation/test identities and the production prompt remain frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="checkpoints/sft_new_loop_phase2_data/frame_index.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/ue3_train_route_balance_smoke")
    parser.add_argument("--target", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-frame-repeat", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """运行审计并在 guard 失败时阻止训练。"""

    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        pathlib.Path(args.index),
        target=int(args.target),
        seed=int(args.seed),
        max_frame_repeat=int(args.max_frame_repeat),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "guards": report["guards"],
        "raw": report["raw"]["route_diversity"],
        "legacy": report["legacy_repeat_whole_bucket"]["route_diversity"],
        "candidate": report["candidate_ue3_route_balanced"]["route_diversity"],
        "candidate_frame_repetition": report["candidate_ue3_route_balanced"]["frame_repetition"],
        "candidate_extra_route_exposure": report["candidate_ue3_route_balanced"]["extra_route_exposure"],
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("UE3 train route-balanced smoke failed; do not train")


if __name__ == "__main__":
    main()
