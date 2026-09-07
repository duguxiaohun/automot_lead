#!/usr/bin/env python3
"""构建全覆盖的 action_prior 标定真值索引：Phase1 四事实 + RS + 逐帧 EVENT 目标类。

Phase1 融合索引本身就是全帧覆盖，直接复用它已审计的 answers 与 RS；EVENT 侧不能
复用 Phase2 索引，因为那份索引为训练做了类别均衡下采样。这里改为按同一 collection
标注重新折叠 EVENT taxonomy，逐帧输出，不做任何采样、均衡或 invalid 增强。

用法::

    python qwen3vl_local/action_prior/build_prior_labels.py
    python qwen3vl_local/action_prior/build_prior_labels.py --output-dir <\u65b0目录>
    python qwen3vl_local/action_prior/build_prior_labels.py --scenarios HighwayCutIn --output-dir /tmp/smoke

产物供 ``--dataset-priors`` 使用；``run_full_pipeline.sh --dataset-priors`` 在默认
路径缺文件时会自动调用本脚本。已存在的索引永不覆盖。
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.action_prior.contracts import digest, file_hash
from qwen3vl_local.action_prior.dataset_labels import LABEL_SCHEMA, PRIOR_SOURCE
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2.build_dataset import (
    _can_construct_invalid_from,
    _event_codes,
    _mismatched_question_domain,
    _rs_label,
    _target_class,
    _target_question_domain,
)
from qwen3vl_local.sft_new_loop_phase2.highway_ue3_audit import (
    HIGHWAY_UE3_SUBTYPE,
    load_highway_ue3_decisions,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIGHWAY_UE3_DECISIONS = (
    ROOT / "qwen3vl_local/sft_new_loop_phase2/highway_ue3_rgb_decisions_v1.jsonl"
)


def read_phase1_index(path, counters):
    """把 Phase1 全帧索引读成 {(scenario, route): {frame: (rs, answers)}}。"""
    table = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answers = row["answers"]
            table.setdefault((row["scenario"], row["route_id"]), {})[
                int(row["frame_id"])
            ] = (
                str(row["rs"]),
                {key: bool(answers[key]) for key in p1.PHASE1_ANSWER_KEYS},
            )
            counters["phase1_index_rows"] += 1
    if not table:
        raise ValueError(f"{path}: empty Phase1 frame index")
    return table


def iter_labels(collection_dir, table, scenarios, overrides, counters):
    """逐 route 流式读取标注，只为 Phase1 索引已收录的帧生成标签。"""
    for result_path in sorted(Path(collection_dir).glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (scenarios and scenario not in scenarios):
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            frames = table.get((scenario, route_id))
            if not frames:
                continue
            for annotation in route.get("annotations", []) or []:
                try:
                    frame_id = int(annotation.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                entry = frames.get(frame_id)
                if entry is None:
                    continue
                rs, phase1_answers = entry
                if _rs_label(annotation) != rs:
                    counters["skipped/rs_disagreement"] += 1
                    continue
                # 高速 cut-in 不能从场景名/R3 自动推定，只采用已 RGB 复核的人工决策。
                override = overrides.get((scenario, route_id, frame_id))
                if override is not None:
                    target = "UE3"
                    counters[f"target_event_class/UE3/{HIGHWAY_UE3_SUBTYPE}"] += 1
                else:
                    target = _target_class(rs, _event_codes(annotation))
                if target is None:
                    counters["skipped/no_event_target_class"] += 1
                    continue
                base = dict(rs=rs, target_event_class=target)
                domain = _target_question_domain(base, target)
                invalid_domain = (
                    _mismatched_question_domain(rs)
                    if _can_construct_invalid_from(base)
                    else None
                )
                if invalid_domain == domain:
                    # U-E3 overlay 会把提问域挪到 ROAD；此时另一域没有干净的 invalid 标签。
                    invalid_domain = None
                counters["labeled_frames"] += 1
                counters[f"road_structure/{rs}"] += 1
                counters[f"target_event_class/{target}"] += 1
                counters[f"question_domain/{domain}"] += 1
                counters["labeled_invalid_domain" if invalid_domain else "unlabeled_other_domain"] += 1
                yield dict(
                    scenario=scenario,
                    route_id=route_id,
                    frame_id=frame_id,
                    road_structure=rs,
                    target_event_class=target,
                    question_domain=domain,
                    invalid_domain=invalid_domain,
                    phase1_answers=phase1_answers,
                )


def main():
    """只写索引，不复制 RGB，也不读取未来轨迹或专家动作。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase1-index",
        default=str(ROOT / "checkpoints/sft_new_loop_phase1_data/frame_index.jsonl"),
    )
    p.add_argument("--collection-dir", default=str(ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints/action_prior_labels"))
    p.add_argument(
        "--highway-ue3-rgb-decisions", default=str(DEFAULT_HIGHWAY_UE3_DECISIONS)
    )
    p.add_argument("--scenarios", default="all")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "prior_labels.jsonl"
    if target.exists():
        raise FileExistsError(f"{target}: use a new output directory; label indices are never overwritten")
    scenarios = (
        None
        if args.scenarios == "all"
        else {item.strip() for item in args.scenarios.split(",") if item.strip()}
    )
    counters = Counter()
    overrides, decision_report = load_highway_ue3_decisions(
        Path(args.highway_ue3_rgb_decisions)
    )
    print(f"[phase1 index] {args.phase1_index}", flush=True)
    table = read_phase1_index(args.phase1_index, counters)
    print(f"[phase1 index] routes={len(table)} frames={counters['phase1_index_rows']}", flush=True)
    temporary = out / ".prior_labels.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in iter_labels(args.collection_dir, table, scenarios, overrides, counters):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if counters["labeled_frames"] % 20000 == 0:
                    print(f"[labels] {counters['labeled_frames']}", flush=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if not counters["labeled_frames"]:
        temporary.unlink(missing_ok=True)
        raise ValueError("no labeled frames; check --phase1-index and --collection-dir")
    temporary.replace(target)
    manifest = dict(
        schema=LABEL_SCHEMA,
        prior_source=PRIOR_SOURCE,
        frame_index=str(target),
        config=vars(args),
        counts=dict(counters),
        sources=dict(
            phase1_index=str(Path(args.phase1_index).resolve()),
            phase1_index_sha256=file_hash(args.phase1_index),
            collection_dir=str(Path(args.collection_dir).resolve()),
            highway_ue3_rgb_decisions=str(Path(args.highway_ue3_rgb_decisions).resolve()),
            highway_ue3_rgb_decisions_sha256=file_hash(args.highway_ue3_rgb_decisions),
            highway_ue3_decision_report=decision_report,
        ),
        coverage_note=(
            "Frames come from the full Phase1 fused index; EVENT classes are refolded per frame "
            "without the Phase2 training balance sampling. Frames the Phase1 index dropped "
            "(visual risk, missing RGB history, unknown RS) stay unlabeled."
        ),
        privileged_label_conditioning=True,
    )
    manifest["identity"] = digest(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"[written] {target}", flush=True)


if __name__ == "__main__":
    main()
