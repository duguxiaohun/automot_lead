#!/usr/bin/env python3
"""Build the fused Phase1 + Phase2 frame index.

The route split, abnormal-route filtering, RGB history layout, visual-risk
filtering, and full-frame RGB review coverage follow the latest
``sft_loop_phase2_augment`` builder.  Phase1 labels come from the finalized
four-question answer table; Phase2 labels come from the primary RS annotation.
Uncertain label policy is not guessed here: this script only consumes the
audited answer table and the per-frame RGB/RS/EVENT annotation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _path in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402
from qwen3vl_local.sft_loop_phase2_augment.visual_audit import (  # noqa: E402
    DEFAULT_COVERAGE_MANIFEST,
    frame_visual_risk,
    load_review_coverage,
)
from qwen3vl_local.sft_new_loop_phase1 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_new_loop_phase1.prompts import (  # noqa: E402
    ANSWER_KEYS,
    PHASE1_ANSWER_KEYS,
    PHASE2_ANSWER_KEYS,
)


RGB_HISTORY_COUNT = 4


def _stable_unit(value: str) -> float:
    """Map text to a stable [0, 1) split value."""

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def _split(scenario: str, route_id: str, seed: int, test_ratio: float, val_ratio: float) -> str:
    """Route-disjoint split shared by Phase1/Phase2."""

    value = _stable_unit(f"{seed}:{scenario}:{route_id}")
    return "test" if value < test_ratio else "val" if value < test_ratio + val_ratio else "train"


def _labels(annotation: Mapping[str, Any]) -> Tuple[str, str]:
    """Read primary RS/EVENT labels from one frame annotation."""

    rs = str(annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label") or "UNKNOWN")
    event = str(annotation.get("primary_event") or (annotation.get("frame_event_annotation") or {}).get("label") or "UNKNOWN")
    return rs, event


def _history(run_dir: pathlib.Path, frame_id: int) -> Optional[list[str]]:
    """Return oldest-to-newest four-frame RGB history with frame-0 left padding."""

    paths = []
    for idx in [max(0, frame_id - offset) for offset in reversed(range(RGB_HISTORY_COUNT))]:
        path = _rgb_path(run_dir, idx)
        if path is None:
            return None
        paths.append(str(path))
    return paths


def _relative_history_paths(paths: Sequence[str], data_root: pathlib.Path) -> List[str]:
    """Store RGB paths relative to data_root so the index can move between machines."""

    root = data_root.expanduser().absolute()
    rel_paths: List[str] = []
    for raw in paths:
        path = pathlib.Path(raw).expanduser().absolute()
        try:
            rel_paths.append(str(path.relative_to(root)))
        except ValueError as exc:
            raise ValueError(f"RGB path {path} is not under data_root {root}") from exc
    return rel_paths


def _town(annotation: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    """Read the XML town used by the RGB audit coverage checks."""

    return str((annotation.get("evidence") or {}).get("xml_town") or route.get("xml_town") or "UNKNOWN")


def _coerce_subgroups(value: Any) -> List[str]:
    """把 notes/annotation 里的 subgroup 字段规整成字符串列表。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_coerce_subgroups(item))
        return out
    return []


def _extract_override_route_ids(override: Mapping[str, Any]) -> List[str]:
    """保留旧 manifest 字段，但禁止从自由文本 evidence 推断 route 覆盖。"""

    _ = override
    return []


def _load_manual_subgroup_notes(paths: Sequence[pathlib.Path]) -> Dict[Tuple[str, str], List[str]]:
    """读取 route/frame 级人工 RGB notes 中的显式 visual/topology subgroup 标记。"""

    out: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scenario = str(obj.get("scenario") or "")
                route_id = str(obj.get("run_id") or obj.get("route_id") or "")
                if not scenario or not route_id:
                    continue
                subgroups: List[str] = []
                for key in ("topology_subgroup", "visual_subgroup", "visual_subgroups", "route_visual_subgroups"):
                    subgroups.extend(_coerce_subgroups(obj.get(key)))
                for subgroup in subgroups:
                    if subgroup not in out[(scenario, route_id)]:
                        out[(scenario, route_id)].append(subgroup)
    return out


def _load_answer_table(path: pathlib.Path) -> Tuple[Dict[Tuple[str, str, str], Dict[str, bool]], List[Dict[str, Any]]]:
    """Load the finalized Phase1 answer table and enforce STATIC_OBSTACLE=U-E2."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "phase1_four_question_answer_table":
        raise ValueError(f"unsupported answer table: {path}")
    table: Dict[Tuple[str, str, str], Dict[str, bool]] = {}
    for row in payload.get("rows", []):
        scenario, rs, event = str(row.get("scenario")), str(row.get("rs")), str(row.get("event"))
        source = row.get("answers") or {}
        if "STATIC_OBSTACLE" in source:
            static_obstacle = bool(source["STATIC_OBSTACLE"])
            expected = event == "U-E2"
            if static_obstacle != expected:
                raise ValueError(f"STATIC_OBSTACLE mismatch for {(scenario, rs, event)}: {static_obstacle} != {expected}")
        elif "OBSTACLE" in source:
            static_obstacle = event == "U-E2"
        else:
            raise ValueError(f"answer table lacks STATIC_OBSTACLE/OBSTACLE for {(scenario, rs, event)}")
        table[(scenario, rs, event)] = {
            "HIGHWAY": bool(source.get("HIGHWAY", False)),
            "STATIC_OBSTACLE": static_obstacle,
            "VULNERABLE": bool(source.get("VULNERABLE", False)),
            "TRAFFIC_LIGHT_ABNORMAL": bool(source.get("TRAFFIC_LIGHT_ABNORMAL", False)),
        }
    overrides: List[Dict[str, Any]] = []
    for raw in payload.get("visual_subgroup_overrides", []) or []:
        item = dict(raw)
        item["explicit_route_ids"] = _extract_override_route_ids(item)
        overrides.append(item)
    return table, overrides


def _apply_visual_subgroup_overrides(
    answers: Mapping[str, bool],
    *,
    overrides: Sequence[Mapping[str, Any]],
    note_subgroups: Mapping[Tuple[str, str], Sequence[str]],
    annotation: Mapping[str, Any],
    scenario: str,
    route_id: str,
    town: str,
    rs: str,
    counters: Counter[str],
) -> Tuple[Dict[str, bool], List[str]]:
    """按显式 RGB subgroup 审计 patch Phase1 答案。"""

    patched = dict(answers)
    applied: List[str] = []
    annotation_subgroups: List[str] = []
    for key in ("topology_subgroup", "visual_subgroup", "visual_subgroups", "route_visual_subgroups"):
        annotation_subgroups.extend(_coerce_subgroups(annotation.get(key)))
    known_subgroups = set(annotation_subgroups)
    known_subgroups.update(note_subgroups.get((scenario, route_id), []))
    for override in overrides:
        override_id = str(override.get("id") or "")
        subgroup = str(override.get("topology_subgroup") or "")
        route_marked = bool(subgroup and subgroup in known_subgroups)
        if not route_marked:
            continue
        if str(override.get("scenario")) != scenario:
            continue
        towns = {str(item) for item in override.get("towns", [])}
        if towns and town not in towns:
            continue
        rs_values = {str(item) for item in override.get("rs_values", [])}
        if rs_values and rs not in rs_values:
            continue
        for key, value in (override.get("answers_patch") or {}).items():
            if key in patched:
                patched[key] = bool(value)
        applied.append(override_id)
        counters[f"visual_subgroup_override/{override_id}"] += 1
    return patched, applied


def _phase2_answers(rs: str) -> Dict[str, bool]:
    """Encode Phase2 RS1/RS2/RS4/RS5 labels; R3/highway is all NO."""

    if rs not in {"R1", "R2", "R3", "R4", "R5"}:
        raise ValueError(f"unsupported RS label for Phase2 answers: {rs}")
    return {key: rs == key.replace("RS", "R") for key in PHASE2_ANSWER_KEYS}


def _focus_availability(counts: Mapping[str, Counter[str]]) -> Dict[str, Dict[str, int]]:
    """Return split -> answer-key:YES/NO availability for all eight keys."""

    return {
        split: {
            f"{key}:{value}": int(counts[split][f"{key}:{value}"])
            for key in ANSWER_KEYS
            for value in ("YES", "NO")
        }
        for split in ("train", "val", "test")
    }


def _assert_required_coverage(availability: Mapping[str, Mapping[str, int]], *, val_ratio: float) -> None:
    """Refuse splits that cannot support per-question YES:NO 1:1 sampling."""

    required = ["train", "test"] + (["val"] if float(val_ratio) > 0.0 else [])
    missing = {
        split: [key for key, count in availability.get(split, {}).items() if int(count) <= 0]
        for split in required
    }
    missing = {split: keys for split, keys in missing.items() if keys}
    if missing:
        raise ValueError(f"route-disjoint fused split lacks required focus bins: {missing}")


def iter_rows(args: argparse.Namespace, risk_stats: Optional[Counter[str]] = None) -> Iterable[Dict[str, Any]]:
    """Yield fused per-frame rows."""

    answer_table, visual_overrides = _load_answer_table(pathlib.Path(args.answer_table))
    note_paths = [pathlib.Path(item) for item in str(args.manual_note_paths).split(",") if item.strip()]
    note_subgroups = _load_manual_subgroup_notes(note_paths)
    collection_dir = pathlib.Path(args.collection_dir)
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    selected = None if args.scenarios == "all" else {item.strip() for item in args.scenarios.split(",") if item.strip()}
    seen = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (selected is not None and scenario not in selected):
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if not route_id or str(route.get("status")) == "data_missing_skip":
                continue
            run_dir = data_root / scenario / route_id
            abnormal, _ = is_abnormal_lead_route(run_dir, scenario)
            if abnormal or not run_dir.is_dir():
                continue
            seen += 1
            if int(args.progress_every_routes) > 0 and seen % int(args.progress_every_routes) == 0:
                print(f"[new-loop-build] routes={seen} last={scenario}/{route_id}", flush=True)
            if int(args.max_routes) > 0 and seen > int(args.max_routes):
                return
            split = _split(scenario, route_id, int(args.split_seed), float(args.test_ratio), float(args.val_ratio))
            for ann in route.get("annotations", []) or []:
                try:
                    frame_id = int(ann.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                rs, event = _labels(ann)
                if rs not in {"R1", "R2", "R3", "R4", "R5"}:
                    continue
                phase1 = answer_table.get((scenario, rs, event))
                if phase1 is None:
                    continue
                risk, reasons = frame_visual_risk(ann)
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_seen"] += 1
                    for reason in reasons:
                        risk_stats[f"reason/{reason}"] += 1
                if risk and not args.include_visual_risk:
                    if risk_stats is not None:
                        risk_stats["risk_frames_excluded"] += 1
                    continue
                if risk and risk_stats is not None:
                    risk_stats["risk_frames_retained"] += 1
                history_abs = _history(run_dir, frame_id)
                if history_abs is None:
                    continue
                history = _relative_history_paths(history_abs, data_root)
                town = _town(ann, route)
                phase1, applied_overrides = _apply_visual_subgroup_overrides(
                    phase1,
                    overrides=visual_overrides,
                    note_subgroups=note_subgroups,
                    annotation=ann,
                    scenario=scenario,
                    route_id=route_id,
                    town=town,
                    rs=rs,
                    counters=risk_stats if risk_stats is not None else Counter(),
                )
                phase2 = _phase2_answers(rs)
                answers = {**phase1, **phase2}
                yield {
                    "dataset_name": DATASET_NAME,
                    "scenario": scenario,
                    "route_id": route_id,
                    "town": town,
                    "split": split,
                    "frame_id": frame_id,
                    "rs": rs,
                    "event": event,
                    "answers": answers,
                    "phase1_answers": {key: answers[key] for key in PHASE1_ANSWER_KEYS},
                    "phase2_answers": {key: answers[key] for key in PHASE2_ANSWER_KEYS},
                    "phase1_highway_vs_phase2_r3": {
                        "phase1_highway": bool(answers["HIGHWAY"]),
                        "phase2_r3_all_no": rs == "R3",
                        "matches": bool(answers["HIGHWAY"]) == (rs == "R3"),
                    },
                    "visual_label_risk": risk,
                    "visual_label_risk_reasons": reasons,
                    "visual_subgroup_overrides_applied": applied_overrides,
                    "manual_note_subgroups": list(note_subgroups.get((scenario, route_id), [])),
                    "history_rgb_paths": history,
                    "latest_rgb_path": history[-1],
                }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """Build ``frame_index.jsonl`` and ``manifest.json``."""

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_coverage, coverage_source = load_review_coverage(
        review_root=pathlib.Path(args.review_root),
        coverage_manifest=pathlib.Path(args.coverage_manifest),
    )
    missing_review = [
        f"{scenario}/{town}"
        for scenario, towns in sorted(review_coverage.items())
        for town, item in sorted(towns.items())
        if int(item.get("completed_routes", 0)) < 1
    ]
    if missing_review:
        raise ValueError(f"fused dataset requires one completed full-frame RGB review per scenario/Town; missing={missing_review[:20]}")

    target = out_dir / "frame_index.jsonl"
    temporary = out_dir / ".frame_index.jsonl.tmp"
    temporary.unlink(missing_ok=True)
    answer_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    counters: Counter[str] = Counter()
    routes: Dict[str, set[str]] = defaultdict(set)
    phase_group_focus_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    cross_phase: Counter[str] = Counter()
    risk_stats: Counter[str] = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in iter_rows(args, risk_stats):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                split = row["split"]
                counters[f"frames/{split}"] += 1
                counters[f"frames/{split}/{row['scenario']}"] += 1
                routes[split].add(f"{row['scenario']}/{row['route_id']}")
                for key in ANSWER_KEYS:
                    value = "YES" if row["answers"][key] else "NO"
                    answer_counts[split][f"{key}:{value}"] += 1
                    phase = "phase1" if key in PHASE1_ANSWER_KEYS else "phase2"
                    phase_group_focus_counts[split][f"{phase}/{value}"] += 1
                cross = row["phase1_highway_vs_phase2_r3"]
                cross_phase[f"{split}/matches/{cross['matches']}"] += 1
                cross_phase[f"{split}/phase1_highway/{cross['phase1_highway']}/phase2_r3/{cross['phase2_r3_all_no']}"] += 1
        availability = _focus_availability(answer_counts)
        _assert_required_coverage(availability, val_ratio=float(args.val_ratio))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)
    manifest = {
        "format": "sft_new_loop_phase1_frame_index",
        "dataset_name": DATASET_NAME,
        "frame_index": str(target),
        "data_root": str(pathlib.Path(args.data_root)),
        "rgb_path_contract": (
            "history_rgb_paths/latest_rgb_path are stored relative to --data-root "
            "(normally lead_data) so train/eval can remap the index on another machine."
        ),
        "answer_table": str(args.answer_table),
        "split_seed": int(args.split_seed),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "include_visual_risk": bool(args.include_visual_risk),
        "counts": dict(counters),
        "route_counts": {split: len(value) for split, value in sorted(routes.items())},
        "focus_bin_availability": availability,
        "phase_group_focus_availability": {split: dict(counter) for split, counter in phase_group_focus_counts.items()},
        "cross_phase_highway_r3_consistency_counts": dict(cross_phase),
        "visual_label_risk_counts": dict(sorted(risk_stats.items())),
        "manual_note_paths": [item for item in str(args.manual_note_paths).split(",") if item.strip()],
        "visual_subgroup_override_contract": (
            "Top-level answer-table visual_subgroup_overrides are applied only when a route/frame carries "
            "the explicit subgroup in structured RGB audit notes/annotations. Free-text audit_evidence is never parsed "
            "as a route-level label source."
        ),
        "full_frame_rgb_review_coverage": {
            "review_root": str(args.review_root),
            "coverage_manifest": str(args.coverage_manifest),
            "coverage_source": coverage_source,
            "scenarios": len(review_coverage),
            "scenario_town_pairs": sum(len(towns) for towns in review_coverage.values()),
            "completed_routes": sum(int(item.get("completed_routes", 0)) for towns in review_coverage.values() for item in towns.values()),
        },
        "sampling_contract": (
            "train/eval expand each frame by an invisible focus key. All eight answer keys are sampled YES:NO=1:1; "
            "because there are four Phase1 keys and four Phase2 keys, Phase1-focus and Phase2-focus cases are also 1:1."
        ),
        "uncertainty_contract": (
            "No new labels are inferred from scenario names. Phase1 answers come from the audited four-question answer table; "
            "Phase2 answers come from per-frame RS labels after abnormal-route filtering and full-frame RGB review coverage checks. "
            "Visual-risk frames are excluded by default following phase2_augment."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--answer-table", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_data"))
    p.add_argument("--review-root", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809"))
    p.add_argument("--coverage-manifest", default=str(DEFAULT_COVERAGE_MANIFEST))
    p.add_argument(
        "--manual-note-paths",
        default=",".join(
            [
                str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/manual_visual_audit_notes.jsonl"),
                str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl"),
                str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl"),
            ]
        ),
    )
    p.add_argument("--scenarios", default="all")
    p.add_argument("--split-seed", type=int, default=20260813)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--max-routes", type=int, default=0)
    p.add_argument("--progress-every-routes", type=int, default=100)
    p.add_argument("--include-visual-risk", action="store_true")
    return p.parse_args()


def main() -> None:
    """CLI entry."""

    manifest = build_dataset(parse_args())
    frames = sum(value for key, value in manifest["counts"].items() if key.count("/") == 1)
    print(f"sft_new_loop_phase1 dataset: frames={frames} output={manifest['frame_index']}")


if __name__ == "__main__":
    main()
