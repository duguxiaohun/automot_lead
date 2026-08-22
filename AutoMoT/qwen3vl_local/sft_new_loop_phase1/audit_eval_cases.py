#!/usr/bin/env python3
"""Sample fused Phase1+Phase2 eval errors and copy the RGB inputs.

This script does not run Qwen. It reads an eval output directory, prefers
`error_cases/**/case.json`, and falls back to `cases_rank*.jsonl` / `cases.jsonl`
with `all_ok=false`. It then buckets common fused error types and copies the RGB
history that the model actually saw into a small review directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]

DEFAULT_EVAL_DIR = _AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_eval"
DEFAULT_OUTPUT_DIR = _AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase1_audit_samples/fused_4rgb"

PHASE1_KEYS = ("HIGHWAY", "STATIC_OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL")
PHASE2_RS_KEYS = ("RS1", "RS2", "RS4", "RS5")
HIERARCHICAL_KEYS = ("RS_HIGHWAY", "GROUP")
ANSWER_KEYS = PHASE1_KEYS + PHASE2_RS_KEYS + HIERARCHICAL_KEYS
TARGETS = (
    "invalid_answer",
    "highway_fn",
    "highway_fp",
    "static_obstacle_fn",
    "static_obstacle_fp",
    "vulnerable_fn",
    "vulnerable_fp",
    "traffic_light_abnormal_fn",
    "traffic_light_abnormal_fp",
    "rs1_fn",
    "rs1_fp",
    "rs2_fn",
    "rs2_fp",
    "rs4_fn",
    "rs4_fp",
    "rs5_fn",
    "rs5_fp",
    "rs_highway_fn",
    "rs_highway_fp",
    "group_fn",
    "group_fp",
    "multi_yes_phase2",
    "subset_unasked_line_leak",
)
TARGET_KEY_NAMES = {
    "HIGHWAY": "highway",
    "STATIC_OBSTACLE": "static_obstacle",
    "VULNERABLE": "vulnerable",
    "TRAFFIC_LIGHT_ABNORMAL": "traffic_light_abnormal",
    "RS1": "rs1",
    "RS2": "rs2",
    "RS4": "rs4",
    "RS5": "rs5",
    "RS_HIGHWAY": "rs_highway",
    "GROUP": "group",
}


def _read_json(path: pathlib.Path) -> Mapping[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_case_jsons(eval_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    yield from sorted((eval_dir / "error_cases").glob("*/*/case.json"))


def _iter_error_payloads(eval_dir: pathlib.Path) -> Iterable[Mapping[str, Any] | None]:
    case_paths = list(_iter_case_jsons(eval_dir))
    if case_paths:
        for case_path in case_paths:
            yield _read_json(case_path)
        return
    for case_jsonl in sorted(eval_dir.glob("cases_rank*.jsonl")) + sorted(eval_dir.glob("cases.jsonl")):
        for line in case_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                yield None
                continue
            if not bool(payload.get("all_ok", False)):
                yield payload


def _yes_count(values: Mapping[str, Any], keys: Sequence[str]) -> int:
    return sum(1 for key in keys if values.get(key) == "YES")


def _expected_output_keys(payload: Mapping[str, Any], gt: Mapping[str, Any]) -> set[str]:
    spec = payload.get("prompt_spec") or payload.get("augment_spec") or {}
    output_keys = spec.get("output_keys")
    if isinstance(output_keys, list):
        return {str(key) for key in output_keys}
    return {str(key) for key in gt.keys()}


def _target_matches(payload: Mapping[str, Any]) -> List[str]:
    gt = payload.get("gt") or {}
    parsed = payload.get("parsed") or {}
    expected_keys = _expected_output_keys(payload, gt)
    matched: List[str] = []

    if any(key in gt and parsed.get(key) not in ("YES", "NO") for key in expected_keys):
        matched.append("invalid_answer")

    for key in ANSWER_KEYS:
        if key not in gt:
            continue
        label = TARGET_KEY_NAMES[key]
        gt_value = gt.get(key)
        pred_value = parsed.get(key)
        if gt_value == "YES" and pred_value != "YES":
            matched.append(f"{label}_fn")
        if gt_value == "NO" and pred_value == "YES":
            matched.append(f"{label}_fp")

    if _yes_count(parsed, PHASE2_RS_KEYS) > 1:
        matched.append("multi_yes_phase2")

    extra_answer_keys = [
        key
        for key, value in parsed.items()
        if key in ANSWER_KEYS and key not in expected_keys and value in ("YES", "NO")
    ]
    if extra_answer_keys:
        matched.append("subset_unasked_line_leak")
    return matched


def _resolve_rgb_path(src: str, *, data_root: pathlib.Path) -> pathlib.Path:
    raw = pathlib.Path(src)
    candidates: List[pathlib.Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        parts = raw.parts
        if "lead_data" in parts:
            lead_idx = parts.index("lead_data")
            tail = parts[lead_idx + 1 :]
            candidates.append(data_root.joinpath(*tail))
    else:
        candidates.append(pathlib.Path.cwd() / raw)
        candidates.append(_AUTOMOT_ROOT / raw)
        parts = raw.parts
        if parts and parts[0] == "lead_data":
            candidates.append(data_root.joinpath(*parts[1:]))
        else:
            candidates.append(data_root / raw)
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return candidates[-1] if candidates else raw


def _copy_case(
    payload: Mapping[str, Any],
    *,
    target: str,
    index: int,
    output_dir: pathlib.Path,
    data_root: pathlib.Path,
) -> Dict[str, Any]:
    scenario = str(payload.get("scenario", "unknown"))
    frame_id = payload.get("frame_id", "x")
    case_index = int(payload.get("case_index", index))
    safe_scenario = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in scenario)
    case_dir = output_dir / target / f"case_{index:03d}_src{case_index:05d}_{safe_scenario}_f{frame_id}"
    rgb_dir = case_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied = 0
    missing: List[str] = []
    selected_indices = payload.get("history_rgb_selected_indices") or []
    selected_paths = payload.get("history_rgb_paths_used") or []
    for source_idx, src in zip(selected_indices, selected_paths):
        src_path = _resolve_rgb_path(str(src), data_root=data_root)
        dst = rgb_dir / f"history_source_{source_idx}_{src_path.name}"
        if src_path.is_file() and src_path.stat().st_size > 0:
            shutil.copy2(src_path, dst)
            copied += 1
        else:
            missing.append(str(src))

    row = {
        "target": target,
        "case_dir": str(case_dir),
        "case_index": payload.get("case_index"),
        "task": payload.get("task") or payload.get("focus_question"),
        "augment_variant": payload.get("augment_variant"),
        "augment_balance_key": payload.get("augment_balance_key"),
        "scenario": payload.get("scenario"),
        "town": payload.get("town"),
        "route_id": payload.get("route_id"),
        "frame_id": payload.get("frame_id"),
        "rs": payload.get("rs"),
        "event": payload.get("event"),
        "gt": payload.get("gt"),
        "parsed": payload.get("parsed"),
        "ok_by_key": payload.get("ok_by_key"),
        "raw_output": payload.get("raw_output"),
        "rgb_copied": copied,
        "rgb_missing": missing,
    }
    (case_dir / "audit_note.md").write_text(_render_note(row), encoding="utf-8")
    return row


def _render_note(row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# {row['target']} case {row['case_index']}",
            "",
            f"- task: `{row.get('task')}`",
            f"- augment_variant: `{row.get('augment_variant')}`",
            f"- augment_balance_key: `{row.get('augment_balance_key')}`",
            f"- scenario: `{row['scenario']}`",
            f"- route: `{row['route_id']}`",
            f"- frame: `{row['frame_id']}`",
            f"- rs/event: `{row['rs']}` / `{row['event']}`",
            f"- gt: `{json.dumps(row['gt'], ensure_ascii=False)}`",
            f"- parsed: `{json.dumps(row['parsed'], ensure_ascii=False)}`",
            f"- ok_by_key: `{json.dumps(row.get('ok_by_key'), ensure_ascii=False)}`",
            f"- rgb_copied: `{row['rgb_copied']}`",
            "",
            "```text",
            str(row.get("raw_output") or ""),
            "```",
            "",
        ]
    )


def build_audit_samples(args: argparse.Namespace) -> Dict[str, Any]:
    eval_dir = pathlib.Path(args.eval_dir)
    output_dir = pathlib.Path(args.output_dir)
    data_root = pathlib.Path(args.data_root)
    rng = random.Random(int(args.seed))
    buckets: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    scan_counts: Counter[str] = Counter()
    bad_json = 0

    source_mode = "error_cases" if any(_iter_case_jsons(eval_dir)) else "cases_jsonl_errors"
    for payload in _iter_error_payloads(eval_dir):
        if payload is None:
            bad_json += 1
            continue
        matched = _target_matches(payload)
        scan_counts["error_cases"] += 1
        for target in matched:
            scan_counts[f"matched/{target}"] += 1
            buckets[target].append(payload)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output dir is not empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    selected_targets = [target for target in args.targets.split(",") if target]
    for target in selected_targets:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choices={TARGETS}")
        cases = list(buckets.get(target, []))
        rng.shuffle(cases)
        for idx, payload in enumerate(cases[: int(args.per_target)]):
            manifest_rows.append(
                _copy_case(payload, target=target, index=idx, output_dir=output_dir, data_root=data_root)
            )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "seed": int(args.seed),
        "per_target": int(args.per_target),
        "targets": selected_targets,
        "source_mode": source_mode,
        "bad_json": bad_json,
        "scan_counts": dict(scan_counts),
        "selected": len(manifest_rows),
        "selected_by_target": dict(Counter(row["target"] for row in manifest_rows)),
        "rgb_missing_cases": sum(1 for row in manifest_rows if row.get("rgb_missing")),
        "manifest_jsonl": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_summary(summary), encoding="utf-8")
    return summary


def _render_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SFT New Loop Phase1 Eval Audit Samples",
        "",
        f"- eval_dir: `{summary['eval_dir']}`",
        f"- selected: `{summary['selected']}`",
        f"- rgb_missing_cases: `{summary['rgb_missing_cases']}`",
        f"- source_mode: `{summary.get('source_mode', 'unknown')}`",
        "",
        "## Selected By Target",
        "",
    ]
    for key, value in sorted((summary.get("selected_by_target") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Scan Counts", ""])
    for key, value in sorted((summary.get("scan_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--per-target", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument(
        "--targets",
        default=",".join(TARGETS),
        help="Comma-separated target buckets to sample.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    result = build_audit_samples(parse_args())
    print(f"audit samples written: {result['output_dir']} selected={result['selected']}")
