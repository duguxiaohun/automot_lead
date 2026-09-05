#!/usr/bin/env python3
"""从 new Phase3 eval 产物中抽取少量错例并补齐真实 RGB。

本脚本不运行 Qwen。它优先读取 ``error_cases/**/case.json``；如果 eval 包里没有
``error_cases/``，则从 ``cases_rank*.jsonl`` / ``cases.jsonl`` 筛出 ``all_ok=false``
的行。随后按常见动作错误类型抽样，再从原始 ``lead_data`` 路径复制模型实际输入的
RGB history，生成一个适合人工快速审计的小目录。
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

DEFAULT_EVAL_DIR = _AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_eval/base_high_level_action_final_4rgb"
DEFAULT_OUTPUT_DIR = _AUTOMOT_ROOT / "checkpoints/sft_new_loop_phase3_audit_samples/base_high_level_action_4rgb"

ACTION_KEYS = ("DECELERATE", "STOP", "RESUME", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT")
INVALID_KEY = "INVALID_ACTION_CONTEXT"
TARGETS = (
    "invalid_answer",
    "decelerate_fn",
    "decelerate_fp",
    "stop_fn",
    "stop_fp",
    "resume_fn",
    "resume_fp",
    "lane_change_left_fn",
    "lane_change_left_fp",
    "lane_change_right_fn",
    "lane_change_right_fp",
    "lane_change_side_swap",
    "longitudinal_multi_yes",
    "invalid_context_fn",
    "invalid_context_fp",
    "invalid_context_not_all_no",
    "no_action_fp",
)


def _read_json(path: pathlib.Path) -> Mapping[str, Any] | None:
    """读取单个 JSON；坏文件跳过并交给 summary 计数。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_case_jsons(eval_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """遍历 eval 目录下的错例 case.json。"""

    yield from sorted((eval_dir / "error_cases").glob("*/*/case.json"))


def _iter_error_payloads(eval_dir: pathlib.Path) -> Iterable[Mapping[str, Any] | None]:
    """优先读取 error_cases；缺失时从 cases*.jsonl 里筛 all_ok=false。"""

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
    """统计指定键中预测为 YES 的个数。"""

    return sum(1 for key in keys if values.get(key) == "YES")


def _target_matches(payload: Mapping[str, Any]) -> List[str]:
    """返回一个 case 命中的错误类型。"""

    gt = payload.get("gt") or {}
    parsed = payload.get("parsed") or {}
    matched: List[str] = []
    asked = [key for key in (*ACTION_KEYS, INVALID_KEY) if key in gt]
    if any(parsed.get(key) not in ("YES", "NO") for key in asked):
        matched.append("invalid_answer")
    for key in ACTION_KEYS:
        if key not in gt:
            continue
        lower = key.lower()
        if gt.get(key) == "YES" and parsed.get(key) != "YES":
            matched.append(f"{lower}_fn")
        if gt.get(key) == "NO" and parsed.get(key) == "YES":
            matched.append(f"{lower}_fp")
    if gt.get("LANE_CHANGE_LEFT") == "YES" and parsed.get("LANE_CHANGE_RIGHT") == "YES":
        matched.append("lane_change_side_swap")
    if gt.get("LANE_CHANGE_RIGHT") == "YES" and parsed.get("LANE_CHANGE_LEFT") == "YES":
        matched.append("lane_change_side_swap")
    if _yes_count(parsed, ("DECELERATE", "STOP", "RESUME")) > 1:
        matched.append("longitudinal_multi_yes")
    if gt.get(INVALID_KEY) == "YES" and parsed.get(INVALID_KEY) != "YES":
        matched.append("invalid_context_fn")
    if gt.get(INVALID_KEY) == "NO" and parsed.get(INVALID_KEY) == "YES":
        matched.append("invalid_context_fp")
    if gt.get(INVALID_KEY) == "YES" and any(parsed.get(key) == "YES" for key in ACTION_KEYS):
        matched.append("invalid_context_not_all_no")
    if str(payload.get("action_signature")) == "NONE" and _yes_count(parsed, ACTION_KEYS) > 0:
        matched.append("no_action_fp")
    return matched


def _resolve_rgb_path(src: str, *, data_root: pathlib.Path) -> pathlib.Path:
    """把 case.json 中的相对 RGB 路径解析到本机真实文件。"""

    raw = pathlib.Path(src)
    candidates: List[pathlib.Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(pathlib.Path.cwd() / raw)
        candidates.append(_AUTOMOT_ROOT / raw)
        candidates.append(data_root / raw)
        parts = raw.parts
        if parts and parts[0] == "lead_data":
            candidates.append(data_root.joinpath(*parts[1:]))
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return candidates[-1] if candidates else raw


def _render_note(row: Mapping[str, Any]) -> str:
    """生成每个错例目录里方便打开阅读的简短说明。"""

    return "\n".join(
        [
            f"# {row['target']} case {row['case_index']}",
            "",
            f"- scenario: `{row['scenario']}`",
            f"- route: `{row['route_id']}`",
            f"- frame: `{row['frame_id']}`",
            f"- true_rs / prompt_road_structure: `{row['true_rs']}` / `{row['prompt_road_structure']}`",
            f"- context_id / question_domain: `{row['context_id']}` / `{row['question_domain']}`",
            f"- action_signature: `{row.get('action_signature') or 'n/a'}`",
            f"- route target xy (x forward, y negative left): `{row.get('goal_ego_xy')}`",
            f"- invalid_source: `{row.get('invalid_source') or 'n/a'}`",
            f"- action_evidence: `{json.dumps(row.get('action_evidence') or {}, ensure_ascii=False)}`",
            f"- gt: `{json.dumps(row['gt'], ensure_ascii=False)}`",
            f"- parsed: `{json.dumps(row['parsed'], ensure_ascii=False)}`",
            f"- history source indices (oldest→newest): `{row.get('history_rgb_selected_indices') or []}`",
            f"- rgb_copied: `{row['rgb_copied']}`",
            "",
            "## Manual RGB adjudication",
            "",
            "Review the copied frames oldest→newest. Judge the newest moment; use older frames only for motion.",
            "",
            "- given situation visually plausible: `YES / NO / AMBIGUOUS`",
            "- ego already braking or already stopped: `YES / NO / UNKNOWN`",
            "- lead gap closing across the history: `YES / NO / N/A`",
            "- ego still between the same two lane boundaries: `YES / NO / UNKNOWN`",
            "- lane change really needed, and on which side: `LEFT / RIGHT / NONE / AMBIGUOUS`",
            "- curved-lane false cue mistaken for a lane change: `YES / NO / N/A`",
            "- route target side agrees with the required lane change: `YES / NO / N/A`",
            "- error owner: `MODEL / LABEL_OR_BOUNDARY / BOTH / FORMAT`",
            "- notes: `...`",
            "",
            "```text",
            str(row.get("raw_output") or ""),
            "```",
            "",
        ]
    )


def _copy_case(
    payload: Mapping[str, Any],
    *,
    target: str,
    index: int,
    output_dir: pathlib.Path,
    data_root: pathlib.Path,
) -> Dict[str, Any]:
    """复制一个错例的 case.json 与 RGB，并返回 manifest 行。"""

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
        "scenario": payload.get("scenario"),
        "town": payload.get("town"),
        "route_id": payload.get("route_id"),
        "frame_id": payload.get("frame_id"),
        "true_rs": payload.get("true_rs"),
        "prompt_road_structure": payload.get("prompt_road_structure"),
        "context_id": payload.get("context_id"),
        "question_domain": payload.get("question_domain"),
        "action_signature": payload.get("action_signature"),
        "action_evidence": payload.get("action_evidence"),
        "goal_ego_xy": payload.get("goal_ego_xy"),
        "invalid_source": payload.get("invalid_source"),
        "invalid_subgroups": payload.get("invalid_subgroups"),
        "event": payload.get("event"),
        "gt": payload.get("gt"),
        "parsed": payload.get("parsed"),
        "raw_output": payload.get("raw_output"),
        "history_rgb_selected_indices": list(selected_indices),
        "history_rgb_paths_used": list(selected_paths),
        "rgb_copied": copied,
        "rgb_missing": missing,
    }
    (case_dir / "audit_note.md").write_text(_render_note(row), encoding="utf-8")
    return row


def _render_summary(summary: Mapping[str, Any]) -> str:
    """生成总览 Markdown。"""

    lines = [
        "# new Phase3 High-Level Action Eval Audit Samples",
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
    lines.extend(["", "## Selected Contexts", ""])
    for key, value in sorted((summary.get("selected_contexts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Selected INVALID Subgroups", ""])
    invalid_subgroups = summary.get("selected_invalid_subgroups") or {}
    if not invalid_subgroups:
        lines.append("- none")
    for dimension, counts in sorted(invalid_subgroups.items()):
        lines.append(f"- `{dimension}`: `{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`")
    lines.extend(["", "## Scan Counts", ""])
    for key, value in sorted((summary.get("scan_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def build_audit_samples(args: argparse.Namespace) -> Dict[str, Any]:
    """主流程：扫描错例、按类型抽样、复制真实 RGB。"""

    eval_dir = pathlib.Path(args.eval_dir)
    output_dir = pathlib.Path(args.output_dir)
    data_root = pathlib.Path(args.data_root)
    rng = random.Random(int(args.seed))
    buckets: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    scan_counts: Counter = Counter()
    bad_json = 0

    source_mode = "error_cases" if any(_iter_case_jsons(eval_dir)) else "cases_jsonl_errors"
    for payload in _iter_error_payloads(eval_dir):
        if payload is None:
            bad_json += 1
            continue
        scan_counts["error_cases"] += 1
        for target in _target_matches(payload):
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

    invalid_subgroup_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in manifest_rows:
        for dimension, value in (row.get("invalid_subgroups") or {}).items():
            invalid_subgroup_counts[str(dimension)][str(value)] += 1
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
        "selected_contexts": dict(Counter(str(row.get("context_id")) for row in manifest_rows)),
        "selected_invalid_subgroups": {
            dimension: dict(sorted(counter.items()))
            for dimension, counter in sorted(invalid_subgroup_counts.items())
        },
        "rgb_missing_cases": sum(1 for row in manifest_rows if row.get("rgb_missing")),
        "manifest_jsonl": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_summary(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--per-target", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    result = build_audit_samples(parse_args())
    print(f"audit samples written: {result['output_dir']} selected={result['selected']}")
