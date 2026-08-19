#!/usr/bin/env python3
"""从 Phase2 augment eval 产物中抽取少量错例并补齐真实 RGB。

本脚本不运行 Qwen。它优先读取 `error_cases/**/case.json`；如果 eval 包里没有
`error_cases/`，则从 `cases_rank*.jsonl` / `cases.jsonl` 筛出 `all_ok=false` 的行。
随后按常见错误类型抽样，再从原始 `lead_data` 路径复制模型实际输入的 RGB history，
生成一个适合人工快速审计的小目录。
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

DEFAULT_EVAL_DIR = _AUTOMOT_ROOT / "checkpoints/sft_loop_phase2_augment_eval/base_rs_augmented_final_4rgb/20260817_172821"
DEFAULT_OUTPUT_DIR = _AUTOMOT_ROOT / "checkpoints/sft_loop_phase2_augment_audit_samples/base_4rgb_20260817"

TARGETS = (
    "invalid_answer",
    "rs1_fn",
    "rs1_fp",
    "rs2_fn",
    "rs2_fp",
    "rs4_fn",
    "rs4_fp",
    "rs5_fn",
    "rs5_fp",
    "highway_fn",
    "highway_fp",
    "multi_yes",
)
RS_KEYS = ("RS1", "RS2", "RS4", "RS5")


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
    """优先读取 error_cases；缺失时从 cases_rank*.jsonl 里筛 all_ok=false。"""

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
    """统计指定 RS 键中预测为 YES 的个数。"""

    return sum(1 for key in keys if values.get(key) == "YES")


def _target_matches(payload: Mapping[str, Any]) -> List[str]:
    """返回一个 case 命中的错误类型。"""

    gt = payload.get("gt") or {}
    parsed = payload.get("parsed") or {}
    matched: List[str] = []
    if any(key in gt and parsed.get(key) not in ("YES", "NO") for key in tuple(RS_KEYS) + ("HIGHWAY", "GROUP", "DETAIL")):
        matched.append("invalid_answer")
    if gt.get("RS1") == "YES" and parsed.get("RS1") != "YES":
        matched.append("rs1_fn")
    if gt.get("RS1") == "NO" and parsed.get("RS1") == "YES":
        matched.append("rs1_fp")
    if gt.get("RS2") == "YES" and parsed.get("RS2") != "YES":
        matched.append("rs2_fn")
    if gt.get("RS2") == "NO" and parsed.get("RS2") == "YES":
        matched.append("rs2_fp")
    if gt.get("RS4") == "YES" and parsed.get("RS4") != "YES":
        matched.append("rs4_fn")
    if gt.get("RS4") == "NO" and parsed.get("RS4") == "YES":
        matched.append("rs4_fp")
    if gt.get("RS5") == "YES" and parsed.get("RS5") != "YES":
        matched.append("rs5_fn")
    if gt.get("RS5") == "NO" and parsed.get("RS5") == "YES":
        matched.append("rs5_fp")
    if gt.get("HIGHWAY") == "YES" and parsed.get("HIGHWAY") != "YES":
        matched.append("highway_fn")
    if gt.get("HIGHWAY") == "NO" and parsed.get("HIGHWAY") == "YES":
        matched.append("highway_fp")
    if _yes_count(parsed, RS_KEYS) > 1:
        matched.append("multi_yes")
    return matched


def _resolve_rgb_path(src: str, *, data_root: pathlib.Path) -> pathlib.Path:
    """把 case.json 中的相对 RGB 路径解析到本机真实文件。"""

    raw = pathlib.Path(src)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(pathlib.Path.cwd() / raw)
        candidates.append(_AUTOMOT_ROOT / raw)
        parts = raw.parts
        if parts and parts[0] == "lead_data":
            candidates.append(data_root.joinpath(*parts[1:]))
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return candidates[-1] if candidates else raw


def _copy_case(payload: Mapping[str, Any], *, target: str, index: int, output_dir: pathlib.Path, data_root: pathlib.Path) -> Dict[str, Any]:
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
        "augment_variant": payload.get("augment_variant"),
        "scenario": payload.get("scenario"),
        "town": payload.get("town"),
        "route_id": payload.get("route_id"),
        "frame_id": payload.get("frame_id"),
        "rs": payload.get("rs"),
        "event": payload.get("event"),
        "gt": payload.get("gt"),
        "parsed": payload.get("parsed"),
        "raw_output": payload.get("raw_output"),
        "rgb_copied": copied,
        "rgb_missing": missing,
    }
    (case_dir / "audit_note.md").write_text(_render_note(row), encoding="utf-8")
    return row


def _render_note(row: Mapping[str, Any]) -> str:
    """生成每个错例目录里方便打开阅读的简短说明。"""

    return "\n".join(
        [
            f"# {row['target']} case {row['case_index']}",
            "",
            f"- scenario: `{row['scenario']}`",
            f"- route: `{row['route_id']}`",
            f"- frame: `{row['frame_id']}`",
            f"- rs/event: `{row['rs']}` / `{row['event']}`",
            f"- gt: `{json.dumps(row['gt'], ensure_ascii=False)}`",
            f"- parsed: `{json.dumps(row['parsed'], ensure_ascii=False)}`",
            f"- rgb_copied: `{row['rgb_copied']}`",
            "",
            "```text",
            str(row.get("raw_output") or ""),
            "```",
            "",
        ]
    )


def build_audit_samples(args: argparse.Namespace) -> Dict[str, Any]:
    """主流程：扫描错例、按类型抽样、复制真实 RGB。"""

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
    """生成总览 Markdown。"""

    lines = [
        "# Phase2 Augment Eval Audit Samples",
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
    """解析命令行参数。"""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--per-target", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument(
        "--targets",
        default="invalid_answer,rs1_fn,rs1_fp,rs2_fn,rs2_fp,rs4_fn,rs4_fp,rs5_fn,rs5_fp,highway_fn,highway_fp,multi_yes",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    result = build_audit_samples(parse_args())
    print(f"audit samples written: {result['output_dir']} selected={result['selected']}")
