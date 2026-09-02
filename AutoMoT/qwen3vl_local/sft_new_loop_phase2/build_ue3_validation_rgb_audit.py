#!/usr/bin/env python3
"""把多 seed 的 UE3 validation 正例整理成四帧 RGB 审计包。

默认保留旧行为：从训练期 fallback step 只导出至少一个 seed 漏判的 case。
``--source-mode eval --include-correct`` 则从独立 eval ``cases*.jsonl`` 导出同一
采样集的全部 UE3 正例，用于同时对比稳定答对与稳定答错的 RGB 证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import tarfile
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw


Identity = Tuple[str, str, int, str]


def _is_yes(value: Any) -> bool:
    """兼容训练 generation record 的 bool 与独立 eval 的 YES/NO 字符串。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "YES"


def _record_balance_key(record: Mapping[str, Any]) -> str:
    """读取训练期或独立 eval 的采样桶名。"""

    return str(record.get("balance_key") or record.get("augment_balance_key") or "")


def _record_ue3_gt(record: Mapping[str, Any]) -> bool:
    """读取 UE3 真值，并兼容两种 case schema。"""

    for field in ("answers", "event_answers", "gt"):
        values = record.get(field)
        if isinstance(values, Mapping) and "UE3" in values:
            return _is_yes(values.get("UE3"))
    return False


def _record_ue3_prediction(record: Mapping[str, Any]) -> bool:
    """读取 UE3 预测。格式失败/缺失按非 YES 处理，与 recall 口径一致。"""

    values = record.get("parsed")
    return isinstance(values, Mapping) and _is_yes(values.get("UE3"))


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    """读取 JSON object。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL，并在坏行处报告精确位置。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_no}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object: {path}:{line_no}")
            yield payload


def _identity(row: Mapping[str, Any]) -> Identity:
    """返回可跨 index、seed 和 generation record 对齐的帧身份。"""

    return (
        str(row.get("scenario", "")),
        str(row.get("route_id", "")),
        int(row.get("frame_id", -1)),
        str(row.get("question_domain", "")),
    )


def _load_index(path: pathlib.Path) -> Dict[Identity, Dict[str, Any]]:
    """读取 validation index，用于补齐模型未见的中间两帧。"""

    rows: Dict[Identity, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        if str(row.get("split")) != "val":
            continue
        key = _identity(row)
        previous = rows.get(key)
        if previous is not None and previous.get("history_rgb_paths") != row.get("history_rgb_paths"):
            raise ValueError(f"ambiguous validation identity in index: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no validation rows in index: {path}")
    return rows


def _resolve_rgb_path(raw: str, data_root: pathlib.Path) -> pathlib.Path:
    """解析相对 RGB 路径，并兼容旧绝对 lead_data 路径。"""

    path = pathlib.Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [data_root / path, pathlib.Path.cwd() / path]
    if "lead_data" in path.parts:
        index = path.parts.index("lead_data")
        candidates.append(data_root.joinpath(*path.parts[index + 1 :]))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve RGB path {raw!r}; tried={[str(item) for item in candidates]}")


def _safe_case_name(identity: Identity) -> str:
    """生成短且稳定的审计目录名。"""

    scenario, route_id, frame_id, domain = identity
    digest = hashlib.sha256("|".join(map(str, identity)).encode("utf-8")).hexdigest()[:10]
    safe_scenario = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scenario)
    return f"{safe_scenario}_f{frame_id}_{domain}_{digest}"


def _contact_sheet(paths: Sequence[pathlib.Path], labels: Sequence[str], output: pathlib.Path, title: str) -> None:
    """将四张 1152×384 stitched RGB 排成 2×2 contact sheet。"""

    if len(paths) != 4:
        raise ValueError(f"four RGB frames required, got {len(paths)}")
    thumb_width = 768
    thumb_height = 256
    header_height = 42
    label_height = 24
    canvas = Image.new("RGB", (thumb_width * 2, header_height + (thumb_height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill="black")
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_width
        y = header_height + (index // 2) * (thumb_height + label_height)
        canvas.paste(tile, (x, y))
        draw.text((x + 8, y + thumb_height + 5), f"{label}: {path.name}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=88, optimize=True)


def _audit_note(case: Mapping[str, Any]) -> str:
    """生成强制逐帧判断、禁止按 scenario 名倒推的人工审计模板。"""

    return "\n".join(
        [
            f"# UE3 validation RGB audit: {case['case_name']}",
            "",
            f"- scenario（仅索引，不是判定证据）: `{case['scenario']}`",
            f"- route/frame: `{case['route_id']}` / `{case['frame_id']}`",
            f"- failed seeds: `{case['failed_seeds']}`",
            f"- correct seeds: `{case['correct_seeds']}`",
            f"- sampled seeds: `{case['sampled_seeds']}`",
            f"- model-visible frames: `{case['model_visible_indices']}`",
            "",
            "## 必须按 t0 → t1 → t2 → t3 逐帧填写",
            "",
            "- t0 actor 与车道边界/ego corridor 的关系：`...`",
            "- t1 相对 t0 的横向位移：`...`",
            "- t2 相对 t1 的横向位移：`...`",
            "- t3 是否仍在进入或即将占用 ego immediate future corridor：`YES / NO / AMBIGUOUS`",
            "- t0/t3 两端点是否足以证明横向进入：`YES / NO / AMBIGUOUS`",
            "- 四帧整体是否支持 UE3=YES：`YES / NO / AMBIGUOUS`",
            "- 是否只是 ego 前进视差、静态停车、事故、施工或既有队列：`YES / NO`",
            "- 是否属于事件 span 起止边界过宽：`YES / NO / AMBIGUOUS`",
            "- visual class：`VISIBLE_ACTIVE / PRE_EVENT / POST_EVENT / DOMAIN_CONFLICT / 2RGB_UNOBSERVABLE / AMBIGUOUS`",
            "- error owner：`MODEL / LABEL_OR_SPAN / 2RGB_INFORMATION / AMBIGUOUS`",
            "- notes：`...`",
            "",
        ]
    )


def _load_fallback_records(train_root: pathlib.Path) -> Dict[Identity, Dict[str, Dict[str, Any]]]:
    """读取各 seed fallback step 的 UE3 正例。"""

    sampled: Dict[Identity, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for seed_dir in sorted(train_root.glob("seed_*")):
        fallback_path = seed_dir / "fallback_generation.json"
        cases_path = seed_dir / "generation_val_cases.jsonl"
        if not fallback_path.is_file() or not cases_path.is_file():
            raise FileNotFoundError(f"missing fallback/cases for {seed_dir}")
        fallback = _read_json(fallback_path)
        selected_step = int(fallback["step"])
        seed = seed_dir.name
        for source_record in _iter_jsonl(cases_path):
            if int(source_record.get("step", -1)) != selected_step:
                continue
            if not _record_balance_key(source_record).endswith("/class/UE3"):
                continue
            if not _record_ue3_gt(source_record):
                continue
            record = dict(source_record)
            record["seed"] = seed
            record["selected_step"] = selected_step
            sampled[_identity(record)][seed] = record
    return sampled


def _load_eval_records(eval_root: pathlib.Path) -> Dict[Identity, Dict[str, Dict[str, Any]]]:
    """读取各 seed 独立 eval 的同一组 UE3 正例。"""

    sampled: Dict[Identity, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    seed_identities: Dict[str, set[Identity]] = {}
    seed_dirs = sorted(path for path in eval_root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* eval directories under {eval_root}")
    for seed_dir in seed_dirs:
        case_paths = sorted(seed_dir.glob("cases*.jsonl"))
        if not case_paths:
            raise FileNotFoundError(f"missing cases*.jsonl for {seed_dir}")
        seed = seed_dir.name
        identities: set[Identity] = set()
        for cases_path in case_paths:
            for source_record in _iter_jsonl(cases_path):
                if not _record_balance_key(source_record).endswith("/class/UE3"):
                    continue
                if not _record_ue3_gt(source_record):
                    continue
                record = dict(source_record)
                record["seed"] = seed
                record["selected_step"] = "final"
                key = _identity(record)
                previous = sampled[key].get(seed)
                if previous is not None and previous != record:
                    raise ValueError(f"conflicting duplicate eval case for {seed}: {key}")
                sampled[key][seed] = record
                identities.add(key)
        if not identities:
            raise ValueError(f"no UE3 positive cases in {seed_dir}")
        seed_identities[seed] = identities
    reference_seed = sorted(seed_identities)[0]
    reference = seed_identities[reference_seed]
    mismatched = {
        seed: {
            "missing_vs_reference": len(reference - identities),
            "extra_vs_reference": len(identities - reference),
        }
        for seed, identities in sorted(seed_identities.items())
        if identities != reference
    }
    if mismatched:
        raise ValueError(
            "eval seeds did not score the same UE3 identities; refusing a biased TP/FN comparison: "
            f"reference={reference_seed} cases={len(reference)} mismatched={mismatched}"
        )
    return sampled


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    """读取多 seed UE3 预测，构建去重的四帧 RGB 审计目录。"""

    experiment_root = pathlib.Path(args.experiment_root)
    train_root = experiment_root / "train_runs"
    index = _load_index(pathlib.Path(args.index))
    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else experiment_root / "ue3_validation_rgb_audit"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite to rebuild")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_mode = str(getattr(args, "source_mode", "fallback"))
    if source_mode == "fallback":
        sampled = _load_fallback_records(train_root)
        source_root = train_root
    elif source_mode == "eval":
        eval_root_arg = str(getattr(args, "eval_root", "") or "")
        source_root = pathlib.Path(eval_root_arg) if eval_root_arg else experiment_root / "route_diverse_validation_rescore"
        sampled = _load_eval_records(source_root)
    else:
        raise ValueError(f"unsupported source_mode={source_mode!r}")
    if not sampled:
        raise ValueError(f"no UE3 positive records found in {source_root}")

    # generation_val_cases.jsonl 在断点恢复后可能重复出现同一 step；先按 identity+seed
    # 去重，再统计分母，避免恢复训练把同一 validation case 重复记账。
    per_seed = Counter()
    per_seed_failures = Counter()
    per_seed_scenario_total: Dict[str, Counter[str]] = defaultdict(Counter)
    per_seed_scenario_failures: Dict[str, Counter[str]] = defaultdict(Counter)
    per_seed_route_total: Dict[str, Counter[str]] = defaultdict(Counter)
    per_seed_route_failures: Dict[str, Counter[str]] = defaultdict(Counter)
    for key, records in sampled.items():
        route_key = f"{key[0]}/{key[1]}"
        for seed, record in records.items():
            per_seed[seed] += 1
            per_seed_scenario_total[seed][key[0]] += 1
            per_seed_route_total[seed][route_key] += 1
            if not _record_ue3_prediction(record):
                per_seed_failures[seed] += 1
                per_seed_scenario_failures[seed][key[0]] += 1
                per_seed_route_failures[seed][route_key] += 1

    route_reports = {
        seed: {
            route: {
                "cases": int(total),
                "false_negatives": int(per_seed_route_failures[seed][route]),
                "recall": 1.0 - float(per_seed_route_failures[seed][route]) / max(1.0, float(total)),
            }
            for route, total in sorted(counts.items())
        }
        for seed, counts in sorted(per_seed_route_total.items())
    }
    route_macro_recall = {
        seed: sum(float(report["recall"]) for report in routes.values()) / max(1, len(routes))
        for seed, routes in sorted(route_reports.items())
    }

    failure_keys = {
        key
        for key, records in sampled.items()
        if any(not _record_ue3_prediction(record) for record in records.values())
    }
    include_correct = bool(getattr(args, "include_correct", False))
    audit_keys = sorted(sampled if include_correct else failure_keys)
    manifest_rows = []
    missing_index = []
    for key in audit_keys:
        source = index.get(key)
        if source is None:
            missing_index.append(key)
            continue
        raw_paths = [str(value) for value in source.get("history_rgb_paths", [])]
        if len(raw_paths) != 4:
            raise ValueError(f"index row does not contain four RGB frames: {key} paths={raw_paths}")
        paths = [_resolve_rgb_path(raw, data_root) for raw in raw_paths]
        records = sampled[key]
        failed_seeds = sorted(
            seed for seed, record in records.items() if not _record_ue3_prediction(record)
        )
        correct_seeds = sorted(seed for seed, record in records.items() if _record_ue3_prediction(record))
        sampled_seeds = sorted(records)
        case_name = _safe_case_name(key)
        case_dir = output_dir / "cases" / case_name
        rgb_dir = case_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        if args.copy_originals:
            for frame_index, path in enumerate(paths):
                destination = rgb_dir / f"t{frame_index}_{path.name}"
                shutil.copy2(path, destination)
                copied_paths.append(str(destination))
        sheet_path = case_dir / "contact_sheet.jpg"
        _contact_sheet(
            paths,
            ["t0 oldest (visible)", "t1 audit-only", "t2 audit-only", "t3 newest (visible)"],
            sheet_path,
            title=(
                f"{key[0]} frame={key[2]} "
                f"TP={','.join(correct_seeds) or '-'} FN={','.join(failed_seeds) or '-'}"
            ),
        )
        row = {
            "case_name": case_name,
            "scenario": key[0],
            "route_id": key[1],
            "frame_id": key[2],
            "question_domain": key[3],
            "failed_seeds": failed_seeds,
            "correct_seeds": correct_seeds,
            "sampled_seeds": sampled_seeds,
            "prediction_pattern": {
                seed: "TP" if seed in correct_seeds else "FN" for seed in sampled_seeds
            },
            "model_visible_indices": [0, 3],
            "history_rgb_paths_all4": [str(path) for path in paths],
            "copied_rgb_paths": copied_paths,
            "contact_sheet": str(sheet_path),
            "predictions": {
                seed: {
                    "step": record.get("selected_step"),
                    "parsed": record.get("parsed"),
                    "raw_output": record.get("raw_output"),
                }
                for seed, record in sorted(records.items())
            },
        }
        (case_dir / "case.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        (case_dir / "audit_note.md").write_text(_audit_note(row), encoding="utf-8")
        manifest_rows.append(row)

    if missing_index:
        raise ValueError(f"generation records missing from validation index: {missing_index[:10]}")
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    unique_scenarios = Counter(row["scenario"] for row in manifest_rows)
    failure_rows = [row for row in manifest_rows if row["failed_seeds"]]
    failure_scenarios = Counter(row["scenario"] for row in failure_rows)
    failure_multiplicity = Counter(len(row["failed_seeds"]) for row in failure_rows)
    all_correct_rows = [row for row in manifest_rows if not row["failed_seeds"]]
    summary = {
        "format": "sft_new_loop_phase2_ue3_validation_rgb_audit_v2",
        "experiment_root": str(experiment_root),
        "source_mode": source_mode,
        "source_root": str(source_root),
        "include_correct": include_correct,
        "index": str(args.index),
        "model_visible_indices": [0, 3],
        "audit_visible_indices": [0, 1, 2, 3],
        "sampled_ue3_by_seed": dict(sorted(per_seed.items())),
        "false_negatives_by_seed": dict(sorted(per_seed_failures.items())),
        "false_negative_rate_by_seed": {
            seed: per_seed_failures[seed] / max(1, total) for seed, total in sorted(per_seed.items())
        },
        "scenario_total_by_seed": {
            seed: dict(sorted(counts.items())) for seed, counts in sorted(per_seed_scenario_total.items())
        },
        "scenario_false_negatives_by_seed": {
            seed: dict(sorted(counts.items())) for seed, counts in sorted(per_seed_scenario_failures.items())
        },
        "ue3_route_reports_by_seed": route_reports,
        "ue3_route_macro_recall_by_seed": route_macro_recall,
        "unique_audit_cases": len(manifest_rows),
        "unique_audit_cases_by_scenario": dict(sorted(unique_scenarios.items())),
        "unique_failure_cases": len(failure_rows),
        "unique_failure_cases_by_scenario": dict(sorted(failure_scenarios.items())),
        "unique_all_seed_correct_cases": len(all_correct_rows),
        "failed_seed_multiplicity": dict(sorted(failure_multiplicity.items())),
        "manifest": str(manifest_path),
        "contract": "Do not change prompt or labels from scenario names; inspect all four RGB frames and fill audit_note.md.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_lines = [
        "# UE3 validation RGB audit",
        "",
        f"- unique failure cases: `{summary['unique_failure_cases']}`",
        f"- unique all-seed-correct controls: `{summary['unique_all_seed_correct_cases']}`",
        f"- unique audited cases: `{summary['unique_audit_cases']}`",
        f"- per-seed UE3 samples: `{summary['sampled_ue3_by_seed']}`",
        f"- per-seed false negatives: `{summary['false_negatives_by_seed']}`",
        f"- per-seed route-macro UE3 recall: `{summary['ue3_route_macro_recall_by_seed']}`",
        f"- unique cases by scenario: `{summary['unique_failure_cases_by_scenario']}`",
        f"- failed-seed multiplicity: `{summary['failed_seed_multiplicity']}`",
        "",
        "必须查看 cases/*/contact_sheet.jpg 或四张原图并填写 audit_note.md；scenario 名不能作为标签证据。",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    if args.archive:
        archive_path = output_dir.with_suffix(".tar.gz")
        summary["archive"] = str(archive_path)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(output_dir, arcname=output_dir.name)
    return summary


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        default="checkpoints/sft_new_loop_phase2_frozen_protocol/v3_frozen_3seed_unseen456_20260831",
    )
    parser.add_argument("--index", default="checkpoints/sft_new_loop_phase2_data/frame_index.jsonl")
    parser.add_argument("--data-root", default="lead_data")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-mode", choices=("fallback", "eval"), default="fallback")
    parser.add_argument(
        "--eval-root",
        default="",
        help="source-mode=eval 时包含 seed_*/cases*.jsonl 的目录",
    )
    parser.add_argument(
        "--include-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="导出所有 UE3 正例，而不仅是至少一个 seed 的假阴性",
    )
    parser.add_argument("--copy-originals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    summary = build_audit(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
