"""LeadMoT closed-loop eval 结果按 LEAD scenario 二次聚合。

逻辑：
- 扫 `<eval_base>/<signature>/eval_per_route/eval_<route_id>.json`（launcher 落盘的逐 route 结果）
- 根据 `data/lead/<Scenario>/<Town>_<route_key>.xml` 建反向映射
- 按 scenario 聚合 leaderboard 给的 score_composed / score_route / score_penalty
- 输出 summary_all.json、summary_report.md、scenario_table.csv、route_results.csv
  与 scenarios/<Scenario>/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict

from .scenario_picker import build_route_to_scenario


_SIG_RE = re.compile(r".+__.+__bev[01](?:__ema[01])?$")


def _load_json(path: pathlib.Path) -> dict | None:
    """容错读取 leaderboard JSON。

    leaderboard 的输出有时会因为中途失败留下半截文件；这里统一吞掉读取错误，
    让聚合阶段尽量处理其它已完成 route，而不是因为单个坏文件整体退出。
    """
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[aggregate] failed to load {path}: {e}", file=sys.stderr)
        return None


def _stat(values: list[float]) -> dict[str, float]:
    """把一组 route 分数压成 webapp/summary.json 都能直接消费的统计量。"""
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


_METRIC_KEYS = ("score_composed", "score_route", "score_penalty")
_SUCCESS_STATUSES = {"Completed", "Perfect"}
_PERFECT_STATUSES = {"Perfect"}

_METRIC_EXPLANATIONS = {
    "planned_routes": "本批计划评测的 route 数，来自 run_manifest.json；跨批次总聚合没有 manifest 时等于已评估 route 数。",
    "evaluated_routes": "已经生成 leaderboard eval_<route_id>.json 且可读取的 route 数。",
    "coverage": "evaluated_routes / planned_routes，表示这批测试实际完成落盘比例。",
    "success_rate": "status 为 Completed 或 Perfect 的 route 数 / planned_routes；缺失 JSON 的 route 计为未成功。",
    "perfect_rate": "status 为 Perfect 的 route 数 / planned_routes；Perfect 表示路线完成且没有 infractions。",
    "score_composed": "CARLA leaderboard 主分数，约等于 route completion score * penalty score，越高越好。",
    "score_route": "路线完成度分数，越高表示越接近完整跑完路线。",
    "score_penalty": "违规惩罚分数，越高表示违规越少；碰撞、闯灯、偏航等会降低它。",
    "infractions": "leaderboard 记录的违规事件条数汇总；不同违规类型仍保留在 CSV/JSON 中。",
}


def _safe_rate(numer: int, denom: int) -> float:
    """计算 0..1 比例；分母为 0 时返回 0，避免报告里出现 NaN。"""
    return float(numer) / float(denom) if denom > 0 else 0.0


def _fmt_float(value: float | None, ndigits: int = 3) -> str:
    """报告表格里的数字格式化。"""
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def _route_id_from_record(record: dict) -> int | None:
    """从 leaderboard record 里解析 route_id。

    有些版本写整数，有些版本写类似 "RouteScenario_1711" 的字符串，所以这里用
    正则兜底抽数字，兼容不同 AutoMoT/leaderboard 输出格式。
    """
    value = record.get("route_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group(0))
    return None


def _route_record(data: dict, route_id: int) -> dict | None:
    """在单个 eval JSON 中找到指定 route 的 record。

    正常每个 eval_<route_id>.json 只含一条记录；保留 route_id 匹配逻辑是为了
    兼容旧版一次写多 route 的 checkpoint JSON。
    """
    records = data.get("_checkpoint", {}).get("records", [])
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and _route_id_from_record(record) == route_id:
                return record
        if len(records) == 1 and isinstance(records[0], dict):
            return records[0]
    return None


def _extract_route_scores(data: dict, route_id: int) -> dict[str, float]:
    """抽取 leaderboard 三个核心分数。

    优先读当前 route 的 `scores`；如果 JSON 是旧格式或只有 global_record，
    则退回 `scores_mean`，保证 webapp 至少能显示可比较的分数。
    """
    record = _route_record(data, route_id)
    scores = record.get("scores", {}) if isinstance(record, dict) else {}
    if not isinstance(scores, dict):
        scores = {}
    out = {k: float(scores[k]) for k in _METRIC_KEYS if isinstance(scores.get(k), (int, float))}
    if out:
        return out

    # Leaderboard global_record stores scores_mean in current AutoMoT. Keep this
    # fallback for partial/old JSONs so aggregation can still show something.
    global_record = data.get("_checkpoint", {}).get("global_record", {})
    scores_mean = global_record.get("scores_mean", {}) if isinstance(global_record, dict) else {}
    if isinstance(scores_mean, dict):
        return {k: float(scores_mean[k]) for k in _METRIC_KEYS if isinstance(scores_mean.get(k), (int, float))}
    return {}


def _extract_route_status(data: dict, route_id: int) -> str:
    """读取 route status；缺字段时标记为 Unknown。"""
    record = _route_record(data, route_id)
    if isinstance(record, dict):
        status = record.get("status")
        if isinstance(status, str) and status:
            return status
    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {}) if isinstance(checkpoint, dict) else {}
    status = global_record.get("status") if isinstance(global_record, dict) else None
    return status if isinstance(status, str) and status else "Unknown"


def _extract_route_infractions(data: dict, route_id: int) -> dict[str, int]:
    """把 leaderboard infractions 压成每类计数。

    route record 里通常是 `{"collision_vehicle": [msg, ...]}`；少数旧格式可能
    已经是数值，这里统一转成 count，方便报告和 CSV 直接读。
    """
    record = _route_record(data, route_id)
    infractions = record.get("infractions", {}) if isinstance(record, dict) else {}
    if not isinstance(infractions, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in infractions.items():
        if isinstance(value, list):
            count = len(value)
        elif isinstance(value, (int, float)):
            count = int(value)
        elif value:
            count = 1
        else:
            count = 0
        if count:
            counts[str(key)] = count
    return counts


def _extract_num_infractions(data: dict, route_id: int) -> int:
    """读取 route 违规总数；缺字段时用各类 infractions 计数求和。"""
    record = _route_record(data, route_id)
    if isinstance(record, dict) and isinstance(record.get("num_infractions"), int):
        return int(record["num_infractions"])
    return sum(_extract_route_infractions(data, route_id).values())


def _find_sig_dirs(eval_base: pathlib.Path,
                   leadmot_ckpt: pathlib.Path | None) -> list[pathlib.Path]:
    """定位要聚合的 ckpt signature 目录。

    signature 形如 `<ckpt_parent>__<ckpt_stem>__bev{0|1}__ema{0|1}`。
    传入 leadmot_ckpt 时优先聚合该 ckpt 相关目录；没匹配到则回退到全部 signature，
    方便旧结果目录仍然能被聚合。
    """
    if not eval_base.is_dir():
        return []
    sigs = [p for p in sorted(eval_base.iterdir()) if p.is_dir() and _SIG_RE.fullmatch(p.name)]
    if leadmot_ckpt is None:
        return sigs
    prefix = f"{leadmot_ckpt.parent.name}__{leadmot_ckpt.stem}__bev"
    matched = [p for p in sigs if p.name.startswith(prefix)]
    return matched or sigs


def _normalize_leadmot_ckpt(path: pathlib.Path | None) -> pathlib.Path | None:
    """把 checkpoint 目录解析成具体权重文件，保持手动 aggregate 与 launcher 一致。"""
    if path is None:
        return None
    root = path.expanduser().resolve()
    if root.is_file():
        return root
    if not root.is_dir():
        return root

    candidates = [
        root / "best.pt",
        root / "latest.pt",
        root / "latest" / "best.pt",
        root / "latest" / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    pools: list[pathlib.Path] = []
    for pattern in ("step-checkpoint-*.pt", "checkpoint-epoch*.pt", "*.pt", "*.safetensors"):
        pools.extend(p for p in root.glob(pattern) if p.is_file())
        latest = root / "latest"
        if latest.is_dir():
            pools.extend(p for p in latest.glob(pattern) if p.is_file())
    return max(pools, key=lambda p: p.stat().st_mtime).resolve() if pools else root


def _load_run_manifest(run_dir: pathlib.Path) -> dict | None:
    """读 run_eval.sh 写出的 run_manifest.json，拿到本批 route_id 列表。"""
    mp = run_dir / "run_manifest.json"
    if not mp.is_file():
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[aggregate] failed to load manifest {mp}: {e}", file=sys.stderr)
        return None


def list_runs(sig_dir: pathlib.Path) -> list[str]:
    """列出某 signature 下所有已经跑过的 run_label。"""
    runs_dir = sig_dir / "runs"
    if not runs_dir.is_dir():
        return []
    return [p.name for p in sorted(runs_dir.iterdir()) if p.is_dir()]


def aggregate_one(sig_dir: pathlib.Path,
                  route_to_scenario: dict[int, list[str]],
                  run_label: str | None = None) -> dict:
    """聚合单个 signature 下的 route 结果。

    输入目录：
      sig_dir/eval_per_route/eval_<route_id>.json    （跨 run 共享）
      sig_dir/runs/<run_label>/run_manifest.json     （本批次清单，可选）

    输出：
      sig_dir/runs/<run_label>/scenarios/<Scenario>/summary.json
      sig_dir/runs/<run_label>/summary_all.json
      （兼容：也在 sig_dir/scenarios/ + summary_all.json 写一份跨批次总聚合）

    run_label=None 时：枚举 sig_dir/runs/* 每个 label 各聚合一次；并额外写一份
    跨批次总聚合到 sig_dir 根（含所有已评估 route）。
    """
    if run_label is not None:
        return _aggregate_for_label(sig_dir, route_to_scenario, run_label)

    # 没传 run_label：跨批次总聚合（落根目录）+ 每个 run_label 各跑一次
    overall = _aggregate_for_label(sig_dir, route_to_scenario, run_label=None,
                                    out_dir=sig_dir, label_for_summary="__all__")
    for label in list_runs(sig_dir):
        try:
            _aggregate_for_label(sig_dir, route_to_scenario, run_label=label)
        except Exception as e:
            print(f"[aggregate] run_label={label}: {e}", file=sys.stderr)
    return overall


def _aggregate_for_label(sig_dir: pathlib.Path,
                          route_to_scenario: dict[int, list[str]],
                          run_label: str | None,
                          out_dir: pathlib.Path | None = None,
                          label_for_summary: str | None = None) -> dict:
    """实际聚合执行：可选 run_label 限定 route 子集，可选 out_dir 自定义输出位置。"""
    # 决定输入 route 集合
    allowed_route_ids: set[int] | None = None
    manifest: dict | None = None
    if run_label is not None:
        run_dir = sig_dir / "runs" / run_label
        manifest = _load_run_manifest(run_dir)
        if manifest is None:
            print(f"[aggregate] run_label={run_label} missing manifest, "
                  f"will use all eval_per_route results", file=sys.stderr)
        else:
            allowed_route_ids = {int(r) for r in manifest.get("route_ids", [])}
        if out_dir is None:
            out_dir = run_dir
    if out_dir is None:
        out_dir = sig_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_routes: dict[str, list[int]] = defaultdict(list)
    route_results: dict[int, dict] = {}
    planned_route_ids: list[int] | None = sorted(allowed_route_ids) if allowed_route_ids is not None else None
    per_route_dir = sig_dir / "eval_per_route"
    if per_route_dir.is_dir():
        for jp in sorted(per_route_dir.glob("eval_*.json")):
            # eval_latest_<id>.json 是 launcher 中间文件，不进入最终统计。
            m = re.search(r"eval_(\d+)\.json$", jp.name)
            if not m:
                continue
            rid = int(m.group(1))
            if allowed_route_ids is not None and rid not in allowed_route_ids:
                continue
            data = _load_json(jp)
            if data is None:
                continue
            route_results[rid] = data
            for scen in route_to_scenario.get(rid, ["__unknown__"]):
                scen_routes[scen].append(rid)
    if planned_route_ids is None:
        planned_route_ids = sorted(route_results)

    scen_planned_routes: dict[str, list[int]] = defaultdict(list)
    for rid in planned_route_ids:
        for scen in route_to_scenario.get(rid, ["__unknown__"]):
            scen_planned_routes[scen].append(rid)

    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    route_rows = _build_route_rows(planned_route_ids, route_results, route_to_scenario)
    evaluated_rows = [row for row in route_rows if row["evaluated"]]
    success_count = sum(1 for row in route_rows if row["status"] in _SUCCESS_STATUSES)
    perfect_count = sum(1 for row in route_rows if row["status"] in _PERFECT_STATUSES)
    missing_route_ids = [int(row["route_id"]) for row in route_rows if not row["evaluated"]]
    overall_bucket = _collect_metric_bucket(route_results)
    overall_infractions = _sum_infraction_counts(route_rows)

    summary = {
        "signature": sig_dir.name,
        "run_label": run_label or label_for_summary or "__all__",
        "planned_routes": len(planned_route_ids),
        "evaluated_routes": len(evaluated_rows),
        "total_routes": len(route_results),  # 兼容旧 webapp 字段：等价于 evaluated_routes
        "unique_scenarios": len(scen_planned_routes or scen_routes),
        "manifest_route_count": len(allowed_route_ids) if allowed_route_ids is not None else None,
        "missing_route_ids": missing_route_ids,
        "coverage": _safe_rate(len(evaluated_rows), len(planned_route_ids)),
        "success_count": success_count,
        "perfect_count": perfect_count,
        "success_rate": _safe_rate(success_count, len(planned_route_ids)),
        "success_rate_evaluated": _safe_rate(success_count, len(evaluated_rows)),
        "perfect_rate": _safe_rate(perfect_count, len(planned_route_ids)),
        "metrics": {k: _stat(v) for k, v in overall_bucket.items()},
        "infractions": overall_infractions,
        "metric_explanations": _METRIC_EXPLANATIONS,
        "routes": route_rows,
        "scenarios": {},
    }
    if manifest is not None:
        summary["manifest"] = {k: manifest[k] for k in ("started_at", "scenarios_filter",
            "route_ids_filter", "random_n", "random_seed", "single_test",
            "finished_at", "attempted_count", "failed_route_count", "failed_routes",
            "worker_fail")
            if k in manifest}

    all_scenarios = set(scen_planned_routes) | set(scen_routes)
    for scen in sorted(all_scenarios):
        planned_ids = sorted(set(scen_planned_routes.get(scen, [])))
        ids = sorted(set(scen_routes.get(scen, [])))
        missing_ids = [rid for rid in planned_ids if rid not in route_results]
        scen_rows = [
            row for row in route_rows
            if scen in row.get("scenarios", []) or (not row.get("scenarios") and scen == "__unknown__")
        ]
        scen_success = sum(1 for row in scen_rows if row["status"] in _SUCCESS_STATUSES)
        scen_perfect = sum(1 for row in scen_rows if row["status"] in _PERFECT_STATUSES)
        bucket: dict[str, list[float]] = defaultdict(list)
        for rid in ids:
            data = route_results.get(rid, {})
            for key, value in _extract_route_scores(data, rid).items():
                bucket[key].append(value)
        scen_info = {
            "scenario": scen,
            "n_routes": len(ids),
            "planned_routes": len(planned_ids),
            "evaluated_routes": len(ids),
            "missing_route_ids": missing_ids,
            "success_count": scen_success,
            "perfect_count": scen_perfect,
            "coverage": _safe_rate(len(ids), len(planned_ids)),
            "success_rate": _safe_rate(scen_success, len(planned_ids)),
            "success_rate_evaluated": _safe_rate(scen_success, len(ids)),
            "perfect_rate": _safe_rate(scen_perfect, len(planned_ids)),
            "route_ids": sorted(ids),
            "metrics": {k: _stat(v) for k, v in bucket.items()},
            "infractions": _sum_infraction_counts(scen_rows),
        }
        summary["scenarios"][scen] = scen_info
        (scenarios_dir / scen).mkdir(parents=True, exist_ok=True)
        with open(scenarios_dir / scen / "summary.json", "w", encoding="utf-8") as f:
            json.dump(scen_info, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_report_files(out_dir, summary)
    return summary


def _collect_metric_bucket(route_results: dict[int, dict]) -> dict[str, list[float]]:
    """汇总所有已评估 route 的 score bucket。"""
    bucket: dict[str, list[float]] = defaultdict(list)
    for rid, data in route_results.items():
        for key, value in _extract_route_scores(data, rid).items():
            bucket[key].append(value)
    return bucket


def _sum_infraction_counts(rows: list[dict]) -> dict[str, int]:
    """把 route rows 中的违规类型计数相加。"""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.get("infraction_counts", {}).items():
            counts[str(key)] += int(value)
    return dict(sorted(counts.items()))


def _build_route_rows(planned_route_ids: list[int],
                      route_results: dict[int, dict],
                      route_to_scenario: dict[int, list[str]]) -> list[dict]:
    """生成 route 级报告行，缺失 eval JSON 的路线也保留为 MISSING。"""
    rows: list[dict] = []
    for rid in planned_route_ids:
        data = route_results.get(rid)
        scenarios = route_to_scenario.get(rid, ["__unknown__"])
        if data is None:
            rows.append({
                "route_id": rid,
                "scenarios": scenarios,
                "evaluated": False,
                "status": "MISSING_EVAL_JSON",
                "scores": {},
                "num_infractions": 0,
                "infraction_counts": {},
            })
            continue
        scores = _extract_route_scores(data, rid)
        infractions = _extract_route_infractions(data, rid)
        rows.append({
            "route_id": rid,
            "scenarios": scenarios,
            "evaluated": True,
            "status": _extract_route_status(data, rid),
            "scores": scores,
            "num_infractions": _extract_num_infractions(data, rid),
            "infraction_counts": infractions,
        })
    return rows


def _write_report_files(out_dir: pathlib.Path, summary: dict) -> None:
    """写人类可读 Markdown 与论文表格友好的 CSV。"""
    _write_summary_markdown(out_dir / "summary_report.md", summary)
    _write_scenario_csv(out_dir / "scenario_table.csv", summary)
    _write_route_csv(out_dir / "route_results.csv", summary)


def _write_scenario_csv(path: pathlib.Path, summary: dict) -> None:
    """写 scenario 级 CSV，方便直接导入表格软件或论文脚本。"""
    fieldnames = [
        "scenario", "planned_routes", "evaluated_routes", "coverage",
        "success_count", "success_rate", "perfect_count", "perfect_rate",
        "score_composed_mean", "score_route_mean", "score_penalty_mean",
        "num_infractions", "missing_route_ids",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scen, info in sorted(summary.get("scenarios", {}).items()):
            metrics = info.get("metrics", {})
            writer.writerow({
                "scenario": scen,
                "planned_routes": info.get("planned_routes", 0),
                "evaluated_routes": info.get("evaluated_routes", 0),
                "coverage": info.get("coverage", 0.0),
                "success_count": info.get("success_count", 0),
                "success_rate": info.get("success_rate", 0.0),
                "perfect_count": info.get("perfect_count", 0),
                "perfect_rate": info.get("perfect_rate", 0.0),
                "score_composed_mean": metrics.get("score_composed", {}).get("mean"),
                "score_route_mean": metrics.get("score_route", {}).get("mean"),
                "score_penalty_mean": metrics.get("score_penalty", {}).get("mean"),
                "num_infractions": sum(info.get("infractions", {}).values()),
                "missing_route_ids": " ".join(str(x) for x in info.get("missing_route_ids", [])),
            })


def _write_route_csv(path: pathlib.Path, summary: dict) -> None:
    """写 route 级 CSV，方便排查某条路线为什么拖低总分。"""
    fieldnames = [
        "route_id", "scenarios", "evaluated", "status",
        "score_composed", "score_route", "score_penalty",
        "num_infractions", "infraction_counts",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("routes", []):
            scores = row.get("scores", {})
            writer.writerow({
                "route_id": row.get("route_id"),
                "scenarios": ";".join(row.get("scenarios", [])),
                "evaluated": int(bool(row.get("evaluated"))),
                "status": row.get("status", ""),
                "score_composed": scores.get("score_composed"),
                "score_route": scores.get("score_route"),
                "score_penalty": scores.get("score_penalty"),
                "num_infractions": row.get("num_infractions", 0),
                "infraction_counts": json.dumps(row.get("infraction_counts", {}), ensure_ascii=False, sort_keys=True),
            })


def _write_summary_markdown(path: pathlib.Path, summary: dict) -> None:
    """写一份可直接阅读/复制到实验记录里的 Markdown 报告。"""
    planned = int(summary.get("planned_routes", 0) or 0)
    evaluated = int(summary.get("evaluated_routes", 0) or 0)
    success = int(summary.get("success_count", 0) or 0)
    perfect = int(summary.get("perfect_count", 0) or 0)
    metrics = summary.get("metrics", {})
    infractions = summary.get("infractions", {})
    run_label = str(summary.get("run_label", ""))
    is_all = run_label == "__all__"

    lines = [
        "# LeadMoT Closed-Loop Evaluation Summary",
        "",
        "## Run",
        "",
        f"- signature: `{summary.get('signature', '')}`",
        f"- run_label: `{run_label}`",
    ]
    if is_all:
        # __all__ 是“该 signature 下所有批次的合并视图”，没有单批 manifest；
        # 这里显式说明，避免读者误以为 coverage=1.0 是“计划全跑且全部成功”。
        lines.append(
            "- 说明：`__all__` 表示跨批次总聚合（合并 sig 目录下所有已经落盘的 "
            "`eval_<route_id>.json`），没有单一 `run_manifest.json`；因此 "
            "`planned_routes == evaluated_routes`，`coverage` 恒为 1.0。"
            "若需查看“计划 N 条、实测多少条”的口径，请看 `runs/<RUN_LABEL>/summary_report.md`。"
        )
    lines.extend([
        f"- planned_routes: {planned}",
        f"- evaluated_routes: {evaluated}",
        f"- missing_routes: {len(summary.get('missing_route_ids', []))}",
        f"- coverage: {_fmt_float(summary.get('coverage'), 3)}",
        f"- success_count: {success}",
        f"- success_rate: {_fmt_float(summary.get('success_rate'), 3)}",
        f"- perfect_count: {perfect}",
        f"- perfect_rate: {_fmt_float(summary.get('perfect_rate'), 3)}",
        "",
    ])
    manifest = summary.get("manifest")
    if isinstance(manifest, dict) and manifest:
        lines.extend([
            "## Test Set",
            "",
            f"- started_at: `{manifest.get('started_at', '-')}`",
            f"- scenarios_filter: `{manifest.get('scenarios_filter', [])}`",
            f"- route_ids_filter: `{manifest.get('route_ids_filter', [])}`",
            f"- random_n / random_seed: `{manifest.get('random_n', 0)}` / `{manifest.get('random_seed', 0)}`",
            f"- single_test: `{manifest.get('single_test', 0)}`",
            f"- finished_at: `{manifest.get('finished_at', '-')}`",
            f"- attempted_count: `{manifest.get('attempted_count', '-')}`",
            f"- failed_route_count: `{manifest.get('failed_route_count', '-')}`",
            f"- failed_routes: `{manifest.get('failed_routes', [])}`",
            f"- worker_fail: `{manifest.get('worker_fail', '-')}`",
            "",
        ])

    lines.extend([
        "## Overall Scores",
        "",
        "| metric | mean | min | max | std | count |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key in _METRIC_KEYS:
        stat = metrics.get(key, {})
        lines.append(
            f"| {key} | {_fmt_float(stat.get('mean'))} | {_fmt_float(stat.get('min'))} | "
            f"{_fmt_float(stat.get('max'))} | {_fmt_float(stat.get('std'))} | {stat.get('count', 0)} |"
        )

    lines.extend([
        "",
        "## Scenario Table",
        "",
        "| scenario | planned | evaluated | coverage | success | perfect | success_rate | score_composed | score_route | score_penalty | infractions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scen, info in sorted(summary.get("scenarios", {}).items()):
        m = info.get("metrics", {})
        lines.append(
            f"| {scen} | {info.get('planned_routes', 0)} | {info.get('evaluated_routes', 0)} | "
            f"{_fmt_float(info.get('coverage'))} | {info.get('success_count', 0)} | "
            f"{info.get('perfect_count', 0)} | {_fmt_float(info.get('success_rate'))} | "
            f"{_fmt_float(m.get('score_composed', {}).get('mean'))} | "
            f"{_fmt_float(m.get('score_route', {}).get('mean'))} | "
            f"{_fmt_float(m.get('score_penalty', {}).get('mean'))} | "
            f"{sum(info.get('infractions', {}).values())} |"
        )

    missing = summary.get("missing_route_ids", [])
    if missing:
        lines.extend([
            "",
            "## Missing Routes",
            "",
            "这些 route 在本批计划中，但没有可读取的 `eval_<route_id>.json`，已计入 coverage / success_rate 分母：",
            "",
            ", ".join(str(x) for x in missing),
            "",
        ])

    lines.extend([
        "",
        "## Infractions",
        "",
    ])
    if infractions:
        lines.extend([
            "| type | count |",
            "|---|---:|",
        ])
        for key, value in sorted(infractions.items()):
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("No infractions recorded in evaluated routes.")

    lines.extend([
        "",
        "## Metric Glossary",
        "",
    ])
    for key, text in _METRIC_EXPLANATIONS.items():
        lines.append(f"- `{key}`: {text}")
    lines.extend([
        "",
        "## Files",
        "",
        "- `summary_all.json`: 完整机器可读聚合结果，含 route/scenario 细节。",
        "- `summary_report.md`: 当前这份人类可读报告。",
        "- `scenario_table.csv`: scenario 级表格，适合论文表格或电子表格继续加工。",
        "- `route_results.csv`: route 级明细，适合定位失败路线、低分路线和违规类型。",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """命令行入口：给 run_eval.sh 或手动排查使用。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-base", type=str, required=True)
    ap.add_argument("--benchmark-root", type=str, default=None)
    ap.add_argument("--leadmot-ckpt", type=str, default=None)
    ap.add_argument("--run-label", type=str, default=None,
                    help="只聚合指定 run_label 批次；省略则聚合所有 runs + 跨批次总聚合")
    args = ap.parse_args()

    eval_base = pathlib.Path(args.eval_base).resolve()
    bench_root = pathlib.Path(args.benchmark_root).resolve() if args.benchmark_root else None
    leadmot_ckpt = _normalize_leadmot_ckpt(pathlib.Path(args.leadmot_ckpt)) if args.leadmot_ckpt else None

    route_to_scenario = build_route_to_scenario(bench_root)
    print(f"[aggregate] route->scenario entries: {len(route_to_scenario)}")

    sig_dirs = _find_sig_dirs(eval_base, leadmot_ckpt)
    if not sig_dirs:
        print(f"[aggregate] no ckpt_signature dirs under {eval_base}", file=sys.stderr)
        sys.exit(1)

    for sd in sig_dirs:
        print(f"[aggregate] processing {sd.name}, run_label={args.run_label or '<all>'}")
        s = aggregate_one(sd, route_to_scenario, run_label=args.run_label)
        print(f"  -> evaluated={s['evaluated_routes']}/{s['planned_routes']}, "
              f"success_rate={s['success_rate']:.3f}, scenarios={s['unique_scenarios']}")


if __name__ == "__main__":
    # 允许独立运行：tools/python -m AutoMoT.qwen3vl_local.eval_carla.aggregate ...
    main()
