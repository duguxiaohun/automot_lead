"""LeadMoT closed-loop eval 结果按 LEAD scenario 二次聚合。

逻辑：
- 扫 `<eval_base>/<signature>/eval_per_route/eval_<route_id>.json`（launcher 落盘的逐 route 结果）
- 根据 `lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml` 建反向映射
- 按 scenario 聚合 leaderboard 给的 score_composed / score_route / score_penalty
- 在每个 ckpt_signature 子目录下输出 scenarios/<Scenario>/summary.json 与 summary_all.json
"""

from __future__ import annotations

import argparse
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

    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "signature": sig_dir.name,
        "run_label": run_label or label_for_summary or "__all__",
        "total_routes": len(route_results),
        "unique_scenarios": len(scen_routes),
        "manifest_route_count": len(allowed_route_ids) if allowed_route_ids else None,
        "scenarios": {},
    }
    if manifest is not None:
        summary["manifest"] = {k: manifest[k] for k in ("started_at", "scenarios_filter",
            "route_ids_filter", "random_n", "random_seed", "single_test")
            if k in manifest}

    for scen, ids in sorted(scen_routes.items()):
        bucket: dict[str, list[float]] = defaultdict(list)
        for rid in ids:
            data = route_results.get(rid, {})
            for key, value in _extract_route_scores(data, rid).items():
                bucket[key].append(value)
        scen_info = {
            "scenario": scen,
            "n_routes": len(ids),
            "route_ids": sorted(ids),
            "metrics": {k: _stat(v) for k, v in bucket.items()},
        }
        summary["scenarios"][scen] = scen_info
        (scenarios_dir / scen).mkdir(parents=True, exist_ok=True)
        with open(scenarios_dir / scen / "summary.json", "w", encoding="utf-8") as f:
            json.dump(scen_info, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


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
    leadmot_ckpt = pathlib.Path(args.leadmot_ckpt).resolve() if args.leadmot_ckpt else None

    route_to_scenario = build_route_to_scenario(bench_root)
    print(f"[aggregate] route->scenario entries: {len(route_to_scenario)}")

    sig_dirs = _find_sig_dirs(eval_base, leadmot_ckpt)
    if not sig_dirs:
        print(f"[aggregate] no ckpt_signature dirs under {eval_base}", file=sys.stderr)
        sys.exit(1)

    for sd in sig_dirs:
        print(f"[aggregate] processing {sd.name}, run_label={args.run_label or '<all>'}")
        s = aggregate_one(sd, route_to_scenario, run_label=args.run_label)
        print(f"  -> total_routes={s['total_routes']}, scenarios={s['unique_scenarios']}")


if __name__ == "__main__":
    # 允许独立运行：tools/python -m AutoMoT.qwen3vl_local.eval_carla.aggregate ...
    main()
