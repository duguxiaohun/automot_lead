"""220 routes 与 LEAD scenario 的反向映射 + 子集筛选。

LEAD 按 `<Scenario>/<route_id>.xml` 组织：
  lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml

AutoMoT 这边按 route_id 列表跑：
  AutoMoT/eval_json/b2d_all_routes_split{1,2}.json
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]   # automot_lead/
_AUTOMOT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # AutoMoT/
_EVAL_JSON_DIR = _AUTOMOT_ROOT / "eval_json"
_BENCHMARK_ROOT = _REPO_ROOT / "lead" / "data" / "benchmark_routes" / "bench2drive220"


def load_all_route_ids() -> list[int]:
    """合并 b2d_all_routes_split{1,2}.json 并补两条历史漏 route。"""
    ids: list[int] = []
    for name in ("b2d_all_routes_split1.json", "b2d_all_routes_split2.json"):
        path = _EVAL_JSON_DIR / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # AutoMoT 的 route 列表在不同版本里有两种 schema：
        #   {"ids": [...]} 或 {"routes": [{"id": ...}, ...]}
        # 这里同时兼容，避免只因为 JSON schema 差异跑不起来。
        ids.extend(data.get("ids") or [int(r["id"]) for r in data.get("routes", [])])
    for mid in (1711, 1773):
        if mid not in ids:
            ids.append(mid)
    return sorted(set(ids))


def build_route_to_scenario(benchmark_root: pathlib.Path | None = None) -> dict[int, list[str]]:
    """扫 lead benchmark_routes，建立 route_id -> [scenario, ...] 反向映射。"""
    root = pathlib.Path(benchmark_root) if benchmark_root else _BENCHMARK_ROOT
    mapping: dict[int, list[str]] = defaultdict(list)
    if not root.is_dir():
        print(f"[scenario_picker] benchmark root not found: {root}")
        return mapping
    for scen_dir in sorted(root.iterdir()):
        if not scen_dir.is_dir():
            continue
        scen = scen_dir.name
        for xml_path in scen_dir.glob("*.xml"):
            try:
                # LEAD 的目录结构是 <Scenario>/<route_id>.xml，文件名就是 route_id。
                rid = int(xml_path.stem)
            except ValueError:
                continue
            mapping[rid].append(scen)
    return mapping


def pick_routes(
    only_scenarios: list[str] | None = None,
    only_route_ids: list[int] | None = None,
    random_n: int | None = None,
    random_seed: int = 0,
    benchmark_root: pathlib.Path | None = None,
) -> list[int]:
    """根据 --scenario / --route-id / --random 过滤后的 route_id 列表。

    三种模式（可叠加）：
    - only_scenarios:   只保留指定 LEAD scenario 名下的 route_id
    - only_route_ids:   只保留指定 route_id
    - random_n:         从筛后剩余里随机抽 N 条；不传则全量
    """
    ids = load_all_route_ids()
    if only_route_ids:
        # route_id 是最强过滤器：用户点名的 route 不应被全量列表外的历史残留误带入。
        keep = set(int(x) for x in only_route_ids)
        ids = [i for i in ids if i in keep]
    if only_scenarios:
        # scenario 过滤依赖 LEAD benchmark_routes，而不是 AutoMoT 的 bench2drive220.xml；
        # 这样能保留 LEAD 原始场景分类。
        mapping = build_route_to_scenario(benchmark_root)
        allowed: set[int] = set()
        targets = {s.lower() for s in only_scenarios}
        for rid, scen_list in mapping.items():
            if any(s.lower() in targets for s in scen_list):
                allowed.add(rid)
        ids = [i for i in ids if i in allowed]
    if random_n is not None and random_n > 0 and len(ids) > random_n:
        import random as _r
        # 固定 seed 的抽样用于快速 smoke / ablation，方便不同 ckpt 公平复现同一批路线。
        rng = _r.Random(random_seed)
        ids = sorted(rng.sample(ids, random_n))
    return ids


def main():
    """CLI：枚举筛选后的 route_id（被 run_eval.sh 调用）。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="append", default=None,
                    help="重复使用：只保留指定 LEAD scenario 名下的 route_id")
    ap.add_argument("--route-id", action="append", default=None,
                    help="重复使用：只保留指定 route_id")
    ap.add_argument("--random", type=int, default=None,
                    help="从筛后剩余里随机抽 N 条；与 --scenario / --route-id 叠加")
    ap.add_argument("--seed", type=int, default=0,
                    help="--random 抽样种子，确保可复现")
    ap.add_argument("--benchmark-root", type=str, default=None)
    ap.add_argument("--list-scenarios", action="store_true",
                    help="只打印所有 scenario 名（带每类 route 数）")
    args = ap.parse_args()

    if args.list_scenarios:
        mapping = build_route_to_scenario(
            pathlib.Path(args.benchmark_root) if args.benchmark_root else None
        )
        cnt: dict[str, int] = defaultdict(int)
        for scen_list in mapping.values():
            for s in scen_list:
                cnt[s] += 1
        for s in sorted(cnt):
            print(f"{s}\t{cnt[s]}")
        return

    route_ids = pick_routes(
        only_scenarios=args.scenario,
        only_route_ids=[int(x) for x in args.route_id] if args.route_id else None,
        random_n=args.random,
        random_seed=args.seed,
        benchmark_root=pathlib.Path(args.benchmark_root) if args.benchmark_root else None,
    )
    for rid in route_ids:
        print(rid)


if __name__ == "__main__":
    main()
