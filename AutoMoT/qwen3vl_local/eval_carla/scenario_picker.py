"""220 routes 与 LEAD scenario 的反向映射 + 子集筛选。

LEAD 按 `<Scenario>/<route_id>.xml` 组织：
  lead/data/benchmark_routes/bench2drive220/<Scenario>/<route_id>.xml

AutoMoT 这边按 route_id 列表跑：
  AutoMoT/eval_json/b2d_all_routes_split{1,2}.json
"""

from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET
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


def find_route_xml(route_id: int, benchmark_root: pathlib.Path | None = None) -> pathlib.Path | None:
    """按 route_id 查 LEAD `<Scenario>/<route_id>.xml` 路线文件。"""
    root = pathlib.Path(benchmark_root) if benchmark_root else _BENCHMARK_ROOT
    if not root.is_dir():
        print(f"[scenario_picker] benchmark root not found: {root}")
        return None
    matches = sorted(root.glob(f"*/{int(route_id)}.xml"))
    return matches[0] if matches else None


def load_route_endpoint(route_id: int, benchmark_root: pathlib.Path | None = None) -> dict[str, object] | None:
    """读取 LEAD route XML 的最后一个 waypoint，作为在线 final destination。

    返回的 endpoint 是 CARLA world frame 坐标；agent 会在每帧转成 ego frame，
    与训练侧 `meta["next_target_points"][-1]` -> ego 的坐标系保持一致。
    """
    xml_path = find_route_xml(route_id, benchmark_root)
    if xml_path is None:
        return None
    tree = ET.parse(xml_path)
    route_elem = tree.find(".//route")
    waypoint_elems = tree.findall(".//waypoints/position")
    if route_elem is None or not waypoint_elems:
        return None
    last = waypoint_elems[-1]
    endpoint = [
        float(last.attrib["x"]),
        float(last.attrib["y"]),
        float(last.attrib.get("z", "0.0")),
    ]
    return {
        "route_id": int(route_id),
        "scenario": xml_path.parent.name,
        "xml_path": str(xml_path),
        "town": route_elem.attrib.get("town"),
        "endpoint": endpoint,
    }


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

    # 双层防御：把空字符串 / 纯空白过滤掉。launcher 应该已经避免传 `--scenario ""`
    # 这种空参数，但万一有调用方手误传了，picker 自己也不应该炸。
    scenarios_clean = [s.strip() for s in (args.scenario or []) if s and s.strip()]
    route_ids_raw = [s.strip() for s in (args.route_id or []) if s and s.strip()]
    route_ids_clean: list[int] = []
    for raw in route_ids_raw:
        try:
            route_ids_clean.append(int(raw))
        except ValueError:
            ap.error(f"--route-id must be an integer, got {raw!r}")
    route_ids = pick_routes(
        only_scenarios=scenarios_clean or None,
        only_route_ids=route_ids_clean or None,
        random_n=args.random,
        random_seed=args.seed,
        benchmark_root=pathlib.Path(args.benchmark_root) if args.benchmark_root else None,
    )
    for rid in route_ids:
        print(rid)


if __name__ == "__main__":
    main()
