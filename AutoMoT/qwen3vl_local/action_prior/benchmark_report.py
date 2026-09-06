"""Bench2Drive 路线/场景/能力汇总；缺失结果不伪装成完整论文成绩。"""

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    """记录实际评测源码和路线版本。"""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def routes(path):
    """按正式 XML 的 id/scenario/town 生成唯一评测计划。"""
    result = {}
    for r in ET.parse(path).getroot().findall("route"):
        key = r.get("id")
        if key in result:
            raise ValueError("duplicate benchmark route id: " + key)
        result[key] = dict(
            scenario=r.find("scenarios/scenario").get("type"), town=r.get("town")
        )
    return result


def success(record):
    """官方 SR：Completed/Perfect 且除 min_speed 外无 infraction。"""
    return record["status"] in ("Completed", "Perfect") and not any(
        value
        for key, value in record["infractions"].items()
        if key != "min_speed_infractions"
    )


def route_record(path, rid):
    """拒绝重复、错路线、半截或非有限得分记录。"""
    data = json.loads(Path(path).read_text())
    records = data.get("_checkpoint", {}).get("records", [])
    found = [
        r
        for r in records
        if re.fullmatch(
            r"(?:RouteScenario_)?" + re.escape(str(rid)) + r"(?:_rep0)?",
            str(r.get("route_id", "")),
        )
    ]
    if len(found) != 1:
        raise ValueError("missing/duplicate route record")
    record = found[0]
    if not record.get("status") or not isinstance(record.get("infractions"), dict):
        raise ValueError("incomplete status/infractions")
    if record["status"] not in ("Completed", "Perfect") and not record[
        "status"
    ].startswith("Failed"):
        raise ValueError("nonterminal route status")
    for key in ("score_composed", "score_route", "score_penalty"):
        if not math.isfinite(float(record["scores"][key])):
            raise ValueError("nonfinite score")
    return record


def abilities():
    """只解析本地官方工具的分类常量，不 import 其自动启动 CARLA 的代码。"""
    path = ROOT / "tools/ability_benchmark.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "Ability" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("official Ability mapping not found")


def offline_junction_threshold(route, map_root):
    """按官方 0.0.4 在静态 OpenDRIVE 上计算 junction 通过比例，不启动仿真。"""
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    town = route.get("town")
    root = Path(map_root)
    candidates = [
        root / "OpenDrive" / f"{town}.xodr",
        root / town / "OpenDrive" / f"{town}.xodr",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"OpenDRIVE for {town} missing under {root}")
    planner = GlobalRoutePlanner(carla.Map(town, path.read_text()), 1.0)
    positions = [
        carla.Location(*[float(p.get(k, "0")) for k in ("x", "y", "z")])
        for p in route.findall("waypoints/position")
    ]
    waypoints = [
        wp
        for start, end in zip(positions, positions[1:])
        for wp, _ in planner.trace_route(start, end)
    ]
    for count, wp in enumerate(waypoints, 1):
        if wp.is_junction:
            return (count + 8) / len(waypoints)
    raise ValueError("No junction found in official traffic-sign route")


def mean(values):
    """没有测量值时明确为空，不补假零。"""
    return sum(values) / len(values) if values else None


def summarize(rows):
    """缺记录按 planned 分母列零贡献，并另外显示 observed 指标与覆盖率。"""
    observed = [r for r in rows if r["available"]]
    n = len(rows)
    return dict(
        planned=n,
        observed=len(observed),
        driving_score=sum(r["driving_score"] for r in observed) / n if n else None,
        success_rate=100 * sum(r["success"] for r in observed) / n if n else None,
        route_completion=(
            sum(r["route_completion"] for r in observed) / n if n else None
        ),
        infraction_score=mean([r["infraction_score"] for r in observed]),
        driving_efficiency=mean(
            [r["efficiency"] for r in observed if r["efficiency"] is not None]
        ),
        efficiency_routes=sum(r["efficiency"] is not None for r in observed),
        comfort=mean([r["comfort"] for r in observed if r["comfort"] is not None]),
        comfort_routes=sum(r["comfort"] is not None for r in observed),
    )


def report(root, routes_xml, map_root=None):
    """汇总原始 leaderboard + 独立 motion telemetry，输出论文表和全分母说明。"""
    root = Path(root)
    plan = json.loads((root / "run_manifest.json").read_text())
    if plan.get("routes_sha256") and plan["routes_sha256"] != sha(routes_xml):
        raise ValueError("report routes XML differs from run manifest")
    catalog = routes(routes_xml)
    selected = {str(k): catalog[str(k)] for k in plan["route_ids"]}
    mapping = abilities()
    module_path = ROOT / "tools/efficiency_smoothness_benchmark.py"
    spec = importlib.util.spec_from_file_location("b2d_motion_metrics", module_path)
    motion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motion)
    xml_rows = {r.get("id"): r for r in ET.parse(routes_xml).getroot().findall("route")}
    map_root = map_root or str(
        Path(os.environ.get("CARLA_ROOT", ".")) / "CarlaUE4/Content/Carla/Maps"
    )
    rows, errors = [], []
    for rid, info in selected.items():
        row = dict(route_id=rid, **info, available=False)
        try:
            record = route_record(root / "eval_per_route" / f"eval_{rid}.json", rid)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        row.update(
            available=True,
            status=record["status"],
            success=success(record),
            driving_score=float(record["scores"]["score_composed"]),
            route_completion=float(record["scores"]["score_route"]),
            infraction_score=float(record["scores"]["score_penalty"]),
            infractions={k: len(v) for k, v in record["infractions"].items()},
            efficiency=None,
            comfort=None,
            traffic_sign_success=None,
        )
        values = []
        for text in record["infractions"].get("min_speed_infractions", []):
            match = re.search(r"\b\d+\.?\d*%", text)
            if match and float(match[0][:-1]) <= 1000:
                values.append(float(match[0][:-1]))
        row["efficiency"] = mean(values)
        # save_name 来自 evaluator；agent 在 signature 子目录下保存。
        save_name = str(record.get("save_name", ""))
        paths = (
            list((root / "rollouts").glob(f"*/{save_name}/metric_info.json"))
            if save_name and Path(save_name).name == save_name
            else []
        )
        try:
            if len(paths) != 1:
                raise ValueError("missing/ambiguous motion telemetry")
            telemetry = json.loads(paths[0].read_text())
            ordered = [telemetry[k] for k in sorted(telemetry, key=int)]
            columns = {
                k: [r[k] for r in ordered]
                for k in (
                    "acceleration",
                    "angular_velocity",
                    "forward_vector",
                    "right_vector",
                    "location",
                    "rotation",
                )
            }
            row["comfort"] = 100 * float(motion.seg_compute_comfort_metric(**columns))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            errors.append(dict(route_id=rid, metric="comfort", reason=str(exc)))
        if info["scenario"] in mapping["Traffic_Signs"]:
            if row["success"]:
                row["traffic_sign_success"] = True
            else:
                try:
                    threshold = offline_junction_threshold(xml_rows[rid], map_root)
                    row["junction_completion_threshold"] = threshold
                    row["traffic_sign_success"] = (
                        row["route_completion"] / 100 > threshold
                        and not record["infractions"].get("stop_infraction")
                        and not record["infractions"].get("red_light")
                    )
                    row["traffic_sign_success"] = bool(row["traffic_sign_success"])
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    errors.append(
                        dict(route_id=rid, metric="traffic_signs", reason=str(exc))
                    )
        # 模型条件统计与同步推理耗时只作附加审计，不混入官方驾驶分。
        if len(paths) == 1:
            for name in ("prior_counts", "latency"):
                extra = paths[0].parent / (name + ".json")
                if extra.is_file():
                    row[name] = json.loads(extra.read_text())
        rows.append(row)
    ability = {}
    for name, scenarios in mapping.items():
        members = [r for r in rows if r["scenario"] in scenarios]
        key = "traffic_sign_success" if name == "Traffic_Signs" else "success"
        known = [r for r in members if r["available"] and r.get(key) is not None]
        ability[name] = dict(
            planned=len(members),
            observed=len(known),
            success_rate=(
                100 * sum(bool(r[key]) for r in known) / len(members)
                if members and len(known) == len(members)
                else None
            ),
        )
    overall = summarize(rows)
    overall["ability_mean"] = (
        mean([v["success_rate"] for v in ability.values()])
        if all(v["success_rate"] is not None for v in ability.values())
        else None
    )
    scenarios = {
        s: summarize([r for r in rows if r["scenario"] == s])
        for s in sorted({r["scenario"] for r in rows})
    }
    complete = (
        len(selected) == 220
        and len(catalog) == 220
        and set(selected) == set(catalog)
        and all(r["available"] for r in rows)
    )
    value = dict(
        protocol="Bench2Drive 0.0.4 ability rules; local evaluator/scorer SHA recorded",
        full_220_records=complete,
        all_metrics_available=complete
        and not errors
        and overall["driving_efficiency"] is not None,
        execution_failures=[
            r["route_id"]
            for r in rows
            if r.get("available")
            and any(
                marker in r.get("status", "").lower()
                for marker in ("crash", "error", "rejected", "setup")
            )
        ],
        overall=overall,
        abilities=ability,
        scenarios=scenarios,
        routes=rows,
        metric_errors=errors,
        missing_routes=[r["route_id"] for r in rows if not r["available"]],
        invalid_records_policy="missing records contribute zero DS/SR to planned denominator; incomplete is provisional",
        efficiency_policy="official min_speed percentages <=1000; mean over eligible routes; missing is null",
        comfort_policy="local official function unchanged; raw CARLA angular velocity; telemetry 10Hz, dt=0.1s; not corrected physical metric",
        training_protocol="LEAD training + Phase1/2; not automatically identical to paper training split/sensors",
        sources={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(routes_xml).resolve(),
                module_path,
                ROOT / "tools/ability_benchmark.py",
                ROOT / "leaderboard/leaderboard/utils/statistics_manager.py",
            )
            if p.is_relative_to(ROOT)
        },
    )
    (root / "benchmark_report.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False)
    )
    for name, data in [
        ("route_results", rows),
        ("scenario_results", [dict(scenario=k, **v) for k, v in scenarios.items()]),
        ("ability_results", [dict(ability=k, **v) for k, v in ability.items()]),
    ]:
        keys = list(dict.fromkeys(k for row in data for k in row))
        with (root / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(data)
    table = [
        "# Bench2Drive report",
        "",
        f'Coverage: {overall["observed"]}/{overall["planned"]}. Complete 220: {complete}.',
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    table += [f'| {k} | {v if v is not None else "N/A"} |' for k, v in overall.items()]
    table += [
        f'| {k} SR (%) | {v["success_rate"] if v["success_rate"] is not None else "N/A"} |'
        for k, v in ability.items()
    ]
    table += [
        "",
        "Missing telemetry/maps or partial route coverage must be resolved before presenting a complete benchmark table. Training data, sensors, PID/safety and runtime version must be disclosed.",
    ]
    (root / "paper_table.md").write_text("\n".join(table) + "\n")
    return value


def main():
    """只汇总已有结果；静态地图读取不启动 CARLA。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument(
        "--routes", default=str(ROOT / "leaderboard/data/bench2drive220.xml")
    )
    p.add_argument("--map-root")
    a = p.parse_args()
    report(a.root, a.routes, a.map_root)


if __name__ == "__main__":
    main()
