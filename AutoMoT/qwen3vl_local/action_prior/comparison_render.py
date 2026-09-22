"""发布多模型轨迹/RGB 拼图、精简 JSON 和分桶汇总；不用生成式图片。"""
from collections import Counter, defaultdict
import hashlib
import html
import os
from pathlib import Path
import shutil
import textwrap

from qwen3vl_local.action_prior.comparison_cases import read_json, write_json, case_id, identity, event_groups

METRICS = ("loss", "route_fm_mse", "waypoint_fm_mse", "route_ade_m", "route_fde_m", "waypoint_ade_m", "waypoint_fde_m", "sampled_trajectory_score")


def linked_copy(source, destination):
    """同一次结果内部硬链接节省重复分类体积；跨设备时复制。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def paired_cases(out, manifest, split):
    """拒绝丢帧、重复帧和 GT 不同，不能静默取交集美化分数。"""
    expected = manifest["splits"][split]["cases"]
    result = {}
    for model in manifest["models"]:
        cases = {}
        root = Path(out) / "_models" / model["id"] / split
        for path in sorted((root / "cases").glob("rank*_case*.json")):
            row = read_json(path)
            cid = case_id(row["sample"])
            if cid in cases or cid not in expected or identity(row["sample"]) != identity(expected[cid]):
                raise ValueError(f"重复/意外帧: {path}")
            cases[cid] = row
        if set(cases) != set(expected):
            raise ValueError(f"模型 {model['id']} 的 {split} 预测未覆盖全部计划 case")
        result[model["id"]] = cases
    for cid in expected:
        first = result[manifest["models"][0]["id"]][cid]
        for model in manifest["models"][1:]:
            other = result[model["id"]][cid]
            if any(first[k] != other[k] for k in ("gt_route", "gt_waypoints")):
                raise ValueError(f"模型 GT 不一致: {cid}")
    return result


def lidar_background(route, frame):
    """读取同 anchor 的 LEAD ego 点云；坐标与 meta 的 CARLA ego 累计点一致。"""
    import numpy as np
    path = Path(route) / "lidar" / f"{frame:04d}.laz"
    if not path.is_file():
        return None, "missing anchor LAZ; metric axes only"
    import laspy
    cloud = laspy.read(path)
    points = np.column_stack((cloud.x, cloud.y, cloud.z))
    points = points[np.isfinite(points).all(axis=1)]
    # 显示下采样只影响背景图，不影响模型输入。
    return points[::max(1, len(points) // 40000)], str(path)


def render_panel(folder, row, predictions, models, inputs, cloud, filename):
    """三视角 RGB 与两个米制俯视图拼接，同一颜色始终对应同一模型。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    # 相同输入只展示一次；不同历史/图数模型全部保留并写清输入归属。
    unique = {}
    for model in models:
        for path in inputs[model["id"]]:
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            unique.setdefault(fingerprint, [path, []])[1].append(f"M{model['number']}:{path.stem}")
    nimages = len(unique)
    header_inches = .27 * (len(models) + 4)
    height = header_inches + 7 + nimages * 5.1
    fig = plt.figure(figsize=(16, height))
    grid = fig.add_gridspec(nimages + 1, 2, height_ratios=[5.1] * nimages + [7])
    for index, (path, owners) in enumerate(unique.values()):
        ax = fig.add_subplot(grid[index, :])
        with Image.open(path) as im:
            ax.imshow(im)
            width = im.width
        for position, label in zip((1/6, 1/2, 5/6), ("LEFT", "FRONT", "RIGHT")):
            ax.text(position * width, 12, label, ha="center", va="top", color="white", bbox=dict(facecolor="black", alpha=.6))
        ax.set_title("Actual RGB: " + ", ".join(owners), fontsize=9)
        ax.axis("off")
    for column, (predkey, gtkey, title) in enumerate((("pred_route", "gt_route", "Route"), ("pred_waypoints", "gt_waypoints", "Future waypoints (~2 s)"))):
        ax = fig.add_subplot(grid[-1, column])
        gt = np.asarray(next(iter(predictions.values()))[gtkey][0])
        trajectories = [gt] + [np.asarray(predictions[m["id"]][predkey][0]) for m in models]
        allpoints = np.concatenate(trajectories + [np.zeros((1, 2))])
        # LEAD 原始 meta 来自 inv(ego_matrix) @ world；CARLA +x 前、+y 右。
        xmin, xmax = min(-6., allpoints[:, 1].min()-3), max(6., allpoints[:, 1].max()+3)
        ymin, ymax = min(-4., allpoints[:, 0].min()-2), max(12., allpoints[:, 0].max()+4)
        if cloud is not None:
            keep = (cloud[:, 1] >= xmin) & (cloud[:, 1] <= xmax) & (cloud[:, 0] >= ymin) & (cloud[:, 0] <= ymax)
            ax.scatter(cloud[keep, 1], cloud[keep, 0], s=.4, c="0.7", alpha=.6, rasterized=True)
        ax.plot(gt[:, 1], gt[:, 0], "k--o", linewidth=2.4, markersize=3, label="GT", zorder=5)
        for model in models:
            data = predictions[model["id"]]
            pts = np.asarray(data[predkey][0])
            metric = "route_ade_m" if column == 0 else "waypoint_ade_m"
            token = data["sample"].get("action_token", {}).get("name", "DISABLED")
            action = data.get("high_level_action") or {}
            text_action = ",".join(action.get("actions", [])) or action.get("status", "OFF")
            ax.plot(pts[:, 1], pts[:, 0], "-o", color=model["color"], markersize=3,
                    label=f"M{model['number']} token={token}, text={text_action}\nADE={data['metrics'][metric]:.3f}m")
        ax.scatter([0], [0], marker="^", s=80, c="black", zorder=6)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="Right (+ego y), m", ylabel="Forward (+ego x), m", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    descriptions = [f"M{m['number']} [{m['color']}] {m['label']} | step={m['selection']['step']} EMA" for m in models]
    title = (f"{row['split']} | {row['scenario']}/{row['run_id']} | frame {row['anchor']}\n"
             f"Event={','.join(event_groups(row))} | high-level={row['action_token']['name']} ({row['action_token']['reason']})\n"
             + "\n".join(textwrap.fill(s, 145) for s in descriptions))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - header_inches / height))
    fig.savefig(Path(folder) / filename, dpi=125)
    plt.close(fig)


def publish(out, manifest):
    """发布每例和每桶结果；详细原始审计保留 _models，日常只读 case.json。"""
    from qwen3vl_local.action_prior.comparison_progress import PreflightProgress
    from qwen3vl_local.action_prior.comparison_errors import error_selection
    out = Path(out)
    progress = PreflightProgress(out, phase="render")
    with progress.stage("initialize CPU plotting libraries"):
        from matplotlib.colors import to_hex
        from matplotlib import colormaps
        from qwen3vl_local.action_prior.comparison_scene import load_scene, render_paper
    total = sum(len(plan["cases"]) for plan in manifest["splits"].values())
    completed = 0
    examined = 0
    error_config = manifest.get("error_filter", dict(enabled=False, threshold_m=1.0))
    filter_report = dict(**error_config, splits={})
    print(f"[render] GPU inference finished; CPU rendering {total} unique cases, "
          f"GT + {len(manifest['models'])} models per comparison; progress: {out / 'render.json'}", flush=True)
    models = manifest["models"]
    for index, model in enumerate(models):
        model["number"] = index + 1
        model["color"] = to_hex(colormaps["turbo"]((index + .5) / len(models)))
    summary, links = {}, []
    calibration_counts = Counter()
    for split, plan in manifest["splits"].items():
        with progress.stage(f"{split}: validate paired predictions / GT"):
            cases = paired_cases(out, manifest, split)
        decisions = {}
        with progress.stage(f"{split}: endpoint error selection"):
            for cid in plan["cases"]:
                decisions[cid] = error_selection({m["id"]: cases[m["id"]][cid] for m in models},
                    enabled=error_config["enabled"], threshold_m=error_config["threshold_m"])
            kept = {cid for cid, decision in decisions.items() if decision["kept"]}
            filter_report["splits"][split] = dict(sampled=len(decisions), retained=len(kept),
                skipped=len(decisions)-len(kept),
                retained_at_threshold_m={str(t): sum(max(d["max_distances_m"].values()) > t for d in decisions.values())
                                         for t in (.5, 1., 1.5, 2.)}, cases=decisions)
            write_json(out / "error_filter.json", filter_report)
        print(f"[render] {split}: error_only={error_config['enabled']} threshold>{error_config['threshold_m']}m; "
              f"retained={len(kept)}/{len(decisions)}", flush=True)
        for cid, label in plan["cases"].items():
            examined += 1
            if cid not in kept:
                continue
            with progress.stage(f"case {examined}/{total} {split}/{cid}: comparison / history / individual models"):
                label = dict(label, visualization_error=decisions[cid])
                folder = out / "_cases" / split / cid
                folder.mkdir(parents=True, exist_ok=False)
                predictions = {m["id"]: cases[m["id"]][cid] for m in models}
                first = predictions[models[0]["id"]]
                cloud, cloud_source = lidar_background(first["sample"]["route_dir"], int(label["anchor"]))
                scene = load_scene(first["sample"]["route_dir"], int(label["anchor"]), manifest.get("camera_config"))
                calibration_counts[scene["audit"]["calibration_resolution"]] += 1
                simple = dict(id=cid, split=split, scenario=label["scenario"], run_id=label["run_id"], frame=int(label["anchor"]),
                    physical_route=label["route_group"], events=event_groups(label), event_status=label["event_balance_status"],
                    high_level_action=label["action_token"]["name"], action_reason=label["action_token"]["reason"],
                    label_source="Phase3 offline oracle; not a model prediction", lidar_background=cloud_source,
                    visualization=scene["audit"], error_filter=decisions[cid], models=[])
                inputs = {}
                anchor_sha256 = None
                for model in models:
                    mid = model["id"]
                    data = predictions[mid]
                    source = out / "_models" / mid / split / "inputs" / cid
                    target = folder / "inputs" / mid
                    paths = sorted(source.glob("input_rgb_*.png"))
                    if not paths:
                        raise ValueError(f"没有实际输入图像: {source}")
                    current_hash = hashlib.sha256(paths[-1].read_bytes()).hexdigest()
                    if anchor_sha256 is not None and current_hash != anchor_sha256:
                        raise ValueError(f"模型当前RGB不同: {cid}/{mid}")
                    anchor_sha256 = current_hash
                    for path in paths:
                        linked_copy(path, target / path.name)
                    inputs[mid] = sorted(target.glob("*.png"))
                    actual_input = read_json(source / "input.json")
                    actual_input["current_rgb_sha256"] = current_hash
                    token = data["sample"].get("action_token")
                    if bool(token) != model["high_level_action_token"] or (token and token != label["action_token"]):
                        raise ValueError(f"实际动作输入与模型/分类合同不同: {mid}/{cid}")
                    simple["models"].append(dict(id=mid, label=model["label"], color=model["color"],
                        checkpoint=model["checkpoint"], step=model["selection"]["step"], ema=True,
                        action_token=token["name"] if token else "DISABLED", text_action=data.get("high_level_action"),
                        input=actual_input, metrics={key: data["metrics"][key] for key in METRICS}))
                write_json(folder / "case.json", simple)
                write_json(folder / "visualization_calibration.json", scene["config"])
                write_json(folder / "trajectories.json", dict(gt_route=first["gt_route"][0], gt_waypoints=first["gt_waypoints"][0],
                    predictions={mid: dict(route=data["pred_route"][0], waypoints=data["pred_waypoints"][0]) for mid, data in predictions.items()}))
                render_paper(folder, label, predictions, models, inputs, scene, cloud)
                print(f"[render] comparison ready ({examined}/{total}): {folder / 'comparison.png'}", flush=True)
                render_panel(folder, label, predictions, models, inputs, cloud, "input_history.png")
                for model in models:
                    render_paper(folder, label, {model["id"]: predictions[model["id"]]}, [model], inputs, scene, cloud, stem=model["id"])
            completed += 1
        with progress.stage(f"{split}: publish event/action folders and summaries"):
            for style, categories in plan["groups"].items():
                for category, ids in categories.items():
                    visible_ids = [cid for cid in ids if cid in kept]
                    folder = out / style / split / category
                    folder.mkdir(parents=True, exist_ok=True)
                    for cid in visible_ids:
                        src = out / "_cases" / split / cid
                        shutil.copytree(src, folder / cid, copy_function=lambda a, b: linked_copy(Path(a), Path(b)))
                        link = str((folder / cid / "comparison.png").relative_to(out))
                        links.append(f'<li><a href="{html.escape(link)}">{html.escape(style + "/" + split + "/" + category + "/" + cid)}</a></li>')
                    metrics = {m["id"]: {key: sum(cases[m["id"]][cid]["metrics"][key] for cid in ids) / len(ids) for key in METRICS} for m in models} if ids else {}
                    visible_metrics = {m["id"]: {key: sum(cases[m["id"]][cid]["metrics"][key] for cid in visible_ids) / len(visible_ids)
                                       for key in METRICS} for m in models} if visible_ids else {}
                    group = dict(coverage=plan["coverage"][style][category], means=metrics,
                        displayed_means=visible_metrics, visualization=dict(sampled=len(ids), retained=len(visible_ids),
                            skipped=len(ids)-len(visible_ids), error_only=error_config["enabled"],
                            threshold_m=error_config["threshold_m"]))
                    if ids:
                        base = models[0]["id"]
                        group["paired_vs_first"] = {m["id"]: dict(
                            mean_delta={key: metrics[m["id"]][key] - metrics[base][key] for key in METRICS},
                            lower_score_cases=sum(cases[m["id"]][cid]["metrics"]["sampled_trajectory_score"] < cases[base][cid]["metrics"]["sampled_trajectory_score"] for cid in ids)) for m in models[1:]}
                    write_json(folder / "summary.json", group)
                    summary[f"{style}/{split}/{category}"] = group
    with progress.stage("write final report / gallery"):
        manifest["visualization_summary"] = dict(unique_cases=sum(calibration_counts.values()),
                                                calibration_counts=dict(calibration_counts))
        write_json(out / "manifest.json", manifest)
        write_json(out / "summary.json", summary)
        lines = ["# 多模型配对轨迹对比", "", manifest["interpretation"], "", "选择仅使用完整 val；评估使用 EMA 和同帧同 seed 噪声。", "",
                 "| 模型 | 条件 | best step | 完整 val 选优分数 |", "|---|---|---:|---:|"]
        for model in models:
            lines.append(f"| M{model['number']} {model['label']} | token={model['high_level_action_token']} | {model['selection']['step']} | {model['selection']['score']:.6f} |")
        lines += ["", f"误差筛选 enabled={error_config['enabled']}，阈值严格 > {error_config['threshold_m']} m；"
                  f"保留 {completed}/{total} 个去重采样case。route/waypoint任一模型-GT或任意模型对的终点欧氏距离触发即保留。",
                  "下表指标仅针对保留案例，是可视化诊断，不代表整体性能。summary.json 的means/paired_vs_first仍为全部原采样案例，"
                  "displayed_means为保留案例；error_filter.json记录触发者和不同阈值的保留数。未命中时不补例，原始预测仍在_models中。"]
        lines += ["", "RGB标定来源（去重case计数）：" + "; ".join(f"{key}={value}" for key, value in sorted(calibration_counts.items())) + "。",
                  "nominal_fallback表示缺录制标定而使用默认值；每例图注及case.json记录来源。RGB采用地面平面近似，不做遮挡判断。"]
        lines += ["", "逐例看 `event/` 或 `action/`，`comparison.png/.pdf` 为RGB投影＋场景俯视同屏比较，`model_*.png/.pdf` 单模型对 GT；`input_history.png` 保留实际历史输入，`case.json` 为简表，坐标数组在 `trajectories.json`。", "",
                  "| 分组 | 保留 / 采样 / 请求 | 采样物理路线 | 模型 | FM loss | route ADE | waypoint ADE | waypoint FDE |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        for name, group in summary.items():
            coverage = group["coverage"]
            for model in models:
                metrics = group["displayed_means"].get(model["id"])
                values = " | ".join(f"{metrics[k]:.5f}" for k in ("loss", "route_ade_m", "waypoint_ade_m", "waypoint_fde_m")) if metrics else "N/A | N/A | N/A | N/A"
                lines.append(f"| {name} | {group['visualization']['retained']}/{coverage['selected']}/{coverage['requested']} | {coverage['selected_physical_routes']} | M{model['number']} | {values} |")
        (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Trajectory comparison</title><h1>Case gallery</h1>'
            + f'<p>Retained {completed}/{total} sampled cases. Error filter: {error_config["enabled"]}, threshold &gt; {error_config["threshold_m"]} m.</p>'
            + '<p>See REPORT.md and case.json for metrics and input labels.</p><ul>' + "\n".join(links) + '</ul>', encoding="utf-8")
    print(f"[render] complete {completed}/{total} cases (retained/sampled); gallery: {out / 'index.html'}", flush=True)
