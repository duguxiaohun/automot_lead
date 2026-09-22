"""论文布局的真实场景叠轨迹；只读录制图像/道路图/框，不依赖CARLA或模型。"""
import copy
import hashlib
import json
import lzma
from pathlib import Path
import pickle

import numpy as np

# 来自本仓只读 lead/lead/common/config_base.py 的 CARLA_LEADERBOARD2_3CAMERAS。
# 明确标为名义采集标定；不同传感器配置须提供自己的JSON，不能从图像猜外参。
DEFAULT_CAMERA_CONFIGURATION = {
    "schema": "action_comparison_cameras_v1",
    "source": "LEAD CARLA_LEADERBOARD2_3CAMERAS nominal collection calibration",
    "ground_z_m": 0.0,
    "views": [
        {"name": "Left", "pos": [.1, -.35, 2.25], "rot": [0., 0., -54.5], "width": 384, "height": 384, "fov": 60.},
        {"name": "Front", "pos": [.35, 0., 2.25], "rot": [0., 0., 0.], "width": 384, "height": 384, "fov": 60.},
        {"name": "Right", "pos": [.1, .35, 2.25], "rot": [0., 0., 54.5], "width": 384, "height": 384, "fov": 60.},
    ],
    # expert_data.py: 256px, bottom=32*4=128px, collection ppm=2。
    # chauffeurnet.py 最后把地图顺时针旋转90度存盘，读图先反向旋回。
    "hdmap": {"size": 256, "pixels_per_meter": 2., "pixels_ev_to_bottom": 128., "stored_rotation_k": -1},
    "third_person": None,
}


def camera_configuration(path=None, *, configuration=None):
    """读取并验证显示标定，记录来源；不加入模型训练合同。"""
    if path is not None and configuration is not None:
        raise ValueError("不能同时传相机配置文件和配置对象")
    result = json.loads(Path(path).read_text()) if path is not None else copy.deepcopy(
        DEFAULT_CAMERA_CONFIGURATION if configuration is None else configuration)
    if result.get("schema") != "action_comparison_cameras_v1" or not result.get("source"):
        raise ValueError("相机配置需要schema=action_comparison_cameras_v1及source来源说明")
    if len(result.get("views", [])) != 3 or not np.isfinite(result.get("ground_z_m", 0.)):
        raise ValueError("当前支持三视角拼接图及有限ground_z_m")
    cameras = result["views"] + ([result["third_person"]] if result.get("third_person") else [])
    for camera in cameras:
        if (not isinstance(camera.get("name"), str) or not camera["name"].strip()
                or len(camera.get("pos", [])) != 3 or len(camera.get("rot", [])) != 3
                or not np.isfinite(camera["pos"] + camera["rot"]).all()
                or not 0 < float(camera.get("fov", 0)) < 180
                or any(type(camera.get(key)) is not int or camera[key] < 1 for key in ("width", "height"))):
            raise ValueError("相机配置需要name、pos/rot三元组、正整数width/height和0<fov<180")
    third = result.get("third_person")
    if third:
        pattern = third.get("image_pattern", "")
        if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("第三人称image_pattern必须为route内相对路径，例如3rd_person/{frame:04d}.jpg")
        pattern.format(frame=0)
    hdmap = result.get("hdmap")
    if hdmap and (type(hdmap.get("size")) is not int or hdmap["size"] < 2
                  or not np.isfinite([hdmap.get("pixels_per_meter", 0), hdmap.get("pixels_ev_to_bottom", 0)]).all()
                  or hdmap["pixels_per_meter"] <= 0 or not 0 <= hdmap["pixels_ev_to_bottom"] <= hdmap["size"]
                  or type(hdmap.get("stored_rotation_k")) is not int):
        raise ValueError("hdmap标定尺寸/米制比例/ego偏移/旋转非法")
    result["configuration_file"] = str(Path(path).resolve()) if path else result.get("configuration_file", "builtin_nominal_lead_three_camera")
    if path:
        result["configuration_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def scene_configuration(route, frame, configuration=None):
    """显式配置优先，否则使用同帧录制标定；缺字段才回退名义标定。"""
    route = Path(route)
    local = route / "comparison_camera.json"
    if local.is_file():
        config = camera_configuration(local)
        config["calibration_resolution"] = "explicit_route_json"
        return config
    config = camera_configuration(configuration=configuration)
    if config["configuration_file"] != "builtin_nominal_lead_three_camera":
        config["calibration_resolution"] = "explicit_global_json"
        return config
    meta_path = route / "metas" / f"{frame:04d}.pkl"
    fallback = "anchor_meta_missing"
    if meta_path.is_file():
        with lzma.open(meta_path, "rb") as handle:
            meta = pickle.load(handle)
        if not isinstance(meta, dict):
            raise ValueError(f"非法录制meta: {meta_path}")
        info = meta.get("sensor_information")
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise ValueError(f"非法sensor_information: {meta_path}")
        if "num_cameras" in info and info["num_cameras"] != 3:
            raise ValueError(f"不支持的录制相机数量，不能回退三相机标定: {meta_path}")
        if "camera_calibration" in info:
            calibration = info["camera_calibration"]
            if info.get("num_cameras") != 3 or not isinstance(calibration, dict):
                raise ValueError(f"不支持的录制相机标定: {meta_path}")
            # base_agent.tick按rgb_1/rgb_2/rgb_3顺序直接拼接，不能用旧录像器的[3,2,1]。
            keys = [str(key) for key in calibration]
            if sorted(keys) != ["1", "2", "3"]:
                raise ValueError(f"录制相机标定必须恰好包含1/2/3: {meta_path}")
            views = []
            for index, name in enumerate(("Left", "Front", "Right"), 1):
                view = copy.deepcopy(calibration[index] if index in calibration else calibration[str(index)])
                if view.get("cropped_height", view.get("height")) != view.get("height"):
                    raise ValueError(f"裁剪相机需显式提供实际显示标定: {meta_path}")
                view.update(name=name, sensor_id=f"rgb_{index}")
                views.append(view)
            config.update(views=views, source="recorded sensor_information.camera_calibration (rgb_1, rgb_2, rgb_3)",
                          configuration_file=str(meta_path.resolve()),
                          configuration_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                          calibration_resolution="recorded_anchor_meta")
            return camera_configuration(configuration=config)
        fallback = "anchor_meta_has_no_camera_calibration"
    config.update(calibration_resolution="nominal_fallback", calibration_fallback_reason=fallback)
    return config


def carla_rotation(rotation):
    """CARLA roll/pitch/yaw相机局部到ego旋转；pitch正为抬头，y正向右。"""
    roll, pitch, yaw = np.deg2rad(rotation)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]])


def project_trajectory(points, camera, ground_z=0.):
    """投影地平面2D轨迹，保留NaN断点避免把相机后方点跨视野连线。"""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    xyz = np.column_stack((points, np.full(len(points), ground_z)))
    # 等价于Bench2Drive/tools/utils.py: get_matrix的逆矩阵→get_image_point；
    # 不直接使用旧common_utils.project_points_to_image的正向欧拉矩阵。
    local = (xyz - np.asarray(camera["pos"])) @ carla_rotation(camera["rot"])
    depth = local[:, 0]
    valid = np.isfinite(local).all(axis=1) & (depth > .1)
    pixels = np.full((len(points), 2), np.nan)
    focal = camera["width"] / (2. * np.tan(np.deg2rad(camera["fov"]) / 2.))
    # CARLA sensor.camera.rgb 的fov是水平视场角，像素为正方形，fx=fy。
    pixels[valid, 0] = camera["width"] / 2. + focal * local[valid, 1] / depth[valid]
    pixels[valid, 1] = camera["height"] / 2. - focal * local[valid, 2] / depth[valid]
    return pixels


def project_polyline(points, camera, ground_z=0.):
    """精确裁剪3D折线到近平面和相机视野；NaN隔开不可见区间。"""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("轨迹含非有限坐标，不能生成投影")
    xyz = np.column_stack((points, np.full(len(points), ground_z)))
    local = (xyz - np.asarray(camera["pos"])) @ carla_rotation(camera["rot"])
    w, h = camera["width"], camera["height"]
    f = w / (2. * np.tan(np.deg2rad(camera["fov"]) / 2.))
    cx, cy = w / 2., h / 2.
    # 视野边界的齐次不等式：normal @ camera_xyz + offset >= 0。
    normals = np.array([[1., 0., 0.], [cx+.5, f, 0.], [w-.5-cx, -f, 0.],
                        [cy+.5, 0., -f], [h-.5-cy, 0., f]])
    offsets = np.array([-.1, 0., 0., 0., 0.])
    result = []
    previous_visible_end = False
    for start, end in zip(local[:-1], local[1:]):
        at_start, at_end = normals @ start + offsets, normals @ end + offsets
        lo, hi = 0., 1.
        for a, b in zip(at_start, at_end):
            if a < 0 and b < 0:
                lo, hi = 1., 0.
                break
            if a < 0:
                lo = max(lo, a / (a-b))
            elif b < 0:
                hi = min(hi, a / (a-b))
        if lo > hi:
            previous_visible_end = False
            continue
        clipped = np.stack((start + lo*(end-start), start + hi*(end-start)))
        pixels = np.column_stack((cx + f*clipped[:, 1]/clipped[:, 0], cy - f*clipped[:, 2]/clipped[:, 0]))
        if result and previous_visible_end and lo == 0.:
            result.append(pixels[1])
        else:
            if result:
                result.append([np.nan, np.nan])
            result.extend(pixels)
        previous_visible_end = hi == 1.
    return np.asarray(result, dtype=float).reshape(-1, 2)


def hdmap_extent(config):
    """像素中心匹配原warpAffine的size-1比例，返回imshow的米制边界。"""
    size, ppm, bottom = config["size"], config["pixels_per_meter"], config["pixels_ev_to_bottom"]
    half_pixel_m = size / (size - 1) / ppm / 2.
    return (-size/2/ppm-half_pixel_m, size/2/ppm+half_pixel_m,
            -bottom/ppm-half_pixel_m, (size-bottom)/ppm+half_pixel_m)


def load_scene(route, frame, configuration=None):
    """读取同帧真实场景，第三人称只有图像与标定均存在才叠轨迹。"""
    from PIL import Image
    route = Path(route)
    config = scene_configuration(route, frame, configuration)
    scene = dict(config=config, hdmap=None, boxes=[], third_person=None,
                 audit=dict(calibration_source=config["source"], calibration_file=config["configuration_file"],
                            calibration_sha256=config.get("configuration_sha256"),
                            calibration_resolution=config["calibration_resolution"],
                            calibration_fallback_reason=config.get("calibration_fallback_reason"),
                            projection="CARLA ego x forward/y right, horizontal FOV, ground-plane approximation; no occlusion test",
                            ground_z_m=config.get("ground_z_m", 0.), sources={}))
    def source(path):
        """保存显示素材的内容身份，绝不注入模型条件。"""
        scene["audit"]["sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    hdpath = route / "hdmap" / f"{frame:04d}.png"
    hdconfig = config.get("hdmap")
    if hdpath.is_file() and hdconfig:
        with Image.open(hdpath) as im:
            array = np.array(im)
        if array.shape != (hdconfig["size"], hdconfig["size"]):
            raise ValueError(f"hdmap尺寸与显示标定不符: {hdpath}")
        if not set(np.unique(array)).issubset({0, 1, 3, 4}):
            raise ValueError(f"不是支持的LEAD道路分类图: {hdpath}")
        array = np.rot90(array, -hdconfig["stored_rotation_k"])
        colors = np.array([[229, 235, 227], [96, 104, 114], [229, 235, 227], [242, 241, 224], [242, 203, 89]], dtype=np.uint8)
        scene["hdmap"] = colors[array]
        source(hdpath)
    boxpath = route / "bboxes" / f"{frame:04d}.pkl"
    if boxpath.is_file():
        with lzma.open(boxpath, "rb") as handle:
            boxes = pickle.load(handle)
        for box in boxes:
            if box.get("class") in ("ego_car", "car", "walker", "pedestrian", "static") and all(key in box for key in ("position", "extent", "yaw")):
                values = [*box["position"][:2], *box["extent"][:2], box["yaw"]]
                if not np.isfinite(values).all() or min(box["extent"][:2]) < 0:
                    raise ValueError(f"非法框几何: {boxpath}")
                scene["boxes"].append(dict(category=box["class"], position=box["position"][:2], extent=box["extent"][:2], yaw=box["yaw"]))
        source(boxpath)
    third = config.get("third_person")
    if third:
        path = route / third["image_pattern"].format(frame=frame)
        if path.is_file():
            with Image.open(path) as im:
                if im.size != (third["width"], third["height"]):
                    raise ValueError(f"第三人称图尺寸与标定不符: {path}")
                scene["third_person"] = np.array(im.convert("RGB"))
            source(path)
    scene["audit"].update(hdmap_available=scene["hdmap"] is not None, actor_boxes=len(scene["boxes"]),
                          schematic_ego_fallback=not bool(scene["boxes"]),
                          third_person_available=scene["third_person"] is not None)
    return scene


def draw_scene_bev(ax, scene, cloud=None):
    """道路、车道线和同帧目标框组成可审计俯视场景。"""
    from matplotlib.patches import Polygon
    if scene["hdmap"] is not None:
        ax.imshow(scene["hdmap"], extent=hdmap_extent(scene["config"]["hdmap"]), origin="upper", interpolation="nearest", zorder=0)
    else:
        ax.set_facecolor("#eef1f3")
        if cloud is not None:
            ax.scatter(cloud[:, 1], cloud[:, 0], s=.3, color="#aab2bb", rasterized=True, zorder=0)
    boxes = scene["boxes"] or [dict(category="ego_car", position=[0., 0.], extent=[2.4, .95], yaw=0.)]
    for box in boxes:
        x, y = box["position"]
        ex, ey = box["extent"]
        c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
        polygon = np.array([[-ex, -ey], [ex, -ey], [ex, ey], [-ex, ey]]) @ np.array([[c, s], [-s, c]]) + [x, y]
        color = "#303944" if box["category"] == "ego_car" else "#f0a15a" if box["category"] in ("walker", "pedestrian") else "#abb8c6"
        ax.add_patch(Polygon(polygon[:, [1, 0]], facecolor=color, edgecolor="white", linewidth=.9, alpha=.95, zorder=1))


def draw_camera(ax, rgb, camera, curves, ground_z):
    """在真实RGB上投影同一组轨迹，白边保证路面/阴影中的可读性。"""
    import matplotlib.patheffects as pe
    ax.imshow(rgb)
    for curve in [*curves[1:], curves[0]]:
        pixels = project_polyline(curve["points"], camera, ground_z)
        line, = ax.plot(pixels[:, 0], pixels[:, 1], color=curve["color"], linestyle=curve["linestyle"], linewidth=2.3, zorder=3)
        line.set_path_effects([pe.Stroke(linewidth=4., foreground="white", alpha=.85), pe.Normal()])
    ax.set(xlim=(-.5, camera["width"]-.5), ylim=(camera["height"]-.5, -.5))
    ax.axis("off")


def render_paper(folder, row, predictions, models, inputs, scene, cloud=None, stem="comparison"):
    """主图同帧场景+GT+所有方法，另存PDF；不把历史图当当前预测视角。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    from qwen3vl_local.action_prior.comparison_cases import event_groups
    folder = Path(folder)
    config = scene["config"]
    first = predictions[models[0]["id"]]
    with Image.open(inputs[models[0]["id"]][-1]) as im:
        rgb = np.array(im.convert("RGB"))
    expected_width = sum(camera["width"] for camera in config["views"])
    if rgb.shape[:2] != (config["views"][0]["height"], expected_width) or len({c["height"] for c in config["views"]}) != 1:
        raise ValueError("RGB尺寸与三相机标定不同；请提供实际 --camera-config，不能缩放后猜内参")
    def curves(predkey, gtkey):
        """相同图例/颜色跨RGB与BEV保持一致。"""
        return [dict(points=first[gtkey][0], color="#161b22", linestyle="--", label="GT")] + [
            dict(points=predictions[m["id"]][predkey][0], color=m["color"], linestyle="-", label=m["label"]) for m in models]
    route = curves("pred_route", "gt_route")
    waypoints = curves("pred_waypoints", "gt_waypoints")
    has_third = scene["third_person"] is not None
    header_height = .8 + .27 * len(models)
    fig = plt.figure(figsize=(15, 10.2 + 4.2 * has_third + header_height))
    grid = fig.add_gridspec(3 if has_third else 2, 6, height_ratios=[3.6, 4.2, 6] if has_third else [3.6, 6], hspace=.13, wspace=.08)
    offset = 0
    for index, camera in enumerate(config["views"]):
        ax = fig.add_subplot(grid[0, index*2:index*2+2])
        draw_camera(ax, rgb[:, offset:offset+camera["width"]], camera, waypoints, config.get("ground_z_m", 0.))
        ax.set_title(f"{camera['name']} · future waypoints", fontsize=11)
        offset += camera["width"]
    if has_third:
        for column, items in enumerate((route, waypoints)):
            ax = fig.add_subplot(grid[1, column*3:column*3+3])
            draw_camera(ax, scene["third_person"], config["third_person"], items, config.get("ground_z_m", 0.))
            ax.set_title("Recorded CARLA third-person · " + ("route" if column == 0 else "waypoints"), fontsize=11)
    for column, items in enumerate((route, waypoints)):
        ax = fig.add_subplot(grid[-1, column*3:column*3+3])
        draw_scene_bev(ax, scene, cloud)
        points = np.concatenate([np.asarray(item["points"]) for item in items] + [np.zeros((1, 2))])
        limits = [min(-10., points[:, 1].min()-4), max(10., points[:, 1].max()+4), min(-7., points[:, 0].min()-3), max(22., points[:, 0].max()+6)]
        for item in [*items[1:], items[0]]:
            points = np.asarray(item["points"])
            ax.plot(points[:, 1], points[:, 0], color=item["color"], linestyle=item["linestyle"], marker="o", markersize=2.5, linewidth=2.2, zorder=4)
        ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="Right (m)", ylabel="Forward (m)")
        ax.set_aspect("equal", adjustable="box")
        kind = "CARLA road / actors" if scene["hdmap"] is not None else "LiDAR / actors" if cloud is not None else "Metric frame / actors"
        ax.set_title(kind + (" · route" if column == 0 else " · future waypoints (~2 s)"), fontsize=11)
        ax.grid(alpha=.12)
    title = f"{row['split']}  |  {row['scenario']}  |  frame {row['anchor']}  |  Event: {', '.join(event_groups(row))}  |  Action: {row['action_token']['name']}"
    error = row.get("visualization_error", {})
    if error.get("enabled"):
        strongest = max(error["triggers"], key=lambda item: item["distance_m"]/item["threshold_m"])
        fig.text(.5, .048, f"Selected: {strongest['trajectory']} {' vs '.join(strongest['pair'])} "
                 f"{strongest['metric'].upper()}={strongest['distance_m']:.2f} m > {strongest['threshold_m']:g} m (all-model selection)",
                 ha="center", fontsize=9)
    fig.suptitle(title, fontsize=12, y=.993)
    labels = ["GT (recorded expert)"]
    handles = [Line2D([0], [0], color="#161b22", linewidth=2.4, linestyle="--")]
    for model in models:
        data = predictions[model["id"]]
        token = data["sample"].get("action_token", {}).get("name", "OFF")
        text_action = data.get("high_level_action") or {}
        text_action = ",".join(text_action.get("actions", [])) or text_action.get("status", "OFF")
        labels.append(f"{model['label']} | step {model['selection']['step']} EMA | token={token}, text={text_action} | WP ADE={data['metrics']['waypoint_ade_m']:.3f}m")
        handles.append(Line2D([0], [0], color=model["color"], linewidth=2.4))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), frameon=False, fontsize=10, ncol=1)
    fig.subplots_adjust(left=.055, right=.98, bottom=.075, top=1-header_height/fig.get_figheight())
    calibration_note = {"recorded_anchor_meta": "recorded frame calibration",
                        "explicit_route_json": "explicit route calibration",
                        "explicit_global_json": "explicit global calibration",
                        "nominal_fallback": "NOMINAL calibration fallback"}[config["calibration_resolution"]]
    fig.text(.5, .012, f"RGB: {calibration_note}; ground-plane projection, no occlusion test.\nBEV: recorded road/actor annotations where available; visualization only.", ha="center", fontsize=8, color="#586270")
    fig.savefig(folder / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
