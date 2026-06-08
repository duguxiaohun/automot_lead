"""Ego-frame 点投影 + 拼接图 overlay 工具。

为什么自己写：LEAD 的 viz_utils / common_utils 依赖 jaxtyping / beartype /
PredictedBoundingBox 等若干内部模块，引进来会拖一大坨。这里只摘 LEAD 的
pinhole 投影公式（lead/lead/common/common_utils.py:project_points_to_image）
重写成无依赖版本。

投影约定（与 LEAD 完全一致）：
- ego frame：x_forward, y_left（CARLA 数据集风格的车体坐标，与 LeadMoT 输出
  对齐：pred_waypoints / pred_route 都是 (x_forward, y_left) ego-frame 累计点）。
  注意 LEAD project_points_to_image 把世界 y 解读为 right，而我们把 LeadMoT
  输出当成 (forward, left)，等价做法是把 y 取反后传入。
- 相机外参 (x, y, z) 是相机相对 ego 的位置；旋转 (roll, pitch, yaw) 度数。
- 相机内参由 FOV 推：focal_y = H / (2 * tan(fov/2))；focal_x 按宽高比换算。
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def _euler_deg_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """与 LEAD common_utils.euler_deg_to_mat 完全等价：Rz @ Ry @ Rx。"""
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_ego_points_to_image(
    points_ego_xy: np.ndarray,
    camera_pos: Sequence[float],
    camera_rot_deg: Sequence[float],
    camera_fov_deg: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 ego-frame 2D 点 (N, 2) [x_forward, y_left] 投到一个相机像素 (N, 2)。

    返回 (proj_xy, inside_mask)，inside_mask True 表示落在 (0..W, 0..H) 内。
    与 LEAD project_points_to_image 数学一致：仅多做一步 y_left -> y_right
    取反（LeadMoT 输出是 ego x_forward / y_left，LEAD 原函数要求 y_right）。
    """
    pts = np.asarray(points_ego_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    # LeadMoT/训练标签使用 y_left；LEAD pinhole 投影使用 y_right。
    # 这里统一取反，后续所有调用都可以继续传 ego-frame (x_forward, y_left)。
    pts3 = np.column_stack([pts[:, 0], -pts[:, 1], np.zeros(len(pts))])

    cam_pos = np.asarray(camera_pos, dtype=np.float64)
    R = _euler_deg_to_mat(camera_rot_deg[0], camera_rot_deg[1], camera_rot_deg[2])
    # 先把点从 ego 原点平移到相机原点，再按相机外参旋到相机坐标。
    pts_translated = pts3 - cam_pos
    pts_camera = (R @ pts_translated.T).T

    # ego(world)(x_fwd, y_right, z_up) -> camera(x_right, y_down, z_forward)
    pts_remap = np.column_stack([
        pts_camera[:, 1],
        -pts_camera[:, 2],
        pts_camera[:, 0],
    ])

    fov_rad = math.radians(float(camera_fov_deg))
    # LEAD 的三相机是方形 384x384，但这里按通用 W/H 写，方便以后复用。
    focal_y = image_h / (2.0 * math.tan(fov_rad / 2.0))
    aspect = image_w / float(image_h)
    focal_x = focal_y * aspect
    cx = image_w / 2.0
    cy = image_h / 2.0

    z = pts_remap[:, 2]
    # z<=0 表示在相机后方；这类点不能参与除法投影。
    valid = z > 1e-6
    proj = np.zeros((len(pts), 2), dtype=np.float64)
    inside = np.zeros((len(pts),), dtype=bool)
    proj[valid, 0] = (focal_x * pts_remap[valid, 0] / z[valid]) + cx
    proj[valid, 1] = (focal_y * pts_remap[valid, 1] / z[valid]) + cy
    inside_xy = (proj[:, 0] >= 0) & (proj[:, 0] < image_w) \
        & (proj[:, 1] >= 0) & (proj[:, 1] < image_h)
    inside = valid & inside_xy
    return proj, inside


def draw_waypoints_on_segment(
    bgr_segment: np.ndarray,
    waypoints_ego: np.ndarray,
    camera_pos: Sequence[float],
    camera_rot_deg: Sequence[float],
    camera_fov_deg: float,
    color: tuple[int, int, int] = (0, 255, 255),
    radius: int = 3,
    line_thickness: int = 2,
) -> np.ndarray:
    """在单相机段 BGR 图像上画一条 ego-frame waypoint 折线 + 圆点。"""
    h, w = bgr_segment.shape[:2]
    proj, inside = project_ego_points_to_image(
        waypoints_ego, camera_pos, camera_rot_deg, camera_fov_deg, w, h
    )
    out = bgr_segment
    # 先画点再连线：即使某段线跨出视野，仍能看到落在视野内的 waypoint。
    for pt, ok in zip(proj, inside):
        if ok:
            cv2.circle(out, (int(pt[0]), int(pt[1])), radius, color, -1,
                       lineType=cv2.LINE_AA)
    for i in range(len(proj) - 1):
        if inside[i] and inside[i + 1]:
            cv2.line(out,
                     (int(proj[i, 0]), int(proj[i, 1])),
                     (int(proj[i + 1, 0]), int(proj[i + 1, 1])),
                     color, line_thickness, lineType=cv2.LINE_AA)
    return out


def draw_target_point_on_segment(
    bgr_segment: np.ndarray,
    tp_ego: np.ndarray,
    camera_pos: Sequence[float],
    camera_rot_deg: Sequence[float],
    camera_fov_deg: float,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """画一个 target_point 圆（按距离做透视半径，参考 LEAD draw_target_points）。"""
    h, w = bgr_segment.shape[:2]
    proj, inside = project_ego_points_to_image(
        tp_ego.reshape(1, 2), camera_pos, camera_rot_deg, camera_fov_deg, w, h
    )
    if not inside[0]:
        return bgr_segment
    # target_point 画成近大远小的圆，视觉上更接近 LEAD 原 debug 图。
    fov_rad = math.radians(float(camera_fov_deg))
    focal = (w / 2.0) / math.tan(fov_rad / 2.0)
    dist = float(np.linalg.norm(
        np.array([tp_ego[0], -tp_ego[1], 0.0]) - np.asarray(camera_pos)
    )) or 1e-3
    pixel_r = int(max(2, min(50, 0.10 * focal / dist)))   # 10cm 球
    x, y = int(proj[0, 0]), int(proj[0, 1])
    cv2.circle(bgr_segment, (x, y), pixel_r + 2, (255, 255, 255), -1,
               lineType=cv2.LINE_AA)
    cv2.circle(bgr_segment, (x, y), pixel_r, color, -1, lineType=cv2.LINE_AA)
    return bgr_segment


def overlay_pred_on_stitched_three_cams(
    stitched_bgr: np.ndarray,
    pred_waypoints_ego: np.ndarray | None,
    target_point_ego: np.ndarray | None,
    cam_calibs_3cam: Sequence[dict],
    fov_deg: float = 60.0,
) -> np.ndarray:
    """在 1152×384 三视角拼接图上分段叠加 pred 与 target_point。

    cam_calibs_3cam 顺序与拼接顺序一致：[左前, 前, 右前]，每个 dict 至少含
    pos / rot 字段（roll, pitch, yaw 度）。
    """
    h, w = stitched_bgr.shape[:2]
    n = len(cam_calibs_3cam)
    seg_w = w // n
    out = stitched_bgr.copy()
    for i, calib in enumerate(cam_calibs_3cam):
        # stitched 图是 [左前, 前, 右前] 横拼；逐段切出来用各自相机外参投影。
        x0 = i * seg_w
        x1 = (i + 1) * seg_w if i < n - 1 else w
        seg = out[:, x0:x1].copy()
        if pred_waypoints_ego is not None and len(pred_waypoints_ego) > 0:
            seg = draw_waypoints_on_segment(
                seg, pred_waypoints_ego, calib["pos"], calib["rot"], fov_deg
            )
        if target_point_ego is not None:
            seg = draw_target_point_on_segment(
                seg, target_point_ego, calib["pos"], calib["rot"], fov_deg
            )
        out[:, x0:x1] = seg
    return out
