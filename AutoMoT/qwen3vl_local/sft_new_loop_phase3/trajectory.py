#!/usr/bin/env python3
"""旧版实验工具，未被当前 build/train/eval 使用。正式标定入口是 trajectory_action.py。

LEAD meta 轨迹读取与旧 Phase3 动作特征提取。

Phase3 的监督目标是 "接下来 4 秒的 high-level 动作"，因此不能只看当前帧的
speed/brake，必须使用 LEAD 采集时保存的特权未来轨迹。本模块只做三件事：

1. 解压并读取 ``metas/<frame>.pkl``（LEAD 用 lzma 压缩，历史文件可能是裸 pickle）；
2. 把 ``next_target_points`` 转到 ego frame，作为学生可见的导航输入；
3. 从 ``future_speeds`` / ``future_positions`` / ``route`` 计算纵向与横向机动特征。

坐标系（2026-09-04 用 ``ego_matrix`` 逆变换 + ParkedObstacleTwoWays 借道段 RGB 逐帧
核对确认）：ego frame 为 **x 向前为正、y 向右为正**，与 CARLA 左手系一致。
``future_positions`` / ``future_speeds`` 是 20Hz、10 秒的特权未来；LEAD 每 5 tick 落盘
一帧，所以数据集帧率是 4Hz，1 个数据集帧 = 5 个 future tick。
"""

from __future__ import annotations

import lzma
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


FUTURE_TICK_S = 0.05
DATASET_FPS = 4.0
ACTION_HORIZON_S = 4.0
ROUTE_LATERAL_WINDOW_M = 3.0


def load_meta(run_dir: pathlib.Path, frame_id: int) -> Optional[dict]:
    """读取一帧 meta；文件缺失或损坏时返回 ``None`` 让上层跳过该帧。"""

    path = pathlib.Path(run_dir) / "metas" / f"{int(frame_id):04d}.pkl"
    if not path.is_file():
        return None
    try:
        with lzma.open(path, "rb") as handle:
            return pickle.loads(handle.read())
    except Exception:
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            return None


def _inverse_conversion_2d(point: Sequence[float], translation: Sequence[float], yaw: float) -> np.ndarray:
    """把 world 2D 点转到 ego frame，与 ``sft_v5.build_dataset`` 同一套公式。"""

    pt = np.asarray(point, dtype=np.float64).reshape(2)
    tr = np.asarray(translation, dtype=np.float64).reshape(2)
    delta = pt - tr
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    return np.asarray([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]], dtype=np.float64)


def ego_to_goal_xy(meta: Mapping[str, Any], *, index: int = -1) -> Optional[Tuple[float, float]]:
    """返回 ``next_target_points[index]`` 的 ego-frame 坐标 (x 前, y 右)。

    ``index=-1`` 是 route 终点（最终目的地），``index=1`` 通常是下一个导航目标点。
    缺字段时返回 ``None``；绝不回退到 ``route[-1]`` 或 ``(0, 0)``，否则会给路口
    左右转 / 匝道样本塞进错误方向信号。
    """

    points = np.asarray(meta.get("next_target_points", []), dtype=np.float64)
    if points.size == 0:
        return None
    points = points.reshape(-1, points.shape[-1])
    if points.shape[-1] < 2 or "pos_global" not in meta or "theta" not in meta:
        return None
    try:
        target = points[int(index), :2]
    except IndexError:
        return None
    pos = np.asarray(meta["pos_global"], dtype=np.float64).reshape(-1)[:2]
    theta = float(np.asarray(meta["theta"], dtype=np.float64).reshape(-1)[0])
    goal = _inverse_conversion_2d(target, pos, theta)
    if not np.all(np.isfinite(goal)):
        return None
    return float(goal[0]), float(goal[1])


def _arc_fit_residual(points: np.ndarray) -> np.ndarray:
    """把一段折线拟合成圆弧/直线，返回逐点有符号横向残差。

    路口转弯、匝道弯道本身就是圆弧，残差接近 0；车道变更是叠加在道路几何上的
    横向阶跃，残差会出现 ~半个车道宽的凸起。因此残差是区分 "转弯" 和 "变道" 的
    关键量，不能只看原始 ``route`` 的 y 值。
    """

    if points.ndim != 2 or len(points) < 6:
        return np.zeros(len(points), dtype=np.float64)
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    design = np.column_stack([x, y, np.ones_like(x)])
    target = x**2 + y**2
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(len(points), dtype=np.float64)
    cx = float(solution[0]) / 2.0
    cy = float(solution[1]) / 2.0
    radius_sq = float(solution[2]) + cx**2 + cy**2
    if not np.isfinite(radius_sq) or radius_sq <= 0.0 or radius_sq > 2.5e9:
        slope, intercept = np.polyfit(x, y, 1)
        return y - (slope * x + intercept)
    radius = float(np.sqrt(radius_sq))
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    # 圆心在自车右侧时，"离圆心更远" 等价于 "更靠左"，统一成 y 向右为正的符号。
    return residual if cy < 0 else -residual


@dataclass(frozen=True)
class TrajectoryFeatures:
    """一帧的 Phase3 纵向 / 横向动作特征。"""

    speed_now: float
    speed_min: float
    speed_end: float
    speed_max: float
    stop_hold_s: float
    lateral_ramp_m: float
    lateral_ramp_signed_m: float
    lateral_residual_m: float
    lateral_round_trip: bool
    lane_change_distance: float
    future_lane_change: bool
    horizon_m: float

    def as_dict(self) -> dict:
        """写进 JSONL / audit 的紧凑字典。"""

        return {
            "speed_now": round(self.speed_now, 3),
            "speed_min": round(self.speed_min, 3),
            "speed_end": round(self.speed_end, 3),
            "speed_max": round(self.speed_max, 3),
            "stop_hold_s": round(self.stop_hold_s, 2),
            "lateral_ramp_m": round(self.lateral_ramp_m, 3),
            "lateral_ramp_signed_m": round(self.lateral_ramp_signed_m, 3),
            "lateral_residual_m": round(self.lateral_residual_m, 3),
            "lateral_round_trip": bool(self.lateral_round_trip),
            "lane_change_distance": (
                None if not np.isfinite(self.lane_change_distance) else round(self.lane_change_distance, 3)
            ),
            "future_lane_change": bool(self.future_lane_change),
            "horizon_m": round(self.horizon_m, 2),
        }


def _stop_hold_seconds(speeds: np.ndarray, threshold: float = 0.5) -> float:
    """返回 horizon 内最长的连续低速（近似停车）持续时间。"""

    best = 0
    current = 0
    for value in speeds:
        if float(value) <= threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best) * FUTURE_TICK_S


def _route_lateral(route: np.ndarray, horizon_m: float) -> Tuple[float, float, float, bool]:
    """返回 (最大 3m 弧长横向阶跃, 有符号阶跃, 全 route 弧拟合残差峰值, 是否往返)。

    阶跃在 horizon 内计算，因为 Phase3 只关心 "接下来 4 秒"；弧拟合残差用完整 50m
    route，避免短窗口把变道本身当成道路曲率吸收掉。
    """

    if route.ndim != 2 or len(route) < 6:
        return 0.0, 0.0, 0.0, False
    residual = _arc_fit_residual(route)
    residual_peak = float(np.max(np.abs(residual))) if residual.size else 0.0

    steps = np.linalg.norm(np.diff(route, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    keep = arc <= max(float(horizon_m), ROUTE_LATERAL_WINDOW_M + 1.0)
    arc = arc[keep]
    lateral = residual[keep] if residual.size == len(route) else route[keep, 1]
    if len(arc) < 4:
        return 0.0, 0.0, residual_peak, False

    best = 0.0
    signed = 0.0
    ramps = []
    for i in range(len(arc)):
        j = int(np.searchsorted(arc, arc[i] + ROUTE_LATERAL_WINDOW_M))
        if j >= len(arc):
            break
        delta = float(lateral[j] - lateral[i])
        ramps.append(delta)
        if abs(delta) > best:
            best = abs(delta)
            signed = delta
    positive = max([r for r in ramps if r > 0.0], default=0.0)
    negative = min([r for r in ramps if r < 0.0], default=0.0)
    round_trip = positive >= 1.0 and abs(negative) >= 1.0
    return best, signed, residual_peak, round_trip


def _lane_identity(meta: Mapping[str, Any]) -> Tuple[str, int]:
    """返回 (road_id, lane_id)，用于判断未来是否真的换了车道。"""

    lane = meta.get("ego_lane_id")
    if lane is None:
        lane = meta.get("lane_id")
    try:
        lane_value = int(lane)
    except (TypeError, ValueError):
        lane_value = 0
    return str(meta.get("road_id", "")), lane_value


def extract_features(
    meta: Mapping[str, Any],
    *,
    future_metas: Sequence[Mapping[str, Any]] = (),
    horizon_s: float = ACTION_HORIZON_S,
) -> Optional[TrajectoryFeatures]:
    """从当前帧 meta（可选加未来数据集帧 meta）提取动作特征。"""

    speeds = np.asarray(meta.get("future_speeds", []), dtype=np.float64).reshape(-1)
    if speeds.size < 2:
        return None
    count = min(len(speeds), int(round(float(horizon_s) / FUTURE_TICK_S)) + 1)
    window = speeds[:count]
    speed_now = float(meta.get("speed", window[0]))
    route = np.asarray(meta.get("route", []), dtype=np.float64)
    horizon_m = float(np.clip(speed_now * float(horizon_s), 15.0, 35.0))
    ramp, ramp_signed, residual_peak, round_trip = _route_lateral(route, horizon_m)

    raw_distance = meta.get("signed_dist_to_lane_change")
    try:
        lane_change_distance = float(raw_distance)
    except (TypeError, ValueError):
        lane_change_distance = float("inf")

    road_now, lane_now = _lane_identity(meta)
    future_lane_change = False
    for future in future_metas:
        road_future, lane_future = _lane_identity(future)
        if road_future == road_now and lane_future != lane_now:
            future_lane_change = True
            break
        if road_future != road_now and abs(lane_future) != abs(lane_now):
            future_lane_change = True
            break

    return TrajectoryFeatures(
        speed_now=speed_now,
        speed_min=float(np.min(window)),
        speed_end=float(window[-1]),
        speed_max=float(np.max(window)),
        stop_hold_s=_stop_hold_seconds(window),
        lateral_ramp_m=ramp,
        lateral_ramp_signed_m=ramp_signed,
        lateral_residual_m=residual_peak,
        lateral_round_trip=round_trip,
        lane_change_distance=lane_change_distance,
        future_lane_change=future_lane_change,
        horizon_m=horizon_m,
    )
