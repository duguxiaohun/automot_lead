#!/usr/bin/env python3
"""Phase3 高层动作标定：只用 LEAD meta 的真实自车轨迹与地图车道身份。

设计口径来自 2026-09-04 对 AccidentTwoWays / HardBreakRoute / EnterActorFlow /
HighwayExit / InvadingTurn 等 route 的逐帧 meta + RGB 复核：

* 纵向动作只看未来真实速度曲线，不看 scenario 名或事件标签；
* 横向动作绝不用航向角或 steer 判定。弯道会让 steer/yaw 长期非零，但不换车道。
  因此变道必须由 OpenDRIVE 车道身份 (``road_id`` + ``lane_id``) 的真实切换触发，
  需要相邻两帧确认新车道，遇到 road 身份切换则停止跨 road 比较；
  planned route 可能本身含绕行，只作为审计辅助，不冒充原车道中心线。
* 借对向车道绕障 (U-E2) 与回原车道 (R-E2) 在 meta 中都表现为同一 ``road_id``
  上的 ``lane_id`` 跨中心线切换，方向由 OpenDRIVE 的 lane id 排序 + 行驶方向决定。
"""

from __future__ import annotations

import lzma
import math
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qwen3vl_local.sft_new_loop_phase3.lateral_rgb_audit import lateral_uncertainty


FRAME_DT_SECONDS = 0.25
ACTION_RULE_VERSION = "ordered_speed_driving_lane_v5"


def validate_action_rule(row: Mapping[str, Any]) -> None:
    """训练/eval 拒绝旧规则索引，防止只换 prompt 却继续学习旧动作标签。"""
    version = (row.get("action_evidence") or {}).get("rule_version")
    if version != ACTION_RULE_VERSION:
        raise ValueError(f"action rule mismatch: {version!r}; rebuild index from raw meta")

# 纵向 horizon：STOP 用更短的即时窗，避免“已经停稳但马上起步”被判成继续停车。
IMMEDIATE_HORIZON_FRAMES = 6
LONGITUDINAL_HORIZON_FRAMES = 8
LATERAL_HORIZON_FRAMES = 12

STOP_SPEED_MPS = 0.5
STOP_RELEASE_SPEED_MPS = 2.0
LONGITUDINAL_MIN_DELTA_MPS = 1.2
LONGITUDINAL_RELATIVE_DELTA = 0.20
LATERAL_MIN_SHIFT_M = 1.0

DIRECTION_LEFT = "LEFT"
DIRECTION_RIGHT = "RIGHT"

# 2026-09-04 的 probe_ego_frame_sign.py 用左/右转 scenario 的 route 折线取证：
# LEAD ego frame 是 CARLA 左手系，x 正为正前方，y 负为左、y 正为右。
EGO_FRAME_LEFT_SIGN = -1.0


def _travel_sign(entry_lane_id: int) -> int:
    """返回自车在该 road 上的行驶方向相对 OpenDRIVE s 轴的符号。

    OpenDRIVE 同一 road 上 lane id 自右向左递增，负 id 车道沿 +s 行驶。这里必须用
    自车“首次合法进入该 road 时”的车道，而不是当前车道：借对向车道绕障时当前
    lane id 会翻到正值，但自车航向没变，用当前 lane id 会把回原车道判成左变道。
    """

    return 1 if int(entry_lane_id) < 0 else -1


def lane_change_direction_from_ids(from_lane: int, to_lane: int, entry_lane: int) -> Optional[str]:
    """由同一 road 上的 lane id 切换推出自车视角的横向方向。"""

    if int(from_lane) == int(to_lane):
        return None
    signed = (int(to_lane) - int(from_lane)) * _travel_sign(entry_lane)
    if signed == 0:
        return None
    return DIRECTION_LEFT if signed > 0 else DIRECTION_RIGHT


def _load_meta(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """读取 xz 压缩的 LEAD meta；坏文件返回 None 而不是抛错。"""

    try:
        with lzma.open(path, "rb") as handle:
            meta = pickle.load(handle)
    except Exception:
        try:
            with path.open("rb") as handle:
                meta = pickle.load(handle)
        except Exception:
            return None
    return meta if isinstance(meta, dict) else None


def _scalar(value: Any, default: float = 0.0) -> float:
    """把 numpy/None/inf 统一成有限 float。"""

    try:
        out = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int_field(value: Any, default: int = 0) -> int:
    """读取整型 meta 字段。"""

    try:
        return int(value)
    except Exception:
        return int(default)


def _ego_frame_xy(point: Sequence[float], origin: Sequence[float], theta: float) -> Tuple[float, float]:
    """把 world 点转到 ego frame，与 sft_base/v3/v4 的 final_goal 公式同源。"""

    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    c = math.cos(-float(theta))
    s = math.sin(-float(theta))
    return (c * dx - s * dy, s * dx + c * dy)


def _signed_lateral_offset_from_polyline(polyline: np.ndarray, point: Sequence[float]) -> Optional[float]:
    """点相对折线的带符号横向偏移；与折线同一坐标系，正号与折线 y 轴同向。"""

    if polyline.ndim != 2 or polyline.shape[0] < 2:
        return None
    px, py = float(point[0]), float(point[1])
    best: Optional[Tuple[float, float]] = None
    for idx in range(polyline.shape[0] - 1):
        a = polyline[idx]
        b = polyline[idx + 1]
        d = b - a
        norm = float(math.hypot(d[0], d[1]))
        if norm < 1e-6:
            continue
        t = ((px - a[0]) * d[0] + (py - a[1]) * d[1]) / (norm * norm)
        t = min(1.0, max(0.0, t))
        proj = a + t * d
        dist = float(math.hypot(px - proj[0], py - proj[1]))
        cross = (d[0] * (py - a[1]) - d[1] * (px - a[0])) / norm
        if best is None or dist < best[0]:
            best = (dist, float(cross))
    return None if best is None else best[1]


@dataclass
class RouteTrajectory:
    """一条 run 的逐帧 meta 轨迹缓存。"""

    run_dir: pathlib.Path
    frames: Tuple[int, ...]
    metas: Dict[int, Dict[str, Any]]
    road_entry_lane: Dict[int, int]

    def lateral_window_issue(self, frame_id: int, horizon: int = LATERAL_HORIZON_FRAMES) -> Optional[str]:
        """Any waypoint 可落到路肩；缺失/非 Driving/跨 road 窗口不能监督横向 NO。

        ego_lane_id 来自另一次 Driving 查询，但没有配套 road_id，不能与 Any 的
        road_id 拼接成虚构身份。原始 xodr 未核验时保守排除这些窗口。
        """
        base = self.metas.get(int(frame_id), {})
        for offset in range(int(horizon) + 2):
            meta = self.metas.get(int(frame_id) + offset)
            if meta is None:
                return "missing_meta"
            if meta.get("lane_type_str") != "Driving":
                return "non_driving_or_unknown_waypoint"
            if meta.get("road_id") != base.get("road_id"):
                return "road_transition"
            if meta.get("lane_id") in (None, 0):
                return "missing_lane_identity"
        return None

    def has(self, frame_id: int) -> bool:
        """当前帧是否可用。"""

        return int(frame_id) in self.metas

    def _future_speeds(self, frame_id: int, horizon: int) -> List[float]:
        """返回 [t, t+1, ..., t+horizon] 的真实速度序列。"""

        out: List[float] = []
        for offset in range(0, int(horizon) + 1):
            meta = self.metas.get(int(frame_id) + offset)
            if meta is None:
                break
            speed = _scalar(meta.get("speed"), float("nan"))
            # 静止 meta 实测有 -0.00022 m/s 等数值抖动，不是缺帧或真实倒车。
            if not math.isfinite(speed) or speed < -0.05:
                break
            out.append(max(0.0, speed))
        return out

    def _future_ego_positions(self, frame_id: int, horizon: int) -> List[Tuple[float, float]]:
        """返回未来位置在当前帧 ego frame 下的坐标。"""

        base = self.metas.get(int(frame_id))
        if base is None:
            return []
        origin = list(base.get("pos_global") or [0.0, 0.0])[:2]
        theta = _scalar(base.get("theta"))
        out: List[Tuple[float, float]] = []
        for offset in range(0, int(horizon) + 1):
            meta = self.metas.get(int(frame_id) + offset)
            if meta is None:
                break
            pos = list(meta.get("pos_global") or [])[:2]
            if len(pos) < 2:
                break
            out.append(_ego_frame_xy(pos, origin, theta))
        return out

    def lateral_shift(self, frame_id: int, horizon: int = LATERAL_HORIZON_FRAMES) -> float:
        """自车未来位置相对当前帧 planned route 折线的最大带符号横向位移。

        planned route 可能已经包含绕行；该偏移仅供审计，不能证明跨越车道线。
        """

        base = self.metas.get(int(frame_id))
        if base is None:
            return 0.0
        route = np.asarray(base.get("route", []), dtype=np.float64)
        if route.ndim != 2 or route.shape[0] < 2:
            return 0.0
        best = 0.0
        for point in self._future_ego_positions(frame_id, horizon)[1:]:
            offset = _signed_lateral_offset_from_polyline(route, point)
            if offset is None:
                continue
            if abs(offset) > abs(best):
                best = float(offset)
        return best

    def lane_change(self, frame_id: int, horizon: int = LATERAL_HORIZON_FRAMES) -> Optional[str]:
        """在 horizon 内检测真实车道身份切换，返回自车视角方向。"""

        if lateral_uncertainty(self.run_dir.parent.name, self.run_dir.name, frame_id, horizon):
            return None
        if self.lateral_window_issue(frame_id, horizon):
            return None
        base = self.metas.get(int(frame_id))
        if base is None:
            return None
        from_lane = _int_field(base.get("lane_id"), 0)
        from_road = _int_field(base.get("road_id"), -10_000)
        # 同一路段再次进入时可能方向相反；只回溯本次连续 road visit。
        entry_lane = from_lane
        for previous in range(int(frame_id) - 1, -1, -1):
            old = self.metas.get(previous)
            if old is None or old.get("road_id") != from_road:
                break
            if old.get("lane_type_str") == "Driving":
                entry_lane = _int_field(old.get("lane_id"), entry_lane)
        if not from_lane or not entry_lane:
            return None
        for offset in range(1, int(horizon) + 1):
            meta = self.metas.get(int(frame_id) + offset)
            if meta is None:
                break
            if _int_field(meta.get("road_id"), -10_001) != from_road:
                break
            to_lane = _int_field(meta.get("lane_id"), 0)
            following = self.metas.get(int(frame_id) + offset + 1, {})
            if (not to_lane or following.get("road_id") != from_road
                    or following.get("lane_id") != to_lane):
                continue
            direction = lane_change_direction_from_ids(from_lane, to_lane, entry_lane)
            if direction is not None:
                return direction
        return None

    def goal_ego_xy(self, frame_id: int) -> Optional[Tuple[float, float]]:
        """当前帧目的地相对坐标；缺字段返回 None，让该帧被跳过。"""

        meta = self.metas.get(int(frame_id))
        if meta is None:
            return None
        points = meta.get("next_target_points") or []
        if not points:
            return None
        origin = list(meta.get("pos_global") or [])[:2]
        if len(origin) < 2:
            return None
        return _ego_frame_xy(list(points[-1])[:2], origin, _scalar(meta.get("theta")))

    def signals(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """返回该帧用于动作标定与审计的全部原始信号。"""

        meta = self.metas.get(int(frame_id))
        if meta is None:
            return None
        speeds = self._future_speeds(frame_id, LONGITUDINAL_HORIZON_FRAMES)
        immediate = self._future_speeds(frame_id, IMMEDIATE_HORIZON_FRAMES)
        goal = self.goal_ego_xy(frame_id)
        lateral_review = lateral_uncertainty(self.run_dir.parent.name, self.run_dir.name, frame_id)
        return {
            "frame_id": int(frame_id),
            "speed": float(speeds[0]) if speeds else 0.0,
            "speed_min": float(min(speeds)) if speeds else 0.0,
            "speed_max": float(max(speeds)) if speeds else 0.0,
            "immediate_speed_min": float(min(immediate)) if immediate else 0.0,
            "immediate_speed_max": float(max(immediate)) if immediate else 0.0,
            "future_speed_count": len(speeds),
            "future_speeds": speeds,
            "lateral_rgb_uncertainty": lateral_review,
            "lateral_window_issue": self.lateral_window_issue(frame_id),
            "lane_type_str": meta.get("lane_type_str"),
            "lateral_observation_complete": not lateral_review and self.lateral_window_issue(frame_id) is None,
            "brake": bool(meta.get("brake")),
            "throttle": _scalar(meta.get("throttle")),
            "speed_limit": _scalar(meta.get("speed_limit"), 8.33),
            "lane_id": _int_field(meta.get("lane_id"), 0),
            "road_id": _int_field(meta.get("road_id"), 0),
            "lane_width": _scalar(meta.get("ego_lane_width"), 3.5),
            "is_junction": bool(meta.get("is_junction")),
            "distance_to_next_junction": _scalar(meta.get("distance_to_next_junction"), float("inf")),
            "changed_route": bool(meta.get("changed_route")),
            "lateral_shift": None,  # 按需审计才算；planned route 不是原车道中心线。
            "lane_change_direction": self.lane_change(frame_id),
            "goal_x": float(goal[0]) if goal else 0.0,
            "goal_y": float(goal[1]) if goal else 0.0,
            "goal_available": goal is not None,
        }


def label_actions(signals: Mapping[str, Any]) -> Optional[Dict[str, bool]]:
    """由未来真实轨迹给出五个 high-level 动作的 YES/NO 标签。

    先检查即时窗持续停车；否则按减速/加速第一次达到阈值的时间选择，三者互斥。
    “先减速再恢复”只给 DECELERATE；变道与纵向动作互相独立。
    """

    if int(signals.get("future_speed_count", 0)) < LONGITUDINAL_HORIZON_FRAMES + 1:
        return None
    speeds = list(signals.get("future_speeds", []))
    if len(speeds) < LONGITUDINAL_HORIZON_FRAMES + 1 or not all(
            math.isfinite(float(v)) and float(v) >= 0 for v in speeds):
        return None
    speed = float(speeds[0])
    threshold = max(LONGITUDINAL_MIN_DELTA_MPS, LONGITUDINAL_RELATIVE_DELTA * max(speed, 1.0))

    immediate = speeds[:IMMEDIATE_HORIZON_FRAMES + 1]
    stopped_pairs = [i for i in range(len(immediate) - 1)
                     if max(immediate[i:i + 2]) <= STOP_SPEED_MPS]
    # 当前已停但持续起步，与先刹到停再起步分开。速度极值不能表达这一区别。
    release_at = next((i for i, v in enumerate(immediate)
                       if v >= STOP_RELEASE_SPEED_MPS), None)
    # 起步后 5.8→5.5 m/s 的巡航调节不是继续停车；旧严格单调条件会误标 STOP。
    # 但重新跌回低速/停住仍保留 STOP。只看 1.5s 即时窗，不偷用远期释放。
    pulling_away = (speed <= STOP_SPEED_MPS and release_at is not None
                    and min(immediate[release_at:]) >= STOP_RELEASE_SPEED_MPS)
    stop = bool(stopped_pairs) and not pulling_away
    decrease_at = next((i for i, v in enumerate(speeds[1:], 1) if speed - v >= threshold), 999)
    # RGB 复核 HardBreakRoute/Town13 f213：单帧速度峰值后前车制动、间距缩小，
    # 不应抢在真正减速前标 RESUME。加速需两个连续采样达到阈值；减速保留即时响应，
    # 防止窗口末端出现制动时因缺少下一帧确认被写成不减速。
    increase_at = next((i for i in range(1, len(speeds) - 1)
                        if min(speeds[i:i + 2]) - speed >= threshold), 999)
    decelerate = not stop and decrease_at < increase_at
    resume = not stop and increase_at < decrease_at

    direction = signals.get("lane_change_direction")
    return {
        "DECELERATE": bool(decelerate),
        "STOP": bool(stop),
        "RESUME": bool(resume),
        "LANE_CHANGE_LEFT": direction == DIRECTION_LEFT,
        "LANE_CHANGE_RIGHT": direction == DIRECTION_RIGHT,
    }


def action_evidence(signals: Mapping[str, Any]) -> Dict[str, Any]:
    """写入索引供后续 RGB/轨迹复核的原始判据。"""

    speed = float(signals["speed"])
    return {
        "speed_mps": round(speed, 3),
        "future_speeds_mps": [round(float(v), 3) for v in signals.get("future_speeds", [])],
        "lateral_observation_complete": bool(signals.get("lateral_observation_complete")),
        "lateral_rgb_uncertainty": signals.get("lateral_rgb_uncertainty"),
        "rule_version": ACTION_RULE_VERSION,
        "resume_confirmation_samples": 2,
        "lane_type_str": signals.get("lane_type_str"),
        "lateral_window_issue": signals.get("lateral_window_issue"),
        "future_speed_min_mps": round(float(signals["speed_min"]), 3),
        "future_speed_max_mps": round(float(signals["speed_max"]), 3),
        "immediate_speed_min_mps": round(float(signals["immediate_speed_min"]), 3),
        "immediate_speed_max_mps": round(float(signals["immediate_speed_max"]), 3),
        "longitudinal_threshold_mps": round(
            max(LONGITUDINAL_MIN_DELTA_MPS, LONGITUDINAL_RELATIVE_DELTA * max(speed, 1.0)), 3
        ),
        "lane_change_direction": signals.get("lane_change_direction") or "",
        "lane_id": int(signals["lane_id"]),
        "road_id": int(signals["road_id"]),
        "route_relative_lateral_shift_m": (
            round(float(signals["lateral_shift"]), 3) if signals.get("lateral_shift") is not None else None),
        "brake": bool(signals["brake"]),
        "is_junction": bool(signals["is_junction"]),
        "horizon_frames": {
            "immediate": IMMEDIATE_HORIZON_FRAMES,
            "longitudinal": LONGITUDINAL_HORIZON_FRAMES,
            "lateral": LATERAL_HORIZON_FRAMES,
            "frame_dt_seconds": FRAME_DT_SECONDS,
        },
    }


def load_route_trajectory(run_dir: pathlib.Path, max_frames: int = 0) -> Optional[RouteTrajectory]:
    """读取一条 run 的全部 meta；缺 metas 目录时返回 None。"""

    metas_dir = pathlib.Path(run_dir) / "metas"
    if not metas_dir.is_dir():
        return None
    metas: Dict[int, Dict[str, Any]] = {}
    frames: List[int] = []
    for path in sorted(metas_dir.glob("*.pkl")):
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        meta = _load_meta(path)
        if meta is None:
            continue
        metas[frame_id] = meta
        frames.append(frame_id)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    if not metas:
        return None
    road_entry_lane: Dict[int, int] = {}
    for frame_id in frames:
        road_id = _int_field(metas[frame_id].get("road_id"), 0)
        if road_id not in road_entry_lane:
            road_entry_lane[road_id] = _int_field(metas[frame_id].get("lane_id"), 0)
    return RouteTrajectory(
        run_dir=pathlib.Path(run_dir),
        frames=tuple(frames),
        metas=metas,
        road_entry_lane=road_entry_lane,
    )
