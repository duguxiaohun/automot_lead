"""LeadMoT closed-loop CARLA leaderboard agent (LEAD 风格 3 摄像头 + 可选双 LiDAR)。

详见 EVAL_CARLA_PLAN.md / EVAL_CARLA_RUN.md。

推理路径
- sensors():  LEAD CARLA_LEADERBOARD2_3CAMERAS 档：3 RGB（384×384 fov=60, 横拼 1152×384）
              + use_bev=True 时启用双 LiDAR（yaw=-90 / yaw=-270, attach 在 (0,0,2.5)）
              + IMU / GPS / Speedometer
- tick():     三视角拼 1152×384；按需把双 LiDAR 转 ego frame 拼；UKF 平滑 GPS/compass；
              route_planner 算 ego-frame target_point / next_target_point
- run_step(): 每 STEP_STRIDE 个 tick 调一次 LeadOfflineMoTRunner.run_step()；
              中间 tick 沿用上一拍 pred_route / pred_future_waypoints 做 PID 跟踪；
              每 tick 同时把 stitched RGB 写 input.mp4、把 overlay 后写 debug.mp4、
              触发 demo.mp4 / grid.mp4 写入

target_point / next_target_point 语义（与离线 build_clip 对齐）
- 未来 1.5s / 3s 的位置，沿 global plan 弧长前瞻（按当前速度估算前瞻距离）
- 离线训练用真值未来位置；在线没真值，按 expected 速度沿规划路径推
- ego frame 约定 (x_forward, y_left)；inverse_conversion_2d(world_xy, gps_xy, theta)
- 低速 fallback：MIN_LOOKAHEAD_M=5m，避免静止时 tp 退化

安全兜底（SafetyMixin，对齐 mot_b2d_agent.py）
- stuck_detector → force_move creep-throttle
- parking_start 检测（前 200 帧位移 < 6m 禁用 force_move）
- parking_escape 状态机（1500 帧窗口位移 < 5m 触发，phase 1 强制大转角 + 中油门）
- 限速 35 km/h

环境变量
    LEADMOT_CKPT      [必填] LeadMoT decoder checkpoint
    LEADMOT_ROPE      默认 mrope
    SENSOR_PROFILE    "3cam"（默认且唯一支持的 LEAD 传感器档）
    STEP_STRIDE       默认 5（即 4Hz 推理）
    TP_LOOKAHEAD_S    默认 1.5
    NTP_LOOKAHEAD_S   默认 3.0
    MIN_LOOKAHEAD_M   默认 5.0（低速 fallback）
    RECORD_INPUT/RECORD_DEBUG/RECORD_DEMO/RECORD_GRID   "1"/"0"，默认 1
    SAVE_PATH         leaderboard 框架透传；agent 自己拼 <ckpt_signature>/<save_name>
"""

from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import sys
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
import carla

# ---- 把 AutoMoT / leaderboard / scenario_runner / Automot 加进 sys.path ----
_THIS_FILE = pathlib.Path(__file__).resolve()
_QWEN_LOCAL_ROOT = _THIS_FILE.parents[1]               # .../AutoMoT/qwen3vl_local
_AUTOMOT_ROOT = _THIS_FILE.parents[2]                   # .../AutoMoT
_LEADERBOARD_ROOT = _AUTOMOT_ROOT / "leaderboard"
_SCENARIO_RUNNER_ROOT = _AUTOMOT_ROOT / "scenario_runner"
_AUTOMOT_PROJECT_ROOT = _AUTOMOT_ROOT / "Automot"

for _p in [_AUTOMOT_ROOT, _LEADERBOARD_ROOT, _SCENARIO_RUNNER_ROOT, _AUTOMOT_PROJECT_ROOT]:
    sp = str(_p)
    if sp not in sys.path and _p.exists():
        sys.path.insert(0, sp)

from leaderboard.autoagents import autonomous_agent

# 复用 mot_b2d_agent.py 已就位的 helpers
from team_code.nav_planner import RoutePlanner, LateralPIDController
from team_code import automot_utils as t_u
from team_code.ukf_utils import (
    bicycle_model_forward,
    measurement_function_hx,
    state_mean,
    measurement_mean,
    residual_state_x,
    residual_measurement_h,
)
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

# 推理引擎（含 Qwen prefill + LeadBEVEncoder + LeadMoT decoder）
from team_code.mot_lead_offline_runner import LeadOfflineMoTRunner

# 本子包内部：视频/可视化/安全兜底。
# leaderboard 通过 importlib.spec_from_file_location 把 agent.py 当成顶层 module
# 加载，因此 relative import (`from .video_recorder ...`) 会失败。
# 此处用 _AUTOMOT_ROOT/qwen3vl_local/... 的 plain import 路径，
# _AUTOMOT_ROOT 已经在上面 sys.path.insert 进去了。
from qwen3vl_local.eval_carla.video_recorder import VideoRecorder
from qwen3vl_local.eval_carla.visualizer import (
    overlay_pred_on_stitched_three_cams,
    render_bev_debug,
)
from qwen3vl_local.eval_carla.safety import SafetyMixin


SAVE_PATH = os.environ.get("SAVE_PATH", None)
IS_BENCH2DRIVE = os.environ.get("IS_BENCH2DRIVE", None)
USE_UKF = True

_DEFAULT_STEP_STRIDE = int(os.environ.get("STEP_STRIDE", "5"))
_DEFAULT_SENSOR_PROFILE = os.environ.get("SENSOR_PROFILE", "3cam").lower()

# target_point / next_target_point 与离线训练 build_clip 对齐：未来 1.5s / 3.0s
# 沿 global plan 弧长的位置（按当前速度估算前瞻距离）。
_TP_LOOKAHEAD_S = float(os.environ.get("TP_LOOKAHEAD_S", "1.5"))
_NTP_LOOKAHEAD_S = float(os.environ.get("NTP_LOOKAHEAD_S", "3.0"))
# 低速 fallback：speed 低于该阈值时 tp 直接取 ego 当前位置，与训练分布一致
# （训练时车在红灯前停着，未来 1.5s 真值 ≈ 当前位置）。
# `_MIN_LOOKAHEAD_M` 现已废弃，保留 env 兼容旧脚本但不再生效。
_LOW_SPEED_TP_THRESHOLD = float(os.environ.get("LOW_SPEED_TP_M_PER_S", "1.0"))
_MIN_LOOKAHEAD_M = float(os.environ.get("MIN_LOOKAHEAD_M", "0.0"))   # legacy, unused

# RGB JPEG round-trip 模拟训练分布（B1 修复）：默认 85，与 LEAD ClosedLoopConfig 一致。
# 设 0 或 >=100 关闭模拟（用 raw CARLA RGB）。
_JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))

# 录像四开关
_RECORD_INPUT = os.environ.get("RECORD_INPUT", "1") == "1"
_RECORD_DEBUG = os.environ.get("RECORD_DEBUG", "1") == "1"
_RECORD_DEMO  = os.environ.get("RECORD_DEMO",  "1") == "1"
_RECORD_GRID  = os.environ.get("RECORD_GRID",  "1") == "1"
# BEV 顶视 debug：信息密度最高的一路。只在 use_bev=True 时有 LiDAR 显示，
# no-BEV 模型也仍写 ego/route/waypoints，但 LiDAR 散点会缺。
_RECORD_BEV_DEBUG = os.environ.get("RECORD_BEV_DEBUG", "1") == "1"
_LEADMOT_USE_EMA = os.environ.get("LEADMOT_USE_EMA", "1") != "0"


def get_entry_point():   # leaderboard 反射入口
    """leaderboard 根据这个名字实例化 agent class。"""
    return "MOTLeadAgent"


# ---------------------------------------------------------------------------
# LEAD 3CAM 标定（来自 lead/lead/common/config_base.py CARLA_LEADERBOARD2_3CAMERAS）
# ---------------------------------------------------------------------------
_LEAD_3CAM_CALIBRATION = [
    {"id": 1, "pos": [0.10, -0.35, 2.25], "rot": [0.0, 0.0, -54.5]},
    {"id": 2, "pos": [0.35,  0.00, 2.25], "rot": [0.0, 0.0,   0.0]},
    {"id": 3, "pos": [0.10,  0.35, 2.25], "rot": [0.0, 0.0,  54.5]},
]
_LEAD_3CAM_W = 1152 // 3
_LEAD_3CAM_H = 384
_LEAD_3CAM_FOV = 60

_LEAD_LIDAR_POS = [0.0, 0.0, 2.5]
_LEAD_LIDAR_YAW_1 = -90.0
_LEAD_LIDAR_YAW_2 = -270.0

# LEAD `config_base.py:270-295`：4 个 radar 传感器，front-left / front / front-right / rear。
# `save_radar_pc_as_lidar=True` + `duplicate_radar_near_ego=True` 是训练默认配置，
# 我们在线必须复现这俩，否则 BEV 输入分布偏离训练。
_LEAD_RADAR_CALIBRATION = [
    {"id": "RADAR1", "pos": [2.6, 0.0, 0.60], "rot": [0.0, 0.0, -45.0],
     "horizontal_fov": 90, "vertical_fov": 0.1},
    {"id": "RADAR2", "pos": [2.6, 0.0, 0.60], "rot": [0.0, 0.0,  45.0],
     "horizontal_fov": 90, "vertical_fov": 0.1},
    {"id": "RADAR3", "pos": [-2.6, 0.0, 0.60], "rot": [0.0, 0.0, 135.0],
     "horizontal_fov": 90, "vertical_fov": 0.1},
    {"id": "RADAR4", "pos": [-2.6, 0.0, 0.60], "rot": [0.0, 0.0, 225.0],
     "horizontal_fov": 90, "vertical_fov": 0.1},
]
_LEAD_DUPLICATE_RADAR_RADIUS = 8.0     # LEAD 默认值
_LEAD_DUPLICATE_RADAR_FACTOR = 5        # LEAD 默认值
_USE_RADAR = os.environ.get("USE_RADAR", "1") != "0"

_LEAD_EGO_EXTENT_X = 2.4508416652679443
_LEAD_EGO_EXTENT_Y = 1.0641621351242065
_LEAD_MIN_X_M = -32.0
_LEAD_MAX_X_M = 64.0
_LEAD_MIN_Y_M = -40.0
_LEAD_MAX_Y_M = 40.0
_LEAD_MIN_Z_M = -4.0
_LEAD_MAX_Z_M = 10.0
_LEAD_POINT_PRECISION_M = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_empty_lead_clip(clip_len: int) -> dict[str, Any]:
    """创建 LeadOfflineMoTRunner.run_clip 期望的最小 clip 字典。

    即便 use_bev=False，runner 的输入准备阶段仍会看到 lidar_points 字段；这里用空点云
    占位，保证 no-BEV 模型不需要真实 LiDAR 也能走同一套数据结构。
    """
    return {
        "rgb": np.zeros((clip_len, _LEAD_3CAM_H, _LEAD_3CAM_W * 3, 3), dtype=np.uint8),
        "lidar_points": [np.zeros((0, 3), dtype=np.float32) for _ in range(clip_len)],
        "pos_global": np.zeros((clip_len, 2), dtype=np.float32),
        "theta": np.zeros((clip_len,), dtype=np.float32),
        "speed": np.zeros((clip_len,), dtype=np.float32),
        "target_point": np.zeros((clip_len, 2), dtype=np.float32),
        "target_point_next": np.zeros((clip_len, 2), dtype=np.float32),
    }


def _jpeg_round_trip(rgb: np.ndarray, quality: int) -> np.ndarray:
    """模拟 LEAD `.jpg` 训练数据的 JPEG 量化（与 sensor_agent.tick line 337-345 一致）。

    LEAD 数据采集把 RGB 存成 .jpg，训练 dataloader 读 .jpg 已经走过 JPEG 解码；
    在线 CARLA 给的是 raw BGRA，比训练分布锐利。这里 encode + decode 一轮，
    把 RGB 拉到训练分布。quality 用 env `JPEG_QUALITY`（默认 85，与 LEAD 默认接近）。
    quality<=0 或 >=100 时跳过，相当于关闭该模拟。
    """
    if quality <= 0 or quality >= 100:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return rgb
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _stitch_three_cams(cam_left_bgra, cam_front_bgra, cam_right_bgra,
                       jpeg_quality: int = 0) -> np.ndarray:
    """三 BGRA -> 1152×384 RGB uint8，顺序 [左前, 前, 右前]。

    jpeg_quality > 0 时在拼接后做一轮 JPEG round-trip 模拟训练分布
    （与 LEAD `sensor_agent.tick()` line 337-345 同源）。
    """
    def _bgra_to_rgb(arr):
        """CARLA camera 输出 BGRA；模型和可视化约定用 RGB。"""
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    rgb_l = _bgra_to_rgb(cam_left_bgra)
    rgb_f = _bgra_to_rgb(cam_front_bgra)
    rgb_r = _bgra_to_rgb(cam_right_bgra)
    stitched = cv2.hconcat([rgb_l, rgb_f, rgb_r])
    if jpeg_quality > 0:
        stitched = _jpeg_round_trip(stitched, jpeg_quality)
    return stitched


def _lidar_sensor_to_ego(points_sensor: np.ndarray, sensor_yaw_deg: float,
                         sensor_pos: list[float]) -> np.ndarray:
    """CARLA LiDAR sensor frame -> ego frame；与 LEAD lidar_to_ego_coordinate 一致。"""
    if points_sensor.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = points_sensor[:, :3].astype(np.float32, copy=False)
    yaw_rad = math.radians(sensor_yaw_deg)
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rot = np.array([[c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
    pts_ego = pts @ rot.T
    pts_ego[:, 0] += float(sensor_pos[0])
    pts_ego[:, 1] += float(sensor_pos[1])
    pts_ego[:, 2] += float(sensor_pos[2])
    pts_ego[:, 2] -= float(sensor_pos[2]) / 2.0
    return pts_ego


def _radar_points_to_ego(raw_radar: np.ndarray, sensor_pos: list[float],
                          sensor_rot: list[float]) -> np.ndarray:
    """CARLA radar (alt, azim, depth, vel) -> ego frame xyz。

    数学与 LEAD `common_utils.radar_points_to_ego` 完全一致。
    CARLA radar raw 是球面坐标 (depth, azimuth, altitude, velocity)，先转笛卡尔，
    再 R_se @ pts + translation，最后做和 LiDAR 同款的 z -= sensor_pos[2]/2 偏移。
    """
    arr = np.asarray(raw_radar, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    # CARLA radar Detection: (velocity, azimuth, altitude, depth)，但 leaderboard
    # 透传给 input_data 的格式是 (alt, azim, depth, vel)，与 LEAD radar_points_to_ego
    # 拿到的 raw 一致：第 0 列=alt, 1=azim, 2=depth, 3=vel。
    alt = arr[:, 0]
    az = arr[:, 1]
    r = arr[:, 2]
    x = r * np.cos(az) * np.cos(alt)
    y = r * np.sin(az) * np.cos(alt)
    z = r * np.sin(alt)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    # 与 LEAD 一致：用 euler Rz @ Ry @ Rx
    roll_r = math.radians(sensor_rot[0])
    pitch_r = math.radians(sensor_rot[1])
    yaw_r = math.radians(sensor_rot[2])
    cr, sr = math.cos(roll_r), math.sin(roll_r)
    cp, sp = math.cos(pitch_r), math.sin(pitch_r)
    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    pts_ego = (R @ pts.T).T + np.asarray(sensor_pos, dtype=np.float32).reshape(1, 3)
    pts_ego[:, 2] -= float(sensor_pos[2]) / 2.0
    return pts_ego.astype(np.float32, copy=False)


def _remove_ground_lsq(points_ego: np.ndarray,
                        z_floor: float = -1.4,
                        plane_inlier_thresh: float = 0.2) -> np.ndarray:
    """轻量去地面：z 硬阈值 + LSQ 单平面拟合移除残留地面 inliers。

    LEAD 训练用 `ransac.remove_ground`（n_segments=8 + numba）做径向分段拟合；
    那个实现依赖 numba 重依赖，按 PROJECT_CONTEXT.md §1 不引入。
    这里用：
    1. 先按 z 阈值过滤明显的地面点（ego frame 地面 z ≈ -1.5m，因为
       `lidar_to_ego_coordinate` 末尾 `z -= pos.z/2 = -1.25` 让 ego 原点偏车顶）。
    2. 再用 LSQ 拟一个 z = a*x + b*y + c 的平面（候选低点子集），
       距离平面 < inlier_thresh 的点视为地面。
    这是近似实现，与 LEAD RANSAC 在远距离斜坡上会有差异；不影响近场关键点。
    """
    if points_ego.shape[0] == 0:
        return points_ego

    # 1. z 阈值粗筛：z < z_floor - 0.2 的肯定是地面，删
    z = points_ego[:, 2]
    hard_ground = z < (z_floor - 0.2)
    pts = points_ego[~hard_ground]
    if pts.shape[0] < 50:
        return pts

    # 2. 取低层候选点（z 在 [z_floor-0.2, z_floor+0.4]）拟平面
    cand_mask = (pts[:, 2] >= z_floor - 0.2) & (pts[:, 2] <= z_floor + 0.4)
    cand = pts[cand_mask]
    if cand.shape[0] < 50:
        return pts
    # LSQ: A @ [a,b,c] = z；A = [x, y, 1]
    A = np.column_stack([cand[:, 0], cand[:, 1], np.ones(cand.shape[0], dtype=np.float32)])
    try:
        coef, *_ = np.linalg.lstsq(A, cand[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return pts
    a, b, c = coef
    # 距离 = |a*x + b*y + c - z| / sqrt(a^2 + b^2 + 1)
    norm = float(math.sqrt(a * a + b * b + 1.0))
    dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c - pts[:, 2]) / max(norm, 1e-6)
    ground_mask = (dist < plane_inlier_thresh) & (pts[:, 2] < z_floor + 0.4)
    return pts[~ground_mask]


_LIDAR_REMOVE_GROUND = os.environ.get("LIDAR_REMOVE_GROUND", "1") != "0"
_LIDAR_GROUND_Z = float(os.environ.get("LIDAR_GROUND_Z", "-1.4"))


def _preprocess_lidar_like_lead(points_ego: np.ndarray) -> np.ndarray:
    """LEAD 风格的 LiDAR 过滤。

    流程对齐 `base_agent.tick()` line 134-227：
    1. 去 ego box（abs(x)>extent_x AND abs(y)>extent_y）
    2. BEV 范围裁切（XYZ in [min_*, max_*]）
    3. 去地面：LEAD 用 ransac.remove_ground；我们用轻量 LSQ 平面拟合
       （env `LIDAR_REMOVE_GROUND=0` 可关）
    4. 0.1m XYZ 量化（与 LEAD `point_precision_*` 等价的简化版）

    Radar 拼接见 `_tick_radar`；本函数只处理纯 LiDAR。
    """
    pts = np.asarray(points_ego, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = pts.reshape(-1, pts.shape[-1])[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    outside_ego = (np.abs(x) > _LEAD_EGO_EXTENT_X) & (np.abs(y) > _LEAD_EGO_EXTENT_Y)
    inside_bev = (
        (_LEAD_MIN_X_M <= x) & (x <= _LEAD_MAX_X_M)
        & (_LEAD_MIN_Y_M <= y) & (y <= _LEAD_MAX_Y_M)
        & (_LEAD_MIN_Z_M <= z) & (z <= _LEAD_MAX_Z_M)
    )
    pts = pts[outside_ego & inside_bev]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if _LIDAR_REMOVE_GROUND:
        pts = _remove_ground_lsq(pts, z_floor=_LIDAR_GROUND_Z)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

    pts = np.round(pts / _LEAD_POINT_PRECISION_M) * _LEAD_POINT_PRECISION_M
    return pts.astype(np.float32, copy=False)


def _resolve_leadmot_checkpoint(path_like: str) -> pathlib.Path:
    """Accept either a checkpoint file or a LeadMoT output directory."""
    root = pathlib.Path(path_like).expanduser().resolve()
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"LEADMOT_CKPT path not found: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"LEADMOT_CKPT is neither file nor directory: {root}")

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
        latest_dir = root / "latest"
        if latest_dir.is_dir():
            pools.extend(p for p in latest_dir.glob(pattern) if p.is_file())
    if pools:
        return max(pools, key=lambda p: p.stat().st_mtime).resolve()
    raise FileNotFoundError(
        f"No LeadMoT checkpoint found under {root}; expected best.pt/latest.pt or checkpoint-*.pt"
    )


def _ckpt_signature(ckpt_path: pathlib.Path, use_bev: bool, use_ema: bool) -> str:
    """把模型路径 + use_bev + raw/EMA 编成输出目录名，避免不同实验互相覆盖。"""
    parent = ckpt_path.parent.name or "unknown"
    stem = ckpt_path.stem or "ckpt"
    return f"{parent}__{stem}__bev{1 if use_bev else 0}__ema{1 if use_ema else 0}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class MOTLeadAgent(SafetyMixin, autonomous_agent.AutonomousAgent):
    """LEAD 风格实时 agent + LeadMoT decoder + 4 路视频录制 + safety mixin。"""

    # ============================================================
    # leaderboard hooks
    # ============================================================
    def setup(self, path_to_conf_file: str):
        """leaderboard 在每条 route 开始前调用。

        这里完成四件事：
        1. 解析 route/save_name；
        2. 显式加载 LeadMoT checkpoint，并从 checkpoint 配置读取 use_bev；
        3. 根据 use_bev 决定是否需要 LiDAR 输入；
        4. 初始化视频、PID、UKF、滑动窗口和安全兜底状态。
        """
        self.track = autonomous_agent.Track.SENSORS

        if IS_BENCH2DRIVE:
            parts = path_to_conf_file.split("+")
            self.save_name = parts[-1] if len(parts) > 1 else "run"
            self.config_path = parts[0]
        else:
            now = datetime.datetime.now()
            self.config_path = path_to_conf_file
            self.save_name = "_".join("%02d" % x for x in (
                now.month, now.day, now.hour, now.minute, now.second
            ))

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.step_stride = max(1, _DEFAULT_STEP_STRIDE)
        self.sensor_profile = _DEFAULT_SENSOR_PROFILE
        if self.sensor_profile != "3cam":
            raise ValueError(
                "LeadMoT closed-loop eval only supports SENSOR_PROFILE=3cam. "
                "The AutoMoT 1cam profile has a different RGB shape and is not "
                "compatible with the LEAD-trained action model."
            )

        # ---- 推理引擎 ----
        ckpt_env = os.environ.get("LEADMOT_CKPT")
        if not ckpt_env:
            raise RuntimeError("MOTLeadAgent requires env var LEADMOT_CKPT")
        self.leadmot_ckpt_path = _resolve_leadmot_checkpoint(ckpt_env)
        if not self.leadmot_ckpt_path.exists():
            raise FileNotFoundError(f"LEADMOT_CKPT not found: {self.leadmot_ckpt_path}")

        rope_type = os.environ.get("LEADMOT_ROPE", "mrope")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[MOTLeadAgent] building LeadOfflineMoTRunner on {device}")
        print(f"[MOTLeadAgent]   ckpt = {self.leadmot_ckpt_path}")
        self.runner = LeadOfflineMoTRunner(
            device=device,
            leadmot_ckpt_path=str(self.leadmot_ckpt_path),
            leadmot_rope_type=rope_type,
            leadmot_use_ema=_LEADMOT_USE_EMA,
        )
        self.runner._ensure_leadmot_qwen_engine()
        self.runner._ensure_leadmot_decoder()
        self.use_bev = bool(self.runner.leadmot_config.use_bev)
        # no-BEV 模型不声明、不读取 LiDAR；BEV 模型才请求 LEAD 双 LiDAR。
        # 注意 runner 内部可能仍构造 BEV encoder，但 forward 会由 decoder_config.use_bev 控制。
        self.need_lidar = self.use_bev
        # 训练时 save_radar_pc_as_lidar=True 默认就把 radar 拼进 LiDAR；
        # 只有 BEV 模型用 LiDAR，所以 use_radar 也只在 use_bev 时声明传感器。
        # env `USE_RADAR=0` 可显式关闭做对照实验。
        self.use_radar = bool(self.use_bev and _USE_RADAR)
        print(f"[MOTLeadAgent] decoder.use_bev = {self.use_bev}, use_radar = {self.use_radar}")
        self.clip_len = 4

        # ---- 输出目录 <eval_base>/<signature>/route<route_id>/ ----
        signature = _ckpt_signature(self.leadmot_ckpt_path, self.use_bev, _LEADMOT_USE_EMA)
        if SAVE_PATH is not None:
            base = pathlib.Path(SAVE_PATH)
        else:
            base = _AUTOMOT_ROOT / "outputs" / "closed_loop_eval"
        self.signature_path = base / signature
        self.save_path = self.signature_path / self.save_name
        self.save_path.mkdir(parents=True, exist_ok=True)
        (self.save_path / "meta").mkdir(parents=True, exist_ok=True)
        (self.save_path / "logs").mkdir(parents=True, exist_ok=True)
        print(f"[MOTLeadAgent] save_path = {self.save_path}")

        # 写一次 config.json，方便外部根据目录回溯模型 / 是否 BEV / 传感器档
        meta = {
            "ckpt_path": str(self.leadmot_ckpt_path),
            "ckpt_input": ckpt_env,
            "use_bev": self.use_bev,
            "requires_lidar": self.need_lidar,
            "rope_type": rope_type,
            "use_ema": _LEADMOT_USE_EMA,
            "sensor_profile": self.sensor_profile,
            "step_stride": self.step_stride,
            "history_seconds": (self.clip_len - 1) * self.step_stride / 20.0,
            "waypoint_dt_seconds": 0.25,
            "lidar_preprocess": {
                "active": self.need_lidar,
                "remove_ego_box": True,
                "inside_bev_only": True,
                "quantize_m": _LEAD_POINT_PRECISION_M,
                "range": {
                    "x": [_LEAD_MIN_X_M, _LEAD_MAX_X_M],
                    "y": [_LEAD_MIN_Y_M, _LEAD_MAX_Y_M],
                    "z": [_LEAD_MIN_Z_M, _LEAD_MAX_Z_M],
                },
            },
            "save_name": self.save_name,
            "config_path": self.config_path,
            "record": {
                "input": _RECORD_INPUT, "debug": _RECORD_DEBUG,
                "demo": _RECORD_DEMO, "grid": _RECORD_GRID,
                "bev_debug": _RECORD_BEV_DEBUG,
            },
            "jpeg_quality": _JPEG_QUALITY,
            "lidar_remove_ground": _LIDAR_REMOVE_GROUND,
            "lidar_ground_z": _LIDAR_GROUND_Z,
            "use_radar": getattr(self, "use_radar", False),
            "low_speed_tp_threshold_m_per_s": _LOW_SPEED_TP_THRESHOLD,
            "warmup_policy": "lead_style_left_pad_clip",
        }
        with open(self.signature_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(self.save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # ---- 视频录制 ----
        self.video = VideoRecorder(
            save_dir=self.save_path,
            record_input=_RECORD_INPUT,
            record_debug=_RECORD_DEBUG,
            record_demo=_RECORD_DEMO,
            record_grid=_RECORD_GRID,
            record_bev_debug=_RECORD_BEV_DEBUG,
            produce_frame_frequency=1,   # 与 CARLA tick 同频；如需稀采可改
        )

        # ---- 控制器 ----
        self.turn_controller = LateralPIDController(
            inference_mode=True, k_p=3.118, speed_offset=1.195, default_lookahead=24,
        )
        self.speed_controller = t_u.PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20)
        self.clip_throttle = 1.0
        self.brake_speed = 0.4
        self.brake_ratio = 1.1

        # ---- 滑动 4 帧 lead_clip 输入 ----
        self.clip_rgb = deque(maxlen=self.clip_len)
        self.clip_pos = deque(maxlen=self.clip_len)
        self.clip_theta = deque(maxlen=self.clip_len)
        self.clip_speed = deque(maxlen=self.clip_len)
        self.clip_tp = deque(maxlen=self.clip_len)
        self.clip_ntp = deque(maxlen=self.clip_len)
        self.clip_steps = deque(maxlen=self.clip_len)
        # use_bev=True 时维护最近 step_stride 个 tick 的 ego-frame LiDAR 点
        self.lidar_sweep_buffer = deque(maxlen=self.step_stride) if self.need_lidar else deque(maxlen=0)

        self.last_pred_route: np.ndarray | None = None
        self.last_pred_waypoints: np.ndarray | None = None
        self.last_inference_step = -1

        # ---- 控制初值 ----
        control = carla.VehicleControl()
        self.prev_control = control
        self.control = control
        self.carla_frame_rate = 1.0 / 20.0

        # ---- UKF ----
        if USE_UKF:
            self.points = MerweScaledSigmaPoints(
                n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x
            )
            self.ukf = UKF(
                dim_x=4, dim_z=4,
                fx=bicycle_model_forward, hx=measurement_function_hx,
                dt=self.carla_frame_rate, points=self.points,
                x_mean_fn=state_mean, z_mean_fn=measurement_mean,
                residual_x=residual_state_x, residual_z=residual_measurement_h,
            )
            self.ukf.P = np.diag([0.5, 0.5, 1e-6, 1e-6])
            self.ukf.R = np.diag([0.5, 0.5, 1e-15, 1e-15])
            self.ukf.Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
            self.filter_initialized = False
            self.state_log = deque(maxlen=20)

        self.previous_compass = None
        self.warmup_steps = (self.clip_len - 1) * self.step_stride
        self.lat_ref, self.lon_ref = 42.0, 2.0

        # 安全兜底（stuck_helper / parking_start / parking_escape / 限速 35km/h）
        self.init_safety_state()

        # 启动 banner：远程 tail -F 时一眼能确认 agent 已就绪
        print(
            f"[MOTLeadAgent] READY route={self.save_name} "
            f"sig={signature} use_bev={self.use_bev} use_radar={self.use_radar} "
            f"clip_len={self.clip_len} step_stride={self.step_stride} "
            f"tp_lookahead={_TP_LOOKAHEAD_S}s ntp_lookahead={_NTP_LOOKAHEAD_S}s "
            f"jpeg_q={_JPEG_QUALITY} ground_removal={_LIDAR_REMOVE_GROUND}",
            flush=True,
        )

    # ============================================================
    # sensors
    # ============================================================
    def sensors(self):
        """声明 CARLA leaderboard 传感器。

        RGB 始终使用 LEAD 3cam 标定；LiDAR 只在 checkpoint.use_bev=True 时启用。
        这样 no-BEV 动作模型不会产生未使用输入，也避免实时 input_data 缺键导致崩溃。
        """
        sensors: list[dict] = []
        cam_ids = ["CAM_LEFT", "CAM_FRONT", "CAM_RIGHT"]
        for cam_id, cfg in zip(cam_ids, _LEAD_3CAM_CALIBRATION):
            sensors.append({
                "type": "sensor.camera.rgb",
                "x": cfg["pos"][0], "y": cfg["pos"][1], "z": cfg["pos"][2],
                "roll": cfg["rot"][0], "pitch": cfg["rot"][1], "yaw": cfg["rot"][2],
                "width": _LEAD_3CAM_W, "height": _LEAD_3CAM_H,
                "fov": _LEAD_3CAM_FOV, "id": cam_id,
            })

        if getattr(self, "need_lidar", True):
            sensors.append({
                "type": "sensor.lidar.ray_cast",
                "x": _LEAD_LIDAR_POS[0], "y": _LEAD_LIDAR_POS[1], "z": _LEAD_LIDAR_POS[2],
                "roll": 0.0, "pitch": 0.0, "yaw": _LEAD_LIDAR_YAW_1, "id": "LIDAR1",
            })
            sensors.append({
                "type": "sensor.lidar.ray_cast",
                "x": _LEAD_LIDAR_POS[0], "y": _LEAD_LIDAR_POS[1], "z": _LEAD_LIDAR_POS[2],
                "roll": 0.0, "pitch": 0.0, "yaw": _LEAD_LIDAR_YAW_2, "id": "LIDAR2",
            })
            # 4 个 radar：LEAD 训练默认 `save_radar_pc_as_lidar=True` 把 radar 拼到 LiDAR。
            # 不接 radar 会让 BEV 远距离/盲区点密度偏少 → 偏离训练分布。
            if getattr(self, "use_radar", False):
                for rcfg in _LEAD_RADAR_CALIBRATION:
                    sensors.append({
                        "type": "sensor.other.radar",
                        "x": rcfg["pos"][0], "y": rcfg["pos"][1], "z": rcfg["pos"][2],
                        "roll": rcfg["rot"][0], "pitch": rcfg["rot"][1],
                        "yaw": rcfg["rot"][2],
                        "horizontal_fov": rcfg["horizontal_fov"],
                        "vertical_fov": rcfg["vertical_fov"],
                        "id": rcfg["id"],
                    })
        sensors.extend([
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "IMU"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "GPS"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ])
        return sensors

    # ============================================================
    # route planner + demo camera setup（首帧）
    # ============================================================
    def _init_first_frame(self):
        """首帧初始化 route planner 与 demo cameras。

        leaderboard 的 global plan 已由框架注入；这里把它交给 AutoMoT RoutePlanner，
        后续 target_point/next_target_point 都沿这个 CARLA global route 做 lookahead。
        """
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            import xml.etree.ElementTree as ET
            world_map = CarlaDataProvider.get_map()
            xodr = world_map.to_opendrive()
            tree = ET.ElementTree(ET.fromstring(xodr))
            for opendrive in tree.iter("OpenDRIVE"):
                for header in opendrive.iter("header"):
                    for georef in header.iter("geoReference"):
                        if georef.text:
                            for item in georef.text.split(" "):
                                if "+lat_0" in item:
                                    self.lat_ref = float(item.split("=")[1])
                                if "+lon_0" in item:
                                    self.lon_ref = float(item.split("=")[1])
        except Exception as e:
            print(f"[MOTLeadAgent] OpenDRIVE parse failed ({e}); using default lat/lon")

        self._route_planner = RoutePlanner(7.5, 50.0, self.lat_ref, self.lon_ref)
        self._route_planner.set_route(self._global_plan_world_coord, gps=False)

        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        # 现在 world / ego vehicle 已就位，spawn demo cameras
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            # ego vehicle 在 leaderboard 框架里通常是 hero actor
            ego = None
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.attributes.get("role_name") == "hero":
                    ego = actor
                    break
            if ego is not None:
                self.video.setup_demo_cameras(world, ego)
            else:
                print("[MOTLeadAgent] hero vehicle not found; demo cameras skipped")
        except Exception as e:
            print(f"[MOTLeadAgent] demo cameras setup failed: {e}")

        self.initialized = True

    # ============================================================
    # tick
    # ============================================================
    def tick(self, input_data) -> dict[str, Any]:
        """把 leaderboard 原始 input_data 转成模型/控制需要的轻量 tick_data。

        输入坐标处理：
        - RGB: 三路 BGRA camera -> RGB -> 横拼 1152x384；
        - LiDAR: use_bev=True 时 sensor frame -> ego frame，再做 LEAD 轻量过滤；
        - GPS/compass/speed: 走 RoutePlanner + UKF 平滑；
        - target_point: global route lookahead world 点 -> ego frame (x_forward, y_left)。
        """
        self.step += 1

        rgb_stitched = _stitch_three_cams(
            input_data["CAM_LEFT"][1],
            input_data["CAM_FRONT"][1],
            input_data["CAM_RIGHT"][1],
            jpeg_quality=_JPEG_QUALITY,
        )   # (384, 1152, 3) RGB uint8（已做 JPEG round-trip 对齐训练分布）

        if self.need_lidar:
            # 两个 LiDAR 的外参和 LEAD 数据采集保持一致；no-BEV 时不会访问这些 key。
            lidar_ego_1 = _lidar_sensor_to_ego(
                input_data["LIDAR1"][1], _LEAD_LIDAR_YAW_1, _LEAD_LIDAR_POS
            )
            if "LIDAR2" in input_data:
                lidar_ego_2 = _lidar_sensor_to_ego(
                    input_data["LIDAR2"][1], _LEAD_LIDAR_YAW_2, _LEAD_LIDAR_POS
                )
                lidar_ego = np.concatenate([lidar_ego_1, lidar_ego_2], axis=0)
            else:
                lidar_ego = lidar_ego_1

            # 4 个 radar 转 ego 后拼到 LiDAR，与 LEAD `base_agent.tick()` line 177-193 同源。
            # 拼完后做 near-ego duplicate（与 duplicate_radar_near_ego=True 一致）。
            if self.use_radar:
                radar_lists: list[np.ndarray] = []
                for rcfg in _LEAD_RADAR_CALIBRATION:
                    raw = input_data.get(rcfg["id"], None)
                    if raw is None:
                        continue
                    pts = _radar_points_to_ego(raw[1], rcfg["pos"], rcfg["rot"])
                    if pts.size > 0:
                        radar_lists.append(pts)
                if radar_lists:
                    radar_all = np.concatenate(radar_lists, axis=0)
                    # 近车 radar 复制 factor 次：放大近距离 radar 信号在 BEV 栅格里的权重，
                    # 与 LEAD `base_agent.tick()` line 182-193 一致。
                    near_mask = (
                        np.linalg.norm(radar_all[:, :2], axis=1)
                        < _LEAD_DUPLICATE_RADAR_RADIUS
                    )
                    radar_near = radar_all[near_mask]
                    if radar_near.size > 0:
                        radar_dup = np.concatenate(
                            [radar_near] * _LEAD_DUPLICATE_RADAR_FACTOR, axis=0
                        )
                        radar_all = np.concatenate([radar_all, radar_dup], axis=0)
                    lidar_ego = np.concatenate([lidar_ego, radar_all], axis=0)

            lidar_ego = _preprocess_lidar_like_lead(lidar_ego)
        else:
            lidar_ego = np.zeros((0, 3), dtype=np.float32)

        gps_full = input_data["GPS"][1]
        compass_raw = input_data["IMU"][1][-1]
        if math.isnan(float(compass_raw)):
            # CARLA IMU 偶发 NaN；AutoMoT helper 也有同样兜底，这里提前处理便于 unwrap。
            compass_raw = 0.0
        compass = t_u.preprocess_compass(compass_raw)
        if self.previous_compass is not None:
            compass = float(np.unwrap([self.previous_compass, compass])[1])
        self.previous_compass = compass
        speed_raw = float(input_data["SPEED"][1]["speed"])

        gps_pos = self._route_planner.convert_gps_to_carla(gps_full)
        gps_xy = gps_pos[:2].copy()

        if USE_UKF:
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_xy[0], gps_xy[1], compass, speed_raw])
                self.filter_initialized = True
            self.ukf.predict(steer=float(self.control.steer),
                             throttle=float(self.control.throttle),
                             brake=float(self.control.brake))
            self.ukf.update(np.array([gps_xy[0], gps_xy[1], compass, speed_raw]))
            gps_xy = self.ukf.x[:2].copy()
            compass_filt = float(self.ukf.x[2])
            speed_filt = float(self.ukf.x[3])
        else:
            compass_filt = compass
            speed_filt = speed_raw

        # 让 RoutePlanner 推进队列：route 内已访问的路点会被 popleft
        # 这里只用 run_step 副作用更新队列；tp/ntp 改走时间 lookahead
        wp_route = self._route_planner.run_step(np.append(gps_xy, gps_pos[2]))

        # tp/ntp 与离线训练 build_clip 对齐：未来 1.5s / 3s 沿剩余 route 弧长的位置
        # 弧长距离 = max(speed * lookahead_s, MIN_LOOKAHEAD_M)，避免低速 tp 退化
        tp_world, far_cmd = self._lookahead_world_point(
            speed_filt, _TP_LOOKAHEAD_S, gps_xy, compass_filt
        )
        ntp_world, _ = self._lookahead_world_point(
            speed_filt, _NTP_LOOKAHEAD_S, gps_xy, compass_filt
        )

        if (np.asarray(tp_world[:2]) != np.asarray(self.target_point_prev[:2])).any():
            self.target_point_prev = tp_world
            self.commands.append(far_cmd.value)
        next_command = self.commands[-2]

        # world -> ego frame；在线没有未来真值，只能用 route lookahead 近似离线 tp/ntp 语义。
        # inverse_conversion_2d(world_xy, current_pos_xy, current_theta) -> ego_xy
        # ego frame 约定 (x_forward, y_left)；与 LeadMoT 训练分布一致。
        ego_tp = t_u.inverse_conversion_2d(np.asarray(tp_world[:2]), gps_xy, compass_filt)
        ego_ntp = t_u.inverse_conversion_2d(np.asarray(ntp_world[:2]), gps_xy, compass_filt)

        return {
            "rgb_stitched": rgb_stitched,
            "lidar_ego": lidar_ego,
            "gps_world": gps_xy.astype(np.float32),
            "compass": float(compass_filt),
            "speed": speed_filt,
            "target_point_ego": ego_tp.astype(np.float32),
            "next_target_point_ego": ego_ntp.astype(np.float32),
            "next_command": int(next_command),
        }

    def _lookahead_world_point(self, speed: float, lookahead_s: float,
                                gps_xy: np.ndarray, compass: float):
        """沿剩余 global route 弧长找未来 lookahead_s 秒位置。

        逻辑：
        1. 低速 fallback：speed < `_LOW_SPEED_TP_THRESHOLD` 时直接返回 ego 当前
           位置 → ego frame 下 tp ≈ (0, 0)。**与离线对齐**：训练时车在红灯前停着，
           未来 1.5s 真值位置 ≈ 当前位置，tp 就是 ~0。如果在线退化成"沿弧长前推 5m"，
           tp ≈ (5, 0) 会让模型误以为该往前走。
        2. 否则目标弧长 = `speed * lookahead_s`（移除原来的 MIN_LOOKAHEAD_M=5 兜底）
        3. 从 ego 当前位置开始，沿 self._route_planner.route deque 顺序累加每个
           路点之间的距离，直到累加值 >= 目标弧长。
        4. 在该段内按比例线性插值，返回 (world_xy, RoadOption)
        5. route 走完时返回最后一个路点；route 为空时沿当前 compass 直推。

        与离线 build_clip 用未来真值位置的语义近似一致：训练用真值未来 1.5s
        位置；推理无真值，只能按 expected 速度沿规划路径推。这是闭环里
        能拿到的最接近训练分布的 navigation hint。
        """
        from agents.navigation.local_planner import RoadOption

        # 低速 fallback：车基本不动时，tp = ego 当前位置（→ ego frame ≈ (0,0)）。
        if speed < _LOW_SPEED_TP_THRESHOLD:
            route_for_cmd = list(self._route_planner.route)
            far_cmd = route_for_cmd[0][1] if route_for_cmd else RoadOption.LANEFOLLOW
            return np.asarray(gps_xy, dtype=np.float32), far_cmd

        # 正常前推：直接 speed * lookahead，不再上 5m 兜底（兜底由上面 low-speed 分支接管）。
        target_dist = float(speed * lookahead_s)

        route = list(self._route_planner.route)  # deque -> snapshot list
        if len(route) == 0:
            unit = np.array([math.cos(compass), math.sin(compass)], dtype=np.float64)
            return (np.asarray(gps_xy) + unit * target_dist).astype(np.float32), \
                   RoadOption.LANEFOLLOW

        prev_xy = np.asarray(gps_xy, dtype=np.float64)
        accum = 0.0
        last_cmd = route[-1][1]
        for pos, cmd in route:
            pos_xy = np.asarray(pos[:2], dtype=np.float64)
            seg = pos_xy - prev_xy
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1e-6 and accum + seg_len >= target_dist:
                t = (target_dist - accum) / seg_len
                interp = prev_xy + t * seg
                return interp.astype(np.float32), cmd
            accum += seg_len
            prev_xy = pos_xy
            last_cmd = cmd
        # 路径走完仍未达到目标距离 → 返回最后一个路点
        return np.asarray(route[-1][0][:2], dtype=np.float32), last_cmd

    # ============================================================
    # run_step
    # ============================================================
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        """leaderboard 每个 CARLA tick 调用的主循环。

        20Hz tick 中只有 `step % STEP_STRIDE == 0` 的采样点会进入模型历史；
        其它 tick 复用最近一次模型输出做 PID。warmup 阶段等待真实历史帧，不复制首帧。
        """
        if not self.initialized:
            self._init_first_frame()

        tick_data = self.tick(input_data)
        self.video.update_step(self.step)

        # 写 lead_clip 滑窗
        if self.need_lidar:
            self.lidar_sweep_buffer.append({
                "points": tick_data["lidar_ego"],
                "pos": tick_data["gps_world"].copy(),
                "theta": np.float32(tick_data["compass"]),
            })

        is_sample_tick = self.step % self.step_stride == 0
        if is_sample_tick:
            self.clip_rgb.append(tick_data["rgb_stitched"])
            self.clip_pos.append(tick_data["gps_world"])
            self.clip_theta.append(np.float32(tick_data["compass"]))
            self.clip_speed.append(np.float32(tick_data["speed"]))
            self.clip_tp.append(tick_data["target_point_ego"])
            self.clip_ntp.append(tick_data["next_target_point_ego"])
            self.clip_steps.append(int(self.step))

        # LEAD 风格 warmup：不等历史，第一个 4Hz 采样点就推理；
        # 历史不足时复制 clip 首元素 left-pad 到 clip_len。LeadMoT dataset 的
        # build_clip 在 anchor 历史不足时也是复制 frame 0（line 1808-1815），
        # 模型分布上见过这种 pad 输入。
        history_ready = len(self.clip_rgb) >= 1
        lidar_ready = (not self.use_bev) or (len(self.lidar_sweep_buffer) >= 1)
        is_inference_tick = history_ready and is_sample_tick and lidar_ready

        # 安全状态：stuck 计数、parking_start 检测、deadlock 检测
        self.update_safety_state(tick_data, history_ready)

        if not history_ready:
            # 还没拿到第一个 4Hz 采样点（即首帧前 5 个 tick 内）：LEAD 第一帧也是 brake=1。
            self._record_videos(tick_data["rgb_stitched"], None)
            ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            ctrl = self.apply_safety_overrides(ctrl, tick_data)
            self.prev_control = ctrl
            self.control = ctrl
            return ctrl

        if is_inference_tick:
            self._run_inference_tick()

        # 录像（每 tick 都写一帧，debug 用最近一次预测画）
        self._record_videos(tick_data["rgb_stitched"], tick_data)

        ctrl = self._waypoints_to_control(tick_data)
        self.prev_control = ctrl
        self.control = ctrl

        # 周期性进度总览（每 20 tick=1s 一行）。对齐 LEAD sensor_agent / AutoMoT mot_b2d_agent
        # 的运行时输出风格，让远程 tail 时能直接看到速度 / 控制 / 推理状态。
        if self.step % 20 == 0:
            speed_kmh = float(tick_data["speed"]) * 3.6
            tp_ego = tick_data["target_point_ego"]
            tp_dist = float(np.linalg.norm(tp_ego))
            wp_avg = (
                float(np.linalg.norm(self.last_pred_waypoints[-1]))
                if self.last_pred_waypoints is not None else 0.0
            )
            lidar_n = (
                int(tick_data["lidar_ego"].shape[0]) if self.need_lidar else 0
            )
            flags = []
            if self.parking_escape_active:
                flags.append(f"escape@phase{self.parking_escape_phase}")
            elif self.force_move > 0:
                flags.append(f"force_move={self.force_move}")
            if self.stuck_detector > 100:
                flags.append(f"stuck={self.stuck_detector}")
            flag_str = " " + " ".join(flags) if flags else ""
            print(
                f"[MOTLeadAgent] tick={self.step:5d} "
                f"speed={speed_kmh:5.1f}km/h "
                f"ctrl=(thr={ctrl.throttle:.2f},str={ctrl.steer:+.2f},brk={ctrl.brake:.2f}) "
                f"tp={tp_dist:5.1f}m wp_end={wp_avg:5.1f}m "
                f"lidar_pts={lidar_n}{flag_str}",
                flush=True,   # 让 tail -F 立刻看到
            )

        return ctrl

    # ----- 推理 -----
    def _run_inference_tick(self):
        """在 4Hz 采样点组装 lead_clip 并调用 LeadOfflineMoTRunner。

        BEV 模型会把最近 step_stride 个 20Hz LiDAR sweep 对齐到当前 anchor ego frame；
        no-BEV 模型则传空点云占位，只让 Qwen prefill + LeadMoT decoder 消费 RGB/导航状态。
        """
        anchor_pos = np.asarray(self.clip_pos[-1], dtype=np.float32).reshape(-1)[:2]
        anchor_theta = float(np.asarray(self.clip_theta[-1]).reshape(-1)[0])
        if self.need_lidar:
            aligned_sweeps = []
            for sweep in self.lidar_sweep_buffer:
                pts = np.asarray(sweep["points"], dtype=np.float32)
                if pts.size == 0:
                    continue
                # runner 内部的对齐函数与离线 build_clip 共用，避免在线/离线 LiDAR 融合公式分叉。
                aligned_sweeps.append(
                    self.runner._align_lidar_points_to_anchor(
                        pts,
                        np.asarray(sweep["pos"], dtype=np.float32).reshape(-1)[:2],
                        float(np.asarray(sweep["theta"]).reshape(-1)[0]),
                        anchor_pos,
                        anchor_theta,
                    )
                )
            if aligned_sweeps:
                anchor_lidar = np.concatenate(aligned_sweeps, axis=0).astype(np.float32)
            else:
                anchor_lidar = np.zeros((0, 3), dtype=np.float32)
            anchor_lidar = _preprocess_lidar_like_lead(anchor_lidar)
        else:
            anchor_lidar = np.zeros((0, 3), dtype=np.float32)

        lead_clip = _make_empty_lead_clip(self.clip_len)
        # LEAD 风格 left-pad：clip 历史不足时复制 frame 0（与 build_clip line 1808-1815 一致）
        # 假设 deque 顺序：index 0 = 最早，index -1 = 最新（anchor）。
        # 当 len(clip)<clip_len 时，前 (clip_len - len) 位补 clip[0]。
        n_have = len(self.clip_rgb)
        n_pad = max(0, self.clip_len - n_have)
        for i in range(self.clip_len):
            src = max(0, i - n_pad)  # 前 n_pad 个都用 clip[0]
            # clip_rgb/pos/theta/... 都只在 4Hz sample tick 入队，因此这里的 4 帧就是训练时
            # 0.25s 间隔的历史，而不是 20Hz 连续帧。
            lead_clip["rgb"][i] = self.clip_rgb[src]
            lead_clip["pos_global"][i] = self.clip_pos[src]
            lead_clip["theta"][i] = self.clip_theta[src]
            lead_clip["speed"][i] = self.clip_speed[src]
            lead_clip["target_point"][i] = self.clip_tp[src]
            lead_clip["target_point_next"][i] = self.clip_ntp[src]
            lead_clip["lidar_points"][i] = (
                anchor_lidar if i == self.clip_len - 1 else np.zeros((0, 3), dtype=np.float32)
            )
        if n_pad > 0:
            print(f"[MOTLeadAgent] step={self.step} warmup-pad: copied frame 0 "
                  f"x{n_pad} to fill clip (LEAD-style, dataset has seen this)")

        t0 = time.time()
        outs = self.runner.run_clip(
            lead_clip,
            rgb_frame_step=1, rgb_frame_count=4,
            bev_frame_step=1, bev_frame_count=1,
        )
        dt = time.time() - t0
        out = outs[-1]
        self.last_pred_route = out["leadmot_route"].numpy()[0]
        self.last_pred_waypoints = out["leadmot_future_waypoints"].numpy()[0]
        self.last_inference_step = self.step

        meta_file = self.save_path / "meta" / f"{self.step:06d}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            # 每次真正调用模型才写 meta，方便把视频帧和推理输入/输出对齐排查。
            json.dump({
                "step": int(self.step),
                "clip_steps": list(self.clip_steps),
                "timestamp_wallclock": time.time(),
                "infer_seconds": dt,
                "input_stats": {
                    "rgb_shape": list(lead_clip["rgb"].shape),
                    "anchor_lidar_points": int(anchor_lidar.shape[0]),
                    "lidar_sweeps": len(self.lidar_sweep_buffer),
                    "requires_lidar": self.need_lidar,
                    "speed": float(self.clip_speed[-1]),
                    "target_point": np.asarray(self.clip_tp[-1]).tolist(),
                    "target_point_next": np.asarray(self.clip_ntp[-1]).tolist(),
                },
                "control_context": {
                    "prev_throttle": float(self.control.throttle),
                    "prev_steer": float(self.control.steer),
                    "prev_brake": float(self.control.brake),
                    "history_seconds": (self.clip_len - 1) * self.step_stride / 20.0,
                    "waypoint_dt_seconds": 0.25,
                },
                "pred_route": self.last_pred_route.tolist(),
                "pred_future_waypoints": self.last_pred_waypoints.tolist(),
            }, f, indent=2)
        wp = self.last_pred_waypoints
        rt = self.last_pred_route
        print(
            f"[MOTLeadAgent] INFER step={self.step:5d} dt={dt*1000:6.1f}ms "
            f"|wp[0]|={np.linalg.norm(wp[0]):.2f}m "
            f"|wp[-1]|={np.linalg.norm(wp[-1]):.2f}m "
            f"|rt[-1]|={np.linalg.norm(rt[-1]):.2f}m",
            flush=True,
        )

    # ----- 录像 -----
    def _record_videos(self, stitched_rgb: np.ndarray, tick_data: dict | None) -> None:
        """写 input/debug/demo/grid 四路视频的一帧。

        input/demo/grid 可以在 warmup 阶段写；debug 需要预测轨迹，所以 tick_data 或
        last_pred_waypoints 缺失时会跳过。
        """
        stitched_bgr = cv2.cvtColor(stitched_rgb, cv2.COLOR_RGB2BGR)
        self.video.write_input_frame(stitched_bgr)

        if _RECORD_DEBUG and tick_data is not None and self.last_pred_waypoints is not None:
            try:
                debug_bgr = overlay_pred_on_stitched_three_cams(
                    stitched_bgr,
                    pred_waypoints_ego=self.last_pred_waypoints,
                    target_point_ego=tick_data["target_point_ego"],
                    cam_calibs_3cam=_LEAD_3CAM_CALIBRATION,
                    fov_deg=_LEAD_3CAM_FOV,
                )
                self.video.write_debug_frame(debug_bgr)
            except Exception as e:
                print(f"[MOTLeadAgent] debug overlay failed: {e}")

        # BEV 顶视 debug：用最近一帧 LiDAR + 上一拍预测画一张 LEAD 风格 BEV pseudo-image。
        # tick_data 为 None（warmup 阶段）时仍画一张含 ego box 的占位图。
        if _RECORD_BEV_DEBUG:
            try:
                lidar_for_bev = tick_data["lidar_ego"] if tick_data is not None else None
                tp_for_bev = tick_data["target_point_ego"] if tick_data is not None else None
                ntp_for_bev = (
                    tick_data["next_target_point_ego"] if tick_data is not None else None
                )
                bev_bgr = render_bev_debug(
                    lidar_ego=lidar_for_bev,
                    pred_waypoints_ego=self.last_pred_waypoints,
                    pred_route_ego=self.last_pred_route,
                    target_point_ego=tp_for_bev,
                    next_target_point_ego=ntp_for_bev,
                )
                self.video.write_bev_debug_frame(bev_bgr)
            except Exception as e:
                print(f"[MOTLeadAgent] bev_debug render failed: {e}")

        # demo / grid 不依赖 stitched_bgr，但 grid 会复用 _last_input
        try:
            self.video.write_demo_frame()
        except Exception as e:
            print(f"[MOTLeadAgent] demo frame write failed: {e}")

    # ----- PID 控制 -----
    def _waypoints_to_control(self, tick_data) -> carla.VehicleControl:
        """把 LeadMoT 输出的 ego-frame future waypoints 转成 CARLA 控制量。

        横向控制复用 AutoMoT 的 LateralPIDController；纵向速度由前两个 waypoint
        估计目标速度，再经过 PID 得到 throttle/brake。最后统一交给 SafetyMixin 兜底。
        """
        wps = self.last_pred_waypoints
        route = self.last_pred_route
        if wps is None or route is None:
            ctrl = carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0)
            return self.apply_safety_overrides(ctrl, tick_data)
        steer = float(self.turn_controller.step(route, tick_data["speed"]))
        steer = float(np.clip(steer, -1.0, 1.0))
        # 8 个未来 waypoints @ 0.25s/帧（waypoint_dt=0.25），跨度约 2s。
        # 按 LEADMOT_PLAN.md §32：闭环用 0.5s 与 1.0s 两个点估 desired speed，
        # 而不是 wp[0]/wp[1] 这种相邻两点（后者对加速/减速过敏，噪声大）。
        # 0.5s 处 = wp[1]，1.0s 处 = wp[3]；distance / 0.5s 即为期望平均速度。
        n = wps.shape[0]
        if n >= 4:
            v_05 = float(np.linalg.norm(wps[1]) / 0.5)         # 0~0.5s 段
            v_10 = float(np.linalg.norm(wps[3] - wps[1]) / 0.5)  # 0.5s~1.0s 段
            tgt_speed = 0.5 * v_05 + 0.5 * v_10                 # 平均
        elif n >= 2:
            tgt_speed = float(np.linalg.norm(wps[1]) / 0.5)
        elif n == 1:
            tgt_speed = float(np.linalg.norm(wps[0]) / 0.25)
        else:
            tgt_speed = 0.0
        tgt_speed = float(np.clip(tgt_speed, 0.0, 12.0))
        speed_err = tgt_speed - tick_data["speed"]
        thr = float(np.clip(self.speed_controller.step(speed_err), 0.0, self.clip_throttle))
        do_brake = tick_data["speed"] > tgt_speed * self.brake_ratio and tgt_speed < self.brake_speed
        if do_brake:
            thr, brake_v = 0.0, 1.0
        else:
            brake_v = 0.0
        ctrl = carla.VehicleControl()
        ctrl.throttle = float(thr)
        ctrl.steer = float(steer)
        ctrl.brake = float(brake_v)
        # stuck_helper / parking_escape / 限速 35km/h 兜底覆盖
        return self.apply_safety_overrides(ctrl, tick_data)

    # ============================================================
    # destroy（路线结束）
    # ============================================================
    def destroy(self):
        """leaderboard route 结束时调用：释放视频、压缩 mp4、清理模型/GPU cache。"""
        try:
            self.video.cleanup_and_compress()
        except Exception as e:
            print(f"[MOTLeadAgent] video cleanup failed: {e}")
        try:
            del self.runner
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
