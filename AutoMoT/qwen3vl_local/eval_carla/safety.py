"""MOTLeadAgent 安全兜底逻辑（从 mot_b2d_agent.py 同步过来的纯逻辑层）。

把 stuck_detector / creep_throttle / parking_start / parking_escape / 限速
全部抽到这一个 mixin，agent.py 自身只关心模型推理和 PID。

设计目标与 mot_b2d_agent.py 完全等价：
- stuck_detector:    speed < 0.1 m/s 累计若干帧 → 触发 force_move creep
- force_move:        creep_duration 帧内强制最低油门 creep_throttle
- parking_start:     前 N 帧（默认 200 = 10s）位移 < 6m → 本局禁用 force_move
- parking_escape:    parking_deadlock_window 帧内最大位移 < 阈值 → 强制大转角 + 中油门拉出
- 限速:              速度 > 35 km/h 强制刹车

agent 在 `setup()` 调 `init_safety_state()` 注册所有字段；
在 `tick()` 末尾调 `update_safety_state(tick_data)` 维护 stuck / parking 计数与快照；
在 `_waypoints_to_control()` 末尾调 `apply_safety_overrides(ctrl, tick_data)` 让安全
策略覆盖 throttle/steer/brake；并把 update_parking_escape_anchor 等小工具暴露给外部。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover
    carla = None   # type: ignore


# ---- 默认参数（与 mot_b2d_agent.py setup 完全一致）----
_DEFAULTS = dict(
    # 限速
    max_speed_kmh=35.0,

    # stuck / creep
    stuck_threshold=300,            # 帧
    creep_duration=14,
    creep_throttle=0.4,

    # parking_start
    parking_start_check_frame=200,  # 帧（10s @ 20fps）
    parking_start_disp_thresh=6.0,  # 米

    # parking_escape deadlock detection
    pos_snapshot_interval=200,           # 帧（10s）
    parking_deadlock_window=1500,        # 帧（125s）
    parking_deadlock_max_disp=5.0,       # 米

    # parking_escape phase 1
    escape_phase1_timer=40,         # 帧（~2s @ 20fps）
    escape_phase1_throttle=0.45,
    escape_phase1_steer=0.65,
    escape_phase1_heading_deg=25.0,
    escape_displacement_ok=6.0,
    escape_cooldown=2400,           # 帧（120s）
)


class SafetyMixin:
    """需要 host class 提供：
       self.step (int)
       self._route_planner (nav_planner.RoutePlanner) 可选；脱困时会尝试 popleft 队列
    """

    # ============================================================
    # init
    # ============================================================
    def init_safety_state(self) -> None:
        """在 agent.setup() 末尾调用一次。"""
        cfg = dict(_DEFAULTS)
        self._safety_cfg = cfg

        # stuck / force_move
        self.stuck_detector = 0
        self.force_move = 0

        # parking_start
        self.parking_start_anchor: np.ndarray | None = None
        self.parking_start_checked = False
        self.parking_start_detected = False

        # parking_escape (deadlock)
        self.parking_escape_active = False
        self.parking_escape_phase = 0
        self.parking_escape_timer = 0
        self.parking_escape_anchor: np.ndarray | None = None
        self.parking_escape_start_compass: float | None = None
        self.parking_escape_attempt = 0
        self.parking_escape_cooldown = 0
        self.parking_escape_direction = 1.0   # +1 = left
        self.pos_snapshots: list[tuple[int, np.ndarray]] = []

    # ============================================================
    # 每 tick 维护：stuck 计数 / 快照 / 死锁检测 / 起步检测
    # ============================================================
    def update_safety_state(self, tick_data: dict[str, Any], warmup_done: bool) -> None:
        """tick 之后调一次。

        - 第一次 warmup_done 时记录 parking_start_anchor
        - 到达 parking_start_check_frame 后判定本局是否 parking_start
        - parking_escape 冷却期减计数
        - 满足条件时检测 deadlock，命中即激活 parking_escape
        - parking_escape 进行中：跟踪 anchor 位移与航向变化
        """
        cfg = self._safety_cfg
        gps_xy = np.asarray(tick_data["gps_world"], dtype=np.float64).reshape(-1)[:2]
        compass = float(tick_data.get("compass", 0.0))

        # --- parking_start ---
        if warmup_done:
            if self.parking_start_anchor is None:
                self.parking_start_anchor = gps_xy.copy()
            if (not self.parking_start_checked
                    and self.step >= cfg["parking_start_check_frame"]):
                disp = float(np.linalg.norm(gps_xy - self.parking_start_anchor))
                self.parking_start_detected = disp < cfg["parking_start_disp_thresh"]
                self.parking_start_checked = True
                if self.parking_start_detected:
                    print(f"[ParkingStart] Detected (disp={disp:.2f}m < "
                          f"{cfg['parking_start_disp_thresh']}m). force_move DISABLED.")
                else:
                    print(f"[ParkingStart] Normal start (disp={disp:.2f}m).")

        # --- 死锁检测：周期记录 + 窗口内最大位移阈值 ---
        if self.parking_escape_cooldown > 0:
            self.parking_escape_cooldown -= 1

        if not self.parking_escape_active and warmup_done:
            if self.step % cfg["pos_snapshot_interval"] == 0:
                self.pos_snapshots.append((self.step, gps_xy.copy()))
                cutoff = self.step - cfg["parking_deadlock_window"]
                self.pos_snapshots = [(s, p) for s, p in self.pos_snapshots if s >= cutoff]

            if (self.parking_escape_cooldown == 0
                    and len(self.pos_snapshots) >= 2):
                oldest_step, oldest_pos = self.pos_snapshots[0]
                time_span = self.step - oldest_step
                if time_span >= cfg["parking_deadlock_window"]:
                    max_disp = max(
                        float(np.linalg.norm(p - oldest_pos)) for _, p in self.pos_snapshots
                    )
                    if max_disp < cfg["parking_deadlock_max_disp"]:
                        print(f"[ParkingDetect] DEADLOCK {time_span} frames "
                              f"({time_span/20:.0f}s), max_disp={max_disp:.2f}m")
                        self._activate_parking_escape(gps_xy, compass)

        # --- escape 进行中：跟踪是否可以提前结束 ---
        if self.parking_escape_active:
            self._check_escape_progress(gps_xy, compass)

    # ============================================================
    # 控制覆盖：在 PID 出 ctrl 后调一次
    # ============================================================
    def apply_safety_overrides(
        self,
        ctrl,          # carla.VehicleControl
        tick_data: dict[str, Any],
    ):
        """统一在 ctrl 上叠加 stuck/force_move/escape/限速覆盖。返回新 ctrl。"""
        cfg = self._safety_cfg
        speed = float(tick_data.get("speed", 0.0))

        # 维护 stuck_detector（与 mot_b2d_agent.py 完全一致）
        if speed < 0.1:
            self.stuck_detector += 1
        elif speed > 0.2:
            self.stuck_detector = 0

        if self.parking_escape_active:
            # 脱困期间关 stuck / force_move，直接 override 控制
            self.stuck_detector = 0
            self.force_move = 0
            d = self.parking_escape_direction   # +1 = left
            if self.parking_escape_phase == 1:
                # CARLA 约定：steer>0 向右，steer<0 向左 → 想向左用 steer = -d * X
                ctrl.steer = float(-d * cfg["escape_phase1_steer"])
                ctrl.throttle = float(cfg["escape_phase1_throttle"])
                ctrl.brake = 0.0
                if self.parking_escape_timer % 20 == 0:
                    print(f"[ParkingEscape] Phase1 override "
                          f"steer={ctrl.steer:.2f} thr={ctrl.throttle:.2f}")
                self.parking_escape_timer -= 1
                if self.parking_escape_timer <= 0:
                    self._end_parking_escape("phase 1 timeout")
        else:
            # 常规 force_move
            if (self.stuck_detector > cfg["stuck_threshold"]
                    and not self.parking_start_detected):
                self.force_move = cfg["creep_duration"]
            if self.force_move > 0:
                ctrl.throttle = float(max(cfg["creep_throttle"], ctrl.throttle))
                ctrl.brake = 0.0
                self.force_move -= 1

        # 顶部限速
        if speed * 3.6 > cfg["max_speed_kmh"]:
            ctrl.throttle = 0.0
            ctrl.brake = 1.0
        return ctrl

    # ============================================================
    # internals
    # ============================================================
    def _activate_parking_escape(self, gps_xy: np.ndarray, compass: float) -> None:
        """进入停车脱困状态。

        这里不直接依赖模型输出，而是短时间强制给一段固定转角/油门。这样做是为了
        覆盖模型在停车场、被障碍物卡住、route target 太近时反复输出小位移的情况。
        """
        cfg = self._safety_cfg
        self.parking_escape_active = True
        self.parking_escape_phase = 1
        self.parking_escape_timer = cfg["escape_phase1_timer"]
        self.parking_escape_anchor = gps_xy.copy()
        self.parking_escape_start_compass = compass
        self.parking_escape_attempt += 1
        self.parking_escape_direction = 1.0   # 默认向左

        # 前推 route_planner 队列，避免脱困时 target_point 仍落在车身附近，
        # 否则模型/PID 可能刚脱困又被很近的旧目标点拉回去。
        try:
            rp = getattr(self, "_route_planner", None)
            if rp is not None:
                n_pop = min(5, max(0, len(rp.route) - 3))
                for _ in range(n_pop):
                    if len(rp.route) > 3:
                        rp.route.popleft()
                        rp.route_distances.popleft()
                print(f"[ParkingEscape] popped {n_pop} route WPs")
        except Exception as e:
            print(f"[ParkingEscape] skip route pop: {e}")

        self.stuck_detector = 0
        self.force_move = 0
        dir_str = "LEFT" if self.parking_escape_direction > 0 else "RIGHT"
        print(f"[ParkingEscape] ACTIVATED (attempt #{self.parking_escape_attempt}, dir={dir_str})")

    def _check_escape_progress(self, gps_xy: np.ndarray, compass: float) -> None:
        """脱困进行中每 tick 检查是否已经足够移动/转向。

        位移或航向变化超过阈值就提前结束，避免固定 2s 控制过度干预正常驾驶。
        """
        cfg = self._safety_cfg
        if self.parking_escape_anchor is None:
            return
        disp = float(np.linalg.norm(gps_xy - self.parking_escape_anchor))
        if (self.parking_escape_start_compass is not None
                and self.parking_escape_phase == 1):
            heading_diff = abs(compass - self.parking_escape_start_compass)
            if heading_diff > math.pi:
                heading_diff = 2 * math.pi - heading_diff
            heading_deg = math.degrees(heading_diff)
            if heading_deg > cfg["escape_phase1_heading_deg"]:
                self._end_parking_escape(
                    f"heading {heading_deg:.1f}° > {cfg['escape_phase1_heading_deg']}°, "
                    f"disp={disp:.1f}m"
                )
                return
        if disp > cfg["escape_displacement_ok"]:
            self._end_parking_escape(f"moved {disp:.1f}m")

    def _end_parking_escape(self, reason: str) -> None:
        """退出脱困并进入冷却期，防止同一段卡住状态里连续触发。"""
        cfg = self._safety_cfg
        print(f"[ParkingEscape] ENDED: {reason}")
        self.parking_escape_active = False
        self.parking_escape_phase = 0
        self.parking_escape_timer = 0
        self.parking_escape_cooldown = cfg["escape_cooldown"]
