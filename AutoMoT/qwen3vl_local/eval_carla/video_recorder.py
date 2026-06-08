"""LeadMoT 闭环评测视频录制：input / debug / demo / grid 四路。

设计照搬 lead/lead/inference/video_recorder.py，但去掉 jaxtyping / beartype
依赖，只保留核心写入 + ffmpeg 压缩 + grid 拼接逻辑。

四路视频
- input.mp4 : 三视角拼接后的 1152×384 RGB（直接喂模型那一份）
- debug.mp4 : input 上叠加 pred_waypoints / target_point 投影
- demo.mp4  : 临时 spawn 两个 CARLA RGB camera（cinematic + BEV）横向拼
- grid.mp4  : demo 上下拼 input（demo 在上，input 在下，等宽）

ffmpeg 压缩（与 LEAD 一致）：
- input/demo/grid: crf=18 preset=slow（高质量）
- debug         : crf=28 preset=slower（低质量小体积）

落盘位置：<route_save_dir>/{input,debug,demo,grid}.mp4

CARLA demo cameras 由 agent 在 _init_route_planner 阶段调
VideoRecorder.setup_demo_cameras(world, vehicle) 注册；listen 回调把帧写到
内部 buffer，每 tick agent 调 write_demo_frame() 触发 cv2.VideoWriter.write。
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import threading
from typing import Optional

import cv2
import numpy as np

try:  # carla 可能在本地静态分析机器上没装；agent 侧使用时一定有
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore

LOG = logging.getLogger(__name__)


DEMO_CAMERAS = [
    # cinematic: 类似 LEAD demo 视频里的第三人称后上方视角，用来肉眼看闭环行为。
    {
        "name": "cinematic",
        "width": 960, "height": 1080, "fov": 100,
        "x": -6.5, "y": 0.0, "z": 6.0,
        "pitch": -30.0, "yaw": 0.0,
    },
    # bev: CARLA 顶视 RGB，不等于模型 BEV 特征，只用于录像诊断。
    {
        "name": "bev",
        "width": 960, "height": 1080, "fov": 100,
        "x": 0.0, "y": 0.0, "z": 22.0,
        "pitch": -90.0, "yaw": 0.0,
    },
]


class VideoRecorder:
    """input / debug / demo / grid 四路视频。

    所有路都按 video_fps 默认值（CARLA tick / produce_frame_frequency = 20 / 1）
    无差别采样。每路独立开关，方便从 launcher 控制。
    """

    def __init__(
        self,
        save_dir: pathlib.Path,
        record_input: bool = True,
        record_debug: bool = True,
        record_demo: bool = True,
        record_grid: bool = True,
        produce_frame_frequency: int = 1,
        compress: bool = True,
    ) -> None:
        """初始化四路视频状态。

        这里只记录开关和路径，不立即创建 VideoWriter；每路第一帧到来时才知道真实尺寸，
        所以 writer 采用 `_ensure_writer` 懒创建。
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.record_input = record_input
        self.record_debug = record_debug
        self.record_demo = record_demo
        self.record_grid = record_grid
        self.produce_frame_frequency = max(1, int(produce_frame_frequency))
        self.video_fps = 20.0 / self.produce_frame_frequency
        self.compress = compress
        self._has_ffmpeg = shutil.which("ffmpeg") is not None and compress

        self._input_writer: Optional[cv2.VideoWriter] = None
        self._debug_writer: Optional[cv2.VideoWriter] = None
        self._demo_writer: Optional[cv2.VideoWriter] = None
        self._grid_writer: Optional[cv2.VideoWriter] = None

        self._last_demo: Optional[np.ndarray] = None
        self._last_input: Optional[np.ndarray] = None

        # demo camera 状态
        self._demo_actors: list = []          # CARLA Actor refs
        self._demo_lock = threading.Lock()
        self._demo_latest: dict[str, np.ndarray] = {}

        self.step = 0

    # ------------------------------------------------------------------
    # demo cameras
    # ------------------------------------------------------------------
    def setup_demo_cameras(self, world, vehicle) -> None:
        """spawn cinematic 与 BEV 两个 RGB camera，并 attach 到 vehicle。"""
        if not (self.record_demo or self.record_grid):
            return
        if carla is None or world is None or vehicle is None:
            LOG.warning("VideoRecorder: cannot spawn demo cameras (carla/world/vehicle missing)")
            return

        bp_lib = world.get_blueprint_library()
        for cfg in DEMO_CAMERAS:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(cfg["width"]))
            bp.set_attribute("image_size_y", str(cfg["height"]))
            bp.set_attribute("fov", str(cfg["fov"]))
            try:
                # 关闭 motion blur，避免视频里车速变化导致 overlay/人工检查变糊。
                bp.set_attribute("motion_blur_intensity", "0.0")
            except Exception:
                pass
            transform = carla.Transform(
                carla.Location(x=cfg["x"], y=cfg["y"], z=cfg["z"]),
                carla.Rotation(pitch=cfg["pitch"], yaw=cfg["yaw"], roll=0.0),
            )
            actor = world.spawn_actor(bp, transform, attach_to=vehicle)
            actor.listen(self._make_demo_listener(cfg["name"]))
            self._demo_actors.append(actor)
            LOG.info(f"VideoRecorder: spawned demo camera {cfg['name']}")

    def _make_demo_listener(self, name: str):
        """为 CARLA camera 创建 listen 回调。

        CARLA sensor 回调和 leaderboard 主循环不在同一线程，所以用 lock 保护
        `_demo_latest`。回调只保存最新帧，真正写视频仍由 run_step 的 tick 节奏控制。
        """
        def _cb(image):
            """CARLA sensor.listen 的实际回调：把 BGRA raw buffer 转成 BGR numpy。"""
            try:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
                with self._demo_lock:
                    self._demo_latest[name] = arr  # BGR
            except Exception as e:
                LOG.warning(f"demo camera {name} callback failed: {e}")
        return _cb

    def _get_concatenated_demo(self) -> Optional[np.ndarray]:
        """横向拼接 cinematic + bev 最新帧。任一缺失则返回 None。"""
        with self._demo_lock:
            cinematic = self._demo_latest.get("cinematic")
            bev = self._demo_latest.get("bev")
            if cinematic is None or bev is None:
                return None
            # 两路相同 H，可直接 hconcat
            cine = cinematic.copy()
            bev_ = bev.copy()
        if cine.shape[0] != bev_.shape[0]:
            target_h = min(cine.shape[0], bev_.shape[0])
            cine = cv2.resize(cine, (cine.shape[1] * target_h // cine.shape[0], target_h))
            bev_ = cv2.resize(bev_, (bev_.shape[1] * target_h // bev_.shape[0], target_h))
        return cv2.hconcat([cine, bev_])

    # ------------------------------------------------------------------
    # frame writers
    # ------------------------------------------------------------------
    def update_step(self, step: int) -> None:
        """同步当前 CARLA tick，供 `_gate()` 判断是否该写帧。"""
        self.step = int(step)

    def _gate(self) -> bool:
        """是否在当前 tick 写视频。

        默认 produce_frame_frequency=1，即 20Hz 全写；调大该值可降低视频体积。
        """
        return self.step % self.produce_frame_frequency == 0

    def _ensure_writer(self, current: Optional[cv2.VideoWriter], path: pathlib.Path,
                       frame: np.ndarray) -> cv2.VideoWriter:
        """懒创建 VideoWriter。

        第一帧到来前不知道实际 frame size；因此等第一帧来了再按图像尺寸开 writer。
        """
        if current is not None:
            return current
        path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            (frame.shape[1], frame.shape[0]),
        )

    def write_input_frame(self, stitched_bgr: np.ndarray) -> None:
        """写模型实际 RGB 输入视频，并缓存给 grid.mp4 使用。"""
        if not self.record_input or not self._gate():
            return
        self._last_input = stitched_bgr
        self._input_writer = self._ensure_writer(
            self._input_writer, self.save_dir / "input.mp4", stitched_bgr
        )
        self._input_writer.write(stitched_bgr)

    def write_debug_frame(self, debug_bgr: np.ndarray) -> None:
        """写带预测轨迹 overlay 的调试视频。"""
        if not self.record_debug or not self._gate():
            return
        self._debug_writer = self._ensure_writer(
            self._debug_writer, self.save_dir / "debug.mp4", debug_bgr
        )
        self._debug_writer.write(debug_bgr)

    def write_demo_frame(self) -> None:
        """写 demo.mp4 与 grid.mp4。

        demo 帧来自异步 CARLA camera；如果当前 tick 还没收到两路最新帧，就跳过，
        不阻塞主推理循环。
        """
        if not (self.record_demo or self.record_grid) or not self._gate():
            return
        demo = self._get_concatenated_demo()
        if demo is None:
            return
        self._last_demo = demo
        if self.record_demo:
            self._demo_writer = self._ensure_writer(
                self._demo_writer, self.save_dir / "demo.mp4", demo
            )
            self._demo_writer.write(demo)
        if self.record_grid and self._last_input is not None:
            grid = self._make_grid(demo, self._last_input)
            if grid is not None:
                self._grid_writer = self._ensure_writer(
                    self._grid_writer, self.save_dir / "grid.mp4", grid
                )
                self._grid_writer.write(grid)

    @staticmethod
    def _make_grid(demo_bgr: np.ndarray, input_bgr: np.ndarray) -> Optional[np.ndarray]:
        """demo 在上 input 在下，按 input 宽度等比缩放 demo。"""
        if demo_bgr is None or input_bgr is None:
            return None
        target_w = input_bgr.shape[1]
        scale = target_w / demo_bgr.shape[1]
        target_h = int(demo_bgr.shape[0] * scale)
        demo_resized = cv2.resize(demo_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return cv2.vconcat([demo_resized, input_bgr])

    # ------------------------------------------------------------------
    # cleanup + ffmpeg compress
    # ------------------------------------------------------------------
    def _compress(self, path: pathlib.Path, crf: int, preset: str) -> None:
        """用 ffmpeg 原地转码压缩 mp4。

        cv2 写出的 mp4v 体积偏大；这里先写 tmp，再 os.replace，避免压缩失败破坏原文件。
        """
        if not self._has_ffmpeg or not path.exists():
            return
        tmp = path.with_suffix(".tmp.mp4")
        try:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-an",
                str(tmp),
            ]
            subprocess.run(cmd, check=True)
            os.replace(tmp, path)
            LOG.info(f"Compressed video: {path}")
        except Exception as e:
            LOG.warning(f"ffmpeg compress failed for {path}: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def cleanup_and_compress(self) -> None:
        """release writers + 销毁 demo cameras + ffmpeg 压缩。"""
        for actor in list(self._demo_actors):
            try:
                # leaderboard 每条 route 结束都要销毁临时 camera，否则下一条 route 会堆 actor。
                if actor.is_alive:
                    actor.stop()
                    actor.destroy()
            except Exception:
                pass
        self._demo_actors.clear()

        writers = [
            (self._input_writer, self.save_dir / "input.mp4", 18, "slow"),
            (self._debug_writer, self.save_dir / "debug.mp4", 28, "slower"),
            (self._demo_writer, self.save_dir / "demo.mp4", 18, "slow"),
            (self._grid_writer, self.save_dir / "grid.mp4", 18, "slow"),
        ]
        for writer, path, crf, preset in writers:
            if writer is not None:
                writer.release()
                self._compress(path, crf, preset)

        self._input_writer = None
        self._debug_writer = None
        self._demo_writer = None
        self._grid_writer = None
