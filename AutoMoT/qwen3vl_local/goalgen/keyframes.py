"""按 (scenario, run_id, subgoal_event) 查关键帧。

keyframes_all_scenarios.json 是 §13 三件套之一（远程产物，本机只读）。
结构（节选）：

```
{
  "scenarios": [...],
  "runs": [
    {
      "scenario": "Accident",
      "run_id": "Town03_Rep0_route_001783_route0_01_11_02_37_46",
      "initial": {"event": "initial", "frame": 0, ...},
      "middle": [
        {"event": "hazard_detect", "frame": 37, ...},
        {"event": "max_brake_or_min_gap", "frame": 61, ...},
        {"event": "recover_or_pass", "frame": 63, ...}
      ],
      "final": {"event": "final", "frame": ..., ...}
    },
    ...
  ]
}
```

这里只提供"查 frame_idx + 读取那一帧 stitched RGB"两个能力，不做任何 mutation。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


# 默认位置：仓库根目录的 keyframes_all_scenarios.json。
_THIS_FILE = pathlib.Path(__file__).resolve()
_DEFAULT_KEYFRAMES_JSON = _THIS_FILE.parents[3] / "keyframes_all_scenarios.json"


class KeyframeIndex:
    """对 keyframes_all_scenarios.json 的精简只读索引。"""

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        # (scenario, run_id) -> run dict
        self._by_run: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for run in payload.get("runs", []):
            key = (run.get("scenario", ""), run.get("run_id", ""))
            self._by_run[key] = run

    @classmethod
    def load(cls, path: Optional[pathlib.Path] = None) -> "KeyframeIndex":
        target = pathlib.Path(path) if path else _DEFAULT_KEYFRAMES_JSON
        if not target.exists():
            raise FileNotFoundError(f"keyframes json not found: {target}")
        with target.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload)

    def find_run(self, scenario: str, run_id: str) -> Optional[Dict[str, Any]]:
        return self._by_run.get((scenario, run_id))

    @staticmethod
    def _iter_events(run: Dict[str, Any]):
        """按 initial / middle[*] / final 顺序产出所有事件字典。"""

        ini = run.get("initial")
        if ini:
            yield ini
        for ev in run.get("middle", []) or []:
            yield ev
        fin = run.get("final")
        if fin:
            yield fin

    def find_frame_for_event(
        self,
        scenario: str,
        run_id: str,
        event: str,
    ) -> Optional[int]:
        """查目标事件在该 run 中的帧索引；找不到返回 None。

        用法（runner）：
            idx = KeyframeIndex.load().find_frame_for_event(
                scenario="Accident",
                run_id="Town03_..._02_37_46",
                event=memory.subgoal,   # 例如 "hazard_detect"
            )
            if idx is None: 跳过真值，用 randn z1 兜底跑 forward。

        线性扫一个 run 内部 5 个事件（initial + 3 middle + final），没必要 hash；
        加 hash 反而让"事件 token 拼错"的错误更难定位。
        """

        run = self.find_run(scenario, run_id)
        if run is None:
            return None
        for ev in self._iter_events(run):
            if ev.get("event") == event:
                frame = ev.get("frame")
                # 容忍 frame 字段缺失的脏数据：返回 None 让 runner 走 fallback。
                return int(frame) if frame is not None else None
        return None


def infer_run_id_from_route(route_dir: str) -> str:
    """LEAD 数据通常 data/<Scenario>/<run_id>/，最后一段就是 run_id。"""

    return pathlib.Path(route_dir).resolve().name


def load_keyframe_rgb(route_dir: str, frame_idx: int) -> Image.Image:
    """读取 route 目录下指定帧的 stitched RGB（三视角 1152x384）。

    LEAD 命名约定 rgb/{frame:04d}.jpg；与 image_io.load_lead_rgb_clip 保持一致。
    """

    try:
        import cv2
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"load_keyframe_rgb requires opencv-python + numpy: {e}")

    # 先按 LEAD 的默认命名约定（4 位数 0-padding）拼路径。
    rgb_dir = pathlib.Path(route_dir) / "rgb"
    rgb_path = rgb_dir / f"{frame_idx:04d}.jpg"
    if not rgb_path.exists():
        # 兜底：少数 route 可能用了不同的命名（例如 0 padding 不是 4 位），按目录
        # 中 sorted 顺序的第 frame_idx 个 .jpg 拿。仍然出错才报。
        files = sorted(rgb_dir.glob("*.jpg"))
        if frame_idx < 0 or frame_idx >= len(files):
            raise FileNotFoundError(f"no such keyframe: {rgb_path}")
        rgb_path = files[frame_idx]

    # cv2.imread 读出来是 BGR；后续 VAE 期望 RGB，所以这里立即转。
    # 不用 PIL.Image.open 是为了和 image_io.load_lead_rgb_clip 保持一致行为
    # （那条链路也用 cv2 + cvtColor，避免不同 JPEG 解码器导致的轻微像素差异）。
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read keyframe: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # VAE.encode 接受 PIL.Image 列表，所以这里转回 PIL。
    return Image.fromarray(rgb, mode="RGB")
