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
            raise FileNotFoundError(f"关键帧 JSON 不存在：{target}")
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
            if idx is None: 跳过真值，用 randn z1 兜底跑前向。

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

    def find_status_for_anchor(
        self,
        scenario: str,
        run_id: str,
        anchor: int,
    ) -> Optional[str]:
        """根据 anchor 帧号反查它落在哪个 status 区间。

        逻辑与 build_dataset.build_run_timeline 完全一致：
          - initial 区间：[initial.frame, middle[0].frame - 1]
          - middle[i] 区间：[middle[i].frame, middle[i+1].frame - 1]（i<2）
                          或 [middle[2].frame, final.frame - 1]（i==2）
          - final 区间：[final.frame, +∞)
        anchor 落在某区间内，那个区间的 event 名就是当前 STATUS。

        找不到 run / 区间结构异常 / anchor 落在 initial.frame 之前都返回 None，
        让 runner 走"打 WARNING 退回默认 initial"的兜底分支。

        用法（runner debug）：
            status = KeyframeIndex.load().find_status_for_anchor(scenario, run_id, anchor)
            if status: memory.status = status
            else: print("[runner] 警告：anchor 落在已知区间之外，使用默认 initial")
        """

        run = self.find_run(scenario, run_id)
        if run is None:
            return None

        # 把 initial / middle / final 摊平成 [(start_frame, event), ...] 升序列表。
        # 数据构建器用 (start, end, event) 区间表；这里只用 start 也够：anchor 落在第 i
        # 个 start 之后但第 i+1 个 start 之前就属于第 i 段。这种"按起点查"对 final 段
        # 也天然成立（它的 start 就是最后一个，anchor >= final.frame 都算 final）。
        events: List[Dict[str, Any]] = []
        ini = run.get("initial")
        if ini and ini.get("frame") is not None:
            events.append(ini)
        for ev in run.get("middle", []) or []:
            if ev.get("frame") is not None:
                events.append(ev)
        fin = run.get("final")
        if fin and fin.get("frame") is not None:
            events.append(fin)
        if not events:
            return None

        starts = [int(ev["frame"]) for ev in events]
        # anchor 在 initial 之前是脏数据（数据构建器也会拒掉这种 run），返回 None。
        if anchor < starts[0]:
            return None
        # 从后往前扫第一个 start <= anchor 的事件；用 bisect 也行但 5 个元素直接扫更易读。
        current: Optional[str] = None
        for ev, start in zip(events, starts):
            if start <= anchor:
                current = ev.get("event")
            else:
                break
        return current


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
        raise RuntimeError(f"load_keyframe_rgb 需要 opencv-python + numpy：{e}")

    # 先按 LEAD 的默认命名约定（4 位数 0-padding）拼路径。
    rgb_dir = pathlib.Path(route_dir) / "rgb"
    rgb_path = rgb_dir / f"{frame_idx:04d}.jpg"
    if not rgb_path.exists():
        # 兜底：少数 route 可能用了不同的命名（例如 0 padding 不是 4 位），按目录
        # 中 sorted 顺序的第 frame_idx 个 .jpg 拿。仍然出错才报。
        files = sorted(rgb_dir.glob("*.jpg"))
        if frame_idx < 0 or frame_idx >= len(files):
            raise FileNotFoundError(f"找不到这个关键帧：{rgb_path}")
        rgb_path = files[frame_idx]

    # cv2.imread 读出来是 BGR；后续 VAE 期望 RGB，所以这里立即转。
    # 不用 PIL.Image.open 是为了和 image_io.load_lead_rgb_clip 保持一致行为
    # （那条链路也用 cv2 + cvtColor，避免不同 JPEG 解码器导致的轻微像素差异）。
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 读取关键帧失败：{rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # VAE.encode 接受 PIL.Image 列表，所以这里转回 PIL。
    return Image.fromarray(rgb, mode="RGB")
