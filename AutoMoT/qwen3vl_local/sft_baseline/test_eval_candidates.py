"""SFT baseline eval Q2 候选构造测试。"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_baseline.eval_candidates import q2_candidates_for_student_rs


@dataclass
class DummyFrame:
    """测试用最小 frame。"""

    frame_id: int
    rs_label: str
    event_label: str
    raw: Dict[str, Any]


def main() -> None:
    """验证 Q2 候选只由学生 RS 的静态全集决定。"""

    frame = DummyFrame(
        frame_id=4,
        rs_label="R4",
        event_label="U-E6",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E1", "R-E2", "R-E3", "R-E4", "U-E4", "U-E6"],
            "frame_event_annotation": {"allowed_events": ["R-E4", "U-E6"]},
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R4", seed=7)
    assert set(candidates) == {"R-E4", "U-E4", "U-E6", "U-E7", "U-E8"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is True

    # 学生 RS 猜错时，候选随学生 RS 语境切换；正确答案不在该语境候选里时才不可达。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R1", seed=7)
    assert set(candidates) == {"R-E1", "R-E2", "U-E1", "U-E2", "U-E3", "U-E4", "U-E5"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is False

    # R3 不开放 UE，但会保留 3 个 regular 子类，避免单候选送分题。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R3", seed=7)
    assert set(candidates) == {"R-E1", "R-E2", "R-E3"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is False

    # R4 + U-E2 在全量共现审计中低于严格阈值，保留为 GT 静态表缺口审计项。
    r4_static_obstacle = DummyFrame(
        frame_id=7,
        rs_label="R4",
        event_label="U-E2",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E4", "U-E2", "U-E6"],
            "frame_event_annotation": {"allowed_events": ["R-E4", "U-E2"]},
            "event_candidates_ordered": ["R-E4", "U-E2"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(r4_static_obstacle, "R4", seed=7)
    assert set(candidates) == {"R-E4", "U-E4", "U-E6", "U-E7", "U-E8"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is False

    # 静态表已补入真实数据确认过的 R2 + U-E4 组合。
    r2_vulnerable_crossing = DummyFrame(
        frame_id=8,
        rs_label="R2",
        event_label="U-E4",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E2", "U-E2", "U-E4"],
            "frame_event_annotation": {"allowed_events": ["R-E2", "U-E4"]},
            "event_candidates_ordered": ["R-E2", "U-E4"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(r2_vulnerable_crossing, "R2", seed=7)
    assert set(candidates) == {"R-E1", "R-E2", "U-E2", "U-E4", "U-E5"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is True

    # 当 eval 现场算出的集合与 build_dataset 预生成集合相同时，必须复用 dataset
    # 顺序，不能因为 eval seed 不同而换顺序。
    ordered_frame = DummyFrame(
        frame_id=6,
        rs_label="R2",
        event_label="U-E5",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E2", "U-E2", "U-E5"],
            "frame_event_annotation": {"allowed_events": ["R-E2", "U-E5"]},
            "event_candidates_ordered": ["U-E5", "U-E2", "R-E1", "R-E2", "U-E4"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(ordered_frame, "R2", seed=12345)
    assert candidates == ["U-E5", "U-E2", "R-E1", "R-E2", "U-E4"], (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is True

    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, None, seed=7)
    assert candidates == ["R-E1"]
    assert source == "invalid_rs_fallback"
    assert reachable is False
    print("[test_eval_candidates] ok")


if __name__ == "__main__":
    main()


