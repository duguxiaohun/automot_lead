"""SFT base eval Q2 候选构造测试。"""

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

from qwen3vl_local.sft_base.eval_candidates import q2_candidates_for_student_rs


@dataclass
class DummyFrame:
    """测试用最小 frame。"""

    frame_id: int
    rs_label: str
    event_label: str
    raw: Dict[str, Any]


def main() -> None:
    """验证 Q2 候选优先采用逐帧 allowed_events，缺失时才按学生 RS fallback。"""

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
    assert set(candidates) == {"RE", "U-E6"}, (candidates, source)
    assert source == "pred_rs_allowed_events"
    assert reachable is True

    # allowed_events 是帧级事实；学生 RS 猜错时也不再通过删掉 UE 选项制造必错。
    # 错误 RS 的影响由 Q2 prompt 的错误 RS 语境承担。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R1", seed=7)
    assert set(candidates) == {"RE", "U-E6"}, (candidates, source)
    assert source == "pred_rs_allowed_events"
    assert reachable is True

    # 即使 R3 静态表没有 UE，allowed_events 里的 UE 事实仍保留。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R3", seed=7)
    assert set(candidates) == {"RE", "U-E6"}, (candidates, source)
    assert reachable is True

    # pred_rs == gt_rs 时，跨 RS 的例外 regular code 不能被静态表过滤掉，否则会
    # 从 dataset 的单候选 ["RE"] 回落成 ["RE", "U-E2"]。
    cross_regular = DummyFrame(
        frame_id=5,
        rs_label="R2",
        event_label="RE",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E2", "R-E4", "U-E2"],
            "frame_event_annotation": {"allowed_events": ["R-E4"]},
            "event_candidates_ordered": ["RE"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(cross_regular, "R2", seed=7)
    assert candidates == ["RE"], (candidates, source)
    assert source == "pred_rs_allowed_events"
    assert reachable is True

    # 静态表过窄时，逐帧 allowed_events 认可的 UE 不能被 pred_rs 静态表砍掉。
    r4_static_obstacle = DummyFrame(
        frame_id=7,
        rs_label="R4",
        event_label="U-E2",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E4", "U-E2", "U-E6"],
            "frame_event_annotation": {"allowed_events": ["R-E4", "U-E2"]},
            "event_candidates_ordered": ["RE", "U-E2"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(r4_static_obstacle, "R4", seed=7)
    assert candidates == ["RE", "U-E2"], (candidates, source)
    assert source == "pred_rs_allowed_events"
    assert reachable is True

    r2_vulnerable_crossing = DummyFrame(
        frame_id=8,
        rs_label="R2",
        event_label="U-E4",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E2", "U-E2", "U-E4"],
            "frame_event_annotation": {"allowed_events": ["R-E2", "U-E4"]},
            "event_candidates_ordered": ["RE", "U-E4"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(r2_vulnerable_crossing, "R2", seed=7)
    assert candidates == ["RE", "U-E4"], (candidates, source)
    assert source == "pred_rs_allowed_events"
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
            "event_candidates_ordered": ["U-E5", "RE"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(ordered_frame, "R2", seed=12345)
    assert candidates == ["U-E5", "RE"], (candidates, source)
    assert source == "pred_rs_allowed_events"
    assert reachable is True

    # 旧数据没有逐帧 allowed_events 时，才回落到 pred_rs 静态候选。
    fallback_frame = DummyFrame(
        frame_id=9,
        rs_label="R4",
        event_label="U-E2",
        raw={
            "run_id": "route",
            "scenario_event_candidates": ["R-E4", "U-E2", "U-E6"],
        },
    )
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(fallback_frame, "R4", seed=7)
    assert set(candidates) == {"RE", "U-E6"}, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is False

    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, None, seed=7)
    assert candidates == ["RE"]
    assert source == "invalid_rs_fallback"
    assert reachable is False
    print("[test_eval_candidates] ok")


if __name__ == "__main__":
    main()
