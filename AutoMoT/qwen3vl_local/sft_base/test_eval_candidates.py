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
    """验证 Q2 候选只由学生 RS 控制。"""

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

    # 学生预测 R1 时，不能继续拿 R4 的窄候选 U-E6；应按 pred_rs 过滤后 fallback。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R1", seed=7)
    assert "U-E6" not in candidates, (candidates, source)
    assert source == "pred_rs_static_candidates"
    assert reachable is False

    # R3 没有 UE 候选，Q2 会退化成单候选 REGULAR；这是端到端惩罚，不是 GT 泄漏。
    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, "R3", seed=7)
    assert candidates == ["RE"], (candidates, source)
    assert reachable is False

    candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, None, seed=7)
    assert candidates == ["RE"]
    assert source == "invalid_rs_fallback"
    assert reachable is False
    print("[test_eval_candidates] ok")


if __name__ == "__main__":
    main()
