"""SFT base dataset / 候选池合同测试。"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base.labels import (
    EVENT_CANDIDATES_BY_RS,
    allowed_events_from_frame,
    collapse_regular_to_re,
    q2_raw_candidates_for_frame,
    q2_raw_candidates,
    stable_event_choice_order,
)


def main() -> None:
    # scenario_candidates 模拟 collection_output 顶层候选。sft_base 当前协议下它只
    # 保留为旧接口兼容；真正 Q2 候选固定取当前 RS 的静态全集。
    scenario_candidates = ["R-E1", "R-E2", "R-E4", "R-E5", "U-E4", "U-E6"]

    r1_raw = q2_raw_candidates(scenario_candidates, "R1")
    assert set(r1_raw) == set(EVENT_CANDIDATES_BY_RS["R1"])
    assert collapse_regular_to_re(r1_raw, "R1") == ["RE", "U-E1", "U-E2", "U-E3", "U-E4"]

    r4_raw = q2_raw_candidates(scenario_candidates, "R4")
    assert set(r4_raw) == set(EVENT_CANDIDATES_BY_RS["R4"])
    assert collapse_regular_to_re(r4_raw, "R4") == ["RE", "U-E2", "U-E4", "U-E6", "U-E7", "U-E8"]

    r3_raw = q2_raw_candidates(scenario_candidates, "R3")
    assert set(r3_raw) == set(EVENT_CANDIDATES_BY_RS["R3"])
    # R3 的正常 highway/ramp 行为可能有多个 R-E，但 prompt 里只训练一个 RE。
    assert collapse_regular_to_re(r3_raw, "R3") == ["RE"], "R3 只折叠 regular 为 RE，不开放 UE"

    o1 = stable_event_choice_order(run_id="route", frame_id=3, rs_label="R4", scenario_candidates=scenario_candidates, seed=7)
    o2 = stable_event_choice_order(run_id="route", frame_id=3, rs_label="R4", scenario_candidates=scenario_candidates, seed=7)
    assert o1 == o2, "frame 级随机必须可复现"
    assert set(o1) == {"RE", "U-E2", "U-E4", "U-E6", "U-E7", "U-E8"}
    assert len(o1) == len(set(o1)), "候选顺序里不能有重复项"

    frame = {
        "frame_event_annotation": {"allowed_events": ["R-E4", "U-E8"]},
        "event_evidence": {"allowed_events": ["R-E4", "U-E6"]},
    }
    # allowed_events 仍按优先级读取，但只用于 GT 解析/审计，不参与 Q2 候选构造。
    assert allowed_events_from_frame(frame) == ["R-E4", "U-E8"]
    allowed_raw = q2_raw_candidates_for_frame(frame, scenario_candidates=scenario_candidates, rs_label="R4")
    assert set(allowed_raw) == set(EVENT_CANDIDATES_BY_RS["R4"]), "Q2 候选必须来自 RS 静态全集"
    o3 = stable_event_choice_order(
        run_id="route",
        frame_id=4,
        rs_label="R4",
        scenario_candidates=scenario_candidates,
        raw_candidates=allowed_raw,
        seed=7,
    )
    assert set(o3) == {"RE", "U-E2", "U-E4", "U-E6", "U-E7", "U-E8"}

    for rs, candidates in EVENT_CANDIDATES_BY_RS.items():
        # 静态表只允许原始 R-E*/U-E*，不能提前混入 prompt 展示用的 RE。
        assert all(code.startswith("R-E") or code.startswith("U-E") for code in candidates), rs
    print("[test_dataset_contract] ok")


if __name__ == "__main__":
    main()
