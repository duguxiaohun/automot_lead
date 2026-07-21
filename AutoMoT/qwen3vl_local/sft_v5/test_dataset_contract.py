"""SFT v5 dataset / 候选池合同测试。"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5.labels import (
    EVENT_CANDIDATES_BY_RS,
    allowed_events_from_frame,
    collapse_regular_to_re,
    q2_raw_candidates_for_frame,
    q2_raw_candidates,
    stable_event_option_map,
)


def main() -> None:
    """验证逐帧 allowed_events 优先、RE 折叠和选项随机可复现。"""

    # scenario_candidates 模拟 collection_output 顶层候选：它描述这条 scenario
    # 理论上可能出现哪些原始事件；真正进入 Q2 前还要按当前 RS 过滤。
    scenario_candidates = ["R-E1", "R-E2", "R-E4", "R-E5", "U-E4", "U-E6"]

    r1_raw = q2_raw_candidates(scenario_candidates, "R1")
    assert set(r1_raw) == {"R-E1", "R-E2", "U-E4"}
    assert collapse_regular_to_re(r1_raw, "R1") == ["RE", "U-E4"]

    r4_raw = q2_raw_candidates(scenario_candidates, "R4")
    assert set(r4_raw) == {"R-E4", "U-E4", "U-E6"}
    assert collapse_regular_to_re(r4_raw, "R4") == ["RE", "U-E4", "U-E6"]

    r3_raw = q2_raw_candidates(scenario_candidates, "R3")
    assert set(r3_raw) == {"R-E1", "R-E2"}
    # R3 的正常 highway/ramp 行为可能有多个 R-E，但 prompt 里只训练一个 RE。
    assert collapse_regular_to_re(r3_raw, "R3") == ["RE"], "R3 只折叠 regular 为 RE，不开放 UE"

    m1 = stable_event_option_map(run_id="route", frame_id=3, rs_label="R4", scenario_candidates=scenario_candidates, seed=7)
    m2 = stable_event_option_map(run_id="route", frame_id=3, rs_label="R4", scenario_candidates=scenario_candidates, seed=7)
    assert m1 == m2, "frame 级随机必须可复现"
    assert set(m1.values()) == {"RE", "U-E4", "U-E6"}

    frame = {
        "frame_event_annotation": {"allowed_events": ["R-E4", "U-E8"]},
        "event_evidence": {"allowed_events": ["R-E4", "U-E6"]},
    }
    # 用户明确要求逐帧 allowed_events 优先；即使 event_evidence 或静态 scenario 表里
    # 有不同 UE，也不能覆盖 frame_event_annotation.allowed_events。
    assert allowed_events_from_frame(frame) == ["R-E4", "U-E8"]
    allowed_raw = q2_raw_candidates_for_frame(frame, scenario_candidates=scenario_candidates, rs_label="R4")
    assert allowed_raw == ["R-E4", "U-E8"], "逐帧 allowed_events 必须优先于 scenario fallback"
    assert collapse_regular_to_re(["R-E2", "U-E8"], "R4") == ["RE", "U-E8"], "逐帧 R-E 不能被当前 RS 静态表过滤"
    # 合并后的 EVENT_FAST 必须始终能在 REGULAR 和 UE 之间比较。
    # 即使 collector 的逐帧 allowed_events 只列 UE，也保留恰好一个 RE。
    ue_only = collapse_regular_to_re(["U-E6", "U-E8"], "R4")
    assert ue_only == ["RE", "U-E6", "U-E8"]
    ue_only_map = stable_event_option_map(
        run_id="route",
        frame_id=5,
        rs_label="R4",
        scenario_candidates=scenario_candidates,
        raw_candidates=["U-E6", "U-E8"],
        seed=7,
    )
    assert list(ue_only_map.values()).count("RE") == 1
    assert set(ue_only_map.values()) == {"RE", "U-E6", "U-E8"}
    m3 = stable_event_option_map(
        run_id="route",
        frame_id=4,
        rs_label="R4",
        scenario_candidates=scenario_candidates,
        raw_candidates=allowed_raw,
        seed=7,
    )
    assert set(m3.values()) == {"RE", "U-E8"}

    for rs, candidates in EVENT_CANDIDATES_BY_RS.items():
        # 静态表只允许原始 R-E*/U-E*，不能提前混入 prompt 展示用的 RE。
        assert all(code.startswith("R-E") or code.startswith("U-E") for code in candidates), rs
    print("[test_dataset_contract] ok")


if __name__ == "__main__":
    main()
