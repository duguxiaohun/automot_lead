"""SFT base simple memory 扰动分布测试。

新 baseline 把 RS 折叠成 HIGHWAY/NON_HIGHWAY，把 EVENT 折叠成 RE/UE。
因此 wrong memory 必须在折叠后的二分类意义上真的错误，而不是只在 R1/R2/R4/R5
或 R-E 子类内部换一个等价标签。
"""

from __future__ import annotations

import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base_simple.labels import event_family_from_label, road_label_from_rs
from qwen3vl_local.sft_base_simple.memory_curriculum import (
    RouteMemoryCorruptor,
    _scaled_event_probs,
    maybe_corrupt_memory,
    wrong_event_for_frame,
    wrong_rs_for_frame,
)
from qwen3vl_local.sft_base_simple.prompts import Memory, build_q1_prompt, update_memory_after_q1, update_memory_after_q2


@dataclass
class DummyFrame:
    """测试用最小 frame 对象。"""

    frame_id: int
    rs_label: str
    event_label: str
    abnormal: bool
    event_candidates: List[str]
    ego_to_goal_xy: Optional[Tuple[float, float]]
    raw: Dict[str, Any]


def _make_frame(idx: int, rs_label: str, event_label: str = "U-E2") -> DummyFrame:
    """构造覆盖全部静态候选的 frame。"""

    return DummyFrame(
        frame_id=idx,
        rs_label=rs_label,
        event_label=event_label,
        abnormal=str(event_label).startswith("U-E"),
        event_candidates=["R-E1", "R-E2", "R-E3", "R-E4", "R-E5", "U-E1", "U-E2", "U-E3", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8"],
        ego_to_goal_xy=(10.0, 0.0),
        raw={},
    )


def _ratio(count: int, total: int) -> float:
    """安全比例。"""

    return float(count) / max(float(total), 1.0)


def main() -> None:
    """运行二分类 memory 合同测试。"""

    rng = random.Random(0)
    assert wrong_rs_for_frame(rng, "R3", "R3") == "R1"
    assert road_label_from_rs(wrong_rs_for_frame(rng, "R1", "R1")) == "HIGHWAY"
    assert event_family_from_label(wrong_event_for_frame(rng, "U-E2", _make_frame(1, "R1", "U-E2"), "R1")) == "RE"
    assert event_family_from_label(wrong_event_for_frame(rng, "R-E1", _make_frame(2, "R1", "R-E1"), "R1")) == "UE"
    eff_wrong, eff_unknown, eff_dropout, is_early = _scaled_event_probs(
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        memory_dropout_prob=0.15,
        early_ue_age=0,
        early_ue_frames=4,
        early_ue_wrong_scale=1.75,
        early_ue_unknown_scale=1.35,
        early_ue_dropout_scale=1.50,
    )
    # early-UE cap=0.85, so wrong+unknown should be close to 0.85 (not 1.0)
    assert is_early and abs((eff_wrong + eff_unknown) - 0.85) < 1e-3, (eff_wrong, eff_unknown)
    # keep should be preserved: keep = 1.0 - wrong - unknown
    assert 0.10 <= (1.0 - eff_wrong - eff_unknown) <= 0.20, f"keep floor broken: {1.0 - eff_wrong - eff_unknown}"
    assert abs(eff_dropout - 0.225) < 1e-6, eff_dropout

    total = 5000
    rs_stream = ["R1"] * 50 + ["R2"] * 12 + ["R3"] * 10 + ["R4"] * 18 + ["R5"] * 10
    corruptor = RouteMemoryCorruptor(
        route_id="route",
        seed=20260724,
        first_frame_unknown=False,
        rs_wrong_prob=0.30,
        rs_unknown_prob=0.40,
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        rs_wrong_event_unknown_prob=0.25,
        memory_dropout_prob=0.15,
        duration_min=3,
        duration_max=5,
    )
    hidden = 0
    road_unknown = road_wrong = road_keep = 0
    event_unknown = event_wrong = event_keep = 0
    for idx in range(total):
        gt_rs = rs_stream[idx % len(rs_stream)]
        gt_event = "R-E1"
        frame = _make_frame(idx + 1, gt_rs, gt_event)
        mem = corruptor.corrupt(Memory(rs_label=gt_rs, event_label=gt_event), frame=frame, frame_pos=idx + 1)
        if mem.hide_priors:
            hidden += 1
            continue
        gt_road = road_label_from_rs(gt_rs)
        mem_road = "UNKNOWN" if mem.rs_label == "UNKNOWN" else road_label_from_rs(mem.rs_label)
        gt_family = event_family_from_label(gt_event)
        mem_family = "UNKNOWN" if mem.event_label == "UNKNOWN" else event_family_from_label(mem.event_label)
        road_unknown += int(mem_road == "UNKNOWN")
        road_wrong += int(mem_road != "UNKNOWN" and mem_road != gt_road)
        road_keep += int(mem_road == gt_road)
        event_unknown += int(mem_family == "UNKNOWN")
        event_wrong += int(mem_family != "UNKNOWN" and mem_family != gt_family)
        event_keep += int(mem_family == gt_family)

    assert 0.10 <= _ratio(hidden, total) <= 0.20, hidden
    assert 0.30 <= _ratio(road_unknown, total) <= 0.45, (road_unknown, road_wrong, road_keep)
    assert 0.20 <= _ratio(road_wrong, total) <= 0.35, (road_unknown, road_wrong, road_keep)
    assert 0.30 <= _ratio(event_unknown, total) <= 0.45, (event_unknown, event_wrong, event_keep)
    assert 0.20 <= _ratio(event_wrong, total) <= 0.40, (event_unknown, event_wrong, event_keep)

    # Test early-UE directional safety: at age=0, memory=RE and should never
    # be pushed to UE. at age=8, the post-perturbation guard should push away
    # from UE most of the time, but some UE may leak through (keep mode).
    age0_ue = 0
    age0_total = 2000
    age8_ue = 0
    age8_total = 2000
    for idx in range(age0_total):
        c = RouteMemoryCorruptor(
            route_id=f"age0-{idx}",
            seed=20260724 + idx,
            first_frame_unknown=False,
            rs_wrong_prob=0.0,
            rs_unknown_prob=0.0,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.0,
            memory_dropout_prob=0.15,
            duration_min=1,
            duration_max=1,
            early_ue_frames=4,
            early_ue_wrong_scale=1.75,
            early_ue_unknown_scale=1.35,
            early_ue_dropout_scale=1.50,
            early_ue_resample_prob=0.70,
        )
        mem = c.corrupt(Memory(rs_label="R1", event_label="R-E1"), frame=_make_frame(idx, "R1", "U-E2"), frame_pos=idx, early_ue_age=0)
        if not mem.hide_priors and mem.event_label != "UNKNOWN":
            family = event_family_from_label(mem.event_label)
            age0_ue += int(family == "UE")
    for idx in range(age8_total):
        c = RouteMemoryCorruptor(
            route_id=f"age8-{idx}",
            seed=20260724 + idx,
            first_frame_unknown=False,
            rs_wrong_prob=0.0,
            rs_unknown_prob=0.0,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.0,
            memory_dropout_prob=0.15,
            duration_min=1,
            duration_max=1,
            early_ue_frames=4,
            early_ue_wrong_scale=1.75,
            early_ue_unknown_scale=1.35,
            early_ue_dropout_scale=1.50,
            early_ue_resample_prob=0.70,
        )
        mem = c.corrupt(Memory(rs_label="R1", event_label="U-E2"), frame=_make_frame(idx, "R1", "U-E2"), frame_pos=idx, early_ue_age=8)
        if not mem.hide_priors and mem.event_label != "UNKNOWN":
            family = event_family_from_label(mem.event_label)
            age8_ue += int(family == "UE")
    # P0 fix: age=0 should never push memory to UE (RE prior is preserved)
    assert age0_ue == 0, f"age=0 PREVIOUS_EVENT landed on UE {age0_ue}/{age0_total} times - keep floor broken"
    # age=8 should still have some UE leakage through keep + sampling luck
    assert 0 <= _ratio(age8_ue, age8_total) <= 0.15, f"age=8 UE rate too high: {_ratio(age8_ue, age8_total)}"

    first_corruptor = RouteMemoryCorruptor(
        route_id="first-route",
        seed=20260724,
        first_frame_unknown=True,
        rs_wrong_prob=0.30,
        rs_unknown_prob=0.40,
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        rs_wrong_event_unknown_prob=0.25,
        memory_dropout_prob=1.0,
    )
    first = first_corruptor.corrupt(Memory(rs_label="R4", event_label="U-E6"), frame=_make_frame(0, "R4", "U-E6"), frame_pos=0)
    assert (first.rs_label, first.event_label, first.hide_priors) == ("UNKNOWN", "UNKNOWN", False), first

    changed = update_memory_after_q1(Memory(rs_label="R1", event_label="U-E1"), student_rs_label="HIGHWAY")
    assert changed.rs_label == "R3" and changed.event_label == "UNKNOWN", changed
    same = update_memory_after_q1(Memory(rs_label="R4", event_label="U-E6"), student_rs_label="NON_HIGHWAY")
    assert same.event_label == "U-E6", same
    event_mem = update_memory_after_q2(Memory(rs_label="R3", event_label="UNKNOWN"), student_event_label="RE")
    assert event_family_from_label(event_mem.event_label) == "RE", event_mem
    event_mem = update_memory_after_q2(Memory(rs_label="R3", event_label="UNKNOWN"), student_event_label="UE")
    assert event_family_from_label(event_mem.event_label) == "UE", event_mem

    q1_prompt = build_q1_prompt(Memory(rs_label="R4", event_label="U-E6"))
    assert "PREVIOUS_ROAD: NON_HIGHWAY" in q1_prompt, q1_prompt
    assert "PREVIOUS_EVENT: UE" in q1_prompt, q1_prompt
    assert "ROAD: <HIGHWAY or NON_HIGHWAY>" in q1_prompt, q1_prompt
    assert "EVENT: <RE or UE>" in q1_prompt, q1_prompt

    frame_mode_a = maybe_corrupt_memory(
        Memory(rs_label="R2", event_label="U-E5"),
        frame=_make_frame(17, "R2", "U-E5"),
        route_id="frame-mode",
        frame_pos=17,
        seed=20260724,
        first_frame_unknown=False,
        rs_wrong_prob=0.30,
        rs_unknown_prob=0.40,
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        rs_wrong_event_unknown_prob=0.25,
        memory_dropout_prob=0.15,
    )
    frame_mode_b = maybe_corrupt_memory(
        Memory(rs_label="R2", event_label="U-E5"),
        frame=_make_frame(17, "R2", "U-E5"),
        route_id="frame-mode",
        frame_pos=17,
        seed=20260724,
        first_frame_unknown=False,
        rs_wrong_prob=0.30,
        rs_unknown_prob=0.40,
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        rs_wrong_event_unknown_prob=0.25,
        memory_dropout_prob=0.15,
    )
    assert (frame_mode_a.rs_label, frame_mode_a.event_label, frame_mode_a.hide_priors) == (
        frame_mode_b.rs_label,
        frame_mode_b.event_label,
        frame_mode_b.hide_priors,
    )
    print("[test_memory_curriculum] ok")


if __name__ == "__main__":
    main()

