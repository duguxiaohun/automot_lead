"""SFT base memory 扰动分布测试。

这个测试不加载 torch / 模型，只守住最容易肉眼漏掉的边际分布：EVENT 不能被
RS unknown 吞成大多数 UNKNOWN，R1 大类也不能确定性反向灌入假 R3 memory。
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base.memory_curriculum import (
    RouteMemoryCorruptor,
    event_memory_pool_for_rs,
    maybe_corrupt_memory,
    resample_event_memory_for_q2,
    wrong_rs_for_frame,
)
from qwen3vl_local.sft_base.prompts import Memory, build_q1_prompt, build_q2_prompt, update_memory_after_q1


@dataclass
class DummyFrame:
    """测试用最小 frame 对象。"""

    frame_id: int
    rs_label: str
    event_label: str
    event_candidates: List[str]
    ego_to_goal_xy: Optional[Tuple[float, float]]
    raw: Dict[str, Any]


def _make_frame(idx: int, rs_label: str, event_label: str = "U-E2") -> DummyFrame:
    """构造覆盖全部静态候选的 frame。"""

    return DummyFrame(
        frame_id=idx,
        rs_label=rs_label,
        event_label=event_label,
        event_candidates=["RE", "U-E1", "U-E2", "U-E3", "U-E4", "U-E5", "U-E6", "U-E7", "U-E8"],
        ego_to_goal_xy=(10.0, 0.0),
        raw={
            "scenario_event_candidates": [
                "R-E1",
                "R-E2",
                "R-E3",
                "R-E4",
                "R-E5",
                "U-E1",
                "U-E2",
                "U-E3",
                "U-E4",
                "U-E5",
                "U-E6",
                "U-E7",
                "U-E8",
            ]
        },
    )


def _ratio(count: int, total: int) -> float:
    """安全比例。"""

    return float(count) / max(float(total), 1.0)


def _mean_non_keep_run(labels: List[str]) -> float:
    """计算非 keep 状态的平均连续段长度。"""

    runs: List[int] = []
    current = None
    length = 0
    for label in labels:
        if label != current:
            if current not in {None, "keep"} and length > 0:
                runs.append(length)
            current = label
            length = 1
        else:
            length += 1
    if current not in {None, "keep"} and length > 0:
        runs.append(length)
    return sum(runs) / max(1, len(runs))


_GT_EVENT_BY_RS = {
    "R1": "U-E2",
    "R2": "U-E5",
    "R3": "U-E3",
    "R4": "U-E6",
    "R5": "U-E7",
}


def main() -> None:
    """运行分布合同测试。"""

    # 近似 full_route 的长尾结构：R1 明显多于 R3，用来抓 R1->R3 反向灌入。
    rs_stream = ["R1"] * 50 + ["R2"] * 12 + ["R3"] * 10 + ["R4"] * 18 + ["R5"] * 10
    total = 20000
    rs_counts = {"unknown": 0, "wrong": 0, "keep": 0}
    event_counts = {"unknown": 0, "wrong": 0, "keep": 0}
    mem_r3_total = 0
    mem_r3_gt_r3 = 0
    hidden_total = 0
    corrupted_snapshots = []
    visible_total = 0
    for idx in range(total):
        gt_rs = rs_stream[idx % len(rs_stream)]
        frame = _make_frame(idx + 1, gt_rs)
        base = Memory(rs_label=gt_rs, event_label=frame.event_label)
        mem = maybe_corrupt_memory(
            base,
            frame=frame,
            route_id="route",
            frame_pos=idx + 1,
            seed=20260724,
            first_frame_unknown=False,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.15,
        )
        corrupted_snapshots.append((mem.rs_label, mem.event_label, mem.hide_priors))
        if mem.hide_priors:
            hidden_total += 1
            assert mem.rs_label == gt_rs and mem.event_label == frame.event_label, mem
            continue
        visible_total += 1
        if mem.rs_label == "UNKNOWN":
            rs_counts["unknown"] += 1
        elif mem.rs_label != gt_rs:
            rs_counts["wrong"] += 1
            if mem.event_label != "UNKNOWN":
                assert mem.event_label in event_memory_pool_for_rs(frame, mem.rs_label), (gt_rs, mem)
        else:
            rs_counts["keep"] += 1
        if mem.event_label == "UNKNOWN":
            event_counts["unknown"] += 1
        elif mem.event_label != frame.event_label:
            event_counts["wrong"] += 1
        else:
            event_counts["keep"] += 1
        if mem.rs_label == "R3":
            mem_r3_total += 1
            mem_r3_gt_r3 += int(gt_rs == "R3")

    assert 0.10 <= _ratio(hidden_total, total) <= 0.20, hidden_total
    assert 0.38 <= _ratio(rs_counts["unknown"], visible_total) <= 0.42, rs_counts
    assert 0.27 <= _ratio(rs_counts["wrong"], visible_total) <= 0.33, rs_counts
    assert 0.27 <= _ratio(rs_counts["keep"], visible_total) <= 0.33, rs_counts
    assert 0.36 <= _ratio(event_counts["unknown"], visible_total) <= 0.44, event_counts
    assert 0.35 <= _ratio(event_counts["wrong"], visible_total) <= 0.43, event_counts
    assert 0.18 <= _ratio(event_counts["keep"], visible_total) <= 0.24, event_counts
    assert _ratio(mem_r3_gt_r3, mem_r3_total) >= 0.18, (mem_r3_gt_r3, mem_r3_total)

    first = maybe_corrupt_memory(
        Memory(rs_label="R4", event_label="U-E6"),
        frame=_make_frame(0, "R4", "U-E6"),
        route_id="route",
        frame_pos=0,
        seed=20260724,
        first_frame_unknown=True,
        rs_wrong_prob=0.30,
        rs_unknown_prob=0.40,
        event_wrong_prob=0.35,
        event_unknown_prob=0.35,
        rs_wrong_event_unknown_prob=0.25,
        memory_dropout_prob=1.0,
    )
    assert (first.rs_label, first.event_label, first.hide_priors) == ("UNKNOWN", "UNKNOWN", False), first

    changed = update_memory_after_q1(Memory(rs_label="R1", event_label="U-E1"), student_rs_label="R4")
    assert changed.rs_label == "R4" and changed.event_label == "UNKNOWN", changed
    same = update_memory_after_q1(Memory(rs_label="R4", event_label="U-E6"), student_rs_label="R4")
    assert same.event_label == "U-E6", same

    q2_event_counts = {"unknown": 0, "wrong": 0, "keep": 0}
    q2_visible_total = 0
    for idx in range(total):
        gt_rs = rs_stream[idx % len(rs_stream)]
        gt_event = _GT_EVENT_BY_RS[gt_rs]
        frame = _make_frame(idx + 1, gt_rs, gt_event)
        base = Memory(rs_label=gt_rs, event_label=gt_event)
        q1_mem = maybe_corrupt_memory(
            base,
            frame=frame,
            route_id="route",
            frame_pos=idx + 1,
            seed=20260724,
            first_frame_unknown=False,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.15,
        )
        after_q1 = update_memory_after_q1(q1_mem, student_rs_label=gt_rs)
        q2_mem = resample_event_memory_for_q2(
            after_q1,
            frame=frame,
            route_id="route",
            seed=20260724,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            keep_event_label=gt_event,
        )
        if q2_mem.hide_priors:
            assert q2_mem.event_label == gt_event, q2_mem
            continue
        q2_visible_total += 1
        assert q2_mem.event_label == "UNKNOWN" or q2_mem.event_label in event_memory_pool_for_rs(frame, q2_mem.rs_label), q2_mem
        if q2_mem.event_label == "UNKNOWN":
            q2_event_counts["unknown"] += 1
        elif q2_mem.event_label == gt_event:
            q2_event_counts["keep"] += 1
        else:
            q2_event_counts["wrong"] += 1
    assert 0.30 <= _ratio(q2_event_counts["unknown"], q2_visible_total) <= 0.40, q2_event_counts
    assert 0.30 <= _ratio(q2_event_counts["wrong"], q2_visible_total) <= 0.40, q2_event_counts
    assert 0.25 <= _ratio(q2_event_counts["keep"], q2_visible_total) <= 0.35, q2_event_counts

    def run_stateful_sequence() -> List[Tuple[str, str, str]]:
        """跑一条 route_state 序列，返回可复现快照。"""

        corruptor = RouteMemoryCorruptor(
            route_id="state-route",
            seed=20260724,
            first_frame_unknown=True,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.0,
            duration_min=3,
            duration_max=5,
        )
        snapshots: List[Tuple[str, str, str]] = []
        for idx in range(2000):
            gt_rs = rs_stream[idx % len(rs_stream)]
            gt_event = _GT_EVENT_BY_RS[gt_rs]
            frame = _make_frame(idx, gt_rs, gt_event)
            base = Memory(rs_label=gt_rs, event_label=gt_event)
            mem = corruptor.corrupt(base, frame=frame, frame_pos=idx)
            after_q1 = update_memory_after_q1(mem, student_rs_label=gt_rs)
            q2_mem = corruptor.resample_event_for_q2(after_q1, frame=frame, keep_event_label=gt_event)
            if idx == 0:
                assert (mem.rs_label, mem.event_label, mem.hide_priors) == ("UNKNOWN", "UNKNOWN", False), mem
            assert q2_mem.event_label == "UNKNOWN" or q2_mem.event_label in event_memory_pool_for_rs(frame, q2_mem.rs_label), q2_mem
            if idx == 0:
                rs_mode = "unknown"
                event_mode = "unknown"
            else:
                rs_mode = corruptor.rs_state.mode
                event_mode = corruptor.q2_event_state.mode
            snapshots.append((rs_mode, event_mode, q2_mem.event_label))
        return snapshots

    stateful_a = run_stateful_sequence()
    stateful_b = run_stateful_sequence()
    assert stateful_a == stateful_b, "route_state memory 必须同 route/seed 可复现"
    rs_modes = [item[0] for item in stateful_a[1:]]
    event_modes = [item[1] for item in stateful_a[1:]]
    assert 3.0 <= _mean_non_keep_run(rs_modes) <= 10.0, _mean_non_keep_run(rs_modes)
    assert 3.0 <= _mean_non_keep_run(event_modes) <= 10.0, _mean_non_keep_run(event_modes)
    assert 0.25 <= _ratio(sum(1 for x in rs_modes if x == "wrong"), len(rs_modes)) <= 0.40, rs_modes[:20]
    assert 0.30 <= _ratio(sum(1 for x in rs_modes if x == "unknown"), len(rs_modes)) <= 0.50, rs_modes[:20]
    assert 0.25 <= _ratio(sum(1 for x in event_modes if x == "wrong"), len(event_modes)) <= 0.45, event_modes[:20]
    assert 0.30 <= _ratio(sum(1 for x in event_modes if x == "unknown"), len(event_modes)) <= 0.50, event_modes[:20]

    # 转折帧不能把“本帧答案”塞进 prompt memory。RS keep 应沿用上一帧 RS；
    # Q2 EVENT keep 应沿用上一帧 EVENT，只允许 wrong 分支偶然撞上当前答案。
    rs_answer_leak = 0
    event_answer_leak = 0
    transition_total = 20000
    for idx in range(transition_total):
        rs_frame = _make_frame(idx + 1, "R4", "U-E6")
        rs_mem = maybe_corrupt_memory(
            Memory(rs_label="R1", event_label="RE"),
            frame=rs_frame,
            route_id="rs-transition",
            frame_pos=idx + 1,
            seed=20260724,
            first_frame_unknown=False,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.0,
        )
        rs_answer_leak += int(rs_mem.rs_label == "R4")

        event_frame = _make_frame(idx + 1, "R4", "U-E6")
        q1_mem = maybe_corrupt_memory(
            Memory(rs_label="R4", event_label="RE"),
            frame=event_frame,
            route_id="event-transition",
            frame_pos=idx + 1,
            seed=20260724,
            first_frame_unknown=False,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.0,
        )
        after_q1 = update_memory_after_q1(q1_mem, student_rs_label="R4")
        q2_mem = resample_event_memory_for_q2(
            after_q1,
            frame=event_frame,
            route_id="event-transition",
            seed=20260724,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            keep_event_label="RE",
        )
        event_answer_leak += int(q2_mem.event_label == "U-E6")
    assert _ratio(rs_answer_leak, transition_total) < 0.15, rs_answer_leak
    assert _ratio(event_answer_leak, transition_total) < 0.15, event_answer_leak

    q1_prompt = build_q1_prompt(Memory(rs_label="R4", event_label="U-E6"))
    q2_prompt = build_q2_prompt(Memory(rs_label="R4", event_label="U-E6"), candidates=["RE", "U-E6"])
    assert "BELIEVED_EVENT" not in q1_prompt, q1_prompt
    assert "BELIEVED_EVENT" in q2_prompt, q2_prompt

    # R1->R3 不能是确定性规则；R3->R1 必须保留为定向对抗。
    import random

    assert wrong_rs_for_frame(random.Random(0), "R3", "R3") == "R1"
    r1_to_r3 = sum(wrong_rs_for_frame(random.Random(i), "R1", "R1") == "R3" for i in range(100))
    assert 0 < r1_to_r3 < 45, r1_to_r3

    # 同 seed / 同输入必须可复现。
    second_pass = []
    for idx in range(total):
        gt_rs = rs_stream[idx % len(rs_stream)]
        frame = _make_frame(idx + 1, gt_rs)
        base = Memory(rs_label=gt_rs, event_label=frame.event_label)
        mem = maybe_corrupt_memory(
            base,
            frame=frame,
            route_id="route",
            frame_pos=idx + 1,
            seed=20260724,
            first_frame_unknown=False,
            rs_wrong_prob=0.30,
            rs_unknown_prob=0.40,
            event_wrong_prob=0.35,
            event_unknown_prob=0.35,
            rs_wrong_event_unknown_prob=0.25,
            memory_dropout_prob=0.15,
        )
        second_pass.append((mem.rs_label, mem.event_label, mem.hide_priors))
    assert corrupted_snapshots == second_pass
    print("[test_memory_curriculum] ok")


if __name__ == "__main__":
    main()
