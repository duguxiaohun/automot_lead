"""SFT v5 memory 状态机小测试。"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5.labels import EventTarget, RSTarget
from qwen3vl_local.sft_v5.prompts import (
    MemoryCurriculumConfig,
    MemoryCurriculumState,
    build_q1_teacher_prompt,
    build_q2_teacher_prompt,
    observe_training_memory,
    prepare_training_memory,
    reset_memory_for_frame,
    should_trigger_q2,
    update_memory_after_q1,
    update_memory_after_q2,
    update_memory_navigation,
)


def main() -> None:
    """按 Q1/Q2 顺序推进 memory，并验证扰动、延迟修复与私有信息隔离合同。"""

    # reset_memory_for_frame 只保留给 reference/兼容初始化；正式训练使用下方
    # curriculum，学生错误 memory 不会在下一帧立刻被脚本覆盖。
    rs = RSTarget("R1", "A", "ordinary road", 0.8, (), {"R1": 0.8})
    mem = reset_memory_for_frame(rs)
    assert mem.rs_label == "R1"
    assert mem.event_label == "RE"
    assert should_trigger_q2(student_rs_label="R1", target_rs_label="R1") is True
    assert should_trigger_q2(student_rs_label="R4", target_rs_label="R1") is False
    assert should_trigger_q2(student_rs_label=None, target_rs_label="R1") is False
    rendered_q1 = mem.format_q1_text()
    assert "PREVIOUS_EVENT_HYPOTHESIS" not in rendered_q1, "Q1 memory 不应提前暴露 EVENT"
    assert "PREVIOUS_RS_HYPOTHESIS: A -" not in rendered_q1, "memory 不应保存 A-E 选项编号"

    mem_with_goal = reset_memory_for_frame(rs, ego_to_goal_xy=(12.3, -1.5))
    assert "EGO_TO_GOAL_XY=(+12.3, -1.5) m" in mem_with_goal.format_q1_text()
    rendered_q2 = mem_with_goal.format_q2_text()
    assert "PREVIOUS_EVENT_HYPOTHESIS" in rendered_q2, "Q2 memory 才需要带 EVENT"
    assert "PREVIOUS_EVENT_HYPOTHESIS: RE -" not in rendered_q2, "memory 不应保存 RE/U-E 标签前缀"

    # teacher model 的可视化输出也应当和 student 一样直接产出分析字段，而不是复读
    # MEMORY / choices / REFERENCE。这里把 prompt 合同固定住，避免 base probe
    # 再出现 q1_teacher_output 只续写输入块的情况。
    event = EventTarget("RE", "R-E1", False, ("R-E1",), ("R-E1",))
    q1_teacher = build_q1_teacher_prompt(mem_with_goal, rs_target=rs, event_target=event, weather_text="clear")
    assert "Start directly with `Scene Description:`" in q1_teacher
    assert "Output exactly these lines:" in q1_teacher
    assert "RS: <A|B|C|D|E>" in q1_teacher
    q2_teacher = build_q2_teacher_prompt(
        mem_with_goal,
        option_map={"A": "RE"},
        q1_abnormal=False,
        event_target=event,
        regular_event_codes=("R-E1",),
    )
    assert "Start directly with `Scene Description:`" in q2_teacher
    assert "EVENT: <option letter>" in q2_teacher

    mem = update_memory_after_q1(mem, student_rs_label="R4", student_abnormal=True)
    assert mem.rs_label == "R4"
    # Q1 只能确认“是否异常”，不能凭空写具体 U-E*；具体事件必须由 Q2 决定。
    assert mem.event_label == "RE", "Q1 abnormal=yes 只等待 Q2，不应凭空写 UE"

    mem = update_memory_after_q2(mem, student_event_label="U-E6")
    assert mem.event_label == "U-E6"

    mem2 = update_memory_after_q2(mem, student_event_label=None)
    assert mem2.event_label == "U-E6", "Q2 非法输出不能污染当前 memory"

    # 测试/eval 的下一帧只能刷新外部导航量，不能因为真值已经变化就强制覆盖学生
    # 的 RS/EVENT。这样才能观察学生在后续多帧中是否会自行纠正。
    carried = update_memory_navigation(mem2, ego_to_goal_xy=(8.0, 2.0))
    assert carried.rs_label == "R4"
    assert carried.event_label == "U-E6"
    assert carried.ego_to_goal_x == 8.0 and carried.ego_to_goal_y == 2.0
    corrected = update_memory_after_q1(
        carried,
        student_rs_label="R2",
        student_abnormal=True,
    )
    assert corrected.rs_label == "R2", "RS 必须由后续 student Q1 输出自行改正"
    assert corrected.event_label == "U-E6", "Q1 abnormal=yes 时保留原 EVENT 等待 Q2"

    mem = update_memory_after_q1(mem, student_rs_label="R4", student_abnormal=False)
    assert mem.event_label == "U-E6", "Q1 不得在 Q2 前脚本化清除错误 EVENT memory"
    mem = update_memory_after_q2(mem, student_event_label="RE")
    assert mem.event_label == "RE", "EVENT 必须由 Q2 自己纠正"

    # 延迟修复：RS 连错达到 patience 后也要等低频 review interval；EVENT 可每帧复核。
    config = MemoryCurriculumConfig(
        rs_error_patience=3,
        event_error_patience=2,
        rs_repair_interval=2,
        event_repair_interval=1,
        rs_corrupt_prob=0.0,
        rs_unknown_prob=0.0,
        event_corrupt_prob=0.0,
        event_unknown_prob=0.0,
        rs_initial_gt_prob=1.0,
        event_initial_gt_prob=1.0,
    )
    state = MemoryCurriculumState()
    delayed, audit = prepare_training_memory(
        None,
        state,
        config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/route",
        frame_id=0,
        epoch=0,
        seed=7,
    )
    assert delayed.rs_label == "R1" and not audit["memory_rs_forced_repair"]
    delayed.rs_label = "R4"
    for _ in range(3):
        after = observe_training_memory(
            state,
            config,
            rs_correct=False,
            event_checked=True,
            event_correct=True,
        )
    assert after["memory_rs_repair_pending"] is True
    # 当前 ordinal=1，不到 RS interval，仍保留错误 R4 供学生自救。
    delayed, audit = prepare_training_memory(
        delayed,
        state,
        config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/route",
        frame_id=1,
        epoch=0,
        seed=7,
    )
    assert delayed.rs_label == "R4" and not audit["memory_rs_forced_repair"]
    # 下一 review ordinal=2 才执行 GT 兜底。
    delayed, audit = prepare_training_memory(
        delayed,
        state,
        config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/route",
        frame_id=2,
        epoch=0,
        seed=7,
    )
    assert delayed.rs_label == "R1" and audit["memory_rs_forced_repair"] is True

    defaults = MemoryCurriculumConfig()
    assert (defaults.rs_error_patience, defaults.rs_repair_interval) == (4, 2)
    assert (defaults.rs_corrupt_prob, defaults.rs_unknown_prob) == (0.06, 0.02)
    assert (defaults.event_error_patience, defaults.event_repair_interval) == (3, 1)
    assert (defaults.event_corrupt_prob, defaults.event_unknown_prob) == (0.10, 0.05)

    # EVENT 使用独立且更快的策略：达到 patience 后，默认 interval=1 的下一帧就修复。
    event_state = MemoryCurriculumState()
    event_mem, _ = prepare_training_memory(
        None,
        event_state,
        config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/event-route",
        frame_id=0,
        epoch=0,
        seed=8,
    )
    event_mem.event_label = "U-E6"
    for _ in range(2):
        event_after = observe_training_memory(
            event_state,
            config,
            rs_correct=True,
            event_checked=True,
            event_correct=False,
        )
    assert event_after["memory_event_repair_pending"] is True
    event_mem, event_audit = prepare_training_memory(
        event_mem,
        event_state,
        config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/event-route",
        frame_id=1,
        epoch=0,
        seed=8,
    )
    assert event_mem.event_label == "RE" and event_audit["memory_event_forced_repair"] is True

    # RS wrong 会让当前帧 EVENT gate 关闭，不能在同一帧再伪造一个未被 Q2 使用的
    # EVENT augmentation。
    wrong_config = MemoryCurriculumConfig(
        rs_corrupt_prob=1.0,
        rs_unknown_prob=0.0,
        event_corrupt_prob=0.0,
        event_unknown_prob=0.0,
        rs_initial_gt_prob=1.0,
        event_initial_gt_prob=1.0,
    )
    wrong, wrong_audit = prepare_training_memory(
        None,
        MemoryCurriculumState(),
        wrong_config,
        gt_rs_label="R4",
        gt_event_label="U-E6",
        accepted_event_labels=("U-E6", "U-E7"),
        event_corruption_choices=("U-E6", "U-E7", "RE"),
        ego_to_goal_xy=(0.0, 0.0),
        route_key="Scenario/wrong",
        frame_id=0,
        epoch=0,
        seed=9,
    )
    assert wrong.rs_label != "R4" and wrong.event_label == "U-E6"
    assert wrong_audit["memory_rs_injected_wrong"] is True
    assert wrong_audit["memory_event_injected_wrong"] is False
    assert wrong_audit["memory_event_gate_ready"] is False

    # RS 可用时，EVENT wrong 优先取本帧可见候选，并排除所有多标签可接受答案。
    event_wrong_config = dataclasses.replace(
        wrong_config,
        rs_corrupt_prob=0.0,
        event_corrupt_prob=1.0,
    )
    event_wrong, event_wrong_audit = prepare_training_memory(
        None,
        MemoryCurriculumState(),
        event_wrong_config,
        gt_rs_label="R4",
        gt_event_label="U-E6",
        accepted_event_labels=("U-E6", "U-E7"),
        event_corruption_choices=("U-E6", "U-E7", "RE"),
        ego_to_goal_xy=(0.0, 0.0),
        route_key="Scenario/event-wrong",
        frame_id=0,
        epoch=0,
        seed=9,
    )
    assert event_wrong.rs_label == "R4" and event_wrong.event_label == "RE"
    assert event_wrong_audit["memory_event_injected_wrong"] is True

    # no-prior 初始化是显式 UNKNOWN，不得再静默 fallback 成 R1 描述。
    unknown_config = MemoryCurriculumConfig(
        rs_corrupt_prob=0.0,
        rs_unknown_prob=0.0,
        event_corrupt_prob=0.0,
        event_unknown_prob=0.0,
        rs_initial_gt_prob=0.0,
        event_initial_gt_prob=0.0,
    )
    unknown, _ = prepare_training_memory(
        None,
        MemoryCurriculumState(),
        unknown_config,
        gt_rs_label="R4",
        gt_event_label="U-E6",
        ego_to_goal_xy=(0.0, 0.0),
        route_key="Scenario/unknown",
        frame_id=0,
        epoch=0,
        seed=9,
    )
    assert unknown.rs_label == "UNKNOWN" and unknown.event_label == "UNKNOWN"
    assert "No reliable previous road-structure hypothesis" in unknown.format_q1_text()
    print("[test_memory_update] ok")


if __name__ == "__main__":
    main()
