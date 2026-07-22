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
    Memory,
    MemoryCurriculumConfig,
    MemoryCurriculumState,
    PROMPT_CONTRACT_VERSION,
    SYSTEM_PROMPT_V5,
    advance_memory_age,
    build_q1_student_prompt,
    build_q1_teacher_prompt,
    build_q1_teacher_target,
    build_q2_student_prompt,
    build_q2_teacher_prompt,
    initialize_student_memory,
    observe_inference_rs_schedule,
    observe_training_memory,
    prepare_training_memory,
    reset_memory_for_frame,
    rs_slow_interval_for_state,
    should_run_rs_slow,
    should_run_event_fast,
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
    assert should_run_event_fast(
        rs_slow_ran=True,
        q1_rs_correct=False,
        memory_rs_label="R1",
        target_rs_label="R1",
    ) is False, "慢帧 Q1 invalid/错误不能被旧正确 memory 掩盖"
    assert should_run_event_fast(
        rs_slow_ran=False,
        q1_rs_correct=False,
        memory_rs_label="R1",
        target_rs_label="R1",
    ) is True, "稳定快帧应复用正确 RS memory"
    rendered_q1 = mem.format_q1_text()
    assert "PREVIOUS_EVENT_HYPOTHESIS" not in rendered_q1, "Q1 memory 不应提前暴露 EVENT"
    assert "PREVIOUS_RS_HYPOTHESIS: A -" not in rendered_q1, "memory 不应保存 A-E 选项编号"
    assert "PREVIOUS_RS_HYPOTHESIS_AGE: 0 frames" in rendered_q1

    mem_with_goal = reset_memory_for_frame(rs, ego_to_goal_xy=(12.3, -1.5))
    assert "EGO_TO_GOAL_XY=(+12.3, -1.5) m" in mem_with_goal.format_q1_text()
    rendered_q2 = mem_with_goal.format_q2_text()
    assert "PREVIOUS_EVENT_HYPOTHESIS" in rendered_q2, "Q2 memory 才需要带 EVENT"
    assert "PREVIOUS_EVENT_HYPOTHESIS_AGE: 0 frames" in rendered_q2
    assert "PREVIOUS_EVENT_HYPOTHESIS: RE -" not in rendered_q2, "memory 不应保存 RE/U-E 标签前缀"

    # teacher model 的可视化输出也应当和 student 一样直接产出分析字段，而不是复读
    # MEMORY / choices / REFERENCE。这里把 prompt 合同固定住，避免 base probe
    # 再出现 q1_teacher_output 只续写输入块的情况。
    event = EventTarget("RE", "R-E1", False, ("R-E1",), ("R-E1",))
    q1_teacher = build_q1_teacher_prompt(mem_with_goal, rs_target=rs, weather_text="clear")
    assert "Use REFERENCE to return the same four-line format" in q1_teacher
    assert "Return exactly:" in q1_teacher
    assert "RS: <option letter A-E>" in q1_teacher
    assert "ABNORMAL:" not in q1_teacher
    assert "ABNORMAL:" not in build_q1_teacher_target(
        rs_target=rs,
        weather_text="clear",
    )
    q2_teacher = build_q2_teacher_prompt(
        mem_with_goal,
        option_map={"A": "RE"},
        event_target=event,
        regular_event_codes=("R-E1",),
    )
    assert "Use REFERENCE to return the same four-line format" in q2_teacher
    assert "EVENT: <option letter>" in q2_teacher
    q2_student = build_q2_student_prompt(
        mem_with_goal,
        option_map={"A": "RE", "B": "U-E6"},
    )
    assert "[RE | REGULAR] regular/normal" in q2_student
    assert "[UE | UNUSUAL] unusual/abnormal" in q2_student
    assert "A. [RE | REGULAR]" in q2_student
    assert "B. [UE | UNUSUAL]" in q2_student
    assert "ABNORMAL:" not in q2_student

    # compact prompt 是显式版本化合同，防止后续又把 system、问题说明和格式占位
    # 各自扩写成重复长文。候选数量会改变 Q2 长度，所以这里只固定一个代表性二选一。
    q1_student = build_q1_student_prompt(mem_with_goal)
    assert PROMPT_CONTRACT_VERSION == "sft_v5_compact_prompt_v1"
    assert len(SYSTEM_PROMPT_V5.split()) <= 70
    assert len(q1_student.split()) <= 160
    assert len(q2_student.split()) <= 175
    assert len(q1_teacher.split()) <= 180
    assert len(q2_teacher.split()) <= 180

    mem = update_memory_after_q1(mem, student_rs_label="R4")
    assert mem.rs_label == "R4"
    # RS_SLOW 不回答 EVENT family，也不能凭空写具体 U-E*。但 EVENT 是 EVENT|RS：
    # 从 R1 切到 R4 后，旧 R1 下的 RE 必须先失效，随后只允许由 Q2 重新建立。
    assert mem.event_label == "UNKNOWN"
    assert mem.event_age_frames == 0

    mem = update_memory_after_q2(mem, student_event_label="U-E6")
    assert mem.event_label == "U-E6"
    assert mem.event_age_frames == 0

    mem2 = update_memory_after_q2(mem, student_event_label=None)
    assert mem2.event_label == "U-E6", "Q2 非法输出不能污染当前 memory"

    # 测试/eval 的下一帧先让两个 hypothesis age 各加 1，再只刷新外部
    # 导航量；两步都不能因为真值变化就覆盖学生 RS/EVENT。
    carried = advance_memory_age(mem2)
    carried = update_memory_navigation(carried, ego_to_goal_xy=(8.0, 2.0))
    assert carried.rs_label == "R4"
    assert carried.event_label == "U-E6"
    assert carried.ego_to_goal_x == 8.0 and carried.ego_to_goal_y == 2.0
    assert carried.rs_age_frames == 1 and carried.event_age_frames == 1
    confirmed_rs = update_memory_after_q1(carried, student_rs_label="R4")
    confirmed_event = update_memory_after_q2(carried, student_event_label="U-E6")
    assert confirmed_rs.rs_age_frames == 1, "重复确认同一 RS 不能伪装成新 memory"
    assert confirmed_event.event_age_frames == 1, "重复确认同一 EVENT 不能重置持续时间"
    corrected = update_memory_after_q1(
        carried,
        student_rs_label="R2",
    )
    assert corrected.rs_label == "R2", "RS 必须由后续 student Q1 输出自行改正"
    assert corrected.rs_age_frames == 0, "RS label 变化后 RS age 必须归零"
    assert corrected.event_age_frames == 0, "RS 变化后条件 EVENT age 必须归零"
    assert corrected.event_label == "UNKNOWN", "新 RS 不得继承旧 RS 下的 EVENT"

    # 同一 RS 的周期复核不会失效 EVENT；只有 RS hypothesis 真正变化才清除上下文。
    mem = update_memory_after_q1(mem, student_rs_label="R4")
    assert mem.event_label == "U-E6", "重复确认同一 RS 必须保留 EVENT"
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
    # 下一 review ordinal=2 才执行默认延迟 GT 修复。从首次错误到这里
    # 已经保留了数帧连续错误 memory，不是下一帧立刻覆盖。
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
    assert audit["memory_rs_repaired_to_ground_truth"] is True
    assert audit["memory_rs_input_age_frames"] == 0, "repair 真正改变 RS 时 age 应归零"
    assert delayed.event_label == "UNKNOWN" and delayed.event_age_frames == 0
    assert audit["memory_event_invalidated_by_rs_change"] is True
    repaired_after = observe_training_memory(
        state,
        config,
        rs_correct=True,
        rs_checked=True,
        event_checked=False,
        event_correct=False,
        rs_forced_repair=True,
    )
    assert repaired_after["memory_rs_self_recovered_after_streak"] is False
    assert repaired_after["memory_rs_recovered_after_forced_repair"] is True

    defaults = MemoryCurriculumConfig()
    assert defaults.rs_slow_interval == 4
    assert defaults.rs_slow_interval_jitter == 1
    assert (defaults.rs_error_patience, defaults.rs_repair_interval) == (4, 2)
    assert (defaults.rs_repair_mode, defaults.event_repair_mode) == (
        "ground_truth",
        "ground_truth",
    )
    assert (defaults.rs_corrupt_prob, defaults.rs_unknown_prob) == (0.05, 0.07)
    assert (defaults.event_error_patience, defaults.event_repair_interval) == (3, 1)
    assert (defaults.event_corrupt_prob, defaults.event_unknown_prob) == (0.20, 0.12)

    # 不传 schedule_key 是 legacy 固定周期单测：稳定 RS 每 4 帧复核，中间帧只跑
    # EVENT_FAST。正式 train/eval/probe 会传 key 并随机成 3/4/5；RS 错误/UNKNOWN
    # 仍会立即切回 RS_SLOW，并在回答错误后保持逐帧 recovery。
    schedule_config = dataclasses.replace(
        defaults,
        rs_corrupt_prob=0.0,
        rs_unknown_prob=0.0,
        event_corrupt_prob=0.0,
        event_unknown_prob=0.0,
        rs_initial_gt_prob=1.0,
        event_initial_gt_prob=1.0,
    )
    schedule_state = MemoryCurriculumState()
    schedule_mem, schedule_audit = prepare_training_memory(
        None,
        schedule_state,
        schedule_config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(0.0, 0.0),
        route_key="Scenario/schedule",
        frame_id=0,
        epoch=0,
        seed=10,
    )
    run_rs, reason = should_run_rs_slow(
        schedule_state,
        schedule_config,
        memory=schedule_mem,
        gt_rs_label="R1",
        frame_ordinal=int(schedule_audit["memory_frame_ordinal"]),
    )
    assert run_rs and reason == "route_start"
    observe_training_memory(
        schedule_state,
        schedule_config,
        rs_correct=True,
        rs_checked=True,
        event_checked=True,
        event_correct=True,
    )
    for frame_ordinal in (1, 2, 3):
        run_rs, reason = should_run_rs_slow(
            schedule_state,
            schedule_config,
            memory=schedule_mem,
            gt_rs_label="R1",
            frame_ordinal=frame_ordinal,
        )
        assert not run_rs and reason == "reuse_stable_rs"
    run_rs, reason = should_run_rs_slow(
        schedule_state,
        schedule_config,
        memory=schedule_mem,
        gt_rs_label="R1",
        frame_ordinal=4,
    )
    assert run_rs and reason == "periodic"

    # 正式调用传入 route key 后，每次 query 之后会从 3/4/5 帧可复现
    # 采样下一间隔；不同 route 不应全部锁在同一相位。
    random_state = MemoryCurriculumState(last_rs_query_ordinal=4)
    interval_draws = {
        rs_slow_interval_for_state(
            random_state,
            defaults,
            schedule_key=f"Scenario/route-{idx}",
            schedule_seed=20260711,
        )
        for idx in range(64)
    }
    assert interval_draws == {3, 4, 5}
    same_route_draws = {
        rs_slow_interval_for_state(
            MemoryCurriculumState(last_rs_query_ordinal=ordinal),
            defaults,
            schedule_key="Scenario/same-route",
            schedule_seed=20260711,
        )
        for ordinal in range(64)
    }
    assert same_route_draws == {3, 4, 5}, "同一 route 的后续 query 也不能锁死同一间隔"
    assert rs_slow_interval_for_state(
        random_state,
        defaults,
        schedule_key="Scenario/repro",
        schedule_seed=7,
    ) == rs_slow_interval_for_state(
        random_state,
        defaults,
        schedule_key="Scenario/repro",
        schedule_seed=7,
    )
    schedule_mem.rs_label = "R4"
    run_rs, reason = should_run_rs_slow(
        schedule_state,
        schedule_config,
        memory=schedule_mem,
        gt_rs_label="R1",
        frame_ordinal=2,
    )
    assert run_rs and reason == "memory_mismatch"

    # EVENT 使用独立且更快的策略：达到 patience 后，默认 interval=1
    # 在下一个 review slot 延迟写回 GT；错误当帧仍不会立刻修复。
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
    assert event_audit["memory_event_repaired_to_ground_truth"] is True
    assert event_audit["memory_event_input_age_frames"] == 0

    # UNKNOWN 软擦除作为显式消融仍可用；它只移除陈旧先验，不保证
    # 学生退出 UNKNOWN，所以不能和正式延迟 GT 修复统计混在一起。
    soft_config = dataclasses.replace(
        config,
        rs_repair_mode="unknown",
        event_repair_mode="unknown",
    )
    soft_state = MemoryCurriculumState(
        frames_seen=2,
        rs_recovery_active=True,
        rs_error_streak=soft_config.rs_error_patience,
        rs_repair_pending=True,
        event_error_streak=soft_config.event_error_patience,
        event_repair_pending=True,
    )
    soft_mem, soft_audit = prepare_training_memory(
        Memory(rs_label="R4", event_label="RE"),
        soft_state,
        soft_config,
        gt_rs_label="R1",
        gt_event_label="RE",
        ego_to_goal_xy=(1.0, 2.0),
        route_key="Scenario/soft-repair",
        frame_id=2,
        epoch=0,
        seed=8,
    )
    assert soft_mem.rs_label == "UNKNOWN"
    assert soft_mem.event_label == "UNKNOWN" and soft_mem.event_age_frames == 0
    assert soft_audit["memory_rs_repaired_to_unknown"] is True
    assert soft_audit["memory_event_invalidated_by_rs_change"] is True
    assert soft_state.event_error_streak == 0 and soft_state.event_repair_pending is False

    # 如果新 RS 语境下的同帧 Q2 仍然答错，应从 streak=1 重新累计，不能继承旧 RS
    # 语境已经耗尽的 patience 并立即触发修复。
    context_state = MemoryCurriculumState(event_error_streak=9, event_repair_pending=True)
    context_after = observe_training_memory(
        context_state,
        config,
        rs_correct=True,
        rs_checked=True,
        event_checked=True,
        event_correct=False,
        event_context_reset=True,
    )
    assert context_after["memory_event_context_reset_by_rs_change"] is True
    assert context_after["memory_event_error_streak_after"] == 1
    assert context_after["memory_event_repair_pending"] is False

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
    assert wrong.rs_label != "R4" and wrong.event_label == "UNKNOWN"
    assert wrong_audit["memory_rs_injected_wrong"] is True
    assert wrong_audit["memory_event_injected_wrong"] is False
    assert wrong_audit["memory_event_invalidated_by_rs_change"] is True
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
    assert "No reliable prior road structure" in unknown.format_q1_text()

    # eval/probe 默认也从 UNKNOWN 起步。无 GT 调度把“UNKNOWN -> 合法 RS”的第一次
    # 变化视为待确认，下一帧再输出相同 RS 后才恢复低频周期；整个过程不需要真值。
    inference_mem = initialize_student_memory(rs, ego_to_goal_xy=(3.0, -2.0))
    assert inference_mem.rs_label == "UNKNOWN" and inference_mem.event_label == "UNKNOWN"
    inference_state = MemoryCurriculumState(frames_seen=1, last_rs_query_ordinal=-1)
    inference_audit = observe_inference_rs_schedule(
        inference_state,
        rs_checked=True,
        memory_rs_label_before="UNKNOWN",
        student_rs_label="R1",
    )
    assert inference_audit["inference_rs_changed"] is True
    assert inference_state.rs_recovery_active is True
    run_confirm, confirm_reason = should_run_rs_slow(
        inference_state,
        defaults,
        memory=Memory(rs_label="R1", event_label="UNKNOWN"),
        gt_rs_label=None,
        frame_ordinal=1,
    )
    assert run_confirm and confirm_reason == "recovery"
    inference_state.frames_seen = 2
    stable_audit = observe_inference_rs_schedule(
        inference_state,
        rs_checked=True,
        memory_rs_label_before="R1",
        student_rs_label="R1",
    )
    assert stable_audit["inference_rs_changed"] is False
    assert inference_state.rs_recovery_active is False
    print("[test_memory_update] ok")


if __name__ == "__main__":
    main()
