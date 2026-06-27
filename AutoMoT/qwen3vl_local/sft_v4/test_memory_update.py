"""SFT v4 memory 状态机测试（纯 Python）。

覆盖：

- ``init_memory`` 默认 0.7 layer-1 正确率，且 road_structure / scene 联合自洽；
- ``update_memory_after_step1`` 的 layer-1 翻转与整链 reset；
- ``update_memory_after_step2`` 的同桶翻转、跨桶拒绝与非法过滤；
- ``update_memory_after_step3`` 的合法/非法 event 过滤；
- ``force_memory_to_gt_chain`` 的弱纠偏语义：road_structure / scene == GT 时全 noop，
  status/subgoal 必须跨帧保留；scene != GT 时走 scene-change reset；
- ``should_trigger_step2`` / ``should_trigger_step3`` 的分层触发条件。
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v4.build_dataset import compute_delta
from qwen3vl_local.sft_v4 import replay
from qwen3vl_local.sft_v4.prompts import (
    CANONICAL_SCENARIO_LABELS,
    Memory,
    ROAD_STRUCTURE_TO_SCENES,
    SCENE_TO_ROAD_STRUCTURE,
    build_step1_user_prompt,
    build_step1_teacher_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    canonicalize_scene,
    first_scene_in_bucket,
    first_subgoal,
    correct_memory_after_step1_skip,
    force_memory_to_gt_chain,
    force_memory_to_gt_scene,
    get_road_structure,
    get_full_sequence,
    inject_phase_b_noise,
    initial_event,
    init_memory,
    should_trigger_step2,
    should_trigger_step3,
    update_memory_after_step1,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_scene,
)


def _check_init_probability_modes() -> None:
    """D22/D27：init_memory 联合采样，内部始终自洽，默认 p=0.7。"""

    if len(CANONICAL_SCENARIO_LABELS) <= 1:
        return  # 单场景退化，无法排除
    gt = "Accident"
    gt_rs = get_road_structure(gt)
    always_ok = init_memory(
        run_id="run_ok",
        sub_scenario_id="sub_ok",
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
        gt_scene=gt,
        p_init_correct=1.0,
    )
    assert always_ok.road_structure == gt_rs
    assert SCENE_TO_ROAD_STRUCTURE[always_ok.scene] == always_ok.road_structure
    for i in range(16):
        always_wrong = init_memory(
            run_id=f"run_{i}",
            sub_scenario_id=f"sub_{i}",
            ego_to_goal_x=0.0,
            ego_to_goal_y=0.0,
            gt_scene=gt,
            p_init_correct=0.0,
        )
        assert always_wrong.road_structure != gt_rs, f"p_init_correct=0 在 i={i} 抽中了 GT road_structure={gt_rs}"
        assert SCENE_TO_ROAD_STRUCTURE[always_wrong.scene] == always_wrong.road_structure
    seen = {
        init_memory(
            run_id=f"mix_{i}",
            sub_scenario_id=f"sub_{i}",
            ego_to_goal_x=0.0,
            ego_to_goal_y=0.0,
            gt_scene=gt,
            p_init_correct=0.5,
        ).road_structure
        for i in range(64)
    }
    assert gt_rs in seen, "p_init_correct=0.5 没有覆盖 layer-1 初始正确样本"
    assert any(rs != gt_rs for rs in seen), "p_init_correct=0.5 没有覆盖 layer-1 初始错误样本"


def _check_step1_update_and_trigger() -> None:
    """D24：step1 更新 ROAD_STRUCTURE，翻转时 reset scene/status/subgoal。"""

    gt = "Accident"
    gt_rs = get_road_structure(gt)
    other_rs = next(rs for rs in ROAD_STRUCTURE_TO_SCENES if rs != gt_rs)
    base = Memory(
        road_structure=gt_rs,
        scene=gt,
        status=initial_event(gt),
        subgoal=first_subgoal(gt),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )

    keep = update_memory_after_step1(base, student_road_structure=gt_rs)
    assert keep.road_structure == gt_rs
    assert keep.scene == gt
    assert should_trigger_step2(
        memory_road_structure_before_step1=base.road_structure,
        memory_road_structure_after_step1=keep.road_structure,
        gt_road_structure=gt_rs,
    )

    flipped = update_memory_after_step1(base, student_road_structure=other_rs)
    assert flipped.road_structure == other_rs
    assert flipped.scene == ROAD_STRUCTURE_TO_SCENES[other_rs][0]
    assert flipped.status == initial_event(flipped.scene)
    assert flipped.subgoal == first_subgoal(flipped.scene)
    assert not should_trigger_step2(
        memory_road_structure_before_step1=base.road_structure,
        memory_road_structure_after_step1=flipped.road_structure,
        gt_road_structure=gt_rs,
    )

    wrong_base = Memory(
        road_structure=other_rs,
        scene=first_scene_in_bucket(other_rs),
        status=initial_event(first_scene_in_bucket(other_rs)),
        subgoal=first_subgoal(first_scene_in_bucket(other_rs)),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )
    corrected = update_memory_after_step1(wrong_base, student_road_structure=gt_rs)
    assert corrected.road_structure == gt_rs
    assert not should_trigger_step2(
        memory_road_structure_before_step1=wrong_base.road_structure,
        memory_road_structure_after_step1=corrected.road_structure,
        gt_road_structure=gt_rs,
    ), "刚纠正 layer-1 的帧不应继续跑 step2"

    ignored = update_memory_after_step1(base, student_road_structure="NOT_A_BUCKET")
    assert ignored.road_structure == base.road_structure
    assert ignored.scene == base.scene


def _check_student_prompt_contracts() -> None:
    """Student-facing prompt contract shared by train/collect/eval/probe/learn.

    All runtime paths import the same builders from prompts.py, so this pure-Python
    check is the guardrail for prompt drift: step1 is road-only; step2/3 carry full
    memory; no student prompt exposes teacher-private answer fields.
    """

    scene = "Accident"
    memory = Memory(
        road_structure=get_road_structure(scene),
        scene=scene,
        status=initial_event(scene),
        subgoal=first_subgoal(scene),
        ego_to_goal_x=12.5,
        ego_to_goal_y=-3.0,
    )

    step1 = build_step1_user_prompt(4, memory=memory)
    step1_teacher = build_step1_teacher_prompt(memory, memory.road_structure)
    shared_headings = (
        "Scene Description:",
        "Relevant Visible Cues:",
        "Evidence Assessment:",
        "Memory Judgment:",
    )
    assert "[STEP1_ROAD_MEMORY]" in step1
    assert "BELIEVED_ROAD_STRUCTURE" in step1
    assert "EGO_TO_GOAL_XY" in step1
    assert all(h in step1 for h in shared_headings)
    assert "ROAD_STRUCTURE: <name>" in step1
    assert "Then write the label line(s) yourself" in step1
    assert "[STEP1_TASK]" in step1
    assert "Change a believed label only when clear visible evidence contradicts it" in step1
    assert "not contradicted" in step1
    assert "A lead vehicle alone does not prove HIGHWAY_MERGE" in step1
    assert "Keep the analysis 60-120 words before the label" in step1
    assert "BELIEVED_SCENE" not in step1
    assert "BELIEVED_STATUS" not in step1
    assert "BELIEVED_SUBGOAL" not in step1
    assert "ANSWER_" not in step1
    assert "GROUND_TRUTH_" not in step1
    assert "REFERENCE_" not in step1
    assert all(h in step1_teacher for h in shared_headings)
    assert "ROAD_STRUCTURE: <name>" in step1_teacher
    assert "Do not write label line(s)" in step1_teacher
    assert "The verdict controls memory update only" in step1_teacher
    assert "not contradicted" in step1_teacher
    assert "correction is weakly grounded" in step1_teacher
    assert "Then write the label line(s) yourself" not in step1_teacher

    step2 = build_step2_student_prompt(memory)
    step2_teacher = build_step2_teacher_prompt(memory, memory.road_structure, scene)
    assert "[MEMORY]" in step2
    assert "BELIEVED_SCENE" in step2
    assert "BELIEVED_STATUS" in step2
    assert "BELIEVED_SUBGOAL" in step2
    assert all(h in step2 for h in shared_headings)
    assert "SCENE: <scenario_name>" in step2
    assert "Then write the label line(s) yourself" in step2
    assert "ANSWER_" not in step2
    assert "GROUND_TRUTH_" not in step2
    assert "REFERENCE_" not in step2
    assert all(h in step2_teacher for h in shared_headings)
    assert "SCENE: <scenario_name>" in step2_teacher
    assert "Do not write label line(s)" in step2_teacher
    assert "Then write the label line(s) yourself" not in step2_teacher

    step3 = build_step3_student_prompt(memory)
    step3_teacher = build_step3_teacher_prompt(
        memory,
        memory.road_structure,
        scene,
        memory.status,
        memory.subgoal,
    )
    assert "[MEMORY]" in step3
    assert "BELIEVED_SCENE" in step3
    assert "BELIEVED_STATUS" in step3
    assert "BELIEVED_SUBGOAL" in step3
    assert "[EVENT_OPTIONS]" in step3
    assert all(h in step3 for h in shared_headings)
    assert "STATUS: <event_name>" in step3
    assert "SUBGOAL: <event_name>" in step3
    assert "Then write the label line(s) yourself" in step3
    assert "ANSWER_" not in step3
    assert "GROUND_TRUTH_" not in step3
    assert "REFERENCE_" not in step3
    assert all(h in step3_teacher for h in shared_headings)
    assert "STATUS: <event_name>" in step3_teacher
    assert "SUBGOAL: <event_name>" in step3_teacher
    assert "Do not write label line(s)" in step3_teacher
    assert "Then write the label line(s) yourself" not in step3_teacher


def _check_delta_formula_allows_zero() -> None:
    """C1：δ 严格按公式允许为 0，不能被静默抬到 1。"""

    assert compute_delta([0, 0, 1, 2, 3], warn=False) == 0
    assert compute_delta([0, 2, 6, 8, 10], warn=False) == 1
    assert compute_delta([0, 50, 100, 120, 140], warn=False) == 10


def _check_scene_flip_branches() -> None:
    """update_memory_after_step2 的 4 种 (合法/非法) × (翻转/保持) 分支。"""

    base = Memory(
        road_structure=get_road_structure("Accident"),
        scene="Accident",
        status=initial_event("Accident"),
        subgoal=first_subgoal("Accident"),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )

    # 1) 输出与当前 scene 相同 → keep，status/subgoal 不动
    m1 = update_memory_after_step2(base, student_scene="Accident")
    assert m1.scene == "Accident"
    assert m1.status == base.status
    assert m1.subgoal == base.subgoal

    # 2) 输出合法的不同 scene → 翻转，status/subgoal 重置
    other = next(s for s in ROAD_STRUCTURE_TO_SCENES[base.road_structure] if s != "Accident")
    m2 = update_memory_after_step2(base, student_scene=other)
    assert m2.scene == other
    assert m2.status == initial_event(other)
    assert m2.subgoal == first_subgoal(other)

    # 2b) 输出跨桶 scene → 忽略，保持内部自洽
    cross_bucket = next(s for s in CANONICAL_SCENARIO_LABELS if SCENE_TO_ROAD_STRUCTURE[s] != base.road_structure)
    m2b = update_memory_after_step2(base, student_scene=cross_bucket)
    assert m2b.scene == base.scene
    assert m2b.road_structure == base.road_structure

    # 3) 非法 scene → 忽略
    m3 = update_memory_after_step2(base, student_scene="NotExistsScene")
    assert m3.scene == base.scene
    assert m3.status == base.status

    # 4) None → 忽略
    m4 = update_memory_after_step2(base, student_scene=None)
    assert m4.scene == base.scene


def _check_scene_alias_canonicalization() -> None:
    """V2 aliases are accepted as input but not exposed as v4 prediction targets."""

    assert canonicalize_scene("EnterActorFlowV2") == "EnterActorFlow"
    assert canonicalize_scene("MergerIntoSlowTrafficV2") == "MergerIntoSlowTraffic"
    assert validate_scene("EnterActorFlowV2")
    assert "EnterActorFlowV2" not in ROAD_STRUCTURE_TO_SCENES["HIGHWAY_MERGE"]

    base = Memory(
        road_structure=get_road_structure("EnterActorFlowV2"),
        scene="EnterActorFlowV2",
        status=initial_event("EnterActorFlowV2"),
        subgoal=first_subgoal("EnterActorFlowV2"),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )
    assert base.scene == "EnterActorFlow"

    kept = update_memory_after_step2(base, student_scene="EnterActorFlowV2")
    assert kept.scene == "EnterActorFlow"
    assert kept.status == base.status

    step2 = build_step2_student_prompt(base)
    assert "- EnterActorFlow:" in step2
    assert "EnterActorFlowV2" not in step2


def _check_step3_update() -> None:
    """update_memory_after_step3 只接受当前 scene 事件序列中的合法 event。"""

    scene = "Accident"
    seq = get_full_sequence(scene)
    base = Memory(
        road_structure=get_road_structure(scene),
        scene=scene,
        status=initial_event(scene),
        subgoal=first_subgoal(scene),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )

    valid_status = seq[1] if len(seq) > 1 else seq[0]
    valid_subgoal = seq[2] if len(seq) > 2 else seq[-1]
    m1 = update_memory_after_step3(base, student_status=valid_status, student_subgoal=valid_subgoal)
    assert m1.status == valid_status
    assert m1.subgoal == valid_subgoal

    # 非法 event 不写入
    m2 = update_memory_after_step3(m1, student_status="not_an_event", student_subgoal=None)
    assert m2.status == m1.status
    assert m2.subgoal == m1.subgoal


def _check_phase_b_force_helper() -> None:
    """D23/D27：Phase B 弱纠偏。road_structure/scene == GT 全 noop；错层才 reset。"""

    gt = "Accident"
    gt_rs = get_road_structure(gt)
    other_rs = next(rs for rs in ROAD_STRUCTURE_TO_SCENES if rs != gt_rs)
    other = ROAD_STRUCTURE_TO_SCENES[other_rs][0]

    # (a) scene 已等于 GT，status/subgoal 已经被 step3 推进过 → 必须保留
    seq = get_full_sequence(gt)
    advanced_status = seq[1] if len(seq) > 1 else seq[0]
    advanced_subgoal = seq[2] if len(seq) > 2 else seq[-1]
    mem_ok = Memory(
        road_structure=gt_rs,
        scene=gt,
        status=advanced_status,
        subgoal=advanced_subgoal,
        ego_to_goal_x=5.0,
        ego_to_goal_y=-1.0,
    )
    forced_ok = force_memory_to_gt_chain(mem_ok, gt_road_structure=gt_rs, gt_scene=gt)
    assert forced_ok.scene == gt
    assert forced_ok.road_structure == gt_rs
    assert forced_ok.status == advanced_status, "Phase B 在 scene==GT 时不应重置 status"
    assert forced_ok.subgoal == advanced_subgoal, "Phase B 在 scene==GT 时不应重置 subgoal"

    # 兼容旧 API 仍委托给三层 helper
    forced_ok_legacy = force_memory_to_gt_scene(mem_ok, gt_scene=gt)
    assert forced_ok_legacy.road_structure == gt_rs

    # (b) road_structure 与 GT 不一致 → 先拉回 layer-1，再走 scene-change reset
    mem_bad = Memory(
        road_structure=other_rs,
        scene=other,
        status=initial_event(other),
        subgoal=first_subgoal(other),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )
    forced_bad = force_memory_to_gt_chain(mem_bad, gt_road_structure=gt_rs, gt_scene=gt)
    assert forced_bad.road_structure == gt_rs
    assert forced_bad.scene == gt
    assert forced_bad.status == initial_event(gt)
    assert forced_bad.subgoal == first_subgoal(gt)


def _check_phase_b_noise_helper() -> None:
    """D17v4：Phase B 噪声可关闭；命中时改成非 GT scene 并重置 event。"""

    gt = "Accident"
    gt_rs = get_road_structure(gt)
    if len(CANONICAL_SCENARIO_LABELS) <= 1:
        return
    base = Memory(
        road_structure=gt_rs,
        scene=gt,
        status=initial_event(gt),
        subgoal=first_subgoal(gt),
        ego_to_goal_x=3.0,
        ego_to_goal_y=-2.0,
    )

    no_noise, injected = inject_phase_b_noise(base, gt_scene=gt, rng=random.Random(0), prob=0.0)
    assert injected is False
    assert no_noise.scene == base.scene
    assert no_noise.status == base.status
    assert no_noise.subgoal == base.subgoal

    noisy, injected = inject_phase_b_noise(base, gt_scene=gt, rng=random.Random(0), prob=1.0)
    assert injected is True
    assert noisy.scene != gt
    assert noisy.road_structure == gt_rs
    assert SCENE_TO_ROAD_STRUCTURE[noisy.scene] == gt_rs
    assert noisy.status == initial_event(noisy.scene)
    assert noisy.subgoal == first_subgoal(noisy.scene)
    assert noisy.ego_to_goal_x == base.ego_to_goal_x
    assert noisy.ego_to_goal_y == base.ego_to_goal_y


def _check_skip_correction_after_step1_skip_helper() -> None:
    """Next-frame skip correction: GT bucket, init status, same-bucket noise only."""

    gt = "Accident"
    gt_rs = get_road_structure(gt)
    wrong_rs = next(rs for rs in ROAD_STRUCTURE_TO_SCENES if rs != gt_rs)
    wrong_scene = ROAD_STRUCTURE_TO_SCENES[wrong_rs][0]
    base = Memory(
        road_structure=wrong_rs,
        scene=wrong_scene,
        status=initial_event(wrong_scene),
        subgoal=first_subgoal(wrong_scene),
        ego_to_goal_x=7.5,
        ego_to_goal_y=-3.25,
    )

    corrected, noisy = correct_memory_after_step1_skip(
        base,
        gt_scene=gt,
        rng=random.Random(0),
        scene_noise_prob=0.0,
    )
    assert noisy is False
    assert corrected.road_structure == gt_rs
    assert corrected.scene == gt
    assert corrected.status == initial_event(gt)
    assert corrected.subgoal == first_subgoal(gt)
    assert corrected.ego_to_goal_x == base.ego_to_goal_x
    assert corrected.ego_to_goal_y == base.ego_to_goal_y

    noisy_mem, noisy = correct_memory_after_step1_skip(
        base,
        gt_scene=gt,
        rng=random.Random(0),
        scene_noise_prob=1.0,
    )
    assert noisy is True
    assert noisy_mem.scene != gt
    assert noisy_mem.road_structure == gt_rs
    assert SCENE_TO_ROAD_STRUCTURE[noisy_mem.scene] == gt_rs
    assert noisy_mem.status == initial_event(noisy_mem.scene)
    assert noisy_mem.subgoal == first_subgoal(noisy_mem.scene)
    assert noisy_mem.ego_to_goal_x == base.ego_to_goal_x
    assert noisy_mem.ego_to_goal_y == base.ego_to_goal_y


def _check_should_trigger_step3() -> None:
    """C4：step3 只在 scene 前后都已稳定为 GT 时触发。"""

    gt = "Accident"
    other = next(s for s in CANONICAL_SCENARIO_LABELS if s != gt)

    # 4 种组合（step2 前 memory.scene、step2 后 memory.scene）：
    # | before | after | 期望触发 |
    # |--------|-------|---------|
    # | GT     | GT    | 是       |
    # | GT     | other | 否       |
    # | other  | other | 否       |
    # | other  | GT    | 否，刚纠正 scene 的帧不继续跑 step3 |
    assert should_trigger_step3(memory_scene_before_step2=gt, memory_scene_after_step2=gt, gt_scene=gt) is True
    assert should_trigger_step3(memory_scene_before_step2=gt, memory_scene_after_step2=other, gt_scene=gt) is False
    assert should_trigger_step3(memory_scene_before_step2=other, memory_scene_after_step2=other, gt_scene=gt) is False
    assert should_trigger_step3(memory_scene_before_step2=other, memory_scene_after_step2=gt, gt_scene=gt) is False
    assert should_trigger_step3(
        memory_scene_before_step2=gt,
        memory_scene_after_step2=gt,
        gt_scene=gt,
        scene_reset_this_frame=True,
    ) is False


def _frame_record_for_replay(*, step2_fired: bool, include_memory_after_step1: bool = True) -> dict:
    """构造最小 v2 trajectory frame，用于 replay schema 回归测试。

    这里故意不读图片、不加载 tokenizer，只验证 jsonl schema 的硬契约：
    - v2 memory 必须显式带 road_structure；
    - step2 未触发时允许没有 step2 target；
    - step2 触发时必须带 memory_after_step1，保证 learner 能复现收窄后的
      SCENE_CHOICES，而不是偷偷回到帧首 memory。
    """

    memory = {
        "road_structure": "ROADSIDE_HAZARD",
        "scene": "Accident",
        "status": initial_event("Accident"),
        "subgoal": first_subgoal("Accident"),
        "ego_to_goal_xy": [0.0, 0.0],
    }
    frame = {
        "kind": "frame",
        "frame_idx": 0,
        "phase": "A",
        "image_paths": ["dummy.jpg"],
        "memory_before": memory,
        "memory_before_frame": memory,
        "teacher_targets": {
            "step1": "I see a roadside hazard.\nROAD_STRUCTURE: ROADSIDE_HAZARD",
            "step2": "The lane is blocked.\nSCENE: Accident" if step2_fired else None,
            "step3": None,
        },
        "student_outputs": {
            "step1": "I see a roadside hazard.\nROAD_STRUCTURE: ROADSIDE_HAZARD",
            "step2": "The lane is blocked.\nSCENE: Accident" if step2_fired else None,
            "step3": None,
        },
        "step2_fired": step2_fired,
        "step3_fired": False,
        "gt": {
            "road_structure": "ROADSIDE_HAZARD",
            "scene": "Accident",
            "status": initial_event("Accident"),
            "subgoal": first_subgoal("Accident"),
        },
    }
    if include_memory_after_step1:
        frame["memory_after_step1"] = memory
    return frame


def _check_replay_schema_v2_step2_gate() -> None:
    """replay schema v2：step2 跳过是合法样本，step2 触发缺状态则必须拒收。"""

    header = {"schema": replay.SCHEMA, "kind": "header", "frame_count": 1}

    # step1 失败时 step2/step3 都跳过，此时没有 step2 target 是合法的。
    replay.validate_trajectory([header, _frame_record_for_replay(step2_fired=False)])

    # step2 触发时，learner 必须用 collector 写下来的 memory_after_step1 重放候选表。
    replay.validate_trajectory([header, _frame_record_for_replay(step2_fired=True)])

    bad = _frame_record_for_replay(step2_fired=True, include_memory_after_step1=False)
    try:
        replay.validate_trajectory([header, bad])
    except ValueError as exc:
        assert "memory_after_step1" in str(exc)
    else:
        raise AssertionError("step2_fired=True but missing memory_after_step1 should be rejected")


def main() -> None:
    """跑全部状态机断言，全部通过则打印 ok=True。"""

    _check_init_probability_modes()
    _check_student_prompt_contracts()
    _check_delta_formula_allows_zero()
    _check_step1_update_and_trigger()
    _check_scene_flip_branches()
    _check_scene_alias_canonicalization()
    _check_step3_update()
    _check_phase_b_force_helper()
    _check_phase_b_noise_helper()
    _check_skip_correction_after_step1_skip_helper()
    _check_should_trigger_step3()
    _check_replay_schema_v2_step2_gate()

    print(json.dumps({"ok": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
