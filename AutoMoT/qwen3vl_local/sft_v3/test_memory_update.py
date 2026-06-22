"""SFT v3 memory 状态机测试（纯 Python）。

覆盖：

- ``init_memory`` 排除 GT scene（D3）；
- ``update_memory_after_step2`` 的 4 种翻转组合；
- ``update_memory_after_step3`` 的合法/非法 event 过滤；
- ``force_memory_to_gt_scene`` 的弱纠偏语义（D2）：scene == GT 时全 noop，
  status/subgoal 必须跨帧保留；scene != GT 时走 scene-change reset；
- ``should_trigger_step3`` 的触发条件（C4）：只看 step2 后 memory.scene 是否 = GT，
  与是否发生过翻转无关。
"""

from __future__ import annotations

import json
import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.prompt_pipeline import SCENARIO_LABELS
from qwen3vl_local.sft_v3.build_dataset import compute_delta
from qwen3vl_local.sft_v3.prompts import (
    Memory,
    first_subgoal,
    force_memory_to_gt_scene,
    get_full_sequence,
    initial_event,
    init_memory,
    should_trigger_step3,
    update_memory_after_step2,
    update_memory_after_step3,
)


def _check_init_excludes_gt() -> None:
    """D3：init_memory 在多次 seed 下都不应抽中 GT scene。"""

    if len(SCENARIO_LABELS) <= 1:
        return  # 单场景退化，无法排除
    gt = "Accident"
    for i in range(64):
        mem = init_memory(
            run_id=f"run_{i}",
            sub_scenario_id=f"sub_{i}",
            ego_to_goal_x=0.0,
            ego_to_goal_y=0.0,
            gt_scene=gt,
        )
        assert mem.scene != gt, f"init_memory 在 i={i} 抽中了 GT scene={gt}"


def _check_delta_formula_allows_zero() -> None:
    """C1：δ 严格按公式允许为 0，不能被静默抬到 1。"""

    assert compute_delta([0, 0, 1, 2, 3], warn=False) == 0
    assert compute_delta([0, 2, 6, 8, 10], warn=False) == 1
    assert compute_delta([0, 50, 100, 120, 140], warn=False) == 10


def _check_scene_flip_branches() -> None:
    """update_memory_after_step2 的 4 种 (合法/非法) × (翻转/保持) 分支。"""

    base = Memory(
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
    other = next(s for s in SCENARIO_LABELS if s != "Accident")
    m2 = update_memory_after_step2(base, student_scene=other)
    assert m2.scene == other
    assert m2.status == initial_event(other)
    assert m2.subgoal == first_subgoal(other)

    # 3) 非法 scene → 忽略
    m3 = update_memory_after_step2(base, student_scene="NotExistsScene")
    assert m3.scene == base.scene
    assert m3.status == base.status

    # 4) None → 忽略
    m4 = update_memory_after_step2(base, student_scene=None)
    assert m4.scene == base.scene


def _check_step3_update() -> None:
    """update_memory_after_step3 只接受当前 scene 事件序列中的合法 event。"""

    scene = "Accident"
    seq = get_full_sequence(scene)
    base = Memory(
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
    """D2：Phase B 弱纠偏。scene == GT 全 noop；scene != GT 才 reset。"""

    gt = "Accident"
    other = next(s for s in SCENARIO_LABELS if s != gt)

    # (a) scene 已等于 GT，status/subgoal 已经被 step3 推进过 → 必须保留
    seq = get_full_sequence(gt)
    advanced_status = seq[1] if len(seq) > 1 else seq[0]
    advanced_subgoal = seq[2] if len(seq) > 2 else seq[-1]
    mem_ok = Memory(
        scene=gt,
        status=advanced_status,
        subgoal=advanced_subgoal,
        ego_to_goal_x=5.0,
        ego_to_goal_y=-1.0,
    )
    forced_ok = force_memory_to_gt_scene(mem_ok, gt_scene=gt)
    assert forced_ok.scene == gt
    assert forced_ok.status == advanced_status, "Phase B 在 scene==GT 时不应重置 status"
    assert forced_ok.subgoal == advanced_subgoal, "Phase B 在 scene==GT 时不应重置 subgoal"

    # (b) scene 与 GT 不一致 → 走 scene-change reset
    mem_bad = Memory(
        scene=other,
        status=initial_event(other),
        subgoal=first_subgoal(other),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )
    forced_bad = force_memory_to_gt_scene(mem_bad, gt_scene=gt)
    assert forced_bad.scene == gt
    assert forced_bad.status == initial_event(gt)
    assert forced_bad.subgoal == first_subgoal(gt)


def _check_should_trigger_step3() -> None:
    """C4：step3 触发判定与 step2 是否翻转无关，只看 scene_after_step2 == gt_scene。"""

    gt = "Accident"
    other = next(s for s in SCENARIO_LABELS if s != gt)

    # 4 种组合（初始 memory.scene、step2 输出）：
    # | 初始 | step2 输出 | 翻转 | scene_after_step2 | 期望触发 |
    # |------|-----------|------|-------------------|---------|
    # | GT   | GT        | 否   | GT                | 是       |
    # | GT   | other     | 是   | other             | 否       |
    # | other| other     | 否   | other             | 否       |
    # | other| GT        | 是   | GT                | 是       |
    assert should_trigger_step3(memory_scene_after_step2=gt, gt_scene=gt) is True
    assert should_trigger_step3(memory_scene_after_step2=other, gt_scene=gt) is False
    assert should_trigger_step3(memory_scene_after_step2=other, gt_scene=gt) is False
    assert should_trigger_step3(memory_scene_after_step2=gt, gt_scene=gt) is True


def main() -> None:
    """跑全部状态机断言，全部通过则打印 ok=True。"""

    _check_init_excludes_gt()
    _check_delta_formula_allows_zero()
    _check_scene_flip_branches()
    _check_step3_update()
    _check_phase_b_force_helper()
    _check_should_trigger_step3()

    print(json.dumps({"ok": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
