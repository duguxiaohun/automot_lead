"""SFT v5 memory 状态机小测试。"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v5.labels import RSTarget
from qwen3vl_local.sft_v5.prompts import (
    reset_memory_for_frame,
    update_memory_after_q1,
    update_memory_after_q2,
)


def main() -> None:
    # memory 的初始值只依赖当前帧 GT RS，事件默认 RE；这也是 Q1 RS 错误后
    # 下一帧恢复的状态，避免错误道路结构继续污染 Q2。
    rs = RSTarget("R1", "A", "ordinary road", 0.8, (), {"R1": 0.8})
    mem = reset_memory_for_frame(rs)
    assert mem.rs_label == "R1"
    assert mem.event_label == "RE"
    rendered_q1 = mem.format_q1_text()
    assert "BELIEVED_EVENT" not in rendered_q1, "Q1 memory 不应提前暴露 EVENT"
    assert "BELIEVED_RS: A -" not in rendered_q1, "memory 不应保存 A-E 选项编号"

    mem_with_goal = reset_memory_for_frame(rs, ego_to_goal_xy=(12.3, -1.5))
    assert "EGO_TO_GOAL_XY=(+12.3, -1.5) m" in mem_with_goal.format_q1_text()
    rendered_q2 = mem_with_goal.format_q2_text()
    assert "BELIEVED_EVENT" in rendered_q2, "Q2 memory 才需要带 EVENT"
    assert "BELIEVED_EVENT: RE -" not in rendered_q2, "memory 不应保存 RE/U-E 标签前缀"

    mem = update_memory_after_q1(mem, student_rs_label="R4", student_abnormal=True)
    assert mem.rs_label == "R4"
    # Q1 只能确认“是否异常”，不能凭空写具体 U-E*；具体事件必须由 Q2 决定。
    assert mem.event_label == "RE", "Q1 abnormal=yes 只等待 Q2，不应凭空写 UE"

    mem = update_memory_after_q2(mem, student_event_label="U-E6")
    assert mem.event_label == "U-E6"

    mem2 = update_memory_after_q2(mem, student_event_label=None)
    assert mem2.event_label == "U-E6", "Q2 非法输出不能污染当前 memory；外层负责下一帧 reset"

    mem = update_memory_after_q1(mem, student_rs_label="R4", student_abnormal=False)
    assert mem.event_label == "RE", "Q1 abnormal=no 应回到当前 RS 下 RE"
    print("[test_memory_update] ok")


if __name__ == "__main__":
    main()
