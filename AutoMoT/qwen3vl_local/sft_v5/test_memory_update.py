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
    rs = RSTarget("R1", "A", "ordinary road", 0.8, (), {"R1": 0.8})
    mem = reset_memory_for_frame(rs)
    assert mem.rs_label == "R1"
    assert mem.event_label == "RE"

    mem = update_memory_after_q1(mem, student_rs_label="R4", student_abnormal=True)
    assert mem.rs_label == "R4"
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
