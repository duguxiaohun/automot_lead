"""SFT v3 memory 状态机测试入口。

SFT v3 直接 import v4 的 prompt / Memory / 状态机实现。因此这里故意复用 v4 测试：
只要 v4 的 ROAD_STRUCTURE -> SCENE -> STATUS/SUBGOAL 契约发生变化，v3 也必须在
同一个测试入口下通过，不能悄悄维护第二份状态机。
"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v4.test_memory_update import main


if __name__ == "__main__":
    main()
