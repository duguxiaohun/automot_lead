"""SFT v3 loss/mask 检查入口。

v3 与 v4 共用 prompt 和 supervised span 契约。本文件故意只做很薄的一层 wrapper：
真正的 ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL mask 检查在 v4 checker 中维护。这样
target span 一旦变动，只需要改一处测试实现，v3 运行本入口就能同步覆盖。
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

from qwen3vl_local.sft_v4.check_loss_mask import main


if __name__ == "__main__":
    main()
