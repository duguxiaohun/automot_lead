"""SFT v3 memory-state test wrapper.

SFT v3 imports the v4 prompt/state-machine implementation directly.  Running
the v4 test here is intentional: it enforces the shared ROAD_STRUCTURE -> SCENE
-> STATUS/SUBGOAL contract for both routes.
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
