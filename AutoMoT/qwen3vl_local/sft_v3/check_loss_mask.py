"""SFT v3 loss/mask check.

v3 uses the same prompt and supervised span contract as v4.  Keep this wrapper
thin so ROAD_STRUCTURE/SCENE/STATUS/SUBGOAL mask changes are made once in the
v4 checker and immediately apply to v3 as well.
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
