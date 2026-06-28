"""SFT v3 GT 泄露 hook 同步测试。

v3 的 prompt 实现直接 re-export v4。当前 v4 已把 ``check_gt_leak_*`` 变成
legacy no-op：teacher target 清洗和学生视角 prompt 约束负责处理私有字段泄露，
训练不再靠字面正则跳 analysis loss。这个测试用于锁住该同步关系，避免 v3 悄悄恢复
旧的 hard-target 泄露过滤逻辑。
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

from qwen3vl_local.sft_v3.prompts import check_gt_leak_scene, check_gt_leak_status_subgoal


def main() -> None:
    """验证 v3 与 v4 一样：泄露检测 hook 当前始终返回 False。

    这里仍构造“明显含答案字面”的文本，是为了确认 v3 没有维护第二套正则。
    """

    a_ok = "I see blocked lane and hazard lights ahead."
    a_bad = "This clearly indicates Accident scene."

    b_ok = "I should keep braking and maintain gap."
    b_bad = "Current status is hazard_detect and subgoal max_brake_or_min_gap."

    r = {
        "scene_no_leak": check_gt_leak_scene(a_ok, "Accident"),
        "scene_leak": check_gt_leak_scene(a_bad, "Accident"),
        "status_no_leak": check_gt_leak_status_subgoal(b_ok, "hazard_detect", "max_brake_or_min_gap"),
        "status_leak": check_gt_leak_status_subgoal(b_bad, "hazard_detect", "max_brake_or_min_gap"),
    }
    ok = all(value is False for value in r.values())
    print(json.dumps({"ok": ok, "results": r}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
