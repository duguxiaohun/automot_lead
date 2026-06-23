"""SFT v4 GT 泄露过滤测试。"""

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

from qwen3vl_local.sft_v4.prompts import check_gt_leak_scene, check_gt_leak_status_subgoal


def main() -> None:
    """验证 teacher 分析文本的 GT 字面泄露检测。

    泄露检测只影响分析 loss：如果 teacher 分析直接说出 GT scene/status/subgoal token，
    对应 L_A2/L_A3 会跳过，但离散值 token loss 仍保留。
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
    ok = (r["scene_no_leak"] is False and r["scene_leak"] is True and r["status_no_leak"] is False and r["status_leak"] is True)
    print(json.dumps({"ok": ok, "results": r}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

