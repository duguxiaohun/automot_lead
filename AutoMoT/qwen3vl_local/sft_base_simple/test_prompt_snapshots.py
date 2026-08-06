"""SFT base simple prompt 快照测试。

本测试把高速/非高速单问 prompt 固化成可 review 的文本快照。prompt 措辞变化
必须显式更新 `prompt_snapshots.txt`。
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

from qwen3vl_local.sft_base_simple.labels import RS_LABELS  # noqa: E402
from qwen3vl_local.sft_base_simple.prompts import Memory, build_q1_prompt  # noqa: E402


def _render_snapshot() -> str:
    """生成全部 RS 的固定 prompt 快照文本。"""

    blocks: list[str] = []
    for rs in RS_LABELS:
        memory = Memory(
            rs_label=rs,
            event_label="U-E2" if rs == "R3" else "R-E1",
            ego_to_goal_x=8.0,
            ego_to_goal_y=-2.0,
        )
        blocks.append(f"### JOINT {rs}\n{build_q1_prompt(memory)}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    """对比 prompt 快照。"""

    expected_path = pathlib.Path(__file__).with_name("prompt_snapshots.txt")
    expected = expected_path.read_text(encoding="utf-8")
    actual = _render_snapshot()
    assert actual == expected, (
        "prompt snapshot mismatch; review the new prompt text and update "
        "qwen3vl_local/sft_base_simple/prompt_snapshots.txt if intentional"
    )
    hidden = Memory(rs_label="UNKNOWN", event_label="UNKNOWN", ego_to_goal_x=8.0, ego_to_goal_y=-2.0, hide_priors=True)
    hidden_prompt = build_q1_prompt(hidden)
    assert "PREVIOUS_ROAD" not in hidden_prompt and "PREVIOUS_EVENT" not in hidden_prompt
    assert "previous memory" not in hidden_prompt.lower()
    assert "hidden for this visual check" in hidden_prompt
    print("[test_prompt_snapshots] ok")


if __name__ == "__main__":
    main()



