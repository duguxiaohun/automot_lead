"""SFT base prompt 快照测试。

本测试把 5 个 RS 的 Q1/Q2 展开 prompt 固化成可 review 的文本快照。prompt
措辞变化必须显式更新 `prompt_snapshots.txt`，避免 RS/EVENT 分工在后续改动中
悄悄漂移。
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

from qwen3vl_local.sft_base.labels import EVENT_CANDIDATES_BY_RS, RS_LABELS  # noqa: E402
from qwen3vl_local.sft_base.prompts import Memory, build_q1_prompt, build_q2_prompt  # noqa: E402


def _render_snapshot() -> str:
    """生成全部 RS 的固定 prompt 快照文本。"""

    blocks: list[str] = []
    for rs in RS_LABELS:
        memory = Memory(
            rs_label=rs,
            event_label=EVENT_CANDIDATES_BY_RS[rs][0],
            ego_to_goal_x=8.0,
            ego_to_goal_y=-2.0,
        )
        blocks.append(f"### Q1 {rs}\n{build_q1_prompt(memory)}")
        blocks.append(f"### Q2 {rs}\n{build_q2_prompt(memory, candidates=EVENT_CANDIDATES_BY_RS[rs])}")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    """对比 prompt 快照。"""

    expected_path = pathlib.Path(__file__).with_name("prompt_snapshots.txt")
    expected = expected_path.read_text(encoding="utf-8")
    actual = _render_snapshot()
    assert actual == expected, (
        "prompt snapshot mismatch; review the new prompt text and update "
        "qwen3vl_local/sft_base/prompt_snapshots.txt if intentional"
    )
    print("[test_prompt_snapshots] ok")


if __name__ == "__main__":
    main()
