"""静态检查 SFT base 的直接答案监督 span。

本脚本不加载模型，只验证 Q1/Q2 target 中的 RS/ABNORMAL/EVENT 值 span 能被
parser 找到，避免 prompt 字段名改动后 loss 悄悄变成 0。
"""

from __future__ import annotations

import pathlib
import re
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base.labels import EventTarget, RSTarget
from qwen3vl_local.sft_base.prompts import (
    build_q1_target,
    build_q2_target,
    parse_q1_output,
    parse_q2_output,
    reset_memory_for_frame,
    target_spans_q1,
    target_spans_q2,
)


def _has_letter_answer(text: str, field: str) -> bool:
    """检测 `RS: D` 这类旧 A/B/C 单字母答案是否复活。"""

    return re.search(rf"(?im)^\s*{field}\s*:\s*[A-Z]\s*$", text or "") is not None


def _assert_nonempty(text: str, spans: dict, keys: list[str]) -> None:
    for key in keys:
        if key not in spans:
            raise AssertionError(f"missing span {key} in {text!r}")
        lo, hi = spans[key]
        if hi <= lo:
            raise AssertionError(f"empty span {key} in {text!r}")


def main() -> None:
    rs = RSTarget(
        label="R4",
        description="Signalized intersection",
        confidence=0.9,
        secondary=(),
        candidates={"R4": 0.9},
    )
    event = EventTarget(label="U-E6", event_code="U-E6", abnormal=True, raw_events=("R-E4", "U-E6"))
    q1 = build_q1_target(rs_target=rs, event_target=event)
    # 答案必须是语义 token；任何单字母选项形式都视为回退到旧协议。
    assert "SIGNAL_INTERSECTION" in q1
    assert not _has_letter_answer(q1, "RS"), q1
    assert parse_q1_output(q1)["rs_label"] == "R4"
    _assert_nonempty(q1, target_spans_q1(q1), ["rs", "abnormal"])
    memory = reset_memory_for_frame(rs)
    candidates = ["RE", "U-E6"]
    q2 = build_q2_target(memory, candidates=candidates, event_target=event)
    assert "RULE_VIOLATION" in q2
    assert not _has_letter_answer(q2, "EVENT"), q2
    assert parse_q2_output(q2, candidates)["event_label"] == "U-E6"
    _assert_nonempty(q2, target_spans_q2(q2), ["event"])

    # 候选外的合法全局 token 必须被判为非法，否则 eval 会把 off-candidate 输出算成有效答案。
    assert parse_q2_output("EVENT: LEAD_BRAKE", candidates)["event_label"] is None
    print("[sft_base check_loss_mask] ok")


if __name__ == "__main__":
    main()

