"""静态检查 SFT baseline 的单问直接答案监督 span。

本脚本不加载模型，只验证 `ROAD` / `EVENT` 两个值 span 能被 parser 找到，
避免 prompt 字段名改动后 loss 悄悄变成 0。
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

from qwen3vl_local.sft_baseline.labels import EventTarget, RSTarget
from qwen3vl_local.sft_baseline.prompts import (
    Memory,
    build_q1_target,
    parse_q1_output,
    target_spans_q1,
)


def _assert_nonempty(text: str, spans: dict, keys: list[str]) -> None:
    for key in keys:
        if key not in spans:
            raise AssertionError(f"missing span {key} in {text!r}")
        lo, hi = spans[key]
        if hi <= lo:
            raise AssertionError(f"empty span {key} in {text!r}")


def main() -> None:
    rs = RSTarget(
        label="R3",
        description="Highway merge/exit",
        confidence=0.9,
        secondary=(),
        candidates={"R3": 0.9},
    )
    event = EventTarget(label="U-E6", event_code="U-E6", abnormal=True, raw_events=("R-E4", "U-E6"))
    q1 = build_q1_target(rs_target=rs, event_target=event)
    assert q1 == "ROAD: HIGHWAY\nEVENT: UE"
    parsed = parse_q1_output(q1)
    assert parsed["road"] == "HIGHWAY"
    assert parsed["event"] == "UE"
    _assert_nonempty(q1, target_spans_q1(q1), ["road", "event"])

    regular = EventTarget(label="R-E1", event_code="R-E1", abnormal=False, raw_events=("R-E1",))
    q1_regular = build_q1_target(
        rs_target=RSTarget(label="R4", description="Intersection", confidence=0.9, secondary=(), candidates={"R4": 0.9}),
        event_target=regular,
    )
    assert q1_regular == "ROAD: NON_HIGHWAY\nEVENT: RE"
    assert parse_q1_output(q1_regular)["road"] == "NON_HIGHWAY"
    assert parse_q1_output(q1_regular)["event"] == "RE"
    hidden = Memory(rs_label="R4", event_label="U-E6", hide_priors=True).format_text()
    assert "PREVIOUS_ROAD" not in hidden and "PREVIOUS_EVENT" not in hidden, hidden
    assert "EGO_TO_GOAL_XY" in hidden, hidden
    print("[sft_baseline check_loss_mask] ok")


if __name__ == "__main__":
    main()



