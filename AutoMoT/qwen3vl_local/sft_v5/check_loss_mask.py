"""静态检查 SFT v5 的监督 span。

本脚本不加载模型，只验证 Q1/Q2 teacher target 中的结构化分析段、
RS/ABNORMAL/EVENT 字符 span 能被 prompt parser 找到，避免后续 prompt 改动导致
离散标签 loss 变成 0。
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

from qwen3vl_local.sft_v5.labels import RSTarget, EventTarget
from qwen3vl_local.sft_v5.prompts import (
    build_q1_teacher_target,
    build_q2_teacher_target,
    reset_memory_for_frame,
    target_spans_q1,
    target_spans_q2,
)


def _assert_nonempty(text: str, spans: dict, keys: list[str]) -> None:
    """检查指定 span 是否存在且非空。"""

    for key in keys:
        if key not in spans:
            raise AssertionError(f"missing span {key} in {text!r}")
        lo, hi = spans[key]
        if hi <= lo:
            raise AssertionError(f"empty span {key} in {text!r}")


def main() -> None:
    # 这个脚本使用手工构造的最小 target，不依赖数据集和模型。
    # 它主要防止后续改 prompt 时把 "RS:" / "EVENT:" 等字段名改掉，
    # 导致训练里的 span 定位失败而离散标签 loss 悄悄变成 0。
    rs = RSTarget(
        label="R4",
        option="D",
        description="Signalized intersection",
        confidence=0.9,
        secondary=(),
        candidates={"R4": 0.9},
    )
    event = EventTarget(label="U-E6", event_code="U-E6", abnormal=True, raw_events=("R-E4", "U-E6"))
    q1 = build_q1_teacher_target(rs_target=rs, event_target=event, weather_text="clear daytime weather")
    # Q1 必须同时监督分析、RS 选择和 ABNORMAL 判断。
    _assert_nonempty(q1, target_spans_q1(q1), ["analysis", "rs", "abnormal"])
    memory = reset_memory_for_frame(rs)
    q2 = build_q2_teacher_target(memory, option_map={"A": "RE", "B": "U-E6"}, event_target=event)
    # Q2 必须同时监督分析和 EVENT 选择。
    _assert_nonempty(q2, target_spans_q2(q2), ["analysis", "event"])
    print("[check_loss_mask] ok")


if __name__ == "__main__":
    main()
