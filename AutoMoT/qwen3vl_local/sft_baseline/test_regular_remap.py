"""SFT baseline regular remap 口径测试。

重点守住 UE 与 regular 的边界：同一帧即使带有 regular 原始标注，只要最终 GT
是 UE，就不能计入 pure-regular remap。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_baseline.audit_rs_event_cooccurrence import audit  # noqa: E402
from qwen3vl_local.sft_baseline.labels import (  # noqa: E402
    EVENT_CANDIDATES_BY_RS,
    REGULAR_EVENT_LABELS,
    RS_LABELS,
    RS_REGULAR_EVENTS,
    canonical_regular_event_for_rs,
    resolve_event_target,
)


def _frame(rs: str, events: list[str], primary: str) -> dict:
    """构造最小 annotation。"""

    return {
        "frame_id": 0,
        "frame_rs_annotation": {"label": rs},
        "frame_event_annotation": {"events": events, "label": primary},
    }


def test_mapping_rules() -> None:
    """逐条验证用户指定的 RS canonical regular 映射。"""

    cases = [
        ("R5", "R-E4", "R-E5"),
        ("R1", "R-E4", "R-E1"),
        ("R3", "R-E4", "R-E1"),
        ("R4", "R-E1", "R-E4"),
    ]
    for rs, raw, expected in cases:
        target = resolve_event_target(_frame(rs, [raw], raw), rs_label=rs)
        assert target.label == expected, (rs, raw, target)
        assert target.event_code == raw, target
        assert target.regular_event_codes == (raw,), target


def test_ue_frame_keeps_regular_audit_but_skips_remap() -> None:
    """UE 帧可以保留 raw regular 标注，但不能进入 pure-regular remap。"""

    target = resolve_event_target(_frame("R4", ["R-E4", "U-E8"], "R-E4"), rs_label="R4")
    assert target.label == "U-E8", target
    assert target.abnormal is True, target
    assert target.event_code == "U-E8", target
    assert target.raw_events == ("R-E4", "U-E8"), target
    assert target.regular_event_codes == ("R-E4",), target

    with tempfile.TemporaryDirectory() as tmp:
        collection = pathlib.Path(tmp)
        payload = {
            "scenario": "Mini",
            "routes": [
                {
                    "route_id": "route0",
                    "status": "success",
                    "annotations": [
                        _frame("R4", ["R-E4", "U-E8"], "R-E4"),
                        _frame("R5", ["R-E4"], "R-E4"),
                        _frame("R1", ["R-E4"], "R-E4"),
                    ],
                }
            ],
        }
        (collection / "Mini_result.json").write_text(json.dumps(payload), encoding="utf-8")
        report = audit(
            argparse.Namespace(
                collection_dir=str(collection),
                min_count=1,
                min_rate=0.0,
                top_k=5,
                focus_combo="R4:R-E4",
            )
        )

    assert report["frames_with_regular_annotation_by_rs"]["R4"] == 1, report
    assert report["pure_regular_frames_by_rs"]["R4"] == 0, report
    assert report["raw_regular_remap_by_rs"]["R4"] == 0, report
    assert report["raw_regular_remap_total"] == 2, report
    assert abs(report["raw_regular_remap_rate"] - (2.0 / 3.0)) < 1e-9, report
    assert report["raw_regular_remap_rate_over_pure_regular"] == 1.0, report
    combos = {(row["rs"], row["raw_event"], row["mapped_event"]) for row in report["raw_regular_remap_breakdown"]}
    assert ("R4", "R-E4", "R-E4") not in combos, report["raw_regular_remap_breakdown"]
    assert ("R5", "R-E4", "R-E5") in combos, report["raw_regular_remap_breakdown"]
    assert ("R1", "R-E4", "R-E1") in combos, report["raw_regular_remap_breakdown"]


def test_mapped_regular_stays_in_rs_candidates() -> None:
    """任意 raw regular 经过 RS 映射后必须仍在该 RS 静态候选内。"""

    for rs in RS_LABELS:
        for raw in REGULAR_EVENT_LABELS:
            mapped = canonical_regular_event_for_rs(rs, raw)
            assert mapped in RS_REGULAR_EVENTS[rs], (rs, raw, mapped)
            assert mapped in EVENT_CANDIDATES_BY_RS[rs], (rs, raw, mapped)


def main() -> None:
    """脚本入口，方便远端直接 python 执行。"""

    test_mapping_rules()
    test_ue_frame_keeps_regular_audit_but_skips_remap()
    test_mapped_regular_stays_in_rs_candidates()
    print("[test_regular_remap] ok")


if __name__ == "__main__":
    main()


