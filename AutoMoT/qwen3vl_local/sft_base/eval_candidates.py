"""SFT base eval 的 Q2 候选构造。

候选构造不依赖模型或 torch，便于单独测试“按学生 RS 而不是 GT RS 出题”的合同。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from qwen3vl_local.sft_base.labels import (
    EVENT_CANDIDATES_BY_RS,
    EVENT_ORDER,
    RS_LABELS,
    allowed_events_from_frame,
    collapse_regular_to_re,
    event_in_candidates,
    q2_raw_candidates,
    stable_event_choice_order,
)


def _dataset_event_candidates(frame: Any, raw_dict: dict[str, Any]) -> List[str]:
    """读取 build_dataset 预生成的 Q2 展示顺序。"""

    attr = getattr(frame, "event_candidates", None)
    if attr:
        return [str(item) for item in attr]
    return [str(item) for item in (raw_dict.get("event_candidates_ordered") or [])]


def _filter_allowed_events_for_pred_rs(allowed: List[str], pred_rs: str) -> List[str]:
    """按学生 RS 过滤逐帧 allowed_events。

    逐帧 allowed_events 里的 regular code 已经是 collector clamp / overlay 的最终结果，
    不再用静态 RS 表过滤；所有 R-E* 都保留并在 prompt 中折成 RE。UE 才是真正与
    RS 语境绑定的候选，只对 U-E* 做静态表 gate。
    """

    pred_rs_raw_set = set(EVENT_CANDIDATES_BY_RS.get(str(pred_rs), []))
    raw: List[str] = []
    for code in allowed:
        text = str(code)
        if text.startswith("R-E"):
            raw.append(text)
        elif text.startswith("U-E") and text in pred_rs_raw_set:
            raw.append(text)
    return raw


def q2_candidates_for_student_rs(
    frame: Any,
    pred_rs: Optional[str],
    *,
    seed: int,
) -> Tuple[List[str], List[str], str, bool]:
    """按学生 RS 生成 Q2 候选，避免 GT RS 候选泄漏。"""

    if pred_rs not in RS_LABELS:
        return ["RE"], [], "invalid_rs_fallback", False
    raw_dict = getattr(frame, "raw", frame if isinstance(frame, dict) else {})
    scenario_candidates = [str(code) for code in (raw_dict.get("scenario_event_candidates") or list(EVENT_ORDER))]
    dataset_candidates = _dataset_event_candidates(frame, raw_dict)
    allowed_raw = _filter_allowed_events_for_pred_rs(allowed_events_from_frame(raw_dict), str(pred_rs))
    source = "pred_rs_allowed_events"
    raw = allowed_raw
    if not raw:
        raw = q2_raw_candidates(scenario_candidates, str(pred_rs))
        source = "pred_rs_static_candidates"
    display = collapse_regular_to_re(raw, str(pred_rs))
    if dataset_candidates and set(display) == set(dataset_candidates):
        ordered = list(dataset_candidates)
    else:
        ordered = stable_event_choice_order(
            run_id=str(raw_dict.get("run_id") or raw_dict.get("route_id") or ""),
            frame_id=int(getattr(frame, "frame_id", raw_dict.get("frame_id", 0))),
            rs_label=str(pred_rs),
            scenario_candidates=scenario_candidates,
            raw_candidates=raw,
            seed=int(seed),
        )
    if set(ordered) != set(display):
        raise AssertionError(f"candidate order set mismatch: ordered={ordered} display={display}")
    regular_codes = [str(code) for code in raw if str(code).startswith("R-E")]
    gt_event = getattr(frame, "event_label", raw_dict.get("event_label"))
    return ordered, regular_codes, source, event_in_candidates(gt_event, ordered)
