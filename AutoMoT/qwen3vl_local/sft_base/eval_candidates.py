"""SFT base eval 的 Q2 候选构造。

候选构造不依赖模型或 torch，便于单独测试“按学生 RS 而不是 GT RS 出题”的合同。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from qwen3vl_local.sft_base.labels import (
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
    """采用逐帧 allowed_events，不再用学生 RS 静态表二次过滤。

    allowed_events 是 collector 给出的帧级事实，可能包含静态表没有覆盖的真实组合
    （如路口静止障碍、窄路行人横穿）。只要它存在，就完整保留；只有缺失时才在
    调用方回落到 `scenario_event_candidates ∩ EVENT_CANDIDATES_BY_RS[pred_rs]`。
    """

    del pred_rs
    return [str(code) for code in allowed]


def q2_candidates_for_student_rs(
    frame: Any,
    pred_rs: Optional[str],
    *,
    seed: int,
) -> Tuple[List[str], List[str], str, bool]:
    """生成 Q2 候选：逐帧 allowed_events 优先，缺失时按学生 RS 静态表 fallback。"""

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
