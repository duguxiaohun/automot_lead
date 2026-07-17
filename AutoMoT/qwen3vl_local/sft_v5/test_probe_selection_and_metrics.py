"""SFT v5 定向小样本选择与大样本指标的纯 CPU 回归测试。

可直接运行 ``python qwen3vl_local/sft_v5/test_probe_selection_and_metrics.py``；测试不
加载 Qwen，适合每次修改选帧策略、case schema 或 FP/FN 分母后快速执行。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

_AUTOMOT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_AUTOMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOMOT_ROOT))

from qwen3vl_local.sft_v5.metrics import (  # noqa: E402
    StudentMetricsAccumulator,
    summarize_student_predictions,
)
from qwen3vl_local.sft_v5.probe import build_probe_selection_plan, dump_probe  # noqa: E402
from qwen3vl_local.sft_v5.train import FrameRow, SequenceRow  # noqa: E402


def _frame(frame_id: int, rs: str, event: str) -> FrameRow:
    """构造无需真实 RGB 的最小 FrameRow。"""

    return FrameRow(
        frame_id=frame_id,
        history_rgb_paths=[f"rgb/{frame_id:04d}.jpg"],
        weather_text="Clear",
        ego_to_goal_xy=(10.0, 0.0),
        rs_label=rs,
        rs_option={"R1": "A", "R2": "B", "R3": "C"}.get(rs, "A"),
        event_label=event,
        event_code="R-E1" if event == "RE" else event,
        abnormal=event != "RE",
        event_option_map={"A": "RE", "B": "U-E1"},
        regular_event_codes=["R-E1"],
        raw={"event_labels_raw": [event]},
    )


def _routes() -> list[SequenceRow]:
    """route0 同时包含 UE 起止和 RS 变换，route1 提供稳定 RE 对照。"""

    return [
        SequenceRow(
            scenario="EventRoute",
            route_id="route_event",
            run_dir="/tmp/route_event",
            split="val",
            frames=[
                _frame(0, "R1", "RE"),
                _frame(1, "R1", "U-E1"),
                _frame(2, "R1", "U-E1"),
                _frame(3, "R1", "RE"),
                _frame(4, "R2", "RE"),
                _frame(5, "R2", "RE"),
            ],
        ),
        SequenceRow(
            scenario="ControlRoute",
            route_id="route_control",
            run_dir="/tmp/route_control",
            split="val",
            frames=[_frame(0, "R3", "RE"), _frame(1, "R3", "RE")],
        ),
    ]


def test_diagnostic_selection_covers_hard_cases() -> None:
    """默认 6 个 case 应各覆盖一个核心诊断类别，而不是顺序取前 6 帧。"""

    plan = build_probe_selection_plan(
        _routes(),
        num_cases=6,
        sample_mode="diagnostic",
        context_radius=1,
        seed=7,
    )
    assert len(plan) == 6
    assert {item.primary_reason for item in plan} == {
        "ue_boundary",
        "ue_positive",
        "ue_nearby_re",
        "rs_transition",
        "rs_nearby",
        "stable_re",
    }
    assert any(item.route_id == "route_control" for item in plan), "稳定 RE 对照应优先跨 route"


def test_metric_false_positive_false_negative_contract() -> None:
    """严格指标必须让 invalid 降低 recall，并正确区分 Q1/Q2 假阳性。"""

    logs = [
        {
            "gt_rs_label": "R1", "gt_event_label": "RE", "gt_abnormal": False,
            "pred_abnormal": False, "q1_rs_correct": True, "q1_abnormal_correct": True,
            "q2_triggered": True, "pred_event_is_ue": False, "q2_event_correct": True,
            "q2_candidate_mismatch": False, "rs_transition": False, "abnormal_transition": False,
        },
        {
            "gt_rs_label": "R1", "gt_event_label": "RE", "gt_abnormal": False,
            "pred_abnormal": True, "q1_rs_correct": True, "q1_abnormal_correct": False,
            "q2_triggered": True, "pred_event_is_ue": True, "q2_event_correct": False,
            "q2_candidate_mismatch": False, "rs_transition": False, "abnormal_transition": False,
        },
        {
            "gt_rs_label": "R2", "gt_event_label": "U-E1", "gt_abnormal": True,
            "pred_abnormal": True, "q1_rs_correct": True, "q1_abnormal_correct": True,
            "q2_triggered": True, "pred_event_is_ue": True, "q2_event_correct": True,
            "q2_candidate_mismatch": False, "rs_transition": True, "abnormal_transition": True,
        },
        {
            "gt_rs_label": "R2", "gt_event_label": "U-E1", "gt_abnormal": True,
            "pred_abnormal": None, "q1_rs_correct": False, "q1_abnormal_correct": False,
            "q2_triggered": False, "pred_event_is_ue": None, "q2_event_correct": False,
            "q2_candidate_mismatch": False, "rs_transition": False, "abnormal_transition": False,
        },
    ]
    summary = summarize_student_predictions(logs)
    assert summary["abnormal_confusion"]["tp"] == 1
    assert summary["abnormal_confusion"]["fp"] == 1
    assert summary["abnormal_confusion"]["invalid"] == 1
    assert summary["abnormal_recall"] == 0.5
    assert summary["abnormal_false_positive_rate"] == 0.5
    assert summary["q2_false_positive_rate"] == 0.5
    assert summary["event_end_to_end_false_positive_rate"] == 0.5
    assert summary["event_acc_when_rs_correct"] == 2 / 3
    assert summary["ue_end_to_end_recall"] == 0.5
    assert summary["metric_definitions"]["abnormal_false_positive_rate"]["direction"] == "lower_is_better"

    streaming = StudentMetricsAccumulator()
    for row in logs:
        streaming.update(row)
    assert streaming.summary() == summary


def test_static_probe_writes_selection_and_complete_case_record() -> None:
    """静态 probe 也必须写定向计划和单文件完整输入输出合同。"""

    route = _routes()[0]
    row = {
        "scenario": route.scenario,
        "route_id": route.route_id,
        "run_dir": route.run_dir,
        "split": route.split,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "history_rgb_paths": frame.history_rgb_paths,
                "weather_text": frame.weather_text,
                "ego_to_goal_xy": list(frame.ego_to_goal_xy or (10.0, 0.0)),
                "rs_label": frame.rs_label,
                "rs_option": frame.rs_option,
                "event_label": frame.event_label,
                "event_code": frame.event_code,
                "abnormal": frame.abnormal,
                "event_option_map": frame.event_option_map,
                "regular_event_codes": frame.regular_event_codes,
                "event_labels_raw": [frame.event_label],
            }
            for frame in route.frames
        ],
    }
    with tempfile.TemporaryDirectory(prefix="sft_v5_directed_probe_") as tmp:
        root = pathlib.Path(tmp)
        index_path = root / "index.jsonl"
        index_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        output_dir = root / "probe"
        args = argparse.Namespace(
            index=str(index_path), output_dir=str(output_dir), num_cases=4,
            max_routes=0, max_frames_per_route=0, sample_mode="diagnostic",
            context_radius=1, seed=7, with_model=False, with_teacher=True,
            with_teacher_model=False, model_dir="unused", teacher_model_dir=None,
            adapter_dir=None, merge_lora=True, max_new_tokens_q1=16, max_new_tokens_q2=16,
        )
        summary = dump_probe(args)
        selection = json.loads((output_dir / "selection_plan.json").read_text(encoding="utf-8"))
        assert selection["sample_mode"] == "diagnostic"
        assert selection["selected_cases"] == 4
        case_records = list(output_dir.glob("route_*/frame_*/case_record.json"))
        assert len(case_records) == 4
        case = json.loads(case_records[0].read_text(encoding="utf-8"))
        assert set(case) == {"selection", "labels", "inputs", "targets", "outputs", "memory", "flags"}
        assert "q1_student_messages" in case["inputs"]
        assert case["flags"]["generation_limits"] == {
            "max_new_tokens_q1": 16,
            "max_new_tokens_q2": 16,
        }
        assert summary["sampling"]["selected_cases"] == 4
        assert summary["generation_limits"]["max_new_tokens_q1"] == 16


def main() -> None:
    """运行定向选帧、FP/FN 指标和完整 case 产物回归。"""

    test_diagnostic_selection_covers_hard_cases()
    test_metric_false_positive_false_negative_contract()
    test_static_probe_writes_selection_and_complete_case_record()
    print("[ok] probe selection and eval metrics")


if __name__ == "__main__":
    main()
