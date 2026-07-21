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
    build_transition_fields,
    build_transition_report,
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


def test_random_selection_is_seeded() -> None:
    """random 必须按 seed 复现同一段连续帧，而不是全数据集散点抽样。"""

    first = build_probe_selection_plan(
        _routes(),
        num_cases=4,
        sample_mode="random",
        context_radius=1,
        seed=7,
        sequence_length=4,
    )
    second = build_probe_selection_plan(
        _routes(), num_cases=4, sample_mode="random", context_radius=1, seed=7,
        sequence_length=4,
    )
    assert [(x.route_index, x.frame_index) for x in first] == [
        (x.route_index, x.frame_index) for x in second
    ]
    assert {item.primary_reason for item in first} == {"random"}
    assert len({item.route_index for item in first}) == 1
    frame_indices = [item.frame_index for item in first]
    assert frame_indices == list(range(frame_indices[0], frame_indices[0] + 4))


def test_rs_transition_selection_is_one_contiguous_change_window() -> None:
    """RS 专项必须返回变化前帧、新 RS 首帧和变化后帧。"""

    plan = build_probe_selection_plan(
        _routes(), num_cases=3, sample_mode="rs_transition", context_radius=2, seed=7
    )
    assert [(item.route_id, item.frame_id) for item in plan] == [
        ("route_event", 3),
        ("route_event", 4),
        ("route_event", 5),
    ]
    assert [item.primary_reason for item in plan] == [
        "rs_before_transition",
        "rs_transition",
        "rs_after_transition",
    ]


def test_ue_transition_selection_contains_entry_and_exit() -> None:
    """UE 专项必须从进入前 RE 跟到 UE 末帧和退出后首个 RE。"""

    plan = build_probe_selection_plan(
        _routes(), num_cases=4, sample_mode="ue_transition", context_radius=1, seed=7
    )
    assert [(item.route_id, item.frame_id) for item in plan] == [
        ("route_event", 0),
        ("route_event", 1),
        ("route_event", 2),
        ("route_event", 3),
    ]
    assert [item.primary_reason for item in plan] == [
        "ue_before_entry",
        "ue_entry",
        "ue_last_frame",
        "ue_exit",
    ]


def test_ue_transition_keeps_full_long_span_beyond_budget() -> None:
    """长 UE 必须保留完整持续区间和前后 RE，不能被 num_cases 从中间截断。"""

    long_route = SequenceRow(
        scenario="LongEventRoute",
        route_id="route_long_event",
        run_dir="/tmp/route_long_event",
        split="val",
        frames=[
            _frame(0, "R1", "RE"),
            *[_frame(frame_id, "R1", "U-E1") for frame_id in range(1, 7)],
            _frame(7, "R1", "RE"),
        ],
    )
    plan = build_probe_selection_plan(
        [long_route],
        num_cases=4,
        sample_mode="ue_transition",
        context_radius=1,
        seed=7,
    )
    assert [item.frame_id for item in plan] == list(range(8))
    assert plan[0].primary_reason == "ue_before_entry"
    assert plan[1].primary_reason == "ue_entry"
    assert plan[-2].primary_reason == "ue_last_frame"
    assert plan[-1].primary_reason == "ue_exit"


def test_transition_modes_do_not_fill_with_unrelated_frames() -> None:
    """专项数据不存在时应返回空计划，不能用稳定 RE 冒充变化帧。"""

    stable_routes = [_routes()[1]]
    assert build_probe_selection_plan(
        stable_routes, num_cases=4, sample_mode="rs_transition", context_radius=1, seed=7
    ) == []
    assert build_probe_selection_plan(
        stable_routes, num_cases=4, sample_mode="ue_transition", context_radius=1, seed=7
    ) == []


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


def test_transition_detection_confusion_and_report() -> None:
    """变化帧指标必须区分 RS/UE 的 TP、FP、FN，并能生成轻量对比报告。"""

    gt_rs = ["R1", "R2", "R2", "R3", "R3"]
    pred_rs = ["R1", "R2", "R1", "R1", "R1"]
    gt_abnormal = [False, True, True, False, False]
    pred_abnormal = [False, True, False, False, True]
    logs = []
    for index in range(len(gt_rs)):
        transitions = build_transition_fields(
            pair_evaluated=index > 0,
            previous_frame_id=index - 1 if index > 0 else None,
            previous_gt_rs_label=gt_rs[index - 1] if index > 0 else None,
            gt_rs_label=gt_rs[index],
            previous_pred_rs_label=pred_rs[index - 1] if index > 0 else None,
            pred_rs_label=pred_rs[index],
            previous_gt_abnormal=gt_abnormal[index - 1] if index > 0 else None,
            gt_abnormal=gt_abnormal[index],
            previous_pred_abnormal=pred_abnormal[index - 1] if index > 0 else None,
            pred_abnormal=pred_abnormal[index],
        )
        logs.append(
            {
                "scenario": "TransitionRoute",
                "route_id": "route_transition",
                "frame_id": index,
                "gt_rs_label": gt_rs[index],
                "pred_rs_label": pred_rs[index],
                "gt_abnormal": gt_abnormal[index],
                "pred_abnormal": pred_abnormal[index],
                "gt_event_label": "U-E1" if gt_abnormal[index] else "RE",
                "q1_rs_correct": gt_rs[index] == pred_rs[index],
                "q1_abnormal_correct": gt_abnormal[index] == pred_abnormal[index],
                "q2_triggered": False,
                "q2_event_correct": False,
                "reset_next": False,
                "rs_transition": bool(transitions["gt_rs_change"]),
                "abnormal_transition": bool(
                    transitions["gt_ue_entry"] or transitions["gt_ue_exit"]
                ),
                **transitions,
            }
        )

    summary = summarize_student_predictions(logs)
    assert summary["rs_change_confusion"]["tp"] == 1
    assert summary["rs_change_confusion"]["fp"] == 1
    assert summary["rs_change_confusion"]["fn"] == 1
    assert summary["rs_change_detection_precision"] == 0.5
    assert summary["rs_change_detection_recall"] == 0.5
    assert summary["rs_change_false_positive_rate"] == 0.5
    assert summary["ue_entry_confusion"]["tp"] == 1
    assert summary["ue_entry_confusion"]["fp"] == 1
    assert summary["ue_entry_detection_precision"] == 0.5
    assert summary["ue_entry_detection_recall"] == 1.0
    assert summary["ue_exit_confusion"]["fp"] == 1
    assert summary["ue_exit_confusion"]["fn"] == 1
    assert summary["ue_exit_detection_recall"] == 0.0

    report = build_transition_report(logs, summary=summary)
    assert report["evaluated_pairs"] == 4
    assert report["informative_cases"] == 4
    assert {case["rs_change_outcome"] for case in report["cases"]} >= {"TP", "FP", "FN"}


def test_static_probe_compact_and_full_artifacts() -> None:
    """默认只写 results.json，full 模式仍保留完整逐帧输入输出合同。"""

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
            max_routes=0, max_frames_per_route=0, sample_mode="random",
            context_radius=1, sequence_length=4, artifact_level="compact",
            seed=7, with_model=False, with_teacher=True,
            with_teacher_model=False, model_dir="unused", teacher_model_dir=None,
            adapter_dir=None, merge_lora=True, max_new_tokens_q1=16, max_new_tokens_q2=16,
        )
        summary = dump_probe(args)
        assert [path.name for path in output_dir.iterdir()] == ["results.json"]
        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        selection = results["sampling"]
        assert selection["sample_mode"] == "random"
        assert "连续短片段" in selection["sample_mode_description"]
        assert selection["selected_cases"] == 4
        assert selection["sequence_length"] == 4
        assert all(
            case["primary_reason_description"] == "随机连续片段中的帧"
            for case in selection["cases"]
        )
        assert len(results["frames"]) == 4
        frame_indices = [item["frame_index"] for item in results["frames"]]
        assert frame_indices == list(range(frame_indices[0], frame_indices[0] + 4))
        assert results["transition_report"]["student_enabled"] is False
        assert results["transition_report"]["evaluated_pairs"] == 0
        assert summary["sampling"]["selected_cases"] == 4
        assert summary["generation_limits"]["max_new_tokens_q1"] == 16

        # full 是显式深度审计开关，不应被默认 compact 的收敛产物删除。
        full_output_dir = root / "probe_full"
        args.output_dir = str(full_output_dir)
        args.artifact_level = "full"
        dump_probe(args)
        assert (full_output_dir / "results.json").exists()
        assert (full_output_dir / "selection_plan.json").exists()
        assert (full_output_dir / "summary.json").exists()
        assert (full_output_dir / "transition_report.json").exists()
        case_records = list(full_output_dir.glob("route_*/frame_*/case_record.json"))
        assert len(case_records) == 4
        case = json.loads(case_records[0].read_text(encoding="utf-8"))
        assert set(case) == {"selection", "labels", "inputs", "targets", "outputs", "memory", "flags"}
        assert "q1_student_messages" in case["inputs"]
        assert case["selection"]["sample_mode_description"]
        assert case["selection"]["primary_reason_description"] == "随机连续片段中的帧"
        assert case["flags"]["generation_limits"] == {
            "max_new_tokens_q1": 16,
            "max_new_tokens_q2": 16,
        }


def main() -> None:
    """运行定向选帧、FP/FN 指标和完整 case 产物回归。"""

    test_random_selection_is_seeded()
    test_rs_transition_selection_is_one_contiguous_change_window()
    test_ue_transition_selection_contains_entry_and_exit()
    test_ue_transition_keeps_full_long_span_beyond_budget()
    test_transition_modes_do_not_fill_with_unrelated_frames()
    test_metric_false_positive_false_negative_contract()
    test_transition_detection_confusion_and_report()
    test_static_probe_compact_and_full_artifacts()
    print("[ok] probe selection and eval metrics")


if __name__ == "__main__":
    main()
