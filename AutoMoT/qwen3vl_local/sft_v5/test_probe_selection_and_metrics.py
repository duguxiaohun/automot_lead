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
from qwen3vl_local.sft_v5.probe import (  # noqa: E402
    build_memory_recovery_report,
    build_probe_selection_plan,
    dump_probe,
)
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
    """random 必须按 seed 复现同一个完整 route ID 的全部帧。"""

    first = build_probe_selection_plan(
        _routes(),
        num_cases=4,
        sample_mode="random",
        context_radius=1,
        seed=7,
        sequence_length=4,
        num_routes=1,
    )
    second = build_probe_selection_plan(
        _routes(), num_cases=4, sample_mode="random", context_radius=1, seed=7,
        sequence_length=4,
        num_routes=1,
    )
    assert [(x.route_index, x.frame_index) for x in first] == [
        (x.route_index, x.frame_index) for x in second
    ]
    assert {item.primary_reason for item in first} == {"random"}
    assert len({item.route_index for item in first}) == 1
    selected_route = _routes()[first[0].route_index]
    frame_indices = [item.frame_index for item in first]
    assert frame_indices == list(range(len(selected_route.frames)))
    assert [item.frame_id for item in first] == [
        frame.frame_id for frame in selected_route.frames
    ]

    # num_cases/sequence_length 都不能把随机 ID 截成短片段。
    tiny_budget = build_probe_selection_plan(
        _routes(),
        num_cases=1,
        sample_mode="random",
        context_radius=1,
        seed=7,
        sequence_length=1,
        num_routes=1,
    )
    assert len(tiny_budget) == len(selected_route.frames)


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


def test_long_context_observes_delayed_correction_window() -> None:
    """RS 变化后必须继续保留多帧，不能只测变化首帧就判定模型不会纠正。"""

    route = SequenceRow(
        scenario="LongRSRoute",
        route_id="route_long_rs",
        run_dir="/tmp/route_long_rs",
        split="val",
        frames=[
            *[_frame(frame_id, "R1", "RE") for frame_id in range(5)],
            *[_frame(frame_id, "R2", "RE") for frame_id in range(5, 18)],
        ],
    )
    plan = build_probe_selection_plan(
        [route],
        num_cases=17,
        sample_mode="rs_transition",
        context_radius=8,
        seed=7,
    )
    selected = [item.frame_id for item in plan]
    assert 4 in selected and 5 in selected
    assert max(selected) >= 13, "RS 变化后至少继续观察 8 帧"


def test_memory_recovery_report_tracks_delayed_student_repair() -> None:
    """reference 只做比较；学生延迟两帧改对时必须报告 delay=2。"""

    logs = []
    for frame_id, rs_match, event_match in (
        (0, True, True),
        (1, False, False),
        (2, False, False),
        (3, True, True),
        (4, True, True),
    ):
        logs.append(
            {
                "scenario": "RecoveryRoute",
                "route_id": "route_recovery",
                "frame_id": frame_id,
                "selection_gap_reset": frame_id == 0,
                "rs_transition": frame_id == 1,
                "abnormal_transition": frame_id == 1,
                "gt_rs_label": "R2" if frame_id >= 1 else "R1",
                "gt_event_label": "U-E1" if frame_id >= 1 else "RE",
                "gt_abnormal": frame_id >= 1,
                "memory_rs_matches_after_q1": rs_match,
                "memory_event_matches_after_q2": event_match,
            }
        )
    report = build_memory_recovery_report(logs)
    assert report["rs_change_cases"][0]["recovery_delay_frames"] == 2
    assert report["event_change_cases"][0]["recovery_delay_frames"] == 2
    assert report["summary"]["recovered_cases"] == 2
    assert report["summary"]["not_recovered_cases"] == 0


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
    # 两个已知错误 memory case：一个继续复制，一个自主恢复，copy/recovery 都应为 0.5。
    logs[0].update(
        memory_rs_input_known_wrong=True,
        memory_rs_copied_when_wrong=True,
        memory_event_input_known_wrong=True,
        memory_event_copied_when_wrong=True,
    )
    logs[1].update(
        memory_rs_input_known_wrong=True,
        memory_rs_recovered=True,
        memory_event_input_known_wrong=True,
        memory_event_recovered=True,
    )
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
    assert summary["rs_wrong_memory_copy_rate"] == 0.5
    assert summary["rs_wrong_or_unknown_memory_recovery_rate"] == 0.5
    assert summary["event_wrong_memory_copy_rate"] == 0.5
    assert summary["event_wrong_or_unknown_memory_recovery_rate"] == 0.5
    assert summary["q2_skipped_rs_wrong"] == 1
    assert summary["q2_skip_due_rs_rate"] == 0.25
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


def test_static_probe_compact_review_and_full_artifacts() -> None:
    """compact 只写结果；review 每帧保存 RGB/输入/输出/memory。"""

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
        rgb_path = root / "source_rgb.jpg"
        rgb_path.write_bytes(b"fake-jpeg-for-copy-contract")
        for frame_payload in row["frames"]:
            frame_payload["history_rgb_paths"] = [str(rgb_path)]
        index_path = root / "index.jsonl"
        index_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        output_dir = root / "probe"
        args = argparse.Namespace(
            index=str(index_path), output_dir=str(output_dir), num_cases=4, num_routes=1,
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
        assert "完整 route ID" in selection["sample_mode_description"]
        assert selection["selected_cases"] == len(route.frames)
        assert selection["sequence_length"] == 4
        assert selection["sequence_length_ignored_for_random"] is True
        assert selection["requested_routes"] == 1
        assert "cases" not in selection, "顶层 results 不应重复逐帧选择记录"
        assert len(results["frames"]) == len(route.frames)
        frame_indices = [item["frame_index"] for item in results["frames"]]
        assert frame_indices == list(range(len(route.frames)))
        compact_case = results["frames"][0]
        # 默认从 UNKNOWN 启动时，第 0 帧给出新 RS 后会进入一次无 GT 的确认态，
        # 因此第 1 帧仍是 RS_SLOW；第 2 帧确认完成后才是首个纯 EVENT_FAST。
        # 这里显式按 schedule_reason 找 fast frame，避免以后调整确认长度时把
        # “第几帧”这种实现细节误当成 prompt/KV 合同本身。
        compact_fast_case = next(
            item
            for item in results["frames"]
            if item["memory"]["q1"]["schedule_reason"] == "reuse_stable_rs"
        )
        # compact 只减少文件数量，不能删掉训练前 base / 训练后 LoRA 人工对比所需的
        # 输入、完整标签、teacher 脚本真值、输出槽位和 memory。
        assert compact_case["ground_truth"]["rs_label"] in {"R1", "R2"}
        assert "resolved_event_target" in compact_case["ground_truth"]
        assert "q1_student_messages" in compact_case["inputs"]
        assert "q1_teacher_messages" in compact_case["inputs"]
        assert "q2_student_user_turn" in compact_case["inputs"]
        assert compact_case["inputs"]["q1_student_messages"][0]["role"] == "system"
        assert compact_case["inputs"]["q2_student_user_turn"]["continued_from"] == "student.q1_output_kv"
        assert compact_fast_case["inputs"]["q2_student_user_turn"]["continued_from"] == "fresh_rgb_prefill"
        assert compact_fast_case["teacher_targets"]["q1"] == ""
        assert compact_case["teacher_targets"]["q1"].startswith("Scene Description:")
        assert compact_case["teacher_targets"]["q2_training"].startswith("Scene Description:")
        assert set(compact_case["memory"]) >= {
            "before", "after", "q1", "q2", "next_frame", "autonomous_change"
        }
        assert compact_case["memory"]["reference_is_comparison_only"] is True
        assert compact_case["memory"]["forced_correction_applied"] is False
        assert results["summary"]["student_initial_memory_mode"] == "unknown"
        assert results["summary"]["event_memory_semantics"] == "event_conditioned_on_rs"
        assert results["summary"]["rs_change_invalidates_event"] is True
        assert results["summary"]["rs_schedule_policy"] == "deployable"
        assert results["summary"]["rs_schedule_uses_ground_truth"] is False
        assert set(compact_case["student"]) >= {"q1_output", "q2_output", "q1_parsed", "q2_parsed"}
        assert set(compact_case["teacher"]) >= {"q1_output", "q2_output", "q1_parsed", "q2_parsed"}
        assert results["frame_artifacts"] == []
        assert results["memory_recovery_report"]["student_enabled"] is False
        assert summary["sampling"]["selected_cases"] == len(route.frames)
        assert summary["generation_limits"]["max_new_tokens_q1"] == 16

        # review 是默认人工入口：每帧只保留 RGB、input、output、memory。
        review_output_dir = root / "probe_review"
        args.output_dir = str(review_output_dir)
        args.artifact_level = "review"
        dump_probe(args)
        review_frames = list(review_output_dir.glob("scenarios/*/frame_*"))
        assert len(review_frames) == len(route.frames)
        expected_review_files = {"input_rgb_00.jpg", "input.json", "output.json", "memory.json"}
        assert {path.name for path in review_frames[0].iterdir()} == expected_review_files
        inputs = json.loads((review_frames[0] / "input.json").read_text(encoding="utf-8"))
        outputs = json.loads((review_frames[0] / "output.json").read_text(encoding="utf-8"))
        memory = json.loads((review_frames[0] / "memory.json").read_text(encoding="utf-8"))
        assert inputs["rgb"][0]["file"] == "input_rgb_00.jpg"
        assert inputs["q1_student_messages"][0]["role"] == "system"
        assert inputs["q2_student_user_turn"]["continued_from"] == "student.q1_output_kv"
        assert outputs["ground_truth"]["scenario"]
        assert outputs["ground_truth"]["route_id"]
        expected_q1 = outputs["ground_truth"]["structured"]["q1"]
        expected_q2 = outputs["ground_truth"]["structured"]["q2"]
        assert expected_q1["rs_option"] in {"A", "B"}
        assert "abnormal" not in expected_q1
        assert expected_q2["event_family"] in {"RE", "UE"}
        assert expected_q2["resolved_for_student_label"] in {"RE", "U-E1"}
        assert expected_q2["accepted_event_labels"]
        assert set(outputs["student"]) >= {"q1_output", "q2_output", "q1_parsed", "q2_parsed"}
        assert set(outputs["teacher"]) >= {"q1_output", "q2_output", "q1_parsed", "q2_parsed"}
        assert outputs["teacher_targets"]["q1"].startswith("Scene Description:")
        assert outputs["correctness"]["q1_rs_correct"] is None
        assert memory["reference_is_comparison_only"] is True
        assert memory["forced_correction_applied"] is False
        assert memory["q1"]["event_context_invalidated_by_rs_change"] is True
        assert memory["q2"]["input"]["student"]["event_label"] == "UNKNOWN"
        assert memory["q2"]["input"]["student"]["event_age_frames"] == 0
        review_results = json.loads((review_output_dir / "results.json").read_text(encoding="utf-8"))
        assert review_results["frames"] == []
        assert len(review_results["frame_artifacts"]) == len(route.frames)

        # full 是显式深度审计开关，不应删除 review 的规范文件和 legacy 产物。
        full_output_dir = root / "probe_full"
        args.output_dir = str(full_output_dir)
        args.artifact_level = "full"
        dump_probe(args)
        assert (full_output_dir / "results.json").exists()
        assert (full_output_dir / "selection_plan.json").exists()
        assert (full_output_dir / "summary.json").exists()
        assert (full_output_dir / "transition_report.json").exists()
        case_records = list(full_output_dir.glob("scenarios/*/frame_*/case_record.json"))
        assert len(case_records) == len(route.frames)
        case = json.loads(case_records[0].read_text(encoding="utf-8"))
        assert set(case) == {"selection", "labels", "inputs", "targets", "outputs", "memory", "flags"}
        assert "q1_student_messages" in case["inputs"]
        assert case["selection"]["sample_mode_description"]
        assert case["selection"]["primary_reason_description"] == "随机完整 route ID 中按时间顺序测试的帧"
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
    test_long_context_observes_delayed_correction_window()
    test_memory_recovery_report_tracks_delayed_student_repair()
    test_metric_false_positive_false_negative_contract()
    test_transition_detection_confusion_and_report()
    test_static_probe_compact_review_and_full_artifacts()
    print("[ok] probe selection and eval metrics")


if __name__ == "__main__":
    main()
