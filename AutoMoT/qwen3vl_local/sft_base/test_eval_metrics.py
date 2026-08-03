"""SFT base eval metrics 源码合同测试。

本测试故意不 import eval.py；eval.py 会 import torch 和模型训练入口，在本地轻量
环境里可能因为 CUDA/torch DLL 初始化失败而无法加载。这里用源码合同守住
RS×候选数分层的关键落点。
"""

from __future__ import annotations

import pathlib


def main() -> None:
    """验证 EVENT 指标同时按候选数、RS、regular 子类和 remap 口径分层输出。"""

    eval_py = pathlib.Path(__file__).with_name("eval.py")
    src = eval_py.read_text(encoding="utf-8")
    required = [
        'rs_bucket_prefix = f"q2_rs_{frame.rs_label}_candidate_count_{candidate_count_bucket}"',
        'rs_prefix = f"q2_rs_{rs}_candidate_count_{n}"',
        'rs_candidate_count_report[rs] = rs_report_by_count',
        '"q2_rs_candidate_count_report": rs_candidate_count_report',
        '"joint_acc": counters["q2_joint_correct"] / q2_total',
        'p.add_argument("--output-html", type=str, default=None)',
        'args.output_html = str(output_dir / "report.html")',
        'def _write_html_report(path: pathlib.Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:',
        'const DATA = {metrics_payload};',
        'renderMatrix("eventMatrix", "GT EVENT × Pred EVENT", DATA.event_confusion_report || {{}});',
        'renderMatrix("ueMultiMatrix", "UE-vs-Regular multi-candidate", binaryMatrix("ue_vs_re_tp_multi_candidate", "ue_vs_re_fp_multi_candidate", "ue_vs_re_fn_multi_candidate", "ue_vs_re_tn_multi_candidate"));',
        '_GLOBAL_EVENT_MAJORITY_LABEL = "R-E1"',
        'counters["event_global_majority_correct"] += int(target.label == _GLOBAL_EVENT_MAJORITY_LABEL)',
        '"event_global_majority_baseline": event_global_majority_baseline',
        '"event_visual_gain_over_global_majority_baseline": event_acc_q2 - event_global_majority_baseline',
        '"event_visual_gain_over_regular_baseline": event_acc_q2 - event_global_majority_baseline',
        '"event_regular_baseline_given_gt_rs": event_regular_baseline_given_gt_rs',
        '"event_gap_to_given_gt_rs_regular_baseline": event_acc_q2 - event_regular_baseline_given_gt_rs',
        '"event_regular_baseline_expected_full_data": REGULAR_ZERO_INFO_BASELINE_END_TO_END',
        '"event_oracle_majority_regular_baseline_end_to_end": event_regular_baseline_oracle_majority_q2',
        'baseline_target = _event_target_from_frame(frame)',
        'counters["regular_majority_static_correct"] += int(baseline_target.label == baseline_label)',
        '"regular_internal_confusion_report": regular_report',
        'counters[f"q2_rs_{frame.rs_label}_multi_re_ue_fp"] += int(pred_ue)',
        '"q2_multi_re_by_rs_report": q2_multi_re_by_rs_report',
        '"q2_multi_ue_by_rs_report": q2_multi_ue_by_rs_report',
        '"ue_fp_on_multi_candidate_re_by_rs": ue_fp_on_multi_candidate_re_by_rs',
        '"gt_event_code_raw": gt_event_code_raw',
        '"gt_regular_remapped": bool(gt_regular_remapped)',
        '"event_raw_regular_remap_report": {',
        '"combos": raw_regular_remap_combo_report',
        "event_report = _multiclass_report(counters, prefix=\"event\", labels=_EVENT_LABELS, pred_labels=_EVENT_CM_LABELS)",
    ]
    for needle in required:
        assert needle in src, needle
    print("[test_eval_metrics] ok")


if __name__ == "__main__":
    main()
