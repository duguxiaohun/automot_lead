"""SFT baseline eval metrics 源码合同测试。

本测试故意不 import eval.py；eval.py 会 import torch 和模型训练入口，在本地轻量
环境里可能因为 CUDA/torch DLL 初始化失败而无法加载。这里用源码合同守住当前
单问 baseline 的关键指标。
"""

from __future__ import annotations

import pathlib


def main() -> None:
    """验证 eval.py 已按 HIGHWAY/NON_HIGHWAY + RE/UE 单问协议输出指标。"""

    eval_py = pathlib.Path(__file__).with_name("eval.py")
    src = eval_py.read_text(encoding="utf-8")
    required = [
        '"road": "road_transition"',
        '"event": "event_transition"',
        '"road_acc": counters["road_correct"] / frames',
        '"event_acc": counters["event_correct"] / frames',
        '"joint_acc": counters["joint_correct"] / frames',
        '"highway_precision": road_report["precision"]',
        '"highway_recall": road_report["recall"]',
        '"highway_f1": road_report["f1"]',
        '"ue_precision": event_report["precision"]',
        '"ue_recall": event_report["recall"]',
        '"ue_f1": event_report["f1"]',
        '"road_change_f1": road_change["f1"]',
        '"event_change_f1": event_change["f1"]',
        '"transition_hit_rate": counters["transition_hit_cases"]',
        'args.output_json = args.output_json or str(output_dir / "metrics.json")',
        'args.output_jsonl = args.output_jsonl or str(output_dir / "frames.jsonl")',
        'args.output_summary = args.output_summary or str(output_dir / "summary.md")',
        '"gt_road": gt_road',
        '"pred_road": pred_road',
        '"gt_event": gt_event',
        '"pred_event": pred_event',
        '"raw_text": text',
        'ROAD: HIGHWAY|NON_HIGHWAY',
        'EVENT: RE|UE',
    ]
    forbidden = [
        '"q2_joint_correct"',
        '"q2_rs_candidate_count_report"',
        'event_visual_gain_over_regular_baseline',
    ]
    for needle in required:
        assert needle in src, needle
    for needle in forbidden:
        assert needle not in src, needle
    print("[test_eval_metrics] ok")


if __name__ == "__main__":
    main()
