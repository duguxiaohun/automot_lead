"""
场景事件采集系统 - 结构分析模块

该模块为 quick_start.py 和 README 中的分析入口提供兼容实现。
它基于采集结果 JSON 做汇总统计，不依赖额外第三方库。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class StructureAnalyzer:
    """对采集结果做轻量统计和报告输出。"""

    def __init__(self, results: Dict[str, Dict[str, Any]]):
        self.results = results or {}
        self._cached_summary: Optional[Dict[str, Any]] = None

    def analyze_multi_angle(self) -> Dict[str, Any]:
        """从多个角度汇总采集结果。"""
        scenario_count = len(self.results)
        success_results = [r for r in self.results.values() if r.get("status") == "success"]
        total_frames = sum(int(r.get("total_frames", 0)) for r in success_results)

        road_counter: Counter[str] = Counter()
        event_counter: Counter[str] = Counter()
        frame_distribution: Dict[str, int] = {}

        for scenario_name, result in self.results.items():
            frame_distribution[scenario_name] = int(result.get("total_frames", 0))
            road_counter.update(result.get("road_candidates", []))
            event_counter.update(result.get("event_candidates", []))

        summary = {
            "scenario_count": scenario_count,
            "success_count": len(success_results),
            "error_count": scenario_count - len(success_results),
            "total_frames": total_frames,
            "average_frames_per_success_scenario": (
                round(total_frames / len(success_results), 2) if success_results else 0.0
            ),
            "road_candidate_distribution": dict(sorted(road_counter.items())),
            "event_candidate_distribution": dict(sorted(event_counter.items())),
            "frame_distribution": dict(sorted(frame_distribution.items())),
        }

        self._cached_summary = summary
        return summary

    def print_summary(self) -> None:
        """打印简要统计结果。"""
        summary = self._cached_summary or self.analyze_multi_angle()

        print("\n" + "=" * 70)
        print("结构分析摘要".center(70))
        print("=" * 70)
        print(f"场景数: {summary['scenario_count']}")
        print(f"成功场景数: {summary['success_count']}")
        print(f"失败场景数: {summary['error_count']}")
        print(f"总帧数: {summary['total_frames']}")
        print(f"成功场景平均帧数: {summary['average_frames_per_success_scenario']}")

        if summary["road_candidate_distribution"]:
            print("\n道路结构候选分布:")
            for name, count in summary["road_candidate_distribution"].items():
                print(f"  - {name}: {count}")

        if summary["event_candidate_distribution"]:
            print("\n事件候选分布:")
            for name, count in summary["event_candidate_distribution"].items():
                print(f"  - {name}: {count}")

        print("=" * 70)

    def generate_report(self, output_path: str) -> Dict[str, Any]:
        """生成 JSON 报告并写入文件。"""
        report = {
            "summary": self.analyze_multi_angle(),
            "results": self.results,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

        return report


def _load_results_from_directory(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """从采集输出目录加载所有场景结果。"""
    results: Dict[str, Dict[str, Any]] = {}
    if not output_dir.exists():
        return results

    for result_file in sorted(output_dir.glob("*_result.json")):
        try:
            with open(result_file, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            scenario_name = result.get("scenario")
            if scenario_name:
                results[scenario_name] = result
        except Exception:
            continue

    return results


def quick_analysis(output_dir: str = "/home/cruser1/lda/AutoMoT/keyframe_filter/collection_output") -> Optional[Dict[str, Any]]:
    """快速分析当前采集结果目录。"""
    results = _load_results_from_directory(Path(output_dir))
    if not results:
        print("\n❌ 没有找到可分析的采集结果")
        return None

    analyzer = StructureAnalyzer(results)
    analyzer.print_summary()
    report = analyzer.generate_report(str(Path(output_dir) / "structure_analysis_report.json"))
    print(f"\n✓ 完整报告已保存: {Path(output_dir) / 'structure_analysis_report.json'}")
    return report