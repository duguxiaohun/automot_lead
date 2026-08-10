#!/usr/bin/env python3
"""把 RGB 审计矩阵转换为第一轮训练的组合级四问答案表。"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Iterable, List, Sequence

from qwen3vl_local.sft_loop_phase1.answer_policy import (
    answer_rationale,
    resolve_group_answers,
    topology_subgroup_overrides,
)


def _load_manifest(path: pathlib.Path) -> Dict[str, Any]:
    """读取并验证一个 phase1 审计 manifest。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "phase1_four_question_audit":
        raise ValueError(f"unsupported phase1 manifest: {path}")
    return payload


def _row_visual_subgroups(scenario: str, rs: str) -> List[Dict[str, Any]]:
    """返回适用于当前 scenario/RS 默认行的显式视觉子组。"""

    out: List[Dict[str, Any]] = []
    for item in topology_subgroup_overrides():
        if str(item.get("scenario")) != str(scenario):
            continue
        rs_values = {str(value) for value in item.get("rs_values", [])}
        if rs_values and str(rs) not in rs_values:
            continue
        out.append(item)
    return out


def build_answer_table(
    manifests: Sequence[pathlib.Path],
    *,
    excluded_scenarios: Iterable[str] = ("noScenarios",),
) -> Dict[str, Any]:
    """为每个已 RGB 抽样的初始组合生成待复核答案表。

    ``noScenarios`` 不是统一的可见驾驶语义：同一初始 RS 会混有城市主干道与
    受限出入道路。因此它默认完全排除，不能在后续重建时又被机械答案策略写回。
    其余行仍只是初始分层键；完成全帧 RGB 审阅后，调用方必须再按视觉子组细分。
    """

    rows: List[Dict[str, Any]] = []
    seen = set()
    excluded = {str(item) for item in excluded_scenarios if str(item)}
    for manifest_path in manifests:
        payload = _load_manifest(manifest_path)
        for group in payload.get("groups", []):
            scenario, rs, event = (str(group.get(key) or "UNKNOWN") for key in ("scenario", "rs", "event"))
            if scenario in excluded:
                continue
            key = (scenario, rs, event)
            if key in seen:
                # 分批全量审计可能为了更密集的 route 抽查而重复包含同一组合。默认答案
                # 合同只由已审核的 scenario/RS/EVENT 决定，保留 CLI 中更早 manifest 的
                # 样本即可；需要的 topology 混合信息在顶层 overrides 中单独表达。
                continue
            seen.add(key)
            answers = resolve_group_answers(scenario, rs, event)
            rows.append(
                {
                    "scenario": scenario,
                    "rs": rs,
                    "event": event,
                    "answers": answers,
                    "rationale": answer_rationale(scenario, rs, event),
                    "frame_count": int(group.get("frame_count", 0)),
                    "towns_seen": group.get("towns_seen", []),
                    "rgb_review_samples": group.get("samples", []),
                    "evidence_sheet": group.get("evidence_sheet"),
                    "visual_subgroups": _row_visual_subgroups(scenario, rs),
                    "label_contract": "rgb_reviewed_default_per_scenario_rs_event; explicit_topology_subgroup_override_may_apply",
                }
            )
    rows.sort(key=lambda row: (row["scenario"], row["rs"], row["event"]))
    return {
        "format": "phase1_four_question_answer_table",
        "answer_order": ["HIGHWAY", "OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL"],
        "group_count": len(rows),
        "excluded_scenarios": sorted(excluded),
        "review_contract": (
            "default rows were validated against full-frame RGB review; a route may use a different answer "
            "only when it carries one of visual_subgroup_overrides with its stated evidence contract"
        ),
        "visual_subgroup_overrides": topology_subgroup_overrides(),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""

    parser = argparse.ArgumentParser(description="Build uniform scenario/RS/EVENT phase1 answers")
    parser.add_argument("--manifests", required=True, help="comma-separated phase1_four_question_matrix.json paths")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-scenarios",
        default="noScenarios",
        help="comma-separated scenario names excluded from the answer table (default: noScenarios)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    paths = [pathlib.Path(item.strip()) for item in str(args.manifests).split(",") if item.strip()]
    if not paths:
        raise ValueError("--manifests is empty")
    excluded = [item.strip() for item in str(args.exclude_scenarios).split(",") if item.strip()]
    table = build_answer_table(paths, excluded_scenarios=excluded)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"phase1 answer table: groups={table['group_count']} output={output}")


if __name__ == "__main__":
    main()
