"""SFT v4 loss mask 静态检查（7 项）。

v4 训练入口实际监督 7 项 loss：

  L_A1 / L_RS1    step1 analysis + ROAD_STRUCTURE 值 token
  L_A2 / L_S2     step2 analysis + SCENE 值 token
  L_A3 / L_S3_status / L_S3_subgoal   step3 analysis + STATUS + SUBGOAL 值 token

每一路都按 ``train.py:_append_token_ids`` 内同款规则切 token：

- analysis token 集 = ``lo < analysis_end and hi > 0 and text[lo:hi].strip()``；
- value token 集 = ``lo < span_hi and hi > span_lo``；
- 同一个 token 不能同时落进 analysis 和 value 两个集合（per-token normalize 分母
  互不污染）。

本检查不需要训练数据，只构造典型样例文本就能验证。没有本地 Qwen tokenizer 时跳过
分词步骤，但仍报字符级 recovered 结果。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_v4.prompts import (  # noqa: E402
    build_step1_teacher_target,
    build_step2_teacher_target,
    build_step3_teacher_target,
    target_spans_road_structure,
    target_spans_scene,
    target_spans_status,
)

# train.py 内的 _analysis_char_end 用同一条正则
_LABEL_LINE_RE = re.compile(
    r"^\s*(ROAD_STRUCTURE|SCENE|STATUS|SUBGOAL)\s*:",
    re.MULTILINE | re.IGNORECASE,
)


def _analysis_char_end(text: str) -> int:
    """与 train.py:_analysis_char_end 完全等价的分析段终止位置。"""

    match = _LABEL_LINE_RE.search(text or "")
    return match.start() if match else len(text or "")


def _load_tokenizer(model_dir: pathlib.Path) -> Any | None:
    """尝试加载本地 Qwen tokenizer，没有就返回 None，让上层跳过 token-level 验证。"""

    if not model_dir.exists():
        return None
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    return processor.tokenizer


def _split_token_positions(
    tok: Any,
    text: str,
    spans: Dict[str, Tuple[int, int]],
) -> Dict[str, List[int]]:
    """复用 train.py:_append_token_ids 的 mask 切分逻辑。

    返回 {"analysis": [...], <span_key>: [...], ...} 的 token 位置索引。
    分析段位置使用 ``strip()`` 排除纯空白 token，与 train.py 一致。
    """

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    analysis_end = _analysis_char_end(text)

    positions: Dict[str, List[int]] = {
        "analysis": [
            j for j, (lo, hi) in enumerate(offsets)
            if hi > 0 and lo < analysis_end and text[lo:hi].strip()
        ]
    }
    for key, (span_lo, span_hi) in spans.items():
        positions[key] = [
            j for j, (lo, hi) in enumerate(offsets)
            if lo < span_hi and hi > span_lo
        ]
    return positions


def _check_route(
    route: str,
    tok: Any | None,
    text: str,
    spans: Dict[str, Tuple[int, int]],
    *,
    expect_analysis: bool,
    expect_values: Dict[str, str],
) -> Dict[str, Any]:
    """对单一路 loss 跑一次 mask 检查并汇总诊断。

    expect_analysis    该路 analysis token 数应 > 0 吗（step1/2/3 都应为 True）。
    expect_values      该路 value token 与字符 span 期望恢复的字符串。
    """

    recovered = {k: text[a:b] for k, (a, b) in spans.items()}
    char_ok = all(recovered.get(k) == v for k, v in expect_values.items())

    if tok is None:
        return {
            "route": route,
            "tokenizer": "skipped",
            "char_recovered": recovered,
            "char_ok": char_ok,
            "ok": char_ok,
        }

    positions = _split_token_positions(tok, text, spans)
    analysis_count = len(positions["analysis"])
    value_counts = {k: len(positions[k]) for k in spans}
    overlap = {
        k: sorted(set(positions["analysis"]) & set(positions[k]))
        for k in spans
    }
    no_overlap = all(len(v) == 0 for v in overlap.values())
    if not no_overlap:
        raise AssertionError(f"{route}: analysis/value token masks overlap: {overlap}")
    analysis_ok = (not expect_analysis) or (analysis_count > 0)
    value_ok = all(v > 0 for v in value_counts.values())
    ok = char_ok and analysis_ok and value_ok and no_overlap

    return {
        "route": route,
        "tokenizer": "ok",
        "char_recovered": recovered,
        "char_ok": char_ok,
        "analysis_token_count": analysis_count,
        "value_token_counts": value_counts,
        "no_overlap": no_overlap,
        "overlap_positions": overlap,
        "ok": ok,
    }


def main() -> None:
    """枚举 7 项 loss 的 mask 检查并返回结构化报告。"""

    parser = argparse.ArgumentParser(description="Check SFT v4 loss mask (7 loss terms)")
    parser.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    tok = _load_tokenizer(pathlib.Path(args.model_dir))

    # step1：分析 + ROAD_STRUCTURE 值
    step1_text = build_step1_teacher_target(
        "\n".join(
            [
                "Scene Description: The frames show an intersection with visible lane structure.",
                "Critical Object Description: No pedestrian or blocking vehicle is dominant.",
                "Reasoning on Intent: Ego should slow and prepare for junction negotiation.",
                "Memory Judgment: The remembered road structure matches the intersection layout.",
            ]
        ),
        "JUNCTION",
    )

    # step2：分析 + SCENE 值
    step2_text = build_step2_teacher_target(
        "\n".join(
            [
                "Scene Description: The frames show a blocked lane in front of ego.",
                "Critical Object Description: The obstacle ahead is the main critical object.",
                "Reasoning on Intent: Ego should prepare to avoid or yield around the hazard.",
                "Memory Judgment: The remembered scene should be corrected to the obstacle scenario.",
            ]
        ),
        "Accident",
    )

    # step3：分析 + STATUS + SUBGOAL 值
    step3_text = build_step3_teacher_target(
        "\n".join(
            [
                "Scene Description: Ego is approaching the hazard with limited free space.",
                "Critical Object Description: The obstacle ahead controls the near-term maneuver.",
                "Reasoning on Intent: Ego should brake and preserve a safe gap.",
                "Memory Judgment: The remembered event should move toward braking for the hazard.",
            ]
        ),
        "hazard_detect",
        "max_brake_or_min_gap",
    )

    spans1 = target_spans_road_structure(step1_text)
    spans2 = target_spans_scene(step2_text)
    spans3 = target_spans_status(step3_text)

    reports = [
        _check_route(
            "L_A1 / L_RS1",
            tok,
            step1_text,
            spans1,
            expect_analysis=True,
            expect_values={"road_structure": "JUNCTION"},
        ),
        _check_route(
            "L_A2 / L_S2",
            tok,
            step2_text,
            spans2,
            expect_analysis=True,
            expect_values={"scene": "Accident"},
        ),
        _check_route(
            "L_A3 / L_S3_status / L_S3_subgoal",
            tok,
            step3_text,
            spans3,
            expect_analysis=True,
            expect_values={
                "status": "hazard_detect",
                "subgoal": "max_brake_or_min_gap",
            },
        ),
    ]

    overall_ok = all(r["ok"] for r in reports)
    print(json.dumps({"reports": reports, "ok": overall_ok}, ensure_ascii=False, indent=2))
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

