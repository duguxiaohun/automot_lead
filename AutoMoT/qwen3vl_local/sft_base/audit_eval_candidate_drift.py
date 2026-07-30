"""审计 eval Q2 候选过滤相对训练候选的偏差。

用法（从 AutoMoT/ 目录）：
  python qwen3vl_local/sft_base/audit_eval_candidate_drift.py \
    --index checkpoints/sft_base_data/val_sequence_index.jsonl

脚本不加载模型，只假设 pred_rs == gt_rs，比较 eval 当前候选构造与 build_dataset
写入的 `event_candidates_ordered` 是否一致，用于决定是否需要把训练侧也统一成同一过滤口径。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.sft_base.eval_candidates import q2_candidates_for_student_rs


@dataclass
class FrameView:
    """audit 用最小 frame 视图。"""

    frame_id: int
    rs_label: str
    event_label: str
    raw: Dict[str, Any]


def _iter_frames(index_path: pathlib.Path, max_frames: int) -> Any:
    """流式读取 sequence index 里的 frame。"""

    seen = 0
    with index_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            for frame in row.get("frames") or []:
                raw = dict(frame)
                raw.setdefault("run_id", row.get("route_id"))
                raw.setdefault("route_id", row.get("route_id"))
                yield FrameView(
                    frame_id=int(frame.get("frame_id", 0)),
                    rs_label=str(frame.get("rs_label") or frame.get("road_structure")),
                    event_label=str(frame.get("event_label") or "RE"),
                    raw=raw,
                )
                seen += 1
                if max_frames > 0 and seen >= max_frames:
                    return


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    """执行候选偏差审计。"""

    total = 0
    set_mismatch = 0
    order_mismatch = 0
    unreachable = 0
    single_candidate = 0
    examples: List[Dict[str, Any]] = []
    for frame in _iter_frames(pathlib.Path(args.index), int(args.max_frames)):
        dataset_candidates = [str(x) for x in (frame.raw.get("event_candidates_ordered") or [])]
        if not dataset_candidates:
            continue
        candidates, _regular, source, reachable = q2_candidates_for_student_rs(frame, frame.rs_label, seed=int(args.seed))
        total += 1
        same_set = set(candidates) == set(dataset_candidates)
        same_order = list(candidates) == list(dataset_candidates)
        set_mismatch += int(not same_set)
        order_mismatch += int(same_set and not same_order)
        unreachable += int(not reachable)
        single_candidate += int(len(candidates) == 1)
        if (not same_set or not reachable) and len(examples) < int(args.max_examples):
            examples.append(
                {
                    "frame_id": frame.frame_id,
                    "rs_label": frame.rs_label,
                    "event_label": frame.event_label,
                    "source": source,
                    "dataset_candidates": dataset_candidates,
                    "eval_candidates": candidates,
                    "reachable": reachable,
                }
            )
    return {
        "index": str(args.index),
        "seed": int(args.seed),
        "total": total,
        "set_mismatch": set_mismatch,
        "set_mismatch_rate": set_mismatch / max(total, 1),
        "order_mismatch_same_set": order_mismatch,
        "order_mismatch_same_set_rate": order_mismatch / max(total, 1),
        "unreachable_when_pred_eq_gt": unreachable,
        "unreachable_when_pred_eq_gt_rate": unreachable / max(total, 1),
        "single_candidate": single_candidate,
        "single_candidate_rate": single_candidate / max(total, 1),
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    """解析参数。"""

    parser = argparse.ArgumentParser(description="Audit sft_base eval candidate drift")
    parser.add_argument("--index", type=str, default="checkpoints/sft_base_data/val_sequence_index.jsonl")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""

    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
