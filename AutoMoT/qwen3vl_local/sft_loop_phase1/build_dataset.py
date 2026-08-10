#!/usr/bin/env python3
"""构建 sft_loop_phase1 的逐帧四问训练/测试索引。

输入来自 `keyframe_filter/collection_output/*_result.json` 的逐帧 RS/EVENT 标定和
Phase1 最终四问答案表。脚本不重新判断 RGB，只把已审计的
`scenario x RS x EVENT -> 四问 YES/NO` 映射到真实 LEAD RGB 帧，并按 route 做稳定
train/test split，避免同一路线相邻帧泄漏到两侧。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_THIS = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS.parents[2]
_PROJECT_ROOT = _THIS.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route  # noqa: E402
from qwen3vl_local.sft_loop_phase1 import DATASET_NAME  # noqa: E402
from qwen3vl_local.sft_loop_phase1.audit_matrix import _iter_routes_stream, _rgb_path  # noqa: E402
from qwen3vl_local.sft_loop_phase1.prompts import ANSWER_KEYS  # noqa: E402


RGB_HISTORY_COUNT = 4


def _stable_unit(seed_text: str) -> float:
    """把字符串稳定映射到 [0, 1)，用于 route split。"""

    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _annotation_labels(annotation: Mapping[str, Any]) -> Tuple[str, str]:
    """读取当前帧 primary RS/EVENT。"""

    rs = str(annotation.get("primary_road_structure") or (annotation.get("frame_rs_annotation") or {}).get("label") or "UNKNOWN")
    event = str(annotation.get("primary_event") or (annotation.get("frame_event_annotation") or {}).get("label") or "UNKNOWN")
    return rs, event


def _town(annotation: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    """读取 XML town；缺失时保留 UNKNOWN。"""

    evidence = annotation.get("evidence") or {}
    return str(evidence.get("xml_town") or route.get("xml_town") or "UNKNOWN")


def _history_rgb_paths(run_dir: pathlib.Path, frame_id: int) -> Optional[List[str]]:
    """返回 oldest -> newest 的 4 帧 RGB history，开头 left-pad 到 frame0。"""

    paths: List[str] = []
    for idx in [max(0, frame_id - offset) for offset in reversed(range(RGB_HISTORY_COUNT))]:
        rgb = _rgb_path(run_dir, idx)
        if rgb is None:
            return None
        paths.append(str(rgb))
    return paths


def _load_answer_table(path: pathlib.Path) -> Dict[Tuple[str, str, str], Dict[str, bool]]:
    """读取最终四问答案表。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "phase1_four_question_answer_table":
        raise ValueError(f"unsupported answer table: {path}")
    table: Dict[Tuple[str, str, str], Dict[str, bool]] = {}
    for row in payload.get("rows", []):
        scenario, rs, event = str(row.get("scenario")), str(row.get("rs")), str(row.get("event"))
        answers = {key: bool((row.get("answers") or {}).get(key, False)) for key in ANSWER_KEYS}
        table[(scenario, rs, event)] = answers
    return table


def _split_for_route(scenario: str, route_id: str, *, seed: int, test_ratio: float, val_ratio: float) -> str:
    """按 route 稳定划分 split。"""

    value = _stable_unit(f"{seed}:{scenario}:{route_id}")
    if value < float(test_ratio):
        return "test"
    if value < float(test_ratio) + float(val_ratio):
        return "val"
    return "train"


def iter_frame_rows(
    *,
    collection_dir: pathlib.Path,
    data_root: pathlib.Path,
    answer_table: Mapping[Tuple[str, str, str], Dict[str, bool]],
    split_seed: int,
    test_ratio: float,
    val_ratio: float,
    scenarios: Optional[set[str]],
    max_routes: int,
) -> Iterable[Dict[str, Any]]:
    """流式产出所有可训练/测试帧。"""

    route_seen = 0
    for result_path in sorted(collection_dir.glob("*_result.json")):
        scenario = result_path.stem.removesuffix("_result")
        if scenario == "noScenarios" or (scenarios is not None and scenario not in scenarios):
            continue
        for route in _iter_routes_stream(result_path):
            route_id = str(route.get("route_id") or "")
            if not route_id or str(route.get("status")) == "data_missing_skip":
                continue
            run_dir = data_root / scenario / route_id
            abnormal, _ = is_abnormal_lead_route(run_dir, scenario)
            if abnormal or not run_dir.is_dir():
                continue
            route_seen += 1
            if max_routes > 0 and route_seen > max_routes:
                return
            split = _split_for_route(
                scenario,
                route_id,
                seed=int(split_seed),
                test_ratio=float(test_ratio),
                val_ratio=float(val_ratio),
            )
            for ann in route.get("annotations", []) or []:
                try:
                    frame_id = int(ann.get("frame_id"))
                except (TypeError, ValueError):
                    continue
                rs, event = _annotation_labels(ann)
                answers = answer_table.get((scenario, rs, event))
                if answers is None:
                    continue
                history = _history_rgb_paths(run_dir, frame_id)
                if history is None:
                    continue
                yield {
                    "dataset_name": DATASET_NAME,
                    "scenario": scenario,
                    "route_id": route_id,
                    "town": _town(ann, route),
                    "split": split,
                    "frame_id": frame_id,
                    "rs": rs,
                    "event": event,
                    "answers": answers,
                    "history_rgb_paths": history,
                    "latest_rgb_path": history[-1],
                }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """构建 jsonl 索引和 manifest。"""

    collection_dir = pathlib.Path(args.collection_dir)
    data_root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_table = _load_answer_table(pathlib.Path(args.answer_table))
    scenarios = None if str(args.scenarios) == "all" else {x.strip() for x in str(args.scenarios).split(",") if x.strip()}

    out_path = output_dir / "frame_index.jsonl"
    counters: Counter[str] = Counter()
    answer_counters: Dict[str, Counter[str]] = {key: Counter() for key in ANSWER_KEYS}
    route_ids: Dict[str, set[str]] = defaultdict(set)
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_frame_rows(
            collection_dir=collection_dir,
            data_root=data_root,
            answer_table=answer_table,
            split_seed=int(args.split_seed),
            test_ratio=float(args.test_ratio),
            val_ratio=float(args.val_ratio),
            scenarios=scenarios,
            max_routes=int(args.max_routes),
        ):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            split = str(row["split"])
            counters[f"frames/{split}"] += 1
            counters[f"frames/{split}/{row['scenario']}"] += 1
            route_ids[split].add(f"{row['scenario']}/{row['route_id']}")
            for key in ANSWER_KEYS:
                answer_counters[key][f"{split}/{'YES' if row['answers'][key] else 'NO'}"] += 1

    manifest = {
        "format": "sft_loop_phase1_frame_index",
        "dataset_name": DATASET_NAME,
        "frame_index": str(out_path),
        "answer_table": str(args.answer_table),
        "split_seed": int(args.split_seed),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "counts": dict(counters),
        "route_counts": {split: len(values) for split, values in sorted(route_ids.items())},
        "answer_counts": {key: dict(counter) for key, counter in answer_counters.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    p = argparse.ArgumentParser(description="Build sft_loop_phase1 frame index")
    p.add_argument("--collection-dir", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output"))
    p.add_argument("--data-root", default=str(_AUTOMOT_ROOT / "lead_data"))
    p.add_argument("--answer-table", default=str(_AUTOMOT_ROOT / "keyframe_filter/collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json"))
    p.add_argument("--output-dir", default=str(_AUTOMOT_ROOT / "checkpoints/sft_loop_phase1_data"))
    p.add_argument("--scenarios", default="all")
    p.add_argument("--split-seed", type=int, default=20260810)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--val-ratio", type=float, default=0.00)
    p.add_argument("--max-routes", type=int, default=0, help="debug only; 0 means all routes")
    return p.parse_args()


def main() -> None:
    """CLI 入口。"""

    manifest = build_dataset(parse_args())
    print(
        "sft_loop_phase1 dataset: "
        f"frames={sum(v for k, v in manifest['counts'].items() if k.startswith('frames/') and k.count('/') == 1)} "
        f"output={manifest['frame_index']}"
    )


if __name__ == "__main__":
    main()
