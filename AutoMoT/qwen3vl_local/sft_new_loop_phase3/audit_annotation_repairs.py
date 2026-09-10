"""独立重放源标注修复，输出包含被隔离帧的逐帧追溯清单。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
from qwen3vl_local.sft_new_loop_phase3.build_dataset import _rs_label, _event_codes
from qwen3vl_local.sft_new_loop_phase3.collection_reader import iter_routes
from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash


def audit(collection_dir, data_root, output_dir, scenarios="all"):
    """审计所有发生变更的帧，不只记录进入训练样本池的幸存者。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir/"annotation_repairs.jsonl"
    temp = target.with_suffix(".jsonl.tmp")
    selected = None if scenarios == "all" else set(scenarios.split(","))
    sources = [p for p in sorted(Path(collection_dir).glob("*_result.json"))
               if p.stem != "noScenarios_result" and
               (selected is None or p.stem.removesuffix("_result") in selected)]
    if not sources:
        raise ValueError("no collection sources found")
    counts, changes, per_scenario = Counter(), Counter(), Counter()
    print(f"[annotation-discover] scenarios={len(sources)}", flush=True)
    with temp.open("w") as out:
        for source in sources:
            scenario = source.stem.removesuffix("_result")
            for route in iter_routes(source):
                run = route.get("route_id")
                if not run or route.get("status") == "data_missing_skip":
                    continue
                directory = Path(data_root)/scenario/run
                if not directory.is_dir() or is_abnormal_lead_route(directory, scenario)[0]:
                    counts["excluded_routes"] += 1
                    continue
                counts["routes"] += 1
                if counts["routes"] % 200 == 0:
                    print(f"[annotation-audit] routes={counts['routes']} changed={counts['changed_frames']}", flush=True)
                for ann in route.get("annotations") or []:
                    try:
                        frame = int(ann["frame_id"])
                    except (KeyError, TypeError, ValueError):
                        counts["invalid_frame_id"] += 1
                        continue
                    counts["frames"] += 1
                    rs, primary, codes, trace = repair_annotation(scenario, run, frame, ann,
                        _rs_label(ann), str(ann.get("primary_event") or "UNKNOWN"), _event_codes(ann))
                    if not trace["changes"]:
                        continue
                    counts["changed_frames"] += 1
                    changes.update(trace["changes"])
                    per_scenario[scenario] += 1
                    out.write(json.dumps(dict(scenario=scenario, route_id=run, frame_id=frame,
                        source_file=source.name, repair=trace, repaired_primary_event=primary,
                        rs_quarantined=rs == "UNKNOWN", automatic_rule_hit=True,
                        manual_rgb_confirmed="rgb_confirmed_mainline_not_signalized_junction" in trace["changes"]),
                        ensure_ascii=False)+"\n")
    temp.replace(target)
    report = dict(mapping_contract_hash=mapping_contract_hash(), counts=dict(counts),
        changes=dict(changes), by_scenario=dict(per_scenario), source_scope="collection_results",
        manual_review_claim="Only explicitly recorded RGB decisions are visually confirmed; automatic hits require review.")
    (output_dir/"annotation_repair_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", type=Path, default=Path(__file__).resolve().parents[2]/"keyframe_filter/collection_output")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scenarios", default="all")
    args = p.parse_args()
    audit(**vars(args))


if __name__ == "__main__":
    main()
