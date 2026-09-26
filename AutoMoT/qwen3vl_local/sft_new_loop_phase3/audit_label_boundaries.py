"""从原 meta 审计动作边界和先后关系；只分桶报告，不修改标签或模型输入。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.action_review import build_action_review, near_stop_review
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID, DOMAIN_MANEUVER
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import primary_choice
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    FRAME_DT_SECONDS, LONGITUDINAL_HORIZON_FRAMES, LATERAL_HORIZON_FRAMES,
    longitudinal_decision, longitudinal_from_signals, load_route_trajectory, label_actions,
)

AUDIT_VERSION = "phase3_boundary_diagnostics_v4_near_stop_segments"
# 固定审查带宽，不是动作阈值，不用于过滤、权重或修改 KEEP。
REVIEW_MARGIN_MPS = 0.10


def speed_boundaries(speeds, *, brake=None, throttle=None):
    """保留精确幅度余量；窗外变化只记审计，绝不触发窗内动作。"""
    decision = longitudinal_decision(speeds, brake=brake, throttle=throttle)
    if not decision["eligible"]:
        return {"eligible": False, "flags": [], "reason": decision["reason"]}
    values = list(map(float, speeds[:LONGITUDINAL_HORIZON_FRAMES + 1]))
    baseline, threshold = decision["baseline_speed_mps"], decision["delta_threshold_mps"]
    drop = baseline - min(values[1:])
    gain = max(min(values[i:i+2]) - baseline for i in range(1, len(values)-1))
    margins = {"drop_margin_mps": threshold-drop, "gain_margin_mps": threshold-gain}
    flags = [f"near_{name}_threshold" for name, value in (("drop", drop), ("gain", gain))
             if abs(threshold-value) <= REVIEW_MARGIN_MPS]
    for name in ("stop_pair_crosses_1_5s_boundary", "isolated_near_stop_in_1_5s",
                 "first_drop_single_sample", "isolated_gain_present", "gain_unconfirmed_at_2s_boundary"):
        if decision[name]:
            flags.append(name)
    near_stop = near_stop_review(values, brake=brake, throttle=throttle)
    flags.extend(near_stop['flags'])
    # 最多额外一秒，同一基准下的窗外邻近变化；不运行变长标定器。
    tail = list(map(float, speeds[9:13]))
    tail_complete = len(tail) == 4 and all(math.isfinite(v) and v >= 0 for v in tail)
    outside_drop = outside_gain = None
    if tail_complete and decision["action"] == "NONE":
        outside_drop = next(((i+9)*FRAME_DT_SECONDS for i,v in enumerate(tail)
                             if baseline-v >= threshold), None)
        extended = values + tail
        outside_gain = next((i*FRAME_DT_SECONDS for i in range(9,13)
                             if min(extended[i-1:i+1])-baseline >= threshold), None)
        if outside_drop is not None:
            flags.append("speed_continuation_drop_just_outside_window")
        if outside_gain is not None:
            flags.append("speed_continuation_gain_just_outside_window")
    return {"eligible": True, "speed_action": decision["action"], **margins,
            "drop_mps": drop, "sustained_gain_mps": gain, "flags": flags,
            "near_stop": near_stop,
            "tail_complete": tail_complete, "outside_drop_s": outside_drop,
            "outside_gain_confirmed_s": outside_gain}


def diagnose(trajectory, frame, context_id):
    """先确认完整域证据，再比较速度判据与首次跨线时刻；时刻仅为标定触发点。"""
    signals = trajectory.signals(frame)
    if signals is None:
        raise ValueError("missing anchor meta")
    labels = label_actions(signals)
    if labels is None:
        raise ValueError("index contains incomplete or ambiguous speed evidence")
    context = CONTEXT_BY_ID[context_id]
    maneuver = context.question_domain == DOMAIN_MANEUVER
    if maneuver and not signals["lateral_observation_complete"]:
        raise ValueError("maneuver index has incomplete lateral evidence")
    report = speed_boundaries(trajectory._future_speeds(frame, 12),
                              brake=signals.get("brake"), throttle=signals.get("throttle"))
    report["primary_action"] = primary_choice(labels, context.action_keys)
    report["lateral_observation_complete"] = signals["lateral_observation_complete"]
    report["lateral_window_issue"] = signals["lateral_window_issue"]
    review = build_action_review(trajectory, frame, context_id, labels, signals)
    report["action_review"] = review
    report["response_flags"] = review["flags"]
    report["crossing_start_s"] = None
    report["speed_action_start_s"] = review["speed_action_start_s"]
    if maneuver and signals["lane_change_direction"]:
        crossing = next(i for i in range(1, LATERAL_HORIZON_FRAMES+1)
                        if trajectory.lane_change(frame, horizon=i))
        report["crossing_start_s"] = crossing * FRAME_DT_SECONDS
        d = longitudinal_from_signals(signals)
        start = {"DECELERATE": d["first_drop_s"], "RESUME": d["gain_start_s"],
                 "STOP": d["stop_start_s"], "NONE": None}[d["action"]]
        report["speed_action_start_s"] = start
        if start is not None and start < report["crossing_start_s"]:
            report["flags"].append("speed_before_crossing")
    return report, labels


def identity(row):
    """兼容候选、独立评测及训练验证；只用所问道路，不回退 true_rs。"""
    spec = row.get("prompt_spec") or {}
    roads = [v for v in (row.get("prompt_road_structure"), row.get("rs"),
                         spec.get("road_structure")) if v not in (None, "")]
    if not roads:
        raise ValueError("missing prompted road structure; true_rs is not a question identity")
    if any(v != roads[0] for v in roads[1:]):
        raise ValueError("conflicting prompted road structures")
    return tuple(row[k] for k in ("scenario", "route_id", "frame_id", "context_id")) + (roads[0],)


def audit(index, data_root, output, cases=None):
    """按路线去重回读；可用同题 eval all_ok 报告各边界桶的实际正确率。"""
    index, output = Path(index), Path(output)
    grouped, seen = defaultdict(list), set()
    counts, bins, metrics = Counter(), defaultdict(Counter), defaultdict(Counter)
    response_bins, response_metrics = defaultdict(Counter), defaultdict(Counter)
    outcomes, matched = {}, set()
    eval_rows = 0
    if cases is not None:
        for line in Path(cases).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            eval_rows += 1
            if type(r.get("all_ok")) is not bool:
                raise ValueError("eval cases require boolean all_ok")
            key = identity(r)
            if key in outcomes and outcomes[key] != r["all_ok"]:
                raise ValueError("conflicting duplicate eval outcomes")
            outcomes[key] = r["all_ok"]
    for line in index.open():
        row = json.loads(line)
        counts["input_rows"] += 1
        if row.get("invalid_action_context") or row.get("gt", {}).get("INVALID_ACTION_CONTEXT") == "YES":
            counts["invalid_rows_skipped"] += 1
            continue
        key = identity(row)
        if key in seen:
            counts["duplicate_rows_skipped"] += 1
            continue
        seen.add(key)
        grouped[key[:2]].append(row)
    print(f"[boundary-discover] routes={len(grouped)} unique_valid={len(seen)}", flush=True)
    records = []
    for i, ((scenario, route), rows) in enumerate(sorted(grouped.items()), 1):
        run = Path(data_root)/scenario/route
        excluded, _ = is_abnormal_lead_route(run, scenario)
        if excluded:
            counts["abnormal_rows_excluded"] += len(rows)
            continue
        trajectory = load_route_trajectory(run)
        if trajectory is None:
            raise ValueError(f"missing raw meta: {run}")
        for row in rows:
            report, labels = diagnose(trajectory, row["frame_id"], row["context_id"])
            context = CONTEXT_BY_ID[row["context_id"]]
            # 候选 raw 五动作/索引按域屏蔽横向必须分别比较。
            expected = row.get("action_labels", row.get("answers"))
            if expected is not None and any(expected.get(k) is not labels[k] for k in context.action_keys):
                raise ValueError("index/raw domain action mismatch")
            stored = row.get("action_evidence", {}).get("future_speeds_exact_mps")
            if stored is not None and stored != trajectory.signals(row["frame_id"])["future_speeds"]:
                raise ValueError("index/raw speed mismatch")
            stored_review = row.get("action_review")
            if stored_review is not None:
                stored_review = {k:v for k,v in stored_review.items() if k != "applies_to_prompt_context"}
                if stored_review != report["action_review"]:
                    raise ValueError("index/raw action review mismatch; rebuild the index")
            counts["audited_unique_valid"] += 1
            flags = report["flags"] or ["no_boundary_flag"]
            for group in ("all", row["context_id"], "action/"+report["primary_action"]):
                bins[group]["total"] += 1
                bins[group]["any_boundary_flag"] += bool(report["flags"])
                for flag in flags:
                    bins[group][flag] += 1
            for group in ("all", row["context_id"], "action/"+report["primary_action"]):
                response_bins[group]["total"] += 1
                response_bins[group]["any_response_flag"] += bool(report["response_flags"])
                for flag in report["response_flags"]:
                    response_bins[group][flag] += 1
            key = identity(row)
            if key in outcomes:
                matched.add(key)
                for flag in report["response_flags"]:
                    response_metrics[flag]["samples"] += 1
                    response_metrics[flag]["exact_hits"] += outcomes[key]
                for flag in ("all", *flags):
                    metrics[flag]["samples"] += 1
                    metrics[flag]["exact_hits"] += outcomes[key]
            records.append({**{k:row[k] for k in ("scenario","route_id","frame_id","context_id")},
                            "prompt_road_structure": identity(row)[-1], **report})
        print(f"[boundary-route] {i}/{len(grouped)} {scenario}/{route}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output/"cases.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    matching = {
        "provided": cases is not None, "input_rows": eval_rows,
        "duplicate_rows": eval_rows-len(outcomes), "unique_cases": len(outcomes),
        "matched_unique_cases": len(matched), "unmatched_unique_cases": len(outcomes)-len(matched),
        "audited_without_case": counts["audited_unique_valid"]-len(matched),
        "status": ("not_requested" if cases is None else "empty_cases" if not outcomes else
                   "no_matches" if not matched else "partial" if len(matched)<len(outcomes) else "all_matched"),
    }
    result = {"audit_version": AUDIT_VERSION, "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
              "review_margin_mps": REVIEW_MARGIN_MPS, "labels_modified": False,
              "scope": "Supplied index only; automatic diagnostics, not new RGB review or noise-rate estimate.",
              "counts": dict(counts), "bins": {k:dict(v) for k,v in bins.items()},
              "response_bins": {k:dict(v) for k,v in response_bins.items()},
              "eval_response_bins": {k:{**v,"exact_accuracy":v["exact_hits"]/v["samples"]} for k,v in response_metrics.items()},
              "eval_matching": matching,
              "eval_bins": {k:{**v,"exact_accuracy":v["exact_hits"]/v["samples"]} for k,v in metrics.items()}}
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if cases is not None:
        print("[boundary-eval-matching] " + json.dumps(matching), flush=True)
        if outcomes and not matched:
            warnings.warn("No evaluation cases matched the audited valid index; "
                          "eval_bins is empty. Check question identities and index scope.", RuntimeWarning)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2]/"lead_data"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="Optional matching eval cases JSONL; never used to change labels")
    args = parser.parse_args()
    report = audit(args.index, args.data_root, args.output, args.cases)
    print(json.dumps(report["counts"]))
