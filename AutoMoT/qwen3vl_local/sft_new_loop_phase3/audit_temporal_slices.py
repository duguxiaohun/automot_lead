"""量化输入时间不足和阈值边界；只作诊断，不删难例或修改动作真值。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import (
    STOP_SPEED_MPS, LONGITUDINAL_MIN_DELTA_MPS, LONGITUDINAL_RELATIVE_DELTA)


def slices(row):
    """边界带固定 0.05m/s 只报告敏感性，不能用于重设分类门槛。"""
    flags = []
    if len(set(row["history_rgb_paths"])) < 4:
        flags.append("repeated_startup_history")
    if row["invalid_action_context"]:
        return flags
    speeds = row["action_evidence"]["future_speeds_exact_mps"]
    if len(speeds) < 9:
        raise ValueError("valid sample missing exact future speed window")
    threshold = max(LONGITUDINAL_MIN_DELTA_MPS, LONGITUDINAL_RELATIVE_DELTA * max(speeds[0], 1))
    if min(abs(v-STOP_SPEED_MPS) for v in speeds[:7]) <= .05:
        flags.append("speed_within_0.05_of_stop_threshold")
    if min(abs(abs(v-speeds[0])-threshold) for v in speeds[1:]) <= .05:
        flags.append("speed_change_within_0.05_of_delta_threshold")
    if row["answers"]["DECELERATE"]:
        first = next(i for i,v in enumerate(speeds) if speeds[0]-v >= threshold)
        if first >= 7:
            flags.append("deceleration_first_crosses_threshold_after_1.5s")
    if speeds[0] <= STOP_SPEED_MPS and row["answers"]["RESUME"]:
        flags.append("stationary_anchor_future_resume")
    return flags


def audit(path):
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    for line in Path(path).open():
        row = json.loads(line)
        split = row["split"]
        counts[split]["total"] += 1
        for flag in slices(row):
            counts[split][flag] += 1
            key = split+"/"+flag
            if len(examples[key]) < 10:
                examples[key].append({k:row[k] for k in
                    ("scenario","route_id","frame_id","context_id","action_signature")})
    return dict(counts={k:dict(v) for k,v in counts.items()}, examples=dict(examples),
        diagnostic_only=True, filtered_rows=0, notes="Overlapping slices; these flags do not prove label error or RGB visibility. No thresholds were fitted.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = audit(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
