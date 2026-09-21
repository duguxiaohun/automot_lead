"""重建后回读原始 meta 核验速度和被问动作；机器核验不声称人工看过 RGB。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, label_actions, recorded_controls, longitudinal_from_signals
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
from qwen3vl_local.sft_new_loop_phase3.preflight import check_index
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import mismatched_road_contexts


def audit(index, data_root, action_output_mode="binary"):
    result = check_index(index, action_output_mode=action_output_mode)
    groups = defaultdict(list)
    for line in Path(index).open():
        row = json.loads(line)
        groups[(row["scenario"], row["route_id"])].append(row)
    checked = 0
    for i, ((scenario, run), rows) in enumerate(sorted(groups.items()), 1):
        directory = Path(data_root) / scenario / run
        if is_abnormal_lead_route(directory, scenario)[0]:
            raise ValueError(f"abnormal route entered index: {directory}")
        trajectory = load_route_trajectory(directory)
        if trajectory is None:
            raise ValueError(f"missing trajectory: {directory}")
        for row in rows:
            signals = trajectory.signals(row["frame_id"])
            if signals is None:
                raise ValueError(f"missing signals: {run}/{row['frame_id']}")
            speeds = signals["future_speeds"]
            evidence = row["action_evidence"]
            if recorded_controls(signals) != recorded_controls(evidence):
                raise ValueError(f"raw anchor controls mismatch: {run}/{row['frame_id']}")
            if longitudinal_from_signals(signals) != evidence["longitudinal_decision"]:
                raise ValueError(f"raw decision mismatch: {run}/{row['frame_id']}")
            if speeds != row["action_evidence"]["future_speeds_exact_mps"]:
                raise ValueError(f"raw speed mismatch: {run}/{row['frame_id']}")
            if row.get("invalid_reason") == "wrong_road_structure":
                allowed = mismatched_road_contexts(
                    true_rs=row["true_rs"], is_junction=signals["is_junction"],
                    distance_to_next_junction=signals["distance_to_next_junction"])
                if (row["context_id"], row["prompt_road_structure"]) not in allowed:
                    raise ValueError(f"unsupported synthetic INVALID: {run}/{row['frame_id']}")
            if not row["invalid_action_context"]:
                labels = label_actions(signals)
                keys = CONTEXT_BY_ID[row["context_id"]].action_keys
                if labels is None or any(labels[k] != row["answers"][k] for k in keys):
                    raise ValueError(f"raw action mismatch: {run}/{row['frame_id']}")
                if "LANE_CHANGE_LEFT" in keys and not signals["lateral_observation_complete"]:
                    raise ValueError(f"unknown lateral supervision: {run}/{row['frame_id']}")
            checked += 1
        if i % 200 == 0:
            print(f"[raw-index-audit] routes={i}/{len(groups)} rows={checked}", flush=True)
    result.update(raw_rows_verified=checked, raw_routes_verified=len(groups),
                  raw_speed_mismatches=0, raw_action_mismatches=0,
                  manual_rgb_confirmation_by_this_program=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument('--action-output-mode', choices=('binary', 'choice'), default='binary')
    args = p.parse_args()
    result = audit(args.index, args.data_root, args.action_output_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "counts"}))


if __name__ == "__main__":
    main()
