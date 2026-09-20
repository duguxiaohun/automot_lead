"""重放 v15 已逐帧复核样本，并检查重建索引；机器核验不代替人工视觉审阅。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
from qwen3vl_local.sft_new_loop_phase3.build_dataset import physical_route_group, development_route_groups
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation, validate_choice_row
from qwen3vl_local.sft_new_loop_phase3.prompts import (
    make_prompt_spec, build_action_target, build_action_prompt, parse_action_output, spec_answers,
)
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, label_actions, action_evidence


def audit(data_root, notes_path, index_path=None):
    """先排除异常路线，再核对历史输入、实际轨迹和正向监督；不写回原始标注。"""
    root = Path(__file__).resolve().parents[3]
    notes = [json.loads(line) for line in Path(notes_path).read_text().splitlines()]
    counts, routes, route_checks = Counter(), set(), []
    print(f"[causal-review-discover] cases={len(notes)}", flush=True)
    for i, note in enumerate(notes):
        run = Path(data_root) / note["scenario"] / note["route_id"]
        excluded, check = is_abnormal_lead_route(run, note["scenario"])
        if excluded:
            raise ValueError(f"abnormal route: {run}")
        route_checks.append(check)
        group = physical_route_group(note["scenario"], note["route_id"])
        if group not in development_route_groups():
            raise ValueError(f"review route is not train-only: {group}")
        routes.add(group)
        hashes = [hashlib.sha256((run/"rgb"/f"{f:04d}.jpg").read_bytes()).hexdigest()
                  for f in note["input_frames"]]
        if hashes != note["input_rgb_sha256"]:
            raise ValueError(f"changed RGB: {run}")
        sheet = root / note["evidence_sheet"]
        if hashlib.sha256(sheet.read_bytes()).hexdigest() != note["evidence_sheet_sha256"]:
            raise ValueError(f"changed reviewed sheet: {sheet}")
        signals = load_route_trajectory(run).signals(note["frame_id"])
        labels = label_actions(signals)
        annotation = choice_annotation(labels, note["context_id"], action_evidence(signals))
        # 历史视觉笔记保留当时的协议版本；重放检查动作和证据，不倒写旧审计记录。
        if labels != note["raw_action_labels"] or any(note.get(k) != v for k,v in annotation.items()
                                                     if k != "primary_action_version"):
            raise ValueError(f"review label mismatch: {run} frame={note['frame_id']}")
        counts[f"{note['context_id']}/{annotation['primary_action']}"] += 1
        print(f"[causal-review] {i+1}/{len(notes)} {note['context_id']} {annotation['primary_action']}", flush=True)
    index_counts = Counter()
    if index_path:
        for line in Path(index_path).open():
            row = json.loads(line)
            validate_choice_row(row)
            binary_spec = make_prompt_spec(variant="all_random_order", answers=row["answers"], seed_key="audit",
                context_id=row["context_id"], road_structure=row["prompt_road_structure"],
                action_output_mode="binary")
            binary_target = build_action_target(binary_spec)
            if parse_action_output(binary_target, spec=binary_spec) != spec_answers(binary_spec):
                raise ValueError("binary target/parser mismatch")
            if row["invalid_action_context"]:
                if row["primary_action"] is not None or row["keep_scope"] is not None:
                    raise ValueError("invalid row must not become KEEP")
                index_counts["INVALID"] += 1
                continue
            annotation = choice_annotation(row["answers"], row["context_id"], row["action_evidence"])
            if any(row.get(k) != v for k,v in annotation.items()):
                raise ValueError("index annotation mismatch")
            spec = make_prompt_spec(variant="all_random_order", answers=row["answers"], seed_key="audit",
                context_id=row["context_id"], road_structure=row["prompt_road_structure"],
                context_detail=row["context_detail"], current_speed_mps=row["current_speed_mps"],
                goal_xy=row["goal_ego_xy"], action_output_mode="choice")
            target = build_action_target(spec)
            if target != row["primary_action"] or parse_action_output(target, spec=spec) != spec_answers(spec):
                raise ValueError("target/parser mismatch")
            scene = build_action_prompt(spec=spec).split("[SCENE_CONTEXT]",1)[1].split("[/SCENE_CONTEXT]",1)[0]
            if "High-level purpose" in scene or "Check from the visible history" in scene:
                raise ValueError("purpose leaked into scene")
            index_counts[f"{row['context_id']}/{target}"] += 1
    return dict(scope="26 reviewed development clips, not full-dataset visual audit or model evaluation",
                reviewed_clips=len(notes), reviewed_frame_panels=sum(len(n["reviewed_frames"]) for n in notes),
                physical_routes=len(routes), route_checks=route_checks, reviewed_labels=dict(counts),
                index_rows=sum(index_counts.values()), index_counts=dict(index_counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2]/"lead_data"))
    parser.add_argument("--notes", default=str(Path(__file__).with_name("causal_action_rgb_notes_20260920.jsonl")))
    parser.add_argument("--index")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.data_root, args.notes, args.index)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"[causal-review-complete] clips={report['reviewed_clips']} index_rows={report['index_rows']}")
