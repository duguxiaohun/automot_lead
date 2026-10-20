"""Read-only local raw-data experiment; not a production index migration.

Recompute native v9 labels on explicitly supplied historical development anchors.
Never reuse obsolete label hashes, move splits, or infer new event classifications.
"""
import argparse
from collections import Counter
from pathlib import Path
import json
from dataclasses import asdict
from . import dataset, history
from .__main__ import audit
from qwen3vl_local.audit_joint import phase3_base_trial as trial, label_quarantine
from qwen3vl_local.audit_joint.phase3_base_dense.trajectory_targets import project
from qwen3vl_local.sft_new_loop_phase3 import train, trajectory_action as rule, build_dataset as builder
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--legacy-candidates',type=Path)
    group.add_argument('--historical-cases',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history-fill',choices=history.FILL_MODES,default=history.DEFAULT_FILL)
    args=p.parse_args()
    output=args.output.resolve();root=args.data_root.resolve(strict=True)
    if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('Output overlaps raw input')
    output.mkdir(parents=True,exist_ok=False)
    source=args.legacy_candidates or args.historical_cases
    sha=trial.file_sha(source)
    raw=[json.loads(x) for x in source.read_text().splitlines()] if args.legacy_candidates else json.loads(source.read_text())
    counts=Counter();rows=[];groups={}
    for r in raw:
        if r.get('invalid_action_context'):continue
        scenario,route=r['scenario'],r['route_id'];event=r['context_id'];frame=r['frame_id']
        if (scenario,route) not in groups:
            trajectory=rule.load_route_trajectory(root/scenario/route)
            quarantines=label_quarantine.load(root,route=(scenario,route))
            groups[scenario,route]=trajectory,quarantines
        trajectory,quarantines=groups[scenario,route]
        signals=trajectory.signals(frame) if trajectory else None
        native=project(signals,event)
        counts[event+'/requested']+=1
        if native['action'] is None:
            counts[event+'/rejected/'+native['reason']]+=1;continue
        if label_quarantine.phase3_risks(dict(scenario=scenario,route_id=route,frame_id=frame,
                context_id=event,action_evidence=native['evidence']),quarantines):
            counts[event+'/rejected/source_quarantine']+=1;continue
        labels=rule.label_actions(signals)
        answers=builder._answers_for(event,labels,invalid=False)
        if args.legacy_candidates:
            split=r['split'];road=r['prompt_road_structure'];detail=r.get('context_detail','')
            paths=r['history_rgb_paths']
        else:
            split='val';road=r['prompt_spec']['road_structure'];detail=r['prompt_spec'].get('context_detail','')
            paths=r['history_rgb_paths_used']
        paths=[train._resolve_rgb_path(x,root) for x in paths]
        if not Path(paths[-1]).is_file():
            counts[event+'/rejected/current_rgb_missing']+=1;continue
        rows.append(train.FrameRow(scenario=scenario,route_id=route,town=r.get('town',''),frame_id=frame,
            true_rs=road,prompt_road_structure=road,context_id=event,
            question_domain=CONTEXT_BY_ID[event].question_domain,
            action_signature=builder.action_signature(labels,context_id=event),event=r.get('event',''),
            split=split,goal_ego_xy=(signals['goal_x'],signals['goal_y']),history_rgb_paths=paths,
            latest_rgb_path=paths[-1],current_speed_mps=signals['speed'],answers=answers,
            invalid_reason='wrong_road_structure',context_detail=detail))
        counts[event+'/accepted']+=1
    distinct={}
    for row in rows:
        key=dataset.identity(row)
        if key in distinct:
            if asdict(row)!=asdict(distinct[key]):raise ValueError('Conflicting duplicate historical anchor')
            counts[row.context_id+'/exact_duplicate_removed']+=1
        else:distinct[key]=row
    rows=list(distinct.values())
    artifact,report=dataset.build(rows,root,output/'data',history_policy='offline_native',history_fill=args.history_fill)
    if sha!=trial.file_sha(source):raise ValueError('Historical source changed')
    if any(r.split=='train' for r in rows):
        sampling=audit(rows,artifact,output/'sampling.json')
    else:sampling=None
    result=dict(source_sha256=sha,source=str(source),scope='local development native-label/data audit; no model scores',
        selection_counts=dict(counts),dataset=report,sampling_status=sampling['status'] if sampling else None,
        fixed_members_preserved=len(rows)==sum(v for k,v in counts.items() if k.endswith('/requested')),
        historical_fixed_val_ids_match=([dataset.identity(r) for r in rows]==trial.reference()['validation']['generation']['selected_ids']) if args.historical_cases else None,
        original_index_rewritten=False,model_training=False,history_policy='offline_native',
        history_fill=args.history_fill,
        all_history_unknown=all(r['history']==['UNKNOWN']*4 for r in artifact['records'].values()),
        sample_rgb_count=1,paired_ablation_note='History ablation must replace the native history tokens with UNKNOWN; no ablation scores computed here',
        history_token_counts=dict(Counter(a for r in artifact['records'].values() for a in r['history'])))
    trial.write(output/'local_check.json',result)
    print(json.dumps(dict(anchors=len(rows),sampling=result['sampling_status'],history_all_unknown=result['all_history_unknown'])))


if __name__=='__main__':main()
