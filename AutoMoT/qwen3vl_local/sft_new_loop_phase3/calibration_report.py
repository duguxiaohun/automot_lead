"""离线标定诊断：回读原始轨迹、RGB SHA及当前阈值边界；不拟合阈值、不重标注。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from .paired_eval import load_cases, temporal_slices
from .regression_gate import scored
from .trajectory_action import load_route_trajectory, longitudinal_from_signals, longitudinal_decision


def audit(paths, root):
    cache={};seen={};errors=[];details=[];groups=defaultdict(Counter);sources={}
    for tag,path in zip('ABCD',paths):
        rows,hashes=load_cases(path);sources.update(hashes)
        for row in rows.values():
            ok,_=scored(row)
            rgb=row['history_rgb_paths_used'];shas=row['history_rgb_sha256']
            if len(rgb)!=len(shas): raise ValueError('missing RGB SHA')
            for p,expected in zip(rgb,shas):
                file=Path(p) if Path(p).is_absolute() else root/p
                if file not in seen: seen[file]=hashlib.sha256(file.read_bytes()).hexdigest()
                if seen[file]!=expected: errors.append([tag,row['case_index'],'RGB',p])
            if row['prompt_spec']['invalid_context']: continue
            key=(row['scenario'],row['route_id']);fr=row['frame_id']
            if key not in cache: cache[key]=load_route_trajectory(root/'lead_data'/key[0]/key[1])
            trajectory=cache[key];s=trajectory.signals(fr);e=row['action_evidence']
            expected={'future_speeds_exact_mps':s['future_speeds'],
                'longitudinal_decision':longitudinal_from_signals(s),
                'lane_change_direction':s['lane_change_direction'] or ''}
            expected.update({k:s[k] for k in ['brake','throttle','lane_id','road_id']})
            for field,value in expected.items():
                actual=e[field] or '' if field=='lane_change_direction' else e[field]
                if actual!=value: errors.append([tag,row['case_index'],field])
            d=expected['longitudinal_decision'];speeds=s['future_speeds'][:9]
            # Counterfactual removal is observability diagnostics, never an alternative target.
            without_controls=longitudinal_decision(speeds)
            control_sensitive=without_controls.get('action')!=d.get('action')
            flags=temporal_slices(row)
            if control_sensitive: flags.append('target_depends_on_unshown_anchor_controls')
            if d.get('first_drop_s') is not None and speeds[-1]>min(speeds): flags.append('drop_then_recovery')
            record=dict(mode=tag,case_index=row['case_index'],scenario=key[0],route_id=key[1],frame_id=fr,
                correct=ok,flags=flags,action=d.get('action'),reason=d.get('reason'),
                stop_confirmed_s=d.get('stop_confirmed_s'),
                stop_confirmation_margin_s=None if d.get('stop_confirmed_s') is None else 1.5-d['stop_confirmed_s'],
                first_drop_s=d.get('first_drop_s'),gain_confirmed_s=d.get('gain_confirmed_s'),
                threshold_mps=d.get('delta_threshold_mps'),
                nearest_stop_speed_margin_mps=min(abs(v-.5) for v in speeds[:7]),
                control_sensitive=control_sensitive)
            details.append(record)
            for flag in ['all_valid',*flags]:
                groups[tag+'/'+flag]['cases']+=1;groups[tag+'/'+flag]['correct']+=int(ok)
    return dict(schema='phase3_calibration_diagnostic_v1',diagnostic_only=True,thresholds_changed=False,
        labels_changed=0,source_sha256=sources,runs=len(cache),metas_read=sum(len(t.metas) for t in cache.values()),
        distinct_input_rgb_verified=len(seen),raw_mismatches=errors,
        groups={k:dict(v) for k,v in sorted(groups.items())},details=details,
        interpretation='Known development cases; metadata precision is not RGB observability. Threshold margins and missing-control counterfactuals are diagnostics only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs=4,type=Path,required=True)
    p.add_argument('--automot-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.cases,args.automot_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['runs','metas_read','distinct_input_rgb_verified','raw_mismatches']}))
    return 1 if result['raw_mismatches'] else 0

if __name__=='__main__': raise SystemExit(main())
