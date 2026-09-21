"""Compare saved pre-v21 audit labels with raw-meta v21 replay; no production writes."""
import argparse,json,sys,pathlib,collections,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory,label_actions,action_evidence,ACTION_RULE_VERSION,action_rule_sha256
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
from qwen3vl_local.sft_new_loop_phase3.action_review import build_action_review
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dirs',nargs='+',required=True,type=pathlib.Path)
    parser.add_argument('--data-root',required=True,type=pathlib.Path)
    parser.add_argument('--output',required=True,type=pathlib.Path)
    args=parser.parse_args()
    counts=collections.Counter();changed=[];inputs=[];seen=set();choices=collections.Counter()
    for folder in args.baseline_dirs:
     for f in sorted(folder.glob('*.json')):
      d=json.loads(f.read_text());route=d['scenario']+'/'+d['route_id']
      if route in seen:continue
      seen.add(route);inputs.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
      t=load_route_trajectory(args.data_root/route);counts['routes']+=1
      for row in d['frames']:
       fid=row['frame_id'];s=t.signals(fid)
       if s is None:continue
       assert [round(v,3) for v in s['future_speeds']]==row['future_speeds'], (route,fid,'baseline speeds changed')
       labels=label_actions(s);counts['frames']+=1;old=row['labels']
       assert old==label_actions({**s,'brake':None,'throttle':None}), (route,fid,'not a v8 baseline')
       if labels!=old:
        assert old is not None and old['STOP'] and labels['RESUME'] and not labels['STOP']
        assert all(labels[k]==old[k] for k in ['DECELERATE','LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT'])
        evidence=action_evidence(s);item={'route':route,'frame':fid,'contexts':row['contexts'],'before':old,'after':labels,'evidence':evidence,'before_choice':row.get('choice'),'after_choice':{},'visual_risk':row['visual_risk']}
        for ctx in row['contexts']:
         try:
          chosen=choice_annotation(labels,ctx,evidence)['primary_action']
         except ValueError:continue
         item['after_choice'][ctx]=chosen
         assert build_action_review(t,fid,ctx,labels,s)['primary_action']==chosen
         choices[chosen]+=1
        changed.append(item);counts['changed_all_frames']+=1
        counts['changed_context_frames']+=bool(row['contexts'])
        counts['changed_context_frames_risk_goal_pass']+=bool(row['contexts']) and not row['visual_risk'] and s['goal_available']
        counts['changed_context_startup_frames']+=bool(row['contexts']) and fid<3
      if counts['routes']%30==0:print(dict(counts),flush=True)
    result={'rule_version':ACTION_RULE_VERSION,'rule_sha256':action_rule_sha256(),'counts':dict(counts),'changed_choice_counts':dict(choices),'changes':changed,'inputs':inputs,'scope':'Supplied development routes; cached v20 labels versus raw meta, no production index rebuild; not a noise-rate estimate'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['changes','inputs']},ensure_ascii=False))


if __name__ == '__main__':
    main()
