"""Screen proceed YES followed by re-yield; this is not an early-release verdict.

Count one specified RGB mode, bind the same route and instance, expose route-end
censoring and keep raw-rule versus admitted supervision counts separate.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from .label_inventory import verify_replay
from qwen3vl_local.sft_new_loop_phase4.identity import file_sha,write_json


def screen(records,route_ends,*,rgb_mode=4,horizon_frames=8):
    if rgb_mode not in (2,4) or type(horizon_frames) is not int or horizon_frames<1:
        raise ValueError('invalid screening mode or horizon')
    groups=defaultdict(list);seen=set()
    for row in records:
        q=row['question']
        if q['rgb_mode']!=rgb_mode:continue
        key=(q['scenario'],q['route_id'],q['episode']['instance_id'])
        if q['question_id'] in seen:raise ValueError('duplicate question identity')
        seen.add(q['question_id']);groups[key].append(row)
    counts=defaultdict(Counter);windows=[]
    for key,items in sorted(groups.items()):
        items.sort(key=lambda row:(row['question']['frame_id'],row['question']['edge']))
        end=route_ends[key[:2]]
        for row in items:
            q=row['question'];anchor=q['frame_id'];event=q['episode']['event']
            if q['edge']!='proceed' or q['rule_target']!='YES':continue
            later=[x for x in items if anchor<x['question']['frame_id']<=anchor+horizon_frames]
            rechecks=[x for x in later if x['question']['edge']=='re_yield' and x['question']['rule_target']=='YES']
            first=min((x['question']['frame_id'] for x in rechecks),default=None)
            status=('rechecked' if first is not None else 'no_recheck_full_window' if anchor+horizon_frames<=end else 'right_censored')
            counts[event]['raw_proceed_yes']+=1;counts[event][status]+=1
            counts[event]['admitted_proceed_yes']+=int(row['admission_target']=='YES')
            windows.append(dict(scenario=key[0],route_id=key[1],instance_id=key[2],event=event,frame_id=anchor,
                question_id=q['question_id'],admission_target=row['admission_target'],route_end=end,
                requested_end=anchor+horizon_frames,first_recheck=first,status=status,
                frames=[dict(frame_id=x['question']['frame_id'],edge=x['question']['edge'],
                    target=x['question']['rule_target'],admission_target=x['admission_target'],facts=x['question']['facts'],
                    state=x['question']['episode']['state'],longitudinal=x['question']['episode']['longitudinal']) for x in [row]+later],
                early_release_verdict='not_adjudicated'))
    return dict(policy='same_instance_recheck_screen_v1',rgb_mode=rgb_mode,horizon_frames=horizon_frames,
        counts={k:dict(v) for k,v in counts.items()},windows=windows,
        scope='development screening only; a later new blocker, hidden input, label error or instance error needs separate adjudication; anchors are not independent events')


def run(replay,output):
    replay=Path(replay);output=Path(output)
    if output.exists():raise FileExistsError('use a fresh screening report')
    verify_replay(replay)
    request=json.loads((replay/'request.json').read_text())
    ends={(r['scenario'],r['route_id']):max(r['rgb_frames']) for r in request['routes']}
    with (replay/'phase4_questions.jsonl').open() as f:report=screen((json.loads(line) for line in f),ends)
    report['source_receipt_sha256']=file_sha(replay/'receipt.json')
    write_json(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.replay,a.output)
