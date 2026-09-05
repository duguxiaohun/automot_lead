"""从真实候选动作切换处建立可复现 RGB 审计队列，显式记录尚未查看的覆盖缺口。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def build_queue(candidate_path, routes_path, previous_notes, output, per_context=3):
    """优先选择此前没人工看过的 route，再轮换 Town/场景；只复用已有连续 RGB 图组。"""
    notes = [json.loads(x) for x in previous_notes.read_text().splitlines() if x.strip()]
    seen_routes = {(x['scenario'], x['route_id']) for x in notes}
    route_info = {}
    for line in routes_path.open():
        r = json.loads(line)
        route_info[(r['scenario'], r['route_id'])] = r
    groups = defaultdict(list)
    for line in candidate_path.open():
        row = json.loads(line)
        groups[(row['context_id'], row['scenario'], row['town'], row['route_id'])].append(row)
    candidates = defaultdict(list)
    for (ctx, scenario, town, route), rows in groups.items():
        info = route_info[(scenario, route)]
        rows.sort(key=lambda x:x['frame_id'])
        changes = []
        for a,b in zip(rows, rows[1:]):
            if b['frame_id'] == a['frame_id'] + 1 and a['action_labels'] != b['action_labels']:
                changes.append(b)
        if not changes:
            changes = rows[len(rows)//2:len(rows)//2+1]
        for row in changes:
            for sheet in info['sheets']:
                m = re.search(r'_f(\d+)_to_f(\d+)\.jpg$', sheet)
                if not m or not (int(m[1]) <= row['frame_id'] <= int(m[2])):
                    continue
                candidates[ctx].append(dict(context_id=ctx, scenario=scenario, town=town,
                    route_id=route, frame_id=row['frame_id'], start_frame=int(m[1]),
                    end_frame=int(m[2]), sheet=sheet, action=row['action_labels'],
                    action_evidence=row['action_evidence'], previously_reviewed_route=(scenario,route) in seen_routes))
                break
    queue=[]
    for ctx, items in sorted(candidates.items()):
        used_towns,used_scenarios,used_routes=set(),set(),set()
        for _ in range(per_context):
            remaining=[r for r in items if r['route_id'] not in used_routes]
            if not remaining:break
            remaining.sort(key=lambda r:(r['previously_reviewed_route'],r['town'] in used_towns,
                r['scenario'] in used_scenarios,r['route_id'],r['frame_id']))
            chosen=remaining[0];queue.append(chosen)
            used_towns.add(chosen['town']);used_scenarios.add(chosen['scenario']);used_routes.add(chosen['route_id'])
    for i,r in enumerate(queue):r['index']=i;r['review_status']='PENDING_RGB'
    output.mkdir(parents=True,exist_ok=True)
    (output/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2))
    coverage=Counter((k[0],k[1],k[2]) for k in groups)
    summary=dict(candidate_source=str(candidate_path), selected=len(queue),
        selected_contexts=dict(Counter(r['context_id'] for r in queue)),
        previously_unreviewed_routes=len({r['route_id'] for r in queue if not r['previously_reviewed_route']}),
        eligible_context_scenario_town=[dict(context=c,scenario=s,town=t,routes=n)
            for (c,s,t),n in sorted(coverage.items())],
        manual_confirmation='none_by_this_program', full_dataset_coverage=False)
    (output/'coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return {k:v for k,v in summary.items() if k!='eligible_context_scenario_town'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--previous-notes',type=Path,default=Path(__file__).with_name('rgb_mapping_review_20260905.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-context',type=int,default=3)
    a=p.parse_args()
    print(json.dumps(build_queue(a.candidates,a.routes,a.previous_notes,a.output,a.per_context),ensure_ascii=False))
