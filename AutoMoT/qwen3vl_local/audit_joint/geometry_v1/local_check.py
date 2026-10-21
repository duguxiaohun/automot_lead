"""Explicit small local source check, using old anchors ONLY as test locations.

This is not the production candidate builder, which starts before old action
filtering. No GPU, old labels, remote access or simulation server is used here.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from . import dataset,sources,planning


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--locations',type=Path,required=True,help='Existing single_frame dataset: test locations only')
    p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();original=json.loads(a.locations.read_text())
    groups=defaultdict(list);counts=Counter()
    for r in original['records'].values():
        if counts[r['event']]>=6:continue
        groups[(r['scenario'],r['route_id'],r['split'])].append(dict(
            frame=r['anchor'],event=r['event'],road_structure='UNSPECIFIED_LOCAL_SMOKE'))
        counts[r['event']]+=1
    routes=[dict(scenario=s,route_id=r,town=r.split('_')[0],split=split,anchors=items)
            for (s,r,split),items in sorted(groups.items())]
    artifact,receipt=dataset.build(routes,original['data_root'],sources.LaneMap(a.maps),a.output,
        input_paths=[a.locations],require_coverage=False)
    reloaded=dataset.load(a.output/'dataset.json')
    if reloaded!=artifact:raise ValueError('reload_changed')
    actual=Counter(r['event'] for r in artifact['records'])
    if set(actual)!=set(counts):raise ValueError('smoke_event_coverage_lost')
    if sum(actual.values()) < sum(counts.values())*.9:raise ValueError('unexpected_smoke_rejections')
    _,plan=planning.plan(artifact['records'],per_event=4,world_size=4,epochs=3)
    receipt.update(smoke_locations_only=True,old_action_values_used=False,
        production_coverage_not_certified=True,plan=plan,
        history_valid=sum(h['valid'] for r in artifact['records'] for h in r['observation']['history']),
        future_valid=sum(t['valid'] for r in artifact['records'] for t in r['targets']))
    dataset.write(a.output/'local_check.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('counts','physical_groups','plan')},ensure_ascii=False))


if __name__=='__main__':main()
