"""Bounded native raw replay for implementation verification, never formal admission."""
import argparse
from collections import Counter
from pathlib import Path
from . import dataset, planning, regions
from ..geometry_v1 import dataset as native, sources


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--maps',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    selected=[]; counts=Counter()
    for route in native.annotation_routes(args.collection,args.data_root):
        if route['split']!='train':continue
        events={a['event'] for a in route['anchors'] if counts[a['event']]<3}
        anchors=[]
        for event in sorted(events):
            rows=[a for a in route['anchors'] if a['event']==event]
            indices=sorted({round(i*(len(rows)-1)/7) for i in range(8)})
            anchors.extend(rows[i] for i in indices);counts[event]+=1
        if anchors:selected.append(dict(route,anchors=anchors))
        if len(counts)==10 and min(counts.values())>=3:break
    args.output.mkdir(parents=True,exist_ok=False)
    dataset.write(args.output/'selected_routes.json',selected)
    paths=[args.collection/(s+'_result.json') for s in sorted({r['scenario'] for r in selected})]
    native_artifact,native_report=native.build(selected,args.data_root,sources.LaneMap(args.maps),
        args.output/'native',input_paths=paths,require_coverage=False)
    study,receipt=dataset.build(args.output/'native/dataset.json',args.maps,args.output/'study')
    projections={}
    for v in regions.VARIANTS:
        view=dataset.load(args.output/'study/dataset.json',variant=v)
        projections[v]=planning.project(view['records'],per_event=4,world_size=4,epochs=3,seed=20261022)
    if any(x!=projections['a'] for x in projections.values()):raise ValueError('representation_changed_shared_plan')
    dataset.write(args.output/'verification.json',dict(native_anchors=native_report['anchors'],
        common_anchors=receipt['anchors'],rejected=receipt['rejected'],support=receipt['support'],
        route_counts=dict(counts),same_plans_all_four=True,projection=projections['a'],
        native_a_values_unchanged=all(
            all(a==b for a,b in zip(next(x for x in native_artifact['records'] if x['id']==r['id'])['targets'],
                                   r['representations']['a']['targets']) if b['valid']) for r in study['records']),
        source_reverification=True,formal_budget_tested=False,training_executed=False,
        scope='scoped raw implementation replay; no new independent RGB annotation or performance claim'))
    print(dict(native_anchors=native_report['anchors'],common_anchors=receipt['anchors'],same_plans_all_four=True))


if __name__=='__main__':main()
