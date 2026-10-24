"""Rebuild the entire existing eligible single cohort's labels using independent frozen rules.

Does not add/remove candidates, change splits, or publish training readiness.
"""
import argparse
from collections import Counter
import gzip
from itertools import groupby
import json
from pathlib import Path
from .audit import APP,sha,write,items,verify_snapshot,label_signature
from .frozen import six,trajectory_action as rule
from ..geometry_v1 import sources
from .frozen import label_quarantine


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ready',type=Path,default=APP/'checkpoints/phase3_single_v3/ready_1280.json')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data');p.add_argument('--maps',type=Path,default=APP/'checkpoints/phase3_single_v3_maps');args=p.parse_args()
    frozen=verify_snapshot();ready=json.loads(args.ready.read_text());ds=Path(ready['dataset']);ds_sha=sha(ds)
    if ready['status']!='ready' or ds_sha!=ready['dataset_sha256']:raise ValueError('ready_mismatch')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'started.json',dict(source_dataset_sha256=ds_sha,frozen=frozen['frozen_files'],code_sha256=sha(__file__),completed=False))
    lanes=sources.LaneMap(args.maps);counts=Counter();seen=set();changes=[];raw_count=0
    with gzip.open(out/'labels.jsonl.gz','wt') as bank,(out/'raw_sources.jsonl').open('x') as proof:
        for (scenario,route),group in groupby(items(ds),key=lambda r:(r['scenario'],r['route_id'])):
            if (scenario,route) in seen:raise ValueError('noncontiguous_route_members')
            seen.add((scenario,route));rows=list(group);root=args.data_root/scenario/route
            paths=sorted((root/'metas').glob('*.pkl'));bound={str(p.relative_to(args.data_root)):sha(p) for p in paths}
            tr=rule.load_route_trajectory(root)
            if tr is None:raise ValueError('missing_trajectory')
            quarantine=label_quarantine.load(args.data_root,route=(scenario,route));cache={};references={}
            for r in rows:
                action=six.build(dict(r,split=r.get('parent_split',r['split'])),tr,quarantine,lanes,native_cache=cache,reference_cache=references)
                sig=dict(history=action['history'],targets=action['targets'])
                same=sig==label_signature(r);eligible=action['anchor_action'] is not None and any(t['valid'] for t in action['targets'])
                counts[r['split']]+=1;counts['same']+=same;counts['eligible']+=eligible
                if not same or not eligible:changes.append(dict(id=r['id'],same=same,eligible=eligible,old=label_signature(r),new=sig))
                bank.write(json.dumps(dict(id=r['id'],split=r['split'],event=r['event'],**sig),separators=(',',':'))+'\n')
            if paths!=sorted((root/'metas').glob('*.pkl')) or bound!={str(p.relative_to(args.data_root)):sha(p) for p in paths}:raise ValueError('raw_source_changed')
            for name,h in bound.items():proof.write(json.dumps(dict(path=name,sha256=h))+'\n')
            proof.flush();raw_count+=len(bound)
            if len(seen)%25==0:print(f'routes={len(seen)} anchors={counts["train"]+counts["val"]+counts["test"]} differences={len(changes)}',flush=True)
    if sha(ds)!=ds_sha:raise ValueError('dataset_changed')
    verify_snapshot()
    write(out/'completed.json',dict(status='reproduced' if not changes else 'differences_require_review',counts=counts,routes=len(seen),raw_metadata_files=raw_count,
        differences=changes,label_bank_sha256=sha(out/'labels.jsonl.gz'),raw_sources_sha256=sha(out/'raw_sources.jsonl'),
        dataset_sha256=ds_sha,maps=lanes.hashes,training_admitted=False,
        scope='all existing local eligible anchors; excluded parent candidates not rescanned',new_rule_thresholds=False))
    print('complete',dict(counts),flush=True)
    return 2 if changes else 0

if __name__=='__main__':raise SystemExit(main())
