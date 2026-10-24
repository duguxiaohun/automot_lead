"""Read-only source-bound pool audit, independent label replay, and RGB evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from .frozen import six, trajectory_action as rule, targets
from ..geometry_v1 import sources
from .frozen import label_quarantine

APP = Path(__file__).resolve().parents[3]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def verify_snapshot():
    root=Path(__file__).parent
    p=json.loads((root/'provenance.json').read_text())
    for name,h in p['frozen_files'].items():
        if sha(root/'frozen'/name)!=h:raise ValueError('frozen_single_source_changed: '+name)
    return p

def items(path):
    try:
        import ijson
    except ImportError as exc:
        import sys, shlex
        raise RuntimeError('Offline streaming audit requires ijson; install with: '+
                           shlex.join([sys.executable,'-m','pip','install','ijson'])) from exc
    with Path(path).open('rb') as f:yield from ijson.items(f,'records.item',use_float=True)

def status(case):
    return 'error' if any(t['valid'] and t['target'][p]==0 for t,p in zip(case['record']['targets'],case['prediction']['token_ids'])) else 'correct'

def label_signature(r):
    return {'history':r['observation']['history'],'targets':r['targets']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ready',type=Path,default=APP/'checkpoints/phase3_single_v3/ready_1280.json')
    p.add_argument('--received',type=Path,default=APP/'checkpoints/phase3_single_received_20261024/received/checkpoints/phase3_single_v3/runs/six_1280')
    p.add_argument('--data-root',type=Path,default=APP/'lead_data')
    p.add_argument('--maps',type=Path,default=APP/'checkpoints/phase3_single_v3_maps')
    args=p.parse_args(); provenance=verify_snapshot()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    ready=json.loads(args.ready.read_text());ds=Path(ready['dataset']);before=sha(ds)
    if before!=ready['dataset_sha256']:raise ValueError('local_dataset_sha_mismatch')
    received=args.received/'test/cases.json';received_sha=sha(received)
    complete=json.loads((args.received/'training_complete.json').read_text())
    if complete['binding']!=provenance['received_binding']:raise ValueError('wrong_received_run')
    cases=json.loads(received.read_text()); buckets=defaultdict(list)
    for c in cases:buckets[c['record']['event'],status(c)].append(c)
    selected=[];groups=set()
    for key,rows in sorted(buckets.items()):
        # Choose diverse physical groups first; deterministic ID ordering within stratum.
        c=min(rows,key=lambda c:(c['record']['physical_group'] in groups,c['record']['id']))
        selected.append(dict(record=c['record'],prediction=c['prediction'],stratum='/'.join(key),source='received_test_development'))
        groups.add(c['record']['physical_group'])
    # Additional hard transitions independent of the event examples.
    for a,b in [('STOP','RESUME'),('KEEP','DECELERATE'),('STOP','LANE_CHANGE_LEFT')]:
        rows=[c for c in cases if c['record']['observation']['history'][-1]['name']==a and any(t['name']==b and t['valid'] and t['target'][pred]==0 for t,pred in zip(c['record']['targets'],c['prediction']['token_ids']))]
        if rows:
            c=min(rows,key=lambda c:(c['record']['physical_group'] in groups,c['record']['id']))
            if c['record']['id'] not in {x['record']['id'] for x in selected}:
                selected.append(dict(record=c['record'],prediction=c['prediction'],stratum=a+'->'+b,source='received_test_development'));groups.add(c['record']['physical_group'])
    stats=Counter();support=Counter();hist=Counter();reservoir={};local_match=Counter();lookup={c['record']['id']:c['record'] for c in cases};val=[]
    # Every local eligible record is counted; controls are chosen without model outputs.
    for n,r in enumerate(items(ds),1):
        stats[r['split']]+=1
        if r['id'] in lookup:
            local_match['same_id']+=1;local_match['same_labels']+=label_signature(r)==label_signature(lookup[r['id']])
        if r['split']=='val':val.append(dict(r,variant='six'))
        for i,t in enumerate(r['targets']):
            support[f"{r['split']}/{r['event']}/{i}/{t['name']}"]+=1
        points=r['six_audit']['history_points']
        for i in range(4):
            pair=points[2*i:2*i+2];hist['slots']+=1
            hist['different_valid_origins']+=all(x['action'] for x in pair) and pair[0]['action']!=pair[1]['action']
            hist['imputed_origins']+=sum(bool(x.get('imputed')) for x in pair)
            if any(x.get('imputed') for x in pair) and r['observation']['history'][i]['valid']:hist['slots_changed_if_no_fill']+=1
            expected=targets.compress_history(pair,r['frame'])
            if expected!=r['observation']['history'][i]['name']:raise ValueError('stored_history_compression_mismatch')
        recomputed=targets.window_targets([dict(x,id=t) for x,t in zip(r['six_audit']['future_points'],[s for w in r['targets'] for s in w['source_ids']])])
        for a,b in zip(recomputed,r['targets']):
            if any(a[k]!=b[k] for k in ('valid','raw','target','allowed_answers')):raise ValueError('stored_future_compression_mismatch')
        if r['split'] in ('train','val'):
            for cls in {t['name'] for t in r['targets'] if t['valid'] and t['name']!='MIXED'}:
                key=(r['split'],cls)
                if key not in reservoir or r['id']<reservoir[key]['id']:reservoir[key]=r
        if n%50000==0:print('pool records',n,flush=True)
    # Six training class controls plus two val controls; full val is retained for paired inference.
    controls=[(k,r) for k,r in sorted(reservoir.items()) if k[0]=='train']
    controls += [(k,r) for k,r in sorted(reservoir.items()) if k in [('val','STOP'),('val','RESUME')]]
    for k,r in controls:
        if r['id'] not in {x['record']['id'] for x in selected}:selected.append(dict(record=dict(r,variant='six'),prediction=None,stratum='/'.join(k),source='local_no_prediction_control'))
    write(out/'validation_records.json',val)
    write(out/'pool_report.json',dict(dataset_sha256=before,counts=stats,history=hist,support=support,received_local_overlap=local_match,
        all_stored_history_and_target_compressions_recomputed=True,raw_all_pool_relabelled=False,labels_changed=0,
        train_val_test_membership_changed=False,validation_size=len(val),source='local_single_v3_not_remote_identical_cohort'))
    write(out/'selection.json',selected)
    grouped=defaultdict(list)
    # Recompute all received test records + independently selected train/val controls from raw metadata.
    replay_records={c['record']['id']:c['record'] for c in cases}
    replay_records.update({x['record']['id']:x['record'] for x in selected})
    for r in replay_records.values():grouped[r['scenario'],r['route_id']].append(r)
    lanes=sources.LaneMap(args.maps);results=[];changes=[];raw_sources={}; traces={}
    for j,((scenario,route),rows) in enumerate(sorted(grouped.items()),1):
        root=args.data_root/scenario/route
        paths=sorted((root/'metas').glob('*.pkl'));bound={str(p.relative_to(args.data_root)):sha(p) for p in paths}
        trajectory=rule.load_route_trajectory(root)
        if trajectory is None:raise ValueError('missing trajectory '+str(root))
        quarantines=label_quarantine.load(args.data_root,route=(scenario,route));native_cache={};references={}
        for r in rows:
            action=six.build(dict(r,split=r.get("parent_split",r["split"])),trajectory,quarantines,lanes,native_cache=native_cache,reference_cache=references)
            sig=dict(history=action['history'],targets=action['targets'])
            if sig!=label_signature(r):changes.append(dict(id=r['id'],old=label_signature(r),recomputed=sig))
            results.append(dict(id=r['id'],same=sig==label_signature(r)))
            if r['id'] in {x['record']['id'] for x in selected}:
                frames=[]
                for f in range(max(0,r['frame']-8),r['frame']+22):
                    m=trajectory.metas.get(f);s=trajectory.signals(f)
                    frames.append(dict(frame=f,time=(f-r['frame'])*.25,speed=s['speed'] if s else None,
                        road=m.get('road_id') if m else None,lane=m.get('lane_id') if m else None,
                        decision=rule.longitudinal_from_signals(s) if s else None,
                        native_action=native_cache.get((f,r['event']),{}).get('action')))
                traces[r['id']]=dict(frames=frames,history=action['audit']['history_points'],future=action['audit']['future_points'])
        if bound!={str(p.relative_to(args.data_root)):sha(p) for p in paths}:raise ValueError('raw_meta_changed')
        raw_sources.update(bound)
        print(f'raw replay route {j}/{len(grouped)} anchors={len(results)} differences={len(changes)}',flush=True)
    if sha(ds)!=before or sha(received)!=received_sha:raise ValueError('audit_input_changed')
    write(out/'raw_replay.json',dict(anchors=len(results),routes=len(grouped),results=results,differences=changes,
        received_test_anchors=len(cases),raw_source_hashes=raw_sources,map_hashes=lanes.hashes,
        remote_raw_byte_equivalence_proven=False,independent_frozen_rules=True))
    write(out/'traces.json',traces)
    sheets(out,selected,traces,args.data_root)
    write(out/'manifest.json',dict(received_sha256=received_sha,local_dataset_sha256=before,
        frozen=provenance['frozen_files'],audit_code_sha256=sha(__file__),selected=len(selected),
        manual_review_performed=False,model_inference=False,training_admitted=False,
        outputs={p.name:sha(p) for p in out.glob('*.json')}))
    print('completed',out,flush=True)

def sheets(out,selected,traces,root):
    from PIL import Image,ImageDraw
    (out/'rgb').mkdir();refs=[]
    for idx,x in enumerate(selected):
        r=x['record'];frames=traces[r['id']]['frames'];canvas=Image.new('RGB',(1536,120+len(frames)//3*200+(200 if len(frames)%3 else 0)), 'white');draw=ImageDraw.Draw(canvas)
        title=f"{idx:02d} {x['stratum']} {r['scenario']} / {r['route_id']} anchor={r['frame']}"
        draw.text((8,5),title,fill='black');draw.text((8,25),'GT: '+str([t['name'] for t in r['targets']])+'  PRED: '+str(x['prediction']['names'] if x['prediction'] else 'not inferred'),fill='black')
        draw.text((8,45),'HISTORY: '+str([h['name'] for h in r['observation']['history']]),fill='black')
        draw.text((8,65),'Each panel is full stitched left/front/right RGB. Future images are AUDIT ONLY, never model inputs.',fill='black')
        refs.append(dict(index=idx,id=r['id'],stratum=x['stratum'],frames=[]))
        for k,f in enumerate(frames):
            p=root/r['scenario']/r['route_id']/'rgb'/f"{f['frame']:04d}.jpg";xx=(k%3)*512;yy=120+(k//3)*200
            if p.is_file():
                h=sha(p)
                with Image.open(p) as im:canvas.paste(im.convert('RGB').resize((512,171)),(xx,yy+25))
                refs[-1]['frames'].append(dict(path=str(p.relative_to(root)),sha256=h,frame=f['frame']))
            d=f['decision'];label=d.get('action') or d.get('reason') if d else 'missing'
            draw.text((xx,yy),f"f={f['frame']} t={f['time']:+.2f} v={f['speed']} lane={f['lane']}",fill='black')
            draw.text((xx,yy+12),str(label),fill='black')
        canvas.save(out/'rgb'/f'{idx:02d}.jpg',quality=90)
    write(out/'rgb_sources.json',refs)

if __name__=='__main__':main()
