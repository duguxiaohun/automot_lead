"""Evidence-bound frame corrections, without rewriting parent labels or sampling.

Review decisions are explicit, portable (relative RGB paths), and image-bound.
A decision never spreads by scenario, route, weather, or model prediction.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from . import CONTRACT
FACTS = ('HIGHWAY', 'STATIC_OBSTACLE', 'VULNERABLE', 'TRAFFIC_LIGHT_ABNORMAL')
ROADS = ('RS1', 'RS2', 'RS4', 'RS5')
ALL = FACTS + ROADS

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def key(row):
    return str(row['scenario']),str(row['route_id']),int(row['frame_id'])

def relative_rgb(value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Review RGB must be relative without traversal')
    parts=p.parts
    if parts and parts[0]=='lead_data': parts=parts[1:]
    if len(parts)!=4 or parts[2]!='rgb': raise ValueError('Expected scenario/route/rgb/file')
    return str(Path(*parts))

def resolve_rgb(root,value):
    # The asset root may itself be a symlink on either server. Content hashes bind it.
    root=Path(root).expanduser().resolve();p=(root/relative_rgb(value)).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError(f'Missing or escaping RGB: {value}')
    return p

def load_reviews(path):
    obj=json.loads(Path(path).read_text())
    if obj.get('contract')!=CONTRACT: raise ValueError('Wrong review contract')
    result={};ids=set()
    for r in obj['decisions']:
        k=key(r)
        if not isinstance(r.get('id'),str) or not r['id'] or r['id'] in ids: raise ValueError('Missing/duplicate review id')
        ids.add(r['id'])
        if k in result: raise ValueError('Duplicate frame decision')
        if r.get('decision') not in {'correct','retain','uncertain'}: raise ValueError('Unknown decision')
        if len(r.get('frame_evidence',[]))!=4 or len(r.get('frame_notes',[]))!=4:
            raise ValueError('Four explicit frame hashes and visual notes required')
        if not all(isinstance(n,str) and n.strip() for n in r['frame_notes']): raise ValueError('Empty visual note')
        if not r.get('reason') or not r.get('reviewer'): raise ValueError('Missing reviewer/reason')
        for f in r['frame_evidence']:
            rel=relative_rgb(f['path'])
            if tuple(Path(rel).parts[:2])!=k[:2]: raise ValueError('RGB from another route')
            if len(f['sha256'])!=64 or any(c not in '0123456789abcdef' for c in f['sha256']): raise ValueError('Invalid SHA')
        expected=r['expected']
        if set(expected['answers'])!=set(ALL) or any(type(x)!=bool for x in expected['answers'].values()):
            raise ValueError('Expected labels must be the full native boolean answer set')
        if expected['rs'] not in {'R1','R2','R3','R4','R5'} or expected['split'] not in {'train','val','test'}:
            raise ValueError('Invalid expected RS/split')
        change=r.get('changes',{})
        if set(change)-{'rs','facts'} or set(change.get('facts',{}))-set(FACTS): raise ValueError('Unsupported label field')
        if any(type(v)!=bool for v in change.get('facts',{}).values()): raise ValueError('Facts must be bool')
        if 'rs' in change and change['rs'] not in {'R1','R2','R3','R4','R5'}: raise ValueError('Invalid new RS')
        if r['decision']=='correct' and not change: raise ValueError('Correction must specify changes')
        if r['decision']!='correct' and change: raise ValueError('Uncertain/retained frames cannot change')
        if type(r.get('development_exposed'))!=bool: raise ValueError('Explicit development exposure required')
        result[k]=r
    return obj,result

def apply(row,decision,data_root):
    if key(row)!=key(decision): raise ValueError('Frame identity mismatch')
    expected=decision['expected']
    for field in ['rs','split','answers']:
        if row[field]!=expected[field]: raise ValueError(f'Stale reviewed {field}: {key(row)}')
    paths=[relative_rgb(p) for p in row['history_rgb_paths']]
    if paths!=[relative_rgb(f['path']) for f in decision['frame_evidence']]: raise ValueError('History frame order mismatch')
    for f in decision['frame_evidence']:
        if sha(resolve_rgb(data_root,f['path']))!=f['sha256']: raise ValueError('Reviewed RGB content changed')
    if decision['decision']!='correct': return copy.deepcopy(row)
    result=copy.deepcopy(row);change=decision['changes'];result['rs']=change.get('rs',row['rs'])
    facts={k:row['answers'][k] for k in FACTS};facts.update(change.get('facts',{}))
    roads={k:result['rs']==k.replace('RS','R') for k in ROADS}
    result.update(answers={**facts,**roads},phase1_answers=facts,phase2_answers=roads)
    if result['answers']==row['answers']: raise ValueError('Correction makes no label change')
    result['phase1_highway_vs_phase2_r3']={'phase1_highway':facts['HIGHWAY'],'phase2_r3_all_no':result['rs']=='R3','matches':facts['HIGHWAY']==(result['rs']=='R3')}
    result['visual_refinement']={'contract':CONTRACT,'review_id':decision['id'],'reason':decision['reason'],
                                 'parent_rs':row['rs'],'parent_answers':row['answers'],
                                 'development_exposed':bool(decision.get('development_exposed',True))}
    return result

def derive(parent_index,reviews,data_root,output):
    parent=Path(parent_index).resolve();out=Path(output).resolve()
    if out==parent.parent or out.is_relative_to(parent.parent) or parent.is_relative_to(out):
        raise ValueError('Output must be separate from parent dataset')
    reviews=Path(reviews).resolve();data_root=Path(data_root).resolve()
    if out.is_relative_to(data_root) or data_root.is_relative_to(out) or reviews.is_relative_to(out):
        raise ValueError('Output overlaps review/RGB input')
    review_sha=sha(reviews);obj,decisions=load_reviews(reviews);parent_sha=sha(parent)
    original=parent.parent/'manifest.json'
    manifest_sha=sha(original) if original.is_file() else None
    manifest=json.loads(original.read_text()) if manifest_sha else {}
    out.mkdir(parents=True,exist_ok=False);seen=set();counts=Counter();changed=[]
    availability=defaultdict(Counter);groups=defaultdict(Counter);consistency=Counter()
    try:
        with parent.open('rb') as source,(out/'frame_index.jsonl').open('xb') as dest:
            for raw in source:
                row=json.loads(raw);k=key(row);counts['rows']+=1;counts['split/'+row['split']]+=1
                new=row
                if k in decisions:
                    if k in seen:raise ValueError('Duplicate reviewed identity in parent')
                    seen.add(k);new=apply(row,decisions[k],data_root)
                split=new['split']
                for label,value in new['answers'].items():
                    availability[split][f'{label}:{"YES" if value else "NO"}']+=1
                    groups[split][f'{"phase1" if label in FACTS else "phase2"}/{"YES" if value else "NO"}']+=1
                highway=new['answers']['HIGHWAY'];r3=new['rs']=='R3'
                consistency[f'{split}/matches/{highway==r3}']+=1
                consistency[f'{split}/phase1_highway/{highway}/phase2_r3/{r3}']+=1
                if new==row:dest.write(raw);counts['unchanged_bytes']+=1
                else:
                    changed.append({'key':list(k),'old':row['answers'],'new':new['answers'],'split':row['split']})
                    dest.write((json.dumps(new,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        missing=set(decisions)-seen
        if missing: raise ValueError(f'Reviewed frames absent from this machine dataset: {sorted(missing)[:5]}')
        if sha(parent)!=parent_sha or sha(reviews)!=review_sha:raise ValueError('Inputs changed during derivation')
        if (sha(original) if original.is_file() else None)!=manifest_sha: raise ValueError('Parent manifest changed')
        for decision in decisions.values():
            for frame in decision['frame_evidence']:
                if sha(resolve_rgb(data_root,frame['path']))!=frame['sha256']:raise ValueError('RGB changed during derivation')
        report={'contract':CONTRACT,'status':'derived_candidate_not_model_validated','parent_index_sha256':parent_sha,
                'parent_manifest_sha256':manifest_sha,'index_sha256':sha(out/'frame_index.jsonl'),'review_sha256':review_sha,'counts':dict(counts),
                'changed_rows':len(changed),'changes':changed,'splits_changed':0,'sampling_changed':False,
                'test_changes_are_development':True,'model_regression_status':'not_run'}
        # Explicit new manifest; native readers get a valid new data identity, never an old ready.
        manifest.update(frame_index=str(out/'frame_index.jsonl'),visual_refinement=report)
        manifest['focus_bin_availability']={k:dict(v) for k,v in availability.items()}
        manifest['phase_group_focus_availability']={k:dict(v) for k,v in groups.items()}
        manifest['cross_phase_highway_r3_consistency_counts']=dict(consistency)
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        (out/'derivation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
    except BaseException as exc:
        (out/'rejected.json').write_text(json.dumps({'status':'rejected','error':str(exc)},ensure_ascii=False)+'\n')
        raise
