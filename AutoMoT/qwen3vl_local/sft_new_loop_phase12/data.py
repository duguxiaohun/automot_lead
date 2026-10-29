"""Join physical annotations, preserve native P2 slots, protect BOTH holdouts."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from qwen3vl_local.sft_new_loop_phase1.run_integrity import sha, write
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity, physical_group
from . import CONTRACT

ORDER = {'train': 0, 'val': 1, 'test': 2}


def relative_rgb(value):
    parts = Path(value).parts
    if 'lead_data' in parts:
        parts = parts[parts.index('lead_data')+1:]
    elif Path(value).is_absolute():
        parts = parts[-4:]
    path = Path(*parts)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 4:
        raise ValueError('Invalid RGB member: '+str(value))
    return path.as_posix()


def resolve_rgb(root, value):
    root = Path(root).resolve();path=(root/relative_rgb(value)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing/escaping RGB: '+str(value))
    return path


def build(phase1, phase2, data_root, output):
    phase1,phase2,output=map(Path,(phase1,phase2,output))
    before={str(p.resolve()):sha(p) for p in (phase1,phase2)}
    rows=[json.loads(l) for l in phase2.open() if l.strip()]
    wanted={identity(r) for r in rows}; groups={physical_group(k) for k in wanted}
    first={}; split_groups=defaultdict(set)
    for raw in phase1.open():
        row=json.loads(raw);key=identity(row);group=physical_group(key)
        if group in groups:split_groups[group].add(row['split'])
        if key in wanted:
            if key in first:raise ValueError('Duplicate physical Phase1 anchor')
            first[key]=row
    for row in rows:split_groups[physical_group(identity(row))].add(row['split'])
    if wanted-set(first):raise ValueError('Missing Phase1 physical labels; no guessed facts allowed')
    conflicts=[];joined=[];transitions=Counter()
    for row in rows:
        key=identity(row);p1=first[key]
        if p1['rs']!=row['true_rs'] or [relative_rgb(p) for p in p1['history_rgb_paths']]!=[relative_rgb(p) for p in row['history_rgb_paths']]:
            conflicts.append(dict(key=key,phase1_rs=p1['rs'],phase2_rs=row['true_rs']))
            continue
        from .prompts import P1
        if set(p1['answers'])!=set(P1.ANSWER_KEYS) or any(type(v)!=bool for v in p1['answers'].values()):
            raise ValueError('Invalid Phase1 boolean annotation')
        split=max(split_groups[physical_group(key)],key=ORDER.__getitem__)
        transitions[row['split']+'->'+split]+=1
        joined.append({**row,'split':split,'history_rgb_paths':[relative_rgb(p) for p in row['history_rgb_paths']],
            'latest_rgb_path':relative_rgb(row['latest_rgb_path']), 'phase1_answers':p1['answers'],
            'phase1_source_split':p1['split'],'original_phase2_split':row['split'],'fusion_contract':CONTRACT})
    if conflicts:raise ValueError('Conflicting physical labels/history, explicit audit needed: '+str(conflicts[:10]))
    output.mkdir(parents=True,exist_ok=False)
    image_hashes={}
    for row in joined:
        for p in row['history_rgb_paths']:
            if p not in image_hashes:
                actual=resolve_rgb(data_root,p)
                with Image.open(actual) as im:im.verify()
                image_hashes[p]=sha(actual)
                if len(image_hashes)%2000==0:print('[fusion-data] RGB checked',len(image_hashes),flush=True)
    if before!={p:sha(p) for p in before}:raise ValueError('Parent data changed while joining')
    index=output/'frame_index.jsonl'
    with index.open('x') as f:
        for row in joined:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    receipt=dict(contract=CONTRACT,parents=before,rows=len(joined),index_sha256=sha(index),
        split_counts=dict(Counter(r['split'] for r in joined)),split_transitions=dict(transitions),
        split_policy='max_holdout_per_native_physical_group_across_both_parents',
        labels_changed=False,original_sources_unchanged=True,original_slots_preserved=True,
        warning='New fusion split protocol. Compare original-test intersections; never compare totals as identical protocols.')
    write(output/'rgb_sha256.json',image_hashes);write(output/'manifest.json',receipt)
    return receipt


def labels(index):
    result={}
    for l in Path(index).open():
        r=json.loads(l);key=identity(r)
        if r.get('fusion_contract')!=CONTRACT:raise ValueError('Not a prepared fusion dataset')
        if key in result and result[key]!=r['phase1_answers']:raise ValueError('Contradictory physical facts')
        result[key]=r['phase1_answers']
    return result
