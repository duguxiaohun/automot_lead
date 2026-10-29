"""Join physical annotations, preserve native P2 slots, protect BOTH holdouts."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from qwen3vl_local.sft_new_loop_phase1.run_integrity import sha, write
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity, physical_group
from . import CONTRACT

ORDER = {'train': 0, 'val': 1, 'test': 2}
VISIBLE_KEYS=('HIGHWAY','STATIC_OBSTACLE','VULNERABLE','TRAFFIC_LIGHT_ABNORMAL')
RS_KEYS={'RS1':'R1','RS2':'R2','RS4':'R4','RS5':'R5'}


def phase1_targets(visible,rs):
    if rs not in ('R1','R2','R3','R4','R5'):raise ValueError('Invalid authoritative Phase2 RS: '+str(rs))
    if any(type(visible.get(k)) is not bool for k in VISIBLE_KEYS):
        raise ValueError('Missing/non-boolean Phase1 visual facts; events cannot substitute for visibility')
    return {**{k:visible[k] for k in VISIBLE_KEYS},**{k:rs==v for k,v in RS_KEYS.items()}}


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


def build(phase1, phase2, data_root, output, *, historical_groups=()):
    from . import missing_labels
    phase1,phase2,output=map(Path,(phase1,phase2,output))
    if output.exists():raise FileExistsError('Fusion output already exists: '+str(output))
    before={str(p.resolve()):sha(p) for p in (phase1,phase2)}
    reviews,review_sha=missing_labels.reviewed()
    control_keys={tuple(c['key']) for r in reviews.values() for c in r['controls']}
    rows=[json.loads(l) for l in phase2.open() if l.strip()]
    wanted={identity(r) for r in rows}; groups={physical_group(k) for k in wanted}
    first={}; controls={}; split_groups=defaultdict(set)
    for raw in phase1.open():
        row=json.loads(raw);key=identity(row);group=physical_group(key)
        if group in groups:split_groups[group].add(row['split'])
        if key in control_keys:
            if key in controls:raise ValueError('Duplicate Phase1 control anchor')
            controls[key]=row
        if key in wanted:
            if key in first:raise ValueError('Duplicate physical Phase1 anchor')
            first[key]=row
    for row in rows:split_groups[physical_group(identity(row))].add(row['split'])
    for group in historical_groups:split_groups[tuple(group)].add('test')
    missing=wanted-set(first);recovery=None
    if missing:
        recovered,recovery=missing_labels.recover(missing,rows,controls,split_groups,data_root,reviews,review_sha)
        first.update(recovered)
        print(f'[fusion-data] recovered {len(recovered)} reviewed visual-fact anchors; RS/event use Phase2; original slots retained',flush=True)
    conflicts=[];joined=[];transitions=Counter();aligned={};phase2_structures={}
    for row in rows:
        key=identity(row);p1=first[key]
        if [relative_rgb(p) for p in p1['history_rgb_paths']]!=[relative_rgb(p) for p in row['history_rgb_paths']]:
            conflicts.append(dict(key=key,reason='RGB histories differ'))
            continue
        rs=row['true_rs'];answers=phase1_targets(p1['answers'],rs)
        if key in phase2_structures and phase2_structures[key]!=rs:
            raise ValueError('Phase2 itself has conflicting RS copies: '+str(key))
        phase2_structures[key]=rs
        if p1.get('rs')!=rs or any(p1['answers'].get(k)!=answers[k] for k in RS_KEYS):
            aligned[key]=dict(key=list(key),phase1_source_rs=p1.get('rs'),phase2_authoritative_rs=rs,
                old_rs_answers={k:p1['answers'].get(k) for k in RS_KEYS},
                new_rs_answers={k:answers[k] for k in RS_KEYS},visual_facts_unchanged=True)
        split=max(split_groups[physical_group(key)],key=ORDER.__getitem__)
        transitions[row['split']+'->'+split]+=1
        joined.append({**row,'split':split,'history_rgb_paths':[relative_rgb(p) for p in row['history_rgb_paths']],
            'latest_rgb_path':relative_rgb(row['latest_rgb_path']), 'phase1_answers':answers,
            'phase1_source_split':p1['split'],'original_phase2_split':row['split'],'fusion_contract':CONTRACT})
    if conflicts:raise ValueError('Conflicting physical RGB history, explicit audit needed: '+str(conflicts[:10]))
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
    if sha(missing_labels.REVIEW)!=review_sha:raise ValueError('Missing-label review changed while joining')
    if recovery:
        for record in recovery['recovered']:
            if any(image_hashes[p]!=expected for p,expected in record['review']['rgb_sha256'].items()):
                raise ValueError('Reviewed RGB changed while joining')
        write(output/'missing_label_recovery.json',recovery)
    index=output/'frame_index.jsonl'
    with index.open('x') as f:
        for row in joined:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    authority=dict(policy='phase2_rs_event_domain_invalid_phase1_visual_facts_v1',
        authoritative_phase2_fields=['true_rs','event','event_codes','question_domain','answers','target_event_class','invalid_source'],
        phase1_visual_keys=list(VISIBLE_KEYS),rs_answer_map=RS_KEYS,
        aligned_physical_frames=list(aligned.values()),original_phase2_slots_and_labels_unchanged=True,
        historical_reference_labels_unchanged=True)
    write(output/'label_authority.json',authority)
    receipt=dict(contract=CONTRACT,parents=before,rows=len(joined),index_sha256=sha(index),
        split_counts=dict(Counter(r['split'] for r in joined)),split_transitions=dict(transitions),
        split_policy='max_holdout_per_native_physical_group_across_both_parents',
        labels_changed=bool(aligned),existing_visual_facts_changed=False,original_sources_unchanged=True,original_slots_preserved=True,
        phase1_rs_aligned_frames=len(aligned),label_authority_sha256=sha(output/'label_authority.json'),
        missing_phase1_annotations_recovered=len(missing),
        missing_label_recovery_sha256=sha(output/'missing_label_recovery.json') if recovery else None,
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
