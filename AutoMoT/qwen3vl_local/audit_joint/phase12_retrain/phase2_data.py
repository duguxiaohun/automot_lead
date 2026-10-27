"""Image-bound train RS revision; preserve holdouts and native training samplers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from qwen3vl_local.audit_joint.phase1_visual_refine.calibration import relative_rgb, resolve_rgb, sha
from qwen3vl_local.sft_new_loop_phase2.build_dataset import _answers_for, _make_row
from qwen3vl_local.sft_new_loop_phase2.invalid_balance import _cycle_sample
from . import CONTRACT

REVIEW = Path(__file__).with_name('phase2_highway_review.json')
SIGNATURE = 'source=UE3|true_rs=R3|question_domain=LOCAL_JUNCTION'


def key(row):
    return row['scenario'], row['route_id'], int(row['frame_id'])


def derive(index, data_root, output, reviews=REVIEW):
    index, output, reviews = map(lambda p: Path(p).resolve(), (index, output, reviews))
    if output.exists() or index.is_relative_to(output) or output.is_relative_to(index.parent):
        raise ValueError('Choose a fresh separate output directory')
    rgb_root=Path(data_root).resolve()
    if output.is_relative_to(rgb_root) or rgb_root.is_relative_to(output) or reviews.is_relative_to(output) or (output.parent/'selected_train_reviews.json').exists():
        raise ValueError('Output overlaps input or existing review evidence')
    parent_sha, manifest_sha, review_sha = sha(index), sha(index.parent/'manifest.json'), sha(reviews)
    review = json.loads(reviews.read_text())
    if review['contract'] != 'phase2_highway_train_rs_review_v1':
        raise ValueError('Wrong Phase2 review contract')
    corrections = {key(r) for r in review['reviewed_candidate_corrections']}
    controls = {key(r) for r in review['controls']}
    if len(corrections)!=23 or len(controls)!=5 or corrections & controls:
        raise ValueError('Expected 23 reviewed corrections and 5 protected controls')
    approved = corrections | controls
    frame_hashes = dict(review['frames'])
    def verify_images():
        for path, digest in frame_hashes.items():
            if sha(resolve_rgb(data_root,path)) != digest:
                raise ValueError('Reviewed RGB changed: '+path)
    verify_images()
    raw = index.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw]
    revised = copy.deepcopy(rows)
    positives = {}; changed = []
    for i, row in enumerate(rows):
        if key(row) not in approved:
            continue
        if row['split'] != 'train':
            raise ValueError('Reviewed training frame appears in holdout')
        if row.get('target_event_class') != 'UE3':
            continue  # Augmented copies validated separately below.
        if key(row) in positives:
            raise ValueError('Duplicate reviewed UE3 positive')
        if row.get('ue3_subtype')!='HIGHWAY_CUTIN' or row['question_domain']!='ROAD_CORRIDOR' or row.get('invalid_source'):
            raise ValueError('Reviewed positive subtype/domain changed')
        if any(type(v)!=bool for v in row['answers'].values()) or row['answers'] != _answers_for('UE3','ROAD_CORRIDOR',invalid=False):
            raise ValueError('Reviewed UE3 answers changed')
        paths=[relative_rgb(p) for p in row['history_rgb_paths']]
        expected=[f"{row['scenario']}/{row['route_id']}/rgb/{f:04d}.jpg" for f in range(row['frame_id']-3,row['frame_id']+1)]
        if paths!=expected or any(p not in frame_hashes for p in paths) or relative_rgb(row['latest_rgb_path'])!=paths[-1]:
            raise ValueError('Reviewed RGB history/order changed')
        state=(row['true_rs'],row['event'],tuple(row['event_codes']))
        if state==('R3','R-E1',('R-E1',)):
            pass
        elif key(row) in corrections and state==('R4','R-E4',('R-E4',)):
            revised[i].update(true_rs='R3',event='R-E1',event_codes=['R-E1'])
            changed.append(list(key(row)))
        else:
            raise ValueError('Unreviewed RS/event state: '+str(key(row))+': '+str(state))
        positives[key(row)]=revised[i]
    if set(positives)!=approved:
        raise ValueError('Missing reviewed positive frames')
    affected = []
    for i,row in enumerate(rows):
        if row['split']=='train' and row.get('invalid_source')==SIGNATURE:
            if key(row) not in approved or row['true_rs']!='R3' or row['question_domain']!='LOCAL_JUNCTION' or row['answers']!=_answers_for('INVALID','LOCAL_JUNCTION',invalid=True):
                raise ValueError('Unexpected member/label in reviewed INVALID subgroup')
            if row.get('event')!='R-E1' or row.get('event_codes')!=['R-E1'] or row.get('target_event_class')!='INVALID':
                raise ValueError('Unexpected INVALID source event')
            positive=positives[key(row)]
            if [relative_rgb(p) for p in row['history_rgb_paths']] != [relative_rgb(p) for p in positive['history_rgb_paths']]:
                raise ValueError('INVALID RGB history differs from verified positive')
            affected.append(i)
        elif key(row) in approved and row.get('target_event_class')!='UE3':
            raise ValueError('Unexpected augmented copy of reviewed frame')
    if not affected:
        raise ValueError('Missing reviewed INVALID subgroup')
    # Newly valid cross-domain examples come only from the already reviewed 28
    # train positives. Preserve exactly the old subgroup slots and all other rows.
    # This is an explicit new data version, NOT an assertion of full-builder replay.
    regenerate=bool(changed) or {key(rows[i]) for i in affected}!=approved
    if regenerate:
        candidates=[]
        for k,row in sorted(positives.items()):
            base=dict(row,rs=row['true_rs'],primary_event=row['event'])
            candidates.append(_make_row(base=base,question_domain='LOCAL_JUNCTION',
                target_class='INVALID',invalid=True,invalid_source=SIGNATURE))
        replacement=_cycle_sample(candidates,len(affected),random.Random('phase2_highway_train_rs_review_v1'))
        for i,new in zip(affected,replacement):
            revised[i]=new
    result=[line if old==new else (json.dumps(new,ensure_ascii=False)+'\n').encode()
            for line,old,new in zip(raw,rows,revised)]
    protected=[i for i,r in enumerate(rows) if not (key(r) in corrections and r.get('target_event_class')=='UE3') and i not in affected]
    assert all(result[i]==raw[i] for i in protected)
    heldout={split:hashlib.sha256(b''.join(line for line,r in zip(raw,rows) if r['split']==split)).hexdigest() for split in ('val','test')}
    assert heldout=={split:hashlib.sha256(b''.join(line for line,r in zip(result,revised) if r['split']==split)).hexdigest() for split in heldout}
    verify_images()
    if (sha(index),sha(index.parent/'manifest.json'),sha(reviews))!=(parent_sha,manifest_sha,review_sha):
        raise ValueError('Input changed during revision')
    receipt=dict(contract=CONTRACT,revision='phase2_highway_train_rs_review_v1',status='derived_model_pending',
        parent_sha256=parent_sha,parent_manifest_sha256=manifest_sha,review_sha256=review_sha,
        corrected_train_positives=changed,already_correct_train_positives=28-len(changed),
        regenerated_invalid_slots=len(affected) if regenerate else 0,
        invalid_unique_candidates=28,invalid_subgroup=SIGNATURE,
        invalid_rebuild='native _make_row plus native _cycle_sample, fixed revision seed; original subgroup slot count/order retained',
        protected_rows_byte_identical=len(protected),heldouts_byte_identical=True,heldout_sha256=heldout,
        changed_rows=sum(a!=b for a,b in zip(raw,result)),rows=len(rows),
        training_sampler_changed=False,training_quota=1024,max_frame_repeat=10,
        source_metadata_unchanged=True,model_inference=False)
    output.mkdir(parents=True,exist_ok=False)
    (output/'frame_index.jsonl').write_bytes(b''.join(result))
    manifest=json.loads((index.parent/'manifest.json').read_text())
    # Parent sampling/answer/route statistics can be stale after reviewed data
    # membership changes; keep them explicitly as parent evidence, never current.
    parent_stats={k:manifest.pop(k) for k in ('sampling','route_counts','answer_counts','visual_label_risk_counts') if k in manifest}
    manifest.update(frame_index=str(output/'frame_index.jsonl'),fresh_training_revision=receipt,parent_statistics=parent_stats)
    for name,value in [('manifest.json',manifest),('training_revision.json',receipt),('preservation.json',receipt)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (output.parent/'selected_train_reviews.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(derive(a.parent,a.data_root,a.output),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
