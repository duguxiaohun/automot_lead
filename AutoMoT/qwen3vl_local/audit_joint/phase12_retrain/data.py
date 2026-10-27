"""Train-only reviewed derivation; holdouts and unreviewed rows stay byte-identical."""
import hashlib
import json
from pathlib import Path
from qwen3vl_local.audit_joint.phase1_visual_refine import calibration as parent
from qwen3vl_local.audit_joint.phase1_visual_refine_v2 import calibration as cal
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from . import CONTRACT


def derive(index,reviews,data_root,output):
    index,reviews,output=map(Path,(index,reviews,output))
    obj,decisions=cal.load_reviews(reviews)
    if not decisions or any(d['expected']['split']!='train' or d['decision']!='correct' for d in decisions.values()):
        raise ValueError('Fresh-training review must contain only explicit train corrections')
    # Different machines can have different route membership. No copied local ready.
    before=parent.sha(index);seen=set()
    heldout={k:hashlib.sha256() for k in ('val','test')}
    for line in index.open('rb'):
        row=json.loads(line);key=parent.key(row)
        if row['split'] in heldout:heldout[row['split']].update(line)
        if key in decisions:
            if key in seen:raise ValueError('Duplicate reviewed physical frame')
            parent.apply(row,decisions[key],data_root);seen.add(key)
    if not seen:raise ValueError('No reviewed train frames on this machine; cannot claim corrected training data')
    if parent.sha(index)!=before:raise ValueError('Parent index changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    selected=output.parent/'selected_train_reviews.json'
    with selected.open('x') as f:
        json.dump({**obj,'decisions':[d for d in obj['decisions'] if parent.key(d) in seen]},f,ensure_ascii=False,indent=2)
    result=cal.derive(index,selected,data_root,output)
    new={k:hashlib.sha256() for k in heldout}
    for line in (output/'frame_index.jsonl').open('rb'):
        row=json.loads(line)
        if row['split'] in new:new[row['split']].update(line)
    expected={k:v.hexdigest() for k,v in heldout.items()}
    if {k:v.hexdigest() for k,v in new.items()}!=expected:raise ValueError('Holdout changed')
    if parent.sha(index)!=before:raise ValueError('Parent changed after derivation')
    record=dict(contract=CONTRACT,status='derived_train_only_model_pending',
        requested_reviews=len(decisions),applied_train_reviews=len(seen),
        absent_reviews=[list(k) for k in sorted(set(decisions)-seen)],
        parent_sha256=before,heldout_sha256=expected,heldouts_byte_identical=True,
        review_sha256=parent.sha(reviews),selected_review_sha256=parent.sha(selected),
        preservation=result['preservation'],sampling_algorithm_changed=False,
        warning='Label corrections can change train-bin membership; no forced sampling of reviewed frames.')
    proof.write(output/'training_revision.json',record)
    manifest=proof.read(output/'manifest.json');manifest['fresh_training_revision']=record
    proof.write(output/'manifest.json',manifest)
    return record
