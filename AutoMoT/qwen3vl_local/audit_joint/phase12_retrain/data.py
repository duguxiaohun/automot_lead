"""Train-only reviewed derivation; holdouts and unreviewed rows stay byte-identical."""
import hashlib
import copy
import json
from pathlib import Path
from qwen3vl_local.audit_joint.phase1_visual_refine import calibration as parent
from qwen3vl_local.audit_joint.phase1_visual_refine_v2 import calibration as cal
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from . import CONTRACT


def local_review(row, decision, data_root):
    """Accept only the reviewed old value or the reviewed target, never a third label.

    Keep the original signed-by-hash review untouched. The selected local review
    explicitly binds this machine's before-state, including already-correct rows.
    """
    expected=decision['expected']; changes=decision['changes']
    current={k:row[k] for k in ('rs','split','answers')}
    target=copy.deepcopy(expected)
    if 'rs' in changes:
        target['rs']=changes['rs']
        for key in parent.ROADS:target['answers'][key]=target['rs']==key.replace('RS','R')
    target['answers'].update(changes.get('facts',{}))
    if row['split']!='train':raise ValueError('Reviewed frame is not in train on this machine')
    if set(row['answers'])!=set(parent.ALL) or any(type(v)!=bool for v in row['answers'].values()):
        raise ValueError('Invalid native boolean answers')
    if any(row['answers'][k]!=(row['rs']==k.replace('RS','R')) for k in parent.ROADS):
        raise ValueError('Inconsistent local rs/answers')
    if row['rs'] not in (expected['rs'],target['rs']):
        raise ValueError('Local rs is neither reviewed old label nor reviewed target')
    for key in parent.ALL:
        if row['answers'][key] not in (expected['answers'][key],target['answers'][key]):
            raise ValueError('Unreviewed answer differs: '+key)
    selected=copy.deepcopy(decision)
    selected['expected']=copy.deepcopy(current)
    remaining={}
    if row['rs']!=target['rs']:remaining['rs']=target['rs']
    facts={k:v for k,v in changes.get('facts',{}).items() if row['answers'][k]!=v}
    if facts:remaining['facts']=facts
    selected.update(decision='correct' if remaining else 'retain',changes=remaining,
                    machine_reconciliation=dict(policy='reviewed_old_or_target_only_v1',
                        original_expected=copy.deepcopy(expected),reviewed_target=target,
                        status='correction_needed' if remaining else 'already_matches_reviewed_target'))
    # Retain decisions still verify identity, ordered frame paths and all four RGB
    # hashes. Already-correct cannot bypass image checks or enter test/val.
    result=parent.apply(row,selected,data_root)
    if any(result[k]!=target[k] for k in ('rs','split','answers')):
        raise ValueError('Reconciled output differs from reviewed target')
    return selected


def derive(index,reviews,data_root,output):
    index,reviews,output=map(Path,(index,reviews,output))
    obj,decisions=cal.load_reviews(reviews)
    if not decisions or any(d['expected']['split']!='train' or d['decision']!='correct' for d in decisions.values()):
        raise ValueError('Fresh-training review must contain only explicit train corrections')
    # Different machines can have different route membership. No copied local ready.
    before=parent.sha(index);review_before=parent.sha(reviews);seen=set();selected_decisions={};conflicts=[]
    heldout={k:hashlib.sha256() for k in ('val','test')}
    for line in index.open('rb'):
        row=json.loads(line);key=parent.key(row)
        if row['split'] in heldout:heldout[row['split']].update(line)
        if key in decisions:
            if key in seen:raise ValueError('Duplicate reviewed physical frame')
            seen.add(key)
            try:selected_decisions[key]=local_review(row,decisions[key],data_root)
            except ValueError as exc:
                conflicts.append(dict(key=list(key),reason=str(exc),
                    local={k:row[k] for k in ('rs','split','answers')},
                    reviewed_expected=decisions[key]['expected'],reviewed_changes=decisions[key]['changes']))
    if conflicts:
        output.parent.mkdir(parents=True,exist_ok=True)
        report=output.parent/'train_review_conflicts.json'
        with report.open('x') as f:
            json.dump(dict(status='rejected',policy='reviewed_old_or_target_only_v1',
                parent_sha256=before,review_sha256=review_before,conflicts=conflicts),f,ensure_ascii=False,indent=2)
        print('[REVIEW_CONFLICT] '+json.dumps(conflicts,ensure_ascii=False),flush=True)
        raise ValueError(f'{len(conflicts)} local train review conflict(s); actual/expected/target saved in {report}')
    if not seen:raise ValueError('No reviewed train frames on this machine; cannot claim corrected training data')
    if parent.sha(index)!=before or parent.sha(reviews)!=review_before:raise ValueError('Parent index or reviews changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    selected=output.parent/'selected_train_reviews.json'
    with selected.open('x') as f:
        json.dump({**obj,'decisions':[selected_decisions[parent.key(d)] for d in obj['decisions'] if parent.key(d) in seen]},f,ensure_ascii=False,indent=2)
    result=cal.derive(index,selected,data_root,output)
    new={k:hashlib.sha256() for k in heldout}
    for line in (output/'frame_index.jsonl').open('rb'):
        row=json.loads(line)
        if row['split'] in new:new[row['split']].update(line)
    expected={k:v.hexdigest() for k,v in heldout.items()}
    if {k:v.hexdigest() for k,v in new.items()}!=expected:raise ValueError('Holdout changed')
    if parent.sha(index)!=before or parent.sha(reviews)!=review_before:raise ValueError('Parent or reviews changed after derivation')
    changed=sum(d['decision']=='correct' for d in selected_decisions.values())
    record=dict(contract=CONTRACT,status='derived_train_only_model_pending',
        requested_reviews=len(decisions),matched_train_reviews=len(seen),applied_train_reviews=changed,
        already_correct_train_reviews=len(seen)-changed,
        reconciliation_policy='reviewed_old_or_target_only_v1',
        absent_reviews=[list(k) for k in sorted(set(decisions)-seen)],
        parent_sha256=before,heldout_sha256=expected,heldouts_byte_identical=True,
        review_sha256=parent.sha(reviews),selected_review_sha256=parent.sha(selected),
        preservation=result['preservation'],sampling_algorithm_changed=False,
        warning='Label corrections can change train-bin membership; no forced sampling of reviewed frames.')
    proof.write(output/'training_revision.json',record)
    manifest=proof.read(output/'manifest.json');manifest['fresh_training_revision']=record
    proof.write(output/'manifest.json',manifest)
    return record
