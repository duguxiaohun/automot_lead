"""2026-10-29 opt-in local revision; original snapshots stay immutable."""
import ast
import hashlib
import json
from pathlib import Path

from qwen3vl_local.audit_joint.phase1_visual_refine import calibration as cal
from . import data

REVISION = 'visual_20261029'
REVISION_P2_EVIDENCE = 'evidence_20261030'
REVISIONS = ('legacy_20261027', REVISION, REVISION_P2_EVIDENCE)
REVIEW = Path(__file__).with_name('phase1_vulnerable_review_20261029.json')

P1_RULE = '''
VULNERABLE witness and absence check:
Inspect the left/front/right camera panels separately in temporal order. A small rider beside a junction mouth or a pedestrian at a curb can count before entering the lane; a clear body/rider silhouette in one available frame is sufficient if the same person remains relevant at the newest moment. Include incidental people and cyclists even when the main interaction is with a vehicle. Do not require a pedestrian crossing event.
Track the witness to the newest frame. Brief occlusion by a passing vehicle does not erase a still-relevant person, but do not carry YES forward after the person has clearly left the forward/turning conflict area. If no available frame resolves a person or rider, a crosswalk, pole, parking meter, headlight reflection, or hidden actor is not positive visual evidence. Do not reconstruct missing middle frames in endpoint mode.'''

P2_RULE = '''
UE3 temporal check (same actor, same road-relative corridor):
First locate ego's corridor from lane boundaries; then compare the other vehicle's nose/body to that corridor in every available frame. Entry may start from either side and need not finish at the lane center. A visible boundary crossing or progressive encroachment toward ego's immediate corridor is sufficient without separately observing ego braking. A vehicle still straddling the boundary while moving into the corridor is not yet a settled lead vehicle.
Finally recheck the newest moment: preserve YES while entry or its visible immediate conflict continues; return NO after the vehicle has settled and neither further entry nor an immediate cut-in effect is visible. Do not extend an event merely because it happened earlier. Distinguish lateral entry from ego passing a parked/queued car or changing lanes itself. With endpoints only, use the changed relation to the lane boundary, not image displacement or apparent growth alone. A stable neighboring vehicle and an already-settled lead are negative controls.'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Reviewed source anchor changed: '+old[:100])
    return source.replace(old,new,1)


def phase1_prompt(source):
    # Called AFTER the previous RS2/RS4 corrections. Only this phase's module
    # changes; shared historical prompt modules and their samplers are untouched.
    for old,new in [
        ('VARIANT_WEIGHTS = dict(phase2_prompts.VARIANT_WEIGHTS)', 'VARIANT_WEIGHTS = {"all_random_order": 1}'),
        ('TRAIN_VARIANT_WEIGHTS = dict(phase2_prompts.TRAIN_VARIANT_WEIGHTS)', 'TRAIN_VARIANT_WEIGHTS = dict(VARIANT_WEIGHTS)'),
        ('VARIANT_ORDER = tuple(phase2_prompts.VARIANT_ORDER)', 'VARIANT_ORDER = tuple(VARIANT_WEIGHTS)'),
        ('SUBSET_COUNTS = tuple(phase2_prompts.SUBSET_COUNTS)', 'SUBSET_COUNTS = ()'),
        ('GROUP_DEFINITIONS = phase2_prompts.GROUP_DEFINITIONS', 'GROUP_DEFINITIONS = {}'),
        ('    phase2_answers = {key:', '    if variant != "all_random_order":\n        raise ValueError("This revision requires all eight direct questions")\n    phase2_answers = {key:'),
        ('{phase1_rules}\n[/PHASE1_VISIBLE_FACT_RULES]', '{phase1_rules}\n'+P1_RULE+'\n[/PHASE1_VISIBLE_FACT_RULES]'),
        ('RS1/RS2/RS4/RS5/RS_HIGHWAY/GROUP/DETAIL', 'RS1/RS2/RS4/RS5'),
        ('_rs2_rs4_trained"', '_direct8_visual_20261029"'),
    ]: source=replace_once(source,old,new)
    ast.parse(source)
    return source


P2_UE1_RULE = '''
UE1 evidence check (2026-10-30):
A brake cue must be a visible change in the lead vehicle: brake lamps that turn on or brighten, a gap that visibly closes, or a lead that visibly stops ahead while ego is still rolling up or has just come to rest behind it. Steady red tail lamps at night, wet-road reflections, headlight glare, a lead that is pulling away, accelerating, or starting from a stop, and an ordinary red-light or traffic queue in which both vehicles have been stationary throughout the history are NO. Judge the newest moment: once the slowdown is over and following has become steady, return NO.'''

P2_UE6_RULE = '''
UE6 violator check (2026-10-30):
Answer YES only when a specific other vehicle is visible in the frames entering, crossing, or sweeping through ego's junction path, or visibly approaching that path at speed from the side, AND an independent cue shows the conflict is against the rules: ego's own signal is visibly green, that vehicle runs a visibly red signal, or lane/priority markings give ego the right of way. Ego being stopped or waiting is not by itself evidence of priority: normal yielding, waiting at a red signal, letting a vehicle that has the right of way pass, and a queue at the junction are NO. Ambulances, fire engines, and police cars are ordinary road users here; flashing lights neither grant them priority nor prove a violation, so judge them by the same signal/priority cue, and one passing legitimately while ego waits at red is NO. If no such vehicle is visible anywhere in the history, answer NO even at a junction where a violation might be expected, and return NO once the vehicle has fully passed and ego's wait is no longer caused by it.'''

P2_UE3_SIDE_RULE = '''
Entry may come from the left or the right lane: on a multi-lane highway judge the vehicle in the right-hand lane exactly like one in the left-hand lane, using its changed heading toward ego's lane and the shrinking gap to the divider.'''


def phase2_prompt_v7(source):
    """Phase2 evidence revision: v6 UE3 rule plus bounded UE1/UE6 evidence clarifications."""
    source=phase2_prompt(source)
    source=replace_once(source,'sft_new_loop_phase2_direct_event_visual_v6_temporal_20261029',
                        'sft_new_loop_phase2_direct_event_visual_v7_evidence_20261030')
    anchor='\nDOMAIN_DESCRIPTIONS = {'
    source=replace_once(source,anchor,
        '\nEVENT_DEFINITIONS["UE3"] += '+repr(P2_UE3_SIDE_RULE)+
        '\nEVENT_DEFINITIONS["UE1"] += '+repr(P2_UE1_RULE)+
        '\nEVENT_DEFINITIONS["UE6"] += '+repr(P2_UE6_RULE)+'\n'+anchor)
    ast.parse(source)
    return source


def phase2_prompt(source):
    source=replace_once(source,'sft_new_loop_phase2_direct_event_visual_v5_highway_ue3',
                        'sft_new_loop_phase2_direct_event_visual_v6_temporal_20261029')
    # Only UE3 receives an extra boundary clarification, preserving UE1/5/6 and
    # the question-domain, INVALID, answer, parser and sampling contracts.
    source=replace_once(source,'EVENT_DEFINITIONS = {','EVENT_DEFINITIONS = {')
    anchor='\nDOMAIN_DESCRIPTIONS = {'
    source=replace_once(source,anchor,'\nEVENT_DEFINITIONS["UE3"] += '+repr(P2_RULE)+'\n'+anchor)
    ast.parse(source)
    return source


def derive(index, data_root, output, reviews=REVIEW):
    """Explicit train-only records; retain controls and uncertain frames too."""
    index,output,reviews=map(Path,(index,output,reviews))
    obj,decisions=cal.load_reviews(reviews)
    if not decisions or any(d['expected']['split']!='train' for d in decisions.values()):
        raise ValueError('Only reviewed train frames are admissible')
    selected=[];seen=set();holdout={s:hashlib.sha256() for s in ('val','test')}
    before=cal.sha(index);review_sha=cal.sha(reviews)
    for raw in index.open('rb'):
        row=json.loads(raw);k=cal.key(row)
        if row['split'] in holdout:holdout[row['split']].update(raw)
        if k in decisions:
            if k in seen:raise ValueError('Duplicate reviewed frame')
            seen.add(k);d=decisions[k]
            if d['decision']=='correct':d=data.local_review(row,d,data_root)
            else:cal.apply(row,d,data_root)
            selected.append(d)
    if not any(d['decision']=='correct' or d.get('machine_reconciliation') for d in selected):
        raise ValueError('No reviewed correction on this machine; cannot claim revised labels')
    if cal.sha(index)!=before or cal.sha(reviews)!=review_sha:raise ValueError('Review input changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    selected_path=output.parent/'selected_train_reviews.json'
    with selected_path.open('x') as f:json.dump({**obj,'decisions':selected},f,ensure_ascii=False,indent=2)
    result=cal.derive(index,selected_path,data_root,output)
    after={s:hashlib.sha256() for s in holdout}
    for raw in (output/'frame_index.jsonl').open('rb'):
        row=json.loads(raw)
        if row['split'] in after:after[row['split']].update(raw)
    heldout={s:h.hexdigest() for s,h in holdout.items()}
    if heldout!={s:h.hexdigest() for s,h in after.items()}:raise ValueError('Holdout changed')
    receipt=dict(revision=REVISION,status='cpu_derived_model_pending',
        requested=len(decisions),matched=len(selected),absent=[list(k) for k in sorted(set(decisions)-seen)],
        review_sha256=review_sha,selected_review_sha256=cal.sha(selected_path),
        parent_training_revision=json.loads((index.parent/'training_revision.json').read_text()),
        preservation=result,heldouts_byte_identical=True,heldout_sha256=heldout,
        sampler_algorithm_changed=False,question_protocol='all_eight_direct',model_inference=False)
    for name in ('training_revision.json','preservation.json'):
        (output/name).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    manifest=json.loads((output/'manifest.json').read_text());manifest['fresh_training_revision']=receipt
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(derive(a.parent,a.data_root,a.output),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
