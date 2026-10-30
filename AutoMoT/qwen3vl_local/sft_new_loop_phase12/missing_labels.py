"""Explicit RGB-bound recovery of reviewed missing TRAIN annotations only."""
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity,physical_group
from .data import relative_rgb,resolve_rgb,VISIBLE_KEYS

REVIEW=Path(__file__).with_name('missing_phase1_review.json')


def reviewed():
    before=proof.sha(REVIEW);obj=proof.read(REVIEW)
    if obj.get('schema')!='fusion_missing_phase1_review_v1':raise ValueError('Wrong missing-label review')
    records={tuple(r['key']):r for r in obj['records']}
    if len(records)!=len(obj['records']):raise ValueError('Duplicate missing-label review')
    from .prompts import P1
    for k,r in records.items():
        paths=[f'{k[0]}/{k[1]}/rgb/{f:04d}.jpg' for f in range(k[2]-3,k[2]+1)]
        if (r['split']!='train' or set(r['answers'])!=set(P1.ANSWER_KEYS)
                or any(type(v) is not bool for v in r['answers'].values())
                or r['history_rgb_paths']!=paths or set(r['rgb_sha256'])!=set(paths)
                or not r['controls']):
            raise ValueError('Incomplete reviewed training anchor')
    if proof.sha(REVIEW)!=before:raise ValueError('Missing-label review changed')
    return records,before


def recover(missing,rows,witnesses,split_groups,data_root,records,review_sha):
    unreviewed=sorted(missing-set(records))
    if unreviewed:
        raise ValueError(f'Missing Phase1 physical labels: {len(missing)} frames; unreviewed={unreviewed[:20]}; no guessed facts allowed')
    recovered={};evidence=[]
    for key in sorted(missing):
        r=records[key]
        if split_groups[physical_group(key)]!={'train'}:
            raise ValueError('Missing-label recovery is train-only; heldout/historical route: '+str(key))
        copies=[row for row in rows if identity(row)==key]
        for row in copies:
            if (row['split']!='train'
                    or [relative_rgb(p) for p in row['history_rgb_paths']]!=r['history_rgb_paths']
                    or relative_rgb(row['latest_rgb_path'])!=r['history_rgb_paths'][-1]):
                raise ValueError('Reviewed missing anchor differs from Phase2 split/history: '+str(key))
        # The reviewed target has its own complete four-image byte binding.
        # Adjacent labels from another collection are diagnostic, not proof of
        # this target's visual facts. Preserve them even when they disagree.
        verified_rgb={}
        for name,expected in r['rgb_sha256'].items():
            actual=proof.sha(resolve_rgb(data_root,name))
            if actual!=expected:
                raise ValueError(f'Reviewed missing-anchor RGB differs from local audit: {name}; expected={expected}; actual={actual}')
            verified_rgb[name]=actual
        controls=[]
        for control in r['controls']:
            witness=witnesses.get(tuple(control['key']))
            expected={k:control['answers'][k] for k in VISIBLE_KEYS}
            actual={k:witness['answers'].get(k) for k in VISIBLE_KEYS} if witness else None
            changed=[k for k in VISIBLE_KEYS if actual is not None and
                     (type(actual[k]) is not bool or actual[k]!=expected[k])]
            state='missing' if witness is None else ('different_labels' if changed else 'matching')
            controls.append(dict(key=control['key'],status=state,expected_visual_facts=expected,
                actual_visual_facts=actual,actual_split=witness['split'] if witness else None,
                differing_keys=changed,used_to_generate_target=False,modified=False))
            if state!='matching':
                print(f"[fusion-data] adjacent annotation {state}: {control['key']}; "
                      f"differing_keys={changed}; preserved; target's four RGB hashes verified",flush=True)
        recovered[key]=dict(scenario=key[0],route_id=key[1],frame_id=key[2],split='train',
            rs=r['rs'],event=r['event'],answers=r['answers'],history_rgb_paths=r['history_rgb_paths'])
        evidence.append(dict(key=list(key),phase2_slots=len(copies),review=r,
            verified_rgb_sha256=verified_rgb,adjacent_controls=controls,
            restored_keys=list(VISIBLE_KEYS),rs_event_authority='phase2',
            applied_only_because_phase1_anchor_missing=True,existing_phase1_rows_changed=0))
    if proof.sha(REVIEW)!=review_sha:raise ValueError('Missing-label review changed')
    return recovered,dict(schema='fusion_missing_phase1_recovery_v2_target_rgb',review_sha256=review_sha,
        recovered=evidence,original_parent_files_unchanged=True,phase2_slots_preserved=True,
        recovery_basis='exact reviewed target identity plus all four RGB SHA; adjacent labels diagnostic only',
        scope='Pinned reviewed train anchors only; not a diagnosis of remote collection filtering')
