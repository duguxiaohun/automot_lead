"""Known visual risks require explicit review of every causal input frame, not inferred labels."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from .data_paths import data_path
from .identity import ROOT,digest,file_sha,write_json

POLICY='known_visual_risk_admission_v1'
LEDGERS=('tenth_audit_exposure_20260930.json','eleventh_audit_exposure_20260930.json',
         'twelfth_thirteenth_audit_exposure_20260930.json','fourteenth_audit_exposure_20260930.json','fifteenth_audit_exposure_20260930.json',
         'sixteenth_audit_exposure_20260930.json','eighteenth_nineteenth_audit_exposure_20261001.json','seventeenth_audit_exposure_20261001.json',
         'twentieth_audit_exposure_20261001.json','twenty_first_audit_exposure_20261001.json','twenty_second_audit_exposure_20261002.json','twenty_third_audit_exposure_20261002.json','thirty_first_audit_exposure_20261005.json','thirty_second_audit_exposure_20261005.json','thirty_third_audit_exposure_20261006.json','thirty_fifth_audit_exposure_20261006.json','thirty_fourth_audit_exposure_20261006.json','thirty_sixth_audit_exposure_20261006.json')
CHECKS=('participant_identity_reviewed','transition_scope_reviewed','instance_boundary_reviewed')


def registry():
    result=defaultdict(list)
    for name in LEDGERS:
        for risk in json.loads((ROOT/name).read_text())['calibration_risks']:
            result[(risk['scenario'],risk['route_id'])].append(dict(
                risk_id=digest(dict(ledger=name,record=risk)),ledger=name,record=risk))
    return dict(result)


def registry_identity(known):
    return digest(sorted((s,r,[v['risk_id'] for v in values]) for (s,r),values in known.items()))


def required_frames(annotation,rgb_mode):
    band=annotation.get('transition_band')
    start,end=(band['review_start'],band['review_end']) if band else (annotation['start'],annotation['end'])
    offsets=(-4,0) if rgb_mode==2 else (-6,-4,-2,0)
    anchors=[f for f in range(start,end+1) if f+offsets[0]>=4 and (not band or f<band['stop']['frame'])]
    return sorted({f+offset for f in anchors for offset in offsets})


def validate_review(review,risks,frames,hashes):
    if not isinstance(review,dict) or review.get('policy')!=POLICY:
        raise ValueError('known visual risk requires explicit per-frame risk_review')
    ids=review.get('risk_ids')
    if not isinstance(ids,list) or sorted(ids)!=sorted(r['risk_id'] for r in risks):
        raise ValueError('risk review does not cover current registered evidence')
    if any(not isinstance(review.get(k),str) or not review[k].strip() for k in ('reviewer','evidence_id')):
        raise ValueError('risk review lacks reviewer/evidence')
    items=review.get('frames')
    if not isinstance(items,dict):raise ValueError('risk review lacks frame records')
    states=[]
    for f,sha in zip(frames,hashes,strict=True):
        item=items.get(str(f))
        if (not isinstance(item,dict) or item.get('status') not in ('usable','uncertain','excluded')
                or not isinstance(item.get('reason'),str) or not item['reason'].strip()
                or type(item.get('observed_until')) is not int or item['observed_until']!=f
                or item.get('rgb_sha256')!=sha):
            raise ValueError(f'incomplete/noncausal risk evidence or RGB mismatch at frame {f}')
        if item['status']=='usable' and any(item.get(k) is not True for k in CHECKS):
            raise ValueError('usable risk frame requires identity, transition scope and instance boundary review')
        states.append(item['status'])
    return 'excluded' if 'excluded' in states else 'uncertain' if 'uncertain' in states else 'usable'


def risk_window(record):
    """An explicit confirmed onset never vetoes an earlier causal observation.

    Historical ledgers retain their original full-envelope policy. New onset
    records require both a before and after RGB source; no guessed onset.
    """
    evidence=record.get('rgb_evidence',[])
    frames=[e['frame'] for e in evidence if type(e.get('frame')) is int]
    onset=record.get('first_affected_frame')
    if onset is not None:
        last_good=record.get('last_unaffected_frame')
        if (record.get('temporal_scope')!='confirmed_discontinuity_onset_v1'
                or type(onset) is not int or type(last_good) is not int
                or onset!=last_good+1 or last_good not in frames or onset not in frames
                or record.get('reported_status')!='original_RGB_confirmed_visible_discontinuity'
                or any(not isinstance(e.get('sha256'),str) or len(e['sha256'])!=64 for e in evidence)):
            raise ValueError('invalid confirmed risk onset evidence')
        return onset,max(frames)
    if record.get('temporal_scope')=='confirmed_discontinuity_onset_v1':
        raise ValueError('confirmed risk scope lacks onset')
    return (min(frames),max(frames)) if frames else None


def teacher_window_check(question,risks):
    """Evidence envelope is a veto window, not proof that other frames are safe.

    All routes use the same automatic geometry/visibility/anomaly checks. Known
    evidence adds local vetoes; records without frame evidence stay unresolved.
    Include the entire teacher history and older STOP proof, not just RGB slots.
    """
    observed={s['frame_id'] for s in question['causal_sources']}
    observed.update(s['frame_id'] for s in question.get('instance_sources',[]))
    observed.update(s['frame_id'] for r in question.get('control_evidence',{}).values() for o in r['observations'] for s in o['sources'])
    affected=[]
    for risk in risks:
        window=risk_window(risk['record'])
        if window is None or observed and min(observed)<=window[1] and max(observed)>=window[0]:affected.append(risk['risk_id'])
    return dict(policy='teacher_known_risk_evidence_windows_v1',affected_risk_ids=affected,
                outside_registered_windows=not affected,scope='evidence envelope only; automatic checks still required on every route')


def manual_risks(risks,frames):
    """Explicit new windows cover the history envelope, including gaps between RGB.

    Legacy route-wide review commitments stay in force. Missing localization is
    unresolved for the whole route, never an exemption from review.
    """
    result=[]
    for risk in risks:
        record=risk['record'];window=risk_window(record)
        if (record.get('manual_scope')!='causal_history_window' or window is None or not frames
                or min(frames)<=window[1] and max(frames)>=window[0]):
            result.append(risk)
    return result


def reviewed_risks(risks,review,frames):
    """Retain explicit review judgments across a band, even outside its window."""
    if not isinstance(review,dict):raise ValueError('known visual risk row lacks per-frame admission')
    ids=review.get('risk_ids')
    if (not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids)
            or not {r['risk_id'] for r in manual_risks(risks,frames)}<=set(ids)
            or not set(ids)<={r['risk_id'] for r in risks}):
        raise ValueError('risk review does not cover current registered evidence')
    return [r for r in risks if r['risk_id'] in ids]


def annotation_review(annotation,data_root,rgb_mode,known):
    risks=known.get((annotation['scenario'],annotation['route_id']),[])
    if not risks:
        if annotation.get('risk_review') is not None:
            raise ValueError('risk_review supplied for route absent from registry')
        return None
    if annotation.get('label_basis') in ('approved_rule_teacher','weak_rule_teacher'):
        q=annotation['teacher_provenance']['question']
        if not teacher_window_check(q,risks)['outside_registered_windows']:raise ValueError('teacher evidence intersects known risk window')
        return None
    frames=required_frames(annotation,rgb_mode)
    review=annotation.get('risk_review')
    risks=reviewed_risks(risks,review,frames) if review is not None else manual_risks(risks,frames)
    if not risks:return None
    # Legacy interval declarations cannot resolve known visual ambiguity.
    if annotation['label_basis']!='reviewed_transition_band':
        raise ValueError('known visual risk requires reviewed_transition_band')
    run=data_path(data_root, f"{annotation['scenario']}/{annotation['route_id']}")
    hashes=[file_sha(run/'rgb'/f'{f:04d}.jpg') for f in frames]
    validate_review(review,risks,frames,hashes)
    return review


def row_review(row,review,known):
    frames=row['observation']['history_frames']
    known_risks=known.get((row['scenario'],row['route_id']),[])
    if not known_risks:
        return None
    risks=reviewed_risks(known_risks,review,frames)
    projected={k:review[k] for k in ('policy','reviewer','evidence_id','risk_ids')}
    projected['frames']={str(f):review['frames'][str(f)] for f in frames}
    status=validate_review(projected,risks,frames,row['image_sha256'])
    return dict(disposition=status,evidence=projected)


def validate_rows(rows,known):
    counts=Counter()
    for row in rows:
        risks=known.get((row['scenario'],row['route_id']),[])
        proof=row.get('risk_review')
        if not risks:
            if proof is not None:raise ValueError('unexpected risk review on unregistered route')
            continue
        if row.get('label_basis') in ('approved_rule_teacher','weak_rule_teacher'):
            if proof is not None or not teacher_window_check(row['teacher_provenance']['question'],risks)['outside_registered_windows']:raise ValueError('teacher risk window not resolved')
            counts['teacher_outside_registered_windows']+=1;continue
        applicable=manual_risks(risks,row['observation']['history_frames'])
        if not applicable and proof is None:
            counts['manual_outside_registered_windows']+=1;continue
        if row.get('label_basis')!='reviewed_transition_band' or not isinstance(proof,dict):
            raise ValueError('known visual risk row lacks per-frame admission')
        risks=reviewed_risks(risks,proof.get('evidence'),row['observation']['history_frames'])
        state=validate_review(proof.get('evidence'),risks,row['observation']['history_frames'],row['image_sha256'])
        if state!=proof.get('disposition') or (state!='usable' and row['target']!='UNKNOWN'):
            raise ValueError('unresolved/excluded risk cannot supply a supervised answer')
        counts[state]+=1
    return dict(policy=POLICY,registry_sha256=registry_identity(known),row_dispositions=dict(counts))


def templates(annotations,data_root,rgb_mode):
    known=registry();out=[]
    for index,ann in enumerate(annotations):
        risks=manual_risks(known.get((ann['scenario'],ann['route_id']),[]),required_frames(ann,rgb_mode))
        if not risks:continue
        run=data_path(data_root, f"{ann['scenario']}/{ann['route_id']}")
        frames=required_frames(ann,rgb_mode)
        review=dict(policy=POLICY,reviewer='',evidence_id='',risk_ids=[r['risk_id'] for r in risks],
                    frames={str(f):dict(status='uncertain',reason='',observed_until=f,
                        rgb_sha256=file_sha(run/'rgb'/f'{f:04d}.jpg'),**{k:False for k in CHECKS}) for f in frames})
        out.append(dict(annotation_index=index,scenario=ann['scenario'],route_id=ann['route_id'],
                        edge=ann['edge'],label_basis_required='reviewed_transition_band',
                        risks=risks,risk_review=review))
    return dict(policy=POLICY,rgb_mode=rgb_mode,review_complete=False,annotations=out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    parser.add_argument('--rgb-mode',type=int,choices=(2,4),default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('use a fresh review template path')
    write_json(args.output,templates(json.loads(args.annotations.read_text()),args.data_root,args.rgb_mode))

if __name__=='__main__':main()
