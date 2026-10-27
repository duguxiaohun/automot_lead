import copy
import hashlib
import json
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint.phase12_retrain import phase2_data as d
from qwen3vl_local.sft_new_loop_phase2.build_dataset import _make_row


@pytest.fixture
def case(tmp_path):
    review=json.loads(d.REVIEW.read_text()); rgb=tmp_path/'rgb'; rgb.mkdir()
    for item in review['frames']:
        p=rgb/item[0];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(item[0].encode())
        item[1]=hashlib.sha256(p.read_bytes()).hexdigest()
    reviewpath=tmp_path/'review.json';reviewpath.write_text(json.dumps(review))
    corrections={d.key(r) for r in review['reviewed_candidate_corrections']}
    keys=sorted(corrections|{d.key(r) for r in review['controls']})
    bases=[]; rows=[]
    for scenario,route,frame in keys:
        rs='R4' if (scenario,route,frame) in corrections else 'R3'
        event='R-E4' if rs=='R4' else 'R-E1'
        paths=[f'{scenario}/{route}/rgb/{f:04d}.jpg' for f in range(frame-3,frame+1)]
        base=dict(scenario=scenario,route_id=route,frame_id=frame,town='Town13',split='train',rs=rs,
            primary_event=event,event_codes=[event],ue3_subtype='HIGHWAY_CUTIN',
            history_rgb_paths=paths,latest_rgb_path=paths[-1],visual_label_risk=False,visual_label_risk_reasons=[])
        bases.append(base);rows.append(_make_row(base=base,question_domain='ROAD_CORRIDOR',target_class='UE3',invalid=False))
    controls=[b for b in bases if b['rs']=='R3']
    for i in range(111):
        rows.append(_make_row(base=controls[i%5],question_domain='LOCAL_JUNCTION',target_class='INVALID',invalid=True,invalid_source=d.SIGNATURE))
    for split in ('train','val','test'):
        rows.append(dict(rows[0],scenario='Unreviewed',split=split))
    parent=tmp_path/'parent';parent.mkdir();index=parent/'frame_index.jsonl'
    index.write_bytes(b''.join((json.dumps(r)+'  \n').encode() for r in rows))
    (parent/'manifest.json').write_text('{"sampling":{"parent_only":true}}')
    return tmp_path,rgb,reviewpath,index,rows


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_preservation_native_invalid_regeneration_and_idempotence(case,layout):
    root,rgb,review,index,rows=case
    link=root/layout;link.symlink_to(rgb,target_is_directory=True)
    before=index.read_bytes();out=root/'new/data'
    receipt=d.derive(index,link,out,review)
    after=(out/'frame_index.jsonl').read_bytes().splitlines(keepends=True)
    old=before.splitlines(keepends=True)
    assert len(receipt['corrected_train_positives'])==23
    assert receipt['already_correct_train_positives']==5 and receipt['regenerated_invalid_slots']==111
    assert receipt['heldouts_byte_identical'] and index.read_bytes()==before
    assert after[-3:]==old[-3:]
    for i,r in enumerate(rows[:28]):
        if r['true_rs']=='R3':assert after[i]==old[i]
        else:
            changed=json.loads(after[i]);expected={**r,'true_rs':'R3','event':'R-E1','event_codes':['R-E1']}
            assert changed==expected
    invalid=[json.loads(x) for x in after[28:139]]
    assert len({d.key(r) for r in invalid})==28
    assert all(r['answers']==rows[28]['answers'] and r['invalid_source']==d.SIGNATURE for r in invalid)
    other=root/'again/data';second=d.derive(out/'frame_index.jsonl',link,other,review)
    assert second['changed_rows']==0 and second['regenerated_invalid_slots']==0
    assert (other/'frame_index.jsonl').read_bytes()==(out/'frame_index.jsonl').read_bytes()
    with pytest.raises(ValueError):d.derive(index,link,out,review)


@pytest.mark.parametrize('bad',['rgb','rs','control','event','answer','history','holdout','missing','invalid_history','invalid_source'])
def test_reject_drift_before_new_dataset(case,bad):
    root,rgb,review,index,rows=case
    r=next(r for r in rows[:28] if r['true_rs']=='R4')
    if bad=='rgb':(rgb/r['history_rgb_paths'][0]).write_bytes(b'changed')
    elif bad=='rs':r['true_rs']='R5'
    elif bad=='control':next(x for x in rows[:28] if x['true_rs']=='R3')['true_rs']='R4'
    elif bad=='event':r['event']='U-E3'
    elif bad=='answer':r['answers']['UE1']=True
    elif bad=='history':r['history_rgb_paths'].reverse()
    elif bad=='holdout':r['split']='val'
    elif bad=='missing':rows.remove(r)
    elif bad=='invalid_history':rows[28]['history_rgb_paths']=list(reversed(rows[28]['history_rgb_paths']))
    else:rows[28]['invalid_source']='unexpected'
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):d.derive(index,rgb,root/'rejected/data',review)
    assert not (root/'rejected/data').exists()


def test_already_correct_positives_with_stale_invalid_pool_are_repaired(case):
    root,rgb,review,index,rows=case
    for r in rows[:28]:r.update(true_rs='R3',event='R-E1',event_codes=['R-E1'])
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receipt=d.derive(index,rgb,root/'revised/data',review)
    assert receipt['corrected_train_positives']==[]
    assert receipt['regenerated_invalid_slots']==111
