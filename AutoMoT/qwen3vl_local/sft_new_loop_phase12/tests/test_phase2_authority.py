import json
import pytest
from qwen3vl_local.sft_new_loop_phase12 import data,audit
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from test_missing_labels import fixture
from test_data_delivery import evidence


@pytest.mark.parametrize('rs',['R1','R2','R3','R4','R5'])
def test_phase2_rs_is_only_structure_authority_and_visual_facts_remain_independent(rs):
    facts={k:True for k in data.VISIBLE_KEYS}
    result=data.phase1_targets(facts,rs)
    assert {k:result[k] for k in facts}==facts
    assert {k:result[k] for k in data.RS_KEYS}=={k:rs==v for k,v in data.RS_KEYS.items()}
    assert len(result)==8


@pytest.mark.parametrize('missing',[False,True])
def test_phase1_rs_event_disagreement_is_logged_not_an_abort(tmp_path,monkeypatch,missing):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    for row in rows:row.update(true_rs='R5',event='R-E5',event_codes=['R-E5'])
    p2.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if not missing:
        with p1.open('a') as f:f.write(json.dumps(base)+'\n')
    before=p1.read_bytes(),p2.read_bytes();out=tmp_path/'joined'
    report=data.build(p1,p2,rgb,out)
    joined=[json.loads(l) for l in (out/'frame_index.jsonl').open()]
    for original,row in zip(rows,joined):
        assert row['phase1_answers']==data.phase1_targets(base['answers'],'R5')
        for key in ('true_rs','event','event_codes','question_domain','target_event_class','answers'):
            assert row[key]==original[key]
    assert (p1.read_bytes(),p2.read_bytes())==before
    assert report['phase1_rs_aligned_frames']==1
    alignment=proof.read(out/'label_authority.json')['aligned_physical_frames'][0]
    assert alignment['phase1_source_rs']=='R4' and alignment['phase2_authoritative_rs']=='R5'


@pytest.mark.parametrize('bad',['unknown_rs','visual_missing','visual_string'])
def test_phase2_authority_does_not_invent_missing_visual_facts(bad):
    facts={k:False for k in data.VISIBLE_KEYS};rs='R1'
    if bad=='unknown_rs':rs='UNKNOWN'
    elif bad=='visual_missing':facts.pop('VULNERABLE')
    else:facts['VULNERABLE']='NO'
    with pytest.raises(ValueError):data.phase1_targets(facts,rs)


def test_bundle_rejects_rs_labels_not_derived_from_phase2(tmp_path):
    evidence(tmp_path);p=tmp_path/'cpu/fusion_labels.json';rows=proof.read(p)
    rows[0]['phase2_rs']='R4';proof.write(p,rows)
    with pytest.raises(ValueError,match='authoritative Phase2'):audit.verify_evaluations(tmp_path,'4rgb')


def test_bundle_rejects_shared_guard_with_old_rs(tmp_path):
    evidence(tmp_path);p=tmp_path/'cpu/guard_members.json';rows=proof.read(p)
    rows[0]['answers']['RS1']=False;rows[0]['answers']['RS4']=True;proof.write(p,rows)
    with pytest.raises(ValueError,match='guard labels'):audit.verify_evaluations(tmp_path,'4rgb')


def test_shared_guard_uses_phase2_rs_without_changing_parent_validation(tmp_path):
    from test_guard import test_guard_excludes_test_repetitions_and_fusion_training
    from qwen3vl_local.sft_new_loop_phase12 import validation
    test_guard_excludes_test_repetitions_and_fusion_training(tmp_path)
    path=tmp_path/'phase1/data/frame_index.jsonl';before=path.read_bytes()
    row=json.loads(path.read_text().splitlines()[0]);answers=data.phase1_targets(row['answers'],'R4')
    with (tmp_path/'data/frame_index.jsonl').open('a') as f:
        f.write(json.dumps(dict(row,true_rs='R4',phase1_answers=answers))+'\n')
    report=validation.prepare(tmp_path,tmp_path/'rgb')
    member=proof.read(tmp_path/'cpu/guard_members.json')[0]
    assert member['answers']==answers and member['rs_source']=='phase2'
    assert report['rs_aligned_records']==1 and path.read_bytes()==before
