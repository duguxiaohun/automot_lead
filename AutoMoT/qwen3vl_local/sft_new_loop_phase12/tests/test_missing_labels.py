import copy
import json
from pathlib import Path
from PIL import Image
import pytest
from qwen3vl_local.sft_new_loop_phase12 import data,missing_labels,prompts
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof


def fixture(tmp_path,monkeypatch):
    rgb=tmp_path/'rgb';paths=[f's/route/rgb/{i:04d}.jpg' for i in range(4)]
    for i,name in enumerate(paths):
        p=rgb/name;p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(8,8),(20*i,10,30)).save(p)
    facts={k:k=='RS4' for k in prompts.P1.ANSWER_KEYS}
    base=dict(scenario='s',route_id='route',frame_id=3,split='train',rs='R4',event='R-E4',
        answers=facts,history_rgb_paths=paths,latest_rgb_path=paths[-1])
    controls=[dict(base,frame_id=i) for i in (2,4)]
    rows=[dict(base,true_rs='R4',question_domain=domain,target_event_class=target,
               answers={'INVALID_EVENT_CONTEXT':target=='INVALID'})
          for domain,target in [('LOCAL_JUNCTION','RE'),('ROAD_CORRIDOR','INVALID')]]
    review=tmp_path/'review.json'
    proof.write(review,dict(schema='fusion_missing_phase1_review_v1',records=[
        dict(key=['s','route',3],split='train',rs='R4',event='R-E4',answers=facts,
             history_rgb_paths=paths,rgb_sha256={p:proof.sha(rgb/p) for p in paths},
             controls=[dict(key=['s','route',r['frame_id']],rs='R4',answers=facts) for r in controls])]))
    monkeypatch.setattr(missing_labels,'REVIEW',review)
    p1=tmp_path/'p1.jsonl';p2=tmp_path/'p2.jsonl'
    p1.write_text(''.join(json.dumps(r)+'\n' for r in controls))
    p2.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return p1,p2,rgb,base,controls,rows


def test_single_recovery_keeps_every_native_phase2_slot_and_parents(tmp_path,monkeypatch):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    original=p1.read_bytes(),p2.read_bytes();out=tmp_path/'joined'
    result=data.build(p1,p2,rgb,out)
    joined=[json.loads(l) for l in (out/'frame_index.jsonl').open()]
    assert len(joined)==2 and [r['answers'] for r in joined]==[r['answers'] for r in rows]
    assert all(r['phase1_answers']==base['answers'] for r in joined)
    assert (p1.read_bytes(),p2.read_bytes())==original
    assert result['missing_phase1_annotations_recovered']==1
    receipt=proof.read(out/'missing_label_recovery.json')
    assert receipt['recovered'][0]['phase2_slots']==2
    assert receipt['review_sha256']==proof.sha(missing_labels.REVIEW)
    assert result['missing_label_recovery_sha256']==proof.sha(out/'missing_label_recovery.json')


def test_existing_correct_or_different_annotations_are_never_overwritten(tmp_path,monkeypatch):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    base['answers']['VULNERABLE']=True
    with p1.open('a') as f:f.write(json.dumps(base)+'\n')
    original=p1.read_bytes();out=tmp_path/'joined';r=data.build(p1,p2,rgb,out)
    assert r['missing_phase1_annotations_recovered']==0
    assert not (out/'missing_label_recovery.json').exists()
    assert all(json.loads(l)['phase1_answers']['VULNERABLE'] for l in (out/'frame_index.jsonl').open())
    assert p1.read_bytes()==original


@pytest.mark.parametrize('bad',['rgb','missing_rgb','rs','history','latest','val','test',
                               'historical','control_split','unreviewed'])
def test_unsafe_recovery_fails_before_a_training_index_is_written(tmp_path,monkeypatch,bad):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch);historical=[]
    if bad=='rgb':(rgb/base['history_rgb_paths'][0]).write_bytes(b'wrong')
    elif bad=='missing_rgb':(rgb/base['history_rgb_paths'][0]).unlink()
    elif bad=='rs':rows[0]['true_rs']='R1'
    elif bad=='history':rows[0]['history_rgb_paths']=list(reversed(base['history_rgb_paths']))
    elif bad=='latest':rows[0]['latest_rgb_path']=base['history_rgb_paths'][0]
    elif bad in ('val','test'):rows[0]['split']=bad
    elif bad=='historical':historical=[('s','route')]
    elif bad=='control_split':controls[0]['split']='val'
    elif bad=='unreviewed':rows[0]['frame_id']=10
    p1.write_text(''.join(json.dumps(r)+'\n' for r in controls))
    p2.write_text(''.join(json.dumps(r)+'\n' for r in rows));out=tmp_path/'joined'
    with pytest.raises(ValueError):data.build(p1,p2,rgb,out,historical_groups=historical)
    assert not (out/'frame_index.jsonl').exists()


@pytest.mark.parametrize('condition',['missing','different_labels'])
def test_adjacent_labels_are_preserved_diagnostics_not_target_supervision(tmp_path,monkeypatch,condition):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    controls=copy.deepcopy(controls)
    if condition=='missing':controls=[]
    else:
        controls[0]['answers']['VULNERABLE']=True
        # Also train the neighbor: its own positive fact must survive the join.
        rows.append(dict(rows[0],frame_id=2))
    p1.write_text(''.join(json.dumps(r)+'\n' for r in controls))
    p2.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original=p1.read_bytes(),p2.read_bytes();out=tmp_path/'joined'
    data.build(p1,p2,rgb,out)
    joined=[json.loads(l) for l in (out/'frame_index.jsonl').open()]
    assert all(r['phase1_answers']==base['answers'] for r in joined if r['frame_id']==3)
    if condition=='different_labels':assert joined[-1]['phase1_answers']['VULNERABLE'] is True
    assert (p1.read_bytes(),p2.read_bytes())==original
    receipt=proof.read(out/'missing_label_recovery.json')
    entry=receipt['recovered'][0]
    assert receipt['schema']=='fusion_missing_phase1_recovery_v2_target_rgb'
    assert entry['verified_rgb_sha256']==entry['review']['rgb_sha256']
    control=entry['adjacent_controls'][0]
    assert control['status']==condition and control['modified'] is False
    assert control['used_to_generate_target'] is False
    if condition=='different_labels':assert control['differing_keys']==['VULNERABLE']


@pytest.mark.parametrize('frame',range(4))
def test_each_target_rgb_remains_required_when_adjacent_labels_are_missing(tmp_path,monkeypatch,frame):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    p1.write_text('')
    (rgb/base['history_rgb_paths'][frame]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='RGB differs from local audit'):
        data.build(p1,p2,rgb,tmp_path/'joined')
    assert not (tmp_path/'joined/frame_index.jsonl').exists()


def test_repaired_join_matches_complete_parent_join_exactly(tmp_path,monkeypatch):
    p1,p2,rgb,base,controls,rows=fixture(tmp_path,monkeypatch)
    data.build(p1,p2,rgb,tmp_path/'repaired')
    with p1.open('a') as f:f.write(json.dumps(base)+'\n')
    data.build(p1,p2,rgb,tmp_path/'complete')
    assert (tmp_path/'repaired/frame_index.jsonl').read_bytes()==(tmp_path/'complete/frame_index.jsonl').read_bytes()
