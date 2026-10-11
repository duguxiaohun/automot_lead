from copy import deepcopy
import json

import pytest

from qwen3vl_local.audit_joint import junction_review as tool


def question(ident='a',junction=1,event='U-E7'):
    return dict(scenario='Scene',route_id='route',frame_id=10,split='train',physical_group='Scene/route',
                rgb_mode=4,episode=dict(instance_id=ident,event=event),
                instance=dict(actor_id=ident,actor_ids=[],junction_id=junction,junction_world=[0.,0.,0.]),
                rule_target='YES')


def frame(qs):
    return dict(scenario='Scene',route_id='route',frame_id=10,split='train',physical_group='Scene/route',questions=qs)


@pytest.mark.parametrize('value',[-1,None,True,'1',1.5])
def test_missing_or_sentinel_junction_cannot_establish_identity(value):
    r=tool.pair_evidence(question(junction=value),question('b',junction=value))
    assert r['declared_junction_relation']=='unknown'
    assert r['merge_authorized'] is False


def test_same_junction_and_coincident_reference_do_not_merge_anchors():
    a,b=question(junction=0),question('b',junction=0)
    b['instance']['actor_ids']=[42]
    r=tool.pair_evidence(a,b)
    assert r['declared_junction_relation']=='same' and r['frozen_reference_distance_m']==0
    assert r['participant_sets']==[[],[42]] and r['merge_authorized'] is False
    b['instance']['junction_world'][0]=float('nan')
    assert tool.pair_evidence(a,b)['frozen_reference_distance_m'] is None


def test_concurrency_is_per_recording_frame_and_event_not_junction_id():
    a,b=question(),question('b');other=question('c',event='R-E5')
    assert len(list(tool.concurrent([frame([a,b,other])])) )==1
    second=frame([b]);second['frame_id']=11;second['questions'][0]['frame_id']=11
    assert list(tool.concurrent([frame([a]),second]))==[]


@pytest.mark.parametrize('damage',['duplicate_frame','foreign_route','conflicting_seed'])
def test_inconsistent_replay_rejected(damage):
    a,b=question(),question('b');r=frame([a,b]);rs=[r]
    if damage=='duplicate_frame':rs.append(deepcopy(r))
    elif damage=='foreign_route':a['route_id']='other'
    else:
        another=deepcopy(a);another['instance']['actor_id']='other';r['questions'].append(another)
    with pytest.raises(ValueError):list(tool.concurrent(rs))


@pytest.mark.parametrize('answers,status',[
    (['YES','YES'],'same_input_same_target'),
    (['UNKNOWN','UNKNOWN'],'same_input_same_target'),
    (['YES','UNKNOWN'],'same_input_abstention_disagreement'),
    (['NO','UNKNOWN'],'same_input_abstention_disagreement'),
    (['YES','NO','UNKNOWN'],'same_input_binary_conflict')])
def test_abstention_disagreement_is_not_binary_contradiction(answers,status):
    assert tool.input_disagreement([dict(rule_target=a) for a in answers])==status


@pytest.mark.parametrize('damage',['replay','request','teacher','membership'])
def test_audit_rejects_unbound_inputs_before_publishing(tmp_path,monkeypatch,damage):
    replay=tmp_path/'replay.jsonl';replay.write_text('')
    request=tmp_path/'request.json';request.write_text(json.dumps({'routes':[]}))
    receipt=tmp_path/'receipt.json';meta=dict(sha256=tool.file_sha(replay),request_sha256=tool.file_sha(request),teacher={},frames=0)
    monkeypatch.setattr(tool.teacher_rules,'identity',lambda:{})
    if damage=='replay':meta['sha256']='f'*64
    elif damage=='request':meta['request_sha256']='f'*64
    elif damage=='teacher':meta['teacher']={'wrong':True}
    else:meta['frames']=1
    receipt.write_text(json.dumps(meta));out=tmp_path/'out'
    with pytest.raises(ValueError):tool.audit(replay,receipt,request,tmp_path/'raw',out)
    assert not out.exists()
