from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import event_scope as e,automatic_context as auto,review_queue as q
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g,privileged_producer as p
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import snapshot,actor
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest


def frames(actors):
    result=[snapshot(f,a) for f,a in zip((8,9,10),actors)]
    for f in result:f['meta'].update(route=[[3.,0.],[10.,0.],[13.,0.],[18.,-8.],[24.,-13.],[30.,-18.]],changed_route=False)
    return result


def test_accelerating_lead_release_is_not_stable_following():
    fs=frames([[actor(position=[x,0.,0.],ego_velocity=[v,0.])] for x,v in ((8,0),(8.2,3),(8.6,6))])
    assert g.following(fs)[0] is False
    assert e.lead_releasing(fs,2) is True
    assert e.release_suggestion(fs,'U-E1',2)['value'] is True
    fs[-1]['actors'][1]['position'][0]=6
    assert e.lead_releasing(fs,2) is not True


def test_later_junction_actors_do_not_block_released_pedestrian_but_local_ones_do():
    fs=frames([[actor(**{'class':'walker'},position=[10.,y,0.],extent=[.3,.3,.9]),actor(3,position=[25.,-13.,0.])] for y in (-2.1,-2.7,-3.3)])
    result=e.release_suggestion(fs,'U-E4',2)
    assert result['value'] is True and 3 in result['outside_scope_actor_ids']
    for f in fs:
        f['actors'].append(actor(4,**{'class':'walker'},position=[9.,0.,0.]));f['image_evidence']['4']={'quality_pass':True}
    result=e.release_suggestion(fs,'U-E4',2)
    assert result['value'] is False and result['blocking_actor_ids']==[4]


@pytest.mark.parametrize('case',['missing','dark','jump','turn'])
def test_no_release_from_missing_dark_jumping_actor_or_ego_turn(case):
    fs=frames([[actor(**{'class':'walker'},position=[10.,y,0.],extent=[.3,.3,.9])] for y in (-2.1,-2.7,-3.3)])
    if case=='missing':fs[-1]['actors']=fs[-1]['actors'][:1]
    if case=='dark':fs[-1]['image_evidence']['2']['quality_pass']=False
    if case=='jump':fs[-1]['actors'][1]['position'][0]=70
    if case=='turn':fs[-1]['meta']['ego_matrix']=[[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    assert e.release_suggestion(fs,'U-E4',2)['value'] is not True


def test_near_junction_or_red_light_alone_does_not_establish_events():
    f=snapshot(10);f['meta'].update(is_junction=True,distance_to_next_junction=0)
    assert auto.junction_candidates(f)==[]
    f['actors'].append(actor(3,**{'class':'traffic_light'},affects_ego=True,state='Red'))
    assert auto.junction_candidates(f)==[]
    f['actors'][-1]['state']='Off'
    assert [v['event'] for v in auto.junction_candidates(f)]==['U-E7']


def test_passed_actor_candidate_is_not_carried_forever():
    fs=frames([[actor(position=[-8.,0.,0.])] for _ in range(3)])
    assert not p.candidates(fs,{('U-E5',2):1})


def approved_fixture():
    queue=json.loads((ROOT/'twenty_fifth_review_queue_20261002.json').read_text())
    review=json.loads((ROOT/'twenty_fifth_review_decisions_20261002.json').read_text())
    return queue,review


def test_review_path_produces_real_new_scope_labels_with_old_annotations_intact():
    queue,review=approved_fixture();new=q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')
    old=json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text());all_=json.loads((ROOT/'reviewed_state_pairs_v9.json').read_text())
    assert len(new)==14 and all_==old+new
    assert sum(a['edge']=='enter' for a in new)==2 and sum(a['edge']=='depart' for a in new)==2
    assert all(a['visible_scope_review'] for a in new if a['edge'] in ('enter','depart'))


@pytest.mark.parametrize('mutation',['blank','pending','history','hash','future','identity','facts','risk','scope'])
def test_unreviewed_or_tampered_evidence_cannot_turn_into_supervision(mutation):
    queue,review=approved_fixture();d=next(d for d in review['decisions'] if next(c for c in queue['cards'] if c['id']==d['card_id'])['edge']=='depart')
    review['decisions']=[d]
    if mutation=='pending':
        d['status']='pending';assert q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')==[];return
    if mutation=='blank':d.pop('facts')
    if mutation=='history':d['rgb_reviews'].pop(next(iter(d['rgb_reviews'])))
    if mutation=='hash':d['rgb_reviews'][next(iter(d['rgb_reviews']))]['rgb_sha256']='0'*64
    if mutation=='future':d['rgb_reviews'][next(iter(d['rgb_reviews']))]['observed_until']=999
    if mutation=='identity':d['event_identity_reviewed']=False
    if mutation=='facts':d['target']='YES'
    if mutation=='risk':d['risk_ids_reviewed']=[]
    if mutation=='scope':d['visible_maneuver_scope_reviewed']=False
    with pytest.raises(ValueError):q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')


def test_review_queue_budget_is_hard_cap_and_does_not_label_unselected(monkeypatch):
    monkeypatch.setattr(q,'card',lambda root,s,r,f,ep,edge:dict(scenario=s,route_id=r,frame_id=f,episode=ep,edge=edge))
    def rows():
        for f in range(10,1000):
            yield dict(scenario='s',route_id='r',frame_id=f,split='train',proposals=[dict(instance=dict(event='U-E1',instance_id='one'),
                edge='proceed',episode=dict(event='U-E1'),proposal=dict(release_suggestion=dict(value=bool(f%2))))])
    queue=q.select(rows(),None,per_event=3)
    assert queue['statistics']['raw_questions']==990 and len(queue['cards'])==2  # one route per answer stratum
    assert queue['supervision_approved'] is False and all('target' not in c for c in queue['cards'])


def test_blank_review_template_never_preapproves_a_label():
    queue,_=approved_fixture();review=q.blank_reviews(queue)
    assert all(d['status']=='pending' and d['target'] is None for d in review['decisions'])
    review['reviewer']='tester'
    assert q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')==[]
