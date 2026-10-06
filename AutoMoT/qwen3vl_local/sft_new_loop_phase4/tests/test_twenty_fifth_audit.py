from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import review_queue as q,visual_review as v,event_scope as e
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest
from qwen3vl_local.sft_new_loop_phase4.tests.test_twenty_fourth_audit import approved_fixture,frames,actor


def evidence():
    queue,review=approved_fixture();cards={c['id']:c for c in queue['cards']}
    d=next(d for d in review['decisions'] if cards[d['card_id']]['frame_id']==73)
    review['decisions']=[d]
    return queue,review,d,cards[d['card_id']]


def test_disputed_six_labels_are_deferred_and_never_relabelled_no():
    queue,review=approved_fixture();anns=q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')
    phases=[next(iter(a['transition_band']['frames'].values()))['phase'] for a in anns]
    assert phases.count('uncertain')==6 and phases.count('catchup')==2
    old=json.loads((ROOT/'reviewed_state_pairs_v8.json').read_text())
    assert len(old)==166  # historical frozen labels are not rewritten


def test_old_checkbox_only_approval_is_not_accepted_by_current_compiler():
    queue=json.loads((ROOT/'twenty_fourth_review_queue_20261002.json').read_text())
    review=json.loads((ROOT/'twenty_fourth_review_decisions_20261002.json').read_text())
    with pytest.raises(ValueError):q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')


@pytest.mark.parametrize('change',['missing','wrong_mode','future','bad_sha','no_current','checkbox','moving_car','seam','subpixel'])
def test_actual_input_and_successor_evidence_are_required(change):
    queue,review,d,c=evidence();m=d['visual_review']['modes']['2']
    if change=='missing':d.pop('visual_review')
    if change=='wrong_mode':m['frames']=[67,69,71,73]
    if change=='future':m['current_cues'][0]['frame']=74
    if change=='bad_sha':m['sha256'][0]='0'*64
    if change=='no_current':m['current_cues']=[]
    if change=='checkbox':m['successor']=None
    if change=='moving_car':m['successor']['same_feature_reviewed']=False
    if change=='seam':m['successor']['points'][1]['xy']=[600,245]
    if change=='subpixel':m['successor']['points'][1]['xy']=[163.1,245]
    with pytest.raises(ValueError):q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')


def test_two_rgb_uncertain_four_rgb_supported_build_separately(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,read_rows
    queue,review,d,c=evidence();d['visual_review']['modes']['2'].update(disposition='uncertain',reason='two input images unresolved')
    anns=q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')
    for mode in (2,4):
        path=tmp_path/str(mode);build(anns,ROOT.parents[1]/'lead_data',path,rgb_mode=mode)
        data,_=load_dataset(path);pending=read_rows(path/'review_queue.jsonl')
        assert len(data['train'])==(mode==4)
        assert len(pending)==(mode==2)
        if pending:assert pending[0]['target']=='UNKNOWN'


def test_rendered_panels_contain_only_mode_input(tmp_path):
    from PIL import Image
    queue,_,_,c=evidence();queue['cards']=[c];out=tmp_path/'panels';q.render(queue,ROOT.parents[1]/'lead_data',out)
    for mode in (2,4):
        with Image.open(out/f"{c['id']}_{mode}rgb.jpg") as im:assert im.size==(1152,404*mode)


@pytest.mark.parametrize('y,result',[(-2.1,None),(-2.7,None),(-3.3,True),(-1.,False)])
def test_vru_one_metre_footprint_margin_boundary_is_unknown(y,result):
    fs=frames([[actor(**{'class':'walker'},position=[10.,b,0.],extent=[.3,.3,.9])] for b in (y+.5,y+.25,y)])
    assert e.release_suggestion(fs,'U-E4',2)['value'] is result


def test_no_vru_yes_from_centre_distance_without_boundary_evidence():
    queue,review,d,c=evidence();m=deepcopy(d['visual_review']['modes']['2'])
    m['vulnerable_clearance']=dict(lower_bound_m=.5,moving_away_or_separated=True,whole_footprint_reviewed=True,boundary_cue=m['current_cues'][0])
    with pytest.raises(ValueError):v.validate(m,frame=73,frames=m['frames'],hashes=m['sha256'],event='U-E4',edge='proceed',phase='readiness',target='YES')


def test_forged_target_cannot_reuse_mode_proof():
    _,_,d,c=evidence();m=d['visual_review']['modes']['2']
    row=dict(evidence_id='semi_review/example',visual_review=dict(policy=v.POLICY,evidence=m,reviewed_target='YES',reviewed_phase='catchup'),
             target='NO',slice='readiness',episode=c['episode'],edge=c['edge'],observation=dict(frame_id=73,history_frames=m['frames']),image_sha256=m['sha256'])
    with pytest.raises(ValueError):v.row_supported(row)


def test_stratification_resists_dense_scenario_and_deduplicates_physical_routes(monkeypatch):
    monkeypatch.setattr(q,'card',lambda root,s,r,f,ep,edge:dict(scenario=s,route_id=r,frame_id=f,episode=ep,edge=edge))
    rows=[]
    for scenario,town,length in [('dense','Town01',100),('rare','Town02',1),('third','Town03',1)]:
        for route in range(3):
            for f in range(10,10+length):
                rows.append(dict(split='train',scenario=scenario,route_id=f'{town}_route_{route}',physical_group=f'{scenario}/{town}/{route}',frame_id=f,
                    proposals=[dict(instance=dict(event='U-E1',instance_id=f'{route}/{answer}'),episode=dict(event='U-E1'),edge=edge,
                        proposal=dict(geometric_target=answer,release_suggestion=dict(value=answer=='YES')))
                        for answer,edge in [('YES','proceed'),('NO','proceed'),('YES','recover_follow')]]))
    out=q.select(rows,None,per_event=9)
    assert len(out['cards'])==9
    assert {c['scenario'] for c in out['cards']}=={'dense','rare','third'}
    assert len({(c['scenario'],c['route_id']) for c in out['cards']})==9
    assert {s['suggestion'] for s in out['statistics']['strata'] if s['selected']}=={'YES','NO'}
    assert all('suggestion' not in c and 'target' not in c for c in out['cards'])


def test_route_floor_counts_readiness_and_unique_physical_routes_only():
    from qwen3vl_local.sft_new_loop_phase4.admission import review_capacity_report
    row=dict(episode=dict(event='U-E1'),edge='proceed',target='YES',slice='readiness',physical_group='one')
    data=dict(train=[deepcopy(row) for _ in range(100)],val=[],test=[])
    data['train'] += [dict(row,slice='catchup',physical_group=str(i)) for i in range(30)]
    report=review_capacity_report(data)
    assert report['train_routes_per_transition_answer']['U-E1/proceed']['YES']==1
    assert not report['numerical_floor_met'] and len(report['missing_evaluation'])==20


def test_independent_evaluation_packets_cannot_use_development_compiler():
    queue,review=approved_fixture();queue['split']='val';review['queue_sha256']=digest(queue)
    with pytest.raises(ValueError,match='independent'):q.compile_reviews(queue,review,ROOT.parents[1]/'lead_data')


def test_signal_name_cannot_prove_ue7_identity():
    _,_,d,_=evidence();m=d['visual_review']['modes']['2']
    with pytest.raises(ValueError,match='UE7'):v.validate(m,frame=73,frames=m['frames'],hashes=m['sha256'],event='U-E7',edge='proceed',phase='readiness',target='YES')
