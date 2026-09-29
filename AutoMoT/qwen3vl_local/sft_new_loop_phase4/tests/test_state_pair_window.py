import copy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.prompts import prompt,messages,parse_answer
from qwen3vl_local.sft_new_loop_phase4.calibration import reviewed_band_labels,band_summary
from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT


def test_state_pair_only_and_no_metadata_text():
    ep=Episode('U-E2','private-instance',direction='LEFT',target_corridor='left bypass',return_corridor='private-future-return')
    obs=dict(frame_id=10,history_frames=[4,6,8,10],speed_mps=12.345)
    text=prompt(ep,'depart',obs)
    assert len(text.splitlines())==3
    assert text.startswith('Current state:') and '\nCandidate next state:' in text
    for hidden in ('12.345','private-instance','private-future-return','Branch:','Return required:','relative frame offsets'):
        assert hidden not in text
    assert 'left bypass' in text and 'entry gap' in text
    obs['speed_mps']=0
    assert prompt(ep,'depart',obs)==text


@pytest.mark.parametrize('target',['UNKNOWN','INVALID','YES because','LANE_CHANGE_LEFT'])
def test_model_output_binary_only(target):
    with pytest.raises(ValueError):parse_answer(target)
    with pytest.raises(ValueError):
        messages(Episode('U-E4','x'),'proceed',dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None,None],target)


def reviewed(case,edge):
    anns=json.loads((ROOT/'reviewed_state_pairs_v3.json').read_text())
    return next(a for a in anns if a['evidence_id']==f'p4_{case:03d}/boundary_v3/{edge}')


@pytest.mark.parametrize('case,edge,before,after',[
    (8,'depart',2,2),(10,'depart',1,2),(13,'depart',3,1),
    (25,'proceed',3,2),(26,'proceed',3,2),(28,'proceed',4,2),
    (42,'proceed',7,4),(46,'proceed',4,5),(70,'proceed',0,4),
    (88,'return',0,3),(91,'return',0,2),(79,'complete',0,2)])
def test_route_specific_boundaries_come_from_review(case,edge,before,after):
    a=reviewed(case,edge)
    s=band_summary(Episode(**a['episode']),edge,a['transition_band'])
    assert (s['before_frames'],s['after_frames'])==(before,after)


def test_real_band_retains_old_pair_and_no_future_images(tmp_path):
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    a=reviewed(13,'pass')
    build([a],root,tmp_path/'data')
    data,_=load_dataset(tmp_path/'data');rows=data['train']
    assert [(r['observation']['frame_id'],r['target'],r['slice']) for r in rows]==[
        (53,'NO','readiness'),(54,'NO','readiness'),(55,'YES','readiness'),(56,'YES','catchup')]
    assert {r['episode']['state'] for r in rows}=={'DEPART'}
    assert len({r['prompt_sha256'] for r in rows})==1
    assert all(max(r['observation']['history_frames'])==r['observation']['frame_id'] for r in rows)


def test_legacy_time_dilation_is_rejected(tmp_path):
    old=json.loads((ROOT/'reviewed_state_pairs_v2.json').read_text())
    a=next(a for a in old if a['label_basis']=='transition_point_window')
    with pytest.raises(ValueError,match='not a Phase4 permission'):
        build([a],ROOT.parents[1]/'lead_data',tmp_path/'data')


@pytest.mark.parametrize('mutation', ['missing_frame','future','past','missing_note','cross_stop',
                                      'no_successor','no_conflict_review','early_milestone',
                                      'unreviewed_reference','bad_stop','unknown_fact'])
def test_bad_or_unreviewed_band_rejected(mutation):
    a=reviewed(13,'pass'); b=a['transition_band']
    if mutation=='missing_frame': del b['frames']['54']
    if mutation=='future': b['frames']['54']['observed_until']=55
    if mutation=='past': b['frames']['54']['observed_until']=53
    if mutation=='missing_note': b['frames']['54']['observation']=''
    if mutation=='cross_stop': b['frames']['57']['phase']='catchup'
    if mutation=='no_successor': del b['frames']['56']['successor_confirmed']
    if mutation=='no_conflict_review': del b['frames']['56']['current_conflict']
    if mutation=='early_milestone': b['frames']['54']['facts']['lateral_entry_complete']=True
    if mutation=='unreviewed_reference': b['reference_frame']=52
    if mutation=='bad_stop': b['stop']['frame']=100
    if mutation=='unknown_fact': b['frames']['55']['facts']['future_action']=True
    with pytest.raises(ValueError): reviewed_band_labels(Episode(**a['episode']),a['edge'],b)


def test_current_blocker_beats_stale_state_catchup():
    a=reviewed(13,'depart');b=a['transition_band']
    b['frames']['54']['current_conflict']=True
    b['frames']['54']['transition_blocking_conflict']=True
    labels=reviewed_band_labels(Episode(**a['episode']),a['edge'],b)
    assert labels[54]==('NO','catchup')
    b['frames']['54']['current_conflict']=False
    b['frames']['54']['transition_blocking_conflict']=False
    b['frames']['54']['facts']={'entry_gap_clear':False}
    assert reviewed_band_labels(Episode(**a['episode']),a['edge'],b)[54][0]=='NO'


def test_unknown_and_end_of_review_never_become_negative():
    a=reviewed(42,'proceed');b=a['transition_band'];ep=Episode(**a['episode'])
    labels=reviewed_band_labels(ep,a['edge'],b)
    assert labels[74]==('UNKNOWN','uncertain') and 87 not in labels
    assert band_summary(ep,a['edge'],b)['right_censored']


def test_readiness_and_catchup_metrics_are_separate():
    from qwen3vl_local.sft_new_loop_phase4.evaluate import metrics
    rows=[dict(target='YES',prediction='NO',event='U-E2',edge='depart',slice='readiness'),
          dict(target='YES',prediction='YES',event='U-E2',edge='depart',slice='catchup')]
    result=metrics(rows)
    assert result['readiness_recall']==0 and result['catchup_recall']==1
    assert 'transition_window_recall' not in result


def test_review_manifest_counts_and_annotations_are_backed_by_viewed_rgb():
    old=json.loads((ROOT/'rgb_review_20260929.json').read_text())
    audit=json.loads((ROOT/'rgb_boundary_review_v3.json').read_text())
    paths={f['rgb'] for r in audit['reviews'] for f in r['frames']}
    prior={f['rgb'] for r in old['reviews'] for f in r['frames']}
    assert len(audit['reviews'])==45 and len(paths)==701
    assert len(paths-prior)==125 and len(paths|prior)==1541
    assert len(audit['contexts'])==10 and len(audit['towns'])==10
    assert set(audit['train_only_groups']) <= set(old['train_only_groups'])
    evidence={(r['scenario'],r['route_id'],f['frame_id']):f['rgb_sha256']
              for r in audit['reviews'] for f in r['frames']}
    for a in json.loads((ROOT/'reviewed_state_pairs_v3.json').read_text()):
        if a['label_basis']!='reviewed_transition_band':continue
        b=a['transition_band'];reviewed_band_labels(Episode(**a['episode']),a['edge'],b)
        for f in b['frames']:
            assert a['frame_sha256'][f]==evidence[a['scenario'],a['route_id'],int(f)]
