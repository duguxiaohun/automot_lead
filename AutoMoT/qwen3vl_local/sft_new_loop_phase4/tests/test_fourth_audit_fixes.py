import copy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.calibration import interval_facts,label_condition,reviewed_band_labels
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay


def initial():
    return dict(event='U-E2',instance_id='x',state='WAIT',longitudinal='HOLD',direction='LEFT',target_corridor='left')


def at_twelve(ep,key,obs):
    return 'YES' if key=='depart' and obs['frame_id']==12 else 'NO'


def observations():
    return [dict(frame_id=10,truth={'depart':'YES'}),dict(frame_id=11,truth={}),
            dict(frame_id=12,truth={'depart':'YES'},execution_committed=True)]


@pytest.mark.parametrize('condition',[False,True,None])
@pytest.mark.parametrize('mode',[2,4])
def test_legacy_catchup_rejected_before_any_dataset_is_published(tmp_path,condition,mode):
    old=json.loads((ROOT/'reviewed_intervals_20260929.json').read_text())
    a=next(a for a in old if a.get('slice')=='catchup')
    a['facts']={'entry_gap_clear':condition}
    # Counterfactual validation data, not a claim about actual RGB contents.
    a['evidence_id']='synthetic_guard_probe_do_not_train'
    a['observation']='synthetic current gap evidence; past execution alone is insufficient'
    if condition is False:
        assert label_condition(Episode(**a['episode']),a['edge'],a['start'],
            interval_facts(a['facts'],a['start'],a['end'],a['evidence_id']))=='NO'
    before=copy.deepcopy(a);output=tmp_path/'rejected'
    with pytest.raises(ValueError,match='legacy catchup requires reviewed_transition_band'):
        dataset.build([a],ROOT.parents[1]/'lead_data',output,rgb_mode=mode)
    assert a==before and not output.exists()


@pytest.mark.parametrize('mode',[2,4])
def test_reviewed_band_builder_honors_current_false_fact_over_observed_successor(tmp_path,mode):
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    anns=json.loads((ROOT/'reviewed_state_pairs_v3.json').read_text())
    a=next(a for a in anns if a['evidence_id']=='p4_013/boundary_v3/depart')
    b=a['transition_band'];changed=set()
    for f,item in b['frames'].items():
        if item['phase']=='catchup':
            item['facts']={'entry_gap_clear':False}
            item['current_conflict']=True
            item['transition_blocking_conflict']=True
            item['observation']='synthetic validation: current entry gap blocked despite observed successor'
            changed.add(int(f))
    assert changed
    target=tmp_path/'synthetic'
    dataset.build([a],root,target,rgb_mode=mode)
    data,_=dataset.load_dataset(target)
    rows=[r for r in data['train'] if r['observation']['frame_id'] in changed]
    assert len(rows)==len(changed) and all(r['target']=='NO' and r['slice']=='catchup' for r in rows)


@pytest.mark.parametrize('unknown',['missing_truth','missing_edge','UNKNOWN','INVALID',None])
def test_unknown_truth_censors_old_interval_and_restarts_all_three_delays(unknown):
    obs=observations()
    if unknown=='missing_truth':del obs[1]['truth']
    elif unknown!='missing_edge':obs[1]['truth']['depart']=unknown
    result=replay(initial(),obs,at_twelve)
    for key in ('answer_delay_frames','accepted_delay_frames','execution_delay_frames','delay_frames'):
        assert result[key]==[0]
    assert result['unconfirmed_transitions']==[dict(edge='depart',end_frame=11,reason='truth_unknown',
        ready_since=10,last_ready_frame=10,answer_frame=None,accepted_frame=None)]
    assert result['steps'][-1]['execution_confirmed']==['depart']


@pytest.mark.parametrize('middle,delay,reason',[('YES',2,None),('NO',0,'condition_closed')])
def test_contiguous_truth_and_explicit_closed_controls(middle,delay,reason):
    obs=observations();obs[1]['truth']['depart']=middle
    result=replay(initial(),obs,at_twelve)
    for k in ('answer_delay_frames','accepted_delay_frames','execution_delay_frames'):
        assert result[k]==[delay]
    assert [r['reason'] for r in result['unconfirmed_transitions']]==([] if reason is None else [reason])


def test_omitted_observation_is_also_a_truth_gap():
    obs=observations();del obs[1]
    result=replay(initial(),obs,at_twelve)
    assert result['execution_delay_frames']==[0]
    record=result['unconfirmed_transitions'][0]
    assert (record['reason'],record['last_ready_frame'],record['end_frame'])==('observation_gap',10,12)


def test_unknown_gap_preserves_old_answer_acceptance_but_not_execution_delay():
    obs=observations()
    result=replay(initial(),obs,lambda ep,key,o:'YES' if key=='depart' else 'NO')
    assert result['answer_delay_frames']==result['accepted_delay_frames']==[0,0]
    assert result['execution_delay_frames']==[0]
    r=result['unconfirmed_transitions'][0]
    assert r['answer_frame']==r['accepted_frame']==r['last_ready_frame']==10
    assert r['reason']=='truth_unknown'


def test_execution_on_unknown_frame_is_observed_but_has_no_exact_readiness_delay():
    obs=observations()[:2];obs[1]['execution_committed']=True
    result=replay(initial(),obs,lambda ep,key,o:'YES' if key=='depart' and o['frame_id']==11 else 'NO')
    assert result['execution_delay_frames']==result['answer_delay_frames']==result['accepted_delay_frames']==[]
    assert result['steps'][-1]['execution_confirmed']==['depart']
    assert result['unconfirmed_transitions'][0]['reason']=='truth_unknown'


def test_unknown_at_sequence_end_is_censored_once():
    result=replay(initial(),observations()[:2],at_twelve)
    assert result['execution_delay_frames']==[]
    assert [r['reason'] for r in result['unconfirmed_transitions']]==['truth_unknown']


def test_edge_no_longer_queried_censors_its_unfinished_readiness():
    init=dict(event='U-E5',instance_id='x',state='PROCEED',longitudinal='RECOVER')
    obs=[dict(frame_id=f,truth={'stable':'YES'}) for f in (10,11,12)]
    result=replay(init,obs,lambda ep,key,o:'YES' if key=='hold' and o['frame_id']==11 else 'NO')
    assert result['execution_delay_frames']==[]
    assert result['unconfirmed_transitions']==[dict(edge='stable',ready_since=10,last_ready_frame=11,
        answer_frame=None,accepted_frame=None,end_frame=12,reason='transition_not_queried')]
