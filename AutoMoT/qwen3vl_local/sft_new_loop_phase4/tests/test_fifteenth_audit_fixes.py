from collections import Counter
from copy import deepcopy
import json
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,coverage_schedule,frame_key,CapacityError
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,file_sha
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row


def independent_rows():
    return [row(e*1000+i,event) for e,event in enumerate(EVENTS) for i in range(100 if e==0 else 10)]


@pytest.mark.parametrize('world',[1,4])
def test_independent_large_event_exhausted_in_required_quota_windows(world):
    rows=independent_rows();history=None;seen=set()
    quota=math.ceil(len(rows)/math.lcm(10,world))*math.lcm(10,world)//10
    for epoch in range(math.ceil(100/quota)):
        selected,audit=plan(rows,epoch=epoch,world_size=world,history=history)
        history=audit['next_history'];seen.update(selected)
        assert set(audit['event_counts'].values())=={quota}
        assert max(Counter(frame_key(rows[i]) for i in selected).values())<=8
    assert len(seen)==len(rows)


@pytest.mark.parametrize('world',[1,4])
def test_real_default_pool_covers_all_questions_within_three_epochs(world):
    # Existing reviewed annotations; no generated facts or modified labels.
    from qwen3vl_local.sft_new_loop_phase4.dataset import read_rows
    folder=ROOT.parents[1]/'lead_data'
    if not folder.exists():pytest.skip('external RGB data unavailable')
    from pathlib import Path
    old=Path('/tmp/phase4_fourteenth_fixes_20260930/final2/train.jsonl')
    if not old.exists():pytest.skip('previous production pool unavailable')
    rows=read_rows(old);report=coverage_schedule(rows,world_size=world)
    assert report['pool']==563 and report['complete']
    assert report['timeline'][2]['seen']==563
    assert not report['missing_ids']


def test_history_roundtrip_reordering_and_random_access_agree():
    rows=independent_rows();history=None;plans=[]
    for epoch in range(7):
        ids,a=plan(rows,epoch=epoch,history=history);plans.append(ids)
        assert ids==plan(rows,epoch=epoch)[0]
        history=json.loads(json.dumps(a['next_history']))
        if epoch==2:saved=deepcopy(history)
    rev=list(reversed(rows));state=saved
    for epoch in range(3,7):
        ids,a=plan(rev,epoch=epoch,history=state);state=a['next_history']
        assert [rev[i]['id'] for i in ids]==[rows[i]['id'] for i in plans[epoch]]
    assert state==history


@pytest.mark.parametrize('fault',['epoch','seed','cap','budget','world','pool','counts','last'])
def test_stale_or_inconsistent_history_rejected_without_mutation(fault):
    rows=independent_rows();_,a=plan(rows,epoch=0);state=a['next_history'];kw=dict(epoch=1,history=state)
    if fault=='epoch':kw['epoch']=2
    elif fault=='seed':kw['seed']=10
    elif fault=='cap':kw['cap']=9
    elif fault=='budget':kw['budget']=200
    elif fault=='world':kw['world_size']=4
    elif fault=='pool':rows[0]['target']='changed'
    elif fault=='counts':state['counts'][rows[0]['id']]=-1
    else:state['last_epoch'][rows[0]['id']]=99
    before=deepcopy(state)
    with pytest.raises(ValueError,match='history'):plan(rows,**kw)
    assert state==before


def test_shared_frame_skips_are_remembered_not_counted_as_exposure():
    rows=[row(i,'U-E1',i//4) for i in range(12)]+[row(100+i,'U-E2',i//4) for i in range(12)]
    history=None;seen=set()
    for epoch in range(14):
        before=deepcopy(history)
        ids,a=plan(rows,epoch=epoch,cap=1,budget=2,history=history)
        assert history==before
        history=a['next_history'];seen.update(ids)
        assert a['max_frame_repeat']==1 and a['event_counts']=={'U-E1':1,'U-E2':1}
        assert sum(history['counts'].values())==2*(epoch+1)
    assert len(seen)==len(rows)
    assert not coverage_schedule(rows,epochs=1,cap=1,budget=2)['complete']
    assert not coverage_schedule(rows,epochs=1,cap=1,budget=4)['feasible']


def test_new_risks_gate_all_causal_inputs_without_adding_labels():
    from qwen3vl_local.sft_new_loop_phase4.risk_review import registry,annotation_review
    from qwen3vl_local.sft_new_loop_phase4.dataset import groups
    audit=json.loads((ROOT/'fifteenth_audit_exposure_20260930.json').read_text());known=registry()
    used={(a['scenario'],a['route_id']) for a in json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())}
    assert len(audit['calibration_risks'])==10 and len({r['route_id'] for r in audit['calibration_risks']})==6
    assert set(audit['train_only_groups'])<=groups()
    for risk in audit['calibration_risks']:
        key=risk['scenario'],risk['route_id'];assert key in known and key not in used
        for f in risk['rgb_evidence']:assert file_sha(ROOT.parents[1]/'lead_data'/f['path'])==f['sha256']
        a=dict(scenario=key[0],route_id=key[1],label_basis='reviewed_transition_band',
               transition_band=dict(review_start=10,review_end=10,stop={'frame':11}))
        for mode in (2,4):
            with pytest.raises(ValueError,match='explicit per-frame'):
                annotation_review(a,ROOT.parents[1]/'lead_data',mode,known)
