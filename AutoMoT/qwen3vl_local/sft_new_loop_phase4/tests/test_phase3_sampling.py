from collections import Counter
from copy import deepcopy
import math
import pytest
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,coverage_schedule
from qwen3vl_local.sft_new_loop_phase4.phase3_sampling import capacity_quota
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def pool():
    return [row(i, e) for e,n in zip(EVENTS,[5000,3,21,3000,44,40,30,23,24,32]) for i in range(n)]


@pytest.mark.parametrize('world',[1,2,3,4,8])
def test_bounded_epochs_equal_events_and_whole_pool_repeat_bound(world):
    rows=pool();ids,a=plan(rows,epoch=0,policy='phase3_balanced',world_size=world)
    expected=math.ceil(10240/math.lcm(10,world))*math.lcm(10,world)
    assert len(ids)==expected and len(ids)%world==0
    assert len(set(Counter(rows[i]['episode']['event'] for i in ids).values()))==1
    counts=Counter(ids)
    for e in EVENTS:
        indices=[i for i,r in enumerate(rows) if r['episode']['event']==e]
        assert max(counts[i] for i in indices)<=math.ceil(a['per_event_quota']/len(indices))
    # The largest event is sampled once without replacement, never upsampled.
    assert a['event_repetition'][next(iter(EVENTS))]['unique_questions']==a['per_event_quota']
    assert a['event_repetition'][next(iter(EVENTS))]['max_question_repeat']==1
    assert not coverage_schedule(rows,epochs=1,policy='phase3_balanced')['complete']


def test_pairing_order_rotation_and_epoch_resume_are_identical():
    rows=pool();other=list(reversed(deepcopy(rows)))
    for r in other:r['model_input_sha256']='rgb4'+r['id']
    history=None;peer=None;orders=[];seen=[]
    for epoch in range(3):
        ids,a=plan(rows,epoch=epoch,policy='phase3_balanced',history=history)
        ids4,b=plan(other,epoch=epoch,policy='phase3_balanced',history=peer)
        assert [rows[i]['id'] for i in ids]==[other[i]['id'] for i in ids4]
        history=a['next_history'];peer=b['next_history'];orders.append(ids);seen.append(a['cumulative_unique_questions'])
    assert seen[2]>seen[1]>seen[0]
    again,a=plan(rows,epoch=2,policy='phase3_balanced')
    assert again==orders[2] and a['next_history']==history
    with pytest.raises(ValueError,match='history'):
        plan(rows,epoch=3,policy='phase3_balanced',history=history,budget=20000)
    history['counts'][rows[0]['id']]+=1
    with pytest.raises(ValueError,match='history'):
        plan(rows,epoch=3,policy='phase3_balanced',history=history)


def test_rare_answer_is_not_repeated_to_fill_half_the_event():
    rows=[row(i,'U-E1') for i in range(100)]
    for r in rows:r.update(edge='proceed',target='NO',slice='readiness')
    rows[0]['target']='YES'
    ids,a=plan(rows,epoch=0,policy='phase3_balanced',budget=150)
    assert Counter(ids)[0]==2 and a['max_question_repeat']==2
    assert len(set(ids))==100
    assert capacity_quota({'YES':1,'NO':99},50)=={'NO':49,'YES':1}


def test_physical_routes_take_turns_before_long_route_dominates():
    rows=[row(i,'U-E1') for i in range(100)]
    for i,r in enumerate(rows):r.update(edge='proceed',target='NO',slice='readiness',physical_group='long' if i<98 else str(i))
    ids,_=plan(rows,epoch=0,policy='phase3_balanced',budget=3)
    assert {rows[i]['physical_group'] for i in ids}=={'long','98','99'}


@pytest.mark.parametrize('budget',[0,-1,10001,True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):plan(pool(),epoch=0,policy='phase3_balanced',budget=budget,world_size=4)
