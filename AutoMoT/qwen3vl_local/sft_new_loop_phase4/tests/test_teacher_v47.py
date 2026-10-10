"""Synthetic evidence/temporal counterexamples; not new visual approvals."""
import random
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_controls as c, teacher_events as t, route_prompts as p
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v44 import cut_in
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v29 import controlled_frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import episode


@pytest.mark.parametrize('explicit',['flag','ids'])
@pytest.mark.parametrize('history',['missing','inside','outside'])
def test_cut_in_identity_requires_known_historical_exterior(explicit,history):
    fs=cut_in(lambda n:4. if n<8 else 0.)
    if explicit=='flag':fs[-1]['actors'][1]['is_cut_in']=True
    else:fs[-1]['meta']['cut_in_actors_ids']=[2]
    if history=='missing':fs[0]['meta']['route']=[]
    elif history=='inside':fs[0]['actors'][1]['position'][1]=0.
    assert t.cut_in_identity(fs,fs[-1]['actors'][1],fs[0]['actors'][1]) is (history=='outside')
    assert ('U-E3' in {s['event'] for s in t.seeds(fs)}) is (history=='outside')


def test_different_axis_occupancy_times_do_not_imply_crossing():
    # x overlaps from 0 to 1.2s, y only from 1.65s. Their swept hull overlaps.
    f=controlled_frames(16.)[-1];stop=f['actors'][-1];f['meta']['ego_lane_width']=3.5
    f['actors']=[f['actors'][0],stop,actor(3,position=[10.,5.2,0.],ego_velocity=[10.,-1.],speed=10.)]
    f['image_evidence']['3']={'quality_pass':True}
    assert c.crossing_clear(f,stop) is True
    f['actors'][-1]['ego_velocity']=[0.,-3.]
    assert c.crossing_clear(f,stop) is False
    f['image_evidence']['3']['quality_pass']=False
    assert c.crossing_clear(f,stop) is None


@pytest.mark.parametrize('velocity,expected',[([0.,0.],None),([-1.,0.],None),([1.,0.],(1.,2.)),([3.,0.],(1/3,1.))])
def test_crossing_stationary_away_endpoint_and_short_occupancy(velocity,expected):
    assert c.crossing_interval([0.,1.,0.,1.],[2.,3.,0.,1.],velocity,2.)==expected


@pytest.mark.parametrize('horizon',[0.,-1.,float('nan'),float('inf')])
def test_crossing_invalid_horizons(horizon):
    with pytest.raises(ValueError):c.crossing_interval([0.,1.,0.,1.],[2.,3.,0.,1.],[1.,0.],horizon)


def test_crossing_interval_matches_dense_independent_aabb_oracle():
    rng=random.Random(47)
    region=[-1.,1.,-2.,2.]
    for _ in range(100):
        x,y=rng.uniform(-8,8),rng.uniform(-8,8);v=[rng.uniform(-6,6),rng.uniform(-6,6)]
        box=[x-.8,x+.8,y-.4,y+.4];interval=c.crossing_interval(box,region,v,2.)
        def hit(dt):return not (box[1]+v[0]*dt<region[0] or box[0]+v[0]*dt>region[1] or box[3]+v[1]*dt<region[2] or box[2]+v[1]*dt>region[3])
        if any(hit(i/500) for i in range(1001)):assert interval is not None
        if interval is not None:assert hit(sum(interval)/2)


@pytest.mark.parametrize('event,edge',[('U-E1','proceed'),('U-E2','depart'),('U-E7','proceed'),('R-E2','enter')])
def test_prompt_context_is_not_transition_evidence(event,edge):
    ep=episode(event);obs=dict(frame_id=10,history_frames=[4,6,8,10],speed_mps=0.)
    messages=p.messages(ep,edge,obs,['rgb']*4)
    assert 'Event names and navigation targets identify the question context' in messages[0]['content']
    assert p.prompt_version(ep,edge).endswith('_v11')
    assert 'stop_sign_hazard' not in str(messages)
