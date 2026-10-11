"""Cancel expired proposals before native replay lifecycle checks."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay, teacher_rules as rules, teacher_events as events
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames


@pytest.mark.parametrize('committed', [False, True])
def test_lifecycle_sees_expired_or_committed_state_before_questions(monkeypatch, committed):
    source={f['frame_id']:f for f in frames()}
    for n in range(11,14):
        f=deepcopy(source[10]);f['frame_id']=n;f['meta']['speed']=0.
        f['meta']['is_junction']=False;f['meta']['distance_to_next_junction']=30.
        f['meta']['ego_matrix'][0][3]=10.
        source[n]=f
    source[10]['meta']['is_junction']=True
    monkeypatch.setattr(replay.g,'load_frame',lambda root,s,r,n:source[n])
    monkeypatch.setattr(replay,'instance_anomalies',lambda *args:[])
    monkeypatch.setattr(rules,'seeds',lambda fs:[dict(event='U-E7',actor_id=2,junction_seen=True,junction_world=[0.,0.,0.])])
    monkeypatch.setattr(events,'update_instance',lambda *args:None)
    monkeypatch.setattr(events,'synchronize_observed',lambda *args:None)
    monkeypatch.setattr(rules,'execution_motion',lambda fs:committed)
    monkeypatch.setattr(replay,'instance_facts',lambda fs,ep,seed:(dict(event_restriction_observed=True,
        release_ready=True,corridor_clear=True,priority_satisfied=True,event_resolved=False,
        stable_progress_established=False,restricted_progress_established=False,stationary_wait_required=False,restriction_present=False),[]))
    route=dict(scenario='Scenario',route_id='r',physical_group='g',split='train',rgb_frames=list(source))
    records=list(replay.route_records(None,route));last=records[-1]
    first=next(r for r in records if r['traces'])['traces'][0]
    assert first['after']['state']=='PROCEED'
    assert bool(first['execution_receipt'])==committed
    if committed:
        assert not last['instance_ends']
        assert any(q['edge']=='complete' for q in last['questions'])
    else:
        assert last['questions']==[]
        assert last['instance_ends']==[dict(instance_id='teacher/Scenario/r/U-E7/2/10',
            reason='observed_junction_exit_censors_stale_wait',completed=False)]
        # Repeated calls to questions() must not double-count expiry.
        assert records[-2]['traces'][0]['before']['state']=='YIELD'
