from copy import deepcopy
import pytest
from qwen3vl_local.audit_joint.instance_lifecycle import inspect


def rows():
    def frame(n, ident):
        r = dict(scenario='s', route_id='r', split='train', frame_id=n,
                 questions=[], instance_ends=[])
        for mode in (2, 4):
            r['questions'].append(dict(scenario='s', route_id='r', split='train', frame_id=n,
                episode=dict(event='U-E5', instance_id=ident),
                instance=dict(actor_id=1, restriction_frame=6 if ident=='old' else 10),
                rgb_mode=mode, edge='proceed', rule_target='NO', slice='readiness'))
        return r
    first, second = frame(7, 'old'), frame(10, 'new')
    second['instance_ends']=[dict(instance_id='old', reason='identity_lost', completed=False)]
    return [first, second]


def test_new_instance_after_retirement_does_not_become_old_completion():
    rs=rows();before=deepcopy(rs);result=inspect(rs)
    assert result['status']=='verified_structure' and rs==before
    assert result['ended']['old']['completed'] is False
    assert result['open_instance_ids']==['new'] and not result['training_authorized']


@pytest.mark.parametrize('defect',['same_frame','later_frame','completed','reused_actor',
    'reused_restriction','duplicate_end','frame_order','route','split','missing_mode','different_target','question_frame'])
def test_bad_lifetime_or_supervision_rejected(defect):
    rs=rows()
    if defect=='same_frame':
        for q in rs[1]['questions']:q['episode']['instance_id']='old'
    elif defect=='later_frame':
        late=deepcopy(rs[0]);late['frame_id']=11
        for q in late['questions']:q['frame_id']=11
        rs.append(late)
    elif defect=='completed':rs[1]['instance_ends'][0]['completed']=True
    elif defect in ('reused_actor','reused_restriction'):
        rs[1]['instance_ends']=[]
        for q in rs[1]['questions']:
            q['episode']['instance_id']='old';q['instance']['restriction_frame']=6
            q['instance']['actor_id' if defect=='reused_actor' else 'restriction_frame']=999
    elif defect=='duplicate_end':rs[1]['instance_ends']*=2
    elif defect=='frame_order':rs.reverse()
    elif defect=='route':rs[1]['route_id']='different'
    elif defect=='split':rs[1]['split']='test'
    elif defect=='missing_mode':rs[0]['questions'].pop()
    elif defect=='different_target':rs[0]['questions'][0]['rule_target']='YES'
    elif defect=='question_frame':rs[0]['questions'][0]['frame_id']=6
    with pytest.raises(ValueError):inspect(rs)


def test_empty_scan_is_not_pass():
    with pytest.raises(ValueError):inspect([])
