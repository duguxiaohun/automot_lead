from copy import deepcopy
import pytest

from qwen3vl_local.audit_joint import rule_baseline as rb,server_finish as sf
from qwen3vl_local.audit_joint.stop_proxy import measure
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v29 import controlled_frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_production_selection_v50 import pool
from qwen3vl_local.sft_new_loop_phase4.teacher_rules import rule_class


@pytest.mark.parametrize('answer',['YES','NO'])
def test_raw_and_filtered_re5_missing_support_share_hard_gate(answer):
    rows=pool('rule_teacher')['train']
    for r in rows:r['edge']={'U-E2':'return','U-E7':'complete'}.get(r['episode']['event'],'proceed')
    rows=[r for r in rows if not (r['episode']['event']=='R-E5' and r['target']==answer)]
    index=dict(routes=[dict(split=r['split'],physical_group=r['physical_group'],targets={
        rule_class(r['episode']['event'],r['edge'],4,r['slice'])+'/'+r['target']:1}) for r in rows])
    source=sf.raw_training_support(index);final=rb.critical_support(rows)
    assert source['critical_support']==final['critical_support']
    key='R-E5/proceed/'+answer
    assert final['missing_critical_support']==[key]
    for policy in rb.COVERAGE_POLICIES:
        assert rb.coverage_decision([key],policy)['blocking_missing_support']==[key]


def test_positive_support_counts_routes_not_repeated_rows():
    row=next(r for r in pool('rule_teacher')['train'] if r['episode']['event']=='R-E5' and r['target']=='YES')
    assert rb.critical_support([row,deepcopy(row)])['critical_support']['R-E5/proceed/YES']==dict(rows=2,physical_groups=1)


def test_export_box_size_changes_proxy_gap_not_center_or_label_authorization():
    frame=controlled_frames(position=2.)[0]
    frame['actors'][-1]['extent']=[1.,2.,.5]
    before=deepcopy(frame);first=measure(frame,50)
    assert frame==before
    frame['actors'][-1]['extent']=[1.5,1.5,.5]
    second=measure(frame,50)
    assert first['center_to_ego_front_m']==second['center_to_ego_front_m']
    assert first['exported_near_face_to_ego_front_m']!=second['exported_near_face_to_ego_front_m']
    assert not first['label_authorization'] and not second['label_authorization']
    assert second['exported_fixed_1p5_shape']
    assert before['actors'][-1]['extent']!=frame['actors'][-1]['extent']


def test_missing_stop_is_unknown_not_completed():
    report=measure(controlled_frames()[0],999)
    assert report['status']=='actor_missing' and not report['label_authorization']
