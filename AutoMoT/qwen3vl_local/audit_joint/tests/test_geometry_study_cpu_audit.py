import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint import geometry_study_cpu_audit as audit
from qwen3vl_local.audit_joint.geometry_study.tests.test_admission import cohort
from qwen3vl_local.audit_joint.geometry_study.tests.test_study import record


def test_capacity_diagnostic_never_overrides_holdout_rejection():
    a=cohort();a['records']=[r for r in a['records'] if not(r['split']=='test' and r['event']=='LEAD_BRAKE')]
    before=copy.deepcopy(a);r=audit.pool_report(a,per_event=1,world_size=1)
    assert r['admission']['status']=='rejected' and r['capacity']['status']=='passed'
    assert len(r['capacity']['plan']['plans'])==3
    assert not r['formal_conditions_passed'] and not r['creates_ready'] and a==before
    assert r['capacity']['plan']['actual_exposure_advanced'] is False


def test_capacity_full_quota_failure_and_missing_train_event_are_explicit():
    a=cohort();r=audit.pool_report(a,per_event=1280)
    assert r['admission']['status']=='admitted' and r['capacity']['status']=='rejected'
    assert not r['formal_conditions_passed'] and r['capacity']['per_event']==1280
    a['records']=[r for r in a['records'] if not(r['split']=='train' and r['event']=='LEAD_BRAKE')]
    r=audit.pool_report(a,per_event=1,world_size=1)
    assert r['capacity']['error'].startswith('missing_training_events')


def test_quality_reports_mixed_events_and_concentration_without_changing_selection():
    a=cohort();before=audit.admission.validation_ids(a['records'])
    for r in a['records']:
        if r['split']=='val' and r['event']=='LEAD_BRAKE':
            for t in r['representations']['six']['targets']:t.update(target=[.5,0,0,0,0,.5],token_id=-1)
    r=audit.pool_report(a,per_event=1,world_size=1)
    s=r['validation_quality']['six'];assert s['mixed_only_events']==['LEAD_BRAKE']
    assert s['mixed_fraction']==.1 and len(s['unsupported_classes'])==5
    assert s['classes']['KEEP']['physical_groups']==9
    assert s['classes']['KEEP']['largest_group_fraction']==pytest.approx(1/9)
    assert not s['reliability_certified'] and r['admission']['fixed_validation_ids']==before
    # One physical group supplying all pure labels is visible, not called reliable.
    b=cohort();b['records']=[x for x in b['records'] if x['split']!='val' or x['event']=='LEAD_BRAKE']
    q=audit.pool_report(b,per_event=1,world_size=1)['validation_quality']['six']
    assert q['single_group_classes']==['KEEP'] and q['classes']['KEEP']['largest_group_fraction']==1


@pytest.mark.parametrize('native,action,reason,kind',[
    ('LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT','native','flipped'),
    ('LANE_CHANGE_RIGHT',None,'ambiguous_current_direction_projection','near_perpendicular'),
    ('LANE_CHANGE_LEFT',None,'missing_map','other_invalid'),
    ('LANE_CHANGE_LEFT','LANE_CHANGE_LEFT','native','unchanged'),
    ('KEEP','KEEP','native',None)])
def test_direction_buckets(native,action,reason,kind):
    assert audit.direction_kind(dict(native_action=native,action=action,reason=reason))==kind


def test_direction_scan_train_only_excluded_anchors_and_imputation_donor(tmp_path,monkeypatch):
    r=record(event='STATIC_BLOCKAGE');r['split']='train'
    test=copy.deepcopy(r);test.update(split='test',route_id='MustNotRead')
    parent=dict(records=[r,test],data_root=str(tmp_path))
    seen=[]
    from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule
    from qwen3vl_local.audit_joint import label_quarantine
    monkeypatch.setattr(rule,'load_route_trajectory',lambda path:seen.append(path.name) or object())
    monkeypatch.setattr(label_quarantine,'load',lambda *a,**k:[])
    monkeypatch.setattr(audit.sources,'LaneMap',lambda p:SimpleNamespace(hashes={}))
    verified=[];monkeypatch.setattr(audit.native,'verify',lambda p:verified.append(p))
    def build(r,trajectory,quarantine,lanes,*,native_cache,reference_cache):
        native_cache[9,r['event']]=dict(evidence={'confirmed_crossing':'donor'},required_end_frame=10)
        reference_cache[9]=dict(reference={'yaw':3.14})
        point=dict(frame=8,native_action='LANE_CHANGE_LEFT',action='LANE_CHANGE_RIGHT',reason='native',
            direction_origin_frame=9,evidence={'wrong':'missing origin'},imputed=True)
        return dict(anchor_action=None,audit=dict(history_points=[point]*3,future_points=[]))
    monkeypatch.setattr(audit.six,'build',build)
    out=tmp_path/'out';out.mkdir()
    result=audit.direction_scan(parent,tmp_path,out,max_cases=1)
    assert seen==[r['route_id']] and verified==[parent]
    assert result['occurrence_counts']['flipped']==3 and len(result['cases'])==1
    assert result['distinct_event_origin_kind_counts']['flipped']==1
    c=result['cases'][0];assert c['anchor_action'] is None and c['native_evidence']=={'confirmed_crossing':'donor'}
    assert c['native_evidence_origin']==9 and not result['visual_review_complete']
    assert (out/'directions.html').is_file()


def test_project_map_view_uses_known_locations_and_rejects_conflicts(tmp_path):
    root=tmp_path/'CARLA_0915/CarlaUE4/Content/Carla/Maps'
    towns=[f'Town{i:02d}' for i in range(1,8)]+['Town10HD','Town11','Town12','Town13','Town15']
    for t in towns:
        p=(root if t in towns[:8] else root/t)/'OpenDrive'/f'{t}.xodr'
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t)
    view=audit.project_maps(tmp_path,tmp_path/'view')
    assert len(list(view.glob('*.xodr')))==12
    assert audit.project_maps(tmp_path,view)==view
    p=root/'Town01/OpenDrive/Town01.xodr';p.parent.mkdir(parents=True);p.write_text('different')
    with pytest.raises(ValueError,match='conflicting_project_maps'):audit.project_maps(tmp_path,view)
