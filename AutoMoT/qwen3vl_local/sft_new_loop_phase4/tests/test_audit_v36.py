"""Paired sampling and causal anomaly boundaries; no label approval."""
from copy import deepcopy
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_data as td,paired_training as pt,risk_review as rr
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question


def test_selection_key_independent_of_mode_and_source_but_not_task():
    a=question();b=deepcopy(a)
    b.update(rgb_mode=4,question_id='new',teacher_sha256='new',rule_class='new');b['input_sources']*=2
    assert td.selection_key(a)==td.selection_key(b)
    b['episode']['instance_id']='different_participant'
    assert td.selection_key(a)!=td.selection_key(b)
    b=deepcopy(a);b['rule_target']='NO'
    assert td.selection_key(a)!=td.selection_key(b)


def onset_risk():
    return dict(risk_id='onset',record=dict(manual_scope='causal_history_window',temporal_scope='confirmed_discontinuity_onset_v1',
        last_unaffected_frame=97,first_affected_frame=98,reported_status='original_RGB_confirmed_visible_discontinuity',
        rgb_evidence=[dict(frame=f,sha256='a'*64) for f in (97,98)]))


@pytest.mark.parametrize('last,expected',[(97,False),(98,True),(99,True),(105,False)])
def test_onset_never_uses_future_disappearance_to_veto_previous_frame(last,expected):
    r=onset_risk();q=dict(causal_sources=[dict(frame_id=f) for f in range(last-6,last+1)])
    assert bool(rr.teacher_window_check(q,[r])['affected_risk_ids'])==expected
    assert bool(rr.manual_risks([r],[last-4,last]))==expected


def test_gap_between_rgb_inputs_still_intersects_anomaly_and_stop_history_counts():
    r=onset_risk()
    assert rr.manual_risks([r],[97,99])
    q=dict(causal_sources=[dict(frame_id=f) for f in range(105,112)],control_evidence={'stop':dict(observations=[dict(sources=[dict(frame_id=98)])])})
    assert rr.teacher_window_check(q,[r])['affected_risk_ids']==['onset']


@pytest.mark.parametrize('field,value',[('first_affected_frame',99),('last_unaffected_frame',96),('reported_status','unconfirmed'),('temporal_scope','guess')])
def test_unproved_onset_rejected(field,value):
    r=onset_risk();r['record'][field]=value
    with pytest.raises(ValueError):rr.risk_window(r['record'])


def test_legacy_envelope_unchanged():
    r=onset_risk();r['record'].pop('temporal_scope');r['record'].pop('last_unaffected_frame');r['record'].pop('first_affected_frame')
    assert rr.manual_risks([r],[93,97])


def row(ident,mode,target='YES'):
    fs=[6,10] if mode==2 else [4,6,8,10]
    return dict(id=ident,physical_group='g',scenario='S',route_id='r',split='train',episode=dict(event='U-E1',instance_id=ident),edge='proceed',slice='readiness',target=target,label_basis='weak_rule_teacher',
        observation=dict(frame_id=10,speed_mps=0.,history_frames=fs),images=[f'{f}.jpg' for f in fs],image_sha256=[str(f) for f in fs],image_rgb_sha256=[str(f) for f in fs])


def test_pair_excludes_unmatched_and_disagreement_without_inventing_labels():
    a=[row('a',2),row('b',2),row('c',2)];b=[row('a',4),row('b',4,'NO'),row('d',4)]
    x,y,r=pt.select(a,b)
    assert [v['id'] for v in x]==[v['id'] for v in y]==['a']
    assert r['exclusions']=={'semantic_or_label_disagreement':1,'missing_in_peer':2}
    assert a[1]['target']=='YES' and b[1]['target']=='NO'


@pytest.mark.parametrize('case',['sha','frame','duplicate','split','empty'])
def test_pair_rejects_mismatched_evidence_or_invalid_inputs(case):
    a=[row('a',2)];b=[row('a',4)]
    if case=='sha':b[0]['image_sha256'][1]='replaced'
    if case=='frame':b[0]['observation']['history_frames'][1]=5
    if case=='duplicate':a*=2
    if case=='split':a[0]['split']='val'
    if case=='empty':b=[]
    with pytest.raises(ValueError):pt.select(a,b)


def test_pair_symmetric_and_order_independent():
    a=[row('b',2),row('a',2)];b=[row('a',4),row('b',4)]
    x,y,p=pt.select(a,b);_,_,q=pt.select(list(reversed(b)),a)
    assert p==q and [r['id'] for r in x]==[r['id'] for r in y]==['a','b']


def test_paired_view_preserves_entire_evaluation_and_binds_both_manifests(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import dataset,admission
    a=dict(train=[row('a',2),row('b',2)],val=[{'reference':'all val'}],test=[{'reference':'all test'}])
    b=dict(train=[row('a',4)],val=[],test=[])
    ma=dict(rgb_mode=2,contract='frozen',seed=1);mb=dict(ma,rgb_mode=4)
    monkeypatch.setattr(dataset,'load_dataset',lambda path:(b,mb))
    monkeypatch.setattr(dataset,'coverage_report',lambda rows: {})
    monkeypatch.setattr(admission,'training_report',lambda *a:dict(trainable=True))
    view,report=pt.load_view(a,ma,'peer')
    assert len(a['train'])==2 and len(view['train'])==1
    assert view['val'] is a['val'] and view['test'] is a['test']
    assert set(report['datasets'])=={'2','4'}
    mb['seed']=2
    with pytest.raises(ValueError,match='seed'):pt.load_view(a,ma,'peer')


def test_shell_forwards_pairing_to_host_check(tmp_path):
    import os,subprocess,json
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nwith open('+repr(str(log))+',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n');fake.chmod(0o755)
    result=subprocess.run(['bash',str(ROOT/'train.sh'),'host-preflight','--paired-with','/peer with spaces'],env=dict(os.environ,PYTHON=str(fake)),capture_output=True)
    assert result.returncode==0
    args=json.loads(log.read_text().splitlines()[-1])
    assert args[args.index('--paired-with')+1]=='/peer with spaces'
