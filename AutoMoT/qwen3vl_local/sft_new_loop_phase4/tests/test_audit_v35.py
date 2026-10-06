"""Local scope and launch regressions; synthetic cases do not approve teachers."""
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
import subprocess
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as t, preflight, train
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import actor
from qwen3vl_local.sft_new_loop_phase4.sampling import CapacityError


@pytest.mark.parametrize('kind',['stop_sign','traffic_light'])
def test_far_control_does_not_block_local_release(kind):
    f=frames()[-1]
    c=actor(8,**{'class':kind},position=[52.,0.,0.],extent=[1.5,1.5,.5],affects_ego=True,state='Red')
    f['actors'].append(c)
    nav=t.local_navigation(f,12.)
    assert t.local_priority(f,nav) is True
    c['position'][0]=10.
    assert t.local_priority(f,nav) is (False if kind=='traffic_light' else None)


@pytest.mark.parametrize('change',['hazard','near_junction','unknown_position','large_trigger'])
def test_signal_exclusion_requires_scope_evidence(change):
    f=frames()[-1];c=actor(8,**{'class':'traffic_light'},position=[52.,0.,0.],affects_ego=True,state='Red')
    f['actors'].append(c)
    if change=='hazard':f['meta']['light_hazard']=True
    if change=='near_junction':f['meta']['distance_to_next_junction']=5.
    if change=='unknown_position':c.pop('position')
    if change=='large_trigger':c['extent']=[50.,2.,1.]
    assert t.local_priority(f,t.local_navigation(f,12.)) is False


def curved_frames():
    fs=frames()
    # The ego lane bends; the actor follows its local tangent, not ego heading.
    angle=.4
    for f in fs:
        f['meta'].update(speed=2.,route=[[0.,0.],[5.,0.],[5.+5*math.cos(angle),5*math.sin(angle)],[5.+10*math.cos(angle),10*math.sin(angle)]])
        f['actors'][1].update(base_type='bicycle',position=[5.+4*math.cos(angle),4*math.sin(angle),0.],yaw=angle,
                              speed=2.,ego_velocity=[2*math.cos(angle),2*math.sin(angle)])
    return fs


def test_curved_cyclist_follow_and_additional_parallel_actor():
    fs=curved_frames()
    assert t.parallel_cyclist(fs,2)
    assert t.seeds(fs)[0]['branch']=='cyclist_follow'
    for f in fs:
        b=deepcopy(f['actors'][1]);b.update(id=3,position=[5.+7*math.cos(.4),7*math.sin(.4),0.])
        f['actors'].append(b);f['image_evidence']['3']=dict(quality_pass=True)
    facts,reasons=t.facts(fs,Episode('U-E4','e',branch='cyclist_follow'),2)
    assert 'additional_local_conflict' not in reasons
    assert facts['restricted_progress_established'] is True
    fs[-1]['actors'][-1]['ego_velocity']=[0.,2.]
    assert t.facts(fs,Episode('U-E4','e',branch='cyclist_follow'),2)[0]['release_ready'] is False


@pytest.mark.parametrize('case',['crossing','opposite','missing_route','near','closing','dark'])
def test_cyclist_exemption_is_not_blanket_permission(case):
    fs=curved_frames()
    for f in fs:
        a=f['actors'][1]
        if case=='crossing':a['ego_velocity']=[0.,2.]
        if case=='opposite':a['yaw']+=math.pi
        if case=='missing_route':f['meta'].pop('route')
        if case=='near':a['position']=[2.,0.,0.];a['yaw']=0.;a['ego_velocity']=[2.,0.]
        if case=='closing':f['meta']['speed']=5.
        if case=='dark':f['image_evidence']['2']['quality_pass']=False
    assert not t.ordinary_cyclist(fs,2)


@pytest.mark.parametrize('flag',['--require-trainable','--require-ready'])
def test_cli_capacity_failure_is_nonzero(monkeypatch,flag):
    monkeypatch.setattr(preflight,'inspect',lambda *a,**kw:dict(data_ready=True,ready=False,sampling_feasible=False,complete_coverage=False))
    monkeypatch.setattr(sys,'argv',['preflight','--dataset','/unused','--model-dir','/model','--data-root','/rgb',flag])
    with pytest.raises(SystemExit) as ex:preflight.main()
    assert ex.value.code==2


def test_direct_train_capacity_rejected_before_loading_model(monkeypatch,tmp_path):
    monkeypatch.setattr(train,'load_dataset',lambda *a,**k:({'train':[{}]},dict(training_admission={})))
    def fail(*a,**kw):raise CapacityError({'requested':10000})
    monkeypatch.setattr(train,'plan',fail)
    monkeypatch.setattr(train,'load_bundle',lambda *a,**kw:pytest.fail('model loaded before capacity check'))
    monkeypatch.setattr(sys,'argv',['train','--dataset','/unused','--output-dir',str(tmp_path/'output'),'--epoch-samples','10000'])
    with pytest.raises(CapacityError):train.main()
    assert not (tmp_path/'output').exists()


def test_shell_forwards_sampling_flags_without_optimizer_flags(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nwith open('+repr(str(log))+',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    import os
    result=subprocess.run(['bash',str(ROOT/'train.sh'),'host-preflight','--epoch-samples=10000','--cap','3','--seed=17','--epochs','2','--max-question-repeat=2','--sampling-policy','event_weighted','--lr','0.001'],env=dict(os.environ,PYTHON=str(fake)),capture_output=True)
    assert result.returncode==0,result.stderr
    calls=[json.loads(x) for x in log.read_text().splitlines()]
    assert len(calls)==2
    args=calls[-1]
    assert '--epoch-samples=10000' in args and args[args.index('--cap')+1]=='3'
    assert '--seed=17' in args and '--max-question-repeat=2' in args and '--lr' not in args
    assert args[args.index('--epochs')+1]=='2'


@pytest.mark.parametrize('mode,world',[('single',1),('ddp',4)])
def test_shell_actual_world_and_capacity_failure_stop_training(tmp_path,mode,world):
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    fake=tmp_path/'python';log=tmp_path/'calls'
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\na=sys.argv[1:]\nwith open('+repr(str(log))+',"a") as f:f.write(json.dumps(a)+"\\n")\nif a[0]=="-c":print("0,1,2,3")\nif any(x.endswith(".preflight") for x in a):sys.exit(2)\n')
    fake.chmod(0o755)
    import os
    result=subprocess.run(['bash',str(ROOT/'train.sh'),mode,'--epoch-samples','10000','--cap=3','--max-question-repeat','2'],env=dict(os.environ,PYTHON=str(fake)),capture_output=True)
    assert result.returncode==2
    calls=[json.loads(x) for x in log.read_text().splitlines()]
    args=calls[-1]
    assert args[1].endswith('.preflight')
    assert args[args.index('--world-size')+1]==str(world)
    assert args[args.index('--epoch-samples')+1]=='10000' and '--cap=3' in args
    assert not any('torch.distributed.run' in c or any(x.endswith('.train') for x in c) for c in calls)


def test_auxiliary_world_diagnostic_does_not_throw_or_override_actual_world():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    rows=[row(i,'U-E1') for i in range(10)]
    actual=preflight.sampling_coverage(rows,epochs=2,world_size=1,budget=10)
    diagnostic=preflight.sampling_coverage(rows,epochs=2,world_size=4,budget=10)
    assert actual['feasible'] is True and actual['complete'] is True
    assert diagnostic['feasible'] is False and 'multiple of 4' in diagnostic['diagnostic']['error']


def test_explicit_legacy_diagnostic_supports_history_free_sampler():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    rows=[row(i,'U-E1') for i in range(10)]
    report=preflight.sampling_coverage(rows,epochs=2,world_size=1,budget=10,policy='legacy_ring')
    assert report['feasible'] and report['complete'] and report['timeline'][-1]['seen']==10


def test_preflight_accepts_actual_single_world_even_when_four_world_diagnostic_fails(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row
    from qwen3vl_local.sft_new_loop_phase4 import branch_support,admission
    rows=[row(i,'U-E1') for i in range(10)]
    data=dict(train=rows,val=[],test=[],review_queue=[])
    manifest=dict(trainable=True,training_admission={},complete_coverage=False,coverage={},risk_review_check={})
    monkeypatch.setattr(preflight,'load_dataset',lambda *a:(data,manifest))
    monkeypatch.setattr(branch_support,'report',lambda *a:{})
    monkeypatch.setattr(admission,'review_capacity_report',lambda *a:dict(numerical_floor_met=False))
    report=preflight.inspect('/unused',epoch_samples=10,world_size=1,epochs=1)
    assert report['sampling_feasible'] is True
    assert report['default_seven_epoch_coverage']['4']['feasible'] is False
    assert report['requested_epoch_coverage']['feasible'] is True
