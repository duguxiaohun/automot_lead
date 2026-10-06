from qwen3vl_local.sft_new_loop_phase4.tests.safety_fixtures import clearances, loop_clearances
import copy
import json
from types import SimpleNamespace
from qwen3vl_local.sft_new_loop_phase4.route_context import RECOVER_FOLLOW
import pytest
from qwen3vl_local.sft_new_loop_phase4.controller import Episode,combine_priors
from qwen3vl_local.sft_new_loop_phase4.dataset import coverage_report
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS,COMMON,event_edges,template_for
from qwen3vl_local.sft_new_loop_phase4.evaluate import replay
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop,handoff,predict_with_bundle
from qwen3vl_local.sft_new_loop_phase4.observation import observation_contract,validate_observation


def step(ep,frame,*yes,**kw):
    return ep.advance(frame,{e.key:'YES' if e.key in yes else 'NO' for e in ep.questions()},maneuver_clearances=clearances(ep,frame,*yes),**kw)


def all_support(include_common=False):
    rows=[]
    variants=[(e,'default',True) for e in EVENTS]+[('U-E2','default',False),('U-E2','in_lane_pass',False),
        ('U-E4','cyclist_follow',False),('U-E4','cyclist_bypass',True),('U-E4','cyclist_bypass',False)]
    for split in ('train','val','test'):
        for event,branch,ret in variants:
            edges=(*event_edges(template_for(event,branch),ret),*((*COMMON, RECOVER_FOLLOW) if include_common else ()))
            for edge in edges:
                for target in ('YES','NO'):
                    rows.append(dict(split=split,episode=dict(event=event,branch=branch,return_required=ret),
                        edge=edge.key,slice='readiness',target=target,physical_group=f'{split}/{event}/{target}'))
    return rows


def test_event_only_coverage_can_never_be_ready():
    report=coverage_report(all_support())
    assert not report['ready']
    for edge in COMMON:
        assert any(k.endswith('/'+edge.key) for k in report['missing_transition_edges'])
    full=coverage_report(all_support(True))
    assert not full['missing_transition_edges'] and not full['missing_event_support']
    assert len(full['missing_segmented_route_support'])==12 and not full['ready']
    rows=all_support(True)
    for r in rows:
        if r['edge']=='stable' and r['target']=='YES':r['slice']='catchup'
    assert not coverage_report(rows)['ready']


def test_reyield_and_variant_support_cannot_be_borrowed():
    rows=all_support(True)
    assert not coverage_report([r for r in rows if not (r['episode']['event']=='U-E5' and r['edge']=='re_yield')])['ready']
    rows=[r for r in rows if not (r['episode']['branch']=='cyclist_follow' and r['edge']=='hold')]
    assert any('cyclist_follow/hold' in k for k in coverage_report(rows)['missing_transition_edges'])


def test_pending_return_does_not_undo_confirmed_stable():
    ep=Episode('U-E2','x',state='PASS',longitudinal='RECOVER',return_direction='RIGHT',return_corridor='route')
    step(ep,10,'return','stable')
    assert (ep.state,ep.longitudinal)==('RETURN','STABLE')
    restored=Episode(**ep.to_dict());restored.questions()
    assert (restored.state,restored.longitudinal)==('PASS','STABLE')
    restored.questions()
    assert restored.unexecuted_observations==1


@pytest.mark.parametrize('event,state', [('R-E2','CROSS'),('R-E3','CROSS'),('U-E2','RETURN'),('U-E4','RETURN')])
def test_geometry_completion_keeps_new_braking(event,state):
    ep=Episode(event,'x',state=state,branch='cyclist_bypass' if event=='U-E4' else 'default',
               direction='LEFT',return_direction='RIGHT',target_corridor='target',return_corridor='route')
    p=step(ep,1,'complete','restrict')
    assert not ep.needs_recheck and ep.state=='DONE' and not ep.finished
    assert p['lateral']=='KEEP' and p['action']=='DECELERATE'
    assert combine_priors([ep])['action']=='DECELERATE'
    step(ep,2,'hold');assert ep.prior()['action']=='STOP'
    step(ep,3,'release',execution_committed=True);assert not ep.finished
    step(ep,4,'stable');assert ep.finished and ep.prior()['action']=='UNCOND'


def test_same_axis_contradiction_still_rejected():
    ep=Episode('U-E5','x',state='PROCEED')
    assert step(ep,1,'complete','re_yield')['status']=='RECHECK'


def test_explicit_wait_and_missing_execution_and_uncertainty_are_distinct():
    ep=Episode('U-E1','wait',longitudinal='HOLD',stall_limit=2)
    for f in range(1,201):step(ep,f)
    assert ep.waiting_observations==200 and not ep.needs_recheck
    ep=Episode('U-E2','pending',longitudinal='HOLD',direction='LEFT',target_corridor='target',stall_limit=2)
    step(ep,1,'depart');step(ep,2,'depart');ep.questions()
    assert ep.needs_recheck and ep.wait_reason=='permission_not_executed'
    assert ep.prior()['action']=='STOP' and combine_priors([ep])['action']=='STOP'
    ep=Episode('U-E1','uncertain',longitudinal='HOLD',stall_limit=2)
    ep.advance(1,{});ep.advance(2,{})
    assert ep.needs_recheck and ep.wait_reason=='uncertain_observation' and ep.prior()['action']=='STOP'
    ep=Episode('U-E1','fault',longitudinal='HOLD')
    step(ep,1,progress_fault=True)
    assert ep.needs_recheck and ep.wait_reason=='external_progress_fault'


def test_answer_order_cannot_erase_stronger_stop():
    ep=Episode('U-E5','x',state='PROCEED')
    answers={e.key:'NO' for e in reversed(ep.questions())};answers.update(hold='YES',re_yield='YES')
    ep.advance(1,answers)
    assert ep.longitudinal=='HOLD'


def initial():
    return dict(event='U-E2',instance_id='x',state='WAIT',longitudinal='HOLD',direction='LEFT',target_corridor='left')


def predictor(ep,key,obs,*images):return 'YES' if key=='depart' else 'NO'


def receipt(frame,started=9):
    return dict(instance_id='x',decision_frame=frame,observed_frame=frame,started_frame=started,
                edges=['depart'],source='causal_tracker',evidence_id=f'causal_track:{frame}',successor_observed=True)


def test_repeated_yes_is_not_zero_execution_delay():
    observations=[dict(visible_truth_scope='visible_maneuver_conditions_v1',frame_id=f,truth={'depart':'YES'},maneuver_clearances=clearances(Episode(**initial()),f,'depart')) for f in (10,11,12)]
    r=replay(initial(),observations,predictor)
    assert r['missed_answers']==0 and r['missed_execution']==3
    assert r['answer_delay_frames']==[0] and r['accepted_delay_frames']==[0]
    assert r['execution_delay_frames']==[] and r['delay_frames']==[] and not r['completed']
    assert r['unconfirmed_transitions'][0]['ready_since']==10
    observations[-1]['execution_receipt']=receipt(12)
    r=replay(initial(),observations,predictor)
    assert r['execution_delay_frames']==[2] and r['unconfirmed_transitions']==[]


def test_rejected_yes_never_counts_as_accepted_or_executed():
    def conflict(ep,key,obs):return 'YES' if key in ('depart','hold') else 'NO'
    init=initial();init['longitudinal']='STABLE'
    r=replay(init,[dict(visible_truth_scope='visible_maneuver_conditions_v1',frame_id=10,truth={'depart':'YES'})],conflict)
    assert r['answer_delay_frames']==[0] and r['accepted_delay_frames']==[]
    assert r['execution_delay_frames']==[] and r['needs_recheck']


def test_catchup_receipt_after_restore_binds_current_edge_not_new_command():
    loop=Phase4Loop(rgb_mode=2);loop.establish(Episode(**initial()),verified=True)
    loop.tick(dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None]*2,predictor,maneuver_clearances=loop_clearances(loop,10,'depart'))
    loop=Phase4Loop.restore(loop.snapshot())
    loop.tick(dict(frame_id=11,history_frames=[7,11],speed_mps=0),[None]*2,predictor,
              execution_receipts=[receipt(11)],maneuver_clearances=loop_clearances(loop,11,'depart'))
    assert loop.episodes['x'].state=='DEPART'
    assert 'pass' in {e.key for e in loop.episodes['x'].questions()}
    assert loop.episodes['x'].history[-1]['execution_receipt']['started_frame']==9
    with pytest.raises(ValueError):loop.episodes['x'].confirm_execution(receipt(11))


@pytest.mark.parametrize('field,value',[('instance_id','other'),('decision_frame',9),('observed_frame',12),
    ('started_frame',12),('edges',['return']),('source','model_yes'),('successor_observed',False),('evidence_id','')])
def test_invalid_catchup_receipts_rejected(field,value):
    ep=Episode(**initial());step(ep,10,'depart');r=receipt(10);r[field]=value
    with pytest.raises(ValueError):ep.confirm_execution(r)
    assert not ep.history[-1]['committed']


def upstream():
    return (dict.fromkeys(('HIGHWAY','STATIC_OBSTACLE','VULNERABLE','TRAFFIC_LIGHT_ABNORMAL','RS1','RS2','RS4','RS5'),False),
            dict.fromkeys(('UE1','UE3','UE5','UE6','INVALID_EVENT_CONTEXT'),False))


def test_missing_context_is_not_valid_false():
    p1,p2=upstream();p2['UE1']=True;spec=[dict(event='U-E1',instance_id='x')]
    assert handoff(p1,p2,[],spec).episodes
    for phase,key in ((p2,'INVALID_EVENT_CONTEXT'),(p2,'UE6'),(p1,'TRAFFIC_LIGHT_ABNORMAL')):
        saved=phase.pop(key)
        with pytest.raises(ValueError,match='missing'):handoff(p1,p2,[],spec)
        phase[key]=saved


@pytest.mark.parametrize('mode,frames',[(2,[4,6,8,10]),(4,[6,10]),(2,[8,10]),(4,[1,4,7,10])])
def test_cross_mode_and_wrong_spacing_rejected_before_predictor(mode,frames):
    c=observation_contract(mode);obs=dict(frame_id=10,history_frames=frames,speed_mps=0)
    with pytest.raises(ValueError):validate_observation(c,obs,len(frames))
    loop=Phase4Loop(rgb_mode=mode);loop.establish(Episode(**initial()),verified=True)
    def forbidden(*args):raise AssertionError('model must not run')
    with pytest.raises(ValueError):loop.tick(obs,[None]*len(frames),forbidden)
    assert loop.episodes['x'].last_frame==-1
    with pytest.raises(ValueError):predict_with_bundle(SimpleNamespace(observation_contract=c),Episode(**initial()),'depart',obs,[None]*len(frames))


def test_inference_bundle_requires_explicit_contract():
    with pytest.raises(ValueError):
        predict_with_bundle(SimpleNamespace(),Episode(**initial()),'depart',
                            dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None]*2)


def test_loop_does_not_exit_after_geometry_completion_with_stop():
    loop=Phase4Loop(rgb_mode=2)
    ep=Episode('R-E2','x',state='CROSS',longitudinal='HOLD',direction='LEFT',target_corridor='target')
    loop.establish(ep,verified=True)
    def answer(ep,key,obs,images):return 'YES' if key=='complete' else 'NO'
    r=loop.tick(dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None]*2,answer)
    assert not r['completed_instances'] and r['prior']['action']=='STOP'
    assert ep.state=='DONE' and not ep.finished
    r=loop.tick(dict(frame_id=11,history_frames=[7,11],speed_mps=0),[None]*2,lambda *a:'NO')
    assert not r['completed_instances'] and ep.last_frame==11


def test_audit_bundle_contains_test_evaluation_and_rejects_symlinks(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.audit_bundle import create
    import zipfile
    run=tmp_path/'run';run.mkdir();(run/'run.json').write_text('{}')
    test=run/'test';test.mkdir();(test/'metrics.json').write_text('{}');(test/'cases.jsonl').write_text('{}\n')
    create(run,tmp_path/'audit.zip')
    with zipfile.ZipFile(tmp_path/'audit.zip') as z:
        assert set(z.namelist())=={'run.json','test/metrics.json','test/cases.jsonl'}
    (test/'cases.jsonl').unlink();(test/'cases.jsonl').symlink_to(run/'run.json')
    with pytest.raises(ValueError,match='symlink'):create(run,tmp_path/'bad.zip')


def test_adapter_without_observation_contract_rejected_before_weights(tmp_path,monkeypatch):
    pytest.importorskip('transformers')
    from qwen3vl_local.sft_new_loop_phase4.model import load_for_inference
    from qwen3vl_local.sft_new_loop_phase4.identity import contract
    from qwen3vl_local.qwen35 import backend
    def forbidden(*a,**k):raise AssertionError('weights must not load')
    monkeypatch.setattr(backend.LocalModel,'from_pretrained',forbidden)
    (tmp_path/'phase4_contract.json').write_text(json.dumps(dict(source=contract())))
    with pytest.raises(ValueError,match='observation contract'):load_for_inference(tmp_path,tmp_path,'cpu')


def test_full_pipeline_wires_build_train_independent_test_and_audit(tmp_path):
    # Stub the process boundary; this verifies orchestration, not GPU training.
    import os,subprocess
    from pathlib import Path
    from qwen3vl_local.sft_new_loop_phase4.identity import ROOT
    wrapper=tmp_path/'python_stub'
    log=tmp_path/'calls.jsonl'
    wrapper.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if '-c' in sys.argv:
 if 'select_gpus' in sys.argv[2]:print('0')
 else:print(str(Path(os.environ['OUTPUT_DIR'])/'epoch_000'))
''')
    wrapper.chmod(0o755)
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').touch()
    env=dict(os.environ,PYTHON=str(wrapper),MODEL_DIR=str(model),CALL_LOG=str(log),OUTPUT_DIR=str(tmp_path/'run'),
             DATA_DIR=str(tmp_path/'data'),PIPELINE_ROOT=str(tmp_path/'pipeline'),MODE='single')
    env.pop('ANNOTATIONS',None)
    subprocess.run(['bash',str(ROOT/'run_full_pipeline.sh')],env=env,check=True,capture_output=True)
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    modules=[c[1] for c in calls if c[0]=='-m']
    assert modules==['qwen3vl_local.sft_new_loop_phase4.'+n for n in
                     ('full_pipeline','launch','preflight','train','evaluate','audit_bundle')]
    construction=next(c for c in calls if 'qwen3vl_local.sft_new_loop_phase4.full_pipeline' in c)
    assert construction[construction.index('--annotations')+1]==str(ROOT/'reviewed_state_pairs_v9.json')
    evaluation=next(c for c in calls if 'qwen3vl_local.sft_new_loop_phase4.evaluate' in c)
    assert evaluation[evaluation.index('--split')+1]=='test'
    assert evaluation[evaluation.index('--output-dir')+1]==str(tmp_path/'run'/'test')
