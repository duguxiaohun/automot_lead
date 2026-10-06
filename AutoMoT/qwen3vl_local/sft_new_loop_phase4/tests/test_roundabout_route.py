"""Route observability, completion scope, causal labels and atomic receipts."""
import copy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset, condition_builder as cb, risk_review
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.route_prompts import prompt, messages
from qwen3vl_local.sft_new_loop_phase4.route_context import route_identity, validate_source_context
from qwen3vl_local.sft_new_loop_phase4.route_calibration import label_condition, interval_facts, reviewed_band_labels
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT, file_sha, contract, digest

OBS=dict(frame_id=20,speed_mps=0.,history_frames=[16,20])
NAME='eighteenth_nineteenth_audit_exposure_20261001.json'


def episode(**changes):
    args=dict(event='R-E5',instance_id='roundabout/entry1',target_corridor='right-hand exit beside the red building',
        route_context=dict(kind='roundabout',completion_boundary='past the exit mouth and circulation conflict',navigation_evidence_id='navigation/current/entry1'))
    args.update(changes)
    return Episode(**args)


def test_target_exit_changes_actual_message_but_not_hidden_instance_or_evidence_id():
    a=episode();b=episode(target_corridor='left-hand exit beside the park')
    assert prompt(a,'proceed',OBS)!=prompt(b,'proceed',OBS)
    b=copy.deepcopy(a);b.instance_id='another';b.route_context['navigation_evidence_id']='another receipt'
    assert messages(a,'proceed',OBS,['a','b'])==messages(b,'proceed',OBS,['a','b'])
    assert len(prompt(a,'proceed',OBS).splitlines())==3
    assert 'past the exit mouth' in prompt(a,'proceed',OBS)
    # Also repair the audit's legacy target-only reproduction.
    a=Episode('R-E5','a',target_corridor='first exit');b=Episode('R-E5','b',target_corridor='second exit')
    assert prompt(a,'proceed',OBS)!=prompt(b,'proceed',OBS)


@pytest.mark.parametrize('change',[dict(target_corridor=''),dict(event='U-E4'),dict(route_context={'kind':'roundabout'}),dict(route_context=[]),dict(target_corridor='exit\nYES')])
def test_roundabout_requires_navigation_and_boundary(change):
    with pytest.raises(ValueError):episode(**change)


@pytest.mark.parametrize('value,expected',[(True,'YES'),(False,'NO'),(None,'UNKNOWN')])
def test_exit_boundary_required_for_completion_even_after_entry(value,expected):
    ep=episode(state='PROCEED',longitudinal='HOLD')
    facts=dict(event_resolved=True,route_exit_reached=value)
    assert label_condition(ep,'complete',20,interval_facts(facts,20,20,'review'))==expected
    text=prompt(ep,'complete',OBS)
    assert 'settled into ordinary following' in text  # Alone is insufficient; exit is also required.
    assert 'visibly reached the specified exit boundary' in text
    assert 'stable progress has been established' not in text
    assert 'longitudinal restrictions stay in effect' in text


@pytest.mark.parametrize('value,expected',[(True,'YES'),(False,'NO'),(None,'UNKNOWN')])
def test_catchup_does_not_replace_exit_evidence(value,expected):
    band=dict(review_start=20,review_end=20,reference_frame=20,reference_kind='milestone_observed',
        reference_observation='visible target exit',start_reason='reviewed current RGB',stop=dict(frame=21,kind='review_end',reason='sequence ends'),
        frames={'20':dict(phase='catchup',observed_until=20,observation='current RGB',successor_confirmed=True,
                         current_conflict=True,transition_blocking_conflict=False,
                         facts=dict(event_resolved=True,route_exit_reached=value))})
    assert reviewed_band_labels(episode(state='PROCEED',longitudinal='HOLD'),'complete',band)[20]==(expected,'catchup')
    del band['frames']['20']['facts']['route_exit_reached']
    assert reviewed_band_labels(episode(state='PROCEED'),'complete',band)[20][0]=='UNKNOWN'


def test_future_exit_evidence_rejected():
    facts=interval_facts(dict(event_resolved=True,route_exit_reached=True),20,20,'review')
    facts['route_exit_reached']['observed_until']=21
    with pytest.raises(ValueError,match='future'):label_condition(episode(state='PROCEED'),'complete',20,facts)


def answers(ep,**yes):
    return {e.key:yes.get(e.key,'NO') for e in ep.questions()}


def test_exit_completion_preserves_stop_and_reyield_preserves_instance():
    ep=episode(state='PROCEED',longitudinal='HOLD')
    ep.advance(20,answers(ep,re_yield='YES'))
    assert ep.state=='YIELD' and ep.prior()['action']=='STOP' and not ep.finished
    ep.state='PROCEED'
    ep.advance(21,answers(ep,complete='YES'))
    assert ep.state=='DONE' and ep.prior()['action']=='STOP' and not ep.finished
    assert not {'proceed','re_yield','complete'} & {e.key for e in ep.questions()}


def test_route_receipt_mismatch_rolls_back_whole_tick_and_matching_snapshot_restores():
    ep=episode();loop=Phase4Loop(rgb_mode=2);loop.establish(ep,verified=True)
    before=loop.snapshot()
    receipt=dict(instance_id=ep.instance_id,source='causal_tracker',evidence_id='tracker/20',successor_observed=True,
                 decision_frame=20,observed_frame=20,started_frame=19,edges=['proceed'],route_context_id='wrong')
    predictor=lambda ep,key,obs,ims:'YES' if key=='proceed' else 'NO'
    with pytest.raises(ValueError,match='route context'):loop.tick(OBS,['a','b'],predictor,execution_receipts=[receipt])
    assert loop.snapshot()==before
    receipt['route_context_id']=route_identity(ep)
    loop.tick(OBS,['a','b'],predictor,execution_receipts=[receipt])
    assert ep.state=='PROCEED' and ep.history[-1]['committed']
    assert Phase4Loop.restore(loop.snapshot()).snapshot()==loop.snapshot()
    ep.needs_recheck=True;before=loop.snapshot()
    with pytest.raises(ValueError,match='new event instance'):
        loop.revalidate(ep.instance_id,episode(target_corridor='different exit'),verified=True)
    assert loop.snapshot()==before


def test_roundabout_requires_bound_execution_confirmation():
    ep=episode()
    with pytest.raises(ValueError,match='route-bound'):ep.advance(20,answers(ep,proceed='YES'),execution_committed=True)
    ep.advance(20,answers(ep,proceed='YES'))
    with pytest.raises(ValueError,match='route context'):ep.acknowledge(20)
    ep.acknowledge(20,route_context_id=route_identity(ep))


def test_compiler_generates_boundary_fact_and_missing_stays_unknown():
    ep=episode(state='PROCEED')
    fields={k:v for k,v in ep.to_dict().items() if k in dataset.EPISODE_FIELDS}
    record=dict(scenario='Synthetic',route_id='Town01_Rep0_1_0_route0',episode=fields,
        frame_id=20,observed_until=20,context_valid=True,facts=dict(event_resolved=True,route_exit_reached=False),
        sources=[dict(path='Synthetic/Town01_Rep0_1_0_route0/rgb/0020.jpg',kind='rgb',frame_id=20,sha256='a'*64)])
    producer=dict(name='test',version='1',source='rgb_review',rubric_sha256=cb.rubric_identity())
    ann=next(a for a in cb.annotations_for_record(record,producer) if a['edge']=='complete')
    assert ann['facts']==record['facts']
    assert label_condition(ep,'complete',20,interval_facts(ann['facts'],20,20,'review'))=='NO'
    assert {'route_context.py','route_prompts.py','route_calibration.py'}<=producer['rubric_sha256'].keys()


def test_old_holdout_rubric_is_unchanged_and_does_not_approve_roundabout_extension():
    plan=json.loads((ROOT/dataset.HOLDOUT_PLAN).read_text())
    for name,sha in plan['frozen_rubric_sha256'].items():assert file_sha(ROOT/name)==sha
    owner=next(iter(dataset.holdout_reservations()))
    record=dict(physical_group=owner,evaluation_annotation_protocol=dataset.HOLDOUT_PLAN,episode=episode().to_dict())
    with pytest.raises(ValueError,match='new prospective'):dataset.check_evaluation_protocol(record,{owner:'val'})
    report=dataset.coverage_report([])
    assert len(report['missing_roundabout_support'])==9 and not report['ready']


def test_audit_registration_no_synthetic_labels_and_all_evidence_hashes():
    ledger=json.loads((ROOT/NAME).read_text());registry=risk_review.registry()
    assert contract()['calibration_assets'][NAME]==file_sha(ROOT/NAME)
    assert len(ledger['calibration_risks'])==22
    assert len({(r['scenario'],r['route_id']) for r in ledger['calibration_risks']})==21
    assert ledger['local_review']==dict(new_labels=0,new_complete_sequences=0,new_visual_frames=0,rgb_hash_checks=44)
    assert len(ledger['new_train_only_groups'])==3
    assert not set(ledger['train_only_groups'])&set(dataset.holdout_reservations())
    for g in ledger['new_train_only_groups']:
        assert all(dataset.split_for(g,dataset.groups(),seed)=='train' for seed in range(10))
    for r in ledger['calibration_risks']:
        assert not r['training_label_approved']
        assert any(v['ledger']==NAME for v in registry[r['scenario'],r['route_id']])
        for image in r['rgb_evidence']:assert file_sha(ROOT.parents[1]/'lead_data'/image['path'])==image['sha256']
    for r in ledger['navigation_context_required_routes']:
        r['episode']=dict(event='R-E5',instance_id='test')
        with pytest.raises(ValueError,match='audited roundabout'):validate_source_context(r)
        r['episode']=episode().to_dict();validate_source_context(r)


def test_roundabout_demo_reyield_exit_stop_then_final_release():
    from qwen3vl_local.sft_new_loop_phase4.demo import roundabout
    trace=roundabout()
    assert trace[1]['result']['prior']['action']=='STOP'
    assert trace[3]['result']['prior']['action']=='STOP'
    assert not trace[3]['result']['completed_instances']
    assert trace[-1]['result']['completed_instances']==['roundabout-demo']


def test_wrong_route_truth_cannot_count_as_covered_replay():
    from qwen3vl_local.sft_new_loop_phase4.evaluate import replay
    ep=episode()
    observations=[dict(frame_id=20,truth={'proceed':'YES'},route_context_id='wrong')]
    result=replay(ep.to_dict(),observations,lambda *args:'NO')
    assert result['covered_questions']==0 and result['missed_execution']==0
    observations[0]['route_context_id']=route_identity(ep)
    result=replay(ep.to_dict(),observations,lambda *args:'NO')
    assert result['covered_questions']==1 and result['missed_execution']==1


@pytest.mark.parametrize('mode',[2,4])
def test_roundabout_actual_builder_loader_and_model_identity(mode,tmp_path):
    import pickle
    from PIL import Image
    from qwen3vl_local.sft_new_loop_phase4.model import load_images
    root=tmp_path/'source';route=root/'Synthetic'/'Town01_Rep0_1_0_route0'
    (route/'rgb').mkdir(parents=True);(route/'metas').mkdir()
    for f in range(4,21):Image.new('RGB',(8,8),(f,0,0)).save(route/'rgb'/f'{f:04d}.jpg')
    (route/'metas'/'0020.pkl').write_bytes(pickle.dumps({'speed':0.}))
    a=dict(scenario='Synthetic',route_id=route.name,
        episode={k:v for k,v in episode(state='PROCEED',longitudinal='HOLD').to_dict().items() if k in dataset.EPISODE_FIELDS},
        edge='complete',start=20,end=20,facts=dict(event_resolved=True,route_exit_reached=True),
        slice='readiness',label_basis='per_frame_conditions',context_valid=True,reviewer='synthetic test',
        evidence_id='synthetic',observation='synthetic fixture, not a real label',frame_sha256={'20':file_sha(route/'rgb'/'0020.jpg')})
    b=copy.deepcopy(a);b['episode']['target_corridor']='other exit beside the park';b['episode']['instance_id']='other'
    b['facts']['route_exit_reached']=False
    out=tmp_path/'data';dataset.build([a,b],root,out,rgb_mode=mode)
    data,_=dataset.load_dataset(out);rows=sum(data.values(),[])
    assert len(rows)==2 and {r['target'] for r in rows}=={'YES','NO'}
    assert len({r['model_input_sha256'] for r in rows})==2
    for row in rows:
        images=load_images(row,root)
        assert len(images)==mode
        for im in images:im.close()
    b['episode']['target_corridor']=a['episode']['target_corridor']
    with pytest.raises(ValueError,match='conflicting answers for identical model input'):
        dataset.build([a,b],root,tmp_path/'conflict',rgb_mode=mode)
