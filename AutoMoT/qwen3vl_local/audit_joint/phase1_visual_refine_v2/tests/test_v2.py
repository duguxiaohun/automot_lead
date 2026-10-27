import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from qwen3vl_local.audit_joint.phase1_visual_refine.tests.test_refine import fixture, records, flip
from qwen3vl_local.audit_joint.phase1_visual_refine import calibration as oldc
from qwen3vl_local.audit_joint.phase1_visual_refine_v2 import CONTRACT, prompts, calibration as c, regression as g, experiment as e, remote
from qwen3vl_local.sft_new_loop_phase1 import prompts as native

def review(f):
    obj=json.loads(f.review.read_text());obj['review_policy']=CONTRACT
    d=obj['decisions'][0];d['verified_fields']=['rs']
    d['field_evidence']={'rs':dict(confidence='confirmed', visible_frames=[0,3], newest_relevance='same local STOP junction remains visible')}
    f.review.write_text(json.dumps(obj));return obj

@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
@pytest.mark.parametrize('candidate',list(prompts.RULES))
def test_single_clause_only_and_system_targets_unchanged(fixture,mode,candidate):
    spec=fixture.item.spec;before=native.build_phase1_target(fixture.row['answers'],spec=spec)
    original=native.build_phase1_prompt(spec=spec,history_rgb_mode=mode)
    text=prompts.build(candidate=candidate,spec=spec,history_rgb_mode=mode)
    assert text!=original
    assert text.replace(prompts.RULES[candidate][1],'',1)==original
    assert prompts.SYSTEM_PROMPT==native.SYSTEM_PROMPT
    assert native.build_phase1_target(fixture.row['answers'],spec=spec)==before

def test_omitted_definition_no_text_change(fixture):
    for seed in range(100):
        spec=native.make_prompt_spec(variant='subset_random',answers=fixture.row['answers'],seed_key=str(seed),focus='RS1')
        old=native.build_phase1_prompt(spec=spec)
        if native.phase2_prompts.RS_DEFINITIONS['RS2'] not in old:
            assert prompts.build(candidate='rs2_same_direction',spec=spec)==old
            break
    else:raise AssertionError('No absent-definition fixture found')

@pytest.mark.parametrize('bad',['unverified','uncertain','no_latest_context','invalid_frame','inconsistent_rs'])
def test_bad_calibration_evidence_rejected(fixture,bad):
    obj=review(fixture);d=obj['decisions'][0]
    if bad=='unverified':d['verified_fields']=[]
    elif bad=='uncertain':d['field_evidence']['rs']['confidence']='uncertain'
    elif bad=='no_latest_context':d['field_evidence']['rs']['newest_relevance']=''
    elif bad=='invalid_frame':d['field_evidence']['rs']['visible_frames']=[4]
    else:d['expected']['answers']['RS2']=True
    fixture.review.write_text(json.dumps(obj))
    with pytest.raises(ValueError):c.load_reviews(fixture.review)

def test_whole_parent_protected_and_tampering_refused(fixture):
    f=fixture;review(f);other=copy.deepcopy(f.row);other['frame_id']=44
    with f.index.open('a') as stream:stream.write(json.dumps(other)+'  \n')
    out=f.tmp/'v2';result=c.derive(f.index,f.review,f.root,out)
    assert result['preservation']['protected_rows_byte_identical']==1
    lines=(out/'frame_index.jsonl').read_text().splitlines(keepends=True)
    bad=json.loads(lines[1]);bad['answers']['VULNERABLE']=True
    (out/'frame_index.jsonl').write_text(lines[0]+json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='Unapproved'):c.verify_derivation(f.index,out/'frame_index.jsonl',f.review,f.root)

def test_recorded_correct_loss_cannot_hide_in_baseline_drift(fixture):
    f=fixture;reference=records(f.frozen);old=copy.deepcopy(reference);flip(old[0])
    report=g.compare(f.frozen,old,old,reference)
    assert report['same_run']['status']=='no_observed_regression'
    assert report['status']=='review_required'
    assert report['against_recorded_baseline']['old_correct_losses']==[0]

def test_answer_loss_even_when_joint_already_wrong_is_blocked(fixture):
    f=fixture;old=records(f.frozen);flip(old[0]);new=copy.deepcopy(old)
    key=list(new[0]['gt'])[1];v=new[0]['gt'][key]
    new[0]['raw_output']=new[0]['raw_output'].replace(key+': '+v,key+': '+('YES' if v=='NO' else 'NO'))
    report=g.compare(f.frozen,old,new,old)
    assert not report['same_run']['old_correct_losses']
    assert report['same_run']['old_correct_answer_losses'] and report['status']=='review_required'

def test_prepare_and_actual_orchestration_with_explicit_generator_double(fixture,monkeypatch):
    f=fixture;reference=f.tmp/'reference';reference.mkdir()
    (reference/'cases_rank0.jsonl').write_bytes(e.reference_bytes(records(f.frozen)))
    (reference/'evaluation_complete.json').write_text(json.dumps(dict(status='verified',phase=1,total_cases=1,
        inputs_sha256=oldc.sha(f.inputs),case_files_sha256={'cases_rank0.jsonl':oldc.sha(reference/'cases_rank0.jsonl')})))
    monkeypatch.setattr(e.run_integrity,'verify_adapter',lambda p:f.frozen['adapter_identity'])
    monkeypatch.setattr(e.native_eval,'_validate_phase1_adapter',lambda *args:{'history_rgb_mode':'4rgb'})
    bundle=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(local_asset_sha256=f.frozen['base_asset_sha256'])))
    monkeypatch.setattr(e.native_eval,'load_eval_bundle',lambda *args,**kw:bundle)
    called=[]
    def generate(bundle,images,prompt,tokens):
        called.append(prompt);assert e.native_eval.SYSTEM_PROMPT==native.SYSTEM_PROMPT
        return native.build_phase1_target(f.row['answers'],spec=f.item.spec)
    monkeypatch.setattr(e.native_eval,'_generate',generate)
    out=f.tmp/'replay';report=e.replay(f.inputs,f.root,f.tmp/'model',f.tmp/'adapter',out,reference,'rs2_same_direction',device='cpu')
    assert len(called)==2 and called[0]!=called[1] and report['status']=='no_observed_regression'
    assert remote.package(out,f.inputs).is_file()
    (out/'candidate.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='binding'):remote.package(out,f.inputs)

def test_bad_reference_rejected_before_model(fixture):
    f=fixture;ref=f.tmp/'ref.jsonl';ref.write_text('')
    with pytest.raises(ValueError,match='completed original'):e.prepare(f.inputs,f.root,ref,'rs2_same_direction')

def test_reference_raw_tampering_rejected(fixture):
    f=fixture;reference=f.tmp/'reference';reference.mkdir()
    raw=reference/'cases_rank0.jsonl';raw.write_bytes(e.reference_bytes(records(f.frozen)))
    (reference/'evaluation_complete.json').write_text(json.dumps(dict(status='verified',phase=1,total_cases=1,
        inputs_sha256=oldc.sha(f.inputs),case_files_sha256={raw.name:oldc.sha(raw)})))
    changed=records(f.frozen);flip(changed[0]);raw.write_bytes(e.reference_bytes(changed))
    with pytest.raises(ValueError,match='changed since'):e.read_reference(reference,f.inputs)

def test_system_restored_on_generation_failure(fixture,monkeypatch):
    old=e.native_eval.SYSTEM_PROMPT
    def fail(*args):raise RuntimeError('synthetic')
    monkeypatch.setattr(e.native_eval,'_generate',fail)
    with pytest.raises(RuntimeError):e.generate_arm(None,[],fixture.item,'4rgb','rs2_same_direction',64)
    assert e.native_eval.SYSTEM_PROMPT==old
