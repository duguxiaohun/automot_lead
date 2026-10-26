import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from qwen3vl_local.audit_joint.phase1_visual_refine import CONTRACT, calibration as c, prompts, regression as g, experiment as e
from qwen3vl_local.sft_new_loop_phase1 import prompts as native, eval as ev
from qwen3vl_local.audit_joint.phase1_visual_refine import rgb_inputs


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'assets'; paths=[]
    for i in range(4):
        path=root/'Scenario'/'Route'/'rgb'/f'{i:04}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(12,6),(i*20,0,0)).save(path);paths.append(str(path.relative_to(root)))
    answers={k:k=='RS1' for k in c.ALL}
    row=dict(scenario='Scenario',route_id='Route',frame_id=3,rs='R1',split='train',answers=answers,
             history_rgb_paths=paths,latest_rgb_path=paths[-1],event='U-E3',town='Town12')
    decision=dict(id='review1',scenario='Scenario',route_id='Route',frame_id=3,decision='correct',changes={'rs':'R5'},
                  expected={k:copy.deepcopy(row[k]) for k in ('rs','split','answers')},
                  frame_evidence=[dict(path=p,sha256=c.sha(root/p)) for p in paths],
                  frame_notes=['visible local STOP junction']*4,reviewer='fixture',reason='fixture',development_exposed=True)
    review=tmp_path/'reviews.json';review.write_text(json.dumps(dict(contract=CONTRACT,decisions=[decision])))
    parent=tmp_path/'parent';parent.mkdir();index=parent/'frame_index.jsonl';index.write_text(json.dumps(row)+'\n')
    (parent/'manifest.json').write_text('{}')
    spec=native.make_prompt_spec(variant='all_random_order',answers=answers,seed_key='fixture',focus='RS1')
    item=ev.WorkItem(ev.FrameRow(idx=0,**row), 'RS1',spec,'RS1:YES','fixture')
    frozen=dict(phase=1,expected_total=1,items=[asdict(item)],history_rgb_mode='4rgb',arguments={'max_new_tokens':64},
                image_sha256={p:c.sha(root/p) for p in paths},adapter_identity={'fixture':True},base_asset_sha256={'weights':'fixture'})
    inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps(frozen))
    return SimpleNamespace(root=root,row=row,decision=decision,review=review,index=index,item=item,frozen=frozen,inputs=inputs,tmp=tmp_path)


def test_derive_changes_only_reviewed_row_and_preserves_parent(fixture):
    f=fixture;other={**f.row,'frame_id':9};raw=json.dumps(other,indent=None)+'  \n'
    with f.index.open('a') as h:h.write(raw)
    before=c.sha(f.index);out=f.tmp/'candidate';report=c.derive(f.index,f.review,f.root,out)
    lines=(out/'frame_index.jsonl').read_text().splitlines(keepends=True)
    assert lines[1]==raw and c.sha(f.index)==before
    new=json.loads(lines[0]);assert new['rs']=='R5' and new['answers']['RS5'] and not new['answers']['RS1']
    assert new['phase2_answers']['RS5'] and new['split']=='train'
    assert report['changed_rows']==1 and report['counts']['unchanged_bytes']==1
    with pytest.raises(FileExistsError):c.derive(f.index,f.review,f.root,out)


@pytest.mark.parametrize('field',['rs','split','answers','history','image'])
def test_stale_review_rejected(fixture,field):
    f=fixture;row=copy.deepcopy(f.row)
    if field=='rs':row['rs']='R2'
    elif field=='split':row['split']='test'
    elif field=='answers':row['answers']['VULNERABLE']=True
    elif field=='history':row['history_rgb_paths'].reverse()
    else:(f.root/row['history_rgb_paths'][0]).write_bytes(b'changed')
    with pytest.raises(ValueError):c.apply(row,f.decision,f.root)


@pytest.mark.parametrize('decision',['retain','uncertain'])
def test_noncorrections_never_change_labels(fixture,decision):
    f=fixture;d=copy.deepcopy(f.decision);d.update(decision=decision,changes={})
    assert c.apply(f.row,d,f.root)==f.row


def test_duplicate_reviews_and_missing_parent_fail(fixture):
    f=fixture;obj=json.loads(f.review.read_text());obj['decisions']*=2;f.review.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='duplicate'):c.load_reviews(f.review)
    obj['decisions']=obj['decisions'][:1];obj['decisions'][0]['frame_id']=500;f.review.write_text(json.dumps(obj))
    before=c.sha(f.index)
    with pytest.raises(ValueError,match='absent'):c.derive(f.index,f.review,f.root,f.tmp/'reject')
    assert c.sha(f.index)==before and (f.tmp/'reject/rejected.json').is_file()


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_asset_root_symlink_and_escape(fixture,layout):
    f=fixture;project=f.tmp/layout/'AutoMoT';project.mkdir(parents=True)
    (project/'lead_data').symlink_to(f.root,target_is_directory=True)
    (project/'checkpoints').symlink_to(f.index.parent,target_is_directory=True)
    assert c.apply(f.row,f.decision,project/'lead_data')['rs']=='R5'
    with pytest.raises(ValueError):c.resolve_rgb(project/'lead_data','../escape')
    with pytest.raises(ValueError):c.derive(project/'checkpoints/frame_index.jsonl',f.review,project/'lead_data',project/'checkpoints/candidate')


@pytest.mark.parametrize('layout',['cr02','dsw'])
@pytest.mark.parametrize('level',['scenario','route','rgb','file'])
@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
def test_external_nested_rgb_links_still_require_frozen_hashes(fixture,layout,level,mode):
    f=fixture;disk=f.tmp/'external_disk';disk.mkdir();project=f.tmp/layout/'AutoMoT';project.mkdir(parents=True)
    (project/'lead_data').symlink_to(f.root,target_is_directory=True)
    relative={'scenario':'Scenario','route':'Scenario/Route','rgb':'Scenario/Route/rgb','file':f.row['history_rgb_paths'][0]}[level]
    logical=f.root/relative;target=disk/logical.name;logical.rename(target);logical.symlink_to(target,target_is_directory=target.is_dir())
    frozen=copy.deepcopy(f.frozen);frozen['history_rgb_mode']=mode;f.inputs.write_text(json.dumps(frozen))
    _,_,report=e.prepare(f.inputs,project/'lead_data')
    assert report['images_verified']==(2 if mode=='2rgb_endpoints' else 4)
    assert c.apply(f.row,f.decision,project/'lead_data')['rs']=='R5'
    (f.root/f.row['history_rgb_paths'][0]).write_bytes(b'wrong image in external disk')
    with pytest.raises(ValueError,match='SHA mismatch'):e.prepare(f.inputs,project/'lead_data')
    with pytest.raises(ValueError,match='content changed'):c.apply(f.row,f.decision,project/'lead_data')


def test_missing_and_broken_rgb_link_show_physical_paths(fixture):
    f=fixture;path=f.root/f.row['history_rgb_paths'][0];path.unlink();missing=f.tmp/'missing_disk/0000.jpg';path.symlink_to(missing)
    with pytest.raises(FileNotFoundError) as error:c.resolve_rgb(f.root,f.row['history_rgb_paths'][0])
    assert str(path) in str(error.value) and str(missing) in str(error.value) and 'symlinks=' in str(error.value)


def test_admit_uses_recorded_root_and_strict_explicit_root(fixture):
    f=fixture;frozen=copy.deepcopy(f.frozen);frozen['arguments']['data_root']=str(f.root);f.inputs.write_text(json.dumps(frozen))
    root,receipt=rgb_inputs.admit(f.inputs,app=f.tmp/'project')
    assert root==f.root and receipt['images']==4
    with pytest.raises(FileNotFoundError,match='No complete frozen RGB'):rgb_inputs.admit(f.inputs,f.tmp/'absent',app=f.tmp)


def test_admit_known_nested_layout_but_no_wrong_hash_fallback(fixture):
    f=fixture;app=f.tmp/'project';(app/'Automot').mkdir(parents=True)
    (app/'Automot/lead_data').symlink_to(f.root,target_is_directory=True)
    root,receipt=rgb_inputs.admit(f.inputs,app=app)
    assert root==app/'Automot/lead_data' and receipt['absent_candidates']
    wrong=app/'lead_data'/f.row['history_rgb_paths'][0];wrong.parent.mkdir(parents=True);wrong.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='SHA mismatch'):rgb_inputs.admit(f.inputs,app=app)


def test_rgb_admission_import_does_not_load_model_runtime(fixture):
    import subprocess,sys
    result=subprocess.run([sys.executable,'-B','-c',
        'import sys; from qwen3vl_local.audit_joint.phase1_visual_refine.rgb_inputs import admit; '
        'assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def records(frozen):
    result={}
    for i,v in enumerate(frozen['items']):
        item=g.item_from_json(1,v);r=g.expected(item,i)
        result[i]={**r,'raw_output':'\n'.join(k+': '+v for k,v in r['gt'].items())}
    return result


def flip(record):
    key=next(iter(record['gt']));value=record['gt'][key]
    record['raw_output']=record['raw_output'].replace(key+': '+value,key+': '+('NO' if value=='YES' else 'YES'))


def test_net_gain_still_rejects_old_correct_loss(fixture):
    frozen=copy.deepcopy(fixture.frozen);frozen['items']*=3;frozen['expected_total']=3
    old=records(frozen);new=copy.deepcopy(old);flip(old[1]);flip(old[2]);flip(new[0])
    report=g.compare(frozen,old,new)
    assert report['joint_correct']==[1,2] and report['old_correct_losses']==[0]
    assert report['status']=='review_required' and report['old_correct_answer_losses']


def test_focus_denominator_is_one_main_question_per_case(fixture):
    old=records(fixture.frozen);report=g.compare(fixture.frozen,old,old)
    assert report['baseline']['FOCUS/RS1']['n']==1
    assert report['baseline']['ALL']['n']==8


def test_source_mutation_rejected_without_requiring_identical_import_set(fixture,monkeypatch):
    f=fixture;source=f.tmp/'source.py';source.write_text('original')
    monkeypatch.setattr(e.run_integrity,'ROOT',f.tmp)
    record={'source.py':c.sha(source)};e.verify_source_identity(record)
    source.write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):e.verify_source_identity(record)


@pytest.mark.parametrize('kind',['missing','identity','gt','spec','invalid_raw'])
def test_comparison_refuses_mismatch_or_format_loss(fixture,kind):
    f=fixture;old=records(f.frozen);new=copy.deepcopy(old)
    if kind=='missing':new.clear()
    elif kind=='identity':new[0]['frame_id']=77
    elif kind=='gt':new[0]['gt']['RS1']='NO'
    elif kind=='spec':new[0]['prompt_spec']={}
    else:new[0]['raw_output']='Reasoning: I think yes.'
    if kind=='invalid_raw':assert g.compare(f.frozen,old,new)['status']=='review_required'
    else:
        with pytest.raises(ValueError):g.compare(f.frozen,old,new)


@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
def test_prepare_and_prompt_preserve_questions_targets(fixture,mode):
    f=fixture;frozen=copy.deepcopy(f.frozen);frozen['history_rgb_mode']=mode;f.inputs.write_text(json.dumps(frozen))
    before=native.phase1_prompt_sha256(history_rgb_mode=mode)
    _,cases,report=e.prepare(f.inputs,f.root)
    assert len(cases)==1 and len(cases[0][1])==(2 if mode=='2rgb_endpoints' else 4)
    spec=f.item.spec;target=native.build_phase1_target(f.row['answers'],spec=spec)
    text=prompts.build(spec=spec,history_rgb_mode=mode)
    assert all(key in text for key in spec.output_keys)
    assert native.build_phase1_target(f.row['answers'],spec=spec)==target
    assert before==native.phase1_prompt_sha256(history_rgb_mode=mode)
    assert report['candidate_prompt_sha256']!=before
    (f.root/f.row['history_rgb_paths'][0]).write_bytes(b'bad')
    with pytest.raises(ValueError,match='SHA'):e.prepare(f.inputs,f.root)


def test_generator_restores_original_system_on_failure(fixture,monkeypatch):
    original=ev.SYSTEM_PROMPT
    def fail(*args):
        assert ev.SYSTEM_PROMPT==prompts.SYSTEM_PROMPT
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(ev,'_generate',fail)
    with pytest.raises(RuntimeError):e.generate_arm(None,[],fixture.item,'4rgb',True,64)
    assert ev.SYSTEM_PROMPT==original


def test_paired_replay_complete_fixture_and_no_overwrite(fixture,monkeypatch):
    f=fixture;monkeypatch.setattr(e.run_integrity,'verify_adapter',lambda p:f.frozen['adapter_identity'])
    monkeypatch.setattr(ev,'_validate_phase1_adapter',lambda *a:{'history_rgb_mode':'4rgb'})
    bundle=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(local_asset_sha256=f.frozen['base_asset_sha256'])))
    monkeypatch.setattr(ev,'load_eval_bundle',lambda *a,**kw:bundle)
    called=[]
    def generate(bundle,images,prompt,tokens):
        called.append(ev.SYSTEM_PROMPT)
        return native.build_phase1_target(f.row['answers'],spec=f.item.spec)
    monkeypatch.setattr(ev,'_generate',generate)
    report=e.replay(f.inputs,f.root,f.tmp/'model',f.tmp/'adapter',f.tmp/'run',device='cpu')
    assert report['status']=='no_observed_regression' and len(called)==2 and called[0]!=called[1]
    assert (f.tmp/'run/completion.json').is_file()
    with pytest.raises(ValueError,match='exists'):e.replay(f.inputs,f.root,f.tmp/'model',f.tmp/'adapter',f.tmp/'run',device='cpu')
