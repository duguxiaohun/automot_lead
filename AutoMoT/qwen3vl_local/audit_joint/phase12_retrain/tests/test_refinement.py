import importlib.util
import json
from pathlib import Path
import sys
import pytest
from qwen3vl_local.audit_joint.phase12_retrain import refinement as r, snapshot, budget, launch
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2


def module(source,tmp_path,name):
    p=tmp_path/(name+'.py');p.write_text(source)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
def test_direct_eight_contract(tmp_path,mode):
    src=Path(p1.__file__).read_text();m=module(r.phase1_prompt(snapshot.training_prompt(src)),tmp_path,'direct8')
    assert m.TRAIN_VARIANT_WEIGHTS==m.VARIANT_WEIGHTS=={'all_random_order':1}
    assert not m.SUBSET_COUNTS and not m.GROUP_DEFINITIONS
    for i in range(32):
        answers={k:bool(i & (1<<(n%5))) for n,k in enumerate(m.ANSWER_KEYS)}
        spec=m.make_prompt_spec(variant='all_random_order',answers=answers,seed_key=str(i))
        target=m.build_phase1_target(answers,spec=spec)
        assert len(spec.output_keys)==8 and set(spec.output_keys)==set(m.ANSWER_KEYS)
        assert m.parse_phase1_output(target,spec=spec)=={k:'YES' if answers[k] else 'NO' for k in spec.output_keys}
        assert all(v is None for v in m.parse_phase1_output(target+'\nGROUP: YES',spec=spec).values())
        assert m.PROMPT_NAME in m.build_phase1_prompt(spec=spec,history_rgb_mode=mode)
    for variant in ('subset_random','hierarchical_probe'):
        with pytest.raises(ValueError):m.make_prompt_spec(variant=variant,answers=answers,seed_key='x')
    assert m.phase1_prompt_sha256(history_rgb_mode=mode)!=p1.phase1_prompt_sha256(history_rgb_mode=mode)
    assert Path(p1.__file__).read_text()==src


def test_phase2_only_ue3_wording_changes(tmp_path):
    src=Path(p2.__file__).read_text();m=module(r.phase2_prompt(src),tmp_path,'event6')
    for k in ('UE1','UE5','UE6'):assert m.EVENT_DEFINITIONS[k]==p2.EVENT_DEFINITIONS[k]
    assert m.EVENT_DEFINITIONS['UE3']==p2.EVENT_DEFINITIONS['UE3']+r.P2_RULE
    assert m.DOMAIN_DESCRIPTIONS==p2.DOMAIN_DESCRIPTIONS
    assert m.VARIANT_WEIGHTS==p2.VARIANT_WEIGHTS
    for domain in p2.QUESTION_DOMAINS:
        a={k:False for k in p2.ANSWER_KEYS};a[p2.DOMAIN_ANSWER_KEYS[domain]]=True
        s=m.make_prompt_spec(variant='all_random_order',answers=a,seed_key='same')
        old=p2.make_prompt_spec(variant='all_random_order',answers=a,seed_key='same')
        assert m.prompt_spec_to_json(s)==p2.prompt_spec_to_json(old)
    assert Path(p2.__file__).read_text()==src


def test_budget_is_finite_and_phase2_independent():
    b=budget.profile(1,'phase1_40k');assert b['expected_updates']==40320
    assert b['quota']*16//4*3==40320
    assert b['total_step_limit']==40320 and b['epoch_step_limit']==13440
    assert budget.profile(2)['expected_updates']==5376
    with pytest.raises(ValueError):budget.profile(2,'phase1_40k')


def test_review_contains_controls_and_does_not_flip_occluded_cyclist():
    obj,ds=r.cal.load_reviews(r.REVIEW)
    assert all(d['expected']['split']=='train' for d in ds.values())
    assert sum(d['decision']=='correct' for d in ds.values())==6
    assert any(d['decision']=='retain' and d['expected']['answers']['VULNERABLE'] for d in ds.values())
    assert any(d['decision']=='retain' and not d['expected']['answers']['VULNERABLE'] for d in ds.values())
    assert any(d['decision']=='uncertain' for d in ds.values())
    for d in ds.values():
        if d['frame_id']==80 and 'Scenario4_11_' in d['route_id']:
            assert d['decision']=='retain' and d['expected']['answers']['VULNERABLE']
        if d['decision']=='correct':assert d['changes']=={'facts':{'VULNERABLE':True}}


def test_source_bundle_includes_native_external_import():
    from qwen3vl_local.sft_new_loop_phase1.run_integrity import sources
    s=sources(2)
    assert 'lead_video_tools/abnormal_duration_filter.py' in s
    assert 'lead_video_tools/__init__.py' in s


def test_portable_train_review_preserves_controls_and_holdout(tmp_path):
    import copy
    rgb=tmp_path/'rgb';rgb.mkdir();parent=tmp_path/'parent';parent.mkdir()
    base=dict(scenario='s',route_id='r',frame_id=4,split='train',rs='R1',
        answers={k:k=='RS1' for k in p1.ANSWER_KEYS},history_rgb_paths=[f's/r/rgb/{i:04}.jpg' for i in range(1,5)])
    for path in base['history_rgb_paths']:
        p=rgb/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.encode())
    d=dict(id='correction',scenario='s',route_id='r',frame_id=4,decision='correct',
        expected={k:base[k] for k in ('rs','split','answers')},changes={'facts':{'VULNERABLE':True}},
        frame_evidence=[dict(path=p,sha256=r.cal.sha(rgb/p)) for p in base['history_rgb_paths']],
        frame_notes=['explicit frame observation']*4,reason='visible rider',reviewer='test fixture',development_exposed=True)
    rows=[base,dict(base,frame_id=8),dict(base,frame_id=9,split='test')]
    original=b''.join((json.dumps(x)+'  \n').encode() for x in rows)
    index=parent/'frame_index.jsonl';index.write_bytes(original)
    (parent/'manifest.json').write_text('{}');(parent/'training_revision.json').write_text('{}')
    reviews=tmp_path/'review.json';reviews.write_text(json.dumps(dict(contract='phase1_visual_refine_v1',decisions=[d])))
    out=tmp_path/'new/data';report=r.derive(index,rgb,out,reviews)
    after=(out/'frame_index.jsonl').read_bytes().splitlines(keepends=True)
    assert after[1:]==original.splitlines(keepends=True)[1:]
    assert report['preservation']['changed_rows']==1 and report['heldouts_byte_identical']
    assert index.read_bytes()==original
    # An already-correct local machine is accepted without changing bytes.
    out2=tmp_path/'again/data';report2=r.derive(out/'frame_index.jsonl',rgb,out2,reviews)
    assert report2['preservation']['changed_rows']==0
    assert (out2/'frame_index.jsonl').read_bytes()==(out/'frame_index.jsonl').read_bytes()
    (rgb/base['history_rgb_paths'][0]).write_bytes(b'tampered')
    with pytest.raises(ValueError):r.derive(index,rgb,tmp_path/'bad/data',reviews)
    assert not (tmp_path/'bad/data/frame_index.jsonl').exists()


def test_standalone_package_can_import_phase2_native_audit(tmp_path):
    import os, shutil, subprocess
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    target=tmp_path/'code';target.mkdir()
    for name in proof.sources(2):
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(proof.ROOT/name,dest)
    command='import sys; sys.path.insert(0,sys.argv[1]); from qwen3vl_local.sft_new_loop_phase2 import highway_ue3_audit; from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route'
    subprocess.run([sys.executable,'-I','-c',command,str(target)],cwd=tmp_path,check=True,capture_output=True)
