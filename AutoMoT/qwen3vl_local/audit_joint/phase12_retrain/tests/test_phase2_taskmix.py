"""Task composition must preserve donors, holdout bytes, and execution layout."""
import copy
import json
from pathlib import Path
import sys

import pytest

from qwen3vl_local.audit_joint.phase12_retrain import phase2_taskmix as mix, refinement, launch, snapshot
from qwen3vl_local.sft_new_loop_phase2 import prompts
from qwen3vl_local.sft_new_loop_phase1 import data_stage
from qwen3vl_local.audit_joint.phase12_retrain.tests.test_phase2_evidence import row
from qwen3vl_local.audit_joint.phase12_retrain.tests.test_refinement import module


@pytest.fixture
def donors(tmp_path):
    indexes=[]
    for source in ('legacy','evidence'):
        d=tmp_path/source;d.mkdir()
        rows=[row(f'{source}_{split}_{cls}',20,cls,split) for split in ('train','val') for cls in mix.TASK_SOURCES]
        rows += [row(f'test_{cls}',20,cls,'test') for cls in mix.TASK_SOURCES]
        index=d/'frame_index.jsonl'
        index.write_text(''.join(json.dumps(r)+'  \n' for r in rows))
        (d/'manifest.json').write_text('{}')
        (d/'training_revision.json').write_text(json.dumps({'revision':'evidence_20261030','policy':{'policy_version':'evidence_policy_v3_20261030'}}))
        (d/'val_original.jsonl').write_text(''.join(json.dumps(r)+'  \n' for r in rows if r['split']=='val'))
        for name in ('holdout_disputes.jsonl','evidence_exclusions.jsonl','val_exclusions.jsonl'):
            (d/name).write_text(''.join(json.dumps({'target_event_class':c})+'\n' for c in mix.TASK_SOURCES))
        (d/'evidence_sources.json').write_text('{}')
        indexes.append(index)
    return indexes


def test_composition_preserves_complete_task_buckets_and_frozen_test(donors,tmp_path):
    before=[p.read_bytes() for p in donors];out=tmp_path/'run/data'
    report=mix.derive(*donors,out)
    actual=(out/'frame_index.jsonl').read_bytes().splitlines(keepends=True)
    expected=[]
    for split in ('train','val'):
        for name,raw in zip(('legacy','evidence'),before):
            expected += [l for l in raw.splitlines(keepends=True) if json.loads(l)['split']==split and mix.TASK_SOURCES[mix.target_class(json.loads(l))]==name]
    expected += [l for l in before[0].splitlines(keepends=True) if json.loads(l)['split']=='test']
    assert actual==expected and [p.read_bytes() for p in donors]==before
    assert report['labels_flipped']==report['new_annotation_decisions']==0
    assert (out/'holdout_disputes.jsonl').read_bytes()==(donors[1].parent/'holdout_disputes.jsonl').read_bytes()
    assert (out/'evidence_parent_receipt.json').read_bytes()==(donors[1].parent/'training_revision.json').read_bytes()
    assert {json.loads(l)['target_event_class'] for l in (out/'evidence_exclusions.jsonl').read_text().splitlines()}=={'UE3','UE6','RE','INVALID'}
    with pytest.raises(ValueError,match='fresh'):mix.derive(*donors,out)


@pytest.mark.parametrize('bad',['test','empty','split','conflict','policy'])
def test_bad_donors_block_before_writing(donors,tmp_path,bad):
    p=donors[0];rows=[json.loads(l) for l in p.read_text().splitlines()]
    if bad=='test':rows[-1]['frame_id']+=1
    elif bad=='empty':rows=[r for r in rows if not (r['split']=='train' and mix.target_class(r)=='UE1')]
    elif bad=='split':
        q=copy.deepcopy(rows[0]);q['split']='test';rows.append(q)
        with donors[1].open('a') as f:f.write(json.dumps(q)+'  \n')
    elif bad=='conflict':
        q=next(r for r in rows if r['split']=='train' and mix.target_class(r)=='UE5')
        q.update({k:rows[0][k] for k in ('scenario','route_id','frame_id','question_domain')})
    else:(donors[1].parent/'training_revision.json').write_text('{}')
    p.write_text(''.join(json.dumps(r)+'  \n' for r in rows))
    with pytest.raises(ValueError):mix.derive(*donors,tmp_path/'bad/data')
    assert not (tmp_path/'bad/data').exists()


def test_prompt_definitions_are_exact_donor_rules(tmp_path):
    source=Path(prompts.__file__).read_text()
    old=module(refinement.phase2_prompt(source),tmp_path,'mix_old')
    new=module(refinement.phase2_prompt_v7(source),tmp_path,'mix_new')
    combined=module(refinement.phase2_prompt_taskmix(source),tmp_path,'mix_combined')
    for key in ('UE1','UE5'):assert combined.EVENT_DEFINITIONS[key]==old.EVENT_DEFINITIONS[key]
    for key in ('UE3','UE6'):assert combined.EVENT_DEFINITIONS[key]==new.EVENT_DEFINITIONS[key]
    assert combined.SYSTEM_PROMPT==old.SYSTEM_PROMPT==new.SYSTEM_PROMPT
    assert combined.DOMAIN_DESCRIPTIONS==old.DOMAIN_DESCRIPTIONS==new.DOMAIN_DESCRIPTIONS
    for domain in prompts.QUESTION_DOMAINS:
        a={k:False for k in prompts.ANSWER_KEYS};a[prompts.DOMAIN_ANSWER_KEYS[domain]]=True
        s=combined.make_prompt_spec(variant='all_random_order',answers=a,seed_key='same')
        assert combined.prompt_spec_to_json(s)==old.prompt_spec_to_json(old.make_prompt_spec(variant='all_random_order',answers=a,seed_key='same'))
        assert combined.parse_event_output(combined.build_event_target(s),spec=s)=={q.output_key:False for q in s.questions}
    assert Path(prompts.__file__).read_text()==source


@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
def test_online_prompt_recovery_matches_new_training_and_rejects_tamper(tmp_path,monkeypatch,mode):
    from qwen3vl_local.action_prior import prompt_versions
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    source=refinement.phase2_prompt_taskmix(Path(prompts.__file__).read_text())
    m=module(source,tmp_path,'taskmix_online_'+mode)
    meta=dict(prompt_name=m.PROMPT_NAME,history_rgb_mode=mode,input_layout='text_before_images_v1',
              production_prompt_sha256=m.event_prompt_sha256(history_rgb_mode=mode))
    restored=prompt_versions.prompt_module(2,meta)
    assert prompt_versions.prompt_source(restored)==source.encode()
    meta['production_prompt_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):prompt_versions.prompt_module(2,meta)


def test_default_prefix_is_revision_specific_and_cache_can_be_disabled(monkeypatch,tmp_path):
    for k in ('AUTOMOT_TEXT_PREFIX','AUTOMOT_TEXT_PREFIX_CACHE'):monkeypatch.delenv(k,raising=False)
    launch.configure_revision('visual_20261029',2)
    import os
    assert 'AUTOMOT_TEXT_PREFIX' not in os.environ
    launch.configure_revision(mix.REVISION,2)
    assert os.environ['AUTOMOT_TEXT_PREFIX']==os.environ['AUTOMOT_TEXT_PREFIX_CACHE']=='1'
    env=launch.environment(launch.APP,2,tmp_path,tmp_path/'index',tmp_path,tmp_path,tmp_path,'phase2_10k')
    assert env['AUTOMOT_TEXT_PREFIX']==env['AUTOMOT_TEXT_PREFIX_CACHE']=='1'
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','0');launch.configure_revision(mix.REVISION,2)
    assert os.environ['AUTOMOT_TEXT_PREFIX_CACHE']=='0'
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','0')
    with pytest.raises(ValueError,match='text_before'):launch.configure_revision(mix.REVISION,2)
    with pytest.raises(ValueError,match='Phase2'):snapshot.prepare(launch.APP,1,mix.REVISION)


def test_dual_score_stage_status_cannot_overwrite_report(tmp_path):
    # Execute the actual nested stage call extracted from worker, with the real
    # supervisor and a tiny child standing in for the score writer.
    import ast
    tree=ast.parse(Path(launch.__file__).read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
              and n.func.id=='stage' and isinstance(n.args[0],ast.Constant) and n.args[0].value.startswith('dual_score'))
    report=tmp_path/'dual_score.json'
    stage_name=call.args[0].value
    data_stage.run([sys.executable,'-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(\'{"modes": {"fixture": true}}\')',str(report)],
                   tmp_path/(stage_name+'.json'),timeout=30)
    assert json.loads(report.read_text())=={'modes':{'fixture':True}}
    assert json.loads((tmp_path/'dual_score_stage.json').read_text())['status']=='completed'


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_taskmix_snapshot_matches_composed_prompt_on_both_server_layouts(tmp_path,layout):
    from qwen3vl_local.audit_joint.phase12_retrain.tests.test_retrain import miniapp
    from qwen3vl_local import runtime_version
    app=miniapp(tmp_path,layout)
    target=app/'qwen3vl_local/sft_new_loop_phase2/prompts.py'
    source=Path(prompts.__file__).read_text();target.write_text(source)
    version=snapshot.prepare(app,2,mix.REVISION)
    assert (version/'app'/target.relative_to(app)).read_text()==refinement.phase2_prompt_taskmix(source)
    assert target.read_text()==source
    assert runtime_version.verify(version)['training_revision']['revision']==mix.REVISION


def test_ready_blocks_cache_capacity_change_before_model_loading(tmp_path,monkeypatch):
    from qwen3vl_local.qwen35 import cache_policy
    from qwen3vl_local.audit_joint.phase12_retrain import CONTRACT
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE','1')
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES','2')
    version=tmp_path/'version'
    (tmp_path/'ready.json').write_text(json.dumps(dict(phase=2,version=str(version),contract=CONTRACT,
        revision=mix.REVISION,input_layout='text_before_images_v1',text_cache_policy=cache_policy.current())))
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX_CACHE_ENTRIES','1')
    with pytest.raises(ValueError,match='deployment policy'):launch.verify_ready(tmp_path,2,version)
