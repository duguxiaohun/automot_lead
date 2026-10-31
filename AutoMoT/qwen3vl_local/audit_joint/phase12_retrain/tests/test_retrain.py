import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from qwen3vl_local.audit_joint.phase1_visual_refine.tests.test_refine import fixture
from qwen3vl_local.audit_joint.phase1_visual_refine_v2.tests.test_v2 import review
from qwen3vl_local.audit_joint.phase12_retrain import CONTRACT, REVIEW_PATH, data, snapshot, launch
from qwen3vl_local import runtime_version as runtime
from qwen3vl_local.sft_new_loop_phase1 import prompts as native


@pytest.mark.parametrize('mode',['2rgb_endpoints','4rgb'])
def test_real_training_prompt_contract_targets_and_sampling_unchanged(fixture,tmp_path,mode):
    source=Path(native.__file__).read_text();changed=snapshot.training_prompt(source)
    path=tmp_path/'prompts.py';path.write_text(changed)
    spec=importlib.util.spec_from_file_location('fresh_prompt_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    item=fixture.item.spec
    text=module.build_phase1_prompt(spec=item,history_rgb_mode=mode)
    for _,clause in snapshot.RULES.values():assert clause in text
    assert module.phase1_prompt_sha256(history_rgb_mode=mode)!=native.phase1_prompt_sha256(history_rgb_mode=mode)
    assert module.build_phase1_target(fixture.row['answers'],spec=item)==native.build_phase1_target(fixture.row['answers'],spec=item)
    assert module.SYSTEM_PROMPT==native.SYSTEM_PROMPT
    assert module.TRAIN_VARIANT_WEIGHTS==native.TRAIN_VARIANT_WEIGHTS
    assert Path(native.__file__).read_text()==source
    with pytest.raises(ValueError):snapshot.training_prompt(changed)


def test_train_derivation_protects_all_holdout_and_unreviewed_bytes(fixture):
    f=fixture;review(f);before=f.index.read_bytes()
    protected=[]
    for i,split in enumerate(['train','val','test']):
        row={**f.row,'split':split,'frame_id':10+i};protected.append((json.dumps(row)+'  \n').encode())
    f.index.write_bytes(before+b''.join(protected))
    original=f.index.read_bytes();out=f.tmp/'run/data'
    result=data.derive(f.index,f.review,f.root,out)
    assert result['applied_train_reviews']==1 and result['heldouts_byte_identical']
    assert f.index.read_bytes()==original
    lines=(out/'frame_index.jsonl').read_bytes().splitlines(keepends=True)
    assert lines[1:]==protected and json.loads(lines[0])['rs']=='R5'
    assert result['preservation']['protected_rows_byte_identical']==3


@pytest.mark.parametrize('bad',['test','val','rgb','stale','absent'])
def test_invalid_or_nontraining_review_never_admitted(fixture,bad):
    f=fixture;obj=review(f)
    if bad in ('test','val'):obj['decisions'][0]['expected']['split']=bad
    elif bad=='rgb':(f.root/f.row['history_rgb_paths'][0]).write_bytes(b'changed')
    elif bad=='stale':obj['decisions'][0]['expected']['answers']['VULNERABLE']=True
    else:obj['decisions'][0]['frame_id']=500
    f.review.write_text(json.dumps(obj))
    with pytest.raises(ValueError):data.derive(f.index,f.review,f.root,f.tmp/'new/data')
    assert not (f.tmp/'new/data/frame_index.jsonl').exists()


def test_partial_machine_membership_is_explicit(fixture):
    f=fixture;obj=review(f);other=copy.deepcopy(obj['decisions'][0]);other.update(id='absent',frame_id=500)
    obj['decisions'].append(other);f.review.write_text(json.dumps(obj))
    result=data.derive(f.index,f.review,f.root,f.tmp/'out/data')
    assert result['applied_train_reviews']==1 and result['requested_reviews']==2
    assert result['absent_reviews']==[['Scenario','Route',500]]


@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_already_correct_remote_rs_keeps_bytes_and_still_checks_rgb(fixture,layout):
    f=fixture;review(f)
    row=copy.deepcopy(f.row);row['rs']='R5'
    row['answers'].update(RS1=False,RS5=True)
    raw=(json.dumps(row)+'  \n').encode();f.index.write_bytes(raw)
    root=f.tmp/layout;root.mkdir();(root/'lead_data').symlink_to(f.root,target_is_directory=True)
    report=data.derive(f.index,f.review,root/'lead_data',root/'run/data')
    assert report['matched_train_reviews']==1 and report['applied_train_reviews']==0
    assert report['already_correct_train_reviews']==1
    assert (root/'run/data/frame_index.jsonl').read_bytes()==raw
    selected=json.loads((root/'run/selected_train_reviews.json').read_text())['decisions'][0]
    assert selected['expected']['rs']=='R5' and selected['decision']=='retain'
    assert selected['machine_reconciliation']['original_expected']['rs']=='R1'
    (f.root/f.row['history_rgb_paths'][0]).write_bytes(b'wrong image')
    with pytest.raises(ValueError):data.derive(f.index,f.review,root/'lead_data',root/'bad/data')


@pytest.mark.parametrize('bad',['third_rs','inconsistent','unreviewed_fact','holdout','order'])
def test_cross_machine_conflict_is_explicit_and_does_not_publish_data(fixture,bad):
    f=fixture;review(f);row=copy.deepcopy(f.row)
    if bad=='third_rs':row['rs']='R4';row['answers'].update(RS1=False,RS4=True)
    elif bad=='inconsistent':row['rs']='R5'
    elif bad=='unreviewed_fact':row['answers']['VULNERABLE']=True
    elif bad=='holdout':row['split']='test'
    else:row['history_rgb_paths'].reverse()
    f.index.write_text(json.dumps(row)+'\n');before=f.index.read_bytes()
    with pytest.raises(ValueError,match='local train review conflict'):data.derive(f.index,f.review,f.root,f.tmp/'reject/data')
    conflict=json.loads((f.tmp/'reject/train_review_conflicts.json').read_text())['conflicts'][0]
    assert conflict['local']['rs']==row['rs'] and conflict['reviewed_expected']['rs']=='R1'
    assert conflict['reviewed_changes']=={'rs':'R5'}
    assert f.index.read_bytes()==before and not (f.tmp/'reject/data').exists()


def test_mixed_partial_applied_fields_and_original_review_are_preserved(fixture):
    f=fixture;obj=review(f);d=obj['decisions'][0]
    d['changes']['facts']={'STATIC_OBSTACLE':True};d['verified_fields'].append('STATIC_OBSTACLE')
    d['field_evidence']['STATIC_OBSTACLE']=copy.deepcopy(d['field_evidence']['rs'])
    f.review.write_text(json.dumps(obj));original=f.review.read_bytes()
    row=copy.deepcopy(f.row);row['rs']='R5';row['answers'].update(RS1=False,RS5=True)
    f.index.write_text(json.dumps(row)+'\n')
    report=data.derive(f.index,f.review,f.root,f.tmp/'mixed/data')
    selected=json.loads((f.tmp/'mixed/selected_train_reviews.json').read_text())['decisions'][0]
    assert selected['changes']=={'facts':{'STATIC_OBSTACLE':True}}
    new=json.loads((f.tmp/'mixed/data/frame_index.jsonl').read_text())
    assert new['answers']['RS5'] and new['answers']['STATIC_OBSTACLE']
    assert report['applied_train_reviews']==1 and f.review.read_bytes()==original


def miniapp(tmp_path,layout):
    app=tmp_path/layout/'AutoMoT';app.mkdir(parents=True)
    for phase in (1,2):
        folder=app/f'qwen3vl_local/sft_new_loop_phase{phase}';folder.mkdir(parents=True)
        (folder/'prompts.py').write_text(Path(native.__file__).read_text() if phase==1 else '# phase2 untouched')
        (folder/'train.py').write_text('# training untouched')
        for name in ('train.sh','run_full_pipeline.sh'):
            (folder/name).write_text(f'echo checkpoints/sft_new_loop_phase{phase}_runs\n')
    (app/REVIEW_PATH).parent.mkdir(parents=True);(app/REVIEW_PATH).write_text('{}')
    (app/'Automot').mkdir();disk=tmp_path/'disk';disk.mkdir();(disk/'weight.bin').write_bytes(b'weight fixture')
    (app/'Automot/checkpoints').symlink_to(disk,target_is_directory=True)
    if layout=='cr02':(app/'checkpoints').symlink_to(disk,target_is_directory=True)
    return app


@pytest.mark.parametrize('phase',[1,2])
@pytest.mark.parametrize('layout',['cr02','dsw'])
def test_snapshot_isolated_assets_reuse_and_tamper_rejection(tmp_path,phase,layout):
    app=miniapp(tmp_path,layout);before=(app/snapshot.PROMPT).read_bytes()
    version=snapshot.prepare(app,phase);record=runtime.verify(version)
    assert snapshot.prepare(app,phase)==version
    assert (app/snapshot.PROMPT).read_bytes()==before
    other=3-phase
    assert (version/f'app/qwen3vl_local/sft_new_loop_phase{other}/train.sh').read_bytes()==(app/f'qwen3vl_local/sft_new_loop_phase{other}/train.sh').read_bytes()
    assert (version/'app/Automot/checkpoints/weight.bin').read_bytes()==b'weight fixture'
    assert (version/'app'/snapshot.PROMPT).read_bytes() != before if phase==1 else (version/'app'/snapshot.PROMPT).read_bytes()==before
    (app/snapshot.PROMPT).write_text('new main-tree source')
    runtime.verify(version)  # old runtime remains valid after source sync
    target=version/f'app/qwen3vl_local/sft_new_loop_phase{phase}/train.py';target.chmod(0o644);target.write_text('tamper')
    with pytest.raises(ValueError,match='changed'):runtime.verify(version)


@pytest.mark.parametrize('phase',[1,2])
def test_stale_shell_cannot_skip_train_change_sampling_or_reuse_adapter(monkeypatch,tmp_path,phase):
    for key,value in dict(RUN_TRAIN='0',PLAN_ONLY='1',ADAPTER_DIR='old',GRAD_ACCUM='9',TRAIN_FOCUS_BALANCE_COUNT='9999',
                          TRAIN_OUTPUT_DIR='old-run',TRAIN_MAX_STEPS='500000',RUN_EVAL_SH='0',EVAL_SPLIT='test',GPU_IDS='0,1,2,3').items():
        monkeypatch.setenv(key,value)
    env=launch.environment(launch.APP,phase,tmp_path,tmp_path/'frame_index.jsonl',tmp_path,tmp_path,tmp_path)
    assert env['RUN_TRAIN']=='1' and env['PLAN_ONLY']=='0' and env['RUN_EVAL_SH']=='1'
    assert env['TRAIN_MAX_STEPS']=='0' and env['TRAIN_NUM_EPOCHS']=='3' and env['GRAD_ACCUM']=='1'
    assert env['TRAIN_FOCUS_BALANCE_COUNT']==('768' if phase==1 else '1024')
    assert env['EVAL_SPLIT']=='val' and env['FINAL_EVAL_SPLIT']=='test'
    assert 'ADAPTER_DIR' not in env and 'TRAIN_OUTPUT_DIR' not in env


def test_wrong_gpu_count_fails_before_gpu(monkeypatch,tmp_path):
    monkeypatch.setenv('GPU_IDS','0')
    with pytest.raises(ValueError,match='four'):launch.environment(launch.APP,1,tmp_path,tmp_path,tmp_path,tmp_path,tmp_path)


@pytest.mark.parametrize('phase',[1,2])
def test_default_gpu_ids_are_bound_without_querying_gpu(monkeypatch,tmp_path,phase):
    monkeypatch.delenv('GPU_IDS',raising=False)
    env=launch.environment(launch.APP,phase,tmp_path,tmp_path,tmp_path,tmp_path,tmp_path)
    assert env['GPU_IDS']=='0,1,2,3'


@pytest.mark.parametrize('phase',[1,2])
@pytest.mark.parametrize('failure',[None,'cpu_admission','fresh_train_eval','historical_isolation'])
def test_actual_worker_order_and_failure_stop_with_explicit_stage_doubles(tmp_path,monkeypatch,phase,failure):
    if phase==2 and failure=='historical_isolation':pytest.skip('Phase2 does not depend on Phase1 historical isolation')
    root=tmp_path/'run';root.mkdir();version=tmp_path/'version';app=version/'app';app.mkdir(parents=True)
    (version/'manifest.json').write_text('{}')
    parent=tmp_path/'parent';parent.mkdir();index=parent/'frame_index.jsonl';index.write_text('{}\n');(parent/'manifest.json').write_text('{}')
    monkeypatch.setattr(launch,'APP',app)
    monkeypatch.setattr(runtime,'verify',lambda p:{'training_revision':{'phase':phase}})
    monkeypatch.setattr(launch.local_assets,'ensure_data',lambda *args:('reuse',{}))
    monkeypatch.setattr(launch,'select_gpus',lambda count,requested:requested.split(','))
    from qwen3vl_local.audit_joint.phase12_retrain import history_holdout
    monkeypatch.setattr(history_holdout,'verify',lambda *args: {})
    # No CUDA/model inference in this orchestration test.
    calls=[]
    def run(cmd,status,**kw):
        name=status.stem;calls.append(name)
        if failure==name:raise RuntimeError('explicit stage failure')
        if name=='train_label_derivation':
            output=Path(cmd[cmd.index('--output')+1]);output.mkdir()
            (output/'frame_index.jsonl').write_text('{}\n');(output/'manifest.json').write_text('{}')
            for p in [root/'selected_train_reviews.json',output/'training_revision.json',output/'preservation.json']:p.write_text('{}')
        if name=='historical_isolation':
            import shutil
            shutil.copytree(root/'labeled_data',root/'data')
            for filename in history_holdout.FILES:(root/'data'/filename).write_text('{}')
        if name=='cpu_admission':
            (root/'cpu').mkdir();(root/'cpu/status.json').write_text('{"status":"ready"}')
        if name=='fresh_train_eval':
            assert kw['env']['RUN_TRAIN']=='1' and kw['env']['HISTORY_RGB_MODES']=='2rgb_endpoints 4rgb'
            assert str(phase) in cmd[-1]
            for mode in launch.MODES:
                p=root/'pipeline'/mode;p.mkdir(parents=True);(p/'fixture_audit_bundle.tar.gz').write_bytes(b'explicit test double')
        status.write_text('{}')
    monkeypatch.setattr(launch.data_stage,'run',run)
    args=SimpleNamespace(run_dir=root,version=version,phase=phase,parent_data=parent,data_root=tmp_path,model_dir=tmp_path,
                         collection_dir=tmp_path,command='all',cpu_timeout=10,train_timeout=30,history_contract=tmp_path/'history.json')
    cwd=Path.cwd()
    try:
        if failure:
            with pytest.raises(RuntimeError):launch.worker(args)
            assert not (root/'completion.json').exists()
            if failure=='cpu_admission':assert 'fresh_train_eval' not in calls
        else:
            launch.worker(args)
            assert calls==['model_check','train_label_derivation']+(['historical_isolation'] if phase==1 else [])+['cpu_admission','fresh_train_eval']
            assert json.loads((root/'completion.json').read_text())['status']=='trained_evaluated_packaged_quality_pending'
            assert (root/'training_revision_audit.tar.gz').is_file()
    finally:os.chdir(cwd)
