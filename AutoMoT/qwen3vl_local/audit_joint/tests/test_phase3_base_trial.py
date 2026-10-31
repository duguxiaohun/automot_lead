"""Controlled experiment contracts; small CPU models are not 4B quality evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint import phase3_base_trial_worker as worker
from qwen3vl_local.sft_new_loop_phase3 import train, sequence_evaluation
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity
from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
from qwen3vl_local.sequence_batch import digest
from qwen3vl_local.qwen35.tests.test_backend import model


def job(tmp_path, stage='quick'):
    return dict(output=str(tmp_path/'run'), index=str(tmp_path/'index'),
        data_root=str(tmp_path/'data'), model_dir=str(tmp_path/'model'),
        stage=stage, stop_after_epoch=1 if stage=='quick' else None, resume=None,
        resume_epoch=0, reference_sha256=trial.file_sha(trial.REFERENCE), release_sha256='fixture')


def fixed_pool(per_event=32):
    rows=[r for r in candidate_rows() if not r.invalid_source]
    rows=[replace(r, route_id=f'{r.route_id}_{i}', frame_id=i,
        answers={k:False for k in r.answers}, action_signature='NONE')
        for r in rows for i in range(per_event)]
    work=[train._make_item(r,seed=20260904,action_output_mode='choice') for r in rows]
    plan=dict(seed=20260904,per_event=per_event,selected_ids=[digest(case_identity(x)) for x in work])
    return rows,work,dict(validation={'generation':plan},seed=20260904,test_plan=plan)


def test_budget_and_args_cannot_inherit_environment(tmp_path,monkeypatch):
    for key,value in {'BATCH_SIZE':'8','GRAD_ACCUM':'16','NUM_EPOCHS':'1'}.items():
        monkeypatch.setenv(key,value)
    args=worker.make_args(job(tmp_path)); p=trial.profile()
    assert args.batch_size==args.grad_accum==1
    assert p['presentations_per_epoch']//(p['world_size']*args.batch_size)*args.num_epochs//args.grad_accum==7680
    assert args.warmup_steps==2000 and args.num_epochs==3 and args.stop_after_epoch==1
    assert args.history_rgb_mode=='4rgb' and args.action_output_mode=='choice'
    assert not args.reuse_frozen_rgb and args.generation_eval_batch_size==1
    assert not args.save_best_val and not args.save_best_generation and args.save_training_state
    complete=worker.make_args(job(tmp_path,'complete'))
    for key in vars(args):
        if key!='stop_after_epoch':assert getattr(args,key)==getattr(complete,key)


def test_fixed_members_order_and_missing_split_duplicate_refused(monkeypatch):
    rows,work,ref=fixed_pool(2)
    selected,audit=trial.select_reference(list(reversed(work)),ref,seed=20260904,limit=2)
    assert selected==work and audit['selected_ids']==ref['validation']['generation']['selected_ids']
    for broken,pattern in [(work[:-1],'missing'),(work+[work[0]],'Duplicate')]:
        with pytest.raises(ValueError,match=pattern):trial.select_reference(broken,ref,seed=20260904,limit=2)
    with pytest.raises(ValueError,match='seed/budget'):trial.select_reference(work,ref,seed=4,limit=2)
    wrong=[train._make_item(replace(r,split='test'),seed=20260904,action_output_mode='choice') for r in rows]
    with pytest.raises(ValueError,match='split'):trial.select_reference(wrong,ref,seed=20260904,limit=2)
    with pytest.raises(ValueError,match='stride'):trial.select_reference(work,ref,seed=20260904,limit=2,stride=2)
    monkeypatch.setattr(sequence_evaluation,'select_work',sequence_evaluation.select_work)
    worker.install_reference(ref)
    actual,report=train._validation_work(rows,target_per_bin=2,seed=20260904,action_output_mode='choice',equal_events=True)
    assert [case_identity(x) for x in actual]==[case_identity(x) for x in work]
    assert report['sampled_cases']==20 and report['effective']==2
    actual,_=trial.select_reference(wrong,ref,seed=20260904,limit=0,split='test')
    assert len(actual)==20


def test_legacy_generator_real_small_qwen_matches_batch_one(model):
    from qwen3vl_local.tests.test_batch_runtime import row,tokenizer
    from qwen3vl_local.batch_runtime import greedy_batch
    tok=tokenizer()
    bundle=SimpleNamespace(model=model,tokenizer=tok,processor=SimpleNamespace(
        batch_decode=lambda ids,**kw:[tok.decode(r) for r in ids]))
    inputs=[row(4)]
    assert worker.legacy_single_generate(bundle,inputs,max_new_tokens=5,phase3=True)==greedy_batch(bundle,inputs,max_new_tokens=5,phase3=True)
    with pytest.raises(ValueError):worker.legacy_single_generate(bundle,inputs*2,max_new_tokens=5,phase3=True)
    with pytest.raises(ValueError):worker.legacy_single_generate(bundle,inputs,max_new_tokens=5,phase3=True,reuse_rgb=True)


def test_summary_scores_keep_and_rejects_changed_saved_predictions(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3.prompts import prompt_spec_to_json,parse_action_output
    _,work,_=fixed_pool()
    reference=copy.deepcopy(trial.reference())
    reference['validation']['generation']['selected_ids']=[digest((*case_identity(x)[:5],'wrong_road_structure')) for x in work]
    reference['baseline_validation']['actions']={'KEEP':dict(support=320,correct=320,predicted=320)}
    monkeypatch.setattr(trial,'reference',lambda:reference)
    j=job(tmp_path); root=Path(j['output'])/'train';root.mkdir(parents=True)
    records=[]
    for i,x in enumerate(work):
        raw='KEEP' if i else 'unparseable'
        records.append(dict(step=2560,scenario=x.row.scenario,route_id=x.row.route_id,
            frame_id=x.row.frame_id,context_id=x.row.context_id,answers=x.row.answers,
            prompt_spec=prompt_spec_to_json(x.spec),parsed=parse_action_output(raw,spec=x.spec),
            raw_output=raw,all_ok=bool(i),format_valid=bool(i)))
    p=root/'generation_val_cases.jsonl'
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    worker.summarize(j)
    result=trial.read(root.parent/'validation_summary.json')
    assert result['correct']==319 and result['actions']['KEEP']['correct']==319
    assert result['actions']['KEEP']['predicted']==319 and not result['test_used']
    records[-1]['parsed']['INVALID_ACTION_CONTEXT']=True
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='parser'):worker.summarize(j)


def test_verify_actual_native_consumption_and_resume_endpoint(tmp_path):
    j=job(tmp_path);root=Path(j['output'])/'train';root.mkdir(parents=True)
    manifest=dict(total_steps_global=7680,batch_size=1,world_size=4,train_work_global=10240,
        learning_rate_schedule=dict(total_optimizer_updates=7680,warmup_updates=2000))
    trial.write(root/'train_run_manifest.json',manifest)
    counts=dict.fromkeys(train.CONTEXT_IDS,1024);counts['INVALID']=0
    trial.write(root/'balance/consumed_epoch_0001.json',dict(status='verified',actual=counts))
    trial.write(root/'training_progress.json',dict(global_step=2560,planned_epochs=3))
    trial.write(root/'training_states/epoch_0001/receipt.json',{})
    assert trial.verify_training(j)['optimizer_updates']==2560
    manifest['learning_rate_schedule']['total_optimizer_updates']=3840
    trial.write(root/'train_run_manifest.json',manifest)
    with pytest.raises(ValueError,match='7680-update'):trial.verify_training(j)
    manifest['learning_rate_schedule']['total_optimizer_updates']=7680
    trial.write(root/'train_run_manifest.json',manifest)
    j.update(stage='complete',resume='previous/train/training_states/epoch_0001',resume_epoch=1)
    trial.write(root/'final/sft_new_loop_phase3_adapter_config.json',dict(global_step=7680))
    for epoch in (2,3):trial.write(root/f'balance/consumed_epoch_{epoch:04d}.json',dict(status='verified',actual=counts))
    assert trial.verify_training(j)['newly_verified_epochs']==[2,3]
    counts[train.CONTEXT_IDS[0]]-=1
    trial.write(root/'balance/consumed_epoch_0003.json',dict(status='verified',actual=counts))
    with pytest.raises(ValueError,match='consumption'):trial.verify_training(j)


def test_report_packages_no_weights_and_preserves_failure(tmp_path):
    trial.write(tmp_path/'job.json',{'fixture':True})
    trial.write(tmp_path/'train/training_states/epoch_0001/receipt.json',{'private_state':True})
    (tmp_path/'train/training_states/epoch_0001/training_state.pt').write_bytes(b'weights')
    (tmp_path/'train.log').write_text('failure details')
    trial.package(tmp_path,dict(status='failed_or_interrupted'))
    receipt=trial.read(tmp_path/'handoff/phase3_base_package.json')
    with zipfile.ZipFile(tmp_path/'handoff/phase3_base_report.zip') as z:
        assert z.testzip() is None and set(z.namelist())==set(receipt['files'])
        assert not any('training_states' in n for n in z.namelist())
        assert z.read('train.log')==b'failure details'
    result=trial.run_stage([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'exit.log',
        dict(__import__('os').environ),timeout=10,reserve_bytes=0)
    assert result['exit_code']==7


def test_trial_args_native_cpu_checkpoint_resume_binding(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import preflight,training_state
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    from qwen3vl_local.audit_joint.phase3_base_dense.baseline import install_baseline
    for name in ('_contiguous_work','_loss_batch','SEQUENCE_POLICY'):
        monkeypatch.setattr(train,name,getattr(train,name))
    install_baseline()
    index=fixture(tmp_path,'valid');base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    def load(*a,**kw):
        m=torch.nn.Linear(1,1)
        return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['weight'])
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(train,'_load_images',lambda *a,**kw:[])
    monkeypatch.setattr(train,'_build_inputs',lambda *a,**kw:{})
    monkeypatch.setattr(train,'_loss_batch',lambda b,p,s:(b.model(torch.ones(len(p),1)).square().mean(),[{} for _ in p]))
    def run(name,resume=None):
        j=job(tmp_path);j.update(output=str(tmp_path/name),index=str(index),model_dir=str(base),resume=resume)
        args=worker.make_args(j)
        # CPU fixture changes only computational size, not the wrapper binding or resume rules.
        args.device='cpu';args.focus_balance_count=4;args.warmup_steps=20
        args.prompt_variant='baseline';args.eval_steps=args.generation_eval_steps=0
        args.save_final=False;args.stop_after_epoch=1 if resume is None else None
        train.train(args)
        return Path(args.output_dir)
    first=run('first');checkpoint=first/'training_states/epoch_0001'
    resumed=run('resumed',str(checkpoint))
    state=torch.load(resumed/'training_states/epoch_0003/training_state.pt',weights_only=True)
    assert state['global_step']==120 and state['scheduler']['last_epoch']==120
    assert state['binding']['configuration']['base_trial']['policy']==trial.POLICY
    assert state['sampling_history']['next_epoch']==3
    before=trial.file_sha(checkpoint/'receipt.json')
    cfg=copy.deepcopy(state['binding']);cfg['configuration']['base_trial']['reference_sha256']='changed'
    with pytest.raises(ValueError,match='mismatch'):training_state.read(checkpoint,cfg)
    assert trial.file_sha(checkpoint/'receipt.json')==before


def test_preflight_real_sequence_plan_and_frozen_validation_without_model(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    rows,work,ref=fixed_pool(32)
    reference=copy.deepcopy(trial.reference())
    reference['validation']['generation']=ref['validation']['generation']
    reference['validation']['loss']=dict(seed=20261913,per_event=16,
        selected_ids=[digest(case_identity(x)) for x in work if x.row.frame_id<16])
    reference['baseline_training_presentations_per_epoch']={'KEEP':10240}
    monkeypatch.setattr(trial,'reference',lambda:reference)
    monkeypatch.setattr(preflight,'check_index',lambda *a,**kw:{'fixture':True})
    monkeypatch.setattr(preflight,'check_model',lambda *a,**kw:{'fixture':True})
    def read(path,split,**kw):return [replace(r,split=split) for r in rows]
    monkeypatch.setattr(train,'_read_rows',read)
    monkeypatch.setattr(sequence_evaluation,'select_work',sequence_evaluation.select_work)
    monkeypatch.setattr(train,'load_model_with_lora',lambda *a,**kw:pytest.fail('preflight loaded model'))
    j=job(tmp_path)
    worker.preflight(j)
    result=trial.read(Path(j['output'])/'preflight.json')
    assert result['new_action_presentations']=={'KEEP':10240}
    assert result['validation']['loss']['sampled_cases']==160
    assert result['validation']['generation']['sampled_cases']==320
    assert result['sampling']['policy']==trial.profile()['sampling_policy']
    assert not result['old_models_evaluated']


def test_build_job_blocks_wrong_dataset_and_old_resume_before_writes(tmp_path,monkeypatch):
    index=tmp_path/'data/frame_index.jsonl';index.parent.mkdir();index.write_text('{}')
    base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    args=SimpleNamespace(index=index,data_root=index.parent,model_dir=base,resume=None,stage='quick')
    output=tmp_path/'new'
    with pytest.raises(ValueError,match='Index differs'):trial.build_job(args,output)
    assert not output.exists()
    ref=copy.deepcopy(trial.reference());ref['index_sha256']=trial.file_sha(index)
    monkeypatch.setattr(trial,'reference',lambda:ref)
    monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    with pytest.raises(ValueError,match='separate'):trial.build_job(args,index.parent/'new')
    old=tmp_path/'old/train/training_states/epoch_0001';old.mkdir(parents=True)
    trial.write(old/'receipt.json',{'binding':{'configuration':{}}})
    args.resume=old;args.stage='complete'
    with pytest.raises(ValueError,match='older run'):trial.build_job(args,output)
    assert not output.exists()
    trial.write(old/'receipt.json',{'binding':{'configuration':{'base_trial':{'policy':trial.POLICY}}}})
    built=trial.build_job(args,output)
    assert built['resume_epoch']==1
    args.stage='quick'
    with pytest.raises(ValueError,match='stage complete'):trial.build_job(args,output)


def test_new_prefix_release_preserves_legacy_gate_and_training_computation(monkeypatch):
    # Historical manifests must reject changed source; never overwrite their SHA
    # just to make a new optional inference protocol look like the old release.
    old_release=trial.RELEASE.with_name('phase3_base_trial_sources.json')
    with monkeypatch.context() as historical:
        historical.setattr(trial,'RELEASE',old_release)
        with pytest.raises(ValueError,match='Published trial source mismatch'):
            trial.verified_sources()
    # Default image-first execution remains launchable against NEW source;
    # old checkpoints still require their original release hash/source.
    assert len(trial.verified_sources())==64
    new_release=old_release.with_name('phase3_text_prefix_sources_r11_20261031.json')
    spec=trial.read(new_release)
    assert spec['parent_release_sha256']==trial.file_sha(old_release.with_name('phase3_text_prefix_sources_r10_20261031.json'))
    monkeypatch.setenv('AUTOMOT_TEXT_PREFIX','1')
    monkeypatch.setattr(trial,'RELEASE',new_release)
    monkeypatch.setattr(trial,'POLICY','phase3_independent_base_7680_text_prefix_v1')
    assert len(trial.verified_sources())==64
    prior=trial.read(Path(trial.__file__).with_name('release_sources_v88_resume_lifecycle_20261019.json'))
    layout_changes={
        'qwen3vl_local/sft_new_loop_phase3/prompts.py',
        'qwen3vl_local/sft_new_loop_phase3/prompt_contract.py',
        'qwen3vl_local/sft_new_loop_phase3/eval.py',
        'qwen3vl_local/sft_new_loop_phase3/preflight.py',
        'qwen3vl_local/batch_runtime.py',
        'qwen3vl_local/qwen35/adapters.py',  # r11 native metadata guard; backend math unchanged.
    }
    for name,sha in prior['files'].items():
        if name.startswith(('qwen3vl_local/sft_new_loop_phase3/','qwen3vl_local/sft_v2/','qwen3vl_local/qwen35/')) or name in ('qwen3vl_local/sequence_batch.py','qwen3vl_local/batch_runtime.py'):
            if name in layout_changes:
                continue  # New layout/rendering/metrics have dedicated tests.
            if name.endswith(('/train.py','/train.sh','/run_full_pipeline.sh')) and '/sft_new_loop_phase3/' in '/'+name:
                continue  # CLI routing/guard changed; computational functions checked below.
            assert trial.file_sha(trial.APP/name)==sha,name
    import ast, subprocess
    old=subprocess.check_output(['git','show','06cda13d36a91253d9f81bba71299f825723ed9e:AutoMoT/qwen3vl_local/sft_new_loop_phase3/train.py'],text=True)
    functions=lambda text:[ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
        and n.name not in ('_save_adapter','_write_run_metadata')]
    current=(trial.APP/'qwen3vl_local/sft_new_loop_phase3/train.py').read_text()
    policy_binding='    from qwen3vl_local.qwen35.cache_policy import start_training\n    start_training(bundle)\n'
    assert current.count(policy_binding)==1
    # Only the explicit startup metadata binding is new; all training math stays exact.
    assert functions(old)==functions(current.replace(policy_binding,''))
    broken=copy.deepcopy(spec);broken['files']['qwen3vl_local/qwen35/text_prefix.py']='0'*64
    monkeypatch.setattr(trial,'read',lambda path:broken)
    with pytest.raises(ValueError,match='Published trial source mismatch'):
        trial.verified_sources()
