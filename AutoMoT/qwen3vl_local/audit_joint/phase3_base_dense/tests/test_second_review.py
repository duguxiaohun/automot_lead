"""No-training intent, matched action mass, raw-frame provenance and fullflow eval admission."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import pickle
import subprocess
import pytest
from PIL import Image
from qwen3vl_local.audit_joint import phase3_base_trial as trial
from qwen3vl_local.audit_joint.phase3_base_dense import data, planning, runtime, evaluation, provenance, shapes, worker, baseline
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import rows, args
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.sft_new_loop_phase3 import train, trajectory_action as rule, build_dataset as builder
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity


@pytest.mark.parametrize('entry',['run_full_pipeline.sh','train.sh','../audit_joint/train_phase3_batch.sh'])
@pytest.mark.parametrize('flag',['SKIP_TRAIN','PIPELINE_CHECK_ONLY','both'])
@pytest.mark.parametrize('cli',[False,True])
def test_check_switch_never_becomes_training(tmp_path,entry,flag,cli):
    fake=tmp_path/'python';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
    env=dict(os.environ,PYTHON=str(fake),INDEX='explicit',OUTPUT_DIR='fresh',SKIP_TRAIN='0',PIPELINE_CHECK_ONLY='0')
    env.pop('LEGACY_PHASE3_SEQUENCE',None)
    for key in ('SKIP_TRAIN','PIPELINE_CHECK_ONLY'):
        if flag in (key,'both'):env[key]='1'
    command=['bash',str(trial.APP/'qwen3vl_local/sft_new_loop_phase3'/entry)]
    if cli:command+=['--index','explicit','--output','fresh','--stage','complete']
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    assert r.returncode==0,r.stderr
    assert '--prepare-only' in r.stdout


def test_check_mode_and_ambiguous_boolean(tmp_path):
    fake=tmp_path/'python';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
    env=dict(os.environ,PYTHON=str(fake),INDEX='explicit',OUTPUT_DIR='fresh')
    command=['bash',str(trial.APP/'qwen3vl_local/audit_joint/phase3_default.sh'),'check']
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--prepare-only' in r.stdout
    env['SKIP_TRAIN']='yes'
    r=subprocess.run(command,env=env,text=True,capture_output=True)
    assert r.returncode==2 and 'ambiguous' in r.stderr


@pytest.mark.parametrize('mode',['future','stream','windows'])
def test_every_action_mass_identical_between_arms_all_epochs(tmp_path,mode):
    source=rows();anchors=source
    extra=[replace(r,frame_id=r.frame_id+offset) for r in source for offset in (100,200)] if mode=='future' else []
    pool=data.Pool(anchors,extra,offsets=(0,100,200))
    cfg=args(tmp_path);cfg.focus_balance_count=1024
    j=dict(mode=mode,forecast_span_seconds=50 if mode=='future' else 2,forecast_step_seconds='25' if mode=='future' else '1')
    for epoch in range(3):
        _,all_plan,_=planning.make_plan(pool,cfg,j,epoch=epoch)
        _,single,_=planning.make_plan(pool,cfg,dict(j,supervision='single'),epoch=epoch)
        assert all_plan['input_plan_sha256']==single['input_plan_sha256']
        assert all_plan['planned_query_loss_mass']==single['planned_query_loss_mass']==all_plan['anchor_action_quotas']


def frozen_pool(monkeypatch,mode='future',missing=False):
    source=rows();original=[];extra=[]
    for split in ('train','val','test'):
        for r in source:
            new=replace(r,split=split,route_id=r.route_id+'_'+split)
            (original if r.frame_id==8 else extra).append(new)
    pool=data.Pool(original,extra,offsets=(0,4,8))
    ref=dict(trial.reference());ref['validation']=dict(ref['validation'])
    for split in ('val','test'):
        ids=[baseline.digest(case_identity(train._make_item(r,seed=20260904,action_output_mode='choice'))) for r in original if r.split==split]
        plan=dict(seed=20260904,per_event=1,selected_ids=ids)
        if split=='test':ref['test_plan']=plan
        else:
            ref['validation']['generation']=plan
            ref['validation']['loss']=dict(plan,seed=20261913)
    monkeypatch.setattr(trial,'reference',lambda:ref)
    if missing:
        # Remove all future/window support for ONE held-out event, leave train intact.
        extra=[r for r in extra if not (r.split=='val' and r.context_id==train.CONTEXT_IDS[0])]
        pool=data.Pool(original,extra,offsets=(0,4,8))
    return pool


@pytest.mark.parametrize('mode',['future','stream','windows'])
def test_frozen_evaluation_targets_and_coverage_gate(monkeypatch,mode):
    pool=frozen_pool(monkeypatch);job=dict(mode=mode,forecast_span_seconds=2,forecast_step_seconds='1')
    report,records=evaluation.build(job,pool)
    assert report['val']['primary_query']==(0 if mode=='future' else 4)
    assert report['val']['coverage'] and set(report['val']['coverage'].values())=={1.}
    assert len(report['val']['ordered_anchor_ids'])==10
    assert len(records)==50  # val/test transfer + matched + fixed native loss
    broken=frozen_pool(monkeypatch,missing=True)
    with pytest.raises(ValueError,match='val fixed-anchor evaluation coverage'):
        evaluation.build(job,broken)


def test_stream_primary_scores_last_query_not_four_frame_transfer():
    cases=[dict(targets=['KEEP']*5,predictions=[dict(action='KEEP',constrained_action='KEEP')]*4+
                [dict(action='STOP',constrained_action='STOP')])]
    assert evaluation.score_matched(cases,4)['correct']==0
    assert evaluation.score_matched(cases,0)['correct']==1


def raw_fixture(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
    base_dir=tmp_path/'base';base_dir.mkdir();index=fixture(base_dir,'valid')
    aux_dir=tmp_path/'aux';aux_dir.mkdir();aux=fixture(aux_dir,'valid')
    original=[json.loads(x) for x in index.read_text().splitlines()];extra=[]
    for row in original:
        if row['invalid_action_context']:continue
        run=base_dir/row['scenario']/row['route_id'];(run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
        for frame in range(50):
            meta=dict(speed=3.+frame*.01,brake=False,throttle=.2,lane_type_str='Driving',
                road_id=1,lane_id=-1,section_id=0,pos_global=[frame,0],next_target_points=[[frame+10,0]],theta=0.)
            (run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(meta))
            Image.new('RGB',(32,32),(frame,20,30)).save(run/'rgb'/f'{frame:04d}.jpg')
        trajectory=rule.load_route_trajectory(run)
        for offset in (12,24):
            frame=row['frame_id']+offset;sig=trajectory.signals(frame);labels=rule.label_actions(sig)
            history=builder._relative_history_paths(builder._history(run,frame),base_dir)
            answers=builder._answers_for(row['context_id'],labels,invalid=False);evidence=rule.action_evidence(sig)
            fresh=dict(row,frame_id=frame,current_speed_mps=evidence['speed_mps'],answers=answers,
                goal_ego_xy=[round(sig['goal_x'],3),round(sig['goal_y'],3)],action_evidence=evidence,
                action_signature=builder.action_signature(labels,context_id=row['context_id']),
                history_rgb_paths=history,latest_rgb_path=history[-1])
            fresh.update(choice_annotation(answers,row['context_id'],evidence));extra.append(fresh)
    aux.write_text(''.join(json.dumps(r)+'\n' for r in original+extra))
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    init=tmp_path/'final';init.mkdir();trial.write(init/'sft_new_loop_phase3_adapter_config.json',dict(sampling_policy=baseline.POLICY,global_step=7680))
    ref=dict(trial.reference(),index_sha256=trial.file_sha(index));ref['validation']=dict(ref['validation'])
    for split in ('val','test'):
        rr=train._read_rows(index,split,data_root=base_dir)
        ids=[baseline.digest(case_identity(train._make_item(r,seed=20260904,action_output_mode='choice'))) for r in rr if not r.invalid_source]
        plan=dict(seed=20260904,per_event=1,selected_ids=ids)
        if split=='test':ref['test_plan']=plan
        else:ref['validation'].update(generation=plan,loss=dict(plan,seed=20261913))
    monkeypatch.setattr(trial,'reference',lambda:ref);monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    cfg=SimpleNamespace(index=index,data_root=base_dir,model_dir=model,resume=None,stage='quick',experiment='future',aux_index=aux,init_adapter=init)
    job=trial.build_job(cfg,tmp_path/'out')
    Image.new('RGB',(32,32)).save(base_dir/'unused.jpg')
    return job,original,extra


def test_real_raw_meta_recomputation_all_splits_and_source_change(tmp_path,monkeypatch,processor):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    pool=runtime.load_pool(j,recompute_aux=True)
    assert pool.aux_added==60 and len(pool.anchor_rows)==len(original)
    proof=provenance.verify_cached(j)
    assert proof['rows_by_split']==dict(train=20,val=20,test=20)
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    from qwen3vl_local.qwen35.backend import LocalProcessor
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(LocalProcessor,'from_pretrained',lambda *a,**kw:processor)
    worker.preflight(j,pool)
    audit=trial.read(Path(j['output'])/'dense_input_shapes.json')
    assert sum(audit['input_usage_counts'].values())==60  # 10 train + 50 evaluation protocols
    assert audit['input_usage_counts']['evaluation/future']==50
    plans=trial.read(Path(j['output'])/'dense_evaluation_plan.json')
    assert plans['val']['matched_anchor_ids']==plans['val']['ordered_anchor_ids']
    # Same indices and proof cannot hide changed raw source content/stat.
    source=Path(j['data_root'])/extra[0]['scenario']/extra[0]['route_id']/'metas/0012.pkl'
    source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Raw auxiliary source changed'):runtime.load_pool(j)


@pytest.mark.parametrize('field',['action_evidence','history_rgb_paths'])
def test_shifted_copy_fails_even_with_existing_raw_future_frames(tmp_path,monkeypatch,field):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    copied=dict(extra[0]);copied[field]=original[0][field]
    content=original+[copied]+extra[1:]
    Path(j['aux_index']).write_text(''.join(json.dumps(r)+'\n' for r in content))
    j['aux_binding']['index_sha256']=trial.file_sha(j['aux_index'])
    # Native metadata/schema validation is insufficient, raw recomputation must reject it.
    with pytest.raises(ValueError,match='raw-frame recomputation mismatch|RGB history'):
        runtime.load_pool(j,recompute_aux=True)


def test_invalid_original_identity_does_not_exempt_new_valid_target(tmp_path,monkeypatch):
    j,original,extra=raw_fixture(tmp_path,monkeypatch)
    # A rejected original row is not evidence for a new valid label at that identity.
    original.append(dict(extra[0],invalid_action_context=True))
    Path(j['index']).write_text(''.join(json.dumps(r)+'\n' for r in original))
    j['index_sha256']=trial.file_sha(j['index'])
    summary=provenance.recompute(j)
    assert summary['rows_by_split']==dict(train=20,val=20,test=20)
    trial.package(Path(j['output']),dict(status='prepared'))
    package=trial.read(Path(j['output'])/'handoff/phase3_base_package.json')
    assert 'dense_aux_summary.json' in package['files']
    assert 'dense_aux_provenance.json' not in package['files']


def test_evaluation_shape_is_admitted_and_larger_eval_rejected(tmp_path,monkeypatch,processor):
    pool=frozen_pool(monkeypatch);job=dict(mode='stream',forecast_span_seconds=2,forecast_step_seconds='1')
    _,records=evaluation.build(job,pool)
    image=tmp_path/'image.jpg';Image.new('RGB',(32,32)).save(image)
    for rec in records.values():rec['images']=[str(image)]*len(rec['images'])
    key=next(k for k in records if '/test/transfer/' in k)
    records[key]['premises']=['long description '*2000]
    with pytest.raises(ValueError,match='per-sequence/total padded token budget'):
        shapes.scan(processor,records,16384)


def test_primary_summary_uses_matched_stream_readout(tmp_path,monkeypatch,processor):
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import training_state
    from qwen3vl_local.audit_joint.phase3_base_dense import objectives
    from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
    pool=frozen_pool(monkeypatch)
    j=job(tmp_path);j.update(mode='stream',supervision='single',forecast_span_seconds=2,
        forecast_step_seconds='1',init_adapter='fixture',init_adapter_files={})
    plan,records=evaluation.build(j,pool)
    trial.write(Path(j['output'])/'dense_evaluation_plan.json',plan)
    input_shapes={key:shapes.measure(processor,rec,[(32,32)]*len(rec['images'])) for key,rec in records.items()}
    trial.write(Path(j['output'])/'dense_input_shapes.json',dict(shapes=input_shapes))
    binding=dict(configuration=dict(dense_contract=runtime.contract(j)))
    trial.write(Path(j['output'])/'train/training_states/epoch_0001/receipt.json',dict(files={},binding=binding))
    m=torch.nn.Linear(1,1);bundle=SimpleNamespace(model=m,unwrap=lambda:m,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    monkeypatch.setattr(runtime,'create_bundle',lambda *a:bundle)
    monkeypatch.setattr(runtime,'load_pool',lambda *a:pool)
    monkeypatch.setattr(training_state,'files',lambda p:{})
    monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(binding=binding,trainable={}))
    monkeypatch.setattr(train,'_load_images',lambda paths:[Image.new('RGB',(32,32)) for _ in paths])
    monkeypatch.setattr(objectives,'readout_loss',lambda *a:(torch.tensor(0.),dict(correct=1,constrained_correct=1)))
    def predictions(bundle,record,images,**kw):
        return [dict(action=a if i<4 else None,constrained_action=a if i<4 else None)
                for i,a in enumerate(record['actions'])]
    monkeypatch.setattr(objectives,'predict',predictions)
    runtime.evaluate(j,'val')
    primary=trial.read(Path(j['output'])/'dense_val_summary.json')
    transfer=trial.read(Path(j['output'])/'dense_val_transfer_summary.json')
    assert primary['query']==4 and primary['correct']==0 and primary['total']==10
    assert primary['metric_role']=='primary_matched_training_readout'
    assert transfer['correct']==transfer['total']==10
    assert transfer['metric_role']=='secondary_four_frame_transfer'


def test_holdout_probe_has_no_loss_backward_or_optimizer(model):
    import torch
    from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import tiny_packed
    from qwen3vl_local.tests.test_batch_runtime import row
    packed=tiny_packed(row(4),[10],[20])
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    def forbidden(*a,**kw):pytest.fail('holdout probe performed an optimizer update')
    bundle=SimpleNamespace(model=model,device='cpu')
    report=worker.probe_record(bundle,dict(usage='evaluation'),packed,SimpleNamespace(step=forbidden,zero_grad=forbidden))
    assert report==[dict(tokens=packed['tokens'],phase='evaluation_forward_only',
                        vision_policy='no_reuse_nonwindow',visual_reuse=None)]
    for name,param in model.named_parameters():
        torch.testing.assert_close(param,before[name],rtol=0,atol=0)
        assert param.grad is None


def test_zero_weight_future_labels_do_not_count_as_dense_supervision(tmp_path):
    source=rows();anchors=[r for r in source if r.frame_id==4]
    extra=[replace(r,answers={k:False for k in r.answers},action_signature='NONE') if r.frame_id==12 else r for r in source if r.frame_id!=4]
    pool=data.Pool(anchors,extra)
    cfg=args(tmp_path);cfg.focus_balance_count=16
    for supervision in ('single','all'):
        with pytest.raises(ValueError,match='effective query coverage'):
            planning.make_plan(pool,cfg,dict(mode='future',supervision=supervision,forecast_span_seconds=2,
                forecast_step_seconds='1'),epoch=0)
