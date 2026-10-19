"""Counterexamples from the dense-route review and matched-control invariants."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import pytest
import torch
from PIL import Image
from qwen3vl_local.audit_joint.phase3_base_dense import data, planning, baseline, shapes, objectives, runtime, worker
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import rows, args, tiny_packed
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.tests.test_batch_runtime import row
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.audit_joint import phase3_base_trial as trial


def settings(mode='stream',**kw):
    return dict(mode=mode,forecast_span_seconds=2,forecast_step_seconds='1',**kw)


def force_action(r,action):
    return replace(r,answers={k:k==action for k in r.answers},action_signature=action if action!='KEEP' else 'NONE')


def test_filtered_pool_cannot_erase_supported_stop(tmp_path):
    source=[force_action(r,'KEEP') for r in rows()]
    stop=next(r for r in source if 'STOP' in train.CONTEXT_BY_ID[r.context_id].action_keys)
    # Sparse STOP has no preceding labels, whereas every event still has KEEP windows.
    source.append(replace(force_action(stop,'STOP'),frame_id=200))
    cfg=args(tmp_path);cfg.focus_balance_count=16
    with pytest.raises(ValueError,match='removed originally supported actions.*STOP'):
        planning.make_plan(data.Pool(source),cfg,settings(),epoch=0)


def test_original_anchor_quotas_and_query_mass_with_matched_control(tmp_path):
    cfg=args(tmp_path);cfg.focus_balance_count=16;pool=data.Pool(rows())
    work,audit,records=planning.make_plan(pool,cfg,settings(),epoch=0)
    original,ref=baseline.independent_work(rows(),cfg,epoch=0,world_size=4)
    assert audit['anchor_action_quotas']==ref['event_action_counts']
    _,control,one=planning.make_plan(pool,cfg,settings(supervision='single'),epoch=0)
    assert audit['ordered_ids']==control['ordered_ids']
    assert audit['input_plan_sha256']==control['input_plan_sha256']
    assert audit['planned_query_loss_mass']==control['planned_query_loss_mass']==audit['anchor_action_quotas']
    assert audit['planned_supervised_queries']==5*len(work)
    assert control['planned_supervised_queries']==len(work)
    mass=Counter()
    for key,value in audit['planned_query_loss_mass'].items():mass[key.split('/')[0]]+=value
    assert mass==dict.fromkeys(train.CONTEXT_IDS,16)
    for identity in records:
        assert objectives.messages(records[identity],list(range(8)))==objectives.messages(one[identity],list(range(8)))
        assert sum(w>0 for w in one[identity]['query_weights'])==1
        assert one[identity]['query_weights'][-1]>0


def test_auxiliary_labels_are_not_new_anchors_and_conflicts_rejected():
    source=rows();anchor=source[0]
    extra=replace(anchor,frame_id=100,history_rgb_paths=[f'/rgb/{i}' for i in range(97,101)])
    pool=data.Pool(source,[extra])
    assert pool.aux_added==1 and pool.at(anchor,100)==extra and pool.anchor_rows==source
    assert extra not in pool.eligible('future','train')[0]
    for invalid in (replace(extra,split='val'),replace(extra,route_id='unknown'),replace(anchor,current_speed_mps=99)):
        with pytest.raises(ValueError,match='route/group|conflicts'):data.Pool(source,[invalid])
    assert data.Pool(source,[anchor]).aux_added==0


def test_future_coverage_every_event_every_position_and_same_control_gate(tmp_path):
    cfg=args(tmp_path);cfg.focus_balance_count=16
    source=[force_action(r,"KEEP") for r in rows()];anchors=[r for r in source if r.frame_id==4]
    aux=[r for r in source if r.frame_id!=4]
    pool=data.Pool(anchors,aux)
    for supervision in ('all','single'):
        _,audit,_=planning.make_plan(pool,cfg,settings('future',supervision=supervision),epoch=0)
        assert set(audit['future_label_coverage'].values())=={1.}
        broken=[r for r in aux if not (r.context_id==anchors[0].context_id and r.frame_id==12)]
        with pytest.raises(ValueError,match='per-event/per-position'):
            planning.make_plan(data.Pool(anchors,broken),cfg,settings('future',supervision=supervision),epoch=0)


def test_rule_horizon_policy_and_configurable_queries():
    opts=planning.options({})
    assert opts['offset_frames']==[0,12,24] and opts['step_source']=='maximum_native_rule_horizon'
    assert not opts['overlapping_native_windows']
    opts=planning.options(dict(forecast_step_seconds='1',forecast_span_seconds=4))
    assert opts['offset_frames']==[0,4,8,12,16] and opts['overlapping_native_windows']
    for opts in ({'forecast_step_seconds':'0'}, {'forecast_step_seconds':'nan'},
                 {'forecast_span_seconds':100},{'min_future_coverage':.01}, {'forecast_span_seconds':2}):
        with pytest.raises(ValueError):planning.options(opts)


@pytest.mark.parametrize('mode',['future','stream','windows'])
def test_exact_header_shape_matches_real_processor_and_total_budget(processor,mode):
    rec=dict(mode=mode,actions=['STOP']*(3 if mode=='future' else 5),offsets_seconds=[0,3,6])
    bundle=SimpleNamespace(processor=processor,tokenizer=processor.tokenizer)
    images=[Image.new('RGB',(1152,384))]*(4 if mode=='future' else 8)
    measured=shapes.measure(processor,rec,[i.size for i in images])
    # Byte tokenizer fixture needs a larger total cap than the released tokenizer.
    p=objectives.pack(bundle,rec,images,max_batch_tokens=32768)
    lengths=[x['tokens'] for x in p] if isinstance(p,list) else [p['tokens']]
    assert measured['sequence_tokens']==lengths
    assert measured['padded_batch_tokens']==len(lengths)*max(lengths)
    if mode=='windows':
        assert measured['padded_batch_tokens']>8192
        with pytest.raises(ValueError,match='total padded'):
            objectives.pack(bundle,rec,images,max_batch_tokens=8192)


def test_full_scan_selects_largest_later_candidate(tmp_path,processor):
    path=tmp_path/'image.png';Image.new('RGB',(64,64)).save(path)
    base=dict(mode='future',images=[str(path)]*4,actions=['KEEP'],offsets_seconds=[0])
    records={'first':dict(base,premises=['short']),'later':dict(base,premises=['much longer '*50])}
    audit=shapes.scan(processor,records,16384)
    assert audit['unique_planned_inputs']==2 and audit['probe_ids']==['later']


def test_single_query_masked_targets_have_zero_gradient(model):
    p=tiny_packed(row(4),[1,8,20],[20,21,22]);p['query_weights']=[1.,0.,0.]
    actual,report=objectives.readout_loss(model,p,'cpu')
    logits=objectives.readout_logits(model,p,'cpu')
    expected=torch.nn.functional.cross_entropy(logits[:1],torch.tensor([20]))
    torch.testing.assert_close(actual,expected)
    assert report['supervised']==1
    changed=dict(p,targets=[20,25,-100])
    torch.testing.assert_close(actual,objectives.readout_loss(model,changed,'cpu')[0])


def test_resume_plans_only_remaining_epochs(tmp_path):
    source=[force_action(r,"KEEP") for r in rows()];anchors=[r for r in source if r.frame_id==4]
    pool=data.Pool(anchors,[r for r in source if r.frame_id!=4])
    cfg=args(tmp_path)
    plans,_=worker.planned_epochs(settings('future',resume_epoch=1),pool,cfg)
    assert list(plans)==['1','2']
    assert all(p['phase']=='plan_not_consumption' for p in plans.values())
    assert plans['1']['epoch']==1 and plans['2']['epoch']==2


@pytest.mark.parametrize('entry',['sft_new_loop_phase3/run_full_pipeline.sh','sft_new_loop_phase3/train.sh','audit_joint/train_phase3_batch.sh'])
def test_old_shell_defaults_route_new_base_before_legacy_work(tmp_path,entry):
    fake=tmp_path/'python';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
    env=dict(os.environ,PYTHON=str(fake),INDEX='/explicit/index',OUTPUT_DIR='/fresh/output')
    env.pop('LEGACY_PHASE3_SEQUENCE',None)
    result=subprocess.run(['bash',str(trial.APP/'qwen3vl_local'/entry)],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'phase3_base_trial' in result.stdout and '/explicit/index' in result.stdout
    assert 'sequence_trial' not in result.stdout
    env['BATCH_SIZE']='2'
    result=subprocess.run(['bash',str(trial.APP/'qwen3vl_local'/entry)],env=env,text=True,capture_output=True)
    assert result.returncode==2 and 'cannot silently carry over' in result.stderr


def test_auxiliary_binding_tamper_rejected_before_model(tmp_path,monkeypatch):
    monkeypatch.setattr(worker.base,'validate_job',lambda j:None)
    index=tmp_path/'extra.jsonl';index.write_text('{}\n')
    manifest=index.with_name('manifest.json');manifest.write_text('{}')
    init=tmp_path/'adapter';init.mkdir()
    j=settings('future',aux_index=str(index),aux_binding=dict(index_sha256=trial.file_sha(index),manifest_sha256=trial.file_sha(manifest)),init_adapter=str(init),init_adapter_files={})
    j['forecast_contract']=planning.options(j)
    worker.validate(j)
    index.write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='Auxiliary labels/manifest changed'):worker.validate(j)


def _consume_rank(rank,root,corrupt):
    import torch.distributed as dist
    from qwen3vl_local.audit_joint.tests.test_phase3_sequence_trial import job
    os.environ['RANK']=str(rank)
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),rank=rank,world_size=4)
    try:
        source=[force_action(r,"KEEP") for r in rows()];anchors=[r for r in source if r.frame_id==4]
        pool=data.Pool(anchors,[r for r in source if r.frame_id!=4])
        j=job(Path(root));j.update(settings('future'),init_adapter='fixture',init_adapter_files={})
        runtime.install(j,pool)
        cfg=runtime.make_args(j);cfg.focus_balance_count=4
        work,audit=train._contiguous_work(anchors,cfg,epoch=0,world_size=4)
        proc=processor.__wrapped__()
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(320))
            def forward(self,input_ids,logits_to_keep,**kw):
                return SimpleNamespace(logits=self.values[None,None].expand(1,len(logits_to_keep),-1))
        m=torch.nn.parallel.DistributedDataParallel(Tiny())
        bundle=SimpleNamespace(model=m,processor=proc,tokenizer=proc.tokenizer,device='cpu')
        train._load_images=lambda paths:[Image.new('RGB',(32,32)) for _ in paths]
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        consumed=Counter()
        for pos in range(rank,len(work),4):
            item=work[pos]
            packed=train._build_inputs(bundle,spec=item.spec,max_length=8192)
            loss,_=train._loss_batch(bundle,[packed],[item.spec]);loss.backward()
            train._append_jsonl(Path(root)/f'epoch_0001_rank{rank}_batches.jsonl',
                dict(ids=[audit['ordered_ids'][pos]],positions=[pos+4 if corrupt and rank==0 else pos]))
            opt.step();opt.zero_grad(set_to_none=True);consumed[item.row.context_id]+=1
        if corrupt:
            try:train.verify_epoch_consumption(work,consumed,device='cpu',world_size=4)
            except ValueError as exc:
                assert 'Actual post-backward' in str(exc)
            else:raise AssertionError('Corrupt actual positions passed')
        else:train.verify_epoch_consumption(work,consumed,device='cpu',world_size=4)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('corrupt',[False,True])
def test_four_cpu_ranks_actual_consumption_and_corruption(tmp_path,corrupt):
    torch.multiprocessing.start_processes(_consume_rank,args=(str(tmp_path),corrupt),nprocs=4,join=True,start_method='spawn')
    report=tmp_path/'run/train/balance/query_consumed_epoch_0001.json'
    if corrupt:assert not report.exists()
    else:
        result=json.loads(report.read_text())
        assert result['ranks']==4 and result['anchor_presentations']==40
        assert result['supervised_queries']==120
        assert sum(result['actual_query_loss_mass'].values())==pytest.approx(40)


def test_new_aux_index_native_admission_without_replacing_frozen_base(tmp_path,monkeypatch,processor):
    from qwen3vl_local.sft_new_loop_phase3.test_entry_admission import fixture
    base_dir=tmp_path/'original';base_dir.mkdir();index=fixture(base_dir,'valid')
    aux_dir=tmp_path/'aux';aux_dir.mkdir();aux=fixture(aux_dir,'valid')
    original=[json.loads(line) for line in aux.read_text().splitlines()]
    extra=[dict(r,frame_id=r['frame_id']+offset) for r in original
           if r['split']=='train' and not r['invalid_action_context'] for offset in (12,24)]
    aux.write_text(''.join(json.dumps(r)+'\n' for r in original+extra))
    model_dir=tmp_path/'model';model_dir.mkdir();(model_dir/'config.json').write_text('{}')
    init=tmp_path/'final';init.mkdir()
    trial.write(init/'sft_new_loop_phase3_adapter_config.json',dict(sampling_policy=baseline.POLICY,global_step=7680))
    ref=dict(trial.reference(),index_sha256=trial.file_sha(index))
    monkeypatch.setattr(trial,'reference',lambda:ref)
    monkeypatch.setattr(trial,'verified_sources',lambda:'fixture')
    cfg=SimpleNamespace(index=index,data_root=base_dir,model_dir=model_dir,resume=None,stage='quick',
        experiment='future',aux_index=aux,init_adapter=init)
    j=trial.build_job(cfg,tmp_path/'out')
    assert j['index_sha256']!=j['aux_binding']['index_sha256']
    worker.validate(j)
    # The former fixture merely shifted frame IDs. It must now fail without raw provenance.
    with pytest.raises((FileNotFoundError,ValueError)):
        runtime.load_pool(j,recompute_aux=True)


def test_dense_final_verifier_requires_actual_query_receipt(tmp_path):
    from qwen3vl_local.audit_joint.tests.test_phase3_base_trial import job
    j=job(tmp_path);j['mode']='future';root=Path(j['output'])/'train'
    trial.write(root/'train_run_manifest.json',dict(total_steps_global=7680,batch_size=1,world_size=4,
        train_work_global=10240,learning_rate_schedule=dict(total_optimizer_updates=7680,warmup_updates=2000)))
    trial.write(root/'balance/consumed_epoch_0001.json',dict(status='verified',actual=dict(dict.fromkeys(train.CONTEXT_IDS,1024),INVALID=0)))
    trial.write(root/'training_progress.json',dict(global_step=2560,planned_epochs=3))
    trial.write(root/'training_states/epoch_0001/receipt.json',{})
    plan=dict(planned_query_exposure={'event/0/KEEP':10240},planned_query_loss_mass={'event/KEEP':10240})
    trial.write(Path(j['output'])/'dense_coverage.json',dict(epoch_plans={'0':plan}))
    with pytest.raises(FileNotFoundError):trial.verify_training(j)
    receipt=dict(status='verified_after_backward',plan_sha256=baseline.digest(plan),
        actual_query_exposure=plan['planned_query_exposure'],actual_query_loss_mass=plan['planned_query_loss_mass'],
        anchor_presentations=10240,ranks=4)
    path=root/'balance/query_consumed_epoch_0001.json';trial.write(path,receipt)
    assert trial.verify_training(j)['newly_verified_epochs']==[1]
    receipt['actual_query_exposure']={'event/0/KEEP':10239};trial.write(path,receipt)
    with pytest.raises(ValueError,match='dense query consumption'):trial.verify_training(j)


def test_prepare_only_does_not_train_and_preserves_packaging_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['trial','--index','unused','--output',str(tmp_path/'new'),'--prepare-only'])
    monkeypatch.setattr(trial,'build_job',lambda a,o:dict(mode='future',resume=None,output=str(o),stage='quick'))
    monkeypatch.setattr(trial.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(trial,'gpu_ids',lambda *a:pytest.fail('CPU prepare selected GPUs'))
    commands=[]
    def stage(cmd,*a,**kw):
        commands.append(cmd)
        return dict(exit_code=0)
    monkeypatch.setattr(trial,'run_stage',stage)
    monkeypatch.setattr(trial,'verify_training',lambda j:pytest.fail('CPU prepare tried to verify training'))
    def broken_package(o,s):
        assert s['status']=='prepared_cpu_no_training'
        raise ValueError('package failure')
    monkeypatch.setattr(trial,'package',broken_package)
    assert trial.main()==2
    assert len(commands)==1 and commands[0][-1]=='preflight'
