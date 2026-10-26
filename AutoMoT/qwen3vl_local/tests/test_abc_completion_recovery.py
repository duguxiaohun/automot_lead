import ast
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
from qwen3vl_local import abc_completion_recovery as recovery
from qwen3vl_local.audit_joint.abc_delivery import sha,write
from qwen3vl_local.action_expert_ablation.geometry_abc import contracts,worker,preflight


def fixture(tmp_path):
    run=tmp_path/'a_bev_only';run.mkdir()
    args=worker.parser().parse_args(['--abc-variant','a','--branch','bev_only','--suite-root',str(tmp_path)])
    config=vars(args);manifest=dict(identity='fixture',splits={'train':'T','val':'V','test':'E'})
    plans=[]
    for epoch in range(1,8):
        ids=[event for event,n in contracts.QUOTAS.items() for _ in range(n)]
        plans.append(dict(ordered_ids=ids,samples=116256,optimizer_steps=1817,quotas=contracts.QUOTAS,physical_repeat_histogram={'1':116256}))
        counts=Counter(ids);values={scope+k:v for scope in ('exposure/','verified_exposure/') for k,v in counts.items()}
        write(run/'epoch_audit'/f'epoch_{epoch:03d}_step{epoch*1817:08d}.json',values)
    projection=dict(plans=plans,total_updates=12719,total_presentations=813792,cumulative_physical_histogram={'7':116256})
    projection['sha256']=preflight.plan_digest(projection)
    gate=dict(code=contracts.source_identity(),dataset='fixture',splits=manifest['splits'],resources={'fixture':True},plan_sha256=projection['sha256'],manifest_sha256='M')
    binding=dict(source=gate['code'],dataset='fixture',splits=manifest['splits'],resources=gate['resources'],plan=projection['sha256'],
        representation=contracts.contract('a'),branch='bev_only',consumption_sources=dict(manifest_sha256='M',protocol='same_hashed_bytes_decode_v1'),numerical_config=worker.numerical_config(args))
    write(run/'config.json',config);write(run/'condition_contract.json',binding);write(run/'training_plan.json',dict(actual_step_limit=12719))
    score=args.route_loss_weight*.2+args.waypoint_loss_weight*.3
    metrics=dict(route_ade_m=.2,waypoint_ade_m=.3,samples=320)
    state=dict(schema='abc_trajectory_checkpoint_v1',condition_contract=binding,dataset_hashes=manifest['splits'],world_size=4,rng_by_rank=[{}]*4,
        args=config,step=12719,cursor=dict(epoch=7,micro=0),best_val=score,decoder={'weight':torch.ones(2)},optimizer={})
    for name in ('best','latest'):torch.save(state,run/(name+'.pt'))
    write(run/'validation/final.json',dict(optimizer_step=12719,checkpoint='latest.pt',weight_view='ema',split='val',metrics=metrics,best_selection_score=score))
    write(run/'validation/epoch_007_step00012719.json',dict(metrics,optimizer_step=12719,validation_full_epoch=True))
    write(run/'training_audit/latest.json',dict(optimizer_step=12719,validation_pending=False,validation=metrics))
    for rank in range(4):
        write(run/'token_diagnostics'/f'step_00012719_rank{rank}.json',dict(rank=rank,optimizer_steps=12719,
            slot_exposure={'samples':203448,'future/0/0':813792,'history/0/0':813792},gradient_norms=[1.]*10))
    return run,manifest,projection,gate


def test_completion_native_verifier_no_optimizer_or_checkpoint_rewrite(tmp_path):
    run,m,p,g=fixture(tmp_path);before=recovery.evidence_inventory(run)
    first=recovery.validate_completed(run,m,p,g,publish=False)
    assert first['validated'] and not (run/'training_complete.json').exists()
    result=recovery.validate_completed(run,m,p,g,publish=True)
    assert result['optimizer_updates_added']==0 and (run/'training_complete.json').is_file()
    assert recovery.evidence_inventory(run)==before
    assert recovery.validate_completed(run,m,p,g,publish=True)==result


@pytest.mark.parametrize('bad',['step','pending','cursor','source','best','exposure','gradient','final','selection','config'])
def test_completion_rejects_incomplete_or_changed_evidence_before_publish(tmp_path,bad):
    run,m,p,g=fixture(tmp_path)
    if bad in ('step','pending','cursor','best','source','config'):
        path=run/('best.pt' if bad=='best' else 'latest.pt');state=torch.load(path,weights_only=False)
        if bad in ('step','best'):state['step']=8500
        elif bad=='pending':state['cursor']['validation_pending']=True
        elif bad=='cursor':state['cursor']['micro']=1
        elif bad=='source':state['condition_contract']['source']={}
        else:state['args']['learning_rate']=999
        torch.save(state,path)
    else:
        path={'exposure':run/'epoch_audit/epoch_001_step00001817.json',
              'gradient':run/'token_diagnostics/step_00012719_rank3.json',
              'final':run/'validation/final.json','selection':run/'validation/epoch_007_step00012719.json'}[bad]
        data=json.loads(path.read_text())
        if bad=='exposure':data['exposure/LEAD_BRAKE']-=1
        elif bad=='gradient':data['optimizer_steps']=12718
        elif bad=='final':data['optimizer_step']=8500
        else:data['waypoint_ade_m']=.1
        write(path,data)
    with pytest.raises((ValueError,KeyError)):recovery.validate_completed(run,m,p,g,publish=True)
    assert not (run/'training_complete.json').exists()


def distributed_child(rank,path,mode):
    from qwen3vl_local.action_prior import training_core as core
    from qwen3vl_local.action_prior.tests.test_training_loop import lightweight_old_helpers
    from qwen3vl_local.action_expert_ablation.geometry_abc.tests.test_core import tiny,record,inputs
    from qwen3vl_local.action_expert_ablation.geometry_abc import runtime
    from qwen3vl_local.action_prior.config import DEFAULTS
    from qwen3vl_local.action_prior import event_balance
    torch.set_num_threads(1);torch.manual_seed(13)
    out=Path(path);dist.init_process_group('gloo',init_method='file://'+str(out/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(seconds=30))
    try:
        old=lightweight_old_helpers();model=tiny();ddp=torch.nn.parallel.DistributedDataParallel(model)
        args=SimpleNamespace(**dict(DEFAULTS,num_epochs=1,grad_accum_steps=1,num_workers=0,high_level_action_token=False,
            action_token_separation_weight=0.,sampling_mode='abc_regions',decoder_dtype='float32',save_steps=1,
            logging_steps=1,val_steps=100,output_dir=str(out),evaluation_distribution='natural'))
        records=[record(i) for i in range(8)];rows=[runtime.sample(r,'/unused') for r in records]
        for row in rows:row['consumed_sources']={k:'0'*64 for k in ('metas','rgb','lidar')}
        event_balance.build_balanced_epoch=lambda *a,**kw:(rows,{})
        plan=dict(samples_per_epoch=8,optimizer_steps_per_epoch=2,actual_step_limit=2,per_event=8)
        opt,scheduler,ema=core.make_optimization(args,model,plan,old)
        class Runtime:
            device=torch.device('cpu');variant='a';branch='bev_only'
            def forward_sample(self,sample,decoder,config,dtype,clip=None,**kw):return decoder(**inputs(model,sample['abc_record']),**kw)
        writer=SimpleNamespace(add_scalar=lambda *a:None,flush=lambda:None,close=lambda:None) if rank==0 else None
        def save(path,cursor,step,best):
            core.save_training_checkpoint(path,model,opt,scheduler,ema,args,{}, {},cursor,step,best,rank,4,
                schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
        kw=dict(args=args,rows={'train':rows,'val':rows[:1]},plan=plan,runtime=Runtime(),model=model,decoder=ddp,
            config=model.config,flow_config=model.flow_config,optimizer=opt,scheduler=scheduler,ema=ema,
            cursor={'epoch':0,'micro':0},step=0,best=float('inf'),device=torch.device('cpu'),dtype=torch.float32,
            rank=rank,world=4,out=out,writer=writer,old=old,hooks=worker.hooks(),
            evaluate_fn=lambda *a,**kw:dict(loss=1.,route_ade_m=1.,waypoint_ade_m=1.,samples=1),checkpoint=save)
        # Helpers used by the test loader don't shard; emulate the native rank
        # loader exactly so the real loop performs two updates per rank.
        torch.cuda.reset_peak_memory_stats=lambda *a:None
        torch.cuda.max_memory_allocated=lambda *a:0
        if mode=='bridge':recovery.pinned_loop(core,**kw)
        else:core.run_training_loop(**kw,destroy_process_group=(mode=='old_default'))
        if mode=='old_default':
            assert not dist.is_initialized()
            with pytest.raises(ValueError,match='process group'):dist.all_gather_object([None]*4,rank)
        else:
            assert dist.is_initialized()
            values=[None]*4;dist.all_gather_object(values,rank);assert values==list(range(4))
        (out/f'passed_rank{rank}.json').write_text(json.dumps(dict(mode=mode,post_loop_collective=mode!='old_default')))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


@pytest.mark.parametrize('mode',['new','bridge','old_default'])
def test_real_four_rank_training_final_validation_save_and_outer_collective(tmp_path,mode):
    import torch.multiprocessing as mp
    mp.spawn(distributed_child,args=(str(tmp_path),mode),nprocs=4,join=True)
    assert len(list(tmp_path.glob('passed_rank*.json')))==4
    assert torch.load(tmp_path/'latest.pt',weights_only=False)['step']==2


def test_command_preserves_six_groups_and_current_server_paths(tmp_path):
    from qwen3vl_local.action_expert_ablation.geometry_abc.audit import GROUPS
    for base in ('home/cruser1/lda','mnt/workspace/cee3sgh'):
        root=tmp_path/base/'AutoMoT/checkpoints/run';config={k:str(tmp_path/base/'AutoMoT'/k) for k in ('data_dir','data_root','model_dir','lead_bev_ckpt')}
        for group in GROUPS:
            cmd=recovery.worker_command(root,config,group,'evaluate',root/'completion_recovery/runner.py')
            assert cmd[cmd.index('--checkpoint')+1]==str(root/group/'best.pt')
            assert cmd[cmd.index('--branch')+1]==group.split('_',1)[1]


@pytest.mark.parametrize('incomplete',[False,True])
def test_suite_preserves_a_and_only_trains_missing_groups(tmp_path,monkeypatch,incomplete):
    from qwen3vl_local.action_expert_ablation.geometry_abc import audit
    from qwen3vl_local.action_prior import host_memory
    root=tmp_path/'run';(root/'a_bev_only').mkdir(parents=True);write(root/'admission.json',{})
    directory=root/'completion_recovery';write(directory/'supplement.json',dict(script_sha256=sha(recovery.__file__),admission_sha256=sha(root/'admission.json')))
    config={k:str(tmp_path/k) for k in ('data_dir','data_root','model_dir','lead_bev_ckpt')}
    monkeypatch.setattr(recovery,'source_gate',lambda root:{})
    monkeypatch.setattr(recovery,'paths',lambda root:config)
    monkeypatch.setattr(recovery,'verify_environment',lambda *a:({},{}))
    def validate(run,*a,publish=False):
        if incomplete:raise ValueError('incomplete_training_cannot_be_finalized')
        if publish:write(run/'training_complete.json',{})
        return {'group':run.name,'optimizer_updates_added':0}
    monkeypatch.setattr(recovery,'validate_completed',validate)
    calls=[]
    def launch(cmd,log):
        op=cmd[cmd.index('--operation')+1];out=Path(cmd[cmd.index('--output-dir')+1]);calls.append((op,out))
        if op=='train':out.mkdir();write(out/'training_complete.json',{})
        else:write(out/'metrics.json',{});write(out/'condition_sensitivity.json',{})
    monkeypatch.setattr(host_memory,'run_logged',launch)
    def build(root,data):
        marker=root/'marker.json';write(marker,{'fixture':True})
        result=audit.pack(root/'delivery/fixture.zip',{'marker.json':marker},summary={})
        import zipfile
        with zipfile.ZipFile(result['package']) as z:assert 'completion_recovery/runner.py' in z.namelist()
        return result
    monkeypatch.setattr(audit,'build',build)
    if incomplete:
        with pytest.raises(ValueError,match='incomplete_training'):recovery.run_inside(root,'suite',[],False)
        assert calls==[]
        assert not (root/'failure_delivery').exists()
        assert (root/'completion_recovery/failure_delivery/diagnostic.zip').exists()
    else:
        recovery.run_inside(root,'suite',[],False)
        assert [p.name for op,p in calls if op=='train']==list(audit.GROUPS[1:])
        assert [p.parent.name for op,p in calls if op=='evaluate']==list(audit.GROUPS)
        assert recovery.read(directory/'status.json')['status']=='completed'
        calls.clear();recovery.run_inside(root,'suite',[],False)
        assert not calls


def test_bridge_exception_closes_iterator_writer_leaves_group_for_outer_cleanup(monkeypatch):
    from qwen3vl_local.action_prior import training_core as core
    events=[]
    class Coordinator:
        def __enter__(self):return self
        def __exit__(self,*a):events.append('signals_restored')
        def close_iterators(self):events.append('iterators_closed')
    monkeypatch.setattr(core,'TerminationCoordinator',Coordinator)
    monkeypatch.setattr(core,'_archive_stale_termination',lambda *a:None)
    def fail(**kw):raise RuntimeError('real failure')
    monkeypatch.setattr(core,'_run_training_loop',fail)
    writer=SimpleNamespace(close=lambda:events.append('writer_closed'))
    with pytest.raises(RuntimeError,match='real failure'):
        recovery.pinned_loop(core,out=Path('/unused'),rank=0,world=1,writer=writer)
    assert events==['signals_restored','iterators_closed','writer_closed'] and core._ACTIVE_TERMINATION is None
