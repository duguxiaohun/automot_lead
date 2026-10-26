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


@pytest.mark.parametrize('branch',['legacy_six','bev_only','qwen_simple','both'])
@pytest.mark.parametrize('incomplete',[False,True])
def test_suite_preserves_a_and_only_trains_missing_groups(tmp_path,monkeypatch,incomplete,branch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import audit
    from qwen3vl_local.action_prior import host_memory
    gate=scope_gate(branch);groups=recovery.selected_groups(gate)
    root=tmp_path/'run';(root/groups[0]).mkdir(parents=True);write(root/'admission.json',gate)
    directory=root/'completion_recovery';write(directory/'supplement.json',dict(script_sha256=sha(recovery.__file__),admission_sha256=sha(root/'admission.json')))
    config={k:str(tmp_path/k) for k in ('data_dir','data_root','model_dir','lead_bev_ckpt')}
    monkeypatch.setattr(recovery,'source_gate',lambda root:gate)
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
        assert [p.name for op,p in calls if op=='train']==list(groups[1:])
        assert [p.parent.name for op,p in calls if op=='evaluate']==list(groups)
        assert recovery.read(directory/'status.json')['status']=='completed'
        assert recovery.read(directory/'status.json')['selected_groups']==list(groups)
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


def scope_gate(branch):
    from qwen3vl_local.action_expert_ablation.geometry_abc.group_scope import groups
    gate={'code':dict(recovery.PROFILES['original_six' if branch=='legacy_six' else 'admitted_scope'])}
    if branch!='legacy_six':gate['selected_groups']=list(groups(branch))
    return gate


@pytest.mark.parametrize('change',['worker','core','audit','scope_code','mixed','reordered','legacy_subset'])
def test_source_and_scope_rejected_before_any_recovery_files(tmp_path,change):
    gate=scope_gate('bev_only');files=dict(gate['code'])
    if change in ('worker','core','audit','scope_code'):
        files[{'worker':recovery.WORKER,'core':recovery.CORE,'audit':recovery.AUDIT,'scope_code':recovery.GROUP_SCOPE}[change]]='changed'
    elif change=='mixed':gate['selected_groups'][-1]='c_qwen_simple'
    elif change=='reordered':gate['selected_groups'].reverse()
    else:gate['code']=dict(recovery.PROFILES['original_six']);files=gate['code']
    write(tmp_path/'admission.json',gate)
    with pytest.raises(ValueError):recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':files})
    assert not (tmp_path/'completion_recovery').exists()


@pytest.mark.parametrize('branch',['legacy_six','bev_only','qwen_simple','both'])
def test_supplement_scope_idempotent_preserves_admission_and_config(tmp_path,branch):
    gate=scope_gate(branch);write(tmp_path/'admission.json',gate);before=sha(tmp_path/'admission.json')
    groups=recovery.selected_groups(gate)
    write(tmp_path/groups[0]/'config.json',dict(suite_root=str(tmp_path),branch=groups[0].split('_',1)[1]))
    assert recovery.paths(tmp_path)['branch']==groups[0].split('_',1)[1]
    path=recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':gate['code']})
    saved=path.read_bytes();proof=recovery.read(path.with_name('supplement.json'))
    assert proof['selected_groups']==list(groups)
    assert recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':gate['code']})==path
    assert path.read_bytes()==saved and sha(tmp_path/'admission.json')==before


@pytest.mark.parametrize('branch',['legacy_six','bev_only'])
@pytest.mark.parametrize('bad_loader',[False,True])
def test_preflight_requires_only_original_probes_and_checks_loader(tmp_path,monkeypatch,branch,bad_loader):
    from qwen3vl_local.action_expert_ablation.geometry_abc import probe_inputs,indexed_rows
    gate=scope_gate(branch);gate.update(resources={'ready':True},dataset='fixture',splits={},
        manifest_sha256='manifest',plan_sha256='plan',processor_sha256='processor')
    write(tmp_path/'admission.json',gate);write(tmp_path/'plan.json',{'sha256':'plan'})
    config=dict(data_dir=str(tmp_path),data_root=str(tmp_path),model_dir='model',lead_bev_ckpt='bev',num_workers=2)
    monkeypatch.setattr(preflight,'resources',lambda *a:gate['resources'])
    monkeypatch.setattr(indexed_rows,'load',lambda *a,**kw:({},dict(identity='fixture',splits={})))
    monkeypatch.setattr(preflight,'verify_development',lambda *a:None)
    monkeypatch.setattr(preflight,'plan_digest',lambda *a:'plan')
    monkeypatch.setattr(probe_inputs,'verify_processor',lambda *a:{})
    monkeypatch.setattr(probe_inputs,'selection',lambda *a:{})
    checked=[];monkeypatch.setattr(probe_inputs,'verify_probe_inputs',lambda *a:checked.append(a[-1].name))
    groups=recovery.selected_groups(gate)
    for group in groups:
        v,b=group.split('_',1);root=tmp_path/'probes'/group;root.mkdir(parents=True)
        (root/'probe.pt').write_bytes(b'fixture')
        write(root/'probe_complete.json',dict(loader_workers=3 if bad_loader else 2,
            admission_sha256=sha(tmp_path/'admission.json'),checkpoint_reload_verified=True,
            post_checkpoint_backward_verified=True,checkpoint_sha256=sha(root/'probe.pt'),
            binding=dict(source=gate['code'],dataset='fixture',resources=gate['resources'],representation=contracts.contract(v),branch=b)))
    if bad_loader:
        with pytest.raises(ValueError,match='original_probe_changed'):recovery.verify_environment(tmp_path,gate,config)
    else:
        recovery.verify_environment(tmp_path,gate,config)
        assert checked==list(groups)


def test_previous_runner_reuse_requires_exact_published_bytes_and_contract(tmp_path,monkeypatch):
    # Small substitute solely for immutable-runner selection; actual published
    # bytes are independently checked in the release audit.
    gate=scope_gate('legacy_six');write(tmp_path/'admission.json',gate)
    target=tmp_path/'completion_recovery/runner.py';target.parent.mkdir();target.write_text('# previous fixture\n')
    expected_sha=sha(target);monkeypatch.setattr(recovery,'PREVIOUS_RUNNER_SHA',expected_sha)
    proof=dict(schema=recovery.SCHEMA,script_sha256=expected_sha,admission_sha256=sha(tmp_path/'admission.json'),
        frozen_version='frozen',source_overrides=recovery.SUPPORTED,
        change='default process group destroyed by worker after performance and completion, not inner loop',
        numerical_loop_unchanged=True,scope='original six groups; completed optimizer runs only; no weight migration')
    write(target.with_name('supplement.json'),proof)
    assert recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':gate['code']})==target
    assert sha(target)==expected_sha
    proof['admission_sha256']='changed';write(target.with_name('supplement.json'),proof)
    with pytest.raises(ValueError):recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':gate['code']})
    write(target.with_name('supplement.json'),dict(proof,admission_sha256=sha(tmp_path/'admission.json')))
    target.write_text('changed')
    with pytest.raises(ValueError):recovery.prepare_supplement(tmp_path,Path('/frozen'),{'files':gate['code']})


@pytest.mark.parametrize('server',['home/cruser1/lda','mnt/workspace/cee3sgh'])
@pytest.mark.parametrize('change',['none','tamper','missing_run'])
def test_bootstrap_native_snapshot_verifier_both_server_layouts(tmp_path,monkeypatch,server,change):
    import shutil
    from qwen3vl_local import runtime_version as version
    app=tmp_path/server/'AutoMoT';(app/'qwen3vl_local').mkdir(parents=True)
    checkpoints=app/'checkpoints'
    # Exercise shared checkpoints as well as the nested cr02 model link.
    if server.startswith('home'):
        assets=tmp_path/'shared_checkpoints';assets.mkdir();checkpoints.symlink_to(assets)
        nested=app/'Automot';nested.mkdir();(nested/'checkpoints').symlink_to(assets)
    else:checkpoints.mkdir()
    script=app/'qwen3vl_local/abc_completion_recovery.py';shutil.copyfile(recovery.__file__,script)
    shutil.copyfile(version.__file__,app/'qwen3vl_local/runtime_version.py')
    files={}
    for name in recovery.PROFILES['admitted_scope']:
        path=app/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture source '+name+'\n');files[name]=sha(path)
    monkeypatch.setattr(recovery,'PROFILES',{'admitted_scope':files})
    frozen=version.prepare(app);root=checkpoints/'run';root.mkdir()
    gate=dict(code=files,selected_groups=['a_bev_only','b_bev_only','c_bev_only'])
    write(root/'admission.json',gate);write(root/'execution_version.json',dict(directory=str(frozen)))
    if change=='tamper':
        path=frozen/'app'/recovery.WORKER;path.chmod(0o644);path.write_text('# tampered\n')
    # A main-tree update must not alter the selected original snapshot.
    (app/recovery.CORE).write_text('# new main tree\n')
    monkeypatch.setattr(recovery,'__file__',str(script));calls=[]
    class Process:
        def __init__(self,command,**kw):calls.append((command,kw))
        def wait(self):return 0
    monkeypatch.setattr(recovery.subprocess,'Popen',Process)
    args=['--run-dir',str(root if change!='missing_run' else checkpoints/'absent'),'--check-only']
    if change!='none':
        with pytest.raises((ValueError,FileNotFoundError)):recovery.main(args)
        assert not calls and not (root/'completion_recovery').exists()
    else:
        with pytest.raises(SystemExit) as exc:recovery.main(args)
        assert exc.value.code==0 and len(calls)==1
        cmd,kw=calls[0]
        assert '--check-only' in cmd and kw['cwd']==frozen/'app'
        assert kw['env']['PYTHONPATH']==str(frozen/'app')
        assert str(root.resolve()) in cmd
        assert recovery.read(root/'completion_recovery/supplement.json')['selected_groups']==gate['selected_groups']
        version.verify(frozen)
