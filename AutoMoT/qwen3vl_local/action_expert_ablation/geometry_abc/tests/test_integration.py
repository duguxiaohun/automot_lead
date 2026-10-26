import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from .test_core import record,tiny,inputs
from qwen3vl_local.action_expert_ablation.geometry_abc import audit,planning,runtime,worker


def test_real_entry_parser_accepts_only_the_frozen_ABC_sampling():
    base=['--abc-variant','a','--branch','bev_only','--suite-root','/unused']
    args=worker.parser().parse_args(base);worker.validate_args(args)
    assert args.sampling_mode=='abc_regions' and args.event_balance_max_frame_repeats==11
    for option in ('--event-balanced','--action-balanced'):
        with pytest.raises(ValueError,match='sampling_mode'):
            worker.validate_args(worker.parser().parse_args(base+[option]))


@pytest.mark.parametrize('branch',['bev_only','qwen_simple'])
def test_native_branch_boundary_receives_exact_phase3_status_and_regions(monkeypatch,branch):
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_prior import image_condition
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    import numpy as np
    m=tiny();r=record();r['observation']['speed_mps']=3.25
    r['observation']['navigation_xy']=[[.2,.4],[7.,8.],[123.,-42.]]
    captured=[];runner=SimpleNamespace(_run_leadmot_qwen_prefill=lambda *a:None)
    original_prefill=runner._run_leadmot_qwen_prefill
    monkeypatch.setattr(image_condition,'current_base_prefill',lambda run,images,text,system:captured.append((images,text,system)))
    if branch=='bev_only':
        inner=common.BevOnlyRuntime.__new__(common.BevOnlyRuntime)
        inner._torch=torch;inner.device=torch.device('cpu');inner.args=SimpleNamespace(rgb_frame_step=1,rgb_frame_count=1)
        runner._build_group_indices=lambda **kwargs:None
        runner._prepare_inference_inputs=lambda *a:([],[],torch.full((1,7),-999.),None,None)
    else:
        inner=common.QwenSimpleRuntime.__new__(common.QwenSimpleRuntime);inner.device=torch.device('cpu')
        class FrozenBoundary:
            def forward_sample(self,sample,decoder,config,dtype,clip):
                runner._run_leadmot_qwen_prefill(['one_current_image'],'old_navigation_prompt')
                kw=inputs(m,r)
                for key in ('geometry_token_ids','geometry_reference_yaw','history_token_ids','history_mask','representation_contract'):
                    kw.pop(key)
                kw.update(speed=torch.tensor([-999.]),target_point=torch.full((1,2),-999.))
                return decoder(**kw)
        inner.inner=FrozenBoundary()
    inner.runner=runner
    rt=runtime.Runtime.__new__(runtime.Runtime)
    rt.inner=inner;rt.device=torch.device('cpu');rt.variant='a';rt.branch=branch;rt.intervention=None
    flow=make_training_flow(torch.ones(1,10,2),torch.ones(1,8,2),m.flow_config)
    out=rt.forward_sample(runtime.sample(r,'/unused'),m,m.config,torch.float32,clip={'rgb':np.zeros((1,2,2,3))},
        flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
    loss=flow_matching_loss(out,flow['flow_target_velocity'],10,.5,1.)[0];loss.backward()
    assert torch.allclose(out['input_status'],torch.tensor([[3.25,.2,.4,7.,8.,123.,-42.]]))
    assert m.conditioner.geometry_embedding.role.weight.grad.norm(dim=1).min()>0
    assert runner._run_leadmot_qwen_prefill is original_prefill
    if branch=='qwen_simple':
        assert len(captured)==1 and captured[0][0]==['one_current_image']
        assert 'ACTION_QUERY' not in captured[0][1] and 'Current speed: 3.25' in captured[0][1]


def test_lossless_members_and_float_roundtrip(tmp_path):
    from qwen3vl_local.audit_joint.geometry_v1.dataset import digest
    r=record();r['id']=digest([r['scenario'],r['route_id'],r['frame'],r['event']])
    lookup,paths=audit.members_archive({'train':[r],'val':[],'test':[]},tmp_path)
    z=np.load(paths['members.npz']);meta=json.loads(paths['membership.json'].read_text())
    route,frame,event,split=z['indices'][0];scenario,rid,_=meta['routes'][route]
    assert digest([scenario,rid,int(frame),meta['events'][event]])==r['id']
    assert z['regions'].shape==(1,3,4) and lookup[r['id']]==0
    for values in ([[1.,2.,3.,4.]],[[1/3,1/7,1/11,1/13]]):
        audit.error_archive(tmp_path/'errors.npz',[0],values)
        assert np.array_equal(np.load(tmp_path/'errors.npz')['errors'].astype(np.float64),values)


def test_output_region_projection_keeps_2D_height_assumption_explicit():
    r=record();r['observation']['reference']['yaw']=0.
    c=dict(geometry_projection=dict(rotation_world_xy=[[1.,0.,0.],[0.,1.,.9]],
        gt_endpoint_xyz=[[.49,0.,3.]]*4),gt_waypoints=[[[.49,0.]]*8],pred_waypoints=[[[4.,0.]]*8])
    predicted,gt2d=audit.reclassified_regions(c,r,'c')
    assert predicted==[4]*4 and gt2d==[3]*4
    c['geometry_projection']['gt_endpoint_xyz']=[[.49,0.,99.]]*4
    assert audit.reclassified_regions(c,r,'c')==(predicted,gt2d)
    c['gt_waypoints'][0][1][0]=5.
    with pytest.raises(ValueError,match='ticks'):audit.reclassified_regions(c,r,'c')


def test_native_missing_history_replay_preserves_gap_reasons(tmp_path):
    import lzma,pickle
    from qwen3vl_local.audit_joint.geometry_v1 import dataset as native
    from qwen3vl_local.audit_joint.geometry_v1.tests.test_geometry import meta,Lanes
    from qwen3vl_local.audit_joint.geometry_abc_v2.dataset import combine
    from qwen3vl_local.action_expert_ablation.geometry_abc import data,preflight
    metas={i:meta(i) for i in range(3)};folder=tmp_path/'Synthetic'/'Town00_Test'/'metas';folder.mkdir(parents=True)
    for i,value in metas.items():
        with lzma.open(folder/f'{i:04d}.pkl','wb') as f:pickle.dump(value,f)
    route=dict(scenario='Synthetic',route_id='Town00_Test',town='Town00',split='train')
    r=native.record(route,dict(frame=2,event='LEAD_BRAKE',road_structure='R1'),metas,Lanes())
    r['physical_group']='Synthetic/Town00_Test';r=data.compact_record(combine(r))
    rows={'train':[r],'val':[],'test':[]};report=preflight.history_check(rows,tmp_path,tmp_path/'history.json')
    assert sum(report['missing_reasons'].values())==3 and report['replayed_physical_frames']==1
    assert all(k.endswith('history_meta_gap') for k in report['missing_reasons'])
    r['representations']['a']['history'][0]['valid']=True
    with pytest.raises(ValueError,match='history_replay'):preflight.history_check(rows,tmp_path,tmp_path/'changed.json')


def test_known_development_exposure_content_survives_relocation_but_not_change(monkeypatch):
    from qwen3vl_local.audit_joint.geometry_v1 import splits
    from qwen3vl_local.action_expert_ablation.geometry_abc.preflight import verify_development
    saved=dict(groups=['S/R'],sources={'/old/development.json':'abc'},manifests=[])
    current=dict(groups=['S/R'],sources={'/new/development.json':'abc'},manifests=[])
    monkeypatch.setattr(splits,'development_contract',lambda:current)
    assert verify_development({'development':saved})['groups']==1
    current['groups'].append('S/R2')
    with pytest.raises(ValueError,match='exposure_changed'):verify_development({'development':saved})


def test_parallel_raw_verifier_keeps_content_member_and_candidate_rejections(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import data,preflight
    from qwen3vl_local.audit_joint.abc_delivery import digest,sha,write
    root=tmp_path/'raw';run=root/'S'/'R'
    for kind in ('metas','rgb','lidar'):(run/kind).mkdir(parents=True)
    source=run/'metas'/'0000.pkl';source.write_bytes(b'original metadata bytes')
    route=dict(scenario='S',route_id='R',town='Town00',anchors=[{'frame':0}]);key=digest(['S','R'])
    directory=tmp_path/'data';path=directory/'routes'/key/'receipt.json'
    receipt=dict(route_sha256=digest(route),members={'metas':['0000.pkl'],'rgb':[],'lidar':[]},
                 bindings={'S/R/metas/0000.pkl':sha(source)})
    write(path,receipt);manifest={'routes':{key:sha(path)}}
    monkeypatch.setattr(data,'candidates',lambda *a:iter([route]))
    assert preflight.verify_raw_sources(directory,manifest,root,'unused',workers=2)['bound_file_appearances']==1
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw_source_changed'):preflight.verify_raw_sources(directory,manifest,root,'unused')
    source.write_bytes(b'original metadata bytes');extra=run/'rgb'/'0001.jpg';extra.touch()
    with pytest.raises(ValueError,match='route_members_changed'):preflight.verify_raw_sources(directory,manifest,root,'unused')
    extra.unlink();monkeypatch.setattr(data,'candidates',lambda *a:iter([]))
    with pytest.raises(ValueError,match='candidate_route_members'):preflight.verify_raw_sources(directory,manifest,root,'unused')


def test_saved_seven_epoch_plan_survives_JSON_integer_histogram_keys(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation.geometry_abc import preflight
    rows=[record(i) for i in range(3)]
    projected=planning.project(rows,quotas={'A':20},epochs=7,cap=11,world=1,accumulation=4)
    restored=json.loads(json.dumps(projected));(tmp_path/'plan.json').write_text(json.dumps(restored))
    assert preflight.plan_digest(restored)==projected['sha256']
    monkeypatch.setattr(planning,'project',lambda rows:projected)
    assert preflight.verify_plan(tmp_path,{'train':rows})==restored
    first=restored['plans'][0]['physical_repeat_histogram'];key=next(iter(first));first[key]+=1
    assert preflight.plan_digest(restored)!=projected['sha256']


def test_training_sampler_bound_to_epoch_and_restored():
    from qwen3vl_local.action_prior import event_balance
    old=event_balance.build_balanced_epoch
    records=[record(i) for i in range(5)]
    p=planning.project(records,quotas={'A':5},epochs=2,cap=1,world=1,accumulation=2)
    rows=[runtime.sample(r,'/tmp/unused') for r in records]
    with worker.planned_sampler(p,2026):
        selected,a=event_balance.build_balanced_epoch(rows,mode='abc_regions',total=5,seed=2027)
        assert [r['abc_record']['id'] for r in selected]==p['plans'][1]['ordered_ids']
        with pytest.raises(ValueError,match='epoch'):event_balance.build_balanced_epoch(rows,mode='abc_regions',total=5,seed=2029)
    assert event_balance.build_balanced_epoch is old


@pytest.mark.parametrize('storage',['list','indexed'])
def test_tail_windows_and_actual_exposure_through_shared_training_loop(tmp_path,monkeypatch,storage):
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    from qwen3vl_local.action_prior.tests.test_training_loop import lightweight_old_helpers
    from qwen3vl_local.action_prior import training_core as core
    from qwen3vl_local.action_prior.config import DEFAULTS
    from qwen3vl_local.action_prior.flow_matching import make_evaluation_flow
    old=lightweight_old_helpers();records=[record(i) for i in range(5)]
    rows=[runtime.sample(r,'/unused') for r in records]
    for row in rows:row['consumed_sources']={k:'0'*64 for k in ('metas','rgb','lidar')}
    if storage=='indexed':
        from qwen3vl_local.action_expert_ablation.geometry_abc.indexed_rows import Store,Rows
        original=runtime.sample
        def with_proof(r,root):
            return dict(original(r,root),consumed_sources={k:'0'*64 for k in ('metas','rgb','lidar')})
        monkeypatch.setattr(runtime,'sample',with_proof)
        path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        from qwen3vl_local.audit_joint.abc_delivery import sha
        rows=Rows(Store(path,sha(path),'train','/unused'))
    args=SimpleNamespace(**dict(DEFAULTS,num_epochs=2,grad_accum_steps=2,num_workers=0,
        high_level_action_token=False,action_token_separation_weight=0.,sampling_mode='abc_regions',
        event_balance_max_frame_repeats=1,decoder_dtype='float32',save_steps=1,logging_steps=1,
        val_steps=100,num_workers_unused=0,output_dir=str(tmp_path),evaluation_distribution='natural'))
    m=tiny();projection=planning.project(records,quotas={'A':5},epochs=2,cap=1,world=1,accumulation=2)
    plan=dict(samples_per_epoch=5,optimizer_steps_per_epoch=3,actual_step_limit=6,per_event=5)
    optimizer,scheduler,ema=core.make_optimization(args,m,plan,old)
    from qwen3vl_local.action_expert_ablation.geometry_abc.gradients import Evidence
    gradients=Evidence(m,optimizer,0)
    class Runtime:
        device=torch.device('cpu');variant='a';branch='bev_only'
        def forward_sample(self,sample,decoder,config,dtype,clip=None,**kwargs):
            return decoder(**inputs(m,sample['abc_record']),**kwargs)
    logs=[];writer=SimpleNamespace(add_scalar=lambda *x:logs.append(x),flush=lambda:None,close=lambda:None)
    saved=[];gradient_reports=[];rt=Runtime();rt.gradient_evidence=gradients;rt.training_evidence_active=True
    def save(path,cursor,step,best):
        saved.append((copy.deepcopy(cursor),step))
        assert m.conditioner.geometry_embedding.token.weight.grad is None
        core.save_training_checkpoint(path,m,optimizer,scheduler,ema,args,{'test':'gradient'}, {},
            cursor,step,best,0,1,schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
        report=gradients.snapshot(step);gradient_reports.append(report)
        (tmp_path/f'gradient_{step}.json').write_text(json.dumps(report))
    def evaluate(*args,**kwargs):return dict(loss=1.,route_ade_m=1.,waypoint_ade_m=1.,samples=1)
    with worker.planned_sampler(projection,2026):
        core.run_training_loop(args=args,rows={'train':rows,'val':rows[:1]},plan=plan,runtime=rt,
            model=m,decoder=m,config=m.config,flow_config=m.flow_config,optimizer=optimizer,scheduler=scheduler,
            ema=ema,cursor={'epoch':0,'micro':0},step=0,best=float('inf'),device=torch.device('cpu'),dtype=torch.float32,
            rank=0,world=1,out=tmp_path,writer=writer,old=old,hooks=worker.hooks(),evaluate_fn=evaluate,checkpoint=save)
    assert saved[-1][1]==6
    assert all(np.isfinite(p['gradient_norms']).all() and max(p['gradient_norms'])>0 for p in gradient_reports)
    assert gradient_reports[-1]['slot_exposure']['samples']==10
    assert sum(v for k,v in gradient_reports[-1]['slot_exposure'].items() if k.startswith('history/'))==40
    assert gradient_reports[-1]['optimizer_steps']==6
    assert torch.load(tmp_path/'latest.pt',weights_only=False)['step']==6
    gradients.close()
    assert [v for tag,v,step in logs if tag=='train/step_samples']==[2,2,1,2,2,1]
    for p in (tmp_path/'epoch_audit').glob('*.json'):
        counts=json.loads(p.read_text());assert sum(v for k,v in counts.items() if k.startswith('exposure/'))==5


def ddp_child(rank,path,output):
    import torch.distributed as dist
    from qwen3vl_local.action_expert_ablation.geometry_abc.gradients import Evidence
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    torch.set_num_threads(1);torch.manual_seed(41)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4)
    try:
        reports={}
        for variant in 'abc':
            m=tiny(variant);ddp=torch.nn.parallel.DistributedDataParallel(m)
            opt=torch.optim.AdamW(m.parameters(),lr=.001)
            evidence=Evidence(m,opt,rank)
            for epoch in range(2):
                flow=make_training_flow(torch.ones(1,10,2),torch.ones(1,8,2),m.flow_config)
                out=ddp(**inputs(m,record(rank),variant),flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
                loss=flow_matching_loss(out,flow['flow_target_velocity'],10,.5,1.)[0]
                opt.zero_grad();loss.backward();evidence.observe(record(rank),variant);opt.step();opt.zero_grad(set_to_none=True)
            assert m.conditioner.geometry_embedding.token.weight.grad is None
            report=evidence.snapshot(2);reports[variant]=report
            assert report['slot_exposure']['samples']==2 and max(report['gradient_norms'])>0
            norm=torch.tensor(report['gradient_norms']);all_norms=[torch.empty_like(norm) for _ in range(4)]
            dist.all_gather(all_norms,norm);assert all(torch.equal(norm,n) for n in all_norms)
            evidence.close()
            value=m.conditioner.geometry_embedding.token.weight.detach();gathered=[torch.empty_like(value) for _ in range(4)]
            dist.all_gather(gathered,value)
            assert all(torch.equal(value,x) for x in gathered)
        Path(output,f'rank{rank}.json').write_text(json.dumps({'rank':rank,'variants':3,'updates_per_variant':2,'gradient_evidence':reports}))
    finally:dist.destroy_process_group()


def test_real_four_gloo_geometry_parameters(tmp_path):
    torch.multiprocessing.start_processes(ddp_child,args=(str(tmp_path/'rendezvous'),str(tmp_path)),
        nprocs=4,join=True,start_method='spawn')
    assert len(list(tmp_path.glob('rank*.json')))==4


def test_native_full_checkpoint_with_geometry_contract_and_optimizer(tmp_path,monkeypatch):
    from qwen3vl_local.action_prior import training_core as core
    from qwen3vl_local.action_prior.tests.test_training_loop import lightweight_old_helpers
    from qwen3vl_local.action_prior.flow_matching import make_training_flow,flow_matching_loss
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    args=worker.parser().parse_args(['--abc-variant','a','--branch','bev_only','--suite-root',str(tmp_path)])
    worker.validate_args(args);old=lightweight_old_helpers();m=tiny();r=record()
    opt,scheduler,ema=core.make_optimization(args,m,dict(samples_per_epoch=116256,optimizer_steps_per_epoch=1817,actual_step_limit=12719),old)
    flow=make_training_flow(torch.ones(1,10,2),torch.ones(1,8,2),m.flow_config)
    out=m(**inputs(m,r),flow_state=flow['flow_state'],flow_time=flow['flow_time'],sample_trajectory=False)
    loss=flow_matching_loss(out,flow['flow_target_velocity'],10,.5,1.)[0]
    loss.backward();opt.step();scheduler.step();ema.update(m)
    binding={'representation':m.get_extra_state()};path=tmp_path/'probe.pt'
    core.save_training_checkpoint(path,m,opt,scheduler,ema,args,binding,{'train':'fixture'},
        {'epoch':0,'micro':1},1,float('inf'),0,1,schema='abc_trajectory_checkpoint_v1',contract_key='condition_contract')
    saved=torch.load(path,weights_only=False);restored=tiny();restored.load_state_dict(saved['decoder'],strict=True)
    restored_ema=old._DecoderEMA(restored);restored_ema.load_state_dict(saved['ema_state_dict'])
    assert saved['condition_contract']==binding and saved['step']==1 and saved['optimizer']['state']
    for key,value in ema.shadow.items():assert torch.equal(value,restored_ema.shadow[key])
    m.eval();restored.eval();noise=torch.randn(1,18,2)
    with ema.apply_to(m),restored_ema.apply_to(restored):
        assert torch.equal(m(**inputs(m,r),flow_sample_noise=noise)['pred_future_waypoints'],
                           restored(**inputs(restored,r),flow_sample_noise=noise)['pred_future_waypoints'])


@pytest.mark.parametrize('name',['qwen_adapter_dir','init_from_ckpt','phase1_adapter','phase2_adapter','lora_bundle','resume'])
def test_fresh_abc_rejects_prior_task_weights(name):
    args=worker.parser().parse_args(['--abc-variant','a','--branch','bev_only','--suite-root','unused'])
    setattr(args,name,'prior_weights')
    with pytest.raises(ValueError):worker.validate_args(args)


@pytest.mark.parametrize('operation',['train','probe','evaluate'])
def test_fresh_abc_checkpoint_only_for_evaluation(operation):
    args=worker.parser().parse_args(['--abc-variant','b','--branch','qwen_simple','--suite-root','unused',
        '--operation',operation,'--checkpoint','own_run/best.pt'])
    if operation=='evaluate':worker.validate_args(args)
    else:
        with pytest.raises(ValueError,match='checkpoint_only_allowed'):worker.validate_args(args)


@pytest.mark.parametrize('name',['adapter_config.json','adapter_model.safetensors'])
def test_abc_resource_admission_rejects_automatic_adapter_loading(tmp_path,monkeypatch,name):
    from qwen3vl_local.action_expert_ablation.geometry_abc import preflight
    from qwen3vl_local.qwen35 import preflight as qwen_check
    from qwen3vl_local.action_expert_ablation import common
    base=tmp_path/'base';base.mkdir();(base/name).write_text('old adapter')
    bev=tmp_path/'bev.pth';bev.write_bytes(b'fixture')
    monkeypatch.setattr(qwen_check,'check',lambda *a,**kw:dict(errors=[]))
    monkeypatch.setattr(common,'leadmot_train',lambda:None)
    result=preflight.resources(base,bev)
    assert not result['ready'] and 'original_Qwen35_base_required_no_prior_adapter' in result['errors']
    assert result['bev'] and not result['qwen']
