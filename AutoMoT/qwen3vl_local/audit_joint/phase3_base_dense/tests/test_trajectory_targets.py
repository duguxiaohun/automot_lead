from collections import Counter
from dataclasses import replace
from pathlib import Path
import pickle
import pytest
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import args
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.audit_joint import phase3_base_trial as trial, phase3_suite as suite
from qwen3vl_local.audit_joint.phase3_base_dense import trajectory_targets as targets, runtime, planning, baseline, evaluation, objectives, worker
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as rule, train


def fixture(tmp_path,monkeypatch):
    job,original,extra=raw_fixture(tmp_path,monkeypatch)
    job.update(forecast_protocol='trajectory',supervision='all',trajectory_coverage_policy='strict')
    return job,original


def test_timeline_from_waypoint_config_not_action_evidence_window():
    cfg=targets.timeline()
    assert cfg['offset_frames']==list(range(9))
    assert cfg['waypoint_dt']==.25 and cfg['waypoint_count']==8
    assert planning.options(dict(forecast_protocol='trajectory'))['offset_frames']==list(range(9))
    assert planning.options({})['offset_frames']==[0,12,24]
    with pytest.raises(ValueError,match='single control'):
        planning.options(dict(forecast_protocol='trajectory',supervision='single'))
    assert suite.parser().parse_args(['--index','x','--output','y']).forecast_protocol=='trajectory'


def test_raw_targets_keep_anchor_context_no_aux_needed_and_reuse_is_bound(tmp_path,monkeypatch):
    job,raw=fixture(tmp_path,monkeypatch);job['aux_index']=None;job['aux_binding']=None
    original=Path(job['index']).read_bytes()
    pool=runtime.load_pool(job,recompute_aux=True)
    assert len(pool.proof['targets'])==240  # eight points, ten events, three splits
    anchor=next(r for r in pool.anchor_rows if pool.action(r) is not None)
    rec=pool.record(anchor,'future')
    assert len(rec['actions'])==9 and None not in rec['actions']
    assert rec['images']==list(anchor.history_rgb_paths)
    assert rec['actions'][0]==pool.action(anchor)
    assert Path(job['index']).read_bytes()==original
    assert runtime.load_pool(job).record(anchor,'future')==rec
    # Physical identity is shared, but event-domain target identities are distinct.
    other='STATIC_BLOCKAGE' if anchor.context_id!='STATIC_BLOCKAGE' else 'LEAD_BRAKE'
    alternate=replace(anchor,context_id=other,prompt_road_structure='STRAIGHT')
    assert targets.physical_key(anchor,8)==targets.physical_key(alternate,8)
    assert targets.target_key(anchor,8)!=targets.target_key(alternate,8)
    meta=Path(job['data_root'])/anchor.scenario/anchor.route_id/'metas/0008.pkl'
    meta.write_bytes(meta.read_bytes()+b'change')
    with pytest.raises(ValueError,match='source changed'):runtime.load_pool(job)


def test_missing_meta_is_unknown_not_keep_and_reports_before_coverage_error(tmp_path,monkeypatch):
    job,raw=fixture(tmp_path,monkeypatch)
    row=next(r for r in raw if not r['invalid_action_context'] and r['split']=='train')
    path=Path(job['data_root'])/row['scenario']/row['route_id']/'metas/0002.pkl';path.unlink()
    pool=runtime.load_pool(job,recompute_aux=True)
    anchor=next(r for r in pool.anchor_rows if r.route_id==row['route_id'])
    rec=pool.record(anchor,'future')
    assert rec['actions'][2] is None and rec['actions'][8] is None
    assert rec['target_reasons'][8]=='anchor_to_target_meta_gap'
    with pytest.raises(ValueError,match='Trajectory intent coverage'):
        planning.make_plan(pool,args(tmp_path),job,epoch=0)
    report=trial.read(Path(job['output'])/'trajectory_train_coverage_epoch0.json')
    assert any('meta_gap' in k for k in report['missing_reasons'])


def test_allowed_future_action_not_removed_by_current_action_quotas_and_overlap_normalized(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    # RESUME is allowed in every anchor domain, even if absent in that
    # event's selected current-action quotas. It must keep future loss weight.
    for t in pool.proof['targets'].values():t['action']='RESUME'
    cfg=args(tmp_path)
    work,plan,recs=planning.make_plan(pool,cfg,job,epoch=0)
    reference,ref=baseline.independent_work([r for r in pool.anchor_rows if r.split=='train' and pool.action(r)],cfg,epoch=0,world_size=4)
    assert plan['ordered_ids']==ref['ordered_ids']
    mass=Counter();physical=Counter();anchor_mass=Counter()
    for item in work:
        rec=pool.record(item.row,'future');weighted=recs[rec['id']]
        for i,(a,w,s) in enumerate(zip(weighted['actions'],weighted['query_weights'],weighted['source_ids'])):
            mass[rec['event']]+=w
            if i:physical[rec['event'],s]+=w
            else:anchor_mass[rec['event'],a]+=w
    assert all(abs(n-cfg.focus_balance_count)<1e-7 for n in mass.values())
    assert physical and all(v>0 for v in physical.values())
    for key,n in plan['anchor_action_quotas'].items():
        assert anchor_mass[tuple(key.split('/'))]==pytest.approx(n/2)
    for event in mass:
        values=[v for (e,s),v in physical.items() if e==event]
        assert max(values)==pytest.approx(min(values))


def test_nine_readouts_targets_absent_from_prompt_and_real_backward(tmp_path,monkeypatch,processor,model):
    from types import SimpleNamespace
    from PIL import Image
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    rec=pool.record(next(r for r in pool.anchor_rows if pool.action(r)),'future')
    images=[Image.new('RGB',(32,32)) for _ in range(4)]
    before=objectives.messages(rec,images)
    modified=dict(rec,actions=['STOP']*9,target_reasons=['SECRET']*9)
    assert objectives.messages(modified,images)==before
    text=str(before)
    assert text.count('All queries retain the current anchor event')==1
    assert 'uses all six choices' not in text
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    rec['query_weights']=[.5]+[.5/8]*8
    packed=objectives.pack(bundle,rec,images)
    assert len(packed['positions'])==9
    loss,report=objectives.readout_loss(model,packed,'cpu')
    loss.backward()
    assert report['supervised']==9


def test_native_full_projection_and_unknown_road_transition(tmp_path,monkeypatch):
    job,raw=fixture(tmp_path,monkeypatch)
    r=next(r for r in raw if not r['invalid_action_context'])
    run=Path(job['data_root'])/r['scenario']/r['route_id']
    trajectory=rule.load_route_trajectory(run)
    assert targets.classify(trajectory,1,'STATIC_BLOCKAGE')['action']=='KEEP'
    path=run/'metas/0006.pkl';meta=pickle.loads(path.read_bytes());meta['road_id']=2;path.write_bytes(pickle.dumps(meta))
    result=targets.classify(rule.load_route_trajectory(run),1,'STATIC_BLOCKAGE')
    assert targets.classify(rule.load_route_trajectory(run),1,'SIGNAL_FAILURE')['action']=='KEEP'
    assert result['action'] is None and result['reason'].startswith('incomplete_lateral_evidence:')


def test_frozen_evaluation_all_future_positions_and_missing_reason(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    report,records=evaluation.build(job,pool)
    assert len(report['val']['coverage'])==90
    assert set(report['val']['coverage'].values())=={1.}
    for t in pool.proof['targets'].values():
        if t['key'][2]=='val':t.update(action=None,reason='fixture_unknown')
    with pytest.raises(ValueError,match='val fixed-anchor'):
        evaluation.build(job,pool)
    assert trial.read(Path(job['output'])/'trajectory_eval_coverage_val.json')['missing_reasons']


def test_raw_projection_reused_explicitly_without_recomputing(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    artifact=Path(job['output'])/'trajectory_targets.json'
    newer=dict(job,output=str(tmp_path/'next'),trajectory_targets=str(artifact))
    monkeypatch.setattr(targets,'build',lambda *a:pytest.fail('must not regenerate explicit targets'))
    other=runtime.load_pool(newer,recompute_aux=True)
    assert other.proof==pool.proof
    summary=trial.read(Path(newer['output'])/'trajectory_targets_summary.json')
    assert summary['reused_from']==str(artifact)
    assert runtime.contract(newer)['trajectory_targets_sha256']==trial.file_sha(artifact)


def test_real_nine_query_preflight_includes_all_fixed_evaluation_shapes(tmp_path,monkeypatch,processor):
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    from qwen3vl_local.qwen35.backend import LocalProcessor
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    monkeypatch.setattr(LocalProcessor,'from_pretrained',lambda *a,**kw:processor)
    worker.preflight(job,pool)
    report=trial.read(Path(job['output'])/'dense_input_shapes.json')
    assert report['input_usage_counts']['evaluation/future']==50
    coverage=trial.read(Path(job['output'])/'dense_coverage.json')
    assert len(coverage['epoch_plans'])==3 and coverage['supplemental_raw_intents_built']
    assert coverage['epoch_plans']['0']['planned_supervised_queries']==10240*9


def test_actual_nine_query_training_consumption_and_checkpoint(tmp_path,monkeypatch,processor):
    import json
    import torch
    from types import SimpleNamespace
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    for name in ('_make_item','_contiguous_work','_loss_batch','SEQUENCE_POLICY','_build_inputs',
                 'load_model_with_lora','_save_adapter','_write_run_metadata','_append_jsonl','verify_epoch_consumption'):
        monkeypatch.setattr(train,name,getattr(train,name))
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(320))
        def forward(self,input_ids,logits_to_keep,**kw):
            return SimpleNamespace(logits=self.values[None,None].expand(1,len(logits_to_keep),-1))
    def load(*a,**kw):
        m=Tiny();return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['values'],
            processor=processor,tokenizer=processor.tokenizer,device='cpu')
    monkeypatch.setattr(train,'load_model_with_lora',load)
    monkeypatch.setattr(runtime,'initialize',lambda b,j:b)
    runtime.install(job,pool)
    cfg=runtime.make_args(job);cfg.device='cpu';cfg.focus_balance_count=1;cfg.warmup_steps=2
    cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0;cfg.save_final=False
    train.train(cfg)
    root=Path(cfg.output_dir)
    state=torch.load(root/'training_states/epoch_0001/training_state.pt',weights_only=True)
    assert state['global_step']==10 and state['trainable']['values'].abs().sum()>0
    report=trial.read(root/'balance/query_consumed_epoch_0001.json')
    assert report['supervised_queries']==90 and report['anchor_presentations']==10
    records=[json.loads(line) for line in (root/'epoch_0001_rank0_batches.jsonl').read_text().splitlines()]
    assert len(records)==10
    assert all(len(r['dense_query_source_ids'])==9 for r in records)
    assert all(r['dense_query_target_reasons'][0]=='frozen_native_anchor' for r in records)


def test_overlap_between_neighboring_anchors_uses_same_physical_identity(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch)
    rows=[]
    for split in ('train','val','test'):
        rows.extend(train._read_rows(Path(job['index']),split,data_root=Path(job['data_root'])))
    originals=[r for r in rows if r.split=='train' and targets.Pool.action(r)]
    rows += [replace(r,frame_id=r.frame_id+1) for r in originals]
    pool=targets.TrajectoryPool(rows,targets.build(job,rows))
    anchor=originals[0]
    first=pool.future(anchor);second=pool.future(replace(anchor,frame_id=anchor.frame_id+1))
    assert first['source_ids'][2:]==second['source_ids'][1:8]
    job['trajectory_coverage_policy']='masked'
    work,plan,recs=planning.make_plan(pool,args(tmp_path),job,epoch=0)
    mass=Counter()
    for item in work:
        rec=pool.record(item.row,'future');rec=recs[rec['id']]
        for s,w in zip(rec['source_ids'][1:],rec['query_weights'][1:]):mass[rec['event'],s]+=w
    for e in plan['future_unique_targets_by_event']:
        values=[v for (event,s),v in mass.items() if event==e]
        assert max(values)==pytest.approx(min(values))


def test_native_stop_keeps_original_event_specific_evidence_requirements(tmp_path,monkeypatch):
    job,raw=fixture(tmp_path,monkeypatch)
    r=next(r for r in raw if not r['invalid_action_context'])
    run=Path(job['data_root'])/r['scenario']/r['route_id']
    for path in (run/'metas').glob('*.pkl'):
        meta=pickle.loads(path.read_bytes());meta.update(speed=0.,brake=True,throttle=0.,road_id=int(path.stem)%2)
        path.write_bytes(pickle.dumps(meta))
    trajectory=rule.load_route_trajectory(run)
    result=targets.classify(trajectory,1,'LEAD_BRAKE')
    assert targets.classify(trajectory,1,'STATIC_BLOCKAGE')['action'] is None
    assert result['action']=='STOP' and not result['evidence']['lateral_observation_complete']


def test_invalid_waypoint_override_rejected():
    with pytest.raises(ValueError,match='span/step overrides'):
        planning.options(dict(forecast_protocol='trajectory',forecast_step_seconds='.5'))


def test_nine_query_ledger_package_keeps_sources_without_raw_targets(tmp_path):
    import zipfile
    root=tmp_path/'run';root.mkdir()
    ledger=root/'epoch_0001_rank0_batches.jsonl'
    # Dense source identity audit may legitimately exceed the former 4 MiB cap.
    ledger.write_bytes(b'{"bounded_fixture":true}\n'*(5*1024*1024//25))
    trial.write(root/'trajectory_targets_summary.json',{'status':'fixture'})
    (root/'trajectory_targets.json').write_text('raw local proof must not be uploaded')
    trial.package(root,{'status':'fixture'})
    with zipfile.ZipFile(root/'handoff/phase3_base_report.zip') as z:
        assert 'trajectory_targets_summary.json' in z.namelist()
        assert ledger.name in z.namelist()
        assert 'trajectory_targets.json' not in z.namelist()


def test_masked_admission_requires_actual_density_and_distinct_targets():
    from collections import defaultdict
    from qwen3vl_local.audit_joint.phase3_base_dense.trajectory_plan import admission
    opts=planning.options(dict(forecast_protocol='trajectory'))
    available={f'E/{i}':32 for i in range(9)}
    valid=Counter({'E/0':32,**{f'E/{i}':4 for i in range(1,9)}})
    identities=defaultdict(set,{f'E/{i}':{'physical_a','physical_b'} for i in range(1,9)})
    result=admission(available,valid,identities,opts)
    assert not result['bad'] and result['mean_future_queries_per_anchor_by_event']['E']==1
    assert admission(available,valid,identities,dict(opts,trajectory_coverage_policy='strict'))['bad']
    valid['E/8']=3
    assert 'E/mean_future_queries' in admission(available,valid,identities,opts)['bad']
    valid['E/8']=4;identities['E/8']={'same_target_repeated_100_times'}
    assert 'E/8' in admission(available,valid,identities,opts)['bad']
    valid['E/8']=0
    assert admission(available,valid,identities,opts)['bad']


def test_distributed_plan_and_external_reuse_do_not_race_on_report_writes(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch);pool=runtime.load_pool(job,recompute_aux=True)
    external=dict(job,trajectory_targets=str(Path(job['output'])/'trajectory_targets.json'))
    monkeypatch.setenv('RANK','1')
    monkeypatch.setattr(trial,'write',lambda *a:pytest.fail('nonzero rank must not rewrite shared artifacts'))
    planning.make_plan(pool,args(tmp_path),job,epoch=0)
    runtime.load_pool(external,recompute_aux=False)


def test_same_future_frame_projects_different_actions_in_fixed_anchor_domains(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
    job,raw=fixture(tmp_path,monkeypatch)
    r=next(r for r in raw if not r['invalid_action_context'])
    run=Path(job['data_root'])/r['scenario']/r['route_id']
    for path in (run/'metas').glob('*.pkl'):
        meta=pickle.loads(path.read_bytes());meta.update(lane_id=-1 if int(path.stem)<5 else -2)
        path.write_bytes(pickle.dumps(meta))
    trajectory=rule.load_route_trajectory(run);sig=trajectory.signals(1)
    labels=rule.label_actions(sig);evidence=rule.action_evidence(sig)
    speed=targets.classify(trajectory,1,'LEAD_BRAKE')
    maneuver=targets.classify(trajectory,1,'STATIC_BLOCKAGE')
    assert speed['action']=='KEEP' and speed['keep_scope']=='speed_only'
    assert maneuver['action'] in ('LANE_CHANGE_LEFT','LANE_CHANGE_RIGHT')
    for event in train.CONTEXT_IDS:
        actual=targets.classify(trajectory,1,event)
        assert actual['action']==choice_annotation(labels,event,evidence)['primary_action']


def test_same_physical_time_keeps_separate_anchor_domain_targets(tmp_path,monkeypatch):
    job,_=fixture(tmp_path,monkeypatch)
    rows=[]
    for split in ('train','val','test'):
        rows.extend(train._read_rows(Path(job['index']),split,data_root=Path(job['data_root'])))
    anchor=next(r for r in rows if r.context_id=='LEAD_BRAKE' and r.split=='train')
    other=replace(anchor,context_id='STATIC_BLOCKAGE')
    rows.append(other)
    pool=targets.TrajectoryPool(rows,targets.build(job,rows))
    a=pool.future(anchor);b=pool.future(other)
    assert a['frames']==b['frames'] and a['source_ids'][1:]!=b['source_ids'][1:]
    assert a['forecast_event_domain']=='LEAD_BRAKE' and b['forecast_event_domain']=='STATIC_BLOCKAGE'
    assert pool.proof['targets'][a['source_ids'][1]]['allowed_actions']==['DECELERATE','STOP','RESUME','KEEP']
    pool.proof['targets'][a['source_ids'][1]]['action']='LANE_CHANGE_LEFT'
    with pytest.raises(ValueError,match='fixed anchor event domain'):pool.future(anchor)
