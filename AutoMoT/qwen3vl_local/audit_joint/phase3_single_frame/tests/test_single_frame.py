from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import math
import pytest
import torch
from PIL import Image
from qwen3vl_local.audit_joint.phase3_single_frame import targets, history, dataset, planning, objective
from qwen3vl_local.audit_joint.phase3_base_dense import baseline
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_second_review import raw_fixture
from qwen3vl_local.audit_joint.phase3_base_dense.tests.test_base_dense import args, rows as balanced_rows
from qwen3vl_local.qwen35.tests.test_processor import processor
from qwen3vl_local.qwen35.tests.test_backend import model
from qwen3vl_local.sft_new_loop_phase3 import train, trajectory_action as rule


def fixture(tmp_path, monkeypatch):
    job, raw, _ = raw_fixture(tmp_path,monkeypatch)
    rows = [r for split in ('train','val','test') for r in train._read_rows(Path(job['index']),split,data_root=Path(job['data_root']))]
    artifact, report = dataset.build(rows,job['data_root'],tmp_path/'single',history_policy='offline_native',index=job['index'])
    return job, rows, artifact, report


def test_raw_and_preferred_targets_missing_pairs_and_short_actions():
    w = targets.distribution(['DECELERATE','LANE_CHANGE_LEFT'],1.5)
    assert w['raw']==[.5,0,0,.5,0,0]
    assert w['target']==[.4,0,0,.6,0,0]
    assert w['allowed_answers']==['DECELERATE','LANE_CHANGE_LEFT']
    assert targets.distribution(['STOP',None])['target'] is None
    assert targets.distribution(['STOP','STOP'])['target']==[0,1,0,0,0,0]
    assert targets.distribution(['STOP','RESUME'])['raw']==targets.distribution(['RESUME','STOP'])['raw']
    for weight in (float('nan'),0,4):
        with pytest.raises(ValueError):targets.distribution(['KEEP','KEEP'],weight)


def test_history_direct_native_labels_allow_post_observation_evidence():
    from qwen3vl_local.audit_joint.phase3_base_dense.trajectory_targets import classify
    metas={i:dict(speed=3.,lane_type_str='Driving',road_id=1,section_id=0,lane_id=-1,
                pos_global=[i,0],next_target_points=[[i+10,0]],theta=0.) for i in range(70)}
    t=rule.RouteTrajectory(Path('/synthetic/route'),tuple(metas),metas,{})
    for context in train.CONTEXT_IDS:
        for offset in range(-7,1):
            p=history.recognize(t,30+offset,30,context)
            native=classify(t,30+offset,context)
            assert p['action']==native['action']=='KEEP'
            assert p['required_end_frame']>30 and p['post_observation_evidence_allowed']
            assert p['reason']==native['reason'] and p['evidence']==native['evidence']
    before=history.recognize(t,30,30,'LEAD_BRAKE')
    altered=copy.deepcopy(t)
    for i in range(31,70):altered.metas[i].update(speed=0.)
    after=history.recognize(altered,30,30,'LEAD_BRAKE')
    assert after['action']==classify(altered,30,'LEAD_BRAKE')['action']
    assert after['action']!=before['action']  # Explicitly permitted offline label lookahead.
    assert targets.compress_history([before,after],30)==after['action']
    assert targets.compress_history([before,dict(after,action=None)],30)=='UNKNOWN'
    missing=history.recognize(t,69,30,'LEAD_BRAKE')
    assert missing['action'] is None  # Actual missing native evidence remains missing.
    with pytest.raises(ValueError):history.recognize(t,22,30,'LEAD_BRAKE','confirmed_native')


def test_source_bound_dataset_keeps_native_rules_and_single_current_image(tmp_path,monkeypatch):
    job, rows, art, report=fixture(tmp_path,monkeypatch)
    assert report['anchors']==30
    assert report['native_index_rewritten'] is False and report['RGB_copied'] is False
    for r in rows:
        if dataset.identity(r) not in art['records']:continue
        rec=art['records'][dataset.identity(r)]
        assert rec['image']==r.history_rgb_paths[-1]
        trajectory=rule.load_route_trajectory(Path(job['data_root'])/r.scenario/r.route_id)
        points=[history.recognize(trajectory,r.frame_id+d,r.frame_id,r.context_id) for d in range(-7,1)]
        assert rec['history_original']==[targets.compress_history(points[i:i+2],r.frame_id) for i in range(0,8,2)]
        assert rec['history']==[targets.compress_history(rec['history_resolved_points'][i:i+2],r.frame_id) for i in range(0,8,2)]
        assert all(p['post_observation_evidence_allowed'] for p in rec['history_points'])
        assert len(rec['windows'])==4 and all(w['valid'] for w in rec['windows'])
    p=tmp_path/'single/dataset.json'
    assert dataset.load(p,index=job['index'])['records']==art['records']
    meta=next(Path(job['data_root']).rglob('0001.pkl'));meta.write_bytes(meta.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source changed'):dataset.load(p,index=job['index'])


def record(action='STOP'):
    return dict(event='LEAD_BRAKE',history=['UNKNOWN']*4,
        history_semantics=history.SEMANTICS['offline_native'],premise='Current measured speed: 3 m/s. Allowed A B C F.',
        windows=[targets.distribution([action,action]) for _ in range(4)])


def test_one_image_four_readouts_real_qwen_backward_and_no_future_inputs(processor,model):
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    rec=record();image=Image.new('RGB',(32,32))
    original=objective.messages(rec,image)
    assert objective.messages(rec,image,history=False)==original  # This fixture explicitly uses unknown history.
    modified=copy.deepcopy(rec);modified['windows']=[targets.distribution(['RESUME','RESUME'])]*4
    modified['future_points']=['SECRET']
    assert objective.messages(modified,image)==original
    known=copy.deepcopy(rec);known['history']=['KEEP','STOP','STOP','RESUME']
    assert objective.messages(known,image)!=objective.messages(known,image,history=False)
    packed=objective.pack(bundle,rec,image)
    assert packed['images']==1 and len(packed['positions'])==4
    value,report=objective.loss(bundle,packed)
    value.backward()
    assert torch.isfinite(value) and report['supervised']==4
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_soft_ce_entropy_not_zero_and_anchor_normalization():
    windows=[targets.distribution(['DECELERATE','STOP'],1)]*2+[targets.distribution([None,'STOP'])]*2
    logits=torch.full((4,6),-60.,requires_grad=True)
    with torch.no_grad():logits[:,:2]=0
    loss,report=objective.loss_from_logits(logits,windows,list(range(6)))
    assert loss.item()==pytest.approx(math.log(2))
    assert report['kl_to_training_target']==pytest.approx(0,abs=1e-6)
    assert report['query_weights']==[.5,.5,0,0]
    loss.backward()
    assert logits.grad[2:].count_nonzero()==0
    with pytest.raises(ValueError):objective.loss_from_logits(logits,[targets.distribution([None,None])]*4,list(range(6)))


def test_event_and_native_action_quotas_survive_incomplete_anchor_filter(tmp_path):
    rows=balanced_rows(); cfg=args(tmp_path);cfg.focus_balance_count=32
    recs={dataset.identity(r):dict(record(),id=dataset.identity(r),event=r.context_id,
        anchor_action=baseline_action(r)) for r in rows}
    for r in recs.values():
        for i,w in enumerate(r['windows']):w['source_ids']=[r['id']+str(2*i),r['id']+str(2*i+1)]
    # One eligible replacement exists; do not move its action quota to KEEP.
    damaged=next(iter(recs.values()));damaged['windows']=[targets.distribution([None,None])]*4
    for epoch in range(3):
        _,old=baseline.independent_work(rows,cfg,epoch=epoch,world_size=4)
        work,plan=planning.plan(rows,recs,cfg,epoch=epoch)
        assert plan['event_action_counts']==old['event_action_counts']
        assert plan['event_counts']==dict.fromkeys(train.CONTEXT_IDS,32)
        assert len(work)==320 and damaged['id'] not in plan['ordered_ids']
        assert not plan['least_seen_priority'] and not plan['contiguous_priority']
    for r in recs.values():
        if r['event']==damaged['event']:r['windows']=[targets.distribution([None,None])]*4
    _,audit=planning.plan(rows,recs,cfg,epoch=0,strict=False)
    assert audit['errors'] and audit['lost_anchor_action_support']
    with pytest.raises(ValueError,match='quotas'):planning.plan(rows,recs,cfg,epoch=0)


def baseline_action(row):
    from qwen3vl_local.audit_joint.phase3_base_dense.data import Pool
    return Pool.action(row)


def test_quality_metrics_do_not_use_lateral_training_preference():
    w=targets.distribution(['DECELERATE','LANE_CHANGE_LEFT'],2)
    r=dict(record(),windows=[w]*4,predictions=['DECELERATE']*4)
    report=objective.metrics([r])
    assert report['set_accuracy']['all']==1 and report['raw_mass_score']['all']==.5
    assert report['action_recall']['LANE_CHANGE_LEFT']==0
    assert report['repeat_last_unknown_anchors']==1 and 'repeat_last' not in report['total']


def test_native_training_consumes_four_queries_one_image_and_saves_checkpoint(tmp_path,monkeypatch,processor):
    import json
    from qwen3vl_local.audit_joint.phase3_single_frame import runtime
    from qwen3vl_local.audit_joint import phase3_base_trial as trial
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    job['dataset']=str(tmp_path/'single/dataset.json')
    for name in ('_make_item','_contiguous_work','_loss_batch','SEQUENCE_POLICY','_build_inputs',
                 '_save_adapter','_write_run_metadata','_validation_work','_loss_one','_append_jsonl','verify_epoch_consumption','select_history_rgb_paths'):
        monkeypatch.setattr(train,name,getattr(train,name))
    monkeypatch.setattr(preflight,'check_model',lambda *a:None)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.zeros(320))
        def forward(self,input_ids,logits_to_keep,**kw):
            assert kw['image_grid_thw'].shape[0]==1
            return SimpleNamespace(logits=self.values[None,None].expand(1,len(logits_to_keep),-1))
    def load(*a,**kw):
        m=Tiny();return SimpleNamespace(model=m,unwrap=lambda:m,lora_target_modules=['values'],
            processor=processor,tokenizer=processor.tokenizer,device='cpu')
    monkeypatch.setattr(train,'load_model_with_lora',load)
    runtime.install(artifact,job['output'])
    cfg=runtime.make_args(job,artifact);cfg.device='cpu';cfg.focus_balance_count=1;cfg.warmup_steps=2
    cfg.prompt_variant='baseline';cfg.eval_steps=cfg.generation_eval_steps=0;cfg.save_final=False
    train.train(cfg)
    root=Path(cfg.output_dir)
    state=torch.load(root/'training_states/epoch_0001/training_state.pt',weights_only=True)
    assert state['global_step']==10 and state['trainable']['values'].abs().sum()>0
    report=trial.read(root/'balance/single_frame_consumed_epoch_0001.json')
    assert sum(report['actual_query_exposure'].values())==40
    batches=[json.loads(line) for line in (root/'epoch_0001_rank0_batches.jsonl').read_text().splitlines()]
    assert len(batches)==10 and all(b['single_frame']['rgb_count']==1 for b in batches)


def test_evaluation_no_grad_or_parameter_change(tmp_path,monkeypatch,processor,model):
    from qwen3vl_local.audit_joint.phase3_single_frame import runtime
    _,_,art,_=fixture(tmp_path,monkeypatch)
    rec=next(iter(art['records'].values()))
    monkeypatch.setattr(train,'_load_images',lambda *a:[Image.new('RGB',(32,32))])
    bundle=SimpleNamespace(model=model,processor=processor,tokenizer=processor.tokenizer,device='cpu')
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    model.train()
    cases,report=runtime.predict(bundle,[rec])
    assert model.training and len(cases[0]['predictions'])==4 and report['total']['all']==4
    assert set(cases[0]['predictions'])<=set(train.CONTEXT_BY_ID[rec['event']].action_keys)|{'KEEP'}
    assert report['distribution_fit']['all/raw_cross_entropy']>=0
    for k,v in model.named_parameters():
        assert v.grad is None
        torch.testing.assert_close(v,before[k])


def test_future_native_actions_unchanged_by_lateral_preference_and_exact_time_pairs(tmp_path,monkeypatch):
    job,rows,art,_=fixture(tmp_path,monkeypatch)
    for row in rows:
        if dataset.identity(row) not in art['records']:continue
        trajectory=rule.load_route_trajectory(Path(job['data_root'])/row.scenario/row.route_id)
        rec=dataset.make_record(row,trajectory,[],history_policy='offline_native',lateral_weight=2.)
        old=art['records'][rec['id']]
        assert rec['future_points']==old['future_points']
        assert [w['raw'] for w in rec['windows']]==[w['raw'] for w in old['windows']]
        assert [w['offsets_seconds'] for w in rec['windows']]==[[.25,.5],[.75,1.],[1.25,1.5],[1.75,2.]]
        assert all(p['anchor_context_id']==row.context_id for p in rec['future_points'])


def test_current_rgb_mutation_is_bound(tmp_path,monkeypatch):
    job,rows,art,_=fixture(tmp_path,monkeypatch)
    image=Path(next(iter(art['records'].values()))['image'])
    image.write_bytes(image.read_bytes()+b'mutation')
    with pytest.raises(ValueError,match='source changed'):dataset.load(tmp_path/'single/dataset.json')


def test_zero_target_evaluation_is_skipped_not_counted_as_correct(tmp_path,monkeypatch,processor):
    from qwen3vl_local.audit_joint.phase3_single_frame import runtime
    job,rows,art,_=fixture(tmp_path,monkeypatch)
    for name in ('_make_item','_contiguous_work','_loss_batch','SEQUENCE_POLICY','_build_inputs',
                 '_save_adapter','_write_run_metadata','_validation_work','_loss_one','_append_jsonl','verify_epoch_consumption','select_history_rgb_paths'):
        monkeypatch.setattr(train,name,getattr(train,name))
    row=next(r for r in rows if r.split=='val' and dataset.identity(r) in art['records'])
    art['records'][dataset.identity(row)]['windows']=[targets.distribution([None,None])]*4
    runtime.install(art,job['output'])
    item=train._make_item(row,seed=20260904,action_output_mode='choice')
    assert train._build_inputs(None,spec=item.spec) is None


def test_cpu_resource_scan_contains_train_and_fixed_evaluation(tmp_path,monkeypatch,processor):
    from qwen3vl_local.audit_joint.phase3_single_frame import worker
    from qwen3vl_local.sft_new_loop_phase3 import preflight
    monkeypatch.setattr(preflight,'check_model',lambda *a:dict(fixture=True))
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    job['dataset']=str(tmp_path/'single/dataset.json')
    needed={k:('train' if r['split']=='train' else 'evaluation') for k,r in artifact['records'].items()}
    monkeypatch.setattr(worker,'planned',lambda *a:({'fixture':True},needed))
    monkeypatch.setattr(train,'_load_images',lambda *a:[Image.new('RGB',(32,32))])
    result=worker.preflight(job,artifact,processor)
    assert result['action_token_identity']['status']=='passed'
    assert set(result['action_token_identity']['checks'])==set(objective.ACTIONS)|{'UNKNOWN'}
    assert result['contract']['inference']['unknown_policy']['random_history_masking'] is False
    assert len(result['shapes'])==sum(1 if v=='train' else 4 for v in needed.values())
    assert {result['shapes'][k]['usage'] for k in result['probe_ids']}=={'train','evaluation'}
    assert all(len(v['positions'])==4 and len(v['grid'])==1 for v in result['shapes'].values())


def test_fixed_eval_insufficient_distinct_pairs_rejected_before_gpu(tmp_path,monkeypatch):
    from qwen3vl_local.audit_joint.phase3_single_frame import worker
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    job['dataset']=str(tmp_path/'single/dataset.json')
    with pytest.raises(ValueError,match='distinct supported pairs'):worker.planned(job,artifact)
    monkeypatch.setenv('WORLD_SIZE','1')
    with pytest.raises(ValueError,match='four ranks'):worker.probe(job,artifact,{})


def gap_fixture(actions, quarantined=(), missing=()):
    def lookup(f):
        action=actions[f] if 0<=f<len(actions) else None
        reason='native' if action is not None else 'mixed_longitudinal_phase'
        if f in quarantined:action=None;reason='source_quarantine'
        return dict(frame=f,action=action,reason=reason,required_end_frame=f+8)
    exists=lambda f:0<=f<len(actions) and f not in missing
    points=[lookup(f) for f in range(len(actions))]
    return points,lookup,exists


@pytest.mark.parametrize('mode,expected',[
    ('none',['KEEP',None,None,'KEEP']),
    ('same_short',['KEEP','KEEP','KEEP','KEEP']),
    ('nearest_short',['KEEP','KEEP','KEEP','KEEP'])])
def test_history_same_endpoint_short_fill_and_original_provenance(mode,expected):
    points,lookup,exists=gap_fixture(['KEEP',None,None,'KEEP'])
    original=copy.deepcopy(points)
    filled=history.fill_short_gaps(points,lookup,exists,mode)
    assert points==original and [p['action'] for p in filled]==expected
    assert not filled[0]['imputed'] and not filled[-1]['imputed']
    if mode!='none':
        for p in filled[1:3]:
            assert p['original_action'] is None and p['original_reason']=='mixed_longitudinal_phase'
            assert p['imputation']['missing_duration_seconds']==.5
            assert p['imputation']['left_frame']==0 and p['imputation']['right_frame']==3
            assert p['imputation']['native_label'] is False


def test_history_short_different_endpoint_nearest_with_later_tie():
    points,lookup,exists=gap_fixture(['STOP',None,None,'RESUME'])
    assert [p['action'] for p in history.fill_short_gaps(points,lookup,exists,'same_short')]==['STOP',None,None,'RESUME']
    out=history.fill_short_gaps(points,lookup,exists,'nearest_short')
    assert [p['action'] for p in out]==['STOP','STOP','RESUME','RESUME']
    assert all(p['imputation']['kind']=='nearest_transition' for p in out[1:3])
    points,lookup,exists=gap_fixture(['STOP',None,'RESUME'])
    assert history.fill_short_gaps(points,lookup,exists,'nearest_short')[1]['action']=='RESUME'


@pytest.mark.parametrize('actions,quarantined,missing',[
    (['KEEP',None,None,None,'KEEP'],(),()),  # Long gap cannot be shortened by cascading fills.
    ([None,None,'KEEP'],(),()),  # No one-sided extrapolation.
    (['KEEP',None,None],(),()),
    (['KEEP',None,'KEEP'],(1,),()),
    (['KEEP',None,'KEEP'],(0,),()),
    (['KEEP',None,'KEEP'],(),(1,)),
])
def test_history_fill_preserves_missing_boundaries_quarantine_and_long_gaps(actions,quarantined,missing):
    points,lookup,exists=gap_fixture(actions,quarantined,missing)
    for mode in ('same_short','nearest_short'):
        out=history.fill_short_gaps(points,lookup,exists,mode)
        assert all(not p['imputed'] for p in out)
        assert [p['action'] for p in out]==[p['action'] for p in points]


def test_history_fill_never_erases_known_short_action_or_changes_future(tmp_path,monkeypatch):
    points,lookup,exists=gap_fixture(['KEEP','STOP','KEEP'])
    assert [p['action'] for p in history.fill_short_gaps(points,lookup,exists,'nearest_short')]==['KEEP','STOP','KEEP']
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    row=replace(rows[0],frame_id=10);trajectory=rule.load_route_trajectory(Path(job['data_root'])/row.scenario/row.route_id)
    fake=lambda t,f,c:dict(action=None if f==row.frame_id-1 else 'KEEP',reason='mixed_longitudinal_phase' if f==row.frame_id-1 else 'native',evidence={})
    monkeypatch.setattr(history,'classify',fake)
    raw=dataset.make_record(row,trajectory,[],history_policy='offline_native',lateral_weight=1.5,history_fill='none')
    filled=dataset.make_record(row,trajectory,[],history_policy='offline_native',lateral_weight=1.5)
    assert filled['history_points']==raw['history_points']
    assert filled['windows']==raw['windows'] and filled['future_points']==raw['future_points']
    assert filled['history_original']==raw['history']
    assert filled['history_changed'][-1] and filled['history_imputed'][-1]
    assert filled['history'][-1]=='KEEP'


def test_nearest_history_is_default_and_comparison_modes_remain_explicit(tmp_path,monkeypatch):
    import sys
    from qwen3vl_local.audit_joint.phase3_single_frame import __main__ as cli
    assert history.DEFAULT_FILL=='nearest_short'
    points,lookup,exists=gap_fixture(['STOP',None,None,'RESUME'])
    assert [p['action'] for p in history.fill_short_gaps(points,lookup,exists)]==['STOP','STOP','RESUME','RESUME']
    job,rows,artifact,_=fixture(tmp_path,monkeypatch)
    assert artifact['config']['history_fill']['mode']=='nearest_short'
    for mode in (None,'same_short','none'):
        output=tmp_path/('cli_'+str(mode))
        argv=['single_frame','build','--index',job['index'],'--data-root',job['data_root'],'--output',str(output)]
        if mode:argv+=['--history-fill',mode]
        monkeypatch.setattr(sys,'argv',argv)
        cli.main()
        built=dataset.load(output/'dataset.json',index=job['index'])
        assert built['config']['history_fill']['mode']==(mode or 'nearest_short')
        for key,r in built['records'].items():
            assert r['future_points']==artifact['records'][key]['future_points']
            assert r['history_points']==artifact['records'][key]['history_points']
