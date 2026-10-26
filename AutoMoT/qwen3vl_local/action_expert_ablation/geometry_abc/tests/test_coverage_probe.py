import copy
import json
from types import SimpleNamespace
import pytest
import torch
from .test_core import record,tiny,inputs
from qwen3vl_local.action_expert_ablation.geometry_abc import diagnostic_coverage as coverage
from qwen3vl_local.action_expert_ablation.geometry_abc import diagnostics,probe_inputs,runtime
from qwen3vl_local.audit_joint.abc_delivery import sha,write


def supported_rows():
    rows=[]
    for region in range(9):
        for group in range(5):
            r=record(region*5+group,split='val');r['physical_group']=f'g{region}_{group}'
            for v in 'abc':
                for h,t in enumerate(r['representations'][v]['targets']):t['token_id']=(region+h)%9
                for h,t in enumerate(r['representations'][v]['history']):t['token_id']=(region+h)%9
            rows.append(r)
    return rows


def cases_for(rows,value=1.):
    return [dict(id=r['id'],physical_group=r['physical_group'],labels=coverage.labels(r,'a','future'),
        history_labels=coverage.labels(r,'a','history'),slot_responses={role:[value]*4 for role in coverage.ROLES},
        results={k:dict(mean_waypoint_shift_from_oracle_m=value) for k in ('replace_regions','permute_horizons')}) for r in rows]


def test_all_supported_cells_have_four_groups_deterministic_and_bounded():
    rows=supported_rows();selected,report=coverage.select(rows,'a')
    assert coverage.select(list(reversed(rows)),'a')==(selected,report)
    assert len(selected)<=report['maximum_records']
    assert all(c['selected_groups']>=4 for c in report['cells'].values() if c['support_records'])
    assert sum(c['status']=='N/A' for c in report['cells'].values())==4
    rows[0]['representations']['a']['history'][0]['valid']=False
    _,report=coverage.select(rows,'a');cell=report['cells']['history/0/9']
    assert cell['selected_records']==1 and cell['sparse_group_support']


def test_one_responsive_case_cannot_hide_inactive_cells_or_samples():
    selected,report=coverage.select(supported_rows(),'a');cases=cases_for(selected,0.)
    cases[0]=cases_for(selected[:1],.1)[0]
    assert diagnostics.summarize(cases)['status']=='observed_condition_response'  # old global check
    result=diagnostics.summarize(cases,report)
    assert result['status']=='condition_use_review_required'
    assert result['role_horizon_region']['review_cells']
    assert any(0<c['nonzero_fraction']<1 for c in result['role_horizon_region']['cells'].values() if c['cases'])
    assert diagnostics.summarize(cases_for(selected),report)['status']=='observed_condition_response'
    with pytest.raises(ValueError,match='members'):diagnostics.summarize(cases[:-1],report)
    cases[0]['slot_responses']['history'][0]=float('nan')
    with pytest.raises(ValueError,match='invalid_condition_response'):diagnostics.summarize(cases,report)


@pytest.mark.parametrize('variant',list('abc'))
def test_actual_runtime_isolates_each_history_and_future_slot_same_noise(variant):
    torch.manual_seed(41);m=tiny(variant);m.eval();r=record();seen=[]
    class Inner:
        runner=SimpleNamespace()
        def forward_sample(self,sample,decoder,config,dtype,**kwargs):
            kw=inputs(m,r,variant)
            for key in ('geometry_token_ids','geometry_reference_yaw','history_token_ids','history_mask','representation_contract'):kw.pop(key)
            kwargs.pop('clip');return decoder(**kw,**kwargs)
    rt=runtime.Runtime.__new__(runtime.Runtime);rt.inner=Inner();rt.device=torch.device('cpu')
    rt.variant=variant;rt.branch='bev_only';rt.intervention=None
    def capture(module,args,kw):seen.append({k:kw[k].clone() for k in ('geometry_token_ids','history_token_ids','history_mask')})
    handle=m.register_forward_pre_hook(capture,with_kwargs=True);noise=torch.randn(1,18,2)
    def run():
        with torch.no_grad():return rt.forward_sample(runtime.sample(r,'/unused'),m,m.config,torch.float32,
            clip={},flow_sample_noise=noise,sample_trajectory=True)['pred_future_waypoints']
    baseline=run();original=seen[-1]
    for role in ('future','history'):
        for slot in range(4):
            rt.intervention=dict(role=role,slot=slot);result=run();actual=seen[-1]
            for key in ('geometry_token_ids','history_token_ids'):
                expected=original[key].clone()
                if key==('geometry_token_ids' if role=='future' else 'history_token_ids'):expected[:,slot]=(expected[:,slot]+1)%9
                assert torch.equal(actual[key],expected)
            assert float((result-baseline).norm())>1e-6
    handle.remove()


def fixture_processor():
    rows=[dict(id=f'train{i}',split='train') for i in range(6)]+[dict(id=s,split=s) for s in ('val','test')]
    extrema={}
    for v in 'abc':
        for i,r in enumerate(rows):
            probe_inputs.update(extrema,r,v,dict(tokens=100+i,visual_tokens=999 if i==4 else 10+i,image_grid_thw=[[1,2,2]]))
    return rows,dict(extrema=extrema)


def test_probe_selects_train_text_and_visual_maxima_holdout_forward_only():
    rows,processor=fixture_processor()
    for v in 'abc':
        selected=probe_inputs.selection(rows,processor,v)
        assert [x['id'] for x in selected['backward']]==['train5','train4','train0','train1']
        assert all(x['split']=='train' for x in selected['backward'])
        assert {x['id'] for x in selected['holdout_forward_only']}=={'val','test'}
    processor['extrema']['a']['train']['tokens']['id']='test'
    with pytest.raises(ValueError,match='split_or_member'):probe_inputs.selection(rows,processor,'a')


def test_probe_consumed_shape_and_rank_evidence_are_rechecked(tmp_path):
    rows,p=fixture_processor();selected=probe_inputs.selection(rows,p,'a')
    receipt=dict(input_selection=selected,processor_sha256='p',binding={'branch':'qwen_simple'},rank_receipts={},
        post_checkpoint_backward_verified=True,loader_workers=0)
    for rank,item in enumerate(selected['backward']):
        dims={k:item.get(k,1) for k in ('tokens','visual_tokens','image_grid_thw')}
        proof=dict(loader=dict(configured_workers=0,decoded_copies=1,formal_training_exposures=0,lifetime='through checkpoint reload'),
            input=dict(id=item['id'],split='train',processor=dims,rgb_shape=[1,2,2,3],lidar_points=[4],decoder_shapes={'bev':[1,2]}),
            post_checkpoint_backward=dict(id=item['id'],split='train',optimizer_updates=0,loss=1.),
            holdout_forward_only=[dict(h,updates=0,decoder_shapes={'bev':[1,2]}) for h in selected['holdout_forward_only'][rank::4]])
        write(tmp_path/f'rank{rank}.json',proof);receipt['rank_receipts'][f'rank{rank}.json']=sha(tmp_path/f'rank{rank}.json')
    probe_inputs.verify_probe_inputs(receipt,selected,'p',tmp_path)
    receipt['post_checkpoint_backward_verified']=False
    with pytest.raises(ValueError,match='post_checkpoint_backward_evidence_required'):
        probe_inputs.verify_probe_inputs(receipt,selected,'p',tmp_path)
    receipt['post_checkpoint_backward_verified']=True
    path=tmp_path/'rank0.json';proof=json.loads(path.read_text());proof['input']['processor']['tokens']-=1
    write(path,proof);receipt['rank_receipts'][path.name]=sha(path)
    with pytest.raises(ValueError,match='shape_changed'):probe_inputs.verify_probe_inputs(receipt,selected,'p',tmp_path)
    proof['input']['processor']['tokens']+=1;proof['holdout_forward_only'][0]['tokens']+=1
    write(path,proof);receipt['rank_receipts'][path.name]=sha(path)
    with pytest.raises(ValueError,match='coverage_changed'):probe_inputs.verify_probe_inputs(receipt,selected,'p',tmp_path)


def test_runtime_probe_observes_actual_qwen_packing_and_restores_hook(monkeypatch):
    from qwen3vl_local.action_prior import image_condition
    r=record();m=tiny();engine=SimpleNamespace(processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)))
    packed={'input_ids':torch.ones(1,123,dtype=torch.long),'image_grid_thw':torch.tensor([[1,4,6]])}
    engine.prepare_inputs=lambda *a,**kw:packed;original=engine.prepare_inputs
    runner=SimpleNamespace(leadmot_qwen_engine=engine,_run_leadmot_qwen_prefill=lambda *a:None)
    monkeypatch.setattr(image_condition,'current_base_prefill',lambda *a:engine.prepare_inputs('actual input'))
    class Inner:
        def forward_sample(self,sample,decoder,config,dtype,**kwargs):
            runner._run_leadmot_qwen_prefill(['image'],'unused')
            kw=inputs(m,r)
            for key in ('geometry_token_ids','geometry_reference_yaw','history_token_ids','history_mask','representation_contract'):kw.pop(key)
            return decoder(**kw)
    inner=Inner();inner.runner=runner;rt=runtime.Runtime.__new__(runtime.Runtime)
    rt.inner=inner;rt.device=torch.device('cpu');rt.variant='a';rt.branch='qwen_simple';rt.intervention=None;rt.args=SimpleNamespace(operation='probe')
    rt.forward_sample(runtime.sample(r,'/unused'),m,m.config,torch.float32,clip={})
    assert rt.last_processor_shape==dict(tokens=123,visual_tokens=6,image_grid_thw=[[1,4,6]])
    assert engine.prepare_inputs is original and rt.last_decoder_shapes['prefix_kv']


def test_production_sensitivity_writes_eight_slot_responses_with_same_noise(tmp_path,monkeypatch):
    from qwen3vl_local.action_expert_ablation import common
    from qwen3vl_local.action_expert_ablation.geometry_abc import loader
    torch.manual_seed(23);m=tiny();r=record(split='val');observed=[]
    class Dataset:
        def __init__(self,*a):pass
        def __getitem__(self,i):return dict(gt_route=torch.ones(10,2),gt_waypoints=torch.ones(8,2),clip={})
    monkeypatch.setattr(loader,'Dataset',Dataset)
    monkeypatch.setattr(common,'leadmot_train',lambda:SimpleNamespace(_compute_planning_metrics=lambda *a:{'fixture_metric':0.}))
    class Inner:
        runner=SimpleNamespace()
        def forward_sample(self,sample,decoder,config,dtype,**kw):
            observed.append(kw['flow_sample_noise'].clone());kw.pop('clip')
            packed=inputs(m,r)
            for key in ('geometry_token_ids','geometry_reference_yaw','history_token_ids','history_mask','representation_contract'):packed.pop(key)
            return decoder(**packed,**kw)
    rt=runtime.Runtime.__new__(runtime.Runtime);rt.inner=Inner();rt.device=torch.device('cpu')
    rt.variant='a';rt.branch='bev_only';rt.intervention=None
    path=tmp_path/'condition.json'
    diagnostics.sensitivity(rt,m,m.config,[runtime.sample(r,'/unused')],SimpleNamespace(seed=42),torch.float32,0,1,path)
    report=json.loads(path.read_text())
    assert len(observed)==12 and all(torch.equal(observed[0],v) for v in observed)
    assert report['summary']==diagnostics.summarize(report['cases'],report['coverage'])
    assert all(len(report['cases'][0]['slot_responses'][role])==4 for role in ('future','history'))
    assert rt.intervention is None and m.training
