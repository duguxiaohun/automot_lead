from collections import Counter
from copy import deepcopy
from itertools import product
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import candidate_pool as cp,dataset
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,frame_key,CapacityError
from qwen3vl_local.sft_new_loop_phase4.evaluate import metrics
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,file_sha
from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS


def row(i,event,frame=None):
    return dict(id=f'{event}-{i}',episode={'event':event},edge=f'edge{i%3}',target='YES' if i%2 else 'NO',
                slice='readiness' if i%3 else 'catchup',scenario='S',route_id='R',physical_group=f'g{i%2}',
                observation={'frame_id':i if frame is None else frame})


@pytest.mark.parametrize('world,budget',[(1,None),(4,None),(1,100),(4,100),(1,10)])
def test_event_equal_on_unequal_pool_with_repetition_and_global_cap(world,budget):
    rows=[row(j+e*100,event) for e,event in enumerate(EVENTS) for j in range(3+e)]
    for epoch in range(4):
        ids,a=plan(rows,epoch=epoch,world_size=world,budget=budget,cap=8,expected_events=EVENTS)
        counts=Counter(rows[i]['episode']['event'] for i in ids)
        assert len(counts)==10 and len(set(counts.values()))==1
        assert len(ids)%world==0 and max(Counter(frame_key(rows[i]) for i in ids).values())<=8
        assert ids==plan(rows,epoch=epoch,world_size=world,budget=budget,cap=8)[0]
        rev=list(reversed(rows));other,_=plan(rev,epoch=epoch,world_size=world,budget=budget,cap=8)
        assert [rows[i]['id'] for i in ids]==[rev[i]['id'] for i in other]
        assert a['event_counts']==counts


@pytest.mark.parametrize('budget',[1,11,19,21])
def test_non_equal_budget_rejected(budget):
    with pytest.raises(ValueError,match='positive multiple'):
        plan([row(i,e) for i,e in enumerate(EVENTS)],epoch=0,budget=budget)


def test_shared_frame_bottleneck_and_reassignment():
    rows=[row(0,'U-E1',0),row(1,'U-E1',1),row(2,'U-E2',0)]
    ids,_=plan(rows,epoch=0,budget=2,cap=1)
    assert {rows[i]['episode']['event'] for i in ids}=={'U-E1','U-E2'}
    assert {frame_key(rows[i])[-1] for i in ids}=={0,1}
    with pytest.raises(CapacityError) as error:plan(rows,epoch=0,budget=4,cap=1)
    assert error.value.diagnostic['global_frame_capacity']==2
    with pytest.raises(ValueError,match='pool mismatch'):plan(rows,epoch=0,expected_events=EVENTS)


def test_small_topologies_match_bruteforce_capacity():
    # Both events can use arbitrary nonempty subsets of three shared frames.
    for mask_a,mask_b in product(range(1,8),repeat=2):
        choices=[[f for f in range(3) if mask&(1<<f)] for mask in (mask_a,mask_b)]
        rows=[row(e*3+f,f'U-E{e+1}',f) for e,fs in enumerate(choices) for f in fs]
        for cap in (1,2):
            for quota in (1,2):
                feasible=any(max(Counter(a+b).values())<=cap
                    for a in product(choices[0],repeat=quota) for b in product(choices[1],repeat=quota))
                try:ids,_=plan(rows,epoch=3,budget=quota*2,cap=cap)
                except CapacityError:assert not feasible
                else:
                    assert feasible and Counter(rows[i]['episode']['event'] for i in ids)=={'U-E1':quota,'U-E2':quota}


def test_small_equal_budget_rotates_each_event_entire_pool():
    rows=[row(e*100+i,event) for e,event in enumerate(EVENTS) for i in range(e+2)]
    seen=set()
    for epoch in range(11):seen.update(plan(rows,epoch=epoch,budget=10,cap=1)[0])
    assert seen==set(range(len(rows)))


def test_event_score_is_invariant_to_subgroup_count_and_missing_is_explicit():
    records=[dict(event='U-E1',edge=f'edge{i}',slice='readiness',target='YES',prediction='YES') for i in range(8)]
    records+=[dict(event='U-E2',edge='enter',slice='readiness',target='YES',prediction='NO')]
    r=metrics(records)
    assert r['event_macro_accuracy_observed']==.5 and r['event_macro_accuracy'] is None
    assert set(r['missing_events'])==set(EVENTS)-{'U-E1','U-E2'}
    for e in r['missing_events']:records.append(dict(event=e,edge='x',slice='readiness',target='NO',prediction='NO'))
    assert metrics(records)['event_macro_accuracy']==.9
    assert metrics([])['event_macro_accuracy_observed'] is None


def inventory(tmp_path):
    root=tmp_path/'source'
    for name in ('Town01_Rep0_1_0_route0','Town01_Rep1_1_0_route0','Town02_Rep0_2_0_route0'):
        for kind,suffix in [('rgb','jpg'),('metas','pkl')]:
            d=root/'Scenario'/name/kind;d.mkdir(parents=True)
            for i in range(4,12):(d/f'{i:04d}.{suffix}').write_bytes(b'fixture only')
    return root,cp.scan(root,tmp_path/'pool.json')


def test_inventory_freezes_all_routes_and_unknown_frames_are_not_no(tmp_path):
    _,pool=inventory(tmp_path)
    assert pool['summary']['routes']==3 and pool['summary']['rgb_frames']==24
    assert pool['routes'][0]['physical_group']==pool['routes'][1]['physical_group']
    assert pool['routes'][0]['split']==pool['routes'][1]['split']
    report=cp.coverage(pool,[])
    assert report['frames_without_questions']==24 and report['frames_with_supervision']==0
    assert not report['complete_causal_production']
    with pytest.raises(FileExistsError):cp.scan(tmp_path/'source',tmp_path/'pool.json')


@pytest.mark.parametrize('fault',['split','route','frame','hash'])
def test_inventory_rejects_labels_outside_frozen_pool(tmp_path,fault):
    _,pool=inventory(tmp_path);r=pool['routes'][0]
    item=dict(scenario=r['scenario'],route_id=r['route_id'],split=r['split'],target='YES',
              observation=dict(frame_id=10,history_frames=[6,10]))
    assert cp.coverage(pool,[item])['frames_with_supervision']==1
    if fault=='split':item['split']='val' if r['split']!='val' else 'train'
    elif fault=='route':item['route_id']='missing'
    elif fault=='frame':item['observation']['history_frames']=[2,10]
    else:pool['routes'][0]['split']='tampered'
    with pytest.raises(ValueError):cp.coverage(pool,[item])


def test_new_risk_routes_registered_without_supervision():
    from qwen3vl_local.sft_new_loop_phase4.risk_review import registry,annotation_review
    audit=json.loads((ROOT/'fourteenth_audit_exposure_20260930.json').read_text());known=registry()
    used={(a['scenario'],a['route_id']) for a in json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())}
    assert len(audit['calibration_risks'])==15
    assert len({r['route_id'] for r in audit['calibration_risks']})==11
    assert set(audit['train_only_groups'])<=dataset.groups()
    for r in audit['calibration_risks']:
        key=r['scenario'],r['route_id'];assert key not in used and key in known
        for f in r['rgb_evidence']:assert file_sha(ROOT.parents[1]/'lead_data'/f['path'])==f['sha256']
        a=dict(scenario=r['scenario'],route_id=r['route_id'],label_basis='reviewed_transition_band',
               transition_band=dict(review_start=10,review_end=10,stop={'frame':11}))
        for mode in (2,4):
            with pytest.raises(ValueError,match='explicit per-frame'):
                annotation_review(a,ROOT.parents[1]/'lead_data',mode,known)


def test_builder_and_loader_bind_pool_without_changing_labels(tmp_path):
    # Small real-RGB fixture copied from an already reviewed training interval.
    import shutil
    ann=json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())[0]
    source=ROOT.parents[1]/'lead_data'/ann['scenario']/ann['route_id']
    dest=tmp_path/'data'/ann['scenario']/ann['route_id']
    for kind in ('rgb','metas'):shutil.copytree(source/kind,dest/kind)
    pool=cp.scan(tmp_path/'data',tmp_path/'inventory.json')
    dataset.build([ann],tmp_path/'data',tmp_path/'built',rgb_mode=2,candidate_pool=pool)
    data,m=dataset.load_dataset(tmp_path/'built')
    assert m['production_coverage']['candidate_routes']==1
    assert m['production_coverage']['frames_without_questions']>0
    assert m['candidate_pool_sha256']==file_sha(tmp_path/'built/candidate_pool.json')
    path=tmp_path/'built/candidate_pool.json';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='candidate pool hash'):dataset.load_dataset(tmp_path/'built')


def test_selector_rejects_old_group_weighted_score(tmp_path,monkeypatch):
    import qwen3vl_local.qwen35.adapters as adapters
    from qwen3vl_local.sft_new_loop_phase4.selection import select_best
    from qwen3vl_local.sft_new_loop_phase4.identity import contract,write_json
    from qwen3vl_local.sft_new_loop_phase4.observation import observation_contract
    c=dict(source=contract(),dataset='test-dataset',observation_contract=observation_contract(2))
    write_json(tmp_path/'run.json',c)
    monkeypatch.setattr(adapters,'validate_adapter',lambda *a,**k:None)
    for epoch,score,metric in [(0,.5,'event_macro_accuracy_observed_v1'),(1,.99,'old_group_macro')]:
        write_json(tmp_path/f'epoch_{epoch:03}'/'phase4_contract.json',c)
        write_json(tmp_path/f'epoch_{epoch:03}_validation.json',dict(count=10,expected_count=10,
             macro_score=score,event_macro_accuracy_observed=score,selection_metric=metric,
             split='val',dataset='test-dataset'))
    assert select_best(tmp_path,{}).name=='epoch_000'
