"""Exact split coverage recovery, independently checked against enumeration."""
import copy
import itertools
import pickle
import random

from PIL import Image
import pytest

from qwen3vl_local.audit_joint.geometry_v1 import coverage_search, dataset, splits, train
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
from .test_geometry import Lanes, meta

MASKS = (699, 376, 892, 983, 915, 254)


def rows_for(masks=MASKS):
    return [dict(id=f'{group}:{event}', scenario='Synthetic', route_id=f'Town00_{group}',
                 frame=8, event=event, split='train', targets=[dict(valid=True)],
                 physical_group=f'Synthetic/Town00_{group}')
            for group, mask in enumerate(masks) for i, event in enumerate(CONTEXT_IDS) if mask & (1 << i)]


def test_six_routes_ten_events_backtrack_greedy_counterexample():
    rows=rows_for()
    report=splits.complete(rows,[])
    assert report['feasibility']=='feasible' and report['search']['method']=='exact_event_mask_backtracking_v1'
    assert 'LEAD_BRAKE' in report['greedy_attempt']['unresolved']['val']
    assert not report['unresolved']
    dataset.validate_splits(rows)
    for split in ('train','val','test'):
        assert {r['event'] for r in rows if r['split']==split}==set(CONTEXT_IDS)
    # Known independently enumerated witness: (train,val,train,val,test,test).
    for color in range(3):
        union=0
        for mask,assigned in zip(MASKS,(0,1,0,1,2,2)):
            if assigned==color:union|=mask
        assert union==1023
    reversed_rows=list(reversed(rows_for()))
    assert splits.complete(reversed_rows,[])==report
    assert {r['id']:r['split'] for r in reversed_rows}=={r['id']:r['split'] for r in rows}


def brute(masks,origins,development,event_count):
    names=('train','val','test')
    movable=[g for g in masks if origins[g]=='train' and g not in development]
    for choices in itertools.product(names,repeat=len(movable)):
        assigned=dict(origins);assigned.update(zip(movable,choices));unions={s:0 for s in names}
        for group,mask in masks.items():unions[assigned[group]]|=mask
        if all(v==(1<<event_count)-1 for v in unions.values()):return True
    return False


def check_against_brute(masks,origins,development,event_count):
    feasible=brute(masks,origins,development,event_count)
    assignment,report=coverage_search.solve(masks,origins,development,event_count=event_count,seed=17)
    assert report['status']==('feasible' if feasible else 'infeasible'),(masks,origins,development,report)
    if feasible:
        assert assignment is not None
        for group in masks:
            if group in development:assert assignment[group]=='train'
            if origins[group]!='train':assert assignment[group]==origins[group]
        for split in ('train','val','test'):
            union=0
            for group,mask in masks.items():
                if assignment[group]==split:union|=mask
            assert union==(1<<event_count)-1
    else:assert assignment is None


def test_exact_search_matches_all_2401_four_route_three_event_graphs():
    for values in itertools.product(range(1,8),repeat=4):
        masks=dict(enumerate(values));origins=dict.fromkeys(masks,'train')
        check_against_brute(masks,origins,set(),3)


def test_fixed_holdouts_and_development_match_independent_enumeration():
    rng=random.Random(20261021)
    for _ in range(200):
        masks={str(i):rng.randrange(1,16) for i in range(6)}
        origins={g:rng.choice(('train','train','train','val','test')) for g in masks}
        development={g for g in masks if origins[g]=='train' and rng.random()<.4}
        check_against_brute(masks,origins,development,4)


def test_unknown_and_infeasible_are_distinct_and_do_not_publish_partial_moves():
    rows=rows_for();before=copy.deepcopy(rows)
    report=splits.complete(rows,[],max_search_states=1)
    assert report['feasibility']=='unknown' and report['search']['visited_states']==1
    assert report['moves']=={} and rows==before
    with pytest.raises(RuntimeError,match='coverage_search_inconclusive'):
        splits.require_coverage(report,'candidate')
    rows=rows_for((1023,1023))
    before=copy.deepcopy(rows);report=splits.complete(rows,[])
    assert report['feasibility']=='infeasible' and rows==before and report['moves']=={}
    with pytest.raises(ValueError,match='proven infeasible'):
        splits.require_coverage(report,'geometry_after_filtering')


def test_duplicate_masks_compressed_without_changing_coverage():
    masks={str(i):1023 for i in range(5000)}
    assignment,report=coverage_search.solve(masks,dict.fromkeys(masks,'train'),set(),event_count=10,seed=17)
    assert report['representatives']==3 and report['event_signatures']==1
    assert report['status']=='feasible' and sum(s!='train' for s in assignment.values())==2


def test_joint_infeasibility_despite_three_routes_for_every_event():
    # Every three-vertex subset would need all three split colors: impossible
    # on four vertices, even though each event has three candidate routes.
    masks=dict(enumerate((7,11,13,14)))
    assignment,report=coverage_search.solve(masks,dict.fromkeys(masks,'train'),set(),event_count=4,seed=17)
    assert not brute(masks,dict.fromkeys(masks,'train'),set(),4)
    assert assignment is None and report['status']=='infeasible' and report['visited_states']>1


def test_greedy_success_preserved_and_invalid_members_move_with_whole_group():
    from qwen3vl_local.sft_new_loop_phase3.split_coverage import complete_context_splits
    legacy=rows_for((1023,1023,1023))
    for r in legacy:r['context_id']=r['event']
    before=copy.deepcopy(legacy)
    old=complete_context_splits(legacy,contexts=CONTEXT_IDS,splits=('train','val','test'),
        development=set(),group_of=lambda r:r['physical_group'],seed=20260920)
    report=splits.complete(before,[])
    assert report['search']['method']=='greedy'
    assert all(report[key]==value for key,value in old.items())
    assert [r['split'] for r in before]==[r['split'] for r in legacy]
    rows=rows_for()
    for g in range(6):
        row=copy.deepcopy(next(r for r in rows if r['route_id']==f'Town00_{g}'))
        row.update(id=f'invalid:{g}',route_id=f'Town00_{g}_10_10_10_10_10',targets=[dict(valid=False)])
        rows.append(row)
    result=splits.complete(rows,[]);assert result['feasibility']=='feasible'
    dataset.validate_splits(rows)
    for g in range(6):assert len({r['split'] for r in rows if r['physical_group']==f'Synthetic/Town00_{g}'})==1


@pytest.mark.parametrize('search_limit',[1,100000])
def test_formal_build_runs_fallback_before_and_after_geometry_filter(tmp_path,search_limit):
    root=tmp_path/'raw';routes=[]
    for group,mask in enumerate(MASKS):
        rid=f'Town00_{group}';run=root/'Synthetic'/rid
        (run/'metas').mkdir(parents=True);(run/'rgb').mkdir()
        for frame in range(9):(run/'metas'/f'{frame:04d}.pkl').write_bytes(pickle.dumps(meta(frame)))
        Image.new('RGB',(32,32)).save(run/'rgb/0008.jpg')
        routes.append(dict(scenario='Synthetic',route_id=rid,town='Town00',split='train',
            anchors=[dict(frame=8,event=e,road_structure='STRAIGHT') for i,e in enumerate(CONTEXT_IDS) if mask & (1<<i)]))
    if search_limit==1:
        with pytest.raises(RuntimeError,match='candidate_coverage_search_inconclusive'):
            dataset.build(routes,root,Lanes(),tmp_path/'built',coverage_search_states=search_limit)
        import json
        report=json.loads((tmp_path/'built/candidate_split_coverage.json').read_text())
        assert report['feasibility']=='unknown' and not (tmp_path/'built/dataset.json').exists()
        return
    artifact,_=dataset.build(routes,root,Lanes(),tmp_path/'built',coverage_search_states=search_limit)
    assert artifact==dataset.load(tmp_path/'built/dataset.json')
    assert artifact['split_coverage']['search']['method']=='exact_event_mask_backtracking_v1'
    import json
    preflight=json.loads((tmp_path/'built/candidate_split_coverage.json').read_text())
    assert preflight['search']['method']=='exact_event_mask_backtracking_v1' and not preflight['unresolved']
    assert len(train.training_members(artifact))>=10
