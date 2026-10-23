import copy
import pytest
from .. import splits
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS

def row(group,event,split='train',frame=1):
    return dict(id=f'{group}/{event}/{frame}',physical_group='Synthetic/'+group,scenario='Synthetic',
        route_id=group,frame=frame,event=event,split=split,targets=[dict(valid=True)],sampling_mass={'six/0/5':1.})

def cohort():
    rows=[]
    for event in CONTEXT_IDS:
        for s in ('train','val','test'):rows.append(row(f'Town00_{s}_{event}',event,s))
    return rows

def test_only_unexposed_train_whole_group_moves_and_input_unchanged():
    rows=cohort();rows=[r for r in rows if not(r['event']=='LEAD_BRAKE' and r['split']=='test')]
    rows.extend([row('Town00_new','LEAD_BRAKE'),row('Town00_new','SIGNAL_FAILURE',frame=2)])
    rows.append(row('Town00_exposed','LEAD_BRAKE'));before=copy.deepcopy(rows)
    protected=['Synthetic/Town00_exposed','Synthetic/Town00_train_LEAD_BRAKE']
    out,report=splits.repair(rows,development=protected,per_event=1,world_size=1)
    assert rows==before and report['status']=='feasible'
    assert report['moves']=={'Synthetic/Town00_new':dict(from_split='train',to_split='test',events=['LEAD_BRAKE','SIGNAL_FAILURE'])}
    assert all(r['split']=='test' and r['parent_split']=='train' for r in out if r['route_id']=='Town00_new')
    assert all(a['split']==b['split'] for a,b in zip(rows,out) if a['split']!='train')
    splits.verify(out,report,protected)
    bad=copy.deepcopy(out);bad[0]['split']='test'
    with pytest.raises(ValueError):splits.verify(bad,report,protected)

def test_no_supply_and_unknown_do_not_publish_partial_moves():
    rows=[r for r in cohort() if not(r['event']=='LEAD_BRAKE' and r['split']=='test')]
    out,r=splits.repair(rows,development=[],per_event=1,world_size=1)
    assert r['status']=='infeasible' and not r['moves'] and out==rows
    rows.append(row('Town00_new','LEAD_BRAKE'))
    out,r=splits.repair(rows,development=[],per_event=1,world_size=1,max_states=1)
    assert r['status']=='unknown' and not r['moves'] and out==rows

def test_exposed_holdout_is_rejected_not_silently_reassigned():
    with pytest.raises(ValueError,match='known_exposure_in_existing_holdout'):
        splits.repair(cohort(),development=['Synthetic/Town00_test_LEAD_BRAKE'],per_event=1,world_size=1)

def test_shared_frame_capacity_is_checked_after_move():
    rows=cohort()
    # All remaining train rows share one physical frame; individual event floors pass.
    for r in rows:
        if r['split']=='train':r.update(route_id='Town00_shared',physical_group='Synthetic/Town00_shared')
    out,report=splits.repair(rows,development=[],per_event=1,world_size=1)
    assert report['status']=='infeasible' and report['shared_capacity_failures']==1 and out==rows

def test_label_values_do_not_select_holdout_groups():
    rows=[r for r in cohort() if not(r['event']=='LEAD_BRAKE' and r['split']=='test')]
    rows.extend(row(f'Town00_extra{i}','LEAD_BRAKE') for i in range(5))
    a=splits.repair(rows,development=[],per_event=1,world_size=1)[1]
    for r in rows:r['targets'][0].update(token_id=0,name='arbitrary');r['sampling_mass']={'six/0/0':1.}
    b=splits.repair(rows,development=[],per_event=1,world_size=1)[1]
    assert a==b

def test_small_assignment_search_matches_independent_exhaustive_feasibility():
    import itertools,random
    rng=random.Random(4);events=list(CONTEXT_IDS[:2])
    for _ in range(40):
        rows=[r for r in cohort() if r['event'] not in events]
        masks=[rng.randrange(1,4) for i in range(4)]
        for i,mask in enumerate(masks):
            for j,event in enumerate(events):
                if mask&(1<<j):rows.append(row(f'Town00_mask{i}',event))
        possible=False
        for choices in itertools.product(('train','val','test'),repeat=4):
            covered={s:0 for s in ('train','val','test')}
            for mask,s in zip(masks,choices):covered[s]|=mask
            train=[mask for mask,s in zip(masks,choices) if s=='train']
            capacity=any(a&1 and b&2 for i,a in enumerate(train) for j,b in enumerate(train) if i!=j)
            if all(m==3 for m in covered.values()) and capacity:possible=True;break
        out,report=splits.repair(rows,development=[],per_event=1,world_size=1)
        assert (report['status']=='feasible')==possible
        assert report['status']!='unknown'
        if not possible:assert out==rows and not report['moves']
