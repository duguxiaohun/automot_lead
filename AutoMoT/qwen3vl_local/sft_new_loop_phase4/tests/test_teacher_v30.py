from collections import Counter
from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4 import candidate_pool as cp,dataset
from qwen3vl_local.sft_new_loop_phase4.identity import digest,ROOT
from qwen3vl_local.sft_new_loop_phase4.risk_review import teacher_window_check
from qwen3vl_local.sft_new_loop_phase4.sampling import plan,CapacityError
from qwen3vl_local.sft_new_loop_phase4.tests.test_fourteenth_audit_fixes import row


def weighted_rows():
    out=[]
    for e,event in enumerate(('U-E1','U-E4')):
        for edge in ('proceed','hold','settle'):
            for answer in ('YES','NO'):
                for j in range(20):
                    r=row(len(out),event);r.update(edge=edge,target=answer);out.append(r)
    return out


@pytest.mark.parametrize('world',[1,4])
def test_weighted_edges_binary_balance_caps_and_resume(world):
    rows=weighted_rows();history=None
    for epoch in range(3):
        ids,a=plan(rows,epoch=epoch,policy='event_weighted',world_size=world,budget=80,history=history,max_question_repeat=2)
        history=a['next_history'];assert ids==plan(rows,epoch=epoch,policy='event_weighted',world_size=world,budget=80,max_question_repeat=2)[0]
        assert len(set(a['event_counts'].values()))==1
        for event in ('U-E1','U-E4'):
            assert a['edge_counts'][event+'/proceed']>=4*a['edge_counts'][event+'/hold']
            for edge in ('proceed','hold','settle'):
                assert a['edge_answer_counts'][event+'/'+edge+'/YES']==a['edge_answer_counts'][event+'/'+edge+'/NO']
        assert a['max_question_repeat']==1 and len(set(ids))==len(ids) and a['max_frame_repeat']<=8
    with pytest.raises(ValueError,match='history'):plan(rows,epoch=3,policy='event_weighted',world_size=world,budget=80,history=history,max_question_repeat=3)


def test_automatic_budget_shrinks_without_exceeding_repeat_cap():
    rows=weighted_rows()
    rows=[r for r in rows if r['episode']['event']=='U-E1']+rows[-2:]
    ids,a=plan(rows,epoch=0,policy='event_weighted',max_question_repeat=2)
    assert len(ids)==8 and a['max_question_repeat']<=2
    assert 'U-E4/settle/YES' in a['missing_cell_answers']
    with pytest.raises(CapacityError):plan(rows,epoch=0,policy='event_weighted',budget=80,max_question_repeat=2)


@pytest.mark.parametrize('frame,blocked',[(19,True),(22,True),(23,False)])
def test_risk_windows_use_all_causal_frames(frame,blocked):
    q=dict(causal_sources=[dict(frame_id=n) for n in range(frame-6,frame+1)])
    risks=[dict(risk_id='a',record=dict(rgb_evidence=[dict(frame=15),dict(frame=16)]))]
    assert teacher_window_check(q,risks)['outside_registered_windows'] is not blocked


def test_risk_windows_include_old_stop_proof_and_unbounded_records():
    q=dict(causal_sources=[dict(frame_id=30)],control_evidence={'1':dict(observations=[dict(sources=[dict(frame_id=15)])])})
    risks=[dict(risk_id='a',record=dict(rgb_evidence=[dict(frame=15)]))]
    assert not teacher_window_check(q,risks)['outside_registered_windows']
    assert not teacher_window_check(q,[dict(risk_id='b',record={})])['outside_registered_windows']


def test_filter_scan_duration_boundary_whitelist_and_missing_meta(tmp_path):
    root=tmp_path/'data'
    for scenario,n,meta in [('S',360,True),('Bad',361,True),('ControlLoss',361,True),('BlockedIntersection',361,True),('Missing',20,False)]:
        route=root/scenario/'Town01_Rep0_1_0_route0_x';(route/'rgb').mkdir(parents=True)
        for i in range(n):(route/'rgb'/f'{i:04d}.jpg').touch()
        if meta:(route/'metas').mkdir();(route/'metas'/'0000.pkl').touch()
    pool=cp.scan(root,tmp_path/'pool.json');cp.validate(pool)
    assert {r['scenario'] for r in pool['routes']}=={'S','ControlLoss','BlockedIntersection'}
    assert {r['exclusion_reason'] for r in pool['excluded_routes']}=={'abnormal_duration','missing_all_metas'}
    bad=deepcopy(pool);bad['routes'].append(bad['excluded_routes'].pop(0));bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):cp.validate(bad)


def test_retired_independent_routes_stay_quarantined():
    plan=json.loads((ROOT/dataset.PRODUCER_CHECK_PLAN).read_text());active=dataset.producer_check_reservations();all_owners=dataset.holdout_reservations()
    assert len(active)==93 and len(plan['excluded_reservations'])==16
    for r in plan['excluded_reservations']:
        assert r['physical_group'] not in active and all_owners[r['physical_group']]==r['split']


def test_reference_pack_rejects_training_access(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.teacher_evaluation import load
    with pytest.raises(ValueError,match='held-out'):load(tmp_path,split='train',rgb_mode=2)


def test_heldout_reference_export_load_and_tamper(tmp_path,monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_privileged_producer import write_source
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question,seal
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as replay,teacher_rules as rules,teacher_evaluation as evaluation
    from qwen3vl_local.sft_new_loop_phase4.identity import file_sha,write_json
    for n in range(4,12):write_source(tmp_path/'data',n)
    monkeypatch.setattr(dataset,'groups',lambda:set())
    monkeypatch.setattr(dataset,'holdout_reservations',lambda:{})
    monkeypatch.setattr(dataset,'split_for',lambda *a,**k:'val')
    pool=cp.scan(tmp_path/'data',tmp_path/'pool.json');q=question();q['split']='val'
    for s in q['causal_sources']:s['sha256']=file_sha(tmp_path/'data'/s['path'])
    q['input_sha256']=digest(q['input_sources']);seal(q)
    header=dict(policy=replay.POLICY,teacher=rules.identity(),candidate_pool_sha256=pool['sha256']);route=pool['routes'][0]
    items=[dict(kind='header',**header)]
    for n in route['rgb_frames']:
        items.append(dict(kind='frame_disposition',**{k:route[k] for k in ('scenario','route_id','physical_group','split')},frame_id=n,
            disposition='questions' if n==10 else 'abstained',questions=[q] if n==10 else [],reasons=['synthetic']))
    path=tmp_path/'route.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in items))
    index=dict(**header,routes=[replay.inspect_route(path,route,header)]);index['sha256']=digest(index);write_json(tmp_path/'index.json',index)
    output=tmp_path/'refs';report=evaluation.export(pool,tmp_path/'index.json',tmp_path/'data',output,rgb_mode=2)
    rows=evaluation.load(output,split='val',rgb_mode=2)
    assert len(rows)==1 and rows[0]['target']=='YES' and report['files']['test']['count']==0
    with pytest.raises(ValueError,match='manifest'):evaluation.load(output,split='val',rgb_mode=4)
    with pytest.raises((KeyError,FileNotFoundError,ValueError)):dataset.load_dataset(output)
    rows[0]['target']='NO';(output/'val.jsonl').write_text(json.dumps(rows[0])+'\n')
    m=json.loads((output/'references.json').read_text());m['files']['val']['sha256']=file_sha(output/'val.jsonl');m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'});write_json(output/'references.json',m)
    with pytest.raises(ValueError,match='detached'):evaluation.load(output,split='val',rgb_mode=2)


def test_conflicting_weak_labels_never_overwrite_manual_reference():
    from qwen3vl_local.sft_new_loop_phase4.teacher_data import reconcile_weak_inputs
    manual=dict(id='m',model_input_sha256='x',target='NO',label_basis='reviewed_transition_band')
    weak=dict(id='w',model_input_sha256='x',target='YES',label_basis='weak_rule_teacher')
    kept,counts=reconcile_weak_inputs([manual,weak]);assert kept==[manual] and counts['manual_input_overlap']==1
    kept,counts=reconcile_weak_inputs([weak,dict(weak,id='w2',target='NO')]);assert kept==[] and counts['conflicting_weak_input']==2


def test_only_scarce_cells_repeat_when_other_cells_have_unused_questions():
    rows=weighted_rows()
    rows=[r for r in rows if r['episode']['event']=='U-E1']+rows[-2:]
    ids,a=plan(rows,epoch=0,policy='event_weighted',budget=16,max_question_repeat=4)
    uses=Counter(ids)
    assert all(n==1 for i,n in uses.items() if rows[i]['episode']['event']=='U-E1')
    assert max(n for i,n in uses.items() if rows[i]['episode']['event']=='U-E4')==4
