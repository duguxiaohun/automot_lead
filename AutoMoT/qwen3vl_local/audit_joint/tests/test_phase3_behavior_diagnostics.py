import copy
import json
from pathlib import Path
import pytest

from qwen3vl_local.audit_joint import phase3_behavior_diagnostics as d
from qwen3vl_local.audit_joint import phase3_matched_image_eval as e


def window(a='STOP',b=None):
    if a is None:return dict(valid=False,raw=None,target=None,allowed_answers=[])
    b=b or a
    raw=[([a,b].count(x)/2) for x in d.ACTIONS]
    weighted=[v*(1.5 if x.startswith('LANE_CHANGE') else 1.) for x,v in zip(d.ACTIONS,raw)]
    return dict(valid=True,raw=raw,target=[v/sum(weighted) for v in weighted],
                allowed_answers=[x for x,v in zip(d.ACTIONS,raw) if v])


def row(i,group=None):
    return dict(id=str(i),split='val',event='LEAD_BRAKE',group=group or 'group'+str(i),frame_id=i,
        speed=3.,brightness=30.,image=f'route{i}/rgb/0001.jpg',rgb_sha256='sha'+str(i),image_size=[120,40],
        base_target='STOP',history=['STOP']*4,predictions=['STOP']*4,windows=[window() for _ in range(4)])


def test_anchor_coverage_separates_missing_and_complete_future():
    a=row(1);a['base_target']='LANE_CHANGE_LEFT'
    b=row(2);b['base_target']='LANE_CHANGE_LEFT';b['windows'][0]=window(None)
    r=d.summarize([a,b]);c=r['anchor_action_coverage']['LANE_CHANGE_LEFT']
    assert c['anchor_action_absent_with_all_four_valid']==1
    assert c['anchor_action_absent_with_partial_future']==1
    assert r['metrics']['all']['windows']==7
    assert r['metrics']['all']['hits']==7


def test_preference_is_not_a_pure_class_weight():
    r=row(1);r['windows']=[window('LANE_CHANGE_LEFT'),window('STOP','LANE_CHANGE_LEFT'),window(),window()]
    p=d.summarize([r])['lateral_preference']
    assert p['pure_lateral_windows']==1
    assert p['changed_target_windows']==1
    assert p['raw_lateral_mass']==1.5
    assert p['training_lateral_mass']==pytest.approx(1.6)


@pytest.mark.parametrize('pred,expected',[
    (['STOP','STOP','RESUME','RESUME'],'on_time_correct_action'),
    (['STOP','RESUME','RESUME','RESUME'],'early'),
    (['STOP','STOP','STOP','RESUME'],'late'),
    (['STOP']*4,'never_changes'),
    (['STOP','STOP','KEEP','KEEP'],'on_time_wrong_action'),
])
def test_transition_timing(pred,expected):
    r=row(1);r['windows']=[window(),window(),window('RESUME'),window('RESUME')];r['predictions']=pred
    assert d.first_transition(r)==expected


def test_timing_never_interpolates_masked_or_mixed_labels():
    r=row(1);r['windows'][1]=window(None)
    assert d.first_transition(r)=='excluded_mixed_or_missing'
    r['windows'][1]=window('STOP','RESUME')
    assert d.first_transition(r)=='excluded_mixed_or_missing'
    r['windows'][1]=window();r['history'][-1]='UNKNOWN'
    assert d.first_transition(r)=='excluded_latest_history_unknown'


def test_maximum_matching_and_donor_capacity():
    rows=[row(1,'A'),row(2,'A'),row(3,'A'),row(4,'B')]
    p=d.matched_plan(rows,11)
    assert len(p['pairs'])==2 and len(p['excluded'])==2
    assert len({x['donor'] for x in p['pairs']})==2
    d.validate_plan(p)
    rows=[row(i) for i in range(4)]
    assert len(d.matched_plan(rows,11)['pairs'])==4


def test_matching_is_label_independent_deterministic_and_cross_route():
    rows=[row(i) for i in range(8)];p=d.matched_plan(rows,29)
    changed=copy.deepcopy(rows)
    for r in changed:r['predictions']=['INVALID']*4;r['windows']=[window('RESUME')]*4;r['base_target']='KEEP'
    assert d.matched_plan(changed,29)==p
    assert all(x['receiver']!=x['donor'] for x in p['pairs'])
    assert d.matched_plan(rows,29)==p


@pytest.mark.parametrize('field,value',[('group','group1'),('rgb_sha256','sha1'),('event','OTHER'),
                                      ('brightness',2.),('speed',22.),('image_size',[30,30])])
def test_incompatible_donor_is_excluded(field,value):
    rows=[row(1),row(2)];rows[1][field]=value
    assert d.matched_plan(rows,11)['pairs']==[]


def test_no_silent_id_split_or_plan_changes():
    with pytest.raises(ValueError,match='duplicate'):d.matched_plan([row(1),row(1)],11)
    r=row(1);r['split']='test'
    with pytest.raises(ValueError,match='validation only'):d.matched_plan([r],11)
    p=d.matched_plan([row(1),row(2)],11);p['pairs'][0]['donor']='1'
    with pytest.raises(ValueError,match='plan changed'):d.validate_plan(p)


def test_pairing_keeps_denominators_and_rejects_target_changes():
    a=row(1);b=copy.deepcopy(a);b['predictions'][0]='RESUME'
    p=d.paired_scores([a],[b],{'1':a})
    assert p['subsets']['all']==dict(windows=4,reference_hits=4,alternative_hits=3,reference_only=1,alternative_only=0,delta=.25)
    b['windows'][0]=window('RESUME')
    with pytest.raises(ValueError,match='labels or history'):d.paired_scores([a],[b],{'1':a})


def test_wrong_image_changes_only_pixels_not_receiver_labels_or_premise():
    rows=[row(1),row(2)];records={r['id']:dict(r,premise='text'+r['id']) for r in rows}
    before=copy.deepcopy(records);calls=[]
    def predict(bundle,recs):
        calls.append(copy.deepcopy(recs))
        return [dict(r) for r in recs],{}
    plan=d.matched_plan(rows,11)
    actual,wrong=e.predict_pairs(None,records,plan,predict=predict)
    assert records==before
    for original,swapped,pair in zip(calls[0],calls[1],plan['pairs']):
        assert swapped==dict(original,image=records[pair['donor']]['image'])
        assert swapped['premise']==original['premise']
    assert all(c['image_mode']=='matched_other_route' for c in wrong)


def test_raw_target_validation_rejects_corrupt_labels():
    r=row(1);r['windows'][0]['allowed_answers']=['RESUME']
    with pytest.raises(ValueError,match='distribution'):d.summarize([r])
    r=row(1);r['windows'][0]=window(None);r['windows'][0]['raw']=[0]*6
    with pytest.raises(ValueError,match='must not carry'):d.summarize([r])


def test_cross_seed_cache_preserves_pairs_and_avoids_repeated_forwards():
    rows=[row(1),row(2)];records={r['id']:r for r in rows};calls=[];ref={};swap={}
    def predict(bundle,recs):
        calls.append([r['image'] for r in recs]);return copy.deepcopy(recs),{}
    p=d.matched_plan(rows,11)
    a,b=e.predict_pairs(None,records,p,predict=predict,reference_cache=ref,swapped_cache=swap)
    c,f=e.predict_pairs(None,records,p,predict=predict,reference_cache=ref,swapped_cache=swap)
    assert len(calls)==2 and a==c and b==f
    assert set(swap)=={(x['receiver'],x['donor']) for x in p['pairs']}


def test_four_process_shards_equal_serial_and_reject_partial_or_duplicate(tmp_path):
    import os
    import subprocess
    import sys
    from qwen3vl_local.audit_joint import phase3_matched_image_launch as launch
    rows=[row(i) for i in range(8)]
    plans=[d.matched_plan(rows,seed) for seed in (11,29,47)]
    records={r['id']:r for r in rows}
    payload=tmp_path/'input.json';payload.write_text(json.dumps(dict(plans=plans,records=records)))
    script=tmp_path/'worker.py'
    script.write_text('''import json,sys
from qwen3vl_local.audit_joint import phase3_matched_image_eval as e
v=json.load(open(sys.argv[1]));rank=int(sys.argv[2]);world=int(sys.argv[3])
ids=e.receiver_shard(v['plans'],rank,world)
def predict(bundle,records):
    return [dict(r,predictions=['STOP' if sum(map(ord,r['image']))%2 else 'KEEP']*4) for r in records],{}
a={};b={}
for p in v['plans']:
    selected=dict(p,pairs=[x for x in p['pairs'] if x['receiver'] in ids])
    e.predict_pairs(None,v['records'],selected,predict=predict,reference_cache=a,swapped_cache=b)
json.dump(dict(rank=rank,world=world,reference=list(a.values()),swapped=list(b.values())),open(sys.argv[4],'w'))
''')
    env=dict(os.environ,PYTHONPATH=str(launch.APP))
    commands=[[sys.executable,str(script),str(payload),str(rank),'4',str(tmp_path/f'rank_{rank}.json')] for rank in range(4)]
    launch.processes(commands,tmp_path,[env]*4,[tmp_path/f'{rank}.log' for rank in range(4)])
    shards=[json.loads((tmp_path/f'rank_{rank}.json').read_text()) for rank in range(4)]
    actual,swapped=e.merge_shards(plans,shards)
    single=tmp_path/'single.json'
    subprocess.run([sys.executable,str(script),str(payload),'0','1',str(single)],env=env,check=True)
    serial=json.loads(single.read_text())
    assert actual=={c['id']:c for c in serial['reference']}
    assert swapped=={(c['id'],c['donor_id']):c for c in serial['swapped']}
    assert len(actual)==8
    original=tmp_path/'original';original.mkdir()
    (original/'single_frame_val_cases.json').write_text(json.dumps(serial['reference']))
    parallel_output=tmp_path/'parallel_metrics';parallel_output.mkdir()
    serial_output=tmp_path/'serial_metrics';serial_output.mkdir()
    e.save_metrics(dict(output=str(original)),plans,parallel_output,actual,swapped)
    sa,sb=e.merge_shards(plans,[serial])
    e.save_metrics(dict(output=str(original)),plans,serial_output,sa,sb)
    for path in parallel_output.glob('*.json'):
        assert path.read_bytes()==(serial_output/path.name).read_bytes()
    for mutation in ('missing','duplicate','rank'):
        bad=copy.deepcopy(shards)
        if mutation=='missing':bad[0]['reference'].pop()
        if mutation=='duplicate':bad[0]['swapped'].append(bad[0]['swapped'][0])
        if mutation=='rank':bad[0]['rank']=1
        with pytest.raises(ValueError):e.merge_shards(plans,bad)


def test_real_plan_four_rank_exact_904_forward_partition():
    root=Path(e.__file__).with_name('phase3_behavior_diagnostic_v1')
    plans=[json.loads((root/f'matched_images_seed_{s}.json').read_text()) for s in (11,29,47)]
    shards=[e.receiver_shard(plans,r,4) for r in range(4)]
    assert [len(s) for s in shards]==[67]*4
    assert sum(len({(p['receiver'],p['donor']) for plan in plans for p in plan['pairs'] if p['receiver'] in ids}) for ids in shards)==636
    assert len(set.union(*shards))==268
