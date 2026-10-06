import copy
import json

import pytest
from PIL import Image
from qwen3vl_local.sft_new_loop_phase4 import condition_builder as cb,dataset
from qwen3vl_local.sft_new_loop_phase4.calibration import label_condition,interval_facts
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.identity import file_sha,ROOT


def producer():
    return dict(name='test causal geometry',version='1',source='causal_geometry_review',rubric_sha256=cb.rubric_identity())


def record():
    return dict(scenario='Scenario',route_id='Town01_Rep0_1_0_route0',episode=dict(event='U-E4',instance_id='pedestrian',state='YIELD',longitudinal='APPROACH'),frame_id=20,observed_until=20,context_valid=True,
                facts={'release_ready':True,'priority_satisfied':True,'corridor_clear':True},
                sources=[dict(path='Scenario/Town01_Rep0_1_0_route0/rgb/0020.jpg',sha256='a'*64,kind='rgb',frame_id=20)])


def answers(r):
    result={}
    for a in cb.annotations_for_record(r,producer()):
        result[a['edge']]=label_condition(Episode(**a['episode']),a['edge'],20,interval_facts(a['facts'],20,20,'test'),context_valid=a['context_valid'])
    return result


def test_shared_rules_clear_corridor_before_motion_and_missing_stays_unknown():
    r=record();a=answers(r)
    assert a['proceed']=='YES' and a['hold']=='UNKNOWN' and a['settle']=='UNKNOWN'
    assert 'complete' not in a
    del r['facts']['priority_satisfied']
    assert answers(r)['proceed']=='UNKNOWN'
    r['facts']['corridor_clear']=False
    assert answers(r)['proceed']=='NO'
    r['context_valid']=None
    assert set(answers(r).values())=={'UNKNOWN'}


@pytest.mark.parametrize('event,branch,ret,edge,state', [
    ('U-E1','default',True,'proceed','YIELD'),('U-E2','default',True,'depart','WAIT'),
    ('U-E2','default',False,'depart','WAIT'),('U-E2','in_lane_pass',False,'proceed','YIELD'),
    ('U-E3','default',True,'proceed','YIELD'),('U-E4','default',True,'proceed','YIELD'),
    ('U-E4','cyclist_follow',False,'proceed','YIELD'),('U-E4','cyclist_bypass',True,'depart','WAIT'),
    ('U-E4','cyclist_bypass',False,'depart','WAIT'),('U-E5','default',True,'proceed','YIELD'),
    ('U-E6','default',True,'proceed','YIELD'),('U-E7','default',True,'proceed','YIELD'),
    ('R-E2','default',True,'enter','WAIT'),('R-E3','default',True,'enter','WAIT'),
    ('R-E5','default',True,'proceed','YIELD')])
def test_no_per_route_rules_needed_for_all_event_templates(event,branch,ret,edge,state):
    r=record();r['episode'].update(event=event,branch=branch,return_required=ret,state=state)
    if state=='WAIT':
        r['episode'].update(direction='LEFT',target_corridor='navigation corridor')
        if ret and (event=='U-E2' or branch=='cyclist_bypass'):
            r['episode'].update(return_direction='RIGHT',return_corridor='original corridor')
        r['facts']=dict(target_corridor_known=True,entry_gap_clear=True,priority_satisfied=True)
    assert answers(r)[edge]=='YES'


@pytest.mark.parametrize('bad', ['future','future_source','wrong_route','control','future_array','unknown_fact','stale_rule','missing_rgb','value_type'])
def test_bad_conditions_do_not_create_labels(bad):
    r=record();p=producer()
    if bad=='future':r['observed_until']=21
    elif bad=='future_source':r['sources'][0]['frame_id']=21
    elif bad=='wrong_route':r['sources'][0]['path']='other/rgb/0020.jpg'
    elif bad=='control':r['brake']=False
    elif bad=='future_array':r['future_speeds']=[1,2,3]
    elif bad=='unknown_fact':r['facts']['throttle']=True
    elif bad=='stale_rule':p['rubric_sha256']['taxonomy.py']='0'*64
    elif bad=='missing_rgb':r['sources']=[]
    else:r['facts']['release_ready']=1
    with pytest.raises(ValueError):cb.annotations_for_record(r,p)


def test_evidence_content_hash_verified_and_duplicate_stream_rejected(tmp_path):
    r=record();path=tmp_path/r['sources'][0]['path'];path.parent.mkdir(parents=True);Image.new('RGB',(8,8),'red').save(path)
    r['sources'][0]['sha256']=file_sha(path)
    stream=tmp_path/'conditions.jsonl'
    header=dict(policy=cb.POLICY,producer=producer())
    stream.write_text('\n'.join(json.dumps(x) for x in [header,r])+'\n')
    anns,report=cb.compile_stream(stream,tmp_path)
    assert len(anns)==3 and report['answers_before_history_filter']=={'YES':1,'UNKNOWN':2}
    for a in anns:cb.validate_annotation(a,tmp_path)
    with stream.open('a') as f:f.write(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='duplicate'):cb.compile_stream(stream,tmp_path)
    Image.new('RGB',(8,8),'blue').save(path)
    with pytest.raises(ValueError,match='content'):cb.annotations_for_record(r,producer(),tmp_path)


def test_edited_compiled_annotation_is_rejected():
    a=cb.annotations_for_record(record(),producer())[0];a['facts']['corridor_clear']=False
    with pytest.raises(ValueError,match='differs'):cb.validate_annotation(a)


def test_real_audited_facts_compile_build_and_load_with_same_yes_no(tmp_path):
    root=ROOT.parents[1]/'lead_data'
    if not root.exists():pytest.skip('external RGB unavailable')
    source=json.loads((ROOT/'reviewed_state_pairs_v7.json').read_text())
    a=next(a for a in source if a['edge']=='proceed' and a['label_basis']=='reviewed_transition_band'
           and any(x['phase']=='readiness' and x.get('facts',{}).get('release_ready') is True for x in a['transition_band']['frames'].values()))
    records=[]
    for f,item in a['transition_band']['frames'].items():
        if item['phase']!='readiness':continue
        r=dict(scenario=a['scenario'],route_id=a['route_id'],episode=a['episode'],frame_id=int(f),observed_until=int(f),context_valid=a['context_valid'],facts=item['facts'],sources=[dict(path=f"{a['scenario']}/{a['route_id']}/rgb/{int(f):04d}.jpg",kind='rgb',frame_id=int(f),sha256=a['frame_sha256'][f])])
        records.append(r)
    p=producer();p.update(name='existing audited facts fixture',source='rgb_review')
    stream=tmp_path/'conditions.jsonl';stream.write_text('\n'.join(json.dumps(x) for x in [dict(policy=cb.POLICY,producer=p),*records])+'\n')
    anns,report=cb.compile_stream(stream,root)
    out=tmp_path/'data';dataset.build(anns,root,out,rgb_mode=4)
    data,m=dataset.load_dataset(out)
    assert {r['target'] for r in data['train'] if r['edge']=='proceed'}=={'YES','NO'}
    for r in sum(data.values(),[]):cb.validate_row(r,root)
    r=copy.deepcopy(data['train'][0]);r['target']='NO' if r['target']=='YES' else 'YES'
    with pytest.raises(ValueError,match='differs'):cb.validate_row(r)
