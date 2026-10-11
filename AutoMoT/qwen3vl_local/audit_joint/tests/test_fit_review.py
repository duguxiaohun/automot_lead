from copy import deepcopy
import pytest
from qwen3vl_local.audit_joint import fit_review as tool
from qwen3vl_local.sft_new_loop_phase4.paired_eval import case_identity
from qwen3vl_local.sft_new_loop_phase4.tests.test_short_rgb import source_row

@pytest.fixture
def pair(tmp_path):
    row=source_row(tmp_path);row['label_basis']='weak_rule_teacher';row['episode']['instance_id']='original-instance'
    def case(r):
        return dict(id=r['id'],target=r['target'],prediction='NO',raw='NO\n',event=r['episode']['event'],edge=r['edge'],slice=r['slice'],reference_kind=r.get('reference_kind','reviewed_rgb'),original_label_basis='weak_rule_teacher',sampled_presentations=3,paired_identity=case_identity(r))
    small=deepcopy(row)
    for k in ('images','image_sha256','image_rgb_sha256'):small[k]=small[k][-2:]
    small['observation']['history_frames']=small['observation']['history_frames'][-2:]
    return case(small),case(row)

def record(case,target=None):
    f=case['paired_identity']['fields'];base={k:f[k] for k in ('scenario','route_id','physical_group','split')};base['frame_id']=f['observation']['frame_id']
    q=dict(base,episode=deepcopy(f['episode']),rgb_mode=4,edge=f['edge'],slice=f['slice'],rule_target=target or f['target'],facts={'condition':True},reasons=[],question_id='q')
    q['history_frames']=f['observation']['history_frames']
    q['input_sources']=[dict(frame_id=n,path=p,sha256=h,kind='rgb') for n,p,h in zip(q['history_frames'],f['images'],f['image_sha256'])]
    return dict(base,questions=[q])

def test_summary_keeps_routes_questions_and_exposures_separate(pair):
    a,b=pair;b['prediction']=b['raw']='YES';r=tool.paired_diagnostics([a],[b])
    assert r['unique_questions']==1 and r['rgb2']['presentations']==3 and r['rgb2']['physical_groups']==1
    assert r['rgb2']['recall']['YES']==0 and r['rgb4']['recall']['YES']==1
    assert r['correctness_transitions']=={'rgb2_0_rgb4_1':1}

@pytest.mark.parametrize('damage',['duplicate','missing','episode','exposure','shared_rgb','raw','future'])
def test_pair_rejects_incompatible_evidence(pair,damage):
    a,b=deepcopy(pair);left=[a];right=[b]
    if damage=='duplicate':left.append(a)
    elif damage=='missing':right=[]
    elif damage=='exposure':b['sampled_presentations']+=1
    elif damage=='raw':b['raw']='YES'
    else:
        f=b['paired_identity']['fields']
        if damage=='episode':f['episode']['instance_id']='other'
        elif damage=='shared_rgb':f['image_sha256'][-1]='f'*64
        else:f['observation']['history_frames'][-1]+=1
        b['paired_identity']=case_identity(f)
    with pytest.raises(ValueError):tool.paired_diagnostics(left,right)

@pytest.mark.parametrize('damage,expected',[
    ('same','rule_target_unchanged'),('target','rule_target_changed'),
    ('absent','current_proposition_not_comparable'),('state','current_proposition_not_comparable'),
    ('rgb','current_proposition_not_comparable'),('phase','current_proposition_not_comparable'),('instance','current_proposition_not_comparable'),
    ('duplicate','current_proposition_not_comparable'),('original_target','original_target_not_reproduced'),
    ('manual','manual_reference_requires_separate_adjudication')])
def test_replay_does_not_relabel_changed_or_missing_propositions(pair,damage,expected):
    _,c=deepcopy(pair);old=record(c);new=deepcopy(old)
    if damage=='target':new['questions'][0]['rule_target']='NO'
    elif damage=='original_target':old['questions'][0]['rule_target']='NO'
    elif damage=='absent':new['questions']=[]
    elif damage=='state':new['questions'][0]['episode']['state']='changed'
    elif damage=='rgb':new['questions'][0]['input_sources'][0]['sha256']='e'*64
    elif damage=='phase':new['questions'][0]['slice']='catchup'
    elif damage=='instance':new['questions'][0]['episode']['instance_id']='other'
    elif damage=='duplicate':new['questions'].append(deepcopy(new['questions'][0]))
    elif damage=='manual':c['original_label_basis']='reviewed_transition_band'
    r=tool.replay_diagnostics([c],[old],[new])['cases'][0]
    assert r['status']==expected and 'new_target' not in r
    if damage=='absent':assert r['after']['status']=='question_absent'
    if damage=='duplicate':assert r['after']['status']=='ambiguous'
    assert tool.replay_diagnostics([c],[],[])['cases'][0]['after']['status']=='not_replayed'

def test_replay_rejects_duplicate_frames_and_cross_route_questions(pair):
    _,c=pair;r=record(c)
    with pytest.raises(ValueError,match='duplicate replay'):tool.replay_diagnostics([c],[r,r],[])
    r['questions'][0]['route_id']='another'
    with pytest.raises(ValueError,match='outside frame'):tool.replay_diagnostics([c],[r],[])


def test_explicit_transferred_reference_is_not_new_human_approval(pair):
    a,b=deepcopy(pair);f=a['paired_identity']['fields']
    f['label_basis']='transferred_rgb4_reference';f['reference_kind']='transferred_reference'
    a['reference_kind']='transferred_reference';a['paired_identity']=case_identity(f)
    assert tool.paired_diagnostics([a],[b])['unique_questions']==1
    a['original_label_basis']='another'
    with pytest.raises(ValueError):tool.paired_diagnostics([a],[b])
