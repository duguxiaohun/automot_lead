from copy import deepcopy
import json
import pytest
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,file_sha,write_json
from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,read_rows,dump_rows
from qwen3vl_local.sft_new_loop_phase4.risk_review import registry,templates,validate_review,CHECKS

DATA=ROOT.parents[1]/'lead_data'


def annotation():
    # Synthetic plumbing facts only; never added to default reviewed annotations.
    risk=next(v[0]['record'] for v in registry().values() if v[0]['record']['manifest_id']=='596')
    band=dict(review_start=89,review_end=91,reference_frame=89,reference_kind='condition_onset',
        reference_observation='SYNTHETIC test condition',start_reason='SYNTHETIC test start',
        stop=dict(frame=92,kind='review_end',reason='SYNTHETIC right censor'),frames={})
    for f in range(89,92):band['frames'][str(f)]=dict(phase='readiness',observed_until=f,
        observation='SYNTHETIC test facts, not approved labels',facts=dict(release_ready=True,corridor_clear=True,priority_satisfied=True))
    return dict(scenario=risk['scenario'],route_id=risk['route_id'],reviewer='test-only',evidence_id='SYNTHETIC',
        observation='SYNTHETIC fixture',context_valid=True,label_basis='reviewed_transition_band',
        episode=dict(event='U-E4',instance_id='synthetic',longitudinal='HOLD'),edge='proceed',transition_band=band,
        frame_sha256={str(f):file_sha(DATA/risk['scenario']/risk['route_id']/'rgb'/f'{f:04d}.jpg') for f in range(89,92)})


def reviewed(mode):
    ann=annotation();draft=templates([ann],DATA,mode)['annotations'][0]['risk_review']
    draft.update(reviewer='synthetic reviewer',evidence_id='SYNTHETIC frame review')
    for record in draft['frames'].values():record.update(status='usable',reason='SYNTHETIC review',**{k:True for k in CHECKS})
    ann['risk_review']=draft
    return ann


@pytest.mark.parametrize('mode',[2,4])
def test_known_risk_missing_review_is_rejected_before_output(tmp_path,mode):
    with pytest.raises(ValueError,match='explicit per-frame risk_review'):
        build([annotation()],DATA,tmp_path/'missing',rgb_mode=mode)
    assert not (tmp_path/'missing').exists()


@pytest.mark.parametrize('mode',[2,4])
def test_template_is_not_approval_and_reviews_all_causal_history(mode):
    a=annotation();draft=templates([a],DATA,mode)
    assert not draft['review_complete']
    proof=draft['annotations'][0]['risk_review']
    assert min(map(int,proof['frames']))==(85 if mode==2 else 83)
    risks=registry()[(a['scenario'],a['route_id'])]
    with pytest.raises(ValueError,match='reviewer/evidence'):
        validate_review(proof,risks,[89],[a['frame_sha256']['89']])


@pytest.mark.parametrize('mode',[2,4])
@pytest.mark.parametrize('state',['uncertain','excluded'])
def test_risky_history_demotes_current_yes_and_survives_loading(tmp_path,mode,state):
    ann=reviewed(mode)
    # Anchor89 itself is reviewed usable; its old image85 is not.
    ann['risk_review']['frames']['85']['status']=state
    m=build([ann],DATA,tmp_path/'risk',rgb_mode=mode);data,_=load_dataset(tmp_path/'risk')
    queued=read_rows(tmp_path/'risk/review_queue.jsonl');r=next(r for r in queued if r['observation']['frame_id']==89)
    assert r['target']=='UNKNOWN' and r['risk_review']['disposition']==state
    assert all(r['observation']['frame_id']!=89 for r in data['train'])
    assert m['risk_review_check']['row_dispositions'][state]>=1 and not m['ready']


@pytest.mark.parametrize('fault',['stale_registry','missing_history','wrong_sha','future','identity','scope','boundary'])
def test_incomplete_risk_review_cannot_be_presented_as_usable(tmp_path,fault):
    a=reviewed(4);proof=a['risk_review'];r=proof['frames']['83']
    if fault=='stale_registry':proof['risk_ids']=['0'*64]
    if fault=='missing_history':del proof['frames']['83']
    if fault=='wrong_sha':r['rgb_sha256']='0'*64
    if fault=='future':r['observed_until']=89
    if fault=='identity':r['participant_identity_reviewed']=False
    if fault=='scope':r['transition_scope_reviewed']=False
    if fault=='boundary':r['instance_boundary_reviewed']=False
    with pytest.raises(ValueError):build([a],DATA,tmp_path/'bad',rgb_mode=4)
    assert not (tmp_path/'bad').exists()


def test_known_route_cannot_use_legacy_interval_even_with_review(tmp_path):
    a=reviewed(2);a.update(label_basis='per_frame_conditions',start=89,end=91)
    with pytest.raises(ValueError,match='requires reviewed_transition_band'):build([a],DATA,tmp_path/'bad',rgb_mode=2)


def test_cutoff_still_excludes_old_question_even_with_usable_review(tmp_path):
    a=reviewed(2);b=a['transition_band'];b['stop']=dict(frame=90,kind='calibration_anomaly',reason='SYNTHETIC anomaly')
    for f in ('90','91'):b['frames'][f]['phase']='excluded'
    build([a],DATA,tmp_path/'data',rgb_mode=2);data,m=load_dataset(tmp_path/'data')
    assert [r['observation']['frame_id'] for r in data['train']]==[89]
    assert m['risk_review_check']['row_dispositions']=={'usable':1}
    from qwen3vl_local.sft_new_loop_phase4.preflight import inspect
    assert inspect(tmp_path/'data')['risk_review']==m['risk_review_check']


@pytest.mark.parametrize('fault',['remove_proof','supervise_excluded'])
def test_loader_enforces_risk_contract_after_manifest_rehash(tmp_path,fault):
    a=reviewed(2)
    if fault=='supervise_excluded':a['risk_review']['frames']['85']['status']='excluded'
    folder=tmp_path/'data';m=build([a],DATA,folder,rgb_mode=2)
    if fault=='remove_proof':
        rows=read_rows(folder/'train.jsonl');del rows[0]['risk_review'];dump_rows(folder/'train.jsonl',rows)
        m['files']['train.jsonl']=file_sha(folder/'train.jsonl')
    else:
        rows=read_rows(folder/'review_queue.jsonl');rows[0]['target']='YES';dump_rows(folder/'review_queue.jsonl',rows)
        m['review_queue_sha256']=file_sha(folder/'review_queue.jsonl')
    write_json(folder/'manifest.json',m)
    with pytest.raises(ValueError,match='risk'):load_dataset(folder)


def test_registry_and_risk_proof_never_rendered_as_model_hints(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key
    a=reviewed(2);build([a],DATA,tmp_path/'data',rgb_mode=2);data,_=load_dataset(tmp_path/'data')
    row=data['train'][0];other=deepcopy(row);other['risk_review']['evidence']['reviewer']='other'
    assert model_input_key(row)==model_input_key(other)
