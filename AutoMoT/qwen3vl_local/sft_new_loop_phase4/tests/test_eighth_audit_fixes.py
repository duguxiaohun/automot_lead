"""Independent restrictions survive uncertainty; hidden IDs cannot hide label conflicts."""
from copy import deepcopy
import json
import pytest
from PIL import Image, PngImagePlugin
from qwen3vl_local.sft_new_loop_phase4.controller import Episode,combine_priors
from qwen3vl_local.sft_new_loop_phase4.runtime import Phase4Loop
from qwen3vl_local.sft_new_loop_phase4.action_prior import export
from qwen3vl_local.sft_new_loop_phase4.dataset import build,load_dataset,read_rows,dump_rows,groups,split_for
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,digest,file_sha,write_json
from qwen3vl_local.sft_new_loop_phase4.input_identity import model_input_key,validate_model_inputs,rgb_content_sha
from qwen3vl_local.sft_new_loop_phase4.model import load_images
from qwen3vl_local.sft_new_loop_phase4.taxonomy import get_edge


def answers(ep,**overrides):
    return dict({q.key:'NO' for q in ep.questions()},**overrides)


@pytest.mark.parametrize('lon',['STABLE','APPROACH','FOLLOW','RECOVER'])
@pytest.mark.parametrize('unknown',['explicit','missing'])
def test_known_hold_survives_unknown_or_missing_permission(lon,unknown):
    ep=Episode('U-E4','a',longitudinal=lon)
    a=answers(ep,hold='YES',proceed='UNKNOWN')
    if unknown=='missing':del a['proceed']
    p=ep.advance(10,a)
    assert p['status']=='UNKNOWN' and p['action']=='STOP'
    assert ep.state=='YIELD' and ep.longitudinal=='HOLD' and not ep.finished
    assert ep.history[-1]['accepted']==['hold'] and ep.history[-1]['partial_restriction']
    assert not ep.history[-1]['pending_start'] and not ep.history[-1]['rollback']
    ep.refresh_uncommitted()
    assert combine_priors([ep])['action']=='STOP' and export([ep])['token_name']=='STOP'
    with pytest.raises(ValueError,match='no pending'):ep.acknowledge(10)
    assert ep.advance(11,answers(ep,proceed='UNKNOWN'))['action']=='STOP'
    assert ep.advance(12,answers(ep,proceed='YES'))['action']=='RESUME'
    ep.refresh_uncommitted()
    assert ep.longitudinal=='HOLD'


@pytest.mark.parametrize('event',['U-E1','U-E3','U-E4','U-E5','U-E6','U-E7','R-E5'])
def test_reyield_under_uncertainty_preserves_prior_stop(event):
    ep=Episode(event,'a',state='PROCEED',longitudinal='HOLD')
    p=ep.advance(10,answers(ep,re_yield='YES',release='UNKNOWN'))
    assert p['action']=='STOP' and p['status']=='UNKNOWN'
    assert (ep.state,ep.longitudinal)==('YIELD','HOLD')


def test_uncertain_restriction_never_grants_progress_or_clears_on_timeout():
    ep=Episode('U-E4','a',stall_limit=2)
    assert ep.advance(10,answers(ep,restrict='YES',proceed='UNKNOWN'))['action']=='DECELERATE'
    assert ep.advance(11,answers(ep,hold='YES',proceed='UNKNOWN'))['action']=='STOP'
    assert ep.needs_recheck and ep.prior()['status']=='RECHECK'
    assert ep.state=='YIELD' and not ep.finished


@pytest.mark.parametrize('bad',['invalid','contradiction'])
def test_untrusted_or_explicitly_contradictory_hold_requires_recheck(bad):
    ep=Episode('U-E4','a')
    a=answers(ep,hold='YES',restrict='UNKNOWN',proceed='YES' if bad=='contradiction' else 'UNKNOWN')
    p=ep.advance(10,a,context_valid=bad!='invalid')
    assert p['status']=='RECHECK' and ep.longitudinal=='STABLE'
    assert not ep.history


def test_runtime_unknown_stop_survives_snapshot_and_export():
    loop=Phase4Loop(rgb_mode=2)
    loop.establish(Episode('U-E4','a',longitudinal='FOLLOW'),verified=True)
    p=loop.tick(dict(frame_id=10,history_frames=[6,10],speed_mps=0),[None]*2,
                lambda ep,key,*args:'YES' if key=='hold' else 'UNKNOWN' if key=='proceed' else 'NO')
    assert p['prior']['action']=='STOP' and p['prior']['status']=='UNKNOWN'
    restored=Phase4Loop.restore(loop.snapshot())
    assert restored.snapshot()==loop.snapshot()
    assert export(list(restored.episodes.values()))['token_name']=='STOP'


def synthetic_annotations():
    a=deepcopy(next(a for a in json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text())
                    if a['label_basis']=='per_frame_conditions' and a['edge']=='proceed'))
    a['end']=a['start'];a['episode']['instance_id']='SYNTHETIC_a';a['evidence_id']='SYNTHETIC_a'
    edge=get_edge(a['episode']['event'],a['edge'])
    a['facts']={k:True for k in edge.criteria}
    b=deepcopy(a);b['episode']['instance_id']='SYNTHETIC_b';b['evidence_id']='SYNTHETIC_b'
    return a,b,edge


@pytest.mark.parametrize('mode',[2,4])
def test_builder_rejects_opposite_answers_across_instances_before_publication(tmp_path,mode):
    a,b,edge=synthetic_annotations();b['facts'][edge.criteria[0]]=False
    output=tmp_path/'conflict'
    with pytest.raises(ValueError,match='conflicting answers for identical model input'):
        build([a,b],ROOT.parents[1]/'lead_data',output,rgb_mode=mode)
    assert not output.exists()


@pytest.fixture
def dataset(tmp_path):
    a,b,_=synthetic_annotations()
    out=tmp_path/'data';build([a,b],ROOT.parents[1]/'lead_data',out,rgb_mode=2)
    data,m=load_dataset(out)
    return out,data['train'],m


def test_same_answer_duplicates_are_visible_without_silent_relabeling(dataset):
    _,rows,m=dataset
    assert len(rows)==2 and rows[0]['model_input_sha256']==rows[1]['model_input_sha256']
    assert m['model_input_check']['same_answer_duplicates']==1
    unknown=deepcopy(rows[0]);unknown['target']='UNKNOWN'
    assert validate_model_inputs(rows+[unknown])==m['model_input_check']


def test_loader_rechecks_conflicts_even_when_file_hash_was_updated(dataset):
    out,rows,m=dataset
    rows[1]['target']='NO';dump_rows(out/'train.jsonl',rows)
    m['files']['train.jsonl']=file_sha(out/'train.jsonl');write_json(out/'manifest.json',m)
    with pytest.raises(ValueError,match='conflicting answers for identical model input'):load_dataset(out)


def test_key_uses_visible_order_and_text_not_hidden_identity(dataset):
    _,rows,_=dataset;r=deepcopy(rows[0]);key=model_input_key(r)
    r['episode']['instance_id']='different';r['observation']['speed_mps']=99
    r['observation']['frame_id']+=20;r['observation']['history_frames']=[f+20 for f in r['observation']['history_frames']]
    assert model_input_key(r)==key
    r['image_rgb_sha256'].reverse()
    assert model_input_key(r)!=key
    r=deepcopy(rows[0]);r['prompt_sha256']=digest('forged text')
    with pytest.raises(ValueError,match='prompt identity'):model_input_key(r)


def test_decoded_identity_ignores_file_metadata_and_verifies_actual_pixels(tmp_path,dataset):
    im=Image.new('RGB',(8,9),'red');a=tmp_path/'a.png';b=tmp_path/'b.png'
    im.save(a);meta=PngImagePlugin.PngInfo();meta.add_text('note','same input')
    im.save(b,pnginfo=meta)
    assert file_sha(a)!=file_sha(b)
    with Image.open(a) as x,Image.open(b) as y:assert rgb_content_sha(x)==rgb_content_sha(y)
    _,rows,_=dataset;r=deepcopy(rows[0]);r['image_rgb_sha256'][0]='0'*64
    with pytest.raises(ValueError,match='decoded RGB'):load_images(r,ROOT.parents[1]/'lead_data')


def test_eighth_exposure_is_train_only():
    audit=json.loads((ROOT/'eighth_audit_exposure_20260930.json').read_text())
    for group in audit['train_only_groups']:
        for seed in (1,2026,20260930):assert split_for(group,groups(),seed)=='train'
    assert not audit['reported_coverage']['all_requested_rgb_audit_complete']
