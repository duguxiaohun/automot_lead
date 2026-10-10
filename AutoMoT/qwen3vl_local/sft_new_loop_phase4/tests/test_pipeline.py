import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase4 import dataset
from qwen3vl_local.sft_new_loop_phase4.identity import ROOT,contract,file_sha
from qwen3vl_local.sft_new_loop_phase4.runtime import handoff
from qwen3vl_local.sft_new_loop_phase4.controller import Episode
from qwen3vl_local.sft_new_loop_phase4.action_prior import export


def test_phase12_handoff_does_not_turn_highway_into_merge():
    p1=dict.fromkeys(('HIGHWAY','STATIC_OBSTACLE','VULNERABLE','TRAFFIC_LIGHT_ABNORMAL','RS1','RS2','RS4','RS5'),False)
    p2=dict.fromkeys(('UE1','UE3','UE5','UE6','INVALID_EVENT_CONTEXT'),False)
    p1['HIGHWAY']=True
    loop=handoff(p1,p2,[],[])
    assert not loop.episodes
    with pytest.raises(ValueError,match='not established'):
        handoff(p1,p2,[],[dict(event='R-E3',instance_id='x')])
    p1['STATIC_OBSTACLE']=True
    loop=handoff(p1,p2,[],[dict(event='U-E2',instance_id='x',branch='in_lane_pass')])
    assert loop.episodes['x'].branch=='in_lane_pass'
    p2['INVALID_EVENT_CONTEXT']=True
    with pytest.raises(ValueError):
        handoff(p1,p2,[],[])


def test_export_abstention_never_becomes_uncond():
    ep=Episode('U-E4','x',uncertain=True)
    prior=export([ep])
    assert not prior['valid'] and prior['token_id'] is None
    ep.uncertain=False;ep.state='DONE'
    assert export([ep])['token_id']==0


@pytest.mark.parametrize('name',['train','evaluate','preflight','demo','dataset','audit_bundle','audit_review','replay'])
def test_direct_cli_outside_repo(name,tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/(name+'.py')),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_review_coverage_and_interval_boundaries():
    review=json.loads((ROOT/'rgb_review_20260929.json').read_text())
    notes=json.loads((ROOT/'review_notes.json').read_text())
    assert review['windows']==len(notes)==92
    assert review['unique_rgb_paths']==1416
    rs={r['review_id']:r for r in review['reviews']}
    for a in json.loads((ROOT/'reviewed_intervals_20260929.json').read_text()):
        r=rs[a['evidence_id']]
        assert {str(f) for f in range(a['start'],a['end']+1)}==set(a['frame_sha256'])
        for f in r['frames']:
            if str(f['frame_id']) in a['frame_sha256']:
                assert f['rgb_sha256']==a['frame_sha256'][str(f['frame_id'])]


def test_real_data_roundtrip_is_train_only_and_tamper_rejected(tmp_path):
    data_root=ROOT.parents[1]/'lead_data'
    if not data_root.exists():
        pytest.skip('LEAD RGB is external; not a synthetic production readiness assertion')
    ann=json.loads((ROOT/'reviewed_state_pairs_v4.json').read_text())
    output=tmp_path/'dataset'
    m=dataset.build(ann,data_root,output)
    assert m['counts']=={'train':340} and not m['ready']
    assert m['review_queue_count']==246
    data,_=dataset.load_dataset(output)
    assert not data['val'] and not data['test']
    assert {r['target'] for r in data['train']}=={'YES','NO'}
    assert any(r['slice']=='catchup' for r in data['train'])
    with pytest.raises(ValueError,match='support'):
        dataset.load_dataset(output,require_complete_coverage=True)
    with (output/'train.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='hash'):
        dataset.load_dataset(output)


def test_real_data_rgb_swap_and_future_timing_labels_rejected(tmp_path):
    ann=json.loads((ROOT/'reviewed_intervals_20260929.json').read_text())[:1]
    data_root=ROOT.parents[1]/'lead_data'
    if not data_root.exists():pytest.skip('external LEAD unavailable')
    bad=copy.deepcopy(ann);bad[0]['label_basis']='future_action_onset'
    with pytest.raises(ValueError,match='permission'):
        dataset.build(bad,data_root,tmp_path/'bad')
    bad=copy.deepcopy(ann);bad[0]['frame_sha256'][str(bad[0]['start'])]='0'*64
    with pytest.raises(ValueError,match='RGB identity'):
        dataset.build(bad,data_root,tmp_path/'bad2')


def test_catchup_yes_cannot_fill_readiness_support():
    r=dict(split='train',episode={'event':'U-E2'},edge='depart',target='YES',slice='catchup',physical_group='x')
    report=dataset.coverage_report([r])
    assert 'train/U-E2/default/return/depart' in report['missing_transition_edges']


def test_gpu_selection_respects_visibility_and_explicit_choice(monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import devices
    monkeypatch.delenv('GPU_IDS',raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-b,2')
    monkeypatch.setattr(devices.subprocess,'check_output',lambda *a,**k:'0, GPU-a, 0, 0\n1, GPU-b, 500, 0\n2, GPU-c, 100, 0\n3, GPU-d, 10, 0\n')
    assert devices.select_gpus()=='2,1'
    monkeypatch.setenv('GPU_IDS','3,0')
    assert devices.select_gpus(1)=='3'
