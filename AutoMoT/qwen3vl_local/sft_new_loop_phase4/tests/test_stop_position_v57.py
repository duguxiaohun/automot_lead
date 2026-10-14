"""Production STOP binding and revocation. Synthetic cases do not approve labels."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from qwen3vl_local.sft_new_loop_phase4 import teacher_controls as controls,teacher_rules as rules
from qwen3vl_local.sft_new_loop_phase4 import stop_position as position
from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher_v29 import controlled_frames
from qwen3vl_local.sft_new_loop_phase4.tests.test_stop_v55 import clean,observe


def test_negative_trigger_gap_does_not_prevent_matched_incoming_boundary_stop():
    fs=clean(controlled_frames());observe(fs)
    f=fs[2];record=f['stop_duties']['50']
    assert controls.g.aabb(f['actors'][-1])[0]-f['actors'][0]['extent'][0]<0
    assert record['front_gap_m']==pytest.approx(2.,abs=.01)
    assert record['satisfied'] and len(record['observations'])==3
    assert record['position_binding']['catalog_key']=='Town12/225/-1'
    assert rules.local_priority(f,rules.local_navigation(f,18.)) is True


@pytest.mark.parametrize('fault',['displaced','shape','lane','town','pose'])
def test_unknown_or_mismatched_initial_boundary_never_falls_back_to_trigger_proxy(fault):
    fs=clean(controlled_frames())
    for f in fs:
        if fault=='displaced':f['actors'][-1]['position'][0]+=.1
        elif fault=='shape':f['actors'][-1]['extent'][0]=1.
        elif fault=='lane':f['meta']['lane_id']=999;f['actors'][0]['lane_id']=999
        elif fault=='town':f['route_id']='Town99_Rep0_1_0_route0'
        elif fault=='pose':f['meta']['ego_matrix'][0][0]=3.
    observe(fs)
    r=fs[-1]['stop_duties']['50']
    assert not r['satisfied'] and not r['identity_verified'] and r['observations']==[]
    assert rules.local_priority(fs[-1],rules.local_navigation(fs[-1],18.)) is None


@pytest.mark.parametrize('fault',['drift','shape','missing','source','gap','foreign_route'])
def test_evidence_loss_revokes_previously_completed_stop_without_erasing_debt(fault):
    fs=clean(controlled_frames());tracker=observe(fs[:3]);assert fs[2]['stop_duties']['50']['satisfied']
    f=fs[3];f['actors'][-1]['affects_ego']=False
    if fault=='drift':f['actors'][-1]['position'][1]+=.1
    elif fault=='shape':f['actors'][-1]['extent'][0]=1.
    elif fault=='missing':f['actors'].pop()
    elif fault=='source':f['source_geometry_issues']=['invalid']
    elif fault=='gap':f['frame_id']+=3
    elif fault=='foreign_route':f['route_id']='Town12_Rep0_9_0_route0'
    f['stop_duties']=tracker.observe(f)
    assert 50 in tracker.entries and not f['stop_duties']['50']['satisfied']
    assert rules.local_priority(f,rules.local_navigation(f,18.)) is None


def test_map_asset_change_revokes_completion(monkeypatch,tmp_path):
    path=tmp_path/'catalog.json';path.write_bytes(position.CATALOG.read_bytes())
    monkeypatch.setattr(position,'catalog',lambda:position.Catalog(path))
    fs=clean(controlled_frames());tracker=observe(fs[:3])
    path.write_bytes(path.read_bytes()+b'\n')
    r=tracker.observe(fs[3])['50']
    assert not r['satisfied'] and 'changed' in r['position_error'] and 50 in tracker.entries


def test_missing_catalog_abstains_in_native_tracker(monkeypatch):
    fs=clean(controlled_frames())
    def missing():raise FileNotFoundError('missing catalog')
    monkeypatch.setattr(position,'catalog',missing);observe(fs)
    assert not fs[-1]['stop_duties']['50']['satisfied']
    assert 'missing catalog' in fs[-1]['stop_duties']['50']['position_error']


@pytest.mark.parametrize('field',['catalog_sha256','map_sha256','signal_id','catalog_key','route_id'])
def test_detached_binding_rejected_by_native_question_validation(field):
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question,seal
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import validate_question
    fs=clean(controlled_frames());observe(fs)
    q=question(route=fs[-1]['route_id']);q['control_evidence']=deepcopy(fs[-1]['stop_duties'])
    validate_question(seal(q))
    q['control_evidence']['50']['position_binding'][field]='wrong'
    with pytest.raises(ValueError):validate_question(seal(q))


def test_impossible_parking_observation_cannot_be_declared_completed():
    from qwen3vl_local.sft_new_loop_phase4.tests.test_teacher import question,seal
    from qwen3vl_local.sft_new_loop_phase4.teacher_replay import validate_question
    fs=clean(controlled_frames());observe(fs);q=question(route=fs[-1]['route_id']);q['control_evidence']=deepcopy(fs[-1]['stop_duties'])
    q['control_evidence']['50']['observations'][0]['front_gap_m']=-1.
    with pytest.raises(ValueError,match='position or speed'):validate_question(seal(q))
