"""Synthetic LEAD metas/bboxes exercise every objective exclusion rule; holdouts stay byte-identical."""
import json
import lzma
import pickle
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint.phase12_retrain import phase2_evidence as ev
from qwen3vl_local.sft_new_loop_phase2.build_dataset import _make_row

SC = 'Synthetic'


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, 'wb') as f:
        pickle.dump(obj, f)


def _actor(aid, x, y, *, speed=8.0, brake=0.0, yaw=0.0, cls='car'):
    return dict(id=aid, position=[x, y, 0.0], speed=speed, brake=brake, yaw=yaw, **{'class': cls})


def make_route(root, route, frames, *, lead=None, scenario_actor=None, cut_in=None, ego_speed=6.0, tl='Green'):
    """lead/scenario_actor/cut_in: callables frame->actor dict (or None)."""
    for f in range(frames):
        meta = dict(speed=ego_speed(f) if callable(ego_speed) else ego_speed, traffic_light_state=tl, vehicle_affecting_id=-1,
                    speed_reduced_by_obj_id=None, scenario_actors_ids=[], cut_in_actors_ids=[], ego_lane_id=4, road_id=1, ego_lane_width=3.5)
        boxes = [dict(id=1, position=[0, 0, 0], speed=meta['speed'], **{'class': 'ego_car'})]
        for fn, key in ((lead, 'vehicle_affecting_id'), (scenario_actor, 'scenario_actors_ids'), (cut_in, 'cut_in_actors_ids')):
            a = fn(f) if fn else None
            if a is None:
                continue
            boxes.append(a)
            if key == 'vehicle_affecting_id':
                meta[key] = a['id']
            else:
                meta[key] = [a['id']]
        _write(root / SC / route / 'metas' / f'{f:04d}.pkl', meta)
        _write(root / SC / route / 'bboxes' / f'{f:04d}.pkl', boxes)


def row(route, frame, target, split='train', **extra):
    paths = [f'{SC}/{route}/rgb/{max(0, f):04d}.jpg' for f in range(frame - 3, frame + 1)]
    base = dict(scenario=SC, route_id=route, frame_id=frame, town='Town12', split=split, rs='R3' if target == 'UE3' else 'R1',
                primary_event='U-E1' if target == 'UE1' else 'R-E1', event_codes=['R-E1'], ue3_subtype=extra.pop('ue3_subtype', ''),
                history_rgb_paths=paths, latest_rgb_path=paths[-1], visual_label_risk=False, visual_label_risk_reasons=[])
    domain = 'LOCAL_JUNCTION' if target == 'UE6' else 'ROAD_CORRIDOR'
    if target == 'INVALID':
        return _make_row(base=base, question_domain='LOCAL_JUNCTION', target_class='INVALID', invalid=True, invalid_source='source=RE|true_rs=R1|question_domain=LOCAL_JUNCTION')
    r = _make_row(base=base, question_domain=domain, target_class=target, invalid=False)
    r.update(extra)
    return r


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'lead'
    # steady lead: constant speed, no brake, gap 15m -> UE1 YES must be excluded
    make_route(root, 'steady', 30, lead=lambda f: _actor(7, 15.0, 0.1))
    # braking lead: brake lamp + speed drop at f>=18
    make_route(root, 'braking', 30, lead=lambda f: _actor(7, 14.0, 0.1, speed=8.0 if f < 18 else 3.0, brake=0.7 if f >= 18 else 0.0))
    # long queue: both stopped from f0 (lead stopped 30 frames) ; recent stop at f=20
    make_route(root, 'queue', 30, lead=lambda f: _actor(7, 9.0, 0.1, speed=0.0), ego_speed=0.0)
    make_route(root, 'recent', 30, lead=lambda f: _actor(7, 9.0, 0.1, speed=8.0 if f < 20 else 0.0), ego_speed=lambda f: 6.0 if f < 20 else 0.0)
    # junction: violator crossing in front at f 10..14, absent before, behind after 20
    def viol(f):
        if f < 8: return None
        if f < 15: return _actor(9, 8.0, 12.0 - (f - 8) * 4.0, speed=16.0)
        if f < 22: return _actor(9, 8.0, -16.0 - (f - 15) * 4.0, speed=16.0)
        return _actor(9, -12.0, -40.0, speed=16.0)
    make_route(root, 'junction', 30, scenario_actor=viol, ego_speed=0.0)
    # highway: adjacent-lane car never entering vs. one that enters
    make_route(root, 'adjacent', 30, cut_in=lambda f: _actor(11, 6.0, -3.6, yaw=0.08))
    make_route(root, 'entering', 30, cut_in=lambda f: _actor(11, 11.0, max(0.2, 3.5 - 0.3 * max(0, f - 10)), yaw=-0.3 if f >= 10 else 0.0))
    return root


def _index(tmp_path, rows):
    parent = tmp_path / 'parent'; parent.mkdir()
    (parent / 'frame_index.jsonl').write_bytes(b''.join((json.dumps(r) + '\n').encode() for r in rows))
    (parent / 'manifest.json').write_text(json.dumps({'counts': {'frames/train': len(rows)}, 'sampling': {'old': True}}))
    return parent / 'frame_index.jsonl'


def test_rules_and_holdout_preservation(world, tmp_path):
    rows = [row('steady', 20, 'UE1'), row('braking', 20, 'UE1'), row('queue', 20, 'UE1'), row('recent', 22, 'UE1'),
            row('braking', 20, 'RE'), row('steady', 20, 'RE'),
            row('junction', 5, 'UE6'), row('junction', 12, 'UE6'), row('junction', 16, 'UE6'), row('junction', 25, 'UE6'),
            row('adjacent', 20, 'UE3', ue3_subtype='HIGHWAY_CUTIN'), row('entering', 14, 'UE3', ue3_subtype='HIGHWAY_CUTIN'),
            row('entering', 20, 'UE3', ue3_subtype='HIGHWAY_CUTIN'),
            row('steady', 2, 'UE5'), row('steady', 20, 'UE5'), row('steady', 20, 'INVALID'),
            row('steady', 20, 'UE1', split='val'), row('junction', 25, 'UE6', split='test')]
    index = _index(tmp_path, rows)
    before = index.read_bytes()
    receipt = ev.derive(index, world, tmp_path / 'out')
    kept = [json.loads(l) for l in (tmp_path / 'out/frame_index.jsonl').read_text().splitlines()]
    excluded = [json.loads(l) for l in (tmp_path / 'out/evidence_exclusions.jsonl').read_text().splitlines()]
    key = lambda r: (r['route_id'], r['frame_id'], r.get('question_domain'), r.get('invalid_source', ''))
    ex = {key(r): r['evidence'] for r in excluded}
    assert ex[('steady', 20, 'ROAD_CORRIDOR', '')] == 'steady_lead'
    assert ex[('queue', 20, 'ROAD_CORRIDOR', '')] == 'long_both_stopped'
    assert ex[('braking', 20, 'ROAD_CORRIDOR', '')] == 'lead_brake_signal' and [r for r in excluded if r['route_id'] == 'braking'][0]['rule'] == 'regular_negative_with_lead_braking'
    assert ex[('junction', 5, 'LOCAL_JUNCTION', '')] == 'actor_absent'
    assert ex[('junction', 25, 'LOCAL_JUNCTION', '')] == 'actor_behind_ego'
    assert ex[('adjacent', 20, 'ROAD_CORRIDOR', '')] == 'adjacent_lane_only'
    assert ex[('steady', 2, 'ROAD_CORRIDOR', '')] == 'history_left_padded'
    kept_keys = {key(r) for r in kept if r['split'] == 'train'}
    assert sum(1 for r in kept if r['route_id'] == 'braking' and r['answers']['UE1']) == 1
    assert sum(1 for r in kept if r['route_id'] == 'braking' and not r['answers']['UE1']) == 0
    assert any(r['route_id'] == 'recent' for r in kept) and any(r['route_id'] == 'entering' for r in kept)
    assert sum(1 for r in kept if r['route_id'] == 'junction' and r['split'] == 'train') == 2  # f12 conflict, f16 just cleared
    assert any(r['route_id'] == 'steady' and r['answers']['UE5'] and r['frame_id'] == 20 for r in kept)
    assert any(r['invalid_event_context'] for r in kept)
    assert receipt['labels_flipped'] == 0 and receipt['holdouts_byte_identical'] and receipt['train_excluded'] == len(excluded) == 7
    assert [r for r in kept if r['split'] != 'train'] == [r for r in rows if r['split'] != 'train']
    assert index.read_bytes() == before
    assert receipt['train_class_counts_after']['UE1'] == 2 and receipt['train_class_counts_after']['UE6'] == 2
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert manifest['counts']['frames/train/UE1'] == 2 and 'sampling' in manifest['parent_statistics']
    assert (tmp_path / 'selected_train_reviews.json').exists()
    disputes = ev.audit(tmp_path / 'out/frame_index.jsonl', world)
    assert {(d['split'], d['route_id'], d['evidence']) for d in disputes} == {('val', 'steady', 'steady_lead'), ('test', 'junction', 'actor_behind_ego')}
    with pytest.raises(ValueError):
        ev.derive(index, world, tmp_path / 'out')


def test_required_class_cannot_be_emptied(world, tmp_path):
    rows = [row('steady', 20, t) for t in ('UE1', 'UE3', 'UE5', 'RE', 'INVALID')] + [row('junction', 5, 'UE6')]
    with pytest.raises(ValueError):
        ev.derive(_index(tmp_path, rows), world, tmp_path / 'empty')
    assert not (tmp_path / 'empty').exists()


def test_missing_meta_is_reported_not_crashed(world, tmp_path):
    rows = [row('nowhere', 20, 'UE1'), row('braking', 20, 'UE1'), row('steady', 20, 'UE3'), row('steady', 20, 'UE5'),
            row('junction', 12, 'UE6'), row('steady', 20, 'RE'), row('steady', 20, 'INVALID')]
    receipt = ev.derive(_index(tmp_path, rows), world, tmp_path / 'out2')
    assert receipt['exclusion_reasons'] == {'ue1_positive_without_braking_evidence/no_meta': 1}
