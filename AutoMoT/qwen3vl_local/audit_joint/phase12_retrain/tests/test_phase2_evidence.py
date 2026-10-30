"""Synthetic LEAD metas/bboxes exercise every objective rule; test stays byte-identical, val is filtered with originals kept."""
import json
import lzma
import pickle
import shutil
from pathlib import Path

import pytest
from qwen3vl_local.audit_joint.phase12_retrain import phase2_evidence as ev
from qwen3vl_local.audit_joint.phase12_retrain import launch
from qwen3vl_local.sft_new_loop_phase2.build_dataset import _make_row
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1 import evaluation_integrity as evaluation

SC = 'Synthetic'


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, 'wb') as f:
        pickle.dump(obj, f)


def _actor(aid, x, y, *, speed=8.0, brake=0.0, yaw=0.0, cls='car'):
    return dict(id=aid, position=[x, y, 0.0], speed=speed, brake=brake, yaw=yaw, **{'class': cls})


def make_route(root, route, frames, *, lead=None, scenario_actor=None, cut_in=None, ego_speed=6.0, tl='Green'):
    for f in range(frames):
        meta = dict(speed=ego_speed(f) if callable(ego_speed) else ego_speed, traffic_light_state=tl, vehicle_affecting_id=-1,
                    speed_reduced_by_obj_id=None, scenario_actors_ids=[], cut_in_actors_ids=[], ego_lane_id=4, road_id=1, ego_lane_width=3.5)
        boxes = [dict(id=1, position=[0, 0, 0], speed=meta['speed'], **{'class': 'ego_car'})]
        for fn, key in ((lead, 'vehicle_affecting_id'), (scenario_actor, 'scenario_actors_ids'), (cut_in, 'cut_in_actors_ids')):
            a = fn(f) if fn else None
            if a is None:
                continue
            boxes.append(a)
            meta[key] = a['id'] if key == 'vehicle_affecting_id' else [a['id']]
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
    make_route(root, 'steady', 30, lead=lambda f: _actor(7, 15.0, 0.1))
    make_route(root, 'braking', 30, lead=lambda f: _actor(7, 14.0, 0.1, speed=8.0 if f < 18 else 3.0, brake=0.7 if f >= 18 else 0.0))
    make_route(root, 'queue', 30, lead=lambda f: _actor(7, 9.0, 0.1, speed=0.0), ego_speed=0.0)
    make_route(root, 'recent', 30, lead=lambda f: _actor(7, 9.0, 0.1, speed=8.0 if f < 20 else 0.0), ego_speed=lambda f: 6.0 if f < 20 else 0.0)
    # lead stopped long ago (whole route); ego cruising toward it at constant speed -> ordinary approach
    make_route(root, 'approach', 30, lead=lambda f: _actor(7, 12.0, 0.1, speed=0.0), ego_speed=4.0)
    # lead stopped long ago; ego still braking hard toward it (8 -> 2 m/s within the window) -> event continues
    make_route(root, 'response', 30, lead=lambda f: _actor(7, 12.0, 0.1, speed=0.0), ego_speed=lambda f: max(2.0, 8.0 - 0.75 * max(0, f - 12)))

    def viol(f):
        if f < 8: return None
        if f < 15: return _actor(9, 8.0, 12.0 - (f - 8) * 4.0, speed=16.0)
        if f < 22: return _actor(9, 8.0, -16.0 - (f - 15) * 4.0, speed=16.0)
        return _actor(9, -12.0, -40.0, speed=16.0)
    make_route(root, 'junction', 30, scenario_actor=viol, ego_speed=0.0)
    make_route(root, 'adjacent', 30, cut_in=lambda f: _actor(11, 6.0, -3.6, yaw=0.08))
    make_route(root, 'entering', 30, cut_in=lambda f: _actor(11, 11.0, max(0.2, 3.5 - 0.3 * max(0, f - 10)), yaw=-0.3 if f >= 10 else 0.0))
    return root


def _index(tmp_path, rows):
    parent = tmp_path / 'parent'; parent.mkdir(parents=True)
    (parent / 'frame_index.jsonl').write_bytes(b''.join((json.dumps(r) + '\n').encode() for r in rows))
    (parent / 'manifest.json').write_text(json.dumps({'counts': {'frames/train': len(rows)}, 'sampling': {'old': True}}))
    return parent / 'frame_index.jsonl'


def _full(extra=()):
    rows = [row('steady', 20, 'UE1'), row('braking', 20, 'UE1'), row('queue', 20, 'UE1'), row('recent', 22, 'UE1'),
            row('approach', 24, 'UE1'), row('response', 24, 'UE1'),
            row('braking', 20, 'RE'), row('steady', 20, 'RE'),
            row('junction', 5, 'UE6'), row('junction', 12, 'UE6'), row('junction', 16, 'UE6'), row('junction', 25, 'UE6'),
            row('adjacent', 20, 'UE3', ue3_subtype='HIGHWAY_CUTIN'), row('entering', 14, 'UE3', ue3_subtype='HIGHWAY_CUTIN'),
            row('entering', 20, 'UE3', ue3_subtype='HIGHWAY_CUTIN'),
            row('steady', 2, 'UE5'), row('steady', 20, 'UE5'), row('steady', 20, 'INVALID')]
    val = [row('steady', 20, t, split='val') for t in ('UE1', 'UE3', 'UE5', 'RE', 'INVALID')] + [row('junction', 12, 'UE6', split='val'),
           row('braking', 20, 'UE1', split='val'), row('junction', 25, 'UE6', split='val')]
    test = [row('steady', 20, 'UE1', split='test'), row('junction', 25, 'UE6', split='test'), row('entering', 20, 'UE3', split='test', ue3_subtype='HIGHWAY_CUTIN')]
    return rows + val + test + list(extra)


def test_rules_val_selection_view_and_test_preservation(world, tmp_path):
    rows = _full()
    index = _index(tmp_path, rows)
    before = index.read_bytes()
    receipt = ev.derive(index, world, tmp_path / 'out')
    out = tmp_path / 'out'
    kept = [json.loads(l) for l in (out / 'frame_index.jsonl').read_text().splitlines()]
    excluded = [json.loads(l) for l in (out / 'evidence_exclusions.jsonl').read_text().splitlines()]
    key = lambda r: (r['route_id'], r['frame_id'], r.get('question_domain'), r.get('invalid_source', ''))
    ex = {key(r): r['evidence'] for r in excluded}
    assert ex[('steady', 20, 'ROAD_CORRIDOR', '')] == 'steady_lead'
    assert ex[('queue', 20, 'ROAD_CORRIDOR', '')] == 'long_both_stopped'
    assert ex[('approach', 24, 'ROAD_CORRIDOR', '')] == 'approaching_long_stopped_lead'
    assert ex[('braking', 20, 'ROAD_CORRIDOR', '')] == 'lead_brake_signal' and [r for r in excluded if r['route_id'] == 'braking'][0]['rule'] == 'regular_negative_with_lead_braking'
    assert ex[('junction', 5, 'LOCAL_JUNCTION', '')] == 'actor_absent'
    assert ex[('junction', 25, 'LOCAL_JUNCTION', '')] == 'actor_behind_ego'
    assert ex[('adjacent', 20, 'ROAD_CORRIDOR', '')] == 'adjacent_lane_only'
    assert ex[('steady', 2, 'ROAD_CORRIDOR', '')] == 'history_left_padded'
    train_kept = [r for r in kept if r['split'] == 'train']
    assert sum(1 for r in train_kept if r['route_id'] == 'braking' and r['answers']['UE1']) == 1
    assert not any(r['route_id'] == 'braking' and not r['answers']['UE1'] for r in train_kept)
    assert any(r['route_id'] == 'response' for r in train_kept) and any(r['route_id'] == 'recent' for r in train_kept)
    assert sum(1 for r in train_kept if r['route_id'] == 'junction') == 2
    assert receipt['labels_flipped'] == 0 and receipt['train_excluded'] == len(excluded) == 8
    assert receipt['train_class_counts_after'] == {'INVALID': 1, 'RE': 1, 'UE1': 3, 'UE3': 2, 'UE5': 1, 'UE6': 2}
    # val: filtered as the selection view; originals kept; test untouched and disputes listed
    val_kept = [r for r in kept if r['split'] == 'val']
    val_ex = [json.loads(l) for l in (out / 'val_exclusions.jsonl').read_text().splitlines()]
    assert {r['evidence'] for r in val_ex} == {'steady_lead', 'actor_behind_ego'} and receipt['val_excluded'] == 2
    assert len(val_kept) == 6 and receipt['val_class_counts_after']['UE1'] == 1
    original_val = (out / 'val_original.jsonl').read_bytes()
    assert original_val == b''.join(l for l in before.splitlines(keepends=True) if json.loads(l)['split'] == 'val')
    test_lines = [l for l in before.splitlines(keepends=True) if json.loads(l)['split'] == 'test']
    assert [l for l in (out / 'frame_index.jsonl').read_bytes().splitlines(keepends=True) if json.loads(l)['split'] == 'test'] == test_lines
    disputes = [json.loads(l) for l in (out / 'holdout_disputes.jsonl').read_text().splitlines()]
    assert {(d['route_id'], d['evidence']) for d in disputes} == {('steady', 'steady_lead'), ('junction', 'actor_behind_ego')}
    assert receipt['test_disputes'] == 2 and receipt['test_byte_identical'] and not receipt['val_byte_identical']
    sources = json.loads((out / 'evidence_sources.json').read_text())
    assert receipt['evidence_sources'] == len(sources['files']) > 20 and all(len(v) == 64 for v in sources['files'].values())
    assert index.read_bytes() == before
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['counts']['frames/train/UE1'] == 3 and manifest['counts']['frames/val'] == 6 and 'sampling' in manifest['parent_statistics']
    with pytest.raises(ValueError):
        ev.derive(index, world, out)


def test_missing_meta_or_bbox_blocks_instead_of_excluding(world, tmp_path):
    rows = _full([row('nowhere', 20, 'UE1')])
    index = _index(tmp_path, rows)
    with pytest.raises(ValueError, match='missing or unreadable'):
        ev.derive(index, world, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()
    (world / SC / 'junction' / 'bboxes' / '0012.pkl').unlink()
    with pytest.raises(ValueError, match='missing or unreadable'):
        ev.derive(_index(tmp_path / 'b', _full()), world, tmp_path / 'blocked2')
    items = ev.audit(_index(tmp_path / 'c', _full()), world)
    assert any(i['rule'] == 'source_missing' and i['route_id'] == 'junction' for i in items)


def test_required_class_cannot_be_emptied(world, tmp_path):
    rows = [row('steady', 20, t) for t in ('UE1', 'UE3', 'UE5', 'RE', 'INVALID')] + [row('junction', 5, 'UE6')]
    rows += [row('steady', 20, t, split='val') for t in ('UE3', 'UE5', 'RE', 'INVALID')] + [row('junction', 12, 'UE6', split='val'), row('braking', 20, 'UE1', split='val')]
    with pytest.raises(ValueError, match='emptied'):
        ev.derive(_index(tmp_path, rows), world, tmp_path / 'empty')
    assert not (tmp_path / 'empty').exists()


def _native_output(root, mode, cases, world=1):
    """Small CPU fixture of native shard layout and its completion binding."""
    d = root / mode / 'eval_review' / 'lora_production'
    d.mkdir(parents=True)
    cases = [dict(c, case_index=i) for i, c in enumerate(cases)]
    names = ['cases.jsonl'] if world == 1 else [f'cases_rank{r}.jsonl' for r in range(world)]
    for rank, name in enumerate(names):
        (d / name).write_text(''.join(json.dumps(c) + '\n' for c in cases[rank::world]))
    metrics = dict(world_size=world, history_rgb_mode=mode, total_cases=len(cases),
                   exact_match_accuracy=sum(c['all_ok'] for c in cases) / len(cases))
    proof.write(d / 'metrics.json', metrics)
    proof.write(d / 'evaluation_inputs.json', dict(schema=proof.SCHEMA, phase=2,
                history_rgb_mode=mode, expected_total=len(cases)))
    proof.write(d / 'evaluation_complete.json', dict(schema=proof.SCHEMA, phase=2, status='verified',
                total_cases=len(cases), scores=evaluation.score_fields(metrics),
                inputs_sha256=ev.sha(d / 'evaluation_inputs.json'), metrics_sha256=ev.sha(d / 'metrics.json'),
                case_files_sha256={n: ev.sha(d / n) for n in names}))
    return d


def test_dual_score_reports_original_and_retained(tmp_path):
    disputes = tmp_path / 'disputes.jsonl'
    disputes.write_text(json.dumps(dict(scenario='S', route_id='r', frame_id=5, question_domain='ROAD_CORRIDOR', rule='x', evidence='y')) + '\n')
    for mode in ('2rgb_endpoints', '4rgb'):
        cases = [dict(scenario='S', route_id='r', frame_id=5, question_domain='ROAD_CORRIDOR', gt={'UE1': 'YES'}, all_ok=mode == '4rgb'),
                 dict(scenario='S', route_id='r', frame_id=6, question_domain='ROAD_CORRIDOR', gt={'UE1': 'NO'}, all_ok=True),
                 dict(scenario='S', route_id='r', frame_id=7, question_domain='LOCAL_JUNCTION', gt={'INVALID_EVENT_CONTEXT': 'YES'}, all_ok=False)]
        _native_output(tmp_path / 'pipeline', mode, cases)
    report = ev.dual_score(tmp_path / 'pipeline', disputes, tmp_path / 'dual.json')
    m2, m4 = report['modes']['2rgb_endpoints'], report['modes']['4rgb']
    assert m2['original']['correct'] == 1 and m2['original']['total'] == 3 and m2['retained_after_removing_disputed'] == {
        'correct': 1, 'total': 2, 'exact': 0.5, 'by_class': {'INVALID': [0, 1], 'RE': [1, 1]}}
    assert m4['original']['correct'] == 2 and m4['disputed_subset']['by_class'] == {'UE1': [1, 1]}
    assert json.loads((tmp_path / 'dual.json').read_text())['disputed_keys'] == 1
    with pytest.raises(ValueError):
        ev.dual_score(tmp_path / 'nothing', disputes, tmp_path / 'x.json')


@pytest.mark.parametrize('layout', ['cr02', 'cee'])
def test_dual_score_ignores_bundle_copies_preserves_sample_duplicates(tmp_path, layout):
    physical = tmp_path / layout / 'outputs'
    root = tmp_path / 'pipeline'
    if layout == 'cr02':
        physical.mkdir(parents=True)
        root.symlink_to(physical, target_is_directory=True)
    c = dict(scenario='S', route_id='r', frame_id=5, question_domain='ROAD_CORRIDOR',
             gt={'UE1': 'YES'}, all_ok=True)
    for mode in ev.MODES:
        # Same logical sample twice is legal; case_index distinguishes presentations.
        d = _native_output(root, mode, [c, c, dict(c, frame_id=6, all_ok=False)], world=2)
        shutil.copytree(d, d.parent / 'example_audit_bundle' / 'lora_production')
    disputes = tmp_path / 'disputes.jsonl'; disputes.write_text('')
    report = ev.dual_score(root, disputes, tmp_path / 'dual.json')
    for m in report['modes'].values():
        assert m['original']['total'] == 3 and m['original']['correct'] == 2
        assert len(m['case_files']) == 2 and all('_audit_bundle' not in p for p in m['case_files'])


@pytest.mark.parametrize('damage', ['mode_missing', 'shard_missing', 'extra_shard', 'duplicate_index',
                                     'truncated', 'score_changed', 'metric_count', 'wrong_mode', 'receipt_missing'])
def test_dual_score_rejects_incomplete_or_changed_native_output(tmp_path, damage):
    root = tmp_path / 'pipeline'
    c = dict(scenario='S', route_id='r', frame_id=5, question_domain='ROAD_CORRIDOR',
             gt={'UE1': 'YES'}, all_ok=True)
    for mode in ev.MODES:
        d = _native_output(root, mode, [c, c, c], world=2)
    if damage == 'mode_missing':shutil.rmtree(root / '4rgb')
    elif damage == 'shard_missing':(d / 'cases_rank1.jsonl').unlink()
    elif damage == 'extra_shard':(d / 'cases_rank2.jsonl').write_text('')
    elif damage == 'duplicate_index':
        lines=(d / 'cases_rank0.jsonl').read_text().splitlines()
        (d / 'cases_rank0.jsonl').write_text(lines[0]+'\n'+lines[0]+'\n')
    elif damage == 'truncated':(d / 'cases_rank1.jsonl').write_text('')
    elif damage == 'score_changed':
        p=d / 'cases_rank1.jsonl';p.write_text(p.read_text().replace('true','false'))
    elif damage in ('metric_count', 'wrong_mode'):
        p=d / 'metrics.json';m=json.loads(p.read_text())
        m['total_cases' if damage == 'metric_count' else 'history_rgb_mode'] = 4 if damage == 'metric_count' else '2rgb_endpoints'
        proof.write(p,m)
    else:(d / 'evaluation_complete.json').unlink()
    disputes=tmp_path / 'disputes.jsonl';disputes.write_text('')
    with pytest.raises(ValueError):ev.dual_score(root, disputes, tmp_path / 'dual.json')
    assert not (tmp_path / 'dual.json').exists()


@pytest.mark.parametrize('source,frame', [('metas',19), ('bboxes',19), ('bboxes',10)])
def test_history_gap_cannot_turn_long_queue_into_recent_stop(world, tmp_path, source, frame):
    assert ev.ue1_evidence(ev.LeadStore(world), SC, 'queue', 20)[0] == 'long_both_stopped'
    (world / SC / 'queue' / source / f'{frame:04d}.pkl').unlink()
    with pytest.raises(ev.MissingSource):ev.ue1_evidence(ev.LeadStore(world), SC, 'queue', 20)
    with pytest.raises(ValueError, match='missing or unreadable'):
        ev.derive(_index(tmp_path, _full()), world, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()


def test_ue6_history_gap_blocks_but_negative_frame_numbers_are_not_required(world):
    ev.ue1_evidence(ev.LeadStore(world), SC, 'steady', 4)
    ev.ue6_evidence(ev.LeadStore(world), SC, 'junction', 8)
    (world / SC / 'junction' / 'bboxes' / '0011.pkl').unlink()
    with pytest.raises(ev.MissingSource):ev.ue6_evidence(ev.LeadStore(world), SC, 'junction', 12)


def test_optional_cached_missing_frame_cannot_satisfy_required_read(world):
    store=ev.LeadStore(world)
    assert store.boxes(SC, 'missing', 12, required=False) is None
    with pytest.raises(ev.MissingSource):store.boxes(SC, 'missing', 12)


def test_parent_pool_must_carry_decision_identity(tmp_path):
    decisions = tmp_path / 'd.jsonl'; decisions.write_text('{}\n')
    parent = tmp_path / 'pool'; parent.mkdir(); index = parent / 'frame_index.jsonl'; index.write_text('')
    (parent / 'manifest.json').write_text(json.dumps({'highway_ue3_rgb_decisions': {'path': 'old'}}))
    with pytest.raises(ValueError, match='not built with the requested'):
        launch.verify_parent_decisions(index, decisions)
    good = {'highway_ue3_rgb_decisions': {'sha256': ev.sha(decisions), 'missing_positive_frames': 0}}
    (parent / 'manifest.json').write_text(json.dumps(good))
    assert launch.verify_parent_decisions(index, decisions) == ev.sha(decisions)
    good['highway_ue3_rgb_decisions']['missing_positive_frames'] = 3
    (parent / 'manifest.json').write_text(json.dumps(good))
    with pytest.raises(ValueError, match='missing RGB-reviewed'):
        launch.verify_parent_decisions(index, decisions)
