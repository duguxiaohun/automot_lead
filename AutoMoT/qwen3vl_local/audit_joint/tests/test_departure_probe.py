from copy import deepcopy
import hashlib

import pytest

from qwen3vl_local.audit_joint import departure_probe as probe
from qwen3vl_local.sft_new_loop_phase4.tests.test_ten_event_teacher import frames, seed


def frozen_seed():
    return dict(seed('U-E5'), identity_world=[[float(x), 0., 0.] for x in range(0, 31, 2)], identity_half_width=1.75)


def test_other_oncoming_actor_remains_a_blocker_when_decisive_actor_is_removed():
    fs = frames('U-E5', True)
    for f in fs:
        other = deepcopy(f['actors'][1]);other['id'] = 3
        f['actors'].append(other)
        f['image_evidence']['3'] = deepcopy(f['image_evidence']['2'])
    result = probe.geometry(fs, frozen_seed())
    assert result['valid_causal_geometry'] is True
    assert result['native_clearance'] is False
    assert result['clearance_without_decisive_actor'] is False
    assert result['labels_changed'] is False
    assert 'answer' not in result


def test_removing_the_only_blocker_is_an_ablation_not_a_label():
    result = probe.geometry(frames('U-E5', True), frozen_seed())
    assert result['native_clearance'] is False
    assert result['clearance_without_decisive_actor'] is True
    assert 'priority and lifecycle are not replayed' in result['scope']


def test_missing_history_or_disappearance_does_not_get_clearance_summary():
    fs = frames('U-E5', True)
    assert 'native_clearance' not in probe.geometry(fs[-2:], frozen_seed())
    fs[-1]['actors'] = fs[-1]['actors'][:1]
    result = probe.geometry(fs, frozen_seed())
    assert result['valid_causal_geometry'] is False
    assert 'native_clearance' not in result


def test_wrong_event_or_heldout_examples_rejected(tmp_path):
    with pytest.raises(ValueError, match='frozen corridor'):
        probe.geometry(frames('U-E5'), seed('U-E5'))
    with pytest.raises(ValueError, match='training examples'):
        probe.inspect_example(dict(split='val'), tmp_path)


def test_original_navigation_and_frozen_instance_are_reported_separately():
    fs = frames('U-E5', True)
    for f in fs:
        f['meta']['route'] = [[float(x), 8.] for x in range(0, 31, 2)]
    result = probe.geometry(fs, frozen_seed())
    last = result['observations'][-1]
    assert last['fixed_corridor_intersection'] is True
    assert last['historical_navigation_intersection'] is False


def test_raw_source_mutation_rejected_before_loading_metadata(tmp_path, monkeypatch):
    from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
    sources = []
    for kind, suffix in [('metas', 'pkl'), ('bboxes', 'pkl'), ('rgb', 'jpg')]:
        name = f's/r/{kind}/0010.{suffix}'
        p = tmp_path/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(b'original')
        sources.append(dict(kind=kind, frame_id=10, path=name, sha256=hashlib.sha256(b'original').hexdigest()))
    (tmp_path/sources[0]['path']).write_bytes(b'changed')
    monkeypatch.setattr(g, 'load_frame', lambda *a: pytest.fail('must reject before unpickling'))
    example = dict(split='train', scenario='s', route_id='r', frame_id=10,
                   frame_window_sources=[dict(frame_id=10, sources=sources)])
    with pytest.raises(ValueError, match='raw source changed'):
        probe.inspect_example(example, tmp_path)


def test_metadata_passage_is_not_a_visible_release():
    fs=frames('U-E5',True)
    for f in fs:f['actors'][1]['position']=[-12.,-3.,0.]
    result=probe.passage_evidence(fs,dict(frozen_seed(),restriction_frame=4))
    assert result['status']=='geometrically_passed_not_visual_release'
    assert result['observations'][-1]['wholly_behind'] is True
    assert not any(r['visible_separation'] for r in result['observations'])
    assert 'answer' not in result and 'permission' not in result


def test_passing_uses_rotated_whole_footprint_not_center_or_length():
    import math
    fs=frames('U-E5',True)
    for f in fs:
        f['actors'][0]['position']=[0.,0.,0.];f['actors'][0]['extent']=[2.,1.,1.]
        f['actors'][1].update(position=[-4.,0.,0.],extent=[1.,3.,1.],yaw=math.pi/2)
    last=probe.passage_evidence(fs,dict(frozen_seed(),restriction_frame=4))['observations'][-1]
    assert last['rear_gap_m']==pytest.approx(-1.) and last['wholly_behind'] is False


@pytest.mark.parametrize('restriction',[None,-1,10,11])
def test_pre_restriction_separation_cannot_be_reported_as_departure(restriction):
    result=probe.passage_evidence(frames('U-E5'),dict(frozen_seed(),restriction_frame=restriction))
    assert not any(r['visible_separation'] for r in result['observations'])


def test_missing_identity_and_frame_gaps_do_not_claim_passage():
    fs=frames('U-E5');fs[-1]['actors']=fs[-1]['actors'][:1]
    assert probe.passage_evidence(fs,frozen_seed())['status']=='identity_discontinuity'
    with pytest.raises(ValueError,match='continuous'):
        probe.passage_evidence(fs[::2],frozen_seed())
