"""Source-bound U-E5 departure diagnostics; never a new supervised answer.

Inspect the recorded instance corridor in every observation. Clearance-only
ablations do not establish priority, readiness, catchup, or completed execution.
"""
from pathlib import Path


def passage_evidence(frames, seed):
    """Separate privileged passing geometry from observed departure evidence.

    This never grants permission or revives a censored episode. In particular,
    an actor entirely behind ego in metadata is not a visible separation.
    """
    from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g, teacher_events as t
    from qwen3vl_local.sft_new_loop_phase4.teacher_rules import actor
    if seed.get('event')!='U-E5':raise ValueError('U-E5 instance required')
    if not frames or not g.history_valid(frames,len(frames)):
        raise ValueError('continuous same-route observations required')
    restriction=seed.get('restriction_frame');observations=[]
    for f in frames:
        obj=actor(f,seed['actor_id']);nav=t.instance_navigation(f,seed)
        r=dict(frame_id=f['frame_id'],present=obj is not None,
               after_restriction=type(restriction) is int and restriction>=0 and f['frame_id']>restriction,
               visible_separation=False)
        if obj is not None and nav is not None and not f.get('source_geometry_issues'):
            separated=t.route_intersection(obj,nav,margin=t.PARAMS['separation_m']) is False
            visible=g.visible(obj,f)
            # Yaw-aware footprint, not center position or unrotated length.
            gap=g.aabb(t.ego(f))[0]-g.aabb(obj)[1]
            partial=f.get('image_evidence',{}).get(str(obj['id']),{}).get('partial_quality_pass')
            r.update(visible=visible,partial_quality=partial,separated=separated,
                rear_gap_m=gap,wholly_behind=gap>0,
                visible_separation=r['after_restriction'] and visible is True and separated)
        observations.append(r)
    anomalies=t.anomalies(frames,seed);current=observations[-1]
    if anomalies or not current['present']:status='identity_discontinuity'
    elif current.get('wholly_behind'):status='geometrically_passed_not_visual_release'
    elif any(x['visible_separation'] for x in observations):status='observed_separation_requires_current_checks'
    elif current.get('partial_quality') is True:status='partial_view_without_separation_evidence'
    elif current.get('visible') is not True:status='decisive_actor_unobserved'
    else:status='no_observed_post_restriction_separation'
    return dict(status=status,observations=observations,anomalies=anomalies,
        scope='diagnostic_only_not_a_supervised_answer_or_permission')


def geometry(frames, seed):
    from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g, teacher_events as t
    from qwen3vl_local.sft_new_loop_phase4.teacher_rules import actor
    if seed.get('event') != 'U-E5' or not seed.get('identity_world'):
        raise ValueError('U-E5 frozen corridor evidence required')
    observations = []
    for frame in frames:
        participant = actor(frame, seed['actor_id'])
        corridor = t.bounded(dict(points=t.local_path(frame, seed['identity_world']),
                                  half_width=seed['identity_half_width']))
        observation = dict(frame_id=frame['frame_id'], participant_present=participant is not None,
                           scene_quality=frame.get('scene_visibility'), corridor_points=corridor['points'])
        if participant is not None:
            observation.update(position=participant['position'], extent=participant['extent'],
                yaw=participant['yaw'], visible=g.visible(participant, frame),
                camera_possible=g.camera_possible(participant, frame),
                pixel_quality=frame.get('image_evidence', {}).get(str(seed['actor_id'])),
                fixed_corridor_intersection=t.route_intersection(participant, corridor, margin=t.PARAMS['separation_m']),
                historical_navigation_intersection=t.route_intersection(participant,
                    t.local_navigation(frame, t.PARAMS['horizon_m']), margin=t.PARAMS['separation_m']))
        observations.append(observation)
    current = frames[-1]
    nav = t.bounded(dict(points=t.local_path(current, seed['identity_world']), half_width=seed['identity_half_width']))
    enough = g.history_valid(frames, 7) and frames[0]['frame_id'] >= 4
    anomalies = t.anomalies(frames, seed) if g.history_valid(frames, len(frames)) else []
    valid = enough and not any(f.get('source_geometry_issues') for f in frames) and not anomalies
    result = dict(observations=observations, history_valid=enough, anomalies=anomalies,
                  valid_causal_geometry=valid, labels_changed=False,
                  passage=passage_evidence(frames,seed),
                  intersection_margin_m=t.PARAMS['separation_m'],
                  scope='diagnostic geometry only; priority and lifecycle are not replayed; no inferred answer')
    if valid:
        result.update(native_clearance=t.clearance(frames, nav, following=True),
            clearance_without_decisive_actor=t.clearance(frames, nav, following=True, ignore=(seed['actor_id'],)),
            native_visible_departure=t.visible_departure(frames, seed, nav))
    return result


def inspect_example(example, data_root):
    from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
    from qwen3vl_local.sft_new_loop_phase4.production_storage import binding, check_binding
    from qwen3vl_local.sft_new_loop_phase4.data_paths import data_path
    if example['split'] != 'train':
        raise ValueError('departure RGB diagnostics accept training examples only')
    root = Path(data_root).absolute()
    resolved = root.resolve(strict=True)
    frames, bindings = [], []
    for observation in example['frame_window_sources']:
        number = observation['frame_id']
        sources = observation['sources']
        expected = {s['path']: s for s in sources}
        if len(expected) != 3 or {s['kind'] for s in sources} != {'metas', 'bboxes', 'rgb'}:
            raise ValueError('incomplete diagnostic frame source binding')
        for source in sources:
            # Validate containment, but keep the logical path for drift checks.
            data_path(root, source['path'])
            record = binding(root / source['path'])
            if record['sha256'] != source['sha256']:
                raise ValueError('diagnostic raw source changed')
            bindings.append(record)
        frame = g.load_frame(root, example['scenario'], example['route_id'], number)
        if {s['path']: s for s in frame['sources']} != expected:
            raise ValueError('diagnostic source identity differs from recorded frame')
        frames.append(frame)
    if not frames or frames[-1]['frame_id'] != example['frame_id']:
        raise ValueError('missing event-end frame evidence')
    report = geometry(frames, example['instance_context']['instance'])
    for record in bindings:
        check_binding(record)
    if root.resolve(strict=True) != resolved:
        raise ValueError('diagnostic data root retargeted')
    report['source_bindings'] = bindings
    return report
