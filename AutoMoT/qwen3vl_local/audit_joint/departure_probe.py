"""Source-bound U-E5 departure diagnostics; never a new supervised answer.

Inspect the recorded instance corridor in every observation. Clearance-only
ablations do not establish priority, readiness, catchup, or completed execution.
"""
from pathlib import Path


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
