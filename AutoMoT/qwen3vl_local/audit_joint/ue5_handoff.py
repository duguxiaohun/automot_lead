"""Causal U-E5 task-handoff proposals, separate from permission to move.

This is an offline policy experiment. It neither changes native teacher labels
nor supplies an R-E1 target. Phase2 RE is a classifier remainder, not clearance.
"""
from qwen3vl_local.sft_new_loop_phase4 import privileged_geometry as g
from qwen3vl_local.sft_new_loop_phase4 import teacher_events as t
from qwen3vl_local.sft_new_loop_phase4.teacher_rules import actor

POLICY = 'ue5_instance_handoff_diagnostic_v1'


def assess(frames, seed):
    """Assess only observations available now; never remove the target actor.

    Two contiguous whole-footprint observations behind ego with a non-closing
    rear gap propose retiring this *instance*, even if its visual proceed edge
    is unanswerable. Current route/priority must be evaluated independently.
    Identity loss, occlusion, speed recovery and scenario flag removal cannot
    establish this proposal. Geometry is privileged, not RGB certification.
    """
    if seed.get('event') != 'U-E5':
        raise ValueError('U-E5 instance required')
    result = dict(policy=POLICY, lifecycle='unresolved', context='unresolved',
                  retire_instance=False, permission_to_move=False,
                  training_authorized=False, labels_changed=False,
                  supervised_target=None, reasons=[], observations=[],
                  scope='offline task-routing proposal; not visual release, completion, R-E1 label or action permission')
    restriction = seed.get('restriction_frame')
    if (not g.history_valid(frames, 7) or frames[0]['frame_id'] < 4 or
            type(restriction) is not int or restriction < 0 or restriction >= frames[-2]['frame_id']):
        result['reasons'].append('missing_causal_post_restriction_history')
        return result
    if any(f.get('source_geometry_issues') for f in frames):
        result['reasons'].append('invalid_geometry')
        return result
    anomalies = t.anomalies(frames, seed)
    if anomalies:
        result.update(anomalies=anomalies)
        result['reasons'].append('identity_or_scene_discontinuity')
        return result
    if any(actor(f, seed['actor_id']) is None for f in frames):
        result['reasons'].append('identity_gap')
        return result
    for f in frames[-2:]:
        obj = actor(f, seed['actor_id'])
        nav = t.instance_navigation(f, seed)
        if nav is None:
            result['reasons'].append('missing_frozen_corridor')
            return result
        gap = g.aabb(t.ego(f))[0] - g.aabb(obj)[1]
        result['observations'].append(dict(frame_id=f['frame_id'], rear_gap_m=gap,
            wholly_behind=gap > 0, visible=g.visible(obj, f),
            overlaps_instance_corridor=t.route_intersection(obj, nav)))
    previous, current = result['observations']
    passed = (previous['wholly_behind'] and current['wholly_behind'] and
              current['rear_gap_m'] >= previous['rear_gap_m'])
    if not passed:
        result['lifecycle'] = 'retain_or_reobserve'
        result['reasons'].append('whole_actor_passage_not_established_or_gap_closing')
        return result
    result.update(lifecycle='passed_instance_handoff_candidate', retire_instance=True)
    # Source/control history is required even when no STOP duty is active.
    # The caller must reconstruct it from route start, not inject an empty map.
    f = frames[-1]
    if any('stop_duties' not in x for x in frames[-3:]):
        result['reasons'].append('missing_control_history')
        return result
    context_history = []
    for end in range(len(frames)-2, len(frames)+1):
        history = frames[max(0,end-7):end]
        observation = history[-1]
        nav = t.bounded(t.navigation(observation))
        clear = priority = None
        if len(history) == 7 and len(nav['points']) >= 2:
            clear = t.clearance(history, nav, following=True)
            priority = t.control_priority(observation, nav)
        # Known independent vetoes survive missing navigation; unknown facts
        # cannot cancel a red light or a still-active STOP obligation.
        if any(observation['meta'].get(k) is True for k in ('light_hazard','stop_sign_hazard')):
            priority = False
        context_history.append(dict(frame_id=observation['frame_id'],
            corridor_clear=clear, priority_satisfied=priority,
            navigation_available=len(nav['points'])>=2))
    clear = context_history[-1]['corridor_clear']
    priority = context_history[-1]['priority_satisfied']
    # Evaluate current evidence including all actors; the passed actor is not
    # manually ignored, and normal following is distinct from an empty road.
    from qwen3vl_local.sft_new_loop_phase4 import teacher_rules as rules
    current_events = sorted({s['event'] for s in rules.seeds(frames)
                             if (s['event'], s['actor_id']) != ('U-E5', seed['actor_id'])})
    result.update(current_corridor_clear=clear, priority_satisfied=priority,
                  current_event_candidates=current_events,
                  control_evidence=f['stop_duties'], context_history=context_history)
    if clear is False or priority is False:
        result['context'] = 'handoff_with_remaining_restriction'
    elif any(x['corridor_clear'] is not True or x['priority_satisfied'] is not True
             for x in context_history):
        result['context'] = 'handoff_requires_reobservation'
    elif current_events:
        result['context'] = 'handoff_to_current_event_candidates'
    else:
        result['context'] = 'background_reclassification_candidate'
    return result
