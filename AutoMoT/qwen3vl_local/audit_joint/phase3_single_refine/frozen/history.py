"""Past labels from unmodified native Phase3 rules on the full offline route.

The user explicitly allows post-observation evidence in historical input labels.
This is an offline-label experiment, not an online-causal history recognizer.
"""
from . import trajectory_action as rule
from qwen3vl_local.audit_joint.phase3_single_refine.frozen.context_taxonomy import CONTEXT_BY_ID, DOMAIN_MANEUVER
from .classification import classify

POLICY = 'offline_native_history_v2'
FILL_POLICY = 'native_history_short_gap_v1'
FILL_MODES = ('none', 'same_short', 'nearest_short')
DEFAULT_FILL = 'nearest_short'
MAX_GAP_FRAMES = 2  # Two missing 0.25-second origins, not two compressed tokens.
SEMANTICS = {
    'offline_native': 'Past native upcoming-action intention labels computed from the full recorded trajectory, including later evidence. These labels are not an executed-action plan that must continue.',
}


def recognize(trajectory, frame, observed_until, context_id, policy='offline_native'):
    if policy not in SEMANTICS:
        raise ValueError('Only user-selected offline_native history is supported')
    maneuver = CONTEXT_BY_ID[context_id].question_domain == DOMAIN_MANEUVER
    horizon = rule.LATERAL_HORIZON_FRAMES+1 if maneuver else rule.LONGITUDINAL_HORIZON_FRAMES
    end = frame+horizon
    result = classify(trajectory, frame, context_id)
    return dict(frame=frame, native_origin_frame=frame, required_end_frame=end,
                observed_until=observed_until, policy=policy,
                post_observation_evidence_allowed=True,
                evidence_window_extends_past_observation=end > observed_until,
                action=result['action'], reason=result['reason'], evidence=result.get('evidence'))


def fill_short_gaps(points, lookup, frame_exists, mode=DEFAULT_FILL):
    """Derive history only. lookup must return quarantined, unfilled native points.

    Both endpoints must exist in the same route/event context. Never propagate a
    previously filled value, extrapolate at route edges, or replace known actions.
    nearest_short supports A ? B, with midpoint ties assigned to the later B.
    """
    if mode not in FILL_MODES:
        raise ValueError('Unknown history fill mode: '+str(mode))
    result = []
    for point in points:
        item = dict(point, original_action=point['action'], original_reason=point['reason'],
                    imputed=False, imputation=None)
        f = point['frame']
        if mode == 'none' or point['action'] is not None or not frame_exists(f) or point['reason']=='source_quarantine':
            result.append(item)
            continue
        left = next((lookup(f-k) for k in range(1,MAX_GAP_FRAMES+1)
                     if lookup(f-k)['action'] is not None), None)
        right = next((lookup(f+k) for k in range(1,MAX_GAP_FRAMES+1)
                      if lookup(f+k)['action'] is not None), None)
        if left is not None and right is not None:
            lo, hi = left['frame'], right['frame']
            gap = hi-lo-1
            same = left['action']==right['action']
            continuous = all(frame_exists(k) and lookup(k)['reason']!='source_quarantine'
                             for k in range(lo,hi+1))
            if gap<=MAX_GAP_FRAMES and continuous and (same or mode=='nearest_short'):
                selected = left if f-lo < hi-f else right
                item.update(action=selected['action'], imputed=True, imputation=dict(
                    policy=FILL_POLICY, mode=mode, kind='same_endpoints' if same else 'nearest_transition',
                    left_frame=lo, right_frame=hi, left_action=left['action'], right_action=right['action'],
                    left_required_end_frame=left['required_end_frame'], right_required_end_frame=right['required_end_frame'],
                    missing_frames=gap, missing_duration_seconds=gap*rule.FRAME_DT_SECONDS,
                    endpoint_separation_seconds=(hi-lo)*rule.FRAME_DT_SECONDS,
                    selected_endpoint_frame=selected['frame'], midpoint_tie='later',
                    native_label=False))
        result.append(item)
    return result
