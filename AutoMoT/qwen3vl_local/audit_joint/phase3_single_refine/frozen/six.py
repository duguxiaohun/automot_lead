"""Native action evidence first; change only the direction reference of D/E."""
import math
from . import history, targets
from .classification import classify
from ...geometry_v1 import sources
from ...geometry_v1.dataset import digest
from . import label_quarantine
from .vocabulary import ACTIONS


def transform(point, trajectory, anchor_yaw, town, lanes, reference_cache=None):
    result = dict(point, original_action=point.get('original_action', point['action']),
                  native_action=point['action'], direction_reference='current_lane')
    if point['action'] not in ACTIONS[3:5]: return result
    # Filled labels inherit the actual native endpoint's spatial reference.
    origin = (point['imputation']['selected_endpoint_frame'] if point.get('imputed') else point['frame'])
    try:
        reference_cache = {} if reference_cache is None else reference_cache
        if origin not in reference_cache:
            reference_cache[origin] = sources.observed(trajectory.metas[origin], town, lanes)
        state = reference_cache[origin]
        projection = math.cos(state['reference']['yaw']-anchor_yaw)
        result['direction_projection'] = projection
        result['direction_origin_frame'] = origin
        if abs(projection) < .1: raise ValueError('ambiguous_current_direction_projection')
        if projection < 0:
            result['action'] = ACTIONS[4] if point['action'] == ACTIONS[3] else ACTIONS[3]
    except (KeyError, TypeError, ValueError) as exc:
        result.update(action=None, reason=str(exc))
    return result


def build(record, trajectory, quarantines, lanes, *, native_cache=None, reference_cache=None):
    anchor, event = record['frame'], record['event']
    scenario, route = record['scenario'], record['route_id']
    yaw = record['observation']['reference']['yaw']; town = route.split('_')[0]
    cache = {}
    native_cache = {} if native_cache is None else native_cache
    def recognize(frame):
        key=(frame,event)
        if key not in native_cache:
            native_cache[key]=history.recognize(trajectory,frame,anchor,event)
        value=native_cache[key]
        return dict(value,observed_until=anchor,
                    evidence_window_extends_past_observation=value['required_end_frame']>anchor)
    def native_at(frame):
        if frame in cache: return cache[frame]
        value = recognize(frame)
        if (label_quarantine.affected(quarantines, scenario, route, frame, value['required_end_frame']) or
            label_quarantine.phase3_risks(dict(scenario=scenario, route_id=route, frame_id=frame,
                context_id=event, action_evidence=value.get('evidence', {})), quarantines)):
            value = dict(value, action=None, reason='source_quarantine')
        cache[frame] = value
        return value
    observed = [native_at(f) for f in range(anchor-7, anchor+1)]
    filled = history.fill_short_gaps(observed, native_at, lambda f: f in trajectory.metas)
    converted = [transform(p, trajectory, yaw, town, lanes, reference_cache) for p in filled]
    historical = []
    for i in range(0, 8, 2):
        action = targets.compress_history(converted[i:i+2], anchor)
        historical.append(dict(valid=action != 'UNKNOWN', token_id=ACTIONS.index(action) if action in ACTIONS else -1,
                               name=action, duration_s=2-i*.25))
    future = []
    for f in range(anchor+1, anchor+9):
        value = dict(recognize(f), frame=f,
                     id=digest([scenario, route, record['split'], f, event]))
        if any(k not in trajectory.metas for k in range(anchor, f+1)):
            value.update(action=None, reason='anchor_to_target_meta_gap')
        if (label_quarantine.affected(quarantines, scenario, route, anchor, f) or
            label_quarantine.phase3_risks(dict(scenario=scenario, route_id=route, frame_id=f,
                context_id=event, action_evidence=value.get('evidence', {})), quarantines)):
            value.update(action=None, reason='source_quarantine')
        future.append(transform(value, trajectory, yaw, town, lanes, reference_cache))
    windows = targets.window_targets(future, lateral_weight=1.5)
    for i, w in enumerate(windows):
        w.update(duration_s=(i+1)*.5, name='MIXED' if len(w['allowed_answers']) > 1 else
                 w['allowed_answers'][0] if w['valid'] else 'UNKNOWN')
        w['token_id'] = ACTIONS.index(w['allowed_answers'][0]) if len(w['allowed_answers']) == 1 else -1
    return dict(history=historical, targets=windows, anchor_action=native_at(anchor)['action'],
                audit=dict(history_points=converted, future_points=future,
                           history_fill=history.DEFAULT_FILL, post_observation_evidence_allowed=True))
