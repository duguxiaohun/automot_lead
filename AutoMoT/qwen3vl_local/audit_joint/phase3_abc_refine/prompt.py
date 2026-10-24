"""Only causal quantities enter the candidate; no change to labels or radii."""
import copy
from ..geometry_abc_v2 import objective
from ..geometry_v1 import codec

ORIGINAL_MESSAGES = objective.messages


def messages(record, image):
    result = copy.deepcopy(ORIGINAL_MESSAGES(record, image))
    if record['variant'] != 'a': return result
    v = codec.speed(record['observation']['speed_mps'])
    delta = codec.tolerance(v)
    rows = [f'For endpoint +{t!r} s: inner={max(0., (v-delta)*t)!r} m, outer={(v+delta)*t!r} m.'
            for t in codec.HORIZONS]
    note = ('Classify each endpoint independently with its own duration and the CURRENT speed. '
            'The radii below are computed from the current observation, not future measurements. '
            'FAR denotes distance relative to that radius, not an acceleration command. '
            'Even if endpoint distance stays unchanged, its region can change when the duration changes. '
            'A current speed of zero does not determine future motion. History describes past endpoints, '
            'not a future plan. Equality belongs to REFERENCE.\n' + '\n'.join(rows))
    # Decoder-only attention cannot let a note after the queries affect them.
    result[1]['content'][-1]['text'] = note + '\n' + result[1]['content'][-1]['text']
    return result


def verify_no_target_leak(record):
    first = messages(record, None)
    changed = copy.deepcopy(record)
    changed['targets'] = [{'valid': True, 'token_id': 8-i} for i in range(4)]
    if messages(changed, None) != first: raise ValueError('future_target_leaks_into_prompt')
    return first
