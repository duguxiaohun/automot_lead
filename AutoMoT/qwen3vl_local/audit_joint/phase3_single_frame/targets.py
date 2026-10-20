"""Keep native future labels; distinguish raw frequency from training preference."""
from collections import Counter
import math
from qwen3vl_local.audit_joint.phase3_base_dense.data import ACTIONS

FUTURE_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8))
HISTORY_PAIRS = ((-7, -6), (-5, -4), (-3, -2), (-1, 0))


def distribution(actions, lateral_weight=1.5):
    if not math.isfinite(lateral_weight) or not 1 <= lateral_weight <= 3:
        raise ValueError('Lateral preference must be finite and in [1,3]')
    if len(actions) != 2:
        raise ValueError('Each half-second window contains exactly two native origins')
    if any(a is None for a in actions):
        return dict(valid=False, raw=None, target=None, allowed_answers=[], entropy=None)
    if any(a not in ACTIONS for a in actions):
        raise ValueError('Unknown native action')
    counts = Counter(actions)
    raw = [counts[a] / 2 for a in ACTIONS]
    weighted = [q * (lateral_weight if a.startswith('LANE_CHANGE_') else 1.)
                for a, q in zip(ACTIONS, raw)]
    total = sum(weighted)
    target = [q / total for q in weighted]
    return dict(valid=True, raw=raw, target=target,
                allowed_answers=[a for a in ACTIONS if counts[a]],
                entropy=-sum(q * math.log(q) for q in target if q))


def compress_history(points, observation_frame):
    """Do not expose a partially known pair as a complete historical window."""
    if len(points) != 2:
        raise ValueError('History pair size')
    for point in points:
        if point.get('action') is None:
            return 'UNKNOWN'
    # Two equal-width observations: majority, ties resolve to latest.
    return points[-1]['action']


def window_targets(points, lateral_weight=1.5):
    if len(points) != 8:
        raise ValueError('Eight fine origins are required; never pad future labels')
    return [dict(distribution([p['action'] for p in points[i:i+2]], lateral_weight),
                 offsets_seconds=[(i+1)*.25, (i+2)*.25],
                 source_ids=[p['id'] for p in points[i:i+2]],
                 reasons=[p['reason'] for p in points[i:i+2]]) for i in range(0, 8, 2)]
