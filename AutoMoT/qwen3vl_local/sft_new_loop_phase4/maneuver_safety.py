"""External maneuver clearance is independent of RGB answers and execution receipts.

The planner/BEV adapter must verify the entire maneuver corridor, including
unseen rear/side approaches. This module validates its receipt, not the sensor
coverage or the truth of an upstream assertion. No production adapter is bundled.
"""
from .route_context import route_identity

GUARDED = frozenset(('depart', 'enter', 'return'))
POLICY = 'current_full_corridor_clearance_v1'


def binding(episode, edge, frame):
    returning = edge == 'return'
    return dict(policy=POLICY, instance_id=episode.instance_id, frame_id=frame,
                edge=edge, segment_id=episode.segment_id, route_context_id=route_identity(episode),
                target_corridor=episode.return_corridor if returning else episode.target_corridor,
                direction=episode.return_direction if returning else episode.direction)


def validate_clearances(episode, frame, records, questions):
    if not isinstance(records, (list, tuple)):
        raise ValueError('maneuver clearances must be a sequence')
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('invalid maneuver clearance')
        key = record.get('edge')
        if key not in GUARDED or key not in questions or key in result:
            raise ValueError('unknown, inapplicable or duplicate maneuver clearance')
        expected = binding(episode, key, frame)
        if (set(record) != set(expected) | {'source', 'evidence_id', 'clear', 'rear_side_coverage_confirmed'}
                or any(record.get(k) != v for k, v in expected.items())
                or type(record.get('frame_id')) is not int
                or record.get('source') not in ('planner', 'bev_safety')
                or not isinstance(record.get('evidence_id'), str) or not record['evidence_id'].strip()
                or type(record.get('clear')) is not bool
                or record.get('rear_side_coverage_confirmed') is not True
                or not expected['target_corridor'] or not expected['direction']):
            raise ValueError('maneuver clearance must bind current instance, edge, target and rear/side coverage')
        result[key] = record
    return result
