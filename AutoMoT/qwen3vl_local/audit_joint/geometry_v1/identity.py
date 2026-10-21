"""A prediction belongs to one causal sample, never to a future-label payload."""
from .dataset import digest


def observation_binding(record):
    fields = ('id', 'scenario', 'route_id', 'frame', 'image', 'event', 'road_structure', 'observation')
    try:
        payload = {key: record[key] for key in fields}
    except KeyError as exc:
        raise ValueError('missing_prediction_sample_identity') from exc
    if type(payload['frame']) is not int or any(not isinstance(payload[k], str) or not payload[k]
            for k in ('id', 'scenario', 'route_id', 'image', 'event', 'road_structure')):
        raise ValueError('invalid_prediction_sample_identity')
    return dict(schema='sample_identity_and_causal_observation_sha256_v1',
                sample_id=record['id'], observation_sha256=digest(payload))
