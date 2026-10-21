"""Native 20 Hz endpoints and observed 4 Hz history; no action-label dependency."""
import hashlib
import lzma
import math
import pickle
from pathlib import Path
import numpy as np

from . import codec


def file_sha(path, *, check_budget=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            check_budget()
            h.update(block)
    return h.hexdigest()


def read_meta(path):
    """Only trusted local LEAD pickle inputs; never downloads or unpickles web data."""
    raw = Path(path).read_bytes()
    return pickle.loads(lzma.decompress(raw) if raw.startswith(b'\xfd7zXZ') else raw)


def pose(meta):
    matrix = np.asarray(meta['ego_matrix'], dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError('invalid_pose')
    if meta.get('dataset_information', {}).get('data_save_freq') != 5:
        raise ValueError('unexpected_frame_frequency')
    return matrix, math.atan2(matrix[1, 0], matrix[0, 0])


class LaneMap:
    """Same map lookup for build and online callers. No simulation server connection."""
    def __init__(self, maps):
        self.root = Path(maps).resolve(strict=True)
        self.maps = {}; self.hashes = {}

    def at(self, town, position, ego_yaw):
        import carla
        if not town or Path(town).name != town:
            raise ValueError('invalid_town')
        if town not in self.maps:
            path = self.root / (town + '.xodr')
            raw = path.read_bytes()
            self.hashes[town] = hashlib.sha256(raw).hexdigest()
            self.maps[town] = carla.Map(town, raw.decode())
        wp = self.maps[town].get_waypoint(carla.Location(*map(float, position)),
            project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            raise ValueError('missing_driving_lane')
        result = codec.reference(math.radians(wp.transform.rotation.yaw), ego_yaw)
        result.update(road_id=wp.road_id, lane_id=wp.lane_id, lane_width_m=wp.lane_width,
            is_junction=wp.is_junction, map_sha256=self.hashes[town], town=town)
        return result


def observed(meta, town, lanes):
    """Reads no future field, including when called for an old history origin."""
    matrix, yaw = pose(meta)
    return dict(position=matrix[:3, 3].tolist(), ego_yaw=yaw,
                speed_mps=codec.speed(meta['speed']),
                reference=lanes.at(town, matrix[:3, 3], yaw))


def history(metas, anchor, town, lanes, discontinuities=()):
    """Past endpoints minus CURRENT origin, all using current speed/lane direction."""
    state = observed(metas[anchor], town, lanes)
    result = []
    for duration in codec.LOOKBACKS:
        first = anchor-round(duration/.25)
        try:
            if any(f not in metas for f in range(first, anchor+1)):
                raise ValueError('history_meta_gap')
            if any(first < f <= anchor for f in discontinuities):
                raise ValueError('source_discontinuity')
            positions = np.array([pose(metas[f])[0][:2, 3] for f in range(first, anchor+1)])
            if np.any(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 25.):
                raise ValueError('history_position_jump')
            item = codec.classify(positions[0]-positions[-1], duration,
                                  state['speed_mps'], state['reference']['yaw'])
            item.update(start_frame=first, end_frame=anchor,
                        reference_relative_yaw=state['reference']['relative_yaw'],
                        origin_frame=anchor, endpoint_frame=first,
                        reference_speed_mps=state['speed_mps'],
                        reference_frame='current', displacement='past_minus_current')
        except (KeyError, TypeError, ValueError) as e:
            item = codec.unavailable(str(e), duration)
        result.append(item)
    return result


def future(meta, state, *, anchor, discontinuities=()):
    """Classify exact ticks 10/20/30/40, freezing the current reference for all."""
    result = []
    try:
        matrix, _ = pose(meta)
        local = np.asarray(meta['future_positions'], dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3 or not len(local):
            raise ValueError('invalid_future_shape')
        if not np.isfinite(local[0]).all() or np.linalg.norm(local[0]) > .1:
            raise ValueError('future_origin_mismatch')
        world = local @ matrix[:3, :3].T
    except (KeyError, TypeError, ValueError) as e:
        return [codec.unavailable(str(e), t) for t in codec.HORIZONS]
    for t in codec.HORIZONS:
        tick = round(t/.05)
        try:
            if len(world) <= tick or not np.isfinite(world[:tick+1]).all():
                raise ValueError('incomplete_future')
            if np.any(np.linalg.norm(np.diff(world[:tick+1, :2], axis=0), axis=1) > 5.):
                raise ValueError('future_position_jump')
            if any(anchor < f <= anchor+round(t/.25) for f in discontinuities):
                raise ValueError('source_discontinuity')
            item = codec.classify(world[tick, :2], t, state['speed_mps'], state['reference']['yaw'])
            item.update(native_tick=tick, end_frame=anchor+round(t/.25))
        except ValueError as e:
            item = codec.unavailable(str(e), t)
        result.append(item)
    return result


def observation(metas, anchor, town, lanes, discontinuities=()):
    """Shared causal runtime input. Navigation is current planned route information."""
    current = metas[anchor]
    state = observed(current, town, lanes)
    # Diagnostic only: do not smooth/map-match using future poses, or silently
    # change the nearest-lane rule at junctions and opposite-lane excursions.
    reference_step_change = None
    if anchor-1 in metas:
        try:
            previous = observed(metas[anchor-1], town, lanes)
            reference_step_change = (state['reference']['yaw']-previous['reference']['yaw']+math.pi)%(2*math.pi)-math.pi
        except (KeyError,TypeError,ValueError):
            pass
    points = np.asarray(current['next_target_points'], dtype=np.float64)
    if points.ndim != 2 or len(points) < 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError('missing_navigation')
    d = points[[0, 1, -1], :2]-np.asarray(state['position'][:2])
    c, s = math.cos(state['ego_yaw']), math.sin(state['ego_yaw'])
    nav = np.stack([d[:, 0]*c+d[:, 1]*s, -d[:, 0]*s+d[:, 1]*c], axis=-1)
    return dict(speed_mps=state['speed_mps'], reference=state['reference'],
                reference_step_change_rad=reference_step_change,
                navigation_xy=nav.tolist(), history=history(metas, anchor, town, lanes, discontinuities)), state
