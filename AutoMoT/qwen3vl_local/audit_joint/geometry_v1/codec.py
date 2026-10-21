"""One definition for offline labels, causal history, prompts and expert input.

CARLA horizontal coordinates: x forward, y right. The reference is the nearest
Driving lane at the origin, oriented toward ego forward, never a future fit.
"""
import math
import numpy as np

from . import VERSION

HORIZONS = (.5, 1., 1.5, 2.)
LOOKBACKS = (2., 1.5, 1., .5)
NAMES = tuple(f'{d}_{p}' for d in ('LEFT', 'CENTER', 'RIGHT')
              for p in ('NEAR', 'REFERENCE', 'FAR'))
LETTERS = tuple('ABCDEFGHI')
UNKNOWN = -1  # Missing observation, never a tenth supervision class.
REFERENCE_SOURCE = 'nearest_driving_lane_oriented_to_ego_v1'


def contract():
    return dict(version=VERSION, names=list(NAMES), horizons=list(HORIZONS),
        lookbacks=list(LOOKBACKS), half_width_m=2., delta_floor_mps=.5,
        delta_fraction=.2, delta_cap_mps=1., negative_speed_limit_mps=-.1,
        reference_source=REFERENCE_SOURCE, radius='horizontal_endpoint_chord',
        boundary='equality belongs to CENTER/REFERENCE; no radius rounding',
        history='past endpoints minus current origin; frozen current speed/lane; positive lookback duration',
        prediction_binding='sample_identity_and_causal_observation_sha256_v1',
        future='exact endpoints relative to frozen current origin/speed/lane',
        supervision='single hard class per valid endpoint; no event action mask',
        missing='UNKNOWN history; masked future; no imputation',
        frame_dt_s=.25, native_dt_s=.05, max_native_step_m=5.)


def speed(value):
    value = float(value)
    if not math.isfinite(value) or value < -.1:
        raise ValueError('invalid_speed')
    return max(value, 0.)


def tolerance(value):
    return min(1., max(.5, .2 * speed(value)))


def reference(lane_yaw, ego_yaw):
    """Return oriented world yaw and relative yaw using only the origin state."""
    if not all(math.isfinite(float(v)) for v in (lane_yaw, ego_yaw)):
        raise ValueError('invalid_reference')
    relative = (float(lane_yaw)-float(ego_yaw)+math.pi) % (2*math.pi)-math.pi
    flipped = abs(relative) > math.pi/2
    if flipped:
        relative = relative-math.copysign(math.pi, relative)
    return dict(yaw=float(ego_yaw)+relative, relative_yaw=relative,
                flipped=flipped, source=REFERENCE_SOURCE)


def classify(displacement_world_xy, duration, start_speed, reference_yaw):
    xy = np.asarray(displacement_world_xy, dtype=np.float64)
    duration = float(duration)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or
            not math.isfinite(duration) or duration <= 0 or
            not math.isfinite(float(reference_yaw))):
        raise ValueError('invalid_geometry')
    v = speed(start_speed); delta = tolerance(v)
    lateral = -math.sin(reference_yaw)*xy[0]+math.cos(reference_yaw)*xy[1]
    radius = float(np.linalg.norm(xy)); q = radius/duration-v
    direction = 0 if lateral < -2 else 2 if lateral > 2 else 1
    lower = max(0., (v-delta)*duration); upper = (v+delta)*duration
    # Algebraically q +/- delta, evaluated in distance space to preserve exact
    # equality at computed radii (e.g. 4.8 m), without rounding measured values.
    progress = 0 if radius < lower else 2 if radius > upper else 1
    # r=0 is not a NEAR boundary when the unclipped inner radius is negative.
    radial_margin = min(abs(radius-(v-delta)*duration), abs(radius-upper)) if v > delta else abs(radius-upper)
    token = 3*direction+progress
    return dict(valid=True, token_id=token, name=NAMES[token], duration_s=duration,
        start_speed_mps=v, delta_mps=delta, q_mps=q, radius_m=radius,
        lateral_m=float(lateral), inner_radius_m=lower, outer_radius_m=upper,
        lateral_margin_m=float(min(abs(lateral-2), abs(lateral+2))),
        radial_margin_m=float(radial_margin))


def unavailable(reason, duration):
    return dict(valid=False, token_id=UNKNOWN, name='UNKNOWN', reason=reason,
                duration_s=float(duration))
