"""Frozen candidate boundaries from the train-only 2026-10-22 audit."""
import math
from . import VERSION
from ..geometry_v1 import codec

VARIANTS = ('six', 'a', 'b', 'c')
ACTIONS = ('DECELERATE', 'STOP', 'RESUME', 'LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT', 'KEEP')
FIXED_NAMES = tuple(f'{d}_{p}' for d in ('LEFT', 'CENTER', 'RIGHT') for p in ('NEAR', 'MIDDLE', 'FAR'))


def names(variant):
    if variant not in VARIANTS: raise ValueError('unknown_representation')
    return ACTIONS if variant == 'six' else codec.NAMES if variant == 'a' else FIXED_NAMES


def bounds(variant, road):
    if variant not in ('b', 'c'): raise ValueError('fixed_boundaries_require_b_or_c')
    # Do not silently interpret missing/unknown road annotations as non-highway.
    if road not in ('R1', 'R2', 'R3', 'R4', 'R5'): raise ValueError('unknown_current_road_category')
    return (20., 24.) if variant == 'b' and road == 'R3' else (3., 13.)


def contract(variant):
    return dict(version=VERSION, variant=variant, names=list(names(variant)),
        geometry_a=codec.contract(), fixed_other_m=[3., 13.], fixed_highway_m=[20., 24.],
        road_group='current R3 highway, R1/R2/R4/R5 other; frozen for past and future',
        fixed_boundaries='all durations share absolute radii; equality MIDDLE; FAR unbounded',
        boundary_status='frozen train-audit candidates; final admitted-pool supports reported, no test fitting',
        six_direction='native confirmed lane-change side vector at local origin projected onto current lane right; abs(cos)<0.1 invalid',
        six_longitudinal='unchanged native per-origin speed and evidence, including STOP/KEEP priority',
        six_decoding='native event action_keys plus KEEP; unrestricted diagnostics retained',
        six_history='offline native intention with later evidence permitted; native short-gap fill then direction transform',
        geometry_history='causal past-position-minus-current, not past maneuver',
        missing='no future fill; common four-query validity mask across representations',
        information_comparison='same current RGB/speed/navigation/event/road; history meanings and offline leakage differ',
        sampling='matched event quota, successful-update exposure, physical frame cap 1/epoch and 3/run',
        validation_selection='maximum macro recall on supported pure classes; earliest epoch breaks ties')


def relabel(item, variant, road):
    """Consume verified native geometric measurements, never discretized centers."""
    if variant == 'a': return dict(item)
    lower, upper = bounds(variant, road)
    if not item['valid']: return dict(item)
    radius, lateral = item['radius_m'], item['lateral_m']
    if not all(math.isfinite(x) for x in (radius, lateral)) or radius < 0:
        raise ValueError('invalid_native_geometry')
    direction = 0 if lateral < -2 else 2 if lateral > 2 else 1
    radial = 0 if radius < lower else 2 if radius > upper else 1
    result = dict(item, token_id=3*direction+radial, name=FIXED_NAMES[3*direction+radial],
        inner_radius_m=lower, outer_radius_m=upper,
        radial_margin_m=min(abs(radius-lower), abs(radius-upper)))
    # These speed-relative diagnostics would misdescribe the fixed-distance rule.
    result.pop('q_mps', None); result.pop('delta_mps', None)
    return result
