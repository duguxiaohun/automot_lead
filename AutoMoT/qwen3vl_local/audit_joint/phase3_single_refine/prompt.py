"""Single-specific clarification. No dependency on current base prompt modules."""
from pathlib import Path

VARIANTS=('original','clarified')
NOTE=(
    'The only observed image is at t=0. Predict the recorded ego behavior, not a recommended maneuver. '
    'Each future origin T requires forecasting the state at T; current speed, navigation and image are not future measurements. '
    'For each origin, a STOP requires two consecutive 0.25-second samples at or below 0.5 m/s, '
    'both within [T,T+1.5s], subject to the immediate confirmed pull-away exception. '
    'An isolated low-speed sample or a stop confirmed after T+1.5s is insufficient. '
    'Speed changes are compared with speed at T over [T,T+2s]: the threshold is max(1.2,0.2*max(v(T),1)) m/s. '
    'A qualifying decrease can occur at one sample including T+2s; an increase needs two consecutive samples within the window. '
    'For a near-zero current pair, the pull-away exception requires released brake and throttle above 0.1 at T, '
    'speed rising by T+0.5s, nondecreasing speed through gain confirmation, retained qualifying gain afterwards, '
    'and speed at least 2 m/s at T+2s. These are label definitions to forecast, not supplied control measurements. '
    'Otherwise a confirmed current wait is STOP even if release follows later. '
    'The lane window ends at T+3s: a future intention label can refer to a crossing later than t=+2s. '
    'Two valid history origins are compressed to the LATER origin label, including ties; they do not assert the same action at both times. '
    'If either history origin remains unavailable after the stated gap handling, the slot is UNKNOWN. '
    'The last history slot includes the label at t=0; it does not specify any future query answer. '
    'Use visible traffic and obstacle positions to forecast persistence or transition. '
    'An approaching conflict does not force STOP if its defined timing is not met, and an open-looking road does not prove release. '
)

def instruction(variant):
    if variant not in VARIANTS:raise ValueError('unknown_single_prompt_variant')
    original=Path(__file__).with_name('frozen').joinpath('instruction.txt').read_text()
    if variant=='original':return original
    old='Reset all action rules at T: speed comparisons use ego speed at T, lane direction uses ego heading at T. '
    new='Reset speed comparisons at T using ego speed at T. Identify a native lane crossing locally at T, then report its side in the CURRENT lane reference as defined below. '
    if original.count(old)!=1:raise ValueError('frozen_prompt_patch_anchor_mismatch')
    old_lane="relative to ego's heading."
    if original.count(old_lane)!=1:raise ValueError('frozen_lane_prompt_patch_anchor_mismatch')
    return original.replace(old,new).replace(old_lane,'identified in the local lane reference and reported in the CURRENT lane reference.')+'\n'+NOTE
