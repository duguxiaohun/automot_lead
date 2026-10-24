"""Independent offline label sensitivity lab. No native training integration.

Candidates alter one longitudinal definition at a time. Original context domain,
lateral evidence and STOP/crossing/speed precedence are retained. A candidate
that becomes mixed/undefined is missing, never coerced to KEEP.
"""
import math
from qwen3vl_local.sft_new_loop_phase3 import trajectory_action as native
from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation

VARIANTS=('native','stop_confirm_175','drop_confirm_two','pullaway_without_controls')
SCHEMA='phase3_base_offline_label_sensitivity_v1'

def decision(speeds, *, brake=None, throttle=None, variant='native'):
    if variant not in VARIANTS:raise ValueError('unknown label hypothesis')
    original=native.longitudinal_decision(speeds,brake=brake,throttle=throttle)
    if variant=='native':return original
    # Incomplete or malformed windows cannot be rehabilitated by a hypothesis.
    if original['reason'] in ('incomplete_speed_window','invalid_speed_sample'):return original
    v=[float(x) for x in speeds[:9]];s=v[0];threshold=max(1.2,.2*max(s,1.))
    drop_n=2 if variant=='drop_confirm_two' else 1
    drop=next((i for i in range(1,10-drop_n) if all(s-x>=threshold for x in v[i:i+drop_n])),None)
    gain=next((i for i in range(1,8) if min(v[i:i+2])-s>=threshold),None)
    stop_end=7 if variant=='stop_confirm_175' else 6
    pairs=[i for i in range(stop_end) if max(v[i:i+2])<=.5]
    controls=native.recorded_controls(dict(brake=brake,throttle=throttle))
    released=controls['brake'] is False and controls['throttle'] is not None and controls['throttle']>.1
    current_pair=bool(pairs and pairs[0]==0)
    pullaway=(current_pair and (released or variant=='pullaway_without_controls') and v[2]>s and v[-1]>=2. and gain is not None
              and all(v[k+1]>=v[k] for k in range(gain+1)) and all(x-s>=threshold for x in v[gain+1:]))
    trace=dict(eligible=True,action=None,reason=None,variant=variant,first_drop_s=None if drop is None else drop*.25,
               gain_start_s=None if gain is None else gain*.25,stop_confirmation_limit_s=stop_end*.25,
               drop_confirmation_samples=drop_n,anchor_controls=controls,confirmed_pullaway=bool(pullaway))
    def done(action,reason,eligible=True):return dict(trace,action=action,reason=reason,eligible=eligible)
    if pullaway:return done('RESUME','current_confirmed_pullaway')
    if current_pair:return done('STOP','current_confirmed_wait')
    if gain is not None and (drop is None or gain<drop):
        peak=s
        for i in range(gain,9):
            peak=max(peak,v[i])
            if peak-v[i]>=threshold and v[i]-s<threshold:return done(None,'mixed_longitudinal_phase',False)
        if pairs and gain<pairs[0]:return done(None,'mixed_longitudinal_phase',False)
    if pairs:return done('STOP','near_stop_confirmed_in_immediate_window')
    if drop is not None and (gain is None or drop<gain):return done('DECELERATE','first_qualifying_drop')
    if gain is not None and (drop is None or gain<drop):return done('RESUME','first_confirmed_gain')
    return done('NONE','no_qualifying_speed_change')

def project(evidence,context_id,variant):
    d=decision(evidence['future_speeds_exact_mps'],brake=evidence.get('brake'),throttle=evidence.get('throttle'),variant=variant)
    if not d['eligible']:return dict(action=None,reason=d['reason'],eligible=False)
    labels={a:d['action']==a for a in ('STOP','DECELERATE','RESUME')}
    labels.update(LANE_CHANGE_LEFT=evidence.get('lane_change_direction')=='LEFT',LANE_CHANGE_RIGHT=evidence.get('lane_change_direction')=='RIGHT')
    e=dict(evidence,longitudinal_decision=d)
    try:a=choice_annotation(labels,context_id,e)['primary_action']
    except ValueError as ex:return dict(action=None,eligible=False,reason=str(ex))
    return dict(action=a,eligible=True,reason=d['reason'])
