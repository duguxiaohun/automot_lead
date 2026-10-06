"""Quarantine long, wholly stationary captures, including duration exceptions.

This is a dataset quality policy, not a claim about traffic legality or the
cause of a standstill. Short waits and routes that eventually move are retained.
"""
import lzma
import pickle
import math
from pathlib import Path
from .identity import file_sha
from .privileged_geometry import finite,world_point

POLICY='whole_route_stationarity_v1'
MIN_FRAMES=500
MAX_SPEED=.1
MAX_DISPLACEMENT=.25


def stationarity_check(route,rgb_frames,meta_frames):
    if len(rgb_frames)<MIN_FRAMES:return dict(policy=POLICY,status='not_long')
    if not set(rgb_frames)<=set(meta_frames) or rgb_frames!=list(range(rgb_frames[0],rgb_frames[-1]+1)):
        return dict(policy=POLICY,status='unknown',reason='incomplete_metadata_or_frame_gap')
    observations=[];origin=None
    for number in rgb_frames:
        path=Path(route)/'metas'/f'{number:04d}.pkl'
        try:
            with lzma.open(path,'rb') as stream:meta=pickle.load(stream)
            speed=meta.get('speed')
            point=world_point(dict(meta=meta),dict(position=[0.,0.,0.]))
        except (OSError,EOFError,ValueError,TypeError,pickle.UnpicklingError,lzma.LZMAError):
            return dict(policy=POLICY,status='unknown',reason='unusable_metadata')
        if not finite(speed) or point is None:return dict(policy=POLICY,status='unknown',reason='missing_speed_or_pose')
        origin=point if origin is None else origin
        if abs(speed)>MAX_SPEED or math.dist(origin[:2],point[:2])>MAX_DISPLACEMENT:
            return dict(policy=POLICY,status='not_stationary',witness_frame=number)
        observations.append(dict(frame_id=number,speed=float(speed),position=point,sha256=file_sha(path)))
    return dict(policy=POLICY,status='whole_route_stationary',observations=observations)


def validate_stationarity(record,frames):
    if not isinstance(record,dict) or record.get('policy')!=POLICY:raise ValueError('missing route quality check')
    status=record.get('status')
    if status not in ('not_long','not_stationary','unknown','whole_route_stationary'):raise ValueError('invalid route quality status')
    if (status=='not_long')!=(len(frames)<MIN_FRAMES):raise ValueError('route quality duration mismatch')
    if status=='not_stationary' and record.get('witness_frame') not in frames:raise ValueError('invalid movement witness')
    if status=='unknown' and not record.get('reason'):raise ValueError('missing route quality abstention reason')
    if status!='whole_route_stationary':return
    obs=record.get('observations',[])
    if [o.get('frame_id') for o in obs]!=frames or frames!=list(range(frames[0],frames[-1]+1)):raise ValueError('incomplete stationarity evidence')
    origin=obs[0]['position']
    for o in obs:
        p=o.get('position');sha=o.get('sha256')
        if (not finite(o.get('speed')) or abs(o['speed'])>MAX_SPEED or not isinstance(p,list) or len(p)!=3
                or not all(finite(v) for v in p) or math.dist(origin[:2],p[:2])>MAX_DISPLACEMENT
                or not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('unsubstantiated whole-route stationarity')
