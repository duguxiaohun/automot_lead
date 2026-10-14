"""Causal stop-area duty evidence. Trigger boxes are not painted stop lines.

Development observation rules only: stationary motion alone never discharges a
stop obligation. A matched incoming-road boundary supplies a fixed position;
the exported trigger is used for identity, never as a painted stopping line.
"""
from . import privileged_geometry as g

PARAMS=dict(stop_speed_mps=.1,stop_observations=3,max_front_gap_m=3.,crossing_horizon_s=2.,crossing_radius_m=15.)


def owned_lane(frame):
    """Meta owns the lane; optional bbox copies must not contradict it."""
    ego=next(a for a in frame['actors'] if a['class']=='ego_car')
    lane=tuple(frame['meta'].get(k) for k in ('road_id','lane_id'))
    if any(type(v) is not int for v in lane):return None
    if any(ego.get(k) is not None and (type(ego[k]) is not int or ego[k]!=v)
           for k,v in zip(('road_id','lane_id'),lane)):return None
    return lane


def same_stop(frame,actor,world):
    current=g.world_point(frame,actor)
    return (current is not None and world is not None
            and sum((x-y)**2 for x,y in zip(current,world))<=1.)


class StopDutyTracker:
    def __init__(self):
        from .stop_position import catalog
        self.entries={};self.last_frame=None
        try:self.catalog=catalog();self.catalog_error=None
        except (OSError,ValueError) as exc:self.catalog=None;self.catalog_error=str(exc)
    def invalidate(self):
        """Unusable source revokes completion evidence, never the obligation."""
        for old in self.entries.values():
            old.update(observations=[],satisfied=False,identity_verified=False,retired=False)
    def observe(self,frame):
        n=frame['frame_id'];m=frame['meta']
        continuous=self.last_frame is None or n==self.last_frame+1
        self.last_frame=n
        valid=n>=4 and not frame.get('source_geometry_issues')
        ego=next(a for a in frame['actors'] if a['class']=='ego_car')
        stops={a['id']:a for a in frame['actors'] if a['class']=='stop_sign'}
        lane=owned_lane(frame)
        # Cleared flags, missing actors and source gaps cannot discharge debt.
        for ident,a in stops.items():
            if n>=4 and a.get('affects_ego') is True and ident not in self.entries:
                self.entries[ident]=dict(lane=lane,world=g.world_point(frame,a),
                    observations=[],satisfied=False,established_frame=n,position_binding=None,
                    position_error=None,front_gap_m=None)
        for ident,old in self.entries.items():
            previously_retired=old.get('retired',False)
            a=stops.get(ident)
            identity=valid and a is not None and same_stop(frame,a,old['world'])
            if not continuous or not valid:
                old['observations']=[];old['satisfied']=False
            old['identity_verified']=bool(identity)
            old['retired']=False
            if not identity:
                old.update(observations=[],satisfied=False,front_gap_m=None,
                           position_error='same physical STOP or source geometry unverified')
                continue
            box=g.aabb(a)
            try:
                if self.catalog is None:raise ValueError(self.catalog_error or 'STOP catalog unavailable')
                if old['position_binding'] is None:
                    if a.get('affects_ego') is not True or lane!=old['lane']:
                        raise ValueError('cannot bind STOP after association or approach changed')
                    binding=self.catalog.match(frame,a)
                    # Binding may become reliable later; its establishment is
                    # explicit and never backfills earlier stationary frames.
                    old['position_binding']=binding;old['established_frame']=n;old['world']=binding['world']
                gap=self.catalog.position(frame,a,old['position_binding'])
                old['front_gap_m']=gap;old['position_error']=None
            except (OSError,ValueError,KeyError,TypeError) as exc:
                old.update(observations=[],satisfied=False,identity_verified=False,
                           position_error=str(exc),front_gap_m=None)
                continue
            if previously_retired and box[1]>=-ego['extent'][0]:
                old['observations']=[];old['satisfied']=False
            # Only a fulfilled duty can retire after the complete trigger has
            # passed behind the ego footprint. Crossing cannot forgive debt.
            if old['satisfied'] and box[1]<-ego['extent'][0]:
                old['retired']=True
                continue
            identified=lane is not None and lane==old['lane']
            before_area=0<=gap<=PARAMS['max_front_gap_m'] and box[2]<=0<=box[3]
            stopped=(identified and before_area and g.finite(m.get('speed'))
                     and abs(m['speed'])<=PARAMS['stop_speed_mps'])
            if not old['satisfied']:
                if stopped:old['observations'].append(dict(frame_id=n,sources=frame['sources'],
                    front_gap_m=gap,speed_mps=m['speed'],catalog_sha256=self.catalog.sha))
                else:old['observations']=[]
                old['observations']=old['observations'][-PARAMS['stop_observations']:]
                old['satisfied']=len(old['observations'])==PARAMS['stop_observations']
        result={str(i):dict(stop_id=i,satisfied=e['satisfied'],
                lane=list(e['lane']) if e['lane'] is not None else None,
                world=e['world'],established_frame=e['established_frame'],
                identity_verified=e['identity_verified'],retired=e['retired'],
                position_basis='incoming_lane_end_plane_not_painted_stop_line',
                position_binding=e['position_binding'],position_error=e['position_error'],front_gap_m=e['front_gap_m'],
                observations=list(e['observations'])) for i,e in self.entries.items()}
        # Do not immediately recreate a still-active, passed control as debt.
        # Clear it only after geometric retirement AND association withdrawal.
        self.entries={i:e for i,e in self.entries.items()
                      if not (e['retired'] and stops[i].get('affects_ego') is False)}
        return result


def crossing_interval(box,region,velocity,horizon_s):
    """Times a translating fixed AABB overlaps the region on both axes.

    This preserves the existing footprint and constant-velocity hypothesis;
    a union of start/end boxes would mix positions at different times.
    """
    if not g.finite(horizon_s) or horizon_s<=0 or not g.vector(velocity,2):
        raise ValueError('crossing prediction requires finite velocity and positive horizon')
    low,high=0.,float(horizon_s)
    for lo,hi,v,rlo,rhi in ((box[0],box[1],velocity[0],region[0],region[1]),
                            (box[2],box[3],velocity[1],region[2],region[3])):
        if abs(v)<1e-12:
            if hi<rlo or lo>rhi:return None
        else:
            start,end=sorted(((rlo-hi)/v,(rhi-lo)/v))
            low=max(low,start);high=min(high,end)
            if low>high:return None
    return low,high


def crossing_clear(frame,stop,nav=None):
    """Check the question's whole local corridor, without a stop-centre radius.

    Current occupancy and continuous motion determine relevance *before*
    visibility. Unknown motion cannot establish irrelevance. This is local
    route clearance, never permission for an entire junction maneuver.
    """
    from .automatic_context import route_intersection
    from .teacher_events import linear_intersection, PARAMS as events
    if nav is None:return None
    points=nav.get('points');width=nav.get('half_width')
    if (not isinstance(points,list) or len(points)<2 or
            any(not g.vector(p,2) for p in points) or
            not g.finite(width) or width<=0 or
            not any(a!=b for a,b in zip(points,points[1:]))):return None
    speed=frame['meta'].get('speed')
    if not g.finite(speed) or frame.get('source_geometry_issues'):return None
    horizon=(events['junction_start_prediction_s'] if abs(speed)<events['junction_start_speed_mps']
             else events['prediction_s'])
    unknown=False
    for a in frame['actors']:
        if not g.physical_actor(a):continue
        current=route_intersection(a,nav,margin=.25)
        if current is None:unknown=True;continue
        if current:
            if g.visible(a,frame) is True:return False
            unknown=True;continue
        if g.obstacle(a):continue
        velocity=a.get('ego_velocity')
        if not g.vector(velocity,2):unknown=True;continue
        hit=linear_intersection(a,nav,velocity,horizon)
        if hit is None:unknown=True;continue
        if hit:
            if g.visible(a,frame) is True:return False
            unknown=True
    return None if unknown else True
