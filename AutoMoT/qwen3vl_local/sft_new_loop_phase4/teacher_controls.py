"""Causal stop-area duty evidence. Trigger boxes are not painted stop lines.

Development observation rules only: stationary motion alone never discharges a
stop obligation. The trigger's near face is an explicit geometric proxy. Real
RGB learnability and the proxy error require independent measurement.
"""
from . import privileged_geometry as g

PARAMS=dict(stop_speed_mps=.1,stop_observations=3,max_front_gap_m=3.,crossing_horizon_s=2.,crossing_radius_m=15.)


class StopDutyTracker:
    def __init__(self):self.entries={};self.last_frame=None
    def observe(self,frame):
        n=frame['frame_id'];m=frame['meta']
        if self.last_frame is not None and n!=self.last_frame+1:self.entries.clear()
        self.last_frame=n
        if n<4 or frame.get('source_geometry_issues'):self.entries.clear();return {}
        ego=next(a for a in frame['actors'] if a['class']=='ego_car')
        active={a['id']:a for a in frame['actors'] if a['class']=='stop_sign' and a.get('affects_ego') is True}
        self.entries={i:e for i,e in self.entries.items() if i in active}
        for ident,a in active.items():
            lane=(m.get('road_id'),m.get('lane_id'));world=g.world_point(frame,a)
            old=self.entries.get(ident)
            if (old is None or old['lane']!=lane or world is None or old['world'] is None
                    or sum((x-y)**2 for x,y in zip(world,old['world']))>1.):
                old=dict(lane=lane,world=world,observations=[],satisfied=False);self.entries[ident]=old
            box=g.aabb(a);gap=box[0]-ego['extent'][0]
            identified=(None not in lane and tuple(ego.get(k) for k in ('road_id','lane_id'))==lane and world is not None)
            before_area=0<=gap<=PARAMS['max_front_gap_m'] and box[2]<=0<=box[3]
            stopped=identified and before_area and abs(m['speed'])<=PARAMS['stop_speed_mps']
            if not old['satisfied']:
                if stopped:old['observations'].append(dict(frame_id=n,sources=frame['sources']))
                else:old['observations']=[]
                old['observations']=old['observations'][-PARAMS['stop_observations']:]
                old['satisfied']=len(old['observations'])==PARAMS['stop_observations']
        return {str(i):dict(stop_id=i,satisfied=e['satisfied'],lane=list(e['lane']),
                position_basis='trigger_area_near_face_not_painted_line',observations=list(e['observations'])) for i,e in self.entries.items()}


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


def crossing_clear(frame,stop):
    """Visible local crossing traffic, using current velocities, never saved futures."""
    x,y,_=stop['position'];width=frame['meta'].get('ego_lane_width')
    if not g.finite(width) or width<=0:return None
    region=[x-4.,x+4.,-width/2-1.,width/2+1.];unknown=False
    for a in frame['actors']:
        if not g.physical_actor(a):continue
        if (a['position'][0]-x)**2+(a['position'][1]-y)**2>PARAMS['crossing_radius_m']**2:continue
        if g.visible(a,frame) is not True:unknown=True;continue
        box=g.aabb(a)
        if g.intersects(box,region):return False
        if g.obstacle(a):continue
        velocity=a.get('ego_velocity')
        if not g.vector(velocity,2):unknown=True;continue
        if crossing_interval(box,region,velocity,PARAMS['crossing_horizon_s']) is not None:return False
    return None if unknown else True
