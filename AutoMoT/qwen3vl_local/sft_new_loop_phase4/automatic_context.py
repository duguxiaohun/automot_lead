"""Current planner geometry and causal event hypotheses, never reviewed truth.

LEAD save_meta exports the remaining planned route in current ego coordinates.
This is not future executed ego motion. Prefer route_original; if absent, accept
route only when changed_route is explicitly False. A bypass-shifted expert route
is not evidence of permission to bypass. Missing map/priority/visibility facts
remain None. No scenario names, controls or future arrays participate.
"""
import math
from .identity import digest
from .privileged_geometry import finite,vector,physical_actor,vulnerable,aabb,visible,camera_possible,world_point

POLICY='current_navigation_hypotheses_v1'


def navigation(frame):
    m=frame['meta'];points=m.get('route_original');source='route_original'
    if points is None:
        if m.get('changed_route') is not False:return dict(points=[],reason='unmodified_navigation_unavailable')
        points=m.get('route');source='route'
    width=m.get('ego_lane_width')
    if (not isinstance(points,list) or len(points)<2 or not finite(width) or width<=0
            or any(not vector(p,2) for p in points)):
        return dict(points=[],reason='navigation_geometry_unavailable')
    # Keep the near current corridor, not a box enclosing a whole turn.
    result=[];distance=0.
    for p in points:
        if result:
            step=math.dist(p,result[-1])
            if step>10:return dict(points=[],reason='navigation_discontinuous')
            if step<1e-5:continue
            distance+=step
        if distance>55:break
        result.append(p)
    if len(result)<2:return dict(points=[],reason='navigation_too_short')
    return dict(points=result,half_width=width/2,source=source,reason='current_planned_corridor_not_execution',
                sha256=digest([source,result,width]))


def route_intersection(actor,nav, *, margin=0.):
    """Actor oriented footprint versus each planned segment's lateral strip.

    Do not extend segment ends into an unbounded straight ego-axis corridor.
    """
    if not nav.get('points'):return None
    x,y=actor['position'][:2];ex,ey=actor['extent'][:2];yaw=actor['yaw']
    for a,b in zip(nav['points'],nav['points'][1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<1e-5:continue
        ux,uy=dx/length,dy/length;rx,ry=x-a[0],y-a[1]
        along=rx*ux+ry*uy;lateral=abs(-rx*uy+ry*ux)
        delta=yaw-math.atan2(dy,dx)
        longitudinal_extent=abs(math.cos(delta))*ex+abs(math.sin(delta))*ey
        lateral_extent=abs(math.sin(delta))*ex+abs(math.cos(delta))*ey
        if (-longitudinal_extent<=along<=length+longitudinal_extent
                and lateral<=nav['half_width']+lateral_extent+margin):return True
    return False


def infer(frames,instances):
    """Auto-produce inspectable partial context; absence is not clearance."""
    current=frames[-1];m=current['meta'];nav=navigation(current)
    lights=[a for a in current['actors'] if a['class']=='traffic_light' and a.get('affects_ego') is True
            and a.get('dummy_traffic_light_bounding_box') is not True]
    stops=[a for a in current['actors'] if a['class']=='stop_sign' and a.get('affects_ego') is True]
    # Traffic-light boxes may be stop-waypoint proxies, NOT image locations of
    # lamp heads. Keep privileged signal diagnostics separate from RGB truth.
    red=any(a.get('state') in ('Red','Yellow') for a in lights)
    green=bool(lights) and all(a.get('state')=='Green' for a in lights)
    obligation=True if red or stops else False if green else None
    actors={a['id']:a for a in current['actors'] if physical_actor(a)}
    progress=[]
    for instance in instances:
        ident=instance.get('actor_id');actor=actors.get(ident)
        tracked=actor is not None and all(any(a['id']==ident for a in f['actors']) for f in frames)
        # Current behind-ego location suggests passage only with continuous
        # actor history; no absence-based completion or stable inference.
        passed=(aabb(actor)[1]<-next(a['extent'][0] for a in current['actors'] if a['class']=='ego_car')) if tracked else None
        progress.append(dict(instance_id=instance['instance_id'],event=instance['event'],actor_id=ident,
                             actor_passed=passed,event_resolved=None,
                             reason='passage_is_not_complete_event_boundary_or_stability'))
    return dict(policy=POLICY,frame_id=current['frame_id'],sources_sha256=digest([f['sources'] for f in frames]),
                navigation=nav,priority=dict(stop_obligation_candidate=obligation,priority_satisfied=None,
                traffic_light_ids=[a['id'] for a in lights],stop_sign_ids=[a['id'] for a in stops],
                reason='applicable_controls_do_not_establish_all_crossing_priority_or_RGB_visibility'),
                boundaries=progress,visible_space_resolved=None,event_identity_established=None,
                signal_failure_established=None,training_approved=False)


def junction_candidates(frame):
    m=frame['meta'];distance=m.get('distance_to_next_junction')
    near=m.get('is_junction') is True or (finite(distance) and 0<=distance<=35)
    if not near:return []
    junction=m.get('junction_id');token=f'junction:{junction if junction is not None else "unknown"}'
    lights=[a for a in frame['actors'] if a['class']=='traffic_light' and a.get('affects_ego') is True]
    stops=[a for a in frame['actors'] if a['class']=='stop_sign' and a.get('affects_ego') is True]
    from .event_scope import local_navigation
    nav=local_navigation(frame,18.)
    crossing=[a for a in frame['actors'] if physical_actor(a) and 0<a['position'][0]<22
              and .5<abs(a['yaw'])<2.5 and route_intersection(a,nav) is True
              and visible(a,frame) is True]
    out=[]
    if crossing:out.append(('U-E6','visible_crossing_conflict_at_near_junction_requires_priority_review'))
    if any(a.get('state') in ('Off','Unknown') for a in lights):
        out.append(('U-E7','unresolved_or_off_signal_requires_failure_review'))
    if stops:out.append(('R-E5','applicable_stop_obligation_requires_unsignalized_identity_review'))
    return [dict(event=event,actor_id=token,reason=reason) for event,reason in out]



def condition_facts(frames,episode,instance,params):
    """Partial automatic facts consumed by the proposal path, never approval.

    Positively visible blockers can establish a veto. Unknown map priority,
    unseen space and unconfirmed event identity cannot establish a positive
    permission. Signal stop-waypoint proxies are not lamp RGB evidence.
    """
    from .privileged_geometry import following,disappearances
    current=frames[-1];nav=navigation(current);facts={}
    if not nav.get('points') or any(f.get('source_geometry_issues') for f in frames):return facts
    if any(r.get('rgb_relevant',True) for a,b in zip(frames,frames[1:]) for r in disappearances(a,b,params)):return facts
    if episode.event in ('U-E1','U-E3','U-E4','U-E5'):
        from .event_scope import release_suggestion
        suggestion=release_suggestion(frames,episode.event,instance.get('actor_id'),params)
        # Local release is a review fact, never complete permission. In
        # particular priority_satisfied remains unknown without separate proof.
        return dict(corridor_clear=suggestion['value'],release_ready=suggestion['value'])
    established,lead_id,_=following(frames,params)
    blocked=False;unresolved=False;must_wait=False
    for actor in current['actors']:
        if not physical_actor(actor) or route_intersection(actor,nav) is not True:continue
        if visible(actor,current,params) is not True:unresolved=True;continue
        if episode.event in ('U-E1','U-E3') and established is True and actor['id']==lead_id:continue
        blocked=True
        gap=aabb(actor)[0]-next(a['extent'][0] for a in current['actors'] if a['class']=='ego_car')
        speed=max(0,current['meta']['speed']);limit=max(2.,speed+speed*speed/6.)
        if vulnerable(actor):limit=max(limit,15.)
        if gap<=limit:must_wait=True
    facts['corridor_clear']=None  # replace the legacy straight-strip veto with route-aware evidence
    if blocked:facts.update(corridor_clear=False,release_ready=False,restriction_present=True)
    if must_wait:facts['stationary_wait_required']=True
    if episode.event in ('U-E1','U-E3'):
        facts['restricted_progress_established']=established
    return facts
