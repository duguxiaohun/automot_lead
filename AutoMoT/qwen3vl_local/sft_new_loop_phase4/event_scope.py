"""Actor-local release suggestions for RGB review, not complete driving permits.

Separate release from stable following. A moving-away lead need not already
satisfy the recover_follow stability test. Other actors in the local conflict
area still veto; distant participants on a later route segment do not.
"""
import math
from .privileged_geometry import (aabb,physical_actor,visible,history_valid,disappearances,
                                  vector,finite,vulnerable,DEFAULTS)

POLICY='actor_local_review_suggestions_v2'


def local_navigation(frame, horizon):
    from .automatic_context import navigation
    nav=navigation(frame)
    if not nav.get('points'):return nav
    points=[];distance=0.
    for p in nav['points']:
        if points:
            step=math.dist(p,points[-1])
            if distance+step>horizon:
                remaining=horizon-distance
                if remaining>1e-6:points.append([a+(b-a)*remaining/step for a,b in zip(points[-1],p)])
                break
            distance+=step
        points.append(p)
    return dict(nav,points=points,review_horizon_m=horizon)


def lead_releasing(frames,actor_id,params=DEFAULTS):
    """A gap opening in observed history is an opportunity, not stable following."""
    if not history_valid(frames,params['history_frames']):return None
    history=frames[-params['history_frames']:];gaps=[];leads=[]
    for frame in history:
        lead=next((a for a in frame['actors'] if a['id']==actor_id),None)
        if (lead is None or lead['class']!='car' or vulnerable(lead)
                or abs(lead['yaw'])>.35 or visible(lead,frame,params) is not True):return None
        if any(frame['meta'].get(k) is None or lead.get(k)!=frame['meta'][k] for k in ('road_id','lane_id')):return None
        ego=next(a for a in frame['actors'] if a['class']=='ego_car')
        gaps.append(aabb(lead)[0]-ego['extent'][0]);leads.append(lead)
    velocity=leads[-1].get('ego_velocity')
    if not vector(velocity,2):return None
    # Conservative proposal thresholds for review; no certified headway policy.
    if (gaps[-1]>=3 and velocity[0]>.5 and abs(velocity[1])<.5
            and gaps[-1]-gaps[0]>=.15 and all(b>=a-.05 for a,b in zip(gaps,gaps[1:]))):return True
    if abs(velocity[0])<.2 and gaps[-1]<10:return False
    return None


def release_suggestion(frames,event,actor_id,params=DEFAULTS):
    from .automatic_context import route_intersection
    result=dict(value=None,actor_id=actor_id,blocking_actor_ids=[],unresolved_actor_ids=[],
                outside_scope_actor_ids=[],scope='local_event_release_only_not_priority_or_execution',policy=POLICY)
    if event not in ('U-E1','U-E3','U-E4','U-E5'):return dict(result,reason='unsupported_event')
    if not history_valid(frames,params['history_frames']):return dict(result,reason='insufficient_history')
    if any(f.get('source_geometry_issues') for f in frames):return dict(result,reason='invalid_geometry')
    if any(r['rgb_relevant'] for a,b in zip(frames,frames[1:]) for r in disappearances(a,b,params)):
        return dict(result,reason='rgb_discontinuity')
    current=frames[-1];actor=next((a for a in current['actors'] if a['id']==actor_id),None)
    if actor is None:return dict(result,reason='missing_actor_not_release')
    if visible(actor,current,params) is not True:return dict(result,reason='actor_not_visually_resolved')
    horizon=min(20.,max(6.,actor['position'][0]+3.))
    nav=local_navigation(current,horizon)
    if len(nav.get('points',[]))<2:return dict(result,reason='missing_navigation')
    result['navigation']=nav
    # The candidate is tied to this actor; clearance cannot be inferred from its
    # absence, from an action timestamp, or from other actors leaving the scene.
    if event in ('U-E1','U-E3'):
        release=lead_releasing(frames,actor_id,params)
        from .privileged_geometry import following
        stable,lead_id,_=following(frames,params)
        if stable is True and lead_id==actor_id:release=True
    else:
        history=[]
        for frame in frames:
            item=next((a for a in frame['actors'] if a['id']==actor_id),None)
            if item is None:return dict(result,reason='actor_identity_gap')
            history.append(item)
        from .visual_review import VRU_MARGIN_M
        occupied=route_intersection(actor,nav,margin=0. if event=='U-E4' else .3)
        if occupied is True:release=False
        elif event=='U-E4' and route_intersection(actor,nav,margin=VRU_MARGIN_M) is True:
            # Outside the lane is not yet a well-resolved positive margin.
            # Do not manufacture a NO for this uncertain boundary region.
            release=None
        else:
            # Lateral separation must grow in the current coordinate history;
            # avoid interpreting a large ego turn as participant departure.
            from .privileged_geometry import world_point
            poses=[world_point(f,next(a for a in f['actors'] if a['class']=='ego_car')) for f in frames]
            matrices=[f['meta'].get('ego_matrix') for f in frames]
            stationary=all(p is not None for p in poses) and math.dist(poses[0],poses[-1])<.5
            same_heading=all(m is not None for m in matrices) and all(abs(matrices[-1][i][j]-matrices[0][i][j])<.03 for i in (0,1) for j in (0,1))
            moving_out=stationary and same_heading and abs(actor['position'][1])-abs(history[0]['position'][1])>.15
            passed=aabb(actor)[1]<-next(a['extent'][0] for a in current['actors'] if a['class']=='ego_car')
            release=True if moving_out or passed else None
    for other in current['actors']:
        if not physical_actor(other) or other['id']==actor_id:continue
        if route_intersection(other,nav) is not True:
            result['outside_scope_actor_ids'].append(other['id']);continue
        if visible(other,current,params) is True:result['blocking_actor_ids'].append(other['id'])
        else:result['unresolved_actor_ids'].append(other['id'])
    if result['blocking_actor_ids']:release=False
    elif result['unresolved_actor_ids'] and release is True:release=None
    return dict(result,value=release,reason='review_local_release_and_separate_priority')
