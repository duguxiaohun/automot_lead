"""Causal weak rules for obstacle, intrusion, junction and navigation events.

Scenario/active-scenario identities retrieve an instance, never a permission.
Current planned routes specify targets only. No future execution, controls,
recorded action labels or a missing participant can establish clearance.
"""
import math
from . import privileged_geometry as g
from .automatic_context import navigation, route_intersection
from .event_scope import local_navigation

EVENTS = ('U-E2','U-E3','U-E5','U-E6','U-E7','R-E2','R-E3','R-E5')
MERGES = {'HighwayExit','EnterActorFlow','EnterActorFlowV2','MergerIntoSlowTraffic','MergerIntoSlowTrafficV2'}
UNSIGNALIZED = {'NonSignalizedJunctionLeftTurn','NonSignalizedJunctionRightTurn',
               'NonSignalizedJunctionLeftTurnEnterFlow','InterurbanActorFlow','InterurbanAdvancedActorFlow','T_Junction'}
PARAMS = dict(horizon_m=22.,prediction_s=2.,separation_m=.75,entry_margin_m=.2,
              entry_heading_rad=.2,stable_observations=3,detour_offset_m=.6)


def ego(frame):
    return next(a for a in frame['actors'] if a['class']=='ego_car')


def world_path(frame, points):
    out=[g.world_point(frame,dict(position=[*p,0.])) for p in points]
    return out if out and all(p is not None for p in out) else []


def local_path(frame, points):
    m=frame['meta'].get('ego_matrix')
    if g.world_point(frame,ego(frame)) is None:return []
    return [[sum((p[i]-m[i][3])*m[i][j] for i in range(3)) for j in (0,1)] for p in points]


def planned(frame, *, original=False, horizon=45.):
    points=frame['meta'].get('route_original',frame['meta'].get('route')) if original else frame['meta'].get('route');width=frame['meta'].get('ego_lane_width')
    if (not isinstance(points,list) or len(points)<2 or not g.finite(width) or width<=0
            or any(not g.vector(p,2) for p in points)):return None
    out=[];distance=0.
    for p in points:
        if out:
            step=math.dist(p,out[-1])
            if step>10:return None
            if step<1e-5:continue
            distance+=step
        if distance>horizon:break
        out.append(p)
    return dict(points=out,half_width=width/2) if len(out)>1 else None


def closest(points, point=(0.,0.)):
    best=None;station=0.
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<1e-5:continue
        u=((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length**2
        if not 0<=u<=1:
            station+=length;continue
        x,y=a[0]+u*dx,a[1]+u*dy
        candidate=dict(distance=math.dist(point,(x,y)),lateral=((point[0]-x)*-dy+(point[1]-y)*dx)/length,
                       heading=math.atan2(dy,dx),station=station+u*length)
        if best is None or candidate['distance']<best['distance']:best=candidate
        station+=length
    return best


def corridor(frame,seed,returning=False):
    points=local_path(frame,seed.get('original_world' if returning else 'target_world',[]))
    return dict(points=[p for p in points if -5<=p[0]<=45],half_width=seed.get('half_width',0),review_horizon_m=PARAMS['horizon_m'])


def source_receipts(frames):
    return [s for f in frames for s in f['sources']]


def navigation_reference(frame):
    if frame['meta'].get('route_original') is None and frame['meta'].get('changed_route') is not False:return None
    nav=planned(frame,original=True,horizon=120.)
    if not nav:return None
    points=world_path(frame,nav['points'])
    if not points:return None
    return dict(points=points,half_width=nav['half_width'],sources=frame['sources'],frame_id=frame['frame_id'])


def update_reference(frame, previous):
    """Keep pre-obstacle navigation while the current planner modifies its path.

    changed_route is an execution-time flag in LEAD; the planned detour can
    appear earlier. Refreshing that reference each frame erases the detour.
    """
    ids=set(frame['meta'].get('scenario_obstacles_ids') or [])
    approaching=any(a['id'] in ids and 0<a['position'][0]<55 for a in frame['actors'])
    if previous and (approaching or frame['meta'].get('changed_route') is True):return previous
    return navigation_reference(frame) or previous


def bounded(nav, horizon=PARAMS['horizon_m']):
    points=[];distance=0.
    for p in nav.get('points',[]):
        if p[0]<-5:continue
        if points:
            step=math.dist(p,points[-1])
            if distance+step>horizon:
                if step>0 and horizon>distance:points.append([a+(b-a)*(horizon-distance)/step for a,b in zip(points[-1],p)])
                break
            distance+=step
        points.append(p)
    return dict(nav,points=points,review_horizon_m=horizon)


def target_context(frame,reference):
    target=planned(frame)
    if not target or not reference:return None
    original=local_path(frame,reference['points'])
    if len(original)<2:return None
    offsets=[]
    for p in target['points']:
        if not 5<=p[0]<=25:continue
        c=closest(original,p)
        if c:offsets.append(c['lateral'])
    if not offsets:return None
    shift=max(offsets,key=abs)
    return dict(target_world=world_path(frame,target['points']),original_world=reference['points'],
                half_width=target['half_width'],shift=shift,
                direction='LEFT' if shift<-.6 else 'RIGHT' if shift>.6 else 'FORWARD',
                instance_sources=reference['sources']+frame['sources'])


def crossed_fixed_corridor(previous,current,old,actor):
    """The actor crosses a fixed world corridor, not an ego/planner path shift."""
    nav=navigation(previous)
    if route_intersection(old,nav) is not False:return False
    points=world_path(previous,nav.get('points',[]))
    if not points:return False
    same=dict(nav,points=local_path(current,points))
    return route_intersection(actor,same) is True


def static_obstacle(frame,a):
    # Scenario obstacle IDs also occur in actor-flow scenes. Identity is not
    # evidence of a static obstacle; moving vehicles must remain dynamic.
    return g.obstacle(a) or (a['id'] in (frame['meta'].get('scenario_obstacles_ids') or [])
        and a['class']=='car' and not g.vulnerable(a) and g.finite(a.get('speed')) and abs(a['speed'])<.2)


def invading_identity(frame,a):
    m=frame['meta']
    return ('InvadingTurn' in (m.get('current_active_scenario_type'),m.get('previous_active_scenario_type'))
            and a['id'] in (m.get('scenario_actors_ids') or []))


def own_corridor(frame,reference):
    # A planner's detour is NOT the ego's original lane. A frozen pre-detour
    # reference is admissible only where it actually has geometry.
    if frame['meta'].get('route_original') is not None:return navigation(frame)
    if reference:return dict(points=local_path(frame,reference['points']),half_width=reference['half_width'])
    if frame['meta'].get('changed_route') is False:return navigation(frame)
    return dict(points=[])


def cut_in_identity(frames,a,old):
    f=frames[-1];m=f['meta']
    explicit=(a.get('is_cut_in') is True
              or a['id'] in (m.get('cut_in_actors_ids') or []))
    if explicit:return True
    # Generic retrieval remains available outside intersections. Ordinary route
    # turns/merges do not prove an anomalous cut-in.
    if any(p['meta'].get('is_junction') is not False or p['meta'].get('current_active_scenario_type') in MERGES for p in frames):return False
    return crossed_fixed_corridor(frames[0],f,old,a)


def lane_predictions(frames,a,nav):
    """Project a persistent parallel lane participant along current road shape.

    None means insufficient lane evidence: retain the conservative generic
    prediction. Empty means a supported adjacent lane stays clear locally.
    Never use saved future actor or ego positions.
    """
    if a['class']!='car' or g.vulnerable(a) or len(frames)<3:return None
    offsets=[];motions=[]
    for f in frames[-3:]:
        obj=next((x for x in f['actors'] if x['id']==a['id']),None)
        path=navigation(f);c=closest(path.get('points',[]),obj['position'][:2]) if obj else None
        if (not c or f['meta'].get('is_junction') is not False or obj.get('road_id')!=f['meta'].get('road_id')
                or obj.get('lane_id') is None or obj.get('lane_id')!=a.get('lane_id')
                or not g.vector(obj.get('ego_velocity'),2)):return None
        angle=math.atan2(math.sin(obj['yaw']-c['heading']),math.cos(obj['yaw']-c['heading']))
        vx,vy=obj['ego_velocity'];h=c['heading']
        longitudinal=vx*math.cos(h)+vy*math.sin(h);lateral=-vx*math.sin(h)+vy*math.cos(h)
        if abs(angle)>.25 or abs(lateral)>.35 or longitudinal<.5:return None
        offsets.append(c['lateral']);motions.append((longitudinal,lateral))
    if max(offsets)-min(offsets)>.5:return None
    c=closest(nav['points'],a['position'][:2])
    if not c:return None
    out=[]
    for dt in (.5,1.,1.5,2.):
        station=c['station']+motions[-1][0]*dt;walk=0.
        for p,q in zip(nav['points'],nav['points'][1:]):
            length=math.dist(p,q)
            if length and walk<=station<=walk+length:
                h=math.atan2(q[1]-p[1],q[0]-p[0]);u=(station-walk)/length
                lateral=c['lateral']+motions[-1][1]*dt
                out.append(dict(a,position=[p[0]+u*(q[0]-p[0])-math.sin(h)*lateral,
                    p[1]+u*(q[1]-p[1])+math.cos(h)*lateral,a['position'][2]],yaw=h));break
            walk+=length
    return out


def seeds(frames):
    if not g.history_valid(frames,7):return []
    f=frames[-1];m=f['meta'];nav=local_navigation(f,PARAMS['horizon_m']);out=[]
    reference=f.get('_teacher_navigation_reference')
    if reference is None:
        reference=next((navigation_reference(p) for p in reversed(frames) if navigation_reference(p)),None)
    context=target_context(f,reference)
    physical=[a for a in f['actors'] if g.physical_actor(a)]
    visible=[a for a in physical if g.visible(a,f) is True]
    obstacle_ids=set(m.get('scenario_obstacles_ids') or [])
    obstacles=[a for a in visible if static_obstacle(f,a)
               and 0<a['position'][0]<35]
    if obstacles and context:
        original=dict(points=local_path(f,context['original_world']),half_width=context['half_width'])
        blocking=[a for a in obstacles if route_intersection(a,original) is True]
        points=local_path(f,context['target_world'])
        side=[p for p in points if (c:=closest(original['points'],p)) and abs(c['lateral'])>=PARAMS['detour_offset_m']]
        direction=context['direction']
        if not blocking or direction=='FORWARD' or len(side)<2:
            # Some exports omit route_original and shift the planned path before
            # changed_route. An explicit current obstacle ID and a path passing
            # its footprint on one side still establish a bypass TARGET. They do
            # not establish permission, a return target, or completed execution.
            identified=[a for a in obstacles if a['id'] in obstacle_ids]
            if identified:
                obstacle=min(identified,key=lambda a:a['position'][0]);c=closest(points,obstacle['position'][:2])
                if c and PARAMS['detour_offset_m']<abs(c['lateral'])<m['ego_lane_width']*2:
                    direction='LEFT' if c['lateral']>0 else 'RIGHT'
                    blocking=[obstacle]
                    side=[p for p in points if p[0]>=max(5.,obstacle['position'][0]-8.)]
        if blocking and direction in ('LEFT','RIGHT') and len(side)>=2:
            group=sorted({a['id'] for a in blocking}|{a['id'] for a in physical if a['id'] in obstacle_ids and static_obstacle(f,a)})
            bypass=dict(context,direction=direction,target_world=world_path(f,side))
            out.append(dict(event='U-E2',actor_id=min(group),actor_ids=group,**bypass,
                episode_context=dict(direction=direction,target_corridor='current planner bypass corridor',
                    return_required=True,return_direction='RIGHT' if direction=='LEFT' else 'LEFT',
                    return_corridor='established original route corridor')))
    own=own_corridor(f,reference)
    for a in visible:
        if a['class']!='car' or g.vulnerable(a) or static_obstacle(f,a) or not 0<a['position'][0]<30:continue
        # Explicit current/past upstream instance identity can recover an
        # invading actor even when the original route export is unavailable.
        invading=invading_identity(f,a)
        path=own if own.get('points') and closest(own['points'],a['position'][:2]) else planned(f) if invading else own
        c=closest(path.get('points',[]),a['position'][:2]) if path else None
        angle=math.atan2(math.sin(a['yaw']-(c['heading'] if c else 0)),math.cos(a['yaw']-(c['heading'] if c else 0)))
        if c and abs(angle)>2.2 and route_intersection(a,path) is True and (
                invading or m.get('changed_route') is False and m.get('is_junction') is False
                and a.get('road_id') is not None and a.get('road_id')==m.get('road_id')
                and a.get('lane_id') is not None and a.get('lane_id')==m.get('lane_id')
                and (m.get('distance_to_next_junction')==math.inf or g.finite(m.get('distance_to_next_junction')) and m['distance_to_next_junction']>25)):
            out.append(dict(event='U-E5',actor_id=a['id'],actor_ids=[a['id']],
                identity_world=world_path(f,path['points']),identity_half_width=path['half_width'],
                identity_basis='upstream_invading_actor_and_overlap' if invading else 'opposing_overlap_original_lane'))
            continue
        old=next((v for v in frames[0]['actors'] if v['id']==a['id']),None)
        c=closest(nav.get('points',[]),a['position'][:2])
        angle=math.atan2(math.sin(a['yaw']-(c['heading'] if c else 0)),math.cos(a['yaw']-(c['heading'] if c else 0)))
        if (c and route_intersection(a,nav) is True and old and g.visible(old,frames[0]) is True
                and abs(angle)<1.2 and g.finite(a.get('speed')) and a['speed']>.5 and cut_in_identity(frames,a,old)):
            out.append(dict(event='U-E3',actor_id=a['id'],actor_ids=[a['id']]))
    distance=m.get('distance_to_next_junction')
    near=m.get('is_junction') is True or (g.finite(distance) and 0<=distance<=25)
    lights=[a for a in f['actors'] if a['class']=='traffic_light' and a.get('affects_ego') is True]
    stops=[a for a in f['actors'] if a['class']=='stop_sign' and a.get('affects_ego') is True and abs(a['position'][0])<25]
    crossing=[a for a in visible if 0<a['position'][0]<25 and .5<abs(a['yaw'])<2.5
              and (route_intersection(a,nav,margin=2.) is True)]
    active=m.get('current_active_scenario_type')
    if near and nav.get('points'):
        # Explicit upstream defect identity, not a normal red light or a file
        # name alone. Lamp color is not used as the failure classifier.
        failed=active=='CrossJunctionDefectTrafficLight'
        if failed:event='U-E7'
        elif not lights and (stops or active in UNSIGNALIZED):event='R-E5'
        elif crossing and active in ('OppositeVehicleRunningRedLight','OppositeVehicleTakingPriority','PriorityAtJunction'):event='U-E6'
        else:event=None
        if event:
            anchor=stops[0] if stops else lights[0] if lights else crossing[0] if crossing else ego(f)
            out.append(dict(event=event,actor_id=anchor['id'],actor_ids=[a['id'] for a in crossing],
                failure_established=failed,junction_id=m.get('junction_id'),junction_seen=m.get('is_junction') is True,
                junction_world=world_path(f,[[max(0.,distance) if g.finite(distance) else 8.,0.]])[0],
                instance_sources=source_receipts(frames)))
    # Planner route is a target, not evidence that entering it is safe. Lane
    # change command plus geometry is required for RE2; ramp identity plus
    # current nearby branch/junction is required for RE3.
    if context:
        commands=m.get('next_commands') or []
        pnav=planned(f)
        lateral_points=[p for p in pnav['points'] if 4<=p[0]<=35 and abs(p[1])>=context['half_width']]
        lateral=bool(len(lateral_points)>1 and (m.get('lane_type_str')=='Shoulder' or any(c in (5,6) for c in commands[:2])))
        if lateral:
            side='LEFT' if lateral_points[0][1]<0 else 'RIGHT'
            context=dict(context,direction=side,target_world=world_path(f,lateral_points))
        if active not in MERGES and lateral and not m.get('changed_route') and (m.get('lane_type_str')=='Shoulder' or any(c in (5,6) for c in commands[:2])):
            out.append(dict(event='R-E2',actor_id=ego(f)['id'],actor_ids=[],**context,
                episode_context=dict(direction=context['direction'],target_corridor='current navigation target lane')))
        if active in MERGES and (near or lateral):
            out.append(dict(event='R-E3',actor_id=ego(f)['id'],actor_ids=[],**context,
                episode_context=dict(direction=context['direction'],target_corridor='current navigation merge or exit corridor')))
    for s in out:
        s['identity_sources']=['current_local_geometry','current_navigation_or_active_instance']
        s.setdefault('instance_sources',source_receipts(frames))
        s['established_by']='causal_geometry_not_source_event_interval'
    return out


def scene_visible(frame):
    return frame.get('scene_visibility',{}).get('quality_pass') is True


def clearance(frames,nav,*,following=False,ignore=()):
    from .teacher_rules import ordinary_lead,ordinary_cyclist
    f=frames[-1];unknown=False
    if not nav.get('points'):return None
    for a in f['actors']:
        if not g.physical_actor(a) or a['id'] in ignore:continue
        if g.camera_possible(a,f) is False:continue  # visible gap only; not rear permission
        if following and (ordinary_lead(frames,a['id']) or ordinary_cyclist(frames,a['id'])):continue
        velocity=a.get('ego_velocity')
        if route_intersection(a,nav,margin=.25) is not True:
            if g.obstacle(a):continue
            if not g.vector(velocity,2):
                if math.hypot(*a['position'][:2])<30:unknown=True
                continue
            predictions=lane_predictions(frames,a,nav)
            if predictions is None:
                predictions=[dict(a,position=[a['position'][0]+velocity[0]*dt,a['position'][1]+velocity[1]*dt,a['position'][2]]) for dt in (.5,1.,1.5,2.)]
            hit=any(route_intersection(predicted,nav,margin=.25) is True for predicted in predictions)
            if not hit:continue
        if g.visible(a,f) is not True:unknown=True
        else:return False
    return None if unknown or not scene_visible(f) else True


def anomalies(frames,seed):
    tracked=set(seed.get('actor_ids',[]))|{seed['actor_id']};out=[]
    for a,b in zip(frames,frames[1:]):
        for issue in g.disappearances(a,b):
            relevant=issue['actor_id'] in tracked
            if issue.get('rgb_relevant'):
                for f in (a,b):
                    obj=next((x for x in f['actors'] if x['id']==issue['actor_id']),None)
                    nav=corridor(f,seed) if seed.get('target_world') else navigation(f)
                    if obj and 0<obj['position'][0]<PARAMS['horizon_m'] and route_intersection(obj,nav) is True:relevant=True
            if relevant:out.append(issue)
    return out


def established(frames,seed,returning=False):
    checks=[]
    for f in frames[-3:]:
        nav=corridor(f,seed,returning);c=closest(nav['points']);e=ego(f)
        if not c or not scene_visible(f):return None
        ex,ey=e['extent'][:2];width=abs(math.sin(c['heading']))*ex+abs(math.cos(c['heading']))*ey
        checks.append(c['distance']+width+PARAMS['entry_margin_m']<nav['half_width']
                      and abs(c['heading'])<PARAMS['entry_heading_rad'])
    return all(checks)


def control_priority(frame,nav):
    """Use current ego lane ownership from metas, where LEAD exports it.

    Ego bboxes often omit road_id/lane_id entirely. Requiring a second copy
    makes valid junction evidence structurally UNKNOWN. Contradictory copies
    still abstain, and all signal/STOP obligations remain in local_priority.
    """
    from .teacher_rules import local_priority
    own=ego(frame);meta=frame['meta']
    for key in ('road_id','lane_id'):
        if type(meta.get(key)) is not int:return None
        if own.get(key) is not None and own[key]!=meta[key]:return None
    enriched=dict(own,road_id=meta['road_id'],lane_id=meta['lane_id'])
    current=dict(frame,actors=[enriched if a['class']=='ego_car' else a for a in frame['actors']])
    return local_priority(current,nav)


def facts(frames,ep,seed):
    from .teacher_rules import conjunction,actor,ordinary_lead
    if not g.history_valid(frames,7) or frames[0]['frame_id']<4:return {},['incomplete_causal_history']
    if any(f.get('source_geometry_issues') for f in frames):return {},['invalid_geometry']
    if anomalies(frames,seed):return {},['instance_rgb_discontinuity']
    f=frames[-1];m=f['meta'];a=actor(f,seed['actor_id']);why=[]
    if a is None:return {},['identity_gap_not_clearance']
    nav=corridor(f,seed) if seed.get('target_world') and ep.state!='DONE' else navigation(f)
    if ep.event=='U-E5' and seed.get('identity_world'):
        nav=dict(points=local_path(f,seed['identity_world']),half_width=seed['identity_half_width'])
    # Bound to this local event, never a later turn 50m down the route.
    nav=bounded(nav)
    if len(nav['points'])<2:return {},['navigation_missing']
    clear=clearance(frames,nav,following=ep.event!='U-E2')
    priority=control_priority(f,nav)
    restricted=False;resolved=False
    follow=any(ordinary_lead(frames,x['id']) and route_intersection(x,nav) is True for x in f['actors'] if x['class']=='car' and not g.vulnerable(x)) if ep.event!='U-E2' else False
    steady=all(p['meta']['speed']>.5 for p in frames[-3:]) and max(p['meta']['speed'] for p in frames[-3:])-min(p['meta']['speed'] for p in frames[-3:])<1.
    if ep.event in ('U-E6','U-E7','R-E5'):
        if ep.event=='U-E7':
            if seed.get('failure_established') is not True:return {},['signal_failure_unestablished']
            # Failure identity removes only the failed signal obligation. STOP
            # duties and current crossing occupancy remain independent vetoes.
            unlit=dict(f,meta=dict(m,light_hazard=False,traffic_light_state=None),
                       actors=[x for x in f['actors'] if x['class']!='traffic_light'])
            priority=control_priority(unlit,nav)
        elif ep.event=='R-E5' and any(x['class']=='traffic_light' and x.get('affects_ego') is True for x in f['actors']):
            return {},['unsignalized_identity_no_longer_supported']
        restricted=clear is False or priority is False
        position=local_path(f,[seed['junction_world']])[0]
        resolved=bool(seed.get('junction_seen') and m.get('is_junction') is False and position[0]<-ego(f)['extent'][0]-2)
    elif ep.event in ('U-E3','U-E5'):
        if g.visible(a,f) is not True and not visible_departure(frames,seed,nav):
            return {},['decisive_actor_not_visible']
        restricted=route_intersection(a,nav) is True and not (ep.event=='U-E3' and ordinary_lead(frames,a['id']))
        if ep.event=='U-E3' and ordinary_lead(frames,a['id']):follow=True
        resolved=clear is True and (follow or route_intersection(a,nav,margin=PARAMS['separation_m']) is False)
    else:restricted=ep.state=='WAIT'  # current navigation/obstacle establishes an entry obligation
    wait=clear is False or priority is False
    result=dict(event_restriction_observed=restricted,release_ready=clear,corridor_clear=clear,
        priority_satisfied=priority,restriction_present=wait if clear is not None and priority is not None else None,
        stationary_wait_required=wait if clear is not None and priority is not None else None,
        restricted_progress_established=conjunction(follow,clear,priority),
        stable_progress_established=conjunction(steady,clear,priority,not follow or ep.longitudinal=='FOLLOW'),
        event_resolved=conjunction(resolved,clear))
    if seed.get('target_world'):
        returning=ep.state=='RETURN';entry=established(frames,seed,returning)
        result.update(target_corridor_known=True,entry_gap_clear=clear,
                      return_gap_clear=clearance(frames,bounded(corridor(f,seed,True)),following=True),
                      lateral_entry_complete=entry,route_corridor_reached=entry)
        objects=[actor(f,i) for i in seed.get('actor_ids',[])]
        result['obstacle_passed']=(all(g.aabb(x)[1]<-ego(f)['extent'][0]-.5 for x in objects) if objects and all(x is not None for x in objects) and scene_visible(f) and (seed.get('obstacle_group_visible') or any(g.visible(x,f) is True for x in objects)) else None)
        # Return needs original-corridor priority/occupancy, not the bypass.
        if ep.state=='PASS' and ep.return_required:
            result['priority_satisfied']=control_priority(f,bounded(corridor(f,seed,True)))
        if ep.state=='WAIT':result['release_ready']=False  # only the event edge grants entry
    if clear is None:why.append('local_visible_space_unresolved')
    if priority is None:why.append('local_priority_unresolved')
    return result,why


def visible_departure(frames,seed,nav):
    """Continuous identity plus a visibly cleared footprint; absence is not exit."""
    from .teacher_rules import actor
    current=actor(frames[-1],seed['actor_id'])
    if current is None or route_intersection(current,nav,margin=PARAMS['separation_m']) is not False:return False
    for frame in frames[-5:-1]:
        obj=actor(frame,seed['actor_id'])
        if obj is not None and g.visible(obj,frame) is True and route_intersection(obj,local_navigation(frame,PARAMS['horizon_m']),margin=PARAMS['separation_m']) is False:
            return True
    return False


def update_instance(frames,seed):
    f=frames[-1];m=f['meta']
    if seed['event']=='U-E2' and not seed.get('obstacle_group_visible'):
        if any(a['id'] in seed.get('actor_ids',[]) and g.visible(a,f) is True for a in f['actors']):
            seed['obstacle_group_visible']=True
            seed['instance_sources']=seed.get('instance_sources',[])+f['sources']
    if seed['event'] in ('U-E6','U-E7','R-E5'):
        seed['junction_seen']=seed.get('junction_seen',False) or m.get('is_junction') is True
        if seed['event']=='U-E7' and not seed.get('negotiated_stop_sources'):
            distance=m.get('distance_to_next_junction')
            if (m.get('is_junction') is True or g.finite(distance) and 0<=distance<=8) and all(abs(p['meta']['speed'])<.1 for p in frames[-3:]):
                seed['negotiated_stop_sources']=source_receipts(frames[-3:])


def retirement(frames,ep,seed):
    """Censor exhausted targets/stale waits; never manufacture a completion."""
    f=frames[-1]
    if seed.get('target_world') and ep.state!='DONE':
        points=local_path(f,seed['original_world'] if ep.state in ('PASS','RETURN') else seed['target_world'])
        if points and max(p[0] for p in points)<-ego(f)['extent'][0]:
            return 'frozen_navigation_exhausted_not_completion'
    if ep.event in ('U-E6','U-E7','R-E5') and ep.state=='YIELD' and seed.get('junction_seen'):
        p=local_path(f,[seed['junction_world']])[0]
        if all(x['meta'].get('is_junction') is False and x['meta'].get('distance_to_next_junction',0)>PARAMS['horizon_m'] for x in frames[-3:]) and p[0]<-ego(f)['extent'][0]-2:
            return 'observed_junction_exit_censors_stale_wait'
    return None


def synchronize_observed(frames,ep,seed):
    """Assimilate executed geometry into OFFLINE state; never a safety receipt.

    An RGB YES still cannot authorize a runtime maneuver without planner/BEV.
    This keeps post-entry milestone questions available in recorded replays.
    """
    if not seed.get('target_world'):return None
    f=frames[-1];before=[ep.state,ep.longitudinal]
    if ep.state=='WAIT' and established(frames,seed) is True:
        # Frozen target must be beyond the original ego footprint at creation.
        if seed.get('direction')=='FORWARD':
            c=closest(corridor(f,seed)['points'])
            if not c or c['station']<8:return None
        ep.state='DEPART' if ep.event=='U-E2' or ep.branch=='cyclist_bypass' else 'CROSS'
    elif ep.state=='PASS' and ep.return_required and established(frames,seed,True) is True:
        ep.state='RETURN'
    else:return None
    ep.longitudinal='RECOVER';ep.uncertain=False;ep.needs_recheck=False;ep.stalled_observations=0
    return dict(before=before,after=[ep.state,ep.longitudinal],sources=source_receipts(frames[-3:]),
                scope='offline_observed_state_only_not_permission_or_label')
