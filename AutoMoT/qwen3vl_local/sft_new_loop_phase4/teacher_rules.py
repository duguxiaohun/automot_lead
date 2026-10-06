"""Versioned UE1/UE4 weak teacher. Current/past evidence only; approval separate.

Scene names and privileged affected IDs retrieve seeds; visible local conflict
must independently establish the instance. No expert controls/future arrays.
"""
import math
from .identity import ROOT,digest,file_sha
from . import privileged_geometry as g
from .event_scope import local_navigation
from .automatic_context import route_intersection

VERSION='ue1_ue4_causal_teacher_v8'
PARAMS=dict(local_horizon_m=18.,seed_horizon_m=30.,vru_margin_m=1.,boundary_band_m=.25,
            lead_growth_m=1.,lead_min_gap_m=3.,min_pixel_change=6.,
            stationary_lead_displacement_m=.75,stationary_ego_displacement_m=.25,
            stable_gap_m=3.,stable_time_gap_s=.5,relative_speed_mps=3.,
            relative_speed_range_mps=1.5,max_closing_speed_mps=.5,max_gap_loss_m=.5,
            receipt_displacement_m=.75,receipt_max_lateral_m=.5,receipt_heading_tolerance=.05,
            lead_min_forward_speed_mps=.5,resumption_observations=5,resumption_distance_m=2.)
SOURCES=('branch_support.py','paired_eval.py','route_quality.py','weighted_sampling.py','teacher_evaluation.py','teacher_controls.py','teacher_rules.py','teacher_replay.py','teacher_approval.py','teacher_data.py','teacher_review.py','teacher_pool.py',
         'privileged_geometry.py','privileged_visibility.py','event_scope.py','automatic_context.py',
         'controller.py','taxonomy.py','route_context.py','route_calibration.py','calibration.py',
         'observation.py','prompts.py','route_prompts.py','risk_review.py',
         'dataset.py','candidate_pool.py','preflight.py','evaluate.py','identity.py')


def identity():
    return dict(version=VERSION,parameters=PARAMS,sources={n:file_sha(ROOT/n) for n in SOURCES})


def rule_class(event,edge,mode,phase='readiness'):
    return f'{VERSION}/{event}/{edge}/rgb{mode}/{phase}'


def actor(frame,ident):
    return next((a for a in frame['actors'] if a['id']==ident),None)


def seeds(frames):
    """Retrieve identity hints, then require present observable local conflict."""
    if not g.history_valid(frames,3):return []
    f=frames[-1];m=f['meta'];nav=local_navigation(f,PARAMS['seed_horizon_m']);out=[]
    for a in f['actors']:
        if not g.physical_actor(a) or not 0<a['position'][0]<PARAMS['seed_horizon_m']:continue
        hints=[]
        if a.get('role_name')=='scenario':hints.append('role_name')
        if a['id'] in (m.get('scenario_obstacles_ids') or []):hints.append('scenario_obstacles_ids')
        if a['id'] in (m.get('vehicle_affecting_id'),m.get('walker_affecting_id')):hints.append('affecting_id')
        if g.visible(a,f) is not True or route_intersection(a,nav) is not True:continue
        if g.vulnerable(a):event='U-E4'
        elif (m.get('road_id') is not None and m.get('lane_id') is not None and a['class']=='car' and a['id'] not in (m.get('scenario_obstacles_ids') or []) and abs(a['yaw'])<.25 and a.get('road_id')==m.get('road_id')
                and a.get('lane_id')==m.get('lane_id')):
            history=[actor(p,a['id']) for p in frames[-3:]]
            if not all(b is not None and g.visible(b,p) is True for b,p in zip(history,frames[-3:])):continue
            # Establish a braking/queued lead, not every normally moving car.
            decelerating=(all(g.finite(b.get('speed')) for b in history)
                          and history[0]['speed']-history[-1]['speed']>.5
                          and a['position'][0]<max(8.,m['speed']*1.2))
            if not (a.get('speed',999)<.5 or decelerating):continue
            event='U-E1'
        else:continue
        # Names are retrieval context, not evidence for release/permission.
        if f.get('scenario') in ('HardBreakRoute','DynamicObjectCrossing','PedestrianCrossing','VehicleTurningRoute'):
            hints.append('source_scenario_prior')
        out.append(dict(event=event,actor_id=a['id'],branch='cyclist_follow' if event=='U-E4' and parallel_cyclist(frames,a['id']) else 'default',identity_sources=hints or ['local_visible_geometry'],
                        established_by='visible_local_conflict_not_source_fragment'))
    # The closest aligned vehicle is the lead. Distant queued cars are not
    # separate indistinguishable lead-braking instances in the same RGB frame.
    leads=[s for s in out if s['event']=='U-E1']
    lead=min(leads,key=lambda s:actor(f,s['actor_id'])['position'][0]) if leads else None
    aligned=[a for a in f['actors'] if a['class']=='car' and not g.vulnerable(a)
             and 0<a['position'][0] and abs(a['yaw'])<.25
             and a.get('road_id')==m.get('road_id') and a.get('lane_id')==m.get('lane_id')
             and route_intersection(a,nav) is True]
    nearest=min(aligned,key=lambda a:a['position'][0]) if aligned else None
    return [s for s in out if s['event']!='U-E1' or s is lead and nearest is not None and s['actor_id']==nearest['id']]


def route_motion(frame, participant):
    """Motion at the participant's current route station; no endpoint extension.

    Ambiguous self-crossings and actors outside the planned strip abstain. The
    returned gap is arc length less both projected footprints, not ego-axis x.
    """
    from .automatic_context import navigation
    nav=navigation(frame)
    if not nav.get('points') or not g.vector(participant.get('ego_velocity'),2):return None
    candidates=[];station=math.dist((0.,0.),nav['points'][0])
    x,y=participant['position'][:2]
    for a,b in zip(nav['points'],nav['points'][1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<1e-5:continue
        ux,uy=dx/length,dy/length
        along=(x-a[0])*ux+(y-a[1])*uy;lateral=-(x-a[0])*uy+(y-a[1])*ux
        angle=math.atan2(uy,ux);yaw=math.atan2(math.sin(participant['yaw']-angle),math.cos(participant['yaw']-angle))
        ex,ey=participant['extent'][:2]
        if 0<=along<=length and abs(lateral)<=nav['half_width']:
            extent=abs(math.cos(yaw))*ex+abs(math.sin(yaw))*ey
            vx,vy=participant['ego_velocity'][:2]
            ego=next(a for a in frame['actors'] if a['class']=='ego_car')
            candidates.append((abs(lateral),angle,dict(yaw=yaw,ego_velocity=[vx*ux+vy*uy,-vx*uy+vy*ux],
                              gap=station+along-extent-ego['extent'][0])))
        station+=length
    if not candidates:return None
    candidates.sort(key=lambda c:c[0]);best=candidates[0]
    if any(c[0]<=best[0]+.25 and abs(math.atan2(math.sin(c[1]-best[1]),math.cos(c[1]-best[1])))>.25 for c in candidates[1:]):return None
    return best[2]


def parallel_cyclist(frames,ident):
    """Retrieve a following branch from observed motion, never scenario name."""
    if len(frames)<3:return False
    for f in frames[-3:]:
        a=actor(f,ident)
        if (a is None or a.get('base_type') not in ('bicycle','motorcycle')
                or g.visible(a,f) is not True):return False
        motion=route_motion(f,a)
        if (motion is None or abs(motion['yaw'])>.25 or motion['ego_velocity'][0]<.5
                or abs(motion['ego_velocity'][1])>.35):return False
    return True


def ordinary_cyclist(frames,ident):
    """A moving, separated parallel cyclist permits following, never overtaking."""
    if not parallel_cyclist(frames,ident):return False
    for f in frames[-3:]:
        motion=route_motion(f,actor(f,ident));speed=f['meta']['speed']
        if (motion['gap']<max(PARAMS['stable_gap_m'],speed*PARAMS['stable_time_gap_s'])
                or motion['ego_velocity'][0]-speed<-PARAMS['max_closing_speed_mps']):return False
    return True


def ordinary_lead(frames,ident):
    """A visible moving following gap is not another stationary obstruction."""
    if len(frames)<3:return False
    for f in frames[-3:]:
        a=actor(f,ident)
        if (a is None or a['class']!='car' or g.vulnerable(a) or g.visible(a,f) is not True
                or abs(a['yaw'])>.25 or not g.vector(a.get('ego_velocity'),2)
                or any(f['meta'].get(k) is None or a.get(k)!=f['meta'][k] for k in ('road_id','lane_id'))):return False
        ego=next(x for x in f['actors'] if x['class']=='ego_car')
        gap=g.aabb(a)[0]-ego['extent'][0];speed=f['meta']['speed']
        if (gap<max(PARAMS['stable_gap_m'],speed*PARAMS['stable_time_gap_s'])
                or a['ego_velocity'][0]<.5 or a['ego_velocity'][0]-speed<-.5
                or abs(a['ego_velocity'][1])>.25):return False
    return True


def lead_retirement(frames,episode,ident):
    """End obsolete identity scope, not a driving permission or complete label.

    Require consecutive visible evidence; missing/teleported actors are handled
    by the anomaly gate first. New leaders must independently seed a restriction.
    """
    if episode.event!='U-E1' or len(frames)<3:return None
    departed=[]
    for f in frames[-3:]:
        a=actor(f,ident)
        if a is None or g.visible(a,f) is not True:return None
        nav=local_navigation(f,min(55.,max(30.,a['position'][0]+3.)))
        outside=route_intersection(a,nav,margin=.5) is False
        ego=next(x for x in f['actors'] if x['class']=='ego_car')
        separated=(g.aabb(a)[1]<-ego['extent'][0] or
                   abs(a['position'][1])>nav.get('half_width',float('inf'))+max(a['extent'][:2])+.5)
        departed.append(outside and separated)
    if all(departed):return 'lead_visibly_left_event_corridor'
    if all(g.aabb(actor(f,ident))[0]>PARAMS['seed_horizon_m'] for f in frames[-3:]):
        return 'lead_visibly_beyond_event_horizon'
    f=frames[-1];a=actor(f,ident)
    replacements=[x for x in f['actors'] if x['id']!=ident and 0<x['position'][0]<a['position'][0]-2
                  and ordinary_lead(frames,x['id'])]
    if replacements:return 'lead_replaced_by_observed_moving_lead'
    return None


def stationary_lead_start(frames,ident):
    """Visible participant motion while ego is still waiting; no expert action.

    Keep the pixel-change gate. Privileged subpixel movement cannot turn an
    unobservable boundary into an RGB YES just to improve class balance.
    """
    fs=frames[-5:]
    if len(fs)!=5 or any(abs(f['meta']['speed'])>=.5 for f in fs):return False
    first,last=fs[0],fs[-1];a,b=actor(first,ident),actor(last,ident)
    p,q=g.world_point(first,a),g.world_point(last,b)
    e0,e1=ego_point(first),ego_point(last)
    if any(x is None for x in (p,q,e0,e1)) or math.dist(e0,e1)>PARAMS['stationary_ego_displacement_m']:return False
    m=last['meta']['ego_matrix'];before=first['meta']['ego_matrix']
    if any(abs(m[i][j]-before[i][j])>.03 for i in range(3) for j in range(3)):return False
    delta=[y-x for x,y in zip(p,q)]
    forward=sum(delta[i]*m[i][0] for i in range(3));lateral=sum(delta[i]*m[i][1] for i in range(3))
    return (forward>=PARAMS['stationary_lead_displacement_m'] and abs(lateral)<.25
            and all(g.vector(actor(f,ident).get('ego_velocity'),2)
                    and actor(f,ident)['ego_velocity'][0]>=.5 for f in fs[-2:])
            and image_change(frames,ident) is True)


def motion_stratum(frames):
    return 'stationary' if abs(frames[-1]['meta']['speed'])<.5 else 'moving'


def image_change(frames,ident):
    """Projected change plus RGB quality veto: a weak teacher, needs calibration."""
    if len(frames)<5:return None
    a,b=frames[-5],frames[-1];old=a.get('image_evidence',{}).get(str(ident),{})
    new=b.get('image_evidence',{}).get(str(ident),{})
    for x in old.get('crops',[]):
        for y in new.get('crops',[]):
            if not x['usable'] or not y['usable']:continue
            u,v=x['bounds'],y['bounds']
            if camera_for_crop(a,u)!=camera_for_crop(b,v) or camera_for_crop(a,u) is None:continue
            if max(abs((u[3]-u[1])-(v[3]-v[1])),math.dist([(u[0]+u[2])/2,(u[1]+u[3])/2],[(v[0]+v[2])/2,(v[1]+v[3])/2]))>=PARAMS['min_pixel_change']:
                return True
    return None


def following_stable(frames,history,gaps):
    """Observed following milestone, not a certified safe headway policy.

    Absolute ego-frame actor velocity must first subtract measured ego speed.
    Five observations must agree: finite gap, no sustained closure, bounded
    relative velocity variation. Merely lowering a headway threshold is unsafe.
    """
    if not all(g.vector(b.get('ego_velocity'),2) for b in history):return None
    relative=[b['ego_velocity'][0]-p['meta']['speed'] for b,p in zip(history,frames)]
    return (all(p['meta']['speed']>.5 and gap>=max(PARAMS['stable_gap_m'],p['meta']['speed']*PARAMS['stable_time_gap_s'])
                and abs(b['ego_velocity'][1])<.25 for p,b,gap in zip(frames,history,gaps))
            and min(relative)>=-PARAMS['max_closing_speed_mps']
            and max(relative)<=PARAMS['relative_speed_mps']
            and max(relative)-min(relative)<=PARAMS['relative_speed_range_mps']
            and all(y>=x-PARAMS['max_gap_loss_m'] for x,y in zip(gaps,gaps[1:]))
            and gaps[-1]>=gaps[0]-PARAMS['max_gap_loss_m'])


def distant_control(frame,control,nav):
    """Exclude only a control demonstrably beyond this local progress horizon.

    Lamp/proxy position alone cannot locate the ego stop line: signals also
    require the map junction to be beyond the horizon. Hazard flags still veto
    in local_priority. Unknown geometry keeps the original control obligation.
    """
    points=nav.get('points')
    if (not points or not g.vector(control.get('position'),3)
            or not g.vector(control.get('extent'),3) or min(control['extent'])<0):return False
    # Circle enclosing the entire requested corridor plus a five metre buffer;
    # taking the full box radius prevents a large trigger being called distant.
    reach=max(math.hypot(*p[:2]) for p in points)+5.
    if math.hypot(*control['position'][:2])-math.hypot(*control['extent'][:2])<=reach:return False
    if control['class']=='stop_sign':return True
    distance=frame['meta'].get('distance_to_next_junction')
    return (frame['meta'].get('is_junction') is False and
            (g.finite(distance) or distance==math.inf) and distance>reach)


def local_priority(frame,nav):
    """Local UE1/4 progress only; never certifies a junction-wide maneuver.

    Known lane controls veto progress. Unmatched/unknown controls abstain;
    applicable green needs matching logical and visible actor evidence. A map
    junction flag alone is neither a restriction nor permission.
    """
    m=frame['meta']
    if not g.finite(nav.get('review_horizon_m')):return None
    if m.get('light_hazard') is True or m.get('stop_sign_hazard') is True:return False
    if m.get('light_hazard') is not False or m.get('stop_sign_hazard') is not False:return None
    distance=m.get('distance_to_next_junction')
    if not (type(m.get('is_junction')) is bool and (g.finite(distance) or distance==math.inf) and distance>=0):return None
    controls=[a for a in frame['actors'] if a['class'] in ('traffic_light','stop_sign')
              and not distant_control(frame,a,nav)]
    if any(type(a.get('affects_ego')) is not bool for a in controls):return None
    active=[a for a in controls if a.get('affects_ego') is True]
    if active:
        stops=[a for a in active if a['class']=='stop_sign']
        if stops:
            from .teacher_controls import crossing_clear
            duties=frame.get('stop_duties',{})
            if any(duties.get(str(a['id']),{}).get('satisfied') is not True for a in stops):return None
            clear=conjunction(*(crossing_clear(frame,a) for a in stops))
            if clear is not True:return clear
            active=[a for a in active if a['class']!='stop_sign']
            if not active:return True
        if any(str(a.get('state')).lower() in ('red','yellow') for a in active):return False
        if (str(m.get('traffic_light_state')).lower()!='green'
                or any(str(a.get('state')).lower()!='green' or g.visible(a,frame) is not True for a in active)):return None
        return True  # local occupancy is checked independently below
    if m['is_junction'] is False and distance>nav['review_horizon_m']+5:return True
    # Near a junction, require identified lane ownership. Source scenario names
    # and the expert's choice to move supply no right-of-way evidence.
    if m.get('road_id') is None or m.get('lane_id') is None:return None
    ego=next(a for a in frame['actors'] if a['class']=='ego_car')
    if ego.get('road_id')!=m['road_id'] or ego.get('lane_id')!=m['lane_id']:return None
    return True


def scoped_anomalies(frames,ident):
    """Anomaly scope follows this participant and its current local corridor.

    Both endpoints matter (including a jump into the region). Decisive actor
    loss always censors; unrelated camera-visible deletions remain in the route
    ledger and in the full-scene safety adapter, without poisoning this event.
    """
    out=[]
    for previous,current in zip(frames,frames[1:]):
        for issue in g.disappearances(previous,current):
            relevant=issue['actor_id']==ident
            if not relevant and issue['rgb_relevant']:
                for frame in (previous,current):
                    decisive=actor(frame,ident);other=actor(frame,issue['actor_id'])
                    if other is None:continue
                    horizon=min(PARAMS['local_horizon_m'],max(6.,decisive['position'][0]+3.)) if decisive else PARAMS['local_horizon_m']
                    nav=local_navigation(frame,horizon)
                    if route_intersection(other,nav,margin=.25) is not False:relevant=True
            if relevant:out.append(issue)
    return out


def facts(frames,episode,ident):
    f=frames[-1];m=f['meta'];a=actor(f,ident);why=[]
    empty={}
    if episode.event not in ('U-E1','U-E4'):return empty,['unsupported_event']
    if not g.history_valid(frames,7) or frames[0]['frame_id']<4:return empty,['incomplete_model_history']
    if any(p.get('source_geometry_issues') for p in frames):return empty,['invalid_geometry']
    if scoped_anomalies(frames,ident):return empty,['instance_rgb_discontinuity']
    if a is None or any(actor(p,ident) is None for p in frames):return empty,['identity_gap_not_clearance']
    if g.visible(a,f) is not True:return empty,['decisive_actor_not_visible']
    nav=local_navigation(f,min(PARAMS['local_horizon_m'],max(6.,a['position'][0]+3.)))
    if not nav.get('points'):return empty,['navigation_missing']
    priority=local_priority(f,nav)
    clear=None;blocked=False;other_unknown=False;other_block=False
    for other in f['actors']:
        if not g.physical_actor(other) or other['id']==ident or route_intersection(other,nav) is not True:continue
        if ordinary_lead(frames,other['id']) or ordinary_cyclist(frames,other['id']):continue
        if g.visible(other,f) is True:other_block=True
        else:other_unknown=True
    stable=False
    following_event=episode.event=='U-E1' or episode.branch=='cyclist_follow'
    if following_event:
        history=[actor(p,ident) for p in frames[-5:]];gaps=[]
        motions=[route_motion(p,b) if b is not None else None for p,b in zip(frames[-5:],history)] if episode.branch=='cyclist_follow' else None
        if motions is not None:
            if any(v is None for v in motions):return empty,['cyclist_route_motion_unresolved']
            history=[dict(b,**{k:v[k] for k in ('yaw','ego_velocity')}) for b,v in zip(history,motions)]
        if any(b is None or g.visible(b,p) is not True or abs(b['yaw'])>.25
               or (episode.event=='U-E1' and (b.get('lane_id')!=p['meta'].get('lane_id') or b.get('road_id')!=p['meta'].get('road_id')))
               for b,p in zip(history,frames[-5:])):return empty,['lead_track_unresolved']
        for p,b in zip(frames[-5:],history):
            ego=next(x for x in p['actors'] if x['class']=='ego_car');gaps.append(g.aabb(b)[0]-ego['extent'][0])
        if motions is not None:gaps=[v['gap'] for v in motions]
        moving_actor=history[-1]
        stable=following_stable(frames[-5:],history,gaps)
        blocked=a.get('speed',999)<.2 and gaps[-1]<8.
        growth=gaps[-1]-gaps[0]
        if stable is True:clear=True
        elif (g.vector(moving_actor.get('ego_velocity'),2) and moving_actor['ego_velocity'][0]>=PARAMS['lead_min_forward_speed_mps']
              and abs(moving_actor['ego_velocity'][1])<.5 and (growth>=PARAMS['lead_growth_m'] or stationary_lead_start(frames,ident)) and gaps[-1]>=PARAMS['lead_min_gap_m']
              and image_change(frames,ident) is True and all(y>=x-.05 for x,y in zip(gaps,gaps[1:]))):clear=True
        elif blocked:clear=False
        else:why.append('lead_release_boundary_or_visual_change_unresolved')
    else:
        blocked=route_intersection(a,nav) is True
        if blocked:clear=False
        elif route_intersection(a,nav,margin=PARAMS['vru_margin_m']+PARAMS['boundary_band_m']) is False:
            first=actor(frames[-5],ident)
            # Compare actor world motion in the current ego orientation so
            # ego translation cannot erase (or invent) outward movement.
            p,q=g.world_point(frames[-5],first),g.world_point(f,a)
            matrix1,matrix2=frames[-5]['meta'].get('ego_matrix'),m.get('ego_matrix')
            same_heading=matrix1 is not None and matrix2 is not None and all(abs(matrix1[i][j]-matrix2[i][j])<.03 for i in (0,1) for j in (0,1))
            lateral=sum((q[i]-p[i])*matrix2[i][1] for i in range(3)) if p is not None and q is not None and same_heading else None
            moving_away=(lateral is not None and lateral*(1 if a['position'][1]>0 else -1)>.25
                         and all(g.visible(actor(p,ident),p) is True for p in frames[-5:]))
            passed=g.aabb(a)[1]<-next(x['extent'][0] for x in f['actors'] if x['class']=='ego_car')
            # Once stopped safely outside the corridor, loss of outward motion
            # is not a renewed conflict. Require sustained whole-footprint
            # separation, continuous identity and observed low actor speed.
            stopped_separated=all(
                actor(p,ident).get('speed',999)<.2 and
                route_intersection(actor(p,ident),local_navigation(p,PARAMS['local_horizon_m']),
                    margin=PARAMS['vru_margin_m']+PARAMS['boundary_band_m']) is False
                for p in frames[-5:])
            if moving_away or passed or stopped_separated:clear=True
        if clear is None:why.append('vru_margin_or_direction_unresolved')
    event_restriction=blocked
    if episode.branch=='cyclist_follow':
        # A moving cyclist calls for reduced progress, not an automatic STOP.
        event_restriction=(clear is not True and gaps[-1]<max(8.,m['speed']*1.2))
    if other_block:clear=False;why.append('additional_local_conflict')
    elif other_unknown:clear=None;why.append('unknown_occupancy_in_conflict_region')
    restriction=True if blocked or other_block else False if clear is True else None
    wait=True if blocked or other_block or priority is False else False if clear is True and priority is True else None
    # Stable ordinary following is not unrestricted driving; no simultaneous
    # stable/recover_follow YES. Completion waits for the relevant milestone.
    follow=conjunction(stable,clear,priority) if following_event else False
    unrestricted=None
    if clear is True and priority is True:
        speeds=[p['meta']['speed'] for p in frames[-5:]]
        steady=min(speeds)>.5 and max(speeds)-min(speeds)<.5
        if not following_event:unrestricted=steady
        else:
            # Only a continuously visible, separated lead can establish free
            # progress. A missing lead or a short gap cannot finish the event.
            separated=all(gap>max(PARAMS['local_horizon_m'],p['meta']['speed']*2.)
                          for gap,p in zip(gaps,frames[-5:]))
            unrestricted=steady and separated and follow is not True
    resolved=conjunction(clear,True if follow is True or unrestricted is True else None if follow is None or unrestricted is None else False)
    release=False if episode.state=='PROCEED' and episode.longitudinal=='FOLLOW' and follow else clear
    return dict(event_restriction_observed=event_restriction,release_ready=release,corridor_clear=clear,priority_satisfied=priority,
        restriction_present=restriction,stationary_wait_required=wait,
        restricted_progress_established=follow if not other_unknown else None,
        stable_progress_established=follow if episode.state=='PROCEED' and episode.longitudinal=='FOLLOW' else unrestricted,
        event_resolved=resolved),why


def ego_point(frame):
    return g.world_point(frame,next(a for a in frame['actors'] if a['class']=='ego_car'))


def observed_resumption(frames,episode,status):
    """Censor a stale waiting instance; never grant permission or label success.

    A prior observed stop and sustained *actual forward* motion are required.
    Integrate in each observation's body frame so a normal curve does not keep
    HOLD alive. Reverse/sideways motion, jumps and gaps cannot establish this.
    The runtime's permission-bound execution receipt remains unchanged.
    """
    if episode.event!='U-E1' or episode.state!='YIELD':return None
    if g.history_valid(frames,3) and all(abs(f['meta']['speed'])<.5 for f in frames[-3:]):
        status['stop']=[dict(frame_id=f['frame_id'],sources=f['sources']) for f in frames[-3:]]
    count=PARAMS['resumption_observations'];fs=frames[-count:]
    stop=status.get('stop')
    if (not stop or not g.history_valid(fs,count) or stop[-1]['frame_id']>=fs[0]['frame_id']
            or any(f['meta']['speed']<=.5 for f in fs)):return None
    distance=0.
    for a,b in zip(fs,fs[1:]):
        p,q=ego_point(a),ego_point(b)
        if p is None or q is None:return None
        matrix=a['meta']['ego_matrix'];other=b['meta']['ego_matrix']
        if sum(matrix[i][0]*other[i][0] for i in range(3))<.9:return None
        delta=[y-x for x,y in zip(p,q)]
        forward=sum(delta[i]*matrix[i][0] for i in range(3))
        lateral=sum(delta[i]*matrix[i][1] for i in range(3))
        limit=max(a['meta']['speed'],b['meta']['speed'])*.5+.25
        if not 0<forward<=limit or abs(lateral)>max(.15,forward*.35):return None
        distance+=forward
    if distance<PARAMS['resumption_distance_m']:return None
    return dict(stop_observations=stop,forward_distance_m=distance,
                motion_observations=[dict(frame_id=f['frame_id'],sources=f['sources']) for f in fs],
                scope='observed execution censors stale teacher state; no permission or supervision')


def execution_motion(frames):
    """Longitudinal execution requires forward motion, not just displacement.

    This is tracker evidence only. It does not certify visibility in sparse RGB.
    """
    if not g.history_valid(frames,5):return False
    first,last=frames[-5],frames[-1]
    p,q=ego_point(first),ego_point(last)
    if p is None or q is None or last['meta']['speed']<=.5:return False
    before,after=first['meta']['ego_matrix'],last['meta']['ego_matrix']
    if any(abs(before[i][j]-after[i][j])>PARAMS['receipt_heading_tolerance'] for i in range(3) for j in range(3)):
        return False
    delta=[b-a for a,b in zip(p,q)]
    longitudinal=sum(delta[i]*before[i][0] for i in range(3))
    lateral=sum(delta[i]*before[i][1] for i in range(3))
    return longitudinal>=PARAMS['receipt_displacement_m'] and abs(lateral)<=PARAMS['receipt_max_lateral_m']


def answer(criteria,values):
    vals=[values.get(k) for k in criteria]
    if any(v is not None and type(v) is not bool for v in vals):raise ValueError('teacher conditions must be bool or None')
    return 'NO' if False in vals else 'YES' if vals and all(x is True for x in vals) else 'UNKNOWN'


def conjunction(*values):
    return False if False in values else True if all(v is True for v in values) else None


def camera_for_crop(frame,bounds):
    cameras=frame['meta'].get('sensor_information',{}).get('camera_calibration',{})
    offset=0
    for key in sorted(cameras,key=lambda x:int(x)):
        width=cameras[key].get('width')
        if type(width) is not int or width<=0:return None
        if offset<=bounds[0]<bounds[2]<=offset+width:return key
        offset+=width
    return None
