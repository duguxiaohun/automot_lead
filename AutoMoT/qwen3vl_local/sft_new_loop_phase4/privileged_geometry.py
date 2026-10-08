"""Causal LEAD geometry projection. Outputs are proposals, never reviewed labels.

Only trusted local dataset pickles are read. Current-frame allowlists deliberately
exclude saved future arrays, controls and expert target speeds. Active scenario
identity is retained for instance retrieval, never used as permission evidence.
"""
import lzma
import math
import pickle
from pathlib import Path
from .data_paths import data_path
from .identity import file_sha, digest

META_FIELDS = ('speed','road_id','lane_id','ego_lane_width','ego_matrix','is_junction',
               'light_hazard','walker_hazard','vehicle_hazard','stop_sign_hazard',
               'rear_danger_8','rear_danger_16','scenario_obstacles_ids','sensor_information',
               'route','route_original','changed_route','is_intersection','junction_id','distance_to_next_junction','traffic_light_state',
               'vehicle_affecting_id','walker_affecting_id','current_active_scenario_type','scenario_actors_ids',
               'previous_active_scenario_type','cut_in_actors_ids','next_commands','lane_type_str','target_lane_width','signed_dist_to_lane_change')
ACTOR_FIELDS = ('id','class','base_type','type_id','position','extent','yaw','speed','ego_velocity','road_id','lane_id',
                'visible_pixels','num_points','matrix','state','affects_ego','dummy_traffic_light_bounding_box','same_lane_as_ego','role_name','is_cut_in')
# Initial proposal thresholds, not calibrated safety limits or formal label rules.
DEFAULTS = dict(min_visible_pixels=20, following_time_gap_s=2.0, min_following_gap_m=5.0,
                max_relative_speed_mps=2.0, max_lateral_speed_mps=0.5, heading_tolerance_rad=0.25,
                near_disappearance_m=45.0, history_frames=3, frame_period_s=0.25,
                jump_margin_m=3.0, max_actor_acceleration_mps2=15.0)


def finite(value):
    return isinstance(value, (float,int)) and not isinstance(value,bool) and math.isfinite(value)


def vector(value, length):
    return isinstance(value,(list,tuple)) and len(value)==length and all(finite(v) for v in value)


def plain(value):
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'item'):return value.item()
    return value


def project(meta, boxes, frame):
    if type(frame) is not int or frame<0 or not isinstance(meta,dict) or not isinstance(boxes,list):
        raise ValueError('invalid causal frame')
    m={k:plain(meta[k]) for k in META_FIELDS if k in meta}
    actors=[];seen=set();issues=[]
    for raw in boxes:
        if not isinstance(raw,dict) or type(raw.get('id')) is not int or raw['id'] in seen:
            raise ValueError('invalid or duplicate actor identity')
        seen.add(raw['id'])
        a={k:plain(raw[k]) for k in ACTOR_FIELDS if k in raw}
        if a.get('class')=='static_prop_car' and vector(a.get('extent'),3) and min(a['extent'])<0 and all(v!=0 for v in a['extent']):
            issues.append(dict(actor_id=a['id'],kind='negative_static_mesh_extent',original_extent=list(a['extent'])))
            a['extent']=[abs(v) for v in a['extent']]  # diagnostic envelope only; never certifies valid geometry
        if (not isinstance(a.get('class'),str) or not vector(a.get('position'),3)
                or not vector(a.get('extent'),3) or min(a['extent'])<=0
                or not finite(a.get('yaw'))):
            raise ValueError('invalid current actor geometry')
        actors.append(a)
    ego=[a for a in actors if a['class']=='ego_car']
    if len(ego)!=1 or not finite(m.get('speed')):
        raise ValueError('missing unique ego or observed speed')
    return dict(frame_id=frame,meta=m,actors=actors,source_geometry_issues=issues)


def load_frame(root, scenario, route_id, frame):
    from .dataset import source_route
    source_route(dict(scenario=scenario,route_id=route_id))
    root=Path(root).resolve();run=data_path(root, f'{scenario}/{route_id}')
    values={};sources=[]
    for kind in ('metas','bboxes'):
        path=run/kind/f'{frame:04d}.pkl'
        with lzma.open(path,'rb') as f:values[kind]=pickle.load(f)
        sources.append(dict(path=str(path.relative_to(root)),frame_id=frame,kind=kind,sha256=file_sha(path)))
    rgb=run/'rgb'/f'{frame:04d}.jpg'
    sources.append(dict(path=str(rgb.relative_to(root)),frame_id=frame,kind='rgb',sha256=file_sha(rgb)))
    result=project(values['metas'],values['bboxes'],frame)
    result.update(scenario=scenario,route_id=route_id,sources=sources)
    from .privileged_visibility import image_evidence,scene_evidence
    result['image_evidence']=image_evidence(rgb,result)
    result['scene_visibility']=scene_evidence(rgb,result)
    result['causal_sha256']=digest({k:result[k] for k in ('frame_id','meta','actors')})
    return result


def aabb(actor):
    x,y,_=actor['position'];ex,ey,_=actor['extent'];yaw=actor['yaw']
    dx=abs(math.cos(yaw))*ex+abs(math.sin(yaw))*ey
    dy=abs(math.sin(yaw))*ex+abs(math.cos(yaw))*ey
    return [x-dx,x+dx,y-dy,y+dy]


def intersects(a,b):
    return a[0]<=b[1] and b[0]<=a[1] and a[2]<=b[3] and b[2]<=a[3]


def visible(actor, frame, params=DEFAULTS):
    """Positive projection/pixel evidence is necessary, not proof of RGB learnability."""
    quality=frame.get('image_evidence',{}).get(str(actor['id']))
    if not quality or quality.get('quality_pass') is not True:return None
    pixels=actor.get('visible_pixels')
    # Static semantic point counts may be zero even for plainly visible cones.
    # Image/projection evidence can support a static conflict proposal; a failed
    # visibility gate never certifies that occupied space is clear.
    if not obstacle(actor):
        if not finite(pixels) or pixels<0:return None
        if pixels<params['min_visible_pixels']:return False
    cameras=frame['meta'].get('sensor_information',{}).get('camera_calibration')
    if not isinstance(cameras,dict):return None
    for camera in cameras.values():
        if not isinstance(camera,dict):continue
        pos,rot,fov=camera.get('pos'),camera.get('rot'),camera.get('fov')
        if not vector(pos,3) or not vector(rot,3) or not finite(fov) or not 0<fov<180:continue
        x,y,z=[actor['position'][i]-pos[i] for i in range(3)]
        # Unknown pitched/rolled cameras cannot use the horizontal approximation.
        if abs(rot[0])+abs(rot[1])>1e-6:continue
        angle=math.atan2(y,x)-math.radians(rot[2]);angle=math.atan2(math.sin(angle),math.cos(angle))
        if math.hypot(x,y)>0 and abs(angle)<math.radians(fov/2):return True
    return None  # Pixel/projection disagreement is not an invisible negative.


def camera_possible(actor,frame):
    """False only when the entire horizontal bounding circle is outside ALL cameras.

    Pixel count/brightness failure means unresolved, not outside view. Bounds
    near a camera or unsupported calibration remain potentially RGB relevant.
    """
    cameras=frame['meta'].get('sensor_information',{}).get('camera_calibration')
    if not isinstance(cameras,dict) or not cameras:return None
    supported=True
    for camera in cameras.values():
        if not isinstance(camera,dict):supported=False;continue
        pos,rot,fov=camera.get('pos'),camera.get('rot'),camera.get('fov')
        if (not vector(pos,3) or not vector(rot,3) or not finite(fov) or not 0<fov<180
                or abs(rot[0])+abs(rot[1])>1e-6):supported=False;continue
        x,y=[actor['position'][i]-pos[i] for i in range(2)];distance=math.hypot(x,y)
        radius=math.hypot(*actor['extent'][:2])
        if distance<=radius:return True
        angle=math.atan2(y,x)-math.radians(rot[2]);angle=math.atan2(math.sin(angle),math.cos(angle))
        if abs(angle)<=math.radians(fov/2)+math.asin(radius/distance):return True
    return False if supported else None


def history_valid(frames, required=3):
    return (len(frames)>=required and all(b['frame_id']==a['frame_id']+1 for a,b in zip(frames,frames[1:]))
            and len({(f.get('scenario'),f.get('route_id')) for f in frames})==1)


def vulnerable(actor):
    return actor['class']=='walker' or actor.get('base_type') in ('bicycle','motorcycle')


def obstacle(actor):
    return actor['class'] in ('static','static_prop_car') or actor.get('type_id','').startswith('static.')


def physical_actor(actor):
    return actor['class'] in ('car','walker') or obstacle(actor)


def world_point(frame, actor):
    """Rigid current ego transform: remove ego translation AND rotation."""
    import numpy as np
    matrix=frame['meta'].get('ego_matrix')
    try:m=np.asarray(matrix,dtype=float)
    except (TypeError,ValueError):return None
    if (m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-4)
            or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-3)
            or not np.isclose(np.linalg.det(m[:3,:3]),1,atol=1e-3)):return None
    return (m@np.array([*actor['position'],1]))[:3].tolist()


def disappearances(previous,current,params=DEFAULTS):
    """Missing IDs and same-ID impossible world motion are review candidates."""
    if (current['frame_id']!=previous['frame_id']+1 or
            (previous.get('scenario'),previous.get('route_id'))!=(current.get('scenario'),current.get('route_id'))):
        raise ValueError('discontinuity check requires adjacent observations on same route')
    actors={a['id']:a for a in current['actors']};out=[]
    for actor in previous['actors']:
        if not physical_actor(actor) or math.dist(actor['position'],[0,0,0])>=params['near_disappearance_m']:continue
        new=actors.get(actor['id']);kind=None;detail={}
        if new is None:kind='near_actor_missing_candidate'
        elif (actor['class'],actor.get('base_type'))!=(new['class'],new.get('base_type')):kind='actor_identity_class_changed'
        else:
            p,q=world_point(previous,actor),world_point(current,new)
            if p is None or q is None:kind='motion_compensation_unavailable'
            else:
                speeds=[math.hypot(*a['ego_velocity']) for a in (actor,new) if vector(a.get('ego_velocity'),2)]
                # Missing velocity cannot certify plausible motion. Static meshes
                # use a zero nominal speed; previous motion bounds detect jumps.
                speed=speeds[0] if speeds else 0 if obstacle(actor) else None
                if speed is None and finite(actor.get('speed')):speed=abs(actor['speed'])
                if speed is None:kind='actor_motion_bound_unavailable'
                else:
                    dt=params['frame_period_s'];distance=math.dist(p,q)
                    limit=speed*dt+.5*params['max_actor_acceleration_mps2']*dt*dt+params['jump_margin_m']
                    if distance>limit:
                        kind='same_id_position_jump_candidate';detail=dict(world_displacement_m=distance,allowed_displacement_m=limit)
        rgb_relevant=(camera_possible(actor,previous) is not False or
                      (new is not None and camera_possible(new,current) is not False))
        if kind:out.append(dict(rgb_relevant=rgb_relevant,
                               impact='potential_rgb_discontinuity' if rgb_relevant else 'inventory_only',
                               actor_id=actor['id'],previous_frame=previous['frame_id'],frame_id=current['frame_id'],
                               kind=kind,previous_visible=visible(actor,previous,params),
                               status='requires_review_not_confirmed_render_disappearance',**detail))
    return out


def following(frames, params=DEFAULTS):
    """Three-valued geometric following proposal; no expert action timing is read."""
    if not history_valid(frames,params['history_frames']):return None, None, 'insufficient_adjacent_history'
    identities=[];valid=[]
    for frame in frames[-params['history_frames']:]:
        m=frame['meta'];width=m.get('ego_lane_width')
        if not finite(width) or width<=0:return None,None,'missing_lane_geometry'
        if m.get('is_junction') is not False:return None,None,'junction_geometry_not_supported'
        ego=next(a for a in frame['actors'] if a['class']=='ego_car')
        # Road/lane ids alone do not establish geometry on curved or crossing roads.
        candidates=[a for a in frame['actors'] if a['class']=='car' and not vulnerable(a) and a.get('road_id')==m.get('road_id')
                    and a.get('lane_id')==m.get('lane_id') and a['position'][0]>0
                    and abs(a['position'][1])+a['extent'][1]<=width/2
                    and abs(a['yaw'])<=params['heading_tolerance_rad']]
        if not candidates:return None,None,'no_tracked_following_vehicle'
        lead=min(candidates,key=lambda a:a['position'][0]);identities.append(lead['id'])
        if visible(lead,frame,params) is not True:return None,lead['id'],'lead_not_visibly_resolved'
        v=lead.get('ego_velocity')
        if not vector(v,2):return None,lead['id'],'missing_actor_velocity'
        gap=aabb(lead)[0]-ego['extent'][0]
        gap_ok=gap>=max(params['min_following_gap_m'],m['speed']*params['following_time_gap_s'])
        valid.append(gap_ok and m['speed']>0.5 and v[0]>0.5
                     and abs(v[0]-m['speed'])<=params['max_relative_speed_mps']
                     and abs(v[1])<=params['max_lateral_speed_mps'])
        if any(m.get(k) is not False for k in ('light_hazard','stop_sign_hazard','walker_hazard')):
            return None,lead['id'],'priority_or_vulnerable_conflict_unresolved'
        corridor=[0,max(5,lead['position'][0]),-width/2,width/2]
        if any(a['class']!='ego_car' and a['id']!=lead['id'] and intersects(aabb(a),corridor)
               for a in frame['actors']):return None,lead['id'],'additional_corridor_participant'
    if len(set(identities))!=1:return None,None,'lead_identity_changed'
    return all(valid),identities[-1],'geometric_proposal_requires_rgb_calibration'


def maneuver_facts(frames, episode, edge_key, context, params=DEFAULTS):
    """Compute gap/passage/entry proposals for explicitly grounded route geometry.

    Navigation/priority evidence is separate from expert controls. No lateral
    direction, priority, completed route or whole-obstacle membership is guessed
    from source event labels or future ego lanes.
    """
    if context is None:return {},['current_navigation_geometry_required']
    current=frames[-1];f=current['frame_id'];returning=edge_key=='return' or episode.state=='RETURN'
    expected=episode.return_corridor if returning else episode.target_corridor
    required={'frame_id','target_corridor','corridor_bounds','target_road_id','target_lane_id',
              'priority_satisfied','visible_space_resolved','obstacle_actor_ids','obstacle_membership_complete',
              'source','evidence_id'}
    if (not isinstance(context,dict) or set(context)!=required or type(context['frame_id']) is not int
            or context['frame_id']!=f or not expected or context['target_corridor']!=expected
            or context['source'] not in ('planner','causal_navigation_review')
            or not isinstance(context['evidence_id'],str) or not context['evidence_id'].strip()
            or not vector(context['corridor_bounds'],4)
            or any(context[k] is not None and type(context[k]) is not bool for k in
                   ('priority_satisfied','visible_space_resolved','obstacle_membership_complete'))
            or not isinstance(context['obstacle_actor_ids'],list)
            or any(type(i) is not int for i in context['obstacle_actor_ids'])
            or len(set(context['obstacle_actor_ids']))!=len(context['obstacle_actor_ids'])
            or any(type(context[k]) is not int for k in ('target_road_id','target_lane_id'))):
        raise ValueError('invalid current navigation geometry/priority evidence')
    corridor=context['corridor_bounds']
    if corridor[0]>=corridor[1] or corridor[2]>=corridor[3]:raise ValueError('invalid navigation corridor')
    conflict=False;unresolved=False
    for actor in current['actors']:
        if not physical_actor(actor) or not intersects(aabb(actor),corridor):continue
        visibility=visible(actor,current,params)
        if visibility is True:conflict=True
        else:unresolved=True
    gap=False if conflict else True if context['visible_space_resolved'] is True and not unresolved else None
    facts=dict(target_corridor_known=True,priority_satisfied=context['priority_satisfied'],
               entry_gap_clear=gap,return_gap_clear=gap)
    if context['obstacle_actor_ids'] and context['obstacle_membership_complete'] is True:
        actors={a['id']:a for a in current['actors']};ego=next(a for a in current['actors'] if a['class']=='ego_car')
        if all(i in actors for i in context['obstacle_actor_ids']):
            facts['obstacle_passed']=all(aabb(actors[i])[1]<-ego['extent'][0] for i in context['obstacle_actor_ids'])
        else:facts['obstacle_passed']=None  # Disappearance is not passage.
    established=[]
    for frame in frames:
        ego=next(a for a in frame['actors'] if a['class']=='ego_car');m=frame['meta']
        established.append(m.get('road_id')==context['target_road_id'] and m.get('lane_id')==context['target_lane_id']
                           and corridor[2]<=-ego['extent'][1] and ego['extent'][1]<=corridor[3])
    # Bounds are current coordinates. Only the current footprint uses them;
    # historical road/lane identity is corroboration, not a mapped past polygon.
    if history_valid(frames,params['history_frames']):
        facts['route_corridor_reached']=all(established)
        facts['lateral_entry_complete']=all(established)  # Candidate only: lateral drift needs RGB calibration.
    return facts,['navigation_bound_geometry_proposal_requires_rgb_calibration']


def condition_proposal(frames, episode, edge_key, params=DEFAULTS, *, geometry_context=None, scene_context=None):
    from .route_context import episode_edge
    from .taxonomy import applicable
    edge=episode_edge(episode,edge_key)
    if not applicable(edge,episode.state,episode.longitudinal):raise ValueError('inapplicable proposal edge')
    facts={k:None for k in edge.criteria};reasons=[];current=frames[-1]
    if any(f.get('source_geometry_issues') for f in frames):
        return dict(facts=facts,target='UNKNOWN',reasons=['invalid_static_source_geometry'],anomalies=[])
    anomalies=[r for a,b in zip(frames,frames[1:]) for r in disappearances(a,b,params)]
    if any(r.get('rgb_relevant',True) for r in anomalies):
        return dict(facts=facts,target='UNKNOWN',reasons=['near_actor_disappearance'],anomalies=anomalies)
    if edge.key=='recover_follow' or 'restricted_progress_established' in facts:
        value,actor,reason=following(frames,params);facts['restricted_progress_established']=value;reasons.append(reason)
    elif edge.key in ('depart','enter','return') or (edge.key in ('pass','complete') and episode.state in ('DEPART','PASS','CROSS','RETURN')):
        derived,why=maneuver_facts(frames,episode,edge_key,geometry_context,params)
        facts.update({k:v for k,v in derived.items() if k in facts});reasons.extend(why)
    else:
        reasons.append('event_identity_route_priority_or_visibility_evidence_required')
        # Visible positive conflicts may veto a clearance proposal; an empty box
        # set or a false hazard flag never establishes free space/priority.
        m=current['meta'];width=m.get('ego_lane_width')
        if finite(width) and width>0:
            corridor=[0,max(5,m['speed']*params['following_time_gap_s']),-width/2,width/2]
            conflict=any(physical_actor(a) and visible(a,current,params) is True
                         and intersects(aabb(a),corridor) for a in current['actors'])
            if conflict and 'corridor_clear' in facts:facts['corridor_clear']=False
        if m.get('light_hazard') is True or m.get('stop_sign_hazard') is True:
            # Expert hazards identify a review obligation, not an RGB negative.
            reasons.append('privileged_priority_hazard_needs_visible_review')
    if scene_context is not None:
        from .privileged_context import scene_facts
        derived,why=scene_facts(frames,episode,scene_context,params)
        facts.update({k:v for k,v in derived.items() if k in facts});reasons.extend(why)
    target='NO' if False in facts.values() else 'YES' if facts and all(v is True for v in facts.values()) else 'UNKNOWN'
    return dict(facts=facts,target=target,reasons=reasons,anomalies=anomalies)
