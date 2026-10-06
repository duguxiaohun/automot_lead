"""Offline geometry adapter for a planner's current maneuver envelope.

Boxes alone do not establish inventory completeness, drivable free space, traffic
priority or the intended route. A current source-bound planner request must
supply those prerequisites. No request means no clearance; rear_danger=False
is never used as a shortcut. The adapter is not a CARLA sensor integration.
"""
import math
from .identity import digest
from .maneuver_safety import binding, GUARDED, validate_clearances
from .privileged_geometry import load_frame, vector, finite, aabb, intersects, disappearances, world_point

POLICY='offline_swept_envelope_v1'


def assess(episode, edge, frames, request):
    current=frames[-1];frame=current['frame_id'];expected=binding(episode,edge,frame)
    if edge not in GUARDED:raise ValueError('safety adapter requires maneuver edge')
    required={'policy','binding','causal_sha256','navigation_evidence_id','coverage_evidence_id',
              'swept_envelope','target_footprint','coverage_radius_m','horizon_s','max_untracked_speed_mps',
              'acceleration_bound_mps2','uncertainty_margin_m','all_actor_inventory_complete',
              'static_drivable_space_verified','priority_verified'}
    if (not isinstance(request,dict) or set(request)!=required or request['policy']!=POLICY
            or request['binding']!=expected or request['causal_sha256']!=current.get('causal_sha256')
            or any(not isinstance(request[k],str) or not request[k].strip() for k in
                   ('navigation_evidence_id','coverage_evidence_id'))):
        raise ValueError('safety request must bind current navigation, sources and maneuver')
    for k in ('all_actor_inventory_complete','static_drivable_space_verified','priority_verified'):
        if type(request[k]) is not bool:raise ValueError('safety prerequisites must be explicit booleans')
    for k in ('coverage_radius_m','horizon_s','max_untracked_speed_mps','acceleration_bound_mps2','uncertainty_margin_m'):
        if not finite(request[k]) or request[k]<=0:raise ValueError('invalid safety envelope limits')
    envelope=request['swept_envelope']
    if not vector(envelope,4) or envelope[0]>=envelope[1] or envelope[2]>=envelope[3]:
        raise ValueError('invalid ego-coordinate swept envelope')
    # Envelope includes the ego footprint throughout the entire maneuver, not
    # just the target lane center or a point at the end of a future trajectory.
    ego=next(a for a in current['actors'] if a['class']=='ego_car');box=aabb(ego)
    if not (envelope[0]<=box[0] and envelope[1]>=box[1] and envelope[2]<=box[2] and envelope[3]>=box[3]):
        raise ValueError('maneuver envelope must contain current ego footprint')
    target=request['target_footprint']
    if (not vector(target,4) or target[0]>=target[1] or target[2]>=target[3]
            or not (envelope[0]<=target[0] and envelope[1]>=target[1] and
                    envelope[2]<=target[2] and envelope[3]>=target[3])):
        raise ValueError('maneuver envelope must contain the navigation target footprint')
    cx,cy=(target[0]+target[1])/2,(target[2]+target[3])/2
    if ((expected['direction']=='LEFT' and cy>=-ego['extent'][1]) or
            (expected['direction']=='RIGHT' and cy<=ego['extent'][1]) or
            (expected['direction']=='FORWARD' and cx<=ego['extent'][0])):
        raise ValueError('target footprint contradicts maneuver direction')
    h=request['horizon_s'];margin=request['uncertainty_margin_m']+0.5*request['acceleration_bound_mps2']*h*h
    radius=max(math.hypot(x,y) for x in envelope[:2] for y in envelope[2:])
    required_radius=radius+(request['max_untracked_speed_mps']+abs(current['meta']['speed']))*h+margin
    reasons=[];blocking=[]
    if any(f.get('source_geometry_issues') for f in frames):reasons.append('invalid_static_source_geometry')
    if any(world_point(f,next(a for a in f['actors'] if a['class']=='ego_car')) is None for f in frames):
        reasons.append('ego_motion_compensation_unavailable')
    if request['coverage_radius_m']<required_radius:reasons.append('inventory_radius_insufficient_for_horizon')
    for k in ('all_actor_inventory_complete','static_drivable_space_verified','priority_verified'):
        if not request[k]:reasons.append(k+'_unconfirmed')
    if len(frames)<2 or any(b['frame_id']!=a['frame_id']+1 for a,b in zip(frames,frames[1:])):
        reasons.append('adjacent_inventory_history_required')
    else:
        if any(disappearances(a,b) for a,b in zip(frames,frames[1:])):reasons.append('near_actor_disappearance')
    for actor in current['actors']:
        # Saved traffic-control boxes are trigger/stop-waypoint proxies, not
        # swept physical obstacles. Priority is an independent prerequisite.
        if actor['class'] in ('ego_car','traffic_light','stop_sign'):continue
        bounds=aabb(actor)
        if actor['class'] in ('car','walker'):
            velocity=actor.get('ego_velocity')
            if not vector(velocity,2):
                reasons.append(f'actor_velocity_unknown:{actor["id"]}');continue
            if math.hypot(*velocity)>request['max_untracked_speed_mps']:
                reasons.append(f'actor_exceeds_motion_bound:{actor["id"]}')
            # The planner envelope is the union of ego footprints in the
            # FIXED current coordinate frame. Actor velocities use that same
            # frame; subtracting ego motion here would move the actor sweep
            # into a different frame and could miss an envelope intersection.
            vx,vy=velocity
            swept=[min(bounds[0],bounds[0]+vx*h)-margin,max(bounds[1],bounds[1]+vx*h)+margin,
                   min(bounds[2],bounds[2]+vy*h)-margin,max(bounds[3],bounds[3]+vy*h)+margin]
        else:
            swept=[bounds[0]-margin,bounds[1]+margin,bounds[2]-margin,bounds[3]+margin]
        if intersects(swept,envelope):blocking.append(actor['id'])
    if blocking:reasons.append('actor_swept_envelope_conflict')
    audit=dict(policy=POLICY,frame_id=frame,source_sha256=digest([f['sources'] for f in frames]),
               request_sha256=digest(request),blocking_actor_ids=blocking,reasons=sorted(set(reasons)),
               inventory_required_radius_m=required_radius,scope='offline_geometry_given_verified_planner_prerequisites')
    # Unknown prerequisites do not become a known negative or positive receipt.
    unknown=[r for r in reasons if r!='actor_swept_envelope_conflict']
    if unknown:return None,audit
    receipt=dict(**expected,source='bev_safety',evidence_id=digest(audit),clear=not blocking,
                 rear_side_coverage_confirmed=True)
    from .route_context import episode_edges
    from .taxonomy import applicable
    validate_clearances(episode,frame,[receipt],{e.key for e in episode_edges(episode)
                        if applicable(e,episode.state,episode.longitudinal)})
    return receipt,audit


class ReplaySafetyAdapter:
    def __init__(self,data_root):self.data_root=data_root

    def __call__(self,episode,observation):
        requests=observation.get('maneuver_safety_requests',[])
        if not isinstance(requests,list):raise ValueError('invalid maneuver safety requests')
        if not requests:return [],[]
        source=observation.get('privileged_source')
        if not isinstance(source,dict) or set(source)!={'scenario','route_id','source_sha256'}:
            raise ValueError('offline safety requires source route and hashes')
        frame=observation['frame_id']
        if frame<1:raise ValueError('offline safety requires prior causal inventory')
        frames=[load_frame(self.data_root,source['scenario'],source['route_id'],f) for f in (frame-1,frame)]
        if source['source_sha256']!=digest([f['sources'] for f in frames]):raise ValueError('offline safety source mismatch')
        if (not observation.get('image_sha256') or
                observation['image_sha256'][-1]!=frames[-1]['sources'][-1]['sha256']):
            raise ValueError('safety inventory must bind the same current RGB as the predictor')
        receipts=[];audits=[];seen=set()
        for request in requests:
            key=request.get('binding',{}).get('edge') if isinstance(request,dict) else None
            if key in seen:raise ValueError('duplicate safety request')
            seen.add(key)
            receipt,audit=assess(episode,key,frames,request);audits.append(audit)
            if receipt is not None:receipts.append(receipt)
        return receipts,audits
