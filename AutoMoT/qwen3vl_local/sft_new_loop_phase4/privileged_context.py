"""Source-bound current state/navigation evidence for proposal production.

This is an input contract, not a producer of navigation, priority, or execution
history. No source event windows or future expert waypoints create this evidence.
"""
from .identity import digest
from .controller import Episode
from .privileged_geometry import vector,finite,aabb,intersects,physical_actor,visible,following,history_valid,vulnerable

POLICY='causal_proposal_context_v1'


def context_index(records):
    index={}
    for record in records:
        required={'policy','scenario','route_id','frame_id','sources_sha256','episode','state_evidence_id','geometry_context','scene_context'}
        if (not isinstance(record,dict) or set(record)!=required or record['policy']!=POLICY
                or type(record['frame_id']) is not int or record['frame_id']<0
                or not isinstance(record['state_evidence_id'],str) or not record['state_evidence_id'].strip()):
            raise ValueError('invalid causal proposal context record')
        ep=Episode(**record['episode']);key=(record['scenario'],record['route_id'],record['frame_id'],ep.instance_id)
        if key in index:raise ValueError('duplicate current instance context')
        index[key]=record
    return index


def bound_context(record,frames):
    f=frames[-1]
    if ((record['scenario'],record['route_id'],record['frame_id'])!=(f['scenario'],f['route_id'],f['frame_id'])
            or record['sources_sha256']!=digest([s['sources'] for s in frames])):
        raise ValueError('context must bind exact causal history and current RGB/source files')
    return Episode(**record['episode'])


def scene_facts(frames,episode,context,params):
    """Geometry propositions with independently grounded identity and priority."""
    if context is None:return {},['current_scene_evidence_required']
    required={'frame_id','episode_sha256','source','evidence_id','corridor_bounds','event_actor_ids',
              'event_identity_established','priority_satisfied','visible_space_resolved','signal_failure_established',
              'event_end_reached','stop_obligation_active'}
    if (not isinstance(context,dict) or set(context)!=required or context['frame_id']!=frames[-1]['frame_id']
            or context['episode_sha256']!=digest(episode.to_dict())
            or context['source'] not in ('planner','causal_scene_review')
            or not isinstance(context['evidence_id'],str) or not context['evidence_id'].strip()
            or not vector(context['corridor_bounds'],4)
            or not isinstance(context['event_actor_ids'],list)
            or any(type(i) is not int for i in context['event_actor_ids'])
            or len(set(context['event_actor_ids']))!=len(context['event_actor_ids'])
            or any(context[k] is not None and type(context[k]) is not bool for k in
                   ('event_identity_established','priority_satisfied','visible_space_resolved','signal_failure_established',
                    'event_end_reached','stop_obligation_active'))):
        raise ValueError('invalid current scene identity/priority evidence')
    corridor=context['corridor_bounds']
    if corridor[0]>=corridor[1] or corridor[2]>=corridor[3]:raise ValueError('invalid scene corridor')
    if context['event_identity_established'] is not True:return {},['event_identity_unestablished']
    if episode.event=='U-E7' and context['signal_failure_established'] is not True:return {},['signal_failure_unestablished']
    current=frames[-1];actors={a['id']:a for a in current['actors'] if physical_actor(a)}
    if any(i not in actors for i in context['event_actor_ids']):return {},['event_actor_absent_not_resolved']
    established,lead_id,_=following(frames,params)
    follow_allowed=episode.event in ('U-E1','U-E3') and established is True
    from .event_scope import lead_releasing
    releasing_ids={i for i in context['event_actor_ids'] if episode.event in ('U-E1','U-E3')
                   and lead_releasing(frames,i,params) is True}
    blocked=False;unknown=False;stop=False
    for a in actors.values():
        if not intersects(aabb(a),corridor):continue
        if visible(a,current,params) is not True:unknown=True;continue
        if (follow_allowed and a['id']==lead_id) or a['id'] in releasing_ids:continue
        blocked=True
        ego=next(v for v in current['actors'] if v['class']=='ego_car')
        gap=aabb(a)[0]-ego['extent'][0];speed=max(0,current['meta']['speed'])
        # Candidate stopping envelope: reaction + comfortable braking distance.
        # A visibly occupied pedestrian corridor can require maintaining a wait
        # before the participant is within two metres of the bumper.
        stop_distance=max(2.,speed+speed*speed/6.)
        if vulnerable(a):stop_distance=max(stop_distance,15.)
        stop |= gap<=stop_distance
    clear=False if blocked else True if context['visible_space_resolved'] is True and not unknown else None
    priority=context['priority_satisfied'];obligation=context['stop_obligation_active']
    # A current boundary/actor review is necessary for release. Empty bboxes,
    # source scenario names, or ordinary following alone do not establish it.
    identity_resolved=context['event_end_reached']
    release=(False if clear is False or priority is False or obligation is True else
             True if clear is True and priority is True and obligation is False else None)
    stable=None
    if history_valid(frames,params['history_frames']) and clear is True:
        speeds=[f['meta']['speed'] for f in frames]
        ego_ids={(f['meta'].get('road_id'),f['meta'].get('lane_id')) for f in frames}
        if len(ego_ids)==1 and all(v is not None for pair in ego_ids for v in pair):
            stable=min(speeds)>.5 and max(speeds)-min(speeds)<=params['max_relative_speed_mps']
    return dict(corridor_clear=clear,priority_satisfied=priority,release_ready=release,
                restriction_present=True if blocked or follow_allowed or obligation is True else False if release is True else None,
                stationary_wait_required=True if stop or obligation is True else False if release is True else None,
                event_resolved=identity_resolved,stable_progress_established=stable,
                restricted_progress_established=established),['scene_bound_geometry_proposal_requires_calibration']


def calibrate_references(references,contexts,root,params):
    """Fresh per-frame RGB references, including visible-scope maneuver edges.

    References never feed the proposal. A context-only prediction is compared
    afterwards, so copying the desired target into the input cannot calibrate it.
    """
    from collections import Counter
    from .privileged_geometry import load_frame,condition_proposal
    from .privileged_producer import producer_identity
    from .maneuver_safety import GUARDED
    from .visible_scope import POLICY as VISIBLE_POLICY
    from .dataset import source_route,groups,holdout_reservations,split_for
    idx=context_index(contexts);counts=Counter();results=[];seen=set();strata={}
    for r in references:
        required={'scenario','route_id','frame_id','episode','edge','target','scope','rgb_history_sha256','reviewer','observation','observed_until'}
        if (not isinstance(r,dict) or set(r)!=required or r['target'] not in ('YES','NO')
                or type(r['frame_id']) is not int or r['frame_id']!=r['observed_until']
                or any(not isinstance(r[k],str) or not r[k].strip() for k in ('reviewer','observation'))):
            raise ValueError('invalid independent current RGB reference')
        ep=Episode(**r['episode']);key=(r['scenario'],r['route_id'],r['frame_id'],ep.instance_id)
        identity=(*key,r['edge'])
        if identity in seen:raise ValueError('duplicate calibration reference')
        seen.add(identity)
        expected_scope=VISIBLE_POLICY if r['edge'] in GUARDED else 'readiness'
        if r['scope']!=expected_scope:raise ValueError('reference scope mismatch')
        group=source_route(r)
        if split_for(group,groups(),reservations=holdout_reservations())!='train':
            raise ValueError('development calibration may not consume frozen validation/test references')
        frames=[load_frame(root,r['scenario'],r['route_id'],f) for f in range(r['frame_id']-params['history_frames']+1,r['frame_id']+1)]
        if r['rgb_history_sha256']!=[f['sources'][-1]['sha256'] for f in frames]:raise ValueError('calibration RGB identity mismatch')
        record=idx.get(key)
        if record:
            bound=bound_context(record,frames)
            if bound.to_dict()!=ep.to_dict():raise ValueError('calibration state/context mismatch')
        proposal=condition_proposal(frames,ep,r['edge'],params,
                                   geometry_context=record['geometry_context'] if record else None,
                                   scene_context=record['scene_context'] if record else None)
        if record is None:
            proposal['target']='UNKNOWN';proposal['reasons'].append('current_state_context_required')
        outcome='abstained' if proposal['target']=='UNKNOWN' else 'correct' if proposal['target']==r['target'] else 'wrong'
        counts['reference']+=1;counts[outcome]+=1
        cell=strata.setdefault(f'{ep.event}/{r["edge"]}',Counter());cell['reference']+=1;cell[outcome]+=1
        results.append(dict(identity=identity,reference=r['target'],proposal=proposal['target'],reasons=proposal['reasons']))
    return dict(producer=producer_identity(params),scope='development_only_new_rubric_references',summary=dict(counts),
                strata={k:dict(v) for k,v in strata.items()},results=results,supervision_approved=False,
                contexts_sha256=digest(contexts),references_sha256=digest(references))
