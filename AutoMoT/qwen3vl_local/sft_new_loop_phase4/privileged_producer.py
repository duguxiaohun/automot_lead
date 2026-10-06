"""Stream causal geometry proposals and compare them with independent RGB labels.

No algorithmic output is declared rgb_review. This separate proposal artifact
cannot enter dataset.build/condition_builder as supervision. The CLI processes
whole routes, respects frozen physical splits, and reports missing event/route
context rather than treating unlabelled frames as negative examples.
"""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'
import argparse
import lzma
import json
from collections import Counter,deque
from pathlib import Path
from .identity import ROOT,digest,file_sha,write_json
from .controller import Episode
from .taxonomy import EVENTS
from .privileged_geometry import load_frame,DEFAULTS,condition_proposal,following,disappearances,finite,aabb,intersects,vulnerable,obstacle,physical_actor

POLICY='phase4_privileged_proposals_v5'


def producer_identity(params=DEFAULTS):
    if set(params)!=set(DEFAULTS) or any(not finite(v) or v<=0 for v in params.values()):
        raise ValueError('invalid proposal parameters')
    if type(params['history_frames']) is not int or params['history_frames']<2:
        raise ValueError('proposal history must contain at least two observations')
    return dict(policy=POLICY,parameters=dict(params),source_sha256={name:file_sha(ROOT/name) for name in
                ('visual_review.py','event_scope.py','review_queue.py','automatic_context.py','privileged_geometry.py','privileged_visibility.py','privileged_context.py','privileged_producer.py','route_context.py','route_calibration.py')},
                calibrated_for_supervision=False)


def candidates(frames, tracks):
    """Tentative actor-scoped identities; ordinary following is not automatically UE1."""
    from .automatic_context import navigation,route_intersection,junction_candidates
    frame=frames[-1];m=frame['meta'];width=m.get('ego_lane_width');active=[];nav=navigation(frame)
    if not finite(width) or width<=0:return active
    prev={a['id']:a for a in frames[-2]['actors']} if len(frames)>1 else {}
    for actor in frame['actors']:
        event=None;reason=None;x,y,_=actor['position']
        if not physical_actor(actor) or not 0<x<50:continue
        if vulnerable(actor) and abs(y)<2*width:event,reason='U-E4','near_corridor_vulnerable_participant'
        elif obstacle(actor) and intersects(aabb(actor),[0,50,-width/2,width/2]):
            event,reason='U-E2','static_obstruction_intersects_corridor_requires_review'
        elif actor['class']=='car' and not vulnerable(actor):
            if (abs(actor['yaw'])>2.5 and (route_intersection(actor,nav,margin=.5) is True if nav.get('points') else
                    m.get('road_id') is not None and m.get('lane_id') is not None
                    and actor.get('road_id')==m['road_id'] and actor.get('lane_id')==m['lane_id']
                    and intersects(aabb(actor),[0,50,-width/2,width/2]))):
                event,reason='U-E5','opposing_clearance_within_half_metre_of_navigation_requires_review'
            elif actor['id'] in (m.get('scenario_obstacles_ids') or []) and actor.get('speed',999)<0.5:
                event,reason='U-E2','tracked_stationary_scenario_obstacle_requires_instance_review'
            elif actor['id'] in prev and abs(actor['yaw'])<0.5 and abs(y)<width:
                p=prev[actor['id']]
                if abs(p['position'][1])>abs(y)+0.15:event,reason='U-E3','lateral_approach_requires_intrusion_review'
                elif actor.get('road_id')==m.get('road_id') and actor.get('lane_id')==m.get('lane_id'):
                    event,reason='U-E1','following_vehicle_does_not_alone_establish_braking_event'
        if event:
            key=(event,actor['id'])
            if key not in tracks:tracks[key]=frame['frame_id']
            active.append(dict(event=event,actor_id=actor['id'],start_frame=tracks[key],reason=reason,
                               instance_id=f"{frame['scenario']}/{frame['route_id']}/{event}/{actor['id']}/{tracks[key]}"))
    active_keys={(a['event'],a['actor_id']) for a in active}
    for (event,ident),start in list(tracks.items()):
        actor=next((a for a in frame['actors'] if a['id']==ident),None)
        if ((event,ident) not in active_keys and actor is not None and -15<actor['position'][0]<=0
                and any(any(a['id']==ident and a['position'][0]>0 for a in f['actors']) for f in frames[:-1])):
            active.append(dict(event=event,actor_id=ident,start_frame=start,reason='tracked_passage_boundary_candidate',
                          instance_id=f"{frame['scenario']}/{frame['route_id']}/{event}/{ident}/{start}"))
    for candidate in junction_candidates(frame):
        key=(candidate['event'],candidate['actor_id'])
        if key not in tracks:tracks[key]=frame['frame_id']
        active.append(dict(**candidate,start_frame=tracks[key],
                      instance_id=f"{frame['scenario']}/{frame['route_id']}/{key[0]}/{key[1]}/{tracks[key]}"))
    current_keys={(a['event'],a['actor_id']) for a in active}
    for key in list(tracks):
        if key not in current_keys:del tracks[key]  # Reappearance starts a new candidate, never silently bridges gaps.
    return active


def route_proposals(root, route, params=DEFAULTS, *, contexts=None):
    from .privileged_context import bound_context
    from .route_context import episode_edges
    from .taxonomy import applicable
    run=Path(root)/route['scenario']/route['route_id']
    frames=route.get('rgb_frames')
    if frames is None:frames=sorted(int(p.stem) for p in (run/'rgb').glob('*.jpg') if p.stem.isdigit())
    history=deque(maxlen=params['history_frames']);tracks={}
    for f in frames:
        try:current=load_frame(root,route['scenario'],route['route_id'],f)
        except (FileNotFoundError,ValueError,EOFError,lzma.LZMAError) as ex:
            history.clear();tracks.clear()
            yield dict(kind='frame_disposition',frame_id=f,disposition='unusable_source',reason=str(ex));continue
        if history and f!=history[-1]['frame_id']+1:history.clear();tracks.clear()
        history.append(current);context=list(history)
        anomalies=disappearances(context[-2],current,params) if len(context)>1 else []
        anomalous_ids={a['actor_id'] for a in anomalies}
        for key in list(tracks):
            if key[1] in anomalous_ids:del tracks[key]
        instances=candidates(context,tracks);proposals=[]
        for instance in instances:
            # Candidate state pairs, explicitly not a reconstructed execution trace.
            states=[('PROCEED','RECOVER','recover_follow')] if instance['event']!='U-E2' else [('WAIT','HOLD','depart'),('PASS','RECOVER','return')]
            if instance['event']!='U-E2':states += [('YIELD','HOLD','proceed')]
            for state,lon,edge in states:
                ep=Episode(instance['event'],instance['instance_id'],state=state,longitudinal=lon)
                result=condition_proposal(context,ep,edge,params)
                from .automatic_context import condition_facts
                automatic_facts=condition_facts(context,ep,instance,params)
                if not result['anomalies'] or not any(a.get('rgb_relevant',True) for a in result['anomalies']):
                    result['facts'].update({k:v for k,v in automatic_facts.items() if k in result['facts']})
                values=result['facts'].values()
                result['geometric_target']='NO' if False in values else 'YES' if values and all(v is True for v in values) else 'UNKNOWN'
                from .event_scope import release_suggestion
                result['release_suggestion']=release_suggestion(context,ep.event,instance['actor_id'],params) if edge=='proceed' else None
                result['target']='UNKNOWN'
                result['reasons'].append('current_instance_state_not_established')
                proposals.append(dict(instance=instance,episode=dict(event=ep.event,instance_id=ep.instance_id,state=state,longitudinal=lon),
                    edge=edge,proposal=result,training_target='UNKNOWN',event_identity_confirmed=False,state_basis='hypothetical_state_pair'))
        for record in (contexts or {}).get((route['scenario'],route['route_id'],f),[]):
            ep=bound_context(record,context)
            for edge in episode_edges(ep):
                if not applicable(edge,ep.state,ep.longitudinal):continue
                result=condition_proposal(context,ep,edge.key,params,geometry_context=record['geometry_context'],
                                          scene_context=record['scene_context'])
                proposals.append(dict(instance=dict(event=ep.event,instance_id=ep.instance_id),episode=record['episode'],
                    edge=edge.key,proposal=result,training_target='UNKNOWN',event_identity_confirmed=False,
                    state_basis='source_bound_current_state',state_evidence_id=record['state_evidence_id']))
        from .automatic_context import infer
        automatic=infer(context,instances)
        # Scenario names only retrieve identity-review packets. They do not
        # establish events or feed any geometry/priority condition as truth.
        hints=[]
        def light_states(frame):
            return sorted((str(a['id']),str(a.get('state'))) for a in frame['actors']
                          if a['class']=='traffic_light' and a.get('affects_ego') is True)
        if (route['scenario']=='CrossJunctionDefectTrafficLight' and f>=10
                and (f==10 or len(context)>1 and light_states(context[-2])!=light_states(current))):
            ident=f"identity-review/{route['scenario']}/{route['route_id']}/signal"
            hints=[dict(instance=dict(event='U-E7',instance_id=ident),
                episode=dict(event='U-E7',instance_id=ident,state='YIELD',longitudinal='HOLD'),
                edge='proceed',proposal=dict(geometric_target='UNKNOWN'),
                reason='source_scenario_retrieval_only_verify_applicable_signal_failure',
                event_identity_confirmed=False,training_target='UNKNOWN')]
        yield dict(kind='frame_disposition',frame_id=f,disposition='pending_calibration',automatic_context=automatic,
                   source_geometry_issues=current.get('source_geometry_issues',[]),
                   causal_sha256=current['causal_sha256'],sources=[s for c in context for s in c['sources']],
                   instances=instances,proposals=proposals,identity_review_hints=hints,anomaly_candidates=anomalies,
                   unavailable_context=['navigation_bound_RE2_RE3','junction_priority_UE6_RE5','established_signal_failure_UE7'],
                   scope='actor identities and condition proposals; no approved labels')


def produce(pool,root,output,*,splits=('train',),route_keys=None,params=DEFAULTS,contexts=()):
    from .candidate_pool import validate
    validate(pool);output=Path(output)
    if output.exists():raise FileExistsError('use a new proposal output')
    if not set(splits)<= {'train','val','test'} or not splits:raise ValueError('invalid proposal splits')
    from .privileged_context import context_index
    context_records=context_index(contexts);by_frame={}
    for key,record in context_records.items():by_frame.setdefault(key[:3],[]).append(record)
    identity=producer_identity(params);selected=[r for r in pool['routes'] if r['split'] in splits
                 and (route_keys is None or (r['scenario'],r['route_id']) in route_keys)]
    if route_keys is not None and {(r['scenario'],r['route_id']) for r in selected}!=set(route_keys):
        raise ValueError('requested route missing or outside selected frozen splits')
    if not selected:raise ValueError('empty proposal selection')
    selected_frames={(r['scenario'],r['route_id']):set(r['rgb_frames']) for r in selected}
    if any(k[:2] not in selected_frames or k[2] not in selected_frames[k[:2]] for k in context_records):
        raise ValueError('context outside selected frozen route/split/frames')
    counts=Counter();events=Counter();strata=Counter();unknown_facts=Counter();output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix(output.suffix+'.pending')
    # Exclusive creation also protects another unfinished producer invocation.
    with pending.open('x') as f:
        f.write(json.dumps(dict(policy=POLICY,producer=identity,candidate_pool_sha256=pool['sha256'],supervision_approved=False,context_sha256=digest(contexts)))+'\n')
        for route in selected:
            for item in route_proposals(root,route,params,contexts=by_frame):
                item.update(scenario=route['scenario'],route_id=route['route_id'],physical_group=route['physical_group'],split=route['split'])
                counts[item['disposition']]+=1
                counts['anomaly_candidates']+=len(item.get('anomaly_candidates',[]))
                counts['rgb_relevant_anomalies']+=sum(a.get('rgb_relevant',True) for a in item.get('anomaly_candidates',[]))
                counts['inventory_only_anomalies']+=sum(not a.get('rgb_relevant',True) for a in item.get('anomaly_candidates',[]))
                counts['frames_with_source_geometry_issues']+=bool(item.get('source_geometry_issues'))
                for p in item.get('proposals',[]):
                    event=p['instance']['event'];events[event]+=1
                    strata['/'.join((event,p['edge'],p['state_basis'],p['proposal']['target']))]+=1
                    for fact,value in p['proposal']['facts'].items():
                        if value is None:unknown_facts[f'{event}/{p["edge"]}/{fact}']+=1
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
    pending.replace(output)
    return dict(policy=POLICY,producer=identity,routes=len(selected),frames=sum(counts[k] for k in ('pending_calibration','unusable_source')),
                dispositions=dict(counts),candidate_questions_by_event=dict(events),
                proposal_strata=dict(strata),unknown_conditions=dict(unknown_facts),
                candidate_pool_sha256=pool['sha256'],contexts_sha256=digest(contexts),
                missing_candidate_events=sorted(set(EVENTS)-set(events)),approved_questions=0,
                full_causal_production=False,formal_data_ready=False,output_sha256=file_sha(output))


def calibrate(annotations,root,*,params=DEFAULTS,scope='train'):
    """Report abstentions and disagreements too; test labels never tune thresholds.

    This function does not optimize parameters or approve a producer. Existing
    audited development RGB is calibration-only; independent accuracy still
    requires a prospectively frozen manual sample.
    """
    from .dataset import source_route,groups,split_for
    from .route_calibration import reviewed_band_labels,label_condition,interval_facts
    from .maneuver_safety import GUARDED
    results=[];cache={};seen=set();strata={};excluded_scope=0
    for ann in annotations:
        group=source_route(ann);split=split_for(group,groups())
        if scope!='all' and split!=scope:continue
        ep=Episode(**ann['episode']);edge=ann['edge']
        band=ann.get('transition_band')
        if band:labels=reviewed_band_labels(ep,edge,band,context_valid=ann.get('context_valid'))
        else:
            labels={f:(label_condition(ep,edge,f,interval_facts(ann.get('facts',{}),ann['start'],ann['end'],ann['evidence_id']),
                       context_valid=ann.get('context_valid')),ann.get('slice','readiness')) for f in range(ann['start'],ann['end']+1)}
        for frame,(target,slice_) in labels.items():
            if slice_!='readiness' or target not in ('YES','NO'):continue
            # Old full-gap references are not ground truth for the narrower
            # visible-gap task. Even abstention coverage must use a valid rubric.
            if edge in GUARDED:
                excluded_scope+=1
                continue
            ident=digest([ann['scenario'],ann['route_id'],frame,ann['episode'],edge])
            if ident in seen:continue
            seen.add(ident)
            context=[];reason=None
            for f in range(frame-params['history_frames']+1,frame+1):
                key=(ann['scenario'],ann['route_id'],f)
                try:
                    if key not in cache:cache[key]=load_frame(root,*key)
                    context.append(cache[key])
                except (FileNotFoundError,ValueError,EOFError,lzma.LZMAError) as ex:reason=str(ex);break
            if reason:proposal=dict(target='UNKNOWN',reasons=[reason])
            else:proposal=condition_proposal(context,ep,edge,params)
            # Exact current image binding of the human reference is mandatory.
            if not context or context[-1]['frame_id']!=frame:
                pass
            elif ann['frame_sha256'].get(str(frame))!=context[-1]['sources'][-1]['sha256']:
                raise ValueError('calibration reference RGB identity mismatch')
            key=f'{split}/{ep.event}/{edge}'
            cell=strata.setdefault(key,Counter());cell['reference']+=1
            cell['abstained' if proposal['target']=='UNKNOWN' else 'correct' if proposal['target']==target else 'wrong']+=1
            results.append(dict(id=ident,split=split,physical_group=group,event=ep.event,edge=edge,frame_id=frame,
                                reference=target,proposal=proposal['target'],reasons=proposal['reasons']))
    summary=Counter()
    for cell in strata.values():summary.update(cell)
    attempted=summary['correct']+summary['wrong']
    return dict(policy=POLICY,producer=producer_identity(params),scope='development calibration, not independent accuracy certification',
                reference_annotations_sha256=digest(annotations),summary=dict(summary),
                excluded_old_maneuver_scope=excluded_scope,
                conditional_error_rate=summary['wrong']/attempted if attempted else None,
                coverage=attempted/summary['reference'] if summary['reference'] else 0,
                strata={k:dict(v) for k,v in strata.items()},results=results,supervision_approved=False,
                independent_manual_evaluation_required=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate-pool',type=Path)
    p.add_argument('--route-list',type=Path,help='JSON list of scenario/route_id objects')
    p.add_argument('--splits',nargs='+',choices=('train','val','test'),default=['train'])
    p.add_argument('--contexts',type=Path,help='JSON list of current state/navigation records bound to causal sources')
    p.add_argument('--calibrate-references',type=Path,help='fresh source-bound RGB reference list; includes visible-scope lateral edges')
    p.add_argument('--calibrate-annotations',type=Path)
    a=p.parse_args()
    contexts=[] if a.contexts is None else json.loads(a.contexts.read_text())
    if a.output.exists():raise FileExistsError('use a new output')
    if a.calibrate_annotations and a.calibrate_references:p.error('choose one reference format')
    if a.contexts and a.calibrate_annotations:p.error('use --calibrate-references with contexts; legacy annotation calibration has no new context')
    if a.calibrate_references:
        from .privileged_context import calibrate_references
        report=calibrate_references(json.loads(a.calibrate_references.read_text()),contexts,a.data_root,DEFAULTS)
        write_json(a.output,report)
    elif a.calibrate_annotations:
        report=calibrate(json.loads(a.calibrate_annotations.read_text()),a.data_root)
        write_json(a.output,report)
    else:
        if not a.candidate_pool:p.error('--candidate-pool required for production')
        routes=None if a.route_list is None else {(r['scenario'],r['route_id']) for r in json.loads(a.route_list.read_text())}
        report=produce(json.loads(a.candidate_pool.read_text()),a.data_root,a.output,splits=a.splits,route_keys=routes,contexts=contexts)
        write_json(a.output.with_suffix('.report.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','strata')},ensure_ascii=False))

if __name__=='__main__':main()
