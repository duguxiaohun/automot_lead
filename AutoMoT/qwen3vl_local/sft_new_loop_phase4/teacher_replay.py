"""Causal teacher replay with resumable, route-local immutable artifacts.

Run with --candidate-pool ... --output-dir ... [--workers 4]. The default
selection is every frozen split. An explicit route list is a partial run.
The accounting ledger records abstentions too; it is not a label approval.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import json
import lzma
from pathlib import Path
from .identity import ROOT, digest, file_sha, write_json
from .controller import Episode, STARTS
from .observation import observation_contract
from . import teacher_rules as rules
from . import privileged_geometry as g

POLICY='causal_episode_teacher_production_v6'
EP_FIELDS=('event','instance_id','branch','state','longitudinal','direction','return_direction',
           'return_required','target_corridor','return_corridor','route_segments','segment_index','route_context')
DISPOSITIONS={'questions','abstained','irrelevant'}


def route_records(root, route):
    history=deque(maxlen=7);active={};seen=set();last_targets={};completed={};motion_status={}
    from .teacher_controls import StopDutyTracker
    controls=StopDutyTracker()
    teacher_sha=digest(rules.identity())
    for number in route['rgb_frames']:
        base=dict(kind='frame_disposition',scenario=route['scenario'],route_id=route['route_id'],
                  physical_group=route['physical_group'],split=route['split'],frame_id=number,
                  questions=[],traces=[],reasons=[],instance_ends=[])
        try:current=g.load_frame(root,route['scenario'],route['route_id'],number)
        except (OSError,ValueError,EOFError,lzma.LZMAError) as ex:
            controls=StopDutyTracker()
            base['instance_ends']=end_instances(active,'unusable_source');history.clear();active.clear();completed.clear();seen.clear()
            yield dict(base,disposition='abstained',reasons=['unusable_source:'+str(ex)]);continue
        if history and number!=history[-1]['frame_id']+1:
            base['instance_ends']=end_instances(active,'observation_gap');history.clear();active.clear();completed.clear();seen.clear()
        current['stop_duties']=controls.observe(current)
        history.append(current);frames=list(history)
        base['sources']=current['sources'];base['causal_sha256']=current['causal_sha256']
        if number==0 or current.get('source_geometry_issues'):
            base['instance_ends']+=end_instances(active,'invalid_geometry_or_initialization');history.clear();active.clear();completed.clear();seen.clear()
            yield dict(base,disposition='abstained',reasons=['initialization_frame' if number==0 else 'invalid_geometry']);continue
        if not g.history_valid(frames,7) or frames[0]['frame_id']<4:
            yield dict(base,disposition='abstained',reasons=['incomplete_causal_history']);continue
        anomalies=[x for a,b in zip(frames,frames[1:]) for x in g.disappearances(a,b)]
        base['anomalies']=anomalies
        seeds=rules.seeds(frames)
        rearm_completed(completed,seen,seeds,current)
        for seed in seeds:
            key=seed['event'],seed['actor_id']
            if key not in active and key not in seen and not rules.scoped_anomalies(frames,seed['actor_id']):
                candidate=Episode(seed['event'],'candidate',branch=seed.get('branch','default'))
                evidence,_=rules.facts(frames,candidate,seed['actor_id'])
                if evidence.get('event_restriction_observed') is not True:
                    base['reasons'].append('seed_without_observed_event_restriction');continue
                seed=dict(seed,restriction_frame=number)
                ident=f"teacher/{route['scenario']}/{route['route_id']}/{seed['event']}/{seed['actor_id']}/{number}"
                active[key]=(Episode(seed['event'],ident,branch=seed.get('branch','default')),seed);seen.add(key)
        for key,(ep,seed) in list(active.items()):
            # A completion established in the preceding observation stays valid;
            # today's deletion still appears in the route anomaly ledger.
            if ep.finished:
                completed[key]=dict(clear_observations=0,last_frame=number)
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason='finished',completed=True))
                del active[key];continue
            if rules.scoped_anomalies(frames,seed['actor_id']):
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason='instance_rgb_discontinuity',completed=False))
                base['reasons'].append('instance_rgb_discontinuity')
                del active[key];seen.discard(key);completed.pop(key,None)
                continue
            if rules.actor(current,seed['actor_id']) is None:
                base['reasons'].append('instance_identity_lost_not_complete')
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason='identity_lost',completed=False));del active[key];seen.discard(key);continue
            execution=rules.observed_resumption(frames,ep,motion_status.setdefault(ep.instance_id,{}))
            if execution:
                base['instance_ends'].append(dict(instance_id=ep.instance_id,
                    reason='observed_resumption_censors_stale_wait',completed=False,evidence=execution))
                completed[key]=dict(clear_observations=0,last_frame=number)
                del active[key];continue
            retired=rules.lead_retirement(frames,ep,seed['actor_id'])
            if retired:
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason=retired,completed=False))
                completed[key]=dict(clear_observations=0,last_frame=number)
                del active[key];continue
            if ep.event=='U-E4' and ep.branch=='default' and rules.parallel_cyclist(frames,seed['actor_id']):
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason='cyclist_follow_branch_requires_new_instance',completed=False))
                del active[key];seen.discard(key);continue
            edges=ep.questions()
            if not edges:
                base['reasons'].append('instance_finished' if ep.finished else 'instance_requires_recheck')
                if ep.finished:completed[key]=dict(clear_observations=0,last_frame=number)
                base['instance_ends'].append(dict(instance_id=ep.instance_id,reason='finished' if ep.finished else 'requires_recheck',completed=ep.finished))
                del active[key];continue
            episode={k:getattr(ep,k) for k in EP_FIELDS}
            values,reasons=rules.facts(frames,ep,seed['actor_id'])
            answers={e.key:rules.answer(e.criteria,values) for e in edges}
            moving=rules.execution_motion(frames)
            for mode in (2,4):
                input_frames=[number+x for x in observation_contract(mode)['frame_offsets']]
                sources=[s for f in frames if f['frame_id'] in input_frames for s in f['sources'] if s['kind']=='rgb']
                for edge in edges:
                    target=answers[edge.key];phase='readiness';why=list(reasons)
                    # Current condition evidence remains readiness after ego motion.
                    # Motion only acknowledges execution; it never manufactures a label.
                    if target=='UNKNOWN' and not why:why=['required_condition_unresolved']
                    question=dict(teacher_sha256=teacher_sha,scenario=route['scenario'],route_id=route['route_id'],physical_group=route['physical_group'],
                        split=route['split'],frame_id=number,episode=episode,edge=edge.key,rgb_mode=mode,
                        history_frames=input_frames,input_sources=sources,input_sha256=digest(sources),
                        causal_sources=[s for f in frames for s in f['sources']],
                        facts={c:values.get(c) for c in edge.criteria},rule_target=target,
                        control_evidence=current.get('stop_duties',{}),
                        ego_motion=rules.motion_stratum(frames),ego_speed=current['meta']['speed'],
                        state_rule_target=answers[edge.key],slice=phase,
                        rule_class=rules.rule_class(ep.event,edge.key,mode,phase),reasons=why,instance=seed,
                        near_transition=last_targets.get((ep.instance_id,mode,edge.key)) not in (None,target))
                    last_targets[(ep.instance_id,mode,edge.key)]=target
                    question['question_id']=digest(question)
                    base['questions'].append(question)
            ep.advance(number,answers)
            receipt=None
            decision=ep.history[-1] if ep.history and ep.history[-1]['frame_id']==number else {}
            pending=decision.get('pending_start',[]) if not decision.get('committed') else []
            if pending and moving and not ep.needs_recheck:
                receipt=dict(instance_id=ep.instance_id,source='causal_tracker',evidence_id=digest(current['sources']),
                    successor_observed=True,decision_frame=number,observed_frame=number,started_frame=number,
                    edges=pending)
                ep.confirm_execution(receipt)
            base['traces'].append(dict(instance_id=ep.instance_id,before=episode,answers=answers,
                after={k:getattr(ep,k) for k in EP_FIELDS},accepted=decision.get('accepted',[]),
                execution_receipt=receipt,wait_reason=ep.wait_reason,finished=ep.finished,needs_recheck=ep.needs_recheck))
        if not base['questions']:
            # No supported instance does not prove the frame irrelevant to the
            # eight event classes not yet implemented by this teacher.
            base['reasons'].append('no_established_UE1_UE4_instance_other_events_unsupported')
        base['disposition']='questions' if base['questions'] else 'abstained'
        yield base


def end_instances(active,reason):
    return [dict(instance_id=ep.instance_id,reason=reason,completed=False) for ep,seed in active.values()]


def rearm_completed(completed,seen,seeds,current):
    """A completed actor may cause another event after a visible clear interval.

    Gaps, invisible/disappeared actors and recheck failures never rearm a seed.
    """
    conflicts={(s['event'],s['actor_id']) for s in seeds}
    for key,status in list(completed.items()):
        actor=rules.actor(current,key[1]);frame=current['frame_id']
        clear=(frame==status['last_frame']+1 and key not in conflicts
               and actor is not None and g.visible(actor,current) is True)
        status['clear_observations']=status['clear_observations']+1 if clear else 0
        status['last_frame']=frame
        if status['clear_observations']>=3:
            seen.discard(key);del completed[key]


def _write_route(args):
    root,route,output,header=args
    output=Path(output)
    if output.exists():
        # Resume only after validating all frame identities and the frozen code.
        return inspect_route(output,route,header)
    pending=output.with_suffix('.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(dict(kind='header',**header))+'\n')
        for row in route_records(root,route):stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    pending.replace(output)
    return inspect_route(output,route,header)


def inspect_route(path,route,header):
    counts=Counter();targets=Counter();seen=[];question_ids=set()
    with Path(path).open() as stream:
        if json.loads(next(stream))!=dict(kind='header',**header):raise ValueError('stale teacher route artifact')
        for line in stream:
            row=json.loads(line);frame=row['frame_id'];seen.append(frame)
            if (row.get('kind')!='frame_disposition' or row['disposition'] not in DISPOSITIONS
                    or any(row.get(k)!=route[k] for k in ('scenario','route_id','physical_group','split'))):
                raise ValueError('foreign/invalid frame disposition')
            if row['disposition']!='questions' and not row.get('reasons'):raise ValueError('abstention/irrelevance needs a reason')
            if bool(row.get('questions'))!=(row['disposition']=='questions'):raise ValueError('question disposition mismatch')
            counts[row['disposition']]+=1
            for q in row.get('questions',[]):
                validate_question(q,expected_teacher_sha=digest(header['teacher']))
                if any(q[k]!=row[k] for k in ('scenario','route_id','physical_group','split','frame_id')):
                    raise ValueError('question outside route/frame')
                if q['question_id'] in question_ids:raise ValueError('duplicate teacher question')
                question_ids.add(q['question_id']);targets[q['rule_class']+'/'+q['rule_target']]+=1
    if seen!=route['rgb_frames']:raise ValueError('missing/duplicate/out-of-order frame dispositions')
    return dict(scenario=route['scenario'],route_id=route['route_id'],physical_group=route['physical_group'],
                split=route['split'],frames=len(seen),dispositions=dict(counts),targets=dict(targets),sha256=file_sha(path),
                artifact=Path(path).name)


def validate_question(q,*,expected_teacher_sha=None):
    from .route_context import episode_edge
    from .taxonomy import applicable
    if q.get('question_id')!=digest({k:v for k,v in q.items() if k!='question_id'}):raise ValueError('teacher question hash mismatch')
    if q.get('teacher_sha256')!=(expected_teacher_sha or digest(rules.identity())):raise ValueError('stale teacher question source identity')
    speed=q.get('ego_speed')
    if (not g.finite(speed) or q.get('ego_motion')!=('stationary' if abs(speed)<.5 else 'moving')):
        raise ValueError('invalid teacher ego motion stratum')
    mode=q['rgb_mode'];spec=observation_contract(mode);f=q['frame_id'];phase=q['slice']
    if type(f) is not int or q['history_frames']!=[f+x for x in spec['frame_offsets']] or q['history_frames'][0]<4:
        raise ValueError('invalid teacher causal input')
    ep=Episode(**q['episode']);edge=episode_edge(ep,q['edge'])
    if not applicable(edge,ep.state,ep.longitudinal):raise ValueError('teacher asked an inapplicable edge')
    if (q['rule_class']!=rules.rule_class(ep.event,edge.key,mode,phase) or ep.event not in ('U-E1','U-E4')
            or q['rule_target'] not in ('YES','NO','UNKNOWN') or phase not in ('readiness','catchup')):
        raise ValueError('invalid teacher rule class/answer')
    if set(q['facts'])!=set(edge.criteria):raise ValueError('teacher condition set mismatch')
    if q['state_rule_target']!=rules.answer(edge.criteria,q['facts']):raise ValueError('teacher state answer disagrees with conditions')
    if q['rule_target']!=q['state_rule_target'] and q['rule_target']!='UNKNOWN':raise ValueError('teacher invented a target')
    if phase=='catchup' and q['rule_target']!='UNKNOWN':raise ValueError('catchup requires an implemented visual successor teacher')
    if q['rule_target']=='UNKNOWN' and not q['reasons']:raise ValueError('missing abstention reason')
    expected=[dict(path=f"{q['scenario']}/{q['route_id']}/rgb/{n:04d}.jpg",frame_id=n,kind='rgb',sha256=s['sha256'])
              for n,s in zip(q['history_frames'],q['input_sources'])]
    if len(expected)!=mode or q['input_sources']!=expected or q['input_sha256']!=digest(expected):raise ValueError('teacher RGB input mismatch')
    causal=q['causal_sources']
    paths=[(s['frame_id'],s['kind']) for s in causal]
    if paths!=[(n,k) for n in range(f-6,f+1) for k in ('metas','bboxes','rgb')]:raise ValueError('missing/future causal teacher evidence')
    for s in causal:
        ext='jpg' if s['kind']=='rgb' else 'pkl'
        if not isinstance(s.get('sha256'),str) or len(s['sha256'])!=64 or any(c not in '0123456789abcdef' for c in s['sha256']):raise ValueError('invalid teacher source SHA')
        if s['path']!=f"{q['scenario']}/{q['route_id']}/{s['kind']}/{s['frame_id']:04d}.{ext}":raise ValueError('causal source path mismatch')
    if any(s not in causal for s in expected):raise ValueError('model input not in causal evidence')
    from .teacher_controls import PARAMS as controls_params
    for ident,record in q.get('control_evidence',{}).items():
        obs=record.get('observations',[])
        if str(record.get('stop_id'))!=ident or type(record.get('satisfied')) is not bool:raise ValueError('invalid stop duty')
        if record['satisfied'] and len(obs)!=controls_params['stop_observations']:raise ValueError('stop duty lacks observations')
        numbers=[x['frame_id'] for x in obs]
        if numbers and (numbers!=list(range(numbers[0],numbers[0]+len(numbers))) or numbers[-1]>f or numbers[0]<0):raise ValueError('future/nonconsecutive stop duty')
        for observation in obs:
            if [s['kind'] for s in observation['sources']]!=['metas','bboxes','rgb']:raise ValueError('stop duty sources missing')
            for source in observation['sources']:
                ext='jpg' if source['kind']=='rgb' else 'pkl'
                if (source['frame_id']!=observation['frame_id'] or source['path']!=f"{q['scenario']}/{q['route_id']}/{source['kind']}/{source['frame_id']:04d}.{ext}"
                        or not isinstance(source.get('sha256'),str) or len(source['sha256'])!=64):raise ValueError('invalid stop duty source')
    return q


def produce(pool,root,output,*,route_keys=None,workers=4):
    from .candidate_pool import validate
    validate(pool)
    if type(workers) is not int or not 1<=workers<=64:raise ValueError('invalid route worker count')
    selected=[r for r in pool['routes'] if route_keys is None or (r['scenario'],r['route_id']) in route_keys]
    if not selected or route_keys is not None and len(selected)!=len(route_keys):raise ValueError('unknown/empty route selection')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    header=dict(policy=POLICY,teacher=rules.identity(),candidate_pool_sha256=pool['sha256'])
    jobs=[(str(root),r,str(output/(digest([r['scenario'],r['route_id']])+'.jsonl')),header) for r in selected]
    import time,sys
    started=time.monotonic();receipts=[]
    def collect(iterator):
        for receipt in iterator:
            receipts.append(receipt)
            if len(receipts)%25==0 or len(receipts)==len(jobs):
                progress=dict(completed_routes=len(receipts),total_routes=len(jobs),elapsed_seconds=round(time.monotonic()-started,1))
                write_json(output/'progress.json',progress)
                print('[teacher] '+json.dumps(progress),file=sys.stderr,flush=True)
    if workers==1:collect(map(_write_route,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:collect(executor.map(_write_route,jobs))
    index=dict(**header,routes=receipts)
    index['sha256']=digest(index);write_json(output/'index.json',index)
    return accounting(pool,output/'index.json')


def accounting(pool,index_path):
    index_path=Path(index_path);index=json.loads(index_path.read_text())
    header=dict(policy=POLICY,teacher=rules.identity(),candidate_pool_sha256=pool['sha256'])
    if index.get('sha256')!=digest({k:v for k,v in index.items() if k!='sha256'}) or any(index.get(k)!=v for k,v in header.items()):
        raise ValueError('teacher production identity mismatch')
    routes={(r['scenario'],r['route_id']):r for r in pool['routes']};seen=set();counts=Counter();targets=Counter()
    for receipt in index['routes']:
        key=receipt['scenario'],receipt['route_id']
        if key in seen or key not in routes:raise ValueError('duplicate/foreign production route')
        seen.add(key);name=receipt['artifact']
        if Path(name).name!=name:raise ValueError('route artifact escapes production bundle')
        actual=inspect_route(index_path.parent/name,routes[key],header)
        if actual!=receipt:raise ValueError('teacher route receipt mismatch')
        counts.update(receipt['dispositions']);targets.update(receipt['targets'])
    return dict(processed_routes=len(seen),processed_frames=sum(counts.values()),dispositions=dict(counts),
        rule_targets=dict(targets),complete_causal_production=bool(routes) and seen==set(routes),
        index_sha256=file_sha(index_path),scope='complete frame accounting, including reasoned abstentions; not label approval')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--candidate-pool',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--route-list',type=Path);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();pool=json.loads(a.candidate_pool.read_text())
    keys=None if a.route_list is None else {(r['scenario'],r['route_id']) for r in json.loads(a.route_list.read_text())}
    print(json.dumps(produce(pool,a.data_root,a.output_dir,route_keys=keys,workers=a.workers),ensure_ascii=False))

if __name__=='__main__':main()
