"""Bounded local Phase3/4 full-frame labeling replay; no model or label approval.

Reads native rules, keeps every requested RGB frame, and preserves abstentions.
Outputs are diagnostic candidates, never a replacement for a dataset manifest.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict


def validate_request(request):
    from qwen3vl_local.sft_new_loop_phase4.dataset import source_route, groups, holdout_reservations, split_for
    root=Path(request['data_root']);seen=set();exposed=groups();reserved=holdout_reservations()
    if not request.get('routes'):raise ValueError('no requested routes')
    for r in request['routes']:
        key=r['scenario'],r['route_id']
        if key in seen:raise ValueError('duplicate requested route')
        seen.add(key)
        group=source_route(r)
        if r['physical_group']!=group or r['split']!=split_for(group,exposed,reservations=reserved):
            raise ValueError('route physical identity or frozen split mismatch')
        if r['split']!='train':raise ValueError('this development replay accepts training routes only')
        frames=r['rgb_frames']
        if (not frames or any(type(x) is not int or x<0 for x in frames)
                or frames!=sorted(set(frames))):raise ValueError('invalid frame inventory')
        actual=sorted(int(p.stem) for p in (root/key[0]/key[1]/'rgb').glob('*.jpg') if p.stem.isdigit())
        if frames!=actual:raise ValueError('request must include every RGB frame, not sampled anchors')
    return request


def run(request_path, output, collection_root):
    # Limits apply only to this dedicated CPU process, never to the host or other jobs.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(4,0,0,0,0)!=0 or libc.prctl(3,0,0,0,0)!=0:
        raise RuntimeError('cannot disable process dumpability')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    os.nice(15)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    request=validate_request(json.loads(Path(request_path).read_text()))
    OUT=Path(output).resolve();OUT.mkdir(parents=True,exist_ok=False)
    APP=Path(__file__).resolve().parents[2]
    collection_root=Path(collection_root).resolve()
    (OUT/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
    from qwen3vl_local.sft_new_loop_phase4.identity import contract, digest, file_sha, write_json
    from qwen3vl_local.sft_new_loop_phase4 import teacher_replay as p4, teacher_rules
    from qwen3vl_local.sft_new_loop_phase4.full_pipeline import default_registry
    from qwen3vl_local.sft_new_loop_phase4.teacher_approval import validate_registry,target
    from qwen3vl_local.sft_new_loop_phase4.risk_review import registry as risk_registry,teacher_window_check
    from qwen3vl_local.sft_new_loop_phase4.dataset import holdout_reservations
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import CONTEXT_TO_EVENT
    from qwen3vl_local.sft_new_loop_phase3 import build_dataset as p3
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import load_route_trajectory, action_evidence, label_actions
    from qwen3vl_local.sft_new_loop_phase3.collection_reader import iter_routes
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapped_contexts,mapping_contract_hash
    from qwen3vl_local.sft_new_loop_phase3.annotation_repair import repair_annotation
    from qwen3vl_local.sft_new_loop_phase3.choice_semantics import choice_annotation
    from qwen3vl_local.sft_new_loop_phase3.history_rgb import history_exclusion_reason
    from qwen3vl_local.sft_new_loop_phase4.privileged_geometry import plain
    ROOT=Path(request['data_root'])
    started=time.monotonic();written=0

    def guard():
        if shutil.disk_usage(OUT).free<20*1024**3:raise RuntimeError('disk reserve below 20 GiB')
        if written>2*1024**3:raise RuntimeError('diagnostic output exceeds 2 GiB budget')
        values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        while int(values['MemAvailable'].split()[0])<6*1024**2:
            print('paused: available RAM below 6 GiB',flush=True);time.sleep(15)
            values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())

    def emit(stream,value):
        nonlocal written
        text=json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+'\n';written+=len(text.encode());stream.write(text)

    def main():
        guard()
        write_json(OUT/'progress.json',dict(status='in_progress',stage='source_annotations',completed_routes=0))
        source=dict(producer=contract(),phase3_mapping_sha256=mapping_contract_hash(),worker_sha256=file_sha(__file__),request_sha256=file_sha(OUT/'request.json'),dumpable=libc.prctl(3,0,0,0,0),resource_limits=dict(memory_limit_gib=6,min_available_ram_gib=6,min_free_disk_gib=20,workers=1,output_jsonl_limit_gib=2))
        write_json(OUT/'source.json',source)
        collection=OUT/'selected_collection';collection.mkdir()
        selected={};source_files={}
        for scenario in sorted({r['scenario'] for r in request['routes']}):
            path=collection_root/f'{scenario}_result.json'
            wanted={r['route_id'] for r in request['routes'] if r['scenario']==scenario}
            rows=[]
            for row in iter_routes(path):
                if row.get('route_id') in wanted:
                    rows.append(row);selected[scenario,row['route_id']]=row
            guard()
            write_json(OUT/'progress.json',dict(status='in_progress',stage='source_annotations',scenario=scenario,completed_scenarios=len(source_files)+1))
            source_files[str(path.resolve())]=dict(sha256=file_sha(path),selected_routes=len(rows))
            write_json(collection/f'{scenario}_result.json',dict(routes=rows,scope='selected diagnostic routes only'))
        write_json(OUT/'annotation_sources.json',source_files)
        # Actual native eligible rows, before quota balancing and invalid-context augmentation.
        old=sys.argv
        try:
            sys.argv=['phase3'];args=p3.parse_args()
        finally:sys.argv=old
        args.collection_dir=str(collection);args.data_root=str(ROOT);args.workers=0;args.progress_every_routes=1
        candidates=defaultdict(list);risk_stats=Counter();candidate_count=0
        with (OUT/'phase3_native_candidates.jsonl').open('x') as f:
            for base in p3.iter_base_frames(args,risk_stats):
                guard();row=p3._make_row(base=base,context_id=base['context_id'],invalid=False)
                row.update(choice_annotation(row['answers'],row['context_id'],row['action_evidence']))
                row['diagnostic_only']=True;row['split_status']='native_hash_before_global_coverage_and_joint_isolation'
                emit(f,row);candidates[base['scenario'],base['route_id'],base['frame_id']].append(row);candidate_count+=1
        write_json(OUT/'phase3_candidate_counts.json',dict(count=candidate_count,risk_stats=dict(risk_stats),scope='native eligible unsampled diagnostic candidates; no new training manifest or approval'))
        approval=validate_registry(default_registry());risks=risk_registry();reserved=holdout_reservations()
        counts=Counter();presence=Counter();funnel=Counter();routes=[];flags=Counter();dispositions=Counter()
        raw=OUT/'phase4_routes';raw.mkdir()
        with (OUT/'joint_timeline.jsonl').open('x') as joint, (OUT/'phase3_timeline.jsonl').open('x') as p3out, (OUT/'phase4_questions.jsonl').open('x') as p4out:
            for i,r in enumerate(request['routes']):
                guard();key=(r['scenario'],r['route_id']);run=ROOT.joinpath(*key)
                trajectory=load_route_trajectory(run)
                annotations={int(a['frame_id']):a for a in selected.get(key,{}).get('annotations',[]) if a.get('frame_id') is not None}
                print('route',i+1,'/',len(request['routes']),*key,'frames',len(r['rgb_frames']),'annotations',len(annotations),flush=True)
                route_counts=Counter();p4_events=Counter();p3_events=Counter();p3_reason=Counter()
                path=raw/(digest(key)+'.jsonl')
                with path.open('x') as artifact:
                    emit(artifact,dict(kind='diagnostic_header',route=r,teacher=teacher_rules.identity(),scope='raw native replay; not a production index'))
                    for record in p4.route_records(ROOT,r):
                        guard();emit(artifact,record);f=record['frame_id'];route_counts[record['disposition']]+=1
                        signals=trajectory.signals(f) if trajectory else None
                        evidence=action_evidence(signals) if signals else None
                        labels=label_actions(signals) if signals else None
                        ann=annotations.get(f);contexts=[];mapping=None;repair=None
                        if ann is not None:
                            rs,primary,codes,repair=repair_annotation(*key,f,ann,p3._rs_label(ann),str(ann.get('primary_event') or 'UNKNOWN'),p3._event_codes(ann))
                            if rs!='UNKNOWN':contexts,mapping=mapped_contexts(*key,f,rs,primary,codes)
                        pp=dict(scenario=key[0],route_id=key[1],frame_id=f,physical_group=r['physical_group'],phase4_split=r['split'],
                          source_annotation_present=ann is not None,source_annotation=ann,contexts=contexts,mapping=mapping,annotation_repair=repair,
                          signals=({k:v for k,v in signals.items() if k!='distance_to_next_junction'} if signals else None),evidence=evidence,raw_labels=labels,
                          native_eligible_candidates=candidates.get((*key,f),[]),rgb_identity=dict(path=str((run/'rgb'/f'{f:04d}.jpg').relative_to(ROOT)),sha256=file_sha(run/'rgb'/f'{f:04d}.jpg')))
                        emit(p3out,pp)
                        if evidence:
                            decision=evidence['longitudinal_decision'];p3_reason[decision['reason']]+=1
                            for flag,value in decision.items():
                                if type(value) is bool and value:flags[flag]+=1
                        p3set={CONTEXT_TO_EVENT[c] for c in contexts};p4set={q['episode']['event'] for q in record['questions']}
                        p3_events.update(p3set);p4_events.update(p4set)
                        status=('phase3_source_unknown' if ann is None or (repair and repair.get('repaired_rs')=='UNKNOWN') else
                            'phase4_abstained_not_event_absence' if not record['questions'] else
                            'same_event_sets' if p3set==p4set else 'event_presence_disagreement_candidate')
                        presence[status]+=1
                        compact=[]
                        for q in record['questions']:
                            p4.validate_question(q,expected_teacher_sha=digest(teacher_rules.identity()))
                            a,reason=target(q,approval)
                            if a!='UNKNOWN' and q['physical_group'] in reserved:a,reason='UNKNOWN','reserved_manual_evaluation'
                            if a!='UNKNOWN' and not teacher_window_check(q,risks.get(key,[]))['outside_registered_windows']:a,reason='UNKNOWN','known_risk_window_requires_separate_review'
                            event=q['episode']['event'];cell='/'.join([r['split'],event,q['edge'],str(q['rgb_mode']),q['slice'],q['rule_target']])
                            counts[cell]+=1;funnel[cell+'/'+reason]+=1
                            summary=dict(id=q['question_id'],instance_id=q['episode']['instance_id'],event=event,edge=q['edge'],rgb_mode=q['rgb_mode'],
                              state=q['episode']['state'],longitudinal=q['episode']['longitudinal'],slice=q['slice'],target=q['rule_target'],state_rule_target=q['state_rule_target'],facts=q['facts'],reasons=q['reasons'],gate_target=a,gate_reason=reason)
                            compact.append(summary);emit(p4out,dict(question=q,admission_target=a,admission_reason=reason,scope='gate only; later deduplication/observable conflicts/manual precedence not applied'))
                        emit(joint,dict(scenario=key[0],route_id=key[1],frame_id=f,physical_group=r['physical_group'],phase3_events=sorted(p3set),phase4_events=sorted(p4set),
                          presence_status=status,phase3_decision=evidence['longitudinal_decision'] if evidence else None,phase4=compact,traces=record['traces'],reasons=record['reasons'],instance_ends=record['instance_ends'],
                          match_status='candidate_same_event_only_participant_and_target_not_adjudicated',contradiction_label=None))
                item=dict(scenario=key[0],route_id=key[1],split=r['split'],frames=sum(route_counts.values()),dispositions=dict(route_counts),phase3_events=dict(p3_events),phase4_events=dict(p4_events),phase3_reasons=dict(p3_reason),raw_file=str(path.relative_to(OUT)),sha256=file_sha(path))
                assert item['frames']==len(r['rgb_frames'])
                routes.append(item);dispositions.update(route_counts)
                write_json(OUT/'progress.json',dict(status='in_progress',stage='replay',completed_routes=i+1,total_routes=len(request['routes']),elapsed_seconds=time.monotonic()-started,last_route=key))
        result=dict(status='complete',routes=routes,total_frames=sum(x['frames'] for x in routes),phase3_candidate_count=candidate_count,
            presence_counts=dict(presence),raw_question_cells=dict(sorted(counts.items())),gate_funnel=dict(sorted(funnel.items())),phase3_flags=dict(flags),
            scope='selected-route full-frame diagnostic; physical action boundaries, participant matches and label correctness not adjudicated',
            elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            no_training=True,no_production_changes=True,blind_review=False)
        assert source['producer']==contract() and source['phase3_mapping_sha256']==mapping_contract_hash()
        write_json(OUT/'summary.json',result)
        write_json(OUT/'receipt.json',dict(files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name not in ('run.log','progress.json','exit_code','receipt.json')}))
        write_json(OUT/'progress.json',dict(status='complete',completed_routes=len(routes),frames=result['total_frames'],elapsed_seconds=result['elapsed_seconds']))
        print('COMPLETE',result['total_frames'],'frames',candidate_count,'phase3 candidates',flush=True)

    try:main()
    except BaseException as error:
        write_json(OUT/'progress.json',dict(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started))
        traceback.print_exc();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--request',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--collection-dir',default='keyframe_filter/collection_output')
    args=parser.parse_args()
    run(args.request,args.output_dir,args.collection_dir)


if __name__=='__main__':
    main()
