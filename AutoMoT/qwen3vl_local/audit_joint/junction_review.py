"""Read-only concurrent junction evidence, never automatic instance merging."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path

from PIL import Image
from .io import file_sha, digest, write_json
from ..sft_new_loop_phase4 import privileged_geometry as g, teacher_rules
from ..sft_new_loop_phase4.controller import Episode
from ..sft_new_loop_phase4.input_identity import rgb_content_sha
from ..sft_new_loop_phase4.route_prompts import messages
from ..sft_new_loop_phase4.teacher_replay import validate_question

EVENTS = {'U-E6', 'U-E7', 'R-E5'}


def input_disagreement(questions):
    targets={q['rule_target'] for q in questions}
    if {'YES','NO'} <= targets:return 'same_input_binary_conflict'
    if 'UNKNOWN' in targets and len(targets)>1:return 'same_input_abstention_disagreement'
    return 'same_input_same_target'


def junction_id(value):
    # CARLA's -1 means no identified junction, not a shared identity.
    return value if type(value) is int and value >= 0 else None


def pair_evidence(a, b):
    x,y = a['instance'],b['instance']
    ja,jb = junction_id(x.get('junction_id')),junction_id(y.get('junction_id'))
    points = [x.get('junction_world'),y.get('junction_world')]
    finite = all(isinstance(p,list) and len(p)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in p) for p in points)
    return dict(instance_ids=[a['episode']['instance_id'],b['episode']['instance_id']],
        declared_junction_relation=('unknown' if ja is None or jb is None else 'same' if ja==jb else 'different'),
        frozen_reference_distance_m=math.dist(*points) if finite else None,
        anchor_ids=[x['actor_id'],y['actor_id']],
        participant_sets=[x.get('actor_ids',[]),y.get('actor_ids',[])],
        status='unresolved_instance_identity',merge_authorized=False)


def concurrent(records):
    seen=set()
    for r in records:
        key=r['scenario'],r['route_id'],r['frame_id']
        if key in seen:raise ValueError('duplicate replay frame')
        seen.add(key)
        groups=defaultdict(dict)
        for q in r['questions']:
            if q['rgb_mode']!=4 or q['episode']['event'] not in EVENTS:continue
            if any(q[k]!=r[k] for k in ('scenario','route_id','frame_id','split','physical_group')):
                raise ValueError('question detached from frame')
            ident=q['episode']['instance_id'];group=groups[q['episode']['event']]
            if ident in group and (group[ident]['instance']!=q['instance'] or group[ident]['episode']!=q['episode']):
                raise ValueError('inconsistent same-frame instance evidence')
            group[ident]=q
        for event,instances in groups.items():
            if len(instances)>1:yield r,event,instances


def audit(replay, receipt, request, data_root, output):
    replay,receipt,request,data_root,output=map(Path,(replay,receipt,request,data_root,output))
    if output.exists():raise FileExistsError('use a new output directory')
    if output.resolve().is_relative_to(data_root.resolve()):raise ValueError('output inside source data')
    paths=[replay,receipt,request]
    bindings={str(p.absolute()):dict(resolved=str(p.resolve()),sha256=file_sha(p)) for p in paths}
    rc=json.loads(receipt.read_text());rq=json.loads(request.read_text())
    teacher=teacher_rules.identity()
    if rc['sha256']!=file_sha(replay) or rc['request_sha256']!=file_sha(request) or rc['teacher']!=teacher:
        raise ValueError('replay/request/current teacher binding differs')
    records=[json.loads(s) for s in replay.open()]
    declared={(r['scenario'],r['route_id']):r['rgb_frames'] for r in rq['routes']}
    actual=defaultdict(list)
    for r in records:actual[r['scenario'],r['route_id']].append(r['frame_id'])
    if dict(actual)!=declared or len(records)!=rc['frames']:raise ValueError('incomplete requested replay')
    root_binding=dict(path=str(data_root.absolute()),resolved=str(data_root.resolve()))
    verified={};resolved_sources={};pixels={};result=[]
    def source(s):
        rel=Path(s['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source path')
        p=data_root/rel
        if s['path'] not in verified:
            resolved_sources[s['path']]=str(p.resolve())
            verified[s['path']]=file_sha(p)
        if verified[s['path']]!=s['sha256'] or str(p.resolve())!=resolved_sources[s['path']]:
            raise ValueError('source bytes or logical path binding changed')
        return p
    def visible_key(q):
        hashes=[]
        for s in q['input_sources']:
            p=source(s)
            if s['sha256'] not in pixels:
                with Image.open(p) as im:pixels[s['sha256']]=rgb_content_sha(im)
            hashes.append(dict(rgb_sha256=pixels[s['sha256']]))
        obs=dict(frame_id=q['frame_id'],history_frames=q['history_frames'],speed_mps=max(0.,q['ego_speed']))
        return digest(messages(Episode(**q['episode']),q['edge'],obs,hashes))
    for r,event,instances in concurrent(records):
        for s in r['sources']:source(s)
        frame=g.load_frame(data_root,r['scenario'],r['route_id'],r['frame_id'])
        if frame['sources']!=r['sources']:raise ValueError('current frame binding differs')
        actors={a['id']:a for a in frame['actors']};anchors=[];inputs=defaultdict(list)
        for q in r['questions']:
            if q['rgb_mode']!=4 or q['episode']['event']!=event:continue
            validate_question(q,expected_teacher_sha=digest(teacher))
            for s in q['causal_sources']+q.get('instance_sources',[]):source(s)
            for record in q.get('control_evidence',{}).values():
                for observation in record['observations']:
                    for s in observation['sources']:source(s)
            inputs[visible_key(q)].append(q)
        for ident,q in instances.items():
            a=actors.get(q['instance']['actor_id'])
            anchors.append(dict(instance_id=ident,seed=q['instance'],episode=q['episode'],
                anchor=None if a is None else {k:a.get(k) for k in ('id','class','position','road_id','lane_id')},
                actor_lookup_status='present' if a else 'missing_not_completion'))
        duplicate=[]
        for key,qs in inputs.items():
            ids={q['episode']['instance_id'] for q in qs}
            if len(ids)>1:
                duplicate.append(dict(rendered_input_sha256=key,question_ids=[q['question_id'] for q in qs],
                    instance_ids=sorted(ids),rule_targets=[q['rule_target'] for q in qs],
                    phases=[q['slice'] for q in qs],reasons=[q['reasons'] for q in qs],
                    status=input_disagreement(qs),
                    requires_admission_review=len({q['rule_target'] for q in qs})>1))
        result.append(dict(scenario=r['scenario'],route_id=r['route_id'],physical_group=r['physical_group'],
            frame_id=r['frame_id'],event=event,
            metadata={k:frame['meta'].get(k) for k in ('is_junction','junction_id','speed','current_active_scenario_type')},
            anchors=anchors,pairs=[pair_evidence(a,b) for a,b in combinations(instances.values(),2)],
            rendered_input_groups=duplicate))
    if bindings!={str(p.absolute()):dict(resolved=str(p.resolve()),sha256=file_sha(p)) for p in paths}:
        raise ValueError('inputs changed during audit')
    for rel,sha in verified.items():
        if file_sha(data_root/rel)!=sha or str((data_root/rel).resolve())!=resolved_sources[rel]:
            raise ValueError('source changed during audit')
    if str(data_root.resolve())!=root_binding['resolved']:raise ValueError('data root binding changed')
    summary=dict(schema='junction_concurrency_development_v1',frame_event_rows=len(result),
        routes=len({(r['scenario'],r['route_id']) for r in result}),
        declared_relations=dict(Counter(p['declared_junction_relation'] for r in result for p in r['pairs'])),
        duplicate_rendered_inputs=dict(Counter(g['status'] for r in result for g in r['rendered_input_groups'])),
        source_files_verified=len(verified),confirmed_merges=0,independent_approval=False,g2_status='not_run',
        scope='concurrent anchors and exact rendered inputs only; junction ID/proximity cannot authorize an instance merge')
    output.mkdir(parents=True)
    write_json(output/'summary.json',summary);write_json(output/'frames.json',result)
    write_json(output/'receipt.json',dict(inputs=bindings,data_root=root_binding,teacher=teacher,
        source_sha256=verified,source_resolved_paths=resolved_sources,
        tool_sha256=file_sha(__file__),files={n:file_sha(output/n) for n in ('summary.json','frames.json')}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','receipt','request','data-root','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),ensure_ascii=False))

if __name__=='__main__':main()
