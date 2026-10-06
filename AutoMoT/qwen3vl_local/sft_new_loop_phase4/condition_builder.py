"""Compile causal condition records into applicable state-pair questions.

This is a condition-to-label compiler, not an RGB detector. Producers provide
current geometry/priority facts, never future action onsets. The taxonomy is
shared across every route; missing criteria remain UNKNOWN. Reviewed catchup
bands remain a separate evidence source and are not fabricated by this module.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='qwen3vl_local.sft_new_loop_phase4'

import argparse
import json
from pathlib import Path
from .controller import Episode
from .route_context import episode_edges, validate_source_context
from .data_paths import data_path
from .identity import ROOT, digest, file_sha, write_json
from .taxonomy import transitions, applicable, EVENTS

POLICY='phase4_causal_conditions_v1'
RUBRIC=('taxonomy.py','prompts.py','calibration.py','observation.py','route_context.py','route_prompts.py','route_calibration.py')
RECORD_KEYS={'scenario','route_id','episode','frame_id','observed_until','facts',
             'context_valid','sources','evaluation_annotation_protocol'}


def rubric_identity():
    return {f:file_sha(ROOT/f) for f in RUBRIC}


def validate_record(record, producer, data_root=None):
    from .dataset import source_route, EPISODE_FIELDS
    if (not isinstance(producer,dict) or set(producer)!={'name','version','source','rubric_sha256'}
            or not all(isinstance(producer.get(k),str) and producer[k].strip() for k in ('name','version'))
            or producer.get('source') not in ('rgb_review','causal_geometry_review')
            or producer.get('rubric_sha256')!=rubric_identity()):
        raise ValueError('condition producer requires identity and current frozen rubric')
    if not isinstance(record,dict) or set(record)-RECORD_KEYS:
        raise ValueError('unknown condition fields; do not supply future/control/action labels')
    source_route(record)
    ep_fields=record['episode']
    if set(ep_fields)-EPISODE_FIELDS:
        raise ValueError('unknown condition episode field')
    validate_source_context(record)
    ep=Episode(**ep_fields)
    frame=record['frame_id']
    if (type(frame) is not int or frame<4 or type(record.get('observed_until')) is not int
            or record['observed_until']!=frame):
        raise ValueError('noncausal condition observation')
    if 'context_valid' not in record or (record['context_valid'] is not None and type(record['context_valid']) is not bool):
        raise ValueError('condition context must be explicit true/false/unknown')
    edges=episode_edges(ep)
    keys={key for edge in edges for key in edge.criteria}
    facts=record.get('facts')
    if (not isinstance(facts,dict) or set(facts)-keys
            or any(v is not None and type(v) is not bool for v in facts.values())):
        raise ValueError('invalid condition facts; no actions or timing-derived labels')
    sources=record.get('sources')
    if not isinstance(sources,list) or not sources:
        raise ValueError('condition source evidence required')
    root=Path(data_root).resolve() if data_root is not None else None
    prefix=f"{record['scenario']}/{record['route_id']}"
    seen=set();current_rgb=None
    for item in sources:
        if not isinstance(item,dict) or set(item)!={'path','sha256','frame_id','kind'}:
            raise ValueError('invalid condition source receipt')
        f=item['frame_id'];kind=item['kind'];sha=item['sha256']
        if type(f) is not int or not 0<=f<=frame or kind not in ('rgb','metas'):
            raise ValueError('noncausal condition source frame')
        expected=f"{prefix}/{kind}/{f:04d}.{'jpg' if kind=='rgb' else 'pkl'}"
        if (item['path']!=expected or expected in seen or not isinstance(sha,str)
                or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('condition source path/identity mismatch')
        seen.add(expected)
        if kind=='rgb' and f==frame:current_rgb=sha
        if root is not None:
            path=data_path(root, expected)
            if file_sha(path)!=sha:
                raise ValueError('condition source content mismatch')
    if current_rgb is None:
        raise ValueError('condition record must bind the current RGB')
    return ep,current_rgb


def annotations_for_record(record,producer,data_root=None):
    ep,current_rgb=validate_record(record,producer,data_root)
    frame=record['frame_id']
    proof=dict(policy=POLICY,producer=producer,record=record)
    evidence=digest(proof)
    out=[]
    for edge in episode_edges(ep):
        if not applicable(edge,ep.state,ep.longitudinal):continue
        facts={k:record['facts'][k] for k in edge.criteria if k in record['facts']}
        a=dict(scenario=record['scenario'],route_id=record['route_id'],episode=record['episode'],
               edge=edge.key,start=frame,end=frame,facts=facts,slice='readiness',
               label_basis='per_frame_conditions',context_valid=record['context_valid'],
               reviewer='condition producer: '+producer['name']+'@'+producer['version'],
               evidence_id='conditions/'+evidence+'/'+edge.key,
               observation='Shared transition criteria evaluated from supplied causal condition evidence; no manual RGB review implied.',
               frame_sha256={str(frame):current_rgb},condition_provenance=proof)
        if 'evaluation_annotation_protocol' in record:
            a['evaluation_annotation_protocol']=record['evaluation_annotation_protocol']
        out.append(a)
    return out


def validate_annotation(annotation,data_root=None):
    proof=annotation.get('condition_provenance')
    if not isinstance(proof,dict) or set(proof)!={'policy','producer','record'} or proof['policy']!=POLICY:
        raise ValueError('invalid condition provenance')
    expected=annotations_for_record(proof['record'],proof['producer'],data_root)
    if annotation not in expected:
        raise ValueError('compiled annotation differs from source condition record')


def validate_row(row,data_root=None):
    from .route_calibration import label_condition,interval_facts
    proof=row['condition_provenance']
    if not isinstance(proof,dict) or set(proof)!={'policy','producer','record'} or proof['policy']!=POLICY:
        raise ValueError('invalid condition provenance')
    annotations=annotations_for_record(proof['record'],proof['producer'],data_root)
    candidates=[a for a in annotations if a['edge']==row['edge']]
    if not candidates:raise ValueError('condition edge is not applicable to this state')
    a=candidates[0];f=a['start'];ep=Episode(**a['episode'])
    target=label_condition(ep,a['edge'],f,interval_facts(a['facts'],f,f,a['evidence_id']),context_valid=a['context_valid'])
    from .maneuver_safety import GUARDED
    # Compiled condition records carry no independent visible-scope RGB review.
    # The v20 quarantine is part of the expected row, not provenance tampering.
    quarantined = a['edge'] in GUARDED
    expected_target = 'UNKNOWN' if quarantined else target
    expected_slice = 'uncertain' if quarantined else a['slice']
    if (any(row[k]!=a[k] for k in ('scenario','route_id','episode','edge','evidence_id','label_basis'))
            or row['slice']!=expected_slice
            or (quarantined and (row.get('visible_scope_review') is not None or
                                row.get('review_reason')!='visible_scope_reaudit_required'))
            or row['observation']['frame_id']!=f or row['target']!=expected_target
            or row['image_sha256'][-1]!=a['frame_sha256'][str(f)]
            or row.get('evaluation_annotation_protocol')!=a.get('evaluation_annotation_protocol')):
        raise ValueError('compiled row differs from source condition record')


def compile_stream(path,data_root):
    from .route_calibration import label_condition,interval_facts
    from collections import Counter
    path=Path(path);annotations=[];counts=Counter();seen=set();records=0
    with path.open() as source:
        header=json.loads(next(source))
        if not isinstance(header,dict) or set(header)!={'policy','producer'} or header['policy']!=POLICY:
            raise ValueError('invalid condition stream header')
        for line in source:
            if not line.strip():continue
            record=json.loads(line)
            key=digest([record.get(k) for k in ('scenario','route_id','episode','frame_id')])
            if key in seen:raise ValueError('duplicate condition record; merge facts explicitly')
            seen.add(key);records+=1
            for a in annotations_for_record(record,header['producer'],data_root):
                ep=Episode(**a['episode']);f=a['start']
                answer=label_condition(ep,a['edge'],f,interval_facts(a['facts'],f,f,a['evidence_id']),context_valid=a['context_valid'])
                counts[answer]+=1;annotations.append(a)
    if not records:raise ValueError('empty condition stream')
    return annotations,dict(policy=POLICY,source_sha256=file_sha(path),producer=header['producer'],
                            records=records,questions_before_history_filter=len(annotations),
                            answers_before_history_filter=dict(counts),
                            raw_rgb_condition_extraction=False,
                            scope='shared criteria compiler; producer facts require separate calibration')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition-stream',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('use a fresh compiled annotation path')
    annotations,report=compile_stream(a.condition_stream,a.data_root)
    write_json(a.output,annotations)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
