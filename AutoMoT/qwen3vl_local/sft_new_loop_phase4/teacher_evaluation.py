"""Held-out teacher references, isolated from training and human accuracy.

No approval is implied. This artifact cannot be loaded as a training dataset.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from .identity import ROOT,contract,digest,file_sha,write_json,check_contract
from . import teacher_rules as rules
from .teacher_replay import accounting,validate_question
from .risk_review import registry,teacher_window_check
from .observation import observation_contract,validate_observation
from .controller import Episode
from .route_prompts import prompt
from .input_identity import rgb_content_sha,model_input_key,validate_model_inputs

POLICY='heldout_teacher_consistency_only_v1'


def export(pool,index_path,root,output,*,rgb_mode):
    from .candidate_pool import validate
    from .dataset import dump_rows
    from .teacher_review import observable
    from qwen3vl_local.sft_new_loop_phase3.trajectory_action import _load_meta
    from PIL import Image
    validate(pool);spec=observation_contract(rgb_mode);coverage=accounting(pool,index_path)
    output=Path(output)
    if output.exists():raise FileExistsError('use a fresh reference pack')
    root=Path(root);index_path=Path(index_path);index=json.loads(index_path.read_text());risks=registry();counts=Counter()
    output.mkdir(parents=True);files={}
    for split in ('val','test'):
        rows=[]
        for receipt in index['routes']:
            if receipt['split']!=split:continue
            by_input=defaultdict(list);pixels={};speeds={}
            with (index_path.parent/receipt['artifact']).open() as stream:
                next(stream)
                for line in stream:
                    for q in json.loads(line).get('questions',[]):
                        if q['rgb_mode']!=rgb_mode:continue
                        if q['rule_target'] not in ('YES','NO'):counts['abstained']+=1;continue
                        if not teacher_window_check(q,risks.get((q['scenario'],q['route_id']),[]))['outside_registered_windows']:counts['risk_window']+=1;continue
                        by_input[observable(q)].append(q)
            for qs in by_input.values():
                if len({q['rule_target'] for q in qs})>1:counts['conflicting_inputs']+=len(qs);continue
                q=min(qs,key=lambda q:q['question_id']);counts['duplicate_inputs']+=len(qs)-1
                frame=q['frame_id']
                if frame not in speeds:
                    source=next(s for s in q['causal_sources'] if s['frame_id']==frame and s['kind']=='metas')
                    path=root/source['path']
                    if file_sha(path)!=source['sha256']:raise ValueError('reference metadata changed')
                    speeds[frame]=max(0.,float(_load_meta(path)['speed']))
                for source in q['input_sources']:
                    if source['sha256'] not in pixels:
                        path=root/source['path']
                        if file_sha(path)!=source['sha256']:raise ValueError('reference RGB changed')
                        with Image.open(path) as im:pixels[source['sha256']]=rgb_content_sha(im)
                obs=dict(frame_id=frame,history_frames=q['history_frames'],speed_mps=speeds[frame])
                r=dict(id='reference/'+q['question_id'],reference_kind='rule_teacher',label_basis='heldout_rule_reference',
                    teacher_question=q,scenario=q['scenario'],route_id=q['route_id'],physical_group=q['physical_group'],split=split,
                    episode=q['episode'],edge=q['edge'],slice=q['slice'],target=q['rule_target'],observation=obs,
                    images=[s['path'] for s in q['input_sources']],image_sha256=[s['sha256'] for s in q['input_sources']],
                    image_rgb_sha256=[pixels[s['sha256']] for s in q['input_sources']],prompt_sha256=digest(prompt(Episode(**q['episode']),q['edge'],obs)))
                r['model_input_sha256']=model_input_key(r);rows.append(r)
        validate_model_inputs(rows);dump_rows(output/(split+'.jsonl'),rows)
        files[split]=dict(sha256=file_sha(output/(split+'.jsonl')),count=len(rows),routes=len({r['physical_group'] for r in rows}))
    body=dict(policy=POLICY,contract=contract(),teacher=rules.identity(),observation_contract=spec,files=files,
              production=coverage,candidate_pool_sha256=pool['sha256'],exclusions=dict(counts),
              scope='teacher consistency only; no independent accuracy, no model selection, no training supervision')
    write_json(output/'references.json',dict(body,sha256=digest(body)));return body


def load(path,*,split,rgb_mode):
    from .dataset import source_route,split_for,groups,holdout_reservations
    if split not in ('val','test'):raise ValueError('teacher references are held-out only')
    path=Path(path);m=json.loads((path/'references.json').read_text());check_contract(m['contract'])
    if (m.get('policy')!=POLICY or m['teacher']!=rules.identity() or m['observation_contract']!=observation_contract(rgb_mode)
            or m.get('sha256')!=digest({k:v for k,v in m.items() if k!='sha256'})):raise ValueError('reference manifest mismatch')
    f=path/(split+'.jsonl')
    if file_sha(f)!=m['files'][split]['sha256']:raise ValueError('reference source changed')
    rows=[json.loads(line) for line in f.open()];risks=registry();owners=holdout_reservations();exposed=groups();teacher_sha=digest(m['teacher'])
    if len(rows)!=m['files'][split]['count']:raise ValueError('reference count mismatch')
    for r in rows:
        q=validate_question(r['teacher_question'],expected_teacher_sha=teacher_sha)
        if (q['rgb_mode']!=rgb_mode or r['reference_kind']!='rule_teacher' or r['label_basis']!='heldout_rule_reference'
                or r['split']!=split or source_route(q)!=q['physical_group'] or split_for(q['physical_group'],exposed,reservations=owners)!=split
                or any(r[k]!=q[k] for k in ('scenario','route_id','physical_group','split','episode','edge','slice'))
                or r['target']!=q['rule_target'] or r['target'] not in ('YES','NO')
                or r['images']!=[s['path'] for s in q['input_sources']] or r['image_sha256']!=[s['sha256'] for s in q['input_sources']]
                or r['observation']['frame_id']!=q['frame_id'] or r['observation']['history_frames']!=q['history_frames']
                or not teacher_window_check(q,risks.get((q['scenario'],q['route_id']),[]))['outside_registered_windows']):raise ValueError('reference detached from held-out teacher evidence')
        validate_observation(m['observation_contract'],r['observation'],len(r['images']))
    validate_model_inputs(rows);return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-pool',type=Path,required=True);p.add_argument('--production-index',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT.parents[1]/'lead_data');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rgb-mode',type=int,choices=(2,4),required=True);a=p.parse_args()
    result=export(json.loads(a.candidate_pool.read_text()),a.production_index,a.data_root,a.output_dir,rgb_mode=a.rgb_mode)
    print(json.dumps(result['files']))

if __name__=='__main__':main()
