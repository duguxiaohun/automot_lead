"""Additional validation-only controls and conservative checkpoint promotion.

Never adds training samples, uses original test rows, or labels an unobserved
subgroup as passing. The comparator is a checkpoint in this NEW training run;
historical-model non-regression remains a separate final test comparison.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import time
from PIL import Image
import torch
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity, physical_group
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.sft_v3.train import _kv_start_state, _student_generate_kv
from . import prompts, evaluate
from .data import relative_rgb, resolve_rgb

WEAK_KEYS = ('VULNERABLE', 'STATIC_OBSTACLE')
PER_POLARITY = 64
MIN_FRAMES = 8
MIN_GROUPS = 2


def select(rows, key, positive, limit=PER_POLARITY):
    groups=defaultdict(list)
    for row in rows:
        if row['answers'][key] is positive:groups[physical_group(identity(row))].append(row)
    for group in groups:groups[group].sort(key=lambda r:proof.digest([key,positive,identity(r)]))
    ordered=sorted(groups,key=lambda g:proof.digest([key,positive,g]));chosen=[]
    while len(chosen)<limit:
        progressed=False
        for group in ordered:
            if groups[group]:chosen.append(groups[group].pop());progressed=True
            if len(chosen)==limit:break
        if not progressed:break
    return chosen


def prepare(root,data_root,*,historical_groups=()):
    root=Path(root);test_groups={tuple(g) for g in historical_groups};val_rows=[];training_groups=set()
    for phase in (1,2):
        for line in (root/f'phase{phase}/data/frame_index.jsonl').open():
            row=json.loads(line);group=physical_group(identity(row))
            if row['split']=='test':test_groups.add(group)
            if phase==1 and row['split']=='val':val_rows.append(row)
    fusion={}
    for line in (root/'data/frame_index.jsonl').open():
        row=json.loads(line)
        if 'phase1_answers' in row:fusion[identity(row)]=row
        if row['split']=='train':training_groups.add(physical_group(identity(row)))
    allowed=[r for r in val_rows if physical_group(identity(r)) not in test_groups|training_groups]
    chosen={};coverage={};images={}
    for key in WEAK_KEYS:
        coverage[key]={}
        for positive in (True,False):
            rows=select(allowed,key,positive)
            count=dict(available=sum(r['answers'][key] is positive for r in allowed),selected=len(rows),
                       physical_groups=len({physical_group(identity(r)) for r in rows}))
            count['adequate_for_guard']=count['selected']>=MIN_FRAMES and count['physical_groups']>=MIN_GROUPS
            coverage[key]['positive' if positive else 'negative']=count
            for row in rows:chosen[identity(row)]=row
    records=[];rs_aligned=0
    for key,row in sorted(chosen.items()):
        paths=[relative_rgb(p) for p in row['history_rgb_paths']]
        answers=row['answers'];rs_source='phase1_only_no_phase2_anchor'
        if key in fusion:
            if paths!=[relative_rgb(p) for p in fusion[key]['history_rgb_paths']]:raise ValueError('Guard/fusion RGB history differs')
            answers=fusion[key]['phase1_answers'];rs_source='phase2'
            rs_aligned+=int(answers!=row['answers'])
        records.append(dict(key=list(key),scenario=row['scenario'],answers=answers,paths=paths,rs_source=rs_source,
            source_split='val',seed_key='weak_guard:'+proof.digest(key),balance_key='phase1_validation_guard'))
        for path in paths:
            if path not in images:
                p=resolve_rgb(data_root,path)
                with Image.open(p) as im:im.verify()
                images[path]=proof.sha(p)
    if not records:raise ValueError('No heldout Phase1 weak/control validation records')
    coverage_ok=all(c['adequate_for_guard'] for q in coverage.values() for c in q.values())
    report=dict(protocol='validation_guard_v3_phase2_authority',source='original Phase1 val members; shared anchors use authoritative Phase2 RS; parent tests, registered historical tests and fusion train excluded by physical_group',
        rs_aligned_records=rs_aligned,
        selected=len(records),coverage=coverage,coverage_adequate=coverage_ok,
        scope='finite validation guard, not a guarantee of population non-regression',
        highway_selection_certified=False,highway_limitation='No independent highway positive validation; original test never used for selection')
    proof.write(root/'cpu/guard_members.json',records);proof.write(root/'cpu/guard_rgb_sha256.json',images)
    proof.write(root/'cpu/guard_coverage.json',report)
    return report


@torch.no_grad()
def run(bundle,root,mode,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data_root=proof.read(root/'ready.json')['data_root'];members=proof.read(root/'cpu/guard_members.json')
    cases=[];bundle.model.eval();start=time.perf_counter()
    with (out/'cases.jsonl').open('x') as handle:
        for number,item in enumerate(members):
            spec=prompts.P1.make_prompt_spec(variant='all_random_order',answers=item['answers'],seed_key=item['seed_key'])
            paths=[item['paths'][i] for i in history_rgb_indices(mode)]
            images=[Image.open(resolve_rgb(data_root,p)).convert('RGB') for p in paths]
            try:
                state=_kv_start_state(bundle,prompts.messages(images,prompts.P1.build_phase1_prompt(spec=spec,history_rgb_mode=mode)))
                raw,_,tokens=_student_generate_kv(bundle,state,160)
            finally:
                for image in images:image.close()
            case=dict(case_index=number,**item,rgb=paths,phase1=dict(text=raw,gt=item['answers'],
                parsed=evaluate._parse(raw,spec,1),generated_tokens=int(tokens.numel())))
            handle.write(json.dumps(case,ensure_ascii=False)+'\n');handle.flush();cases.append(case)
            if (number+1)%16==0:print(f'[weak-validation:{mode}] {number+1}/{len(members)}',flush=True)
    report=evaluate.metrics(cases,(1,));report['seconds']=time.perf_counter()-start
    proof.write(out/'metrics.json',report);return report


def read_cases(folder):return [json.loads(l) for l in (Path(folder)/'cases.jsonl').open() if l.strip()]


def regressions(before,after,phases):
    if len(before)!=len(after) or not before:raise ValueError('Incomplete validation comparison')
    losses=[]
    for old,new in zip(before,after):
        if any(old[k]!=new[k] for k in ('case_index','key','rgb')):raise ValueError('Validation membership changed')
        for p in phases:
            a,b=old[f'phase{p}'],new[f'phase{p}']
            if a['gt']!=b['gt']:raise ValueError('Validation labels changed')
            for key,truth in a['gt'].items():
                if a['parsed'].get(key) is truth and b['parsed'].get(key) is not truth:
                    losses.append(dict(case_index=new['case_index'],key=new['key'],phase=p,question=key))
    return losses


def decision(candidate,incumbent,coverage):
    candidate=Path(candidate);metrics=proof.read(candidate/'metrics.json')
    guard=proof.read(candidate/'weak_guard/metrics.json')['phase1']['questions']
    losses=[]
    if incumbent is not None:
        incumbent=Path(incumbent)
        for suffix,phases in (('',(1,2)),('weak_guard',(1,))):
            for loss in regressions(read_cases(incumbent/suffix),read_cases(candidate/suffix),phases):
                losses.append(dict(validation_set=suffix or 'fusion',**loss))
    scores=[metrics[f'phase{p}']['exact_accuracy'] for p in (1,2)]
    # Both positive recall and negative controls matter; casewise regression veto
    # prevents higher aggregate scores from erasing previously correct answers.
    score=[min(guard[k]['recall'] for k in WEAK_KEYS),min(scores),sum(scores)/2,
           sum(guard[k]['f1'] for k in WEAK_KEYS)/len(WEAK_KEYS)]
    return dict(score=score,eligible=not losses,old_correct_losses=losses,
        comparison_reference=str(incumbent.name) if incumbent else 'first_trained_validation_no_prior_checkpoint',
        coverage_adequate=coverage['coverage_adequate'],historical_model_protection_verified=False,
        quality_review_required=True,
        limitation='Guard covers this new run only; highway val absent and historical test comparison pending')
