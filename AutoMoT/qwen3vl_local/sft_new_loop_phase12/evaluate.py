"""Paired independent inference, strict parsing and recomputable audit metrics."""
import json
import time
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
import torch
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity
from qwen3vl_local.sft_new_loop_phase1.run_integrity import digest, write
from . import prompts, cache, plan
from .data import labels, relative_rgb


def _parse(text,spec,phase):
    values=prompts.P1.parse_phase1_output(text,spec=spec) if phase==1 else prompts.P2.parse_event_output(text,spec=spec)
    return {k: (v=='YES' if isinstance(v,str) else v) for k,v in values.items()}


def metrics(cases,phases=(1,2)):
    result={}
    for phase in phases:
        questions=defaultdict(Counter);groups=defaultdict(lambda:Counter(total=0,exact=0));exact=valid=0
        for case in cases:
            arm=case[f'phase{phase}'];gt=arm['gt'];pred=arm['parsed']
            ok=all(type(pred.get(k))==bool for k in gt);correct=ok and all(pred[k]==v for k,v in gt.items())
            valid+=ok;exact+=correct;group=groups[case['scenario']];group['total']+=1;group['exact']+=correct
            for k,v in gt.items():
                c=questions[k];p=pred.get(k);c['total']+=1
                if type(p)!=bool:c['invalid']+=1
                if p is v:c['correct']+=1
                if v:c['positive']+=1;c['tp' if p is True else 'fn']+=1
                else:c['negative']+=1;c['fp' if p is True else ('tn' if p is False else 'invalid_negative')]+=1
        for c in questions.values():
            c['accuracy']=c['correct']/c['total'];c['recall']=c['tp']/max(1,c['positive'])
            c['precision']=c['tp']/max(1,c['tp']+c['fp']);c['f1']=2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn'])
        result[f'phase{phase}']=dict(total=len(cases),exact=exact,exact_accuracy=exact/max(1,len(cases)),
            format_valid=valid,format_valid_rate=valid/max(1,len(cases)),questions=dict(questions),scenarios=dict(groups))
    if tuple(phases)==(1,2):
        result['both_exact']=sum(all(c[f'phase{p}']['parsed']==c[f'phase{p}']['gt'] for p in (1,2)) for c in cases)
    return result


def benchmark_summary(cases):
    trials=[t for c in cases for t in c['performance']['paired_benchmark']['trials']]
    separate=sum(t['independent_seconds'] for t in trials);shared=sum(t['shared_seconds'] for t in trials)
    return dict(cases=sum(bool(c['performance']['paired_benchmark']['trials']) for c in cases),
        trials=len(trials),shared_seconds=shared,independent_seconds=separate,
        saving_fraction=1-shared/separate if separate else None,scope='pre-encoded model computation; not end-to-end application speed')


@torch.no_grad()
def run(bundle,index,data_root,mode,split,output,*,check_cache=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    items=plan.evaluation(index,data_root,split);facts=labels(index);bundle.model.eval()
    prepared=[]
    for item in items:
        answers=facts[identity(item.row)];specs=prompts.specs(item,answers)
        prepared.append(dict(**plan.descriptor(item),scenario=item.row.scenario,
            rgb=[relative_rgb(item.row.history_rgb_paths[i]) for i in history_rgb_indices(mode)],
            phase1_gt=answers,phase2_gt=prompts.P2.spec_answers(specs[1]),
            phase1_order=list(specs[0].output_keys),phase2_order=list(specs[1].output_keys)))
    write(output/'inputs.json',dict(split=split,mode=mode,prompt=prompts.identity(mode),items=prepared,
                                 members_sha256=digest([plan.descriptor(i) for i in items])))
    cases=[];start=time.perf_counter()
    # Distribute performance probes across class/subgroup strata, not just the first UE bucket.
    benchmark_indices=[];seen=set()
    for number,item in enumerate(prepared):
        group=(item['balance_key'],item['ue3_subtype'])
        if group not in seen and len(benchmark_indices)<8:benchmark_indices.append(number);seen.add(group)
    for number in range(len(prepared)):
        if len(benchmark_indices)==8:break
        if number not in benchmark_indices:benchmark_indices.append(number)
    if bundle.device.type=='cuda':torch.cuda.reset_peak_memory_stats(bundle.device)
    with (output/'cases.jsonl').open('x') as handle:
        for number,(item,receipt) in enumerate(zip(items,prepared)):
            answers=facts[identity(item.row)];first,second=prompts.specs(item,answers)
            ims=[Image.open(item.row.history_rgb_paths[i]).convert('RGB') for i in history_rgb_indices(mode)]
            try:report=cache.generate(bundle,ims,prompts.texts(first,second,mode),check_full=check_cache and number==0,
                verify_outputs=True,benchmark_repeats=2 if split=='test' and number in benchmark_indices else 0)
            finally:
                for im in ims:im.close()
            case=dict(case_index=number,**receipt,performance={k:v for k,v in report.items() if k!='branches'})
            for p,spec in enumerate((first,second),1):
                branch=report['branches'][p-1];case[f'phase{p}']=dict(**branch,gt=receipt[f'phase{p}_gt'],
                    parsed=_parse(branch['text'],spec,p))
            handle.write(json.dumps(case,ensure_ascii=False)+'\n');handle.flush();cases.append(case)
            print(f'[fusion-eval:{split}:{mode}] {number+1}/{len(items)}',flush=True)
    result=metrics(cases);result['performance']=dict(seconds=time.perf_counter()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated(bundle.device) if bundle.device.type=='cuda' else 0,
        device=str(bundle.device),model_dtype=str(next(bundle.model.parameters()).dtype),mode=mode,
        shared_prefix=True,cache_equivalence_first_case=check_cache,
        peak_memory_scope='whole evaluation including independent baseline and diagnostics; not shared-only memory')
    result['performance']['paired_benchmark']=benchmark_summary(cases)
    write(output/'metrics.json',result);return result
