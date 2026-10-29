"""Original heldout membership for fair per-task comparisons, always test-only."""
import json
from pathlib import Path
import time
from PIL import Image
import torch
from qwen3vl_local.sft_new_loop_phase1 import eval as native1, run_integrity as proof
from qwen3vl_local.sft_new_loop_phase2 import eval as native2
from qwen3vl_local.sft_new_loop_phase1.training_policy import identity
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.sft_v3.train import _kv_start_state,_student_generate_kv
from . import prompts,evaluate
from .data import relative_rgb,resolve_rgb


def items(root,data_root,phase):
    native=native1 if phase==1 else native2
    rows=native._read_rows(Path(root)/f'phase{phase}/data/frame_index.jsonl','test',data_root=Path(data_root))
    return native._balanced_cases(rows,cases_per_bin=64,seed=20260810)


def prepare(root,data_root):
    records={};images={}
    for phase in (1,2):
        records[phase]=[]
        for i in items(root,data_root,phase):
            paths=[relative_rgb(p) for p in i.row.history_rgb_paths]
            records[phase].append(dict(key=identity(i.row),answers=i.row.answers,paths=paths,
                balance_key=i.balance_key,original_asked_keys=list(i.spec.output_keys),
                seed_key=i.spec.phase2_spec.seed_key if phase==1 else i.spec.seed_key,
                true_rs=getattr(i.row,'true_rs',getattr(i.row,'rs','')),event=i.row.event,
                ue3_subtype=getattr(i.row,'ue3_subtype',''),invalid_source=getattr(i.row,'invalid_source',''),
                domain=getattr(i.row,'question_domain',''),town=i.row.town))
            for path in paths:
                if path not in images:
                    p=resolve_rgb(data_root,path)
                    with Image.open(p) as im:im.verify()
                    images[path]=proof.sha(p)
    proof.write(Path(root)/'cpu/reference_members.json',records)
    proof.write(Path(root)/'cpu/reference_rgb_sha256.json',images)


@torch.no_grad()
def run(bundle,root,mode,phase):
    root=Path(root);ready=proof.read(root/'ready.json');out=root/'evaluation'/mode/f'original_phase{phase}_test'
    out.mkdir(parents=True,exist_ok=False);selected=items(root,ready['data_root'],phase)
    frozen=proof.read(root/'cpu/reference_members.json')[str(phase)]
    if len(selected)!=len(frozen):raise ValueError('Reference test membership changed')
    cases=[];bundle.model.eval();start=time.perf_counter()
    with (out/'cases.jsonl').open('x') as handle:
        for number,(item,expected) in enumerate(zip(selected,frozen)):
            if list(identity(item.row))!=expected['key'] or item.row.answers!=expected['answers']:
                raise ValueError('Reference test labels changed')
            if phase==1:
                spec=prompts.P1.make_prompt_spec(variant='all_random_order',answers=item.row.answers,
                    seed_key=item.spec.phase2_spec.seed_key)
                text=prompts.P1.build_phase1_prompt(spec=spec,history_rgb_mode=mode);gt=item.row.answers
            else:
                spec=item.spec;text=prompts.P2.build_event_prompt(spec=spec,history_rgb_mode=mode);gt=prompts.P2.spec_answers(spec)
            images=[Image.open(item.row.history_rgb_paths[i]).convert('RGB') for i in history_rgb_indices(mode)]
            try:
                state=_kv_start_state(bundle,prompts.messages(images,text))
                raw,_,tokens=_student_generate_kv(bundle,state,160)
            finally:
                for im in images:im.close()
            case=dict(case_index=number,scenario=item.row.scenario,**expected,
                output_keys=list(spec.output_keys),rgb=[expected['paths'][i] for i in history_rgb_indices(mode)],
                **{f'phase{phase}':dict(text=raw,gt=gt,parsed=evaluate._parse(raw,spec,phase),generated_tokens=int(tokens.numel()))})
            cases.append(case);handle.write(json.dumps(case,ensure_ascii=False)+'\n');handle.flush()
            print(f'[fusion-original-phase{phase}:{mode}] {number+1}/{len(selected)}',flush=True)
    report=evaluate.metrics(cases,(phase,));report.update(seconds=time.perf_counter()-start,
        warning='P1 original case membership with new eight-question protocol; compare common questions separately',
        mode=mode,prompt=prompts.identity(mode),source_members_sha256=proof.digest(frozen))
    proof.write(out/'metrics.json',report);return report
