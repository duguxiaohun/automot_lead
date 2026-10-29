"""Original heldout membership for fair per-task comparisons, always test-only."""
import json
from pathlib import Path
import time
from PIL import Image
import torch
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_v3.train import _kv_start_state,_student_generate_kv
from . import prompts,evaluate
from .data import resolve_rgb


def prepare(root,data_root):
    from . import history
    root=Path(root);obj=history.load(root/'cpu/historical_contract.json')
    images=history.verify_images(obj,data_root)
    records={mode:{str(phase):history.primary(obj,phase,mode)['records'] for phase in (1,2)}
             for mode in ('2rgb_endpoints','4rgb')}
    proof.write(root/'cpu/reference_members.json',records)
    proof.write(root/'cpu/reference_rgb_sha256.json',images)
    proof.write(root/'cpu/historical_isolation.json',history.assert_isolation(root/'data/frame_index.jsonl',obj))


@torch.no_grad()
def run(bundle,root,mode,phase):
    root=Path(root);ready=proof.read(root/'ready.json');out=root/'evaluation'/mode/f'original_phase{phase}_test'
    out.mkdir(parents=True,exist_ok=False)
    frozen=proof.read(root/'cpu/reference_members.json')[mode][str(phase)]
    cases=[];bundle.model.eval();start=time.perf_counter()
    with (out/'cases.jsonl').open('x') as handle:
        for number,expected in enumerate(frozen):
            if expected['case_index']!=number:raise ValueError('Historical case order changed')
            if phase==1:
                spec=prompts.P1.make_prompt_spec(variant='all_random_order',answers=expected['answers'],
                    seed_key=expected['seed_key'])
                text=prompts.P1.build_phase1_prompt(spec=spec,history_rgb_mode=mode);gt=expected['answers']
            else:
                spec=prompts.P2.make_prompt_spec(variant='all_random_order',answers=expected['answers'],seed_key=expected['seed_key'])
                text=prompts.P2.build_event_prompt(spec=spec,history_rgb_mode=mode);gt=prompts.P2.spec_answers(spec)
            images=[Image.open(resolve_rgb(ready['data_root'],p)).convert('RGB') for p in expected['rgb']]
            try:
                state=_kv_start_state(bundle,prompts.messages(images,text))
                raw,_,tokens=_student_generate_kv(bundle,state,160)
            finally:
                for im in images:im.close()
            case=dict(**expected,output_keys=list(spec.output_keys),
                **{f'phase{phase}':dict(text=raw,gt=gt,parsed=evaluate._parse(raw,spec,phase),generated_tokens=int(tokens.numel()))})
            cases.append(case);handle.write(json.dumps(case,ensure_ascii=False)+'\n');handle.flush()
            print(f'[fusion-original-phase{phase}:{mode}] {number+1}/{len(frozen)}',flush=True)
    report=evaluate.metrics(cases,(phase,));report.update(seconds=time.perf_counter()-start,
        warning='P1 original case membership with new eight-question protocol; compare common questions separately',
        mode=mode,prompt=prompts.identity(mode),source_members_sha256=proof.digest(frozen))
    proof.write(out/'metrics.json',report)
    from .compare import compare
    hashes=proof.read(root/'cpu/reference_rgb_sha256.json')
    baseline=[dict(case_index=r['case_index'],historical_case_id=r['historical_case_id'],key=r['key'],rgb=r['rgb'],
        **{f'phase{phase}':dict(gt=r['baseline_gt'],parsed=r['baseline_parsed'])}) for r in frozen]
    comparison=compare(baseline,cases,hashes,hashes,phase)
    proof.write(out/'historical_comparison.json',comparison)
    return report
