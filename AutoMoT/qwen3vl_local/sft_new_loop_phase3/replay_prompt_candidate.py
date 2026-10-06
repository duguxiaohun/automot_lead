"""原包全量同题候选回放：prepare 不需模型；replay 加载本地完整模型和已校验 v23/候选 adapter。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .paired_eval import load_cases
from .prompt_candidate import CANDIDATE_NAME, fingerprint, prompt_for_case, spec_from_case
from .prompts import CHOICE_SYSTEM_PROMPT, SYSTEM_PROMPT, parse_action_output
from .regression_gate import scored, validate_prompt_messages


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def requests(cases, root, variant):
    for row in cases.values():
        scored(row)
        validate_prompt_messages(row)
        prompt = prompt_for_case(row,variant)
        paths = [Path(p) if Path(p).is_absolute() else root/p for p in row['history_rgb_paths_used']]
        if len(paths) != len(row['history_rgb_sha256']) or [sha(p) for p in paths] != row['history_rgb_sha256']:
            raise ValueError('input RGB fingerprint mismatch')
        spec=spec_from_case(row)
        system=CHOICE_SYSTEM_PROMPT if spec.action_output_mode=='choice' else SYSTEM_PROMPT
        yield row, dict(case_index=row['case_index'], images=[str(p.resolve()) for p in paths],
            image_sha256=row['history_rgb_sha256'], system=system, prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), variant=variant,
            contract_sha256=fingerprint(variant,row['history_rgb_mode'],spec.action_output_mode))


def prepare(cases, root, variant, output):
    # 先核完整集合，再创建新目录，避免把部分准备当完成；不覆盖已有结果。
    items=list(requests(cases,root,variant))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'requests.jsonl').open('w') as f:
        for _,request in items: f.write(json.dumps(request,ensure_ascii=False)+'\n')
    return items


def run(args):
    cases,hashes=load_cases(args.cases)
    modes={(r['history_rgb_mode'],r['prompt_spec']['action_output_mode']) for r in cases.values()}
    if len(modes)!=1: raise ValueError('one RGB/output contract required per replay')
    if args.command=='replay':
        if not args.model_dir.is_dir() or not (args.model_dir/'config.json').is_file():
            raise FileNotFoundError('complete local model missing; prepare works without weights')
        if args.adapter_dir is None: raise ValueError('matching v23 adapter is required')
        # validate the original training contract; no mismatch bypass or edited adapter metadata.
        from .eval import _validate_action_adapter
        cfg=_validate_action_adapter(args.adapter_dir,args.model_dir)
        from .prompt_contract import adapter_prompt_variant
        trained_variant = adapter_prompt_variant(cfg)
        if trained_variant != 'baseline' and trained_variant != args.variant:
            raise ValueError('candidate adapter must use its trained prompt variant')
        if (cfg['history_rgb_mode'],cfg['action_output_mode'])!=next(iter(modes)):
            raise ValueError('adapter RGB/output contract mismatch')
    items=prepare(cases,args.automot_root,args.variant,args.output)
    manifest=dict(schema='phase3_prompt_replay_v1',variant=args.variant,cases=len(items),
        source_sha256=hashes,requests_sha256=sha(args.output/'requests.jsonl'),
        generation_completed=False,development_only=True,
        model_dir=str(args.model_dir),adapter_dir=str(args.adapter_dir),
        max_new_tokens=args.max_new_tokens,source_code_sha256={p.name:sha(p) for p in
            [Path(__file__),Path(__file__).with_name('prompt_candidate.py')]})
    manifest_path=args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.command=='prepare':
        print(json.dumps(dict(prepared=len(items),generated=0,output=str(args.output))))
        return
    import torch
    from PIL import Image
    from .eval import load_eval_bundle, _kv_start_state, _student_generate_kv
    torch.manual_seed(20261006)
    device=torch.device(args.device)
    bundle=load_eval_bundle(args.model_dir,args.adapter_dir,device,merge_lora=True)
    manifest['adapter_prompt_variant']=trained_variant
    manifest['evaluation_kind']='trained_prompt' if trained_variant==args.variant else 'old_adapter_prompt_transfer'
    manifest['adapter_config_sha256']=sha(args.adapter_dir/'sft_new_loop_phase3_adapter_config.json')
    manifest['model_config_sha256']=sha(args.model_dir/'config.json')
    # Single process, fresh prefill; no GT, future images, raw metas or notes in messages.
    with (args.output/'cases.jsonl').open('w') as f:
        for row,request in items:
            images=[]
            for path in request['images']:
                with Image.open(path) as im: images.append(im.convert('RGB'))
            messages=[dict(role='system',content=request['system']),dict(role='user',content=[
                *[dict(type='image',image=im) for im in images],dict(type='text',text=request['prompt'])])]
            with torch.inference_mode():
                state=_kv_start_state(bundle,messages)
                raw,_,_=_student_generate_kv(bundle,state,args.max_new_tokens)
            spec=spec_from_case(row)
            result=parse_action_output(raw,spec=spec)
            parsed={k:'INVALID' if v is None else ('YES' if v else 'NO') for k,v in result.items()}
            valid=all(v is not None for v in result.values())
            out=deepcopy(row)
            # Remove stale secondary scores and original-model messages from copied metadata.
            for key in list(out):
                if key.startswith('answer_only_'): del out[key]
            out.update(raw_output=raw,parsed=parsed,strict_format_valid=valid,
                all_ok=valid and parsed==row['gt'],ok_by_key={k:parsed[k]==v for k,v in row['gt'].items()},
                action_user_prompt=request['prompt'],prompt_experiment=request,
                actual_chat_messages=[dict(role='system',content=request['system']),dict(role='user',content=[
                    *[dict(type='image',source_path=p) for p in row['history_rgb_paths_used']],
                    dict(type='text',text=request['prompt'])])])
            f.write(json.dumps(out,ensure_ascii=False)+'\n');f.flush()
    manifest.update(generation_completed=True,result_sha256=sha(args.output/'cases.jsonl'))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(generated=len(items),output=str(args.output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','replay'])
    p.add_argument('--cases',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--variant',choices=['baseline',CANDIDATE_NAME],default=CANDIDATE_NAME)
    p.add_argument('--automot-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--model-dir',type=Path,default=Path(__file__).resolve().parents[2]/'checkpoints/Qwen3.5-4B')
    p.add_argument('--adapter-dir',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-new-tokens',type=int,default=256)
    args=p.parse_args()
    if args.max_new_tokens<1: p.error('--max-new-tokens must be positive')
    run(args)

if __name__=='__main__': main()
