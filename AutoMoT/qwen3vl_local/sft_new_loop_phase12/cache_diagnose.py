"""Read-only, single-GPU diagnostics against a failed run's frozen source/data.

No optimizer, no adapters loaded, no edits to the failed run or its receipts.
This file is supplemental diagnostic code, not a replacement training source.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def compare_logits(actual,expected):
    import torch
    if actual.shape!=expected.shape:raise ValueError('Logit shapes differ')
    atol,rtol=(.08,.03) if actual.dtype==torch.bfloat16 else (2e-5,2e-4)
    count=bad=nonfinite=0;maximum=total=0.
    # Avoid allocating comparison buffers for >1 billion logits at once.
    for offset in range(0,actual.shape[1],16):
        a=actual[:,offset:offset+16].float();b=expected[:,offset:offset+16].float()
        delta=(a-b).abs();finite=torch.isfinite(a)&torch.isfinite(b)
        count+=a.numel();bad+=int((~torch.isclose(a,b,atol=atol,rtol=rtol)).sum())
        nonfinite+=int((~finite).sum())
        safe=delta[finite]
        if safe.numel():maximum=max(maximum,float(safe.max()));total+=float(safe.sum())
    return dict(elements=count,mismatches=bad,nonfinite=nonfinite,max_abs=maximum,
        mean_abs_finite=total/max(1,count-nonfinite),atol=atol,rtol=rtol,
        all_close=bad==0 and nonfinite==0,last_argmax_matches=bool(torch.equal(actual[:,-1].argmax(-1),expected[:,-1].argmax(-1))))


def measure(bundle,inputs,cache,max_new_tokens=160):
    import torch
    with torch.no_grad():
        n,prefix=cache.prefill_prefix(bundle.model,*inputs);comparisons=[]
        for item in inputs:
            split=cache.fork(bundle.model,item,n,prefix)
            full=bundle.model(**item,use_cache=False,return_dict=True)
            comparisons.append(compare_logits(split.logits,full.logits[:,n:]))
            del split,full
        del prefix
        shared=cache._pair(bundle,inputs,max_new_tokens,True)
        independent=cache._pair(bundle,inputs,max_new_tokens,False)
        error=None
        try:cache.assert_answers(shared,independent)
        except ValueError as exc:error=str(exc)
    return dict(actual_dtype=str(next(bundle.model.parameters()).dtype),prefix_tokens=n,
        input_tokens=[int(x['input_ids'].shape[1]) for x in inputs],logits=comparisons,
        full_answers_identical=error is None,answer_difference=error,shared=shared,independent=independent)


def worker(args,version):
    # Import the ORIGINAL frozen model/prompts/cache, not today's working tree.
    sys.path.insert(0,str(version/'app'))
    import torch
    from qwen3vl_local import runtime_version
    from qwen3vl_local.sft_new_loop_phase12 import integrity,plan,data,prompts,cache
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
    from qwen3vl_local.sft_v2.train import load_model_with_lora
    runtime_version.verify(version)
    ready=integrity.verify_ready(args.source_run,execution=True)
    index=args.source_run/'data/frame_index.jsonl';facts=data.labels(index)
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    # Genuine FP32 control: disable TF32 and cast the already-loaded module.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(plan.SEED)
    bundle=load_model_with_lora(args.model_dir,device=device,lora_rank=16,lora_alpha=32,lora_dropout=.05,
        lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
    bundle.model.eval();cases=[];seen=set()
    for item in plan.evaluation(index,ready['data_root'],'val'):
        if item.row.question_domain in seen:continue
        seen.add(item.row.question_domain);cases.append(item)
    if len(cases)!=2:raise ValueError('Both domains required')
    report=dict(status='running',source_run=str(args.source_run),version=str(version),mode=args.mode,
        ready_sha256=proof.sha(args.source_run/'ready.json'),source_manifest_sha256=proof.sha(version/'manifest.json'),
        diagnostic_sha256=proof.sha(Path(__file__)),base_model=str(args.model_dir),environment=proof.environment(),
        base_asset_sha256=getattr(bundle.model.config,'local_asset_sha256',None),device=str(device),
        optimizer_updates=0,adapter_loaded=False,tf32=False,results=[],quality_certified=False)
    proof.write(args.output/'report.json',report)
    for dtype in (torch.bfloat16,torch.float32):
        # Preserve original FP32 rotary buffers in the initial BF16 model.
        # Calling model.bfloat16() here would change those buffers too.
        if dtype==torch.float32:bundle.model.float()
        if next(bundle.model.parameters()).dtype!=dtype:raise ValueError('Requested dtype not applied')
        for item in cases:
            answers=facts[plan.identity(item.row)];first,second=prompts.specs(item,answers)
            images=plan.native._load_images([item.row.history_rgb_paths[i] for i in history_rgb_indices(args.mode)])
            try:
                texts=prompts.texts(first,second,args.mode)
                inputs=[cache._device(cache.encode(bundle,images,text),device) for text in texts]
                print(f'[cache-diagnose] {dtype} domain={item.row.question_domain} start',flush=True)
                result=measure(bundle,inputs,cache)
                result.update(key=list(plan.identity(item.row)),domain=item.row.question_domain,
                    prompt_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts])
                report['results'].append(result);proof.write(args.output/'report.json',report)
                print(f'[cache-diagnose] {dtype} domain={item.row.question_domain} logits={result["logits"]} answers_identical={result["full_answers_identical"]}',flush=True)
                del inputs
            finally:
                for image in images:image.close()
    runtime_version.verify(version)
    if proof.sha(args.source_run/'ready.json')!=report['ready_sha256']:raise ValueError('Source ready changed')
    report['status']='diagnostic_complete_not_training_or_cache_certification';proof.write(args.output/'report.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,default=Path('checkpoints/Qwen3.5-4B'))
    p.add_argument('--mode',choices=('2rgb_endpoints','4rgb'),default='2rgb_endpoints')
    p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();args.source_run=args.source_run.resolve();args.model_dir=args.model_dir.resolve()
    version=Path(json.loads((args.source_run/'execution.json').read_text())['version'])
    if args.worker:return worker(args,version)
    # Pick exactly one idle GPU; never change compute mode or other processes.
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[r.split(',')[0].strip() for r in gpu.splitlines() if float(r.split(',')[1])<256 and float(r.split(',')[2])<=5]
    requested=os.environ.get('GPU_IDS');selected=requested if requested else (idle[0] if idle else '')
    if selected not in idle:raise ValueError('One idle physical GPU required; GPU_IDS must be one idle ID')
    output=(args.output or Path('checkpoints/phase12_cache_diagnose')/f'run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}').resolve()
    if output.is_relative_to(args.source_run):raise ValueError('Do not write diagnostics inside the failed run')
    archive=Path(str(output)+'.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    output.mkdir(parents=True,exist_ok=False)
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--source-run',str(args.source_run),
        '--model-dir',str(args.model_dir),'--mode',args.mode,'--output',str(output)]
    # Frozen native evaluation imports also honor GPU_IDS. Pin both so those
    # imports cannot auto-select a different GPU before model initialization.
    env={**os.environ,'CUDA_VISIBLE_DEVICES':selected,'GPU_IDS':selected,
         'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(name,None)
    try:
        data_stage.run(command,output/'worker.json',timeout=1800,
            env=env)
    finally:
        # Also package partial failure evidence. No RGB, weights or source-run edits.
        with tarfile.open(archive,'x:gz') as tar:
            for name in ('report.json','worker.json','worker.log'):
                if (output/name).is_file():tar.add(output/name,arcname=name,recursive=False)
            tar.add(Path(__file__),arcname='cache_diagnose.py',recursive=False)
        print('[DIAGNOSTIC_BUNDLE] '+str(archive),flush=True)


if __name__=='__main__':main()
