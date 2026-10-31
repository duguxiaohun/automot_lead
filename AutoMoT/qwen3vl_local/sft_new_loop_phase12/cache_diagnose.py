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


def _helpers():
    # Supplemental evidence code stays beside this script; model/cache/prompts
    # below come exclusively from the failed run's verified frozen app.
    import importlib.util
    spec=importlib.util.spec_from_file_location('_diagnostic_cache_checks',Path(__file__).with_name('cache_checks.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_policy(root,mode):
    """Read saved policy before imports/GPU selection; never infer layout from shell."""
    execution=json.loads((root/'execution.json').read_text())
    ready=json.loads((root/'ready.json').read_text())
    training=json.loads((root/'runs'/mode/'training_inputs.json').read_text())
    version=Path(execution['version']).resolve()
    if version!=Path(ready['version']).resolve():raise ValueError('Execution/ready frozen source differs')
    if not (version/'app').is_dir():raise FileNotFoundError('Original frozen app missing: '+str(version))
    layout=ready.get('input_layout','images_before_text_legacy')
    if layout not in ('images_before_text_legacy','text_before_images_v1'):
        raise ValueError('Unknown saved input layout')
    if training['prompt'].get('input_layout','images_before_text_legacy')!=layout:
        raise ValueError('Training/ready input layout mismatch')
    if training.get('mode',mode)!=mode:raise ValueError('Training RGB mode mismatch')
    if training['ready_sha256']!=hashlib.sha256((root/'ready.json').read_bytes()).hexdigest():
        raise ValueError('Training/ready SHA mismatch')
    probe=root/'runs'/mode/'gpu_prefix_probe.json'
    saved=json.loads(probe.read_text()) if probe.is_file() else {}
    if saved.get('input_layout',layout)!=layout:raise ValueError('Probe/ready input layout mismatch')
    flag=saved.get('text_cache_enabled',layout=='text_before_images_v1')
    if type(flag)!=bool or (flag and layout!='text_before_images_v1'):
        raise ValueError('Invalid saved cache switch')
    return dict(version=str(version),input_layout=layout,text_cache_enabled=flag,
        cache_switch_source='probe_receipt' if 'text_cache_enabled' in saved else 'default_not_recorded_in_old_run',
        base_model=training['base_model'],source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'execution.json',root/'ready.json',root/'runs'/mode/'training_inputs.json',probe) if p.is_file()})


def _input_sha(item):
    import torch
    h=hashlib.sha256()
    for key,value in sorted(item.items()):
        if not isinstance(value,torch.Tensor):raise ValueError('Non-tensor encoded input')
        h.update(str((key,str(value.dtype),tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _token_text(bundle,ids):
    import torch
    return bundle.processor.batch_decode(torch.tensor([ids],dtype=torch.long),skip_special_tokens=False)[0]


def _cached_output(bundle,item,inputs,cache):
    from qwen3vl_local.qwen35 import text_prefix
    if text_prefix.enabled():return text_prefix.prefill(bundle,item)
    n,prefix=cache.prefill_prefix(bundle.model,*inputs)
    return cache.fork(bundle.model,item,n,prefix)


def _prime(bundle,pairs,cache,max_new_tokens):
    """Restore preceding probe cases, including their independent/diagnostic forwards."""
    from qwen3vl_local.qwen35 import text_prefix
    for inputs in pairs:
        cache._pair(bundle,inputs,max_new_tokens,True)
        cache._pair(bundle,inputs,max_new_tokens,False)
        with cache.isolated_text_cache(bundle):
            prefix=None
            if not text_prefix.enabled():n,prefix=cache.prefill_prefix(bundle.model,*inputs)
            for item in inputs:
                out=text_prefix.prefill(bundle,item) if text_prefix.enabled() else cache.fork(bundle.model,item,n,prefix)
                full=bundle.model(**item,use_cache=False,return_dict=True)
                del out,full


def trace_first_divergence(bundle,inputs,cache,arm,max_new_tokens,warmup_inputs=()):
    """Replay equal token history in lockstep; report penalized decision scores."""
    import torch
    from qwen3vl_local.sft_v3.train import KVState,_apply_repetition_penalty,_REPETITION_PENALTY,_append_token_ids,_eos_token_ids
    item=inputs[arm-1];ids=item['input_ids'];common=[];eos=_eos_token_ids(bundle)
    def state(out):
        # Clone only last logits, so a tiny view cannot retain gigabytes of
        # prompt logits while the other prefill / FP32 control executes.
        return KVState(ids,ids,item['attention_mask'],out.past_key_values,out.rope_deltas,out.logits[:,-1,:].clone())
    with torch.no_grad(),cache.isolated_text_cache(bundle):
        _prime(bundle,warmup_inputs,cache,max_new_tokens)
        for prior in inputs[:arm-1]:
            out=_cached_output(bundle,prior,inputs,cache)
            cache._decode(bundle,prior,out,max_new_tokens)
            del out
        a=state(_cached_output(bundle,item,inputs,cache))
        b=state(bundle.model(**item,use_cache=True,return_dict=True))
        for step in range(max_new_tokens):
            x=_apply_repetition_penalty(a.next_logits,torch.unique(a.decoded_input_ids),_REPETITION_PENALTY)
            y=_apply_repetition_penalty(b.next_logits,torch.unique(b.decoded_input_ids),_REPETITION_PENALTY)
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                return dict(status='nonfinite',arm=arm,token_index=step)
            left,right=int(x.argmax(-1).item()),int(y.argmax(-1).item())
            if left!=right:
                def scores(logits):
                    values,tokens=logits.float()[0].topk(min(5,logits.shape[-1]))
                    return dict(top=[dict(token_id=int(t),text=_token_text(bundle,[int(t)]),score=float(v))
                                     for v,t in zip(values,tokens)],margin=float(values[0]-values[1]),
                        cached_choice_score=float(logits[0,left]),independent_choice_score=float(logits[0,right]))
                return dict(status='diverged',arm=arm,token_index=step,common_token_ids=common,
                    common_text=_token_text(bundle,common),cached_token_id=left,independent_token_id=right,
                    cached_is_eos=left in eos,independent_is_eos=right in eos,
                    cached=scores(x),independent=scores(y),repetition_penalty=_REPETITION_PENALTY,
                    max_abs_next_logits=float((x.float()-y.float()).abs().max()))
            if left in eos:return dict(status='identical_to_stop',arm=arm,generated_tokens=len(common),stop_token_id=left)
            common.append(left)
            token=torch.tensor([[left]],device=ids.device,dtype=ids.dtype)
            a,_=_append_token_ids(bundle,a,token);b,_=_append_token_ids(bundle,b,token)
    return dict(status='identical_to_limit',arm=arm,generated_tokens=len(common))


def measure(bundle,inputs,cache,max_new_tokens=160,*,warmup_inputs=(),on_evidence=None):
    import torch
    from qwen3vl_local.qwen35 import text_prefix
    helpers=_helpers()
    with torch.no_grad():
        def pair(shared):
            with cache.isolated_text_cache(bundle):
                if shared:
                    _prime(bundle,warmup_inputs,cache,max_new_tokens)
                return cache._pair(bundle,inputs,max_new_tokens,shared)
        # Native order first, then reverse order to expose unstable repeated arms.
        shared=pair(True);independent=pair(False)
        difference=helpers.first_difference(shared,independent)
        # Keep the original validation gate, including terminal-evidence checks.
        error=None
        try:cache.assert_answers(shared,independent)
        except ValueError as exc:error=str(exc)
        result=dict(actual_dtype=str(next(bundle.model.parameters()).dtype),input_layout=text_prefix.identity(),
            text_cache_enabled=text_prefix.cache_enabled(),prefix_tokens=shared['prefix_tokens'],priming_cases=len(warmup_inputs),
            max_new_tokens=max_new_tokens,
            input_tokens=[int(x['input_ids'].shape[1]) for x in inputs],input_sha256=[_input_sha(x) for x in inputs],
            full_answers_identical=error is None,answer_difference=error,first_difference=difference,
            shared=shared,independent=independent,cached_repeat=None,independent_repeat=None,
            cached_repeat_identical=None,independent_repeat_identical=None,
            logits=[],decision_trace=None)
        if on_evidence:on_evidence(result)
        full_repeat=pair(False);cached_repeat=pair(True)
        result.update(cached_repeat=cached_repeat,independent_repeat=full_repeat,
            cached_repeat_identical=helpers.first_difference(shared,cached_repeat) is None,
            independent_repeat_identical=helpers.first_difference(independent,full_repeat) is None)
        if on_evidence:on_evidence(result)
        if difference is not None:
            result['decision_trace']=trace_first_divergence(bundle,inputs,cache,difference['arm'],max_new_tokens,warmup_inputs)
            if on_evidence:on_evidence(result)
        # Prompt-logit diagnostics retain their old tolerances but are not a
        # replacement for the complete generated-output gate.
        with cache.isolated_text_cache(bundle):
            _prime(bundle,warmup_inputs,cache,max_new_tokens)
            for item in inputs:
                out=_cached_output(bundle,item,inputs,cache)
                n=item['input_ids'].shape[1]-out.logits.shape[1]
                full=bundle.model(**item,use_cache=False,return_dict=True)
                result['logits'].append(compare_logits(out.logits,full.logits[:,n:]))
                if on_evidence:on_evidence(result)
                del out,full
    return result


def precision_summary(bf16,fp32):
    if any(not r[k] for r in (bf16,fp32) for k in ('cached_repeat_identical','independent_repeat_identical')):
        return 'unstable_repeated_outputs_investigate_before_precision_attribution'
    a,b=bf16['full_answers_identical'],fp32['full_answers_identical']
    if not a and b:return 'precision_sensitive_on_this_case_not_root_cause_certification'
    if not b:return 'mismatch_persists_or_appears_in_fp32_inspect_state_and_position_paths'
    return 'mismatch_not_reproduced_on_these_inputs'


def worker(args,version):
    args.output=args.output.resolve()
    if args.output.is_relative_to(args.source_run.resolve()) or args.output.is_relative_to(version.resolve()):
        raise ValueError('Diagnostic output must be outside original run/source')
    policy=source_policy(args.source_run,args.mode)
    if str(version.resolve())!=policy['version']:raise ValueError('Frozen version changed')
    os.environ['AUTOMOT_TEXT_PREFIX']='1' if policy['input_layout']=='text_before_images_v1' else '0'
    os.environ['AUTOMOT_TEXT_PREFIX_CACHE']='1' if policy['text_cache_enabled'] else '0'
    sys.path.insert(0,str(version/'app'))
    import torch
    from qwen3vl_local import runtime_version
    from qwen3vl_local.sft_new_loop_phase12 import integrity,plan,data,prompts,cache
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
    from qwen3vl_local.sft_v2.train import load_model_with_lora
    from qwen3vl_local.qwen35.vendor import modeling_qwen3_5 as vendor
    report=dict(status='running',source_run=str(args.source_run),version=str(version),mode=args.mode,
        policy=policy,diagnostic_sha256={p.name:proof.sha(p) for p in
            (Path(__file__),Path(__file__).with_name('cache_checks.py'))},results=[],
        optimizer_updates=0,adapter_loaded=False,quality_certified=False,cache_certified=False)
    report_path=args.output/'report.json'
    if report_path.exists():raise FileExistsError(report_path)
    proof.write(report_path,report)
    try:
        runtime_version.verify(version)
        for module in (integrity,plan,prompts,cache,vendor):
            if not Path(module.__file__).resolve().is_relative_to((version/'app').resolve()):
                raise ValueError('Mixed live/frozen diagnostic imports: '+module.__name__)
        ready=integrity.verify_ready(args.source_run,execution=True)
        training=proof.read(args.source_run/'runs'/args.mode/'training_inputs.json')
        if training['prompt']!=prompts.identity(args.mode):raise ValueError('Frozen prompt/RGB contract mismatch')
        if not hasattr(cache,'isolated_text_cache'):raise ValueError('Frozen cache lacks isolated diagnostic support')
        index=args.source_run/'data/frame_index.jsonl';facts=data.labels(index)
        device=torch.device('cuda',0);torch.cuda.set_device(device)
        torch.manual_seed(plan.SEED);torch.cuda.manual_seed_all(plan.SEED)
        bundle=load_model_with_lora(args.model_dir,device=device,lora_rank=16,lora_alpha=32,lora_dropout=.05,
            lora_vision_scope='off',strict_vision_scope=True,gradient_checkpointing=True)
        bundle.model.eval();cases=[];seen=set()
        for item in plan.evaluation(index,ready['data_root'],'val'):
            if item.row.question_domain in seen:continue
            seen.add(item.row.question_domain);cases.append(item)
        if len(cases)!=2:raise ValueError('Both domains required')
        report.update(source_manifest_sha256=proof.sha(version/'manifest.json'),base_model=str(args.model_dir),
            environment=proof.environment(),base_asset_sha256=getattr(bundle.model.config,'local_asset_sha256',None),
            fast_path_available=vendor.is_fast_path_available,
            model_code_sha256=proof.sha(Path(vendor.__file__)),cache_code_sha256=proof.sha(Path(cache.__file__)),
            fp32_control='FP32 compute after promoting BF16-loaded weights; not an original-FP32 weight reload')
        proof.write(report_path,report)
        for dtype in (torch.bfloat16,torch.float32):
            warmup_inputs=[]
            if dtype==torch.float32:
                # Preserve original rotary buffers for baseline BF16. FP32
                # disables TF32 explicitly and keeps the same rounded weights.
                bundle.model.float()
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if next(bundle.model.parameters()).dtype!=dtype:raise ValueError('Requested dtype not applied')
            for item in cases:
                case=dict(dtype=str(dtype),domain=item.row.question_domain,key=list(plan.identity(item.row)),status='running')
                report['results'].append(case);proof.write(report_path,report)
                answers=facts[plan.identity(item.row)];first,second=prompts.specs(item,answers)
                paths=[item.row.history_rgb_paths[i] for i in history_rgb_indices(args.mode)]
                images=plan.native._load_images(paths)
                try:
                    texts=prompts.texts(first,second,args.mode)
                    inputs=[cache._device(cache.encode(bundle,images,text),device) for text in texts]
                    if any(x['input_ids'].shape[1]>8192 for x in inputs):raise ValueError('Native context limit exceeded')
                    case.update(prompt_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts],
                        image_sha256={str(p):proof.sha(p) for p in paths},question_order=[list(s.output_keys) for s in (first,second)],
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
                    print(f'[cache-diagnose] layout={policy["input_layout"]} {dtype} domain={item.row.question_domain} start',flush=True)
                    def retain(evidence):
                        case.update(evidence);proof.write(report_path,report)
                    case.update(measure(bundle,inputs,cache,args.max_new_tokens,warmup_inputs=warmup_inputs,on_evidence=retain));case['status']='measured'
                    proof.write(report_path,report)
                    print(f'[cache-diagnose] {dtype} domain={item.row.question_domain} answers_identical={case["full_answers_identical"]} first_difference={case["first_difference"]}',flush=True)
                    warmup_inputs.append(inputs)
                finally:
                    for image in images:image.close()
        runtime_version.verify(version)
        if source_policy(args.source_run,args.mode)!=policy:raise ValueError('Failed run evidence changed')
        # Identical processing is required for the precision comparison.
        for i in range(2):
            if report['results'][i]['input_sha256']!=report['results'][i+2]['input_sha256']:
                raise ValueError('Encoded input changed across precision controls')
        report['precision_comparison']={r['domain']:precision_summary(r,report['results'][i+2])
                                        for i,r in enumerate(report['results'][:2])}
        report['status']='diagnostic_complete_not_training_or_cache_certification'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if report['results'] and report['results'][-1]['status']=='running':
            report['results'][-1].update(status='failed',error=report['error'])
        raise
    finally:proof.write(report_path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--mode',choices=('2rgb_endpoints','4rgb'),default='2rgb_endpoints')
    p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--check-only',action='store_true',help='Read saved layout/source on CPU; no GPU or model load')
    p.add_argument('--max-new-tokens',type=int,default=160)
    args=p.parse_args();args.source_run=args.source_run.resolve()
    if not 1<=args.max_new_tokens<=512:p.error('Use 1..512 generated tokens')
    policy=source_policy(args.source_run,args.mode);version=Path(policy['version'])
    args.model_dir=(args.model_dir or Path(policy['base_model'])).resolve()
    if args.worker:
        if args.output is None:p.error('worker requires output')
        args.output=args.output.resolve()
        if args.output.is_relative_to(args.source_run) or args.output.is_relative_to(version):
            raise ValueError('Diagnostic output must be outside original run/source')
        return worker(args,version)
    if args.check_only:
        print(json.dumps(dict(status='saved_policy_read_only',policy=policy,model_dir=str(args.model_dir),
                              source_and_weights_verified=False,inference_executed=False),ensure_ascii=False,indent=2))
        return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    idle=[r.split(',')[0].strip() for r in gpu.splitlines() if float(r.split(',')[1])<256 and float(r.split(',')[2])<=5]
    requested=os.environ.get('GPU_IDS');selected=requested if requested else (idle[0] if idle else '')
    if selected not in idle:raise ValueError('One idle physical GPU required; GPU_IDS must be one idle ID')
    output=(args.output or Path('checkpoints/phase12_cache_diagnose')/f'run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}').resolve()
    if output.is_relative_to(args.source_run) or output.is_relative_to(version):
        raise ValueError('Diagnostic output must be outside original run/source')
    archive=Path(str(output)+'.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    output.mkdir(parents=True,exist_ok=False)
    from qwen3vl_local.sft_new_loop_phase1 import data_stage
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--source-run',str(args.source_run),
        '--model-dir',str(args.model_dir),'--mode',args.mode,'--output',str(output),'--max-new-tokens',str(args.max_new_tokens)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':selected,'GPU_IDS':selected,
         'AUTOMOT_TEXT_PREFIX':'1' if policy['input_layout']=='text_before_images_v1' else '0',
         'AUTOMOT_TEXT_PREFIX_CACHE':'1' if policy['text_cache_enabled'] else '0',
         'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):env.pop(name,None)
    try:
        data_stage.run(command,output/'worker.json',timeout=1800,env=env)
    finally:
        with tarfile.open(archive,'x:gz') as tar:
            for name in ('report.json','worker.json','worker.log'):
                if (output/name).is_file():tar.add(output/name,arcname=name,recursive=False)
            for name in ('cache_diagnose.py','cache_checks.py'):
                tar.add(Path(__file__).with_name(name),arcname=name,recursive=False)
        print('[DIAGNOSTIC_BUNDLE] '+str(archive),flush=True)


if __name__=='__main__':main()
