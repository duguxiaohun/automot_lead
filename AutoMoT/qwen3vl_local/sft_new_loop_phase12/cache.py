"""Inference only: clone the entire Qwen3.5 hybrid image-prefix state per arm."""
import time
import contextlib
import torch
from qwen3vl_local.engine import _clone_cache
from qwen3vl_local.mrope_utils import qwen3vl_incremental_forward
from qwen3vl_local.sft_v3.train import KVState, _student_generate_kv
from .prompts import messages
from . import cache_checks
from qwen3vl_local.qwen35 import text_prefix

BENCHMARK_PROTOCOL='paired_prefix_timing_v2'

def benchmark_protocol():
    return 'paired_text_prefix_cold_pair_v2' if text_prefix.enabled() else BENCHMARK_PROTOCOL


@contextlib.contextmanager
def isolated_text_cache(bundle):
    """Diagnostics/replays cannot warm or change the production stream cache."""
    if not text_prefix.enabled():
        yield
        return
    previous=getattr(bundle,'_text_prefix_cache',None)
    source=previous or text_prefix.TextPrefixCache()
    bundle._text_prefix_cache=text_prefix.TextPrefixCache(source.max_entries,source.max_bytes,source.max_tokens)
    try:
        yield
    finally:
        if previous is None:delattr(bundle,'_text_prefix_cache')
        else:bundle._text_prefix_cache=previous
BENCHMARK_SCOPE='model prefill + complete decode + cache clone; identical pre-encoded inputs; excludes processor and diagnostic forwards'


def encode(bundle, images, text, target=None):
    chat=bundle.processor.apply_chat_template(messages(images,text,target),tokenize=False,
                                              add_generation_prompt=target is None)
    return bundle.processor(text=[chat],images=images,return_tensors='pt',padding=False)


def prefix_length(a,b,vision_end_id):
    x,y=a['input_ids'],b['input_ids']
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=1 or y.shape[0]!=1:raise ValueError('Prefix fork requires unpadded batch size one')
    ends=torch.where(x[0]==vision_end_id)[0]
    if not len(ends):raise ValueError('No image boundary')
    n=int(ends[-1])+1
    if n>=min(x.shape[1],y.shape[1]) or not torch.equal(x[:,:n],y[:,:n]):
        raise ValueError('Different system/image prefix; refusing cache reuse')
    if not all(torch.all(v.get('attention_mask',torch.ones_like(v['input_ids']))==1) for v in (a,b)):
        raise ValueError('Padded cache branches are not supported')
    for key in ('pixel_values','image_grid_thw'):
        if key not in a or key not in b or not torch.equal(a[key],b[key]):
            raise ValueError('Image preprocessing differs: '+key)
    if torch.any(y[:,n:]==vision_end_id):raise ValueError('Unexpected extra image in task suffix')
    return n


def _device(inputs,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}


def prefill_prefix(model,a,b):
    n=prefix_length(a,b,model.config.vision_end_token_id)
    inputs=dict(a)
    for key in ('input_ids','attention_mask','mm_token_type_ids','token_type_ids'):
        if key in inputs:inputs[key]=inputs[key][:,:n]
    outputs=model(**inputs,use_cache=True,return_dict=True)
    return n,outputs


def fork(model,inputs,n,prefix):
    return qwen3vl_incremental_forward(model,feed_ids=inputs['input_ids'][:,n:],
        attention_mask=inputs['attention_mask'],past_key_values=_clone_cache(prefix.past_key_values),
        prefix_len=n,rope_deltas=prefix.rope_deltas.clone() if prefix.rope_deltas is not None else None)


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def _decode(bundle,item,out,max_new_tokens):
    from qwen3vl_local.sft_v3.train import _apply_repetition_penalty, _REPETITION_PENALTY
    ids=item['input_ids'];state=KVState(ids,ids,item['attention_mask'],out.past_key_values,
        out.rope_deltas,out.logits[:,-1,:])
    text,state,tokens=_student_generate_kv(bundle,state,max_new_tokens)
    # Native generation excludes EOS from returned tokens. Bind the terminal decision too.
    stop=None
    if tokens.numel()<max_new_tokens:
        logits=_apply_repetition_penalty(state.next_logits,torch.unique(state.decoded_input_ids),_REPETITION_PENALTY)
        stop=int(logits.argmax(-1).item())
    return dict(text=text,generated_tokens=int(tokens.numel()),token_ids=tokens.flatten().tolist(),
                stop_token_id=stop,reached_token_limit=tokens.numel()==max_new_tokens)


def _pair(bundle,inputs,max_new_tokens,shared):
    _sync(bundle.device);start=time.perf_counter();n=0;prefix=None
    text_shared=shared and text_prefix.enabled()
    if shared and not text_shared:n,prefix=prefill_prefix(bundle.model,*inputs)
    _sync(bundle.device);prefill_s=time.perf_counter()-start;results=[]
    for item in inputs:
        _sync(bundle.device);arm_start=time.perf_counter()
        if text_shared:
            out=text_prefix.prefill(bundle,item)
            n=text_prefix.bank(bundle).last['prefix_tokens'] if text_prefix.cache_enabled() else 0
        else:
            out=fork(bundle.model,item,n,prefix) if shared else bundle.model(**item,use_cache=True,return_dict=True)
        answer=_decode(bundle,item,out,max_new_tokens)
        del out
        _sync(bundle.device);answer['seconds']=time.perf_counter()-arm_start
        if text_shared and text_prefix.cache_enabled():answer['text_cache']=dict(text_prefix.bank(bundle).last)
        results.append(answer)
    if prefix is not None and prefix.past_key_values.get_seq_length()!=n:raise ValueError('Branch mutated common prefix')
    _sync(bundle.device)
    return dict(branches=results,prefix_tokens=n,prefill_seconds=prefill_s,seconds=time.perf_counter()-start)


def assert_answers(cached,independent):
    if len(cached['branches'])!=2 or len(independent['branches'])!=2:raise ValueError('Missing paired output')
    for number,(a,b) in enumerate(zip(cached['branches'],independent['branches']),1):
        for branch in (a,b):
            ids=branch['token_ids'];stop=branch['stop_token_id'];limit=branch['reached_token_limit']
            if (not isinstance(ids,list) or any(type(i)!=int or i<0 for i in ids)
                    or branch['generated_tokens']!=len(ids) or type(limit)!=bool
                    or (stop is not None if limit else type(stop)!=int)):
                raise ValueError('Invalid generated-token/termination evidence')
        for key in ('token_ids','stop_token_id','reached_token_limit','text'):
            if a[key]!=b[key]:
                raise cache_checks.CacheParityError(cached,independent)


@torch.no_grad()
def generate(bundle,images,texts,max_new_tokens=160,*,check_full=False,verify_outputs=False,benchmark_repeats=0,other_images=None):
    if bundle.model.training:raise ValueError('Shared prefix is inference-only; switch model to eval')
    if max_new_tokens<=0 or benchmark_repeats<0:raise ValueError('Invalid generation/benchmark budget')
    device=bundle.device;_sync(device);start=time.perf_counter()
    inputs=[_device(encode(bundle,images,t),device) for t in texts]
    if len(inputs)!=2:raise ValueError('Exactly two independent branches required')
    if any(x['input_ids'].shape[1]>8192 for x in inputs):raise ValueError('Prompt exceeds trained context limit')
    _sync(device);encode_seconds=time.perf_counter()-start
    # Exactly one production pass, before diagnostics or timing replay. Its cold
    # starts and question switches are included in the primary stream measurement.
    before=dict(text_prefix.bank(bundle).stats) if text_prefix.cache_enabled() else {}
    cached=_pair(bundle,inputs,max_new_tokens,True)
    after=dict(text_prefix.bank(bundle).stats) if text_prefix.cache_enabled() else {}
    checked=check_full or verify_outputs or bool(benchmark_repeats)
    independent=None;trials=[];comparisons=[];recomputed=None;structures=[];resident_stream=None
    text_acceptance=text_prefix.cache_enabled()
    if checked:
        independent=_pair(bundle,inputs,max_new_tokens,False)
        if text_acceptance:
            assert_answers(independent,independent)
            # Rebuild each text prefix independently, without borrowing production state.
            # This is a correctness control, NEVER the speed baseline.
            branches=[]
            for item in inputs:
                with isolated_text_cache(bundle):
                    out=text_prefix.prefill(bundle,item)
                    branches.append(_decode(bundle,item,out,max_new_tokens))
                    del out
            recomputed=dict(branches=branches)
        try:assert_answers(cached,recomputed if text_acceptance else independent)
        except cache_checks.CacheParityError as exc:
            exc.evidence.update(input_layout=text_prefix.identity(),text_cache_enabled=text_prefix.cache_enabled(),
                dtype=str(next(bundle.model.parameters()).dtype),device=str(device),
                input_tokens=[int(item['input_ids'].shape[1]) for item in inputs],max_new_tokens=max_new_tokens)
            raise
    if check_full:
        if text_acceptance:
            import sys
            from .cache_structure import measure,verify_reuse,measure_stream
            if other_images is None:raise ValueError('Text acceptance requires alternate real images')
            others=[_device(encode(bundle,other_images,t),device) for t in texts]
            resident_stream=measure_stream(bundle,inputs,others,sys.modules[__name__],max_new_tokens=max_new_tokens)
            for item,other in zip(inputs,others):
                structure=measure(bundle,item,sys.modules[__name__],other_item=other)
                structures.append(structure)
                try:verify_reuse(structure,require_cross=True)
                except ValueError as exc:
                    exc.evidence=dict(text_reuse_structure=structures,cached=cached,
                                      independent=independent,recomputed_split=recomputed)
                    raise
        with isolated_text_cache(bundle):
            prefix=None
            if not text_prefix.enabled():n,prefix=prefill_prefix(bundle.model,*inputs)
            for item in inputs:
                if text_prefix.enabled():
                    out=text_prefix.prefill(bundle,item)
                    n=text_prefix.bank(bundle).last['prefix_tokens'] if text_prefix.cache_enabled() else 0
                else:
                    out=fork(bundle.model,item,n,prefix)
                full=bundle.model(**item,use_cache=False,return_dict=True)
                diagnostic=cache_checks.logits(out.logits,full.logits[:,n:],diagnostic_only=text_acceptance)
                comparisons.append(diagnostic)
                if not diagnostic['all_close']:
                    print('[cache-logits:DIAGNOSTIC] prompt logits differ; see versioned acceptance contract: '+str(diagnostic),flush=True)
                del full,out
            del prefix
    if benchmark_repeats:
        # Text-prefix probes start from EMPTY isolated caches, including rebuild cost.
        # Legacy image-prefix probes keep their historical timing contract.
        for repeat in range(benchmark_repeats):
            trial={}
            for shared in ((False,True) if repeat%2==0 else (True,False)):
                with isolated_text_cache(bundle):
                    answer=_pair(bundle,inputs,max_new_tokens,shared)
                assert_answers((cached if shared else independent) if text_acceptance else cached,answer)
                trial['shared_seconds' if shared else 'independent_seconds']=answer['seconds']
            trials.append(trial)
    for number,branch in enumerate(cached['branches']):
        branch['full_comparison']=comparisons[number] if comparisons else {}
    shared_s=sum(t['shared_seconds'] for t in trials)
    separate_s=sum(t['independent_seconds'] for t in trials)
    return dict(branches=cached['branches'],prefix_tokens=cached['prefix_tokens'],
        prefill_seconds=cached['prefill_seconds'],cached_pair_seconds=cached['seconds'],
        independent_pair_seconds=independent['seconds'] if independent else None,
        encoding_seconds=encode_seconds,total_seconds=time.perf_counter()-start,
        full_forward_equivalence_checked=check_full,full_answer_equivalence_checked=checked,
        full_forward_equivalence_passed=all(r['all_close'] for r in comparisons) if check_full else None,
        cache_acceptance_protocol=cache_checks.protocol(),
        full_answer_equivalence_passed=(cache_checks.first_difference(cached,independent) is None) if checked else None,
        full_answer_difference=cache_checks.first_difference(cached,independent) if checked else None,
        same_partition_equivalence_checked=checked and text_acceptance,
        recomputed_split_branches=recomputed['branches'] if recomputed else None,
        text_reuse_structure=structures,resident_text_stream=resident_stream,
        independent_branches=independent['branches'] if independent else None,
        paired_benchmark=dict(trials=trials,warmup_pairs_per_method=1 if trials and not text_prefix.enabled() else 0,
            protocol=benchmark_protocol(),scope=BENCHMARK_SCOPE,batch_size=1,same_encoded_inputs=True,
            synchronized=device.type=='cuda',do_sample=False,repetition_penalty=1.05,
            execution_order=[['independent','shared'] if i%2==0 else ['shared','independent'] for i in range(len(trials))],
            saving_fraction=1-shared_s/separate_s if separate_s else None,
            device=str(device),dtype=str(next(bundle.model.parameters()).dtype),max_new_tokens=max_new_tokens),
        stream_benchmark=dict(protocol='ordered_production_stream_v1',
            shared_seconds=cached['seconds'],independent_seconds=independent['seconds'] if independent else None,
            encoding_seconds=encode_seconds,cache_before=before,cache_after=after,
            cache_enabled=text_prefix.cache_enabled(),requests=len(inputs),
            scope='one production pass in original order; includes cache builds, switches and clone; excludes processor, baseline and diagnostics'),
        text_prefix=text_prefix.report(bundle),
        cache_scope=('process_local_cross_image_text_prefix' if text_prefix.enabled() else
                     'one_sample_same_model_same_adapter_inference_only'))


def verify_benchmark(meta,*,device,dtype,max_new_tokens=160):
    trials=meta['trials'];count=len(trials)
    expected=dict(protocol=benchmark_protocol(),scope=BENCHMARK_SCOPE,batch_size=1,
        same_encoded_inputs=True,synchronized=str(device).startswith('cuda'),do_sample=False,
        repetition_penalty=1.05,device=device,dtype=dtype,max_new_tokens=max_new_tokens,
        warmup_pairs_per_method=1 if count and not text_prefix.enabled() else 0,
        execution_order=[['independent','shared'] if i%2==0 else ['shared','independent'] for i in range(count)])
    for key,value in expected.items():
        if key not in meta or type(meta[key])!=type(value) or meta[key]!=value:
            raise ValueError('Missing/inconsistent benchmark condition: '+key)
    if count not in (0,2):raise ValueError('Expected two alternating timing trials')
    for trial in trials:
        if (set(trial)!={'shared_seconds','independent_seconds'}
                or any(type(v) not in (float,int) or not 0<v<float('inf') for v in trial.values())):
            raise ValueError('Invalid benchmark time')
    baseline=sum(t['independent_seconds'] for t in trials)
    saving=1-sum(t['shared_seconds'] for t in trials)/baseline if baseline else None
    if 'saving_fraction' not in meta or meta['saving_fraction']!=saving:raise ValueError('Benchmark saving mismatch')
    return True
