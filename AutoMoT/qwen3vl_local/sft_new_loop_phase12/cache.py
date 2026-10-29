"""Inference only: clone the entire Qwen3.5 hybrid image-prefix state per arm."""
import time
import torch
from qwen3vl_local.engine import _clone_cache
from qwen3vl_local.mrope_utils import qwen3vl_incremental_forward
from qwen3vl_local.sft_v3.train import KVState, _student_generate_kv
from .prompts import messages


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
    if shared:n,prefix=prefill_prefix(bundle.model,*inputs)
    _sync(bundle.device);prefill_s=time.perf_counter()-start;results=[]
    for item in inputs:
        _sync(bundle.device);arm_start=time.perf_counter()
        out=fork(bundle.model,item,n,prefix) if shared else bundle.model(**item,use_cache=True,return_dict=True)
        answer=_decode(bundle,item,out,max_new_tokens)
        del out
        _sync(bundle.device);answer['seconds']=time.perf_counter()-arm_start;results.append(answer)
    if shared and prefix.past_key_values.get_seq_length()!=n:raise ValueError('Branch mutated common prefix')
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
                raise ValueError(f'Shared cache changes generated answer arm={number} field={key}; '
                                 f'shared={str(a[key])[:512]} independent={str(b[key])[:512]}')


@torch.no_grad()
def generate(bundle,images,texts,max_new_tokens=160,*,check_full=False,verify_outputs=False,benchmark_repeats=0):
    if bundle.model.training:raise ValueError('Shared prefix is inference-only; switch model to eval')
    if max_new_tokens<=0 or benchmark_repeats<0:raise ValueError('Invalid generation/benchmark budget')
    device=bundle.device;_sync(device);start=time.perf_counter()
    inputs=[_device(encode(bundle,images,t),device) for t in texts]
    if len(inputs)!=2:raise ValueError('Exactly two independent branches required')
    if any(x['input_ids'].shape[1]>8192 for x in inputs):raise ValueError('Prompt exceeds trained context limit')
    _sync(device);encode_seconds=time.perf_counter()-start
    comparisons=[]
    if check_full:
        n,prefix=prefill_prefix(bundle.model,*inputs)
        for item in inputs:
            out=fork(bundle.model,item,n,prefix)
            full=bundle.model(**item,use_cache=False,return_dict=True)
            atol,rtol=(.08,.03) if out.logits.dtype==torch.bfloat16 else (2e-5,2e-4)
            torch.testing.assert_close(out.logits,full.logits[:,n:],atol=atol,rtol=rtol)
            same=torch.equal(out.logits[:,-1].argmax(-1),full.logits[:,-1].argmax(-1))
            comparisons.append(dict(atol=atol,rtol=rtol,max_abs_difference=float((out.logits-full.logits[:,n:]).abs().max()),next_token_matches=same))
            if not same:raise ValueError('Cache split changes next-token decision')
            del full,out
        del prefix
    # Timed pairs contain no diagnostic full-forward work or processor calls.
    cached=_pair(bundle,inputs,max_new_tokens,True)
    checked=check_full or verify_outputs or bool(benchmark_repeats)
    independent=None;trials=[]
    if checked:
        independent=_pair(bundle,inputs,max_new_tokens,False);assert_answers(cached,independent)
    if benchmark_repeats:
        # Above pairs warm both paths. Alternate execution order to reduce warmup/order bias.
        for repeat in range(benchmark_repeats):
            trial={}
            for shared in ((False,True) if repeat%2==0 else (True,False)):
                answer=_pair(bundle,inputs,max_new_tokens,shared);assert_answers(cached,answer)
                trial['shared_seconds' if shared else 'independent_seconds']=answer['seconds']
            trials.append(trial)
    for number,branch in enumerate(cached['branches']):
        branch['full_comparison']=comparisons[number] if comparisons else {}
    shared_s=sum(t['shared_seconds'] for t in trials)
    separate_s=sum(t['independent_seconds'] for t in trials)
    return dict(branches=cached['branches'],prefix_tokens=cached['prefix_tokens'],
        prefill_seconds=cached['prefill_seconds'],cached_pair_seconds=cached['seconds'],
        encoding_seconds=encode_seconds,total_seconds=time.perf_counter()-start,
        full_forward_equivalence_checked=check_full,full_answer_equivalence_checked=checked,
        independent_branches=independent['branches'] if independent else None,
        paired_benchmark=dict(trials=trials,warmup_pairs_per_method=1 if trials else 0,
            scope='model prefill + complete decode + cache clone; identical pre-encoded inputs; excludes processor and diagnostic forwards',
            saving_fraction=1-shared_s/separate_s if separate_s else None,
            device=str(device),dtype=str(next(bundle.model.parameters()).dtype),max_new_tokens=max_new_tokens),
        cache_scope='one_sample_same_model_same_adapter_inference_only')
