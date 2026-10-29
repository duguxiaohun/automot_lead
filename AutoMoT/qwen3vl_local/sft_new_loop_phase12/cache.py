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


@torch.no_grad()
def generate(bundle,images,texts,max_new_tokens=160,*,check_full=False):
    if bundle.model.training:raise ValueError('Shared prefix is inference-only; switch model to eval')
    device=bundle.device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    sync();start=time.perf_counter()
    inputs=[_device(encode(bundle,images,t),device) for t in texts]
    if len(inputs)!=2:raise ValueError('Exactly two independent branches required')
    if any(x['input_ids'].shape[1]>8192 for x in inputs):raise ValueError('Prompt exceeds trained context limit')
    n,prefix=prefill_prefix(bundle.model,*inputs)
    sync();prefill_s=time.perf_counter()-start
    results=[]
    for item in inputs:
        sync();arm_start=time.perf_counter();out=fork(bundle.model,item,n,prefix)
        ids=item['input_ids'];state=KVState(ids,ids,item['attention_mask'],out.past_key_values,
            out.rope_deltas,out.logits[:,-1,:])
        comparison={}
        if check_full:
            full=bundle.model(**item,use_cache=False,return_dict=True)
            atol,rtol=(.08,.03) if out.logits.dtype==torch.bfloat16 else (2e-5,2e-4)
            torch.testing.assert_close(out.logits,full.logits[:,n:],atol=atol,rtol=rtol)
            same_next=torch.equal(out.logits[:,-1].argmax(-1),full.logits[:,-1].argmax(-1))
            comparison=dict(atol=atol,rtol=rtol,max_abs_difference=float((out.logits-full.logits[:,n:]).abs().max()),next_token_matches=same_next)
            if not same_next:raise ValueError('Cache split changes the next-token decision; refusing silent inference drift')
            del full
        text,_,tokens=_student_generate_kv(bundle,state,max_new_tokens)
        sync();results.append(dict(text=text,generated_tokens=int(tokens.numel()),seconds=time.perf_counter()-arm_start,
                                  full_comparison=comparison))
    if prefix.past_key_values.get_seq_length()!=n:raise ValueError('Branch mutated common prefix')
    return dict(branches=results,prefix_tokens=n,prefill_seconds=prefill_s,
        total_seconds=time.perf_counter()-start,full_forward_equivalence_checked=check_full,
        cache_scope='one_sample_same_model_same_adapter_inference_only')
