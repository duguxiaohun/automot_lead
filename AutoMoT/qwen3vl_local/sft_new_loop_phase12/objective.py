"""Full-gradient supervision; inference cache is never used for training."""
import torch
from qwen3vl_local.sft_new_loop_phase2 import train as native
from qwen3vl_local.qwen35.backend import assistant_header_text
from qwen3vl_local.sft_v2.train import _find_subsequence, _assert_inside_assistant_turn
from .cache import encode


def pack(bundle,images,text,target,spec,max_length=8192):
    inputs=encode(bundle,images,text,target)
    ids=inputs['input_ids'][0]
    if len(ids)>max_length:raise ValueError(f'Context exceeds {max_length}: {len(ids)}; no truncation or skip')
    target_ids,weights,components=native._target_token_weights(bundle,target,spec=spec,format_loss_weight=.25)
    expanded=ids.tolist();header=bundle.tokenizer(assistant_header_text(bundle.processor),add_special_tokens=False)['input_ids']
    start=_find_subsequence(expanded,header,0,last=True)+len(header)
    _assert_inside_assistant_turn(expanded,start,header,0)
    if expanded[start:start+len(target_ids)]!=target_ids:raise ValueError('Assistant supervision boundary mismatch')
    w=torch.zeros_like(ids,dtype=torch.float32);c=torch.zeros_like(ids)
    w[start:start+len(weights)]=torch.tensor(weights);c[start:start+len(weights)]=torch.tensor(components)
    end=start+len(weights);ends=native._assistant_end_token_ids(bundle)
    if end>=len(ids) or int(ids[end]) not in ends:raise ValueError('Missing assistant terminator supervision')
    w[end]=.25;c[end]=-1
    if any(not torch.any(c==i) for i in range(1,len(spec.output_keys)+1)):
        raise ValueError('At least one requested answer has no semantic supervision')
    return dict(input_ids=ids,attention_mask=inputs['attention_mask'][0],labels=ids.clone(),loss_weights=w,
                component_ids=c,vision={k:v for k,v in inputs.items() if k not in ('input_ids','attention_mask')})


def loss(bundle,packed,spec):
    # Native weighted token objective works for any output_keys, including P1's eight.
    value,stats=native._loss_one(bundle,packed,spec)
    if stats.get('denom',0)<=0:raise ValueError('Empty supervised loss')
    return value,stats
