"""Differentiable sharing of text-only prefix state across strict image windows.

Only identical tokens before the FIRST image are cached. Each suffix gets its
own KV/conv/DeltaNet branch; no image-bearing state crosses window boundaries.
All caches and their autograd graph live inside one optimizer forward.
"""
import copy
import torch


def fork_batch(cache,batch):
    branch=copy.copy(cache)
    for name in ('key_cache','value_cache','conv_states','recurrent_states'):
        if not hasattr(cache,name):raise ValueError('Unsupported hybrid cache')
        values=getattr(cache,name)
        setattr(branch,name,[None if x is None else torch.cat([x]*batch,dim=0) for x in values])
    return branch


def forward(model,batch,positions,prefix):
    raw=model.module if isinstance(model,torch.nn.parallel.DistributedDataParallel) else model
    native=raw.get_base_model() if hasattr(raw,'get_base_model') else raw
    if getattr(native.config,'model_type',None)!='qwen3_5':raise ValueError('Expected native Qwen3.5')
    if raw.training and any(getattr(m,'gradient_checkpointing',False) for m in raw.modules()):
        raise ValueError('Mutable hybrid cache requires checkpointing disabled in windows_prefix mode')
    if raw.training and any(isinstance(m,torch.nn.Dropout) and m.p>0 or
            getattr(m,'attention_dropout',0)>0 for m in raw.modules()):
        raise ValueError('Shared-prefix equivalence requires deterministic language dropout')
    ids=batch['input_ids'];count,length=ids.shape
    if not 0<prefix<min(positions):raise ValueError('Invalid shared prefix boundary')
    if (not torch.equal(ids[:,:prefix],ids[:1,:prefix].expand(count,-1)) or
            not bool(batch['attention_mask'][:,:prefix].all()) or
            bool(batch['mm_token_type_ids'][:,:prefix].any())):
        raise ValueError('Shared prefix must be identical, valid and image-free')
    if bool(((ids[:,:prefix]==native.config.image_token_id)|
             (ids[:,:prefix]==native.config.vision_start_token_id)).any()):
        raise ValueError('Image-bearing state must never enter shared prefix')
    position_ids,_=native.model.get_rope_index(ids,image_grid_thw=batch['image_grid_thw'],
        attention_mask=batch['attention_mask'],mm_token_type_ids=batch['mm_token_type_ids'])
    if not torch.equal(position_ids[:,:,:prefix],position_ids[:,:1,:prefix].expand(-1,count,-1)):
        raise ValueError('Shared text prefix rotary positions differ')
    # Prefix bypasses DDP's forward bookkeeping, while the single suffix forward
    # prepares its reducer for the entire shared autograd graph (tested on gloo).
    pre=raw(input_ids=ids[:1,:prefix],attention_mask=batch['attention_mask'][:1,:prefix],
        position_ids=position_ids[:,:1,:prefix],cache_position=torch.arange(prefix,device=ids.device),
        logits_to_keep=1,use_cache=True,return_dict=True)
    cache=fork_batch(pre.past_key_values,count)
    suffix=dict(batch)
    for key in ('input_ids','mm_token_type_ids','token_type_ids'):
        if key in suffix:suffix[key]=suffix[key][:,prefix:]
    suffix.update(position_ids=position_ids[:,:,prefix:],past_key_values=cache,
        cache_position=torch.arange(prefix,length,device=ids.device),
        logits_to_keep=torch.tensor(positions,device=ids.device)-prefix,use_cache=True,return_dict=True)
    out=model(**suffix)
    return out,dict(language_cache_scope='identical_text_before_first_image_only',
        shared_prefix_tokens=prefix,language_tokens_computed=prefix+count*(length-prefix),
        language_tokens_without_sharing=count*length,cache_lifetime='this_forward_only',
        image_language_states_shared=False)
