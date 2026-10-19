"""Experimental strict four-frame fusion of independently encoded language KV.

Frame language encodings share only an image-free text prefix. Full-attention
KV for the four selected frames is joined for each readout. Query convolution
and DeltaNet recurrence restart from the text-prefix state, NEVER a frame or an
older window's state. This is a new protocol, not native four-image equivalence.
"""
import torch
from .prefix_cache import fork_batch
from qwen3vl_local.sequence_batch import collate
from qwen3vl_local.batch_runtime import frozen_rgb_reuse

FRAME_POSITION_STRIDE=512
PROTOCOL='independent_frame_kv_query_prefix_recurrence_v1'


def boundaries(ids,start,end):
    starts=(ids==start).nonzero().flatten().tolist();ends=(ids==end).nonzero().flatten().tolist()
    if len(starts)!=4 or len(ends)!=4 or any(a>=b for a,b in zip(starts,ends)):
        raise ValueError('Frame bank needs four complete image spans per window')
    return list(zip(starts,[b+1 for b in ends]))


def parts(packed,start,end):
    """Strip observed frames and query text; labels never enter frame encoders."""
    slots={};queries=[];prefix=None
    for window,p in enumerate(packed):
        x=p['inputs'];ids=x['input_ids'][0];spans=boundaries(ids,start,end)
        current=ids[:spans[0][0]]
        if prefix is None:prefix=current
        if not torch.equal(prefix,current):raise ValueError('Frame bank text prefixes differ')
        chunks=x['pixel_values'].split([int(g.prod()) for g in x['image_grid_thw']])
        for j,(a,b) in enumerate(spans):
            frame=dict(input_ids=ids[a:b].unsqueeze(0),attention_mask=torch.ones((1,b-a),dtype=torch.long),
                mm_token_type_ids=x['mm_token_type_ids'][:,a:b],pixel_values=chunks[j],
                image_grid_thw=x['image_grid_thw'][j:j+1])
            slot=window+j
            if slot in slots:
                if any(not torch.equal(frame[k],slots[slot][k]) for k in frame):
                    raise ValueError('Frame bank overlap changed')
            else:slots[slot]=frame
        qstart=spans[-1][1]
        queries.append(dict(ids=ids[qstart:],position=p['positions'][0]-qstart,latest_slot=window+3))
        if queries[-1]['position']<0:raise ValueError('Readout must follow its four images')
    if sorted(slots)!=list(range(8)):raise ValueError('Frame bank needs exactly eight slots')
    return prefix,[slots[i] for i in range(8)],queries


def geometry(prefix,frames,queries,query_positions):
    F=max(frames);Q=max(queries)
    B=max(prefix+sum(frames[i:i+4]) for i in range(5))
    return dict(frame_block_tokens=frames,query_suffix_tokens=queries,
        model_logit_rows=9+5*len(set(query_positions)),shared_prefix_tokens=prefix,
        language_tokens_computed=prefix+8*F+5*Q,
        frame_attention_work=8*F*(prefix+F),query_attention_work=5*Q*(B+Q),
        live_cache_token_slots=prefix+8*(prefix+F)+5*(B+Q),
        max_sequence_tokens=max(prefix+F,B+Q),
        max_rotary_position=max(prefix+(i+4)*FRAME_POSITION_STRIDE+n-1 for i,n in enumerate(queries)))


def describe(packed,start,end):
    prefix,frames,queries=parts(packed,start,end)
    return geometry(len(prefix),[f['input_ids'].shape[1] for f in frames],
                    [len(q['ids']) for q in queries],[q['position'] for q in queries])


def forward(bundle,packed,*,source_slots=False,share_frames=True):
    model=bundle.model
    raw=model.module if isinstance(model,torch.nn.parallel.DistributedDataParallel) else model
    native=raw.get_base_model() if hasattr(raw,'get_base_model') else raw
    if getattr(native.config,'model_type',None)!='qwen3_5':raise ValueError('Frame bank requires native Qwen3.5')
    if raw.training and any(getattr(m,'gradient_checkpointing',False) for m in raw.modules()):
        raise ValueError('Frame bank hybrid cache requires checkpointing disabled')
    if raw.training and any(isinstance(m,torch.nn.Dropout) and m.p>0 or
            getattr(m,'attention_dropout',0)>0 for m in raw.modules()):
        raise ValueError('Frame bank requires deterministic language dropout')
    prefix,frames,queries=parts(packed,native.config.vision_start_token_id,native.config.vision_end_token_id)
    device=bundle.device;prefix=prefix.to(device);P=len(prefix)
    if P==0 or bool((prefix==native.config.image_token_id).any()):raise ValueError('Invalid frame-bank text prefix')
    pre=raw(input_ids=prefix[None],attention_mask=torch.ones((1,P),dtype=torch.long,device=device),
        position_ids=torch.arange(P,device=device).view(1,1,-1).expand(3,1,-1),
        use_cache=True,logits_to_keep=1,return_dict=True)
    # Recompute reference repeats each of the four frame encoders per window.
    # It deliberately keeps the exact SAME protocol/coordinates as shared mode.
    frame_slots=list(range(8)) if share_frames else [i+j for i in range(5) for j in range(4)]
    rows=[frames[s] for s in frame_slots]
    fb={k:v.to(device) for k,v in collate(rows,bundle.tokenizer.pad_token_id).items()}
    F=fb['input_ids'].shape[1]
    positions=[]
    for slot,frame in zip(frame_slots,rows):
        ids=torch.cat((prefix[None],frame['input_ids'].to(device)),dim=1)
        mm=torch.cat((torch.zeros((1,P),dtype=torch.long,device=device),frame['mm_token_type_ids'].to(device)),dim=1)
        pos,_=native.model.get_rope_index(ids,image_grid_thw=frame['image_grid_thw'].to(device),
            attention_mask=torch.ones_like(ids),mm_token_type_ids=mm)
        local=pos[:,:,P:]
        if int(local.max())-P>=FRAME_POSITION_STRIDE:raise ValueError('Image rotary span exceeds fixed frame slot')
        local=local+slot*FRAME_POSITION_STRIDE
        positions.append(torch.nn.functional.pad(local,(0,F-local.shape[-1])))
    fb.update(position_ids=torch.cat(positions,dim=1),past_key_values=fork_batch(pre.past_key_values,len(rows)),
        cache_position=torch.arange(P,P+F,device=device),
        attention_mask=torch.cat((torch.ones((len(rows),P),dtype=torch.long,device=device),fb['attention_mask']),dim=1),
        use_cache=True,logits_to_keep=1,return_dict=True)
    with frozen_rgb_reuse(raw,enabled=not source_slots) as vision:
        encoded=raw(**fb)
    if source_slots:vision.update(images=len(rows),encoded_images=len(rows),reused_images=0,reason='source_slots')
    if vision['images']!=len(rows) or not 1<=vision['encoded_images']<=8:
        raise ValueError('Unexpected frame-bank visual encoding path')
    cache=fork_batch(pre.past_key_values,5)
    lengths=[P+sum(frames[j]['input_ids'].shape[1] for j in range(i,i+4)) for i in range(5)]
    B=max(lengths)
    for layer in cache.transformer_layers:
        for name in ('key_cache','value_cache'):
            original=getattr(pre.past_key_values,name)[layer]
            bank=getattr(encoded.past_key_values,name)[layer]
            branches=[]
            for i in range(5):
                selected=[]
                for j in range(4):
                    slot=i+j;source=slot if share_frames else i*4+j
                    n=frames[slot]['input_ids'].shape[1]
                    selected.append(bank[source:source+1,:,P:P+n])
                joined=torch.cat([original,*selected],dim=2)
                branches.append(torch.nn.functional.pad(joined,(0,0,0,B-joined.shape[2])))
            getattr(cache,name)[layer]=torch.cat(branches,dim=0)
    # No frame recurrence is carried here. The five queries have only the
    # shared TEXT prefix's conv/recurrent state plus their own observed premises.
    Q=max(len(q['ids']) for q in queries)
    ids=torch.full((5,Q),bundle.tokenizer.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros((5,B+Q),dtype=torch.long,device=device);pos=torch.zeros((3,5,Q),dtype=torch.long,device=device)
    for i,q in enumerate(queries):
        n=len(q['ids']);ids[i,:n]=q['ids'].to(device);mask[i,:lengths[i]]=1;mask[i,B:B+n]=1
        origin=P+(q['latest_slot']+1)*FRAME_POSITION_STRIDE
        pos[:,i,:n]=torch.arange(origin,origin+n,device=device)
    if int(pos.max())>=8192:raise ValueError('Frame bank rotary positions exceed 8192')
    keep=sorted(set(q['position'] for q in queries))
    out=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,
        cache_position=torch.arange(B,B+Q,device=device),logits_to_keep=torch.tensor(keep,device=device),
        use_cache=True,return_dict=True)
    logits=torch.stack([out.logits[i,keep.index(q['position'])] for i,q in enumerate(queries)]).float()
    return logits,dict(vision,protocol=PROTOCOL,frame_language_encodings=len(rows),shared_prefix_tokens=P,
        image_language_states_shared=share_frames,query_recurrence='text_prefix_only',
        window_source_slots=[list(range(i,i+4)) for i in range(5)],
        model_logit_rows=1+len(rows)+5*len(keep),
        language_tokens_computed=P+len(rows)*F+5*Q,
        live_cache_token_slots=P+len(rows)*(P+F)+5*(B+Q),
        cache_lifetime='this_forward_only')
