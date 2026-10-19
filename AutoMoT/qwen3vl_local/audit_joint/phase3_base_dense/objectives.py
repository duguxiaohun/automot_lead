"""Parallel choice readouts; target letters are never fed into the transformer.

stream: one causal sequence, eight images, five readouts (prefixes 4..8).
windows: five independent four-image contexts; only frozen vision is shared.
future: current four images, simultaneous forecasts at configured time anchors.
"""
import torch
import torch.nn.functional as F
from .data import ACTIONS, LETTERS
from qwen3vl_local.sft_new_loop_phase3.prompts import SPEED_ACTION_RULES, LANE_ACTION_RULES, PRIMARY_CHOICE_RULES

INSTRUCTION = ('Images are chronological, sampled every 0.25 seconds. At each action query, '
    'choose the appropriate driving action using only images before that query. '
    'A=DECELERATE; B=STOP; C=RESUME; D=LANE_CHANGE_LEFT; E=LANE_CHANGE_RIGHT; F=KEEP. '
    'KEEP means none of the listed actions qualifies. No previous or current ground-truth action is supplied. '
    'Predict the recorded ego behavior, not a recommended safe action. '
    'An action query asks for a choice at its stated time, not the duration of a token. '
    +SPEED_ACTION_RULES+' '+LANE_ACTION_RULES+' '+PRIMARY_CHOICE_RULES)


def choice_ids(tokenizer):
    ids=[tokenizer(x,add_special_tokens=False)['input_ids'] for x in LETTERS]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=6:
        raise ValueError('A-F must be six distinct existing single tokens; vocabulary is not resized')
    return [x[0] for x in ids]


def query(i, offset=0):
    return f'\nChoose the action at +{offset:g} seconds from the latest image.\n[ACTION_QUERY_{i}]\n'


def messages(record, images):
    content=[]
    if record['mode']=='future':
        content.extend(dict(type='image',image=image) for image in images)
        if record.get('premises'):
            content.append(dict(type='text',text=record['premises'][0]+'\nThese premises describe only the current observation.'))
        for i,offset in enumerate(record['offsets_seconds']):
            content.append(dict(type='text',text=query(i,offset)))
    else:
        for i,image in enumerate(images):
            content.append(dict(type='image',image=image))
            if i>=3:
                if record.get('premises'):
                    content.append(dict(type='text',text=record['premises'][i-3]))
                content.append(dict(type='text',text=query(i-3)))
    return [dict(role='system',content=INSTRUCTION),dict(role='user',content=content)]


def locate_queries(tokenizer, input_ids, count):
    # Decode the processor-expanded image-token stream, then obtain offsets.
    # Round-trip equality is mandatory: never silently misalign supervision.
    values=input_ids.tolist()
    text=tokenizer.decode(values,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=values:
        raise ValueError('Tokenizer expanded-input round trip changed IDs')
    positions=[]
    for i in range(count):
        marker=f'[ACTION_QUERY_{i}]'
        if text.count(marker)!=1:
            raise ValueError('Ambiguous/missing query marker')
        end=text.index(marker)+len(marker)
        indices=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if a<end<=b]
        if len(indices)!=1:
            raise ValueError('Cannot locate choice readout')
        positions.append(indices[0])
    if positions!=sorted(set(positions)):
        raise ValueError('Noncausal readout order')
    return positions


def pack(bundle,record,images,*,max_length=8192,max_pixels=442368,max_batch_tokens=16384):
    if len(images)!=(4 if record['mode']=='future' else 8):
        raise ValueError('Wrong number of images')
    if record['mode']=='windows':
        rows=[]
        for i in range(5):
            one=dict(record,mode='future',actions=record['actions'][i:i+1],offsets_seconds=[0])
            if record.get('premises'):one['premises']=record['premises'][i:i+1]
            if 'query_weights' in record:one['query_weights']=record['query_weights'][i:i+1]
            rows.append(pack(bundle,one,images[i:i+4],max_length=max_length,max_pixels=max_pixels))
        # Real right-padded tensor footprint, not just the unpadded sum.
        if len(rows)*max(p['tokens'] for p in rows)>max_batch_tokens:
            raise ValueError('Strict windows exceed total padded batch token budget')
        return rows
    msg=messages(record,images)
    text=bundle.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=False)
    # Qwen3.5 uses patch 16 and merge 2. The processor enforces the pixel cap;
    # the token guard below checks actual output rather than estimating from image count.
    inputs=bundle.processor(text=[text],images=images,return_tensors='pt',padding=False,
                            images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max_pixels}})
    if inputs['input_ids'].shape[1]>max_length:
        raise ValueError('Dense input exceeds token budget; no truncation of images or targets')
    grid=inputs['image_grid_thw']
    if len(grid)!=len(images) or any(int(g.prod())>max_pixels//256+64 for g in grid):
        raise ValueError('Processor did not respect image budget')
    positions=locate_queries(bundle.tokenizer,inputs['input_ids'][0],len(record['actions']))
    token_ids=choice_ids(bundle.tokenizer)
    targets=[-100 if x is None else token_ids[ACTIONS.index(x)] for x in record['actions']]
    result=dict(inputs=inputs,positions=positions,targets=targets,choices=token_ids,
                tokens=int(inputs['input_ids'].shape[1]),images=len(images))
    if 'query_weights' in record:result['query_weights']=record['query_weights']
    return result


def readout_logits(model,packed,device):
    inputs={k:v.to(device) for k,v in packed['inputs'].items()}
    pos=torch.tensor(packed['positions'],device=device)
    out=model(**inputs,logits_to_keep=pos,use_cache=False,return_dict=True)
    return out.logits[0].float()


def readout_loss(model,packed,device):
    logits=readout_logits(model,packed,device)
    labels=torch.tensor(packed['targets'],device=device)
    weights=torch.tensor(packed.get('query_weights',[1.]*len(labels)),device=device,dtype=torch.float32)
    valid=labels.ne(-100)&weights.gt(0)
    if not valid.any():
        raise ValueError('Dense sample has no supported target')
    losses=F.cross_entropy(logits[valid],labels[valid],reduction='none')
    loss=(losses*weights[valid]).sum() if 'query_weights' in packed else losses.mean()
    choices=torch.tensor(packed['choices'],device=device)
    constrained=choices[logits[:,choices].argmax(-1)]
    predicted=logits.argmax(-1)
    return loss,dict(supervised=int(valid.sum()),correct=int((predicted[valid]==labels[valid]).sum()),
                    predictions=predicted.detach().cpu().tolist(),targets=packed['targets'],
                    query_weights=weights.tolist(),loss_mass=float(weights[valid].sum()),
                    constrained_correct=int((constrained[valid]==labels[valid]).sum()),
                    valid_vocabulary=int(torch.isin(logits.argmax(-1),choices)[valid].sum()),
                    tokens=packed['tokens'],images=packed['images'])


def window_logits(bundle,packed):
    """Five language contexts in parallel, eight unique frozen vision encodings.

    Right padding uses Qwen's native 2-D padding mask. Each query is gathered at
    its own position; labels never enter any language context. No language KV reuse.
    """
    from qwen3vl_local.sequence_batch import collate
    from qwen3vl_local.batch_runtime import frozen_rgb_reuse
    if len(packed)!=5 or any(len(p['positions'])!=1 for p in packed):
        raise ValueError('Expected five strict windows')
    rows=[p['inputs'] for p in packed]
    batch={k:v.to(bundle.device) for k,v in collate(rows,bundle.tokenizer.pad_token_id).items()}
    positions=sorted(set(p['positions'][0] for p in packed))
    with frozen_rgb_reuse(bundle.model,enabled=True) as reuse:
        out=bundle.model(**batch,logits_to_keep=torch.tensor(positions,device=bundle.device),
                        use_cache=False,return_dict=True)
    if not (1<=reuse.get('encoded_images',0)<=8) or reuse.get('images')!=20 or reuse.get('encoded_images',0)+reuse.get('reused_images',0)!=20:
        raise ValueError('Strict window vision reuse did not share the eight-image source')
    logits=torch.stack([out.logits[i,positions.index(p['positions'][0])] for i,p in enumerate(packed)]).float()
    return logits,dict(reuse)


def window_loss(bundle,packed):
    logits,reuse=window_logits(bundle,packed)
    labels=torch.tensor([p['targets'][0] for p in packed],device=bundle.device)
    weights=torch.tensor([p.get('query_weights',[1.])[0] for p in packed],device=bundle.device)
    valid=labels.ne(-100)&weights.gt(0)
    if not valid.any():raise ValueError('No supported window target')
    choices=torch.tensor(packed[0]['choices'],device=bundle.device)
    constrained=choices[logits[:,choices].argmax(-1)]
    predicted=logits.argmax(-1)
    losses=F.cross_entropy(logits[valid],labels[valid],reduction='none')
    loss=(losses*weights[valid]).sum() if 'query_weights' in packed[0] else losses.mean()
    return loss,dict(supervised=int(valid.sum()),correct=int((predicted[valid]==labels[valid]).sum()),
        predictions=predicted.detach().cpu().tolist(),targets=labels.cpu().tolist(),
        query_weights=weights.tolist(),loss_mass=float(weights[valid].sum()),
        constrained_correct=int((constrained[valid]==labels[valid]).sum()),
        tokens=sum(p['tokens'] for p in packed),images=8,visual_reuse=dict(reuse))


@torch.no_grad()
def predict(bundle,record,images,*,max_batch_tokens=16384,expected_tokens=None):
    """Label-free inference for the exact training protocol; no generate() swap.

    Caller supplies the same observed premises and image order. Targets are
    explicitly erased before packing, and no loss or teacher forcing is used.
    """
    count=len(record['offsets_seconds']) if record['mode']=='future' else 5
    unlabeled=dict(record,actions=[None]*count)
    packed=pack(bundle,unlabeled,images,max_batch_tokens=max_batch_tokens)
    lengths=[p['tokens'] for p in packed] if isinstance(packed,list) else [packed['tokens']]
    if expected_tokens is not None and lengths!=expected_tokens:
        raise ValueError('Evaluation input changed since resource admission')
    logits,_=window_logits(bundle,packed) if isinstance(packed,list) else (
        readout_logits(bundle.model,packed,bundle.device),None)
    ids=choice_ids(bundle.tokenizer)
    raw=logits.argmax(-1).tolist()
    constrained=logits[:,ids].argmax(-1).tolist()
    return [dict(action=ACTIONS[ids.index(token)] if token in ids else None,
                 raw_token_id=token,constrained_action=ACTIONS[c],format_valid=token in ids)
            for token,c in zip(raw,constrained)]
