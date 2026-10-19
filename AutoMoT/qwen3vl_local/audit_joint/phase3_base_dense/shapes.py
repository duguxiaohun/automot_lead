"""Exact processor-expanded token counts, without loading model weights or RGB arrays."""
from pathlib import Path
from PIL import Image
from . import objectives, baseline


def sequence_shape(processor, record, sizes):
    text=processor.apply_chat_template(objectives.messages(record,['local-image']*len(sizes)),
                                       tokenize=False,add_generation_prompt=False)
    patches=[]
    for width,height in sizes:
        n=processor.image_processor.get_number_of_image_patches(height,width,
            images_kwargs={'min_pixels':1024,'max_pixels':442368})
        patches.append(n)
        text=text.replace(processor.image_token,'<|placeholder|>'*(n//processor.image_processor.merge_size**2),1)
    text=text.replace('<|placeholder|>',processor.image_token)
    tokens=len(processor.tokenizer(text,add_special_tokens=False)['input_ids'])
    return dict(tokens=tokens,patches=sum(patches))


def measure(processor,record,sizes):
    if record['mode']=='windows':
        children=[]
        for i in range(5):
            one=dict(record,mode='future',actions=record['actions'][i:i+1],offsets_seconds=[0])
            if record.get('premises'):one['premises']=record['premises'][i:i+1]
            children.append(sequence_shape(processor,one,sizes[i:i+4]))
    else:children=[sequence_shape(processor,record,sizes)]
    lengths=[s['tokens'] for s in children]
    return dict(sequence_tokens=lengths,max_sequence_tokens=max(lengths),
        padded_batch_tokens=len(lengths)*max(lengths),unpadded_batch_tokens=sum(lengths),
        vision_patch_presentations=sum(s['patches'] for s in children))


def cost(shape):
    return tuple(shape[k] for k in ('max_sequence_tokens','padded_batch_tokens','vision_patch_presentations'))


def scan(processor,records,max_batch_tokens):
    headers={};shapes={};frontier=[]
    for identity,rec in sorted(records.items()):
        sizes=[]
        for path in rec['images']:
            if path not in headers:
                p=Path(path);before=p.stat()
                with Image.open(p) as im:size=im.size
                after=p.stat()
                stamp=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
                if stamp(before)!=stamp(after):raise ValueError('Image changed during shape audit: '+path)
                headers[path]=dict(size=list(size),stat=stamp(after))
            sizes.append(headers[path]['size'])
        shape=measure(processor,rec,sizes)
        if shape['max_sequence_tokens']>8192 or shape['padded_batch_tokens']>max_batch_tokens:
            raise ValueError(f'Planned input {identity} exceeds per-sequence/total padded token budget: {shape}')
        shapes[identity]=shape
        # Probe all nondominated cost shapes. Equal shapes need only one witness.
        value=cost(shape)
        if any(all(a>=b for a,b in zip(cost(shapes[x]),value)) for x in frontier):continue
        frontier=[x for x in frontier if not all(a>=b for a,b in zip(value,cost(shapes[x])))]
        frontier.append(identity)
    if not shapes:raise ValueError('Empty planned input shape audit')
    return dict(status='all_planned_inputs_checked',unique_planned_anchors=len(shapes),
        shapes=shapes,probe_ids=frontier,header_manifest_sha256=baseline.digest(headers),
        per_sequence_limit=8192,padded_batch_limit=max_batch_tokens,
        note='Exact token shape audit; CUDA/DDP memory measured separately. Not a future OOM guarantee.')
