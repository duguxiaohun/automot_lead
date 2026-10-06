"""Identify visible model inputs, excluding hidden annotation and instance metadata."""
import hashlib
from .controller import Episode
from .identity import digest
from .route_prompts import messages, prompt_version

POLICY = 'ordered_decoded_rgb_messages_v1'


def rgb_content_sha(image):
    """Match model.load_images: RGB pixels plus dimensions, not JPEG metadata."""
    rgb = image.convert('RGB')
    h = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def model_input_key(row):
    hashes = row['image_rgb_sha256']
    if (not isinstance(hashes,list) or len(hashes)!=len(row['images'])
            or len(hashes)!=len(row['image_sha256'])
            or any(not isinstance(h,str) or len(h)!=64 or
                   any(c not in '0123456789abcdef' for c in h) for h in hashes)):
        raise ValueError('invalid decoded RGB identity')
    # Same message construction as training/inference, with image objects replaced
    # by their decoded content identity. Neither target nor instance_id is rendered.
    msgs = messages(Episode(**row['episode']),row['edge'],row['observation'],
                    [dict(rgb_sha256=h) for h in hashes])
    if digest(msgs[-1]['content'][-1]['text']) != row['prompt_sha256']:
        raise ValueError('prompt identity mismatch')
    return digest(dict(policy=POLICY,prompt_version=prompt_version(Episode(**row['episode']),row['edge']),messages=msgs))


def validate_model_inputs(rows):
    seen = {}
    count = 0
    for row in rows:
        key = model_input_key(row)
        if row.get('model_input_sha256') != key:
            raise ValueError('model input identity mismatch')
        if row['target'] not in ('YES','NO'):
            continue
        count += 1
        if key in seen and seen[key]['target'] != row['target']:
            old = seen[key]
            raise ValueError('conflicting answers for identical model input: '
                             f"{old['id']} ({old.get('evidence_id')}) vs "
                             f"{row['id']} ({row.get('evidence_id')}); review participant scope")
        seen.setdefault(key,row)
    return dict(policy=POLICY,questions=count,unique_inputs=len(seen),
                same_answer_duplicates=count-len(seen),conflicting_inputs=0)
