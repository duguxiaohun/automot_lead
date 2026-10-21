"""One current RGB and four parallel A-I endpoint readouts, with causal history."""
import math
import torch
import torch.nn.functional as F

from . import codec
from .identity import observation_binding

INSTRUCTION = (
    'Predict recorded ego endpoint regions, not driving commands. '
    + '; '.join(f'{a}={b}' for a,b in zip(codec.LETTERS,codec.NAMES)) + '. '
    'At all four future times freeze the CURRENT origin, speed v0 and nearest Driving-lane tangent, '
    'oriented toward ego forward. Its relative angle is supplied; x is forward, y is right. '
    'Lateral displacement below -2 m is LEFT, above +2 m is RIGHT, otherwise CENTER. '
    'r is horizontal straight-line distance from the origin, not path length. '
    'q=r/t-v0; delta=min(1,max(0.5,0.2*v0)) m/s. q<-delta is NEAR; q>delta is FAR; '
    'otherwise REFERENCE. Equality belongs to CENTER/REFERENCE. '
    'NEAR/REFERENCE/FAR are not literal braking/constant speed/acceleration or STOP. '
    'History labels describe past positions at -2,-1.5,-1,-0.5 seconds MINUS the CURRENT '
    'position, using the SAME current speed and current lane direction as the future. '
    'For history r/t uses the POSITIVE lookback duration. LEFT/RIGHT locates the past point '
    'relative to now, not the direction of the past maneuver. UNKNOWN is missing information. '
    'All nine answers are available for every event. Return one letter at each query. '
    'Do not assume the event determines a fixed sequence.'
)


def choice_ids(tokenizer):
    ids = [tokenizer(a, add_special_tokens=False)['input_ids'] for a in codec.LETTERS]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=9:
        raise ValueError('A-I must be distinct existing single tokens')
    return [x[0] for x in ids]


def messages(record, image):
    obs = record['observation']; ref = obs['reference']
    if ref['source'] != codec.REFERENCE_SOURCE: raise ValueError('reference_source_mismatch')
    if len(obs['history']) != 4: raise ValueError('four_history_slots_required')
    lines = [f"Current speed: {obs['speed_mps']:.9g} m/s.",
        f"Current lane tangent relative to ego: {ref['relative_yaw']:.9g} radians.",
        f"Current navigation [target,next,final] in ego meters: {obs['navigation_xy']}.",
        f"Observed event: {record['event']}; road: {record['road_structure']}."]
    for i, (duration, item) in enumerate(zip(codec.LOOKBACKS, obs['history'])):
        if item['duration_s'] != duration: raise ValueError('history_duration_mismatch')
        if item['valid']:
            if type(item['token_id']) is not int or not 0 <= item['token_id'] < 9:
                raise ValueError('invalid_history_token')
            letter = codec.LETTERS[item['token_id']]
            if (item.get('reference_frame') != 'current' or item.get('displacement') != 'past_minus_current'
                    or item['reference_speed_mps'] != obs['speed_mps']
                    or item['reference_relative_yaw'] != ref['relative_yaw']):
                raise ValueError('history_current_reference_mismatch')
            detail = 'past position minus CURRENT origin; current speed and lane reference'
        else: letter, detail = 'UNKNOWN', 'past position evidence unavailable'
        lines.append(f'History endpoint at -{duration:g}s; {detail}:\n[GEOMETRY_HISTORY_{i}]\n{letter}\n[/GEOMETRY_HISTORY_{i}]')
    for i, duration in enumerate(codec.HORIZONS):
        lines.append(f'Endpoint at +{duration:g}s from CURRENT origin.\n[ACTION_QUERY_{i}]')
    return [dict(role='system', content=INSTRUCTION), dict(role='user', content=[
        dict(type='image', image=image), dict(type='text', text='\n'.join(lines))])]


def locate(tokenizer, input_ids):
    ids = input_ids.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids: raise ValueError('expanded_prompt_roundtrip_changed')
    positions = []
    for i in range(4):
        marker = f'[ACTION_QUERY_{i}]'
        if text.count(marker)!=1: raise ValueError('ambiguous_query_marker')
        end = text.index(marker)+len(marker)
        found = [j for j,(a,b) in enumerate(encoded['offset_mapping']) if a<end<=b]
        if len(found)!=1: raise ValueError('query_alignment_failed')
        positions.append(found[0])
    if positions != sorted(set(positions)): raise ValueError('query_order_mismatch')
    return positions, text, encoded['offset_mapping']


def pack(bundle, record, image, *, max_length=8192, max_pixels=442368):
    text = bundle.processor.apply_chat_template(messages(record,image), tokenize=False, add_generation_prompt=False)
    inputs = bundle.processor(text=[text], images=[image], return_tensors='pt', padding=False,
        images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max_pixels}})
    if inputs['input_ids'].shape[1] > max_length: raise ValueError('input_budget_exceeded_no_truncation')
    if len(inputs['image_grid_thw'])!=1: raise ValueError('single_current_RGB_required')
    positions, text, offsets = locate(bundle.tokenizer, inputs['input_ids'][0])
    choices = choice_ids(bundle.tokenizer)
    for i, item in enumerate(record['observation']['history']):
        start = f'[GEOMETRY_HISTORY_{i}]\n'; end = f'\n[/GEOMETRY_HISTORY_{i}]'
        if text.count(start)!=1 or text.count(end)!=1: raise ValueError('history_marker_mismatch')
        a, b = text.index(start)+len(start), text.index(end)
        expected = codec.LETTERS[item['token_id']] if item['valid'] else 'UNKNOWN'
        if text[a:b] != expected: raise ValueError('history_text_changed')
        found = [j for j,(x,y) in enumerate(offsets) if x<b and y>a]
        if item['valid'] and (len(found)!=1 or tuple(offsets[found[0]])!=(a,b) or
                int(inputs['input_ids'][0,found[0]])!=choices[item['token_id']]):
            raise ValueError('history_output_token_identity_mismatch')
    return dict(inputs=inputs, positions=positions, choices=choices,
                tokens=inputs['input_ids'].shape[1])


def readout(bundle, packed):
    inputs = {k:v.to(bundle.device) for k,v in packed['inputs'].items()}
    out = bundle.model(**inputs, logits_to_keep=torch.tensor(packed['positions'], device=bundle.device),
                       use_cache=False, return_dict=True)
    if out.logits.shape[:2] != (1,4): raise ValueError('four_parallel_readouts_required')
    return out.logits[0].float()


def loss_from_logits(logits, targets, choices):
    if logits.ndim!=2 or logits.shape[0]!=4 or len(targets)!=4 or len(choices)!=9:
        raise ValueError('geometry_loss_shape')
    valid = [i for i,t in enumerate(targets) if t['valid']]
    if not valid: raise ValueError('no_valid_targets')
    if any(type(targets[i]['token_id']) is not int or not 0<=targets[i]['token_id']<9 for i in valid):
        raise ValueError('invalid_target_id')
    ids = torch.tensor([choices[targets[i]['token_id']] for i in valid], device=logits.device)
    # Full vocabulary CE: invalid output symbols are penalized, no event mask.
    return F.cross_entropy(logits[valid].float(), ids)


@torch.no_grad()
def predict(bundle, record, image):
    binding = observation_binding(record)
    was_training = bundle.model.training; bundle.model.eval()
    try:
        packed = pack(bundle, record, image); logits = readout(bundle, packed)
    finally:
        bundle.model.train(was_training)
    raw = logits.argmax(-1).tolist(); choices = packed['choices']
    ids = logits[:,choices].argmax(-1).tolist()
    return dict(token_ids=ids, names=[codec.NAMES[i] for i in ids],
        raw_token_ids=raw, format_valid=[i in choices for i in raw],
        probabilities=torch.softmax(logits, -1)[:,choices].tolist(),
        source='predicted', contract=codec.contract(), reference=dict(record['observation']['reference']),
        sample_binding=binding)
