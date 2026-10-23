"""One current image, shared existing letter tokens, four nine-region softmax classification losses."""
import torch
import torch.nn.functional as F
from . import regions
from .dataset import digest, mass
from ..geometry_v1 import codec
from ..geometry_v1.objective import locate, readout, INSTRUCTION as ADAPTIVE_INSTRUCTION


def instruction(variant):
    mapping = '; '.join(f'{chr(65+i)}={name}' for i,name in enumerate(regions.names(variant)))
    if variant == 'a': text = ADAPTIVE_INSTRUCTION
    elif variant in ('b', 'c'):
        text = (f'Predict recorded ego endpoint regions, not commands. {mapping}. '
            'Freeze the CURRENT origin and nearest Driving-lane tangent, oriented toward ego forward. '
            'x is forward, y is right. Lateral displacement below -2 m is LEFT, above +2 m is RIGHT, '
            'otherwise CENTER. r is horizontal straight-line distance, not path length. '
            'Use the supplied inner and outer radii for ALL past and future durations. '
            'r below inner is NEAR, above outer is FAR, otherwise MIDDLE, including equality. '
            'There is no duration or speed scaling and FAR has no upper limit. '
            'MIDDLE means middle distance, not constant speed. These are not braking/acceleration labels. '
            'Past labels locate past endpoints MINUS the CURRENT position, not past maneuver directions. '
            'All nine labels are available for every event. UNKNOWN means missing evidence. ')
    else: raise ValueError('unknown_representation')
    return text + (' History is evidence, not a plan to copy. Use the current image to reassess transitions. '
                   'Return one letter at each query; do not assume a fixed sequence for an event.')


def choice_ids(tokenizer, variant):
    count = len(regions.names(variant))
    ids = [tokenizer(chr(65+i), add_special_tokens=False)['input_ids'] for i in range(count)]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != count:
        raise ValueError('distinct_existing_letter_tokens_required')
    return [x[0] for x in ids]


def allowed_indices(record):
    return list(range(len(regions.names(record['variant']))))


def observation_binding(record):
    keys = ('id', 'scenario', 'route_id', 'frame', 'image', 'event', 'road_structure', 'variant', 'observation')
    return dict(schema='study_sample_observation_and_representation_v1', sample_id=record['id'],
                observation_sha256=digest({k:record[k] for k in keys}))


def messages(record, image):
    v = record['variant']; obs = record['observation']; ref = obs['reference']
    if ref['source'] != codec.REFERENCE_SOURCE or len(obs['history']) != 4:
        raise ValueError('reference_or_history_contract_mismatch')
    lines = [f"Current speed: {obs['speed_mps']:.9g} m/s.",
        f"Current lane tangent relative to ego: {ref['relative_yaw']:.9g} radians.",
        f"Current navigation [target,next,final] in ego meters: {obs['navigation_xy']}.",
        f"Observed event: {record['event']}; road: {record['road_structure']}."]
    if v in ('b','c'):
        lower,upper = regions.bounds(v, record['road_structure'])
        lines.append(f'Fixed inner radius: {lower:g} m; outer radius: {upper:g} m. Current road category frozen.')
    for i,item in enumerate(obs['history']):
        if item['valid'] and (type(item['token_id']) is not int or not 0 <= item['token_id'] < len(regions.names(v))):
            raise ValueError('invalid_history_token')
        token = chr(65+item['token_id']) if item['valid'] else 'UNKNOWN'
        when = f'Past endpoint {-2+i*.5:g}s minus CURRENT origin'
        lines.append(f'{when}:\n[STUDY_HISTORY_{i}]\n{token}\n[/STUDY_HISTORY_{i}]')
    for i,t in enumerate(codec.HORIZONS):
        query = f'Endpoint at +{t:g}s from CURRENT origin'
        lines.append(f'{query}.\n[ACTION_QUERY_{i}]')
    return [dict(role='system', content=instruction(v)), dict(role='user', content=[
        dict(type='image', image=image), dict(type='text', text='\n'.join(lines))])]


def pack(bundle, record, image, *, max_length=8192, max_pixels=442368):
    text = bundle.processor.apply_chat_template(messages(record,image), tokenize=False, add_generation_prompt=False)
    inputs = bundle.processor(text=[text], images=[image], return_tensors='pt', padding=False,
        images_kwargs={'size':{'shortest_edge':1024,'longest_edge':max_pixels}})
    if inputs['input_ids'].shape[1] > max_length: raise ValueError('input_budget_exceeded_no_truncation')
    if len(inputs['image_grid_thw']) != 1: raise ValueError('single_current_RGB_required')
    positions,text,offsets = locate(bundle.tokenizer, inputs['input_ids'][0])
    choices = choice_ids(bundle.tokenizer, record['variant'])
    for i,item in enumerate(record['observation']['history']):
        start = f'[STUDY_HISTORY_{i}]\n'; end = f'\n[/STUDY_HISTORY_{i}]'
        if text.count(start) != 1 or text.count(end) != 1: raise ValueError('history_marker_mismatch')
        a,b = text.index(start)+len(start), text.index(end)
        expected = chr(65+item['token_id']) if item['valid'] else 'UNKNOWN'
        if text[a:b] != expected: raise ValueError('history_text_changed')
        found = [j for j,(x,y) in enumerate(offsets) if x<b and y>a]
        if item['valid'] and (len(found)!=1 or tuple(offsets[found[0]])!=(a,b) or
                int(inputs['input_ids'][0,found[0]]) != choices[item['token_id']]):
            raise ValueError('history_output_token_identity_mismatch')
    return dict(inputs=inputs, positions=positions, choices=choices, tokens=inputs['input_ids'].shape[1])


def loss_from_logits(logits, targets, choices):
    if logits.ndim != 2 or logits.shape[0] != 4 or len(targets) != 4 or len(choices) != 9:
        raise ValueError('study_loss_shape')
    if len(set(choices)) != 9 or any(type(c) is not int or c < 0 or c >= logits.shape[1] for c in choices):
        raise ValueError('distinct_in_range_region_token_ids_required')
    valid = [i for i,t in enumerate(targets) if t['valid']]
    if not valid: raise ValueError('no_valid_targets')
    q = torch.tensor([mass(targets[i],len(choices)) for i in valid],device=logits.device,dtype=torch.float32)
    if q.shape != (len(valid),len(choices)) or not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(-1),torch.ones(len(valid),device=q.device)):
        raise ValueError('invalid_target_distribution')
    return -(q*F.log_softmax(logits[valid][:,choices].float(),dim=-1)).sum(-1).mean()


@torch.no_grad()
def predict(bundle, record, image):
    was_training=bundle.model.training; bundle.model.eval()
    try:
        packed=pack(bundle,record,image); logits=readout(bundle,packed)
    finally: bundle.model.train(was_training)
    choices=packed['choices']; allowed=allowed_indices(record)
    allowed_tokens=[choices[i] for i in allowed]
    ids=[allowed[i] for i in logits[:,allowed_tokens].argmax(-1).tolist()]
    raw=logits.argmax(-1).tolist()
    return dict(token_ids=ids, names=[regions.names(record['variant'])[i] for i in ids],
        raw_token_ids=raw, raw_class_ids=[choices.index(i) if i in choices else -1 for i in raw],
        unrestricted_token_ids=logits[:,choices].argmax(-1).tolist(), allowed_class_ids=allowed,
        letter_format_valid=[i in choices for i in raw], format_valid=[i in allowed_tokens for i in raw],
        probabilities=torch.softmax(logits[:,choices].float(),-1).tolist(),
        probability_space='nine_regions',
        vocabulary_region_mass=torch.softmax(logits.float(),-1)[:,choices].sum(-1).tolist(), source='predicted',
        contract=regions.contract(record['variant']), reference=dict(record['observation']['reference']),
        sample_binding=observation_binding(record))
