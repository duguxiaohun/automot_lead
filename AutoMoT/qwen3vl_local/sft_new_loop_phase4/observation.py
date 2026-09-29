"""Readable training/inference RGB contract, checked before model execution."""


def observation_contract(rgb_mode):
    if type(rgb_mode) is not int or rgb_mode not in (2,4):
        raise ValueError('invalid RGB mode')
    return dict(version='phase4_causal_rgb_v1',rgb_mode=rgb_mode,
                frame_offsets=[-4,0] if rgb_mode==2 else [-6,-4,-2,0],
                frame_rate_hz=4,min_frame=4,order='oldest_to_current')


def check_observation_contract(recorded):
    if not isinstance(recorded,dict) or recorded != observation_contract(recorded.get('rgb_mode')):
        raise ValueError('missing or incompatible observation contract')
    return recorded


def validate_observation(recorded, observation, image_count):
    c=check_observation_contract(recorded)
    frame=observation['frame_id']
    frames=observation['history_frames']
    if (type(frame) is not int or any(type(f) is not int for f in frames)
            or frames != [frame+o for o in c['frame_offsets']]
            or frames[0] < c['min_frame'] or image_count != c['rgb_mode']):
        raise ValueError('RGB mode/history spacing violates training observation contract')
