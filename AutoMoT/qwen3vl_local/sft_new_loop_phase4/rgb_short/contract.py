from pathlib import Path

from ..identity import contract as producer_contract, file_sha
from ..observation import observation_contract as original_observation

ROOT = Path(__file__).resolve().parent
POLICY = 'phase4_rgb2_short_student_v1'


def contract():
    return dict(policy=POLICY, producer=producer_contract(),
                exposure_normalizer_sha256=file_sha(ROOT.parents[1] / 'audit_joint/splits.py'), sources={
        name: file_sha(ROOT / name) for name in
        ('__init__.py', 'contract.py', 'data.py', 'model.py', 'train.py', 'evaluate.py', 'build.py', 'run.sh')})


def observation_contract(mode):
    if type(mode) is not int or mode not in (2, 4):
        raise ValueError('student RGB mode must be 2 or 4')
    if mode == 4:
        return original_observation(4)
    return dict(version=POLICY, rgb_mode=2, frame_offsets=[-2, 0],
                frame_rate_hz=4, min_frame=4, order='oldest_to_current')


def check_observation_contract(recorded):
    if not isinstance(recorded, dict) or recorded != observation_contract(recorded.get('rgb_mode')):
        raise ValueError('short-student observation contract mismatch')
    return recorded


def validate_observation(recorded, observation, image_count):
    spec = check_observation_contract(recorded)
    frame = observation['frame_id']
    frames = observation['history_frames']
    if (type(frame) is not int or any(type(f) is not int for f in frames)
            or frames != [frame + offset for offset in spec['frame_offsets']]
            or frames[0] < spec['min_frame'] or image_count != spec['rgb_mode']):
        raise ValueError('short-student RGB history mismatch')


def check_contract(recorded):
    if recorded != contract():
        raise ValueError('student source/producer contract changed; use the original source or a new experiment')
