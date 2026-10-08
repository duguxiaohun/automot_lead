"""Check RGB2/RGB4 bindings and complete training pairs without loading rows en masse."""
import json
from pathlib import Path

from .io import digest, file_sha


def check_pair(paths, expected_manifest_sha=None):
    from qwen3vl_local.sft_new_loop_phase4.paired_training import semantic
    from qwen3vl_local.sft_new_loop_phase4.observation import observation_contract
    roots = {mode: Path(paths[str(mode)]).resolve() for mode in (2, 4)}
    if roots[2] == roots[4]:
        raise ValueError('RGB2 and RGB4 must be distinct dataset directories')
    hashes = {mode: file_sha(root / 'manifest.json') for mode, root in roots.items()}
    if expected_manifest_sha is not None and hashes != expected_manifest_sha:
        raise ValueError('paired manifests changed after artifact verification')
    manifests = {mode: json.loads((root / 'manifest.json').read_text()) for mode, root in roots.items()}
    for mode, manifest in manifests.items():
        if manifest['rgb_mode'] != mode or manifest['observation_contract'] != observation_contract(mode):
            raise ValueError(f'wrong RGB{mode} dataset mode/observation contract')
    for field in ('contract', 'seed', 'candidate_pool_sha256', 'production_index_sha256',
                  'teacher_registry_sha256', 'exposure_sha256'):
        if manifests[2].get(field) != manifests[4].get(field):
            raise ValueError('paired dataset binding differs: ' + field)
    def fingerprints(mode):
        path = roots[mode] / 'train.jsonl'
        count = 0
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['split'] != 'train' or row['target'] not in ('YES', 'NO'):
                    raise ValueError('invalid paired training split/answer')
                frames = row['observation']['history_frames']
                current = row['observation']['frame_id']
                if frames != [current + offset for offset in observation_contract(mode)['frame_offsets']]:
                    raise ValueError('wrong paired history frames')
                images = dict(zip(frames, zip(row['images'], row['image_sha256'], row['image_rgb_sha256'], strict=True), strict=True))
                shared = [[frame, images[frame]] for frame in (current - 4, current)]
                count += 1
                yield row['id'], digest([semantic(row), shared])
        if count != manifests[mode]['counts']['train']:
            raise ValueError('paired training count differs from manifest')
        if file_sha(path) != manifests[mode]['files']['train.jsonl']:
            raise ValueError('paired training input SHA changed')
    index = {}
    for ident, sha in fingerprints(2):
        if ident in index:
            raise ValueError('duplicate RGB2 training ID')
        index[ident] = sha
    count = len(index)
    for ident, sha in fingerprints(4):
        if index.pop(ident, None) != sha:
            raise ValueError('unpaired/duplicate ID, differing semantics or shared RGB: ' + str(ident))
    if not count or index:
        raise ValueError('incomplete or empty RGB2/RGB4 training pairing')
    if any(file_sha(roots[mode] / 'manifest.json') != sha for mode, sha in hashes.items()):
        raise ValueError('paired manifests changed during pairing')
    return dict(status='verified', pairs=count, mode_directories={str(k): str(v) for k, v in roots.items()},
                scope='All training IDs, semantics and shared RGB identities; does not evaluate label accuracy or model outputs.')
