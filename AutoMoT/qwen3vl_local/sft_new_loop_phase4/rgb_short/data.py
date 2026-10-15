"""Small, source-bound input views over one immutable native RGB4 dataset.

The original four-image labels remain reference targets. Two-image visibility
is unreviewed; no teacher proof or human RGB certification is fabricated.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from .. import dataset as native
from ..controller import Episode
from ..identity import digest, file_sha
from ..input_identity import model_input_key
from ..route_prompts import prompt
from ..observation import observation_contract as producer_observation, validate_observation as validate_producer
from .contract import POLICY, contract, observation_contract, validate_observation


def short_row(original):
    validate_producer(producer_observation(4), original['observation'], len(original['images']))
    row = deepcopy(original)
    row['observation']['history_frames'] = original['observation']['history_frames'][2:]
    for field in ('images', 'image_sha256', 'image_rgb_sha256'):
        if len(original[field]) != 4:
            raise ValueError('source must have exactly four image identities')
        row[field] = original[field][2:]
    # Keep the original proof only as provenance, not as approval of the new input.
    row['student_reference'] = dict(source_row_sha256=digest(original), source_id=original['id'],
        original_label_basis=original['label_basis'], original_reference_kind=original.get('reference_kind', 'reviewed_rgb'),
        actual_input_human_reviewed=False, teacher_recomputed=False)
    for field in ('teacher_provenance', 'condition_provenance', 'visual_review'):
        row.pop(field, None)
    row['reference_kind'] = 'transferred_reference'
    row['label_basis'] = 'transferred_rgb4_reference'
    row['prompt_sha256'] = digest(prompt(Episode(**row['episode']), row['edge'], row['observation']))
    row['model_input_sha256'] = model_input_key(row)
    validate_observation(observation_contract(2), row['observation'], len(row['images']))
    return row


def exposure_inventory():
    from qwen3vl_local.audit_joint.splits import group_of
    p3 = native.ROOT.parent / 'sft_new_loop_phase3'
    groups, files = set(), {}
    for directory, pattern, key in ((p3, 'development_route_groups*.json', 'groups'),
                                    (native.ROOT, '*.json', 'train_only_groups')):
        for path in sorted(directory.glob(pattern)):
            value = json.loads(path.read_text())
            if isinstance(value, dict) and isinstance(value.get(key), list) and value[key]:
                files[str(path.relative_to(native.ROOT.parent))] = file_sha(path)
                groups.update(group_of(row) for row in value[key])
    if not files:
        raise ValueError('missing development exposure inventory')
    return groups, files


def legacy_paired_view(data, mode, exposed):
    """Apply one exclusion set to both modes; never move an old split's rows."""
    transformed, targets = {}, defaultdict(set)
    for split, rows in data.items():
        for row in rows:
            if row['id'] in transformed:
                raise ValueError('duplicate source row ID')
            small = short_row(row)
            transformed[row['id']] = small
            targets[(split, small['model_input_sha256'])].add(small['target'])
    result, exclusions = {}, []
    for split, rows in data.items():
        result[split] = []
        for row in rows:
            small = transformed[row['id']]
            reasons = []
            if split != 'train' and row['physical_group'] in exposed:
                reasons.append('registered_development_exposure')
            if len(targets[(split, small['model_input_sha256'])]) > 1:
                reasons.append('conflicting_targets_for_short_input')
            if reasons:
                exclusions.append(dict(id=row['id'], split=split, physical_group=row['physical_group'], reasons=reasons))
            else:
                result[split].append(small if mode == 2 else row)
    return result, exclusions


def image_identities(data, data_root):
    """Bind new RGB4 pixels from existing files once, never fabricate frames."""
    from PIL import Image
    from ..data_paths import data_path
    from ..input_identity import rgb_content_sha
    identities={}
    for rows in data.values():
        for row in rows:
            last=Path(row['images'][-1]);frame=row['observation']['frame_id']
            for offset in (-3,-2,-1,0):
                rel=last.with_name(f'{frame+offset:04d}'+last.suffix).as_posix()
                if rel in identities:continue
                path=data_path(Path(data_root).resolve(),rel)
                before=file_sha(path)
                with Image.open(path) as image:rgb=rgb_content_sha(image.convert('RGB'))
                if before!=file_sha(path):raise ValueError('RGB changed while binding view')
                identities[rel]=dict(sha256=before,rgb_sha256=rgb)
    return identities


def four_row(original, identities):
    """Transfer old sparse-RGB4 target to new consecutive RGB4, explicitly."""
    row=short_row(original)  # strips old visibility/teacher approval, retaining provenance
    frame=original['observation']['frame_id'];last=Path(original['images'][-1])
    row['observation']['history_frames']=[frame+i for i in (-3,-2,-1,0)]
    row['images']=[last.with_name(f'{f:04d}'+last.suffix).as_posix() for f in row['observation']['history_frames']]
    row['image_sha256']=[identities[p]['sha256'] for p in row['images']]
    row['image_rgb_sha256']=[identities[p]['rgb_sha256'] for p in row['images']]
    # Existing matching frames must agree with the native source identity.
    known=dict(zip(original['images'],original['image_sha256']))
    if any(known[p]!=h for p,h in zip(row['images'],row['image_sha256']) if p in known):
        raise ValueError('new view disagrees with original RGB source SHA')
    row['student_reference']['input_change']='sparse_four_to_consecutive_four'
    row['prompt_sha256']=digest(prompt(Episode(**row['episode']),row['edge'],row['observation']))
    row['model_input_sha256']=model_input_key(row)
    validate_observation(observation_contract(4),row['observation'],4)
    return row


def paired_view(data, mode, exposed, *, identities=None, data_root=None):
    """One exclusion set for both modes, including new-input target conflicts."""
    if identities is None:
        if data_root is None:return legacy_paired_view(data,mode,exposed)
        identities=image_identities(data,data_root)
    transformed={};targets=defaultdict(set)
    for split,rows in data.items():
        for row in rows:
            if row['id'] in transformed:raise ValueError('duplicate source row ID')
            pair={2:short_row(row),4:four_row(row,identities)}
            transformed[row['id']]=pair
            for m,r in pair.items():targets[(split,m,r['model_input_sha256'])].add(r['target'])
    result={};exclusions=[]
    for split,rows in data.items():
        result[split]=[]
        for row in rows:
            pair=transformed[row['id']];reasons=[]
            if split!='train' and row['physical_group'] in exposed:reasons.append('registered_development_exposure')
            for m,r in pair.items():
                if len(targets[(split,m,r['model_input_sha256'])])>1:reasons.append(f'conflicting_targets_for_rgb{m}_input')
            if reasons:exclusions.append(dict(id=row['id'],split=split,physical_group=row['physical_group'],reasons=reasons))
            else:result[split].append(pair[mode])
    return result,exclusions


def parent_binding(parent, manifest):
    names = {'manifest.json', 'review_queue.jsonl', *manifest['files']}
    for key, name in [('candidate_pool_sha256', 'candidate_pool.json'),
                      ('teacher_registry_sha256', 'teacher_registry.json'),
                      ('production_reference_sha256', 'production_reference.json'),
                      ('production_index_sha256', 'production/index.json')]:
        if manifest.get(key):
            names.add(name)
    return {name: dict(sha256=file_sha(parent / name), resolved_path=str((parent / name).resolve()))
            for name in sorted(names)}


def create(parent, output, *, data_root=None):
    parent, output = Path(parent).absolute(), Path(output).absolute()
    if output.resolve().is_relative_to(parent.resolve()):
        raise ValueError('view output must be outside the original dataset')
    if output.exists():
        raise FileExistsError(output)
    data, manifest = native.load_dataset(parent)
    if manifest['rgb_mode'] != 4:
        raise ValueError('short input must derive from native RGB4, not old RGB2')
    exposed, files = exposure_inventory()
    data_root=Path(data_root or native.ROOT.parents[1]/'lead_data')
    identities=image_identities(data,data_root)
    selected, exclusions = paired_view(data, 2, exposed, identities=identities)
    descriptor = dict(schema=POLICY, source=contract(), parent=dict(logical_path=str(parent),
        resolved_path=str(parent.resolve()), files=parent_binding(parent, manifest)),
        exposure_files=files, counts={s: len(rows) for s, rows in selected.items()}, exclusions=exclusions,
        selected_ids={s: digest([row['id'] for row in rows]) for s, rows in selected.items()},
        observation_contracts={str(mode): observation_contract(mode) for mode in (2, 4)},
        production_copied=False, joint_independence_certified=False,
        reference_scope='RGB4 targets transferred to shorter student inputs; no new human visibility or teacher approval')
    if descriptor['parent']['files'] != parent_binding(parent, manifest):
        raise ValueError('parent dataset changed during view preparation')
    output.mkdir(parents=True, exist_ok=False)
    from qwen3vl_local.audit_joint.io import write_json
    # Sidecar is an immutable input binding, not another copy of the RGB corpus.
    write_json(output/'rgb_identities.json',identities)
    descriptor['rgb_identities_sha256']=file_sha(output/'rgb_identities.json')
    write_json(output / 'student_view.json', descriptor)
    return descriptor


def load_dataset(path, *, rgb_mode=2, require_trainable=False, require_complete_coverage=False):
    observation_contract(rgb_mode)
    descriptor = json.loads((Path(path) / 'student_view.json').read_text())
    if descriptor.get('schema') != POLICY or descriptor.get('source') != contract():
        raise ValueError('student view source contract mismatch')
    for mode in (2, 4):
        if descriptor.get('observation_contracts', {}).get(str(mode)) != observation_contract(mode):
            raise ValueError('student view offsets mismatch')
    binding = descriptor['parent']
    parent = Path(binding['logical_path'])
    if str(parent.resolve()) != binding['resolved_path']:
        raise ValueError('parent dataset logical path was redirected')
    manifest = json.loads((parent / 'manifest.json').read_text())
    if parent_binding(parent, manifest) != binding['files']:
        raise ValueError('parent dataset bytes changed')
    data, manifest = native.load_dataset(parent)
    if manifest['rgb_mode'] != 4:
        raise ValueError('parent is not RGB4')
    exposed, files = exposure_inventory()
    if files != descriptor['exposure_files']:
        raise ValueError('exposure inventory changed; prepare a new view')
    identity_path=Path(path)/'rgb_identities.json'
    if file_sha(identity_path)!=descriptor.get('rgb_identities_sha256'):
        raise ValueError('student RGB identity sidecar changed')
    identities=json.loads(identity_path.read_text())
    data, exclusions = paired_view(data, rgb_mode, exposed, identities=identities)
    if (exclusions != descriptor['exclusions'] or
            {s: len(rows) for s, rows in data.items()} != descriptor['counts'] or
            {s: digest([row['id'] for row in rows]) for s, rows in data.items()} != descriptor['selected_ids']):
        raise ValueError('student view membership changed')
    coverage = native.coverage_report(sum(data.values(), []))
    admission = native.training_report(data, coverage)
    admission['evaluation_scope'] = 'development reference agreement; short-input visibility and joint independence unverified'
    admission['transferred_supervision_rows'] = len(data['train'])
    from collections import Counter
    admission['original_label_basis_counts'] = dict(Counter(
        row.get('student_reference', {}).get('original_label_basis', row['label_basis'])
        for row in data['train']))
    admission['original_weak_supervision_rows'] = admission['original_label_basis_counts'].get('weak_rule_teacher', 0)
    if require_trainable and not admission['trainable']:
        raise ValueError('student view training prerequisites failed: ' + '; '.join(admission['errors']))
    if require_complete_coverage:
        raise ValueError('student view has no independent complete-coverage certification')
    selected_manifest = dict(manifest, rgb_mode=rgb_mode, observation_contract=observation_contract(rgb_mode),
                             coverage=coverage, training_admission=admission, student_view=descriptor,
                             trainable=admission['trainable'], ready=admission['trainable'],
                             complete_coverage=False, formal_data_ready=False,
                             reference_scope=admission['evaluation_scope'])
    return data, selected_manifest
