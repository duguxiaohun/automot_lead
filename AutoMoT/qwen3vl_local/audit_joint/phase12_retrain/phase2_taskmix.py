"""Compose existing task buckets byte-for-byte; no new annotation decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import CONTRACT
from .phase2_evidence import sha, target_class

REVISION = 'taskmix_20261031'
TASK_SOURCES = {'UE1': 'legacy', 'UE3': 'evidence', 'UE5': 'legacy',
                'UE6': 'evidence', 'RE': 'evidence', 'INVALID': 'evidence'}
SOURCES = {
    'legacy': {'revision': 'visual_20261029', 'reference_run': 'p2_20261030_020025_2885626',
               'definition': 'v6; v1 parent UE1/UE5 buckets unchanged by the old UE3-only RS derivation'},
    'evidence': {'revision': 'evidence_20261030', 'reference_run': 'p2_20261030_200229_3219354',
                 'policy_version': 'evidence_policy_v3_20261030'},
}


def _rows(path):
    return [(line, json.loads(line)) for line in path.read_bytes().splitlines(keepends=True)]


def derive(legacy_index, evidence_index, output):
    legacy_index, evidence_index, output = map(lambda p: Path(p).resolve(), (legacy_index, evidence_index, output))
    if output.exists() or (output.parent / 'selected_train_reviews.json').exists():
        raise ValueError('Choose a fresh composition output')
    for p in (legacy_index, evidence_index):
        if output.is_relative_to(p.parent) or p.is_relative_to(output):
            raise ValueError('Composition output overlaps an input')
    parents = {'legacy': legacy_index, 'evidence': evidence_index}
    bindings = {n: {'index': str(p), 'index_sha256': sha(p), 'manifest_sha256': sha(p.parent/'manifest.json')}
                for n, p in parents.items()}
    evidence_receipt = json.loads((evidence_index.parent/'training_revision.json').read_text())
    bindings['evidence']['receipt_sha256']=sha(evidence_index.parent/'training_revision.json')
    bindings['evidence']['val_original_sha256']=sha(evidence_index.parent/'val_original.jsonl')
    if (evidence_receipt.get('revision') != SOURCES['evidence']['revision']
            or evidence_receipt.get('policy', {}).get('policy_version') != SOURCES['evidence']['policy_version']):
        raise ValueError('Expected the existing evidence v3 view, not a newly defined policy')
    raw = {n: _rows(p) for n, p in parents.items()}
    for rows in raw.values():
        if any(r['split'] not in ('train', 'val', 'test') or target_class(r) not in TASK_SOURCES for _, r in rows):
            raise ValueError('Unexpected split or task')
    test = {n: b''.join(line for line, r in rows if r['split']=='test') for n, rows in raw.items()}
    if not test['legacy'] or test['legacy'] != test['evidence']:
        raise ValueError('Parent test rows differ; never silently change the main comparison')
    selected = []
    counts = Counter()
    for split in ('train', 'val'):
        for source, rows in raw.items():
            for line, row in rows:
                cls = target_class(row)
                if row['split']==split and TASK_SOURCES[cls]==source:
                    selected.append(line)
                    counts[f'frames/{split}'] += 1
                    counts[f'frames/{split}/{cls}'] += 1
        for cls in TASK_SOURCES:
            if not counts[f'frames/{split}/{cls}']:
                raise ValueError(f'Empty selected task: {split}/{cls}')
    selected.extend(line for line, r in raw['legacy'] if r['split']=='test')
    for _, r in raw['legacy']:
        if r['split']=='test':
            counts['frames/test'] += 1
            counts[f'frames/test/{target_class(r)}'] += 1
    # Keep multiplicity, but reject conflicting labels or cross-split frames.
    identities, splits = {}, {}
    for line in selected:
        r=json.loads(line); frame=(r['scenario'],r['route_id'],int(r['frame_id']))
        identity=(*frame,r['question_domain'],r.get('invalid_source',''))
        if frame in splits and splits[frame]!=r['split']:
            raise ValueError('Cross-split physical frame in composed parents')
        if identity in identities and identities[identity]!=r['answers']:
            raise ValueError('Conflicting selected task labels')
        splits[frame]=r['split'];identities[identity]=r['answers']
    evidence_original = _rows(evidence_index.parent/'val_original.jsonl')
    original_val = b''.join(line for line,r in raw['legacy'] if r['split']=='val' and TASK_SOURCES[target_class(r)]=='legacy')
    original_val += b''.join(line for line,r in evidence_original if TASK_SOURCES[target_class(r)]=='evidence')
    artifacts = {}
    for name in ('holdout_disputes.jsonl','evidence_sources.json','evidence_exclusions.jsonl','val_exclusions.jsonl'):
        p=evidence_index.parent/name
        bindings['evidence'][name+'_sha256']=sha(p)
        if name in ('evidence_exclusions.jsonl','val_exclusions.jsonl'):
            artifacts[name]=b''.join(line for line,r in _rows(p) if TASK_SOURCES[r['target_event_class']]=='evidence')
        else:artifacts[name]=p.read_bytes()
    # No observations are reclassified: use the same pretraining diagnostic list.
    artifacts['val_original.jsonl']=original_val
    artifacts['evidence_parent_receipt.json']=(evidence_index.parent/'training_revision.json').read_bytes()
    for n,p in parents.items():
        if sha(p)!=bindings[n]['index_sha256'] or sha(p.parent/'manifest.json')!=bindings[n]['manifest_sha256']:
            raise ValueError('Parent changed while composing')
    for name in artifacts:
        if name not in ('val_original.jsonl','evidence_parent_receipt.json') and sha(evidence_index.parent/name)!=bindings['evidence'][name+'_sha256']:
            raise ValueError('Parent evidence changed while composing')
    if (sha(evidence_index.parent/'training_revision.json')!=bindings['evidence']['receipt_sha256']
            or sha(evidence_index.parent/'val_original.jsonl')!=bindings['evidence']['val_original_sha256']):
        raise ValueError('Parent evidence receipt or validation rows changed while composing')
    receipt = dict(contract=CONTRACT, revision=REVISION, status='composed_new_training_required',
        task_sources=TASK_SOURCES, source_versions=SOURCES, parents=bindings, counts=dict(sorted(counts.items())),
        labels_flipped=0, new_annotation_decisions=0, rows_preserved_byte_for_byte=True,
        test_byte_identical=True, test_sha256=hashlib.sha256(test['legacy']).hexdigest(),
        selection_split='val task buckets from the same sources as train', training_sampler_changed=False,
        input_layout='text_before_images_v1', text_prefix_cache_default=True,
        note='Task-wise candidate composition, not a claim that independently best task scores add up; disputes remain the frozen evidence-v3 diagnostic view.')
    manifest=json.loads((evidence_index.parent/'manifest.json').read_text())
    manifest.update(frame_index=str(output/'frame_index.jsonl'),counts=dict(sorted(counts.items())),fresh_training_revision=receipt,
                    parent_statistics={'legacy_manifest_sha256':bindings['legacy']['manifest_sha256'],
                                       'evidence_manifest_sha256':bindings['evidence']['manifest_sha256']})
    output.mkdir(parents=True)
    (output/'frame_index.jsonl').write_bytes(b''.join(selected))
    for name,content in artifacts.items():(output/name).write_bytes(content)
    for name,value in [('manifest.json',manifest),('training_revision.json',receipt),('preservation.json',receipt),('task_sources.json',receipt)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (output.parent/'selected_train_reviews.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',required=True);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(derive(a.legacy,a.evidence,a.output),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
