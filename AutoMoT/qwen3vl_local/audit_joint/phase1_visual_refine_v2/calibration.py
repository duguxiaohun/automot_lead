"""Field-scoped evidence admission and whole-parent preservation verification."""
import json
from itertools import zip_longest
from pathlib import Path
from qwen3vl_local.audit_joint.phase1_visual_refine import calibration as parent
from . import CONTRACT

def load_reviews(path):
    obj, decisions = parent.load_reviews(path)
    if obj.get('review_policy') != CONTRACT:
        raise ValueError('Explicit v2 field evidence policy required')
    for d in decisions.values():
        expected = d['expected']
        if any(expected['answers'][k] != (expected['rs'] == k.replace('RS', 'R')) for k in parent.ROADS):
            raise ValueError('Inconsistent parent RS/answers; no implicit road-label repair')
        fields = d.get('verified_fields', [])
        if not isinstance(fields, list) or len(set(fields)) != len(fields) or set(fields)-({'rs'} | set(parent.FACTS)):
            raise ValueError('Invalid verified fields')
        changes = d.get('changes', {})
        changed = set(changes.get('facts', {})) | ({'rs'} if 'rs' in changes else set())
        if not changed <= set(fields):
            raise ValueError('Correction outside verified fields')
        for field in changed:
            evidence = d.get('field_evidence', {}).get(field, {})
            frames = evidence.get('visible_frames', [])
            if (evidence.get('confidence') != 'confirmed' or not frames
                    or any(type(i) is not int or i not in range(4) for i in frames)
                    or not isinstance(evidence.get('newest_relevance'), str)
                    or not evidence['newest_relevance'].strip()):
                raise ValueError('Correction needs confirmed field evidence and newest-frame relevance')
    return obj, decisions

def verify_derivation(parent_index, output_index, reviews, data_root):
    _, decisions = load_reviews(reviews)
    n = unchanged = changed = 0
    seen = set()
    with Path(parent_index).open('rb') as left, Path(output_index).open('rb') as right:
        for old, new in zip_longest(left, right):
            if old is None or new is None:
                raise ValueError('Derivation added/dropped rows')
            row = json.loads(old); target = json.loads(new); k = parent.key(row)
            wanted = row
            if k in decisions:
                if k in seen:
                    raise ValueError('Duplicate reviewed identity')
                seen.add(k); wanted = parent.apply(row, decisions[k], data_root)
            if target != wanted:
                raise ValueError(f'Unapproved field/member change: {k}')
            if wanted == row:
                if old != new:
                    raise ValueError(f'Protected row bytes changed: {k}')
                unchanged += 1
            else:
                allowed = set(decisions[k]['changes'].get('facts', {}))
                if 'rs' in decisions[k]['changes']:
                    allowed |= set(parent.ROADS)
                if any(row['answers'][q] != target['answers'][q] for q in set(parent.ALL)-allowed):
                    raise ValueError('Unreviewed answer changed')
                changed += 1
            n += 1
    if seen != set(decisions):
        raise ValueError('Reviewed member missing; no cross-machine label assumptions')
    return dict(rows=n, changed_rows=changed, protected_rows_byte_identical=unchanged,
                sampling_changed=False, split_changed=False, model_inference=False)

def derive(parent_index, reviews, data_root, output):
    load_reviews(reviews)
    before = {str(p):parent.sha(p) for p in (parent_index, reviews)}
    report = parent.derive(parent_index, reviews, data_root, output)
    try:
        protected = verify_derivation(parent_index, Path(output)/'frame_index.jsonl', reviews, data_root)
        if any(parent.sha(p) != digest for p, digest in before.items()):
            raise ValueError('Input changed during preservation check')
        result = dict(status='calibration_verified_model_pending', review_policy=CONTRACT,
                      preservation=protected, parent_derivation=report,
                      index_sha256=parent.sha(Path(output)/'frame_index.jsonl'))
        (Path(output)/'preservation.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    except BaseException as exc:
        (Path(output)/'rejected.json').write_text(json.dumps(dict(status='rejected', error=str(exc)))+'\n')
        raise
