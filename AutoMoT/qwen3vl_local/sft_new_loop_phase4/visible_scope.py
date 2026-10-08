"""A narrower visible-gap rubric requires fresh causal review, never relabeling."""
from .maneuver_safety import GUARDED

POLICY = 'visible_maneuver_conditions_v1'


def reviewed(row):
    proof = row.get('visible_scope_review')
    if row['edge'] not in GUARDED:
        if proof is not None:
            raise ValueError('visible scope review is only for maneuver permission')
        return True
    if proof is None:
        if row.get('label_basis') in ('weak_rule_teacher','approved_rule_teacher'):
            from .teacher_replay import validate_question
            q=validate_question(row.get('teacher_provenance',{}).get('question',{}))
            if (q.get('visible_condition_scope')!=POLICY or q['edge']!=row['edge']
                    or q['rule_target']!=row['target'] or q['history_frames']!=row['observation']['history_frames']
                    or [s['sha256'] for s in q['input_sources']]!=row['image_sha256']):
                raise ValueError('teacher visible maneuver evidence mismatch')
            return True  # rule evidence, explicitly not a human visible_scope_review
        return False
    if (not isinstance(proof, dict) or proof.get('policy') != POLICY
            or any(not isinstance(proof.get(k), str) or not proof[k].strip()
                   for k in ('reviewer', 'evidence_id'))
            or not isinstance(proof.get('frames'), dict)):
        raise ValueError('invalid visible scope review')
    for frame, sha in zip(row['observation']['history_frames'], row['image_sha256'], strict=True):
        item = proof['frames'].get(str(frame))
        if (not isinstance(item, dict) or item.get('rgb_sha256') != sha
                or type(item.get('observed_until')) is not int or item['observed_until'] != frame
                or item.get('visible_conditions_reviewed') is not True
                or not isinstance(item.get('observation'), str) or not item['observation'].strip()):
            raise ValueError('visible scope requires every causal RGB and current condition review')
    current = proof['frames'][str(row['observation']['frame_id'])]
    if current.get('target') != row['target']:
        raise ValueError('visible scope must explicitly confirm the current frame answer under the new rubric')
    return True


def validate_rows(rows):
    for row in rows:
        if not reviewed(row) and (row['target'] != 'UNKNOWN' or row['slice'] != 'uncertain'
                                 or row.get('review_reason') != 'visible_scope_reaudit_required'):
            raise ValueError('old maneuver labels require visible scope re-audit')
