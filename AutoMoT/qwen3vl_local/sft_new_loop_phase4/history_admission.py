"""Quarantine demonstrated concurrent stop-history ambiguity before abstentions drop.

No instance merge or label rewrite. Equal encoded RGB implies equal pixels; a
copy with different encoding is not assumed equal. Unrelated UNKNOWNs do not
trigger this specific history gate.
"""
from collections import defaultdict
from .controller import Episode
from .identity import digest
from .route_prompts import messages

POLICY = 'concurrent_post_stop_history_quarantine_v1'


def quarantine(questions):
    groups = defaultdict(list)
    for q in questions:
        if q['episode']['event'] not in ('U-E6', 'U-E7', 'R-E5') or q['edge'] != 'proceed':
            continue
        ep = dict(q['episode'])
        ep.pop('instance_id', None)
        # Same logical target as well as same rendered input: hidden route and
        # branch differences never establish equivalence of two propositions.
        obs = dict(frame_id=q['frame_id'], history_frames=q['history_frames'], speed_mps=max(0., q['ego_speed']))
        rendered = messages(Episode(**q['episode']), q['edge'], obs,
                            [dict(rgb_sha256=s['sha256']) for s in q['input_sources']])
        key = digest([q['scenario'], q['route_id'], q['frame_id'], ep, q['edge'], rendered])
        groups[key].append(q)
    dropped = set()
    records = []
    for key, values in groups.items():
        stale = [q for q in values if q['rule_target'] == 'UNKNOWN'
                 and 'post_stop_release_requires_visual_catchup' in q.get('reasons', [])
                 and q.get('motion_phase_evidence', {}).get('target') == 'NO']
        if not stale or not any(q['rule_target'] == 'YES' for q in values):
            continue
        if len({q['episode']['instance_id'] for q in values}) < 2:
            continue
        ids = [q['question_id'] for q in values]
        dropped.update(ids)
        records.append(dict(input_target_sha256=key, question_ids=sorted(ids),
                            reason='concurrent_post_stop_history_unknown_yes',
                            instance_ids=sorted({q['episode']['instance_id'] for q in values}),
                            inherited_history=False, labels_rewritten=False))
    return [q for q in questions if q['question_id'] not in dropped], records
