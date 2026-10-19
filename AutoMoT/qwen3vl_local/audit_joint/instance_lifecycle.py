"""Check native route question lifetimes without turning censoring into labels.

Consumes one complete route in frame order. Verification is structural only;
source authenticity and semantic correctness require the producer/SHA/RGB audit.
"""
from collections import Counter


def inspect(records):
    ended = {}
    identities = {}
    route = None
    last = -1
    frames = questions = 0
    end_counts, targets = Counter(), Counter()
    for row in records:
        identity = (row['scenario'], row['route_id'], row['split'])
        if route is None:
            route = identity
        if identity != route:
            raise ValueError('Lifecycle scan must contain one route and split')
        frame = row['frame_id']
        if type(frame) is not int or frame <= last:
            raise ValueError('Lifecycle frames must be strictly increasing')
        last = frame
        frames += 1
        for end in row['instance_ends']:
            ident, reason = end['instance_id'], end['reason']
            if not isinstance(ident, str) or not ident or not isinstance(reason, str) or not reason:
                raise ValueError('Invalid instance end identity/reason')
            if ident in ended:
                raise ValueError('Instance ended more than once')
            if type(end['completed']) is not bool or end['completed'] != (reason == 'finished'):
                raise ValueError('Censored/retired instance cannot be successful completion')
            ended[ident] = dict(frame_id=frame, reason=reason, completed=end['completed'])
            end_counts[reason] += 1
        paired = {}
        for q in row['questions']:
            ep, seed = q['episode'], q['instance']
            ident = ep['instance_id']
            if ident in ended:
                raise ValueError('Question at/after instance retirement')
            if (q['frame_id'] != frame or
                    (q['scenario'], q['route_id'], q['split']) != route):
                raise ValueError('Question does not bind current route/frame')
            binding = (ep['event'], seed['actor_id'], seed['restriction_frame'])
            if ident in identities and identities[ident] != binding:
                raise ValueError('Instance reused for a different event/actor/restriction')
            identities[ident] = binding
            mode = q['rgb_mode']
            if type(mode) is not int or mode not in (2, 4):
                raise ValueError('Invalid RGB mode')
            key = (ident, q['edge'])
            members = paired.setdefault(key, {})
            if mode in members:
                raise ValueError('Duplicate instance/edge/mode')
            members[mode] = (q['rule_target'], q['slice'])
            questions += 1
            targets[f"{ep['event']}/{q['edge']}/{q['rule_target']}"] += 1
        if any(set(m) != {2, 4} or m[2] != m[4] for m in paired.values()):
            raise ValueError('Native two/four-view lifecycle supervision differs')
    if not frames:
        raise ValueError('Empty route does not verify lifecycle')
    return dict(status='verified_structure', route=route, frames=frames, questions=questions,
                end_counts=dict(end_counts), targets=dict(targets), ended=ended,
                open_instance_ids=sorted(set(identities)-set(ended)),
                labels_changed=False, training_authorized=False,
                scope='native lifetime and paired-label structure only; open route tails are censored, not completion or release')
