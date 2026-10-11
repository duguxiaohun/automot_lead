"""Descriptive fit strata and conservative joins to native replay evidence.

Call after verify_fit with the original runtime/run. These helpers never certify
query completeness, label accuracy, cross-stage participant identity or G2.
"""
from collections import Counter, defaultdict

from .train_fit import validate_prediction, natural
from ..sft_new_loop_phase4.paired_eval import validate
from ..sft_new_loop_phase4.rgb_short.diagnostics import binary_diagnostics


INPUT_FIELDS = {'observation', 'images', 'image_sha256', 'image_rgb_sha256',
                'model_input_sha256', 'prompt_sha256', 'label_basis', 'reference_kind'}


def checked_cases(cases):
    result = {}
    for case in cases:
        validate(case)
        validate_prediction(case)
        if case['id'] in result:
            raise ValueError('duplicate fit ID')
        if case['paired_identity']['fields']['split'] != 'train' or natural(case['sampled_presentations']) < 1:
            raise ValueError('expected actually sampled training case')
        result[case['id']] = case
    return result


def measurement(cases):
    report = binary_diagnostics(cases)
    report['correct'] = sum(c['target'] == c['prediction'] for c in cases)
    report['presentations'] = sum(c['sampled_presentations'] for c in cases)
    groups = defaultdict(list)
    for c in cases:
        groups[c['paired_identity']['fields']['physical_group']].append(c)
    report['physical_groups'] = len(groups)
    report['answer_physical_groups'] = {
        a: len({c['paired_identity']['fields']['physical_group'] for c in cases if c['target'] == a})
        for a in ('YES', 'NO')}
    report['group_macro_recall'] = {
        a: (sum(values)/len(values) if values else None)
        for a in ('YES', 'NO')
        for values in [[sum(c['prediction'] == a for c in rs if c['target'] == a)
                        / sum(c['target'] == a for c in rs)
                        for rs in groups.values() if any(c['target'] == a for c in rs)]]}
    return report


def paired_diagnostics(rgb2, rgb4):
    left, right = checked_cases(rgb2), checked_cases(rgb4)
    if not left or left.keys() != right.keys():
        raise ValueError('paired fit sets must be nonempty and equal; no intersection-only comparison')
    strata = defaultdict(list)
    transitions = Counter()
    for key, a in left.items():
        b = right[key]
        fa, fb = a['paired_identity']['fields'], b['paired_identity']['fields']
        provenance_equal = all(fa[k] == fb[k] for k in ('label_basis','reference_kind'))
        transferred = (fa['label_basis'] == 'transferred_rgb4_reference' and
                       fa['reference_kind'] == 'transferred_reference' and
                       a['original_label_basis'] == b['original_label_basis'] == fb['label_basis'])
        if ({k:v for k,v in fa.items() if k not in INPUT_FIELDS} !=
                {k:v for k,v in fb.items() if k not in INPUT_FIELDS} or
                not (provenance_equal or transferred) or
                a['original_label_basis'] != b['original_label_basis'] or
                a['sampled_presentations'] != b['sampled_presentations'] or
                {k:v for k,v in fa['observation'].items() if k != 'history_frames'} !=
                {k:v for k,v in fb['observation'].items() if k != 'history_frames'}):
            raise ValueError('paired semantic identity or exposure differs')
        histories = []
        for fields, mode in ((fa, 2), (fb, 4)):
            frames = fields['observation']['history_frames']
            if (len(frames) != mode or frames != sorted(set(frames)) or
                    frames[-1] != fields['observation']['frame_id']):
                raise ValueError('invalid paired causal frame window')
            histories.append(dict(zip(frames, zip(fields['images'], fields['image_sha256'], fields['image_rgb_sha256']))))
        if not histories[0].items() <= histories[1].items():
            raise ValueError('shared RGB identity differs')
        strata[(a['event'], a['edge'], a['slice'], a['original_label_basis'])].append(key)
        transitions[f"rgb2_{int(a['prediction']==a['target'])}_rgb4_{int(b['prediction']==b['target'])}"] += 1
    return dict(schema='paired_fit_development_diagnostics_v1', unique_questions=len(left),
        rgb2=measurement(list(left.values())), rgb4=measurement(list(right.values())),
        correctness_transitions=dict(transitions), strata=[dict(
            cell=list(cell), rgb2=measurement([left[k] for k in ids]),
            rgb4=measurement([right[k] for k in ids])) for cell,ids in sorted(strata.items())],
        cell_fields=['event','edge','phase','original_label_basis'],
        scope='same questions and shared RGB, different temporal inputs; descriptive in-sample agreement, not independent accuracy or label approval')


def replay_index(records):
    result = {}
    for record in records:
        key = record['scenario'], record['route_id'], record['frame_id']
        if key in result:
            raise ValueError('duplicate replay frame')
        for q in record['questions']:
            if any(q[k] != record[k] for k in ('scenario','route_id','frame_id','split','physical_group')):
                raise ValueError('replay question outside frame identity')
        result[key] = record
    return result


def replay_diagnostics(cases, before, after):
    """Same-instance same-proposition drift only, never same-frame substitution.

    An absent question is NOT a NO. Even exact native matches are rule evidence,
    not a new training label. Different state/slice must remain non-comparable.
    """
    cases = checked_cases(cases)
    indexes = [replay_index(before), replay_index(after)]
    results = []
    for case in cases.values():
        fields = case['paired_identity']['fields']
        key = fields['scenario'], fields['route_id'], fields['observation']['frame_id']
        item = dict(id=case['id'], scenario=key[0], route_id=key[1], frame_id=key[2],
                    edge=case['edge'], reference=case['target'], prediction=case['prediction'],
                    original_label_basis=case['original_label_basis'])
        matches = []
        for name, index in zip(('before','after'), indexes):
            record = index.get(key)
            candidates = [] if record is None else [q for q in record['questions']
                if q['rgb_mode'] == len(fields['images']) and q['edge'] == fields['edge']
                and q['episode']['instance_id'] == fields['episode']['instance_id']]
            exact = [q for q in candidates if q['episode'] == fields['episode']
                     and q['slice'] == fields['slice'] and q['split'] == fields['split']
                     and q['physical_group'] == fields['physical_group']]
            expected_rgb = list(zip(fields['observation']['history_frames'], fields['images'], fields['image_sha256']))
            bound = [q for q in exact if q.get('history_frames') == fields['observation']['history_frames']
                     and [(s['frame_id'],s['path'],s['sha256']) for s in q.get('input_sources',[])
                          if s['kind']=='rgb'] == expected_rgb]
            status = ('not_replayed' if record is None else 'ambiguous' if len(exact)>1 else
                      'exact_proposition' if len(bound)==1 else 'input_evidence_changed' if exact else
                      'proposition_changed' if candidates else 'question_absent')
            item[name] = dict(status=status, candidates=[dict(question_id=q['question_id'],
                episode=q['episode'], phase=q['slice'], rule_target=q['rule_target'],
                facts=q['facts'], reasons=q['reasons']) for q in candidates])
            matches.append(bound[0] if len(exact)==len(bound)==1 else None)
        a,b = matches
        if item['before']['status'] == 'not_replayed' or item['after']['status'] == 'not_replayed':
            status = 'outside_replay_scope'
        elif case['original_label_basis'] != 'weak_rule_teacher':
            status = 'manual_reference_requires_separate_adjudication'
        elif a is None:
            status = 'original_proposition_not_reproduced'
        elif a['rule_target'] != case['target']:
            status = 'original_target_not_reproduced'
        elif b is None:
            status = 'current_proposition_not_comparable'
        else:
            status = 'rule_target_unchanged' if a['rule_target']==b['rule_target'] else 'rule_target_changed'
        item['status'] = status
        results.append(item)
    return dict(schema='fit_native_replay_drift_v1', counts=dict(Counter(r['status'] for r in results)),
        cases=results, scope='diagnostic join only; missing/changed instances or states are not negative labels; manual references are not adjudicated by weak rules')
