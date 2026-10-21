"""Independent Phase3 diagnostics; never changes frozen labels/prompts/training.

Scores received validation predictions and prepares label-independent matched
image interventions. Inference requires the original admitted dataset, runtime,
and completed checkpoint; there is no source-hash bypass or implicit retraining.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ACTIONS = ('DECELERATE', 'STOP', 'RESUME', 'LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT', 'KEEP')
POLICY = 'phase3_behavior_diagnostics_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def indexed(rows):
    result = {r['id']: r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError('Empty or duplicate diagnostic identities')
    if any(r['split'] != 'val' for r in rows):
        raise ValueError('These development diagnostics accept validation only')
    return result


def validate_case(row):
    if len(row['windows']) != 4 or len(row['predictions']) != 4 or len(row['history']) != 4:
        raise ValueError('Expected exactly four windows, predictions and history slots')
    for w in row['windows']:
        if type(w['valid']) is not bool:
            raise ValueError('Window validity must be boolean')
        if w['valid']:
            q = w['raw']
            if (len(q) != 6 or any(not math.isfinite(v) or v < 0 for v in q)
                    or not math.isclose(sum(q), 1.)
                    or w['allowed_answers'] != [a for a, v in zip(ACTIONS, q) if v > 0]):
                raise ValueError('Invalid raw target distribution/allowed answers')
            target = w['target']
            if (len(target) != 6 or any(not math.isfinite(v) or v < 0 for v in target)
                    or not math.isclose(sum(target), 1.)):
                raise ValueError('Invalid training target distribution')
        elif w['raw'] is not None or w['allowed_answers']:
            raise ValueError('Invalid window must not carry a scored target')


def first_transition(row):
    """Exact timing is reported only for four pure, valid future windows.

    This is intention-query timing, not an executed-action duration. Do not
    infer a transition through a masked or mixed interval.
    """
    if any(not w['valid'] or len(w['allowed_answers']) != 1 for w in row['windows']):
        return 'excluded_mixed_or_missing'
    last = row['history'][-1]
    if last not in ACTIONS:
        return 'excluded_latest_history_unknown'
    true = next((i for i, w in enumerate(row['windows']) if w['allowed_answers'][0] != last), None)
    if true is None:
        return 'no_target_transition'
    pred = next((i for i, p in enumerate(row['predictions']) if p != last), None)
    if pred is None:
        return 'never_changes'
    if row['predictions'][pred] not in ACTIONS:
        return 'invalid_action'
    if pred != true:
        return 'early' if pred < true else 'late'
    return 'on_time_correct_action' if row['predictions'][pred] == row['windows'][true]['allowed_answers'][0] else 'on_time_wrong_action'


def summarize(rows, records=None):
    indexed(rows)
    totals = Counter(); hits = Counter(); mass = Counter(); coverage = defaultdict(Counter)
    transitions = Counter(); preference = Counter(); route_scores = defaultdict(lambda: [0, 0])
    coverage_examples = []
    for r in rows:
        validate_case(r)
        valid = [w for w in r['windows'] if w['valid']]
        anchor = r['base_target']; cov = coverage[anchor]
        cov['anchors'] += 1
        cov['eligible_anchors'] += bool(valid)
        present = any(anchor in w['allowed_answers'] for w in valid)
        cov['anchor_action_present_in_valid_future'] += present
        cov['anchor_action_absent_with_all_four_valid'] += len(valid) == 4 and not present
        cov['anchor_action_absent_with_partial_future'] += len(valid) < 4 and not present
        if not present and anchor.startswith('LANE_CHANGE'):
            coverage_examples.append(dict(id=r['id'],action=anchor,valid_windows=len(valid),
                                          group=r['group'],frame=r['frame_id']))
        if records is not None:
            rec = records[r['id']]
            if rec['windows'] != r['windows']:
                raise ValueError('Replayed record targets differ from received predictions')
            cov['anchor_present_in_uncompressed_future'] += any(p['action'] == anchor for p in rec['future_points'])
        transitions[first_transition(r)] += 1
        last = next((a for a in reversed(r['history']) if a in ACTIONS), None)
        for i, (w, p) in enumerate(zip(r['windows'], r['predictions'])):
            if not w['valid']:
                continue
            pure = len(w['allowed_answers']) == 1
            keys = ['all', f'query_{i}', 'pure' if pure else 'mixed', 'event/' + r['event'],
                    'dark_mean_le16' if r['brightness'] <= 16 else 'brighter_mean_gt16',
                    'history_unknown' if last is None else
                    'history_compatible' if last in w['allowed_answers'] else 'history_incompatible']
            if pure:
                keys.append('pure_action/' + w['allowed_answers'][0])
            hit = p in w['allowed_answers']
            q = w['raw'][ACTIONS.index(p)] if p in ACTIONS else 0.
            for key in keys:
                totals[key] += 1; hits[key] += hit; mass[key] += q
            route_scores[r['group']][0] += hit; route_scores[r['group']][1] += 1
            preference['valid_windows'] += 1
            preference['pure_lateral_windows'] += pure and w['allowed_answers'][0].startswith('LANE_CHANGE')
            preference['changed_target_windows'] += any(not math.isclose(a, b) for a, b in zip(w['raw'], w['target']))
            preference['raw_lateral_mass'] += sum(w['raw'][j] for j in (3, 4))
            preference['training_lateral_mass'] += sum(w['target'][j] for j in (3, 4))
            preference['anchor_normalized_raw_lateral_mass'] += sum(w['raw'][j] for j in (3,4))/len(valid)
            preference['anchor_normalized_training_lateral_mass'] += sum(w['target'][j] for j in (3,4))/len(valid)
    return dict(policy=POLICY,anchors=len(rows),zero_valid_anchors=sum(not any(w['valid'] for w in r['windows']) for r in rows),
        metrics={k:dict(windows=n,hits=hits[k],set_accuracy=hits[k]/n,raw_mass_score=mass[k]/n) for k,n in totals.items()},
        anchor_action_coverage={k:dict(v) for k,v in coverage.items()},lost_lateral_examples=coverage_examples,
        transition_timing=dict(transitions),lateral_preference=dict(preference),
        route_scores={g:dict(hits=v[0],windows=v[1],accuracy=v[0]/v[1]) for g,v in route_scores.items()},
        caveats=['Future intent targets are not executed half-second actions.',
                 'Absent anchor action with masked futures does not prove that action never occurs.',
                 'The 1.5 factor changes relative mass inside mixed windows, not pure-window loss weight.',
                 'Timing excludes incomplete/mixed targets and unknown latest history; nearby anchors are correlated.'])


def speed_bin(v):
    if not math.isfinite(v) or v < 0:
        raise ValueError('Invalid current speed')
    return next(i for i, edge in enumerate((.5, 2., 8., 20., float('inf'))) if v <= edge)


def matched_plan(rows, seed):
    """Maximum-cardinality one-to-one donor matching; no labels used for edges.

    Match same event, speed bin, dark/non-dark bin and image dimensions. Exclude
    same physical group, path or byte-identical image. Missing support stays
    excluded and all comparison denominators must use this same matched subset.
    """
    byid = indexed(rows)
    for r in rows:
        if not math.isfinite(r['brightness']) or not 0 <= r['brightness'] <= 255:
            raise ValueError('Invalid image brightness')
        speed_bin(r['speed'])
        if Path(r['image']).is_absolute() or '..' in Path(r['image']).parts:
            raise ValueError('Intervention images must be relative to the explicit data root')
        if len(r['image_size']) != 2 or any(type(v) is not int or v <= 0 for v in r['image_size']):
            raise ValueError('Invalid image dimensions')
    edges = {}
    for r in rows:
        compatible = [d for d in rows if d['event'] == r['event'] and d['group'] != r['group']
                      and d['rgb_sha256'] != r['rgb_sha256'] and d['image'] != r['image']
                      and d['image_size'] == r['image_size'] and speed_bin(d['speed']) == speed_bin(r['speed'])
                      and (d['brightness'] <= 16) == (r['brightness'] <= 16)]
        # Cost uses observable metadata only; deterministic seeded jitter varies
        # similarly matched donors without consulting prediction correctness.
        jitter = lambda d: int(digest([seed,r['id'],d['id']])[:12],16)/16**12
        edges[r['id']] = [d['id'] for d in sorted(compatible,key=lambda d:
            (abs(d['speed']-r['speed'])/5 + abs(d['brightness']-r['brightness'])/64 + jitter(d),d['id']))]
    owner = {}
    def augment(receiver, seen):
        for donor in edges[receiver]:
            if donor in seen:
                continue
            seen.add(donor)
            if donor not in owner or augment(owner[donor], seen):
                owner[donor] = receiver
                return True
        return False
    order = sorted(byid,key=lambda x:(len(edges[x]),digest([seed,x])))
    for receiver in order:
        augment(receiver,set())
    inverse = {r:d for d,r in owner.items()}
    pairs = [dict(receiver=r['id'],donor=inverse[r['id']]) for r in rows if r['id'] in inverse]
    fields = ('id','split','event','group','image','rgb_sha256','image_size','speed','brightness')
    metadata = [{k:r[k] for k in fields} for r in rows]
    payload = dict(policy=POLICY,seed=seed,metadata=metadata,pairs=pairs,
        excluded=[dict(id=r['id'],reason='no_compatible_donor' if not edges[r['id']] else 'one_to_one_capacity')
                  for r in rows if r['id'] not in inverse],
        matching='same event/speed bin/brightness bin/image dimensions; different physical group; unique donor',
        labels_used_for_matching=False,full_validation_coverage=len(pairs)==len(rows),inference_performed=False)
    return dict(payload,plan_sha256=digest(payload))


def validate_plan(plan):
    payload = {k:v for k,v in plan.items() if k != 'plan_sha256'}
    if digest(payload) != plan['plan_sha256']:
        raise ValueError('Intervention plan changed')
    # Recompute the complete matching, not just the checksum supplied beside it.
    if matched_plan(plan['metadata'],plan['seed']) != plan:
        raise ValueError('Intervention plan does not follow the matching policy')
    return indexed(plan['metadata'])


def paired_scores(reference, alternative, metadata):
    if [c['id'] for c in reference] != [c['id'] for c in alternative]:
        raise ValueError('Paired IDs/order differ')
    result = defaultdict(Counter); routes = defaultdict(Counter)
    for a,b in zip(reference,alternative):
        if a['windows'] != b['windows'] or a['history'] != b['history']:
            raise ValueError('Image intervention changed labels or history')
        validate_case(a);validate_case(b)
        last=next((x for x in reversed(a['history']) if x in ACTIONS),None)
        for w,p,q in zip(a['windows'],a['predictions'],b['predictions']):
            if not w['valid']:continue
            x=p in w['allowed_answers'];y=q in w['allowed_answers']
            keys=['all','pure' if len(w['allowed_answers'])==1 else 'mixed']
            if last and last not in w['allowed_answers']:keys.append('history_incompatible')
            for key in keys:
                result[key].update(windows=1,reference_hits=int(x),alternative_hits=int(y),
                                   reference_only=int(x and not y),alternative_only=int(y and not x))
            routes[metadata[a['id']]['group']].update(windows=1,reference_hits=int(x),alternative_hits=int(y))
    return dict(subsets={k:dict(v,delta=(v['reference_hits']-v['alternative_hits'])/v['windows']) for k,v in result.items()},
                routes={k:dict(v) for k,v in routes.items()},
                interpretation='Paired input intervention on matched validation subset; not causal proof or retraining result')


def prepare(rows_path, records_path, data_root, output):
    from PIL import Image
    output=Path(output)
    if output.exists():raise ValueError('Use a new diagnostic output; existing results are preserved')
    rows=json.loads(Path(rows_path).read_text())['rows'];indexed(rows)
    records=json.loads(Path(records_path).read_text())
    for r in rows:
        p=Path(data_root)/r['image']
        if file_sha(p)!=r['rgb_sha256']:raise ValueError('Source RGB changed: '+str(p))
        with Image.open(p) as im:r['image_size']=list(im.size)
    report=summarize(rows,records)
    plans=[matched_plan(rows,seed) for seed in (11,29,47)]
    output.mkdir(parents=True)
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for p in plans:
        (output/f'matched_images_seed_{p["seed"]}.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
    (output/'receipt.json').write_text(json.dumps(dict(rows_sha256=file_sha(rows_path),records_sha256=file_sha(records_path),
        diagnostic_source_sha256=file_sha(__file__),inference_performed=False),indent=2))
    return dict(anchors=len(rows),matched={p['seed']:len(p['pairs']) for p in plans},inference_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--rows',required=True,type=Path);p.add_argument('--records',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.rows,a.records,a.data_root,a.output)))


if __name__ == '__main__':main()
