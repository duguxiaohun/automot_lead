"""Conservative, paired regression gate. Raw outputs are parsed by the native parser.

Labels come only from the frozen input. Corrected-label diagnostics must never
replace this comparison or hide a loss on the original evaluation contract.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from qwen3vl_local.sft_new_loop_phase1.evaluation_integrity import item_from_json
from qwen3vl_local.sft_new_loop_phase1 import prompts as native


def read_cases(path):
    path = Path(path)
    files = sorted(path.glob('cases_rank*.jsonl')) if path.is_dir() else [path]
    if not files: raise ValueError('No case files')
    result = {}
    for file in files:
        for line in file.open():
            row = json.loads(line); index = row['case_index']
            if type(index) is not int or index in result: raise ValueError('Duplicate/invalid case index')
            result[index] = row
    return result


def expected(item, index):
    row = item.row
    return dict(case_index=index, scenario=row.scenario, route_id=row.route_id,
                frame_id=row.frame_id, prompt_spec=native.prompt_spec_to_json(item.spec),
                gt={key: 'YES' if value else 'NO' for key, _, value in native.spec_metric_items(item.spec)})


def metrics(count):
    total = count['n']; tp, fp, fn = (count[k] for k in ('tp', 'fp', 'fn'))
    return dict(n=total, accuracy=count['ok']/total if total else None,
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                format_valid=count['valid']/total if total else None)


def compare(frozen, baseline, candidate):
    if frozen['phase'] != 1 or len(frozen['items']) != frozen['expected_total']:
        raise ValueError('Wrong or incomplete frozen Phase1 input')
    ids = set(range(frozen['expected_total']))
    if not ids or set(baseline) != ids or set(candidate) != ids:
        raise ValueError('Full identical membership required; missing/extra cases')
    audit = bool(frozen.get('arguments', {}).get('audit_prompt', False))
    counters = [defaultdict(Counter), defaultdict(Counter)]
    losses, gains, answer_losses, answer_gains = [], [], [], []
    joint = [0, 0]
    for index, data in enumerate(frozen['items']):
        item = item_from_json(1, data); exp = expected(item, index); oks = []
        parsed = []
        for side, records in enumerate((baseline, candidate)):
            record = records[index]
            if any(record.get(k) != v for k, v in exp.items()):
                raise ValueError(f'Case identity/spec/GT differs at {index}')
            if not isinstance(record.get('raw_output'), str): raise ValueError('Raw output required')
            pred = native.parse_phase1_output(record['raw_output'], spec=item.spec, audit=audit)
            parsed.append(pred); ok = all(pred.get(k) == v for k, v in exp['gt'].items())
            oks.append(ok); joint[side] += ok
            for output, metric, truth in native.spec_metric_items(item.spec):
                value = pred.get(output); valid = value in ('YES', 'NO'); good = value == exp['gt'][output]
                main_output = item.focus if item.focus in item.spec.output_keys else 'DETAIL'
                groups = (metric, 'ALL') + (('FOCUS/'+item.focus,) if output == main_output else ())
                for group in groups:
                    c = counters[side][group]; c['n'] += 1; c['ok'] += good; c['valid'] += valid
                    c['tp'] += truth and value == 'YES'; c['fp'] += not truth and value == 'YES'
                    c['fn'] += truth and value != 'YES'
        if oks[0] and not oks[1]: losses.append(index)
        if oks[1] and not oks[0]: gains.append(index)
        for output, truth in exp['gt'].items():
            if parsed[0].get(output) == truth and parsed[1].get(output) != truth:
                answer_losses.append(dict(case_index=index, key=output))
            if parsed[1].get(output) == truth and parsed[0].get(output) != truth:
                answer_gains.append(dict(case_index=index, key=output))
    reports = [{k:metrics(v) for k,v in side.items()} for side in counters]
    declines = [dict(group=k, metric=m, baseline=v, candidate=reports[1][k][m])
                for k, stats in reports[0].items() for m,v in stats.items()
                if m != 'n' and v is not None and reports[1][k][m] < v - 1e-12]
    return dict(status='review_required' if losses or answer_losses or declines else 'no_observed_regression',
                scope='frozen labels, full paired membership; not a guarantee for unseen RGB',
                total=len(ids), joint_correct=joint, joint_accuracy=[x/len(ids) for x in joint],
                old_correct_losses=losses, gains=gains, old_correct_answer_losses=answer_losses,
                answer_gains=answer_gains, declines=declines, baseline=reports[0], candidate=reports[1])
