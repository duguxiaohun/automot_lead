"""Strict same-case/RGB-SHA/question comparisons; preserve all old-correct losses."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def value(v):
    if type(v)==bool:return v
    return {'YES':True,'NO':False}.get(v)


def rgb(path):return Path(*Path(path).parts[-4:]).as_posix()


def member(row):
    return tuple(row.get('key',[row.get('scenario'),row.get('route_id'),row.get('frame_id')]))


def signature(row,phase):
    arm=row.get(f'phase{phase}',row)
    return (member(row),tuple(rgb(p) for p in row.get('rgb',row.get('history_rgb_paths_used',[]))),
            tuple(sorted(arm['gt'])))


def hashes_by_rgb(hashes):
    result={}
    for path,sha in hashes.items():
        key=rgb(path)
        if key in result and result[key]!=sha:raise ValueError('Conflicting RGB hash aliases: '+key)
        result[key]=sha
    return result


def bind_native_baseline(rows, inputs, phase):
    """Recover IDs only from verified native input/completion/case evidence."""
    from . import history
    from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
    inputs = Path(inputs)
    source = history.import_inputs(inputs)
    if source['phase'] != phase:
        raise ValueError('Historical baseline phase differs from requested comparison')
    originals = {}
    for name in source['case_shards']:
        for line in (inputs.parent/name).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                originals[row['case_index']] = row
    bound = []
    for row in rows:
        row = dict(row)
        supplied_id = row.pop('historical_case_id', None)
        expected_id = f"{source['inputs_sha256']}:{row['case_index']}"
        if originals.get(row['case_index']) != row or supplied_id not in (None, expected_id):
            raise ValueError('Baseline case is not the verified original historical output')
        bound.append(dict(row, historical_case_id=expected_id))
    # Detect input replacement while validating/reading native shards.
    if proof.sha(inputs) != source['inputs_sha256']:
        raise ValueError('Historical input changed during comparison')
    return bound


def compare(old,new,old_hashes,new_hashes,phase):
    previous={r['case_index']:r for r in old}
    if len(previous)!=len(old) or len({r['case_index'] for r in new})!=len(new):raise ValueError('Duplicate case index')
    by_member=defaultdict(list)
    for row in old:by_member[member(row)].append(row)
    bound={r['historical_case_id']:r for r in old if r.get('historical_case_id')}
    if len(bound)!=sum(bool(r.get('historical_case_id')) for r in old):raise ValueError('Duplicate bound historical case id')
    old_hashes=hashes_by_rgb(old_hashes);new_hashes=hashes_by_rgb(new_hashes)
    counts=Counter();questions=defaultdict(Counter);losses=[];matched_joint=Counter();used=set()
    for row in new:
        candidates=by_member.get(member(row),[])
        if not candidates:counts['no_old_physical_member']+=1;continue
        if row.get('historical_case_id'):
            exact=bound.get(row['historical_case_id'])
            if exact is None or member(exact)!=member(row):counts['historical_binding_mismatch']+=1;continue
            candidates=[exact]
        same=[r for r in candidates if signature(r,phase)==signature(row,phase)]
        if same:candidates=same
        elif len(candidates)>1:
            # An index alone cannot disambiguate repeated tasks on one physical frame.
            counts['ambiguous_old_member']+=1;continue
        if len(candidates)!=1:counts['ambiguous_old_member']+=1;continue
        oldrow=candidates[0]
        if oldrow['case_index'] in used:counts['duplicate_new_match']+=1;continue
        oldkey=oldrow.get('key',[oldrow.get('scenario'),oldrow.get('route_id'),oldrow.get('frame_id')])
        if list(oldkey)!=row['key']:counts['different_member_at_index']+=1;continue
        before=[rgb(p) for p in oldrow.get('rgb',oldrow.get('history_rgb_paths_used',[]))]
        after=[rgb(p) for p in row['rgb']]
        if not before or before!=after:counts['different_rgb_order']+=1;continue
        if any(p not in old_hashes or p not in new_hashes or old_hashes[p]!=new_hashes[p] for p in before):
            counts['missing_or_different_rgb_sha']+=1;continue
        a=oldrow.get(f'phase{phase}',oldrow);b=row[f'phase{phase}'];common=sorted(set(a['gt'])&set(b['gt']))
        if not common:counts['no_common_question']+=1;continue
        if any(value(a['gt'][q]) is None or value(a['gt'][q])!=value(b['gt'][q]) for q in common):
            counts['different_labels']+=1;continue
        counts['matched_cases']+=1;old_exact=new_exact=True
        used.add(oldrow['case_index'])
        if oldrow['case_index']!=row['case_index']:counts['renumbered_matches']+=1
        counts['matched_questions']+=len(common)
        counts['new_questions_without_baseline']+=len(set(b['gt'])-set(a['gt']))
        counts['old_questions_not_asked']+=len(set(a['gt'])-set(b['gt']))
        for key in common:
            truth=value(b['gt'][key]);old_ok=value(a['parsed'].get(key)) is truth;new_ok=value(b['parsed'].get(key)) is truth
            label=('both_correct' if new_ok else 'old_correct_loss') if old_ok else ('fixed' if new_ok else 'both_wrong')
            questions[key][label]+=1;questions[key]['total']+=1;old_exact &= old_ok;new_exact &= new_ok
            if label=='old_correct_loss':losses.append(dict(case_index=row['case_index'],key=row['key'],question=key,rgb=after))
        matched_joint['total']+=1;matched_joint['old_exact_common_questions']+=old_exact;matched_joint['new_exact_common_questions']+=new_exact
    matched=counts['matched_cases'];complete=bool(old and new) and matched==len(old)==len(new)
    evidence_complete=complete and not counts['new_questions_without_baseline'] and not counts['old_questions_not_asked']
    return dict(scope='unique_physical_member_ordered_RGB_SHA_and_common_question_labels',counts=dict(counts),
        coverage=dict(old_cases=len(old),new_cases=len(new),matched_cases=matched,
                      old_fraction=matched/max(1,len(old)),new_fraction=matched/max(1,len(new)),
                      full_case_coverage=complete,full_question_coverage=evidence_complete),
        questions=dict(questions),joint_common_questions=dict(matched_joint),old_correct_losses=losses,
        status='regression_detected' if losses else ('no_regression_on_complete_comparison' if evidence_complete else 'insufficient_comparable_evidence'),
        review_required=bool(losses) or not evidence_complete,warning='New P1 all-eight totals and historical mixed-question totals are different protocols; unmatched cases/questions prevent full non-regression certification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-cases',type=Path,required=True)
    p.add_argument('--baseline-inputs',type=Path,required=True);p.add_argument('--new-cases',type=Path,required=True)
    p.add_argument('--new-rgb-bindings',type=Path,required=True);p.add_argument('--phase',type=int,choices=(1,2),required=True)
    a=p.parse_args();read=lambda f:[json.loads(l) for l in f.open() if l.strip()]
    old,new=read(a.baseline_cases),read(a.new_cases)
    if any(r.get('historical_case_id') for r in new):
        old=bind_native_baseline(old,a.baseline_inputs,a.phase)
    result=compare(old,new,json.loads(a.baseline_inputs.read_text())['image_sha256'],
        json.loads(a.new_rgb_bindings.read_text()),a.phase)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['review_required']:raise SystemExit(2)

if __name__=='__main__':main()
