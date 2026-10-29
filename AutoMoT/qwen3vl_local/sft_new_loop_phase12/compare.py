"""Strict same-case/RGB-SHA/question comparisons; preserve all old-correct losses."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def value(v):
    if type(v)==bool:return v
    return {'YES':True,'NO':False}.get(v)


def rgb(path):return Path(*Path(path).parts[-4:]).as_posix()


def compare(old,new,old_hashes,new_hashes,phase):
    previous={r['case_index']:r for r in old}
    if len(previous)!=len(old) or len({r['case_index'] for r in new})!=len(new):raise ValueError('Duplicate case index')
    old_hashes={rgb(k):v for k,v in old_hashes.items()};new_hashes={rgb(k):v for k,v in new_hashes.items()}
    counts=Counter();questions=defaultdict(Counter);losses=[];matched_joint=Counter()
    for row in new:
        oldrow=previous.get(row['case_index'])
        if oldrow is None:counts['no_old_case_index']+=1;continue
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
        for key in common:
            truth=value(b['gt'][key]);old_ok=value(a['parsed'].get(key)) is truth;new_ok=value(b['parsed'].get(key)) is truth
            label=('both_correct' if new_ok else 'old_correct_loss') if old_ok else ('fixed' if new_ok else 'both_wrong')
            questions[key][label]+=1;questions[key]['total']+=1;old_exact &= old_ok;new_exact &= new_ok
            if label=='old_correct_loss':losses.append(dict(case_index=row['case_index'],key=row['key'],question=key,rgb=after))
        matched_joint['total']+=1;matched_joint['old_exact_common_questions']+=old_exact;matched_joint['new_exact_common_questions']+=new_exact
    return dict(scope='exact_case_index_and_physical_member_and_RGB_SHA_and_common_question_labels',counts=dict(counts),
        questions=dict(questions),joint_common_questions=dict(matched_joint),old_correct_losses=losses,
        review_required=bool(losses),warning='New P1 all-eight totals and historical mixed-question totals are different protocols; unmatched cases excluded explicitly.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-cases',type=Path,required=True)
    p.add_argument('--baseline-inputs',type=Path,required=True);p.add_argument('--new-cases',type=Path,required=True)
    p.add_argument('--new-rgb-bindings',type=Path,required=True);p.add_argument('--phase',type=int,choices=(1,2),required=True)
    a=p.parse_args();read=lambda f:[json.loads(l) for l in f.open() if l.strip()]
    result=compare(read(a.baseline_cases),read(a.new_cases),json.loads(a.baseline_inputs.read_text())['image_sha256'],
        json.loads(a.new_rgb_bindings.read_text()),a.phase)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
