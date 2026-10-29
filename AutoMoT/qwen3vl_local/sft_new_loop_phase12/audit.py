"""Reparse complete original text; validate members, labels, RGB and score denominators."""
import argparse
import json
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from . import prompts,evaluate,cache,validation,CONTRACT,UPDATES


def verify_evaluations(root,mode):
    root=Path(root);members=proof.read(root/'cpu/evaluation_members.json')
    facts={tuple(r['key']):r for r in proof.read(root/'cpu/fusion_labels.json')}
    reference=proof.read(root/'cpu/reference_members.json')
    validations=sorted((root/'runs'/mode).glob('val_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
    if [int(p.name.rsplit('_',1)[1]) for p in validations]!=[*range(1000,UPDATES,1000),UPDATES]:
        raise ValueError('Incomplete scheduled validation evidence')
    paths=[root/'evaluation'/mode/'fusion_test',root/'evaluation'/mode/'original_phase1_test',
           root/'evaluation'/mode/'original_phase2_test']
    paths+=sorted((root/'runs'/mode).glob('val_step_*'))
    guard=proof.read(root/'cpu/guard_members.json')
    paths += [p/'weak_guard' for p in sorted((root/'runs'/mode).glob('val_step_*'))]
    for folder in paths:
        cases=[json.loads(l) for l in (folder/'cases.jsonl').open() if l.strip()]
        phases=(1,2)
        if folder.name=='weak_guard':phases=(1,);expected=guard
        elif folder.name.startswith('original_phase'):
            phase=int(folder.name[len('original_phase')]);phases=(phase,);expected=reference[str(phase)]
        else:expected=members['val' if folder.name.startswith('val_') else 'test']
        if len(cases)!=len(expected):raise ValueError('Incomplete evaluation: '+str(folder))
        for number,(case,item) in enumerate(zip(cases,expected)):
            if case['case_index']!=number or case['key']!=item['key'] or case['balance_key']!=item['balance_key']:
                raise ValueError('Evaluation membership/order mismatch')
            label=facts[tuple(item['key'])] if len(phases)==2 else item
            if case['rgb']!=[label['paths'][i] for i in history_rgb_indices(mode)]:raise ValueError('RGB order differs')
            for phase in phases:
                answers=label['answers'] if phase==1 else item['answers']
                module=prompts.P1 if phase==1 else prompts.P2
                spec=module.make_prompt_spec(variant='all_random_order',answers=answers,seed_key=item['seed_key'])
                gt=answers if phase==1 else module.spec_answers(spec)
                arm=case[f'phase{phase}']
                if arm['gt']!=gt or arm['parsed']!=evaluate._parse(arm['text'],spec,phase):
                    raise ValueError('Ground truth/parser mismatch')
            if folder.name=='fusion_test':
                perf=case['performance']
                if not perf['full_answer_equivalence_checked']:raise ValueError('Missing full-answer cache parity')
                cache.assert_answers({'branches':[case['phase1'],case['phase2']]},
                                     {'branches':perf['independent_branches']})
                for trial in perf['paired_benchmark']['trials']:
                    if (set(trial)!={'shared_seconds','independent_seconds'}
                            or any(not isinstance(v,(float,int)) or not 0<v<float('inf') for v in trial.values())):
                        raise ValueError('Invalid benchmark timing')
        calculated=evaluate.metrics(cases,phases);recorded=proof.read(folder/'metrics.json')
        if any(recorded[k]!=v for k,v in calculated.items()):raise ValueError('Metrics do not reproduce original text')
        if folder.name=='fusion_test':
            measured=evaluate.benchmark_summary(cases)
            if measured['cases']!=min(8,len(cases)) or measured['trials']!=2*measured['cases']:
                raise ValueError('Incomplete paired performance measurement')
            if any(len(c['performance']['paired_benchmark']['trials']) not in (0,2) for c in cases):
                raise ValueError('Missing alternating timing trial')
            if recorded['performance']['paired_benchmark']!=measured:raise ValueError('Benchmark totals mismatch')
    incumbent=None;best_score=None
    for folder in validations:
        expected=validation.decision(folder,incumbent,proof.read(root/'cpu/guard_coverage.json'))
        promote=incumbent is None or (expected['eligible'] and expected['score']>best_score)
        if proof.read(folder/'selection_review.json')!=dict(**expected,promoted=promote):
            raise ValueError('Checkpoint promotion does not reproduce validation guard')
        if promote:incumbent=folder;best_score=expected['score']
    if validations:
        selected=proof.read(root/'runs'/mode/'selection.json')
        if (selected['step']!=int(incumbent.name.rsplit('_',1)[1]) or selected['score']!=best_score
                or selected['adapter']!=f"best_step_{selected['step']}"):
            raise ValueError('Selected checkpoint bypassed validation protection')
    return dict(status='all_complete_original_outputs_recomputed',evaluations=len(paths))


def verify_bundle(root):
    root=Path(root);manifest=proof.read(root/'bundle_manifest.json')
    if manifest['contract']!=CONTRACT:raise ValueError('Wrong bundle contract')
    for name,expected in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or proof.sha(root/name)!=expected:
            raise ValueError('Bundle evidence changed: '+name)
    return verify_evaluations(root,manifest['mode'])


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);a=p.parse_args()
    print(json.dumps(verify_bundle(a.bundle),ensure_ascii=False))

if __name__=='__main__':main()
