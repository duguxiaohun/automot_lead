"""Reparse complete original text; validate members, labels, RGB and score denominators."""
import argparse
import json
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from . import prompts,evaluate,CONTRACT


def verify_evaluations(root,mode):
    root=Path(root);members=proof.read(root/'cpu/evaluation_members.json')
    facts={tuple(r['key']):r for r in proof.read(root/'cpu/fusion_labels.json')}
    reference=proof.read(root/'cpu/reference_members.json')
    paths=[root/'evaluation'/mode/'fusion_test',root/'evaluation'/mode/'original_phase1_test',
           root/'evaluation'/mode/'original_phase2_test']
    paths+=sorted((root/'runs'/mode).glob('val_step_*'))
    for folder in paths:
        cases=[json.loads(l) for l in (folder/'cases.jsonl').open() if l.strip()]
        phases=(1,2)
        if folder.name.startswith('original_phase'):
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
        calculated=evaluate.metrics(cases,phases);recorded=proof.read(folder/'metrics.json')
        if any(recorded[k]!=v for k,v in calculated.items()):raise ValueError('Metrics do not reproduce original text')
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
