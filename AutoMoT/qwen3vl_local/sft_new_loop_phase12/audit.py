"""Reparse complete original text; validate members, labels, RGB and score denominators."""
import argparse
import json
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase1 import run_integrity as proof
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from . import prompts,evaluate,cache,cache_checks,validation,history,integrity,CONTRACT,UPDATES
from .data import phase1_targets


def verify_evaluations(root,mode):
    root=Path(root);members=proof.read(root/'cpu/evaluation_members.json')
    weight_binding=integrity.verify_evaluation_binding(root,mode)
    probe=proof.read(root/'runs'/mode/'gpu_prefix_probe.json')
    if probe.get('status')!='passed_before_optimizer_updates' or len(probe.get('reports',[]))!=2:
        raise ValueError('Missing two-domain pre-training output parity probe')
    for report in probe['reports']:
        cache_checks.verify_output_evidence(report,report['branches'],require_logits=True)
    facts={tuple(r['key']):r for r in proof.read(root/'cpu/fusion_labels.json')}
    historical=history.load(root/'cpu/historical_contract.json')
    reference=proof.read(root/'cpu/reference_members.json')[mode]
    for phase in (1,2):
        if reference[str(phase)]!=history.primary(historical,phase,mode)['records']:
            raise ValueError('Reference differs from bound historical membership')
    validations=sorted((root/'runs'/mode).glob('val_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
    if [int(p.name.rsplit('_',1)[1]) for p in validations]!=[*range(1000,UPDATES,1000),UPDATES]:
        raise ValueError('Incomplete scheduled validation evidence')
    paths=[root/'evaluation'/mode/'fusion_test',root/'evaluation'/mode/'original_phase1_test',
           root/'evaluation'/mode/'original_phase2_test']
    paths+=sorted((root/'runs'/mode).glob('val_step_*'))
    guard=proof.read(root/'cpu/guard_members.json')
    from qwen3vl_local.sft_new_loop_phase1.training_policy import physical_group
    if any(physical_group(tuple(r['key'])) in history.groups(historical) for r in members['val']):
        raise ValueError('Historical test route used by fusion validation')
    for row in facts.values():
        if phase1_targets(row['answers'],row['phase2_rs'])!=row['answers']:
            raise ValueError('Bundled Phase1 labels contradict authoritative Phase2 RS')
        if physical_group(tuple(row['key'])) in history.groups(historical) and row['split']!='test':
            raise ValueError('Historical test route leaked into fusion train/val')
    if any(physical_group(tuple(r['key'])) in history.groups(historical) for r in guard):
        raise ValueError('Historical test route used by validation guard')
    if any(tuple(r['key']) in facts and r['answers']!=facts[tuple(r['key'])]['answers'] for r in guard):
        raise ValueError('Shared validation guard labels contradict authoritative fusion labels')
    paths += [p/'weak_guard' for p in sorted((root/'runs'/mode).glob('val_step_*'))]
    for folder in paths:
        recorded=proof.read(folder/'metrics.json')
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
            if folder.name.startswith('original_phase') and case.get('historical_case_id')!=item['historical_case_id']:
                raise ValueError('Historical case provenance changed')
            for phase in phases:
                answers=label['answers'] if phase==1 else item['answers']
                module=prompts.P1 if phase==1 else prompts.P2
                spec=module.make_prompt_spec(variant='all_random_order',answers=answers,seed_key=item['seed_key'])
                gt=answers if phase==1 else module.spec_answers(spec)
                arm=case[f'phase{phase}']
                if arm['gt']!=gt or arm['parsed']!=evaluate._parse(arm['text'],spec,phase):
                    raise ValueError('Ground truth/parser mismatch')
            if folder.name=='fusion_test' or folder.name.startswith('val_step_'):
                perf=case['performance']
                cache_checks.verify_output_evidence(perf,[case['phase1'],case['phase2']],require_logits=number==0)
                cache.verify_benchmark(perf['paired_benchmark'],device=recorded['performance']['device'],
                                       dtype=recorded['performance']['model_dtype'])
                for arm in [case['phase1'],case['phase2'],*perf['independent_branches']]:
                    if arm['generated_tokens']>160 or arm['reached_token_limit']!=(arm['generated_tokens']==160):
                        raise ValueError('Generated length contradicts timing conditions')
        calculated=evaluate.metrics(cases,phases);recorded=proof.read(folder/'metrics.json')
        if any(recorded[k]!=v for k,v in calculated.items()):raise ValueError('Metrics do not reproduce original text')
        if folder.name.startswith('original_phase'):
            from .compare import compare
            phase=phases[0];hashes=proof.read(root/'cpu/reference_rgb_sha256.json')
            baseline=[dict(case_index=r['case_index'],historical_case_id=r['historical_case_id'],key=r['key'],rgb=r['rgb'],
                **{f'phase{phase}':dict(gt=r['baseline_gt'],parsed=r['baseline_parsed'])}) for r in expected]
            if proof.read(folder/'historical_comparison.json')!=compare(baseline,cases,hashes,hashes,phase):
                raise ValueError('Historical regression report mismatch')
        if (folder.name=='fusion_test' or folder.name.startswith('val_step_')) and cache.text_prefix.enabled():
            if recorded['performance']['stream_benchmark']!=evaluate.stream_summary(cases):
                raise ValueError('Production stream totals mismatch')
        if folder.name=='fusion_test':
            if any(recorded['performance'][k]!=weight_binding[k] for k in ('device','model_dtype')):
                raise ValueError('Benchmark conditions differ from actual evaluation model binding')
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
