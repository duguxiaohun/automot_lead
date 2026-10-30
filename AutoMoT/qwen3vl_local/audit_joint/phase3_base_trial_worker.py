"""Isolated adapters around the unchanged native Phase3 trainer/evaluator."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from . import phase3_base_trial as trial
from qwen3vl_local.audit_joint.phase3_base_dense.baseline import install_baseline, independent_work


def validate_job(job):
    if (job['policy'] != trial.POLICY or job['profile'] != trial.profile()
            or job['reference_sha256'] != trial.file_sha(trial.REFERENCE)
            or job['release_sha256'] != trial.verified_sources()
            or trial.file_sha(job['index']) != job['index_sha256']
            or job['index_sha256'] != trial.reference()['index_sha256']):
        raise ValueError('Trial/source/data binding changed')
    if job['stage'] not in ('quick', 'complete') or job['stop_after_epoch'] != (1 if job['stage'] == 'quick' else None):
        raise ValueError('Invalid trial stage')


def install_reference(ref, split='val'):
    from qwen3vl_local.sft_new_loop_phase3 import sequence_evaluation
    sequence_evaluation.select_work = lambda work, *, seed, limit=0, stride=1: trial.select_reference(
        work, ref, seed=seed, limit=limit, stride=stride, split=split)


def legacy_single_generate(bundle, rows, *, max_new_tokens, phase3=False, reuse_rgb=False):
    """Use the historical greedy KV loop, independently on each of four ranks."""
    import torch
    from qwen3vl_local.sft_v3.train import KVState, _student_generate_kv
    if not phase3 or reuse_rgb or len(rows) != 1:
        raise ValueError('Controlled trial generation requires one Phase3 item without visual reuse')
    row = rows[0]
    model = bundle.model.module if hasattr(bundle.model, 'module') else bundle.model
    with torch.no_grad():
        from qwen3vl_local.qwen35 import text_prefix
        out = (text_prefix.prefill(bundle,row) if text_prefix.enabled() else
               model(**row, use_cache=True, return_dict=True))
        state = KVState(decoded_input_ids=row['input_ids'], cache_input_ids=row['input_ids'],
            attention_mask=row['attention_mask'], past_key_values=out.past_key_values,
            rope_deltas=getattr(out, 'rope_deltas', None), next_logits=out.logits[:, -1, :])
        return [_student_generate_kv(bundle, state, max_new_tokens)[0]]


def make_args(job):
    from qwen3vl_local.sft_new_loop_phase3 import train
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        args = train.parse_args()
    finally:
        sys.argv = saved
    for key, value in dict(index=job['index'], data_root=job['data_root'], model_dir=job['model_dir'],
        output_dir=str(Path(job['output'])/'train'), resume=job['resume'],
        stop_after_epoch=job['stop_after_epoch'], save_training_state=True,
        split='train', eval_split='val', history_rgb_mode='4rgb', action_output_mode='choice',
        prompt_variant=trial.reference()['prompt_variant'], batch_size=1, grad_accum=1,
        anchor_stride=1, train_route_diverse=True, num_epochs=3, max_steps=0, max_frames=0, max_eval_frames=0,
        focus_balance_count=1024, learning_rate=1e-5, warmup_steps=2000,
        seed=20260904, lora_rank=16, lora_alpha=32, lora_dropout=.05,
        weight_decay=0., max_grad_norm=1., format_loss_weight=.25, max_length=8192,
        gradient_checkpointing=True, reuse_frozen_rgb=False, lora_vision_scope='off',
        eval_steps=2560, generation_eval_steps=2560, eval_balance_count=16,
        generation_eval_balance_count=32, generation_eval_sampling_seed=20260904,
        generation_eval_batch_size=1, generation_eval_max_new_tokens=64,
        save_steps=0, save_best_generation=False, save_best_val=False, save_final=True,
        device='cuda', no_tb=True, sampling_only=False).items():
        setattr(args, key, value)
    # Native recovery serializes all additional args into its strict binding.
    # External selector/generation adapters cannot change silently on resume.
    args.base_trial = dict(policy=trial.POLICY, profile=trial.profile(),
        reference_sha256=job['reference_sha256'], release_sha256=job['release_sha256'],
        worker_sha256=trial.file_sha(Path(__file__)),
        coordinator_sha256=trial.file_sha(Path(trial.__file__)),
        generation='original_single_item_kv_loop_four_rank',
        validation='historical_frozen_ids', old_models_evaluated=False)
    if job.get('suite_binding'):
        args.base_trial['suite_binding']=job['suite_binding']
    return args


def preflight(job):
    from importlib.metadata import version
    from qwen3vl_local.sft_new_loop_phase3 import train, training_state
    from qwen3vl_local.sft_new_loop_phase3.preflight import check_index, check_model
    from qwen3vl_local.sft_new_loop_phase3.sampling import primary_action_distribution
    args = make_args(job)
    ref = trial.reference()
    reports = dict(dataset=check_index(args.index, action_output_mode='choice', prompt_variant=args.prompt_variant),
                   model=check_model(args.model_dir))
    rows = train._read_rows(Path(args.index), 'train', data_root=Path(args.data_root))
    selected, audit = independent_work(rows, args, epoch=0, world_size=4)
    if len(selected) != 10240 or audit['policy'] != args.base_trial['profile']['sampling_policy']:
        raise ValueError('Unexpected independent sampling budget/policy')
    old = train._balanced_work(rows, target_per_bin=1024, seed=args.seed,
        route_diverse=True, action_output_mode='choice')
    counts = lambda items: primary_action_distribution(dict(Counter(x.row.action_signature for x in items)))['counts']
    old_counts, new_counts = counts(old), counts(selected)
    if old_counts != ref['baseline_training_presentations_per_epoch']:
        raise ValueError('Old action projection on the same index changed')
    val_rows = train._read_rows(Path(args.index), 'val', data_root=Path(args.data_root))
    install_reference(ref)
    args.validation_sampling = {}
    for kind, limit, seed in [('loss', 16, args.seed+1009), ('generation', 32, args.seed)]:
        work, report = train._validation_work(val_rows, equal_events=True, history_rgb_mode='4rgb',
            target_per_bin=limit, seed=seed, action_output_mode='choice')
        args.validation_sampling[kind] = report
    # Full native resume verification is also performed before optimizer loading
    # in train(); this CPU pass checks the exact configuration and tensor receipt.
    if args.resume:
        from qwen3vl_local.sft_new_loop_phase3.training_schedule import resolve_warmup
        args.training_index_sha256 = job['index_sha256']
        args.learning_rate_schedule = resolve_warmup(7680, steps=2000, ratio=args.warmup_ratio)
        # Native training attaches same-RS metadata only for binary, not this trial.
        expected = training_state.binding(args, 4, 2560)
        recovered = training_state.read(Path(args.resume), expected)
        if recovered['next_epoch'] != job['resume_epoch']:
            raise ValueError('Resume directory and tensor epoch disagree')
        reports['resumed_next_epoch'] = recovered['next_epoch']
    reports.update(profile=trial.profile(), baseline_action_presentations=old_counts,
        environment=dict(python=sys.version, torch=version('torch'), transformers=version('transformers')),
        new_action_presentations=new_counts,
        action_delta={k:new_counts.get(k, 0)-v for k,v in old_counts.items()},
        sampling={k:v for k,v in audit.items() if k not in ('next_history', 'ordered_ids', 'segments')},
        validation=args.validation_sampling, old_models_evaluated=False,
        interpretation='CPU sampling/exposure audit; not model-quality evidence')
    trial.write(Path(job['output'])/'preflight.json', reports)
    print(json.dumps(dict(status='preflight_passed', action_delta=reports['action_delta'])), flush=True)


def summarize(job):
    from qwen3vl_local.sft_new_loop_phase3.prompts import (
        make_prompt_spec, parse_action_output, choice_target_action, prompt_spec_to_json,
    )
    root = Path(job['output'])/'train'
    path = root/('generation_val_cases.jsonl' if job['stage'] == 'quick' else 'final_generation_val_cases.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    step = 2560 if job['stage'] == 'quick' else 7680
    records = [r for r in records if r['step'] == step]
    if len(records) != 320:
        raise ValueError('Expected exactly 320 frozen validation outputs')
    identities, actions, events, seen = [], {}, Counter(), set()
    correct = valid = 0
    for row in records:
        saved = row['prompt_spec']
        spec = make_prompt_spec(variant=saved['variant'], answers=row['answers'],
            seed_key=saved['seed_key'], context_id=saved['context_id'], road_structure=saved['road_structure'],
            goal_xy=saved.get('goal_xy'), context_detail=saved.get('context_detail', ''),
            current_speed_mps=saved.get('current_speed_mps'), action_output_mode='choice')
        if prompt_spec_to_json(spec) != saved:
            raise ValueError('Saved validation spec changed')
        identity = (row['scenario'], row['route_id'], row['frame_id'], row['context_id'])
        if identity in seen:
            raise ValueError('Duplicate validation output')
        seen.add(identity)
        target = choice_target_action(spec)
        parsed = parse_action_output(row['raw_output'], spec=spec)
        if parsed != row['parsed']:
            raise ValueError('Raw prediction disagrees with saved parser output')
        is_valid = all(v is not None for v in parsed.values())
        prediction = row['raw_output'].strip()
        ok = is_valid and prediction == target
        if bool(row['all_ok']) != ok or bool(row['format_valid']) != is_valid:
            raise ValueError('Saved validation correctness mismatch')
        correct += ok; valid += is_valid; events[row['context_id']] += 1
        cell = actions.setdefault(target, dict(support=0, correct=0, predicted=0))
        cell['support'] += 1; cell['correct'] += ok
        if is_valid:
            actions.setdefault(prediction, dict(support=0, correct=0, predicted=0))['predicted'] += 1
        # The pinned index's valid rows use the native reader's default reason.
        # Include road premise and reason, exactly as native case_identity does.
        from qwen3vl_local.sequence_batch import digest
        identities.append(digest((*identity, spec.road_structure, 'wrong_road_structure')))
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_IDS
    if events != dict.fromkeys(CONTEXT_IDS, 32):
        raise ValueError('Validation event quotas differ from the frozen plan')
    if identities != trial.reference()['validation']['generation']['selected_ids']:
        raise ValueError('Validation identities/order differ from the frozen plan')
    for cell in actions.values():
        cell['recall'] = cell['correct']/cell['support'] if cell['support'] else None
        cell['precision'] = cell['correct']/cell['predicted'] if cell['predicted'] else None
    baseline = trial.reference()['baseline_validation']
    deltas = {}
    for action, old in baseline['actions'].items():
        current = actions.get(action, dict(support=0, correct=0, predicted=0))
        if current['support'] != old['support']:
            raise ValueError('Frozen validation target support changed: '+action)
        deltas[action] = dict(correct=current['correct']-old['correct'],
            predicted=current['predicted']-old['predicted'],
            recall_pp=100*(current['correct']-old['correct'])/old['support'] if old['support'] else None)
    report = dict(step=step, optimizer_updates=step, planned_optimizer_updates=7680,
        correct=correct, total=320, exact=correct/320, format_valid=valid/320,
        actions=actions, event_counts=dict(events), reference_sha256=job['reference_sha256'],
        historical_three_epoch_validation=baseline,
        action_delta_to_historical_three_epoch=deltas,
        delta_to_historical_three_epoch_pp=(correct-baseline['correct'])/320*100,
        same_training_budget_as_baseline=job['stage'] == 'complete',
        test_used=False, old_models_evaluated=False,
        conclusion='new first-epoch diagnostic; recovery not established' if job['stage'] == 'quick'
                   else 'full-budget validation comparison; review action guards and test separately')
    trial.write(Path(job['output'])/'validation_summary.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--operation', required=True, choices=['preflight', 'train', 'summarize', 'test'])
    a = parser.parse_args()
    job = trial.read(a.job)
    validate_job(job)
    if a.operation == 'preflight':
        preflight(job)
    elif a.operation == 'summarize':
        summarize(job)
    elif a.operation == 'train':
        if os.environ.get('WORLD_SIZE') != '4':
            raise ValueError('Training requires four ranks')
        from qwen3vl_local.sft_new_loop_phase3 import train
        from qwen3vl_local import batch_runtime
        install_reference(trial.reference())
        install_baseline()
        batch_runtime.greedy_batch = legacy_single_generate
        train.train(make_args(job))
    else:
        if job['stage'] != 'complete':
            raise ValueError('No test during the quick stage')
        trial.verify_training(job)
        from qwen3vl_local.sft_new_loop_phase3 import eval as evaluate
        install_reference(trial.reference(), split='test')
        sys.argv = [sys.argv[0], '--index', job['index'], '--data-root', job['data_root'],
            '--model-dir', job['model_dir'], '--adapter-dir', str(Path(job['output'])/'train/final'),
            '--output-dir', str(Path(job['output'])/'test'), '--device', 'cuda', '--split', 'test',
            '--history-rgb-mode', '4rgb', '--action-output-mode', 'choice', '--batch-size', '1',
            '--no-reuse-frozen-rgb', '--no-timestamp-output', '--no-save-error-rgb',
            '--expected-total-cases', '380']
        evaluate.evaluate(evaluate.parse_args())


if __name__ == '__main__':
    main()
