"""Early base input/resource checks and an adapter-compatible final test.

Keep the frozen base trainer/release unchanged. No old model is evaluated.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys

from . import phase3_base_trial as trial, phase3_base_trial_worker as base
from .phase3_base_dense import baseline
from .phase3_suite import check_model_snapshot

TEST_MAX_NEW_TOKENS = 256  # Preserve the frozen evaluator's historical default.


def test_arguments(job):
    # Native LoRA evaluation obtains RGB mode and prompt from the saved adapter.
    # Passing --history-rgb-mode with --adapter-dir is explicitly forbidden.
    return ['--index', job['index'], '--data-root', job['data_root'],
        '--model-dir', job['model_dir'], '--adapter-dir', str(Path(job['output'])/'train/final'),
        '--output-dir', str(Path(job['output'])/'test'), '--device', 'cuda', '--split', 'test',
        '--action-output-mode', 'choice', '--batch-size', '1', '--no-reuse-frozen-rgb',
        '--no-timestamp-output', '--no-save-error-rgb', '--expected-total-cases', '380',
        '--max-new-tokens', str(TEST_MAX_NEW_TOKENS)]


def items(job):
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    args = base.make_args(job)
    rows = {s: train._read_rows(Path(job['index']), s, data_root=Path(job['data_root']))
            for s in ('train', 'val', 'test')}
    found = {}
    def add(kind, work):
        for item in work:
            key = baseline.digest(dict(kind=kind, spec=asdict(item.spec), images=item.row.history_rgb_paths))
            found[key] = (kind, item)
    history = None
    for epoch in range(3):
        work, report = baseline.independent_work(rows['train'], args, epoch=epoch, world_size=4, history=history)
        history = report['next_history']; add('train', work)
    base.install_reference(trial.reference())
    for kind, count, seed in [('loss', args.eval_balance_count, args.seed+1009),
                              ('generation', args.generation_eval_balance_count, args.seed)]:
        work, _ = train._validation_work(rows['val'], equal_events=True, history_rgb_mode='4rgb',
            target_per_bin=count, seed=seed, action_output_mode='choice')
        add(kind, work)
    candidates = evaluate._balanced_cases(rows['test'], cases_per_bin=0, seed=args.seed,
        action_output_mode='choice', require_invalid_coverage=True)
    work, _ = trial.select_reference(candidates, trial.reference(), seed=args.seed, limit=0, split='test')
    add('test', work)
    return found


def pack(bundle, kind, item, args):
    from qwen3vl_local.sft_new_loop_phase3 import train
    from qwen3vl_local.sft_new_loop_phase3.prompts import build_action_target
    from qwen3vl_local.batch_runtime import encode_messages
    images = train._load_images(item.row.history_rgb_paths)
    if kind in ('train', 'loss'):
        result = train._build_inputs(bundle, images=images, history_rgb_mode='4rgb',
            prompt_variant=args.prompt_variant, target=build_action_target(item.spec), spec=item.spec,
            max_length=args.max_length, format_loss_weight=args.format_loss_weight)
        if result is None:
            raise ValueError('Base input exceeds token budget before training: '+item.spec.seed_key)
        if not bool(result['loss_weights'].gt(0).any()):
            raise ValueError('Base input has no supervised tokens')
        grid = result['vision']['image_grid_thw']
    else:
        result = encode_messages(bundle, train.build_action_messages(images=images, spec=item.spec,
            audit=False, history_rgb_mode='4rgb', prompt_variant=args.prompt_variant))
        grid = result['image_grid_thw']
        decode_limit = TEST_MAX_NEW_TOKENS if kind == 'test' else args.generation_eval_max_new_tokens
        if result['input_ids'].shape[-1]+decode_limit > args.max_length:
            raise ValueError('Base generation input exceeds token budget before training')
    if len(grid) != 4:
        raise ValueError('Base requires exactly four RGB inputs')
    return result, dict(kind=kind, tokens=int(result['input_ids'].shape[-1]),
                       patches=int(grid.prod(-1).sum()), grid=grid.tolist())


def frontier(shapes):
    selected = []
    for kind in sorted({s['kind'] for s in shapes.values()}):
        costs = {}
        for key, shape in shapes.items():
            if shape['kind'] == kind:
                costs.setdefault((shape['tokens'], shape['patches']), key)
        for cost, key in costs.items():
            if not any(other != cost and all(a >= b for a, b in zip(other, cost)) for other in costs):
                selected.append(key)
    return selected


def prepare(job, processor=None):
    from qwen3vl_local.qwen35.backend import LocalProcessor
    base.preflight(job)
    if processor is None:
        processor = LocalProcessor.from_pretrained(job['model_dir'], local_files_only=True)
    bundle = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer, device='cpu')
    args = base.make_args(job); shapes = {}
    work = items(job)
    for i, (key, (kind, item)) in enumerate(work.items(), 1):
        _, shapes[key] = pack(bundle, kind, item, args)
        if i % 500 == 0:
            print(f'[base inputs] {i}/{len(work)}', flush=True)
    report = dict(shapes=shapes, probe_ids=frontier(shapes),
                  scope='all three epoch inputs, frozen loss/val/test; no old model inference')
    trial.write(Path(job['output'])/'base_input_admission.json', report)
    return report


def probe(job):
    import torch
    from qwen3vl_local.sft_new_loop_phase3 import train, eval as evaluate
    args = base.make_args(job)
    admitted = trial.read(Path(job['output'])/'base_input_admission.json')
    rank, local, world = train.setup_distributed()
    device = torch.device('cuda', local)
    try:
        if world != 4:
            raise ValueError('Base probe requires four ranks')
        b = train.load_model_with_lora(Path(job['model_dir']), device=device, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_vision_scope='off',
            strict_vision_scope=True, gradient_checkpointing=args.gradient_checkpointing)
        raw = b.model
        b.model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[local], find_unused_parameters=False)
        optimizer = torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],
            lr=args.learning_rate, betas=(.9, .95), weight_decay=args.weight_decay)
        work = items(job); results = []
        torch.cuda.reset_peak_memory_stats(device)
        for key in admitted['probe_ids']:
            kind, item = work[key]; packed, shape = pack(b, kind, item, args)
            if shape != admitted['shapes'][key]:
                raise ValueError('Base input shape changed after CPU admission')
            if kind == 'train':
                b.model.train()
                for _ in range(2):
                    loss, _ = baseline.single_loss(b, [packed], [item.spec])
                    if not torch.isfinite(loss): raise ValueError('Nonfinite base probe loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all()
                           for p in b.model.parameters() if p.requires_grad):
                        raise ValueError('Missing/nonfinite base probe gradient')
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                del loss
            elif kind == 'loss':
                b.model.eval()
                with torch.no_grad(): baseline.single_loss(b, [packed], [item.spec])
            else:
                raw.eval()
                view = SimpleNamespace(model=raw, processor=b.processor, tokenizer=b.tokenizer, device=device)
                if kind == 'generation':
                    base.legacy_single_generate(view, [packed], max_new_tokens=args.generation_eval_max_new_tokens, phase3=True)
                else:
                    evaluate._generate(view, train._load_images(item.row.history_rgb_paths), spec=item.spec,
                        audit=False, history_rgb_mode='4rgb', max_new_tokens=TEST_MAX_NEW_TOKENS,
                        prompt_variant=args.prompt_variant)
            results.append(dict(key=key, kind=kind))
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device); peak = torch.cuda.max_memory_reserved(device)
        result = dict(rank=rank, free_bytes=free, total_bytes=total, peak_reserved_bytes=peak,
                      passed=free >= 8*1024**3 and peak <= .85*total, results=results)
        ranks = [None]*world; torch.distributed.all_gather_object(ranks, result)
        passed = all(r['passed'] for r in ranks)
        if rank == 0:
            trial.write(Path(job['output'])/'base_probe.json', dict(status='passed' if passed else 'blocked',
                ranks=ranks, model_saved=False, admission_sha256=trial.file_sha(Path(job['output'])/'base_input_admission.json')))
        if not passed: raise ValueError('Base GPU probe has insufficient headroom')
    finally:
        train.cleanup_distributed()


def summarize(job):
    if trial.reference().get('reference_kind') != 'new_common_dataset_v1' and not job.get('coverage_contract'):
        return base.summarize(job)
    import json
    from collections import Counter
    from qwen3vl_local.sft_new_loop_phase3.prompts import (
        make_prompt_spec, parse_action_output, choice_target_action, prompt_spec_to_json,
    )
    root = Path(job['output'])/'train'
    path = root/('generation_val_cases.jsonl' if job['stage'] == 'quick' else 'final_generation_val_cases.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    step = job['profile']['steps_per_epoch'] if job['stage'] == 'quick' else job['profile']['optimizer_updates']
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
    report = dict(step=step, optimizer_updates=step, planned_optimizer_updates=job['profile']['optimizer_updates'],
        correct=correct, total=320, exact=correct/320, format_valid=valid/320,
        actions=actions, event_counts=dict(events), reference_sha256=job['reference_sha256'],
        historical_scores_comparable=trial.reference().get('historical_scores_comparable',True),
        old_models_evaluated=False, test_used=False,
        conclusion='Fixed-member validation; sampling/budget changes must be accounted for in comparisons')
    trial.write(Path(job['output'])/'validation_summary.json',report)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('operation', choices=('prepare', 'probe', 'summarize', 'test'))
    p.add_argument('--job', type=Path, required=True)
    a = p.parse_args(); job = trial.read(a.job); base.validate_job(job)
    check_model_snapshot(Path(job['model_dir']), job['model_snapshot'])
    if a.operation == 'prepare': prepare(job)
    elif a.operation == 'probe': probe(job)
    elif a.operation == 'summarize': summarize(job)
    else:
        trial.verify_training(job)
        from qwen3vl_local.sft_new_loop_phase3 import eval as evaluate
        base.install_reference(trial.reference(), split='test')
        sys.argv = [sys.argv[0], *test_arguments(job)]
        evaluate.evaluate(evaluate.parse_args())


if __name__ == '__main__':
    main()
