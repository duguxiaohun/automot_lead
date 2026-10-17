"""Full-split development reference agreement, without inheriting RGB4 review approval."""
import argparse
from pathlib import Path

from ..dataset import dump_rows
from ..evaluate import _reference_metrics
from ..identity import digest, write_json
from ..paired_eval import case_identity
from .diagnostics import diagnostic_report
from .data import load_dataset
from .model import generate, generate_batch, load_for_inference


def evaluate_rows(bundle, rows, data_root, max_length=8192, *, batch_size=1, reuse_rgb=False):
    if batch_size < 1:
        raise ValueError('evaluation batch_size must be positive')
    from qwen3vl_local.batch_runtime import distributed_batches
    def predict(selected):
        if batch_size==1 and not reuse_rgb:
            return [generate(bundle,row,data_root,max_length) for row in selected]
        return generate_batch(bundle,selected,data_root,max_length,reuse_rgb=reuse_rgb)
    predictions,execution=distributed_batches(rows,batch_size,predict,identities=[r['id'] for r in rows])
    cases = []
    for index, (row, (answer, raw)) in enumerate(zip(rows,predictions)):
        original_kind = row.get('student_reference', {}).get('original_reference_kind', row.get('reference_kind', 'reviewed_rgb'))
        cases.append(dict(id=row['id'], target=row['target'], prediction=answer, raw=raw,
                          event=row['episode']['event'], edge=row['edge'], slice=row['slice'],
                          reference_kind=row.get('reference_kind', 'reviewed_rgb'),
                          original_reference_kind=original_kind, paired_identity=case_identity(row)))
        if index % 20 == 0:
            print(f'[student generate] {index+1}/{len(rows)}', flush=True)
    def agreement(selected):
        metrics = _reference_metrics(selected)
        return dict(count=metrics['count'], agreement=metrics['accuracy'],
                    event_macro_agreement_observed=metrics['event_macro_accuracy_observed'],
                    event_agreement=metrics['event_accuracy'], missing_events=metrics['missing_events'],
                    malformed=metrics['malformed'], confusion=metrics['confusion'])
    reviewed = agreement([row for row in cases if row['original_reference_kind'] != 'rule_teacher'])
    teacher = agreement([row for row in cases if row['original_reference_kind'] == 'rule_teacher'])
    return dict(distributed_execution=execution,generation_batch_size=batch_size, frozen_rgb_reuse_requested=reuse_rgb, count=len(cases), transferred_reviewed_reference=reviewed, teacher_consistency=teacher,
                diagnostics=diagnostic_report(cases),
                selection_agreement=reviewed['event_macro_agreement_observed'],
                selection_metric='transferred_reference_event_macro_agreement_v1',
                independent_accuracy_certified=False,
                scope='development reference agreement; no short-input human visibility or joint independence approval'), cases


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--rgb-mode', type=int, choices=(2, 4), default=2)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--batch-size', type=int, default=None, help='Default: recorded training evaluation batch')
    p.add_argument('--reuse-frozen-rgb', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--evaluation-distribution', choices=('equal_events','natural'), default='equal_events')
    a = p.parse_args()
    from qwen3vl_local.batch_runtime import launch_four_evaluation
    launch=launch_four_evaluation('qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate',a.device)
    if launch is not None:raise SystemExit(launch)
    if a.batch_size is not None and a.batch_size < 1:
        p.error("batch-size must be positive")
    import json
    data, manifest = load_dataset(a.dataset, rgb_mode=a.rgb_mode)
    record = json.loads((a.adapter / 'phase4_contract.json').read_text())
    if record['dataset'] != digest(manifest) or record['observation_contract'] != manifest['observation_contract']:
        raise ValueError('adapter dataset/input mismatch')
    from .evaluation_policy import restore, score_cases
    data = restore(data, record, short=True, natural=a.evaluation_distribution=='natural',data_root=a.data_root)
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    device = a.device
    import os
    from qwen3vl_local.batch_runtime import initialize_evaluation,distributed_identity
    if device == 'cuda' and int(os.environ.get('WORLD_SIZE','1'))==1:
        import os
        from ..devices import select_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = select_gpus(1)
        device = 'cuda:0'
    device=initialize_evaluation(device)
    rank,world=distributed_identity()
    bundle = load_for_inference(a.model_dir, a.adapter, device)
    metrics, cases = evaluate_rows(bundle, data[a.split], a.data_root, a.max_length,
        batch_size=a.batch_size if a.batch_size is not None else record['training'].get('eval_batch_size',1),
        reuse_rgb=a.reuse_frozen_rgb if a.reuse_frozen_rgb is not None else record['training'].get('reuse_frozen_rgb',False))
    selection = record['training']['evaluation']
    ids = selection['plans'][a.split]['selected_ids']
    by_id = {r['id']:r for r in cases}
    metrics.update(score_cases([by_id[i] for i in ids], selection['mode'], ids))
    metrics['evaluation_plan'] = selection['plans'][a.split]
    metrics['evaluation_distribution'] = a.evaluation_distribution
    metrics['evaluation_scope'] = selection['training_admission']['evaluation_scope']
    metrics.update(split=a.split, dataset=digest(manifest), expected_count=len(data[a.split]))
    if rank!=0:
        return
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir / 'metrics.json', metrics)
    dump_rows(a.output_dir / 'cases.jsonl', cases)


if __name__ == '__main__':
    try:main()
    finally:
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
