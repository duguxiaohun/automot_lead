"""Full-split development reference agreement, without inheriting RGB4 review approval."""
import argparse
from pathlib import Path

from ..dataset import dump_rows
from ..evaluate import _reference_metrics
from ..identity import digest, write_json
from ..paired_eval import case_identity
from .diagnostics import diagnostic_report
from .data import load_dataset
from .model import generate, load_for_inference


def evaluate_rows(bundle, rows, data_root, max_length=8192):
    cases = []
    for index, row in enumerate(rows):
        answer, raw = generate(bundle, row, data_root, max_length)
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
    return dict(count=len(cases), transferred_reviewed_reference=reviewed, teacher_consistency=teacher,
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
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    import json
    data, manifest = load_dataset(a.dataset, rgb_mode=a.rgb_mode)
    record = json.loads((a.adapter / 'phase4_contract.json').read_text())
    if record['dataset'] != digest(manifest) or record['observation_contract'] != manifest['observation_contract']:
        raise ValueError('adapter dataset/input mismatch')
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    device = a.device
    if device == 'cuda':
        import os
        from ..devices import select_gpus
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = select_gpus(1)
        device = 'cuda:0'
    bundle = load_for_inference(a.model_dir, a.adapter, device)
    metrics, cases = evaluate_rows(bundle, data[a.split], a.data_root, a.max_length)
    metrics.update(split=a.split, dataset=digest(manifest), expected_count=len(data[a.split]))
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir / 'metrics.json', metrics)
    dump_rows(a.output_dir / 'cases.jsonl', cases)


if __name__ == '__main__':
    main()
