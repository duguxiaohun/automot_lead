"""Read-only checks used by both full pipelines before expensive training."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

from PIL import Image

from .local_assets import inspect_index


def check_images(rows, *, mode, checked):
    indices = (0, 3) if mode == "2rgb_endpoints" else (0, 1, 2, 3)
    for row in rows:
        if len(row.history_rgb_paths) != 4:
            raise ValueError(f"Invalid four-frame history: {row.scenario}/{row.route_id}/{row.frame_id}")
        for i in indices:
            path = row.history_rgb_paths[i]
            if path in checked:
                continue
            try:
                if Path(path).stat().st_size <= 0:
                    raise ValueError("empty image")
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"RGB preflight failed BEFORE training: {path}: {exc}") from exc
            checked.add(path)
            if len(checked) % 10000 == 0:
                print(f"[pipeline-preflight] verified RGB headers={len(checked)}", flush=True)


def check_data(args):
    if args.cases_per_bin < 0 or args.expected_excluded < 0 or args.expected_total < 0:
        raise ValueError("Evaluation counts must be nonnegative before training")
    if not 0 <= args.highway_ue3_fraction <= 1:
        raise ValueError("Evaluation highway fraction must be in [0, 1]")
    receipt = inspect_index(args.index, args.phase)
    module = importlib.import_module(f"qwen3vl_local.sft_new_loop_phase{args.phase}.eval")
    counts, checked, eval_counts = {}, set(), {}
    for split in ("train", "val", "test"):
        rows = module._read_rows(Path(args.index), split=split, data_root=Path(args.data_root))
        counts[split] = len(rows)
        check_images(rows, mode=args.history_rgb_mode, checked=checked)
        if split in {"val", args.eval_split}:
            if split == args.eval_split and getattr(args, 'max_frames', 0) > 0:
                rows = module._read_rows(Path(args.index), split=split, data_root=Path(args.data_root),
                                         max_frames=args.max_frames)
            if args.phase == 2:
                if split == args.eval_split and args.exclude_cases:
                    rows, _ = module._exclude_prior_cases(
                        rows, paths=[Path(x) for x in args.exclude_cases.split()], expected_excluded_cases=args.expected_excluded)
                module._require_highway_ue3_rows(rows, split=split)
                cases = module._balanced_cases(rows, cases_per_bin=args.cases_per_bin,
                    seed=args.seed, highway_ue3_fraction=args.highway_ue3_fraction,
                    route_diverse=args.route_diverse)
            else:
                cases = module._balanced_cases(rows, cases_per_bin=args.cases_per_bin, seed=args.seed)
            if split == args.eval_split and args.expected_total and len(cases) != args.expected_total:
                raise ValueError(f"Final eval count mismatch before training: {len(cases)} != {args.expected_total}")
            eval_counts[split] = dict(Counter(item.balance_key for item in cases))
    return dict(status="data_checked", model_checked=False,
                dataset=receipt, split_rows=counts, image_headers_verified=len(checked),
                history_rgb_mode=args.history_rgb_mode, final_eval_balance=eval_counts,
                limits="Image headers checked, not a guarantee against future file changes, truncated pixel payloads, GPU/OOM or hardware failures.")


def check(args):
    from qwen3vl_local.qwen35.preflight import check as check_model
    model = check_model(args.model_dir)
    if not model['ready']:
        raise ValueError("Local model/runtime preflight: " + "; ".join(model['errors']))
    import peft  # noqa: F401
    from qwen3vl_local.qwen35.backend import LocalProcessor
    LocalProcessor.from_pretrained(args.model_dir)
    report = check_data(args)
    report.update(status="passed_before_training", model_checked=True, model_dir=str(Path(args.model_dir).resolve()))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', type=int, choices=(1,2), required=True)
    p.add_argument('--model-dir', default='checkpoints/Qwen3.5-4B')
    p.add_argument('--data-only', action='store_true', help='CPU data checks only; no model/runtime admission')
    p.add_argument('--device', choices=('cpu',), default='cpu', help='Preflight is CPU-only')
    p.add_argument('--index', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--history-rgb-mode', choices=('4rgb','2rgb_endpoints'), required=True)
    p.add_argument('--eval-split', choices=('val','test'), default='test')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--cases-per-bin', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260810)
    p.add_argument('--highway-ue3-fraction', type=float, default=.125)
    p.add_argument('--route-diverse', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--exclude-cases', default='')
    p.add_argument('--expected-excluded', type=int, default=0)
    p.add_argument('--expected-total', type=int, default=0)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = check_data(args) if args.data_only else check(args)
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print('[pipeline-preflight] '+json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
