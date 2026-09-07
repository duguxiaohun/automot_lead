"""按scenario在CPU进程中扫描原始RGB/meta，主进程仍执行统一平衡与分割审计。"""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace


def scan_one(payload):
    """单进程只写自己的候选缓存；不复制RGB，不把机器扫描当视觉确认。"""
    from qwen3vl_local.sft_new_loop_phase3.build_dataset import iter_base_frames
    args_dict, scenario = payload
    args = SimpleNamespace(**args_dict)
    args.scenarios = scenario
    args.workers = 0
    root = Path(args.output_dir) / 'raw_scans'
    root.mkdir(parents=True, exist_ok=True)
    target = root / (scenario + '.jsonl')
    temporary = target.with_suffix('.tmp')
    stats = Counter()
    pairs = set()
    count = 0
    with temporary.open('w') as handle:
        for row in iter_base_frames(args, stats, observed_scenario_town_pairs=pairs):
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    temporary.replace(target)
    info = dict(scenario=scenario, candidates=count, stats=dict(stats), pairs=sorted(pairs), path=str(target))
    target.with_suffix('.summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f'[phase3-scan-complete] {scenario} candidates={count}', flush=True)
    return info


def parallel_frames(args, risk_stats, observed_pairs):
    """固定scenario顺序消费结果，保证并行和串行的采样顺序一致。"""
    scenarios = [p.stem.removesuffix('_result') for p in sorted(Path(args.collection_dir).glob('*_result.json'))]
    selected = None if args.scenarios == 'all' else set(args.scenarios.split(','))
    scenarios = [s for s in scenarios if s != 'noScenarios' and (selected is None or s in selected)]
    if args.max_routes or args.max_routes_per_scenario or args.candidate_cache:
        raise ValueError('--workers is for full raw scans; do not combine with route caps or candidate-cache')
    print(f'[phase3-discover] scenarios={len(scenarios)} workers={args.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for info in pool.map(scan_one, [(vars(args), s) for s in scenarios]):
            if risk_stats is not None:
                risk_stats.update(info['stats'])
            if observed_pairs is not None:
                observed_pairs.update(tuple(p) for p in info['pairs'])
            with Path(info['path']).open() as handle:
                for line in handle:
                    yield json.loads(line)
