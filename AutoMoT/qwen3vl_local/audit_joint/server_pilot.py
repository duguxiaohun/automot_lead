"""Bounded v52 data pilot; never compile/publish a partial replay or start training."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import zipfile

from .io import file_sha

LIMIT = 30_000_000


def select(routes, limit=30, max_frames=400):
    """Deterministic round robin over split then scenario; no label-based selection."""
    buckets = defaultdict(lambda: defaultdict(deque))
    for row in sorted(routes, key=lambda r: (r['scenario'], r['route_id'])):
        if 0 < len(row['rgb_frames']) <= max_frames:
            buckets[row['split']][row['scenario']].append(row)
    selected = []
    scenarios = {split: deque(sorted(groups)) for split, groups in buckets.items()}
    while len(selected) < limit:
        progress = False
        for split in sorted(buckets):
            if not scenarios[split] or len(selected) >= limit:
                continue
            name = scenarios[split].popleft()
            selected.append(buckets[split][name].popleft())
            if buckets[split][name]:
                scenarios[split].append(name)
            progress = True
        if not progress:
            break
    return selected


def run(data_root, output):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay
    from qwen3vl_local.sft_new_loop_phase4.identity import write_json
    output = Path(output)
    # Frozen candidates are retained for later sequential batches in production/.
    pool = candidate_pool.scan(data_root, output/'candidates.json')
    rows = select(pool['routes'])
    if not rows:
        raise ValueError('no eligible routes <=400 frames; do not silently run the full pool')
    keys = [dict(scenario=r['scenario'], route_id=r['route_id']) for r in rows]
    write_json(output/'pilot_routes.json', keys)
    write_json(output/'pilot_selection.json', dict(policy='split_scenario_round_robin_v1',
        candidates_sha256=file_sha(output/'candidates.json'), routes=len(rows),
        frames=sum(len(r['rgb_frames']) for r in rows), max_routes=30, max_route_frames=400,
        split_routes=dict(Counter(r['split'] for r in rows)),
        scope='bounded pilot, excludes longer routes; not full-pool coverage or capacity certification'))
    teacher_replay.produce(pool, data_root, output/'production', workers=1,
        compression='gzip', route_keys={(r['scenario'], r['route_id']) for r in rows})


def package(output, members, *, uncompressed_limit=None):
    """Only caller-built, bounded diagnostics; never walk checkpoint directories."""
    if uncompressed_limit is None:uncompressed_limit=LIMIT
    blobs = {name: json.dumps(value, ensure_ascii=False, indent=2).encode()+b'\n'
             for name, value in members.items()}
    if sum(map(len, blobs.values())) > uncompressed_limit:
        raise ValueError(f'diagnostics exceed uncompressed limit {uncompressed_limit}; no truncated success package')
    import hashlib
    receipt = dict(files={name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()})
    path = Path(output)/'pilot_report.zip'
    with path.open('xb') as stream, zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in blobs.items():
            z.writestr(name, data)
    if path.stat().st_size > LIMIT:
        raise ValueError('package exceeds 30MB')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} == receipt['files']
    receipt.update(zip_sha256=file_sha(path), bytes=path.stat().st_size, status='verified')
    (Path(output)/'pilot_package.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def report(output, exit_code):
    from qwen3vl_local.sft_new_loop_phase4 import candidate_pool, teacher_replay
    from qwen3vl_local.sft_new_loop_phase4.production_storage import open_route
    output = Path(output); members = {}; errors = []
    for name in ('source_check.json', 'pilot_selection.json', 'pilot_routes.json', 'pilot_runtime.json'):
        if (output/name).is_file():
            members[name] = json.loads((output/name).read_text())
    summary = dict(command_exit_code=exit_code, training_ready=False,
        sufficient_for_full_build=None, total_required_bytes=None,
        scope='pilot observations only; raw rule counts, not admitted training/holdout counts')
    try:
        pool = candidate_pool.validate(json.loads((output/'candidates.json').read_text()))
        summary.update(candidate_summary=pool['summary'], candidates_sha256=file_sha(output/'candidates.json'))
        index_path = output/'production/index.json'
        accounting = teacher_replay.accounting(pool, index_path)
        summary['accounting'] = {k:v for k,v in accounting.items() if k != 'history_ambiguous_question_ids'}
        summary['history_ambiguous_questions'] = len(accounting['history_ambiguous_question_ids'])
        index = json.loads(index_path.read_text()); cells = Counter(); sizes = []
        for r in index['routes']:
            path = index_path.parent/r['artifact']; frames = 0
            with open_route(path) as stream:
                next(stream)
                for line in stream:
                    row = json.loads(line); frames += 1
                    for q in row['questions']:
                        cells['/'.join(map(str, (q['rgb_mode'],q['episode']['event'],q['edge'],q['slice'],q['rule_target'])))] += 1
            sizes.append(dict(scenario=r['scenario'], route_id=r['route_id'], frames=frames,
                              bytes=path.stat().st_size, sha256=file_sha(path)))
        summary.update(raw_condition_cells=dict(sorted(cells.items())), route_sizes=sizes,
                       production_index_sha256=file_sha(index_path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    log = output/'pilot.log'
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-100_000))
            members['log_tail.json'] = dict(log_sha256=file_sha(log), tail=stream.read().decode(errors='replace'))
    summary.update(errors=errors, status='pilot_completed' if not errors and exit_code == 0 else 'incomplete')
    members['pilot_summary.json'] = summary
    (output/'pilot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return package(output, members)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run','report'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('lead_data'))
    p.add_argument('--exit-code', type=int, default=0)
    a = p.parse_args()
    if a.command == 'run':
        import resource
        import time
        start = time.monotonic()
        try:
            run(a.data_root, a.output)
        finally:
            if a.output.is_dir():
                (a.output/'pilot_runtime.json').write_text(json.dumps(dict(
                    elapsed_seconds=time.monotonic()-start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
    else:
        print(json.dumps(report(a.output, a.exit_code)))


if __name__ == '__main__':
    main()
