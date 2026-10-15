"""Read-only first-epoch evidence collection; never train, reload or rebuild data."""
import argparse
import json
import math
from pathlib import Path

from .io import digest, file_sha
from .server_pilot import LIMIT, package


NAMES = (
    'epoch_000_samples.jsonl', 'epoch_000_sampling.json', 'epoch_000_runtime.json',
    'epoch_000_validation.json', 'epoch_000_cases.jsonl', 'epoch_000_checkpoint.json',
    'final_generation.json', 'best.json', 'latest.json',
    'reload_check/checkpoint_reload.json', 'reload_check/metrics.json',
    'reload_check/cases.jsonl', 'test_fixed/metrics.json', 'test_fixed/cases.jsonl',
    *(f'epoch_000_rank{rank}_steps.jsonl' for rank in range(4)),
)
METADATA = ('run.json', 'epoch_000/phase4_contract.json',
            'epoch_000/adapter_config.json', 'epoch_000/selection.json')


def compact(value):
    """Only metadata may be projected; never apply to samples, steps or cases."""
    if isinstance(value, list):
        if len(value) > 20000:
            return dict(omitted_large_list=True, count=len(value), canonical_sha256=digest(value))
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def check_consumption(samples, sampling, ranks):
    """Check every completed microbatch against the hashed plan, not just totals.

    Event assignments are inherited from the plan: sample records do not carry
    event labels. This is not an independent dataset/event or GPU validation.
    """
    from qwen3vl_local.sft_new_loop_phase4.taxonomy import EVENTS
    from qwen3vl_local.sft_new_loop_phase4.rgb_short.diagnostics import sample_binding

    try:
        if len(samples) != 10240 or sampling['event_counts'] != dict.fromkeys(EVENTS, 1024):
            raise ValueError('expected the fixed ten-event 10240-presentation plan')
        if sampling['ordered_samples'] != sample_binding(samples, 4):
            raise ValueError('ordered sample hashes differ from the plan')
        batch_size=sampling.get('batch_size',1)
        if type(batch_size) is not int or batch_size<1 or len(samples)%(4*batch_size):
            raise ValueError('invalid per-rank batch size')
        for position,sample in enumerate(samples):
            expected_position=dict(position=position,rank=(position//batch_size)%4,
                                   local_index=position//(4*batch_size))
            if any(type(sample[k]) is not int or sample[k]!=v for k,v in expected_position.items()):
                raise ValueError('invalid ordered sample position/rank/local_index')
            if 'batch_slot' in sample and sample['batch_slot']!=position%batch_size:
                raise ValueError('invalid batch slot')
        if len(ranks)!=4:raise ValueError('expected all four rank logs')
        for rank,steps in enumerate(ranks):
            expected=[r for r in samples if r['rank']==rank]
            if len(steps)*batch_size!=len(expected):
                raise ValueError(f'rank {rank}: missing or extra completed microbatches')
            for j,step in enumerate(steps):
                batch=expected[j*batch_size:(j+1)*batch_size]
                if type(step['epoch']) is not int or step['epoch']!=0:
                    raise ValueError('invalid completed epoch')
                if 'ids' in step:
                    if (step.get('ids')!=[r['id'] for r in batch] or
                        step.get('positions')!=[r['position'] for r in batch] or
                        step.get('batch_size')!=batch_size or step.get('rank')!=rank or step.get('local_index')!=j):
                        raise ValueError(f'rank {rank}: batch ID/order differs from sampled plan')
                    accumulation=sampling.get('accumulation')
                    if type(accumulation) is not int or accumulation<1:
                        raise ValueError('missing accumulation contract for batched receipt')
                    update=(j+1)%accumulation==0 or j+1==len(steps)
                    if step.get('optimizer_update')!=update or step.get('optimizer_steps')!=((j+1)//accumulation+(int(j+1==len(steps) and (j+1)%accumulation!=0))):
                        raise ValueError('optimizer update count differs from batch plan')
                elif batch_size!=1 or any(step[k]!=batch[0][k] for k in ('id','position','rank','local_index')):
                    raise ValueError(f'rank {rank}: step ID/order differs from sampled plan')
                if type(step['loss']) not in (int,float) or not math.isfinite(step['loss']):
                    raise ValueError(f'rank {rank}: nonfinite completed loss')
        return dict(status='matched_recorded_plan', presentations=len(samples),
                    event_counts_from_recorded_plan=sampling['event_counts'],
                    scope='All four step logs match the hashed plan. Event labels are inherited '
                          'from that plan; no fresh dataset, optimizer or label-quality certification.')
    except (KeyError, TypeError, ValueError) as exc:
        return dict(status='incomplete_or_inconsistent', error=str(exc))


def collect(root, output=None):
    root = Path(root).absolute()
    output = Path(output) if output is not None else root / 'handoff'
    if not root.is_dir():
        raise ValueError('training output directory missing')
    if output.exists():
        raise FileExistsError('preserve existing handoff; use a new --output directory')
    members, references, missing, errors, modes = {}, {}, [], [], {}
    # Leave space for diagnostics even if an unexpected file exceeds the budget.
    used = 0
    error_count = 0

    def error(record):
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(record)

    def put(name, value):
        nonlocal used
        size = len(json.dumps(value, ensure_ascii=False, indent=2).encode()) + 1
        if used + size > LIMIT - 2_000_000:
            error(dict(member=name, error='package_budget_exceeded',
                       omitted_bytes=size, canonical_sha256=digest(value)))
            return False
        members[name] = value
        used += size
        return True

    def read(path, name, *, metadata=False, tail=False):
        relative = str(path.relative_to(root))
        if not path.is_file():
            missing.append(relative)
            return None
        try:
            before = dict(resolved_path=str(path.resolve()), sha256=file_sha(path),
                          bytes=path.stat().st_size)
            references[relative] = before
            if tail:
                with path.open('rb') as stream:
                    stream.seek(max(0, before['bytes'] - 60000))
                    value = dict(original_sha256=before['sha256'], tail=stream.read(60000).decode(errors='replace'))
            else:
                if before['bytes'] > 128_000_000:
                    raise ValueError('input exceeds 128MB read limit; omitted with SHA, not certified')
                with path.open('rb') as stream:
                    raw = stream.read(128_000_001)
                if len(raw) > 128_000_000:
                    raise ValueError('input grew beyond read limit')
                text = raw.decode('utf-8')
                if path.suffix == '.jsonl':
                    value = []
                    for line_number, line in enumerate(text.splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError('expected an object row')
                            value.append(row)
                        except ValueError as exc:
                            # Preserve readable rows as evidence, never count the partial
                            # source as complete even if its remaining rows match a plan.
                            error(dict(path=relative, line=line_number, error=str(exc)))
                else:
                    value = json.loads(text)
                    if not isinstance(value, dict):
                        raise ValueError('expected a JSON object')
            put(name, compact(value) if metadata else value)
            return value
        except (OSError, ValueError, RuntimeError) as exc:
            error(dict(path=relative, error=f'{type(exc).__name__}: {exc}'))
            return None

    for mode in (2, 4):
        run = root / f'run_first_epoch_rgb{mode}'
        start_errors = error_count
        data = {name: read(run / name, f'rgb{mode}/{name}') for name in NAMES}
        check = check_consumption(data['epoch_000_samples.jsonl'], data['epoch_000_sampling.json'],
                                  [data[f'epoch_000_rank{rank}_steps.jsonl'] for rank in range(4)])
        if error_count > start_errors:
            check = dict(status='incomplete_or_inconsistent', error='unreadable or omitted run evidence',
                         recorded_plan_check=check)
        modes[str(mode)] = check
        for name in METADATA:
            read(run / name, f'rgb{mode}/projected/{name}', metadata=True)
        read(root / f'rgb{mode}.log', f'rgb{mode}/log_tail.json', tail=True)
    admissions = sorted(root.glob('admission_*.json'))
    if not admissions:
        missing.append('admission_*.json')
    for path in admissions:
        # Whitelisted admission metadata retains explicit unsupported capabilities.
        read(path, path.name, metadata=True)
    # Recheck logical paths as well as bytes: retargeted links are input drift.
    for relative, before in references.items():
        path = root / relative
        try:
            if str(path.resolve()) != before['resolved_path'] or file_sha(path) != before['sha256']:
                raise ValueError('file changed during collection; wait for the job to stop')
        except (OSError, ValueError, RuntimeError) as exc:
            error(dict(path=relative, error=str(exc)))
    consumption_failed = any(m['status'] != 'matched_recorded_plan' for m in modes.values())
    receipt = dict(status='incomplete' if missing or errors or consumption_failed else 'collected_not_semantically_verified',
                   root=str(root), original_files=references, missing=missing, errors=errors,
                   error_count=error_count, error_details_limit=100,
                   collector_sha256=file_sha(Path(__file__)),
                   consumption=modes, training_accepted=False,
                   scope='Metadata/step/case evidence only. JSONL is stored as arrays; malformed '
                         'rows are diagnosed, not certified. No weights, RGB or optimizer tensors. '
                         'Package SHA verification is not training or label-quality acceptance.')
    members['collection_receipt.json'] = receipt
    output.mkdir(parents=True, exist_ok=False)
    packed = package(output, members)
    (output / 'pilot_report.zip').rename(output / 'phase4_epoch_report.zip')
    (output / 'pilot_package.json').rename(output / 'phase4_epoch_package.json')
    return dict(output=str(output), bytes=packed['bytes'], status=receipt['status'],
                missing=len(missing), errors=error_count, consumption=modes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = collect(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'collected_not_semantically_verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
