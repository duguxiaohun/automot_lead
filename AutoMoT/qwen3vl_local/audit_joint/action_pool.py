"""Export Action's native runtime pool before epoch sampling, on the data host."""
import json
from pathlib import Path

from .io import file_sha, write_json, source_files
from .splits import group_of


def export_contract():
    code = Path(__file__).resolve().parents[1]
    paths = source_files([code / 'action_prior', Path(__file__)])
    return {str(p.relative_to(code)): file_sha(p) for p in paths if p.suffix in {'.py', '.json', '.jsonl'}}


def export_pool(argv, output):
    from qwen3vl_local.action_prior.config import parser, read_rows
    if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
        raise ValueError('Action argv must be an explicit JSON array of CLI arguments')
    args = parser().parse_args(argv)
    if not args.event_balance_index:
        raise ValueError('current Action audit requires the native full-frame event-balance index')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Native readers apply development isolation, support reassignment, history,
    # duration and input contracts. No hand-written second assignment policy.
    paths = [Path(args.data_dir) / f'{split}.jsonl' for split in ('train', 'val', 'test')]
    paths += [Path(args.event_balance_index), Path(args.event_balance_index).with_name('manifest.json')]
    hashes = {str(p.absolute()): file_sha(p) for p in paths}
    contract = export_contract()
    counts, groups = {}, {}
    target = output / 'effective_pool_audit.jsonl'
    with target.open('x') as stream:
        for split in ('train', 'val', 'test'):
            rows = read_rows(args, split)
            counts[split] = len(rows)
            for row in rows:
                group = group_of(row)
                if row['split'] != split or groups.setdefault(group, split) != split:
                    raise ValueError('native effective pool leaks a physical group across splits')
                keep = {k: row[k] for k in ('scenario', 'run_id', 'route_group', 'split', 'anchor',
                        'event_balance_original_split', 'event_balance_status') if k in row}
                stream.write(json.dumps(keep, sort_keys=True) + '\n')
    if any(file_sha(p) != sha for p, sha in hashes.items()):
        raise RuntimeError('Action source changed during export')
    if export_contract() != contract:
        raise RuntimeError('Action implementation changed during export')
    write_json(output / 'effective_pool_manifest.json', dict(
        schema='joint_action_effective_pool_v1', argv=argv, input_sha256=hashes, source_contract=contract,
        index_sha256=file_sha(target), counts=counts,
        physical_groups=len(groups), split_support=getattr(args, 'event_balance_split_support', None),
        scope='All native read_rows before epoch sampling, including sampler-ineligible rows; conservative isolation superset.',
        model_loaded=False, original_splits_modified=False))
    return dict(index=str(target), counts=counts, physical_groups=len(groups))
