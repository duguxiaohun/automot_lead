"""Export Action's native runtime pool before epoch sampling, on the data host."""
import json
from pathlib import Path

from .io import file_sha, write_json, source_files
from .splits import group_of


def export_contract():
    code = Path(__file__).resolve().parents[1]
    from qwen3vl_local.action_prior.phase3_release import active_release
    _, release, _ = active_release()
    external = [code.parent / name for name in release.get('external_dependencies', {})]
    external.append(code.parent / 'lead_video_tools/abnormal_duration_filter.py')
    paths = source_files([code / 'action_prior', Path(__file__), Path(__file__).with_name('splits.py'),
                         Path(__file__).with_name('io.py'), *external])
    return {str(p.relative_to(code.parent)): file_sha(p) for p in paths if p.suffix in {'.py', '.json', '.jsonl'}}


PATH_ARGS = ('data_root', 'data_dir', 'event_balance_index')


def bind_paths(args, base):
    paths = {name: str((base / getattr(args, name)).absolute()) for name in PATH_ARGS}
    for name, value in paths.items():
        setattr(args, name, value)
    return paths


def dependencies(args, split_paths=None, *, path_base=None):
    """Record all raw routes, including ones removed by the native reader."""
    from lead_video_tools.abnormal_duration_filter import is_abnormal_lead_route
    root = Path(args.data_root).resolve()
    paths = [Path(split_paths[split]) if split_paths else Path(args.data_dir) / f'{split}.jsonl'
             for split in ('train', 'val', 'test')]
    mapping = Path(args.event_balance_index)
    paths += [mapping, mapping.with_name('manifest.json')]
    if getattr(args, 'high_level_action_token', False):
        value = json.loads(mapping.with_name('manifest.json').read_text())
        candidate = Path(path_base or Path.cwd()) / value['candidate_index']
        if not candidate.is_file():
            candidate = mapping.parent / candidate.name
        paths.append(candidate)
    dependency_paths = dict(raw_splits={split: str(paths[i].absolute())
                                       for i, split in enumerate(('train', 'val', 'test'))},
                            other_inputs=[str(p.absolute()) for p in paths[3:]])
    hashes = {str(p.absolute()): file_sha(p) for p in paths}
    routes = {}
    for path in paths[:3]:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                group = group_of(row)
                key = row['scenario'], row['run_id']
                if key not in routes:
                    route = root / key[0] / key[1]
                    if not route.is_dir():
                        raise FileNotFoundError(route)
                    blocked, info = is_abnormal_lead_route(route, key[0])
                    routes[key] = dict(scenario=key[0], run_id=key[1], physical_group=group,
                                       resolved_route=str(route.resolve()), excluded=blocked, duration=info)
    if any(file_sha(p) != sha for p, sha in hashes.items()):
        raise RuntimeError('Action inputs changed during dependency scan')
    return dict(data_root=str(root), dependency_paths=dependency_paths, input_sha256=hashes,
                route_filter_inventory=[routes[key] for key in sorted(routes)])


def validate_export(value, index, expected_split_paths=None):
    from qwen3vl_local.action_prior.config import parser
    if value.get('schema') != 'joint_action_effective_pool_v3' or not value.get('export_cwd') or not value.get('resolved_args'):
        raise ValueError('Action export path base is not recorded; re-export to a new directory, do not guess cwd')
    if value.get('source_contract') != export_contract():
        raise ValueError('Action effective-pool source contract/schema mismatch; re-export with native reader')
    if file_sha(index) != value['index_sha256']:
        raise ValueError('Action effective-pool index SHA mismatch')
    argv = value['argv']
    if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
        raise ValueError('invalid recorded Action argv')
    try:
        args = parser().parse_args(argv)
    except SystemExit as error:
        raise ValueError('invalid recorded Action runtime arguments') from error
    base = Path(value['export_cwd'])
    if not base.is_absolute():
        raise ValueError('Action export path base must be absolute')
    if bind_paths(args, base) != value['resolved_args']:
        raise ValueError('Action recorded path bindings disagree with argv/export cwd')
    if expected_split_paths is not None and (set(expected_split_paths) != {'train', 'val', 'test'}
                                            or not all(isinstance(p, str) and p for p in expected_split_paths.values())):
        raise ValueError('Action audit requires the requested raw train/val/test paths')
    current = dependencies(args, expected_split_paths, path_base=base)
    expected = dict(value)
    if expected_split_paths is not None:
        # Bind by split content, allowing relocation without requiring old files.
        expected['dependency_paths'] = dict(value['dependency_paths'],
                                           raw_splits=current['dependency_paths']['raw_splits'])
        expected_hashes = dict(value['input_sha256'])
        raw_hashes = {split: expected_hashes.pop(str((Path(args.data_dir) / f'{split}.jsonl').absolute()))
                      for split in ('train', 'val', 'test')}
        for split, sha in raw_hashes.items():
            new = str(Path(expected_split_paths[split]).absolute())
            if new in expected_hashes and expected_hashes[new] != sha:
                raise ValueError('Action input path collision')
            expected_hashes[new] = sha
        expected['input_sha256'] = expected_hashes
    if any(expected.get(key) != result for key, result in current.items()):
        raise ValueError('Action runtime inputs/route frame counts/filter outcomes changed; re-export')
    return current


def export_pool(argv, output):
    from qwen3vl_local.action_prior.config import parser, read_rows
    if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
        raise ValueError('Action argv must be an explicit JSON array of CLI arguments')
    args = parser().parse_args(argv)
    if not args.event_balance_index:
        raise ValueError('current Action audit requires the native full-frame event-balance index')
    base = Path.cwd()
    resolved_args = bind_paths(args, base)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Native readers apply development isolation, support reassignment, history,
    # duration and input contracts. No hand-written second assignment policy.
    inputs = dependencies(args, path_base=base)
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
    if dependencies(args, path_base=base) != inputs:
        raise RuntimeError('Action inputs/route filter inventory changed during export')
    if export_contract() != contract:
        raise RuntimeError('Action implementation changed during export')
    write_json(output / 'effective_pool_manifest.json', dict(
        schema='joint_action_effective_pool_v3', argv=argv, export_cwd=str(base), resolved_args=resolved_args, **inputs, source_contract=contract,
        index_sha256=file_sha(target), counts=counts,
        physical_groups=len(groups), split_support=getattr(args, 'event_balance_split_support', None),
        scope='All native read_rows before epoch sampling, including sampler-ineligible rows; conservative isolation superset.',
        model_loaded=False, original_splits_modified=False))
    return dict(index=str(target), counts=counts, physical_groups=len(groups))
