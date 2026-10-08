"""G0 artifact capture and verification, without model loading or network access."""
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone

from . import SCHEMA
from .io import digest, file_sha, snapshot_file, source_files, write_json, write_rows
from .splits import audit_sources
from .storage import GIB, plan as storage_plan, capture_input, copied_artifact


def environment():
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'flash-linear-attention', 'causal-conv1d', 'flash-attn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not_installed'
    core_pattern_path = Path('/proc/sys/kernel/core_pattern')
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                core_limits=list(resource.getrlimit(resource.RLIMIT_CORE)),
                core_pattern=core_pattern_path.read_text().strip() if core_pattern_path.is_file() else 'unavailable',
                packages=packages, actual_kernel_bindings='not_run', gpu_forward_backward_generate='not_run',
                note='Package presence is not evidence that the model used a fast kernel.')


def git_state(root):
    def run(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=30).strip()
    return dict(head=run('rev-parse', 'HEAD'),
                status=run('status', '--porcelain=v1', '--untracked-files=normal').splitlines(),
                diff_stat=run('diff', 'HEAD', '--stat').splitlines(),
                note='HEAD is context only; actual source identity is the byte snapshot.')


def native_contracts(phase3_variant):
    from qwen3vl_local.sft_new_loop_phase3.prompt_contract import action_prompt_sha256, prompt_name
    from qwen3vl_local.sft_new_loop_phase3.source_mapping import mapping_contract_hash
    from qwen3vl_local.sft_new_loop_phase4.identity import contract
    from qwen3vl_local.qwen35.backend import backend_contract
    from qwen3vl_local.sft_new_loop_phase4.teacher_rules import identity as teacher_contract
    return dict(phase3=dict(prompt_variant=phase3_variant, prompt_name=prompt_name(phase3_variant),
                           mapping_contract_hash=mapping_contract_hash(),
                           prompt_hashes={f'{mode}/{output}': action_prompt_sha256(
                               prompt_variant=phase3_variant, audit=False, history_rgb_mode=mode,
                               action_output_mode=output)
                               for mode in ('2rgb_endpoints', '4rgb') for output in ('binary', 'choice')}),
                phase4=contract(), teacher=teacher_contract(), backend=backend_contract())


def check_native_artifact(spec, contracts):
    """Validate native recorded contracts without substituting current hashes."""
    path = Path(spec['path'])
    if spec.get('prepare_error'):
        raise ValueError('prepare input error: ' + spec['prepare_error'])
    kind = spec.get('check')
    if kind is None:
        return dict(check='hash_only_no_native_compatibility_claim')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('artifact must be a JSON object')
    if kind == 'json_object':
        return dict(check='parsed_json_object_no_native_compatibility_claim')
    if kind == 'phase4_manifest':
        if spec.get('expected_rgb_mode') is not None and value.get('rgb_mode') != spec['expected_rgb_mode']:
            raise ValueError('dataset RGB mode differs from requested slot')
        from qwen3vl_local.sft_new_loop_phase4.identity import contract_differences
        differences = contract_differences(value.get('contract'), contracts['phase4'])
        if differences:
            raise ValueError(' | '.join(differences))
        for name, sha in value['files'].items():
            if Path(name).name != name or file_sha(path.parent / name) != sha:
                raise ValueError(f'dataset SHA mismatch: {name}')
        for name, field in [('review_queue.jsonl', 'review_queue_sha256'),
                            ('teacher_registry.json', 'teacher_registry_sha256')]:
            if value.get(field) and file_sha(path.parent / name) != value[field]:
                raise ValueError(f'dataset SHA mismatch: {name}')
        return dict(rgb_mode=value['rgb_mode'], counts=value['counts'],
                    full_production=bool(value.get('production_coverage')),
                    teacher_selection=value.get('teacher_selection'),
                    observation_contract=value['observation_contract'])
    if kind == 'phase3_adapter_metadata':
        from qwen3vl_local.sft_new_loop_phase3.prompt_contract import adapter_prompt_variant
        variant = adapter_prompt_variant(value)
        expected = contracts['phase3']
        if variant != expected['prompt_variant']:
            raise ValueError('selected Phase3 prompt variant differs from adapter')
        key = f"{value['history_rgb_mode']}/{value['action_output_mode']}"
        if value['production_prompt_sha256'] != expected['prompt_hashes'][key]:
            raise ValueError('Phase3 prompt SHA differs from adapter')
        if value['mapping_contract_hash'] != expected['mapping_contract_hash']:
            raise ValueError('Phase3 mapping contract differs from adapter')
        return dict(prompt_variant=variant, history_rgb_mode=value['history_rgb_mode'],
                    action_output_mode=value['action_output_mode'], metadata_only=True,
                    training_index_sha256=value['training_index_sha256'])
    if kind == 'action_effective_manifest':
        from .action_pool import validate_export
        dependencies = validate_export(value, spec.get('index_path', path.with_name('effective_pool_audit.jsonl')),
                                       spec.get('expected_split_paths'))
        if spec.get('expected_data_root') and Path(spec['expected_data_root']).resolve() != Path(dependencies['data_root']).resolve():
            raise ValueError('Action actual data root differs from audit request')
        return dict(counts=value['counts'], scope=value['scope'])
    raise ValueError(f'unknown native check: {kind}')


def run(config, output):
    output = Path(output).absolute()
    root = Path(config['project_root']).resolve()
    files = source_files(config['source_roots'])
    if any(output == Path(p).absolute() or Path(p).absolute() in output.parents
           for p in config['source_roots']):
        raise ValueError('output must be outside source roots')
    if any(not p.is_relative_to(root) for p in files):
        raise ValueError('source snapshots must use logical paths inside project_root')
    mode = config.get('storage_mode', 'references')
    storage = storage_plan(config, output, mode, config.get('min_free_bytes', 2 * GIB))
    if not storage['sufficient']:
        raise OSError(f"insufficient free disk for G0: {storage}; no capture created")
    config = dict(config, storage_mode=mode, min_free_bytes=storage['min_free_bytes'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', config)
    source_manifest = {}
    for path in files:
        source_manifest[str(path.relative_to(root))] = snapshot_file(path, output)
    write_json(output / 'source_manifest.json', source_manifest)
    contracts = native_contracts(config['phase3_prompt_variant'])
    artifacts = []
    for spec in config['artifacts']:
        spec = dict(spec)
        if spec.get('check') == 'action_effective_manifest':
            # Also bind requests written before expected_split_paths was added.
            raw = {s['id']: s['path'] for s in config['split_sources']}
            spec['expected_split_paths'] = {split: raw.get(f'action_raw_{split}')
                                            for split in ('train', 'val', 'test')}
        item = dict(spec)
        artifacts.append(item)
        path = Path(spec['path'])
        if not path.is_file():
            item.update(status='missing')
            continue
        item.update(capture_input(path, output, copy=copied_artifact(path, mode)))
        if spec.get('expected_sha256') and spec['expected_sha256'] != item['sha256']:
            item.update(status='invalid', error='historical artifact SHA mismatch')
            continue
        try:
            item.update(status='verified', native=check_native_artifact(spec, contracts))
            if file_sha(path) != item['sha256']:
                raise ValueError('artifact changed during native verification')
        except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
            item.update(status='invalid', error=str(error))
    model = dict(path=config['model_dir'], status='missing')
    if Path(config['model_dir']).is_dir():
        try:
            from qwen3vl_local.qwen35.identity import base_asset_hashes
            from qwen3vl_local.qwen35.backend import local_model_dir
            local_model_dir(config['model_dir'])
            model.update(status='hashed', assets=base_asset_hashes(config['model_dir']),
                         weights_copied=False, execution='not_run')
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            model.update(status='invalid', error=str(error))
    checked = {item['id']: item for item in artifacts}
    pairing = dict(status='not_requested')
    if config.get('phase4_pair'):
        try:
            if any(checked.get(f'phase4_rgb{mode}_manifest', {}).get('status') != 'verified' for mode in (2, 4)):
                raise ValueError('both verified dataset manifests are required')
            from .pairing import check_pair
            pairing = check_pair(config['phase4_pair'], {mode: checked[f'phase4_rgb{mode}_manifest']['sha256'] for mode in (2, 4)})
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            pairing = dict(status='invalid', error=str(error))
    validated_sources = []
    for spec in config['split_sources']:
        spec = dict(spec)
        if spec['id'] == 'action_effective_index':
            spec['required_artifacts'] = sorted(set(spec.get('required_artifacts', [])) | {'action_effective_manifest'})
        invalid = [name for name in spec.get('required_artifacts', [])
                   if checked.get(name, {}).get('status') != 'verified']
        if invalid:
            spec['validation_error'] = 'required artifact not verified: ' + ', '.join(invalid)
        if spec['id'].startswith('phase4_rgb') and pairing['status'] == 'invalid':
            spec['full_pool'] = False
        validated_sources.append(spec)
    split_report, ledger = audit_sources(validated_sources)
    # Default: retain identities, not a second full dataset/adapter copy.
    for spec, result in zip(config['split_sources'], split_report['sources'], strict=True):
        if result.get('observation_status') == 'observed' or result['status'] == 'audited':
            frozen = capture_input(spec['path'], output, copy=mode == 'full')
            if frozen['sha256'] != result['sha256']:
                raise RuntimeError('split input changed after audit')
            result['snapshot'] = frozen
    write_json(output / 'split_cross_audit.json', split_report)
    write_rows(output / 'exposure_ledger.jsonl', ledger)
    # Recheck the entire source set after native imports and audits, not just each
    # file during copying. An edit halfway through must not produce a valid receipt.
    if source_files(config['source_roots']) != files or any(
            file_sha(root / name) != record['sha256'] for name, record in source_manifest.items()):
        raise RuntimeError('source tree changed during G0; choose a new output and retry')
    blockers = [dict(kind='asset', id=a['id'], status=a['status']) for a in artifacts if a['status'] != 'verified']
    if pairing['status'] == 'invalid':
        blockers.append(dict(kind='phase4_pairing', **pairing))
    if model['status'] != 'hashed':
        blockers.append(dict(kind='model', status=model['status']))
    blockers += [dict(kind='missing_full_pool', **v) for v in split_report['missing_full_pools']]
    blockers += [dict(kind='split_source', id=v['id'], status=v['status'])
                 for v in split_report['sources'] if v['status'] != 'audited']
    if split_report['conflicts']:
        blockers.append(dict(kind='cross_task_isolation', conflict_pairs=len(split_report['conflicts'])))
    blockers += [dict(kind='execution', status='not_run', item=item) for item in
                 ('same_checkpoint_full_evaluation', 'fixed_update_numeric_performance', 'four_rank_kernel_probe')]
    manifest = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
                    status='in_progress', reproducible_baseline_ready=False,
                    source_snapshot_sha256=digest({k: v['sha256'] for k, v in source_manifest.items()}),
                    source_file_count=len(source_manifest), git=git_state(root), environment=environment(),
                    contracts=contracts, model=model, artifacts=artifacts, blockers=blockers,
                    phase4_pairing=pairing,
                    storage=storage, external_assets_required=True, full_input_snapshot=mode == 'full',
                    stages={k: 'not_run' for k in ('E0', 'E1', 'G1', 'G2')},
                    invariants=dict(network_used=False, old_artifacts_modified=False, new_labels=0,
                                    reservations_written=0, gpu_training_started=False),
                    scope='Local G0 capture and declared-pool audit; no model/evaluation/label certification.')
    write_json(output / 'baseline_manifest.json', manifest)
    # A completed bundle has a receipt covering every output. Failures leave an
    # incomplete directory without a receipt and cannot be mistaken for success.
    receipt = {str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'receipt.json', dict(schema=SCHEMA, files=receipt))
    return manifest, split_report


def verify_bundle(output):
    root = Path(output).resolve()
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt.get('schema') != SCHEMA:
        raise ValueError('unknown receipt schema')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} - {'receipt.json'}
    if actual != set(receipt['files']):
        raise ValueError('bundle inventory changed')
    for name, expected in receipt['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or file_sha(path) != expected:
            raise ValueError(f'bundle hash mismatch: {name}')
    return dict(status='verified', files=len(actual), scope='bundle bytes only, not semantic approval')
