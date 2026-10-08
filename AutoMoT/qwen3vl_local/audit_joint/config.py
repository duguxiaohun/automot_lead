"""Construct an explicit local audit request from native files, not guessed results."""
import json
from pathlib import Path

SPLIT_ROLES = dict(train='train_pool', val='dev_val', test='final_test')


def make_config(args):
    root = Path(args.project_root).resolve()
    automot = root / 'AutoMoT'
    code = automot / 'qwen3vl_local'
    p3, p4 = code / 'sft_new_loop_phase3', code / 'sft_new_loop_phase4'
    sources, artifacts, diagnostics = [], [], {}
    common_root = str(Path(args.data_root).absolute())
    roots = {
        'phase3': str(Path(getattr(args, 'phase3_data_root', None) or common_root).absolute()),
        'phase4': common_root,
        'action': str(Path(getattr(args, 'action_data_root', None) or common_root).absolute()),
    }

    def read_object(path, object_required=True):
        path = Path(path)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text())
            if object_required and not isinstance(value, dict):
                raise ValueError('expected a JSON object')
            return value
        except (OSError, ValueError) as error:
            diagnostics[str(path.absolute())] = f'{type(error).__name__}: {error}'
            return {}

    def mapping(value, key, path):
        item = value.get(key) or {}
        if not isinstance(item, dict):
            diagnostics[str(Path(path).absolute())] = f'{key} must be an object'
            return {}
        return item

    def source(ident, phase, path, scope, **kwargs):
        kwargs.setdefault('data_root', roots[phase])
        sources.append(dict(id=ident, phase=phase, path=str(Path(path).absolute()), scope=scope, **kwargs))

    def artifact(ident, path, **kwargs):
        if Path(path).suffix == '.json':
            read_object(path)
            kwargs.setdefault('check', 'json_object')
        artifacts.append(dict(id=ident, path=str(Path(path).absolute()), **kwargs))

    # Development exposure is cumulative; a historical file is not a fresh test.
    for phase, directory, pattern, key in [
            ('phase3', p3, 'development_route_groups*.json', 'groups'),
            ('phase4', p4, '*.json', 'train_only_groups')]:
        for path in sorted(directory.glob(pattern)):
            value = read_object(path, object_required=False)
            if isinstance(value, dict) and isinstance(value.get(key), list) and value[key]:
                source(f'{phase}_exposure_{path.stem}', phase, path, 'registered development exposure; cumulative',
                       records_key=key, role='development_exposure')
    for name, key, role in [
            ('producer_manual_check_plan_v22_20261008.json', 'reservations', 'independent_review'),
            ('producer_manual_check_plan_v22_20261008.json', 'excluded_reservations', 'quarantine')]:
        source(f'phase4_{key}', 'phase4', p4 / name, 'frozen producer review reservation, not labels',
               records_key=key, role=role)
    source('phase4_formal_holdout', 'phase4', p4 / 'formal_holdout_plan_20260930.json',
           'native annotation reservations, not new manual judgments', records_key='reservations',
           split_roles=SPLIT_ROLES)

    # The caller supplies actual full pools. Absence stays explicit; default paths
    # are documented production locations, never replaced by sampled audit cases.
    index_name = ('sft_new_loop_phase3_data_v23' if args.phase3_prompt_variant == 'baseline'
                  else 'sft_new_loop_phase3_data_rgb_stage_20261006_isolated')
    p3_index = Path(args.phase3_index or automot / 'checkpoints' / index_name / 'frame_index.jsonl')
    source('phase3_full_index', 'phase3', p3_index, 'entire admitted training/validation/test index',
           split_roles=SPLIT_ROLES, full_pool=True, required_artifacts=['phase3_data_manifest'])
    artifact('phase3_data_manifest', p3_index.parent / 'manifest.json')
    pair = {}
    for mode, supplied in [('2', args.phase4_data2), ('4', args.phase4_data4)]:
        directory = Path(supplied) if supplied else automot / f'checkpoints/phase4_v42_full/data{mode}'
        manifest = directory / 'manifest.json'
        pair[mode] = str(directory.absolute())
        recorded = read_object(manifest)
        if manifest.is_file() and recorded.get('rgb_mode') != int(mode):
            diagnostics.setdefault(str(manifest.absolute()), f'expected RGB{mode} manifest, got {recorded.get("rgb_mode")}')
        artifact(f'phase4_rgb{mode}_manifest', manifest, check='phase4_manifest', expected_rgb_mode=int(mode))
        for split, role in SPLIT_ROLES.items():
            # No candidate_pool/production coverage => development subset, not full production.
            full = mapping(recorded, 'production_coverage', manifest).get('complete_causal_production') is True
            source(f'phase4_rgb{mode}_{split}', 'phase4', directory / f'{split}.jsonl',
                   'native dataset; full production only when manifest declares complete causal coverage',
                   role=role, expected_split=split, full_pool=full,
                   expected_sha256=mapping(recorded, 'files', manifest).get(f'{split}.jsonl'),
                   required_artifacts=[f'phase4_rgb{mode}_manifest'],
                   data_root=str(Path(args.data_root).absolute()), verify_images=args.verify_images)
    action = Path(args.action_data or automot / 'checkpoints/action_prior_data')
    artifact('action_data_manifest', action / 'manifest.json')
    for split, role in SPLIT_ROLES.items():
        source(f'action_raw_{split}', 'action', action / f'{split}.jsonl',
               'raw Action pool; runtime development isolation/support reassignment must also be audited',
               role='raw_inventory', expected_split=split, full_pool=False)
    effective_path = Path(args.action_effective_index or action / 'effective_pool_audit.jsonl')
    effective_manifest = effective_path.with_name('effective_pool_manifest.json')
    effective = read_object(effective_manifest)
    if effective.get('data_root') is not None and not isinstance(effective['data_root'], str):
        diagnostics[str(effective_manifest.absolute())] = 'data_root must be a string'
        effective = {}
    if effective.get('data_root') and not getattr(args, 'action_data_root', None):
        roots['action'] = str(Path(effective['data_root']).absolute())
        for item in sources:
            if item['phase'] == 'action':
                item['data_root'] = roots['action']
    artifact('action_effective_manifest', effective_manifest, check='action_effective_manifest',
             index_path=str(effective_path.absolute()), expected_data_root=roots['action'])
    source('action_effective_index', 'action', effective_path,
           'complete native runtime pool only after manifest/dependency verification',
           split_roles=SPLIT_ROLES, full_pool=effective.get('schema') == 'joint_action_effective_pool_v2',
           required_artifacts=['action_effective_manifest'], expected_sha256=effective.get('index_sha256'))
    if args.candidate_val:
        source('phase4_new_val_candidates', 'phase4', args.candidate_val,
               'candidates only; no reservation performed', role='candidate_dev_val',
               data_root=str(Path(getattr(args, 'candidate_data_root', None) or common_root).absolute()))

    # Historical candidate metadata and error packs remain regression evidence.
    audit_dirs = sorted((automot / 'checkpoints').glob('sft_new_loop_phase3_20261008_*_audit_bundle/audit_bundle'))
    selected_index_hashes = set()
    for directory in audit_dirs:
        ident = directory.parent.name
        metadata = directory / 'adapter_metadata/sft_new_loop_phase3_adapter_config.json'
        saved = read_object(metadata)
        selected = bool(saved) and saved.get('prompt_variant', 'baseline') == args.phase3_prompt_variant
        if selected:
            if isinstance(saved.get('training_index_sha256'), str):
                selected_index_hashes.add(saved['training_index_sha256'])
            else:
                diagnostics[str(metadata.absolute())] = 'missing training_index_sha256'
        artifact(ident + '_metadata', metadata, **({'check': 'phase3_adapter_metadata'} if selected else {}))
        artifact(ident + '_run', directory / 'adapter_metadata/train_run_manifest.json')
        for path in sorted((directory / 'lora_production').glob('cases_rank*.jsonl')):
            source(ident + '_' + path.stem, 'phase3', path, 'previously inspected historical evaluation; no fresh independent score',
                   role='historical_regression')
    # Require actual adapter weights separately from the small copied metadata.
    if args.phase3_adapter:
        adapter = Path(args.phase3_adapter)
        selected_metadata = adapter / 'sft_new_loop_phase3_adapter_config.json'
        saved = read_object(selected_metadata)
        if isinstance(saved.get('training_index_sha256'), str):
            selected_index_hashes = {saved['training_index_sha256']}
        elif selected_metadata.is_file() and str(selected_metadata.absolute()) not in diagnostics:
            diagnostics[str(selected_metadata.absolute())] = 'missing training_index_sha256'
        artifact('phase3_selected_adapter_metadata', adapter / 'sft_new_loop_phase3_adapter_config.json', check='phase3_adapter_metadata')
        artifact('phase3_selected_adapter_weights', adapter / 'adapter_model.safetensors')
        for name in ('adapter_config.json', 'qwen35_backend.json', 'qwen35_base_assets.json'):
            artifact('phase3_selected_' + name, adapter / name)
    else:
        artifact('phase3_selected_adapter_weights', automot / 'checkpoints/PHASE3_SELECTED_ADAPTER_NOT_PROVIDED/adapter_model.safetensors')
    artifact('phase4_v42_verification', p4 / 'teacher_v42_verification_20261008.json')
    if len(selected_index_hashes) > 1:
        diagnostics[str(p3_index.absolute())] = 'historical candidate indices disagree; select one adapter explicitly'
        selected_index_hashes.clear()
    if selected_index_hashes:
        next(s for s in sources if s['id'] == 'phase3_full_index')['expected_sha256'] = selected_index_hashes.pop()
    if Path(pair['2']).resolve() == Path(pair['4']).resolve():
        diagnostics[str((Path(pair['2']) / 'manifest.json').absolute())] = 'RGB2/RGB4 refer to the same resolved directory'
        diagnostics[str((Path(pair['4']) / 'manifest.json').absolute())] = 'RGB2/RGB4 refer to the same resolved directory'
    # Invalid JSON is evidence, not a prepare-time crash. Preserve its path and
    # let capture hash the bytes and record an invalid artifact.
    for item in artifacts:
        if item['path'] in diagnostics:
            item['prepare_error'] = diagnostics[item['path']]
    known = {item['path'] for item in artifacts}
    for number, (path, error) in enumerate(sorted(diagnostics.items())):
        if path not in known:
            artifact(f'prepare_invalid_{number}', path, prepare_error=error)
    for item in sources:
        if item['path'] in diagnostics:
            item['validation_error'] = diagnostics[item['path']]
    return dict(schema='joint_audit_request_v1', project_root=str(root),
                phase3_prompt_variant=args.phase3_prompt_variant,
                source_roots=[str(code), str(automot / 'lead_video_tools'), str(automot / 'keyframe_filter')],
                model_dir=str(Path(args.model_dir or automot / 'checkpoints/Qwen3.5-4B').absolute()),
                artifacts=artifacts, split_sources=sources, phase4_pair=pair,
                prepare_diagnostics=[dict(path=p, error=e) for p, e in sorted(diagnostics.items())],
                note='All operations local. Missing full pools/models remain blockers; no data or labels rewritten.')
