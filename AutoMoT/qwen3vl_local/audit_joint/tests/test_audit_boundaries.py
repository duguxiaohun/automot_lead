import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qwen3vl_local.audit_joint import guard
from qwen3vl_local.audit_joint.action_pool import dependencies, export_contract, export_pool, validate_export
from qwen3vl_local.audit_joint.baseline import run, verify_bundle
from qwen3vl_local.audit_joint.config import make_config
from qwen3vl_local.audit_joint.handoff import pack, verify_package
from qwen3vl_local.audit_joint.io import file_sha
from qwen3vl_local.audit_joint.pairing import check_pair
from qwen3vl_local.audit_joint.splits import audit_sources
from qwen3vl_local.audit_joint.storage import verify_inputs


def request_args(root):
    automot = root / 'AutoMoT'
    for folder in ('qwen3vl_local/sft_new_loop_phase3', 'qwen3vl_local/sft_new_loop_phase4',
                   'lead_video_tools', 'keyframe_filter', 'lead_data'):
        (automot / folder).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(project_root=root, data_root=automot / 'lead_data',
                           phase3_prompt_variant='baseline', phase3_index=None, phase3_adapter=None,
                           phase4_data2=None, phase4_data4=None, action_data=None,
                           action_effective_index=None, candidate_val=None, model_dir=None, verify_images=False)


def capture(config, output):
    with patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}), \
            patch('qwen3vl_local.audit_joint.baseline.git_state', return_value={}):
        return run(config, output)


def pool(path, phase, role, route, root=None, **kw):
    path.write_text(json.dumps(dict(scenario='Scene', route_id=route, **kw)) + '\n')
    return dict(id=path.stem, path=str(path), phase=phase, role=role, scope='fixture',
                **({'data_root': str(root)} if root else {}))


def test_resolved_route_alias_is_a_cross_task_conflict(tmp_path):
    data = tmp_path / 'data'
    (data / 'Scene').mkdir(parents=True)
    actual = tmp_path / 'physical'
    actual.mkdir()
    for name in ('a', 'b', 'c'):
        (data / 'Scene' / name).symlink_to(actual, target_is_directory=True)
    report, _ = audit_sources([
        pool(tmp_path / 'train.jsonl', 'phase3', 'train_pool', 'a', data),
        pool(tmp_path / 'val.jsonl', 'phase4', 'dev_val', 'b', data),
        pool(tmp_path / 'candidate.jsonl', 'phase4', 'candidate_dev_val', 'c', data),
    ])
    assert report['status'] == 'conflicts_found'
    assert report['conflicts'][0]['logical_groups'] == ['Scene/a', 'Scene/b', 'Scene/c']
    assert report['candidate_filter'][0]['status'] == 'excluded'
    assert report['candidate_filter'][0]['reasons'][0]['match'] == 'same_resolved_route'


def test_image_sha_alone_does_not_merge_route_identity(tmp_path):
    shared = dict(images=['image'], image_sha256=['a' * 64], image_rgb_sha256=['b' * 64])
    report, _ = audit_sources([
        pool(tmp_path / 'train.jsonl', 'phase3', 'train_pool', 'a', **shared),
        pool(tmp_path / 'val.jsonl', 'phase4', 'dev_val', 'b', **shared),
    ])
    assert not report['conflicts']
    assert not report['resolved_group_components']
    assert report['aliases'][0]['conclusion'] == 'potential_duplicate_content'


def test_resolved_alias_union_is_transitive_via_physical_groups(tmp_path):
    for name in ('x', 'y'):
        (tmp_path / name / 'Scene').mkdir(parents=True)
        (tmp_path / name / 'actual').mkdir()
    for root, aliases in [('x', ('a', 'b')), ('y', ('b', 'c'))]:
        for name in aliases:
            (tmp_path / root / 'Scene' / name).symlink_to(tmp_path / root / 'actual')
    specs = [pool(tmp_path / (str(i) + '.jsonl'), phase, role, route, tmp_path / root)
             for i, (root, route, phase, role) in enumerate([
                 ('x', 'a', 'phase3', 'train_pool'), ('x', 'b', 'action', 'raw_inventory'),
                 ('y', 'b', 'action', 'raw_inventory'), ('y', 'c', 'phase4', 'final_test')])]
    report, _ = audit_sources(specs)
    assert report['conflicts'][0]['logical_groups'] == ['Scene/a', 'Scene/b', 'Scene/c']


def test_config_wires_distinct_actual_roots(tmp_path):
    args = request_args(tmp_path)
    args.phase3_data_root = tmp_path / 'p3'
    args.action_data_root = tmp_path / 'action'
    args.candidate_data_root = tmp_path / 'candidate'
    args.candidate_val = tmp_path / 'candidates.jsonl'
    config = make_config(args)
    specs = {s['id']: s for s in config['split_sources']}
    assert specs['phase3_full_index']['data_root'] == str(args.phase3_data_root)
    assert specs['action_effective_index']['data_root'] == str(args.action_data_root)
    assert specs['phase4_new_val_candidates']['data_root'] == str(args.candidate_data_root)


@pytest.mark.parametrize('exit_code', [0, 9])
def test_guard_kills_worker_when_leader_exits(tmp_path, exit_code):
    pidfile = tmp_path / 'worker.pid'
    worker = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
    leader = ('import subprocess,sys; from pathlib import Path; '
              'p=subprocess.Popen([sys.executable,"-c",sys.argv[2]]); '
              'Path(sys.argv[1]).write_text(str(p.pid)); sys.exit(int(sys.argv[3]))')
    pid = None
    try:
        with patch.object(guard, 'core_pattern', return_value='core'):
            result = guard.run([sys.executable, '-c', leader, str(pidfile), worker, str(exit_code)], tmp_path, 0, .01)
        pid = int(pidfile.read_text())
        assert result == exit_code
        # An orphan may remain a zombie until init reaps it, but cannot write.
        for _ in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z'):
                break
            time.sleep(.01)
        else:
            pytest.fail('worker survived the guard')
    finally:
        if pid is None and pidfile.exists():
            pid = int(pidfile.read_text())
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def model_capture(tmp_path):
    args = request_args(tmp_path)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'fixture')
    (model / 'config.json').write_text('{}')
    (model / 'tokenizer.json').write_text('{}')
    args.model_dir = model
    with patch('qwen3vl_local.qwen35.backend.local_model_dir', return_value=model):
        capture(make_config(args), tmp_path / 'capture')
    return model, tmp_path / 'capture'


@pytest.mark.parametrize('mutation', ['weights', 'delete', 'add', 'tokenizer'])
def test_verify_inputs_detects_base_asset_mutation(tmp_path, mutation):
    model, output = model_capture(tmp_path)
    assert verify_inputs(output)['status'] == 'verified'
    if mutation == 'delete':
        (model / 'model.safetensors').unlink()
    else:
        target = {'weights': 'model.safetensors', 'add': 'processor_config.json', 'tokenizer': 'tokenizer.json'}[mutation]
        (model / target).write_text('{"changed":true}')
    result = verify_inputs(output)
    assert result['status'] == 'failed'
    assert result['model']['checked_files'] == 3
    if mutation == 'add':
        assert result['model']['added'] == ['processor_config.json']
    assert verify_bundle(output)['status'] == 'verified'


@pytest.mark.parametrize('explicit', [False, True])
def test_action_default_and_explicit_index_require_manifest(tmp_path, explicit):
    args = request_args(tmp_path)
    data = tmp_path / 'action'
    data.mkdir()
    args.action_data = data
    index = data / 'effective_pool_audit.jsonl'
    index.write_text(''.join(json.dumps(dict(scenario='Scene', run_id=s, split=s)) + '\n' for s in ('train', 'val', 'test')))
    if explicit:
        args.action_effective_index = index
    config = make_config(args)
    manifest, report = capture(config, tmp_path / 'capture')
    assert any(a['id'] == 'action_effective_manifest' and a['status'] == 'missing' for a in manifest['artifacts'])
    assert len([r for r in report['missing_full_pools'] if r['phase'] == 'action']) == 3
    assert next(r for r in report['sources'] if r['id'] == 'action_effective_index')['status'] == 'invalid'


@pytest.mark.parametrize('target', ['phase4', 'action', 'history', 'selected', 'phase3_manifest'])
def test_corrupt_metadata_produces_blocked_handoff(tmp_path, target):
    args = request_args(tmp_path)
    if target == 'phase4':
        args.phase4_data2 = tmp_path / 'data2'
        bad = args.phase4_data2 / 'manifest.json'
    elif target == 'action':
        args.action_data = tmp_path / 'action'
        bad = args.action_data / 'effective_pool_manifest.json'
    elif target == 'selected':
        args.phase3_adapter = tmp_path / 'adapter'
        bad = args.phase3_adapter / 'sft_new_loop_phase3_adapter_config.json'
    elif target == 'history':
        bad = tmp_path / 'AutoMoT/checkpoints/sft_new_loop_phase3_20261008_fixture_audit_bundle/audit_bundle/adapter_metadata/sft_new_loop_phase3_adapter_config.json'
    else:
        args.phase3_index = tmp_path / 'phase3/frame_index.jsonl'
        bad = args.phase3_index.parent / 'manifest.json'
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"truncated":')
    config = make_config(args)
    assert config['prepare_diagnostics']
    manifest, report = capture(config, tmp_path / 'capture')
    entry = next(a for a in manifest['artifacts'] if a['path'] == str(bad))
    assert entry['status'] == 'invalid'
    assert entry['sha256'] == file_sha(bad)
    assert not report['full_pool_coverage_complete']
    assert verify_package(pack(tmp_path / 'capture')['archive'])['status'] == 'verified'


def action_fixture(tmp_path):
    data, mapping, root = tmp_path / 'data', tmp_path / 'mapping', tmp_path / 'rgb'
    data.mkdir(); mapping.mkdir()
    for split in ('train', 'val', 'test'):
        (data / f'{split}.jsonl').write_text(json.dumps(dict(scenario='Scene', run_id=split, split=split)) + '\n')
        rgb = root / 'Scene' / split / 'rgb'
        rgb.mkdir(parents=True)
        (rgb / '0000.jpg').write_bytes(b'count only')
    (mapping / 'index.jsonl').write_text('{}\n')
    (mapping / 'manifest.json').write_text('{}')
    args = SimpleNamespace(data_root=str(root), data_dir=str(data), event_balance_index=str(mapping / 'index.jsonl'))
    def native_rows(args, split):
        return [dict(scenario='Scene', run_id=split, split=split, anchor=4)]
    argv = ['--data-root', str(root), '--data-dir', str(data), '--event-balance-index', str(mapping / 'index.jsonl')]
    with patch('qwen3vl_local.action_prior.config.read_rows', side_effect=native_rows):
        export_pool(argv, tmp_path / 'export')
    manifest = json.loads((tmp_path / 'export/effective_pool_manifest.json').read_text())
    return args, manifest, tmp_path / 'export/effective_pool_audit.jsonl'


def test_action_route_counts_are_rechecked(tmp_path):
    args, manifest, index = action_fixture(tmp_path)
    validate_export(manifest, index)
    (Path(args.data_root) / 'Scene/train/rgb/0001.jpg').write_bytes(b'count only')
    with pytest.raises(ValueError, match='frame counts/filter outcomes'):
        validate_export(manifest, index)


def test_action_contract_covers_duration_filter_and_declared_release_dependencies():
    contract = export_contract()
    assert 'lead_video_tools/abnormal_duration_filter.py' in contract
    assert 'keyframe_filter/evidence_guards.py' in contract


def test_action_filter_outcome_drift_and_source_drift_are_rejected(tmp_path):
    args, manifest, index = action_fixture(tmp_path)
    with patch('lead_video_tools.abnormal_duration_filter.is_abnormal_lead_route', return_value=(True, {'changed': True})):
        with pytest.raises(ValueError, match='filter outcomes'):
            validate_export(manifest, index)
    changed = dict(manifest['source_contract'])
    changed['lead_video_tools/abnormal_duration_filter.py'] = '0' * 64
    with patch('qwen3vl_local.audit_joint.action_pool.export_contract', return_value=changed):
        with pytest.raises(ValueError, match='contract/schema'):
            validate_export(manifest, index)


@pytest.mark.parametrize('explicit', [False, True])
def test_action_complete_pool_requires_verified_dependencies(tmp_path, explicit):
    args, manifest, index = action_fixture(tmp_path)
    config_args = request_args(tmp_path)
    config_args.action_data = Path(args.data_dir)
    if explicit:
        config_args.action_effective_index = index
    else:
        shutil.copyfile(index, Path(args.data_dir) / index.name)
        shutil.copyfile(index.with_name('effective_pool_manifest.json'), Path(args.data_dir) / 'effective_pool_manifest.json')
    config = make_config(config_args)
    _, report = capture(config, tmp_path / 'valid_capture')
    assert not [r for r in report['missing_full_pools'] if r['phase'] == 'action']
    (Path(args.data_root) / 'Scene/train/rgb/changed.jpg').touch()
    manifest, report = capture(config, tmp_path / 'stale_capture')
    assert next(a for a in manifest['artifacts'] if a['id'] == 'action_effective_manifest')['status'] == 'invalid'
    assert len([r for r in report['missing_full_pools'] if r['phase'] == 'action']) == 3


def paired_fixture(tmp_path):
    from qwen3vl_local.sft_new_loop_phase4.observation import observation_contract
    paths = {}
    for mode in (2, 4):
        root = tmp_path / str(mode)
        root.mkdir()
        paths[str(mode)] = str(root)
        frames = [10 + d for d in observation_contract(mode)['frame_offsets']]
        row = dict(id='pair', physical_group='Scene/route', scenario='Scene', route_id='route',
                   episode={}, edge='hold', slice='readiness', target='YES', label_basis='per_frame_conditions',
                   split='train', observation=dict(frame_id=10, speed_mps=0, history_frames=frames),
                   images=[f'{f}.jpg' for f in frames], image_sha256=[str(f) * 64 for f in frames],
                   image_rgb_sha256=[str(f) * 64 for f in frames])
        (root / 'train.jsonl').write_text(json.dumps(row) + '\n')
        manifest = dict(rgb_mode=mode, observation_contract=observation_contract(mode),
                        counts={'train': 1}, files={'train.jsonl': file_sha(root / 'train.jsonl')})
        (root / 'manifest.json').write_text(json.dumps(manifest))
    return paths


@pytest.mark.parametrize('mutation', ['same_path', 'wrong_mode', 'binding', 'answer', 'shared_rgb'])
def test_pairing_rejects_wrong_slot_or_mismatched_training(tmp_path, mutation):
    paths = paired_fixture(tmp_path)
    assert check_pair(paths)['pairs'] == 1
    root = Path(paths['4'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if mutation == 'same_path':
        paths['2'] = paths['4']
    elif mutation == 'wrong_mode':
        manifest['rgb_mode'] = 2
    elif mutation == 'binding':
        manifest['seed'] = 99
    else:
        row = json.loads((root / 'train.jsonl').read_text())
        if mutation == 'answer':
            row['target'] = 'NO'
        else:
            row['image_rgb_sha256'][-1] = 'f' * 64
        (root / 'train.jsonl').write_text(json.dumps(row) + '\n')
        manifest['files']['train.jsonl'] = file_sha(root / 'train.jsonl')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        check_pair(paths)


def test_same_dataset_is_rejected_during_prepare_before_large_file_reads(tmp_path):
    paths = paired_fixture(tmp_path)
    args = request_args(tmp_path)
    args.phase4_data2 = args.phase4_data4 = Path(paths['4'])
    config = make_config(args)
    specs = [a for a in config['artifacts'] if a['id'].startswith('phase4_rgb')]
    assert len(specs) == 2
    assert all('same resolved directory' in a['prepare_error'] for a in specs)


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_action_raw_pool_must_match_exported_split_content(tmp_path, split):
    args, _, index = action_fixture(tmp_path)
    other = tmp_path / 'other'
    shutil.copytree(args.data_dir, other)
    (other / f'{split}.jsonl').write_text(json.dumps(dict(scenario='Scene', run_id=split, split=split, anchor=99)) + '\n')
    config_args = request_args(tmp_path)
    config_args.action_data = other
    config_args.action_effective_index = index
    manifest, report = capture(make_config(config_args), tmp_path / 'capture')
    item = next(a for a in manifest['artifacts'] if a['id'] == 'action_effective_manifest')
    assert item['status'] == 'invalid'
    assert 'runtime inputs' in item['error']
    assert len([r for r in report['missing_full_pools'] if r['phase'] == 'action']) == 3


def test_action_raw_pool_relocation_preserving_bytes_is_allowed(tmp_path):
    args, _, index = action_fixture(tmp_path)
    moved = tmp_path / 'moved'
    Path(args.data_dir).rename(moved)
    config_args = request_args(tmp_path)
    config_args.action_data = moved
    config_args.action_effective_index = index
    config = make_config(config_args)
    # Older requests are bound using the raw source paths as well.
    next(a for a in config['artifacts'] if a['id'] == 'action_effective_manifest').pop('expected_split_paths')
    manifest, report = capture(config, tmp_path / 'capture')
    assert next(a for a in manifest['artifacts'] if a['id'] == 'action_effective_manifest')['status'] == 'verified'
    assert not [r for r in report['missing_full_pools'] if r['phase'] == 'action']


@pytest.mark.parametrize('kind', ['index', 'adapter'])
@pytest.mark.parametrize('mutation', ['different_bytes', 'same_bytes', 'dangling', 'parent_link'])
def test_verify_inputs_checks_logical_reference_binding(tmp_path, kind, mutation):
    from qwen3vl_local.audit_joint.storage import capture_input
    _, output = model_capture(tmp_path)
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir(); new_dir.mkdir()
    name = 'index.jsonl' if kind == 'index' else 'adapter_model.safetensors'
    old, new = old_dir / name, new_dir / name
    old.write_bytes(b'original')
    new.write_bytes(b'changed' if mutation == 'different_bytes' else b'original')
    link = tmp_path / ('link_dir' if mutation == 'parent_link' else name)
    link.symlink_to(old_dir if mutation == 'parent_link' else old)
    logical = link / name if mutation == 'parent_link' else link
    record = capture_input(logical, output, copy=False)
    assert record['logical_path'] == str(logical)
    # Build a receipt-covered fixture reference in the appropriate location.
    target = output / ('split_cross_audit.json' if kind == 'index' else 'baseline_manifest.json')
    value = json.loads(target.read_text())
    if kind == 'index':
        value['sources'].append(dict(snapshot=record))
    else:
        value['artifacts'].append(record)
    target.write_text(json.dumps(value))
    receipt = json.loads((output / 'receipt.json').read_text())
    receipt['files'][target.name] = file_sha(target)
    (output / 'receipt.json').write_text(json.dumps(receipt))
    assert verify_inputs(output)['status'] == 'verified'
    link.unlink()
    link.symlink_to(new_dir if mutation == 'parent_link' else (tmp_path / 'absent' if mutation == 'dangling' else new))
    result = verify_inputs(output)
    assert result['status'] == 'failed'
    reference = result['references'][0]
    assert reference['target_status'] == 'verified'
    assert reference['binding_status'] == ('unavailable' if mutation == 'dangling' else 'retargeted')
    assert verify_bundle(output)['status'] == 'verified'


@pytest.mark.parametrize('failure', ['missing_manifest', 'corrupt_manifest', 'wrong_sha'])
def test_uncertified_source_keeps_conflicts_and_excludes_candidates(tmp_path, failure):
    args = request_args(tmp_path)
    args.phase3_index = tmp_path / 'p3/frame_index.jsonl'
    args.phase3_index.parent.mkdir()
    (args.data_root / 'Scene/route').mkdir(parents=True)
    args.phase3_index.write_text(json.dumps(dict(scenario='Scene', route_id='route', split='train')) + '\n')
    if failure == 'corrupt_manifest':
        (args.phase3_index.parent / 'manifest.json').write_text('{')
    if failure == 'wrong_sha':
        (args.phase3_index.parent / 'manifest.json').write_text('{}')
    config = make_config(args)
    if failure == 'wrong_sha':
        next(s for s in config['split_sources'] if s['id'] == 'phase3_full_index')['expected_sha256'] = '0' * 64
    config['split_sources'] += [
        pool(tmp_path / 'val.jsonl', 'phase4', 'dev_val', 'route', args.data_root),
        pool(tmp_path / 'candidate.jsonl', 'phase4', 'candidate_dev_val', 'route', args.data_root)]
    _, report = capture(config, tmp_path / 'capture')
    source = next(s for s in report['sources'] if s['id'] == 'phase3_full_index')
    assert source['status'] == 'invalid'
    assert source['observation_status'] == 'observed'
    assert source['snapshot']['logical_path'] == str(args.phase3_index)
    conflict = next(c for c in report['conflicts'] if c['kind'] == 'training_holdout_overlap')
    assert conflict['evidence_status'] == 'suspected'
    assert any(e['source'] == 'phase3_full_index' and e['source_status'] == 'invalid' for e in conflict['sources'])
    assert report['candidate_filter'][0]['status'] == 'excluded'
    assert any(r['phase'] == 'phase3' and r['role'] == 'train_pool' for r in report['missing_full_pools'])
    assert verify_package(pack(tmp_path / 'capture')['archive'])['status'] == 'verified'


def test_uncertified_manual_val_does_not_count_as_support(tmp_path):
    source = pool(tmp_path / 'val.jsonl', 'phase4', 'dev_val', 'route',
                  episode={'event': 'UE1'}, label_basis='per_frame_conditions',
                  observation={'speed_mps': 0}, target='YES', edge='proceed')
    source.update(validation_error='manifest invalid', full_pool=True)
    report, ledger = audit_sources([source])
    assert report['sources'][0]['observation_status'] == 'observed'
    assert not report['manual_val_support']
    assert not report['manual_val_observed_events']
    assert ledger[0]['uses'][0]['source_status'] == 'invalid'
    assert 'certification_errors' not in ledger[0]['uses'][0]  # Reasons stored once per source.
    assert len(report['missing_full_pools']) == 9
