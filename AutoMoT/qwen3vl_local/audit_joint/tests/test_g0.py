import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from qwen3vl_local.audit_joint import SCHEMA
from qwen3vl_local.audit_joint.io import file_sha, snapshot_file, source_files, write_json
from qwen3vl_local.audit_joint.splits import audit_sources, group_of
from qwen3vl_local.audit_joint.baseline import check_native_artifact, run, verify_bundle


class G0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, rows, phase='phase3', role='train_pool', **kwargs):
        path = self.root / (name + '.jsonl')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return dict(id=name, phase=phase, role=role, path=str(path), scope='test fixture', **kwargs)

    def row(self, route='Town12_Rep0_42_0_route0_01_10_12_30_00', **extra):
        return dict(scenario='Scene', route_id=route, **extra)

    def test_rep_timestamp_isolation_across_tasks(self):
        specs = [self.source('train', [self.row()]), self.source('test', [self.row(
            'Town12_Rep7_42_0_route0_02_11_13_35_05')], phase='phase4', role='final_test')]
        report, ledger = audit_sources(specs)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(report['conflicts'][0]['kind'], 'training_holdout_overlap')
        self.assertFalse(report['independent_admission_ready'])

    def test_full_pool_includes_unsampled_route(self):
        specs = [self.source('pool', [self.row('unused'), self.row('sampled')], full_pool=True),
                 self.source('holdout', [self.row('unused')], phase='action', role='dev_val')]
        report, _ = audit_sources(specs)
        self.assertEqual(report['conflicts'][0]['physical_groups'], ['Scene/unused'])

    def test_declared_group_cannot_override_native_identity(self):
        spec = self.source('bad', [self.row(physical_group='Scene/safe')])
        report, ledger = audit_sources([spec])
        self.assertEqual(report['sources'][0]['status'], 'invalid')
        self.assertEqual(ledger, [])

    def test_invalid_last_row_does_not_commit_partial_source(self):
        spec = self.source('bad', [self.row(), {'garbage': True}], full_pool=True)
        report, ledger = audit_sources([spec])
        self.assertEqual(ledger, [])
        self.assertEqual(len(report['missing_full_pools']), 9)

    def test_canonical_group_validation(self):
        for group in ('Scene/../route', 'Scene/Town_Rep1_route_route0_01_01_01_01_01', '/route'):
            with self.assertRaises(ValueError):
                group_of(group)
        self.assertEqual(group_of('Scene/Town_route'), 'Scene/Town_route')

    def test_missing_pool_never_passes(self):
        spec = dict(id='missing', phase='action', role='train_pool', path=str(self.root / 'absent'),
                    scope='missing full pool', full_pool=True)
        report, _ = audit_sources([spec])
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(len(report['missing_full_pools']), 9)

    def test_empty_pool_not_complete(self):
        report, _ = audit_sources([self.source('empty', [], full_pool=True)])
        self.assertEqual(report['sources'][0]['status'], 'invalid')

    def test_wrong_source_sha_fails(self):
        report, _ = audit_sources([self.source('pool', [self.row()], expected_sha256='0' * 64)])
        self.assertEqual(report['sources'][0]['status'], 'invalid')

    def test_duplicate_and_malformed_source_ids_fail(self):
        spec = self.source('same', [self.row()])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit_sources([spec, spec])
        spec['id'] = 'bad/id'
        with self.assertRaisesRegex(ValueError, 'identifiers'):
            audit_sources([spec])

    def test_split_mismatch_and_unknown_role_fail(self):
        specs = [self.source('wrongsplit', [self.row(split='train')], role='final_test', expected_split='test'),
                 self.source('wrongrole', [self.row()], role='something')]
        report, ledger = audit_sources(specs)
        self.assertTrue(all(r['status'] == 'invalid' for r in report['sources']))
        self.assertEqual(ledger, [])

    def test_candidate_excludes_all_protected_uses(self):
        roles = ['train_pool', 'development_exposure', 'historical_regression',
                 'independent_review', 'final_test', 'quarantine']
        specs = [self.source(role, [self.row(role)], role=role) for role in roles]
        specs.append(self.source('candidates', [self.row(role) for role in roles] + [self.row('fresh')],
                                 role='candidate_dev_val', phase='phase4'))
        report, _ = audit_sources(specs)
        decisions = {r['physical_group']: r for r in report['candidate_filter']}
        for role in roles:
            self.assertEqual(decisions['Scene/' + role]['status'], 'excluded')
            self.assertEqual(decisions['Scene/' + role]['reasons'][0]['source'], role)
        self.assertEqual(decisions['Scene/fresh']['status'], 'eligible_pending_complete_audit')
        self.assertEqual(report['reservations_written'], 0)

    def test_symlink_alias_not_hidden_by_renaming(self):
        data = self.root / 'data'
        scene = data / 'Scene'
        scene.mkdir(parents=True)
        external = self.root / 'external'
        external.mkdir()
        (scene / 'a').symlink_to(external, target_is_directory=True)
        (scene / 'b').symlink_to(external, target_is_directory=True)
        specs = [self.source('train', [self.row('a')], data_root=str(data)),
                 self.source('candidate', [self.row('b')], role='candidate_dev_val', data_root=str(data))]
        report, _ = audit_sources(specs)
        self.assertEqual(report['aliases'][0]['kind'], 'resolved_route')
        self.assertEqual(report['candidate_filter'][0]['status'], 'excluded')

    def test_shared_pixels_are_review_not_assumed_same_physical_route(self):
        image = dict(images=['rgb.jpg'], image_sha256=['1' * 64], image_rgb_sha256=['2' * 64])
        specs = [self.source('train', [self.row('a', **image)]),
                 self.source('candidate', [self.row('b', **image)], role='candidate_dev_val')]
        report, ledger = audit_sources(specs)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(len(report['aliases']), 2)
        self.assertTrue(all(a['conclusion'] == 'potential_duplicate_content' for a in report['aliases']))
        self.assertFalse(report['sources'][0]['image_hashes_verified'])
        self.assertEqual(report['candidate_filter'][0]['status'], 'excluded')

    def test_unknown_and_weak_references_do_not_fill_manual_support(self):
        rows = [self.row('a', label_basis='reviewed_transition_band', episode={'event': 'U-E1'},
                         edge='proceed', target='YES', observation={'speed_mps': .2}),
                self.row('b', label_basis='heldout_rule_reference', episode={'event': 'U-E2'},
                         edge='depart', target='YES'),
                self.row('c', episode={'event': 'U-E3'}, edge='proceed', target='NO')]
        report, _ = audit_sources([self.source('val2', rows, phase='phase4', role='dev_val')])
        self.assertEqual(report['manual_val_observed_events'], ['U-E1'])
        self.assertIn('U-E2', report['manual_val_missing_events'])
        self.assertIn('val2/U-E1/proceed/YES/stationary', report['manual_val_support'])

    def test_json_reservation_preserves_purpose(self):
        path = self.root / 'reserve.json'
        write_json(path, {'reservations': [self.row(split='test')]})
        spec = dict(id='reserve', phase='phase4', path=str(path), scope='reserved',
                    records_key='reservations', role='independent_review')
        report, _ = audit_sources([spec, self.source('final', [self.row()], role='final_test')])
        self.assertEqual(report['conflicts'][0]['kind'], 'holdout_role_overlap')

    def test_snapshot_captures_actual_bytes_and_does_not_rewrite_original(self):
        src = self.root / 'untracked.py'
        src.write_text('dirty = True\n')
        entry = snapshot_file(src, self.root / 'snapshot')
        self.assertEqual((self.root / 'snapshot' / entry['blob']).read_bytes(), src.read_bytes())
        old = entry['sha256']
        src.write_text('dirty = False\n')
        self.assertNotEqual(file_sha(src), old)
        self.assertEqual(file_sha(self.root / 'snapshot' / entry['blob']), old)

    def test_source_snapshot_ignores_generated_corpus_and_weights(self):
        source = self.root / 'src'
        (source / 'collection_output').mkdir(parents=True)
        (source / 'collection_output' / 'large.json').write_text('{}')
        (source / 'model.safetensors').write_bytes(b'weights')
        (source / 'code.py').write_text('pass')
        self.assertEqual(source_files([source]), [source / 'code.py'])

    def test_binary_artifact_is_hashed_without_json_parsing(self):
        path = self.root / 'adapter.safetensors'
        path.write_bytes(b'\xff\x00')
        self.assertEqual(check_native_artifact({'path': str(path)}, {})['check'],
                         'hash_only_no_native_compatibility_claim')

    def test_native_phase4_contract_mismatch_is_not_repaired(self):
        path = self.root / 'manifest.json'
        write_json(path, {'contract': {'task': 'old'}})
        before = file_sha(path)
        with self.assertRaisesRegex(ValueError, 'task'):
            check_native_artifact({'path': str(path), 'check': 'phase4_manifest'}, {'phase4': {'task': 'new'}})
        self.assertEqual(file_sha(path), before)

    def test_receipt_rejects_mutation_and_added_file(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        (bundle / 'report.json').write_text('{}')
        write_json(bundle / 'receipt.json', dict(schema=SCHEMA, files={'report.json': file_sha(bundle / 'report.json')}))
        self.assertEqual(verify_bundle(bundle)['status'], 'verified')
        (bundle / 'extra').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            verify_bundle(bundle)
        (bundle / 'extra').unlink()
        (bundle / 'report.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'hash'):
            verify_bundle(bundle)

    def test_capture_reports_missing_assets_and_refuses_overwrite(self):
        src = self.root / 'src'
        src.mkdir()
        (src / 'model.py').write_text('pass')
        config = dict(project_root=str(self.root), source_roots=[str(src)], artifacts=[],
                      model_dir=str(self.root / 'missing_model'), split_sources=[], phase3_prompt_variant='baseline')
        output = self.root / 'result'
        with patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}), \
                patch('qwen3vl_local.audit_joint.baseline.git_state', return_value={}):
            manifest, report = run(config, output)
            self.assertFalse(manifest['reproducible_baseline_ready'])
            self.assertEqual(manifest['stages']['E1'], 'not_run')
            self.assertEqual(verify_bundle(output)['status'], 'verified')
            with self.assertRaises(FileExistsError):
                run(config, output)

    def test_raw_action_split_does_not_fabricate_effective_leakage(self):
        report, _ = audit_sources([
            self.source('raw', [self.row(split='val')], phase='action', role='raw_inventory'),
            self.source('effective', [self.row(split='train')], phase='action', role='train_pool'),
        ])
        self.assertEqual(report['conflicts'], [])

    def test_source_change_mid_capture_leaves_no_success_receipt(self):
        src = self.root / 'src'
        src.mkdir()
        code = src / 'code.py'
        code.write_text('before')
        config = dict(project_root=str(self.root), source_roots=[str(src)], artifacts=[],
                      model_dir=str(self.root / 'absent'), split_sources=[], phase3_prompt_variant='baseline')
        def change_source(variant):
            code.write_text('after')
            return {}
        with patch('qwen3vl_local.audit_joint.baseline.native_contracts', side_effect=change_source):
            with self.assertRaisesRegex(RuntimeError, 'source tree changed'):
                run(config, self.root / 'out')
        self.assertFalse((self.root / 'out/receipt.json').exists())

    def test_action_export_uses_native_runtime_pool_without_sampling(self):
        from qwen3vl_local.audit_joint.action_pool import export_pool
        data, mapping = self.root / 'action', self.root / 'map'
        data.mkdir()
        mapping.mkdir()
        for split in ('train', 'val', 'test'):
            (data / f'{split}.jsonl').write_text(json.dumps(dict(scenario='Scene', run_id=split)) + '\n')
            (self.root / 'rgb_data' / 'Scene' / split / 'rgb').mkdir(parents=True)
        (mapping / 'index.jsonl').write_text('{}\n')
        (mapping / 'manifest.json').write_text('{}')
        args = SimpleNamespace(data_dir=str(data), data_root=str(self.root / 'rgb_data'),
                               event_balance_index=str(mapping / 'index.jsonl'))
        def native_rows(args, split):
            return [dict(scenario='Scene', run_id=split, route_group='Scene/' + split, split=split,
                         anchor=n, event_balance_original_split='val' if split == 'train' else split)
                    for n in range(4)]
        with patch('qwen3vl_local.action_prior.config.parser') as parser, \
                patch('qwen3vl_local.action_prior.config.read_rows', side_effect=native_rows) as read:
            parser.return_value.parse_args.return_value = args
            result = export_pool(['--data-dir', str(data)], self.root / 'export')
        self.assertEqual(read.call_count, 3)
        self.assertEqual(result['counts'], dict(train=4, val=4, test=4))
        saved = json.loads((self.root / 'export/effective_pool_manifest.json').read_text())
        self.assertEqual(saved['index_sha256'], file_sha(result['index']))
        self.assertFalse(saved['model_loaded'])

    def test_prepare_keeps_baseline_and_candidate_indices_distinct(self):
        from qwen3vl_local.audit_joint.config import make_config
        project = Path(__file__).resolve().parents[4]
        args = SimpleNamespace(project_root=project, phase3_prompt_variant='baseline',
                               data_root=project / 'AutoMoT/lead_data', phase3_index=None,
                               phase4_data2=None, phase4_data4=None, action_data=None,
                               action_effective_index=None, candidate_val=None,
                               phase3_adapter=None, model_dir=None, verify_images=False)
        baseline = make_config(args)
        self.assertTrue(next(s for s in baseline['split_sources'] if s['id'] == 'phase3_full_index')['path'].endswith(
            'sft_new_loop_phase3_data_v23/frame_index.jsonl'))
        args.phase3_prompt_variant = 'v23_rgb_stage_candidate_20261006'
        candidate = make_config(args)
        spec = next(s for s in candidate['split_sources'] if s['id'] == 'phase3_full_index')
        self.assertTrue(spec['path'].endswith('sft_new_loop_phase3_data_rgb_stage_20261006_isolated/frame_index.jsonl'))
        self.assertTrue(all(s['role'] == 'raw_inventory' for s in candidate['split_sources'] if s['id'].startswith('action_raw_')))


if __name__ == '__main__':
    unittest.main()
