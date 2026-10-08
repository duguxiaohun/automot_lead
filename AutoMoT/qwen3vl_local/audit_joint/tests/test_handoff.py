import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from qwen3vl_local.audit_joint import SCHEMA
from qwen3vl_local.audit_joint.baseline import verify_bundle
from qwen3vl_local.audit_joint.handoff import pack, verify_package
from qwen3vl_local.audit_joint.io import file_sha, snapshot_file, write_json


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root / 'capture'
        self.capture.mkdir()
        code = self.root / 'code.py'
        code.write_bytes(os.urandom(18000))
        source = snapshot_file(code, self.capture)
        weights = self.root / 'adapter_model.safetensors'
        weights.write_bytes(os.urandom(24000))
        weight = snapshot_file(weights, self.capture)
        raw = self.root / 'train.jsonl'
        raw.write_text('{"raw_training": true}\n')
        snapshot_file(raw, self.capture)
        self.baseline = dict(reproducible_baseline_ready=False,
                             blockers=[dict(kind='execution', status='not_run')],
                             stages={'E0': 'not_run', 'E1': 'not_run'},
                             artifacts=[dict(id='weights', path=str(weights), **weight)])
        for name, value in [('request.json', {}), ('baseline_manifest.json', self.baseline),
                            ('source_manifest.json', {'project/code.py': source}),
                            ('split_cross_audit.json', {'conflicts': ['real_conflict']})]:
            write_json(self.capture / name, value)
        (self.capture / 'exposure_ledger.jsonl').write_text('{"all_groups": true}\n')
        write_json(self.capture / 'receipt.json', dict(schema=SCHEMA, files={
            p.relative_to(self.capture).as_posix(): file_sha(p)
            for p in self.capture.rglob('*') if p.is_file()}))

    def test_full_reports_source_bytes_and_pending_stages_preserved(self):
        report = pack(self.capture)
        archive = Path(report['archive'])
        self.assertLessEqual(report['bytes'], 30_000_000)
        self.assertEqual(verify_package(archive)['status'], 'verified')
        self.assertEqual(verify_bundle(self.capture)['status'], 'verified')
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('handoff_manifest.json'))
            self.assertFalse(manifest['baseline_ready'])
            self.assertEqual(manifest['stages']['E1'], 'not_run')
            self.assertIn('source/project/code.py', z.namelist())
            self.assertFalse(any('blobs/' in p or 'safetensors' in p or 'train.jsonl' in p for p in z.namelist()))
            self.assertEqual(z.read('g0/split_cross_audit.json'), (self.capture / 'split_cross_audit.json').read_bytes())

    def test_budget_omits_only_source_bytes_explicitly(self):
        report = pack(self.capture, max_bytes=7000)
        self.assertLessEqual(report['bytes'], 7000)
        self.assertFalse(report['source_bytes_included'])
        with zipfile.ZipFile(report['archive']) as z:
            m = json.loads(z.read('handoff_manifest.json'))
            self.assertIn('compressed_size_budget', m['source_omission_reason'])
            self.assertIn('g0/source_manifest.json', z.namelist())
            self.assertIn('g0/exposure_ledger.jsonl', z.namelist())

    def test_oversize_results_fail_without_truncation_or_published_zip(self):
        result = self.root / 'run'
        result.mkdir()
        (result / 'cases.jsonl').write_text(os.urandom(24000).hex())
        with self.assertRaisesRegex(ValueError, 'no cases/metrics truncated'):
            pack(self.capture, results=[('run', result)], max_bytes=7000)
        self.assertFalse(self.root.joinpath('capture.audit.zip').exists())
        self.assertFalse(list(self.root.glob('.audit-pack-*')))
        self.assertEqual(verify_bundle(self.capture)['status'], 'verified')

    def test_results_complete_and_exclusions_inventoried(self):
        result = self.root / 'run'
        result.mkdir()
        (result / 'cases_rank0.jsonl').write_text('{"case_index": 0}\n' * 300)
        (result / 'train_metrics.jsonl').write_text('{"loss": 1}\n')
        (result / 'frame_index.jsonl').write_text('raw dataset')
        (result / 'model.safetensors').write_bytes(b'weight')
        (result / 'external').symlink_to(self.capture, target_is_directory=True)
        report = pack(self.capture, results=[('phase3', result)])
        with zipfile.ZipFile(report['archive']) as z:
            self.assertEqual(z.read('results/phase3/cases_rank0.jsonl'), (result / 'cases_rank0.jsonl').read_bytes())
            m = json.loads(z.read('handoff_manifest.json'))
            self.assertEqual(len(m['results']['phase3']['omitted']), 3)
            self.assertIn('not_performed', m['results']['phase3']['semantic_validation'])

    def test_tampered_capture_rejected(self):
        (self.capture / 'exposure_ledger.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            pack(self.capture)

    def test_no_overwrite_and_archive_outside_capture(self):
        report = pack(self.capture)
        sha = file_sha(report['archive'])
        with self.assertRaises(FileExistsError):
            pack(self.capture)
        self.assertEqual(file_sha(report['archive']), sha)
        with self.assertRaisesRegex(ValueError, 'outside immutable'):
            pack(self.capture, self.capture / 'audit.zip')

    def test_no_result_directory_no_silent_success(self):
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            pack(self.capture, results=[('phase3', self.root / 'missing')])
        with self.assertRaisesRegex(ValueError, 'unique letters'):
            pack(self.capture, results=[('../escape', self.root)])

    def test_size_limit_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            pack(self.capture, max_bytes=30_000_001)

    def test_verify_rejects_altered_or_additional_archive_members(self):
        report = pack(self.capture)
        with zipfile.ZipFile(report['archive'], 'a') as z:
            z.writestr('unexpected.json', '{}')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            verify_package(report['archive'])

    def test_changed_results_during_pack_not_published(self):
        result = self.root / 'run'
        result.mkdir()
        metrics = result / 'metrics.json'
        metrics.write_text('{}')
        from qwen3vl_local.audit_joint import handoff
        original = handoff.result_files
        calls = []
        def mutate(root):
            calls.append(root)
            if len(calls) == 2:
                metrics.write_text('{"changed": true}')
            return original(root)
        with patch.object(handoff, 'result_files', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during packaging'):
                pack(self.capture, results=[('run', result)])
        self.assertFalse(self.root.joinpath('capture.audit.zip').exists())

    def test_cli_run_with_blockers_still_publishes_package_and_returns_two(self):
        from qwen3vl_local.audit_joint.__main__ import main
        config = self.root / 'request.json'
        config.write_text('{"schema": "joint_audit_request_v1"}')
        manifest = dict(self.baseline, source_file_count=1)
        args = ['audit_joint', 'run', '--config', str(config), '--output', str(self.capture)]
        with patch('sys.argv', args), patch('qwen3vl_local.audit_joint.__main__.run',
                                           return_value=(manifest, {'status': 'blocked', 'conflicts': [1]})):
            self.assertEqual(main(), 2)
        self.assertEqual(verify_package(self.root / 'capture.audit.zip')['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
