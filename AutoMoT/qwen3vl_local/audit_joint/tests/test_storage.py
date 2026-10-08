import json
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from qwen3vl_local.audit_joint.baseline import run, verify_bundle
from qwen3vl_local.audit_joint.io import file_sha
from qwen3vl_local.audit_joint.storage import plan, verify_inputs, inventory
from qwen3vl_local.audit_joint.handoff import pack, verify_package
from qwen3vl_local.audit_joint import guard


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'src'
        source.mkdir()
        (source / 'code.py').write_text('pass\n')
        self.weights = self.root / 'adapter_model.safetensors'
        self.weights.write_bytes(b'fake weight' * 400000)
        metadata = self.root / 'adapter_config.json'
        metadata.write_text('{}')
        self.index = self.root / 'train.jsonl'
        self.index.write_text('{"scenario":"Scene","route_id":"route","split":"train"}\n' * 200)
        self.config = dict(project_root=str(self.root), source_roots=[str(source)],
                           artifacts=[dict(id='weights', path=str(self.weights)), dict(id='config', path=str(metadata))],
                           model_dir=str(self.root / 'absent'), phase3_prompt_variant='baseline',
                           split_sources=[dict(id='train', phase='phase3', path=str(self.index),
                                               role='train_pool', scope='test fixture', full_pool=True)])

    def capture(self, name, mode):
        with patch('qwen3vl_local.audit_joint.baseline.native_contracts', return_value={}), \
                patch('qwen3vl_local.audit_joint.baseline.git_state', return_value={}):
            return run(dict(self.config, storage_mode=mode), self.root / name)

    def test_reference_mode_retains_reports_without_weight_or_index_copy(self):
        manifest, splits = self.capture('capture', 'references')
        output = self.root / 'capture'
        self.assertTrue(manifest['external_assets_required'])
        self.assertEqual(manifest['artifacts'][0]['storage'], 'external_reference')
        self.assertEqual(manifest['artifacts'][1]['storage'], 'snapshot')
        self.assertNotIn('blob', splits['sources'][0]['snapshot'])
        self.assertFalse((output / 'blobs' / file_sha(self.weights)).exists())
        self.assertFalse((output / 'blobs' / file_sha(self.index)).exists())
        self.assertEqual(verify_inputs(output)['status'], 'verified')
        archive = pack(output)
        self.assertEqual(verify_package(archive['archive'])['status'], 'verified')

    def test_external_drift_is_visible_without_rewriting_receipt(self):
        self.capture('capture', 'references')
        output = self.root / 'capture'
        sha = file_sha(output / 'receipt.json')
        self.index.write_text('changed')
        self.assertEqual(verify_bundle(output)['status'], 'verified')
        report = verify_inputs(output)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed', [r['status'] for r in report['references']])
        self.assertEqual(file_sha(output / 'receipt.json'), sha)
        self.weights.unlink()
        self.assertIn('unavailable', [r['status'] for r in verify_inputs(output)['references']])

    def test_full_mode_explicitly_copies_and_plan_predicts_savings(self):
        small = plan(self.config, self.root / 'ref')
        full = plan(self.config, self.root / 'full', 'full')
        self.assertEqual(full['copy_bytes_upper_bound'] - small['copy_bytes_upper_bound'],
                         self.weights.stat().st_size + self.index.stat().st_size)
        manifest, _ = self.capture('full', 'full')
        self.assertTrue(manifest['full_input_snapshot'])
        self.assertTrue((self.root / 'full/blobs' / file_sha(self.weights)).exists())

    def test_low_space_rejected_before_output_creation(self):
        with patch('qwen3vl_local.audit_joint.storage.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'no capture created'):
                self.capture('capture', 'references')
        self.assertFalse((self.root / 'capture').exists())

    def test_large_json_artifact_is_also_reference_only(self):
        large = self.root / 'large.json'
        large.write_text(json.dumps({'large': 'x' * 3_000_000}))
        self.config['artifacts'].append(dict(id='large', path=str(large)))
        manifest, _ = self.capture('capture', 'references')
        self.assertEqual(manifest['artifacts'][-1]['storage'], 'external_reference')

    def test_guard_inherits_zero_limits_and_preserves_exit_status(self):
        original = resource.getrlimit(resource.RLIMIT_CORE)
        marker = self.root / 'limits.json'
        code = ('import json,resource,sys; from pathlib import Path; '
                'Path(sys.argv[1]).write_text(json.dumps(resource.getrlimit(resource.RLIMIT_CORE))); '
                'sys.exit(7)')
        with patch.object(guard, 'core_pattern', return_value='core.%p'):
            code = guard.run([sys.executable, '-c', code, str(marker)], self.root, 0, .01)
        self.assertEqual(code, 7)
        self.assertEqual(json.loads(marker.read_text()), [0, 0])
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CORE), original)

    def test_guard_refuses_piped_core_collector_without_launch(self):
        with patch.object(guard, 'core_pattern', return_value='|/collector'), patch.object(guard.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'No job launched'):
                guard.run(['python'], self.root)
        launch.assert_not_called()

    def test_guard_low_disk_before_launch(self):
        with patch.object(guard, 'core_pattern', return_value='core'), \
                patch.object(guard.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)), \
                patch.object(guard.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(OSError, 'no job launched'):
                guard.run(['python'], self.root, 2)
        launch.assert_not_called()

    def test_guard_stops_owned_job_if_disk_drops(self):
        process = Mock(pid=12345)
        with patch.object(guard, 'core_pattern', return_value='core'), \
                patch.object(guard.shutil, 'disk_usage', side_effect=[SimpleNamespace(free=10), SimpleNamespace(free=1)]), \
                patch.object(guard.subprocess, 'Popen', return_value=process), \
                patch.object(guard, 'stop_group') as stop:
            self.assertEqual(guard.run(['python'], self.root, 2), 75)
        stop.assert_called_once_with(process)

    def test_inventory_identifies_elf_core_without_deletion_or_following_links(self):
        folder = self.root / 'checkpoints'
        folder.mkdir()
        (folder / 'large.bin').write_bytes(b'x' * 12000)
        (folder / 'external').symlink_to(self.root, target_is_directory=True)
        (folder / 'core.py').write_text('not a crash dump')
        header = b'\x7fELF' + bytes([2, 1]) + bytes(10) + bytes([4, 0])
        core = self.root / 'core.123'
        core.write_bytes(header)
        report = inventory(folder, self.root)
        self.assertEqual(report['files'], 2)
        self.assertEqual(report['confirmed_elf_core_count'], 1)
        self.assertEqual(report['largest_elf_cores'][0]['path'], str(core))
        self.assertEqual(core.read_bytes(), header)


if __name__ == '__main__':
    unittest.main()
