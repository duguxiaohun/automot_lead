import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import subprocess
import sys
import zipfile

from qwen3vl_local.audit_joint import phase3_recovery_inspect as inspect


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'checkpoints/phase3_pipeline/one/train'
        self.run.mkdir(parents=True)
        (self.run/'train_run_manifest.json').write_text(json.dumps(dict(
            history_rgb_mode='2rgb_short', action_output_mode='choice', epoch_recovery_enabled=True)))
        (self.run/'latest_training_state.json').write_text(json.dumps(dict(
            checkpoint=str(self.run/'training_states/epoch_0002'), next_epoch=2, global_step=2560)))
        self.ckpt = self.run/'training_states/epoch_0002'
        self.ckpt.mkdir(parents=True)
        (self.ckpt/'training_state.pt').write_bytes(b'never deserialize me')
        source = self.root/'qwen3vl_local/sequence_batch.py'
        source.parent.mkdir()
        source.write_text('new source')
        (self.ckpt/'receipt.json').write_text(json.dumps(dict(
            policy='phase3_epoch_training_state_v1',
            binding=dict(sources={'sequence_batch.py': 'old-hash'}, steps_per_epoch=1280),
            files={'training_state.pt': inspect.digest(self.ckpt/'training_state.pt')})))

    def test_readonly_discovery_and_source_drift(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        value = inspect.collect(self.root)
        self.assertEqual(len(value['runs']), 1)
        run = value['runs'][0]
        self.assertEqual(run['latest_pointer']['global_step'], 2560)
        self.assertEqual(run['checkpoints'][0]['source_differences'][0]['path'], 'sequence_batch.py')
        self.assertFalse(run['checkpoints'][0]['hashes_checked'])
        self.assertFalse(run['resume_authorized'])
        self.assertEqual(value['deletion_candidates'], [])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_explicit_hash_and_corruption(self):
        value = inspect.collect(self.root, self.run, True)
        self.assertEqual(value['runs'][0]['checkpoints'][0]['file_hash_errors'], [])
        (self.ckpt/'training_state.pt').write_bytes(b'corrupt')
        value = inspect.collect(self.root, self.run, True)
        self.assertEqual(value['runs'][0]['checkpoints'][0]['file_hash_errors'][0]['error'], 'SHA mismatch')

    def test_never_follow_discovery_alias_or_select_newest(self):
        alias = self.root/'checkpoints/phase3_alias'
        alias.symlink_to(self.run, target_is_directory=True)
        second = self.root/'checkpoints/phase3_other/train'
        second.mkdir(parents=True)
        (second/'train_run_manifest.json').write_text('{}')
        value = inspect.collect(self.root)
        self.assertEqual(len(value['runs']), 2)
        self.assertTrue(all(not r['resume_authorized'] for r in value['runs']))

    def test_partial_checkpoint_and_last_rank_records(self):
        (self.run/'epoch_0003_rank1_batches.jsonl').write_text('{"step":1158}\n{"step":1159}\n')
        pending = self.run/'training_states/epoch_0003.pending'
        pending.mkdir()
        (pending/'receipt.json').write_text('{')
        value = inspect.inspect_run(self.root, self.run)
        self.assertTrue(value['checkpoints'][1]['pending'])
        self.assertTrue(value['checkpoints'][1]['receipt_error'])
        self.assertEqual(value['rank_last_records']['epoch_0003_rank1_batches.jsonl'][-1], '{"step":1159}')

    def test_pipeline_explicit_recorded_run(self):
        (self.run.parent/'pipeline_manifest.json').write_text(json.dumps(dict(run_root=str(self.run))))
        value = inspect.collect(self.root, self.run.parent)
        self.assertEqual(value['runs'][0]['run'], str(self.run))

    def test_extra_checkpoint_file_and_bad_receipt_are_not_accepted(self):
        (self.ckpt/'unexpected.pt').write_bytes(b'extra')
        item = inspect.checkpoint_metadata(self.root, self.ckpt, False)
        self.assertEqual(item['extra_files'], ['unexpected.pt'])
        (self.ckpt/'receipt.json').write_text('{"binding":null,"files":[]}')
        item = inspect.checkpoint_metadata(self.root, self.ckpt, False)
        self.assertTrue(item['metadata_errors'])

    def test_checkpoint_link_outside_tree_is_rejected(self):
        (self.ckpt/'training_state.pt').unlink()
        outside = self.root/'private.pt'
        outside.write_bytes(b'not a checkpoint')
        (self.ckpt/'training_state.pt').symlink_to(outside)
        item = inspect.checkpoint_metadata(self.root, self.ckpt, True)
        self.assertEqual(item['file_hash_errors'][0]['error'], 'outside checkpoint')

    def test_failure_handoff_preserves_good_run_and_reports_missing_run(self):
        with tempfile.TemporaryDirectory() as destination:
            output = Path(destination)/'handoff'
            before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
            with patch.object(inspect, 'process_inventory', return_value={'processes': []}):
                code = inspect.package_inspection(self.root, [self.run, Path('checkpoints/missing')], output)
            self.assertEqual(code, 2)
            receipt = json.loads((output/'phase3_recovery_package.json').read_text())
            archive = output/'phase3_recovery_report.zip'
            self.assertEqual(receipt['archive_sha256'], inspect.digest(archive))
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(bundle.namelist(), ['phase3_recovery_report.json'])
                data = bundle.read('phase3_recovery_report.json')
                self.assertEqual(hashlib.sha256(data).hexdigest(), receipt['members']['phase3_recovery_report.json']['sha256'])
            report = json.loads(data)
            self.assertEqual(len(report['runs']), 1)
            self.assertEqual(len(report['errors']), 1)
            self.assertFalse(receipt['deletion_authorized'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                inspect.package_inspection(self.root, [self.run], output)

    def test_output_cannot_modify_project_or_follow_dangling_link(self):
        with self.assertRaises(ValueError):
            inspect.package_inspection(self.root, [self.run], self.root/'new_report')
        self.assertFalse((self.root/'new_report').exists())
        with tempfile.TemporaryDirectory() as destination:
            alias = Path(destination)/'alias'
            alias.symlink_to(Path(destination)/'missing')
            with self.assertRaises(FileExistsError):
                inspect.package_inspection(self.root, [self.run], alias)

    def test_real_child_timeout_and_independent_process_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'slow.py'
            script.write_text('import time; time.sleep(30)')
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                with patch.object(inspect, '__file__', str(script)):
                    result = inspect.bounded_run(self.root, self.run, timeout=0.1)
                self.assertEqual(result['inspection_error'], 'inspection timeout')
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_cli_requires_runs_for_package_and_rejects_bad_timeout(self):
        for flags in (['--output', str(self.root/'new')], ['--run-timeout', '0']):
            result = subprocess.run([sys.executable, '-B', inspect.__file__, '--root', str(self.root), *flags],
                                    text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
