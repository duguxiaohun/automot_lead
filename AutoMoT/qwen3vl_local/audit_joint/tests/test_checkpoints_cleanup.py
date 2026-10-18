import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from qwen3vl_local.audit_joint import checkpoints_cleanup as cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cp = self.root/'checkpoints'
        self.cp.mkdir()
        for name in ('old_a', 'old_b', 'old_c', 'keep'):
            (self.cp/name).mkdir()
        self.addCleanup(patch.stopall)
        patch.object(cleanup, 'CANDIDATES', ['old_a', 'old_b', 'old_c']).start()
        patch.object(cleanup, 'live_processes', return_value=[]).start()

    def plan(self, **kwargs):
        return cleanup.plan(self.root, **kwargs)

    def test_transitive_reference_and_actual_permanent_delete(self):
        (self.cp/'keep/config.json').write_text('{"data":"../old_a/data"}')
        (self.cp/'old_a/config.json').write_text('{"parent":"../old_b"}')
        (self.cp/'old_c/weight.bin').write_bytes(b'old weights')
        report, snapshot = self.plan()
        self.assertEqual(report['delete'], ['old_c'])
        self.assertEqual(report['kept_referenced'], ['old_a', 'old_b'])
        cleanup.execute(self.root, report, snapshot)
        self.assertEqual(sorted(p.name for p in self.cp.iterdir()), ['keep', 'old_a', 'old_b'])
        self.assertEqual(report['deleted'], ['old_c'])
        self.assertEqual(list(self.root.iterdir()), [self.cp])  # no backup

    def test_obsolete_cycle_is_deleted(self):
        (self.cp/'old_a/config').write_text('old_b')
        (self.cp/'old_b/config').write_text('old_a')
        self.assertEqual(self.plan()[0]['delete'], ['old_a', 'old_b', 'old_c'])

    def test_symlink_reference(self):
        (self.cp/'keep/link').symlink_to('../old_a')
        self.assertEqual(self.plan()[0]['kept_referenced'], ['old_a'])

    def test_symlink_to_checkpoint_ancestor_preserves_all(self):
        (self.cp/'keep/link').symlink_to('../..')
        self.assertEqual(self.plan()[0]['delete'], [])

    def test_intermediate_symlink_directory_is_preserved(self):
        (self.cp/'keep/link').symlink_to('../old_a/link')
        (self.cp/'old_a/link').symlink_to('../old_b')
        self.assertEqual(self.plan()[0]['kept_referenced'], ['old_a', 'old_b'])

    def test_candidate_symlink_blocks(self):
        (self.cp/'old_c').rmdir()
        (self.cp/'old_c').symlink_to('keep')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.plan()

    def test_zip_pickle_extensionless_and_gzip_refs(self):
        with zipfile.ZipFile(self.cp/'keep/optimizer.pt', 'w') as z:
            z.writestr('archive/data.pkl', b'old_a')
            z.writestr('archive/version', b'old_b')
            z.writestr('archive/data/0', b'tensor')
        with gzip.open(self.cp/'keep/source.json.gz', 'wb') as f:
            f.write(b'old_c')
        self.assertEqual(self.plan()[0]['delete'], [])

    def test_safetensors_metadata(self):
        header = json.dumps({'__metadata__': {'source': 'old_a'}}).encode()
        (self.cp/'keep/weight.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'tensors')
        self.assertEqual(self.plan()[0]['kept_referenced'], ['old_a'])

    def test_corrupt_retained_gzip_blocks(self):
        (self.cp/'keep/bad.gz').write_bytes(b'invalid gzip')
        with self.assertRaises(OSError):
            self.plan()
        self.assertTrue((self.cp/'old_a').exists())

    def test_unreachable_obsolete_corrupt_file_does_not_block(self):
        (self.cp/'old_a/bad.gz').write_bytes(b'invalid gzip')
        self.assertEqual(len(self.plan()[0]['delete']), 3)

    def test_changed_file_blocks_before_any_delete(self):
        meta = self.cp/'keep/config.json'
        meta.write_text('{}')
        report, snapshot = self.plan()
        meta.write_text('{"parent":"old_a"}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'old_c').exists())

    def test_added_root_blocks_before_any_delete(self):
        report, snapshot = self.plan()
        (self.cp/'new_run').mkdir()
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'old_a').exists())

    def test_process_start_blocks_before_delete(self):
        report, snapshot = self.plan()
        with patch.object(cleanup, 'live_processes', return_value=[{'pid': 123}]):
            with self.assertRaisesRegex(ValueError, 'process started'):
                cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'old_a').exists())

    def test_nested_mount_blocks(self):
        with patch.object(cleanup.os.path, 'ismount', side_effect=lambda p: p == self.cp/'old_a'):
            with self.assertRaisesRegex(ValueError, 'filesystem'):
                self.plan()

    def test_scan_budget_exhaustion_blocks(self):
        (self.cp/'keep/config').write_bytes(b'x'*100)
        with self.assertRaisesRegex(ValueError, 'byte budget'):
            self.plan(max_bytes=50)
        with self.assertRaisesRegex(ValueError, 'time limit'):
            self.plan(timeout=-1)
        self.assertTrue((self.cp/'old_a').exists())

    def test_only_candidate_list_can_be_deleted(self):
        report, snapshot = self.plan()
        report['delete'] = ['keep']
        with self.assertRaisesRegex(ValueError, 'authorized'):
            cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'keep').exists())


if __name__ == '__main__':
    unittest.main()
