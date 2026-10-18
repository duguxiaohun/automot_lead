import gzip
import json
from pathlib import Path
import struct
import random
import subprocess
import sys
import time
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from qwen3vl_local.audit_joint import checkpoints_cleanup as cleanup


class ReferenceMatcherTests(unittest.TestCase):
    def test_actual_candidate_names_exactly_match_previous_scanner(self):
        names = cleanup.CANDIDATES
        matcher = cleanup.ReferenceMatcher(names)
        rng = random.Random(18)
        cases = [b'', b'202610'*300, b'phase4_short_smoke_rgb4', b'unknown_20261010']
        for _ in range(100):
            parts = [rng.randbytes(100)]
            for name in rng.sample(names, rng.randrange(len(names)+1)):
                parts.extend((name.encode(), b'\0', rng.randbytes(150)))
            cases.append(b''.join(parts))
        for data in cases:
            expected = {n for n in names if n.encode() in data}
            self.assertEqual(matcher.find(data), expected)
            known = set(names[::2])
            self.assertEqual(matcher.find(data, known), expected-known)

    def test_general_names_and_overlapping_patterns(self):
        names = ['a', 'aba', 'baba', 'old_a', 'old_b', 'old_b_long']
        data = b'ababa old_b_long old_a'
        self.assertEqual(cleanup.ReferenceMatcher(names).find(data),
                         {n for n in names if n.encode() in data})


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

    def test_unrelated_baseline_self_link_is_reported_and_preserved(self):
        loop = self.cp/'keep/run_20260805_181121'
        loop.symlink_to(loop.name)
        report, snapshot = self.plan()
        self.assertEqual(len(report['symlink_cycles']), 1)
        self.assertEqual(len(report['delete']), 3)
        cleanup.execute(self.root, report, snapshot)
        self.assertTrue(loop.is_symlink())

    def test_cycle_through_old_directory_keeps_every_intermediate(self):
        (self.cp/'keep/link').symlink_to('../old_a/link')
        (self.cp/'old_a/link').symlink_to('../old_b/link')
        (self.cp/'old_b/link').symlink_to('../old_a/link')
        report, _ = self.plan()
        self.assertEqual(report['kept_referenced'], ['old_a', 'old_b'])
        self.assertEqual(report['delete'], ['old_c'])

    def test_link_traverses_parent_before_dotdot(self):
        (self.cp/'keep/link').symlink_to('../old_a/../old_b')
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

    def test_cross_chunk_boundary_reference_is_preserved(self):
        (self.cp/'keep/data.jsonl').write_bytes(b'x'*(1024**2-2)+b'old_a')
        self.assertEqual(self.plan()[0]['kept_referenced'], ['old_a'])

    def test_only_candidate_list_can_be_deleted(self):
        report, snapshot = self.plan()
        report['delete'] = ['keep']
        with self.assertRaisesRegex(ValueError, 'authorized'):
            cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'keep').exists())

    def feedback(self):
        folder = self.cp/'automot_cleanup.example'
        folder.mkdir()
        script = folder/'checkpoints_cleanup.py'
        script.write_text('old_a old_b old_c')
        (folder/'launch.log').touch()
        output = folder/'cleanup_report.json'
        excluded = cleanup.feedback_files(self.root, output, script)
        output.touch()
        return folder, excluded

    def test_current_directory_feedback_can_grow_without_self_reference(self):
        folder, excluded = self.feedback()
        report, snapshot = self.plan(feedback_files=excluded)
        (folder/'launch.log').write_text('old_a old_b old_c')
        self.assertEqual(report['delete'], ['old_a', 'old_b', 'old_c'])
        cleanup.execute(self.root, report, snapshot)
        self.assertTrue((folder/'cleanup_report.json').exists())

    def test_current_feedback_cannot_hide_new_reference_file(self):
        folder, excluded = self.feedback()
        report, snapshot = self.plan(feedback_files=excluded)
        (folder/'other.json').write_text('old_a')
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.execute(self.root, report, snapshot)
        self.assertTrue((self.cp/'old_a').exists())

    def test_existing_assets_cannot_be_excluded_as_feedback(self):
        folder, _ = self.feedback()
        output = folder/'cleanup_report.json'
        output.unlink()
        (folder/'unknown.json').write_text('old_a')
        with self.assertRaisesRegex(ValueError, 'only this script'):
            cleanup.feedback_files(self.root, output, folder/'checkpoints_cleanup.py')
        with self.assertRaisesRegex(ValueError, 'fresh automot_cleanup'):
            cleanup.feedback_files(self.root, self.cp/'old_a/cleanup_report.json', folder/'checkpoints_cleanup.py')

    def test_trace_only_selected_owner_and_all_matching_files(self):
        (self.cp/'keep/config.json').write_text('{"parent":"old_a"}')
        with gzip.open(self.cp/'keep/log.gz', 'wb') as f:
            f.write(b'historical old_a was examined')
        (self.cp/'old_b/unreadable.gz').write_bytes(b'not a gzip')
        evidence = []
        edges, _, _, _, _ = cleanup.inventory(self.cp, {'old_a', 'old_b'}, time.monotonic()+10,
            1024**2, owners=['keep'], reference_trace=evidence)
        self.assertEqual(edges, {'keep': {'old_a'}})
        self.assertEqual({x['file'] for x in evidence}, {'keep/config.json', 'keep/log.gz'})
        self.assertTrue(all('old_a' in x['excerpt'] for x in evidence))
        self.assertTrue((self.cp/'old_a').exists())

    def test_trace_archive_member_and_symlink(self):
        (self.cp/'keep/link').symlink_to('../old_a')
        with zipfile.ZipFile(self.cp/'keep/data.zip', 'w') as z:
            z.writestr('meta.json', '{"source":"old_b"}')
            z.writestr('old_c/note', 'no content reference')
        evidence = []
        edges, _, _, _, _ = cleanup.inventory(self.cp, {'old_a', 'old_b', 'old_c'}, time.monotonic()+10,
            1024**2, owners=['keep'], reference_trace=evidence)
        self.assertEqual(edges['keep'], {'old_a', 'old_b', 'old_c'})
        self.assertTrue(any(x.get('archive_member') == 'meta.json' and x['candidate'] == 'old_b' for x in evidence))
        self.assertTrue(any(x.get('evidence_kind') == 'archive_member_name' and x['candidate'] == 'old_c' for x in evidence))
        self.assertTrue(any('symlink_target' in x and x['candidate'] == 'old_a' for x in evidence))

    def test_trace_cli_cannot_apply_deletion(self):
        result = subprocess.run([sys.executable, '-S', '-B', cleanup.__file__, '--root', str(self.root),
            '--output', str(self.root/'trace.json'), '--trace-retained', '--apply'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('not allowed with argument', result.stderr)
        self.assertTrue((self.cp/'old_a').exists())


if __name__ == '__main__':
    unittest.main()
