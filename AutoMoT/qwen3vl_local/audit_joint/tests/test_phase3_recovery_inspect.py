import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
