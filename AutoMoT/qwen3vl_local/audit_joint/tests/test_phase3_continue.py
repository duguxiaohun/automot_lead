import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from qwen3vl_local.audit_joint import phase3_continue as c
from qwen3vl_local.audit_joint import dataset_catalog as catalog


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'AutoMoT'; self.root.mkdir()
        self.output = self.root.parent/'continued'

    def test_gpu_count_alias_duplicates_and_autoselect(self):
        for value in ('0,00,1,2', '0,1', '0,1,2,٣', '0,1,2,2', '0,1,2,-1'):
            with self.assertRaises(ValueError): c.gpu_ids(value)
        self.assertEqual(c.gpu_ids('03,2,1,0'), '3,2,1,0')
        with patch.object(subprocess, 'check_output', return_value='0,20,0\n1,2,0\n2,0,0\n3,0,2\n4,99,0\n'):
            self.assertEqual(c.gpu_ids(''), '2,3,1,0')

    def test_preserved_legacy_sequence_has_original_sha_and_default_plan(self):
        from qwen3vl_local.audit_joint import phase3_legacy_sequence as legacy
        from qwen3vl_local import sequence_batch as current
        self.assertEqual(c.sha(Path(legacy.__file__)), c.LEGACY_SHA)
        rows = [dict(id=str(i), event='event', sequence='route', frame=i,
                     action='STOP', split='train', group='route') for i in range(40)]
        old_history = new_history = None
        for epoch in range(3):
            old = legacy.plan(rows, per_event=32, world_size=4, batch_size=2, stride=1, epoch=epoch, history=old_history)
            new = current.plan(rows, per_event=32, world_size=4, batch_size=2, stride=1, epoch=epoch, history=new_history)
            self.assertEqual(old, new)
            old_history = old[1]['next_history']; new_history = new[1]['next_history']

    def test_full_checkpoint_sha_missing_extra_and_symlink(self):
        folder = self.root/'checkpoint'; folder.mkdir()
        p = folder/'training_state.pt'; p.write_bytes(b'weights')
        (folder/'receipt.json').write_text('{}')
        expected = {p.name:c.sha(p)}
        c.verify_files(folder, expected)
        p.write_bytes(b'changed')
        with self.assertRaises(ValueError): c.verify_files(folder, expected)
        p.unlink(); p.symlink_to(self.root/'outside')
        with self.assertRaises(ValueError): c.verify_files(folder, expected)
        with self.assertRaises(ValueError): c.verify_files(folder, {'../outside':'x'})

    def fixture(self):
        q = self.root/'qwen3vl_local'
        for name in ('sft_new_loop_phase3', 'sft_v2', 'qwen35'):
            (q/name).mkdir(parents=True); (q/name/'__init__.py').write_text('')
        for name in ('sequence_batch.py', 'batch_runtime.py', 'image_io.py', 'mrope_utils.py', 'cache_utils.py'):
            (q/name).write_text('# fixture\n')
        (q/'sft_new_loop_phase3/server_only.py').write_text('# server extra\n')
        (q/'sft_new_loop_phase3/eval.sh').write_text('#!/bin/bash\n')
        legacy = Path(c.__file__).with_name('phase3_legacy_sequence.py')
        (q/'sequence_batch.py').write_bytes(legacy.read_bytes())
        sources = c.sources(self.root)
        (q/'sequence_batch.py').write_text('# changed by Phase4\n')
        data = self.root/'checkpoints/shared'; data.mkdir(parents=True)
        (data/'frame_index.jsonl').write_text('{}\n')
        (data/'manifest.json').write_text('{"source_scope":"collection_results"}')
        (self.root/'lead_data').mkdir()
        (self.root/'checkpoints/base').mkdir()
        for name in ('one', 'two'): (self.root/name).mkdir()
        config = dict(index='checkpoints/shared/frame_index.jsonl', history_rgb_mode='4rgb',
            action_output_mode='choice', num_epochs=3, sampling_only=False, max_steps=0,
            training_index_sha256=c.sha(data/'frame_index.jsonl'), data_root='lead_data', model_dir='checkpoints/base')
        binding = dict(sources=sources, configuration=config, world_size=4, steps_per_epoch=1280)
        p3 = q/'sft_new_loop_phase3'
        pipeline = dict(group='4rgb_choice', historical_regression_enabled=False,
            source_sha256={p.name:c.sha(p) for p in p3.iterdir() if p.suffix in ('.py','.sh','.json')},
            data_identity=dict(index_sha256=c.sha(data/'frame_index.jsonl'),manifest_sha256=c.sha(data/'manifest.json')))
        checkpoint = self.root/'checkpoints/failed/train/training_states/epoch_0002'; checkpoint.mkdir(parents=True)
        (checkpoint/'training_state.pt').write_bytes(b'CPU fixture, no GPU')
        c.write(checkpoint/'receipt.json', dict(policy='phase3_epoch_training_state_v1', binding=binding,
                                             files={'training_state.pt':c.sha(checkpoint/'training_state.pt')}))
        c.write(checkpoint.parent.parent.parent/'pipeline_manifest.json', pipeline)
        return binding, pipeline, checkpoint

    def test_isolated_runtime_exact_restore_and_original_untouched(self):
        binding, pipeline, _ = self.fixture()
        runtime = self.root.parent/'runtime'
        c.prepare_runtime(self.root, runtime, binding, pipeline)
        self.assertEqual(c.sources(runtime), binding['sources'])
        self.assertEqual((self.root/'qwen3vl_local/sequence_batch.py').read_text(), '# changed by Phase4\n')
        self.assertEqual((runtime/'checkpoints').resolve(), self.root/'checkpoints')
        self.assertTrue((runtime/'qwen3vl_local/sft_new_loop_phase3/server_only.py').exists())

    def test_runtime_rejects_unrelated_drift(self):
        binding, pipeline, _ = self.fixture()
        (self.root/'qwen3vl_local/sft_v2/__init__.py').write_text('changed')
        with self.assertRaises(ValueError): c.prepare_runtime(self.root, self.root.parent/'runtime', binding, pipeline)

    def test_original_pipeline_test_hash_restored_without_modifying_worktree(self):
        binding, pipeline, _ = self.fixture()
        original = self.root/'qwen3vl_local/sft_new_loop_phase3/test_epoch_consumption.py'
        original.write_text('# newer test fixture\n')
        pipeline['source_sha256'][original.name] = c.LEGACY_TEST_SHA
        runtime = self.root.parent/'runtime'
        c.prepare_runtime(self.root, runtime, binding, pipeline)
        self.assertEqual(c.sha(runtime/'qwen3vl_local/sft_new_loop_phase3'/original.name), c.LEGACY_TEST_SHA)
        self.assertEqual(original.read_text(), '# newer test fixture\n')

    def run_fixture(self, fail=None):
        _, _, checkpoint = self.fixture()
        args = SimpleNamespace(root=self.root, output=self.output, checkpoint=checkpoint,
            completed_choice='one', completed_binary='two', gpu_ids='0,1,2,3')
        calls = []
        def execute(command, *, cwd, env, log):
            calls.append((log.stem, command, env))
            log.write_text('fixture process result\n')
            if log.stem == fail: raise RuntimeError('injected stage failure')
        with patch.object(c, 'completed', side_effect=lambda root,run,mode: {'mode':mode}), \
                patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30)), \
                patch.object(c, 'execute', side_effect=execute), patch.dict(sys.modules, {'dataset_catalog':catalog}):
            result = c.run(args)
        return result, calls

    def test_stage_order_no_data_generation_or_two_image_rerun(self):
        result, calls = self.run_fixture()
        self.assertEqual(result, 0)
        self.assertEqual([r[0] for r in calls], ['checkpoint_cpu_verify','choice_train','choice_test_audit',
            'choice_package','binary_train','binary_test_audit','binary_package'])
        job = c.read(self.output/'4rgb_choice/job.json')
        self.assertEqual(job['binding']['configuration']['num_epochs'], 3)
        self.assertEqual(job['binding']['configuration']['history_rgb_mode'], '4rgb')
        self.assertIn('--max_restarts=0', calls[1][1])
        self.assertEqual(calls[2][2]['RUN_AUDITS'], '1')
        self.assertEqual(calls[2][2]['EVAL_GPU_COUNT'], '4')
        self.assertTrue((self.output/'handoff/phase3_continue_report.zip').is_file())
        self.assertFalse(c.read(self.output/'workflow.json')['data_rebuilt'])

    def test_failure_stops_later_groups_and_packs_diagnostics(self):
        result, calls = self.run_fixture('choice_train')
        self.assertEqual(result, 2)
        self.assertEqual(len(calls), 2)
        record = c.read(self.output/'workflow.json')
        self.assertEqual(record['active_stage'], 'choice_train')
        self.assertEqual(record['status'], 'failed_or_interrupted')
        with zipfile.ZipFile(self.output/'handoff/phase3_continue_report.zip') as z:
            self.assertIn('choice_train.log', z.namelist())
            self.assertFalse(any(name.endswith('.pt') for name in z.namelist()))

    def test_actual_owned_subprocess_and_nonzero(self):
        log = self.root/'process.log'
        c.execute([sys.executable, '-c', 'print("done")'], cwd=self.root, env=os.environ.copy(), log=log)
        self.assertEqual(log.read_text().strip(), 'done')
        with self.assertRaises(RuntimeError):
            c.execute([sys.executable, '-c', 'raise SystemExit(7)'], cwd=self.root,
                      env=os.environ.copy(), log=self.root/'failed.log')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'native checkpoint read requires torch')
    def test_actual_native_checkpoint_cpu_worker(self):
        binding, _, checkpoint = self.fixture()
        from qwen3vl_local.sft_new_loop_phase3 import training_state
        target = self.root/'qwen3vl_local/sft_new_loop_phase3/training_state.py'
        target.write_bytes(Path(training_state.__file__).read_bytes())
        # Construct a real small checkpoint with native RNG/optimizer/scheduler.
        seed = self.root/'seed.py'
        seed.write_text('''import json, torch
from argparse import Namespace
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase3 import training_state as r
cfg=json.loads(Path('config.json').read_text())
b=r.binding(Namespace(**cfg),4,1280)
m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters())
s=torch.optim.lr_scheduler.LambdaLR(o,lambda _:1.)
p=Path('native_checkpoint');p.mkdir()
v=dict(binding=b,next_epoch=2,global_step=2560,sampling_history={'next_epoch':2},
       rng=[r.rng_state('cpu') for _ in range(4)],trainable={n:p.detach() for n,p in m.named_parameters()},
       optimizer=o.state_dict(),scheduler=s.state_dict())
torch.save(v,p/'training_state.pt')
(p/'receipt.json').write_text(json.dumps(dict(policy=r.POLICY,binding=b,files=r.files(p))))
''')
        c.write(self.root/'config.json', binding['configuration'])
        env = dict(os.environ, PYTHONPATH=str(self.root))
        subprocess.run([sys.executable, str(seed)], cwd=self.root, env=env, check=True, capture_output=True)
        checkpoint = self.root/'native_checkpoint'
        binding = c.read(checkpoint/'receipt.json')['binding']
        job = self.root/'job.json'
        c.write(job, dict(runtime=str(self.root), checkpoint=str(checkpoint), binding=binding))
        result = subprocess.run([sys.executable, str(Path(c.__file__).resolve()), '--worker', str(job), '--preflight'],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"global_step": 2560', result.stdout)
        with (checkpoint/'training_state.pt').open('ab') as stream: stream.write(b'damage')
        bad = subprocess.run([sys.executable, str(Path(c.__file__).resolve()), '--worker', str(job), '--preflight'],
                             env=env, text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('changed or incomplete', bad.stderr)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torchrun requires torch')
    def test_real_four_worker_dispatch_keeps_script_arguments(self):
        worker = self.root/'dispatch.py'
        worker.write_text('import os,sys\nfrom pathlib import Path\n'
                          'Path("rank_"+os.environ["RANK"]).write_text(" ".join(sys.argv[1:]))\n')
        c.execute([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '--max_restarts=0', '--log-dir', str(self.root/'rank_logs'), '--tee', '3', '--',
                   str(worker), '--worker', 'job.json'], cwd=self.root, env=os.environ.copy(), log=self.root/'launch.log')
        self.assertEqual(len(list(self.root.glob('rank_[0-3]'))), 4)
        for p in self.root.glob('rank_[0-3]'):
            self.assertEqual(p.read_text(), '--worker job.json')


if __name__ == '__main__': unittest.main()
