from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from qwen3vl_local.audit_joint import dataset_catalog as c


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root/'data'; data.mkdir()
        self.index = data/'frame_index.jsonl'; self.index.write_text('{}\n')
        (data/'manifest.json').write_text('{"source_scope":"collection_results","action_rule_version":"v1"}')
        self.catalog = self.root/'catalog'; self.sources = {'teacher.py':'source-sha'}

    def register(self):
        return c.register(self.index, self.catalog, 'phase3', producer_sources=self.sources,
                          root=self.root, origin='existing_original_run')

    def test_repeated_registration_no_copy_no_new_versions(self):
        before = {p.name:p.read_bytes() for p in self.index.parent.iterdir()}
        first = self.register(); second = self.register()
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.catalog/'phase3/versions').glob('*.json'))), 1)
        self.assertEqual(Path(first['current']).resolve(), self.index.parent)
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.index.parent.iterdir()})
        record = c.resolve(self.catalog, 'phase3', producer_sources=self.sources)
        self.assertTrue(record['generation_provenance_unknown'])
        self.assertIn('hostname', record['registration'])

    def test_changed_bytes_or_producer_refuse_reuse(self):
        self.register()
        with self.assertRaises(ValueError): c.resolve(self.catalog, 'phase3', producer_sources={'teacher.py':'changed'})
        self.index.write_text('changed')
        with self.assertRaises(ValueError): c.resolve(self.catalog, 'phase3')

    def test_no_overwrite_real_current_directory(self):
        (self.catalog/'phase3/current').mkdir(parents=True)
        with self.assertRaises(ValueError): self.register()
        self.assertTrue((self.catalog/'phase3/current').is_dir())

    def test_link_drift_rejected(self):
        self.register()
        link = self.catalog/'phase3/current'; link.unlink(); link.symlink_to(self.root)
        with self.assertRaises(ValueError): c.resolve(self.catalog, 'phase3')

    def test_phase4_binds_holdout_and_external_replay(self):
        folder = self.index.parent
        def make(name, text):
            p = folder/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
            return c.file_binding(p)['sha256']
        files = {name:make(name,'{}\n') for name in ('train.jsonl','val.jsonl','test.jsonl')}
        queue_sha = make('review_queue.jsonl','')
        replay = self.root/'original_replay'; replay.mkdir()
        (replay/'index.json').write_text('{}')
        (replay/'route.jsonl.gz').write_bytes(b'original replay')
        def ref(p):
            value = c.file_binding(p)
            return dict(logical_path=value['path'],resolved_path=value['resolved'],sha256=value['sha256'])
        reference = dict(schema='shared_production_reference_v1',index=ref(replay/'index.json'),
                         routes={'route.jsonl.gz':ref(replay/'route.jsonl.gz')})
        reference_sha = make('production_reference.json',json.dumps(reference))
        index_sha = make('production/index.json','{}')
        (folder/'manifest.json').write_text(json.dumps(dict(files=files,review_queue_sha256=queue_sha,
            production_reference_sha256=reference_sha,production_index_sha256=index_sha)))
        result = c.register(folder/'train.jsonl',self.catalog,'phase4',producer_sources=self.sources,root=self.root)
        record = c.resolve(self.catalog,'phase4')
        self.assertIn(str(replay),record['required_paths'])
        (replay/'route.jsonl.gz').write_bytes(b'changed')
        with self.assertRaises(ValueError): c.resolve(self.catalog,'phase4')

    def contract(self):
        return dict(schema='causal_shards_v1', shared_sources={'history.py':'a'}, split_identity='split',
                    input_identity='dataset', shards={e:dict(sources={e+'.py':'rule'},input_sha256='raw') for e in ('event1','event2')})

    def test_local_rule_change_only_invalidates_independent_shard(self):
        old = self.contract(); new = deepcopy(old); new['shards']['event2']['sources']['event2.py'] = 'new'
        result = c.incremental_plan(old, new)
        self.assertEqual(result['reuse'], ['event1']); self.assertEqual(result['rebuild'], ['event2'])

    def test_shared_teacher_input_split_change_invalidates_all(self):
        old = self.contract()
        for key in ('shared_sources', 'input_identity', 'split_identity'):
            new = deepcopy(old); new[key] = {'changed':'sha'}
            result = c.incremental_plan(old, new)
            self.assertEqual(result['reuse'], []); self.assertEqual(len(result['rebuild']), 2)

    def test_missing_causal_dependencies_never_reused(self):
        old = self.contract(); new = deepcopy(old); new['shards']['event1']['sources'] = {}
        with self.assertRaises(ValueError): c.incremental_plan(old, new)


if __name__ == '__main__': unittest.main()
